# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from ..data_utils import data_utils


class InputEmbedding(nn.Module):
	"""
	Component to compute the embedding of each terminal node.
	"""
	def __init__(self, args, term_vocab, term_vocab_list):
		super(InputEmbedding, self).__init__()
		self.dataProcessor = data_utils.HalideDataProcessor()
		self.term_vocab = term_vocab
		self.term_vocab_list = term_vocab_list
		self.term_vocab_size = args.term_vocab_size
		self.embedding_size = args.embedding_size
		self.hidden_size = args.LSTM_hidden_size
		self.cuda_flag = args.cuda

		self.char_embedding = nn.Embedding(self.term_vocab_size, self.embedding_size)
		self.token_embedding = nn.LSTM(input_size=self.embedding_size,
			hidden_size=self.hidden_size,
			num_layers=1,
			batch_first=True)

	def forward(self, raw_input_tokens, eval_mode=False):
		input_tokens = []
		for raw_inp in raw_input_tokens:
			input_tokens.append(self.dataProcessor.token_to_ids(raw_inp, self.term_vocab))
		input_tokens = np.array(input_tokens)
		input_tokens = data_utils.np_to_tensor(input_tokens, 'int', self.cuda_flag, eval_mode)
		if len(input_tokens.size()) < 2:
			input_tokens = input_tokens.unsqueeze(0)
		init_embedding = self.char_embedding(input_tokens)
		batch_size = input_tokens.size()[0]
		init_h = Variable(torch.zeros(1, batch_size, self.hidden_size))
		init_c = Variable(torch.zeros(1, batch_size, self.hidden_size))
		if self.cuda_flag:
			init_h = init_h.cuda()
			init_c = init_c.cuda()
		init_state = (init_h, init_c)
		embedding_outputs, embedding_states = self.token_embedding(init_embedding, init_state)
		return embedding_states


class TreeLSTM(nn.Module):
	"""
	Tree LSTM to embed each node in the tree. It is used for expression simplification.
	"""
	def __init__(self, args, term_vocab, term_vocab_list, op_vocab, op_vocab_list):
		super(TreeLSTM, self).__init__()
		self.batch_size = args.batch_size
		self.hidden_size = args.LSTM_hidden_size
		self.embedding_size = args.embedding_size
		self.dropout_rate = args.dropout_rate
		self.cuda_flag = args.cuda
		self.term_vocab_size = args.term_vocab_size
		self.term_vocab = term_vocab
		self.term_vocab_list = term_vocab_list
		self.op_vocab_size = args.op_vocab_size
		self.op_vocab = op_vocab
		self.op_vocab_list = op_vocab_list
		self.input_embedding = InputEmbedding(args, term_vocab, term_vocab_list)

		self.ih = nn.ModuleList([nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=True) for _ in range(3)]) for _ in range(self.op_vocab_size)])
		self.fh = nn.ModuleList([nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=True) for _ in range(3)]) for _ in range(self.op_vocab_size)])
		self.oh = nn.ModuleList([nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=True) for _ in range(3)]) for _ in range(self.op_vocab_size)])
		self.uh = nn.ModuleList([nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=True) for _ in range(3)]) for _ in range(self.op_vocab_size)])

		
	def calc_root(self, ops, child_h, child_c):
		i = []
		o = []
		u = []
		fc = []
		h = []
		c = []
		for idx in range(len(ops)):
			op = self.op_vocab[ops[idx]]
			i_cur = Variable(torch.zeros(1, self.hidden_size))
			o_cur = Variable(torch.zeros(1, self.hidden_size))
			u_cur = Variable(torch.zeros(1, self.hidden_size))
			f_cur = []
			if self.cuda_flag:
				i_cur = i_cur.cuda()
				o_cur = o_cur.cuda()
				u_cur = u_cur.cuda()
			for child_idx in range(len(child_h[idx])):
				i_cur += self.ih[op][child_idx](child_h[idx][child_idx])
				o_cur += self.oh[op][child_idx](child_h[idx][child_idx])
				u_cur += self.uh[op][child_idx](child_h[idx][child_idx])
				f_cur.append(self.fh[op][child_idx](child_h[idx][child_idx]))
			i_cur = F.sigmoid(i_cur)
			o_cur = F.sigmoid(o_cur)
			u_cur = F.tanh(u_cur)
			f_cur = torch.cat(f_cur, 0)
			f_cur = F.sigmoid(f_cur)
			fc_cur = F.torch.mul(f_cur, torch.cat(child_c[idx], 0))
			fc_cur = F.torch.sum(fc_cur, 0)
			i.append(i_cur.unsqueeze(0))
			o.append(o_cur.unsqueeze(0))
			u.append(u_cur.unsqueeze(0))
			fc.append(fc_cur.unsqueeze(0).unsqueeze(0))
		i = torch.cat(i, 0)
		o = torch.cat(o, 0)
		u = torch.cat(u, 0)
		fc = torch.cat(fc, 0)
		c = F.torch.mul(i, u) + fc
		h = F.torch.mul(o, F.tanh(c))
		return h, c


	def calc_embedding(self, tree_managers, eval_mode=False):
		queue_term = []
		queue_nonterm = []
		head_term = 0
		head_nonterm = 0
		max_num_trees = 0

		for tree_manager_idx in range(len(tree_managers)):
			tree_manager = tree_managers[tree_manager_idx]
			max_num_trees = max(max_num_trees, tree_manager.num_trees)
			for idx in range(tree_manager.num_trees):
				cur_tree = tree_manager.get_tree(idx)
				canCompute = True
				children_h = []
				children_c = []
				for child_idx in cur_tree.children:
					child = tree_manager.get_tree(child_idx)
					if child.state is None:
						canCompute = False
						break
					else:
						child_h, child_c = child.state
						children_h.append(child_h)
						children_c.append(child_c)
				if canCompute:
					if len(children_h) == 0:
						queue_term.append((tree_manager_idx, idx, cur_tree.root))
					else:
						queue_nonterm.append((tree_manager_idx, idx, cur_tree.root, children_h, children_c))

		while head_term < len(queue_term):
			encoder_inputs = []
			tree_idxes = []
			while head_term < len(queue_term):
				tree_manager_idx, idx, root = queue_term[head_term]
				tree_idxes.append((tree_manager_idx, idx))
				encoder_inputs.append(root)
				head_term += 1
			if len(encoder_inputs) == 0:
				break
			encoder_outputs = self.input_embedding(encoder_inputs, eval_mode)
			for i in range(len(tree_idxes)):
				tree_manager_idx, cur_idx = tree_idxes[i]
				tree_manager = tree_managers[tree_manager_idx]
				child_h = encoder_outputs[0][:, i, :]
				child_c = encoder_outputs[1][:, i, :]
				tree_managers[tree_manager_idx].trees[cur_idx].state = child_h, child_c
				cur_tree = tree_manager.get_tree(cur_idx)
				if cur_tree.parent != -1:
					parent_tree = tree_manager.get_tree(cur_tree.parent)
					canCompute = True
					children_h = []
					children_c = []
					for child_idx in parent_tree.children:
						child = tree_manager.get_tree(child_idx)
						if child.state is None:
							canCompute = False
							break
						else:
							child_h, child_c = child.state
							children_h.append(child_h)
							children_c.append(child_c)
					if canCompute:
						queue_nonterm.append((tree_manager_idx, cur_tree.parent, parent_tree.root, children_h, children_c))

		while head_nonterm < len(queue_nonterm):
			encoder_inputs = []
			children_h = []
			children_c = []
			tree_idxes = []
			while head_nonterm < len(queue_nonterm):
				tree_manager_idx, idx, root, child_h, child_c = queue_nonterm[head_nonterm]
				cur_tree = tree_managers[tree_manager_idx].get_tree(idx)
				if cur_tree.state is None:
					tree_idxes.append((tree_manager_idx, idx))
					encoder_inputs.append(root)
					children_h.append(child_h)
					children_c.append(child_c)
				head_nonterm += 1
			if len(encoder_inputs) == 0:
				break
			encoder_outputs = self.calc_root(encoder_inputs, children_h, children_c)
			for i in range(len(tree_idxes)):
				tree_manager_idx, cur_idx = tree_idxes[i]
				tree_manager = tree_managers[tree_manager_idx]
				child_h = encoder_outputs[0][i]
				child_c = encoder_outputs[1][i]
				tree_managers[tree_manager_idx].trees[cur_idx].state = child_h, child_c
				cur_tree = tree_manager.get_tree(cur_idx)
				if cur_tree.parent != -1:
					parent_tree = tree_manager.get_tree(cur_tree.parent)
					canCompute = True
					children_h = []
					children_c = []
					for child_idx in parent_tree.children:
						child = tree_manager.get_tree(child_idx)
						if child.state is None:
							canCompute = False
							break
						else:
							child_h, child_c = child.state
							children_h.append(child_h)
							children_c.append(child_c)
					if canCompute:
						queue_nonterm.append((tree_manager_idx, cur_tree.parent, parent_tree.root, children_h, children_c))
		return tree_managers


	def update_embedding(self, tree_managers, init_queues, eval_mode=False):
		queue_term = []
		queue_nonterm = []
		head_term = 0
		head_nonterm = 0

		for tree_manager_idx in range(len(tree_managers)):
			tree_manager = tree_managers[tree_manager_idx]
			init_queue = init_queues[tree_manager_idx]
			for idx in init_queue:
				if idx == -1:
					continue
				cur_tree = tree_manager.get_tree(idx)
				canCompute = True
				children_h = []
				children_c = []
				for child_idx in cur_tree.children:
					child = tree_manager.get_tree(child_idx)
					if child.state is None:
						canCompute = False
						break
					else:
						child_h, child_c = child.state
						children_h.append(child_h)
						children_c.append(child_c)
				if len(cur_tree.children) == 0:
					queue_term.append((tree_manager_idx, idx, cur_tree.root))
				elif canCompute:
					queue_nonterm.append((tree_manager_idx, idx, cur_tree.root, children_h, children_c))

		while head_term < len(queue_term):
			encoder_inputs = []
			tree_idxes = []
			while head_term < len(queue_term):
				tree_manager_idx, idx, root = queue_term[head_term]
				tree_idxes.append((tree_manager_idx, idx))
				encoder_inputs.append(root)
				head_term += 1
			if len(encoder_inputs) == 0:
				break
			encoder_outputs = self.input_embedding(encoder_inputs, eval_mode)
			for i in range(len(tree_idxes)):
				tree_manager_idx, cur_idx = tree_idxes[i]
				tree_manager = tree_managers[tree_manager_idx]
				child_h = encoder_outputs[0][:, i, :]
				child_c = encoder_outputs[1][:, i, :]
				tree_managers[tree_manager_idx].trees[cur_idx].state = child_h, child_c
				cur_tree = tree_manager.get_tree(cur_idx)
				if cur_tree.parent != -1:
					parent_tree = tree_manager.get_tree(cur_tree.parent)
					canCompute = True
					children_h = []
					children_c = []
					for child_idx in parent_tree.children:
						child = tree_manager.get_tree(child_idx)
						if child.state is None:
							canCompute = False
							break
						else:
							child_h, child_c = child.state
							children_h.append(child_h)
							children_c.append(child_c)
					if canCompute:
						queue_nonterm.append((tree_manager_idx, cur_tree.parent, parent_tree.root, children_h, children_c))

		while head_nonterm < len(queue_nonterm):
			encoder_inputs = []
			children_h = []
			children_c = []
			tree_idxes = []
			while head_nonterm < len(queue_nonterm):
				tree_manager_idx, idx, root, child_h, child_c = queue_nonterm[head_nonterm]
				cur_tree = tree_managers[tree_manager_idx].get_tree(idx)
				tree_idxes.append((tree_manager_idx, idx))
				encoder_inputs.append(root)
				children_h.append(child_h)
				children_c.append(child_c)
				head_nonterm += 1
			if len(encoder_inputs) == 0:
				break
			encoder_outputs = self.calc_root(encoder_inputs, children_h, children_c)
			for i in range(len(tree_idxes)):
				tree_manager_idx, cur_idx = tree_idxes[i]
				tree_manager = tree_managers[tree_manager_idx]
				child_h = encoder_outputs[0][i]
				child_c = encoder_outputs[1][i]
				tree_managers[tree_manager_idx].trees[cur_idx].state = child_h, child_c
				cur_tree = tree_manager.get_tree(cur_idx)
				if cur_tree.parent != -1:
					parent_tree = tree_manager.get_tree(cur_tree.parent)
					canCompute = True
					children_h = []
					children_c = []
					for child_idx in parent_tree.children:
						child = tree_manager.get_tree(child_idx)
						if child.state is None:
							canCompute = False
							break
						else:
							child_h, child_c = child.state
							children_h.append(child_h)
							children_c.append(child_c)
					if canCompute:
						queue_nonterm.append((tree_manager_idx, cur_tree.parent, parent_tree.root, children_h, children_c))

		return tree_managers
				

