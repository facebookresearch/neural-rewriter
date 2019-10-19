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


class SeqLSTM(nn.Module):
	"""
	LSTM to embed the input as a sequence.
	"""
	def __init__(self, args):
		super(SeqLSTM, self).__init__()
		self.batch_size = args.batch_size
		self.hidden_size = args.LSTM_hidden_size
		self.embedding_size = args.embedding_size
		self.num_layers = args.num_LSTM_layers
		self.dropout_rate = args.dropout_rate
		self.cuda_flag = args.cuda
		self.encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_rate)


	def calc_embedding(self, dag_managers, eval_mode=False):
		encoder_input = []
		encoder_input_idx = []
		max_node_cnt = 0
		batch_size = len(dag_managers)

		for dag_manager in dag_managers:
			cur_encoder_input = []
			cur_encoder_input_idx = []
			max_node_cnt = max(max_node_cnt, dag_manager.num_jobs)
			for st in range(dag_manager.max_schedule_time + 1):
				for idx in dag_manager.schedule[st]:
					cur_encoder_input_idx.append(idx)
					cur_encoder_input.append(dag_manager.get_node(idx).embedding)
			encoder_input.append(cur_encoder_input)
			encoder_input_idx.append(cur_encoder_input_idx)

		for i in range(batch_size):
			while len(encoder_input[i]) < max_node_cnt:
				encoder_input[i].append([0.0 for _ in range(self.embedding_size)])

		encoder_input = np.array(encoder_input)
		encoder_input = data_utils.np_to_tensor(encoder_input, 'float', self.cuda_flag, eval_mode)
		init_h = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
		init_c = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
		if self.cuda_flag:
			init_h = init_h.cuda()
			init_c = init_c.cuda()
		init_state = (init_h, init_c)
		encoder_output, encoder_state = self.encoder(encoder_input, init_state)

		for dag_manager_idx, dag_manager in enumerate(dag_managers):
			for i, node_idx in enumerate(encoder_input_idx[dag_manager_idx]):
				dag_managers[dag_manager_idx].nodes[node_idx].state = (encoder_output[dag_manager_idx, i, :].unsqueeze(0), encoder_output[dag_manager_idx, i, :].unsqueeze(0))
			init_h = Variable(torch.zeros(1, self.hidden_size))
			init_c = Variable(torch.zeros(1, self.hidden_size))
			if self.cuda_flag:
				init_h = init_h.cuda()
				init_c = init_c.cuda()
			init_state = (init_h, init_c)
			dag_managers[dag_manager_idx].nodes[0].state = init_state
		return dag_managers


class DagLSTM(nn.Module):
	"""
	LSTM to embed the input as a DAG.
	"""
	def __init__(self, args):
		super(DagLSTM, self).__init__()
		self.batch_size = args.batch_size
		self.hidden_size = args.LSTM_hidden_size
		self.embedding_size = args.embedding_size
		self.dropout_rate = args.dropout_rate
		self.cuda_flag = args.cuda

		self.ix = nn.Linear(self.embedding_size, self.hidden_size, bias=True)
		self.ih = nn.Linear(self.hidden_size, self.hidden_size)
		self.fx = nn.Linear(self.embedding_size, self.hidden_size, bias=True)
		self.fh = nn.Linear(self.hidden_size, self.hidden_size)
		self.ox = nn.Linear(self.embedding_size, self.hidden_size, bias=True)
		self.oh = nn.Linear(self.hidden_size, self.hidden_size)
		self.ux = nn.Linear(self.embedding_size, self.hidden_size, bias=True)
		self.uh = nn.Linear(self.hidden_size, self.hidden_size)

		
	def calc_root(self, inputs, child_h, child_c):
		child_h_sum = torch.sum(child_h, 1)
		i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
		o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
		u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))

		fx = self.fx(inputs)
		fx = fx.unsqueeze(1)
		fx = fx.repeat(1, child_h.size()[1], 1)
		f = self.fh(child_h)
		f = f + fx
		f = F.sigmoid(f)
		fc = F.torch.mul(f, child_c)
		fc = torch.sum(fc, 1)
		c = F.torch.mul(i, u) + fc
		h = F.torch.mul(o, F.tanh(c))
		return h, c


	def calc_embedding(self, dag_managers, eval_mode=False):
		queue = []
		head = 0

		for dag_manager_idx in range(len(dag_managers)):
			for idx in range(dag_managers[dag_manager_idx].num_nodes):
				dag_managers[dag_manager_idx].nodes[idx].state = None

		for dag_manager_idx in range(len(dag_managers)):
			dag_manager = dag_managers[dag_manager_idx]
			root_node = dag_manager.get_node(0)
			children_h = []
			children_c = []
			queue.append((dag_manager_idx, 0, root_node.embedding, children_h, children_c))

		while head < len(queue):
			encoder_inputs = []
			children_h = []
			children_c = []
			dag_idxes = []
			max_children_size = 1
			while head < len(queue):
				dag_manager_idx, idx, embedding, child_h, child_c = queue[head]
				cur_node = dag_managers[dag_manager_idx].get_node(idx)
				dag_idxes.append((dag_manager_idx, idx))
				encoder_inputs.append(embedding)
				children_h.append(child_h)
				children_c.append(child_c)
				max_children_size = max(max_children_size, len(child_h))
				head += 1
			if len(encoder_inputs) == 0:
				break
			encoder_inputs = np.array(encoder_inputs)
			encoder_inputs = data_utils.np_to_tensor(encoder_inputs, 'float', self.cuda_flag, eval_mode)

			for idx in range(len(children_h)):
				while len(children_h[idx]) < max_children_size:
					init_child_h = Variable(torch.zeros(1, self.hidden_size))
					init_child_c = Variable(torch.zeros(1, self.hidden_size))
					if self.cuda_flag:
						init_child_h = init_child_h.cuda()
						init_child_c = init_child_c.cuda()
					children_h[idx].append(init_child_h)
					children_c[idx].append(init_child_c)
				children_h[idx] = torch.cat(children_h[idx], 0).unsqueeze(0)
				children_c[idx] = torch.cat(children_c[idx], 0).unsqueeze(0)

			children_h = torch.cat(children_h, 0)
			children_c = torch.cat(children_c, 0)

			encoder_outputs = self.calc_root(encoder_inputs, children_h, children_c)
			for i in range(len(dag_idxes)):
				dag_manager_idx, cur_idx = dag_idxes[i]
				dag_manager = dag_managers[dag_manager_idx]
				child_h = encoder_outputs[0][i].unsqueeze(0)
				child_c = encoder_outputs[1][i].unsqueeze(0)
				dag_managers[dag_manager_idx].nodes[cur_idx].state = child_h, child_c
				cur_node = dag_manager.get_node(cur_idx)
				if len(cur_node.children) > 0:
					for child_idx in cur_node.children:
						child_node = dag_manager.get_node(child_idx)
						canCompute = True
						children_h = []
						children_c = []
						for parent_idx in child_node.parents:
							parent = dag_manager.get_node(parent_idx)
							if parent.state is None:
								canCompute = False
								break
							else:
								child_h, child_c = parent.state
								children_h.append(child_h)
								children_c.append(child_c)
						if canCompute:
							queue.append((dag_manager_idx, child_idx, child_node.embedding, children_h, children_c))
		return dag_managers


	def update_embedding(self, dag_managers, init_queues, eval_mode=False):
		queue = []
		head = 0
		for dag_manager_idx in range(len(dag_managers)):
			init_queue = init_queues[dag_manager_idx]
			for idx in init_queue:
				dag_managers[dag_manager_idx].nodes[idx].state = None
		for dag_manager_idx in range(len(dag_managers)):
			dag_manager = dag_managers[dag_manager_idx]
			init_queue = init_queues[dag_manager_idx]
			for idx in init_queue:
				cur_node = dag_manager.get_node(idx)
				canCompute = True
				children_h = []
				children_c = []
				for parent_idx in cur_node.parents:
					parent = dag_manager.get_node(parent_idx)
					if parent.state is None:
						canCompute = False
						break
					else:
						child_h, child_c = parent.state
						children_h.append(child_h)
						children_c.append(child_c)
				if canCompute:
					queue.append((dag_manager_idx, idx, cur_node.embedding, children_h, children_c))

		while head < len(queue):
			encoder_inputs = []
			children_h = []
			children_c = []
			dag_idxes = []
			max_children_size = 1
			while head < len(queue):
				dag_manager_idx, idx, embedding, child_h, child_c = queue[head]
				cur_node = dag_managers[dag_manager_idx].get_node(idx)
				if cur_node.state is None:
					dag_idxes.append((dag_manager_idx, idx))
					encoder_inputs.append(embedding)
					children_h.append(child_h)
					children_c.append(child_c)
					max_children_size = max(max_children_size, len(child_h))
				head += 1
			if len(encoder_inputs) == 0:
				break
			encoder_inputs = np.array(encoder_inputs)
			encoder_inputs = data_utils.np_to_tensor(encoder_inputs, 'float', self.cuda_flag, eval_mode)

			for idx in range(len(children_h)):
				while len(children_h[idx]) < max_children_size:
					init_child_h = Variable(torch.zeros(1, self.hidden_size))
					init_child_c = Variable(torch.zeros(1, self.hidden_size))
					if self.cuda_flag:
						init_child_h = init_child_h.cuda()
						init_child_c = init_child_c.cuda()
					children_h[idx].append(init_child_h)
					children_c[idx].append(init_child_c)
				children_h[idx] = torch.cat(children_h[idx], 0).unsqueeze(0)
				children_c[idx] = torch.cat(children_c[idx], 0).unsqueeze(0)

			children_h = torch.cat(children_h, 0)
			children_c = torch.cat(children_c, 0)
			encoder_outputs = self.calc_root(encoder_inputs, children_h, children_c)
			for i in range(len(dag_idxes)):
				dag_manager_idx, cur_idx = dag_idxes[i]
				dag_manager = dag_managers[dag_manager_idx]
				child_h = encoder_outputs[0][i].unsqueeze(0)
				child_c = encoder_outputs[1][i].unsqueeze(0)
				dag_managers[dag_manager_idx].nodes[cur_idx].state = child_h, child_c
				cur_node = dag_manager.get_node(cur_idx)
				if len(cur_node.children) > 0:
					for child_idx in cur_node.children:
						child_node = dag_manager.get_node(child_idx)
						canCompute = True
						children_h = []
						children_c = []
						for parent_idx in child_node.parents:
							parent = dag_manager.get_node(parent_idx)
							if parent.state is None:
								canCompute = False
								break
							else:
								child_h, child_c = parent.state
								children_h.append(child_h)
								children_c.append(child_c)
						if canCompute:
							queue.append((dag_manager_idx, child_idx, child_node.embedding, children_h, children_c))

		return dag_managers