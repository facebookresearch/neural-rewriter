# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import operator
import random
import time
from multiprocessing.pool import ThreadPool

import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .data_utils import data_utils
from .modules import HalideInputEncoder, mlp
from .rewriter import HalideRewriter
from .BaseModel import BaseModel

eps = 1e-3
log_eps = np.log(eps)


class HalideModel(BaseModel):
	"""
	Model for expression simplification.
	"""
	def __init__(self, args, term_vocab, term_vocab_list, op_vocab, op_vocab_list):
		super(HalideModel, self).__init__(args)
		self.term_vocab = term_vocab
		self.term_vocab_list = term_vocab_list
		self.op_vocab = op_vocab
		self.op_vocab_list = op_vocab_list
		self.term_vocab_size = args.term_vocab_size
		self.op_vocab_size = args.op_vocab_size
		self.embedding_size = args.embedding_size
		self.num_actions = args.num_actions
		self.reward_thres = -0.05
		self.rewriter = HalideRewriter(args, term_vocab, term_vocab_list, op_vocab, op_vocab_list)
		self.input_encoder = HalideInputEncoder.TreeLSTM(args, term_vocab, term_vocab_list, op_vocab, op_vocab_list)
		self.policy = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 2, self.MLP_hidden_size, self.num_actions, self.cuda_flag, self.dropout_rate)
		self.value_estimator = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 2, self.MLP_hidden_size, 1, self.cuda_flag, self.dropout_rate)

		if args.optimizer == 'adam':
			self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
		elif args.optimizer == 'sgd':
			self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
		elif args.optimizer == 'rmsprop':
			self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
		else:
			raise ValueError('optimizer undefined: ', args.optimizer)


	def rewrite(self, tm, ac_logprobs, trace_rec, expr_rec, candidate_rewrite_pos, pending_actions, eval_flag, max_search_pos, reward_thres=None):
		if len(candidate_rewrite_pos) == 0:
			return [], [], [], [], [], []

		candidate_rewrite_pos.sort(reverse=True, key=operator.itemgetter(0))
		if not eval_flag:
			sample_exp_reward_tensor = []
			for idx, (cur_pred_reward, cur_pred_reward_tensor, cur_ac_prob, rewrite_pos, tensor_idx) in enumerate(candidate_rewrite_pos):
				sample_exp_reward_tensor.append(cur_pred_reward_tensor)
			sample_exp_reward_tensor = torch.cat(sample_exp_reward_tensor, 0)
			sample_exp_reward_tensor = torch.exp(sample_exp_reward_tensor * 10)
			sample_exp_reward = sample_exp_reward_tensor.data.cpu().numpy()

		expr = expr_rec[-1]
		extra_reward_rec = []
		extra_action_rec = []
		candidate_tree_managers = []
		candidate_update_tree_idxes = []
		candidate_rewrite_rec = []
		candidate_expr_rec = []
		candidate_pending_actions = []

		if len(pending_actions) > 0:
			for idx, (pred_reward, cur_pred_reward_tensor, cur_ac_prob, rewrite_pos, tensor_idx) in enumerate(candidate_rewrite_pos):
				if len(candidate_tree_managers) > 0 and idx >= max_search_pos:
					break
				if reward_thres is not None and pred_reward < reward_thres:
					if eval_flag:
						break
					elif np.random.random() > self.cont_prob:
						continue
				init_expr = tm.to_string(rewrite_pos)
				op_idx = pending_actions[0]
				op_list = self.rewriter.get_rewrite_seq(op_idx)
				op = self.rewriter.get_rewrite_op(op_list[0])
				new_tm, cur_update_tree_idxes = op(tm, rewrite_pos)
				if len(cur_update_tree_idxes) == 0:
					extra_action_rec.append((ac_logprobs[tensor_idx], op_idx))
					continue
				cur_expr = str(new_tm)
				if cur_expr in candidate_expr_rec:
					continue
				candidate_expr_rec.append(cur_expr)
				candidate_update_tree_idxes.append(cur_update_tree_idxes)
				candidate_tree_managers.append(new_tm)
				candidate_rewrite_rec.append((ac_logprobs[tensor_idx], pred_reward, cur_pred_reward_tensor, rewrite_pos, init_expr, int(op_idx)))
				candidate_pending_actions.append(pending_actions[1:])
				if len(candidate_tree_managers) >= max_search_pos:
					break
			if len(candidate_tree_managers) > 0:
				return candidate_tree_managers, candidate_update_tree_idxes, candidate_rewrite_rec, candidate_pending_actions, extra_reward_rec, extra_action_rec

		if not eval_flag:
			sample_rewrite_pos_dist = Categorical(sample_exp_reward_tensor)
			sample_rewrite_pos = sample_rewrite_pos_dist.sample(sample_shape=[len(candidate_rewrite_pos)])
			#sample_rewrite_pos = torch.multinomial(sample_exp_reward_tensor, len(candidate_rewrite_pos))
			sample_rewrite_pos = sample_rewrite_pos.data.cpu().numpy()
			indexes = np.unique(sample_rewrite_pos, return_index=True)[1]
			sample_rewrite_pos = [sample_rewrite_pos[i] for i in sorted(indexes)]
			sample_rewrite_pos = sample_rewrite_pos[:self.num_sample_rewrite_pos]
			sample_exp_reward = [sample_exp_reward[i] for i in sample_rewrite_pos]
			sample_rewrite_pos = [candidate_rewrite_pos[i] for i in sample_rewrite_pos]
		else:
			sample_rewrite_pos = candidate_rewrite_pos.copy()

		for idx, (pred_reward, cur_pred_reward_tensor, cur_ac_prob, rewrite_pos, tensor_idx) in enumerate(sample_rewrite_pos):
			if len(candidate_tree_managers) > 0 and idx >= max_search_pos:
				break
			if reward_thres is not None and pred_reward < reward_thres:
				if eval_flag:
					break
				elif np.random.random() > self.cont_prob:
					continue
			init_expr = tm.to_string(rewrite_pos)
			if eval_flag:
				_, candidate_acs = torch.sort(cur_ac_prob)
				candidate_acs = candidate_acs.data.cpu().numpy()
				candidate_acs = candidate_acs[::-1]
			else:
				candidate_acs_dist = Categorical(cur_ac_prob)
				candidate_acs = candidate_acs_dist.sample(sample_shape=[self.num_actions])
				#candidate_acs = torch.multinomial(cur_ac_prob, self.num_actions)
				candidate_acs = candidate_acs.data.cpu().numpy()
				indexes = np.unique(candidate_acs, return_index=True)[1]
				candidate_acs = [candidate_acs[i] for i in sorted(indexes)]
			cur_active = False
			cur_ac_prob = cur_ac_prob.data.cpu().numpy()
			for i, op_idx in enumerate(candidate_acs):
				if (expr, init_expr, op_idx) in trace_rec:
					continue
				op_list = self.rewriter.get_rewrite_seq(op_idx)
				op = self.rewriter.get_rewrite_op(op_list[0])
				new_tm, cur_update_tree_idxes = op(tm, rewrite_pos)
				if len(cur_update_tree_idxes) == 0:
					extra_action_rec.append((ac_logprobs[tensor_idx], op_idx))
					continue
				cur_expr = str(new_tm)
				if cur_expr in candidate_expr_rec:
					continue
				candidate_expr_rec.append(cur_expr)
				candidate_update_tree_idxes.append(cur_update_tree_idxes)
				candidate_tree_managers.append(new_tm)
				candidate_rewrite_rec.append((ac_logprobs[tensor_idx], pred_reward, cur_pred_reward_tensor, rewrite_pos, init_expr, int(op_list[0])))
				candidate_pending_actions.append(op_list[1:])
				cur_active = True
				if len(candidate_tree_managers) >= max_search_pos:
					break
			if not cur_active:
				extra_reward_rec.append(cur_pred_reward_tensor)
		return candidate_tree_managers, candidate_update_tree_idxes, candidate_rewrite_rec, candidate_pending_actions, extra_reward_rec, extra_action_rec


	def batch_rewrite(self, tree_managers, ac_logprobs, trace_rec, expr_rec, candidate_rewrite_pos, pending_actions, eval_flag, max_search_pos, reward_thres):
		candidate_tree_managers = []
		candidate_update_tree_idxes = []
		candidate_rewrite_rec = []
		candidate_pending_actions = []
		extra_reward_rec = []
		extra_action_rec = []
		for i in range(len(tree_managers)):
			cur_candidate_tree_managers, cur_candidate_update_tree_idxes, cur_candidate_rewrite_rec, cur_candidate_pending_actions, cur_extra_reward_rec, cur_extra_action_rec = self.rewrite(tree_managers[i], ac_logprobs, trace_rec[i], expr_rec[i], candidate_rewrite_pos[i], pending_actions[i], eval_flag, max_search_pos, reward_thres)
			candidate_tree_managers.append(cur_candidate_tree_managers)
			candidate_update_tree_idxes.append(cur_candidate_update_tree_idxes)
			candidate_rewrite_rec.append(cur_candidate_rewrite_rec)
			candidate_pending_actions.append(cur_candidate_pending_actions)
			extra_reward_rec = extra_reward_rec + cur_extra_reward_rec
			extra_action_rec = extra_action_rec + cur_extra_action_rec
		return candidate_tree_managers, candidate_update_tree_idxes, candidate_rewrite_rec, candidate_pending_actions, extra_reward_rec, extra_action_rec


	def calc_dependency(self, tm, cur_idx=None):
		if cur_idx is None:
			cur_idx = tm.root
			tm.trees[cur_idx].depth = 0
		tm.trees[cur_idx].dependency_parent = cur_idx
		cur_tree = tm.get_tree(cur_idx)
		if len(cur_tree.children) == 0:
			return []
		nonterm_idxes = []
		nonterm_idxes += [cur_idx]
		for child in cur_tree.children:
			tm.trees[child].depth = cur_tree.depth + 1
			child_tree = tm.get_tree(child)
			if child_tree.parent != cur_idx:
				raise ValueError('invalid edge: ' + str(cur_idx) + ' ' + cur_tree.root + ' ' + str(cur_tree.children) + ' ' + str(child) + ' ' + str(child_tree.parent))
			nonterm_idxes += self.calc_dependency(tm, child)
		tm.trees[cur_idx].dependency_parent = tm.root
		return nonterm_idxes


	def forward(self, batch_data, eval_flag=False):
		tree_managers = []
		batch_size = len(batch_data)
		for trace, tm in batch_data:
			tree_managers.append(tm)
		tree_managers = self.input_encoder.calc_embedding(tree_managers, eval_flag)

		active = True
		reduce_steps = 0

		trace_rec = [[] for _ in range(batch_size)]
		rewrite_rec = [[] for _ in range(batch_size)]
		tm_rec = [[] for _ in range(batch_size)]
		expr_rec = [[] for _ in range(batch_size)]
		extra_reward_rec = []
		extra_action_rec = []

		for idx in range(batch_size):
			expr_rec[idx].append(str(tree_managers[idx]))
			trace_rec[idx].append((expr_rec[idx][-1], '', -1))
			tm_rec[idx].append(tree_managers[idx])

		pending_actions = [[] for _ in range(batch_size)]
		while active and ((self.max_reduce_steps is None) or reduce_steps < self.max_reduce_steps):
			active = False
			reduce_steps += 1
			nonterm_idxes = []
			tree_embeddings = []
			root_embeddings = []
			for tm_idx in range(batch_size):
				tm = tree_managers[tm_idx]
				cur_nonterm_idxes = self.calc_dependency(tm)
				if len(cur_nonterm_idxes) == 0:
					continue
				for tree_idx in cur_nonterm_idxes:
					cur_tree = tm.get_tree(tree_idx)
					nonterm_idxes.append((tm_idx, tree_idx))
					tree_embeddings.append(cur_tree.state[0])
					root_embedding = tm.get_tree(cur_tree.dependency_parent).state[0]
					root_embeddings.append(root_embedding)
			if len(nonterm_idxes) == 0:
				break
			ac_logits = []
			pred_rewards = []
			for st in range(0, len(nonterm_idxes), self.batch_size):
				cur_tree_embeddings = tree_embeddings[st: st + self.batch_size]
				cur_tree_embeddings = torch.cat(cur_tree_embeddings, 0)
				cur_root_embeddings = root_embeddings[st: st + self.batch_size]
				cur_root_embeddings = torch.cat(cur_root_embeddings, 0)
				cur_inputs = torch.cat([cur_root_embeddings, cur_tree_embeddings], 1)
				cur_ac_logits = self.policy(cur_inputs)
				cur_pred_rewards = self.value_estimator(cur_inputs)
				ac_logits.append(cur_ac_logits)
				pred_rewards.append(cur_pred_rewards)
			ac_logits = torch.cat(ac_logits, 0)
			ac_logprobs = nn.LogSoftmax()(ac_logits)
			ac_probs = nn.Softmax()(ac_logits)
			pred_rewards = torch.cat(pred_rewards, 0)
			candidate_rewrite_pos = [[] for _ in range(batch_size)]
			for idx, (tm_idx, tree_idx) in enumerate(nonterm_idxes):
				candidate_rewrite_pos[tm_idx].append((pred_rewards[idx].data[0], pred_rewards[idx], ac_probs[idx], tree_idx, idx))

			update_tree_idxes = [[] for _ in range(batch_size)]
			candidate_tree_managers, candidate_update_tree_idxes, candidate_rewrite_rec, candidate_pending_actions, cur_extra_reward_rec, cur_extra_action_rec = self.batch_rewrite(tree_managers, ac_logprobs, trace_rec, expr_rec, candidate_rewrite_pos, pending_actions, eval_flag, max_search_pos=1, reward_thres=self.reward_thres)
			for tm_idx in range(batch_size):
				cur_candidate_tree_managers = candidate_tree_managers[tm_idx]
				cur_candidate_update_tree_idxes = candidate_update_tree_idxes[tm_idx]
				cur_candidate_rewrite_rec = candidate_rewrite_rec[tm_idx]
				cur_candidate_pending_actions = candidate_pending_actions[tm_idx]
				if len(cur_candidate_tree_managers) > 0:
					active = True
					cur_tree_manager = cur_candidate_tree_managers[0]
					cur_update_tree_idxes = cur_candidate_update_tree_idxes[0]
					cur_rewrite_rec = cur_candidate_rewrite_rec[0]
					cur_pending_actions = cur_candidate_pending_actions[0]
					tree_managers[tm_idx] = cur_tree_manager
					update_tree_idxes[tm_idx] = cur_update_tree_idxes
					ac_logprob, pred_reward, cur_pred_reward_tensor, rewrite_pos, init_expr, applied_op = cur_rewrite_rec
					trace_rec[tm_idx][-1] = (expr_rec[tm_idx][-1], init_expr, applied_op)
					rewrite_rec[tm_idx].append(cur_rewrite_rec)
					pending_actions[tm_idx] = cur_pending_actions
					if cur_pending_actions[0] < 0:
						ac_logprob_st, pred_reward_st, cur_pred_reward_tensor_st, rewrite_pos_st, init_expr_st, applied_op_st = rewrite_rec[tm_idx][cur_pending_actions[0]]
						expr_st, init_expr_st, applied_op_st = trace_rec[tm_idx][cur_pending_actions[0]]
						rewrite_rec[tm_idx][cur_pending_actions[0]] = (ac_logprob_st, pred_reward_st, cur_pred_reward_tensor_st, rewrite_pos_st, init_expr_st, cur_pending_actions[1])
						trace_rec[tm_idx][cur_pending_actions[0]] = (expr_st, init_expr_st, cur_pending_actions[1])
						if cur_pending_actions[0] < -1:
							rewrite_rec[tm_idx] = rewrite_rec[tm_idx][:cur_pending_actions[0] + 1]
							trace_rec[tm_idx] = trace_rec[tm_idx][:cur_pending_actions[0] + 1]
							expr_rec[tm_idx] = expr_rec[tm_idx][:cur_pending_actions[0] + 1]
							tm_rec[tm_idx] = tm_rec[tm_idx][:cur_pending_actions[0] + 1]
						pending_actions[tm_idx] = []
			extra_reward_rec = extra_reward_rec + cur_extra_reward_rec
			extra_action_rec = extra_action_rec + cur_extra_action_rec
			if not active:
				break
			updated_tm = self.input_encoder.update_embedding(tree_managers, update_tree_idxes, eval_flag)
			for i in range(batch_size):
				tree_managers[i] = updated_tm[i]
				if len(update_tree_idxes[i]) > 0:
					expr_rec[i].append(str(updated_tm[i]))
					trace_rec[i].append((expr_rec[i][-1], '', -1))
					tm_rec[i].append(updated_tm[i])

		total_policy_loss = data_utils.np_to_tensor(np.zeros(1), 'float', self.cuda_flag)
		total_value_loss = data_utils.np_to_tensor(np.zeros(1), 'float', self.cuda_flag)
		
		pred_actions_rec = []
		pred_actions_logprob_rec = []
		pred_value_rec = []
		value_target_rec = []
		pred_dependency_rec = []
		dependency_target_rec = []
		total_reward = 0
		for tm_idx, cur_trace_rec in enumerate(trace_rec):
			pred_trace_len = []
			for i, (expr, init_expr, op_idx) in enumerate(cur_trace_rec):
				pred_trace_len.append(len(expr))
			max_reward = 0
			for idx, (ac_logprob, pred_reward, cur_pred_reward_tensor, rewrite_pos, init_expr, applied_op) in enumerate(rewrite_rec[tm_idx]):
				cur_reward = pred_trace_len[idx] - pred_trace_len[idx + 1] - 1
				max_reward = max(max_reward, pred_trace_len[0] - pred_trace_len[idx + 1])
				decay_coef = 1.0
				num_rollout_steps = len(pred_trace_len) - idx - 1
				for i in range(idx + 1, idx + 1 + num_rollout_steps):
					cur_reward = max(decay_coef * (min(pred_trace_len[idx] - pred_trace_len[i] - (i - idx), len(init_expr))), cur_reward)
					decay_coef *= self.gamma
				cur_reward = cur_reward * 1.0 / len(init_expr)
				cur_reward_tensor = data_utils.np_to_tensor(np.array([cur_reward], dtype=np.float32), 'float', self.cuda_flag, eval_flag)
				if ac_logprob.data.cpu().numpy()[0] > log_eps or cur_reward - pred_reward > 0:
					ac_mask = np.zeros(self.num_actions)
					ac_mask[applied_op] = cur_reward - pred_reward
					ac_mask = data_utils.np_to_tensor(ac_mask, 'float', self.cuda_flag, eval_flag)
					ac_mask = ac_mask.unsqueeze(0)
					pred_actions_rec.append(ac_mask)
					pred_actions_logprob_rec.append(ac_logprob.unsqueeze(0))
				pred_value_rec.append(cur_pred_reward_tensor)
				value_target_rec.append(cur_reward_tensor)
			total_reward += max_reward

		for cur_pred_reward in extra_reward_rec:
			pred_value_rec.append(cur_pred_reward)
			value_target = data_utils.np_to_tensor(np.zeros(1), 'float', self.cuda_flag)
			value_target_rec.append(value_target)

		if len(pred_actions_rec) > 0:
			pred_actions_rec = torch.cat(pred_actions_rec, 0)
			pred_actions_logprob_rec = torch.cat(pred_actions_logprob_rec, 0)
			pred_value_rec = torch.cat(pred_value_rec, 0)
			value_target_rec = torch.cat(value_target_rec, 0)
			pred_value_rec = pred_value_rec.unsqueeze(1)
			value_target_rec = value_target_rec.unsqueeze(1)
			total_policy_loss = -torch.sum(pred_actions_logprob_rec * pred_actions_rec)
			total_value_loss = F.smooth_l1_loss(pred_value_rec, value_target_rec, size_average=False)
		total_policy_loss /= batch_size
		total_value_loss /= batch_size
		total_loss = total_policy_loss + total_value_loss * self.value_loss_coef
		total_reward = total_reward * 1.0 / batch_size
		return total_loss, total_reward, trace_rec, tm_rec
