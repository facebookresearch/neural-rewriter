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
from .modules import jspInputEncoder, mlp
from .rewriter import jspRewriter
from .BaseModel import BaseModel

eps = 1e-3
log_eps = np.log(eps)


class jspModel(BaseModel):
	"""
	Model for job scheduling.
	"""
	def __init__(self, args):
		super(jspModel, self).__init__(args)
		self.input_format = args.input_format
		self.max_resource_size = args.max_resource_size
		self.job_horizon = args.job_horizon
		self.num_res = args.num_res
		self.max_time_horizon = args.max_time_horizon
		self.max_job_len = args.max_job_len
		self.embedding_size = args.embedding_size
		self.num_actions = self.job_horizon * 2
		self.reward_thres = -0.01
		if self.input_format == 'seq':
			self.input_encoder = jspInputEncoder.SeqLSTM(args)
		else:
			self.input_encoder = jspInputEncoder.DagLSTM(args)
		self.policy_embedding = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * 2, self.MLP_hidden_size, self.LSTM_hidden_size, self.cuda_flag, self.dropout_rate)
		self.policy = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size * (self.job_horizon * 2), self.MLP_hidden_size, self.num_actions, self.cuda_flag, self.dropout_rate)
		self.value_estimator = mlp.MLPModel(self.num_MLP_layers, self.LSTM_hidden_size, self.MLP_hidden_size, 1, self.cuda_flag, self.dropout_rate)
		self.rewriter = jspRewriter()

		if args.optimizer == 'adam':
			self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
		elif args.optimizer == 'sgd':
			self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
		elif args.optimizer == 'rmsprop':
			self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
		else:
			raise ValueError('optimizer undefined: ', args.optimizer)


	def rewrite(self, dm, trace_rec, candidate_rewrite_pos, eval_flag, max_search_pos, reward_thres=None):
		
		candidate_rewrite_pos.sort(reverse=True, key=operator.itemgetter(0))
		if not eval_flag:
			sample_exp_reward_tensor = []
			for idx, (cur_pred_reward, cur_pred_reward_tensor, rewrite_pos) in enumerate(candidate_rewrite_pos):
				sample_exp_reward_tensor.append(cur_pred_reward_tensor)
			sample_exp_reward_tensor = torch.cat(sample_exp_reward_tensor, 0)
			sample_exp_reward_tensor = torch.exp(sample_exp_reward_tensor * 10)
			sample_exp_reward = sample_exp_reward_tensor.data.cpu().numpy()

		candidate_dag_managers = []
		candidate_update_node_idxes = []
		candidate_rewrite_rec = []
		extra_reward_rec = []
		
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

		for idx, (pred_reward, cur_pred_reward_tensor, rewrite_pos) in enumerate(sample_rewrite_pos):
			if len(candidate_dag_managers) > 0 and idx >= max_search_pos:
				break
			if reward_thres is not None and pred_reward < reward_thres:
				if eval_flag:
					break
				elif np.random.random() > self.cont_prob:
					continue
			parent_idxes = dm.get_parent_idxes(rewrite_pos, self.job_horizon)
			children_idxes = dm.get_children_idxes(rewrite_pos, self.job_horizon)
			policy_embedding_inputs = []
			cur_input = dm.get_node(rewrite_pos).state[0]
			cur_inputs = []
			for i in parent_idxes:
				policy_embedding_inputs.append(dm.get_node(i).state[0])
				cur_inputs.append(cur_input.clone())
			while len(policy_embedding_inputs) < self.job_horizon:
				zero_state = Variable(torch.zeros(1, self.LSTM_hidden_size))
				if self.cuda_flag:
					zero_state = zero_state.cuda()
				policy_embedding_inputs.append(zero_state)
				cur_inputs.append(zero_state.clone())
			for i in children_idxes:
				policy_embedding_inputs.append(dm.get_node(i).state[0])
				cur_inputs.append(cur_input.clone())
			while len(policy_embedding_inputs) < self.job_horizon * 2:
				zero_state = Variable(torch.zeros(1, self.LSTM_hidden_size))
				if self.cuda_flag:
					zero_state = zero_state.cuda()
				policy_embedding_inputs.append(zero_state)
				cur_inputs.append(zero_state.clone())
			policy_embedding_inputs = torch.cat(policy_embedding_inputs, 0)
			cur_inputs = torch.cat(cur_inputs, 0)
			policy_embedding_inputs = torch.cat([cur_inputs, policy_embedding_inputs], 1)
			policy_inputs = self.policy_embedding(policy_embedding_inputs)
			policy_inputs = policy_inputs.view(1, self.LSTM_hidden_size * (self.job_horizon * 2))
			ac_logits = self.policy(policy_inputs)
			ac_logprobs = nn.LogSoftmax()(ac_logits)
			ac_probs = nn.Softmax()(ac_logits)
			ac_logits = ac_logits.squeeze(0)
			ac_logprobs = ac_logprobs.squeeze(0)
			ac_probs = ac_probs.squeeze(0)
			if eval_flag:
				_, candidate_acs = torch.sort(ac_logprobs, descending=True)
				candidate_acs = candidate_acs.data.cpu().numpy()
			else:
				candidate_acs_dist = Categorical(ac_probs)
				candidate_acs = candidate_acs_dist.sample(sample_shape=[ac_probs.size()[0]])
				#candidate_acs = torch.multinomial(ac_probs, ac_probs.size()[0])
				candidate_acs = candidate_acs.data.cpu().numpy()
				indexes = np.unique(candidate_acs, return_index=True)[1]
				candidate_acs = [candidate_acs[i] for i in sorted(indexes)]
			cur_active = False
			for i, op_idx in enumerate(candidate_acs):
				if op_idx < self.job_horizon:
					if op_idx >= len(parent_idxes):
						continue
					neighbor_idx = parent_idxes[op_idx]
				else:
					if op_idx - self.job_horizon >= len(children_idxes):
						continue
					neighbor_idx = children_idxes[op_idx - self.job_horizon]
				if (rewrite_pos, neighbor_idx) in trace_rec or (neighbor_idx, rewrite_pos) in trace_rec:
					continue
				new_dm, cur_update_node_idxes = self.rewriter.move(dm, rewrite_pos, neighbor_idx)
				if len(cur_update_node_idxes) == 0:
					continue
				candidate_update_node_idxes.append(cur_update_node_idxes)
				candidate_dag_managers.append(new_dm)
				candidate_rewrite_rec.append((ac_logprobs, pred_reward, cur_pred_reward_tensor, rewrite_pos, op_idx, neighbor_idx))
				cur_active = True
				if len(candidate_dag_managers) >= max_search_pos:
					break
			if not cur_active:
				extra_reward_rec.append(cur_pred_reward_tensor)
		return candidate_dag_managers, candidate_update_node_idxes, candidate_rewrite_rec, extra_reward_rec


	def batch_rewrite(self, dag_managers, trace_rec, candidate_rewrite_pos, eval_flag, max_search_pos, reward_thres):
		candidate_dag_managers = []
		candidate_update_node_idxes = []
		candidate_rewrite_rec = []
		extra_reward_rec = []
		for i in range(len(dag_managers)):
			cur_candidate_dag_managers, cur_candidate_update_node_idxes, cur_candidate_rewrite_rec, cur_extra_reward_rec = self.rewrite(dag_managers[i], trace_rec[i], candidate_rewrite_pos[i], eval_flag, max_search_pos, reward_thres)
			candidate_dag_managers.append(cur_candidate_dag_managers)
			candidate_update_node_idxes.append(cur_candidate_update_node_idxes)
			candidate_rewrite_rec.append(cur_candidate_rewrite_rec)
			extra_reward_rec = extra_reward_rec + cur_extra_reward_rec
		return candidate_dag_managers, candidate_update_node_idxes, candidate_rewrite_rec, extra_reward_rec


	def forward(self, batch_data, eval_flag=False):
		dag_managers = []
		batch_size = len(batch_data)
		for dm in batch_data:
			dag_managers.append(dm)
		dag_managers = self.input_encoder.calc_embedding(dag_managers, eval_flag)

		active = True
		reduce_steps = 0

		trace_rec = [[] for _ in range(batch_size)]
		rewrite_rec = [[] for _ in range(batch_size)]
		dm_rec = [[] for _ in range(batch_size)]
		extra_reward_rec = []
		
		for idx in range(batch_size):
			dm_rec[idx].append(dag_managers[idx])

		while active and ((self.max_reduce_steps is None) or reduce_steps < self.max_reduce_steps):
			active = False
			reduce_steps += 1
			node_idxes = []
			node_embeddings = []
			root_embeddings = []
			for dm_idx in range(batch_size):
				dm = dag_managers[dm_idx]
				root_embedding = dm.get_node(0).state[0]
				for i in range(1, dm.num_nodes):
					cur_node = dm.get_node(i)
					node_idxes.append((dm_idx, i))
					node_embeddings.append(cur_node.state[0])
					root_embeddings.append(root_embedding.clone())
			pred_rewards = []
			for st in range(0, len(node_idxes), self.batch_size):
				cur_node_embeddings = node_embeddings[st: st + self.batch_size]
				cur_node_embeddings = torch.cat(cur_node_embeddings, 0)
				cur_pred_rewards = self.value_estimator(cur_node_embeddings)
				pred_rewards.append(cur_pred_rewards)
			pred_rewards = torch.cat(pred_rewards, 0)
			candidate_rewrite_pos = [[] for _ in range(batch_size)]
			for idx, (dm_idx, node_idx) in enumerate(node_idxes):
				candidate_rewrite_pos[dm_idx].append((pred_rewards[idx].data[0], pred_rewards[idx], node_idx))

			update_node_idxes = [[] for _ in range(batch_size)]
			candidate_dag_managers, candidate_update_node_idxes, candidate_rewrite_rec, cur_extra_reward_rec = self.batch_rewrite(dag_managers, trace_rec, candidate_rewrite_pos, eval_flag, max_search_pos=1, reward_thres=self.reward_thres)
			for dm_idx in range(batch_size):
				cur_candidate_dag_managers = candidate_dag_managers[dm_idx]
				cur_candidate_update_node_idxes = candidate_update_node_idxes[dm_idx]
				cur_candidate_rewrite_rec = candidate_rewrite_rec[dm_idx]
				if len(cur_candidate_dag_managers) > 0:
					active = True
					cur_dag_manager = cur_candidate_dag_managers[0]
					cur_update_node_idxes = cur_candidate_update_node_idxes[0]
					cur_rewrite_rec = cur_candidate_rewrite_rec[0]
					dag_managers[dm_idx] = cur_dag_manager
					update_node_idxes[dm_idx] = cur_update_node_idxes
					ac_logprob, pred_reward, cur_pred_reward_tensor, rewrite_pos, applied_op, neighbor_idx = cur_rewrite_rec
					rewrite_rec[dm_idx].append(cur_rewrite_rec)
					trace_rec[dm_idx].append((rewrite_pos, neighbor_idx))
			if not active:
				break

			updated_dm = self.input_encoder.calc_embedding(dag_managers, eval_flag)

			for i in range(batch_size):
				dag_managers[i] = updated_dm[i]
				if len(update_node_idxes[i]) > 0:
					dm_rec[i].append(updated_dm[i])

		total_policy_loss = data_utils.np_to_tensor(np.zeros(1), 'float', self.cuda_flag)
		total_value_loss = data_utils.np_to_tensor(np.zeros(1), 'float', self.cuda_flag)
		
		pred_actions_rec = []
		pred_actions_logprob_rec = []
		pred_value_rec = []
		value_target_rec = []
		total_reward = 0
		total_completion_time = 0
		total_slow_down = 0
		for dm_idx, cur_dm_rec in enumerate(dm_rec):
			pred_avg_slow_down = []
			pred_avg_completion_time = []
			for dm in cur_dm_rec:
				pred_avg_slow_down.append(dm.avg_slow_down)
				pred_avg_completion_time.append(dm.avg_completion_time)
			min_slow_down = pred_avg_slow_down[0]
			min_completion_time = pred_avg_completion_time[0]
			best_reward = min_slow_down
			for idx, (ac_logprob, pred_reward, cur_pred_reward_tensor, rewrite_pos, applied_op, neighbor_idx) in enumerate(rewrite_rec[dm_idx]):
				cur_reward = pred_avg_slow_down[idx] - pred_avg_slow_down[idx + 1] - 0.01
				best_reward = min(best_reward, pred_avg_slow_down[idx + 1])
				min_slow_down = min(min_slow_down, pred_avg_slow_down[idx + 1])
				min_completion_time = min(min_completion_time, pred_avg_completion_time[idx + 1])

				if self.gamma > 0.0:
					decay_coef = 1.0
					num_rollout_steps = len(cur_dm_rec) - idx - 1
					for i in range(idx + 1, idx + 1 + num_rollout_steps):
						cur_reward = max(decay_coef * (pred_avg_slow_down[idx] - pred_avg_slow_down[i] - (i - idx) * 0.01), cur_reward)
						decay_coef *= self.gamma

				cur_reward = cur_reward * 1.0 / pred_avg_slow_down[0]
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
			total_reward += best_reward
			total_completion_time += min_completion_time
			total_slow_down += min_slow_down

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
		total_completion_time = total_completion_time * 1.0 / batch_size
		total_slow_down = total_slow_down * 1.0 / batch_size
		return total_loss, total_reward, total_completion_time, dm_rec
