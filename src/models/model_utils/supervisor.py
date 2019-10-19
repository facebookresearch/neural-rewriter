# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import argparse
import sys
import os
import torch
import re
import json
import time
import torch.multiprocessing as mp

from torch.nn.utils import clip_grad_norm

from ..data_utils import data_utils
from ..data_utils.parser import *

CKPT_PATTERN = re.compile('^ckpt-(\d+)$')


class Supervisor(object):
	"""
	The base class to manage the high-level model execution processes. The concrete classes for different applications are derived from it.
	"""
	def __init__(self, model, args):
		self.processes = args.processes
		self.model = model
		self.keep_last_n = args.keep_last_n
		self.dropout_rate = args.dropout_rate
		self.global_step = 0
		self.batch_size = args.batch_size
		self.model_dir = args.model_dir


	def load_pretrained(self, load_model):
		print("Read model parameters from %s." % load_model)
		checkpoint = torch.load(load_model)
		self.model.load_state_dict(checkpoint)


	def save_model(self):
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		global_step_padded = format(self.global_step, '08d')
		ckpt_name = 'ckpt-' + global_step_padded
		path = os.path.join(self.model_dir, ckpt_name)
		ckpt = self.model.state_dict()
		torch.save(ckpt, path)

		if self.keep_last_n is not None:
			ckpts = []
			for file_name in os.listdir(self.model_dir):
				matched_name = CKPT_PATTERN.match(file_name)
				if matched_name is None or matched_name == ckpt_name:
					continue
				step = int(matched_name.group(1))
				ckpts.append((step, file_name))
			if len(ckpts) > self.keep_last_n:
				ckpts.sort()
				os.unlink(os.path.join(self.model_dir, ckpts[0][1]))


class HalideSupervisor(Supervisor):
	"""
	Management class for expression simplification.
	"""
	def __init__(self, model, args, term_vocab, term_vocab_list, op_vocab, op_vocab_list):
		super(HalideSupervisor, self).__init__(model, args)
		self.DataProcessor = data_utils.HalideDataProcessor()
		self.parser = HalideParser()
		self.term_vocab = term_vocab
		self.term_vocab_list = term_vocab_list
		self.op_vocab = op_vocab
		self.op_vocab_list = op_vocab_list


	def train(self, batch_data):
		self.model.dropout_rate = self.dropout_rate
		self.model.optimizer.zero_grad()
		avg_loss, avg_reward, trace_rec, tm_rec = self.model(batch_data)
		self.global_step += 1
		if avg_reward != 0:
			avg_loss.backward()
			self.model.train()
		return avg_loss.item(), avg_reward


	def batch_eval(self, eval_data, output_trace_flag, output_trace_option, process_idx):
		cum_loss = 0
		cum_expr_reward = 0
		cum_gt_expr_reward = 0
		cum_tree_reward = 0
		cum_gt_tree_reward = 0
		data_size = len(eval_data)
		trace_rec = []
		for batch_idx in range(0, data_size, self.batch_size):
			batch_data = self.DataProcessor.get_batch(eval_data, self.batch_size, batch_idx)
			cur_avg_loss, cur_avg_expr_reward, cur_trace_rec, cur_tm_rec = self.model(batch_data, eval_flag=True)
			cum_loss += cur_avg_loss.item() * len(batch_data)
			cum_expr_reward += cur_avg_expr_reward * len(batch_data)
			cur_gt_expr_reward = 0
			cur_gt_tree_reward = 0
			for idx, (trace, tm) in enumerate(batch_data):
				gt = len(trace[0]) - len(trace[-1])
				cur_gt_expr_reward += gt
				num_nodes_0 = tm.num_trees
				final_tm = self.parser.parse(trace[-1])
				num_nodes_1 = final_tm.num_trees
				cur_gt_tree_reward += num_nodes_0 - num_nodes_1
				init_expr = cur_trace_rec[idx][0][0]
				pred_expr = cur_trace_rec[idx][-1][0]
				pred_reward = len(init_expr) - len(pred_expr)
				if output_trace_flag == 'complete' or output_trace_flag == 'fail' and len(trace[-1]) < len(pred_expr) \
				or output_trace_flag == 'succeed' and len(pred_expr) < len(trace[-1]):
					if output_trace_option != 'pred':
						self.DataProcessor.print_gt_trace(trace)
					self.DataProcessor.print_pred_trace(cur_trace_rec[idx])
					print('end of a sample')
					print('')

			cur_cum_tree_reward = 0
			for tm_rec in cur_tm_rec:
				cur_tree_reward = 0
				num_nodes_0 = tm_rec[0].num_trees
				for final_tm in tm_rec[1:]:
					num_nodes_1 = final_tm.num_valid_nodes()
					if num_nodes_0 - num_nodes_1 > cur_tree_reward:
						cur_tree_reward = num_nodes_0 - num_nodes_1
				cur_cum_tree_reward += cur_tree_reward

			trace_rec = trace_rec + cur_trace_rec
			cum_tree_reward += cur_cum_tree_reward
			cum_gt_expr_reward += cur_gt_expr_reward
			cum_gt_tree_reward += cur_gt_tree_reward
			print('process start idx: %d batch idx: %d pred expr reward: %.4f pred tree reward: %.4f gt expr reward: %.4f gt tree reward: %.4f' % \
				(process_idx, batch_idx, cur_avg_expr_reward, cur_cum_tree_reward * 1.0 / len(batch_data), cur_gt_expr_reward * 1.0 / len(batch_data), cur_gt_tree_reward * 1.0 / len(batch_data)))
		return cum_loss, cum_expr_reward, cum_tree_reward, cum_gt_expr_reward, cum_gt_tree_reward, trace_rec


	def eval(self, data, output_trace_flag, output_trace_option, output_trace_file, max_eval_size=None):
		data_size = len(data)
		if max_eval_size is not None:
			data_size = min(data_size, max_eval_size)
		eval_data = data[:data_size]
		if self.processes == 1:
			cum_loss, cum_expr_reward, cum_tree_reward, cum_gt_expr_reward, cum_gt_tree_reward, trace_rec = self.batch_eval(eval_data, output_trace_flag, output_trace_option, 0)
		else:
			cum_loss = 0
			cum_expr_reward = 0
			cum_tree_reward = 0
			cum_gt_expr_reward = 0
			cum_gt_tree_reward = 0
			trace_rec = []
			try:
				mp.set_start_method('spawn')
			except RuntimeError:
				pass
			pool = mp.Pool(processes=self.processes)
			res = []
			batch_per_process = data_size // self.processes
			if data_size % self.processes > 0:
				batch_per_process += 1
			for st in range(0, data_size, batch_per_process):
				res += [pool.apply_async(self.batch_eval, (eval_data[st: st + batch_per_process], output_trace_flag, output_trace_option, st))]
			for i in range(len(res)):
				cur_cum_loss, cur_cum_expr_reward, cur_cum_tree_reward, cur_cum_gt_expr_reward, cur_cum_gt_tree_reward, cur_trace_rec = res[i].get()
				cum_loss += cur_cum_loss
				cum_expr_reward += cur_cum_expr_reward
				cum_tree_reward += cur_cum_tree_reward
				cum_gt_expr_reward += cur_cum_gt_expr_reward
				cum_gt_tree_reward += cur_cum_gt_tree_reward
				trace_rec = trace_rec + cur_trace_rec

		avg_loss = cum_loss / data_size
		avg_expr_reward = cum_expr_reward * 1.0 / data_size
		avg_tree_reward = cum_tree_reward  * 1.0 / data_size
		gt_expr_reward = cum_gt_expr_reward * 1.0 / data_size
		gt_tree_reward = cum_gt_tree_reward * 1.0 / data_size
		print('average: pred expr reward: %.4f pred tree reward: %.4f gt expr reward: %.4f gt tree reward: %.4f' % (avg_expr_reward, avg_tree_reward, gt_expr_reward, gt_tree_reward))
		if output_trace_file is not None:
			fout_res = open(output_trace_file, 'w')
			json.dump(trace_rec, fout_res)
		return avg_loss, avg_expr_reward


class jspSupervisor(Supervisor):
	"""
	Management class for job scheduling.
	"""
	def __init__(self, model, args):
		super(jspSupervisor, self).__init__(model, args)
		self.DataProcessor = data_utils.jspDataProcessor(args)

		
	def train(self, batch_data):
		self.model.dropout_rate = self.dropout_rate
		self.model.optimizer.zero_grad()
		avg_loss, avg_reward, avg_completion_time, dm_rec = self.model(batch_data)
		self.global_step += 1
		if avg_reward != 0:
			avg_loss.backward()
			self.model.train()
		return avg_loss.item(), avg_reward


	def batch_eval(self, eval_data, output_trace_flag, process_idx):
		cum_loss = 0
		cum_reward = 0
		cum_completion_time = 0
		cum_gt_reward = 0
		data_size = len(eval_data)

		for batch_idx in range(0, data_size, self.batch_size):
			batch_data = self.DataProcessor.get_batch(eval_data, self.batch_size, batch_idx)
			cur_avg_loss, cur_avg_reward, cur_avg_completion_time, dm_rec = self.model(batch_data, eval_flag=True)
			cum_loss += cur_avg_loss.item() * len(batch_data)
			cum_reward += cur_avg_reward * len(batch_data)
			cum_completion_time += cur_avg_completion_time * len(batch_data)
			if output_trace_flag == 'complete':
				for cur_dm_rec in dm_rec:
					for i, job in enumerate(cur_dm_rec[-1].nodes[1:]):
						print(i)
						print(job.st_time)
						print(job.job_len)
						print(job.resource_size)
						print(job.schedule_time)
						print('')
			print('process start idx: %d batch idx: %d pred reward: %.4f pred completion time: %.4f' \
				% (process_idx, batch_idx, cur_avg_reward, cur_avg_completion_time))
		return cum_loss, cum_reward, cum_completion_time


	def eval(self, data, output_trace_flag, max_eval_size=None):
		data_size = len(data)
		if max_eval_size is not None:
			data_size = min(data_size, max_eval_size)
		eval_data = data[:data_size]
		if self.processes == 1:
			cum_loss, cum_reward, cum_completion_time = self.batch_eval(eval_data, output_trace_flag, 0)
		else:
			cum_loss = 0
			cum_reward = 0
			cum_completion_time = 0
			try:
				mp.set_start_method('spawn')
			except RuntimeError:
				pass
			pool = mp.Pool(processes=self.processes)
			res = []
			batch_per_process = data_size // self.processes
			if data_size % batch_per_process > 0:
				batch_per_process += 1
			for st in range(0, data_size, batch_per_process):
				res += [pool.apply_async(self.batch_eval, (eval_data[st: st + batch_per_process], output_trace_flag, st))]
			for i in range(len(res)):
				cur_cum_loss, cur_cum_reward, cur_cum_completion_time = res[i].get()
				cum_loss += cur_cum_loss
				cum_reward += cur_cum_reward
				cum_completion_time  += cur_cum_completion_time

		avg_loss = cum_loss / data_size
		avg_reward = cum_reward / data_size
		avg_completion_time = cum_completion_time * 1.0 / data_size
		print('average pred reward: %.4f' % avg_reward)
		print('average completion time: %.4f' % avg_completion_time)
		return avg_loss, avg_reward


class vrpSupervisor(Supervisor):
	"""
	Management class for vehicle routing.
	"""
	def __init__(self, model, args):
		super(vrpSupervisor, self).__init__(model, args)
		self.DataProcessor = data_utils.vrpDataProcessor()


	def train(self, batch_data):
		self.model.dropout_rate = self.dropout_rate
		self.model.optimizer.zero_grad()
		avg_loss, avg_reward, dm_rec = self.model(batch_data)
		self.global_step += 1
		if type(avg_loss) != float:
			avg_loss.backward()
			self.model.train()
		return avg_loss.item(), avg_reward


	def batch_eval(self, eval_data, output_trace_flag, process_idx):
		cum_loss = 0
		cum_reward = 0
		data_size = len(eval_data)

		for batch_idx in range(0, data_size, self.batch_size):
			batch_data = self.DataProcessor.get_batch(eval_data, self.batch_size, batch_idx)
			cur_avg_loss, cur_avg_reward, dm_rec = self.model(batch_data, eval_flag=True)
			cum_loss += cur_avg_loss.item() * len(batch_data)
			cum_reward += cur_avg_reward * len(batch_data)
			if output_trace_flag == 'complete':
				for cur_dm_rec in dm_rec:
					for i in range(len(cur_dm_rec)):
						print('step ' + str(i))
						dm = cur_dm_rec[i]
						print(dm.tot_dis[-1])
						for j in range(len(dm.vehicle_state)):
							cur_pos, cur_capacity = dm.vehicle_state[j]
							cur_node = dm.get_node(cur_pos)
							print(cur_node.x, cur_node.y, cur_node.demand, cur_capacity, dm.tot_dis[j])
						print('')
			print('process start idx: %d batch idx: %d pred reward: %.4f' \
				% (process_idx, batch_idx, cur_avg_reward))
		return cum_loss, cum_reward


	def eval(self, data, output_trace_flag, max_eval_size=None):
		data_size = len(data)
		if max_eval_size is not None:
			data_size = min(data_size, max_eval_size)
		eval_data = data[:data_size]
		if self.processes == 1:
			cum_loss, cum_reward = self.batch_eval(eval_data, output_trace_flag, 0)
		else:
			cum_loss = 0
			cum_reward = 0
			try:
				mp.set_start_method('spawn')
			except RuntimeError:
				pass
			pool = mp.Pool(processes=self.processes)
			res = []
			batch_per_process = data_size // self.processes
			if data_size % batch_per_process > 0:
				batch_per_process += 1
			for st in range(0, data_size, batch_per_process):
				res += [pool.apply_async(self.batch_eval, (eval_data[st: st + batch_per_process], output_trace_flag, st))]
			for i in range(len(res)):
				cur_cum_loss, cur_cum_reward = res[i].get()
				cum_loss += cur_cum_loss
				cum_reward += cur_cum_reward

		avg_loss = cum_loss / data_size
		avg_reward = cum_reward / data_size
		print('average pred reward: %.4f' % avg_reward)
		return avg_loss, avg_reward