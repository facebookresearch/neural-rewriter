# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from torch.autograd import Variable
from .utils import *


class Node(object):
	"""
	Class to represent each node in a directed acyclic graph (DAG). It can be used for job scheduling.
	"""
	def __init__(self, state=None):
		self.children = []
		self.parents = []
		if state is None:
			self.state = None
		else:
			self.state = state[0].clone(), state[1].clone()


	def add_child(self, child):
		self.children += [child]


	def add_parent(self, parent):
		self.parents += [parent]


	def del_child(self, child):
		self.children.remove(child)


	def del_parent(self, parent):
		self.parents.remove(parent)


class JobNode(Node):
	"""
	Class to represent each job node for job scheduling.
	"""
	def __init__(self, resource_size, start_time, job_len, schedule_time=None, embedding=None, state=None):
		super(JobNode, self).__init__(state)
		self.resource_size = resource_size
		self.st_time = start_time
		self.job_len = job_len
		if schedule_time is None:
			self.schedule_time = None
			self.ed_time = None
			self.completion_time = None
			self.slow_down = None
		else:
			self.update_schedule_time(schedule_time)
		if embedding is None:
			self.embedding = None
		else:
			self.embedding = embedding.copy()


	def update_schedule_time(self, t):
		self.schedule_time = t
		self.ed_time = self.schedule_time + self.job_len
		self.completion_time = self.ed_time - self.st_time
		if self.job_len > 0:
			self.slow_down = self.completion_time * 1.0 / self.job_len
		else:
			self.slow_down = 0


	def update_embedding(self, embedding):
		self.embedding = embedding


class DagManager(object):
	"""
	Class to maintain the state for problems with DAG-structured data. Can be used for job scheduling.
	"""
	def __init__(self):
		self.nodes = []
		self.num_nodes = 0
		self.root = 0


	def get_node(self, idx):
		return self.nodes[idx]


	def add_edge(self, x, y):
		self.nodes[x].add_child(y)
		self.nodes[y].add_parent(x)


	def del_edge(self, x, y):
		self.nodes[x].del_child(y)
		self.nodes[y].del_parent(x)


	def clear_states(self):
		for idx in range(self.num_nodes):
			self.nodes[idx].state = None
			self.nodes[idx].rev_state = None


class JobScheduleManager(DagManager):
	"""
	Class to maintain the state for job scheduling problems.
	"""
	def __init__(self, num_res, max_time_horizon, max_job_len, max_resource_size):
		super(JobScheduleManager, self).__init__()
		self.num_res = num_res
		self.max_time_horizon = max_time_horizon
		self.max_job_len = max_job_len
		self.max_resource_size = max_resource_size
		self.max_schedule_time = 0
		self.max_ed_time = 0
		self.embedding_size = (self.max_job_len + 1) * self.num_res + 1
		self.nodes.append(JobNode(resource_size=0, start_time=0, job_len=0, schedule_time=0, embedding=[0.0 for _ in range(self.embedding_size)]))
		self.num_jobs = 0
		self.resource_map = np.zeros((self.max_time_horizon, self.num_res))
		self.schedule = [[] for _ in range(self.max_time_horizon)]
		self.terminate = [[] for _ in range(self.max_time_horizon)]


	def clone(self):
		res = JobScheduleManager(self.num_res, self.max_time_horizon, self.max_job_len, self.max_resource_size)
		res.root = self.root
		res.nodes = []
		for i, node in enumerate(self.nodes):
			res.nodes.append(JobNode(resource_size=node.resource_size, start_time=node.st_time, job_len=node.job_len, schedule_time=node.schedule_time, embedding=node.embedding, state=node.state))
			if i != 0:
				res.schedule[node.schedule_time].append(i)
				res.terminate[node.ed_time].append(i)
			for child in node.children:
				res.nodes[i].add_child(child)
			for parent in node.parents:
				res.nodes[i].add_parent(parent)
		res.num_nodes = self.num_nodes
		res.num_jobs = self.num_jobs
		res.resource_map = self.resource_map.copy()
		res.avg_slow_down = self.avg_slow_down
		res.avg_completion_time = self.avg_completion_time
		res.max_schedule_time = self.max_schedule_time
		res.max_ed_time = self.max_ed_time
		return res


	def add_job(self, node_idx, cur_time):
		job = self.nodes[node_idx]
		ed_time = cur_time + job.job_len
		for t in range(cur_time, ed_time):
			self.resource_map[t] += job.resource_size
		if job.schedule_time == cur_time:
			return
		if job.schedule_time is not None:
			self.schedule[job.schedule_time].remove(node_idx)
			self.terminate[job.ed_time].remove(node_idx)
		self.schedule[cur_time].append(node_idx)
		self.terminate[ed_time].append(node_idx)
		self.nodes[node_idx].update_schedule_time(cur_time)


	def update_embedding(self, node_idx):
		job = self.nodes[node_idx]
		embedding = []
		embedding.append(job.slow_down)
		embedding += [job.resource_size[i] * 1.0 / self.max_resource_size for i in range(self.num_res)]
		for t in range(job.schedule_time, job.ed_time):
			embedding += [self.resource_map[t][i] * 1.0 / self.max_resource_size for i in range(self.num_res)]
		if len(embedding) < self.embedding_size:
			embedding += [0.0 for _ in range(self.embedding_size - len(embedding))]
		self.nodes[node_idx].update_embedding(embedding)


	def update_stat(self):
		self.avg_completion_time = 0.0
		self.avg_slow_down = 0.0
		self.max_schedule_time = 0
		self.max_ed_time = 0
		for node in self.nodes:
			self.avg_completion_time += node.completion_time
			self.avg_slow_down += node.slow_down
			self.max_schedule_time = max(self.max_schedule_time, node.schedule_time)
			self.max_ed_time = max(self.max_ed_time, node.ed_time)
		self.avg_slow_down = self.avg_slow_down * 1.0 /  self.num_jobs
		self.avg_completion_time = self.avg_completion_time * 1.0 / self.num_jobs


	def get_parent_idxes(self, st, job_horizon):
		res = [st]
		st_job = self.get_node(st)
		idx = 0
		scheduled_time = []
		scheduled_time.append(st_job.ed_time)
		while len(res) < job_horizon + 1 and idx < len(res):
			cur_job = self.get_node(res[idx])
			for parent in cur_job.parents:
				if not (parent in res):
					res.append(parent)
			idx += 1
		return res[1:job_horizon + 1]


	def get_children_idxes(self, st, job_horizon):
		res = [st]
		st_job = self.get_node(st)
		idx = 0
		while len(res) < job_horizon + 1 and idx < len(res):
			cur_job = self.get_node(res[idx])
			for child in cur_job.children:
				if not (child in res):
					res.append(child)
			idx += 1
		return res[1:job_horizon + 1]


	def runnable(self, cur_job, schedule_time):
		for t in range(cur_job.job_len):
			for j in range(self.num_res):
				if self.resource_map[t + schedule_time][j] + cur_job.resource_size[j] > self.max_resource_size:
					return False
		return True


	def calc_min_schedule_time(self, min_schedule_time, cur_job_idx):
		cur_job = self.get_node(cur_job_idx)
		cur_time_horizon = min_schedule_time
		if self.runnable(cur_job, min_schedule_time):
			return min_schedule_time
		new_schedule_time = self.max_time_horizon
		for i, node in enumerate(self.nodes):
			if i == cur_job_idx:
				continue
			if node.ed_time is None:
				continue
			if node.ed_time <= min_schedule_time or node.ed_time >= new_schedule_time:
				continue
			if self.runnable(cur_job, node.ed_time):
				new_schedule_time = node.ed_time
		return new_schedule_time