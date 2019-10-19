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
import copy

from ..data_utils import data_utils
from ..data_utils import Dag

class jspRewriter(object):
	"""
	Rewriter for job scheduling.
	"""
	def move(self, dm, cur_idx, neighbor_idx):
		cur_job = dm.get_node(cur_idx)
		neighbor_job = dm.get_node(neighbor_idx)
		new_schedule_time = max(neighbor_job.ed_time, cur_job.st_time)
		old_schedule_time = cur_job.schedule_time
		if new_schedule_time == old_schedule_time:
			return dm, []
		min_stop_time = min(new_schedule_time, neighbor_job.ed_time)
		res = dm.clone()
		for t in range(cur_job.schedule_time, cur_job.ed_time):
			res.resource_map[t] -= cur_job.resource_size
		res.add_job(cur_idx, new_schedule_time)
		res.max_schedule_time = max(res.max_schedule_time, new_schedule_time)
		res.max_ed_time = max(res.max_ed_time, new_schedule_time + cur_job.job_len)

		updated_schedule_time = min(min_stop_time, old_schedule_time)
		scheduled_node_idxes = [cur_idx]

		time_step = updated_schedule_time
		while time_step <= res.max_schedule_time:
			temp_old_schedule = res.schedule[time_step].copy()
			for temp_job_idx in temp_old_schedule:
				temp_job = res.get_node(temp_job_idx)
				temp_old_schedule_time = temp_job.schedule_time
				new_schedule_time = temp_job.st_time
				for t in range(temp_job.schedule_time, temp_job.ed_time):
					res.resource_map[t] -= temp_job.resource_size
				new_schedule_time = res.calc_min_schedule_time(new_schedule_time, temp_job_idx)
				res.add_job(temp_job_idx, new_schedule_time)
				if not temp_job_idx in scheduled_node_idxes:
					scheduled_node_idxes.append(temp_job_idx)
				res.max_schedule_time = max(res.max_schedule_time, new_schedule_time)
			time_step += 1

		for idx in scheduled_node_idxes:
			job = res.get_node(idx)
			old_parents = job.parents.copy()
			old_children = job.children.copy()
			for parent_idx in old_parents:
				res.del_edge(parent_idx, idx)
			if job.schedule_time == job.st_time:
				res.add_edge(res.root, idx)
			else:
				schedule_idx = res.schedule[job.schedule_time].index(idx)
				if schedule_idx == 0:
					res.add_edge(res.terminate[job.schedule_time][-1], idx)
				else:
					res.add_edge(res.schedule[job.schedule_time][schedule_idx - 1], idx)
			res.update_embedding(idx)
			res.nodes[idx].parents.sort()
			dm.nodes[idx].parents.sort()
			if res.nodes[idx].parents != dm.nodes[idx].parents or res.nodes[idx].embedding != dm.nodes[idx].embedding:
				updated_schedule_time = min(updated_schedule_time, res.nodes[idx].schedule_time)
		res.update_stat()

		updated_node_idxes = []
		for time_step in range(updated_schedule_time, res.max_schedule_time + 1):
			for idx in res.schedule[time_step]:
				updated_node_idxes.append(idx)
		return res, updated_node_idxes
		