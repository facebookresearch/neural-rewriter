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


class vrpRewriter(object):
	"""
	Rewriter for vehicle routing.
	"""
	def move(self, dm, cur_route_idx, neighbor_route_idx):
		min_update_idx = min(cur_route_idx, neighbor_route_idx)
		res = dm.clone()
		old_vehicle_state = res.vehicle_state[:]
		old_vehicle_state[cur_route_idx], old_vehicle_state[neighbor_route_idx] = old_vehicle_state[neighbor_route_idx], old_vehicle_state[cur_route_idx]
		if old_vehicle_state[neighbor_route_idx][0] == 0:
			del old_vehicle_state[neighbor_route_idx]
		res.vehicle_state = res.vehicle_state[:min_update_idx]
		res.route = res.route[:min_update_idx]
		res.tot_dis = res.tot_dis[:min_update_idx]
		cur_node_idx, cur_capacity = res.vehicle_state[-1]
		for t in range(min_update_idx, len(old_vehicle_state)):
			new_node_idx, new_capacity = old_vehicle_state[t]
			new_node = res.get_node(new_node_idx)
			if new_node_idx != 0 and cur_capacity < new_node.demand:
				res.add_route_node(0)
			res.add_route_node(new_node_idx)
			cur_capacity = res.vehicle_state[-1][1]
		return res
