# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import os
import sys
import argparse
import pyparsing as pyp
from .Seq import *
from .Tree import *
from .Dag import *
from .utils import *

'''
Halide Grammar:
<Var> ::= v0 | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 | v10 | v11 | v12
<Term> ::= <Var> | <Const>
<Expr> ::= <BinaryExpr> | <UnaryExpr> | <SelectExpr> | <Term>
<BinaryExpr> ::= ( <Expr> BinaryOp <Expr> )
<UnaryExpr> ::= ! <Expr>
<SelectExpr> ::= max(<Expr>, <Expr>) | min(<Expr>, <Expr>) | select(<Expr>, <Expr>, <Expr>)
<Expr> ::= <BinaryExpr> | <UnaryExpr> | <SelectExpr> | <Term>
'''

class HalideTokenizer(object):

	def __init__(self):
		self.vars = ['v']
		self.num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
		self.alg_ops = ['+', '-', '*', '/', '%']
		self.cmp_ops = ['<', '<=', '==']
		self.bool_ops = ['&&', '||']
		self.rel_ops = self.cmp_ops + self.bool_ops + ['!']
		self.ops = self.alg_ops + self.rel_ops
		self.keywords = ['max', 'min', 'select']
		self.chars = ['(', ')', ',']

	def preprocess(self, raw_inp):
		Var = pyp.Combine('v' + pyp.Word(pyp.nums))
		Const = pyp.Combine(pyp.Optional(pyp.Literal('-')) + pyp.Word(pyp.nums))
		Term = Var | Const
		BinaryOp = pyp.oneOf('+ - * / % < <= == && ||')
		UnaryOp = pyp.oneOf('!')
		Expr = pyp.Forward()
		BinaryExpr = pyp.Group(pyp.Literal('(') + Expr + BinaryOp + Expr + pyp.Literal(')'))
		UnaryExpr = pyp.Group(UnaryOp + Expr)
		SelectExpr = pyp.Group(pyp.oneOf('min max') + pyp.Literal('(') + Expr + pyp.Literal(',') + Expr + pyp.Literal(')')) \
			| pyp.Group(pyp.Literal('select')  + pyp.Literal('(') + Expr + pyp.Literal(',') + Expr + pyp.Literal(',') + Expr + pyp.Literal(')'))
		Expr << (BinaryExpr | UnaryExpr | SelectExpr | Term)
		return Expr.parseString(raw_inp)

class HalideParser(object):
	def __init__(self):
		self.tokenizer = HalideTokenizer()
		self.EOF = '<EOF>'

	def debug(self, msg):
		if self.is_debug:
			print(msg)

	def parse(self, inp, debug=False):
		self.is_debug = debug
		inp = self.tokenizer.preprocess(inp)
		tm = TreeManager()
		tm, root_idx = self.parseExpr(tm, inp, -1, 0)
		return tm

	def parseExpr(self, tm, inp, parent, depth):
		if type(inp) is str:
			e1 = tm.create_tree(inp, parent, depth)
			return tm, e1
		elif len(inp) == 1:
			return self.parseExpr(tm, inp[0], parent, depth)
		elif (inp[0] == '!'):
			op = inp[0]
			op_node_idx = tm.create_tree(op, parent, depth)
			tm, e1 = self.parseExpr(tm, inp[1], op_node_idx, depth + 1)
			tm.update_edge(op_node_idx, e1)
		elif inp[0] in self.tokenizer.keywords:
			op = inp[0]
			op_node_idx = tm.create_tree(op, parent, depth)
			for expr in inp[1:]:
				if expr in self.tokenizer.chars:
					continue
				tm, e = self.parseExpr(tm, expr, op_node_idx, depth + 1)
				tm.update_edge(op_node_idx, e)
		else:
			op = inp[2]
			op_node_idx = tm.create_tree(op, parent, depth)
			tm, e1 = self.parseExpr(tm, inp[1], op_node_idx, depth + 1)
			tm, e2 = self.parseExpr(tm, inp[3], op_node_idx, depth + 1)
			tm.update_edge(op_node_idx, e1)
			tm.update_edge(op_node_idx, e2)
		return tm, op_node_idx

class jspDependencyParser(object):
	def __init__(self, args):
		self.num_res = args.num_res
		self.max_time_horizon = args.max_time_horizon
		self.max_job_len = args.max_job_len
		self.max_resource_size = args.max_resource_size
		self.job_horizon = args.job_horizon
		self.base_alg = args.base_alg

	def ejf(self, dm):
		for job_idx in range(1, dm.num_nodes):
			job = dm.get_node(job_idx)
			min_schedule_time = dm.calc_min_schedule_time(job.st_time, job_idx)
			dm.add_job(job_idx, min_schedule_time)
		return dm

	def sjf(self, dm):
		pending_job = []
		for job_idx in range(1, dm.num_nodes + 1):
			if job_idx == dm.num_nodes:
				pending_job_cap = 1
			else:
				pending_job_cap = self.job_horizon

			while len(pending_job) >= pending_job_cap:
				schedule_idx = -1
				schedule_time = -1
				min_job_len = -1
				for pending_job_idx in pending_job:
					cur_pending_job = dm.get_node(pending_job_idx)
					cur_min_schedule_time = dm.calc_min_schedule_time(cur_pending_job.st_time, pending_job_idx)
					if schedule_idx == -1 or cur_min_schedule_time < schedule_time or cur_min_schedule_time  == schedule_time and cur_pending_job.job_len < min_job_len:
						schedule_idx = pending_job_idx
						schedule_time = cur_min_schedule_time
						min_job_len = cur_pending_job.job_len
				dm.add_job(schedule_idx, schedule_time)
				dm.max_ed_time = max(dm.max_ed_time, dm.get_node(schedule_idx).ed_time)
				pending_job.remove(schedule_idx)

			if job_idx == dm.num_nodes:
				break
			pending_job.append(job_idx)

		return dm

	def random_schedule(self, dm):
		pending_job = []
		for job_idx in range(1, dm.num_nodes + 1):
			if job_idx == dm.num_nodes:
				pending_job_cap = 1
			else:
				pending_job_cap = self.job_horizon

			while len(pending_job) >= pending_job_cap:
				schedule_idx = np.random.choice(pending_job)
				cur_schedule_job = dm.get_node(schedule_idx)
				schedule_time = dm.calc_min_schedule_time(cur_schedule_job.st_time, schedule_idx)
				dm.add_job(schedule_idx, schedule_time)
				dm.max_ed_time = max(dm.max_ed_time, dm.get_node(schedule_idx).ed_time)
				pending_job.remove(schedule_idx)

			if job_idx == dm.num_nodes:
				break
			pending_job.append(job_idx)

		return dm

	def parse(self, job_seq, debug=False):
		self.is_debug = debug
		dm = JobScheduleManager(self.num_res, self.max_time_horizon, self.max_job_len, self.max_resource_size)
		for inp in job_seq:
			dm.nodes.append(JobNode(resource_size=inp['resource_size'], start_time=inp['start_time'], job_len=inp['job_len']))
		dm.num_nodes = len(dm.nodes)
		dm.num_jobs = len(job_seq)
		if self.base_alg == 'EJF':
			dm = self.ejf(dm)
		elif self.base_alg == 'SJF':
			dm = self.sjf(dm)
		else:
			dm = self.random_schedule(dm)

		for job_idx in range(1, dm.num_nodes):
			job = dm.get_node(job_idx)
			dm.update_embedding(job_idx)
			if job.schedule_time == job.st_time:
				dm.add_edge(dm.root, job_idx)
			else:
				for i in dm.terminate[job.schedule_time]:
					dm.add_edge(i, job_idx)
		dm.update_stat()
		return dm


class vrpParser(object):
	def parse(self, problem, debug=False):
		self.is_debug = debug
		dm = VrpManager(problem['capacity'])
		dm.nodes.append(VrpNode(x=problem['depot'][0], y=problem['depot'][1], demand=0, px=problem['depot'][0], py=problem['depot'][1], capacity=problem['capacity'], dis=0.0))
		for customer in problem['customers']:
			dm.nodes.append(VrpNode(x=customer['position'][0], y=customer['position'][1], demand=customer['demand'], px=customer['position'][0], py=customer['position'][1], capacity=problem['capacity'], dis=0.0))
		dm.num_nodes = len(dm.nodes)
		cur_capacity = problem['capacity']
		pending_nodes = [i for i in range(0, dm.num_nodes)]
		dm.add_route_node(0)
		cur_capacity = dm.vehicle_state[-1][1]
		while len(pending_nodes) > 1:
			dis = []
			demands = []
			pre_node_idx = dm.vehicle_state[-1][0]
			pre_node = dm.get_node(pre_node_idx)
			for i in pending_nodes:
				cur_node = dm.get_node(i)
				dis.append(dm.get_dis(pre_node, cur_node))
				demands.append(cur_node.demand)
			for i in range(len(pending_nodes)):
				for j in range(i + 1, len(pending_nodes)):
					if dis[i] > dis[j] or dis[i] == dis[j] and demands[i] > demands[j]:
						pending_nodes[i], pending_nodes[j] = pending_nodes[j], pending_nodes[i]
						dis[i], dis[j] = dis[j], dis[i]
						demands[i], demands[j] = demands[j], demands[i]
			for i in pending_nodes:
				if i == 0:
					if cur_capacity == problem['capacity']:
						continue
					dm.add_route_node(0)
					break
				else:
					cur_node = dm.get_node(i)
					if cur_node.demand > cur_capacity:
						continue
					dm.add_route_node(i)
					pending_nodes.remove(i)
					break
			cur_capacity = dm.vehicle_state[-1][1]
		dm.add_route_node(0)
		return dm
