# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import math
import random
import sys
import os
import json
import numpy as np
import time
import torch

import arguments as arguments
import models as models
import models.data_utils.data_utils as data_utils
from models.rewriter import HalideRewriter

argParser = arguments.get_arg_parser("Halide")
args = argParser.parse_args()

DataProcessor = data_utils.HalideDataProcessor()
term_vocab, term_vocab_list = DataProcessor.load_term_vocab()
op_vocab, op_vocab_list = DataProcessor.load_ops()
args.term_vocab_size = len(term_vocab)
args.op_vocab_size = len(op_vocab)
rewriter = HalideRewriter(args, term_vocab, term_vocab_list, op_vocab, op_vocab_list)

expr_rec = {}


def get_nonterm_idxes(tm, cur_idx=None):
	if cur_idx is None:
		cur_idx = tm.root
	cur_tree = tm.get_tree(cur_idx)
	if len(cur_tree.children) == 0:
		return []
	nonterm_idxes = []
	nonterm_idxes += [cur_idx]
	for child in cur_tree.children:
		child_tree = tm.get_tree(child)
		if child_tree.parent != cur_idx:
			raise ValueError('invalid edge: ' + str(cur_idx) + ' ' + cur_tree.root + ' ' + str(cur_tree.children) + ' ' + str(child) + ' ' + str(child_tree.parent))
		nonterm_idxes += get_nonterm_idxes(tm, child)
	return nonterm_idxes


def rewrite(tm, init_expr, len_tm, num_nodes_tm, depth):
	expr_rec[init_expr] = 1
	min_len = len_tm
	min_num_nodes = num_nodes_tm
	res_tm = tm
	if depth >= args.max_reduce_steps:
		return res_tm, min_len, min_num_nodes
	nonterm_idxes = get_nonterm_idxes(tm)
	candidate_tm = []
	for i in nonterm_idxes:
		cur_tree = tm.get_tree(i)
		for j in range(args.num_actions):
			op_list = rewriter.get_rewrite_seq(j)
			op = rewriter.get_rewrite_op(op_list[0])
			new_tm, update_tree_idxes = op(tm, i)
			if len(update_tree_idxes) == 0:
				continue
			new_expr = new_tm.to_string(new_tm.root)
			new_len = len(new_expr)
			new_num_nodes = new_tm.num_valid_nodes()
			if (new_expr in expr_rec) or len(expr_rec) >= args.num_sample_rewrite_pos and new_len >= min_len:
				continue
			q_idx = len(candidate_tm) - 1
			while q_idx >= 0 and new_len < candidate_tm[q_idx][0]:
				q_idx -= 1
			candidate_tm = candidate_tm[:q_idx + 1] + [(new_len, new_num_nodes, new_expr, new_tm)] + candidate_tm[q_idx + 1: args.num_sample_rewrite_pos]

	for i in range(len(candidate_tm)):
		new_tm, new_len, new_num_nodes = rewrite(candidate_tm[i][3], candidate_tm[i][2], candidate_tm[i][0], candidate_tm[i][1], depth + 1)
		if new_len < min_len:
			res_tm = new_tm
			min_len = new_len
			min_num_nodes = new_num_nodes
	return res_tm, min_len, min_num_nodes


def evaluate(args):
	print('Search:')

	test_data = data_utils.load_dataset(args.test_dataset, args)
	if args.test_min_len is not None:
		test_data = DataProcessor.prune_dataset(test_data, min_len=args.test_min_len)
		DataProcessor.calc_data_stat(test_data)
	data_size = len(test_data)
	test_data = test_data[:data_size]

	cum_expr_reward = 0
	cum_gt_reward = 0
	cum_tree_reward = 0

	for batch_idx in range(0, data_size, args.batch_size):
		batch_data = DataProcessor.get_batch(test_data, args.batch_size, batch_idx)
		for i, sample in enumerate(batch_data):
			gt_trace, tm = sample
			global expr_rec
			expr_rec = {}
			init_expr = tm.to_string(tm.root)
			len_tm = len(init_expr)
			num_nodes_tm = tm.num_trees
			res_tm, res_len, res_num_nodes = rewrite(tm, init_expr, len_tm, num_nodes_tm, 0)
			cur_expr_reward = len(init_expr) - res_len
			cur_tree_reward = num_nodes_tm - res_num_nodes
			cur_gt_reward = len(gt_trace[0]) - len(gt_trace[-1])
			cum_expr_reward += cur_expr_reward
			cum_tree_reward += cur_tree_reward
			cum_gt_reward += cur_gt_reward
			print('sample %d cur expr reward: %.4f cur tree reward: %.4f gt reward: %.4f avg expr reward: %.4f avg tree reward: %.4f avg gt reward: %.4f' \
				% (batch_idx + i, cur_expr_reward, cur_tree_reward, cur_gt_reward, cum_expr_reward * 1.0 / (batch_idx + i + 1), cum_tree_reward * 1.0 / (batch_idx + i + 1), cum_gt_reward * 1.0 / (batch_idx + i + 1)))
	cum_expr_reward = cum_expr_reward * 1.0 / data_size
	cum_tree_reward = cum_tree_reward * 1.0 / data_size
	cum_gt_reward = cum_gt_reward * 1.0 / data_size
	print('avg search expr reward: %.4f tree reward: %.4f avg gt reward: %.4f' % (cum_expr_reward, cum_tree_reward, cum_gt_reward))


if __name__ == "__main__":
	evaluate(args)