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
from ..data_utils import Tree
from ..data_utils.utils import *


class HalideRewriter(object):
	"""
	The rewriter for expression simplification, according to the rewriting rules of Halide.
	"""
	def __init__(self, args, term_vocab, term_vocab_list, op_vocab, op_vocab_list):
		self.term_vocab = term_vocab
		self.term_vocab_list = term_vocab_list
		self.op_vocab = op_vocab
		self.op_vocab_list = op_vocab_list
		self.term_vocab_size = args.term_vocab_size
		self.op_vocab_size = args.op_vocab_size
		self.rewrite_rules = [self.not_symbol_rewrite, self.simple_bool_rewrite, self.eq_bool_rewrite, \
		self.var_bound_rewrite, self.alg_const_calculation, self.cmp_const_calculation, \
		self.left_association, self.association, \
		self.muldiv_elimination, self.div_reduction, self.muldiv_to_mod_transformation, \
		self.muldiv_association, self.muldiv_distribution, \
		self.select_simplification, self.select_association, self.select_distribution, \
		self.minmax_simplification, self.minmax_alg_distribution, self.minmax_distribution]
		self.rewrite_seqs = [[i, -1, i] for i in range(len(self.rewrite_rules))]
		self.add_rewrite_seqs()


	def add_rewrite_seqs(self):
		self.rewrite_seqs += [[12, 4, -2, len(self.rewrite_seqs)]]
		self.rewrite_seqs += [[9, 4, -2, len(self.rewrite_seqs)]]
		self.rewrite_seqs += [[7, 5, -2, len(self.rewrite_seqs)]]
		self.rewrite_seqs += [[17, 4, -2, len(self.rewrite_seqs)]]
		self.rewrite_seqs += [[18, 5, -2, len(self.rewrite_seqs)]]
		self.rewrite_seqs += [[10, 4, -2, len(self.rewrite_seqs)]]
		self.rewrite_seqs += [[15, 0, -2, len(self.rewrite_seqs)]]
		self.rewrite_seqs += [[15, 5, -2, len(self.rewrite_seqs)]]
		self.rewrite_seqs += [[11, 4, -2, len(self.rewrite_seqs)]]
		self.rewrite_seqs += [[11, 11, -2, len(self.rewrite_seqs)]]


	def get_rewrite_op(self, idx):
		return self.rewrite_rules[idx]


	def get_rewrite_seq(self, idx):
		return self.rewrite_seqs[idx]


	def not_symbol_rewrite(self, tm, tree_idx): #return (tm, tree idxes to update embedding)
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if op != '!':
			return tm, []
		child = tm.get_tree(cur_tree.children[0])
		if child.is_const:
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			res.update_edge(parent_idx, cur_tree.children[0], child_idx)
			if res.trees[cur_tree.children[0]].root == '1':
				res.trees[cur_tree.children[0]].root = '0'
			else:
				res.trees[cur_tree.children[0]].root = '1'
			return res, [cur_tree.children[0]]
		if child.root == '!': # !!v0 -> v0
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			res.update_edge(parent_idx, child.children[0], child_idx)
			return res, [parent_idx]
		if child.root == '<':
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			res.update_edge(parent_idx, cur_tree.children[0], child_idx)
			res.trees[cur_tree.children[0]].children.reverse()
			return res, [cur_tree.children[0]]
		return tm, []


	def simple_bool_rewrite(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['&&', '||']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)
		if ltree.is_const and rtree.is_const:
			res = tm.clone()
			res.trees[tree_idx].root = calc(op, ltree.root, rtree.root)
			res.trees[tree_idx].is_const = True
			res.trees[tree_idx].children = []
			return res, [tree_idx]
		if rtree.is_const:
			if rtree.root == '1' and op == '||':
				res = tm.clone()
				res.trees[tree_idx].root = '1'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
			if rtree.root == '0' and op == '&&':
				res = tm.clone()
				res.trees[tree_idx].root = '0'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
			if (rtree.root == '1' and op == '&&') or (rtree.root == '0' and op == '||'):
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				res.update_edge(parent_idx, ltree_idx, child_idx)
				return res, [parent_idx]
		if ltree.is_const:
			if ltree.root == '1' and op == '||':
				res = tm.clone()
				res.trees[tree_idx].root = '1'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
			if ltree.root == '0' and op == '&&':
				res = tm.clone()
				res.trees[tree_idx].root = '0'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
			if (ltree.root == '1' and op == '&&') or (ltree.root == '0' and op == '||'):
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				res.update_edge(parent_idx, rtree_idx, child_idx)
				return res, [parent_idx]
		same_tree_idx = tm.find_subtree(ltree_idx, rtree_idx, op)
		if same_tree_idx == -1:
			return tm, []
		res = tm.clone()
		parent_idx = cur_tree.parent
		child_idx = res.find_child_idx(parent_idx, tree_idx)
		res.update_edge(parent_idx, ltree_idx, child_idx)
		return res, [parent_idx]


	def eq_bool_rewrite(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['&&', '||']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)
		if ltree.root == '!':
			ltree_child_idx = ltree.children[0]
			ltree_child = tm.get_tree(ltree_child_idx)
			if tm.equal_tree(ltree_child, rtree):
				res = tm.clone()
				if op == '||':
					res.trees[tree_idx].root = '1'
				else:
					res.trees[tree_idx].root = '0'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
		if rtree.root == '!':
			rtree_child_idx = rtree.children[0]
			rtree_child = tm.get_tree(rtree_child_idx)
			if tm.equal_tree(rtree_child, ltree):
				res = tm.clone()
				if op == '||':
					res.trees[tree_idx].root = '1'
				else:
					res.trees[tree_idx].root = '0'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
		if (ltree.root == rtree.root) and (ltree.root in ['<', '<=']):
			ltree_lchild_idx = ltree.children[0]
			ltree_lchild = tm.get_tree(ltree_lchild_idx)
			ltree_rchild_idx = ltree.children[1]
			ltree_rchild = tm.get_tree(ltree_rchild_idx)
			rtree_lchild_idx = rtree.children[0]
			rtree_lchild = tm.get_tree(rtree_lchild_idx)
			rtree_rchild_idx = rtree.children[1]
			rtree_rchild = tm.get_tree(rtree_rchild_idx)
			if tm.equal_tree(ltree_lchild, rtree_rchild) and tm.equal_tree(ltree_rchild, rtree_lchild):
				res = tm.clone()
				if op == '||':
					res.trees[tree_idx].root = '1'
				else:
					res.trees[tree_idx].root = '0'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
		if (op == '&&') and (ltree.root in ['<', '<=']) and (rtree.root in ['<', '<=']):
			ltree_lchild_idx = ltree.children[0]
			ltree_lchild = tm.get_tree(ltree_lchild_idx)
			ltree_rchild_idx = ltree.children[1]
			ltree_rchild = tm.get_tree(ltree_rchild_idx)
			rtree_lchild_idx = rtree.children[0]
			rtree_lchild = tm.get_tree(rtree_lchild_idx)
			rtree_rchild_idx = rtree.children[1]
			rtree_rchild = tm.get_tree(rtree_rchild_idx)
			if tm.equal_tree(ltree_lchild, rtree_rchild) and tm.equal_tree(ltree_rchild, rtree_lchild):
				res = tm.clone()
				res.trees[tree_idx].root = '0'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
		return tm, []


	def var_bound_rewrite(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['&&', '||']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)
		l_reverse = False
		r_reverse = False
		if ltree.root == '!':
			l_reverse = True
			ltree_idx = ltree.children[0]
			ltree = tm.get_tree(ltree_idx)
		if rtree.root == '!':
			r_reverse = True
			rtree_idx = rtree.children[0]
			rtree = tm.get_tree(rtree_idx)		
		if not ((ltree.root in ['<', '<=']) and (rtree.root in ['<', '<='])):
			return tm, []
		ltree_lchild_idx = ltree.children[0]
		ltree_lchild = tm.get_tree(ltree_lchild_idx)
		ltree_rchild_idx = ltree.children[1]
		ltree_rchild = tm.get_tree(ltree_rchild_idx)
		rtree_lchild_idx = rtree.children[0]
		rtree_lchild = tm.get_tree(rtree_lchild_idx)
		rtree_rchild_idx = rtree.children[1]
		rtree_rchild = tm.get_tree(rtree_rchild_idx)

		if tm.equal_tree(ltree_lchild, rtree_lchild) and ltree_rchild.root == '+' and rtree_rchild.root == '+':
			ltree_rchild_ltree_idx = ltree_rchild.children[0]
			ltree_rchild_ltree = tm.get_tree(ltree_rchild_ltree_idx)
			ltree_rchild_rtree_idx = ltree_rchild.children[1]
			ltree_rchild_rtree = tm.get_tree(ltree_rchild_rtree_idx)
			rtree_rchild_ltree_idx = rtree_rchild.children[0]
			rtree_rchild_ltree = tm.get_tree(rtree_rchild_ltree_idx)
			rtree_rchild_rtree_idx = rtree_rchild.children[1]
			rtree_rchild_rtree = tm.get_tree(rtree_rchild_rtree_idx)
			if ltree_rchild_rtree.is_const and rtree_rchild_rtree.is_const \
			and (not ltree_rchild_ltree.is_const) and tm.equal_tree(ltree_rchild_ltree, rtree_rchild_ltree):
				res = tm.clone()
				res.trees[ltree_rchild_idx].root = '-'
				res.trees[rtree_rchild_idx].root = '-'
				res.update_edge(ltree_idx, ltree_rchild_idx, 0)
				res.update_edge(rtree_idx, rtree_rchild_idx, 0)
				res.update_edge(ltree_idx, ltree_rchild_rtree_idx, 1)
				res.update_edge(rtree_idx, rtree_rchild_rtree_idx, 1)
				res.update_edge(ltree_rchild_idx, ltree_lchild_idx, 0)
				res.update_edge(ltree_rchild_idx, ltree_rchild_ltree_idx, 1)
				res.update_edge(rtree_rchild_idx, rtree_lchild_idx, 0)
				res.update_edge(rtree_rchild_idx, rtree_rchild_ltree_idx, 1)
				return res, [ltree_rchild_idx, rtree_rchild_idx]

		if tm.equal_tree(ltree_lchild, rtree_rchild) and ltree_rchild.root == '+' and rtree_lchild.root == '+':
			ltree_rchild_ltree_idx = ltree_rchild.children[0]
			ltree_rchild_ltree = tm.get_tree(ltree_rchild_ltree_idx)
			ltree_rchild_rtree_idx = ltree_rchild.children[1]
			ltree_rchild_rtree = tm.get_tree(ltree_rchild_rtree_idx)
			rtree_lchild_ltree_idx = rtree_lchild.children[0]
			rtree_lchild_ltree = tm.get_tree(rtree_lchild_ltree_idx)
			rtree_lchild_rtree_idx = rtree_lchild.children[1]
			rtree_lchild_rtree = tm.get_tree(rtree_lchild_rtree_idx)
			if ltree_rchild_rtree.is_const and rtree_lchild_rtree.is_const \
			and (not ltree_rchild_ltree.is_const) and tm.equal_tree(ltree_rchild_ltree, rtree_lchild_ltree):
				res = tm.clone()
				res.trees[ltree_rchild_idx].root = '-'
				res.trees[rtree_lchild_idx].root = '-'
				res.update_edge(ltree_idx, ltree_rchild_idx, 0)
				res.update_edge(rtree_idx, rtree_lchild_idx, 1)
				res.update_edge(ltree_idx, ltree_rchild_rtree_idx, 1)
				res.update_edge(rtree_idx, rtree_lchild_rtree_idx, 0)
				res.update_edge(ltree_rchild_idx, ltree_lchild_idx, 0)
				res.update_edge(ltree_rchild_idx, ltree_rchild_ltree_idx, 1)
				res.update_edge(rtree_lchild_idx, rtree_rchild_idx, 0)
				res.update_edge(rtree_lchild_idx, rtree_lchild_ltree_idx, 1)
				return res, [ltree_rchild_idx, rtree_lchild_idx]

		if tm.equal_tree(ltree_rchild, rtree_lchild) and ltree_lchild.root == '+' and rtree_rchild.root == '+':
			ltree_lchild_ltree_idx = ltree_lchild.children[0]
			ltree_lchild_ltree = tm.get_tree(ltree_lchild_ltree_idx)
			ltree_lchild_rtree_idx = ltree_lchild.children[1]
			ltree_lchild_rtree = tm.get_tree(ltree_lchild_rtree_idx)
			rtree_rchild_ltree_idx = rtree_rchild.children[0]
			rtree_rchild_ltree = tm.get_tree(rtree_rchild_ltree_idx)
			rtree_rchild_rtree_idx = rtree_rchild.children[1]
			rtree_rchild_rtree = tm.get_tree(rtree_rchild_rtree_idx)
			if ltree_lchild_rtree.is_const and rtree_rchild_rtree.is_const \
			and (not ltree_lchild_ltree.is_const) and tm.equal_tree(ltree_lchild_ltree, rtree_rchild_ltree):
				res = tm.clone()
				res.trees[ltree_lchild_idx].root = '-'
				res.trees[rtree_rchild_idx].root = '-'
				res.update_edge(ltree_idx, ltree_lchild_idx, 1)
				res.update_edge(rtree_idx, rtree_rchild_idx, 0)
				res.update_edge(ltree_idx, ltree_lchild_rtree_idx, 0)
				res.update_edge(rtree_idx, rtree_rchild_rtree_idx, 1)
				res.update_edge(ltree_lchild_idx, ltree_rchild_idx, 0)
				res.update_edge(ltree_lchild_idx, ltree_lchild_ltree_idx, 1)
				res.update_edge(rtree_rchild_idx, rtree_lchild_idx, 0)
				res.update_edge(rtree_rchild_idx, rtree_rchild_ltree_idx, 1)
				return res, [ltree_lchild_idx, rtree_rchild_idx]

		if tm.equal_tree(ltree_rchild, rtree_rchild) and ltree_lchild.root == '+' and rtree_lchild.root == '+':
			ltree_lchild_ltree_idx = ltree_lchild.children[0]
			ltree_lchild_ltree = tm.get_tree(ltree_lchild_ltree_idx)
			ltree_lchild_rtree_idx = ltree_lchild.children[1]
			ltree_lchild_rtree = tm.get_tree(ltree_lchild_rtree_idx)
			rtree_lchild_ltree_idx = rtree_lchild.children[0]
			rtree_lchild_ltree = tm.get_tree(rtree_lchild_ltree_idx)
			rtree_lchild_rtree_idx = rtree_lchild.children[1]
			rtree_lchild_rtree = tm.get_tree(rtree_lchild_rtree_idx)
			if ltree_lchild_rtree.is_const and rtree_lchild_rtree.is_const \
			and (not ltree_lchild_ltree.is_const) and tm.equal_tree(ltree_lchild_ltree, rtree_lchild_ltree):
				res = tm.clone()
				res.trees[ltree_lchild_idx].root = '-'
				res.trees[rtree_lchild_idx].root = '-'
				res.update_edge(ltree_idx, ltree_lchild_idx, 1)
				res.update_edge(rtree_idx, rtree_lchild_idx, 1)
				res.update_edge(ltree_idx, ltree_lchild_rtree_idx, 0)
				res.update_edge(rtree_idx, rtree_lchild_rtree_idx, 0)
				res.update_edge(ltree_lchild_idx, ltree_rchild_idx, 0)
				res.update_edge(ltree_lchild_idx, ltree_lchild_ltree_idx, 1)
				res.update_edge(rtree_lchild_idx, rtree_rchild_idx, 0)
				res.update_edge(rtree_lchild_idx, rtree_lchild_ltree_idx, 1)
				return res, [ltree_lchild_idx, rtree_lchild_idx]

		if ltree_lchild.is_const + ltree_rchild.is_const != 1:
			return tm, []
		if ltree_lchild.is_const:
			var = ltree_rchild
		else:
			var = ltree_lchild
		if rtree_lchild.is_const + rtree_rchild.is_const != 1:
			return tm, []
		r_max_value = None
		r_min_value = None
		l_max_value = None
		l_min_value = None
		if rtree_lchild.is_const:
			if not tm.equal_tree(var, rtree_rchild):
				return tm, []
			if r_reverse:
				r_max_value = int(rtree_lchild.root)
			else:
				r_min_value = int(rtree_lchild.root)
		if rtree_rchild.is_const:
			if not tm.equal_tree(var, rtree_lchild):
				return tm, []
			if r_reverse:
				r_min_value = int(rtree_rchild.root)
			else:
				r_max_value = int(rtree_rchild.root)
		if ltree_lchild.is_const:
			if l_reverse:
				l_max_value = int(ltree_lchild.root)
			else:
				l_min_value = int(ltree_lchild.root)
		else:
			if l_reverse:
				l_min_value = int(ltree_rchild.root)
			else:
				l_max_value = int(ltree_rchild.root)
		if (l_min_value is not None) and (r_min_value is not None):
			if l_min_value < r_min_value:
				if op == '&&':
					select_tree_idx = 1
				else:
					select_tree_idx = 0
			elif r_min_value < l_min_value:
				if op == '&&':
					select_tree_idx = 0
				else:
					select_tree_idx = 1
			else:
				if ltree.root == '<=' and (not l_reverse):
					if op == '&&':
						select_tree_idx = 1
					else:
						select_tree_idx = 0
				else:
					if op == '&&':
						select_tree_idx = 0
					else:
						select_tree_idx = 1
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			if select_tree_idx == 0:
				res.update_edge(parent_idx, cur_tree.children[0], child_idx)
			else:
				res.update_edge(parent_idx, cur_tree.children[1], child_idx)
			return res, [parent_idx]
		elif (l_min_value is not None) and (r_max_value is not None):
			if l_min_value > r_max_value:
				if op == '&&':
					res = tm.clone()
					res.trees[tree_idx].root = '0'
					res.trees[tree_idx].is_const = True
					res.trees[tree_idx].children = []
					return res, [tree_idx]
				else:
					return tm, []
			if l_min_value <= r_max_value:
				if op == '&&':
					if l_min_value == r_max_value:
						if ltree.root == '<=' and rtree.root == '<=':
							res = tm.clone()
							parent_idx = cur_tree.parent
							child_idx = res.find_child_idx(parent_idx, tree_idx)
							res.update_edge(parent_idx, ltree_idx, child_idx)
							res.trees[ltree_idx].root = '=='
							return res, [ltree_idx]
						else:
							res = tm.clone()
							res.trees[tree_idx].root = '0'
							res.trees[tree_idx].is_const = True
							res.trees[tree_idx].children = []
							return res, [tree_idx]
				else:
					if ltree.root == '<=' or rtree.root == '<=':
						res = tm.clone()
						res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					else:
						if l_min_value == r_max_value:
							res = tm.clone()
							res.trees[tree_idx].root = '!'
							res.trees[ltree_idx].root = '=='
							res.update_edge(tree_idx, ltree_idx, 0)
							res.trees[tree_idx].children = res.trees[tree_idx].children[:1]
							return res, [ltree_idx]
		elif (l_max_value is not None) and (r_min_value is not None):
			if r_min_value > l_max_value:
				if op == '&&':
					res = tm.clone()
					res.trees[tree_idx].root = '0'
					res.trees[tree_idx].is_const = True
					res.trees[tree_idx].children = []
					return res, [tree_idx]
				else:
					return tm, []
			if l_max_value >= r_min_value:
				if op == '&&':
					if l_max_value == r_min_value:
						if ltree.root == '<=' and rtree.root == '<=':
							res = tm.clone()
							parent_idx = cur_tree.parent
							child_idx = res.find_child_idx(parent_idx, tree_idx)
							res.update_edge(parent_idx, ltree_idx, child_idx)
							res.trees[ltree_idx].root = '=='
							return res, [ltree_idx]
						else:
							res = tm.clone()
							res.trees[tree_idx].root = '0'
							res.trees[tree_idx].is_const = True
							res.trees[tree_idx].children = []
							return res, [tree_idx]
				else:
					if ltree.root == '<=' or rtree.root == '<=':
						res = tm.clone()
						res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					else:
						if l_max_value == r_min_value:
							res = tm.clone()
							res.trees[tree_idx].root = '!'
							res.trees[ltree_idx].root = '=='
							res.update_edge(tree_idx, ltree_idx, 0)
							res.trees[tree_idx].children = res.trees[tree_idx].children[:1]
							return res, [ltree_idx]
		elif (l_max_value is not None) and (r_max_value is not None):
			if l_max_value > r_max_value:
				if op == '&&':
					select_tree_idx = 1
				else:
					select_tree_idx = 0
			elif r_max_value > l_max_value:
				if op == '&&':
					select_tree_idx = 0
				else:
					select_tree_idx = 1
			else:
				if ltree.root == '<=' and (not l_reverse):
					if op == '&&':
						select_tree_idx = 1
					else:
						select_tree_idx = 0
				else:
					if op == '&&':
						select_tree_idx = 0
					else:
						select_tree_idx = 1
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			if select_tree_idx == 0:
				res.update_edge(parent_idx, cur_tree.children[0], child_idx)
			else:
				res.update_edge(parent_idx, cur_tree.children[1], child_idx)
			return res, [parent_idx]
		return tm, []


	def alg_const_calculation(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['+', '-', '*', '/', '%']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)

		if ltree.is_const and rtree.is_const:
			res = tm.clone()
			res.trees[tree_idx].root = calc(op, ltree.root, rtree.root)
			res.trees[tree_idx].is_const = True
			res.trees[tree_idx].children = []
			return res, [tree_idx]

		if op == '+':
			if ltree.root == '0':
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				res.update_edge(parent_idx, rtree_idx, child_idx)
				return res, [parent_idx]
			if rtree.root == '0':
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				res.update_edge(parent_idx, ltree_idx, child_idx)
				return res, [parent_idx]
			if tm.equal_tree(ltree, rtree):
				res = tm.clone()
				res.trees[tree_idx].root = '*'
				res.trees[rtree_idx].root = '2'
				res.trees[rtree_idx].is_const = True
				res.trees[rtree_idx].children = []
				return res, [rtree_idx]
			if ltree.is_const:
				rtree_const_idx = tm.find_alg_const(rtree_idx, '+')
				if rtree_const_idx is not None:
					res = tm.clone()
					rtree_const = res.get_tree(rtree_const_idx)
					rtree_const_parent_idx = rtree_const.parent
					rtree_const_parent = res.get_tree(rtree_const_parent_idx)
					if rtree_const_parent.root == '-' and rtree_const_parent.children[1] == rtree_const_idx:
						set_value = int(rtree_const.root) - int(ltree.root)
					else:
						set_value = int(ltree.root) + int(rtree_const.root)
					parent_idx = cur_tree.parent
					child_idx = res.find_child_idx(parent_idx, tree_idx)
					res.update_edge(parent_idx, rtree_idx, child_idx)
					res.trees[rtree_const_idx].root = str(set_value)
					return res, [rtree_const_idx]
			if rtree.is_const:
				ltree_const_idx = tm.find_alg_const(ltree_idx, '+')
				if ltree_const_idx is not None:
					res = tm.clone()
					ltree_const = res.get_tree(ltree_const_idx)
					ltree_const_parent_idx = ltree_const.parent
					ltree_const_parent = res.get_tree(ltree_const_parent_idx)
					if ltree_const_parent.root == '-' and ltree_const_parent.children[1] == ltree_const_idx:
						set_value = int(ltree_const.root) - int(rtree.root)
					else:
						set_value = int(rtree.root) + int(ltree_const.root)
					parent_idx = cur_tree.parent
					child_idx = res.find_child_idx(parent_idx, tree_idx)
					res.update_edge(parent_idx, ltree_idx, child_idx)
					res.trees[ltree_const_idx].root = str(set_value)
					return res, [ltree_const_idx]
			if ltree.is_const and int(ltree.root) < 0:
				res = tm.clone()
				res.trees[tree_idx].children.reverse()
				res.trees[tree_idx].root = '-'
				res.trees[ltree_idx].root = str(-int(ltree.root))
				return res, [ltree_idx]
			if rtree.is_const and int(rtree.root) < 0:
				res = tm.clone()
				res.trees[tree_idx].root = '-'
				res.trees[rtree_idx].root = str(-int(rtree.root))
				return res, [rtree_idx]
			if ltree.root == '-':
				ltree_ltree_idx = ltree.children[0]
				ltree_ltree = tm.get_tree(ltree_ltree_idx)
				if ltree_ltree.root == '0':
					res = tm.clone()
					ltree_rtree_idx = ltree.children[1]
					res.trees[tree_idx].root = '-'
					res.update_edge(tree_idx, rtree_idx, 0)
					res.update_edge(tree_idx, ltree_rtree_idx, 1)
					return res, [tree_idx]
			if ltree.root == '+' and not rtree.is_const:
				ltree_rtree_idx = ltree.children[1]
				ltree_rtree = tm.get_tree(ltree_rtree_idx)
				if ltree_rtree.is_const:
					res = tm.clone()
					res.update_edge(ltree_idx, rtree_idx, 1)
					res.update_edge(tree_idx, ltree_rtree_idx, 1)
					return res, [ltree_idx]
			if ltree.is_const and not rtree.is_const:
				res = tm.clone()
				res.trees[tree_idx].children.reverse()
				return res, [tree_idx]

			same_tree_idx = tm.find_subtree(ltree_idx, rtree_idx, '-')
			if same_tree_idx == -1:
				return tm, []
			res = tm.clone()
			same_tree = res.get_tree(same_tree_idx)
			same_tree_parent_idx = same_tree.parent
			same_tree_parent = res.get_tree(same_tree_parent_idx)
			same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
			parent_tree_child_idx = res.find_child_idx(same_tree_parent.parent, same_tree_parent_idx)
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			res.update_edge(same_tree_parent.parent, same_tree_parent.children[1 - same_tree_child_idx], parent_tree_child_idx)
			res.update_edge(parent_idx, res.trees[tree_idx].children[0], child_idx)
			if same_tree_parent.parent == tree_idx:
				return res, [parent_idx]
			else:
				return res, [same_tree_parent.parent]

		if op == '-':
			if rtree.root == '0':
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				res.update_edge(parent_idx, ltree_idx, child_idx)
				return res, [parent_idx]
			if tm.equal_tree(ltree, rtree):
				res = tm.clone()
				res.trees[tree_idx].root = '0'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
			if rtree.is_const:
				ltree_const_idx = tm.find_alg_const(ltree_idx, '+')
				if ltree_const_idx is not None:
					res = tm.clone()
					ltree_const = res.get_tree(ltree_const_idx)
					ltree_const_parent_idx = ltree_const.parent
					ltree_const_parent = res.get_tree(ltree_const_parent_idx)
					if ltree_const_parent.root == '-' and ltree_const_parent.children[1] == ltree_const_idx:
						set_value = int(ltree_const.root) + int(rtree.root)
					else:
						set_value = int(ltree_const.root) - int(rtree.root)
					parent_idx = cur_tree.parent
					child_idx = res.find_child_idx(parent_idx, tree_idx)
					res.update_edge(parent_idx, ltree_idx, child_idx)
					res.trees[ltree_const_idx].root = str(set_value)
					return res, [ltree_const_idx]
			if rtree.is_const and int(rtree.root) < 0:
				res = tm.clone()
				res.trees[tree_idx].root = '+'
				res.trees[rtree_idx].root = str(-int(rtree.root))
				return res, [rtree_idx]
			same_tree_idx = tm.find_subtree(ltree_idx, rtree_idx, '+')
			if same_tree_idx != -1:
				res = tm.clone()
				same_tree = res.get_tree(same_tree_idx)
				same_tree_parent_idx = same_tree.parent
				same_tree_parent = res.get_tree(same_tree_parent_idx)
				same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
				parent_tree_child_idx = res.find_child_idx(same_tree_parent.parent, same_tree_parent_idx)
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				if same_tree_parent.root == '-' and same_tree_child_idx == 0:
					if same_tree.root != '0':
						res.trees[same_tree_idx].root = '0'
						res.trees[same_tree_idx].is_const = True
						res.trees[same_tree_idx].children = []
						res.update_edge(parent_idx, res.trees[tree_idx].children[0], child_idx)
						return res, [same_tree_idx]
				else:
					res.update_edge(same_tree_parent.parent, same_tree_parent.children[1 - same_tree_child_idx], parent_tree_child_idx)
					res.update_edge(parent_idx, res.trees[tree_idx].children[0], child_idx)
					if same_tree_parent.parent == tree_idx:
						return res, [parent_idx]
					else:
						return res, [same_tree_parent.parent]
			return tm, []

		if op == '*':
			if ltree.root == '0' or rtree.root == '0':
				res = tm.clone()
				res.trees[tree_idx].root = '0'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
			elif ltree.root == '1':
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				res.update_edge(parent_idx, rtree_idx, child_idx)
				return res, [parent_idx]
			elif rtree.root == '1':
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				res.update_edge(parent_idx, ltree_idx, child_idx)
				return res, [parent_idx]
			if ltree.root == '*' and not rtree.is_const:
				ltree_rtree_idx = ltree.children[1]
				ltree_rtree = tm.get_tree(ltree_rtree_idx)
				if ltree_rtree.is_const:
					res = tm.clone()
					res.update_edge(ltree_idx, rtree_idx, 1)
					res.update_edge(tree_idx, ltree_rtree_idx, 1)
					return res, [ltree_idx]
			if ltree.is_const and not rtree.is_const:
				res = tm.clone()
				res.trees[tree_idx].children.reverse()
				return res, [tree_idx]
			if rtree.is_const and ltree.root == '*':
				ltree_ltree_idx = ltree.children[0]
				ltree_ltree = tm.get_tree(ltree_ltree_idx)
				ltree_rtree_idx = ltree.children[1]
				ltree_rtree = tm.get_tree(ltree_rtree_idx)
				if ltree_rtree.is_const:
					res = tm.clone()
					res.trees[rtree_idx].root = str(int(rtree.root) * int(ltree_rtree.root))
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					return res, [rtree_idx]
			return tm, []

		if op == '/':
			if ltree.root == '0':
				res = tm.clone()
				res.trees[tree_idx].root = '0'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
			elif rtree.root == '1':
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				res.update_edge(parent_idx, ltree_idx, child_idx)
				return res, [parent_idx]
			elif tm.equal_tree(ltree, rtree):
				res = tm.clone()
				res.trees[tree_idx].root = '1'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
			same_tree_idx = tm.find_subtree(ltree_idx, rtree_idx, '*')
			if same_tree_idx != -1:
				res = tm.clone()
				same_tree = res.get_tree(same_tree_idx)
				same_tree_parent_idx = same_tree.parent
				same_tree_parent = res.get_tree(same_tree_parent_idx)
				same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
				parent_tree_child_idx = res.find_child_idx(same_tree_parent.parent, same_tree_parent_idx)
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				res.update_edge(same_tree_parent.parent, same_tree_parent.children[1 - same_tree_child_idx], parent_tree_child_idx)
				res.update_edge(parent_idx, res.trees[tree_idx].children[0], child_idx)
				if same_tree_parent.parent == tree_idx:
					return res, [parent_idx]
				else:
					return res, [same_tree_parent.parent]
			if ltree.root == '/' and rtree.is_const:
				ltree_ltree_idx = ltree.children[0]
				ltree_ltree = tm.get_tree(ltree_ltree_idx)
				ltree_rtree_idx = ltree.children[1]
				ltree_rtree = tm.get_tree(ltree_rtree_idx)
				if ltree_rtree.is_const:
					res = tm.clone()
					res.trees[rtree_idx].root = str(int(rtree.root) * int(ltree_rtree.root))
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					return res, [rtree_idx]
			return tm, []

		if op == '%':
			if ltree.root == '0' or rtree.root == '1' or tm.equal_tree(ltree, rtree):
				res = tm.clone()
				res.trees[tree_idx].root = '0'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
			same_tree_idx = tm.find_subtree(ltree_idx, rtree_idx, '*')
			if same_tree_idx != -1:
				res = tm.clone()
				res.trees[tree_idx].root = '0'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]	
			same_tree_idx = tm.find_subtree(ltree_idx, rtree_idx, '+')
			if same_tree_idx != -1:
				res = tm.clone()
				same_tree = res.get_tree(same_tree_idx)
				same_tree_parent_idx = same_tree.parent
				same_tree_parent = res.get_tree(same_tree_parent_idx)
				same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
				parent_tree_child_idx = res.find_child_idx(same_tree_parent.parent, same_tree_parent_idx)
				if same_tree_parent.root == '-' and same_tree_child_idx == 0:
					if same_tree.root != '0':
						res.trees[same_tree_idx].root = '0'
						res.trees[same_tree_idx].is_const = True
						res.trees[same_tree_idx].children = []
						return res, [same_tree_idx]
				else:
					res.update_edge(same_tree_parent.parent, same_tree_parent.children[1 - same_tree_child_idx], parent_tree_child_idx)
					return res, [same_tree_parent.parent]
			if ltree.root == '*' and rtree.is_const:
				ltree_ltree_idx = ltree.children[0]
				ltree_ltree = tm.get_tree(ltree_ltree_idx)
				ltree_rtree_idx = ltree.children[1]
				ltree_rtree = tm.get_tree(ltree_rtree_idx)
				if ltree_rtree.is_const:
					mod = int(ltree_rtree.root) % int(rtree.root)
					if mod == 0:
						res = tm.clone()
						res.trees[tree_idx].root = '0'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					if  mod != int(ltree_rtree.root):
						res = tm.clone()
						res.trees[ltree_rtree_idx].root = str(mod)
						return res, [ltree_rtree_idx]
			if rtree.is_const:
				times_tree_idx = tm.find_times_term(ltree_idx, int(rtree.root))
				if times_tree_idx != -1:
					res = tm.clone()
					times_tree = res.get_tree(times_tree_idx)
					times_tree_parent_idx = times_tree.parent
					times_tree_parent = res.get_tree(times_tree_parent_idx)
					times_tree_child_idx = res.find_child_idx(times_tree_parent_idx, times_tree_idx)
					parent_times_tree_child_idx = res.find_child_idx(times_tree_parent.parent, times_tree_parent_idx)
					if times_tree_parent.root == '-' and times_tree_child_idx == 0:
						if times_tree.root != '0':
							res.trees[times_tree_idx].root = '0'
							res.trees[times_tree_idx].is_const = True
							res.trees[times_tree_idx].children = []
							return res, [times_tree_idx]
					else:
						res.update_edge(times_tree_parent.parent, times_tree_parent.children[1 - times_tree_child_idx], parent_times_tree_child_idx)
						return res, [times_tree_parent.parent]
				const_tree_idx = tm.find_alg_const(ltree_idx, '+')
				if const_tree_idx != None:
					const_tree = tm.get_tree(const_tree_idx)
					if int(const_tree.root) % int(rtree.root) != int(const_tree.root):
						res = tm.clone()
						res.trees[const_tree_idx].root = str(int(const_tree.root) % int(rtree.root))
						return res, [const_tree_idx]

		return tm, []


	def cmp_const_calculation(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['<', '<=', '==']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)

		if ltree.is_const and rtree.is_const:
			res = tm.clone()
			res.trees[tree_idx].root = calc(op, ltree.root, rtree.root)
			res.trees[tree_idx].is_const = True
			res.trees[tree_idx].children = []
			return res, [tree_idx]

		if tm.equal_tree(ltree, rtree):
			if op in ['==', '<=']:
				res = tm.clone()
				res.trees[tree_idx].root = '1'
			else: #op == '<'
				res = tm.clone()
				res.trees[tree_idx].root = '0'
			res.trees[tree_idx].is_const = True
			res.trees[tree_idx].children = []
			return res, [tree_idx]

		same_tree_idx = tm.find_subtree(ltree_idx, rtree_idx, '+')
		if same_tree_idx != -1:
			res = tm.clone()
			same_tree = res.get_tree(same_tree_idx)
			same_tree_parent_idx = same_tree.parent
			same_tree_parent = res.get_tree(same_tree_parent_idx)
			same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
			parent_tree_child_idx = res.find_child_idx(same_tree_parent.parent, same_tree_parent_idx)
			res.trees[rtree_idx].root = '0'
			res.trees[rtree_idx].is_const = True
			res.trees[rtree_idx].children = []
			if same_tree_parent.root == '-' and same_tree_child_idx == 0:
				if same_tree.root != '0':
					res.trees[same_tree_idx].root = '0'
					res.trees[same_tree_idx].is_const = True
					res.trees[same_tree_idx].children = []
					return res, [same_tree_idx, rtree_idx]
			else:
				res.update_edge(same_tree_parent.parent, same_tree_parent.children[1 - same_tree_child_idx], parent_tree_child_idx)
				return res, [same_tree_parent.parent, rtree_idx]

		same_tree_idx = tm.find_subtree(rtree_idx, ltree_idx, '+')
		if same_tree_idx != -1:
			res = tm.clone()
			same_tree = res.get_tree(same_tree_idx)
			same_tree_parent_idx = same_tree.parent
			same_tree_parent = res.get_tree(same_tree_parent_idx)
			same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
			parent_tree_child_idx = res.find_child_idx(same_tree_parent.parent, same_tree_parent_idx)
			res.trees[ltree_idx].root = '0'
			res.trees[ltree_idx].is_const = True
			res.trees[ltree_idx].children = []
			if same_tree_parent.root == '-' and same_tree_child_idx == 0:
				if same_tree.root != '0':
					res.trees[same_tree_idx].root = '0'
					res.trees[same_tree_idx].is_const = True
					res.trees[same_tree_idx].children = []
					return res, [same_tree_idx, ltree_idx]
			else:
				res.update_edge(same_tree_parent.parent, same_tree_parent.children[1 - same_tree_child_idx], parent_tree_child_idx)
				return res, [same_tree_parent.parent, ltree_idx]

		if ltree.root in ['+', '-', '*', '/'] and rtree.is_const:
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			if ltree_ltree.is_const:
				res = tm.clone()
				if ltree.root == '+':
					res.update_edge(tree_idx, ltree_rtree_idx, 0)
					res.trees[rtree_idx].root = str(int(rtree.root) - int(ltree_ltree.root))
					return res, [rtree_idx]
				elif ltree.root == '-':
					res.update_edge(tree_idx, ltree_rtree_idx, 0)
					res.trees[rtree_idx].root = str(int(ltree_ltree.root) - int(rtree.root))
					res.trees[tree_idx].children.reverse()
					return res, [rtree_idx]
				elif ltree.root == '*':
					if int(ltree_ltree.root) == 0:
						res.trees[ltree_idx].root = '0'
						res.trees[ltree_idx].is_const = True
						res.trees[ltree_idx].children = []
						return res, [ltree_idx]
					else:
						if int(rtree.root) % int(ltree_ltree.root) != 0 and op == '==':
							res.trees[tree_idx].root = '0'
							res.trees[tree_idx].is_const = True
							res.trees[tree_idx].children = []
							return res, [tree_idx]
						res.update_edge(tree_idx, ltree_rtree_idx, 0)
						if int(rtree.root) % int(ltree_ltree.root) == 0 or op == '<=' and int(ltree_ltree.root) > 0 or op == '<' and int(ltree_ltree.root) < 0:
							res.trees[rtree_idx].root = str(int(rtree.root) // int(ltree_ltree.root))
						else:
							res.trees[rtree_idx].root = str(int(rtree.root) // int(ltree_ltree.root) + 1)
						if int(ltree_ltree.root) < 0:
							res.trees[tree_idx].children.reverse()
						return res, [rtree_idx]
				elif ltree.root == '/':
					if int(ltree_ltree.root) == 0:
						res.trees[ltree_idx].root = '0'
						res.trees[ltree_idx].is_const = True
						res.trees[ltree_idx].children = []
						return res, [ltree_idx]
			if ltree_rtree.is_const:
				res = tm.clone()
				if ltree.root == '+':
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					res.trees[rtree_idx].root = str(int(rtree.root) - int(ltree_rtree.root))
					return res, [rtree_idx]
				elif ltree.root == '-':
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					res.trees[rtree_idx].root = str(int(ltree_rtree.root) + int(rtree.root))
					return res, [rtree_idx]
				elif ltree.root == '*':
					if int(ltree_rtree.root) == 0:
						res.trees[ltree_idx].root = '0'
						res.trees[ltree_idx].is_const = True
						res.trees[ltree_idx].children = []
						return res, [ltree_idx]
					else:
						if int(rtree.root) % int(ltree_rtree.root) != 0 and op == '==':
							res.trees[tree_idx].root = '0'
							res.trees[tree_idx].is_const = True
							res.trees[tree_idx].children = []
							return res,[tree_idx]
						res.update_edge(tree_idx, ltree_ltree_idx, 0)
						if int(rtree.root) % int(ltree_rtree.root) == 0 or op == '<=' and int(ltree_rtree.root) > 0 or op == '<' and int(ltree_rtree.root) < 0:
							res.trees[rtree_idx].root = str(int(rtree.root) // int(ltree_rtree.root))
						else:
							res.trees[rtree_idx].root = str(int(rtree.root) // int(ltree_rtree.root) + 1)
						if int(ltree_rtree.root) < 0:
							res.trees[tree_idx].children.reverse()
						return res, [rtree_idx]
				elif ltree.root == '/' and op == '<':
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					res.trees[rtree_idx].root = str(int(rtree.root) * int(ltree_rtree.root))
					if int(ltree_rtree.root) < 0:
						res.trees[tree_idx].children.reverse()
					return res, [rtree_idx]
				elif ltree.root == '/' and op == '<=' and int(ltree_rtree.root) > 0:
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					res.trees[rtree_idx].root = str((int(rtree.root) + 1) * int(ltree_rtree.root) - 1)
					return res, [rtree_idx]
			if ltree.root == '-' and rtree.root == '0':
				res = tm.clone()
				res.update_edge(tree_idx, ltree_ltree_idx, 0)
				res.update_edge(tree_idx, ltree_rtree_idx, 1)
				return res, [tree_idx]

		if rtree.root in ['+', '-', '*', '/'] and ltree.is_const:
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = tm.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = tm.get_tree(rtree_rtree_idx)
			if rtree_ltree.is_const:
				res = tm.clone()
				if rtree.root == '+':
					res.update_edge(tree_idx, rtree_rtree_idx, 1)
					res.trees[ltree_idx].root = str(int(ltree.root) - int(rtree_ltree.root))
					return res, [ltree_idx]
				elif rtree.root == '-':
					res.update_edge(tree_idx, rtree_rtree_idx, 1)
					res.trees[ltree_idx].root = str(int(rtree_ltree.root) - int(ltree.root))
					res.trees[tree_idx].children.reverse()
					return res, [ltree_idx]
				elif rtree.root == '*':
					if int(rtree_ltree.root) == 0:
						res.trees[rtree_idx].root = '0'
						res.trees[rtree_idx].is_const = True
						res.trees[rtree_idx].children = []
						return res, [rtree_idx]
					else:
						if int(ltree.root) % int(rtree_ltree.root) != 0 and op == '==':
							res.trees[tree_idx].root = '0'
							res.trees[tree_idx].is_const = True
							res.trees[tree_idx].children = []
							return res, [tree_idx]
						res.update_edge(tree_idx, rtree_rtree_idx, 1)
						if int(ltree.root) % int(rtree_ltree.root) == 0 or op == '<' and int(rtree_ltree.root) > 0 or op == '<=' and int(rtree_ltree.root) < 0:
							res.trees[ltree_idx].root = str(int(ltree.root) // int(rtree_ltree.root))
						else:
							res.trees[ltree_idx].root = str(int(ltree.root) // int(rtree_ltree.root) + 1)
						if int(rtree_ltree.root) < 0:
							res.trees[tree_idx].children.reverse()
						return res, [ltree_idx]
				elif rtree.root == '/':
					if int(rtree_ltree.root) == 0:
						res.trees[rtree_idx].root = '0'
						res.trees[rtree_idx].is_const = True
						res.trees[rtree_idx].children = []
						return res, [rtree_idx]
					
			if rtree_rtree.is_const:
				res = tm.clone()
				if rtree.root == '+':
					res.update_edge(tree_idx, rtree_ltree_idx, 1)
					res.trees[ltree_idx].root = str(int(ltree.root) - int(rtree_rtree.root))
					return res, [ltree_idx]
				elif rtree.root == '-':
					res.update_edge(tree_idx, rtree_ltree_idx, 1)
					res.trees[ltree_idx].root = str(int(rtree_rtree.root) + int(ltree.root))
					return res, [ltree_idx]
				elif rtree.root == '*':
					if int(rtree_rtree.root) == 0:
						res.trees[rtree_idx].root = '0'
						res.trees[rtree_idx].is_const = True
						res.trees[rtree_idx].children = []
						return res, [rtree_idx]
					else:
						if int(ltree.root) % int(rtree_rtree.root) != 0 and op == '==':
							res.trees[tree_idx].root = '0'
							res.trees[tree_idx].is_const = True
							res.trees[tree_idx].children = []
							return res, [tree_idx]
						res.update_edge(tree_idx, rtree_ltree_idx, 1)
						if int(ltree.root) % int(rtree_rtree.root) == 0 or op == '<' and int(rtree_rtree.root) > 0 or op == '<=' and int(rtree_rtree.root) < 0:
							res.trees[ltree_idx].root = str(int(ltree.root) // int(rtree_rtree.root))
						else:
							res.trees[ltree_idx].root = str(int(ltree.root) // int(rtree_rtree.root) + 1)
						if int(rtree_rtree.root) < 0:
							res.trees[tree_idx].children.reverse()
						return res, [ltree_idx]
				elif rtree.root == '/' and op == '<=':
					res.update_edge(tree_idx, rtree_ltree_idx, 1)
					res.trees[ltree_idx].root = str(int(ltree.root) * int(rtree_rtree.root))
					if int(rtree_rtree.root) < 0:
						res.trees[tree_idx].children.reverse()
					return res, [ltree_idx]
				elif rtree.root == '/' and op == '<' and int(rtree_rtree.root) > 0:
					res.update_edge(tree_idx, rtree_ltree_idx, 1)
					res.trees[ltree_idx].root = str((int(ltree.root) + 1) * int(rtree_rtree.root) - 1)
					return res, [ltree_idx]
			if ltree.root == '0' and rtree.root == '-':
				res = tm.clone()
				res.update_edge(tree_idx, rtree_rtree_idx, 0)
				res.update_edge(tree_idx, rtree_ltree_idx, 1)
				return res, [tree_idx]

		if ltree.root in ['+', '-']:
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			left_reduced = False
			same_tree_idx = tm.find_subtree(rtree_idx, ltree_ltree_idx, '+')
			if same_tree_idx == -1:
				if ltree.root == '+':
					same_tree_idx = tm.find_subtree(rtree_idx, ltree_rtree_idx, '+')
				else:
					same_tree_idx = tm.find_subtree(rtree_idx, ltree_rtree_idx, '-')
			else:
				left_reduced = True
			if same_tree_idx != -1:
				res = tm.clone()
				same_tree = res.get_tree(same_tree_idx)
				same_tree_parent_idx = same_tree.parent
				same_tree_parent = res.get_tree(same_tree_parent_idx)
				same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
				parent_tree_child_idx = res.find_child_idx(same_tree_parent.parent, same_tree_parent_idx)
				if left_reduced:
					if ltree.root == '-':
						res.trees[ltree_ltree_idx].root = '0'
						res.trees[ltree_ltree_idx].is_const = True
						res.trees[ltree_ltree_idx].children = []
					else:
						res.update_edge(tree_idx, ltree_rtree_idx, 0)
				else:
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
				if same_tree_parent.root == '-' and same_tree_child_idx == 0:
					if same_tree.root != '0':
						res.trees[same_tree_idx].root = '0'
						res.trees[same_tree_idx].is_const = True
						res.trees[same_tree_idx].children = []
						if left_reduced and ltree.root == '-':
							return res, [ltree_ltree_idx, same_tree_idx]
						else:
							return res, [same_tree_idx]
				else:
					res.update_edge(same_tree_parent.parent, same_tree_parent.children[1 - same_tree_child_idx], parent_tree_child_idx)
					if left_reduced and ltree.root == '-':
						return res, [ltree_ltree_idx, same_tree_parent.parent]
					else:
						return res, [same_tree_parent.parent, tree_idx]

			left_reduced = False
			same_tree_idx = tm.find_subtree(rtree_idx, ltree_ltree_idx, '-')
			if same_tree_idx == -1:
				if ltree.root == '+':
					same_tree_idx = tm.find_subtree(rtree_idx, ltree_rtree_idx, '-')
				else:
					same_tree_idx = tm.find_subtree(rtree_idx, ltree_rtree_idx, '+')
			else:
				left_reduced = True
			if same_tree_idx != -1:
				res = tm.clone()
				same_tree = res.get_tree(same_tree_idx)
				same_tree_parent_idx = same_tree.parent
				same_tree_parent = res.get_tree(same_tree_parent_idx)
				same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
				if left_reduced:
					res.update_edge(tree_idx, ltree_rtree_idx, 0)
				else:
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
				new_tree_idx = res.create_tree('*', same_tree_parent_idx, same_tree.depth)
				new_tree_rtree_idx = res.create_tree('2', new_tree_idx, same_tree.depth + 1)
				res.update_edge(new_tree_idx, same_tree_idx, 0)
				res.update_edge(new_tree_idx, new_tree_rtree_idx, 1)
				res.update_edge(same_tree_parent_idx, new_tree_idx, same_tree_child_idx)
				return res, [new_tree_rtree_idx]

		if rtree.root in ['+', '-']:
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = tm.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = tm.get_tree(rtree_rtree_idx)
			left_reduced = False
			same_tree_idx = tm.find_subtree(ltree_idx, rtree_ltree_idx, '+')
			if same_tree_idx == -1:
				if rtree.root == '+':
					same_tree_idx = tm.find_subtree(ltree_idx, rtree_rtree_idx, '+')
				else:
					same_tree_idx = tm.find_subtree(ltree_idx, rtree_rtree_idx, '-')
			else:
				left_reduced = True
			if same_tree_idx != -1:
				res = tm.clone()
				same_tree = res.get_tree(same_tree_idx)
				same_tree_parent_idx = same_tree.parent
				same_tree_parent = res.get_tree(same_tree_parent_idx)
				same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
				parent_tree_child_idx = res.find_child_idx(same_tree_parent.parent, same_tree_parent_idx)
				if left_reduced:
					if rtree.root == '-':
						res.trees[rtree_ltree_idx].root = '0'
						res.trees[rtree_ltree_idx].is_const = True
						res.trees[rtree_ltree_idx].children = []
					else:
						res.update_edge(tree_idx, rtree_rtree_idx, 1)
				else:
					res.update_edge(tree_idx, rtree_ltree_idx, 1)
				if same_tree_parent.root == '-' and same_tree_child_idx == 0:
					if same_tree.root != '0':
						res.trees[same_tree_idx].root = '0'
						res.trees[same_tree_idx].is_const = True
						res.trees[same_tree_idx].children = []
						if rtree.root == '-' and left_reduced:
							return res, [same_tree_idx, rtree_ltree_idx]
						else:
							return res, [same_tree_idx]
				else:
					res.update_edge(same_tree_parent.parent, same_tree_parent.children[1 - same_tree_child_idx], parent_tree_child_idx)
					if rtree.root == '-' and left_reduced:
						return res, [same_tree_parent.parent, rtree_ltree_idx]
					else:
						return res, [same_tree_parent.parent, tree_idx]

			left_reduced = False
			same_tree_idx = tm.find_subtree(ltree_idx, rtree_ltree_idx, '-')
			if same_tree_idx == -1:
				if rtree.root == '+':
					same_tree_idx = tm.find_subtree(ltree_idx, rtree_rtree_idx, '-')
				else:
					same_tree_idx = tm.find_subtree(ltree_idx, rtree_rtree_idx, '+')
			else:
				left_reduced = True
			if same_tree_idx != -1:
				res = tm.clone()
				same_tree = res.get_tree(same_tree_idx)
				same_tree_parent_idx = same_tree.parent
				same_tree_parent = res.get_tree(same_tree_parent_idx)
				same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
				if left_reduced:
					res.update_edge(tree_idx, rtree_rtree_idx, 1)
				else:
					res.update_edge(tree_idx, rtree_ltree_idx, 1)
				new_tree_idx = res.create_tree('*', same_tree_parent_idx, same_tree.depth)
				new_tree_rtree_idx = res.create_tree('2', new_tree_idx, same_tree.depth + 1)
				res.update_edge(new_tree_idx, same_tree_idx, 0)
				res.update_edge(new_tree_idx, new_tree_rtree_idx, 1)
				res.update_edge(same_tree_parent_idx, new_tree_idx, same_tree_child_idx)
				return res, [new_tree_rtree_idx]

		if ltree.root == '-' and rtree.root == '-':
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = tm.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = tm.get_tree(rtree_rtree_idx)
			if ltree_ltree.root == '0' and rtree_rtree.root != '0':
				res = tm.clone()
				res.update_edge(ltree_idx, rtree_rtree_idx, 0)
				res.update_edge(tree_idx, rtree_ltree_idx, 1)
				return res, [ltree_idx]
			if rtree_ltree.root == '0' and ltree_rtree.root != '0':
				res = tm.clone()
				res.update_edge(rtree_idx, ltree_rtree_idx, 0)
				res.update_edge(tree_idx, ltree_ltree_idx, 0)
				return res, [rtree_idx]
		
		ltree_const_idx = tm.find_alg_const(ltree_idx, '+')
		rtree_const_idx = tm.find_alg_const(rtree_idx, '+')
		if (ltree_const_idx is not None) and (rtree_const_idx is not None):
			res = tm.clone()
			rtree_const = res.get_tree(rtree_const_idx)
			rtree_const_parent_idx = rtree_const.parent
			rtree_const_parent = res.get_tree(rtree_const_parent_idx)
			if rtree_const_parent.root == '-' and rtree_const_parent.children[1] == rtree_const_idx:
				set_value = -int(rtree_const.root)
			else:
				set_value = int(rtree_const.root)
			ltree_const = res.get_tree(ltree_const_idx)
			ltree_const_parent_idx = ltree_const.parent
			ltree_const_parent = res.get_tree(ltree_const_parent_idx)
			if ltree_const_parent.root == '-' and ltree_const_parent.children[1] == ltree_const_idx:
				set_value += int(ltree_const.root)
			else:
				set_value -= int(ltree_const.root)
			if rtree_const_parent.root == '-' and rtree_const_parent.children[1] == rtree_const_idx:
				set_value = -set_value
			ltree_const_child_idx = res.find_child_idx(ltree_const_parent_idx, ltree_const_idx)
			ltree_const_parent_child_idx = res.find_child_idx(ltree_const_parent.parent, ltree_const_parent_idx)
			res.trees[rtree_const_idx].root = str(set_value)
			if ltree_const_parent.root == '-' and ltree_const_child_idx == 0:
				res.trees[ltree_const_idx].root = '0'
				return res, [rtree_const_idx, ltree_const_idx]
			else:
				res.update_edge(ltree_const_parent.parent, ltree_const_parent.children[1 - ltree_const_child_idx], ltree_const_parent_child_idx)
				return res, [rtree_const_idx, ltree_const_parent.parent]

		ltree_const_idx = tm.find_alg_const(ltree_idx, '*')
		rtree_const_idx = tm.find_alg_const(rtree_idx, '*')
		if (ltree_const_idx is not None) and (rtree_const_idx is not None):
			ltree_const = tm.get_tree(ltree_const_idx)
			ltree_const_parent_idx = ltree_const.parent
			ltree_const_parent = tm.get_tree(ltree_const_parent_idx)
			rtree_const = tm.get_tree(rtree_const_idx)
			rtree_const_parent_idx = rtree_const.parent
			rtree_const_parent = tm.get_tree(rtree_const_parent_idx)
			if int(ltree_const.root) == 0:
				res = tm.clone()
				res.trees[ltree_idx].root = '0'
				res.trees[ltree_idx].is_const = True
				res.trees[ltree_idx].children = []
				return res, [ltree_idx]
			if op != '==' or int(rtree_const.root) % int(ltree_const.root) == 0:
				res = tm.clone()
				set_value = int(rtree_const.root) // int(ltree_const.root)
				ltree_const_child_idx = res.find_child_idx(ltree_const_parent_idx, ltree_const_idx)
				ltree_const_parent_child_idx = res.find_child_idx(ltree_const_parent.parent, ltree_const_parent_idx)
				res.update_edge(ltree_const_parent.parent, ltree_const_parent.children[1 - ltree_const_child_idx], ltree_const_parent_child_idx)
				res.trees[rtree_const_idx].root = str(set_value)
				if int(ltree_const.root) < 0:
					res.trees[tree_idx].children.reverse()
				return res, [rtree_const_idx, ltree_const_parent.parent]

		left_bounded = True
		right_bounded = True
		left_min_value = 10000
		left_max_value = -10000
		right_min_value = 10000
		right_max_value = -10000

		if ltree.is_const:
			left_min_value = int(ltree.root)
			left_max_value = int(ltree.root)
		elif ltree.root in ['+', '-']:
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			if ltree_ltree.is_const:
				left_min_value = int(ltree_ltree.root)
				left_max_value = int(ltree_ltree.root)
			elif ltree_ltree.root == '%':
				ltree_ltree_ltree_idx = ltree_ltree.children[0]
				ltree_ltree_ltree = tm.get_tree(ltree_ltree_ltree_idx)
				ltree_ltree_rtree_idx = ltree_ltree.children[1]
				ltree_ltree_rtree = tm.get_tree(ltree_ltree_rtree_idx)
				if ltree_ltree_rtree.is_const and int(ltree_ltree_rtree.root) > 0:
					left_min_value = 0
					left_max_value = int(ltree_ltree_rtree.root) - 1
				else:
					left_bounded = False
			else:
				left_bounded = False
			if ltree_rtree.is_const:
				left_min_value = eval(str(left_min_value) + ltree.root + ltree_rtree.root)
				left_max_value = eval(str(left_max_value) + ltree.root + ltree_rtree.root)
			elif ltree_rtree.root == '%':
				ltree_rtree_ltree_idx = ltree_rtree.children[0]
				ltree_rtree_ltree = tm.get_tree(ltree_rtree_ltree_idx)
				ltree_rtree_rtree_idx = ltree_rtree.children[1]
				ltree_rtree_rtree = tm.get_tree(ltree_rtree_rtree_idx)
				if ltree_rtree_rtree.is_const and int(ltree_rtree_rtree.root) > 0:
					if ltree.root == '+':
						left_min_value = left_min_value
						left_max_value = left_max_value + int(ltree_rtree_rtree.root) - 1
					else:
						if left_bounded and ltree_ltree.root == '%' and tm.equal_tree(ltree_ltree_ltree, ltree_rtree_ltree):
							d = gcd(int(ltree_ltree_rtree.root), int(ltree_rtree_rtree.root))
							left_min_value = left_min_value - int(ltree_rtree_rtree.root) + d
							left_max_value = left_max_value - d + 1
						else:
							left_min_value = left_min_value - int(ltree_rtree_rtree.root) + 1
							left_max_value = left_max_value
				else:
					left_bounded = False
			else:
				left_bounded = False
		elif ltree.root == '%':
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			if ltree_rtree.is_const and int(ltree_rtree.root) > 0:
				left_min_value = 0
				left_max_value = int(ltree_rtree.root) - 1
			else:
				left_bounded = False
		else:
			left_bounded = False

		if rtree.is_const:
			right_min_value = int(rtree.root)
			right_max_value = int(rtree.root)
		elif rtree.root in ['+', '-']:
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = tm.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = tm.get_tree(rtree_rtree_idx)
			if rtree_ltree.is_const:
				right_min_value = int(rtree_ltree.root)
				right_max_value = int(rtree_ltree.root)
			elif rtree_ltree.root == '%':
				rtree_ltree_ltree_idx = rtree_ltree.children[0]
				rtree_ltree_ltree = tm.get_tree(rtree_ltree_ltree_idx)
				rtree_ltree_rtree_idx = rtree_ltree.children[1]
				rtree_ltree_rtree = tm.get_tree(rtree_ltree_rtree_idx)
				if rtree_ltree_rtree.is_const and int(rtree_ltree_rtree.root) > 0:
					right_min_value = 0
					right_max_value = int(rtree_ltree_rtree.root) - 1
				else:
					right_bounded = False
			else:
				right_bounded = False
			if rtree_rtree.is_const:
				right_min_value = eval(str(right_min_value) + rtree.root + rtree_rtree.root)
				right_max_value = eval(str(right_max_value) + rtree.root + rtree_rtree.root)
			elif rtree_rtree.root == '%':
				rtree_rtree_ltree_idx = rtree_rtree.children[0]
				rtree_rtree_ltree = tm.get_tree(rtree_rtree_ltree_idx)
				rtree_rtree_rtree_idx = rtree_rtree.children[1]
				rtree_rtree_rtree = tm.get_tree(rtree_rtree_rtree_idx)
				if rtree_rtree_rtree.is_const and int(rtree_rtree_rtree.root) > 0:
					if rtree.root == '+':
						right_min_value = right_min_value
						right_max_value = right_max_value + int(rtree_rtree_rtree.root) - 1
					else:
						if right_bounded and rtree_ltree.root == '%' and tm.equal_tree(rtree_ltree_ltree, rtree_rtree_ltree):
							d = gcd(int(rtree_ltree_rtree.root), int(rtree_rtree_rtree.root))
							right_min_value = right_min_value - int(rtree_rtree_rtree.root) + d
							right_max_value = right_max_value - d + 1
						else:
							right_min_value = right_min_value - int(rtree_rtree_rtree.root) + 1
							right_max_value = right_max_value
				else:
					right_bounded = False
			else:
				right_bounded = False
		elif rtree.root == '%':
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = tm.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = tm.get_tree(rtree_rtree_idx)
			if rtree_rtree.is_const and int(rtree_rtree.root) > 0:
				right_min_value = 0
				right_max_value = int(rtree_rtree.root) - 1
			else:
				right_bounded = False
		else:
			right_bounded = False

		if left_bounded and right_bounded and op != '==':
			equ = str(left_max_value) + op + str(right_min_value)
			if eval(equ) is True:
				res = tm.clone()
				res.trees[tree_idx].root = '1'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]
			equ = str(left_min_value) + op + str(right_max_value)
			if eval(equ) is False:
				res = tm.clone()
				res.trees[tree_idx].root = '0'
				res.trees[tree_idx].is_const = True
				res.trees[tree_idx].children = []
				return res, [tree_idx]

		if op != '==':
			return tm, []

		same_tree_idx = tm.find_subtree(ltree_idx, rtree_idx, '*')
		if same_tree_idx != -1:
			res = tm.clone()
			same_tree = res.get_tree(same_tree_idx)
			same_tree_parent_idx = same_tree.parent
			same_tree_parent = res.get_tree(same_tree_parent_idx)
			same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
			parent_tree_child_idx = res.find_child_idx(same_tree_parent.parent, same_tree_parent_idx)
			res.trees[rtree_idx].root = '1'
			res.trees[rtree_idx].is_const = True
			res.trees[rtree_idx].children = []
			res.update_edge(same_tree_parent.parent, same_tree_parent.children[1 - same_tree_child_idx], parent_tree_child_idx)
			return res, [same_tree_parent.parent, rtree_idx]

		same_tree_idx = tm.find_subtree(rtree_idx, ltree_idx, '*')
		if same_tree_idx != -1:
			res = tm.clone()
			same_tree = res.get_tree(same_tree_idx)
			same_tree_parent_idx = same_tree.parent
			same_tree_parent = res.get_tree(same_tree_parent_idx)
			same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
			parent_tree_child_idx = res.find_child_idx(same_tree_parent.parent, same_tree_parent_idx)
			res.trees[ltree_idx].root = '1'
			res.trees[ltree_idx].is_const = True
			res.trees[ltree_idx].children = []
			res.update_edge(same_tree_parent.parent, same_tree_parent.children[1 - same_tree_child_idx], parent_tree_child_idx)
			return res, [same_tree_parent.parent, ltree_idx]

		return tm, []


	def left_association(self, tm, tree_idx): #v0 op (v1 op v2) -> (v0 op v1) op v2
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['+', '-', '*', '&&', '||']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)
		if (op == '+') and (rtree.root in ['+', '-']):
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			res.update_edge(tree_idx, rtree.children[0], 1)
			res.update_edge(rtree_idx, tree_idx, 0)
			res.update_edge(parent_idx, rtree_idx, child_idx)
			return res, [tree_idx]
		if (op == '-') and (rtree.root in ['+', '-']):
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			res.update_edge(tree_idx, rtree.children[0], 1)
			res.update_edge(rtree_idx, tree_idx, 0)
			if rtree.root == '+':
				res.trees[rtree_idx].root = '-'
			else:
				res.trees[rtree_idx].root = '+'
			res.update_edge(parent_idx, rtree_idx, child_idx)
			return res, [tree_idx]
		if op == rtree.root:
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			res.update_edge(tree_idx, rtree.children[0], 1)
			res.update_edge(rtree_idx, tree_idx, 0)
			res.update_edge(parent_idx, rtree_idx, child_idx)
			return res, [tree_idx]
		return tm, []


	def association(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['+', '-', '&&', '||']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)

		if op in ['+', '-']:
			if not (ltree.root == '*' and rtree.root == '*'):
				return tm, []
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = tm.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = tm.get_tree(rtree_rtree_idx)
			if tm.equal_tree(ltree_ltree, rtree_ltree) or tm.equal_tree(ltree_ltree, rtree_rtree):
				res = tm.clone()
				res.trees[tree_idx].root = '*'
				res.update_edge(tree_idx, ltree_ltree_idx, 0)
				res.trees[rtree_idx].root = op
				res.update_edge(rtree_idx, ltree_rtree_idx, 0)
				if tm.equal_tree(ltree_ltree, rtree_rtree):
					res.update_edge(rtree_idx, rtree_ltree_idx, 1)
				return res, [rtree_idx]
			elif tm.equal_tree(ltree_rtree, rtree_ltree) or tm.equal_tree(ltree_rtree, rtree_rtree):
				res = tm.clone()
				res.trees[tree_idx].root = '*'
				res.update_edge(tree_idx, ltree_rtree_idx, 0)
				res.trees[rtree_idx].root = op
				res.update_edge(rtree_idx, ltree_ltree_idx, 0)
				if tm.equal_tree(ltree_rtree, rtree_rtree):
					res.update_edge(rtree_idx, rtree_ltree_idx, 1)
				return res, [rtree_idx]
			return tm, []

		if op in ['&&', '||']:
			if ltree.root != rtree.root or (not ltree.root in ['&&', '||', '!']):
				return tm, []
			if ltree.root == '!':
				res = tm.clone()
				ltree_child_idx = ltree.children[0]
				ltree_child = res.get_tree(ltree_child_idx)
				rtree_child_idx = rtree.children[0]
				rtree_child = res.get_tree(rtree_child_idx)
				res.update_edge(tree_idx, ltree_child_idx, 0)
				res.update_edge(tree_idx, rtree_child_idx, 1)
				if op == '&&':
					res.trees[tree_idx].root = '||'
				else:
					res.trees[tree_idx].root = '&&'
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				res.update_edge(ltree_idx, tree_idx, 0)
				res.update_edge(parent_idx, ltree_idx, child_idx)
				return res, [tree_idx]

			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = tm.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = tm.get_tree(rtree_rtree_idx)
			if tm.equal_tree(ltree_ltree, rtree_ltree) or tm.equal_tree(ltree_ltree, rtree_rtree):
				res = tm.clone()
				res.trees[tree_idx].root = ltree.root
				res.update_edge(tree_idx, ltree_ltree_idx, 0)
				res.trees[rtree_idx].root = op
				res.update_edge(rtree_idx, ltree_rtree_idx, 0)
				if tm.equal_tree(ltree_ltree, rtree_rtree):
					res.update_edge(rtree_idx, rtree_ltree_idx, 1)
				return res, [rtree_idx]
			elif tm.equal_tree(ltree_rtree, rtree_ltree) or tm.equal_tree(ltree_rtree, rtree_rtree):
				res = tm.clone()
				res.trees[tree_idx].root = ltree.root
				res.update_edge(tree_idx, ltree_rtree_idx, 0)
				res.trees[rtree_idx].root = op
				res.update_edge(rtree_idx, ltree_ltree_idx, 0)
				if tm.equal_tree(ltree_rtree, rtree_rtree):
					res.update_edge(rtree_idx, rtree_ltree_idx, 1)
				return res, [rtree_idx]
			return tm, []

		return tm, []


	def muldiv_elimination(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['<', '<=', '==']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)

		if ltree.root == '+':
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			if ltree_ltree.root == '*' and ltree_rtree.is_const and rtree.root == '*':
				ltree_ltree_ltree_idx = ltree_ltree.children[0]
				ltree_ltree_rtree_idx = ltree_ltree.children[1]
				ltree_ltree_ltree = tm.get_tree(ltree_ltree_ltree_idx)
				ltree_ltree_rtree = tm.get_tree(ltree_ltree_rtree_idx)
				rtree_ltree_idx = rtree.children[0]
				rtree_rtree_idx = rtree.children[1]
				rtree_ltree = tm.get_tree(rtree_ltree_idx)
				rtree_rtree = tm.get_tree(rtree_rtree_idx)
				if ltree_ltree_rtree.is_const and rtree_rtree.is_const and ltree_ltree_rtree.root == rtree_rtree.root and int(rtree_rtree.root) > 0:
					res = tm.clone()
					if int(ltree_rtree.root) % int(rtree_rtree.root) != 0 and op == '==':
						res.trees[tree_idx].root = '0'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(ltree_idx, ltree_ltree_ltree_idx, 0)
					res.update_edge(tree_idx, rtree_ltree_idx, 1)
					if int(ltree_rtree.root) % int(rtree_rtree.root) == 0 or op == '<':
						res.trees[ltree_rtree_idx].root = str(int(ltree_rtree.root) // int(rtree_rtree.root))
					else:
						res.trees[ltree_rtree_idx].root = str(int(ltree_rtree.root) // int(rtree_rtree.root) + 1)
					return res, [ltree_rtree_idx]

		if rtree.root == '+':
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = tm.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = tm.get_tree(rtree_rtree_idx)
			if rtree_ltree.root == '*' and rtree_rtree.is_const and ltree.root == '*':
				rtree_ltree_ltree_idx = rtree_ltree.children[0]
				rtree_ltree_rtree_idx = rtree_ltree.children[1]
				rtree_ltree_ltree = tm.get_tree(rtree_ltree_ltree_idx)
				rtree_ltree_rtree = tm.get_tree(rtree_ltree_rtree_idx)
				ltree_ltree_idx = ltree.children[0]
				ltree_rtree_idx = ltree.children[1]
				ltree_ltree = tm.get_tree(ltree_ltree_idx)
				ltree_rtree = tm.get_tree(ltree_rtree_idx)
				if rtree_ltree_rtree.is_const and ltree_rtree.is_const and rtree_ltree_rtree.root == ltree_rtree.root and int(ltree_rtree.root) > 0:
					res = tm.clone()
					if int(rtree_rtree.root) % int(ltree_rtree.root) != 0 and op == '==':
						res.trees[tree_idx].root = '0'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(rtree_idx, rtree_ltree_ltree_idx, 0)
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					if int(rtree_rtree.root) % int(ltree_rtree.root) == 0 or op == '<=':
						res.trees[rtree_rtree_idx].root = str(int(rtree_rtree.root) // int(ltree_rtree.root))
					else:
						res.trees[rtree_rtree_idx].root = str(int(rtree_rtree.root) // int(ltree_rtree.root) + 1)
					return res, [rtree_rtree_idx]

		if not (ltree.root in ['*', '/']) or not (rtree.root in ['*', '/']):
			return tm, []

		ltree_ltree_idx = ltree.children[0]
		ltree_ltree = tm.get_tree(ltree_ltree_idx)
		ltree_rtree_idx = ltree.children[1]
		ltree_rtree = tm.get_tree(ltree_rtree_idx)
		rtree_ltree_idx = rtree.children[0]
		rtree_ltree = tm.get_tree(rtree_ltree_idx)
		rtree_rtree_idx = rtree.children[1]
		rtree_rtree = tm.get_tree(rtree_rtree_idx)
		if ltree_ltree.is_const + ltree_rtree.is_const != 1 or rtree_ltree.is_const + rtree_rtree.is_const != 1:
			return tm, []

		if ltree.root == rtree.root and ltree.root == '/':
			if tm.equal_tree(ltree_ltree, rtree_ltree):
				res = tm.clone()
				if ltree_ltree.is_const:
					v = int(ltree_ltree.root)
					if v == 0:
						if op == '<':
							res.trees[tree_idx].root = '0'
						else:
							res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(tree_idx, ltree_rtree_idx, 0)
					res.update_edge(tree_idx, rtree_rtree_idx, 1)
					if v > 0:
						res.trees[tree_idx].children.reverse()
					return res, [tree_idx]
				else:
					v1 = int(ltree_rtree.root)
					v2 = int(rtree_rtree.root)
					if v1 == v2:
						if op == '<':
							res.trees[tree_idx].root = '0'
						else:
							res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					res.trees[rtree_idx].root = '0'
					res.trees[rtree_idx].is_const = True
					res.trees[rtree_idx].children = []
					if v1 > v2:
						res.trees[tree_idx].children.reverse()
					return res, [rtree_idx]
			elif tm.equal_tree(ltree_rtree, rtree_rtree):
				res = tm.clone()
				if ltree_rtree.is_const:
					v = int(ltree_rtree.root)
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					res.update_edge(tree_idx, rtree_ltree_idx, 1)
					if v < 0:
						res.trees[tree_idx].children.reverse()
					return res, [tree_idx]
				else:
					v1 = int(ltree_ltree.root)
					v2 = int(rtree_ltree.root)
					if v1 == v2:
						if op == '<':
							res.trees[tree_idx].root = '0'
						else:
							res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(tree_idx, ltree_rtree_idx, 0)
					res.trees[rtree_idx].root = '0'
					res.trees[rtree_idx].is_const = True
					res.trees[rtree_idx].children = []
					if v1 < v2:
						res.trees[tree_idx].children.reverse()
					return res, [rtree_idx]
		elif ltree.root == '*' and rtree.root == '/':
			if tm.equal_tree(ltree_ltree, rtree_ltree):
				res = tm.clone()
				if ltree_ltree.is_const:
					v = int(ltree_ltree.root)
					if v == 0:
						if op == '<':
							res.trees[tree_idx].root = '0'
						else:
							res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(tree_idx, ltree_rtree_idx, 0)
					res.trees[rtree_ltree_idx].root = '1'
					res.trees[rtree_ltree_idx].is_const = True
					res.trees[rtree_ltree_idx].children = []
					if v < 0:
						res.trees[tree_idx].children.reverse()
					return res, [rtree_ltree_idx]
				else:
					v1 = int(ltree_rtree.root)
					v2 = 1.0 / int(rtree_rtree.root)
					if v1 == v2:
						if op == '<':
							res.trees[tree_idx].root = '0'
						else:
							res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					res.trees[rtree_idx].root = '0'
					res.trees[rtree_idx].is_const = True
					res.trees[rtree_idx].children = []
					if v1 < v2:
						res.trees[tree_idx].children.reverse()
					return res, [rtree_idx]
			elif tm.equal_tree(ltree_rtree, rtree_ltree):
				res = tm.clone()
				if ltree_rtree.is_const:
					v = int(ltree_rtree.root)
					if v == 0:
						if op == '<':
							res.trees[tree_idx].root = '0'
						else:
							res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					res.trees[rtree_ltree_idx].root = '1'
					res.trees[rtree_ltree_idx].is_const = True
					res.trees[rtree_ltree_idx].children = []
					if v < 0:
						res.trees[tree_idx].children.reverse()
					return res, [rtree_ltree_idx]
				else:
					v1 = int(ltree_ltree.root)
					v2 = 1.0 / int(rtree_rtree.root)
					if v1 == v2:
						if op == '<':
							res.trees[tree_idx].root = '0'
						else:
							res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(tree_idx, ltree_rtree_idx, 0)
					res.trees[rtree_idx].root = '0'
					res.trees[rtree_idx].is_const = True
					res.trees[rtree_idx].children = []
					if v1 < v2:
						res.trees[tree_idx].children.reverse()
					return res, [rtree_idx]
		elif ltree.root == '/' and rtree.root == '*':
			if tm.equal_tree(rtree_ltree, ltree_ltree):
				res = tm.clone()
				if rtree_ltree.is_const:
					v = int(rtree_ltree.root)
					if v == 0:
						if op == '<':
							res.trees[tree_idx].root = '0'
						else:
							res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(tree_idx, rtree_rtree_idx, 1)
					res.trees[ltree_ltree_idx].root = '1'
					res.trees[ltree_ltree_idx].is_const = True
					res.trees[ltree_ltree_idx].children = []
					if v < 0:
						res.trees[tree_idx].children.reverse()
					return res, [ltree_ltree_idx]
				else:
					v1 = int(ltree_rtree.root)
					v2 = 1.0 / int(rtree_rtree.root)
					if v1 == v2:
						if op == '<':
							res.trees[tree_idx].root = '0'
						else:
							res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					res.trees[rtree_idx].root = '0'
					res.trees[rtree_idx].is_const = True
					res.trees[rtree_idx].children = []
					if v1 < v2:
						res.trees[tree_idx].children.reverse()
					return res, [rtree_idx]
			elif tm.equal_tree(rtree_rtree, ltree_ltree):
				res = tm.clone()
				if rtree_rtree.is_const:
					v = int(rtree_rtree.root)
					if v == 0:
						if op == '<':
							res.trees[tree_idx].root = '0'
						else:
							res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(tree_idx, rtree_ltree_idx, 1)
					res.trees[ltree_ltree_idx].root = '1'
					res.trees[ltree_ltree_idx].is_const = True
					res.trees[ltree_ltree_idx].children = []
					if v < 0:
						res.trees[tree_idx].children.reverse()
					return res, [ltree_ltree_idx]
				else:
					v1 = int(ltree_rtree.root)
					v2 = 1.0 / int(rtree_ltree.root)
					if v1 == v2:
						if op == '<':
							res.trees[tree_idx].root = '0'
						else:
							res.trees[tree_idx].root = '1'
						res.trees[tree_idx].is_const = True
						res.trees[tree_idx].children = []
						return res, [tree_idx]
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					res.trees[rtree_idx].root = '0'
					res.trees[rtree_idx].is_const = True
					res.trees[rtree_idx].children = []
					if v1 < v2:
						res.trees[tree_idx].children.reverse()
					return res, [rtree_idx]
			return tm, []

		return tm, []


	def div_reduction(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if op != '/':
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)
		if ltree.root == '*' and rtree.is_const and int(rtree.root) > 0:
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			if ltree_rtree.is_const and int(ltree_rtree.root) > 0:
				g = gcd(int(rtree.root), int(ltree_rtree.root))
				if g > 1:
					res = tm.clone()
					res.trees[ltree_rtree_idx].root = str(int(ltree_rtree.root) // g)
					res.trees[rtree_idx].root = str(int(rtree.root) // g)
					return res, [ltree_rtree_idx, rtree_idx]
		return tm, []


	def muldiv_to_mod_transformation(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['<', '<=', '==', '-']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)

		idxes = tm.find_muldiv_term(ltree_idx)
		for tmp_idx in idxes:
			tmp_tree = tm.get_tree(tmp_idx)
			tmp_tree_ltree_idx = tmp_tree.children[0]
			tmp_tree_ltree = tm.get_tree(tmp_tree_ltree_idx)
			tmp_tree_rtree_idx = tmp_tree.children[1]
			tmp_tree_rtree = tm.get_tree(tmp_tree_rtree_idx)
			tmp_tree_ltree_ltree_idx = tmp_tree_ltree.children[0]
			tmp_tree_ltree_ltree = tm.get_tree(tmp_tree_ltree_ltree_idx)
			same_tree_idx = tm.find_subtree(rtree_idx, tmp_tree_ltree_ltree_idx, '+')
			if same_tree_idx != -1:
				res = tm.clone()
				same_tree = res.get_tree(same_tree_idx)
				same_tree_parent_idx = same_tree.parent
				same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
				new_tree_idx = res.create_tree('%', same_tree_parent_idx, same_tree.depth)
				res.update_edge(new_tree_idx, same_tree_idx, 0)
				res.update_edge(new_tree_idx, tmp_tree_rtree_idx, 1)
				res.update_edge(same_tree_parent_idx, new_tree_idx, same_tree_child_idx)
				tmp_tree_parent_idx = tmp_tree.parent
				tmp_tree_parent = res.get_tree(tmp_tree_parent_idx)
				tmp_tree_child_idx = res.find_child_idx(tmp_tree_parent_idx, tmp_idx)
				parent_tmp_tree_child_idx = res.find_child_idx(tmp_tree_parent.parent, tmp_tree_parent_idx)
				if tmp_tree_parent.root == '-' and tmp_tree_child_idx == 0 or tmp_idx == ltree_idx:
					res.trees[tmp_idx].root = '0'
					res.trees[tmp_idx].is_const = True
					res.trees[tmp_idx].children = []
					return res, [tmp_idx, new_tree_idx]
				else:
					res.update_edge(tmp_tree_parent.parent, tmp_tree_parent.children[1 - tmp_tree_child_idx], parent_tmp_tree_child_idx)
					return res, [tmp_tree_parent.parent, new_tree_idx]
			if tmp_tree_ltree_ltree.root == '+':
				tmp_tree_ltree_ltree_ltree_idx = tmp_tree_ltree_ltree.children[0]
				tmp_tree_ltree_ltree_ltree = tm.get_tree(tmp_tree_ltree_ltree_ltree_idx)
				tmp_tree_ltree_ltree_rtree_idx = tmp_tree_ltree_ltree.children[1]
				tmp_tree_ltree_ltree_rtree = tm.get_tree(tmp_tree_ltree_ltree_rtree_idx)
				same_tree_idx = tm.find_subtree(rtree_idx, tmp_tree_ltree_ltree_ltree_idx, '+')
				if tmp_tree_ltree_ltree_rtree.is_const and same_tree_idx != -1:
					res = tm.clone()
					same_tree = res.get_tree(same_tree_idx)
					same_tree_parent_idx = same_tree.parent
					same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
					new_tree_idx = res.create_tree('%', same_tree_parent_idx, same_tree.depth)
					new_tree_ltree_idx = res.create_tree('+', new_tree_idx, same_tree.depth + 1)
					new_tree_ltree_rtree_idx = res.create_tree(tmp_tree_ltree_ltree_rtree.root, new_tree_ltree_idx, same_tree.depth + 2)
					res.update_edge(new_tree_ltree_idx, same_tree_idx, 0)
					res.update_edge(new_tree_ltree_idx, new_tree_ltree_rtree_idx, 1)
					res.update_edge(new_tree_idx, new_tree_ltree_idx, 0)
					res.update_edge(new_tree_idx, tmp_tree_rtree_idx, 1)
					res.update_edge(same_tree_parent_idx, new_tree_idx, same_tree_child_idx)
					tmp_tree_parent_idx = tmp_tree.parent
					tmp_tree_parent = res.get_tree(tmp_tree_parent_idx)
					tmp_tree_child_idx = res.find_child_idx(tmp_tree_parent_idx, tmp_idx)
					res.update_edge(tmp_tree_parent_idx, tmp_tree_ltree_ltree_rtree_idx, tmp_tree_child_idx)
					return res, [tmp_tree_parent_idx, new_tree_ltree_rtree_idx]

		idxes = tm.find_muldiv_term(rtree_idx)
		for tmp_idx in idxes:
			tmp_tree = tm.get_tree(tmp_idx)
			tmp_tree_ltree_idx = tmp_tree.children[0]
			tmp_tree_ltree = tm.get_tree(tmp_tree_ltree_idx)
			tmp_tree_rtree_idx = tmp_tree.children[1]
			tmp_tree_rtree = tm.get_tree(tmp_tree_rtree_idx)
			tmp_tree_ltree_ltree_idx = tmp_tree_ltree.children[0]
			tmp_tree_ltree_ltree = tm.get_tree(tmp_tree_ltree_ltree_idx)
			same_tree_idx = tm.find_subtree(ltree_idx, tmp_tree_ltree_ltree_idx, '+')
			if same_tree_idx != -1:
				res = tm.clone()
				same_tree = res.get_tree(same_tree_idx)
				same_tree_parent_idx = same_tree.parent
				same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
				new_tree_idx = res.create_tree('%', same_tree_parent_idx, same_tree.depth)
				res.update_edge(new_tree_idx, same_tree_idx, 0)
				res.update_edge(new_tree_idx, tmp_tree_rtree_idx, 1)
				res.update_edge(same_tree_parent_idx, new_tree_idx, same_tree_child_idx)
				tmp_tree_parent_idx = tmp_tree.parent
				tmp_tree_parent = res.get_tree(tmp_tree_parent_idx)
				tmp_tree_child_idx = res.find_child_idx(tmp_tree_parent_idx, tmp_idx)
				parent_tmp_tree_child_idx = res.find_child_idx(tmp_tree_parent.parent, tmp_tree_parent_idx)
				if tmp_tree_parent.root == '-' and tmp_tree_child_idx == 0 or tmp_idx == rtree_idx:
					res.trees[tmp_idx].root = '0'
					res.trees[tmp_idx].is_const = True
					res.trees[tmp_idx].children = []
					return res, [tmp_idx, new_tree_idx]
				else:
					res.update_edge(tmp_tree_parent.parent, tmp_tree_parent.children[1 - tmp_tree_child_idx], parent_tmp_tree_child_idx)
					return res, [tmp_tree_parent.parent, new_tree_idx]
			if tmp_tree_ltree_ltree.root == '+':
				tmp_tree_ltree_ltree_ltree_idx = tmp_tree_ltree_ltree.children[0]
				tmp_tree_ltree_ltree_ltree = tm.get_tree(tmp_tree_ltree_ltree_ltree_idx)
				tmp_tree_ltree_ltree_rtree_idx = tmp_tree_ltree_ltree.children[1]
				tmp_tree_ltree_ltree_rtree = tm.get_tree(tmp_tree_ltree_ltree_rtree_idx)
				same_tree_idx = tm.find_subtree(ltree_idx, tmp_tree_ltree_ltree_ltree_idx, '+')
				if tmp_tree_ltree_ltree_rtree.is_const and same_tree_idx != -1:
					res = tm.clone()
					same_tree = res.get_tree(same_tree_idx)
					same_tree_parent_idx = same_tree.parent
					same_tree_child_idx = res.find_child_idx(same_tree_parent_idx, same_tree_idx)
					new_tree_idx = res.create_tree('%', same_tree_parent_idx, same_tree.depth)
					new_tree_ltree_idx = res.create_tree('+', new_tree_idx, same_tree.depth + 1)
					new_tree_ltree_rtree_idx = res.create_tree(tmp_tree_ltree_ltree_rtree.root, new_tree_ltree_idx, same_tree.depth + 2)
					res.update_edge(new_tree_ltree_idx, same_tree_idx, 0)
					res.update_edge(new_tree_ltree_idx, new_tree_ltree_rtree_idx, 1)
					res.update_edge(new_tree_idx, new_tree_ltree_idx, 0)
					res.update_edge(new_tree_idx, tmp_tree_rtree_idx, 1)
					res.update_edge(same_tree_parent_idx, new_tree_idx, same_tree_child_idx)
					tmp_tree_parent_idx = tmp_tree.parent
					tmp_tree_parent = res.get_tree(tmp_tree_parent_idx)
					tmp_tree_child_idx = res.find_child_idx(tmp_tree_parent_idx, tmp_idx)
					res.update_edge(tmp_tree_parent_idx, tmp_tree_ltree_ltree_rtree_idx, tmp_tree_child_idx)
					return res, [tmp_tree_parent_idx, new_tree_ltree_rtree_idx]

		idxes = tm.find_muldiv_term(ltree_idx)
		for tmp_idx in idxes:
			tmp_tree = tm.get_tree(tmp_idx)
			tmp_tree_ltree_idx = tmp_tree.children[0]
			tmp_tree_rtree_idx = tmp_tree.children[1]
			tmp_tree_ltree = tm.get_tree(tmp_tree_ltree_idx)
			tmp_tree_rtree = tm.get_tree(tmp_tree_rtree_idx)
			tmp_tree_ltree_ltree_idx = tmp_tree_ltree.children[0]
			tmp_tree_ltree_ltree = tm.get_tree(tmp_tree_ltree_ltree_idx)
			res = tm.clone()
			res.trees[tmp_idx].root = '-'
			new_rtree_ltree_idx = res.clone_tree(res, tmp_tree_ltree_ltree_idx)
			new_rtree_idx = res.create_tree('%', tmp_idx, tmp_tree.depth + 1)
			res.update_edge(new_rtree_idx, new_rtree_ltree_idx, 0)
			res.update_edge(new_rtree_idx, tmp_tree_rtree_idx, 1)
			res.update_edge(tmp_idx, tmp_tree_ltree_ltree_idx, 0)
			res.update_edge(tmp_idx, new_rtree_idx, 1)
			return res, [new_rtree_idx]

		idxes = tm.find_muldiv_term(rtree_idx)
		for tmp_idx in idxes:
			tmp_tree = tm.get_tree(tmp_idx)
			tmp_tree_ltree_idx = tmp_tree.children[0]
			tmp_tree_rtree_idx = tmp_tree.children[1]
			tmp_tree_ltree = tm.get_tree(tmp_tree_ltree_idx)
			tmp_tree_rtree = tm.get_tree(tmp_tree_rtree_idx)
			tmp_tree_ltree_ltree_idx = tmp_tree_ltree.children[0]
			tmp_tree_ltree_ltree = tm.get_tree(tmp_tree_ltree_ltree_idx)
			res = tm.clone()
			res.trees[tmp_idx].root = '-'
			new_rtree_ltree_idx = res.clone_tree(res, tmp_tree_ltree_ltree_idx)
			new_rtree_idx = res.create_tree('%', tmp_idx, tmp_tree.depth + 1)
			res.update_edge(new_rtree_idx, new_rtree_ltree_idx, 0)
			res.update_edge(new_rtree_idx, tmp_tree_rtree_idx, 1)
			res.update_edge(tmp_idx, tmp_tree_ltree_ltree_idx, 0)
			res.update_edge(tmp_idx, new_rtree_idx, 1)
			return res, [new_rtree_idx]

		return tm, []


	def muldiv_association(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['+', '-']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)
		if ltree.is_const and int(ltree.root) > 0 and rtree.root in ['*', '/']:
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = tm.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = tm.get_tree(rtree_rtree_idx)
			if rtree.root == '/' and rtree_rtree.is_const:
				res = tm.clone()
				set_value = int(ltree.root) * int(rtree_rtree.root)
				res.trees[ltree_idx].root = str(set_value)
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				res.update_edge(parent_idx, rtree_idx, child_idx)
				res.update_edge(tree_idx, rtree_ltree_idx, 1)
				res.update_edge(rtree_idx, tree_idx, 0)
				return res, [ltree_idx]
			if rtree.root == '*' and rtree_ltree.is_const:
				if int(rtree_ltree.root) == 0:
					res = tm.clone()
					res.trees[rtree_idx].root = '0'
					res.trees[rtree_idx].is_const = True
					res.trees[rtree_idx].children = []
					return res, [rtree_idx]
				new_tree_value = int(ltree.root) // int(rtree_ltree.root)
				if  new_tree_value != 0:
					res = tm.clone()
					new_tree_idx = res.create_tree(op, rtree_idx, rtree.depth + 1)
					new_tree = res.get_tree(new_tree_idx)
					new_ltree_idx = res.create_tree(str(new_tree_value), new_tree_idx, new_tree.depth + 1)
					res.update_edge(new_tree_idx, new_ltree_idx, 0)
					res.update_edge(new_tree_idx, rtree_rtree_idx, 1)
					res.update_edge(rtree_idx, new_tree_idx, 1)
					res.trees[ltree_idx].root = str(int(ltree.root) % int(rtree_ltree.root))
					return res, [ltree_idx, new_ltree_idx]
			if rtree.root == '*' and rtree_rtree.is_const:
				if int(rtree_rtree.root) == 0:
					res = tm.clone()
					res.trees[rtree_idx].root = '0'
					res.trees[rtree_idx].is_const = True
					res.trees[rtree_idx].children = []
					return res, [rtree_idx]
				new_tree_value = int(ltree.root) // int(rtree_rtree.root)
				if  new_tree_value != 0:
					res = tm.clone()
					new_tree_idx = res.create_tree(op, rtree_idx, rtree.depth + 1)
					new_tree = res.get_tree(new_tree_idx)
					new_ltree_idx = res.create_tree(str(new_tree_value), new_tree_idx, new_tree.depth + 1)
					res.update_edge(new_tree_idx, new_ltree_idx, 0)
					res.update_edge(new_tree_idx, rtree_ltree_idx, 1)
					res.update_edge(rtree_idx, new_tree_idx, 0)
					res.trees[ltree_idx].root = str(int(ltree.root) % int(rtree_rtree.root))
					return res, [ltree_idx, new_ltree_idx]
		if rtree.is_const and ltree.root in ['*', '/']:
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			if ltree.root == '/' and ltree_rtree.is_const:
				res = tm.clone()
				set_value = int(rtree.root) * int(ltree_rtree.root)
				res.trees[rtree_idx].root = str(set_value)
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				res.update_edge(parent_idx, ltree_idx, child_idx)
				res.update_edge(tree_idx, ltree_ltree_idx, 0)
				res.update_edge(ltree_idx, tree_idx, 0)
				return res, [rtree_idx]
			if ltree.root == '*' and ltree_ltree.is_const:
				if int(ltree_ltree.root) == 0:
					res = tm.clone()
					res.trees[ltree_idx].root = '0'
					res.trees[ltree_idx].is_const = True
					res.trees[ltree_idx].children = []
					return res, [ltree_idx]
				new_tree_value = int(rtree.root) // int(ltree_ltree.root)
				if  new_tree_value != 0:
					res = tm.clone()
					new_tree_idx = res.create_tree(op, ltree_idx, ltree.depth + 1)
					new_tree = res.get_tree(new_tree_idx)
					new_rtree_idx = res.create_tree(str(new_tree_value), new_tree_idx, new_tree.depth + 1)
					res.update_edge(new_tree_idx, new_rtree_idx, 1)
					res.update_edge(new_tree_idx, ltree_rtree_idx, 0)
					res.update_edge(ltree_idx, new_tree_idx, 1)
					res.trees[rtree_idx].root = str(int(rtree.root) % int(ltree_ltree.root))
					return res, [rtree_idx, new_rtree_idx]
			if ltree.root == '*' and ltree_rtree.is_const:
				if int(ltree_rtree.root) == 0:
					res = tm.clone()
					res.trees[ltree_idx].root = '0'
					res.trees[ltree_idx].is_const = True
					res.trees[ltree_idx].children = []
					return res, [ltree_idx]
				new_tree_value = int(rtree.root) // int(ltree_rtree.root)
				if  new_tree_value != 0:
					res = tm.clone()
					new_tree_idx = res.create_tree(op, ltree_idx, ltree.depth + 1)
					new_tree = res.get_tree(new_tree_idx)
					new_rtree_idx = res.create_tree(str(new_tree_value), new_tree_idx, new_tree.depth + 1)
					res.update_edge(new_tree_idx, new_rtree_idx, 1)
					res.update_edge(new_tree_idx, ltree_ltree_idx, 0)
					res.update_edge(ltree_idx, new_tree_idx, 0)
					res.trees[rtree_idx].root = str(int(rtree.root) % int(ltree_rtree.root))
					return res, [rtree_idx, new_rtree_idx]
		return tm, []


	def muldiv_distribution(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['*', '/']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)
		if op == '*':
			if ltree.is_const and rtree.root in ['+', '-']:
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				rtree_ltree_idx = rtree.children[0]
				rtree_ltree = res.get_tree(rtree_ltree_idx)
				rtree_rtree_idx = rtree.children[1]
				rtree_rtree = res.get_tree(rtree_rtree_idx)
				new_rtree_idx = res.create_tree('*', rtree_idx, cur_tree.depth + 1)
				new_rtree_ltree_idx = res.create_tree(ltree.root, new_rtree_idx, cur_tree.depth + 2)
				res.update_edge(new_rtree_idx, new_rtree_ltree_idx, 0)
				res.update_edge(new_rtree_idx, rtree_rtree_idx, 1)
				res.update_edge(tree_idx, rtree_ltree_idx, 1)
				res.update_edge(parent_idx, rtree_idx, child_idx)
				res.update_edge(rtree_idx, tree_idx, 0)
				res.update_edge(rtree_idx, new_rtree_idx, 1)
				return res, [new_rtree_ltree_idx, tree_idx]
			if rtree.is_const and ltree.root in ['+', '-']:
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				ltree_ltree_idx = ltree.children[0]
				ltree_ltree = res.get_tree(ltree_ltree_idx)
				ltree_rtree_idx = ltree.children[1]
				ltree_rtree = res.get_tree(ltree_rtree_idx)
				new_rtree_idx = res.create_tree('*', ltree_idx, cur_tree.depth + 1)
				new_rtree_rtree_idx = res.create_tree(rtree.root, new_rtree_idx, cur_tree.depth + 2)
				res.update_edge(new_rtree_idx, ltree_rtree_idx, 0)
				res.update_edge(new_rtree_idx, new_rtree_rtree_idx, 1)
				res.update_edge(tree_idx, ltree_ltree_idx, 0)
				res.update_edge(parent_idx, ltree_idx, child_idx)
				res.update_edge(ltree_idx, tree_idx, 0)
				res.update_edge(ltree_idx, new_rtree_idx, 1)
				return res, [new_rtree_rtree_idx, tree_idx]
			return tm, []
		if op == '/' and rtree.is_const and ltree.root in ['+', '-']:
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			if tm.is_times(ltree_ltree_idx, int(rtree.root)) or tm.is_times(ltree_rtree_idx, int(rtree.root)):
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				ltree_ltree_idx = ltree.children[0]
				ltree_ltree = res.get_tree(ltree_ltree_idx)
				ltree_rtree_idx = ltree.children[1]
				ltree_rtree = res.get_tree(ltree_rtree_idx)
				new_rtree_idx = res.create_tree('/', ltree_idx, cur_tree.depth + 1)
				new_rtree_rtree_idx = res.create_tree(rtree.root, new_rtree_idx, cur_tree.depth + 2)
				res.update_edge(new_rtree_idx, ltree_rtree_idx, 0)
				res.update_edge(new_rtree_idx, new_rtree_rtree_idx, 1)
				res.update_edge(tree_idx, ltree_ltree_idx, 0)
				res.update_edge(parent_idx, ltree_idx, child_idx)
				res.update_edge(ltree_idx, tree_idx, 0)
				res.update_edge(ltree_idx, new_rtree_idx, 1)
				return res, [new_rtree_rtree_idx, tree_idx]
			return tm, []
		return tm, []


	def select_simplification(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if op != 'select':
			return tm, []
		cond_idx = cur_tree.children[0]
		cond = tm.get_tree(cond_idx)
		ltree_idx = cur_tree.children[1]
		rtree_idx = cur_tree.children[2]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)
		if cond.is_const:
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			if cond == '0':
				res.update_edge(parent_idx, rtree_idx, child_idx)
			else:
				res.update_edge(parent_idx, ltree_idx, child_idx)
			return res, [parent_idx]
		elif tm.equal_tree(ltree, rtree):
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			res.update_edge(parent_idx, ltree_idx, child_idx)
			return res, [parent_idx]
		if not (cond.root in ['<', '<=', '!']):
			return tm, []
		if cond.root == '!':
			res = tm.clone()
			cond_child_idx = cond.children[0]
			res.update_edge(tree_idx, cond_child_idx, 0)
			res.update_edge(tree_idx, ltree_idx, 2)
			res.update_edge(tree_idx, rtree_idx, 1)
			return res, [tree_idx]
		elif cond.root == '<=':
			res = tm.clone()
			res.trees[cond_idx].root = '<'
			res.trees[cond_idx].children.reverse()
			res.update_edge(tree_idx, ltree_idx, 2)
			res.update_edge(tree_idx, rtree_idx, 1)
			return res, [tree_idx]
		elif cond.root == '<':
			cond_l_child_idx = cond.children[0]
			cond_r_child_idx = cond.children[1]
			cond_l_child = tm.get_tree(cond_l_child_idx)
			cond_r_child = tm.get_tree(cond_r_child_idx)
			if tm.equal_tree(cond_l_child, ltree) and tm.equal_tree(cond_r_child, rtree):
				res = tm.clone()
				res.trees[tree_idx].root = 'min'
				res.update_edge(tree_idx, ltree_idx, 0)
				res.update_edge(tree_idx, rtree_idx, 1)
				res.trees[tree_idx].children.pop()
				return res, [tree_idx]
			elif tm.equal_tree(cond_r_child, ltree) and tm.equal_tree(cond_l_child, rtree):
				res = tm.clone()
				res.trees[tree_idx].root = 'max'
				res.update_edge(tree_idx, ltree_idx, 0)
				res.update_edge(tree_idx, rtree_idx, 1)
				res.trees[tree_idx].children.pop()
				return res, [tree_idx]
			return tm, []


	def select_association(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		if len(cur_tree.children) != 2:
			return tm, []
		op = cur_tree.root
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)
		if ltree.root != rtree.root or ltree.root != 'select':
			return tm, []
		l_cond_idx = ltree.children[0]
		l_cond = tm.get_tree(l_cond_idx)
		r_cond_idx = rtree.children[0]
		r_cond = tm.get_tree(r_cond_idx)
		if tm.equal_tree(l_cond, r_cond):
			res = tm.clone()
			new_ltree_idx = res.create_tree(op, ltree_idx, ltree.depth + 1)
			res.update_edge(new_ltree_idx, ltree.children[0], 0)
			res.update_edge(new_ltree_idx, rtree.children[0], 1)
			new_rtree_idx = res.create_tree(op, ltree_idx, ltree.depth + 1)
			res.update_edge(new_rtree_idx, ltree.children[1], 0)
			res.update_edge(new_rtree_idx, rtree.children[1], 1)
			res.update_edge(ltree_idx, new_ltree_idx, 1)
			res.update_edge(ltree_idx, new_rtree_idx, 2)
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			res.update_edge(parent_idx, ltree_idx, child_idx)
			return res, [new_ltree_idx, new_rtree_idx]
		
		if l_cond.root != r_cond.root or l_cond.root != '<':
			return tm, []

		l_cond_ltree_idx = l_cond.children[0]
		l_cond_ltree = tm.get_tree(l_cond_ltree_idx)
		l_cond_rtree_idx = l_cond.children[1]
		l_cond_rtree = tm.get_tree(l_cond_rtree_idx)
		r_cond_ltree_idx = r_cond.children[0]
		r_cond_ltree = tm.get_tree(r_cond_ltree_idx)
		r_cond_rtree_idx = r_cond.children[1]
		r_cond_rtree = tm.get_tree(r_cond_rtree_idx)
		if tm.equal_tree(l_cond_ltree, r_cond_rtree) and tm.equal_tree(l_cond_rtree, r_cond_ltree):
			res = tm.clone()
			new_ltree_idx = res.create_tree(op, ltree_idx, ltree.depth + 1)
			res.update_edge(new_ltree_idx, ltree.children[0], 0)
			res.update_edge(new_ltree_idx, rtree.children[1], 1)
			new_rtree_idx = res.create_tree(op, ltree_idx, ltree.depth + 1)
			res.update_edge(new_rtree_idx, ltree.children[1], 0)
			res.update_edge(new_rtree_idx, rtree.children[0], 1)
			res.update_edge(ltree_idx, new_ltree_idx, 1)
			res.update_edge(ltree_idx, new_rtree_idx, 2)
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			res.update_edge(parent_idx, ltree_idx, child_idx)
			return res, [new_ltree_idx, new_rtree_idx]
		return tm, []


	def select_distribution(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['<', '<=']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		rtree_idx = cur_tree.children[1]
		ltree = tm.get_tree(ltree_idx)
		rtree = tm.get_tree(rtree_idx)
		if ltree.root != 'select' and rtree.root != 'select':
			return tm, []

		if ltree.root == 'select':
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			cond_idx = ltree.children[0]
			cond = res.get_tree(cond_idx)
			ltree_ltree_idx = ltree.children[1]
			ltree_ltree = res.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[2]
			ltree_rtree = res.get_tree(ltree_rtree_idx)
			new_tree_idx = res.create_tree('||', parent_idx, cur_tree.depth)
			new_ltree_idx = res.create_tree('&&', new_tree_idx, cur_tree.depth + 1)
			res.update_edge(tree_idx, ltree_ltree_idx, 0)
			res.update_edge(new_ltree_idx, cond_idx, 0)
			res.update_edge(new_ltree_idx, tree_idx, 1)
			new_rtree_idx = res.create_tree('&&', new_tree_idx, cur_tree.depth + 1)
			new_cond_idx = res.create_tree('!', new_rtree_idx, cur_tree.depth + 2)
			clone_cond_idx = res.clone_tree(res, cond_idx)
			res.update_edge(new_cond_idx, clone_cond_idx, 0)
			new_rtree_rtree_idx = res.create_tree(op, new_rtree_idx, cur_tree.depth + 2)
			new_rtree_rtree_rtree_idx = res.clone_tree(res, rtree_idx)
			res.update_edge(new_rtree_rtree_idx, ltree_rtree_idx, 0)
			res.update_edge(new_rtree_rtree_idx, new_rtree_rtree_rtree_idx, 1)
			res.update_edge(new_rtree_idx, new_cond_idx, 0)
			res.update_edge(new_rtree_idx, new_rtree_rtree_idx, 1)
			res.update_edge(new_tree_idx, new_ltree_idx, 0)
			res.update_edge(new_tree_idx, new_rtree_idx, 1)
			res.update_edge(parent_idx, new_tree_idx, child_idx)
			return res, [tree_idx, new_cond_idx, new_rtree_rtree_idx]
		else:
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			cond_idx = rtree.children[0]
			cond = res.get_tree(cond_idx)
			rtree_ltree_idx = rtree.children[1]
			rtree_ltree = res.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[2]
			rtree_rtree = res.get_tree(rtree_rtree_idx)
			new_tree_idx = res.create_tree('||', parent_idx, cur_tree.depth)
			new_ltree_idx = res.create_tree('&&', new_tree_idx, cur_tree.depth + 1)
			res.update_edge(tree_idx, rtree_ltree_idx, 1)
			res.update_edge(new_ltree_idx, cond_idx, 0)
			res.update_edge(new_ltree_idx, tree_idx, 1)
			new_rtree_idx = res.create_tree('&&', new_tree_idx, cur_tree.depth + 1)
			new_cond_idx = res.create_tree('!', new_rtree_idx, cur_tree.depth + 2)
			clone_cond_idx = res.clone_tree(res, cond_idx)
			res.update_edge(new_cond_idx, clone_cond_idx, 0)
			new_rtree_rtree_idx = res.create_tree(op, new_rtree_idx, cur_tree.depth + 2)
			new_rtree_rtree_ltree_idx = res.clone_tree(res, ltree_idx)
			res.update_edge(new_rtree_rtree_idx, new_rtree_rtree_ltree_idx, 0)
			res.update_edge(new_rtree_rtree_idx, rtree_rtree_idx, 1)
			res.update_edge(new_rtree_idx, new_cond_idx, 0)
			res.update_edge(new_rtree_idx, new_rtree_rtree_idx, 1)
			res.update_edge(new_tree_idx, new_ltree_idx, 0)
			res.update_edge(new_tree_idx, new_rtree_idx, 1)
			res.update_edge(parent_idx, new_tree_idx, child_idx)
			return res, [tree_idx, new_cond_idx, new_rtree_rtree_idx]

		return tm, []


	def minmax_simplification(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['min', 'max']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		ltree = tm.get_tree(ltree_idx)
		rtree_idx = cur_tree.children[1]
		rtree = tm.get_tree(rtree_idx)

		same_tree_idx = tm.find_subtree(ltree_idx, rtree_idx, op)
		if same_tree_idx != -1:
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			res.update_edge(parent_idx, ltree_idx, child_idx)
			return res, [parent_idx]

		if op == rtree.root:
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			res.update_edge(tree_idx, rtree.children[0], 1)
			res.update_edge(rtree_idx, tree_idx, 0)
			res.update_edge(parent_idx, rtree_idx, child_idx)
			return res, [tree_idx]

		if ltree.root in ['+', '-']:
			select_left = None
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			if tm.equal_tree(ltree_ltree, rtree) and ltree_rtree.is_const:
				v = int(ltree_rtree.root)
				if ltree.root == '-':
					v = -v
				if op == 'min' and v > 0 or op == 'max' and v < 0:
					select_left = False
				else:
					select_left = True
			if ltree.root == '+' and tm.equal_tree(ltree_rtree, rtree) and ltree_ltree.is_const:
				v = int(ltree_ltree.root)
				if op == 'min' and v > 0 or op == 'max' and v < 0:
					select_left = False
				else:
					select_left = True
			if select_left is not None:
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				if select_left is True:
					res.update_edge(parent_idx, ltree_idx, child_idx)
				else:
					res.update_edge(parent_idx, rtree_idx, child_idx)
				return res, [parent_idx]

		if rtree.root in ['+', '-']:
			select_left = None
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = tm.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = tm.get_tree(rtree_rtree_idx)
			if tm.equal_tree(rtree_ltree, ltree) and rtree_rtree.is_const:
				v = int(rtree_rtree.root)
				if rtree.root == '-':
					v = -v
				if op == 'min' and v > 0 or op == 'max' and v < 0:
					select_left = True
				else:
					select_left = False
			if rtree.root == '+' and tm.equal_tree(rtree_rtree, ltree) and rtree_ltree.is_const:
				v = int(rtree_ltree.root)
				if op == 'min' and v > 0 or op == 'max' and v < 0:
					select_left = True
				else:
					select_left = False
			if select_left is not None:
				res = tm.clone()
				parent_idx = cur_tree.parent
				child_idx = res.find_child_idx(parent_idx, tree_idx)
				if select_left is True:
					res.update_edge(parent_idx, ltree_idx, child_idx)
				else:
					res.update_edge(parent_idx, rtree_idx, child_idx)
				return res, [parent_idx]

		if ltree.root != rtree.root or not (ltree.root in ['min', 'max']):
			return tm, []

		ltree_ltree_idx = ltree.children[0]
		ltree_ltree = tm.get_tree(ltree_ltree_idx)
		ltree_rtree_idx = ltree.children[1]
		ltree_rtree = tm.get_tree(ltree_rtree_idx)
		rtree_ltree_idx = rtree.children[0]
		rtree_ltree = tm.get_tree(rtree_ltree_idx)
		rtree_rtree_idx = rtree.children[1]
		rtree_rtree = tm.get_tree(rtree_rtree_idx)
		if tm.equal_tree(ltree_ltree, rtree_ltree) or tm.equal_tree(ltree_ltree, rtree_rtree):
			res = tm.clone()
			res.trees[tree_idx].root = ltree.root
			res.update_edge(tree_idx, ltree_ltree_idx, 0)
			res.trees[rtree_idx].root = op
			res.update_edge(rtree_idx, ltree_rtree_idx, 0)
			if tm.equal_tree(ltree_ltree, rtree_rtree):
				res.update_edge(rtree_idx, rtree_ltree_idx, 1)
			return res, [rtree_idx]
		if tm.equal_tree(ltree_rtree, rtree_ltree) or tm.equal_tree(ltree_rtree, rtree_rtree):
			res = tm.clone()
			res.trees[tree_idx].root = ltree.root
			res.update_edge(tree_idx, ltree_rtree_idx, 0)
			res.trees[rtree_idx].root = op
			res.update_edge(rtree_idx, ltree_ltree_idx, 0)
			if tm.equal_tree(ltree_rtree, rtree_rtree):
				res.update_edge(rtree_idx, rtree_ltree_idx, 1)
			return res, [rtree_idx]
		return tm, []


	def minmax_alg_distribution(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		if len(cur_tree.children) != 2:
			return tm, []
		op = cur_tree.root
		ltree_idx = cur_tree.children[0]
		ltree = tm.get_tree(ltree_idx)
		rtree_idx = cur_tree.children[1]
		rtree = tm.get_tree(rtree_idx)
		if not (ltree.root in ['min', 'max']) and not (rtree.root in ['min', 'max']):
			return tm, []

		if  (ltree.root in ['min', 'max']) and (rtree.root in ['min', 'max']) and ltree.root != rtree.root:
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = tm.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = tm.get_tree(ltree_rtree_idx)
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = tm.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = tm.get_tree(rtree_rtree_idx)

			if op in ['+', '*', '==', 'max', 'min']:
				if tm.equal_tree(ltree_ltree, rtree_ltree) and tm.equal_tree(ltree_rtree, rtree_rtree) or \
				tm.equal_tree(ltree_rtree, rtree_ltree) and tm.equal_tree(ltree_ltree, rtree_rtree):
					res = tm.clone()
					res.update_edge(tree_idx, ltree_ltree_idx, 0)
					res.update_edge(tree_idx, ltree_rtree_idx, 1)
					return res, [tree_idx]

		if not (op in ['+', '-']):
			return tm, []

		if ltree.root in ['min', 'max']:
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = res.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = res.get_tree(ltree_rtree_idx)
			new_rtree_idx = res.create_tree(op, ltree_idx, cur_tree.depth + 1)
			new_rtree_rtree_idx = res.clone_tree(res, rtree_idx)
			res.update_edge(new_rtree_idx, ltree_rtree_idx, 0)
			res.update_edge(new_rtree_idx, new_rtree_rtree_idx, 1)
			res.update_edge(tree_idx, ltree_ltree_idx, 0)
			res.update_edge(parent_idx, ltree_idx, child_idx)
			res.update_edge(ltree_idx, tree_idx, 0)
			res.update_edge(ltree_idx, new_rtree_idx, 1)
			return res, [new_rtree_idx, tree_idx]

		if rtree.root in ['min', 'max']:
			res = tm.clone()
			parent_idx = cur_tree.parent
			child_idx = res.find_child_idx(parent_idx, tree_idx)
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = res.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = res.get_tree(rtree_rtree_idx)
			new_rtree_idx = res.create_tree(op, rtree_idx, cur_tree.depth + 1)
			new_rtree_ltree_idx = res.clone_tree(res, ltree_idx)
			res.update_edge(new_rtree_idx, new_rtree_ltree_idx, 0)
			res.update_edge(new_rtree_idx, rtree_rtree_idx, 1)
			res.update_edge(tree_idx, rtree_ltree_idx, 1)
			res.update_edge(parent_idx, rtree_idx, child_idx)
			res.update_edge(rtree_idx, tree_idx, 0)
			res.update_edge(rtree_idx, new_rtree_idx, 1)
			if op == '-':
				if rtree.root == 'min':
					res.trees[rtree_idx].root = 'max'
				else:
					res.trees[rtree_idx].root = 'min'
			return res, [new_rtree_idx, tree_idx]

		return tm, []


	def minmax_distribution(self, tm, tree_idx):
		cur_tree = tm.get_tree(tree_idx)
		op = cur_tree.root
		if not (op in ['<', '<=']):
			return tm, []
		ltree_idx = cur_tree.children[0]
		ltree = tm.get_tree(ltree_idx)
		rtree_idx = cur_tree.children[1]
		rtree = tm.get_tree(rtree_idx)

		if ltree.root == 'min':
			res = tm.clone()
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = res.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = res.get_tree(ltree_rtree_idx)
			rtree_clone_idx = res.clone_tree(res, rtree_idx)
			new_ltree_idx = res.create_tree(op, tree_idx, cur_tree.depth + 1)
			new_rtree_idx = res.create_tree(op, tree_idx, cur_tree.depth + 1)
			res.update_edge(new_ltree_idx, ltree_ltree_idx, 0)
			res.update_edge(new_ltree_idx, rtree_idx, 1)
			res.update_edge(new_rtree_idx, ltree_rtree_idx, 0)
			res.update_edge(new_rtree_idx, rtree_clone_idx, 1)
			res.trees[tree_idx].root = '||'
			res.update_edge(tree_idx, new_ltree_idx, 0)
			res.update_edge(tree_idx, new_rtree_idx, 1)
			return res, [new_ltree_idx, new_rtree_idx]

		if rtree.root == 'max':
			res = tm.clone()
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = res.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = res.get_tree(rtree_rtree_idx)
			ltree_clone_idx = res.clone_tree(res, ltree_idx)
			new_ltree_idx = res.create_tree(op, tree_idx, cur_tree.depth + 1)
			new_rtree_idx = res.create_tree(op, tree_idx, cur_tree.depth + 1)
			res.update_edge(new_ltree_idx, ltree_idx, 0)
			res.update_edge(new_ltree_idx, rtree_ltree_idx, 1)
			res.update_edge(new_rtree_idx, ltree_clone_idx, 0)
			res.update_edge(new_rtree_idx, rtree_rtree_idx, 1)
			res.trees[tree_idx].root = '||'
			res.update_edge(tree_idx, new_ltree_idx, 0)
			res.update_edge(tree_idx, new_rtree_idx, 1)
			return res, [new_ltree_idx, new_rtree_idx]

		if ltree.root == 'max':
			res = tm.clone()
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = res.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = res.get_tree(ltree_rtree_idx)
			rtree_clone_idx = res.clone_tree(res, rtree_idx)
			new_ltree_idx = res.create_tree(op, tree_idx, cur_tree.depth + 1)
			new_rtree_idx = res.create_tree(op, tree_idx, cur_tree.depth + 1)
			res.update_edge(new_ltree_idx, ltree_ltree_idx, 0)
			res.update_edge(new_ltree_idx, rtree_idx, 1)
			res.update_edge(new_rtree_idx, ltree_rtree_idx, 0)
			res.update_edge(new_rtree_idx, rtree_clone_idx, 1)
			res.trees[tree_idx].root = '&&'
			res.update_edge(tree_idx, new_ltree_idx, 0)
			res.update_edge(tree_idx, new_rtree_idx, 1)
			return res, [new_ltree_idx, new_rtree_idx]

		if rtree.root == 'min':
			res = tm.clone()
			rtree_ltree_idx = rtree.children[0]
			rtree_ltree = res.get_tree(rtree_ltree_idx)
			rtree_rtree_idx = rtree.children[1]
			rtree_rtree = res.get_tree(rtree_rtree_idx)
			ltree_clone_idx = res.clone_tree(res, ltree_idx)
			new_ltree_idx = res.create_tree(op, tree_idx, cur_tree.depth + 1)
			new_rtree_idx = res.create_tree(op, tree_idx, cur_tree.depth + 1)
			res.update_edge(new_ltree_idx, ltree_idx, 0)
			res.update_edge(new_ltree_idx, rtree_ltree_idx, 1)
			res.update_edge(new_rtree_idx, ltree_clone_idx, 0)
			res.update_edge(new_rtree_idx, rtree_rtree_idx, 1)
			res.trees[tree_idx].root = '&&'
			res.update_edge(tree_idx, new_ltree_idx, 0)
			res.update_edge(tree_idx, new_rtree_idx, 1)
			return res, [new_ltree_idx, new_rtree_idx]

		return tm, []


