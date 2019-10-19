# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch.autograd import Variable

from .utils import *

eps = 1e-6


class Tree(object):
	"""
	The class to represent a single tree. It can be used for expression simplification.
	"""
	def __init__(self, root, parent, depth, state=None):
		self.root = root
		self.is_const = False
		self.update_tok_type()
		self.children = []
		self.parent = parent
		self.depth = depth
		if state is None:
			self.state = None
		else:
			self.state = state[0].clone(), state[1].clone()


	def add_child(self, child, child_idx=-1):
		while len(self.children) <= child_idx:
			self.children += [child]
		if child_idx == -1:
			self.children += [child]
		else:
			self.children[child_idx] = child


	def update_tok_type(self):
		if is_var(self.root):
			self.tok_type = 'var'
		elif is_int(self.root):
			self.tok_type = 'const'
			self.is_const = True
		elif self.root == '!':
			self.tok_type = 'unary_op'
		elif self.root in ['min', 'max', 'select']:
			self.tok_type = 'select'
		else:
			self.tok_type = 'binary_op'


	def num_tokens(self):
		res = 1
		for c in self.children:
			res += c.num_tokens()
		return res


class TreeManager(object):
	"""
	The class to maintain tree-structured input data. It is used for expression simplification.
	"""
	def __init__(self):
		self.clear()


	def clear(self):
		self.trees = []
		self.num_trees = 0
		self.root = 0


	def create_tree(self, root, parent, depth, state=None):
		self.trees.append(Tree(root, parent, depth, state))
		self.num_trees += 1
		return self.num_trees - 1


	def get_tree(self, idx):
		return self.trees[idx]


	def num_valid_nodes(self, cur_idx=None):
		if cur_idx is None:
			cur_idx = self.root
		tot = 1
		cur_tree = self.get_tree(cur_idx)
		if len(cur_tree.children) == 0:
			return tot
		for child in cur_tree.children:
			tot += self.num_valid_nodes(child)
		return tot


	def clone(self):
		res = TreeManager()
		res.root = self.root
		res.num_trees = 0
		for i, tree in enumerate(self.trees):
			res.create_tree(tree.root, tree.parent, tree.depth, tree.state)
			for child in tree.children:
				res.trees[i].children.append(child)
		return res


	def clone_tree(self, tm, ref_tree_idx=-1):
		if ref_tree_idx == -1:
			ref_tree_idx = tm.root
		ref_tree = tm.get_tree(ref_tree_idx)
		root = self.create_tree(ref_tree.root, -1, 0, ref_tree.state)
		for child in ref_tree.children:
			c = self.clone_tree(tm, child)
			self.update_edge(root, c)
		return root


	def equal_tree(self, t1, t2):
		if t1.root != t2.root:
			return False
		if t1.state is not None and t2.state is not None:
			dis_vec = torch.abs(t1.state[0] - t2.state[0]) + torch.abs(t1.state[1] - t2.state[1])
			dis, _ = dis_vec.max(1)
			if dis.data[0] > eps:
				return False
		for child_idx in range(len(t1.children)):
			t1_c = self.get_tree(t1.children[child_idx])
			t2_c = self.get_tree(t2.children[child_idx])
			if not self.equal_tree(t1_c, t2_c):
				return False
		return True


	def update_edge(self, parent, child, child_idx=-1):
		if parent == -1:
			self.root = child
		else:
			self.trees[parent].add_child(child, child_idx)
		self.trees[child].parent = parent


	def find_child_idx(self, parent, child):
		if parent == -1:
			return -1
		parent_tree = self.get_tree(parent)
		for child_idx in range(len(parent_tree.children)):
			if parent_tree.children[child_idx] == child:
				return child_idx
		raise ValueError('invalid edge: ' + str(parent) + ' ' + parent_tree.root + ' ' + str(child) + ' ' + str(parent_tree.children))
		return -1


	def find_subtree(self, tree_idx, subtree_idx, op):
		cur_tree = self.get_tree(tree_idx)
		subtree = self.get_tree(subtree_idx)
		if self.equal_tree(cur_tree, subtree):
			return tree_idx

		if op == '-':
			if cur_tree.root == '-':
				lchild = self.get_tree(cur_tree.children[0])
				if not self.equal_tree(lchild, subtree):
					res = self.find_subtree(cur_tree.children[0], subtree_idx, op)
					if res != -1:
						return res
				res = self.find_subtree(cur_tree.children[1], subtree_idx, '+')
				if res != -1:
					return res
			if cur_tree.root == '+':
				for child in cur_tree.children:
					child_tree = self.get_tree(child)
					if not self.equal_tree(child_tree, subtree):
						res = self.find_subtree(child, subtree_idx, op)
						if res != -1:
							return res
			return -1

		if op == '+' and cur_tree.root == '-':
			res = self.find_subtree(cur_tree.children[0], subtree_idx, op)
			if res != -1:
				return res
			rchild = self.get_tree(cur_tree.children[1])
			if not self.equal_tree(rchild, subtree):
				res = self.find_subtree(cur_tree.children[1], subtree_idx, '-')
				if res != -1:
					return res

		if cur_tree.root != op:
			return -1
		for child in cur_tree.children:
			res = self.find_subtree(child, subtree_idx, op)
			if res != -1:
				return res
		return -1


	def find_alg_const(self, tree_idx, op):
		cur_tree = self.get_tree(tree_idx)
		if cur_tree.is_const:
			return tree_idx
		if op == '+':
			if cur_tree.root == '+':
				for child in cur_tree.children:
					cur_idx = self.find_alg_const(child, op)
					if cur_idx is not None:
						return cur_idx
			elif cur_tree.root == '-':
				if self.get_tree(cur_tree.children[1]).is_const:
					return cur_tree.children[1]
				cur_idx = self.find_alg_const(cur_tree.children[0], op)
				return cur_idx
		if op == '*':
			if cur_tree.root == '*':
				for child in cur_tree.children:
					cur_idx = self.find_alg_const(child, op)
					if cur_idx is not None:
						return cur_idx
		return None


	def find_minmax_const(self, tree_idx, op):
		cur_tree = self.get_tree(tree_idx)
		if cur_tree.is_const:
			return tree_idx
		if cur_tree.root != op:
			return None
		res_idx = None
		res = None
		for child in cur_tree.children:
			cur_idx = self.find_minmax_const(child, op)
			if cur_idx is not None:
				if res_idx is None:
					res_idx = cur_idx
					res_tree = self.get_tree(res_idx)
					res = int(res_tree.root)
				else:
					cur_tree = self.get_tree(cur_idx)
					t = int(cur_tree.root)
					if op == 'max' and t > res or op == 'min' and t < res:
						res = t
						res_idx = cur_idx
		return res_idx


	def find_muldiv_term(self, tree_idx):
		cur_tree = self.get_tree(tree_idx)
		if not cur_tree.root in ['+', '-', '*']:
			return []
		ltree_idx = cur_tree.children[0]
		ltree = self.get_tree(ltree_idx)
		rtree_idx = cur_tree.children[1]
		rtree = self.get_tree(rtree_idx)
		if cur_tree.root == '*' and rtree.is_const and int(rtree.root) > 0 and ltree.root == '/':
			ltree_ltree_idx = ltree.children[0]
			ltree_ltree = self.get_tree(ltree_ltree_idx)
			ltree_rtree_idx = ltree.children[1]
			ltree_rtree = self.get_tree(ltree_rtree_idx)
			if ltree_rtree.root == rtree.root:
				return [tree_idx]
		res = []
		if cur_tree.root == '+':
			for child in cur_tree.children:
				cur_res = self.find_muldiv_term(child)
				res = res + cur_res
		elif cur_tree.root == '-':
			res = self.find_muldiv_term(cur_tree.children[0])
		return res


	def is_times(self, tree_idx, c):
		cur_tree = self.get_tree(tree_idx)
		if cur_tree.is_const and int(cur_tree.root) % c == 0:
			return True
		if cur_tree.root != '*':
			return False
		ltree_idx = cur_tree.children[0]
		ltree = self.get_tree(ltree_idx)
		rtree_idx = cur_tree.children[1]
		rtree = self.get_tree(rtree_idx)
		if rtree.is_const and int(rtree.root) % c == 0:
			return True
		return False


	def find_times_term(self, tree_idx, c):
		if self.is_times(tree_idx, c):
			return tree_idx
		cur_tree = self.get_tree(tree_idx)
		if not cur_tree.root in ['+', '-']:
			return -1
		ltree_idx = cur_tree.children[0]
		ltree = self.get_tree(ltree_idx)
		rtree_idx = cur_tree.children[1]
		rtree = self.get_tree(rtree_idx)
		res = self.find_times_term(ltree_idx, c)
		if res != -1:
			return res
		res = self.find_times_term(rtree_idx, c)
		return res


	def to_string(self, tree_idx, tok_map=None, log_out=None):
		self.trees[tree_idx].update_tok_type()
		r = ''
		if self.trees[tree_idx].tok_type == 'var' or self.trees[tree_idx].tok_type == 'const':
			if tok_map and (self.trees[tree_idx].root in tok_map):
				r += tok_map(self.trees[tree_idx].root)
			else:
				r += str(self.trees[tree_idx].root)
			return r
		if self.trees[tree_idx].tok_type == 'binary_op':
			r += '('
			r += self.to_string(self.trees[tree_idx].children[0], tok_map, log_out)
			if not self.trees[tree_idx].root in ['*', '/']:
				r += ' '
			if tok_map and (self.trees[tree_idx].root in tok_map):
				r += tok_map(self.trees[tree_idx].root)
			else:
				r += str(self.trees[tree_idx].root)
			if not self.trees[tree_idx].root in ['*', '/']:
				r += ' '
			r += self.to_string(self.trees[tree_idx].children[1], tok_map, log_out)
			r += ')'
			return r
		if self.trees[tree_idx].tok_type == 'unary_op':
			r = ''
			if tok_map and (self.trees[tree_idx].root in tok_map):
				r += tok_map(self.trees[tree_idx].root)
			else:
				r += str(self.trees[tree_idx].root)
			r += '('
			r += self.to_string(self.trees[tree_idx].children[0], tok_map, log_out)
			r += ')'
			return r
		if self.trees[tree_idx].tok_type == 'select':
			if tok_map and (self.trees[tree_idx].root in tok_map):
				r += tok_map(self.trees[tree_idx].root)
			else:
				r += str(self.trees[tree_idx].root)
			r += '('
			r += self.to_string(self.trees[tree_idx].children[0], tok_map, log_out)
			for c in self.trees[tree_idx].children[1:]:
				r += ', ' + self.to_string(c, tok_map, log_out)
			r += ')'
			return r


	def __str__(self):
		return self.to_string(self.root)