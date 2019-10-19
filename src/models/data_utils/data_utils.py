# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import collections
import json
import os
import random
import sys
import time
import six
import numpy as np
import copy
import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from .parser import *

_PAD = b"_PAD"

PAD_ID = 0
START_VOCAB_SIZE = 1
max_token_len = 5

def np_to_tensor(inp, output_type, cuda_flag, volatile_flag=False):
	if output_type == 'float':
		inp_tensor = Variable(torch.FloatTensor(inp), volatile=volatile_flag)
	elif output_type == 'int':
		inp_tensor = Variable(torch.LongTensor(inp), volatile=volatile_flag)
	else:
		print('undefined tensor type')
	if cuda_flag:
		inp_tensor = inp_tensor.cuda()
	return inp_tensor

def load_dataset(filename, args):
	with open(filename, 'r') as f:
		samples = json.load(f)
	print('Number of data samples in ' + filename + ': ', len(samples))
	return samples

class HalideDataProcessor(object):
	def __init__(self):
		self.parser = HalideParser()
		self.tokenizer = self.parser.tokenizer
	def load_term_vocab(self):
		vocab = {}
		vocab_list = []
		vocab[_PAD] = PAD_ID
		vocab_list.append(_PAD)
		for i in range(10):
			vocab[str(i)] = len(vocab)
			vocab_list.append(str(i))
		vocab['v'] = len(vocab)
		vocab_list.append('v')
		vocab['-'] = len(vocab)
		vocab_list.append('-')
		return vocab, vocab_list

	def load_ops(self):
		ops_list = self.tokenizer.ops + self.tokenizer.keywords
		ops = {}
		for op in ops_list:
			ops[op] = len(ops)
		return ops, ops_list

	def token_to_ids(self, token, vocab):
		token_ids = [vocab.get(c) for c in token]
		token_ids = [PAD_ID for _ in range(max_token_len - len(token_ids))] + token_ids
		return token_ids

	def prune_dataset(self, init_data, min_len=None, max_len=None):
		data = []
		for trace in init_data:
			expr_len = len(trace[0])
			if min_len is not None and expr_len < min_len:
				continue
			if max_len is not None and expr_len > max_len:
				continue
			data.append(trace)
		return data

	def get_batch(self, data, batch_size, start_idx=None):
		data_size = len(data)
		if start_idx is not None:
			batch_idxes = [i for i in range(start_idx, min(data_size, start_idx + batch_size))]
		else:
			batch_idxes = np.random.choice(len(data), batch_size)
		batch_data = []
		for idx in batch_idxes:
			trace = data[idx]
			tm = self.parser.parse(trace[0])
			batch_data.append((trace, tm))
		return batch_data

	def print_gt_trace(self, trace):
		print('ground truth: ')
		for trace_step in trace:
			print(trace_step)
			print('')

	def print_pred_trace(self, trace_rec):
		print('prediction: ')
		for trace_step in trace_rec:
			print(trace_step)
			print('')

class jspDataProcessor(object):
	def __init__(self, args):
		self.parser = jspDependencyParser(args)

	def get_batch(self, data, batch_size, start_idx=None):
		data_size = len(data)
		if start_idx is not None:
			batch_idxes = [i for i in range(start_idx, min(data_size, start_idx + batch_size))]
		else:
			batch_idxes = np.random.choice(len(data), batch_size)
		batch_data = []
		for idx in batch_idxes:
			job_seq = data[idx]
			dm = self.parser.parse(job_seq)
			batch_data.append(dm)
		return batch_data


class vrpDataProcessor(object):
	def __init__(self):
		self.parser = vrpParser()

	def get_batch(self, data, batch_size, start_idx=None):
		data_size = len(data)
		if start_idx is not None:
			batch_idxes = [i for i in range(start_idx, min(data_size, start_idx + batch_size))]
		else:
			batch_idxes = np.random.choice(len(data), batch_size)
		batch_data = []
		for idx in batch_idxes:
			problem = data[idx]
			dm = self.parser.parse(problem)
			batch_data.append(dm)
		return batch_data