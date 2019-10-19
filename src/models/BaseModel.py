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

import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F


class BaseModel(nn.Module):
	"""
	Base neural rewriter model. The concrete architectures for different applications are derived from it.
	"""
	def __init__(self, args):
		super(BaseModel, self).__init__()
		self.processes = args.processes
		self.batch_size = args.batch_size
		self.LSTM_hidden_size = args.LSTM_hidden_size
		self.MLP_hidden_size = args.MLP_hidden_size
		self.num_MLP_layers = args.num_MLP_layers
		self.gradient_clip = args.gradient_clip
		self.lr = args.lr
		self.dropout_rate = args.dropout_rate
		self.max_reduce_steps = args.max_reduce_steps
		self.num_sample_rewrite_pos = args.num_sample_rewrite_pos
		self.num_sample_rewrite_op = args.num_sample_rewrite_op
		self.value_loss_coef = args.value_loss_coef
		self.gamma = args.gamma
		self.cont_prob = args.cont_prob
		self.cuda_flag = args.cuda
		

	def init_weights(self, param_init):
		for param in self.parameters():
			param.data.uniform_(-param_init, param_init)


	def lr_decay(self, lr_decay_rate):
		self.lr *= lr_decay_rate
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr


	def train(self):
		if self.gradient_clip > 0:
			clip_grad_norm(self.parameters(), self.gradient_clip)
		self.optimizer.step()

