# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class MLPModel(nn.Module):
	"""
	Multi-layer perceptron module.
	"""
	def __init__(self, num_layers, input_size, hidden_size, output_size, cuda_flag, dropout_rate=0.0, activation=None):
		super(MLPModel, self).__init__()
		self.num_layers = num_layers
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_rate = dropout_rate
		self.cuda_flag = cuda_flag
		self.dropout = nn.Dropout(p=self.dropout_rate)
		self.model = nn.Sequential(
			nn.Linear(self.input_size, self.hidden_size),
			nn.ReLU())
		for _ in range(self.num_layers):
			self.model = nn.Sequential(
				self.model,
				nn.Linear(self.hidden_size, self.hidden_size),
				nn.ReLU())
		self.model = nn.Sequential(
			self.model,
			nn.Linear(self.hidden_size, self.output_size))
		if activation is not None:
			self.model = nn.Sequential(
				self.model,
				activation)

	
	def forward(self, inputs):
		return self.model(inputs)