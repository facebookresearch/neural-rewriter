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

from ..data_utils import data_utils

class SeqLSTM(nn.Module):
	"""
	LSTM to embed the input as a sequence.
	"""
	def __init__(self, args):
		super(SeqLSTM, self).__init__()
		self.batch_size = args.batch_size
		self.hidden_size = args.LSTM_hidden_size
		self.embedding_size = args.embedding_size
		self.num_layers = args.num_LSTM_layers
		self.dropout_rate = args.dropout_rate
		self.cuda_flag = args.cuda
		self.encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_rate, bidirectional=True)


	def calc_embedding(self, seq_managers, eval_mode=False):
		encoder_input = []
		max_node_cnt = 0
		batch_size = len(seq_managers)

		for seq_manager in seq_managers:
			encoder_input.append(seq_manager.route[:])
			max_node_cnt = max(max_node_cnt, len(seq_manager.route))

		for i in range(batch_size):
			while len(encoder_input[i]) < max_node_cnt:
				encoder_input[i].append([0.0 for _ in range(self.embedding_size)])

		encoder_input = np.array(encoder_input)
		encoder_input = data_utils.np_to_tensor(encoder_input, 'float', self.cuda_flag, eval_mode)
		encoder_output, encoder_state = self.encoder(encoder_input)

		for seq_manager_idx, seq_manager in enumerate(seq_managers):
			seq_managers[seq_manager_idx].encoder_outputs = encoder_output[seq_manager_idx]

		return seq_managers