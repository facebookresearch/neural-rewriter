# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import argparse
import sys
import os
import re
import json
import pandas as pd


class Logger(object):
	"""
	The class for recording the training process.
	"""
	def __init__(self, args):
		self.log_interval = args.log_interval
		self.log_name = args.log_name
		self.best_reward = 0
		self.records = []


	def write_summary(self, summary):
		print("global-step: %(global_step)d, avg-reward: %(avg_reward).3f" % summary)
		self.records.append(summary)
		df = pd.DataFrame(self.records)
		df.to_csv(self.log_name, index=False)
		self.best_reward = max(self.best_reward, summary['avg_reward'])
		