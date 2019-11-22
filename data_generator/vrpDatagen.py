# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import numpy as np
import argparse
import json

argParser = argparse.ArgumentParser()
argParser.add_argument('--res_file', type=str, default='vrp_20_30.json')
argParser.add_argument('--res_train_file', type=str, default='vrp_20_30_train.json')
argParser.add_argument('--res_val_file', type=str, default='vrp_20_30_val.json')
argParser.add_argument('--res_test_file', type=str, default='vrp_20_30_test.json')
argParser.add_argument('--num_samples', type=int, default=100000)
argParser.add_argument('--seed', type=int, default=None)
argParser.add_argument('--num_customers', type=int, default=20)
argParser.add_argument('--max_demand', type=int, default=9)
argParser.add_argument('--position_range', type=float, default=1.0)
argParser.add_argument('--capacity', type=int, default=30, choices=[20, 30, 40, 50])

args = argParser.parse_args()


def sample_pos():
	return np.random.rand(), np.random.rand()


def main():
	np.random.seed(args.seed)
	samples = []
	for _ in range(args.num_samples):
		cur_sample = {}
		cur_sample['customers'] = []
		cur_sample['capacity'] = args.capacity
		dx, dy = sample_pos()
		cur_sample['depot'] = (dx, dy)
		for i in range(args.num_customers):
			cx, cy = sample_pos()
			demand = np.random.randint(1, args.max_demand + 1)
			cur_sample['customers'].append({'position': (cx, cy), 'demand': demand})
		samples.append(cur_sample)

	path = '../data/vrp/'
	if not os.path.exists(path):
		os.makedirs(path)

	data_size = len(samples)
	print(data_size)
	fout_res = open(path+args.res_file, 'w')
	json.dump(samples, fout_res)

	fout_train = open(path+args.res_train_file, 'w')
	train_data_size = int(data_size * 0.8)
	json.dump(samples[:train_data_size], fout_train)

	fout_val = open(path+args.res_val_file, 'w')
	val_data_size = int(data_size * 0.9) - train_data_size
	json.dump(samples[train_data_size: train_data_size + val_data_size], fout_val)

	fout_test = open(path+args.res_test_file, 'w')
	test_data_size = data_size - train_data_size - val_data_size
	json.dump(samples[train_data_size + val_data_size:], fout_test)


main()
