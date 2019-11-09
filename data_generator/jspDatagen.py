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
argParser.add_argument('--res_file', type=str, default='../data/jsp/jsp_r10.json')
argParser.add_argument('--res_train_file', type=str, default='../data/jsp/jsp_r10_train.json')
argParser.add_argument('--res_val_file', type=str, default='../data/jsp/jsp_r10_val.json')
argParser.add_argument('--res_test_file', type=str, default='../data/jsp/jsp_r10_test.json')

argParser.add_argument('--num_samples', type=int, default=100000)
argParser.add_argument('--seed', type=int, default=None)
argParser.add_argument('--num_res', type=int, default=10)
argParser.add_argument('--max_resource_size', type=int, default=10)
argParser.add_argument('--time_horizon', type=int, default=50)
argParser.add_argument('--job_horizon', type=int, default=10)
argParser.add_argument('--job_small_chance', type=float, default=0.8)
argParser.add_argument('--new_job_rate', type=float, default=0.7)
argParser.add_argument('--job_len_big_lower', type=int, default=10)
argParser.add_argument('--job_len_big_upper', type=int, default=15)
argParser.add_argument('--job_len_small_lower', type=int, default=1)
argParser.add_argument('--job_len_small_upper', type=int, default=3)
argParser.add_argument('--dominant_res_lower', type=int, default=5)
argParser.add_argument('--dominant_res_upper', type=int, default=10)
argParser.add_argument('--other_res_lower', type=int, default=1)
argParser.add_argument('--other_res_upper', type=int, default=2)

argParser.add_argument('--uniform_short', action='store_true')
argParser.add_argument('--uniform_long', action='store_true')
argParser.add_argument('--uniform_resource', action='store_true')
argParser.add_argument('--dynamic_new_job_rate', action='store_true')

args = argParser.parse_args()


def sample_job():
	if np.random.rand() < args.job_small_chance:
		cur_job_len = np.random.randint(args.job_len_small_lower, args.job_len_small_upper + 1)
	else:
		cur_job_len = np.random.randint(args.job_len_big_lower, args.job_len_big_upper + 1)
	cur_resource_size = np.zeros(args.num_res)
	if args.uniform_resource:
		if np.random.rand() < 0.5:
			dominant_res = []
		else:
			dominant_res = range(args.num_res)
	else:
		dominant_res = np.random.randint(low=0, high=args.num_res, size=args.num_res // 2)
	for i in range(args.num_res):
		if i in dominant_res:
			cur_resource_size[i] = np.random.randint(args.dominant_res_lower, args.dominant_res_upper + 1)
		else:
			cur_resource_size[i] = np.random.randint(args.other_res_lower, args.other_res_upper + 1)
	return cur_job_len, cur_resource_size


def main():
	np.random.seed(args.seed)
	samples = []
	for _ in range(args.num_samples):
		cur_sample = []
		if args.uniform_short:
			args.job_small_chance = 1.0
		elif args.uniform_long:
			args.job_small_chance = 0.0
		while len(cur_sample) == 0:
			for i in range(args.time_horizon):
				if args.dynamic_new_job_rate:
					args.new_job_rate = np.random.rand()
				if np.random.rand() < args.new_job_rate:
					cur_job_len, cur_resource_size = sample_job()
					cur_sample.append({'start_time': i, 'job_len': cur_job_len, 'resource_size': list(cur_resource_size)})
		samples.append(cur_sample)

	data_size = len(samples)
	print(data_size)
	fout_res = open(args.res_file, 'w')
	json.dump(samples, fout_res)

	fout_train = open(args.res_train_file, 'w')
	train_data_size = int(data_size * 0.8)
	json.dump(samples[:train_data_size], fout_train)

	fout_val = open(args.res_val_file, 'w')
	val_data_size = int(data_size * 0.9) - train_data_size
	json.dump(samples[train_data_size: train_data_size + val_data_size], fout_val)

	fout_test = open(args.res_test_file, 'w')
	test_data_size = data_size - train_data_size - val_data_size
	json.dump(samples[train_data_size + val_data_size:], fout_test)


main()
