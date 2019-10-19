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
import time

argParser = argparse.ArgumentParser()
argParser.add_argument('--input_file', type=str, default='../data/jsp/jsp_r20_test.json')
argParser.add_argument('--alg', type=str, default='SJF', choices=['random', 'EJF', 'SJF', 'offline'])
argParser.add_argument('--num_res', type=int, default=20)
argParser.add_argument('--job_horizon', type=int, default=10)
argParser.add_argument('--max_resource_size', type=int, default=10)
argParser.add_argument('--max_time_horizon', type=int, default=1000)

args = argParser.parse_args()


def add_job(cur_job, scheduled_job, used_resources):
	scheduled_job.append(cur_job)
	cur_resources = cur_job['resource_size']
	st_time = cur_job['schedule_time']
	ed_time = st_time + cur_job['job_len']
	for t in range(st_time, ed_time):
		for j in range(args.num_res):
			used_resources[t][j] += cur_resources[j]
	return scheduled_job, used_resources


def calc_min_schedule_time(used_resources, cur_job):
	tmp_used_resources = used_resources.copy()
	min_schedule_time = cur_job['schedule_time']
	cur_resources = cur_job['resource_size']
	runnable = True
	cur_time_horizon = min_schedule_time
	while not runnable or cur_time_horizon <= min_schedule_time + cur_job['job_len'] - 1:
		runnable = True
		for j in range(args.num_res):
			if tmp_used_resources[cur_time_horizon][j] + cur_resources[j] > args.max_resource_size:
				runnable = False
				break
		if not runnable:
			min_schedule_time = cur_time_horizon + 1
		cur_time_horizon += 1

	return min_schedule_time


def random_schedule(job_seq):
	used_resources = np.zeros((args.max_time_horizon, args.num_res))
	scheduled_job = []
	pending_job = []
	for job_idx in range(len(job_seq) + 1):
		if job_idx < len(job_seq):
			cur_job = job_seq[job_idx].copy()
			st_time = cur_job['start_time']
			job_len = cur_job['job_len']
			cur_resources = cur_job['resource_size']
			cur_job['schedule_time'] = st_time
		else:
			st_time = -1
		schedule_time = -1
		if job_idx == len(job_seq):
			pending_job_cap = 1
		else:
			pending_job_cap = args.job_horizon
		while len(pending_job) >= pending_job_cap:
			schedule_idx = np.random.choice(len(pending_job))
			schedule_time = calc_min_schedule_time(used_resources, pending_job[schedule_idx])
			pending_job[schedule_idx]['schedule_time'] = schedule_time
			scheduled_job, used_resources = add_job(pending_job[schedule_idx], scheduled_job, used_resources)
			pending_job = pending_job[:schedule_idx] + pending_job[schedule_idx + 1:]

		if job_idx == len(job_seq):
			break
		pending_job.append(cur_job)

	return scheduled_job


def ejf(job_seq):
	used_resources = np.zeros((args.max_time_horizon, args.num_res))
	scheduled_job = []
	for job_idx in range(len(job_seq)):
		cur_job = job_seq[job_idx].copy()
		st_time = cur_job['start_time']
		job_len = cur_job['job_len']
		cur_resources = cur_job['resource_size']
		cur_job['schedule_time'] = st_time
		min_schedule_time = st_time
		cur_job['schedule_time'] = max(cur_job['schedule_time'], min_schedule_time)
		cur_job['schedule_time'] = calc_min_schedule_time(used_resources, cur_job)
		cur_completion_time = cur_job['schedule_time'] + job_len - st_time
		scheduled_job, used_resources = add_job(cur_job, scheduled_job, used_resources)
	return scheduled_job


def sjf(job_seq):
	used_resources = np.zeros((args.max_time_horizon, args.num_res))
	scheduled_job = []
	pending_job = []
	for job_idx in range(len(job_seq) + 1):
		if job_idx < len(job_seq):
			cur_job = job_seq[job_idx].copy()
			st_time = cur_job['start_time']
			job_len = cur_job['job_len']
			cur_resources = cur_job['resource_size']
			cur_job['schedule_time'] = st_time
		else:
			st_time = -1
		schedule_time = -1
		if job_idx == len(job_seq):
			pending_job_cap = 1
		else:
			pending_job_cap = args.job_horizon
		while len(pending_job) >= pending_job_cap:
			schedule_idx = -1
			schedule_time = -1
			for i in range(len(pending_job)):
				cur_min_schedule_time = calc_min_schedule_time(used_resources, pending_job[i])
				if schedule_idx == -1 or cur_min_schedule_time < schedule_time or cur_min_schedule_time == schedule_time and pending_job[i]['job_len'] < pending_job[schedule_idx]['job_len']:
					schedule_idx = i
					schedule_time = cur_min_schedule_time
			pending_job[schedule_idx]['schedule_time'] = schedule_time
			scheduled_job, used_resources = add_job(pending_job[schedule_idx], scheduled_job, used_resources)
			pending_job = pending_job[:schedule_idx] + pending_job[schedule_idx + 1:]

		if job_idx == len(job_seq):
			break
		pending_job.append(cur_job)

	return scheduled_job


def offline(job_seq):
	used_resources = np.zeros((args.max_time_horizon, args.num_res))
	scheduled_job = []
	pending_job = []
	for job_idx in range(len(job_seq) + 1):
		if job_idx < len(job_seq):
			cur_job = job_seq[job_idx].copy()
			st_time = cur_job['start_time']
			job_len = cur_job['job_len']
			cur_resources = cur_job['resource_size']
			cur_job['schedule_time'] = st_time
		else:
			st_time = -1
		schedule_time = -1
		if job_idx == len(job_seq):
			pending_job_cap = 1
		else:
			pending_job_cap = len(job_seq)
		while len(pending_job) >= pending_job_cap:
			schedule_idx = -1
			schedule_time = -1
			for i in range(len(pending_job)):
				cur_min_schedule_time = calc_min_schedule_time(used_resources, pending_job[i])
				if schedule_idx == -1 or cur_min_schedule_time < schedule_time or cur_min_schedule_time == schedule_time and pending_job[i]['job_len'] < pending_job[schedule_idx]['job_len']:
					schedule_idx = i
					schedule_time = cur_min_schedule_time
			pending_job[schedule_idx]['schedule_time'] = schedule_time
			scheduled_job, used_resources = add_job(pending_job[schedule_idx], scheduled_job, used_resources)
			pending_job = pending_job[:schedule_idx] + pending_job[schedule_idx + 1:]

		if job_idx == len(job_seq):
			break
		pending_job.append(cur_job)

	return scheduled_job


def calc_reward(res):
	avg_slow_down = 0.0
	avg_completion_time = 0.0
	for cur_job in res:
		st_time = cur_job['start_time']
		job_len = cur_job['job_len']
		cur_completion_time = cur_job['schedule_time'] + job_len - st_time
		avg_slow_down += cur_completion_time * 1.0 / job_len
		avg_completion_time += cur_completion_time
	avg_slow_down /= len(res)
	avg_completion_time /= len(res)
	return avg_slow_down, avg_completion_time	


if __name__ == "__main__":
	with open(args.input_file, 'r') as fin:
		samples = json.load(fin)
	avg_slow_down = 0.0
	avg_completion_time = 0.0
	for i, cur_sample in enumerate(samples):
		if args.alg == 'random':
			res = random_schedule(cur_sample)
		if args.alg == 'EJF':
			res = ejf(cur_sample)
		elif args.alg == 'SJF':
			res = sjf(cur_sample)
		elif args.alg == 'offline':
			res = offline(cur_sample)
		cur_avg_slow_down, cur_avg_completion_time = calc_reward(res)
		avg_slow_down += cur_avg_slow_down
		avg_completion_time += cur_avg_completion_time
		print('sample %d slow down: %.4f completion time: %.4f' % (i, cur_avg_slow_down, cur_avg_completion_time))
	
	avg_slow_down /= len(samples)
	avg_completion_time /= len(samples)
	print('average slow down: %.4f average completion time: %.4f' % (avg_slow_down, avg_completion_time))