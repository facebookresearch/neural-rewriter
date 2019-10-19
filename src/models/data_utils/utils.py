# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


def is_int(s):
	return s.lstrip('+-').isdigit()


def is_var(s):
	return s.startswith('v')


def gcd(x, y):
	while y != 0:
		t = y
		y = x % y
		x = t
	return x


def calc(op, c1, c2):
	c1 = int(c1)
	c2 = int(c2)
	if op == '+':
		return str(c1 + c2)
	elif op == '-':
		return str(c1 - c2)
	elif op == '*':
		return str(c1 * c2)
	elif op == '/':
		return str(c1 // c2)
	elif op == '%':
		return str(c1 % c2)
	elif op == '<':
		if c1 < c2:
			return str(1)
		else:
			return str(0)
	elif op == '<=':
		if c1 <= c2:
			return str(1)
		else:
			return str(0)
	elif op == '==':
		if c1 == c2:
			return str(1)
		else:
			return str(0)
	elif op == '&&':
		if c1 and c2:
			return str(1)
		else:
			return str(0)
	elif op == '||':
		if c1 or c2:
			return str(1)
		else:
			return str(0)
	else:
		raise ValueError('undefined op: ' + op)
