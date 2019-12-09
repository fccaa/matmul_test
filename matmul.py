#!/usr/bin/env python3

import numpy
import time
import argparse

parser = argparse.ArgumentParser(description='sum the integers at the command line')
parser.add_argument("-s", metavar="int", type=int, help="Number of Sample", default=10)
parser.add_argument("-N", metavar="int", type=int, help="Maximum size of matrix", default=10000)
parser.add_argument("-fp32", help="Do matrix multiplication with fp32", action="store_true")

args = parser.parse_args()

N=10
while N < args.N:
	GFLOPS = 2.0 * N * N * N / pow(1000.0, 3)
	A = numpy.random.rand(N,N) + 1.0
	B = numpy.random.rand(N,N) + 1.0
	time_record = numpy.array([])

	for i in range(args.s):
		time.sleep(1)
		if args.fp32:
			t1 = time.time()
			C = numpy.matmul(A.astype(numpy.float32), B.astype(numpy.float32))
			t2 = time.time()
		else:
			t1 = time.time()
			C = numpy.matmul(A.astype(numpy.float64), B.astype(numpy.float64))
			t2 = time.time()
		time_record = numpy.append(time_record, numpy.fabs(t2-t1))

	perf_max = GFLOPS / time_record.min()
	perf_avg = (GFLOPS / time_record).mean()
	print(N, perf_avg, perf_max, flush=True)
	N = int(N*1.2)
