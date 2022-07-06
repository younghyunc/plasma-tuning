#! /usr/bin/env python

# Author: Younghyun Cho <younghyun@berkeley.edu>

import os
import sys
import ctypes
import numpy as np

from ctypes import c_int, c_double, c_longlong, POINTER

def test_plasma_dgemm(plasmalib):

    print ("Test PLASMA DGEMM")

    A = np.array([[0, 1], [2, 3]], dtype="float64")
    B = np.array([[0, 1], [2, 3]], dtype="float64")
    C = np.array([[0, 0], [0, 0]], dtype="float64")

    value = c_longlong(0)
    print ("value!: ", value.value)

    event_name = "perf::PERF_COUNT_HW_CPU_CYCLES"

    plasmalib.Plasma_DGEMM_Profiling(\
            c_int(2),\
            c_int(2),\
            c_int(2),\
            c_double(1),\
            A.ctypes.data_as(POINTER(c_double)),\
            c_int(2),\
            B.ctypes.data_as(POINTER(c_double)),\
            c_int(2),\
            c_double(1),\
            C.ctypes.data_as(POINTER(c_double)),\
            c_int(2),\
            c_int(256),\
            c_int(64),
            ctypes.c_char_p(event_name.encode('utf-8')),
            ctypes.pointer(value))

    print (C)
    print ("value: ", value.value)

    return

def test_plasma_dgels(plasmalib):

    print ("Test PLASMA DGELS")

    m = 10
    n = 2
    nrhs = 1

    A = np.random.rand(m, n)
    B = np.random.rand(m, nrhs)

    print (A)
    print (B)

    plasmalib.Plasma_DGELS(\
            c_int(m),\
            c_int(n),\
            c_int(nrhs),\
            A.ctypes.data_as(POINTER(c_double)),\
            B.ctypes.data_as(POINTER(c_double)),\
            c_int(256),\
            c_int(64))

    print ("Solution")
    print (B[0:n])

    return

if __name__ == "__main__":
    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    print ("omp_num_threads: ", omp_num_threads)

    plasmalib = ctypes.CDLL("./library/plasma_tuning.so", mode=ctypes.RTLD_GLOBAL)

    test_plasma_dgemm(plasmalib)
    test_plasma_dgels(plasmalib)

