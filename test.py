#! /usr/bin/env python

# Author: Younghyun Cho <younghyun@berkeley.edu>

import os
import sys
import ctypes
import numpy as np

from ctypes import c_int, c_double, POINTER

def test_plasma_dgemm(plasmalib):

    print ("Test PLASMA DGEMM")

    A = np.array([[0, 1], [2, 3]], dtype="float64")
    B = np.array([[0, 1], [2, 3]], dtype="float64")
    C = np.array([[0, 0], [0, 0]], dtype="float64")

    plasmalib.Plasma_DGEMM(\
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
            c_int(64))

    print (C)

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

    plasmalib = ctypes.CDLL("./library/plasma_tuning.so", mode=ctypes.RTLD_GLOBAL)

    test_plasma_dgemm(plasmalib)
    test_plasma_dgels(plasmalib)

