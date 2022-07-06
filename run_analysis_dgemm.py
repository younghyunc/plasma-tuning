#! /usr/bin/env python

# Author: Younghyun Cho <younghyun@berkeley.edu>

import os
import sys
import ctypes
import json
import time
import numpy as np

from ctypes import c_int, c_double, c_longlong, POINTER

events = [\
    "perf::PERF_COUNT_HW_CPU_CYCLES",
    "perf::CYCLES",
    "perf::CPU-CYCLES",
    "perf::PERF_COUNT_HW_INSTRUCTIONS",
    "perf::INSTRUCTIONS",
    "perf::PERF_COUNT_HW_CACHE_REFERENCES",
    "perf::CACHE-REFERENCES",
    "perf::PERF_COUNT_HW_CACHE_MISSES",
    "perf::CACHE-MISSES",
    "perf::PERF_COUNT_HW_BRANCH_INSTRUCTIONS",
    "perf::BRANCH-INSTRUCTIONS",
    "perf::BRANCHES",
    "perf::PERF_COUNT_HW_BRANCH_MISSES",
    "perf::BRANCH-MISSES",
    "perf::PERF_COUNT_HW_STALLED_CYCLES_FRONTEND",
    "perf::STALLED-CYCLES-FRONTEND",
    "perf::IDLE-CYCLES-FRONTEND",
    "perf::PERF_COUNT_HW_STALLED_CYCLES_BACKEND",
    "perf::STALLED-CYCLES-BACKEND",
    "perf::IDLE-CYCLES-BACKEND",
    "perf::PERF_COUNT_SW_CPU_CLOCK",
    "perf::CPU-CLOCK",
    "perf::PERF_COUNT_SW_TASK_CLOCK",
    "perf::TASK-CLOCK",
    "perf::PERF_COUNT_SW_PAGE_FAULTS",
    "perf::PAGE-FAULTS",
    "perf::FAULTS",
    "perf::PERF_COUNT_SW_CONTEXT_SWITCHES",
    "perf::CONTEXT-SWITCHES",
    "perf::CS",
    "perf::PERF_COUNT_SW_CPU_MIGRATIONS",
    "perf::CPU-MIGRATIONS",
    "perf::MIGRATIONS",
    "perf::PERF_COUNT_SW_PAGE_FAULTS_MIN",
    "perf::MINOR-FAULTS",
    "perf::PERF_COUNT_SW_PAGE_FAULTS_MAJ",
    "perf::MAJOR-FAULTS",
    "perf::PERF_COUNT_SW_CGROUP_SWITCHES",
    "perf::CGROUP-SWITCHES",
    "perf::PERF_COUNT_HW_CACHE_L1D",
    "perf::L1-DCACHE-LOADS",
    "perf::L1-DCACHE-LOAD-MISSES",
    "perf::L1-DCACHE-PREFETCHES",
    "perf::PERF_COUNT_HW_CACHE_L1I",
    "perf::L1-ICACHE-LOADS",
    "perf::L1-ICACHE-LOAD-MISSES",
    "perf::PERF_COUNT_HW_CACHE_DTLB",
    "perf::DTLB-LOADS",
    "perf::DTLB-LOAD-MISSES",
    "perf::PERF_COUNT_HW_CACHE_ITLB",
    "perf::ITLB-LOADS",
    "perf::ITLB-LOAD-MISSES",
    "perf::PERF_COUNT_HW_CACHE_BPU",
    "perf::BRANCH-LOADS",
    "perf::BRANCH-LOAD-MISSES",
    "perf_raw::r0000",
    "L1_ITLB_MISS_L2_ITLB_HIT",
    "RETIRED_SSE_AVX_FLOPS",
    "DIV_CYCLES_BUSY_COUNT",
    "DIV_OP_COUNT",
    "RETIRED_BRANCH_INSTRUCTIONS",
    "RETIRED_FAR_CONTROL_TRANSFERS",
    "RETIRED_INDIRECT_BRANCH_INSTRUCTIONS_MISPREDICTED",
    "RETIRED_BRANCH_INSTRUCTIONS_MISPREDICTED",
    "RETIRED_TAKEN_BRANCH_INSTRUCTIONS",
    "RETIRED_TAKEN_BRANCH_INSTRUCTIONS_MISPREDICTED",
    "RETIRED_CONDITIONAL_BRANCH_INSTRUCTIONS",
    "RETIRED_UOPS",
    "RETIRED_FUSED_INSTRUCTIONS",
    "RETIRED_INSTRUCTIONS",
    "RETIRED_NEAR_RETURNS",
    "RETIRED_NEAR_RETURNS_MISPREDICTED",
    "RETIRED_BRANCH_MISPREDICTED_DIRECTION_MISMATCH",
    "INSTRUCTION_CACHE_REFILLS_FROM_L2",
    "INSTRUCTION_CACHE_REFILLS_FROM_SYSTEM",
    "L2_PREFETCH_HIT_L2",
    "L2_PREFETCH_HIT_L3",
    "L2_PREFETCH_MISS_L3",
    "BAD_STATUS_2",
    "RETIRED_CLFLUSH_INSTRUCTIONS",
    "RETIRED_CPUID_INSTRUCTIONS",
    "SMI_RECEIVED",
    "INTERRUPT_TAKEN",
    "MISALIGNED_LOADS",
    "CYCLES_NOT_IN_HALT",
    "TLB_FLUSHES",
    "PREFETCH_INSTRUCTIONS_DISPATCHED",
    "STORE_TO_LOAD_FORWARD",
    "STORE_COMMIT_CANCELS_2",
    "L1_BTB_CORRECTION",
    "L2_BTB_CORRECTION",
    "DYNAMIC_INDIRECT_PREDICTIONS",
    "DECODER_OVERRIDE_BRANCH_PRED",
    "UOPS_QUEUE_EMPTY",
    "DISPATCH_RESOURCE_STALL_CYCLES_0",
    "FP_DISPATCH_FAULTS"
]

def run_dgemm_analysis(A, B, m, n, k, alpha, beta, nb, ib):

    logfile = "plasma_dgemm.json"

    json_data_arr = {
        "func_eval": []
    }

    if not os.path.exists(logfile):
        with open(logfile, "w") as f_out:
            json.dump(json_data_arr, f_out, indent=2)

    with open(logfile, "r") as f_in:
        json_data_arr = json.load(f_in)

    for func_eval in json_data_arr["func_eval"]:
        if func_eval["task_parameter"]["m"] == m and\
           func_eval["task_parameter"]["n"] == n and\
           func_eval["task_parameter"]["k"] == k and\
           func_eval["tuning_parameter"]["nb"] == nb and\
           func_eval["tuning_parameter"]["ib"] == ib:
            return

    profiling_result = {}
    for event_name in events:
        C = np.zeros((m, k), dtype="float64")

        value = c_longlong(0)

        plasmalib.Plasma_DGEMM_Profiling(\
                c_int(m),\
                c_int(n),\
                c_int(k),\
                c_double(alpha),\
                A.ctypes.data_as(POINTER(c_double)),\
                c_int(m),\
                B.ctypes.data_as(POINTER(c_double)),\
                c_int(n),\
                c_double(beta),\
                C.ctypes.data_as(POINTER(c_double)),\
                c_int(k),\
                c_int(nb),\
                c_int(ib),
                ctypes.c_char_p(event_name.encode('utf-8')),
                ctypes.pointer(value))

        profiling_result[event_name] = value.value

    runtimes = []
    for i in range(3):
        C = np.zeros((m, k), dtype="float64")

        tic = time.time()
        plasmalib.Plasma_DGEMM(\
                c_int(m),\
                c_int(n),\
                c_int(k),\
                c_double(alpha),\
                A.ctypes.data_as(POINTER(c_double)),\
                c_int(m),\
                B.ctypes.data_as(POINTER(c_double)),\
                c_int(n),\
                c_double(beta),\
                C.ctypes.data_as(POINTER(c_double)),\
                c_int(k),\
                c_int(nb),\
                c_int(ib))
        runtime = time.time() - tic
        runtimes.append(runtime)

    point = {
        "task_parameter": {
            "m": m,
            "n": n,
            "k": k
        },
        "machine_configuration": {
            "machine_name": "local",
            "epyc": {
                "cores": 48,
                "nodes": 1,
                "processor_name": "AMD EPYC 7402P 24-Core Processor"
            }
        },
        "software_configuration": {
            "plasma": [21,8,29]
        },
        "constant": {
            "alpha": alpha,
            "beta": beta,
            "num_threads": int(os.environ.get("OMP_NUM_THREADS"))
        },
        "tuning_parameter": {
            "nb": nb,
            "ib": ib
        },
        "evaluation_result": {
            "runtime": np.average(runtimes)
        },
        "evaluation_details": {
            "runtime": {
                "evaluations": runtimes,
                "objective_scheme": "average"
            }
        },
        "profiling_result": profiling_result,
        "source": "measure"
    }

    json_data_arr["func_eval"].append(point)
    with open(logfile, "w") as f_out:
        json.dump(json_data_arr, f_out, indent=2)

    return

def analysis_dgemm(plasmalib):

    m = 4096
    n = 4096
    k = 4096

    alpha = 2.0
    beta = 3.0

    A = np.zeros((m, n), dtype="float64")
    A[:] = np.random.randn(*A.shape)

    B = np.zeros((n, k), dtype="float64")
    B[:] = np.random.randn(*B.shape)

    for nb in range(32, 512+32, 32):
        for ib in range(32, nb+32, 32):
            run_dgemm_analysis(A, B, m, n, k, alpha, beta, nb, ib)

    return

if __name__ == "__main__":

    plasmalib = ctypes.CDLL("./library/plasma_tuning.so", mode=ctypes.RTLD_GLOBAL)

    analysis_dgemm(plasmalib)

