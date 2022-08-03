// Author: Younghyun Cho <younghyun@berkeley.edu>
//

#include <iostream>
#include <algorithm>
#include <papi.h>
#include <omp.h>
#include <unistd.h>
#include <pthread.h>
#include "plasma.h"
#include "plasma_tuning.h"

typedef double plasma_time_t;

int StartProfiling(char* event_name)
{
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "Error inititalizing PAPI! %s\n", PAPI_strerror(retval));
    }

    int num_threads = omp_get_num_procs();
    for (int i = 0; i < num_threads; i++) {
        eventset[i] = PAPI_NULL;
        //printf("eventset[%d]: %d\n", i, eventset[i]);
    }

    #pragma omp parallel
    {
        retval = PAPI_thread_init(pthread_self);
        if ( retval != PAPI_OK ) {
            fprintf(stderr, "Error thread init\n");
        }

        int thread_id = omp_get_thread_num();

        retval = PAPI_create_eventset(&eventset[thread_id]);
        if (retval != PAPI_OK) {
            fprintf(stderr,"Error creating eventset! %s\n", PAPI_strerror(retval));
        }

        retval = PAPI_add_named_event(eventset[thread_id], event_name);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Error adding event: %s\n", PAPI_strerror(retval));
        }

        PAPI_reset(eventset[thread_id]);
        retval = PAPI_start(eventset[thread_id]);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Error starting: %s\n", PAPI_strerror(retval));
        }
    }

    return 0;
}

int StopProfiling(long long int* value)
{
    int num_threads = omp_get_num_procs();
    long long* count = (long long*)malloc(num_threads * sizeof(long long));
    for (int i = 0; i < num_threads; i++)
        count[i] = 0;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int retval = PAPI_stop(eventset[thread_id], &count[thread_id]);
        if (retval != PAPI_OK) {
            fprintf(stderr, "Error stopping: %s\n", PAPI_strerror(retval));
        }

        PAPI_cleanup_eventset(eventset[thread_id]);
        PAPI_destroy_eventset(&eventset[thread_id]);
    }

    long long value_ = 0;
    for (int i = 0; i < num_threads; i++)
        value_ += count[i];
    *value = value_;

    printf("profiled value: %lld\n", value_);

    free(count);

    return 0;
}

int Plasma_DGEMM_Profiling(int m, int n, int k,
        double alpha, double* A, int lda,
        double* B, int ldb,
        double beta, double* C, int ldc,
        int nb, int ib, char* event_name, long long int* value)
{
    std::cout << "profiling event code: " << event_name << std::endl;
    StartProfiling(event_name);

    std::cout << "[Plasma_DGEMM] calling plasma_dgemm" << std::endl;
    std::cout << m << " " << n << " " << k << std::endl;
    std::cout << lda << " " << ldb << " " << ldc << std::endl;

    plasma_init();

    //plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, nb);
    plasma_set(PlasmaIb, ib);

    plasma_enum_t transa = PlasmaNoTrans;
    plasma_enum_t transb = PlasmaNoTrans;

    plasma_time_t start = omp_get_wtime();

    int ret = plasma_dgemm(
        transa, transb,
        m, n, k,
        alpha, A, lda,
               B, ldb,
         beta, C, ldc);

    printf("ret: %d\n", ret);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    printf("time: %lf\n", time);

    StopProfiling(value);

    return ret;
}

int Plasma_DGEMM(int m, int n, int k,
        double alpha, double* A, int lda,
        double* B, int ldb,
        double beta, double* C, int ldc,
        int nb, int ib)
{
    std::cout << "[Plasma_DGEMM] calling plasma_dgemm" << std::endl;
    std::cout << m << " " << n << " " << k << std::endl;
    std::cout << lda << " " << ldb << " " << ldc << std::endl;

    plasma_init();

    //plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, nb);
    plasma_set(PlasmaIb, ib);

    plasma_enum_t transa = PlasmaNoTrans;
    plasma_enum_t transb = PlasmaNoTrans;

    plasma_time_t start = omp_get_wtime();

    int ret = plasma_dgemm(
        transa, transb,
        m, n, k,
        alpha, A, lda,
               B, ldb,
         beta, C, ldc);

    printf("ret: %d\n", ret);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    printf("time: %lf\n", time);

    return ret;
}

int Plasma_DGEQRF_Profiling(int m, int n, double* A, int nb, int ib,
        char* event_name, long long* value)
{
    std::cout << "[Plasma_DGEQRF] calling plasma_dgeqrf" << std::endl;

    StartProfiling(event_name);

    int lda = std::max(1, m);

    plasma_init();

    //plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, nb);
    plasma_set(PlasmaIb, ib);

    plasma_enum_t trans = PlasmaNoTrans;

    plasma_time_t start = omp_get_wtime();

    plasma_desc_t T;
    int ret = plasma_dgeqrf(m, n, A, lda, &T);
    printf("ret: %d\n", ret);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    plasma_desc_destroy(&T);
    printf("time: %lf\n", time);

    plasma_finalize();

    StopProfiling(value);

    return ret;
}

int Plasma_DGEQRF(int m, int n, double* A, int nb, int ib)
{
    std::cout << "[Plasma_DGEQRF] calling plasma_dgeqrf" << std::endl;

    int lda = std::max(1, m);

    plasma_init();

    //plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, nb);
    plasma_set(PlasmaIb, ib);

    plasma_enum_t trans = PlasmaNoTrans;

    plasma_time_t start = omp_get_wtime();

    plasma_desc_t T;
    int ret = plasma_dgeqrf(m, n, A, lda, &T);
    printf("ret: %d\n", ret);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    plasma_desc_destroy(&T);
    printf("time: %lf\n", time);

    plasma_finalize();

    return ret;
}

int Plasma_DGELS(int m, int n, int nrhs,
        double* A, double* B,
        int nb, int ib)
{
    std::cout << "[Plasma_DGELS] calling plasma_dgels" << std::endl;

    int pada = 0;
    int padb = 0;

    int lda = std::max(1, m + pada);
    int ldb = std::max(1, std::max(m, n) + padb);

    plasma_init();

    //plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, nb);
    plasma_set(PlasmaIb, ib);

    plasma_enum_t trans = PlasmaNoTrans;

    plasma_time_t start = omp_get_wtime();

    plasma_desc_t T;
    int ret = plasma_dgels(
        trans, m, n, nrhs,
        A, lda,
        &T,
        B, ldb);
    printf("ret: %d\n", ret);

    plasma_time_t stop = omp_get_wtime();
    plasma_time_t time = stop-start;

    plasma_desc_destroy(&T);
    printf("time: %lf\n", time);

    plasma_finalize();

    return ret;
}

