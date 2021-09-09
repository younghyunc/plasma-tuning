// Author: Younghyun Cho <younghyun@berkeley.edu>
//

#include <iostream>
#include <algorithm>
#include <omp.h>
#include "plasma.h"
#include "plasma_tuning.h"

typedef double plasma_time_t;

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

