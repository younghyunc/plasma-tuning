// Author: Younghyun Cho <younghyun@berkeley.edu>
//

#ifdef __cplusplus
extern "C"
{
#endif

int Plasma_DGEMM(int m, int n, int k,
        double alpha, double* A, int lda,
        double* B, int ldb,
        double beta, double* C, int ldc,
        int nb, int ib);

int Plasma_DGELS(int m, int n, int nrhs,
        double* A, double* B,
        int nb, int ib);

#ifdef __cplusplus
}
#endif
