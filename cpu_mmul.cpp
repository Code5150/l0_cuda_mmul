//
// Created by Vladislav on 25.09.2021.
//
#include <cstdio>
#include "cpu_mmul.h"
#include "omp.h"

void cpu_mmul(const float *A, const float *B, float *C, int N) {
    size_t i = 0, j = 0, k = 0;
    auto start_time = omp_get_wtime();
#pragma omp parallel for simd collapse(3) lastprivate(i, j, k) schedule(static)
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            for (k = 0; k < N; ++k)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
        }
    }
    auto end_time = omp_get_wtime();
    printf("Время выполнения на CPU: %f с\n", end_time-start_time);
}