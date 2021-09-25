//
// Created by Vladislav on 25.09.2021.
//

#ifndef CUDA_MMUL_CPU_MMUL_H
#define CUDA_MMUL_CPU_MMUL_H

void cpu_mmul(const float *A, const float *B, float *C, int N);
void hello_omp(int th_n);

#endif //CUDA_MMUL_CPU_MMUL_H
