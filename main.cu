#include <cstdio>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas.h"
#include <cassert>
#include <ctime>
#include <iostream>
#include "cpu_mmul.h"

#define ULLCAST(X) static_cast<unsigned long long>(X)
//#define LAB_DEBUG

template<typename t>
inline size_t malloc_size(size_t s) {
    return sizeof(t) * s;
}

/**
 * c = alpha*transa(a)*transb(b) + beta*c
 * [lda X ldb] * [ldb X ldc] = [lda X ldc]
 */
void cuda_mmul(cublasHandle_t *chandle, float* c_buffer, float* a, float* b, float *c,
               int lda, int ldb, int ldc, int m, int n, int k)  {
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm_v2(
            *chandle,
            //Transposing matrices from row-major to column-major
            CUBLAS_OP_T, CUBLAS_OP_T,
            m, n, k,
            &alpha,
            a, lda,
            b, ldb,
            &beta,
            c_buffer, ldc
            );

    //Transposing result back to row-major
    cublasSgeam(*chandle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n,
                &alpha,
                c_buffer, ldc,
                &beta,
                b, ldb,
                c, ldc
                );
}

void verify_result(const float *a, const float *b, const float *c, int N) {
    // For every row...
    for (int i = 0; i < N; i++) {
        // For every column...
        for (int j = 0; j < N; j++) {
            // For every element in the row-column pair
            float tmp = 0;
            for (int k = 0; k < N; k++) {
                // Accumulate the partial results
                tmp += a[i * N + k] * b[k * N + j];
            }
            // Check against the CPU result
            assert(tmp - c[i * N + j] < 1e-3);
            if(tmp - c[i * N + j] > 1e-3) {
                printf("Verification failed. tmp - c[i * N + j] > 1e-3\n");
                return;
            }
        }
    }
    printf("Результаты корректны\n");
}
#ifdef LAB_DEBUG
void print_matrix(const float* matrix, const size_t N) {
    for(size_t i = 0; i < N; ++i) {
        for(size_t j = 0; j < N; ++j) {
            printf("%f ", matrix[i*N + j]);
        }
        printf("\n");
    }
}
#endif

void scan_matrix_dim(int* matrix_dim) {
    printf("Введите размер матрицы (от 100 до 2000): ");
    scanf_s("%d", matrix_dim);
    while (*matrix_dim < 100 || *matrix_dim > 2000) {
        printf("Неверный размер матрицы. %d не входит в интервал [100;2000]\n", *matrix_dim);
        printf("Введите размер матрицы (от 100 до 2000): ");
        scanf_s("%d", matrix_dim);
    }
}

enum Options {
    NONE,
    EXIT = 0,
    PAR_MUL_GPU = 1,
    PAR_MUL_CPU = 2
};

int main() {
    setlocale(LC_ALL, "ru_RU.UTF-8");

    curandGenerator_t gen;
    cublasHandle_t HANDLE;
    cublasCreate_v2(&HANDLE);

    float *a, *b, *c, *c_buffer;
    float *cpuA, *cpuB, *cpuC;

    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, ULLCAST(clock()));

    int matrix_dim = 0;
    Options option = Options::NONE;
    bool main_cycle = true;
    while(main_cycle) {
        printf("Список действий:\n");
        printf("0 - выход из программы\n");
        printf("1 - умножение матриц на GPU\n");
        printf("2 - умножение матриц на CPU\n");
        printf("Выберите действие:");
        scanf_s("%d", &option);
        switch (option) {
            case Options::PAR_MUL_GPU: {
                scan_matrix_dim(&matrix_dim);

                size_t matrix_size = matrix_dim*matrix_dim;
                size_t m_size = malloc_size<float>(matrix_size);

                cpuA = new float[matrix_size] {};
                cpuB = new float[matrix_size] {};
                cpuC = new float[matrix_size] {};

                curandGenerateUniform(gen, cpuA, matrix_size);
                curandGenerateUniform(gen, cpuB, matrix_size);

                cudaMalloc((void**)&a, m_size);
                cudaMalloc((void**)&b, m_size);
                cudaMalloc((void**)&c, m_size);
                cudaMalloc((void**)&c_buffer, m_size);

#ifdef LAB_DEBUG
                print_matrix(cpuA, matrix_dim);
                print_matrix(cpuB, matrix_dim);
#endif

                cudaMemcpy(a, cpuA, m_size, cudaMemcpyHostToDevice);
                cudaMemcpy(b, cpuB, m_size, cudaMemcpyHostToDevice);

                // Initialize events
                cudaEvent_t start, stop;
                float elapsedTime;

                // Create events
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                // Record events
                cudaEventRecord(start, nullptr);

                cuda_mmul(&HANDLE, c_buffer, a, b, c,
                          matrix_dim, matrix_dim, matrix_dim,
                          matrix_dim, matrix_dim, matrix_dim);

                cudaEventRecord(stop, nullptr);

                // Waiting to kernel finish
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsedTime, start, stop);
                printf("Время выполнения на GPU: %.6f мс\n", elapsedTime);

#ifdef LAB_DEBUG
                cudaMemcpy(cpuA, a, m_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(cpuB, b, m_size, cudaMemcpyDeviceToHost);
                print_matrix(cpuA, matrix_dim);
                print_matrix(cpuB, matrix_dim);
#endif
                cudaMemcpy(cpuC, c, m_size, cudaMemcpyDeviceToHost);
#ifdef LAB_DEBUG
                print_matrix(cpuC, matrix_dim);
#endif
                verify_result(cpuA, cpuB, cpuC, matrix_dim);

                cudaFree(a);
                cudaFree(b);
                cudaFree(c);
                cudaFree(c_buffer);

                delete[] cpuA;
                delete[] cpuB;
                delete[] cpuC;

                break;
            }
            case Options::PAR_MUL_CPU: {
                scan_matrix_dim(&matrix_dim);

                size_t matrix_size = matrix_dim*matrix_dim;

                cpuA = new float[matrix_size] {};
                cpuB = new float[matrix_size] {};
                cpuC = new float[matrix_size] {0.0f};

                curandGenerateUniform(gen, cpuA, matrix_size);
                curandGenerateUniform(gen, cpuB, matrix_size);

#ifdef LAB_DEBUG
                print_matrix(a, matrix_size);
                print_matrix(b, matrix_size);
#endif

                cpu_mmul(cpuA, cpuB, cpuC, matrix_dim);

#ifdef LAB_DEBUG
                print_matrix(c, matrix_size);
#endif
                verify_result(cpuA, cpuB, cpuC, matrix_dim);

                delete[] cpuA;
                delete[] cpuB;
                delete[] cpuC;

                break;
            }
            case Options::EXIT:{
                main_cycle = false;
                break;
            }
            default: {
                printf("Данной опции не существует. Попробуйте ещё раз.\n");
                break;
            }
        }
    }

    return 0;
}
