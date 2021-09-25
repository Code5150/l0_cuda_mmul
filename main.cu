#include <cstdio>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas.h"
#include <cassert>
#include <ctime>
#include <iostream>

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

enum Options {
    NONE,
    EXIT = 0,
    PAR_MUL = 1
};

int main() {

    curandGenerator_t gen;
    cublasHandle_t HANDLE;
    cublasCreate_v2(&HANDLE);

    float *a, *b, *c, *c_buffer;
    float *cpuA, *cpuB, *cpuC;
    setlocale(LC_ALL, "ru_RU.UTF-8");

    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, ULLCAST(clock()));

    int matrix_dim = 0;
    Options option = Options::NONE;
    bool main_cycle = true;
    while(main_cycle) {
        std::cout << "Список действий:" << std::endl;
        std::cout << "0 - выход из программы" << std::endl;
        std::cout << "1 - умножение матриц" << std::endl;
        std::cout << "Выберите действие:" << std::endl;
        scanf_s("%d", &option);
        switch (option) {
            case Options::PAR_MUL: {
                std::cout << "Введите размер матрицы (от 100 до 2000): ";
                scanf_s("%d", &matrix_dim);
                std::cout << std::endl;
                while (matrix_dim < 100 || matrix_dim > 2000) {
                    std::cout <<"Неверный размер матрицы. "  << std::endl;
                    std::cout <<" Введите размер матрицы (от 100 до 2000): ";
                    scanf_s("%d", &matrix_dim);
                    std::cout << std::endl;
                }

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
                printf("Время выполнения на GPU: %.10f milliseconds\n", elapsedTime);

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
            case Options::EXIT:{
                main_cycle = false;
                break;
            }
            default: {
                std::cout << "Данной опции не существует. Попробуйте ещё раз.\n";
                break;
            }
        }
    }

    return 0;
}
