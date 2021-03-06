#include <cstdio>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas.h"
#include <cassert>
#include <ctime>
#include <iostream>
#include "cpu_mmul.h"

#define ULLCAST(X) static_cast<unsigned long long>(X)
#define INTCAST(X) static_cast<int>(X)
//#define LAB_DEBUG

constexpr const int BLOCK_SIZE = 500;

template<typename t>
inline size_t malloc_size(size_t s) {
    return sizeof(t) * s;
}

/**
 * c = alpha*transa(a)*transb(b) + beta*c
 * [lda X ldb] * [ldb X ldc] = [lda X ldc]
 */
/*void cublasMmul(cublasHandle_t *chandle, float* c_buffer, float* a, float* b, float *c,
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
}*/

//Considering matrix B as transposed
__global__ void cudaMmul(const float* a, const float* b, float *c, int N, int matrix_size) {
    unsigned int blockBegin = blockIdx.x * BLOCK_SIZE;// + blockIdx.y * N;
    unsigned int blockShared = blockBegin + blockIdx.y * N;

    __shared__ float aShared[BLOCK_SIZE];

    //if(threadIdx.x == 0)printf("Block %d, %d begins from %d\n",blockIdx.x, blockIdx.y, blockBegin);
    //Copy values from A block and initial values from
    aShared[threadIdx.x] = a[blockShared + threadIdx.x];
    __syncthreads();
    float sum;
    for(unsigned int i = blockBegin; i < matrix_size; i += N) {
        sum = 0.0f;
        sum += aShared[threadIdx.x] * b[i + threadIdx.x];
        __syncthreads();
        unsigned int ind = blockIdx.y*N + (i / N);
        atomicAdd(&c[ind], sum);
    }

}

void transpose(cublasHandle_t *chandle, float* result_buffer, float* b, float *result,
               int ldb, int ldc, int m, int n) {
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgeam(*chandle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n,
                &alpha,
                result_buffer, ldc,
                &beta,
                b, ldb,
                result, ldc
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
            //assert(tmp - c[i * N + j] < 1e-3);
            if(tmp - c[i * N + j] > 1e-3) {
                printf("Verification failed. tmp - c[i * N + j] > 1e-3\n");
                return;
            }
        }
    }
    printf("???????????????????? ??????????????????\n");
}
#ifdef LAB_DEBUG
void print_matrix(const float* matrix, const size_t N) {
    for(size_t i = 0; i < N; ++i) {
        for(size_t j = 0; j < N; ++j) {
            printf("%f ", matrix[i*N + j]);
        }
        printf("\n");
    }
    printf("\n");
}
#endif

void scan_matrix_dim(int* matrix_dim) {
    printf("?????????????? ???????????? ?????????????? (???? 100 ???? 2000): ");
    scanf_s("%d", matrix_dim);
    while (*matrix_dim < 0 || *matrix_dim > 2000) {
        printf("???????????????? ???????????? ??????????????. %d ???? ???????????? ?? ???????????????? [100;2000]\n", *matrix_dim);
        printf("?????????????? ???????????? ?????????????? (???? 100 ???? 2000): ");
        scanf_s("%d", matrix_dim);
    }
}

cudaDeviceProp showProperties() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);//?????????????????????? ???????????????????? GPU ?? ?????????????? 0

    printf("Device name : %s\n", deviceProp.name);
    printf("Total global memory : %llu MB\n",
           deviceProp.totalGlobalMem / 1024 / 1024);
    printf("Shared memory per block : %zu\n",
           deviceProp.sharedMemPerBlock);
    printf("Registers per block : %d\n",
           deviceProp.regsPerBlock);
    printf("Warp size : %d\n", deviceProp.warpSize);
    printf("Memory pitch : %zu\n", deviceProp.memPitch);
    printf("Max threads per block : %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("Max threads dimensions : x = %d, y = %d, z = %d\n",
           deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("Max grid size: x = %d, y = %d, z = %d\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("Clock rate: %d\n", deviceProp.clockRate);
    printf("Total constant memory: %zu\n",
           deviceProp.totalConstMem);
    printf("Compute capability: %d.%d\n",
           deviceProp.major, deviceProp.minor);
    printf("Texture alignment: %zu\n",
           deviceProp.textureAlignment);
    printf("Device overlap: %d\n",
           deviceProp.deviceOverlap);
    printf("Multiprocessor count: %d\n",
           deviceProp.multiProcessorCount);
    printf("Kernel execution timeout enabled: %s\n",
           deviceProp.kernelExecTimeoutEnabled ? "true" :
           "false");
    printf("Can map host memory: %s\n",
           deviceProp.canMapHostMemory ? "true" :
           "false");
    printf("Device has Compute Capability %d.%d\n",
           deviceProp.major, deviceProp.minor);

    return deviceProp;
}

enum Options {
    NONE,
    EXIT = 0,
    PAR_MUL_GPU = 1,
    PAR_MUL_CPU = 2,
    DEVICE_INFO = 3
};

int main() {
    setlocale(LC_ALL, "ru_RU.UTF-8");

    curandGenerator_t gen;
    cublasHandle_t HANDLE;
    cublasCreate_v2(&HANDLE);

    float *a, *b, *c, *buffer;
    float *cpuA, *cpuB, *cpuC;

    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, ULLCAST(clock()));

    int matrix_dim = 0;
    Options option = Options::NONE;
    bool main_cycle = true;
    while(main_cycle) {
        printf("???????????? ????????????????:\n");
        printf("0 - ?????????? ???? ??????????????????\n");
        printf("1 - ?????????????????? ???????????? ???? GPU\n");
        printf("2 - ?????????????????? ???????????? ???? CPU\n");
        printf("3 - ???????????????????? ?? GPU\n");
        printf("???????????????? ????????????????:");
        scanf_s("%d", &option);
        switch (option) {
            case Options::PAR_MUL_GPU: {
                scan_matrix_dim(&matrix_dim);

                int matrix_size = matrix_dim*matrix_dim;
                size_t m_size = malloc_size<float>(matrix_size);

                cpuA = new float[matrix_size] {};
                cpuB = new float[matrix_size] {};
                cpuC = new float[matrix_size] {0.0f};

                curandGenerateUniform(gen, cpuA, matrix_size);
                curandGenerateUniform(gen, cpuB, matrix_size);

                cudaMalloc((void**)&a, m_size);
                cudaMalloc((void**)&b, m_size);
                cudaMalloc((void**)&c, m_size);
                cudaMalloc((void**)&buffer, m_size);

#ifdef LAB_DEBUG
                print_matrix(cpuA, matrix_dim);
                print_matrix(cpuB, matrix_dim);
#endif

                cudaMemcpy(a, cpuA, m_size, cudaMemcpyHostToDevice);
                cudaMemcpy(buffer, cpuB, m_size, cudaMemcpyHostToDevice);

                // Initialize events
                cudaEvent_t start, stop;
                float elapsedTime;

                // Create events
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                // Record events
                cudaEventRecord(start, nullptr);

                transpose(&HANDLE, buffer, a, b, matrix_dim, matrix_dim, matrix_dim, matrix_dim);

                /*cuda_mmul(&HANDLE, buffer, a, b, c,
                          matrix_dim, matrix_dim, matrix_dim,
                          matrix_dim, matrix_dim, matrix_dim);*/
                cudaMmul<<<dim3(matrix_dim / BLOCK_SIZE, matrix_dim), BLOCK_SIZE>>>(a, b, c, matrix_dim, INTCAST(matrix_size));

                cudaEventRecord(stop, nullptr);

                // Waiting to kernel finish
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&elapsedTime, start, stop);
                printf("?????????? ???????????????????? ???? GPU: %.6f ????\n", elapsedTime);

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
                cudaFree(buffer);

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
            case Options::DEVICE_INFO: {
                showProperties();
                break;
            }
            case Options::EXIT:{
                main_cycle = false;
                break;
            }
            default: {
                printf("???????????? ?????????? ???? ????????????????????. ???????????????????? ?????? ??????.\n");
                break;
            }
        }
    }

    return 0;
}
