/**
 * @file main.cu
 * @author Jia-Baos (18383827268@163.com)
 * @brief
 * @version 0.1
 * @date 2026-01-15
 *
 * @copyright Copyright (c) 2026
 *
 */
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <limits>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            std::cerr << "CUDA error in " << __FILE__ << ":"         \
                      << __LINE__ << ": " << cudaGetErrorString(err) \
                      << std::endl;                                  \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

void matmulCPU(float *A, float *B, float *C, int M, int N, int K)
{
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {

            float sum{};
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void matmulGPU_native(float *A, float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 防止线程访问超出矩阵数组边界的内存地址，避免出现非法内存访问错误，同时处理矩阵尺寸无法被线程块尺寸整除的场景
    if (row < M && col < N) {
        float sum{};
        for (size_t k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }

    // // 行方向：从初始 row 开始，每次跨步 stride_row，覆盖所有符合条件的行
    // for (int r = row; r < M; r += stride_row) {
    //     // 列方向：从初始 col 开始，每次跨步 stride_col，覆盖所有符合条件的列
    //     for (int c = col; c < N; c += stride_col) {
    //         float sum{};
    //         for (size_t k = 0; k < K; k++) {
    //             sum += A[r * K + k] * B[k * N + c];
    //         }
    //         C[r * N + c] = sum;
    //     }
    // }
}

#define TILE_SIZE 16
__global__ void matmulGPU_tiled(float *A, float *B, float *C, int M, int N, int K)
{

    // 指定 blockDim.x blockDim.y 为 TILE_SIZE
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // shared_memory
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;   // 乘法计算后 row，col 位置位置的数值
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE; // 向上取整得到 K 维度需要拆分的小块数量

    for (int t = 0; t < numTiles; t++) {
        // load A and B to shared memory
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        int b_row = t * TILE_SIZE + ty;
        if (col < N && b_row < K) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();    // 同步整个 block 中的线程

        // use shared memory to calculate tiles value
        for (size_t k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();    // 保证计算完成后再更新 As Bs
    }

    // write back
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void matmulGPU_tiled4(float *A, float *B, float *C, int M, int N, int K)
{
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int block_row = blockIdx.y * TILE_SIZE * 4;
    int block_col = blockIdx.x * TILE_SIZE * 4;

    // shared_memory
    __shared__ float As[TILE_SIZE * 4][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE * 4];

    float sum[4][4] = { { 0.0f } };
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // load A and B to shared memory
        int tile_k = t * TILE_SIZE;

        for (size_t i = 0; i < 4; i++) {
            int a_row = block_row + ty * 4 + i;
            int a_col = tile_k + tx;

            As[ty * 4 + i][tx] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }

        for (size_t j = 0; j < 4; j++) {
            int b_row = tile_k + ty;
            int b_col = block_col + tx * 4 + j;

            Bs[ty][tx * 4 + j] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }

        block.sync();

        for (size_t k = 0; k < TILE_SIZE; k++) {
            float a_reg[4];
            float b_reg[4];

            for (size_t i = 0; i < 4; i++) {
                a_reg[i] = As[ty * 4 + i][k];
            }

            for (size_t j = 0; j < 4; j++) {
                b_reg[j] = Bs[k][tx * 4 + j];
            }

            for (size_t i = 0; i < 4; i++) {
                for (size_t j = 0; j < 4; j++) {
                    sum[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        block.sync();
    }

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            int c_row = block_row + ty * 4 + i;
            int c_col = block_col + tx * 4 + j;

            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = sum[i][j];
            }
        }
    }
}

void initMatRandom(float *mat, int rows, int cols)
{
    for (size_t i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

bool verfyResult(float *C_cpu, float *C_gpu, int M, int N)
{
    const float epsilon = 1e-3;
    size_t errCount{};

    for (size_t i = 0; i < M * N; i++) {

        float diff = std::abs(C_cpu[i] - C_gpu[i]);
        if (diff > epsilon) {
            errCount++;
        }
    }

    if (errCount > 0) {

        std::cerr << "oops we found: " << errCount << std::endl;
        return false;
    }

    return true;
}

int main()
{
    int M = 1024;
    int N = 1024;
    int K = 1024;

    float *h_A = (float *)malloc(M * K * sizeof(float));
    float *h_B = (float *)malloc(K * N * sizeof(float));
    float *h_C = (float *)malloc(M * N * sizeof(float));
    float *h_C_native = (float *)malloc(M * N * sizeof(float));
    float *h_C_tiled = (float *)malloc(M * N * sizeof(float));
    float *h_C_tiled4 = (float *)malloc(M * N * sizeof(float));

    float *d_A;
    float *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    srand(time(NULL));
    initMatRandom(h_A, M, K);
    initMatRandom(h_B, K, N);

    // CPU version
    clock_t start_cpu = clock();
    matmulCPU(h_A, h_B, h_C, M, N, K);
    clock_t end_cpu = clock();
    double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC * 1000;
    std::cout << "matmulCPU time: " << cpu_time << " ms" << std::endl;

    // GPU native version
    dim3 block_dim(16, 16);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x, (M + block_dim.y - 1) / block_dim.y);

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start_1;
    cudaEvent_t stop_1;

    cudaEventCreate(&start_1);
    cudaEventCreate(&stop_1);

    cudaEventRecord(start_1);

    matmulGPU_native<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop_1);
    cudaEventSynchronize(stop_1);

    cudaDeviceSynchronize();

    float gpu_time_1;
    cudaEventElapsedTime(&gpu_time_1, start_1, stop_1);
    std::cout << "matmulGPU_native, time: " << gpu_time_1 << " ms" << std::endl;

    cudaMemcpy(h_C_native, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start_1);
    cudaEventDestroy(stop_1);

    if (verfyResult(h_C, h_C_native, M, N)) {
        std::cout << "result is correct" << std::endl;
    } else {
        std::cout << "result is uncorrect" << std::endl;
    }

    cudaEvent_t start_2;
    cudaEvent_t stop_2;

    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);

    cudaEventRecord(start_2);

    matmulGPU_tiled<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop_2);
    cudaEventSynchronize(stop_2);

    cudaDeviceSynchronize();

    float gpu_time_2;
    cudaEventElapsedTime(&gpu_time_2, start_2, stop_2);
    std::cout << "matmulGPU_tiled, time: " << gpu_time_2 << " ms" << std::endl;

    cudaMemcpy(h_C_tiled, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start_2);
    cudaEventDestroy(stop_2);

    if (verfyResult(h_C, h_C_tiled, M, N)) {
        std::cout << "result is correct" << std::endl;
    } else {
        std::cout << "result is uncorrect" << std::endl;
    }

    dim3 block_dim_cg(16, 16);
    dim3 grid_dim_cg((N + 64 - 1) / 64, (M + 64 - 1) / 64);

    cudaEvent_t start_3;
    cudaEvent_t stop_3;

    cudaEventCreate(&start_3);
    cudaEventCreate(&stop_3);

    cudaEventRecord(start_3);

    matmulGPU_tiled4<<<grid_dim_cg, block_dim_cg>>>(d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop_3);
    cudaEventSynchronize(stop_3);

    cudaDeviceSynchronize();
    
    float gpu_time_3;
    cudaEventElapsedTime(&gpu_time_3, start_3, stop_3);
    std::cout << "matmulGPU_tiled4, time: " << gpu_time_3 << " ms" << std::endl;

    cudaMemcpy(h_C_tiled4, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start_3);
    cudaEventDestroy(stop_3);

    if (verfyResult(h_C, h_C_tiled4, M, N)) {
        std::cout << "result is correct" << std::endl;
    } else {
        std::cout << "result is uncorrect" << std::endl;
    }

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_native);
    free(h_C_tiled);
    free(h_C_tiled4);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}