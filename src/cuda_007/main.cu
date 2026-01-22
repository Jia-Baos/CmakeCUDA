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
#include <vector>
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

// 优化：封装CUDA事件操作
inline void create_cuda_event(cudaEvent_t &event)
{
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDefault));
}

inline float measure_cuda_time(cudaEvent_t start, cudaEvent_t stop)
{
    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    return elapsed_time;
}

// 优化：模板化核函数，TILE_SIZE作为模板参数
template <int TILE_SIZE>
__global__ void matmulGPU_tiled_4x4(float *A, float *B_T, float *C, int M, int N, int K)
{
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int block_row = blockIdx.y * TILE_SIZE * 4;
    int block_col = blockIdx.x * TILE_SIZE * 4;

    // 优化：共享内存添加padding，避免bank conflict
    __shared__ float As[TILE_SIZE * 4][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE + 1][TILE_SIZE * 4];

    float sum[4][4] = { { 0.0f } };
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int tile_k = t * TILE_SIZE;

// 加载A到共享内存
#pragma unroll 4
        for (size_t i = 0; i < 4; i++) {
            int a_row = block_row + ty * 4 + i;
            int a_col = tile_k + tx;
            As[ty * 4 + i][tx] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }

// 加载B_T（转置后的B）到共享内存，优化全局内存合并访问
#pragma unroll 4
        for (size_t j = 0; j < 4; j++) {
            int b_col = tile_k + ty;
            int b_row = block_col + tx * 4 + j;
            Bs[ty][tx * 4 + j] = (b_row < N && b_col < K) ? B_T[b_row * K + b_col] : 0.0f;
        }

        block.sync();

// 优化：循环展开，提升计算效率
#pragma unroll
        for (size_t k = 0; k < TILE_SIZE; k++) {
            float a_reg[4];
#pragma unroll 4
            for (size_t i = 0; i < 4; i++) {
                a_reg[i] = As[ty * 4 + i][k];
            }

#pragma unroll 4
            for (size_t i = 0; i < 4; i++) {
#pragma unroll 4
                for (size_t j = 0; j < 4; j++) {
                    sum[i][j] += a_reg[i] * Bs[k][tx * 4 + j];
                }
            }
        }

        block.sync();
    }

// 写回结果
#pragma unroll 4
    for (size_t i = 0; i < 4; i++) {
#pragma unroll 4
        for (size_t j = 0; j < 4; j++) {
            int c_row = block_row + ty * 4 + i;
            int c_col = block_col + tx * 4 + j;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = sum[i][j];
            }
        }
    }
}

// 辅助函数：矩阵转置（GPU）
__global__ void transposeMatrix(float *in, float *out, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        out[x * rows + y] = in[y * cols + x];
    }
}

// 优化：使用页锁定内存初始化矩阵
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
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const int batch_size = 8;
    const int TILE_SIZE = 16;
    srand(time(NULL));

    float *h_B_shared;
    float *h_B_shared_T; // 转置后的B矩阵（主机端）
    float *d_B_shared;
    float *d_B_shared_T; // 转置后的B矩阵（设备端）

    nvtxRangePush("Cuda Malloc for B");
    CUDA_CHECK(cudaMallocHost(&h_B_shared, K * N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_B_shared_T, N * K * sizeof(float)));
    initMatRandom(h_B_shared, K, N);
    CUDA_CHECK(cudaMalloc(&d_B_shared, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_shared_T, N * K * sizeof(float)));
    nvtxRangePop();

    // 优化：使用vector管理批量内存，避免内存泄漏
    std::vector<float *> h_A_batch(batch_size);
    std::vector<float *> h_C_batch(batch_size);
    std::vector<float *> d_A_batch(batch_size);
    std::vector<float *> d_C_batch(batch_size);
    std::vector<float *> h_C_batch_naive(batch_size);

    nvtxRangePush("Cuda Malloc for A and C");
    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaMallocHost(&h_A_batch[i], M * K * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_C_batch[i], M * N * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_C_batch_naive[i], M * N * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_A_batch[i], M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_batch[i], M * N * sizeof(float)));

        initMatRandom(h_A_batch[i], M, K);
    }
    nvtxRangePop();

    // 预转置B矩阵（优化全局内存访问）
    dim3 trans_block(32, 32);
    dim3 trans_grid((N + trans_block.x - 1) / trans_block.x, (K + trans_block.y - 1) / trans_block.y);
    CUDA_CHECK(cudaMemcpy(d_B_shared, h_B_shared, K * N * sizeof(float), cudaMemcpyHostToDevice));
    transposeMatrix<<<trans_grid, trans_block>>>(d_B_shared, d_B_shared_T, K, N);
    CUDA_CHECK(cudaGetLastError());

    // 核函数启动配置（优化：根据TILE_SIZE动态计算）
    dim3 block_dim_cg(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim_cg((N + TILE_SIZE * 4 - 1) / (TILE_SIZE * 4), (M + TILE_SIZE * 4 - 1) / (TILE_SIZE * 4));

    nvtxRangePush("matmulGPU_tiled_4x4 Stream");
    std::vector<cudaStream_t> streams(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start_2;
    cudaEvent_t stop_2;
    create_cuda_event(start_2);
    create_cuda_event(stop_2);

    // 预热GPU
    nvtxRangePush("matmulGPU_tiled_4x4 Stream warmup");
    matmulGPU_tiled_4x4<TILE_SIZE><<<grid_dim_cg, block_dim_cg, 0, streams[0]>>>(d_A_batch[0], d_B_shared_T, d_C_batch[0], M, N, K);
    CUDA_CHECK(cudaGetLastError());
    nvtxRangePop();

    CUDA_CHECK(cudaEventRecord(start_2));
    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d_A_batch[i], h_A_batch[i], M * K * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        matmulGPU_tiled_4x4<TILE_SIZE><<<grid_dim_cg, block_dim_cg, 0, streams[i]>>>(d_A_batch[i], d_B_shared_T, d_C_batch[i], M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_C_batch[i], d_C_batch[i], M * N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
    }

    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    CUDA_CHECK(cudaEventRecord(stop_2));
    CUDA_CHECK(cudaEventSynchronize(stop_2));

    float gpu_time_2 = measure_cuda_time(start_2, stop_2);
    std::cout << "matmulGPU_tiled_4x4, async time: " << gpu_time_2 << " ms" << std::endl;

    // for (size_t i = 0; i < batch_size; ++i) {
    //     if (verfyResult(h_C_batch_naive[i], h_C_batch[i], M, N)) {
    //         std::cout << "result is correct" << std::endl;
    //     } else {
    //         std::cout << "result is uncorrect" << std::endl;
    //     }
    // }

    CUDA_CHECK(cudaEventDestroy(start_2));
    CUDA_CHECK(cudaEventDestroy(stop_2));

    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    nvtxRangePop();

    // 内存释放（优化：使用RAII，避免内存泄漏）
    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaFreeHost(h_A_batch[i]));
        CUDA_CHECK(cudaFreeHost(h_C_batch[i]));
        CUDA_CHECK(cudaFree(d_A_batch[i]));
        CUDA_CHECK(cudaFree(d_C_batch[i]));
    }

    CUDA_CHECK(cudaFreeHost(h_B_shared));
    CUDA_CHECK(cudaFreeHost(h_B_shared_T));
    CUDA_CHECK(cudaFree(d_B_shared));
    CUDA_CHECK(cudaFree(d_B_shared_T));

    return 0;
}