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

__global__ void matmulGPU_tiled_4x4(float *A, float *B, float *C, int M, int N, int K)
{
    // 通过 Cooperative Groups（协作组）编程模型，获取当前线程所属的块级协作组
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
    srand(time(NULL));

    const int batch_size = 8;
    float *h_A_batch[batch_size];
    float *h_C_batch[batch_size];
    float *d_A_batch[batch_size];
    float *d_C_batch[batch_size];

    float *h_B_shared;
    float *d_B_shared;
    
    h_B_shared = (float *)malloc(K * N * sizeof(float));
    initMatRandom(h_B_shared, K, N);
    CUDA_CHECK(cudaMalloc(&d_B_shared, K * N * sizeof(float)));

    for (size_t i = 0; i < batch_size; ++i){
        h_A_batch[i] = (float*)malloc(M * K * sizeof(float));
        h_C_batch[i] = (float*)malloc(M * N * sizeof(float));

        CUDA_CHECK(cudaMalloc(&d_A_batch[i], M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_batch[i], M * N * sizeof(float)));

        initMatRandom(h_A_batch[i], M, K);;
    }

    // version1: sync
    dim3 block_dim_cg(16, 16);
    dim3 grid_dim_cg((N + 64 - 1) / 64, (M + 64 - 1) / 64);

    cudaEvent_t start_1;
    cudaEvent_t stop_1;

    cudaEventCreate(&start_1);
    cudaEventCreate(&stop_1);

    cudaMemcpy(d_B_shared, h_B_shared, K * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_1);

    for (size_t i = 0; i < batch_size; ++i){
        cudaMemcpy(d_A_batch[i], h_A_batch[i], M * K * sizeof(float), cudaMemcpyHostToDevice);
        matmulGPU_tiled_4x4<<<grid_dim_cg, block_dim_cg>>>(d_A_batch[i], d_B_shared, d_C_batch[i], M, N, K);
        cudaMemcpy(h_C_batch[i], d_C_batch[i], M * N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop_1);
    cudaEventSynchronize(stop_1);

    // cudaDeviceSynchronize();
    
    float gpu_time_1;
    cudaEventElapsedTime(&gpu_time_1, start_1, stop_1);
    std::cout << "matmulGPU_tiled_4x4, sync time: " << gpu_time_1 << " ms" << std::endl;

    cudaEventDestroy(start_1);
    cudaEventDestroy(stop_1);


    // version2: async
    cudaStream_t streams[batch_size];
    for (size_t i = 0; i < batch_size; ++i){
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start_2;
    cudaEvent_t stop_2;

    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);

    cudaMemcpy(d_B_shared, h_B_shared, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // 预热GPU（可选，消除首次启动开销）
    matmulGPU_tiled_4x4<<<grid_dim_cg, block_dim_cg>>>(d_A_batch[0], d_B_shared, d_C_batch[0], M, N, K);
    matmulGPU_tiled_4x4<<<grid_dim_cg, block_dim_cg>>>(d_A_batch[0], d_B_shared, d_C_batch[0], M, N, K);
    matmulGPU_tiled_4x4<<<grid_dim_cg, block_dim_cg>>>(d_A_batch[0], d_B_shared, d_C_batch[0], M, N, K);

    cudaEventRecord(start_2);

    for (size_t i = 0; i < batch_size; ++i){
        cudaMemcpyAsync(d_A_batch[i], h_A_batch[i], M * K * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        matmulGPU_tiled_4x4<<<grid_dim_cg, block_dim_cg, 0, streams[i]>>>(d_A_batch[i], d_B_shared, d_C_batch[i], M, N, K);
        cudaMemcpyAsync(h_C_batch[i], d_C_batch[i], M * N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    for (size_t i = 0; i < batch_size; ++i){
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    cudaEventRecord(stop_2);
    cudaEventSynchronize(stop_2);

    // cudaDeviceSynchronize();
    
    float gpu_time_2;
    cudaEventElapsedTime(&gpu_time_2, start_2, stop_2);
    std::cout << "matmulGPU_tiled_4x4, async time: " << gpu_time_2 << " ms" << std::endl;

    cudaEventDestroy(start_2);
    cudaEventDestroy(stop_2);

    for (size_t i = 0; i < batch_size; ++i){
        cudaStreamDestroy(streams[i]);
    }

    // version3: cuda graph
    cudaStream_t graph_streams_pool[batch_size];
    for (size_t i = 0; i < batch_size; ++i){
        cudaStreamCreate(&graph_streams_pool[i]);
    }

    cudaGraph_t graph_par;
    cudaGraphExec_t graphExec_par;

    // setp1: capture stream
    cudaStreamBeginCapture(graph_streams_pool[0], cudaStreamCaptureModeGlobal);

    // step2: fork-join
    cudaEvent_t fork_event;
    cudaEventCreate(&fork_event);
    cudaEventRecord(fork_event, graph_streams_pool[0]);

    for (size_t i = 0; i < batch_size; ++i){
        cudaStreamWaitEvent(graph_streams_pool[i], fork_event, 0);
    }

    // step3: 使用 stream 并行处理所有的 branch
    for (size_t i = 0; i < batch_size; ++i) {
        int stream_idx = i % batch_size;    // 轮流使用 stream
        cudaMemcpyAsync(d_A_batch[i], h_A_batch[i], M * K * sizeof(float), cudaMemcpyHostToDevice, graph_streams_pool[stream_idx]);
        matmulGPU_tiled_4x4<<<grid_dim_cg, block_dim_cg, 0, graph_streams_pool[stream_idx]>>>(d_A_batch[i], d_B_shared, d_C_batch[i], M, N, K);
        cudaMemcpyAsync(h_C_batch[i], d_C_batch[i], M * N * sizeof(float), cudaMemcpyDeviceToHost, graph_streams_pool[stream_idx]);
    }

    // step4: 创建 join event，会合所有 streams
    cudaEvent_t join_event;
    cudaEventCreate(&join_event);

    for (size_t i = 0; i < batch_size; ++i){
        cudaEventRecord(join_event, graph_streams_pool[i]);
        cudaStreamWaitEvent(graph_streams_pool[0], join_event, 0);
    }

    // step5: 结束捕捉，实例化 graph
    cudaStreamEndCapture(graph_streams_pool[0], &graph_par);
    cudaGraphInstantiate(&graphExec_par, graph_par, NULL, NULL, 0);

    // step6: launch，graph 内部是并行的
    cudaEvent_t start_3;
    cudaEvent_t stop_3;

    cudaEventCreate(&start_3);
    cudaEventCreate(&stop_3);

    cudaMemcpy(d_B_shared, h_B_shared, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 预热Graph
    cudaGraphLaunch(graphExec_par, graph_streams_pool[0]);
    cudaStreamSynchronize(graph_streams_pool[0]);

    cudaEventRecord(start_3);
    cudaGraphLaunch(graphExec_par, graph_streams_pool[0]);
    cudaStreamSynchronize(graph_streams_pool[0]);
    cudaEventRecord(stop_3);
    cudaEventSynchronize(stop_3);

    // cudaDeviceSynchronize();
    
    float gpu_time_3;
    cudaEventElapsedTime(&gpu_time_3, start_3, stop_3);
    std::cout << "matmulGPU_tiled_4x4, graph time: " << gpu_time_3 << " ms" << std::endl;

    cudaEventDestroy(start_3);
    cudaEventDestroy(stop_3);
    cudaEventDestroy(join_event);
    cudaEventDestroy(fork_event);

    for (size_t i = 0; i < batch_size; ++i){
        cudaStreamDestroy(graph_streams_pool[i]);
    }

    for (size_t i = 0; i < batch_size; ++i) {
        free(h_A_batch[i]);
        free(h_C_batch[i]);
        CUDA_CHECK(cudaFree(d_A_batch[i]));
        CUDA_CHECK(cudaFree(d_C_batch[i]));
    }

    free(h_B_shared);
    CUDA_CHECK(cudaFree(d_B_shared));

    return 0;
}