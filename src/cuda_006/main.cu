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
    int M_val = M;
    int N_val = N;
    int K_val = K;
    const int batch_size = 16;
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

    nvtxRangePush("matmulCPU");
    clock_t start_cpu = clock();
    for (size_t i = 0; i < batch_size; ++i) {
        matmulCPU(h_A_batch[i], h_B_shared, h_C_batch_naive[i], M, N, K);
    }
    clock_t end_cpu = clock();
    double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC * 1000;
    std::cout << "matmulCPU time: " << cpu_time << " ms" << std::endl;
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

    // version1: sync
    nvtxRangePush("matmulGPU_tiled_4x4");
    cudaEvent_t start_1;
    cudaEvent_t stop_1;

    create_cuda_event(start_1);
    create_cuda_event(stop_1);

    CUDA_CHECK(cudaEventRecord(start_1));
    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaMemcpy(d_A_batch[i], h_A_batch[i], M * K * sizeof(float), cudaMemcpyHostToDevice));
        matmulGPU_tiled_4x4<TILE_SIZE><<<grid_dim_cg, block_dim_cg>>>(d_A_batch[i], d_B_shared_T, d_C_batch[i], M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C_batch[i], d_C_batch[i], M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventRecord(stop_1));
    CUDA_CHECK(cudaEventSynchronize(stop_1));

    float gpu_time_1 = measure_cuda_time(start_1, stop_1);
    std::cout << "matmulGPU_tiled_4x4, sync time: " << gpu_time_1 << " ms" << std::endl;

    for (size_t i = 0; i < batch_size; ++i) {
        if (verfyResult(h_C_batch_naive[i], h_C_batch[i], M, N)) {
            std::cout << "result is correct" << std::endl;
        } else {
            std::cout << "result is uncorrect" << std::endl;
        }
    }

    CUDA_CHECK(cudaEventDestroy(start_1));
    CUDA_CHECK(cudaEventDestroy(stop_1));
    nvtxRangePop();

    // version2: async
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
    matmulGPU_tiled_4x4<TILE_SIZE><<<grid_dim_cg, block_dim_cg, 0, streams[0]>>>(d_A_batch[0], d_B_shared_T, d_C_batch[0], M, N, K);
    CUDA_CHECK(cudaGetLastError());

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

    for (size_t i = 0; i < batch_size; ++i) {
        if (verfyResult(h_C_batch_naive[i], h_C_batch[i], M, N)) {
            std::cout << "result is correct" << std::endl;
        } else {
            std::cout << "result is uncorrect" << std::endl;
        }
    }

    CUDA_CHECK(cudaEventDestroy(start_2));
    CUDA_CHECK(cudaEventDestroy(stop_2));

    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    nvtxRangePop();

    // version3: cuda graph
    nvtxRangePush("matmulGPU_tiled_4x4 Graph");
    const int stream_size_3 = 8;
    std::vector<cudaStream_t> graph_streams_pool(stream_size_3);
    for (size_t i = 0; i < stream_size_3; ++i) {
        CUDA_CHECK(cudaStreamCreate(&graph_streams_pool[i]));
    }

    cudaGraph_t graph_par;
    cudaGraphExec_t graphExec_par;

    // 优化：简化Graph捕捉流程，移除冗余的fork/join事件
    CUDA_CHECK(cudaStreamBeginCapture(graph_streams_pool[0], cudaStreamCaptureModeGlobal));
    for (size_t i = 0; i < batch_size; ++i) {
        int stream_idx = i % stream_size_3;
        CUDA_CHECK(cudaMemcpyAsync(d_A_batch[i], h_A_batch[i], M * K * sizeof(float), cudaMemcpyHostToDevice, graph_streams_pool[stream_idx]));
        matmulGPU_tiled_4x4<TILE_SIZE><<<grid_dim_cg, block_dim_cg, 0, graph_streams_pool[stream_idx]>>>(d_A_batch[i], d_B_shared_T, d_C_batch[i], M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_C_batch[i], d_C_batch[i], M * N * sizeof(float), cudaMemcpyDeviceToHost, graph_streams_pool[stream_idx]));
    }
    CUDA_CHECK(cudaStreamEndCapture(graph_streams_pool[0], &graph_par));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec_par, graph_par, NULL, NULL, 0));

    // 执行Graph并计时
    cudaEvent_t start_3, stop_3;
    create_cuda_event(start_3);
    create_cuda_event(stop_3);

    // 重置Device端数据（避免缓存命中导致时间失真）
    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaMemsetAsync(d_C_batch[i], 0, M * N * sizeof(float), graph_streams_pool[0]));
    }
    CUDA_CHECK(cudaStreamSynchronize(graph_streams_pool[0]));

    // 预热Graph
    CUDA_CHECK(cudaGraphLaunch(graphExec_par, graph_streams_pool[0]));
    CUDA_CHECK(cudaStreamSynchronize(graph_streams_pool[0]));

    CUDA_CHECK(cudaEventRecord(start_3));

    CUDA_CHECK(cudaGraphLaunch(graphExec_par, graph_streams_pool[0]));
    CUDA_CHECK(cudaStreamSynchronize(graph_streams_pool[0]));

    CUDA_CHECK(cudaEventRecord(stop_3));
    CUDA_CHECK(cudaEventSynchronize(stop_3));

    float gpu_time_3 = measure_cuda_time(start_3, stop_3);
    std::cout << "matmulGPU_tiled_4x4, graph time: " << gpu_time_3 << " ms" << std::endl;

    for (size_t i = 0; i < batch_size; ++i) {
        if (verfyResult(h_C_batch_naive[i], h_C_batch[i], M, N)) {
            std::cout << "result is correct" << std::endl;
        } else {
            std::cout << "result is uncorrect" << std::endl;
        }
    }

    // 资源释放
    CUDA_CHECK(cudaEventDestroy(start_3));
    CUDA_CHECK(cudaEventDestroy(stop_3));
    CUDA_CHECK(cudaGraphDestroy(graph_par));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec_par));

    for (size_t i = 0; i < stream_size_3; ++i) {
        CUDA_CHECK(cudaStreamDestroy(graph_streams_pool[i]));
    }
    nvtxRangePop();

    // version4: cuda graph 2
    nvtxRangePush("matmulGPU_tiled_4x4 Graph version 2");
    const int stream_size_4 = 16; // set to 16 will make error: oops we found: 1004192
    std::vector<cudaStream_t> graph_streams_pool_4(stream_size_4);
    for (size_t i = 0; i < stream_size_4; ++i) {
        CUDA_CHECK(cudaStreamCreate(&graph_streams_pool_4[i]));
    }

    // 每张图一个 exec
    std::vector<cudaGraphExec_t> graphExec_par_4(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        cudaGraph_t graph;
        CUDA_CHECK(cudaGraphCreate(&graph, 0));

        cudaGraphNode_t h2d, ker, d2h;
        // 2.1 H2D
        cudaMemcpy3DParms p1 = { 0 };
        p1.srcPtr = make_cudaPitchedPtr(h_A_batch[i], M * K * sizeof(float), M * K, 1);
        p1.dstPtr = make_cudaPitchedPtr(d_A_batch[i], M * K * sizeof(float), M * K, 1);
        p1.extent = make_cudaExtent(M * K * sizeof(float), 1, 1);
        p1.kind = cudaMemcpyHostToDevice;
        CUDA_CHECK(cudaGraphAddMemcpyNode(&h2d, graph, nullptr, 0, &p1));

        // 2.2 Kernel
        void *args[] = { &d_A_batch[i], &d_B_shared_T, &d_C_batch[i], &M_val, &N_val, &K_val };
        cudaKernelNodeParams kpar = { 0 };
        kpar.func = (void *)matmulGPU_tiled_4x4<TILE_SIZE>;
        kpar.gridDim = grid_dim_cg;
        kpar.blockDim = block_dim_cg;
        kpar.kernelParams = args;
        CUDA_CHECK(cudaGraphAddKernelNode(&ker, graph, &h2d, 1, &kpar));

        // 2.3 D2H
        cudaMemcpy3DParms p2 = { 0 };
        p2.srcPtr = make_cudaPitchedPtr(d_C_batch[i], M * N * sizeof(float), M * N, 1);
        p2.dstPtr = make_cudaPitchedPtr(h_C_batch[i], M * N * sizeof(float), M * N, 1);
        p2.extent = make_cudaExtent(M * N * sizeof(float), 1, 1);
        p2.kind = cudaMemcpyDeviceToHost;
        CUDA_CHECK(cudaGraphAddMemcpyNode(&d2h, graph, &ker, 1, &p2));

        CUDA_CHECK(cudaGraphInstantiate(&graphExec_par_4[i], graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(graph)); // 实例化完即可销毁原图
    }

    // 执行Graph并计时
    cudaEvent_t start_4, stop_4;
    create_cuda_event(start_4);
    create_cuda_event(stop_4);

    // 重置Device端数据（避免缓存命中导致时间失真）
    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaMemsetAsync(d_C_batch[i], 0, M * N * sizeof(float), graph_streams_pool_4[0]));
    }
    CUDA_CHECK(cudaStreamSynchronize(graph_streams_pool_4[0]));

    CUDA_CHECK(cudaEventRecord(start_4));

    // 一次性把全部图发射到不同流
    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graphExec_par_4[i], graph_streams_pool_4[i]));
    }

    // 等所有流完成
    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(graph_streams_pool_4[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop_4));
    CUDA_CHECK(cudaEventSynchronize(stop_4));

    float gpu_time_4 = measure_cuda_time(start_4, stop_4);
    std::cout << "matmulGPU_tiled_4x4, graph 2 time: " << gpu_time_4 << " ms" << std::endl;

    for (size_t i = 0; i < batch_size; ++i) {
        if (verfyResult(h_C_batch_naive[i], h_C_batch[i], M, N)) {
            std::cout << "result is correct" << std::endl;
        } else {
            std::cout << "result is uncorrect" << std::endl;
        }
    }

    // 资源释放
    CUDA_CHECK(cudaEventDestroy(start_4));
    CUDA_CHECK(cudaEventDestroy(stop_4));
    for (size_t i = 0; i < batch_size; ++i) {
        if (graphExec_par_4[i]) {
            cudaGraphExecDestroy(graphExec_par_4[i]);
            graphExec_par_4[i] = nullptr;
        }
    }

    for (size_t i = 0; i < stream_size_4; ++i) {
        CUDA_CHECK(cudaStreamDestroy(graph_streams_pool_4[i]));
    }
    nvtxRangePop();

    // 内存释放（优化：使用RAII，避免内存泄漏）
    for (size_t i = 0; i < batch_size; ++i) {
        CUDA_CHECK(cudaFreeHost(h_A_batch[i]));
        CUDA_CHECK(cudaFreeHost(h_C_batch[i]));
        CUDA_CHECK(cudaFreeHost(h_C_batch_naive[i]));
        CUDA_CHECK(cudaFree(d_A_batch[i]));
        CUDA_CHECK(cudaFree(d_C_batch[i]));
    }

    CUDA_CHECK(cudaFreeHost(h_B_shared));
    CUDA_CHECK(cudaFreeHost(h_B_shared_T));
    CUDA_CHECK(cudaFree(d_B_shared));
    CUDA_CHECK(cudaFree(d_B_shared_T));

    return 0;
}