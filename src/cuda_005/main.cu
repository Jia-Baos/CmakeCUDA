/**
 * @file main.cu
 * @author Jia-Baos (18383827268@163.com)
 * @brief test dms, which can't work due to the limitations of the kind 0f GPU
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
#include <nvtx3/nvToolsExt.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

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

// cpu version
int findMaxCPU(int *data, int n)
{
    int max_val = INT_MIN;
    for (int i = 0; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    return max_val;
}

// GPU version 1
__global__ void findMaxGPU_native(int *data, int n, int *result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // 总线程数

    int local_max = INT_MIN;

    // avoid branch divergence，单个线程读多次数据
    for (size_t i = idx; i < n; i += stride) {
        if (data[i] > local_max) {
            local_max = data[i];
        }
    }

    atomicMax(result, local_max);
}

// GPU version 2
__global__ void findMaxGPU_shared(int *data, int n, int *result)
{
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = blockDim.x * gridDim.x; // 总线程数

    int local_max = INT_MIN;

    // avoid branch divergence
    for (size_t i = gid; i < n; i += nThreads) {
        if (data[i] > local_max) {
            local_max = data[i];
        }
    }

    shared_data[tid] = local_max;

    __syncthreads();

    // 归约
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_data[tid + stride] > shared_data[tid]) {
                shared_data[tid] = shared_data[tid + stride];
            }
        }
        __syncthreads();
    }

    // 最后原子操作更新全局最大值
    if (tid == 0) {
        atomicMax(result, shared_data[0]);
    }
}

// GPU version 3
__global__ void findMaxGPU_warp(int *data, int n, int *result)
{
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = blockDim.x * gridDim.x; // 总线程数

    // 计算 thread block 内线程所处的 warp lane 和在所处 warp lane 中的 idx
    int warp_id = tid / 32;
    int lane = tid % 32;

    // avoid branch divergence, each thread find its max
    int local_max = INT_MIN;
    for (size_t i = gid; i < n; i += nThreads) {
        if (data[i] > local_max) {
            local_max = data[i];
        }
    }

    // warp shuffle reduce，聚合线程束内的最大值
    for (size_t offset = 16; offset > 0; offset >>= 1) {
        // 在同一个线程束内，将当前线程（lane）后面第offset个线程的local_max值，直接拷贝到当前线程的neighbor变量中
        // 无需共享内存 / 全局内存中转，直接在线程束内线程间传递数据，延迟远低于共享内存
        int neighbor = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = max(local_max, neighbor);
    }

    // collect all warp results to shared memory
    __shared__ int warp_maxs[8];        // blockDims.x / warp size -> 256 / 32
    if (lane == 0) {                    //
        warp_maxs[warp_id] = local_max; // 线程束内的最大值位于头部，即 lane = 0 处
    }

    __syncthreads();

    // the last warp does the reduce for all warp_maxs
    int block_max = INT_MIN;
    if (warp_id == 0) {
        if (lane < 8) {
            // 在第 0 个线程束内，挑选 8 个线程来分摊加载共享内存中 8 个线程束的最大值，为后续的线程束洗牌归约做准备
            block_max = warp_maxs[lane];
        }

        for (size_t offset = 16; offset > 0; offset >>= 1) {
            int neighbor = __shfl_down_sync(0xffffffff, block_max, offset);
            block_max = max(block_max, neighbor);
        }
    }

    // thread 0 update the global max
    if (tid == 0) {
        atomicMax(result, block_max);
    }
}

// 3060 系列显卡不支持 cluster_group 和 dsm 特性
#define CLUSTER_SIZE 8
__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) findMaxGPU_dsm(int *data, int n, int *result)
{
    // cp handle
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);

    // shared memory
    extern __shared__ int smem[];
    int *warp_results = smem;
    int *cluster_results = &smem[block.size()/32];

    int tid = block.thread_rank();
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int nThreads = blockDim.x * gridDim.x; // 总线程数

    int warp_id = warp.meta_group_rank();   // warp 在当前 block 中的 id，0 ～ 7
    int warp_num = warp.meta_group_size();  // 线程束的数量

    // avoid branch divergence, each thread find its max
    int local_max = INT_MIN;
    for (size_t i = gid; i < n; i += nThreads) {
        if (data[i] > local_max) {
            local_max = data[i];
        }
    }

    // each warp reduce
    int warp_max = cooperative_groups::reduce(warp, local_max, cooperative_groups::greater<int>());

    // warp lane 0/leader write into shared memory
    if(warp.thread_rank() == 0){
        warp_results[warp_id] = warp_max;
    }

    block.sync();

    // 8 warps，第一个 warp 规约所有 warp 的结果
    int block_max = INT_MIN;
    if(warp_id == 0){
        if(warp.thread_rank() < warp_num){
            block_max = warp_results[warp.thread_rank()];
        }

        block_max = cooperative_groups::reduce(warp, block_max, cooperative_groups::greater<int>());

        if (warp.thread_rank() == 0){
            cluster_results[0] = block_max;
        }
    }

    cluster.sync();

    // dsm cluster 级别 reduce
    if (tid == 0){
        int cluster_rank = cluster.block_rank();
        int cluster_size = cluster.num_blocks();

        int cluster_max = cluster_results[0];

        for (size_t i = 0; i < cluster_size; i++){
            if (i != cluster_rank){
                int * remote_sm = cluster.map_shared_rank(cluster_results, i);
                cluster_max = max(cluster_max, remote_sm[0]);
            }
        }

        // cluster 0 update the block global result
        if (cluster_rank == 0){
            atomicMax(result, cluster_max);
        }

    }
}

int main()
{
    const int N = 10000000;
    std::cout << "data size: " << N * sizeof(int) / (1024.0 * 1024.0) << " MB" << std::endl;

    int *d_data;
    int *d_gpu_result_native;
    int *d_gpu_result_shared;
    int *d_gpu_result_warp;
    int *d_gpu_result_dsm;

    // cudaMallocManaged分配的内存（堆内存），必须使用 cudaFree 释放
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gpu_result_native, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gpu_result_shared, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gpu_result_warp, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gpu_result_dsm, sizeof(int)));

    srand(time(NULL));
    int *h_data = (int *)malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++) {
        h_data[i] = rand() % 100000; // 0-99999
    }

    // 设置一个最大值
    int know_max_pos = N / 2;
    h_data[know_max_pos] = 999999;

    clock_t start_cpu = clock();
    int cpu_max = findMaxCPU(h_data, N);
    clock_t end_cpu = clock();
    double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC * 1000;
    std::cout << "CPU max: " << cpu_max << ", time: " << cpu_time << " ms" << std::endl;

    cudaMemcpy(d_data, h_data, sizeof(int) * N, cudaMemcpyHostToDevice);

    int h_gpu_result_native = INT_MIN;
    int threads_per_block = 256;
    int blocks_per_grid = 1024;
    // int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    cudaMemcpy(d_gpu_result_native, &h_gpu_result_native, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start_1;
    cudaEvent_t stop_1;

    cudaEventCreate(&start_1);
    cudaEventCreate(&stop_1);

    cudaEventRecord(start_1);

    findMaxGPU_native<<<blocks_per_grid, threads_per_block>>>(d_data, N, d_gpu_result_native);

    cudaEventRecord(stop_1);
    cudaEventSynchronize(stop_1);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_gpu_result_native, d_gpu_result_native, sizeof(int), cudaMemcpyDeviceToHost);

    float gpu_time_1;
    cudaEventElapsedTime(&gpu_time_1, start_1, stop_1);
    std::cout << "findMaxGPU_native, GPU max: " << h_gpu_result_native << ", time: " << gpu_time_1 << " ms" << std::endl;

    cudaEventDestroy(start_1);
    cudaEventDestroy(stop_1);

    int h_gpu_result_shared = INT_MIN;
    int shared_memory_size = threads_per_block * sizeof(int);

    cudaMemcpy(d_gpu_result_shared, &h_gpu_result_shared, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start_2;
    cudaEvent_t stop_2;

    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);

    cudaEventRecord(start_2);

    findMaxGPU_shared<<<blocks_per_grid, threads_per_block, shared_memory_size>>>(d_data, N, d_gpu_result_shared);

    cudaEventRecord(stop_2);
    cudaEventSynchronize(stop_2);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_gpu_result_shared, d_gpu_result_shared, sizeof(int), cudaMemcpyDeviceToHost);

    float gpu_time_2;
    cudaEventElapsedTime(&gpu_time_2, start_2, stop_2);
    std::cout << "findMaxGPU_shared, GPU max: " << h_gpu_result_shared << ", time: " << gpu_time_2 << " ms" << std::endl;

    cudaEventDestroy(start_2);
    cudaEventDestroy(stop_2);

    int h_gpu_result_warp = INT_MIN;

    cudaMemcpy(d_gpu_result_warp, &h_gpu_result_warp, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start_3;
    cudaEvent_t stop_3;

    cudaEventCreate(&start_3);
    cudaEventCreate(&stop_3);

    cudaEventRecord(start_3);

    findMaxGPU_warp<<<blocks_per_grid, threads_per_block, shared_memory_size>>>(d_data, N, d_gpu_result_warp);

    cudaEventRecord(stop_3);
    cudaEventSynchronize(stop_3);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_gpu_result_warp, d_gpu_result_warp, sizeof(int), cudaMemcpyDeviceToHost);

    float gpu_time_3;
    cudaEventElapsedTime(&gpu_time_3, start_3, stop_3);
    std::cout << "findMaxGPU_warp, GPU max: " << h_gpu_result_warp << ", time: " << gpu_time_3 << " ms" << std::endl;

    cudaEventDestroy(start_3);
    cudaEventDestroy(stop_3);

    int h_gpu_result_dsm = INT_MIN;
    
    int cluster_per_grid = blocks_per_grid / CLUSTER_SIZE;
    int actual_blocks = cluster_per_grid / CLUSTER_SIZE;
    int cluster_shared_memory_size = (threads_per_block / 32  + 1) * sizeof(int);

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(actual_blocks, 1, 1);
    config.blockDim = dim3(threads_per_block, 1, 1);
    config.dynamicSmemBytes = cluster_shared_memory_size;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = CLUSTER_SIZE;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.attrs = attribute;
    config.numAttrs = 1;

    cudaMemcpy(d_gpu_result_dsm, &h_gpu_result_dsm, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start_4;
    cudaEvent_t stop_4;

    cudaEventCreate(&start_4);
    cudaEventCreate(&stop_4);

    cudaEventRecord(start_4);

    cudaLaunchKernelEx(&config, findMaxGPU_dsm, d_data, int(N), d_gpu_result_dsm);
    // findMaxGPU_dsm<<<blocks_per_grid, threads_per_block, shared_memory_size>>>(d_data, N, d_gpu_result_dsm);

    cudaEventRecord(stop_4);
    cudaEventSynchronize(stop_4);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_gpu_result_dsm, d_gpu_result_dsm, sizeof(int), cudaMemcpyDeviceToHost);

    float gpu_time_4;
    cudaEventElapsedTime(&gpu_time_4, start_4, stop_4);
    std::cout << "findMaxGPU_dsm, GPU max: " << h_gpu_result_warp << ", time: " << gpu_time_4 << " ms" << std::endl;

    cudaEventDestroy(start_4);
    cudaEventDestroy(stop_4);

    free(h_data);
    cudaFree(d_data);
    cudaFree(d_gpu_result_native);
    cudaFree(d_gpu_result_shared);
    cudaFree(d_gpu_result_warp);
    cudaFree(d_gpu_result_dsm);

    return 0;
}