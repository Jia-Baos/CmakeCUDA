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

    int local_max = INT_MIN;    // 单个线程独立遍历，计算线程私有局部最大值

    // avoid branch divergence
    for (size_t i = idx; i < n; i += stride) {
        if (data[i] > local_max) {
            local_max = data[i];
        }
    }

    // 每个线程都直接调用 atomicMax(result, local_max)，对全局内存的 result 执行原子操作
    atomicMax(result, local_max);
}

// GPU version 2
// 先 thread block 内计算 block 内的最大值，之后通过 atomicMax 计算 block 间的最大值
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

    shared_data[tid] = local_max;   // thread block 中的线程可以访问共享内存

    __syncthreads();    // 确保当前线程块内所有线程都已完成写入操作

    // 归约
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { // 每次循环中，只有 tid < stride 的线程参与操作
            if (shared_data[tid + stride] > shared_data[tid]) {
                shared_data[tid] = shared_data[tid + stride];
            }
        }
        __syncthreads();
    }   // 线程结束，shared_data[0] 中存储的是当前线程块内所有线程的 local_max 中的最大值

    // 最后原子操作更新全局最大值
    if (tid == 0) {
        atomicMax(result, shared_data[0]);
    }
}

int main()
{
    const int N = 10000000;
    std::cout << "data size: " << N * sizeof(int) / (1024.0 * 1024.0) << " MB" << std::endl;

    int *data;
    int *gpu_result_native;
    int *gpu_result_shared;

    // cudaMallocManaged分配的内存（堆内存），必须使用 cudaFree 释放
    CUDA_CHECK(cudaMallocManaged(&data, N * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&gpu_result_native, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&gpu_result_shared, sizeof(int)));

    srand(time(NULL));
    for (size_t i = 0; i < N; i++) {
        data[i] = rand() % 100000; // 0-99999
    }

    // 设置一个最大值
    int know_max_pos = N / 2;
    data[know_max_pos] = 999999;

    clock_t start_cpu = clock();
    int cpu_max = findMaxCPU(data, N);
    clock_t end_cpu = clock();
    double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC * 1000;
    std::cout << "CPU max: " << cpu_max << ", time: " << cpu_time << " ms" << std::endl;

    *gpu_result_native = INT_MIN;
    int threads_per_block = 256;
    int blocks_per_grid = 1024;
    // int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start_1;
    cudaEvent_t stop_1;

    cudaEventCreate(&start_1);
    cudaEventCreate(&stop_1);

    cudaEventRecord(start_1);

    findMaxGPU_native<<<blocks_per_grid, threads_per_block>>>(data, N, gpu_result_native);

    cudaEventRecord(stop_1);
    cudaEventSynchronize(stop_1);

    cudaDeviceSynchronize();

    float gpu_time_1;
    cudaEventElapsedTime(&gpu_time_1, start_1, stop_1);
    std::cout << "findMaxGPU_native, GPU max: " << *gpu_result_native << ", time: " << gpu_time_1 << " ms" << std::endl;

    cudaEventDestroy(start_1);
    cudaEventDestroy(stop_1);

    *gpu_result_shared = INT_MIN;
    int shared_memory_size = threads_per_block * sizeof(int);
    cudaEvent_t start_2;
    cudaEvent_t stop_2;

    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);

    cudaEventRecord(start_2);

    findMaxGPU_shared<<<blocks_per_grid, threads_per_block, shared_memory_size>>>(data, N, gpu_result_shared);

    cudaEventRecord(stop_2);
    cudaEventSynchronize(stop_2);

    cudaDeviceSynchronize();

    float gpu_time_2;
    cudaEventElapsedTime(&gpu_time_2, start_2, stop_2);
    std::cout << "findMaxGPU_shared, GPU max: " << *gpu_result_shared << ", time: " << gpu_time_2 << " ms" << std::endl;

    cudaEventDestroy(start_2);
    cudaEventDestroy(stop_2);

    cudaFree(data);
    cudaFree(gpu_result_native);
    cudaFree(gpu_result_shared);

    return 0;
}