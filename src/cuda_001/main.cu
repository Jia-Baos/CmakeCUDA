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
#include <iostream>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

void printDeviceInfo()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cout << "Number of CUDA Devices: " << nDevices << std::endl;

    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device Number: " << i << std::endl;
        std::cout << "  Device name: " << prop.name << std::endl;
        std::cout << "  Device compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Device global memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "  Number of SM: " << prop.multiProcessorCount << std::endl;
    
        std::cout << "  Maximum blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "  Maximum threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << "  Max Threads Dim: [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max Grid Dim: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Shared Memory per Block Optin: " << prop.sharedMemPerBlockOptin / 1024 << " KB" << std::endl;
        std::cout << "  Shared Memory per SM: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
        
        std::cout << "  Constant Memory: " << prop.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  Persisting L2 Max Cache Size: " << prop.persistingL2CacheMaxSize / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Access Policy Max Window Size: " << prop.accessPolicyMaxWindowSize / (1024 * 1024) << " MB" << std::endl;

        std::cout << "  Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "  Async Engine Count: " << (prop.asyncEngineCount ? "Yes" : "No") << std::endl;
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate << "KHz" << std::endl;
        std::cout << "  Peak Memory Bandwidth: " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << "GB/s" << std::endl;
    }
}

__global__ void cuda_hello(void)
{
    printf("Hello World from GPU!\n");
}

__global__ void vectorAdd(int *a, int *b, int *c, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        c[index] = a[index] + b[index];
        // printf("Thread %d in block %d, the id is %d\n", threadIdx.x, blockIdx.x, index);
    }
}

int main()
{
    printDeviceInfo();

    // cuda_hello<<<5, 2>>>();
    // cudaError_t err = cudaDeviceReset();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
    // }

    // const int N = 608 * 256;
    const int N = 1024 * 256;
    int h_a[N], h_b[N], h_c[N];
    nvtxRangePush("Init Array");
    for (int i = 0; i < N; i++)
    {
        h_a[i] = i;
        h_b[i] = i * i;
    }
    nvtxRangePop();

    int *d_a;
    int *d_b;
    int *d_c;
    nvtxRangePush("Cuda Malloc");
    cudaMalloc(&d_a, sizeof(int) * N);
    cudaMalloc(&d_b, sizeof(int) * N);
    cudaMalloc(&d_c, sizeof(int) * N);
    nvtxRangePop();

    nvtxRangePush("Memory Copy H2D");
    cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * N, cudaMemcpyHostToDevice);
    nvtxRangePop();

    // 限制1：每个线程块可以使用的线程数限制
    // 限制2：每个线程块可以使用的共享内存限制
    // 限制3：每个线程块可以使用的寄存器数量限制
    // 对于当前设备 38 个SM，每SM最多 16 个 thread block，每块最多 1024 个线程
    nvtxRangePush("Kernel Launch vectorAdd");
    // vectorAdd<<<608, 256>>>(d_a, d_b, d_c, N);
    vectorAdd<<<256, 1024>>>(d_a, d_b, d_c, N);
    vectorAdd<<<512, 512>>>(d_a, d_b, d_c, N);
    vectorAdd<<<1024, 256>>>(d_a, d_b, d_c, N);
    vectorAdd<<<2048, 128>>>(d_a, d_b, d_c, N);
    vectorAdd<<<262144, 1>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePush("Memory Copy D2H");
    cudaMemcpy(h_c, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePush("Print Results");
    for (int i = 0; i < 20; i++)
    {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }
    nvtxRangePop();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    nvtxMarkA("End of Main");

    return 0;
}