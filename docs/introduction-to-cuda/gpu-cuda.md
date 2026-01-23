# GPU CUDA

## GPU 结构

```
GPU（整个显卡）
    ├── GPC（Graphics Processing Cluster）- 多个
    │   ├── SM（Streaming Multiprocessor）- 每个 GPC 包含多个 SM
    │   │   ├── CUDA Core（计算单元）- 每个 SM 包含多个 CUDA 核心
    │   │   ├── Tensor Core - AI 计算单元
    │   │   ├── RT Core - 光线追踪单元
    │   │   ├── 共享内存（Shared Menory）
    │   │   ├── 寄存器文件（Register File）
    │   │   └── 调度器（Warp Scheduler）
    │   └── L1 缓存、纹理单元等
    └── L2 缓存、显存控制器等全局资源
```

## CUDA 开发环境配置

```bash
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# CUDA 缓存相关
export CUDA_CACHE_PATH=$HOME/.cuda_cache    # NVCC 编译缓存路径
export CUDA_CACHE_NAXSIZE=4294967296    # 4 GB缓存大小，（磁盘，不占用显存）
export CUDA_CACHE_DISABLE=0 # 0 =  启用缓存，1 = 禁用缓存

# GPU 可见性
export CUDA_DEVICE_ORDER=PCI_BUS_ID # 按 PCI 物理顺序选择 GPU
export CUDA_VISIBLE_DEVICES=0    # 仅使用第一块 GPU

# NVCC 编译器相关
export CUDA_NVCC_FLAGS="-arch=sm_120"   # 根据显卡性能实际调整

```

## CUDA 硬件与工具探索

[techpowerup](https://www.techpowerup.com/gpu-specs/)

### Nsight Compute

    1. 分析数据采集
    ```bash
    // osrt 全称 Operating System Runtime（操作系统运行时），追踪操作系统层面的性能数据，包括进程 / 线程创建与调度、文件 I/O 操作、系统调用、线程同步等底层运行信息，用于排查系统层面的性能瓶颈。

    // cuda 追踪 NVIDIA CUDA 相关的所有核心性能数据，包括 CUDA 内核函数（Kernel）的启动与执行、设备内存分配 / 释放、流（Stream）操作、CUDA API 调用（如 cudaMalloc、cudaLaunchKernel 等）、GPU 设备间数据传输等，是分析 CUDA 程序性能的核心追踪项。

    // nvtx 追踪用户通过 NVTX 库标记的自定义性能事件（如函数调用、代码块执行、业务流程阶段等），用于将自定义的业务逻辑与 GPU/CUDA 性能数据关联，更精准地定位业务代码中的性能问题。

    // --stats=true 启用自动统计分析功能，在性能追踪完成后，自动对采集到的原始数据进行汇总、统计和分析，生成结构化的统计报告。

    // -f --force-overwrite 的简写参数，作用是强制覆盖已存在的同名输出文件，避免因文件已存在而中断

    nsys profile --trace=osrt,cuda,nvtx --stats=true -f true -o report_data ./demo_exec
    ```

    2. 分析数据采集
    ```bash
    // 需要 sudo 权限
    // ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling //permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
    
    // 需要选择与驱动版本相匹配的 ncu
    sudo /usr/local/cuda-12.6/bin/ncu --set full -o report_data ./cuda_001

    // sudo /usr/local/NVIDIA-Nsight-Compute-2025.3/ncu --set full -o report_data ./demo_exec
    ```


## CUDA 参数计算

1. FLOPS 每秒浮点运算次数
    ```
    // FLOPS = CPU 核数 * 单核主频 * CPU 单个周期浮点计算能力
    82.58 TFLOPS = 82575360 FLOPS = 16384 * 2520 MHz * 2（乘加）
    ```

2. memory bandwidth 计算平台的带宽上限，一个计算平台每秒所能完成的内存交换量，单位是 Byte/s
    ```
    // bandwidth = 内存频率 * Prefetch * 内存位宽 / 8
    1.01 TB/s = 1313MHz * 16 * 384 bit / 8
    ```

3. CUDA 核心计算库（CCCL）提供了一个方便的工具，用于进行上限除法，以计算内核启动所需的块数。该工具通过包含头部 来实现。```cuda::ceil_div<cuda/cmath>```

    ```C++
    // vectorLength is an integer storing number of elements in the vector
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    ```

    这里每块256个线程的选择是任意的，但这通常是个不错的起点。