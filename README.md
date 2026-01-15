# CmakeCUDA

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

## Nsight Compute

    1. 分析数据采集
    ```
    nsys profile --trace=osrt,cuda --stats=true -f true -o report_data ./demo_exec
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