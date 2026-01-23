# Thread Block Clusters

## 问题

1. ```thread_block```-block 内协作
2. ```thread_block_tile``` -warp 内协作
3. 能否让不同 block 之间协作？

传统限制
1. 不同 block 之间无法同步
2. 不同 block 的 shared_memory 互不可见
3. 只能通过 global memory 通信（慢）

## 核心概念

### 什么是 Thread Block Cluster

层次结构
```C++
Grid（整个 kernel）
    Cluster（多个 Block 的组）
        Block（多个线程）
            Warp（32 个线程）
                Thread（单个线程）
```

示例
```C++
// 传统：Grid = 很多个独立的 Block
kernel<<<100, 256>>>(); // 100 个 Block，互不协作

// Clusters：Grid = 多个 cluster，每个 cluster 包含多个block
dim3 grid(100, 1, 1);
dim3 block(256, 1, 1);
dim3 clusters(4, 1, 1); // 每个 cluster 有 4 个block

// 100 个 block 分成 25 个 cluster，每个 cluster 4 个 block
kernel<<<grid, block, 0, 0, cluster>>>();
```

### 为什么需要 Clusters？

特性1：Cluster 内同步

```C++
#include <cude/cluster>

__global__ void __cluster__dims__(4, 1, 1)  // 声明 cluster 大小
clusterKernel(...){
    // 获取 cluster 对象
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();

    // 同步整个 cluster 的所有 block
    cluster.sync();

    // 现在所有 block 都到达了这个点
}
```

特性2：分布式共享内存（Distributed Shared Memory）

```C++
__global__ void __cluster__dims__(4, 1, 1)  // 声明 cluster 大小
clusterKernel(...){
   __shared__ int local_data[256];  // 本 block 的 shared memory

   // 访问 cluster 内其他 block 的 shared memory
   int *remote_data = cluster.map_shared_rank(local_data, 1);
   // reomte_data 指向 cluster 中 rank 为 1 的 block 的 shared_memory

   // 读取其他 block 的数据
   int value = remote_data[threadIdx.x];
}
```

### CLuster 的硬件限制

限制
1. Cluster 大小：通常 1-8 个 block
2. 所有 cluster 必须大小相同
3. 需要足够的 SM 资源（shared memory，registers）（所以是否 cluster 中的 block 被强行分配到同一个 SM 中，所以可以互相读取共享内存）

### Cluster 编程模型

```C++
__global__ void __cluster__dims__(4, 1, 1)  // 声明 cluster 大小
matmulCluster(float *A, float *B, float *C, int N){
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();

    // 获取 cluster 内的位置
    int cluster_rank = cluster.block_rank();    // cluster 中的第几个 block（0-3）
    int num_blocks = cluster.num_blocks();  // cluster 总共有几个 block（4）

    // 分布式 shared_memory
    __shared__ float tile[TILE_SIZE][TILE_SIZE]；

    // 每个 block 加载不同的 tile
    load_tile(tile, A, cluster_rank);

    // 确保所有 block 都加载完毕
    cluster.sync();

    // 访问其他 block 的 tile
    for (size_t i = 0; i < num_blocks; i++){
        float *remote_tile = cluster.map_shared_rank(tile, 1);
        // 使用 remote_tile 计算...
    }

    // 计算前完成同步
    cluser.sync();
}
```

## 常见陷阱

### Cluster 大小限制

```C++
// size 过大（硬件限制）
__global__ void __cluster__dims__(32, 1, 1)  kernel() {}

// 合理范围（0-8）
__global__ void __cluster__dims__(4, 1, 1)  kernel() {}
```

### 资源不足

```C++
// 如果 shared memory 过大， cluster 可能无法调度
__shared_memory__ float huge[10000];    // 可能导致 cluster 无法形成
```

### 忘记同步

```C++
data[i] = value;
// cluster.sync();  // 缺少同步

int *remote = cluser.map_shared_rank(data, 1);
// 可能读到未初始化的数据
```

## CUDA 同步与内存访问延迟对比

### 内存访问延迟

| 内存类型 | 延迟（时钟周期） | 说明 |
| :--------: | :-----: | :-----: |
| Register | 1 | 寄存器访问，零开销 |
| Shared Memory | 20-30 | SM 内部，bank conflict 时更高 |
| DSMEM（Cluster 内跨 Block） | 30-50 | SM 互连，不经过L2 |
| L1 Cache Hit | 30-40 | SM 本地 L1 |
| L2 Cache Hit | 150-200 | 片上共享 L2 |
| Global Memory（L2 Miss） | 400-800 | 访问 HBM/GDDR |
| Global Memory Atomic（L2 Hit） | 200-400 | 原子操作，L2 命中 |
| Global Memory Atomic（L2 Miss） | 400-800 | 原子操作，需访问 DRAM |

### 同步机制延迟

| 同步机制 | 延迟（时钟周期） | 范围 | 实现方式 | 
| :--------: | :-----: | :-----: | :-----: |
| __syncwarp() | ~5 | Warp 内 32 个线程 | 硬件指令 |
| __syncthreads() | 20-50 | Block 内所有线程 | 硬件 Barrier |
| cluster.sync() | 100-300 | Cluster 内所有 Block | 硬件 Barrier（Hopper+） |
| grid.sync() | 1000-5000+ | Grid 内所有 Block | 软件实现（全局内存 + Atomic） |
| Kernel Launch | 5-30us | N/A | CPU-GPU交互 |
| cudaStreamSynchronize | 10-50us | Stream 级别 | CPU-GPU交互 |
| cudaDeviceSynchronize | 10-100us | 设备级别 | CPU-GPU交互 |

### Cluster vs Grid 同步对比

| 特性 | Cluster Sync | Grid Sync |
| :--------: | :-----: | :-----: |
| 延迟 | 100-300 周期 | 1000-5000+ 周期 |
| 范围 | 2-16 个 Block（通过 GPC） | 所有 Block |
| 实现 | 硬件原生 Barrier | 软件（Aromic + Memory Fence） |
| 可扩展性 | 固定，受 Cluster 大小限制 | 任意 Block 数量 |
| 架构要求 | Hopper（SM90）及以上 | Volta（SM70）及以上 |

### 常见操作延迟速查

| 操作 | 延迟 | 备注 |
| :--------: | :-----: | :-----: |
| 一次 DSMEM 读取 | ～30 周期 | Cluster 内跨 Block |
| 一次 Global Atomic Add | ～400 周期 | 竞争是更高 |
| Cluster Barrier | ～200 周期 | 等待最慢的 Block |
| Grid Barrier | ～2000 周期 | Block 数量多时更高 |
| 新 Kernel 启动 | ～10 us | 约 15000-20000 周期 @1.5GHz|

### 优化建议

```C++
Warp 内通信         5 cycles
Block 内同步        30 cycles
DSMEM 访问          30 cycles
Cluster 同步        200 cycles
L2 访问             200 cycles
Global Atomic      400 cycles
Grid 同步           2000+cycles
Kernel 启动         15000+ cycles
```

### 最佳实践

1. 批量操作后同步：减少同步次数，摊销同步成本
2. 避免频繁 Grid Sync：考虑拆分为多个 Kernel
3. 利用 DSMEM：Cluster 内跨 Block 通信比全局内存快 10x+