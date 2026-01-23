# Cooperative Groups

## 问题

1. ```__syncthreads``` 粒度太粗（必须是整个 Block）
2. 直接用 ```Warp``` 函数太底层（硬编码 32）
3. 能否有更灵活的线程组织方式

## 优化方案

1. 数据 tilling：优化内存访问，减少全局内存带宽
2. 线程 tilling：优化线程协作，提供灵活的线程组织

## 核心概念

### ```Cooperative Groups``` 是什么

传统方式的局限
```C++
// 只能同步整个 Block
__syncthreads;

// Warp 大小硬编码
#define WARP_SIZE 32
val = __shfl_down_sync(0xffffffff, var, offset);
```

```Cooperative Groups```方式

```C++
#include <cooperative_groups.h>

// 获取当前线程所在的 Block
cooperative_groups::thread_block block = cooperative_groups::this_thread_block();

// 等价于 __syncthreads
block.sync();

// 创建灵活的 tile
cooperative_groups::thread_block_tile<32> tile = cooperative_groups::tiled_partition<32>(block);    // 16 个线程一组
tile.sync();    // 只同步这 16 个线程
```

优势
- 显示表达意图（代码更清晰）
- 灵活的分组大小
- 类型安全
- 为未来硬件特性准备

### 核心对象

```thread_block```整个 Block

```C++
cooperative_groups::thread_block block = cooperative_groups::this_thread_block();

// 常用方法
block.sync();   // 同步
block.thread_rank();    // 线程在 block 中的索引
block.size();   // block 的总线程数
```

```tiled_partition```固定大小的Tile

```C++
// 将 block 分成多个tile，每个 SIZE 个线程
auto tile = cooperative_groups::tiled_partition<SIZE>(block);

// SIZE 必须是 2 的幂：1, 2, 4, 8, 16, 32
tile.sync();    // 同步 tile 内的线程
tile.shlf_down(val, offset);    // tile 内 shuffle
tile.thread_rank(); // 线程在 tile 中的索引
tile.size();    // tile 大小
```

### 用```Cooperative Groups```重写归约

传统 Warp 归约

```C++
__device__ int warpReduceMax(int val){
    for (size_t offset = 16; offset > 0; offset >> 1){
        int neighbor = __shfl_down__sync(0xffffffff, val, offset);
        val = max(val, neighbor);
    }
    return val;
}
```

```Cooperative Groups``` 版本

```C++
template<int TILE_SIZE>
__device__ int warpReduceMax(cooperative_groups::thread_block_tile<TILE_SIZE> tile, int val){
    for (size_t offset = TILE_SIZE / 2; offset > 0; offset >> 1){
        int neighbor = tile.shfl_down(val, offset);
        val = max(val, neighbor);
    }
    return val;
}

// 使用

auto tile32 = cooperative_groups::tiled_partition<SIZE>(block);
int results = tileReduceMax(tile32, local_max);
```

优势
1. 可以修改为 tile16，tile8 等，无需重写函数
2. 不需要硬编码 mask（0xffffffff）
3. 编译器可以做更好的优化

### 多级归约策略

```Cooperative Groups```的威力

```C++
__global__ void findMaxCG(int *data, int n, int * result){
    auto block = cooperative_groups::this_thread_block();

    // 第一级：每个线程处理多个元素
    int local_max = INT_MIN;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        local_max = max(local_max, data[i]);
    }

    // 第二级：tile 内归约（Warp级别）
    auto tile32 = cooperative_groups::tiled_partition<32>(block);
    local_max = tileReduceMax(tile32, local_max);

    // 第三级：每个 Warp 的第一个线程写 shared_memory
    __shared__ int shared[32];  // 最多 32 个 warps
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0){
        shared[warp_id] = local_max;
    }
    block.sync();

    // 第四级：第一个 warp 做最后的归约
    if (threadIdx.x < 32){
        int val = (threadIdx.x < (blockDim.x / 32)) ? shared[threadIDx.x] : INT_MIN;
        val = tileReduceMax(tile32, val);

        if (threadIdx.x == 0){
            atomicMax(result, val);
        }
    }

}
```

### Cooperative Groups 的层次

```C++
cooperative_groups::grid_group  // 整个 Grid （需要特殊启动）

cooperative_groups::thread_block    // 一个 Block 的所有线程

cooperative_groups::tiled_partition<32> // 32 个线程（Warp）

cooperative_groups::tiled_partition<16> // 16 个线程

cooperative_groups::tiled_partition<8>  // 8 个线程

cooperative_groups::tiled_partition<4>  // 4 个线程
```

每一层都有
1. ```.sync()```-同步
2. ```.thread_rank()```-线程索引
3. ```.size()```-组大小
4. ```.shfl_*()```-shuffle通信

### Cooperative Groups 通信函数

```C++
// CG 版本
cooperative_groups::thread_block_tile<16> half_warp = cooperative_groups::tiled_partition<16>(block);    // 16 个线程一组
cooperative_groups::reduce(half_warp, local_max, cooperative_groups::greater<int>());  // 16 线程归约

// 等价于手写版本（width = 16）
for (size_t offset = 8; offset > 0; offset >> 1){
    int neighbor = __shfl_down__sync(0xffff, local_max, offset);  // width = 16
    local_max = max(local_max, neighbor);
}

// CG 版本
cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);    // 32 个线程一组
cooperative_groups::reduce(warp, local_max, cooperative_groups::greater<int>());  // 16 线程归约

// 编译器展开后
for (size_t offset = 16; offset > 0; offset >> 1){
    int neighbor = __shfl_down__sync(0xffffffff, local_max, offset);
    local_max = max(local_max, neighbor);
}

// 不同的归约操作
cooperative_groups::reduce(warp, val, cooperative_groups::plus<int>()); // 求和
cooperative_groups::reduce(warp, val, cooperative_groups::greater<int>());  // 最大值
cooperative_groups::reduce(warp, val, cooperative_groups::less<int>()); // 最小值
cooperative_groups::reduce(warp, val, cooperative_groups::bit_and<int>());  // 按位与
cooperative_groups::reduce(warp, val, cooperative_groups::bit_or<int>());   // 按位或
cooperative_groups::reduce(warp, val, cooperative_groups::bit_xor<int>());  // 按位异或
```