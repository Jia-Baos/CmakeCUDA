# Warp

## 问题

1. 共享内存还是有延迟（20-30 个时钟周期）
2. ```__syncthreads``` 有开销
3. 速度能否进一步提升

## 优化方案

1. Warp 级编程，直接在寄存器间通信

## 核心概念

### 什么是 Warp

定义：
1. 32 个线程的执行单元
2. GPU 硬件调度的最小单位
3. SIMT（Single Instruction Multiple Threads）

特点：

```C++
// 一个 Block （256线程） = 8 个 Warps（每个 32 个线程）
// 线程 0-31：  Warp0
// 线程 32-63：  Warp0
// ...
// 线程 224-255：  Warp7
```

关键：
1. Warp 内的 32 个线程自动同步执行

### ```Shuffle```指令

传统通信

```C++
__shared__ int data[256];
data[tid] = value;  // 写共享内存
__syncthreads();    // 同步
int neighbor = data[tid - 1]    // 读共享内存
```

```Shuffle```通信

```C++
int value = ...;

// 直接从右边线程的寄存器读取，无需内存
int neighbor = __shfl_down__sync(0xffffffff, value, 1);
```

优势
1. 速度：1 个时钟周期 vs 20+ 个时钟周期
2. 无需共享内存
3. 无需 ```__syncthreads```

### ```Warp Shuffle```函数族

```C++
// 从指定 lane 读取，所有线程从 lane 0 读取
__shfl__sync(mask, var, srcLane);

// 从右边 delta 个线程读，归约常用
__shfl_down_sync(mask, var, delta);

// 从左边 delta 个线程读，前缀和
__shfl_up_sync(mask, var, delta);

// 按位异或交换，蝶式归约
__shfl_xor_sync(mask, war, laneMask);
```
