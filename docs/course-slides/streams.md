# Steams


## 问题

1. GPU 程序的“等待浪费”，彼此串行

## 优化方案

1. 流水线并行

## 核心概念

### CUDA Stream（流）

定义
1. 独立的操作队列
2. 同一 stream 内的操作：按顺序执行
3. 不同 stream 的操作：可以并发执行

API
```C++
// 创建 stream
cudaStream_t stream;
cudaStreamCreate(&stream);

// 在 stream 中执行操作
cudaMemcpyAsync(..., stream);   // 异步传输
kernel<<<grid, block, 0, stream>>>();   // 指定 stream
cudaMemcpyAsync(..., stream);   // 异步传回
```

### 异步操作 vs 同步操作

同步（默认）
1. CPU 阻塞，等待操作完成
2. 安全、简单、但慢

```C++
cudaMemcpy(...);    // CPU等待传输完成才继续
```

异步
1. CPU 立即返回，操作在后台执行
2. 高效，但需要正确管理依赖

```C++
cudaMemcpyAsync(..., stream);   // CPU 立即返回
```
