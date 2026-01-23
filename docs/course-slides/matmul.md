# Matrix Mutiple

## 核心概念

### 内存布局

```C++
// 行主序
C[2][1] = array[2 * cols + 1];
```

### GPU 映射：一个线程算一个 C 的元素

线程```(i, j)```的任务
1. 读取 A 的第 i 行（K 个元素）
2. 读取 B 的第 j 列（K 个元素）
3. 计算点积得到 C[i][j]

### 性能指标

FLOPS = Floating Point Operations Per Second（每秒浮点操作数）

矩阵乘法的操作数
- 每个 C 元素：K 次乘法 + K 次加法 = 2K 次操作
- 总共 M  x N x 2K 次操作

示例（1024 x 1024）
- 操作数：2 x 1024 x 1024 = 2,147,483,648 = 2.15 GFLOP
- 如果耗时 20ms = 0.02s
- 性能：2.15 / 0.02 = 107.5 GFLOPS

对比：
- RTX 5090 FP32峰值：～90,000 GFLOPS（90 TFLOPS）
- 朴素实现：～100 GFLOPS
- 利用率：0.1%（很低）


