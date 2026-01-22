# CmakeCUDA

[在线绘图](https://excalidraw.com/)

[学习链接](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#)

1. 3060 Ti 只有 1 个 copy engine + 1 个 compute engine

2. 一个 GPU kernel 可以执行的必要条件是它在 stream 的队列首位且存在执行 kernel 的空闲硬件资源，所以只有一个 copy engine 时，必须等待前一个流的内容拷贝完成后才能继续拷贝，这期间只能被阻塞。

3. 不同引擎的操作可以完全并行（比如流 1 跑核函数，流 2 跑 memcpy），但同一引擎的操作会受引擎数量限制：
    - 如果 GPU 只有 1 个拷贝引擎，那么多个流的cudaMemcpyAsync只能串行执行（哪怕是不同流）；
    - 新架构 GPU（如 Ampere）有多个拷贝引擎，才能支持多个流的 memcpy 并行。