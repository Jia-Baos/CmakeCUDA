# CmakeCUDA

[在线绘图](https://excalidraw.com/)

[学习链接](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#)

3060 Ti 只有 1 个 copy engine + 1 个 compute engine

一个 GPU kernel 可以执行的必要条件是它在 stream 的队列首位且存在执行 kernel 的空闲硬件资源