# CUDA 与 C++

## Kernels

可以在 GPU 上执行且可从主机调用的函数称为内核。内核被编写为可同时由多个并行线程运行。

### Kernel 规范

内核的代码通过声明指定符来指定。这向编译器表明该函数将以允许从内核启动时调用的方式编译为 GPU 函数。内核启动是一种启动内核运行的作，通常从 CPU 启动。核是具有返回类型的函数。```__global__``` ```void```

```C++
// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{

}
```

### 启动 kernel

kernel 并行执行线程的线程数量在 kernel 启动时指定。这称为执行配置。同一 kernel 的不同调用可能使用不同的执行配置，例如不同数量的线程或线程块。

三重倒V形符号的前两个参数分别是 grid dimensions 和 the thread block dimensions。


```C++
 __global__ void vecAdd(float* A, float* B, float* C)
 {

 }

int main()
{
    ...
    // Kernel invocation
    vecAdd<<<1, 256>>>(A, B, C);
    ...
}
```

上述代码启动一个包含 256 个线程的单 Thread Block。每个线程执行完全相同的内核代码

每个 Thread Block 的线程数量是有限制的，因为一个 Thread Block 的所有线程都存在于同一流式多处理器（SM），并且必须共享该多处理器的资源。在当前的 GPU 上，一个 Thread Block 最多可包含 1024 个线程。如果资源允许，可以同时在一个 SM 上调度多个 Thread Block。

内核启动相对于宿主线程是异步的。也就是说，内核会被设置为在 GPU 上执行，但主机代码不会等内核完成（甚至开始）在 GPU 上的执行后才继续执行。必须使用 GPU 和 CPU 之间的某种同步来确定内核是否已完成。最基础的版本是完全同步整个 GPU。

### Thread 与 Grid 的索引

- threadIdx：给出 Thread Block 内线程的索引。 Thread Block 中的每个线程都有不同的索引。
- blockDim：给出 Thread Block 的尺寸，该尺寸在内核启动的执行配置中指定。
- blockIdx：给出 Grid 中 Thread Block 的索引。每个 Thread Block 的索引都不同。
- gridDim：给出 Grid 的尺寸，该尺寸在内核启动时的执行配置中规定。

### 边界检查

上述示例假设矢量长度是 Thread Block 长度的整数倍，此处为 256 个 Thread。为了让内核能够处理任意向量长度，我们可以添加检查内存访问未超出数组边界，如下所示，然后启动一个 Thread Block ，该块中会有一些非活跃线程。

```C++
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x

     if(workIndex < vectorLength)
     {
         // Perform computation
         C[workIndex] = A[workIndex] + B[workIndex];
     }
}
```

使用上述内核代码，可以启动比需要更多的线程，而不会导致数组被越界访问。当 ```workIndex``` 超过 ```vectorLength``` 时，线程会退出且不再进行任何工作。在一个没有工作功能的块中启动额外的线程不会产生较高的开销，但应避免启动没有线程工作的线程块。该核现在可以处理非块大小整数倍的向量长度。

所需的 Thread Block 数量可以计算为线程数量的上限，这里指向量长度，除以每个 Thread Block 的线程数。即线程数的整数除以每个 Thread Block 线程数，向上取整。以下给出一种常用的单整数除法表达方式。在整数除法之前加 ```threads - 1```，表现为 ``` ceiling function```，只有当向量长度不能被每个 Thread Block 的线程数整除时，才会再加一个 Thread Block。

```C++
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = (vectorLength + threads-1)/threads;
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

CUDA 核心计算库（CCCL）提供了一个方便的工具，用于进行上限除法，以计算内核启动所需的块数。该工具通过包含头部 来实现。```cuda::ceil_div<cuda/cmath>```

```C++
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = cuda::ceil_div(vectorLength, threads);
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

这里每个 Block 256 个线程的选择是任意的，但这通常是个不错的起点。

## GPU 计算中的内存

### 统一内存

统一内存是 CUDA 运行时的一个功能，允许 NVIDIA 驱动管理主机与设备之间的数据传输。内存可通过 ```cudaMallocManaged``` API 或带有指定词 ```__managed__``` 的变量声明来分配。NVIDIA 驱动会确保每当GPU或CPU试图访问内存时，都能访问。

下面的代码展示了一个完整的函数，用于启动 ```vecAdd``` 内核，该函数使用统一内存来存储将用于 GPU 的输入和输出向量。```cudaMallocManaged``` 分配缓冲区，这些缓冲区可以从 CPU 或 GPU 访问。这些缓冲区通过 ```cudaFree``` 释放。

### 显式内存管理

明确管理内存分配和内存空间间的数据迁移有助于提升应用性能，尽管这会使代码变得冗长。可使用 ```cudaMalloc``` 在 GPU 上进行内存分配，使用```cudaFree``` 释放所前者所分配的内存。

```cudaMemcpy``` API 是同步的。也就是说，直到复制完成后才会返回。

推荐使用 ```cudaMallocHost``` 在 CPU 上分配内存。这在主机上分配了页面锁定内存（page-locked memory），可以提升复制性能，并且对于异步内存传输是必要的。一般来说，对于 CPU 缓冲区和 GPU 之间的数据传输，使用页面锁定内存是个好习惯。如果部分系统中过多的主机内存被页面锁定，性能可能会下降。最佳实践是只对用于发送或接收 GPU 数据的缓冲区进行页面锁定。

### 内存管理与应用性能

显式内存管理更为繁琐，要求程序员指定主机与设备之间的拷贝。这是显式内存管理的优点和缺点：它提供了更多控制，决定何时在主机和设备之间拷贝数据，内存驻留在哪里，以及具体分配哪里。显式内存管理可以提供控制内存传输并与其他计算重叠的性能机会。

### CPU 与 GPU 同步

内核启动相对于调用它们的 CPU 线程是异步的。这意味着 CPU 线程的控制流会在内核尚未完成之前继续执行，甚至可能在内核启动前就已开始。为了保证内核在主机代码中继续前完成了执行，需要某种同步机制。

同步 GPU 与主机线程最简单的方法是使用 ```cudaDeviceSynchronize```，该方法阻挡主机线程，直到 GPU 上所有之前发布的工作完成。在较大的应用中，GPU上可能有多个流在执行工作，```cudaDeviceSynchronize``` 会等待所有流的工作完成。


### 线程的同步

线程通常需要与其他线程合作和通信以完成工作。Thread Block 内的线程可以通过共享内存共享数据，并同步以协调内存访问。

Thread Block 中最基本的同步机制是 ```__syncthreads()``` 内在同步，它作为一个屏障（barrier ），所有线程必须在此等待，线程才能继续。

为了高效协作，共享内存应是靠近每个处理器核心的低延迟内存（类似于 L1 缓存），且 ```__syncthreads``` 期望轻量级。仅同步单个 Thread Block 内的线程。CUDA 编程模型不支持 Thread Block 间同步。合作组（Cooperative Groups）提供设置除单个 Thread Block 外的同步域的机制。

最佳性能通常使 同步 保持在 Thread Block 内。 Thread Block 仍可利用原子内存函数处理常见结果。

### 运行时初始化

CUDA 运行时为系统中的每个设备创建一个 CUDA Context。该 Context 是该设备的主 Context ，并在需要该设备激活 Context 的第一个运行时函数处初始化。Context 在应用的所有宿主线程之间共享。作为 Context 创建的一部分，设备代码会在必要时按时编译并加载到设备内存中。这一切都是透明的。CUDA 运行时创建的主要 Context 可以通过驱动程序 API 访问，以实现互作性。

自 CUDA 12.0 起，```cudaInitDevice``` 和 ```cudaSetDevice``` 调用 运行时 及 与指定设备关联的主要 Context 的初始化。如果运行时 API 请求在这些调用之前发生，运行时会隐式使用设备 0，并根据需要自行完成初始化，以处理这些请求。这一点在对运行时函数调用进行计时，以及解读首次调用运行时 API 所返回的错误码时，至关重要。在 CUDA 12.0 版本之前，```cudaSetDevice``` 函数并不会触发运行时的初始化流程。

```cudaDeviceReset``` 函数会销毁当前设备的 Primary Context。如果在 Primary Context被销毁之后调用 CUDA 运行时 API，系统将会为该设备创建一个新的 Primary Context。

### CUDA 中的错误检查

```C++
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)



CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));
```

### 错误状态

CUDA 运行时会为每个 Host Thread 维护一个 ```cudaError_t``` 类型的错误状态。该状态的默认值为 ```cudaSuccess```（表示无错误），且每当发生错误时，该状态值会被覆盖更新。```cudaGetLastError``` 函数会返回当前的错误状态，随后将该状态重置为 ```cudaSuccess```。与之相对，```cudaPeekLastError``` 函数会返回当前错误状态，但不会对其进行重置。

使用 ```<<<>>>```语法启动的内核不会返回 ```cudaError_t``` 类型的错误码。一个良好的编程实践是：在内核启动后立即检查错误状态，以此检测内核启动过程中产生的即时错误，或是内核启动之前产生的异步错误。需要注意的是，若在内核启动后立即检查错误状态得到了 ```cudaSuccess```，这并不意味着内核已经成功执行，甚至不代表内核已经开始执行。该结果仅能验证：传递给运行时的 ```kernel launch parameters``` 和 ```execution configuration``` 未触发任何错误，且该错误状态并非内核启动之前遗留的错误或异步错误。

### 异步错误

CUDA 内核启动操作以及许多运行时 API 都是 Asynchronous。每当发生错误时，CUDA 错误状态都会被设置并覆盖更新。这意味着，在异步操作执行过程中发生的错误，只有在下次检查错误状态时才会被报告。正如前文所述，这种检查可能是调用 ```cudaGetLastError```、```cudaPeekLastError```，也可能是调用任何返回 ```cudaError_t``` 类型值的 CUDA API。


当 CUDA 运行时 API 函数返回错误时，错误状态并不会被清除。这意味着，诸如内核进行非法内存访问这类异步错误产生的错误码，会在每一次 CUDA 运行时 API 调用中被返回，直到通过调用 ```cudaGetLastError``` 清除该错误状态为止。

```C++
 vecAdd<<<blocks, threads>>>(devA, devB, devC);
// check error state after kernel launch
CUDA_CHECK(cudaGetLastError());
// wait for kernel execution to complete
// The CUDA_CHECK will report errors that occurred during execution of the kernel
CUDA_CHECK(cudaDeviceSynchronize());
```

### ```CUDA_LOG_FILE```

将错误信息写入到本地文件

### Device and Host Functions

```__global__``` 说明符用于标记内核函数（kernel）的入口点。也就是说，它标记的是一个将在 GPU 上以并行方式调用执行的函数。大多数情况下，内核函数是从主机端（host）启动的，但通过动态并行（dynamic parallelism）技术，也可以在一个内核函数内部启动另一个内核函数。

```__device__``` 说明符表明，某个函数应当为 GPU 进行编译，且仅能被其他 ```__device__``` 函数或 ```__global__``` 函数调用。一个函数（包括类成员函数、函数对象、Lambda 表达式）可以同时被标记为 ```__device__``` 和 ```__host__```。

### Variable Specifiers

CUDA 说明符可用于静态变量声明，以控制变量的存储位置（内存布局）。

- ```__device__```：指定变量存储在 Global Memory 中。
- ```__constant__```：指定变量存储在 Constant Memory 中。
- ```__managed__```：指定变量存储在 Unified Memory 中。
- ```__shared__```：指定变量存储在 Shared Memory 中。

当在 ```__device__``` 函数或 ```__global__`` 函数内部声明变量且不使用任何说明符时，该变量会在可能的情况下被分配到 Registers 中，必要时则会被分配到 Local Memory 中。

而在 ```__device__``` 函数或 ```__global__``` 函数外部声明变量且不使用任何说明符时，该变量会被分配到 System Memory，即主机端内存中。

### Detecting Device Compilation

当一个函数被标记为 ```__host__``` ```__device__``` 双重修饰时，这是在指示编译器为该函数同时生成 GPU 版本代码和 CPU 版本代码。在这类双重修饰的函数中，有时我们需要使用预处理器来指定仅在函数的 GPU 版本或仅在 CPU 版本中生效的代码。检查宏 ```__CUDA_ARCH__``` 是否被定义，是实现这一需求最常用的方式。

### Thread Block Clusters

从 Compute Capability 9.0 及以上版本开始，CUDA 编程模型引入了一个可选的层级结构  Thread Block Clusters，该结构由多个 Thread Blocks 组成。类似于一个 Thread Block 内的所有线程保证会在单个 SM 上协同调度执行」，一个集群内的所有 Thread Block 也保证会在 GPU 的单个 GPU 处理集群（GPU Processing Cluster，GPC）上协同调度执行。

一个集群中的线程块数量可以由用户自定义，在 CUDA 中，8 个线程块是受支持的可移植集群最大尺寸（即跨兼容设备的通用最大尺寸）。需要注意的是，对于那些硬件规格过小、无法支持 8 个多处理器（SM）的 GPU 硬件或多实例 GPU（MIG）配置，其最大集群尺寸会相应减小。识别这类小规格配置，以及识别那些支持超过 8 个线程块的大规格配置，是与架构相关的（architecture-specific），且可通过 ```cudaOccupancyMaxPotentialClusterSize``` API 进行查询。

一个集群内的所有 Thread Blocks 都保证会被协同调度，在单个 GPU 处理集群（GPC）上同时执行，并且允许集群内的线程块通过 Cooperative Groups API 中的 ```cluster.sync()``` 函数，执行由硬件支持的同步操作。Cluster Group 还提供了成员函数，可分别通过 ```num_threads()``` 和 ```num_blocks()``` API，以线程数量和 Block 数量为维度查询 Cluster Group 的大小。线程或 Thread Block 在 Cluster Group 中的位置，可分别通过 ```dim_threads()``` 和 ```dim_blocks()``` API 进行查询。

属于同一个集群的  Thread Blocks 可以访问分布式共享内存（distributed shared memory）—— 即该集群内所有 Thread Blocks 的共享内存之和。集群内的 Thread Block 能够对分布式共享内存中的任意地址执行读取、写入，以及原子操作。


传统线程块的调度与硬件限制：不同线程块可能被调度到 GPU 的不同流式多处理器（SM）上执行（甚至不同 GPC 上），而传统共享内存是绑定在单个 SM 上的片上内存，无法跨 SM 共享，自然也就无法支撑跨线程块的通信。

分布式共享内存，它不是为单个线程块分配独立共享内存，而是将集群内所有线程块的 __shared__ 内存资源聚合起来，形成一个全局可见、统一编址的内存池；此外，这个内存池对集群内的所有线程块开放访问权限，不再有单个线程块的私有边界，因此集群内的任意线程块，都可以直接读写该内存池中的任意地址，实现了跨线程块的共享内存通信。

### Launching with Clusters

Thread Block Clusters 可通过两种方式在内核中启用：一种是使用编译期内核属性  ```__cluster_dims__(X,Y,Z)```，另一种是调用 CUDA 内核启动 API ```cudaLaunchKernelEx```。下方示例展示了如何通过编译期内核属性来启动 Thread Block Clusters。通过该内核属性设置的集群 Size 会在编译阶段被固定，后续即可通过传统的 ```<<<, >>>``` 语法启动该内核。若某个内核采用了编译期集群 Size，那么在启动该内核时，集群 Size 将无法被修改。

```C++
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```