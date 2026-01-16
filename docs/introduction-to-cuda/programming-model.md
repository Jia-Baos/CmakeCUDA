# Programming Model

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

## GPU 硬件模型

像任何编程模型一样，CUDA 依赖于底层硬件的概念模型。在 CUDA 编程中，GPU 可以被视为一组流式多处理器（SM），这些 SM 被组织成称为图形处理集群（GPC）的组。每个 SM 包含一个本地寄存器文件、统一的数据缓存以及多个执行计算的功能单元。统一的数据缓存为共享内存和 L1 缓存提供物理资源。统一数据缓存的分配可通过运行时配置 L1 和共享内存。不同类型的内存大小以及 SM 中功能单元数量可能因 GPU 架构而异。

<div align="center">

<img src="../imgs-cuda/gpu-cpu-system-diagram.png" />
<p>GPU 包含多个流式多处理器（SM），每个 SM 包含许多功能单元。图形处理集群（GPC）是 SM 的集合。GPU 是一组连接到 GPU 内存的 GPC。CPU 通常有多个核心和一个连接系统内存的内存控制器。CPU 和 GPU 通过 PCIe 或 NVLINK 等互连连接连接。</p>

</div>

### Thread Block 和 Grid

当应用程序启动内核时，它会使用许多线程，通常是数百万线程。这些线程被组织成块。A block of threads 被称为 Thread Block，这或许并不令人意外。Thread Block 被组织成 Grid。```Grid 中所有 Thread Block 的大小和尺寸都相同```。

<div align="center">

<img src="../imgs-cuda/grid-of-thread-blocks.png" />
<p>Grid of Thread Blocks。每个箭头代表一个线程（箭头数量并不代表实际线程的数量）。</p>

</div>

```Thread Block 的所有线程都在同一个 SM 中执行```。这使得 Thread Block 内的线程能够高效地通信和同步。Thread Block 内的线程都可以访问芯片上的共享内存，用于 Thread Block 之间的线程信息交换。


一个Grid可能由数百万个Thread Block组成，而执行Grid的GPU可能只有数十甚至数百个SM。Thread Block的所有线程都由单个SM执行，并且在大多数情况下会在该SM上运行到完成。Thread Block之间无法保证调度，因此Thread Block不能依赖其他Thread Block的结果，因为在该Thread Block完成之前，其他Thread Block可能无法调度。

<div align="center">

<img src="../imgs-cuda/thread-block-scheduling.png" />
<p>每个 SM 有一个或多个活跃 Thread Block。在这个例子中，每个 SM 同时调度三个 Thread Block。对于 Grid 中 Thread Block 分配给 SM 的顺序没有保证。</p>

</div>

CUDA 编程模型使任意大的 Grid 能够在任何大小的 GPU 上运行，无论它只有一个 SM 还是数千个 SM。为此，CUDA 编程模型（除少数例外）要求不同 Thread Block 中线程之间不存在数据依赖关系。也就是说，线程不应依赖于同一 Grid 中不同 Thread Block 内的线程结果或与线程同步。Thread Block 内的所有线程同时运行在同一个 SM 上。Grid 内的不同 Thread Block 在可用 SM 之间被调度，并可按任意顺序执行。```简而言之，CUDA 编程模型要求可以任意顺序执行 Thread Block，无论是并行还是串联```。

## Thread Block Clusters

除了 Thread Block 外，具备 9.0 及以上计算能力的 GPU 还有一种称为 Cluster 的可选分组层级。Cluster 是一组 Thread Block，像 Thread Block 和 Grid 一样，可以布局为一维、二维或三维。指定 Cluster 不会改变 Grid 中的 Grid 尺寸或 Thread Block 的索引。

<div align="center">

<img src="../imgs-cuda/grid-of-clusters.png" />
<p>当指定 Cluster 时，Thread Block 在 Grid 中的位置相同，但在包含 Cluster 中也有一定位置。</p>

</div>

指定 Cluster 会将相邻的 Thread Block 分组成 Cluster ，并在 Cluster 层面提供一些额外的同步和通信机会。具体来说， Cluster 中的所有 Thread Block 都在一个 GPC 中执行。```由于 Thread Block 同时调度且在同一 GPC 内，不同 Thread Block 内但同一 Cluster 内的线程可以通过合作组（Cooperative Groups）提供的软件接口相互通信和同步（可以在更细粒度上实现线程的同步）```。 Cluster 中的线程可以访问 Cluster 中所有块的共享内存，这被称为分布式共享内存（distributed shared memory）。 Cluster 的最大大小取决于硬件，且不同设备之间有所不同。

<div align="center">

<img src="../imgs-cuda/thread-block-scheduling-with-clusters.png" />
<p>当指定 Cluster 时，Cluster 中的 Thread Blocks 按其 Cluster 形状在 Grid 中排列。Cluster 的 Thread Blocks 会同时调度在单个 GPC 的 SM 上。</p>

</div>

## Warps and SIMT

在一个 Thread Block 内，线程被组织成 32 个线程组成的 Group，称为 Warps。Warp 在单指令多线程（SIMT）范式中执行内核代码。在 SIMT 中，Warp 中的所有线程执行的是相同的内核代码，但每个线程可能遵循代码中的不同分支。也就是说，虽然程序中的所有线程执行相同的代码，但线程不必遵循相同的执行路径。

当线程被 Warp 执行时，它们会被分配一条 Warp Lane。Warp Lane 编号为 0 到 31，Thread Block 中的线程以可预测的方式分配给 Warps。

Warp 中的所有线程同时执行同一条指令。如果  Warp 中的某些线程执行时遵循控制流分支，而其他线程不遵循，则不遵循分支的线程将被 masked，而执行该分支的线程则被执行。例如，如果条件句只对 Warp 中一半的线程成立，那么在激活线程执行这些指令时， Warp 的另一半会被 masked。当 Warp 中不同的线程遵循不同的代码路径时，这有时称为 Warp Divergence。因此，当 Warp 内的线程沿同一控制流路径时，GPU 的利用率会被最大化。

<div align="center">

<img src="../imgs-cuda/active-warp-lanes.png" />
<p>在这个例子中，只有线程索引为偶数的线程执行 if 语句中的正文，其他线程在执行正文时被 masked</p>

</div>

在 SIMT 模型中，Warp 中的所有线程都同步通过内核。硬件执行可能有所不同。不鼓励利用 Warp 执行如何映射到真实硬件的知识。CUDA 编程模型和 SIMT 表示，Warp 中的所有线程都会一起推进代码。只要遵循编程模型，硬件可以以对程序透明的方式优化 masked lane。如果程序违反了该模型，可能会导致不同 GPU 硬件中存在未定义的行为。

虽然编写 CUDA 代码时不必考虑 Warp，但理解 Warp 执行模型有助于理解全局内存聚合（global memory coalescing）和共享内存库访问模式（ shared memory bank access patterns）等概念。一些高级编程技术利用 Thread Block 内的 Warp 专用化，以限制线程发散并最大化利用率。这种及其他优化利用了线程在执行时被分组为 Warp 的知识。

Warp 执行的一个含义是，Thread Block 最好指定为线程总数，即 32 的倍数。使用任意数量的线程是合法的，但当总数不是 32 的倍数时，Thread Block 的最后一个 Warp 会有一些执行过程中未使用的 lane。这很可能导致该 Warp 的功能单元利用率和内存访问不优。

SIMT 常被拿来与单指令多数据（SIMD）并行处理比较，但它们存在一些重要区别。在 SIMD 中，执行遵循单一的控制流路径，而在 SIMT 中，每个线程允许遵循自己的控制流路径。因此，SIMT 不像 SIMD 那样有固定的数据宽度。

## GPU 内存

### 异构系统中的DRAM存储器

GPU 和 CPU 都直接连接 DRAM 芯片。在拥有多个 GPU 的系统中，每个 GPU 都有自己的内存。```从设备代码的角度来看，连接到 GPU 的 DRAM 被称为全局内存，因为它对 GPU 中的所有 SM 都可访问。这些术语并不意味着它在系统内的任何地方都能访问```。连接到 CPU 的 DRAM 称为系统内存或主机内存。

像 CPU 一样，GPU 使用虚拟内存寻址。在所有当前支持的系统中，CPU 和 GPU 使用统一的虚拟内存空间。这意味着系统中每个 GPU 的虚拟内存地址范围都是唯一且与 CPU 及系统中其他所有 GPU 不同的。对于给定的虚拟内存地址，可以确定该地址是在 GPU 内存中还是系统内存中，并且在拥有多个 GPU 的系统中，可以确定哪个 GPU 内存包含该地址。

有 CUDA API 用于分配 GPU 内存、CPU 内存，并在 CPU 与 GPU 之间、GPU 内部或多 GPU 系统中的 GPU 之间复制分配。数据的局部性可以在需要时被显式控制。

### GPU 中的片上存储器

除了全局内存外，每个 GPU 还配备了一些片上内存。```每个 SM 都有自己的寄存器文件和共享内存。这些内存是 SM 的一部分，可以从内存内执行的线程极快地访问，但其他 SM 中运行的线程无法访问```。

寄存器文件存储线程本地变量，这些变量通常由编译器分配。```共享内存对 Thread Block 或 Cluster 内的所有线程均可访问。共享内存可用于 Thread Block 或 Cluster 的线程之间的数据交换```。

寄存器文件和 SM 中的统一数据缓存具有有限大小。```寄存器文件、共享内存空间和 L1 缓存在 Thread Block 内的所有线程之间共享```。Thread Block 和 Grid 之间线程的数据共享的唯一方案是```全局内存```。

```要将 Thread Block 调度到 SM，每个线程所需的寄存器总数乘以 Thread Block 中的线程数，必须小于或等于SM中可用的寄存器数```。如果 Thread Block 所需的寄存器数量超过寄存器文件大小，内核无法启动，必须减少 Thread Block 中的线程数以使 Thread Block 可启动。

共享内存分配是在 Thread Block 层面进行的。也就是说，与按线程分配的寄存器分配不同，共享内存的分配是整个 Thread Block 共有的。

### 缓存

除了可编程存储器外，GPU 还拥有 L1 和 L2 缓存。每个 SM 都有一个 L1 缓存，它是统一数据缓存的一部分。一个更大的 L2 缓存由 GPU 内所有 SM 共享。每个 SM 还有一个独立的常量缓存，用于缓存在全局内存中被声明为内核寿命内常数的值。编译器也可以将内核参数放入常量内存。这可以通过允许内核参数独立于L1数据缓存缓存，提升内核性能。

### 统一内存

当应用程序在 GPU 或 CPU 上显式分配内存时，该内存仅对运行在该设备上的代码访问。也就是说，CPU 内存只能从 CPU 代码访问，GPU 内存只能从运行在 GPU 上的内核访问。CUDA API 用于在 CPU 和 GPU 之间复制内存，在正确的时间显式地将数据复制到正确的内存。

CUDA 的一个功能称为统一内存，允许应用程序进行内存分配，这些内存可从 CPU 或 GPU 访问。CUDA 运行时或底层硬件在需要时允许访问或将数据迁移到正确位置。即使是统一内存，也通过尽量减少内存迁移，并尽可能访问直接连接到内存的处理器的数据来实现最佳性能。

系统的硬件特性决定了内存空间之间数据的访问和交换方式。统一内存部分介绍了统一内存系统的不同类别。统一内存部分包含了关于统一内存在所有情况下使用和行为的更多细节。

1. 在某些情况下，使用如 CUDA 动态并行等功能时，线程块可能会被挂起到内存中。这意味着 SM 的状态存储在系统管理的 GPU 内存区域，SM 被释放用于执行其他线程块。这类似于 CPU 上的上下文交换。这并不常见。

2. 一个例外是映射内存，即分配了属性的 CPU 内存，允许 GPU 直接访问它。然而，映射访问通过 PCIe 或 NVLINK 连接实现。GPU 无法通过并行性掩盖更高的延迟和较低的带宽，因此映射内存无法有效替代统一内存或将数据置入相应内存空间。