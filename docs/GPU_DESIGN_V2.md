# 🚀 GPU 加速后端设计文档 v2

> **版本**：2.0（重新设计）  
> **目标**：为 neuralnet.cpp 添加 Vulkan Compute GPU 加速，解决旧版 GPU-resident pipeline 的正确性问题，实现真正的零层间 PCIe 传输。

---

## 📋 目录

1. [设计背景](#-设计背景)
2. [旧版问题回顾](#-旧版问题回顾)
3. [新架构设计](#-新架构设计)
4. [核心组件详解](#-核心组件详解)
5. [数据流与同步机制](#-数据流与同步机制)
6. [与现有代码的集成](#-与现有代码的集成)
7. [分阶段实施计划](#-分阶段实施计划)
8. [测试与验证策略](#-测试与验证策略)

---

## 🎯 设计背景

### 项目现状

当前 `rewrite` 分支是纯 CPU 实现，使用 SmartPolicy 自适应并行策略。虽然 CPU 多线程性能优秀，但对于大规模矩阵运算（如 Transformer 注意力计算），GPU 的数千核心可以提供数量级的吞吐提升。

### 旧版 GPU 后端（`feature/vulkan-compute` 分支）

已在历史分支中实现完整的 Vulkan 计算后端，包括：
- MemoryPool 子分配器
- StagingRing 环形缓冲区
- 双轨制架构（Staging Path + GPU-Resident Path）
- 三个 compute shader（matmul、matmul_tiled、elementwise）

**但 GPU-resident path 存在正确性 bug**，多层链式推理准确率仅 ~10%（预期 95%）。

---

## 🔴 旧版问题回顾

### Bug 1：跨 Command Buffer 缺少 Fence 同步（致命）

**位置**：`GpuBackend::matmul_gpu()` 和 `GpuBackend::elementwise_gpu()`

```cpp
// 旧版代码（有 bug）
vkQueueSubmit(device_.compute_queue(), 1, &submit_info, VK_NULL_HANDLE); // 无 fence
return GpuTensor{C}; // 立即返回，GPU 可能还没算完
```

**问题**：
- Vulkan 规范要求：跨 command buffer 的数据依赖必须通过 fence/semaphore 同步
- `matmul_gpu` 提交后立即返回 `GpuTensor`，但 GPU 可能还在计算
- 下一层 `elementwise_gpu` 立即读取未完成的数据，导致竞态条件

### Bug 2：Pipeline Barrier 跨 Command Buffer 无效（致命）

**位置**：`matmul_gpu()` 中的 `vkCmdPipelineBarrier`

```cpp
// 旧版代码（有 bug）
VkBufferMemoryBarrier barrier{};
barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
vkCmdPipelineBarrier(cmd, ...); // 只在当前 command buffer 内有效
```

**问题**：
- `vkCmdPipelineBarrier` 的作用域**仅限于当前 command buffer**
- 不同 command buffer 提交之间的数据依赖需要 fence/semaphore
- 旧版在同一 mutex 下提交，但 GPU 端执行是异步的

### Bug 3：`pending_ops_` 资源清理竞态（中等）

**位置**：`GpuBackend::flush_pending_ops()`

**问题**：
- `matmul_gpu` 和 `elementwise_gpu` 将资源推入 `pending_ops_`，但不设置 fence
- 多层链式调用时，资源（command buffer、descriptor set）持续累积
- 只在 `download_blocking()` 时才清理，可能导致资源耗尽

### Bug 4：权重缓存训练时不失效（中等）

**位置**：`Linear::forward_gpu()`

```cpp
// 旧版代码（有 bug）
if (!gpu_weights_) { // 懒加载，只在第一次上传
    gpu_weights_ = GpuTensor::from_matrix(W_, backend);
}
// 训练时 W_ 被 optimizer.step() 更新，但 gpu_weights_ 不会失效
```

**问题**：
- 训练循环中，`optimizer.step()` 更新 CPU 端的 `W_` 和 `b_`
- 但 `gpu_weights_` 和 `gpu_bias_` 仍然是旧值
- 下一次 `forward_gpu` 使用过期权重，导致训练不收敛

---

## 🏗️ 新架构设计

### 设计原则

1. **严格分层**：遵循 L0(硬件) → L1(代数) → L2(计算) → L3(模型) 调用规则
2. **同步优先**：每个 GPU 操作完成后显式等待，确保正确性后再优化性能
3. **零手动内存管理**：使用 RAII 封装所有 Vulkan 资源
4. **渐进式集成**：先实现 Staging Path，再启用 GPU-Resident Path
5. **可测试性**：每个组件独立可测，支持 CPU/GPU 结果对比

### 分层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│ L5 用户入口层                                                    │
│   src/mnist_train.cpp, src/text_train.cpp, ...                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ L4 构建层                                                        │
│   domain_mnist.hpp, domain_gpt.hpp                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ L3 实现层                                                        │
│   model_container.hpp (Model)                                   │
│   - forward(): 自动选择 CPU 或 GPU-resident 路径                │
│   - GPU 路径: Upload → [Layer1.forward_gpu → ...] → Download   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ L2 计算层                                                        │
│   compute_layer.hpp (Layer, Linear, ReLU, GeLU, ...)            │
│   - forward(): CPU 路径（现有）                                  │
│   - forward_gpu(): GPU-resident 路径（新增）                    │
│   - backward(): 始终走 CPU 路径（梯度计算通常在 CPU）           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ L1 代数层                                                        │
│   algebra_matrix.hpp (Matrix)                                   │
│   - multiply_to(): 自动 dispatch CPU 或 GPU (staging path)     │
│   - apply_relu_inplace(): 自动 dispatch                         │
│   - apply_gelu_inplace(): 自动 dispatch                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ L0 硬件层                                                        │
│   config.hpp (SmartPolicy)                                      │
│   core_threadpool.hpp (ThreadPool)                              │
│   backend/vk_backend.hpp (GpuBackend)  ← 新增                   │
│   backend/gpu_tensor.hpp (GpuTensor)   ← 新增                   │
│   backend/gpu_sync.hpp (GpuOpHandle)   ← 新增                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧩 核心组件详解

### 组件 1：`GpuBackend` — Vulkan 后端单例

**文件**：`include/neuralnet.cpp/backend/vk_backend.hpp`

**职责**：
- 管理 Vulkan 实例、设备、队列（RAII）
- 管理 MemoryPool 子分配器
- 管理 StagingRing 环形缓冲区
- 提供 GPU 操作 API（matmul、elementwise）

**关键 API**：

```cpp
class GpuBackend {
public:
    // 单例访问
    [[nodiscard]] static GpuBackend& instance();
    
    // 初始化（懒加载，首次调用时初始化）
    [[nodiscard]] Result<void> initialize();
    
    // 阻塞式上传：CPU → GPU
    [[nodiscard]] Result<void> upload_blocking(
        GpuTensor& dst, std::span<const Scalar> cpu_data);
    
    // 阻塞式下载：GPU → CPU
    [[nodiscard]] Result<void> download_blocking(
        const GpuTensor& src, std::span<Scalar> cpu_data);
    
    // 纯 GPU 矩阵乘法（阻塞等待完成）
    [[nodiscard]] Result<GpuTensor> matmul_gpu(
        const GpuTensor& A, const GpuTensor& B);
    
    // 纯 GPU 逐元素运算（阻塞等待完成）
    [[nodiscard]] Result<GpuTensor> elementwise_gpu(
        const GpuTensor& primary, const GpuTensor* secondary,
        uint32_t op, uint32_t rows = 0, uint32_t cols = 0);
    
    // Staging Path：CPU span → GPU → 计算 → GPU → CPU span
    [[nodiscard]] Result<void> matmul_direct(
        std::span<const Scalar> a, std::span<const Scalar> b,
        std::span<Scalar> c, std::size_t M, std::size_t N, std::size_t K);
    
    // 查询状态
    [[nodiscard]] bool is_initialized() const noexcept;
    [[nodiscard】 bool gpu_available() const noexcept;
};
```

**同步机制（修复 Bug 1 & 2）**：

```cpp
Result<GpuTensor> GpuBackend::matmul_gpu(const GpuTensor& A, const GpuTensor& B) {
    // ... 录制 command buffer ...
    
    // 1. 创建 fence
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(device_.device(), &fence_info, nullptr, &fence);
    
    // 2. 提交到队列，附带 fence
    VkSubmitInfo submit_info{};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    {
        std::lock_guard lock(queue_mutex_);
        vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence);
    }
    
    // 3. 等待 GPU 完成（阻塞）
    vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(device_.device(), 1, &fence);
    
    // 4. 清理资源
    vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
    vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
    
    return C; // 确保 GPU 计算已完成
}
```

### 组件 2：`GpuTensor` — GPU 矩阵抽象

**文件**：`include/neuralnet.cpp/backend/gpu_tensor.hpp`

**职责**：
- 封装 GPU 缓冲区（`GpuBuffer`）
- 提供形状信息（rows, cols）
- 支持与 `Matrix` 的转换

**关键 API**：

```cpp
class GpuTensor {
private:
    std::shared_ptr<GpuBuffer> buffer_;
    std::size_t rows_;
    std::size_t cols_;

public:
    GpuTensor() = default;
    
    // 从 CPU Matrix 创建（上传数据）
    [[nodiscard]] static Result<GpuTensor> from_matrix(
        const Matrix& cpu_mat, GpuBackend& backend);
    
    // 创建空的 GPU Tensor（用于输出）
    [[nodiscard]] static Result<GpuTensor> create_empty(
        std::size_t rows, std::size_t cols, GpuBackend& backend);
    
    // 转换为 CPU Matrix（下载数据）
    [[nodiscard]] Result<Matrix> to_matrix(GpuBackend& backend) const;
    
    // 访问器
    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard】 std::size_t cols() const noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const GpuBuffer& buffer() const noexcept;
};
```

### 组件 3：`GpuOpHandle` — GPU 操作句柄

**文件**：`include/neuralnet.cpp/backend/gpu_sync.hpp`

**职责**：
- 封装 fence 和相关资源
- 支持 wait + cleanup 语义
- 用于需要延迟等待的场景（未来优化）

**关键 API**：

```cpp
class GpuOpHandle {
private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> cmd_buffers_;
    std::vector<VkDescriptorSet> desc_sets_;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;

public:
    GpuOpHandle() = default;
    ~GpuOpHandle();
    
    // 等待 GPU 完成并清理资源
    void wait_and_cleanup();
    
    // 检查是否已完成（非阻塞）
    [[nodiscard]] bool is_ready() const;
    
    // 移动语义（支持返回值优化）
    GpuOpHandle(GpuOpHandle&& other) noexcept;
    GpuOpHandle& operator=(GpuOpHandle&& other) noexcept;
    
    // 禁止拷贝
    GpuOpHandle(const GpuOpHandle&) = delete;
    GpuOpHandle& operator=(const GpuOpHandle&) = delete;
};
```

### 组件 4：`MemoryPool` — GPU 内存子分配器

**文件**：`include/neuralnet.cpp/backend/memory_pool.hpp`

**职责**：
- 预分配大块 GPU 内存（128MB+）
- 提供 O(log n) 子分配/释放
- 自动合并空闲区域

**关键 API**：

```cpp
class MemoryPool {
public:
    static constexpr VkDeviceSize DEFAULT_BLOCK_SIZE = 128ull * 1024 * 1024;
    
    struct Allocation {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
        VkMemoryPropertyFlags property_flags = 0;
        [[nodiscard]] bool valid() const noexcept;
    };
    
    MemoryPool(VkDevice device, VkPhysicalDevice physical_device,
               VkDeviceSize block_size = DEFAULT_BLOCK_SIZE);
    ~MemoryPool();
    
    // 分配内存
    [[nodiscard]] Result<Allocation> allocate(
        VkMemoryRequirements requirements,
        VkMemoryPropertyFlags preferred_flags,
        VkMemoryPropertyFlags fallback_flags = 0);
    
    // 释放内存
    void free(const Allocation& alloc);
};
```

### 组件 5：`StagingRing` — 环形 Staging 缓冲区

**文件**：`include/neuralnet.cpp/backend/staging_ring.hpp`

**职责**：
- 管理 HOST_VISIBLE 缓冲区用于 CPU↔GPU 数据传输
- 环形分配，避免频繁创建/销毁
- Fence 同步确保数据安全

**关键 API**：

```cpp
class StagingRing {
public:
    static constexpr std::size_t DEFAULT_REGION_SIZE = 256ull * 1024 * 1024;
    static constexpr std::size_t DEFAULT_NUM_REGIONS = 4;
    
    StagingRing() = default;
    ~StagingRing();
    
    // 初始化
    [[nodiscard]] Result<void> initialize(
        VkDevice device, VkPhysicalDevice physical_device,
        VkCommandPool cmd_pool, MemoryPool* pool);
    
    // 获取一个 staging region
    [[nodiscard]] std::size_t acquire();
    
    // 上传数据到 staging region
    [[nodiscard]] Result<void> upload(
        std::size_t region_idx, std::span<const Scalar> data, VkDeviceSize offset);
    
    // 从 staging region 下载数据
    [[nodiscard]] Result<void> download(
        std::size_t region_idx, std::span<Scalar> data, VkDeviceSize offset);
    
    // 获取缓冲区句柄
    [[nodiscard]] VkBuffer buffer(std::size_t region_idx) const noexcept;
    
    // 获取 fence
    [[nodiscard]] VkFence fence(std::size_t region_idx) const noexcept;
};
```

### 组件 6：Compute Shaders

**文件**：`shaders/matmul.comp`, `shaders/matmul_tiled.comp`, `shaders/elementwise.comp`

**matmul.comp** — 16×16 分块矩阵乘法：
- WorkGroup: 16×16 (256 线程)
- 每个 WorkGroup 计算 16×16 输出块
- Shared Memory 缓存 A 和 B 的分块
- 多轮迭代 K 维度

**matmul_tiled.comp** — 4×4 粗化分块矩阵乘法：
- WorkGroup: 16×16 (256 线程)
- 每个 WorkGroup 计算 64×64 输出块（每线程 4×4 元素）
- 寄存器分块，大幅提升计算访存比
- Shared Memory Padding 避免 Bank Conflict

**elementwise.comp** — 逐元素运算：
- WorkGroup: 256 线程
- 支持 ReLU (op=0), QuickGeLU (op=1), BiasAdd (op=2)
- Push Constants 选择操作类型

---

## 🔄 数据流与同步机制

### Staging Path（自动 fallback）

```
┌─────────────────────────────────────────────────────────────────┐
│                    Staging Path 数据流                          │
│                                                                 │
│  CPU Matrix A ──┐                                               │
│                 │ Upload (memcpy to staging)                    │
│                 ▼                                               │
│  ┌─────────────────────────────────────────┐                   │
│  │ Staging Ring (HOST_VISIBLE)             │                   │
│  └─────────────────────────────────────────┘                   │
│                 │ Copy (vkCmdCopyBuffer)                        │
│                 ▼                                               │
│  ┌─────────────────────────────────────────┐                   │
│  │ GPU Buffer A (DEVICE_LOCAL)             │                   │
│  └─────────────────────────────────────────┘                   │
│                 │ Compute (vkCmdDispatch)                       │
│                 ▼                                               │
│  ┌─────────────────────────────────────────┐                   │
│  │ GPU Buffer C (DEVICE_LOCAL)             │                   │
│  └─────────────────────────────────────────┘                   │
│                 │ Copy (vkCmdCopyBuffer)                        │
│                 ▼                                               │
│  ┌─────────────────────────────────────────┐                   │
│  │ Staging Ring (HOST_VISIBLE)             │                   │
│  └─────────────────────────────────────────┘                   │
│                 │ Download (memcpy from staging)                │
│                 ▼                                               │
│  CPU Matrix C ◄─┘                                               │
│                                                                 │
│  同步点：vkWaitForFences (每次操作完成后)                       │
└─────────────────────────────────────────────────────────────────┘
```

**特点**：
- 每层都有 CPU↔GPU 传输
- 每次操作都阻塞等待完成
- 简单、正确、易于调试
- 适用于小矩阵或调试阶段

### GPU-Resident Path（零层间 PCIe）

```
┌─────────────────────────────────────────────────────────────────┐
│                 GPU-Resident Path 数据流                        │
│                                                                 │
│  CPU Input ──┐                                                  │
│              │ Upload (仅一次，输入层)                           │
│              ▼                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    GPU VRAM                               │  │
│  │                                                          │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │  │
│  │  │ Layer 1     │───►│ Layer 2     │───►│ Layer 3     │  │  │
│  │  │ matmul+relu │    │ matmul+gelu │    │ matmul      │  │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘  │  │
│  │       │                  │                  │            │  │
│  │       ▼                  ▼                  ▼            │  │
│  │   GpuTensor          GpuTensor          GpuTensor       │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│              │                                                  │
│              │ Download (仅一次，输出层)                         │
│              ▼                                                  │
│  CPU Output ◄─┘                                                 │
│                                                                 │
│  同步点：每层 matmul_gpu/elementwise_gpu 完成后 fence wait      │
└─────────────────────────────────────────────────────────────────┘
```

**特点**：
- 只在输入/输出各一次 PCIe 传输
- 中间层全部在 GPU 显存流转
- 每层操作完成后 fence wait，确保数据一致性
- 适用于大矩阵、多层网络

### 同步机制详解

**核心原则**：每个 `matmul_gpu` 或 `elementwise_gpu` 调用都必须：

1. **录制 command buffer**
2. **创建 fence**
3. **提交到队列**（带 fence）
4. **等待 fence**（阻塞）
5. **销毁 fence**
6. **清理资源**（command buffer、descriptor set）

```cpp
// 完整的同步流程
Result<GpuTensor> GpuBackend::matmul_gpu(const GpuTensor& A, const GpuTensor& B) {
    // 1. 分配输出 Tensor
    auto C = GpuTensor::create_empty(M, N, *this);
    
    // 2. 录制 command buffer
    VkCommandBuffer cmd = ...;
    vkBeginCommandBuffer(cmd, &begin_info);
    
    // 3. 添加 pipeline barrier（确保输入数据就绪）
    VkBufferMemoryBarrier input_barrier{};
    input_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    input_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, ...);
    
    // 4. 绑定 pipeline 和 descriptor set
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, ...);
    
    // 5. 推送常量
    vkCmdPushConstants(cmd, ...);
    
    // 6. Dispatch
    vkCmdDispatch(cmd, (N + 15) / 16, (M + 15) / 16, 1);
    
    // 7. 添加输出 barrier（确保写入完成）
    VkBufferMemoryBarrier output_barrier{};
    output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    output_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(cmd, ...);
    
    // 8. 结束录制
    vkEndCommandBuffer(cmd);
    
    // 9. 创建 fence
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    vkCreateFence(device_.device(), &fence_info, nullptr, &fence);
    
    // 10. 提交到队列
    VkSubmitInfo submit_info{};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    {
        std::lock_guard lock(queue_mutex_);
        vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence);
    }
    
    // 11. 等待完成
    vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, UINT64_MAX);
    
    // 12. 清理
    vkDestroyFence(device_.device(), 1, &fence);
    vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
    vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
    
    return C; // 确保 GPU 计算已完成
}
```

---

## 🔗 与现有代码的集成

### 集成点 1：`config.hpp` — 添加 GPU 配置

```cpp
struct SmartPolicy {
    // 现有代码...
    
    // GPU 加速阈值：矩阵面积超过此值时自动走 GPU
    inline static constexpr std::size_t GPU_THRESHOLD = 64 * 64; // 4096 元素
    
    // 是否启用 GPU 后端（运行时开关）
    inline static bool gpu_enabled = false;
    
    // GPU 操作计数（性能统计）
    inline static std::atomic<uint64_t> gpu_matmul_count{0};
    inline static std::atomic<uint64_t> cpu_matmul_count{0};
};
```

### 集成点 2：`algebra_matrix.hpp` — Matrix 类扩展

```cpp
class Matrix {
private:
    std::vector<Scalar> data_{};
    std::size_t rows_{0};
    std::size_t cols_{0};
    
    // 现有代码...

public:
    // 现有 multiply_to 实现...
    void multiply_to(Matrix& result, const Matrix& other) const {
        const std::size_t M = rows_;
        const std::size_t K = cols_;
        const std::size_t N = other.cols_;
        
#ifdef NN_HAS_VULKAN
        // GPU 加速路径（Staging Path）
        if (SmartPolicy::gpu_enabled && M * N >= SmartPolicy::GPU_THRESHOLD) {
            auto& backend = GpuBackend::instance();
            if (backend.is_initialized() || backend.initialize()) {
                auto mm = backend.matmul_direct(
                    span(), other.span(), result.span(), M, N, K);
                if (mm) {
                    SmartPolicy::gpu_matmul_count.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            }
            // GPU 失败，静默 fallback 到 CPU 路径
        }
        SmartPolicy::cpu_matmul_count.fetch_add(1, std::memory_order_relaxed);
#endif
        
        // CPU 路径（现有实现）
        // ...
    }
    
    // 新增：GPU-resident 版本（供 Layer::forward_gpu 使用）
#ifdef NN_HAS_VULKAN
    [[nodiscard]] Result<GpuTensor> to_gpu_tensor(GpuBackend& backend) const {
        return GpuTensor::from_matrix(*this, backend);
    }
#endif
};
```

### 集成点 3：`compute_layer.hpp` — Layer 类扩展

```cpp
class Layer {
public:
    // 现有接口...
    virtual Result<Matrix> forward(const Matrix& input) = 0;
    virtual Result<Matrix> backward(const Matrix& grad_output) = 0;
    
#ifdef NN_HAS_VULKAN
    // 新增：GPU-resident 前向传播
    // 默认实现：Download → CPU 计算 → Upload（fallback）
    virtual Result<GpuTensor> forward_gpu(const GpuTensor& input, GpuBackend& backend) {
        // Fallback：Download → CPU 计算 → Upload
        auto cpu_in = input.to_matrix(backend);
        if (!cpu_in) return std::unexpected(cpu_in.error());
        
        auto cpu_out = forward(*cpu_in);
        if (!cpu_out) return std::unexpected(cpu_out.error());
        
        return GpuTensor::from_matrix(*cpu_out, backend);
    }
    
    // 新增：使 GPU 权重缓存失效（训练时调用）
    virtual void invalidate_gpu_cache() {}
#endif
};

class Linear final : public Layer {
private:
    Matrix w_;
    Matrix b_;
    // ... 现有成员 ...
    
#ifdef NN_HAS_VULKAN
    std::optional<GpuTensor> gpu_weights_;
    std::optional<GpuTensor> gpu_bias_;
#endif

public:
    // 现有 forward 实现...
    
#ifdef NN_HAS_VULKAN
    Result<GpuTensor> forward_gpu(const GpuTensor& input, GpuBackend& backend) override {
        // 1. 上传权重（每次都上传，确保训练时权重最新）
        // 优化：可以用 dirty flag 判断是否需要重新上传
        gpu_weights_.reset();
        gpu_bias_.reset();
        
        auto w_res = GpuTensor::from_matrix(w_, backend);
        if (!w_res) return std::unexpected(w_res.error());
        gpu_weights_ = std::move(*w_res);
        
        auto b_res = GpuTensor::from_matrix(b_, backend);
        if (!b_res) return std::unexpected(b_res.error());
        gpu_bias_ = std::move(*b_res);
        
        // 2. 纯 GPU 矩阵乘法（阻塞等待）
        auto mm_res = backend.matmul_gpu(input, *gpu_weights_);
        if (!mm_res) return std::unexpected(mm_res.error());
        
        // 3. GPU 端 Bias Add（阻塞等待）
        auto ba_res = backend.elementwise_gpu(
            *mm_res, &*gpu_bias_, 2u,
            static_cast<uint32_t>(w_.rows()),
            static_cast<uint32_t>(input.cols()));
        if (!ba_res) return std::unexpected(ba_res.error());
        
        return ba_res;
    }
    
    void invalidate_gpu_cache() override {
        gpu_weights_.reset();
        gpu_bias_.reset();
    }
#endif
};

class ReLU final : public Layer {
public:
    // 现有 forward 实现...
    
#ifdef NN_HAS_VULKAN
    Result<GpuTensor> forward_gpu(const GpuTensor& input, GpuBackend& backend) override {
        return backend.elementwise_gpu(input, nullptr, 0u); // op=0: ReLU
    }
#endif
};

class GeLU final : public Layer {
public:
    // 现有 forward 实现...
    
#ifdef NN_HAS_VULKAN
    Result<GpuTensor> forward_gpu(const GpuTensor& input, GpuBackend& backend) override {
        return backend.elementwise_gpu(input, nullptr, 1u); // op=1: QuickGeLU
    }
#endif
};
```

### 集成点 4：`model_container.hpp` — Model 类扩展

```cpp
class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    
public:
    // 现有 forward 实现...
    [[nodiscard]] Result<Matrix> forward(const Matrix& input) {
        if (layers_.empty()) {
            return std::unexpected(Error{"Model has no layers"});
        }
        
#ifdef NN_HAS_VULKAN
        // GPU-resident 路径
        if (SmartPolicy::gpu_enabled) {
            auto& backend = GpuBackend::instance();
            if (backend.is_initialized() || backend.initialize()) {
                // 1. 上传输入到 GPU（唯一一次 CPU→GPU）
                auto gpu_in = GpuTensor::from_matrix(input, backend);
                if (gpu_in) {
                    // 2. 逐层 forward_gpu（全程 GPU 显存流转）
                    GpuTensor gpu_out = std::move(*gpu_in);
                    bool gpu_ok = true;
                    
                    for (std::size_t i = 0; i < layers_.size(); ++i) {
                        auto layer_res = layers_[i]->forward_gpu(gpu_out, backend);
                        if (!layer_res) {
                            gpu_ok = false;
                            break;
                        }
                        gpu_out = std::move(*layer_res);
                    }
                    
                    if (gpu_ok) {
                        // 3. 下载输出到 CPU（唯一一次 GPU→CPU）
                        auto cpu_out = gpu_out.to_matrix(backend);
                        if (cpu_out) return cpu_out;
                    }
                }
                // GPU 路径失败，静默 fallback 到 CPU 路径
            }
        }
#endif
        
        // CPU 路径（现有实现）
        auto result = layers_.front()->forward(input);
        if (!result) return std::unexpected(result.error());
        Matrix out = std::move(*result);
        
        for (std::size_t i = 1; i < layers_.size(); ++i) {
            result = layers_[i]->forward(out);
            if (!result) return std::unexpected(result.error());
            out = std::move(*result);
        }
        
        return out;
    }
};
```

### 集成点 5：`compute_optimizer.hpp` — 优化器扩展

```cpp
class Adam final : public Optimizer {
public:
    // 现有 step 实现...
    Result<void> step() override {
        // ... 更新参数 ...
        
#ifdef NN_HAS_VULKAN
        // 使所有层的 GPU 权重缓存失效
        // 注意：需要 Model 提供访问 layers 的接口
        // 或者在 Optimizer 中维护一个 Layer* 列表
        for (auto* layer : layers_) {
            layer->invalidate_gpu_cache();
        }
#endif
        
        return {};
    }
};
```

### 集成点 6：`CMakeLists.txt` — 构建系统扩展

```cmake
# ── Vulkan GPU 加速（可选） ────────────────────────────────────────
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    message(STATUS "Vulkan found: ${Vulkan_LIBRARY}")
    
    # 查找 SPIR-V 编译器
    find_program(GLSLC glslc HINTS ${Vulkan_GLSLC_EXECUTABLE})
    
    # 定义着色器列表
    set(GPU_SHADERS matmul matmul_tiled elementwise)
    
    foreach(SHADER_NAME ${GPU_SHADERS})
        set(SRC ${CMAKE_SOURCE_DIR}/shaders/${SHADER_NAME}.comp)
        set(SPV ${CMAKE_BINARY_DIR}/${SHADER_NAME}.spv)
        set(HPP ${CMAKE_BINARY_DIR}/generated/${SHADER_NAME}_spv.hpp)
        set(FUNC "nn_${SHADER_NAME}_spirv_bytecode")
        
        # 编译 GLSL → SPIR-V
        add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${GLSLC} -o ${SPV} ${SRC}
            DEPENDS ${SRC}
            COMMENT "[GPU] Compiling ${SHADER_NAME}.comp → ${SHADER_NAME}.spv"
        )
        
        # 嵌入 SPIR-V → C++ 头文件
        add_custom_command(
            OUTPUT ${HPP}
            COMMAND ${CMAKE_COMMAND}
                -DSPV_INPUT=${SPV}
                -DHPP_OUTPUT=${HPP}
                -DFUNC_NAME=${FUNC}
                -P ${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake
            DEPENDS ${SPV} ${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake
            COMMENT "[GPU] Embedding ${SHADER_NAME} SPIR-V into C++ header"
        )
        
        list(APPEND GPU_SHADER_HPPS ${HPP})
    endforeach()
    
    if(NOT GLSLC)
        message(WARNING "No SPIR-V compiler found (glslc). "
                        "GPU acceleration disabled. Install Vulkan SDK.")
        set(Vulkan_FOUND OFF)
    endif()
    
    if(Vulkan_FOUND)
        add_custom_target(compile_gpu_shaders DEPENDS ${GPU_SHADER_HPPS})
        
        # 为所有目标启用 Vulkan
        set(NN_GPU_TARGETS
            mnist_train mnist_infer
            text_train text_infer
            compute_bench
        )
        
        foreach(TARGET ${NN_GPU_TARGETS})
            target_compile_definitions(${TARGET} PRIVATE NN_HAS_VULKAN)
            target_include_directories(${TARGET} PRIVATE ${CMAKE_BINARY_DIR}/generated)
            target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan)
            add_dependencies(${TARGET} compile_gpu_shaders)
        endforeach()
        
        message(STATUS "GPU acceleration enabled (Vulkan compute)")
    endif()
else()
    message(STATUS "Vulkan not found. GPU acceleration disabled.")
endif()
```

---

## 📅 分阶段实施计划

### Phase 1：基础设施（1-2 周）

| 任务 | 文件 | 说明 |
|------|------|------|
| 1.1 创建 backend 目录 | `include/neuralnet.cpp/backend/` | 新建目录结构 |
| 1.2 实现 MemoryPool | `backend/memory_pool.hpp` | GPU 内存子分配器 |
| 1.3 实现 StagingRing | `backend/staging_ring.hpp` | 环形 staging 缓冲区 |
| 1.4 实现 GpuBackend 基础 | `backend/vk_backend.hpp` | 设备初始化、队列管理 |
| 1.5 创建 shaders 目录 | `shaders/` | 三个 compute shader |
| 1.6 创建 cmake 脚本 | `cmake/embed_spirv.cmake` | SPIR-V 嵌入 |
| 1.7 修改 CMakeLists.txt | `CMakeLists.txt` | 添加 Vulkan 依赖 |

**里程碑**：能编译通过，Vulkan 设备初始化成功

### Phase 2：Staging Path（1 周）

| 任务 | 文件 | 说明 |
|------|------|------|
| 2.1 实现 upload_blocking | `backend/vk_backend.hpp` | CPU → GPU 阻塞上传 |
| 2.2 实现 download_blocking | `backend/vk_backend.hpp` | GPU → CPU 阻塞下载 |
| 2.3 实现 matmul_direct | `backend/vk_backend.hpp` | Staging path 矩阵乘法 |
| 2.4 实现 elementwise_direct | `backend/vk_backend.hpp` | Staging path 逐元素运算 |
| 2.5 修改 config.hpp | `config.hpp` | 添加 GPU 配置项 |
| 2.6 修改 algebra_matrix.hpp | `algebra_matrix.hpp` | 添加 GPU dispatch |
| 2.7 编写 gpu_test.cpp | `src/gpu_test.cpp` | 单层正确性测试 |

**里程碑**：Staging path 工作，单层 matmul 正确

### Phase 3：GPU-Resident Path（1-2 周）

| 任务 | 文件 | 说明 |
|------|------|------|
| 3.1 实现 GpuTensor | `backend/gpu_tensor.hpp` | GPU 矩阵抽象 |
| 3.2 实现 matmul_gpu | `backend/vk_backend.hpp` | 纯 GPU 矩阵乘法（带 fence） |
| 3.3 实现 elementwise_gpu | `backend/vk_backend.hpp` | 纯 GPU 逐元素运算（带 fence） |
| 3.4 修改 compute_layer.hpp | `compute_layer.hpp` | 添加 forward_gpu 接口 |
| 3.5 实现 Linear::forward_gpu | `compute_layer.hpp` | Linear 层 GPU 实现 |
| 3.6 实现 ReLU/GeLU::forward_gpu | `compute_layer.hpp` | 激活层 GPU 实现 |
| 3.7 修改 model_container.hpp | `model_container.hpp` | 添加 GPU-resident forward |

**里程碑**：多层链式推理正确，准确率 ≥ 93%

### Phase 4：训练支持（1 周）

| 任务 | 文件 | 说明 |
|------|------|------|
| 4.1 实现 invalidate_gpu_cache | `compute_layer.hpp` | 权重缓存失效机制 |
| 4.2 修改 optimizer | `compute_optimizer.hpp` | step() 后使缓存失效 |
| 4.3 训练正确性测试 | `src/mnist_train.cpp` | MNIST 训练收敛测试 |

**里程碑**：GPU-resident 训练正确收敛

### Phase 5：性能优化（可选，2-3 周）

| 任务 | 说明 |
|------|------|
| 5.1 Batch submit | 多层 command buffer 一次性提交 |
| 5.2 Double buffering | 两组 command buffer 交替使用 |
| 5.3 Async compute | 多 compute queue 并行 |
| 5.4 LayerNorm/Softmax shader | 专用归约 shader |
| 5.5 权重 dirty flag | 避免每次都上传权重 |

**里程碑**：GPU 利用率 ≥ 80%，性能提升 ≥ 5×

---

## 🧪 测试与验证策略

### 单元测试

| 测试 | 输入 | 验证 |
|------|------|------|
| test_matmul_16x16 | 16×16 方阵 | `max_abs_err < 1e-5` (float) |
| test_matmul_256x256 | 256×256 方阵 | `max_abs_err < 1e-4` |
| test_matmul_4096x4096 | 4096×4096 方阵 | `max_abs_err < 1e-3` |
| test_elementwise_relu | 1000 元素 | 逐元素比较 |
| test_elementwise_gelu | 1000 元素 | `max_abs_err < 1e-6` |
| test_elementwise_biasadd | 512×64 | 行广播正确性 |

### 集成测试

| 测试 | 网络 | 验证 |
|------|------|------|
| test_2layer_chain | Linear→ReLU→Linear | CPU/GPU 输出一致 |
| test_5layer_mlp | 784→512→256→128→64→10 | MNIST 准确率 ≥ 93% |
| test_gpt_inference | GPT-2 small | 生成文本语义一致 |
| test_training_loop | MNIST 10 epochs | 收敛曲线一致 |

### 性能基准

| 指标 | Staging Path | GPU-Resident | 目标 |
|------|-------------|-------------|------|
| 单层 matmul (4096²) | 5ms | 0.5ms | 10× 加速 |
| 5 层 MLP 前向 | 25ms | 3ms | 8× 加速 |
| PCIe 传输量 (5 层) | 5×2×数据量 | 1×2×数据量 | 5× 减少 |
| GPU 利用率 | 93% | ≥ 80% | - |

---

## 📎 参考资料

- [Vulkan Compute Pipeline 规范](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html)
- [Vulkan 同步机制详解](https://www.khronos.org/blog/understanding-vulkan-synchronization)
- [SPIR-V Storage Buffer 规范](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
- [GPU 矩阵乘法优化](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- 仓库记忆：`/memories/repo/gpu-backend.md`（v4-v6.1 设计历史）
- 修复计划：`docs/GPU_FIX_PLAN.md`（旧版 bug 分析）
