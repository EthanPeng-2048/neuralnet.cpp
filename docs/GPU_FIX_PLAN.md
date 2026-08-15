# 🔧 GPU-Resident Pipeline 修复计划

> **目标**：修复 v6 GPU-resident 路径的正确性 bug，使多层链式推理在 GPU 显存中全程流转，零层间 PCIe 传输。

---

## 📋 目录

1. [现状回顾](#-现状回顾)
2. [Bug 根因分析](#-bug-根因分析)
3. [修复方案](#-修复方案)
4. [分阶段实施计划](#-分阶段实施计划)
5. [测试策略](#-测试策略)

---

## 📊 现状回顾

### 当前分支 (`rewrite`) 状态

**当前工作区是纯 CPU 实现**，没有任何 GPU 代码。GPU 后端代码存在于 `feature/vulkan-compute` 分支，但该分支经历了多次重构，最新提交已将 Vulkan 后端移除。

### 历史实现（`feature/vulkan-compute` 分支，commit `8456fe7`）

已实现的架构：

| 组件 | 文件 | 状态 |
|------|------|------|
| Vulkan RAII 后端 | `include/neuralnet.cpp/backend/vk_backend.hpp` | ✅ 已实现 |
| GpuTensor 方法实现 | `include/neuralnet.cpp/backend/gpu_tensor_impl.hpp` | ✅ 已实现 |
| matmul shader (16×16) | `shaders/matmul.comp` | ✅ 已实现 |
| matmul_tiled shader (4×4 粗化) | `shaders/matmul_tiled.comp` | ✅ 已实现 |
| elementwise shader | `shaders/elementwise.comp` | ✅ 已实现 |
| SPIR-V 嵌入 | `cmake/embed_spirv.cmake` | ✅ 已实现 |
| 双轨制 GPU 流水线 | `model.hpp` → `forward_gpu` | ✅ 已实现 |

### 两条路径对比

```
┌─────────────────────────────────────────────────────────────┐
│                    Staging Path (v4, 工作正常)               │
│                                                             │
│  Layer 1: CPU → Staging → GPU → 计算 → GPU → Staging → CPU │
│  Layer 2: CPU → Staging → GPU → 计算 → GPU → Staging → CPU │
│  Layer 3: CPU → Staging → GPU → 计算 → GPU → Staging → CPU │
│                                                             │
│  ✅ 正确（每层等待完成），但每层都有 PCIe 传输               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              GPU-Resident Path (v6, 有正确性 Bug)           │
│                                                             │
│  Upload: CPU → GPU (仅一次)                                 │
│  Layer 1: GPU → 计算 → GpuTensor (不等待)                   │
│  Layer 2: GPU → 计算 → GpuTensor (不等待)                   │
│  Layer 3: GPU → 计算 → GpuTensor (不等待)                   │
│  Download: GPU → CPU (仅一次)                               │
│                                                             │
│  ❌ 多层链式结果错误 (~10% 准确率 vs 预期 95%)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Bug 根因分析

经过对 `vk_backend.hpp`、`layer.hpp`、`model.hpp`、`gpu_tensor_impl.hpp` 以及三个 compute shader 的完整代码审查，识别出以下 **5 个独立 Bug**，它们共同导致多层链式推理结果错误：

### Bug 1：跨 Command Buffer 提交缺少队列级同步（致命）

**位置**：`GpuBackend::matmul_gpu()` 和 `GpuBackend::elementwise_gpu()`

**问题**：`matmul_gpu()` 录制 command buffer A（读 buffer_1, 写 buffer_2），提交到 compute queue 后立即返回 `GpuTensor C`。随后 `elementwise_gpu()` 录制 command buffer B（读 buffer_2, 写 buffer_3），再次提交。**两个 command buffer 之间没有任何 fence 或 semaphore 同步**。

```cpp
// matmul_gpu() 末尾 —— 提交后不等待
VkSubmitInfo submit_info{};
submit_info.commandBufferCount = 1;
submit_info.pCommandBuffers = &cmd;
vkQueueSubmit(device_.compute_queue(), 1, &submit_info, VK_NULL_HANDLE);
// ⚠️ 没有 fence！返回 GpuTensor，GPU 可能还没算完

// elementwise_gpu() —— 立即录制新的 command buffer 读取 matmul 的输出
// ⚠️ 此时 matmul 的 GPU 计算可能还没完成！
vkCmdCopyBuffer(cmd, src_buffer, ...); // 读取未完成写入的数据
```

**Vulkan 规范**：同一个 queue 内的 command buffer 按提交顺序执行，**但** command buffer B 必须在 command buffer A **完全提交到 queue** 之后才能提交。当前代码虽然在同一 mutex 保护下提交，但 GPU 端的执行是异步的——buffer A 的 SHADER_WRITE 可能还没完成，buffer B 的 SHADER_READ 就开始了。

**正确做法**：在 `matmul_gpu` 返回前，用 `vkQueueSubmit` + fence 保证 GPU 完成，或在两个 command buffer 之间插入 `VkSemaphore`。

### Bug 2：管线屏障 (Pipeline Barrier) 跨 Command Buffer 无效（致命）

**位置**：`matmul_gpu()` 中的 `VkBufferMemoryBarrier`

```cpp
// matmul_gpu() 录制的管线屏障
VkBufferMemoryBarrier input_barriers[2]{};
input_barriers[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
input_barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
vkCmdPipelineBarrier(cmd,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // srcStageMask
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // dstStageMask
    ...);
```

**问题**：`vkCmdPipelineBarrier` 的作用域**仅限于当前 command buffer 内部**。它保证的是：在同一个 command buffer 中，之前的 SHADER_WRITE 在后续 SHADER_READ 之前完成。但它**不保证**不同 command buffer 之间的顺序——后一个 command buffer 的 SHADER_READ 可能在前一个 command buffer 的 SHADER_WRITE 之前执行。

**Vulkan 规范**：跨 command buffer 的同步必须使用 `VkSemaphore`（queue 间）或 `VkFence`（CPU-GPU）。

### Bug 3：`pending_ops_` 资源清理机制存在竞态条件（中等）

**位置**：`GpuBackend::flush_pending_ops()`

```cpp
void flush_pending_ops()
{
    auto& p = pending_ops_;
    if (p.has_fence) {
        vkWaitForFences(device_.device(), 1, &p.fence, VK_TRUE, 10'000'000'000ULL);
        vkDestroyFence(device_.device(), p.fence, nullptr);
        p.has_fence = false;
    }
    // ...
}
```

**问题**：
1. `matmul_gpu` 和 `elementwise_gpu` 将 command buffer 和 descriptor set 推入 `pending_ops_`，但**不设置 fence**（`has_fence` 始终为 false）。
2. 只有 `download_blocking()` 调用 `flush_pending_ops()`，但如果 GPU-resident 路径有 10 层，前 9 层的资源（command buffer、descriptor set）一直累积，直到最后 download 时才清理。
3. 更严重的是：`pending_ops_` 是单例的，只有一组 `cmd_buffers`、`desc_sets`、`fence`。如果两次 `matmul_gpu` 调用之间没有 flush，第二次的 command buffer 会被追加到同一个 vector，但两次操作可能引用**相同的 descriptor set**（因为 `gpu_tensor_pool_` 可能回收后重用同一 set），导致数据竞争。

### Bug 4：`GpuTensor` 权重缓存在训练时不会失效（中等）

**位置**：`Linear::forward_gpu()`

```cpp
Result<GpuTensor> forward_gpu(const GpuTensor& input, GpuBackend& backend) override
{
    // 懒加载权重到 GPU（只在第一次调用时上传，之后常驻显存）
    if (!gpu_weights_) {
        auto w_res = GpuTensor::from_matrix(W_, backend);
        gpu_weights_ = std::move(*w_res);
    }
    // ...
}
```

**问题**：在训练循环中，`optimizer.step()` 更新的是 CPU 端的 `W_` 和 `b_`，但 `gpu_weights_` 和 `gpu_bias_` 仍然是旧值。下一次 `forward_gpu` 调用时，因为 `gpu_weights_` 已有值（`has_value() == true`），不会重新上传，导致**前向传播使用过期的权重**。

**影响**：此 bug 仅影响 GPU-resident **训练**路径。推理路径（加载预训练模型后）不受影响，因为权重在推理期间不变。

### Bug 5：`Scalar` 精度不匹配导致数值误差累积（低）

**位置**：`detail::convert_scalar_to_float()` / `detail::convert_float_to_scalar()`

```cpp
// CPU 使用 double（Scalar = double），GPU 使用 float
inline void convert_scalar_to_float(
    std::span<const Scalar> src, float* __restrict dst) noexcept
{
    std::transform(src.begin(), src.end(), dst,
                   [](Scalar v) { return static_cast<float>(v); });
}
```

**问题**：每次 upload/download 都有 double→float→double 的精度损失。在 staging path 中，每层都有一次转换，累计误差约 `N_layers × ε_float`。在 GPU-resident path 中，只在输入/输出各转换一次，误差应该更小。但如果 `Scalar = float`，此问题不存在。

**影响**：此 bug 不会导致 10% 准确率的大幅下降，但会引入微小的数值差异。主要 bug 仍是 Bug 1 和 Bug 2。

---

## 🛠️ 修复方案

### Fix 1：引入 `GpuFence` 封装队列级同步

**新增**：`include/neuralnet.cpp/backend/gpu_sync.hpp`

```cpp
// GPU 操作句柄：封装 fence，支持 wait + cleanup
class GpuOpHandle {
    VkDevice device_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> cmd_buffers_;
    std::vector<VkDescriptorSet> desc_sets_;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
public:
    GpuOpHandle() = default;
    ~GpuOpHandle() { cleanup(); }
    
    // 提交后等待 GPU 完成
    void wait_and_cleanup() {
        if (fence_ != VK_NULL_HANDLE) {
            vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
            vkDestroyFences(device_, 1, &fence_);
            fence_ = VK_NULL_HANDLE;
        }
        cleanup();
    }
    
    void cleanup() {
        if (!cmd_buffers_.empty())
            vkFreeCommandBuffers(device_, cmd_pool_, cmd_buffers_.size(), cmd_buffers_.data());
        if (!desc_sets_.empty())
            vkFreeDescriptorSets(device_, desc_pool_, desc_sets_.size(), desc_sets_.data());
        cmd_buffers_.clear();
        desc_sets_.clear();
    }
};
```

### Fix 2：为 `matmul_gpu` / `elementwise_gpu` 添加 fence 同步

**修改**：`vk_backend.hpp` — `matmul_gpu()` 和 `elementwise_gpu()`

```cpp
// 修改前（无 fence）：
vkQueueSubmit(device_.compute_queue(), 1, &submit_info, VK_NULL_HANDLE);
return C;

// 修改后（有 fence）：
VkFenceCreateInfo fence_info{};
fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
VkFence fence = VK_NULL_HANDLE;
vkCreateFence(device_.device(), &fence_info, nullptr, &fence);
vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence);

// 方案 A：阻塞等待（简单，正确性优先）
vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, UINT64_MAX);
vkDestroyFence(device_.device(), 1, &fence);
vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
return C;

// 方案 B：返回 handle（高性能，延迟等待）
return GpuOpHandle{device_.device(), fence, {cmd}, {desc_set}, ...};
// 下一层的 matmul_gpu 在录制前先 wait 上一个 handle
```

**推荐方案 A（阻塞等待）**，理由：
- 每层计算耗时远大于 fence wait 的开销（GPU 上千个核心并行）
- 简单、正确、易于调试
- 后续可优化为 batch submit（多层 command buffer 一次性提交 + 单次 fence wait）

### Fix 3：移除 `pending_ops_`，改为每个操作独立管理生命周期

**修改**：`vk_backend.hpp`

```cpp
// 删除：
struct PendingGpuOps { ... };
PendingGpuOps pending_ops_;
void flush_pending_ops() { ... }

// 每个 GpuTensor 操作完成后立即清理资源
// matmul_gpu / elementwise_gpu 中：
vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, UINT64_MAX);
vkDestroyFence(device_.device(), 1, &fence);
vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
```

### Fix 4：添加权重缓存失效机制

**修改**：`layer.hpp` — `Linear` 类

```cpp
class Linear final : public Layer {
    std::optional<GpuTensor> gpu_weights_;
    std::optional<GpuTensor> gpu_bias_;
    bool gpu_weights_dirty_ = true;  // 新增：权重是否需要重新上传

    // 当 optimizer.step() 修改 W_ 或 b_ 时，标记 dirty
    // 方案：在 parameters() 返回的 reference_wrapper 中包装一个 dirty flag
    // 或者更简单：每次 forward_gpu 都检查
    
    Result<GpuTensor> forward_gpu(const GpuTensor& input, GpuBackend& backend) override
    {
        // 简单方案：每次都重新上传（有性能开销但正确）
        // 优化方案：用 hash 或 version 号判断是否需要更新
        gpu_weights_.reset();
        gpu_bias_.reset();
        auto w_res = GpuTensor::from_matrix(W_, backend);
        if (!w_res) return std::unexpected(w_res.error());
        gpu_weights_ = std::move(*w_res);
        // ...
    }
};
```

**推荐方案**：在 `Layer` 基类中添加 `invalidate_gpu_cache()` 虚函数，由 `optimizer.step()` 在参数更新后调用。

### Fix 5：记录精度转换统计（可选）

**修改**：`vk_backend.hpp`

```cpp
// 在 convert_scalar_to_float / convert_float_to_scalar 中添加统计
struct PrecisionStats {
    std::atomic<uint64_t> conversions{0};
    std::atomic<uint64_t> max_ulp_error{0};  // 最大 ULP 误差
};
inline PrecisionStats g_precision_stats;
```

---

## 📅 分阶段实施计划

### Phase 1：移植 GPU 后端到 `rewrite` 分支（基础）

**前置条件**：安装 Vulkan SDK 1.4.x

| 任务 | 文件 | 说明 |
|------|------|------|
| 1.1 创建 backend 目录 | `include/neuralnet.cpp/backend/` | 从 `8456fe7` 提取 `vk_backend.hpp`、`gpu_tensor_impl.hpp` |
| 1.2 创建 shaders 目录 | `shaders/` | 从 `8456fe7` 提取 `matmul.comp`、`matmul_tiled.comp`、`elementwise.comp` |
| 1.3 创建 cmake 脚本 | `cmake/embed_spirv.cmake` | SPIR-V → C++ 头文件嵌入 |
| 1.4 修改 CMakeLists.txt | `CMakeLists.txt` | 添加 Vulkan 可选依赖、shader 编译管线、`NN_HAS_VULKAN` 定义 |
| 1.5 修改 config.hpp | `include/neuralnet.cpp/config.hpp` | 添加 `GPU_THRESHOLD`、`gpu_enabled`、`gpu_matmul_count`、`cpu_matmul_count` |
| 1.6 适配文件路径 | 所有 backend 文件 | 将 `../nn_config.hpp` 改为 `../config.hpp`，适配扁平化 include 结构 |
| 1.7 编译验证 | — | `cmake -B build && cmake --build build --target gpu_test` |

### Phase 2：修复核心同步 Bug（关键）

| 任务 | 文件 | 修复 |
|------|------|------|
| 2.1 修复 `matmul_gpu` 同步 | `vk_backend.hpp` | 提交 command buffer 后添加 fence wait + 资源清理 |
| 2.2 修复 `elementwise_gpu` 同步 | `vk_backend.hpp` | 同上 |
| 2.3 删除 `pending_ops_` | `vk_backend.hpp` | 移除有问题的延迟清理机制 |
| 2.4 单层正确性测试 | `gpu_test.cpp` | 验证单层 matmul + elementwise 结果正确 |

### Phase 3：集成到 Matrix 类（双轨制）

| 任务 | 文件 | 说明 |
|------|------|------|
| 3.1 添加 GPU dispatch 到 `multiply_to` | `algebra_matrix.hpp` | `#ifdef NN_HAS_VULKAN` 分支调用 `matmul_direct()` |
| 3.2 添加 GPU dispatch 到 `apply_relu_inplace` | `algebra_matrix.hpp` | 调用 `elementwise_direct()` |
| 3.3 添加 GPU dispatch 到 `apply_gelu_inplace` | `algebra_matrix.hpp` | 调用 `elementwise_direct()` |
| 3.4 Staging path 验证 | `gpu_test.cpp` | 验证 staging path（自动 fallback）正确 |

### Phase 4：启用 GPU-Resident 流水线

| 任务 | 文件 | 说明 |
|------|------|------|
| 4.1 添加 `forward_gpu` 到 Layer | `compute_layer.hpp` | `Linear`、`ReLU`、`GeLU`、`LayerNorm` 等 |
| 4.2 添加 `forward_gpu` 到 Model | `model_container.hpp` | GPU upload → 逐层 `forward_gpu` → download |
| 4.3 添加权重缓存失效 | `compute_layer.hpp` | `optimizer.step()` 后重置 `gpu_weights_` |
| 4.4 多层链式测试 | `gpu_test.cpp` | 2 层 → 5 层 → 10 层网络，对比 CPU/GPU 结果 |

### Phase 5：性能优化（可选）

| 任务 | 说明 |
|------|------|
| 5.1 Batch submit | 将多层 command buffer 一次性提交，单次 fence wait |
| 5.2 Double buffering | 使用两组 command buffer 交替录制/执行 |
| 5.3 Async compute | 使用多个 compute queue 并行执行独立操作 |
| 5.4 LayerNorm / Softmax shader | 为归约操作编写专用 shader |

---

## 🧪 测试策略

### 单元测试

| 测试 | 输入 | 验证 |
|------|------|------|
| `test_matmul_single` | 16×16 方阵 | `max_abs_err < 1e-5` (float) |
| `test_matmul_large` | 8192×8192 | `max_abs_err < 1e-3` (float) |
| `test_elementwise_relu` | 1000 元素 | 逐元素比较 |
| `test_elementwise_gelu` | 1000 元素 | `max_abs_err < 1e-6` |
| `test_elementwise_biasadd` | 512×64 | 行广播正确性 |

### 集成测试

| 测试 | 网络 | 验证 |
|------|------|------|
| `test_2layer_chain` | Linear→ReLU→Linear | CPU/GPU 输出一致 |
| `test_5layer_mlp` | 784→512→256→128→64→10 | MNIST 准确率 ≥ 93% |
| `test_gpt_inference` | GPT-2 small | 生成文本语义一致 |
| `test_training_loop` | MNIST 10 epochs | 收敛曲线一致 |

### 性能基准

| 指标 | Staging Path | GPU-Resident | 目标 |
|------|-------------|-------------|------|
| 单层 matmul (4096×4096) | 5ms | 0.5ms | 10× 加速 |
| 5 层 MLP 前向 | 25ms | 3ms | 8× 加速 |
| PCIe 传输量 (5 层) | 5×2×数据量 | 1×2×数据量 | 5× 减少 |

---

## 📎 参考资料

- [Vulkan Compute Pipeline 规范](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html)
- [Vulkan 同步机制详解](https://www.khronos.org/blog/understanding-vulkan-synchronization)
- [SPIR-V Storage Buffer 规范](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
- 仓库记忆：`/memories/repo/gpu-backend.md`（v4-v6.1 设计历史）
