# 🚀 Vulkan 后端性能优化方案

> 基于 ggml-vulkan (llama.cpp) 架构分析，针对当前 `vk_backend.hpp` 的性能优化方案。

---

## ✅ 已完成的优化（2026-07-17）

| 优化 | 状态 | 说明 |
|------|------|------|
| **分层合规** | ✅ | 删除 `forward_gpu`，Layer 只调 Matrix 语义 API |
| **Elementwise GPU dispatch** | ✅ | `apply_relu_inplace` / `apply_gelu_inplace` 内部调 `elementwise_direct` |
| **Elementwise LRU cache** | ✅ | `ElemKey` → 预录制命令缓冲，与 matmul 同级优化 |
| **Model 简化** | ✅ | 删除 GPU 路由代码，Model 只调 Layer::forward |
| **layer_base.hpp 清理** | ✅ | 删除 dead code 中的 forward_gpu |

### 调用链（优化后）

```
Model::forward(input)
  → Layer::forward(input)                    // L3 → L2
    → Matrix::multiply_to(result, weights)   // L2 → L1, 内部 GPU dispatch
    → Matrix::apply_relu_inplace()           // L2 → L1, 内部 GPU dispatch
  → Layer::forward(input)
    → ...
```

**每个操作的 GPU 路径：**
```
Matrix::multiply_to()
  → GpuBackend::matmul_direct()              // L1 → L0
    → staging upload → pre-recorded cmd → submit+wait → staging download

Matrix::apply_relu_inplace()
  → GpuBackend::elementwise_direct()         // L1 → L0
    → staging upload → pre-recorded cmd → submit+wait → staging download
```

---

## 📊 性能问题分析

### 问题总览

| 严重度 | 问题 | 影响 | ggml 做法 |
|--------|------|------|-----------|
| 🔴 P0 | 逐层独立 QueueSubmit | CPU 开销爆炸 | 合并为单次 Submit |
| 🔴 P0 | 逐层分配新 Command Buffer | 分配 + 录制开销 | 复用 Command Buffer |
| 🟠 P1 | 逐操作分配新 Descriptor Set | 频繁分配/释放 | 复用 Descriptor Set |
| 🟠 P1 | 逐层分配新 GpuTensor/Buffer | 显存碎片化 | 预分配 double-buffer |
| 🟡 P2 | Elementwise 每次 alloc 输出 buffer | 不必要的分配 | In-place 或预分配 |
| 🟡 P2 | LayerNorm/Softmax 未 GPU 化 | CPU-GPU 来回传输 | GPU 端融合 |
| 🔵 P3 | 无 Cooperative Matrix 支持 | 未利用 Tensor Core | 多路径硬件适配 |
| 🔵 P3 | Tile/Workgroup 硬编码 | 无法适配不同 GPU | Specialization Constants |

---

### P0-1：逐层独立 QueueSubmit（最严重的性能瓶颈）

**现状：** 一次前向传播的调用链：

```
Model::forward()
  → upload_blocking()                    # vkQueueSubmit × 1
  → Linear::forward_gpu()
      → matmul_gpu()                     # vkQueueSubmit × 1
      → elementwise_gpu(BiasAdd)         # vkQueueSubmit × 1
  → GeLU::forward_gpu()
      → elementwise_gpu(GeLU)            # vkQueueSubmit × 1
  → Linear::forward_gpu()
      → matmul_gpu()                     # vkQueueSubmit × 1
      → elementwise_gpu(BiasAdd)         # vkQueueSubmit × 1
  → GeLU::forward_gpu()
      → elementwise_gpu(GeLU)            # vkQueueSubmit × 1
  → ... (重复 5 层)
  → download_blocking()                  # vkQueueSubmit × 1
```

**一次 MNIST 前向传播 = ~16 次 vkQueueSubmit**

每次 `vkQueueSubmit` 有固定的 CPU 开销：
- 驱动命令验证和解析（~10-50μs / 次）
- 队列锁竞争（`queue_mutex_`）
- Fence 信号/等待

**ggml-vulkan 做法：** 单次前向传播中所有操作录制到**同一个 Command Buffer**，最后一次性 `vkQueueSubmit`：

```cpp
// ggml-vulkan: 所有 op 录制到一个 command buffer
auto ctx = ggml_vk_start_recording(device);
for (auto& op : graph->ops) {
    ggml_vk_dispatch_pipeline(ctx, pipeline, ...);  // 只录制，不提交
}
ggml_vk_submit(device, ctx);  // 一次性提交
```

---

### P0-2：逐层分配新 Command Buffer

**现状：** `matmul_gpu()` 和 `elementwise_gpu()` 每次调用：

```cpp
VkCommandBufferAllocateInfo cmd_alloc{};
vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd);  // 分配
vkBeginCommandBuffer(cmd, &begin_info);                         // 开始录制
// ... 录制命令 ...
vkEndCommandBuffer(cmd);                                        // 结束录制
vkQueueSubmit(..., cmd, ...);                                   // 提交
// cmd 留在 pending_ops_ 中，等 flush 时才释放
```

**问题：**
- `vkAllocateCommandBuffers` 有驱动开销
- 命令缓冲区积压在 `pending_ops_` 中，直到 `download_blocking()` 才释放
- 5 层网络积压 ~15 个 Command Buffer + Descriptor Set

**ggml-vulkan 做法：** 维护 Command Buffer 池，fence 完成后回收复用：

```cpp
struct vk_command_pool {
    vk::CommandPool pool;
    std::deque<vk_command_buffer> buffers;  // 可复用的 buffer 池
};

struct vk_command_buffer {
    vk::CommandBuffer buffer;
    vk::Fence fence;
    bool submitted;
};
```

---

### P1-1：逐操作分配新 Descriptor Set

**现状：** `matmul_gpu()` 和 `elementwise_gpu()` 每次调用都分配新 Descriptor Set：

```cpp
vkAllocateDescriptorSets(device_.device(), &desc_alloc, &desc_set);
vkUpdateDescriptorSets(device_.device(), 3, writes, 0, nullptr);
// desc_set 积压在 pending_ops_ 中
```

**问题：** 5 层网络 × ~3 ops = ~15 个 Descriptor Set 同时活跃

**ggml-vulkan 做法：** Pipeline 绑定的 Descriptor Set 按尺寸缓存复用。

---

### P1-2：逐层分配新 GpuTensor/Buffer

**现状：** `matmul_gpu()` 和 `elementwise_gpu()` 每次创建输出：

```cpp
auto C_res = GpuTensor::create_empty(M, N, *this);  // 每次分配新 buffer
```

**问题：** 5 层网络产生 ~15 个中间 GpuTensor，每个对应一个 VkBuffer + VkDeviceMemory 分配

**ggml-vulkan 做法：** 使用预分配的 `vk_subbuffer`，张量是大 buffer 的子区域：

```cpp
struct vk_subbuffer {
    vk_buffer_ref buffer;  // 指向父 buffer
    uint64_t offset;       // 子区域偏移
    uint64_t size;         // 子区域大小
};
```

---

### P2-1：Pipeline Barrier 过度使用

**现状：** `matmul_gpu()` 中：

```cpp
// 输入 barrier
vkCmdPipelineBarrier(cmd, TRANSFER | COMPUTE, COMPUTE, ...);

// dispatch
vkCmdDispatch(cmd, ...);

// 输出 barrier
vkCmdPipelineBarrier(cmd, COMPUTE, TRANSFER | COMPUTE, ...);
```

**问题：** 在非阻塞流水线中，跨 dispatch 的 barrier 不应使用 `vkCmdPipelineBarrier`（host-side 同步），而应使用 `vkCmdPipelineBarrier` 但需要精简 src/dst stage。

**ggml-vulkan 做法：** 只在真正需要数据依赖的地方加 barrier，同一 command buffer 内连续 compute dispatch 之间不需要 TRANSFER stage。

---

### P2-2：Elementwise 产生不必要的额外 Buffer

**现状：** `elementwise_gpu()` 为 ReLU/GeLU 每次分配新输出 buffer：

```cpp
auto out_res = GpuTensor::create_empty(primary.rows(), primary.cols(), *this);
// 新 buffer → 旧 buffer 立刻变成垃圾
```

**问题：** 5 层 × 2 次 elementwise = 10 个中间 buffer，其中大部分在下一步立刻被丢弃

**ggml-vulkan 做法：** 支持 in-place 操作，输出写回原 buffer：

```glsl
// in-place: binding 0 既是输入也是输出
buf_out[idx] = max(buf_a[idx], 0.0);  // 写回 binding 0
```

---

### P3-1：LayerNorm/Softmax 未 GPU 化

**现状：** `LayerNorm` 和 `Softmax` 在 CPU 端执行，导致：

```
GPU: matmul → download → CPU: LayerNorm → upload → GPU: GeLU → ...
```

每经过一个 LayerNorm 就多一次 PCIe 往返（~1-5ms）。

**ggml-vulkan 做法：** 所有操作都在 GPU 端执行，包括 RMSNorm、Softmax、RoPE。

---

### P3-2：无硬件 Tensor Core 支持

**现状：** `matmul_tiled.comp` 使用纯标量 FMA：

```glsl
for (uint k = 0u; k < BK; ++k) {
    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 4u; ++j)
            acc[i][j] += a_vals[i] * b_vals[j];  // 标量 FMA
}
```

**ggml-vulkan 做法：** 检测硬件能力，选择最优路径：
- 无扩展 → 标量 tiled（与你当前一致）
- `GL_KHR_cooperative_matrix` → WMMA/Tensor Core
- `GL_NV_cooperative_matrix2` → NVIDIA 专用优化

---

## 🔧 修改方案

### Phase 1：Command Buffer 批量录制（P0，最高优先级）

**目标：** 一次前向传播只用 1 次 `vkQueueSubmit`

**修改文件：** `vk_backend.hpp`

**核心改动：**

1. 新增 `GpuCommandContext` — 批量录制上下文：

```cpp
class GpuCommandContext {
private:
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> desc_sets_;      // 本批次使用的 desc sets
    std::vector<VkCommandBuffer> child_cmds_;     // 本批次的子命令
    bool recording_ = false;

public:
    // 开始录制（分配一个 command buffer）
    Result<void> begin(VkDevice device, VkCommandPool pool);

    // 录制 matmul 命令（不提交）
    Result<void> record_matmul(
        const GpuTensor& A, const GpuTensor& B, GpuTensor& C,
        const VulkanPipeline& pipeline, VkDescriptorPool desc_pool);

    // 录制 elementwise 命令（不提交）
    Result<void> record_elementwise(
        const GpuTensor& primary, const GpuTensor* secondary,
        GpuTensor& output, uint32_t op,
        const VulkanPipeline& pipeline, VkDescriptorPool desc_pool,
        uint32_t rows = 0, uint32_t cols = 0);

    // 结束录制并提交（一次 vkQueueSubmit）
    Result<void> submit(VkQueue queue, VkFence fence);

    // 清理资源
    void cleanup(VkDevice device, VkCommandPool cmd_pool, VkDescriptorPool desc_pool);
};
```

2. `GpuBackend` 新增批量 API：

```cpp
class GpuBackend {
public:
    // 获取当前录制上下文（懒创建）
    GpuCommandContext& begin_context();

    // 提交当前上下文
    Result<void> submit_context(VkFence fence = VK_NULL_HANDLE);

    // 同步等待（在 download 时调用）
    Result<void> wait_and_flush();
};
```

3. Layer 调用改为录制模式：

```cpp
// 旧：每次调用都提交
auto c = backend.matmul_gpu(a, b);   // vkQueueSubmit
auto d = backend.elementwise_gpu(c);  // vkQueueSubmit

// 新：只录制，最后统一提交
auto& ctx = backend.begin_context();
ctx.record_matmul(a, b, c, pipeline, pool);
ctx.record_elementwise(c, nullptr, d, 0, pipeline, pool);
backend.submit_context();  // 一次 vkQueueSubmit
```

**预期收益：** ~16 次 Submit → 1 次 Submit，CPU 开销减少 90%+

---

### Phase 2：Command Buffer 池化（P0）

**目标：** 消除频繁的 `vkAllocateCommandBuffers` / `vkFreeCommandBuffers`

**修改文件：** `vk_backend.hpp`

**核心改动：**

```cpp
class CommandBufferPool {
private:
    VkDevice device_;
    VkCommandPool pool_;
    std::vector<VkCommandBuffer> available_;
    std::vector<VkCommandBuffer> in_flight_;
    std::vector<VkFence> fences_;

public:
    // 获取可用的 command buffer（fence 完成后回收）
    VkCommandBuffer acquire();

    // 标记为 in-flight
    void mark_submitted(VkCommandBuffer cmd, VkFence fence);

    // 回收已完成的 command buffers
    void reclaim_completed();
};
```

**预期收益：** 消除 ~15 次/前向传播的 `vkAllocateCommandBuffers` 开销

---

### Phase 3：中间 Tensor 预分配（P1）

**目标：** 消除逐层 `GpuBuffer::create_device_local` 的开销

**修改文件：** `vk_backend.hpp`, `gpu_tensor_impl.hpp`

**核心改动：**

1. 双缓冲策略（类似 GGML 的 `vk_subbuffer`）：

```cpp
class GpuTensorArena {
private:
    // 预分配两个大 buffer，交替使用
    std::shared_ptr<GpuBuffer> buffers_[2];
    std::size_t write_idx_ = 0;

public:
    // 从 arena 中分配一个 tensor（零分配开销）
    GpuTensor allocate(std::size_t rows, std::size_t cols);

    // 交换读写 buffer（每层结束后调用）
    void swap();
};
```

2. 对于固定大小的网络（如 MNIST），可以预计算每层的 tensor 大小，一次性分配。

**预期收益：** 消除 ~15 次/前向传播的显存分配开销

---

### Phase 4：Elementwise In-place 优化（P2）

**目标：** ReLU/GeLU 直接写回输入 buffer，避免额外分配

**修改文件：** `vk_backend.hpp`, `elementwise.comp`

**核心改动：**

1. Shader 支持 in-place（binding 0 = 输入，binding 2 = 输出，可以是同一 buffer）：

```glsl
// 已支持（binding 0 和 binding 2 可以是同一个 buffer）
buf_out[idx] = max(buf_a[idx], 0.0);
```

2. 后端 API 支持 in-place：

```cpp
// In-place 版本：输出写回 input tensor
Result<void> elementwise_gpu_inplace(
    GpuTensor& inout, uint32_t op, uint32_t rows = 0, uint32_t cols = 0);
```

**预期收益：** 消除 ~10 次/前向传播的 buffer 分配

---

### Phase 5：Pipeline Barrier 优化（P2）

**目标：** 精简 barrier，减少不必要的同步

**修改文件：** `vk_backend.hpp`

**核心改动：**

```cpp
// 旧：每次 dispatch 前后都有 barrier
vkCmdPipelineBarrier(cmd, TRANSFER | COMPUTE, COMPUTE, ...);  // 输入 barrier
vkCmdDispatch(cmd, ...);
vkCmdPipelineBarrier(cmd, COMPUTE, TRANSFER | COMPUTE, ...);  // 输出 barrier

// 新：连续 compute dispatch 之间不需要 barrier（Vulkan 保证 compute-to-compute 顺序）
vkCmdDispatch(cmd, matmul...);
vkCmdDispatch(cmd, biasadd...);   // 无 barrier
vkCmdDispatch(cmd, activation...); // 无 barrier
// 只在最后一次 dispatch 后加 barrier（供 download 或下一层使用）
```

**关键规则：**
- 同一 command buffer 内，连续 compute dispatch 之间**不需要** barrier（Vulkan spec 保证顺序执行）
- 只在数据被不同 pipeline stage 访问时才需要 barrier
- 只在最后一次 dispatch 后、download 之前加输出 barrier

**预期收益：** 减少 ~10 次/前向传播的 barrier 开销

---

### Phase 6：LayerNorm/Softmax GPU 化（P3，长期）

**目标：** 消除 LayerNorm 的 CPU-GPU 来回传输

**修改文件：** 新增 `shaders/layernorm.comp`, `vk_backend.hpp`, `layer.hpp`

**核心改动：**

1. 新增 LayerNorm compute shader：

```glsl
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Input { float input[]; };
layout(std430, binding = 1) readonly buffer Gamma { float gamma[]; };
layout(std430, binding = 2) readonly buffer Beta { float beta[]; };
layout(std430, binding = 3) writeonly buffer Output { float output[]; };

layout(push_constant) uniform PC {
    uint N;      // 特征维度
    uint batch;  // batch size
    float eps;
};

shared float shared_sum[256];
shared float shared_sq_sum[256];

void main() {
    uint f = gl_WorkGroupID.x;    // 特征索引
    uint tid = gl_LocalInvocationID.x;

    // 1. 计算均值和方差（并行归约）
    float local_sum = 0.0;
    float local_sq_sum = 0.0;
    for (uint b = tid; b < batch; b += 256) {
        float v = input[f * batch + b];
        local_sum += v;
        local_sq_sum += v * v;
    }
    shared_sum[tid] = local_sum;
    shared_sq_sum[tid] = local_sq_sum;
    barrier();

    // 归约
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        barrier();
    }

    float mean = shared_sum[0] / float(batch);
    float var = shared_sq_sum[0] / float(batch) - mean * mean;
    float inv_std = 1.0 / sqrt(var + eps);

    // 2. 归一化 + 仿射
    for (uint b = tid; b < batch; b += 256) {
        float v = input[f * batch + b];
        output[f * batch + b] = (v - mean) * inv_std * gamma[f] + beta[f];
    }
}
```

2. 后端 API：

```cpp
Result<GpuTensor> layernorm_gpu(
    const GpuTensor& input, const GpuTensor& gamma, const GpuTensor& beta,
    float eps);
```

3. Layer 层改用 GPU：

```cpp
Result<GpuTensor> LayerNorm::forward_gpu(const GpuTensor& input, GpuBackend& backend) {
    return backend.layernorm_gpu(input, gpu_gamma_, gpu_beta_, epsilon_);
}
```

**预期收益：** 消除 LayerNorm 处的 CPU-GPU 往返（~1-5ms/层）

---

### Phase 7：Specialization Constants（P3，长期）

**目标：** Tile/Workgroup 大小可配置，适配不同 GPU

**修改文件：** `matmul_tiled.comp`, `vk_backend.hpp`

**核心改动：**

1. Shader 使用 Specialization Constants：

```glsl
layout(constant_id = 0) const uint BM = 64u;
layout(constant_id = 1) const uint BN = 64u;
layout(constant_id = 2) const uint BK = 16u;
```

2. 运行时根据 GPU 能力选择最优配置：

```cpp
struct GpuCaps {
    uint32_t max_workgroup_size;
    uint32_t max_shared_memory;
    uint32_t subgroup_size;
    bool has_cooperative_matrix;
};

VkSpecializationInfo select_tile_config(const GpuCaps& caps, uint32_t M, uint32_t N, uint32_t K);
```

---

### Phase 8：Cooperative Matrix 支持（P3，长期）

**目标：** 利用硬件 Tensor Core 加速矩阵乘法

**修改文件：** 新增 `shaders/matmul_coopmat.comp`, `vk_backend.hpp`

**核心改动：**

1. 检测硬件能力：

```cpp
bool has_cooperative_matrix(VkPhysicalDevice dev) {
    // 检查 VK_KHR_cooperative_matrix 扩展
    uint32_t ext_count;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &ext_count, exts.data());
    for (auto& ext : exts) {
        if (strcmp(ext.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0)
            return true;
    }
    return false;
}
```

2. 新增 CoopMat shader：

```glsl
#version 450
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require

void main() {
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> A;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> B;
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> C;

    // 加载 tile 到 cooperative matrix
    coopMatLoad(A, buf_a, ...);
    coopMatLoad(B, buf_b, ...);

    // 硬件加速矩阵乘累加
    C = coopMatMulAdd(A, B, C);

    // 写回结果
    coopMatStore(C, buf_c, ...);
}
```

3. Pipeline 选择：

```cpp
auto& pipeline = has_coopmat ? matmul_coopmat_pipeline_ :
                  has_tiled ? matmul_tiled_pipeline_ :
                  matmul_pipeline_;
```

---

## 📋 实施优先级

```
Phase 1 ─ Command Buffer 批量录制 (P0)  ─── 影响最大，改动中等
    │
Phase 2 ─ Command Buffer 池化 (P0)      ─── 影响大，改动小
    │
Phase 3 ─ 中间 Tensor 预分配 (P1)       ─── 影响中等，改动中等
    │
Phase 4 ─ Elementwise In-place (P2)     ─── 影响中等，改动小
    │
Phase 5 ─ Pipeline Barrier 优化 (P2)    ─── 影响小，改动小
    │
Phase 6 ─ LayerNorm GPU 化 (P3)         ─── 影响大，改动大
    │
Phase 7 ─ Specialization Constants (P3) ─── 影响中等，改动中等
    │
Phase 8 ─ Cooperative Matrix (P3)       ─── 影响大，改动大
```

### 建议实施顺序

**第一批（性能收益最大）：**
- Phase 1 + Phase 2 同时实施（Command Buffer 批量录制 + 池化）
- 预期：前向传播 CPU 开销减少 80-90%

**第二批（消除内存压力）：**
- Phase 3 + Phase 4 同时实施（Tensor 预分配 + In-place）
- 预期：显存分配次数减少 90%

**第三批（补全 GPU 覆盖）：**
- Phase 6（LayerNorm GPU 化）
- 预期：消除 LayerNorm 的 PCIe 往返

**第四批（硬件优化）：**
- Phase 7 + Phase 8（Spec Constants + CoopMat）
- 预期：矩阵乘法吞吐提升 2-4×

---

## ⚠️ 风险与注意事项

1. **Phase 1 需要重构 Layer 接口**：当前 `forward_gpu` 返回 `Result<GpuTensor>`，改为录制模式需要改变调用语义
2. **Phase 3 需要预知网络结构**：Arena 预分配需要知道每层的 tensor 大小
3. **Phase 6 需要新的 shader**：LayerNorm 的并行归约需要仔细处理 shared memory
4. **Phase 8 需要硬件检测**：CoopMat 扩展在不同 GPU 上支持程度不同

---

*文档生成日期：2026-07-17*
*参考：ggml-vulkan (llama.cpp) 架构分析*
