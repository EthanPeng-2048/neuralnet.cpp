# g02 引擎核心（engine/tensor/gpu_engine/cuda_engine）代码审查

## 模块概览

**文件清单与职责：**
- `compute_engine.hpp`（413 行）：ComputeEngine 纯虚接口定义，含所有原语签名、枚举（UnaryOp/BinaryOp/CompareOp/ReduceOp）、批处理控制、表达式录制、offload API
- `compute_tensor.hpp`（236 行）：Tensor 统一跨设备张量，shared_ptr 内部存储，CPU/GPU/CUDA 三路互斥，零拷贝 reshape，图 IR virtual_tag
- `compute_gpu_engine.hpp`（999 行）：GpuEngine final 实现，封装 GpuBackend，IR-C 图 IR 融合执行（begin_expr/end_expr），AOT 闭合世界 eval_expr/eval_expr_reduce，P2-12 图级缓存跨 step 复用
- `compute_cuda_engine.hpp`（464 行）：CudaEngine final 实现，**已停用死代码**（`#ifdef NN_HAS_CUDA` 永假），所有原语返回"未实现"错误

**整体质量评估：**
GpuEngine 实现质量高——真原地操作（add_inplace/scale_inplace/broadcast_*_inplace 直接写回 A 的 buffer）、batch 录制生命周期安全（GpuBuffer 析构 defer_buffer_destroy + end_batch flush_pending_destroys）、IR-C 图 IR 隔离（thread_local recording_graph_owner + node_outputs）。接口签名与 CpuEngine/GpuEngine 完全一致（26 个纯虚原语全部覆盖）。铁律核对：✅ 无 throw/try/catch，✅ 无 new/delete，✅ 引擎只提供原语不含算法，✅ batch-major 布局在扫描原语注释中明确约定。

## 发现

### P0

无。

### P1

**P1-1 `compute_cuda_engine.hpp:211` — `add_inplace` 函数签名缺失（死代码编译错误）**

```cpp
// line 209:     }  ← batched_matmul 结尾
// line 210:
// line 211:     {  ← add_inplace 函数体开始，但签名被删除
// line 212:         if (A.rows() != B.rows() || A.cols() != B.cols())
// line 213:             return std::unexpected(Error{"add_inplace: shape mismatch"});
```

`add_inplace` 的 override 签名行（`[[nodiscard]] Result<void> add_inplace(Tensor& A, const Tensor& B) override`）缺失，函数体直接以 `{` 开始。当前因 `NN_HAS_CUDA` 永假不编译，但若有人尝试启用 CUDA 后端，此文件将无法通过编译。与 docs/17 指针审查中 6 处裸指针债对应——该文件整体应删除。

**修复建议：** 删除整个 `compute_cuda_engine.hpp`（及 `compute_cuda_backend.hpp`、`compute_tensor.hpp` 中 `#ifdef NN_HAS_CUDA` 分支），与 docs/17 路线图一致。若保留，至少补上缺失的 `add_inplace` 签名。

### P2

**P2-1 `compute_engine.hpp:148` / `compute_gpu_engine.hpp:127` — `create_offload_buffer` 参数名 `bytes` 实为 float 元素数**

```cpp
// compute_engine.hpp:148
[[nodiscard]] virtual Result<Tensor> create_offload_buffer(std::size_t /*bytes*/)
{
    return Tensor::cpu(1, 1);
}

// compute_gpu_engine.hpp:127
[[nodiscard]] Result<Tensor> create_offload_buffer(std::size_t bytes) override
{
    auto g = GpuTensor::create_host_visible_empty(1, bytes, backend_);
    // ...
}
```

参数名为 `bytes`，但 `create_host_visible_empty(1, bytes, ...)` 将其作为 float 元素数（第二个参数是 cols）。调用方 `compute_layer_gpt.hpp:165` 传入的 `total` 也是 float 元素数之和。语义内部一致，但参数名误导——传入 `1024` 实际分配 4KB 而非 1KB。

**修复建议：** 将参数名改为 `elem_count` 或 `num_floats`，注释标注"float 单位"。

**P2-2 `compute_gpu_engine.hpp:890-892` — 图级缓存 overflow 全量清空**

```cpp
// line 890:             if (gcache.size() >= fused::GRAPH_PLAN_CACHE_MAX)
// line 891:                 gcache.clear();
// line 892:             gcache.emplace(gkey, std::move(plans));
```

`GRAPH_PLAN_CACHE_MAX = 512`（`expr_graph.hpp:475`）。缓存满时全量清空，导致下一次每个 Layer 的首次 end_expr 重新执行融合分析（CPU 开销，非 GPU）。训练热路径会出现一次性延迟尖峰。

**修复建议：** 可接受（512 远大于典型 Layer 种类数，实际很少触发）。若需平滑，可改为 LRU 淘汰或双缓存轮替。

### P3

**P3-1 `compute_cuda_engine.hpp` — 死代码状态确认，建议删除**

`#ifdef NN_HAS_CUDA` 全文件包裹（line 8, 464）。`NN_HAS_CUDA` 在整个项目中无 `#define`（grep 确认 0 处定义）。文件内实现不完整（eval_expr 返回硬错误、scan 原语返回"未实现"、add_inplace 签名缺失）。与 `compute_tensor.hpp` 中 6 处 `#ifdef NN_HAS_CUDA` 分支、`nn.hpp` 中 1 处条件包含、`cli_engine_factory.hpp` 中 2 处条件路径共 10 处引用对应。

**修复建议：** 按 docs/17 路线图删除 `compute_cuda_engine.hpp`、`compute_cuda_backend.hpp`，清理 `compute_tensor.hpp` 和 `nn.hpp` 中的 `#ifdef NN_HAS_CUDA` 分支。`cli_engine_factory.hpp` 中 CUDA 路径改为 `#error` 提示。

**P3-2 `compute_tensor.hpp:194-227` — CPU reshape 复制数据（设计已知，非 bug）**

```cpp
// line 210-220
// CPU 的 Matrix 无"零拷贝视图"能力：reshape 只改 Tensor 元数据
// 会让 cpu_matrix() 仍持有旧形状，导致 add_inplace/scale_inplace
// 等按 Tensor 逻辑形状匹配、却按 Matrix 实际形状运算的调用维度
// 不一致而断言失败。这里复制数据到新形状的 Matrix
auto m = std::make_shared<Matrix>(new_rows, new_cols);
const auto src = cpu_data_->span();
auto dst = m->span();
std::copy(src.begin(), src.end(), dst.begin());
t.cpu_data_ = std::move(m);
```

CPU reshape 是深拷贝（非零拷贝），GPU 保持零拷贝共享 buffer。注释已充分解释原因（Matrix 无视图能力）。行为正确，但在 CPU 路径上 reshape 频繁调用会有性能开销。

**P3-3 `compute_gpu_engine.hpp:437-438` — add_inplace 防御性 CPU 路径**

```cpp
// line 436-438
// 原地模式下 A 的 buffer 已被更新；若 ensure_gpu 上传了新 Tensor（防御路径），替换 A
if (A.is_cpu())
    A = std::move(*a_gpu);
```

纯 GPU 架构下 A 应始终为 GPU Tensor，此 CPU 分支为防御性兜底。`scale_inplace`（line 456-457）、`axpy_inplace`（line 481-482）、`zero`（line 496-497）、`broadcast_row_inplace`（line 618-619）、`broadcast_col_inplace`（line 637-638）均有相同模式。逻辑正确，但增加了代码体积。可考虑统一为一个 `ensure_gpu_and_replace(Tensor& t, Result<Tensor> gpu)` 辅助方法。

## 已知项核对

| 已知项 | 状态 | 本组涉及文件 |
|--------|------|-------------|
| col_reduce 并行非逐字节一致 | ✅ 仍存在（已知未修） | 不涉及本组（CpuEngine col_reduce 用 Matrix::col_reduce） |
| VK_TIMEOUT 重试未实现 | ✅ 仍存在（end_batch line 997-1013 直接返回错误） | compute_vk_backend.hpp:997 |
| CpuEmitter 已删除 | ✅ 确认（grep 无 CpuEmitter 引用） | 不涉及本组 |
| GpuBuffer 析构 defer_buffer_destroy | ✅ 正确（compute_vk_backend.hpp:3098-3127） | 生命周期安全 |
| pending_destroys_ 延迟销毁 | ✅ 正确（end_batch line 1029 调用 flush_pending_destroys） | 生命周期安全 |
| IR-C 图 IR thread_local 隔离 | ✅ 正确（fused::recording_graph_owner thread_local） | compute_gpu_engine.hpp:181 |
| 图级缓存 thread_local | ✅ 正确（fused::graph_plan_cache thread_local） | compute_gpu_engine.hpp:873 |
| AOT 闭合世界 eval_expr 未命中硬报错 | ✅ 正确（line 800-802） | compute_gpu_engine.hpp |
| batch-major 列序约定 | ✅ 扫描原语注释明确（compute_engine.hpp:267-280） | 接口文档充分 |

## 待验证

1. **`compute_vk_backend.hpp:scan_prefix_outer_gpu` 的 dk≤64 限制**（line 1674）：GPU shader 有 dk≤64 硬限制，CPU 实现无此限制。若 Layer 传入 dk>64，CPU 路径正常但 GPU 路径硬报错。需确认上层 Layer（compute_layer_rapt.hpp）是否在构造时校验 dk≤64。
2. **`compute_gpu_engine.hpp:execute_fused_graph` 的 `gcache.find(gkey)` 碰撞风险**：FNV-1a 64-bit 哈希碰撞概率极低（512 条目下 birthday bound ≈ 2^28），但理论上存在。碰撞会导致错误的 kernel 计划被复用。建议：可加 spec 指针二次验证（成本可忽略）。
3. **`compute_cuda_engine.hpp:46` `create_tensor` 失败返回空 Tensor**：接口签名 `Tensor create_tensor(...)` 非 `Result<Tensor>`，分配失败返回 `Tensor()`（空/invalid）。调用方需检查 `valid()`，但接口未强制。需确认所有调用方是否检查。
