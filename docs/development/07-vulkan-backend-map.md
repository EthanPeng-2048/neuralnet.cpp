# Vulkan 后端结构地图（P2-7）

> 面向"第一次读后端代码"的人：先建立调用链与所有权模型，再动手改。
> 本文只描述**当前代码**与**已确定的重构方向**；每一项拆分的落地记录进
> `docs/history/history.md`。

## 1. 为什么需要这张图

后端拆分过一轮（RV-08），但那是**物理拆分**：四个 `.inc` 片段被
`class GpuBackend { ... }` 在类体内按序 `#include`。文件数变多了，读者面对的
仍然是"一个类、四段成员定义、彼此共享全部私有状态"：

| 位置 | 行数 | 读者实际看到的内容 |
|---|---:|---|
| `backend/compute_vk_backend.hpp` | 1073 | 设备/张量声明 + 44 个 pipeline 成员 + 帧环/池/计时状态 |
| `backend/compute_vk_backend_destroy.inc` | 994 | 延迟销毁 + `initialize()`（约 750 行，内含 40 处 pipeline 创建样板） |
| `backend/compute_vk_backend_frame.inc` | 785 | 帧环、batch、上传下载、matmul |
| `backend/compute_vk_backend_pipelines.inc` | 673 | 扫描原语 + 描述符/屏障/solo fence 辅助 |
| `backend/compute_vk_backend_ops.inc` | 2688 | 15 个原语 dispatch + 融合执行 |
| `compute_gpu_engine.hpp` | 2610 | 引擎适配、cast 回退、融合图执行 |

`.inc` **不可单独编译、不可单独索引**：IDE 的 "go to definition" 与
`clangd` 只能把它们当成类体的一部分。这就是"文件拆了、类没拆"的代价。

## 2. 调用链（一次 GPU 原语）

```text
Layer / DSL
  │
  ├─ ComputeEngine::matmul()                 (NVI，compute_engine.hpp)
  │    └─ GpuEngine::matmul_impl()           (compute_gpu_engine.hpp)
  │         │  形状选择 / 精度回退 / K-split 决策
  │         └─ GpuBackend::matmul_gpu()      (compute_vk_backend_frame.inc)
  │              ├─ acquire_cmd()            → 帧环 or solo fence
  │              ├─ alloc_desc_set() + vkUpdateDescriptorSets
  │              ├─ record_input_barriers()
  │              ├─ ts_dispatch()            → vkCmdDispatch（+ 可选 timestamp）
  │              ├─ record_output_barrier()
  │              └─ submit_and_wait() / 留在帧内
  │
  └─ dsl::compute()                          (expr_dsl.hpp)
       └─ GpuEngine::eval_expr*_impl()
            └─ GpuBackend::run_fused_gpu()  (AOT 融合 kernel)
                 └─ fused_kernels_[key]     （一个 FusedKernel 记录）
```

读代码时的关键分界：

- **引擎层（`GpuEngine`）**回答"用哪个原语、什么精度、要不要回退"；
- **后端层（`GpuBackend`）**回答"Vulkan 怎么录、怎么交、什么时候等"。

当前这两件事在 `GpuEngine::*_impl` 里仍混有 `cast_if_f16` / `has_*_pipeline`
之类的后端能力判断；重构方向是把"能力查询"收进一个显式的 capability 对象，
引擎只读事实、不读 pipeline 句柄。

## 3. 所有权与生命周期

| 对象 | 拥有者 | 生命周期要点 |
|---|---|---|
| `VulkanDevice` | `GpuBackend::device_` | 进程级单例（`GpuBackend::instance()` 故意泄漏） |
| `MemoryPool` ×2（持久/瞬态） | `GpuBackend` | 所有 `GpuBuffer` 的子分配都来自其一 |
| `StagingRing` | `GpuBackend` | 阻塞上传/下载路径；时间线信号量可选 |
| `VulkanPipeline` ×44 | `GpuBackend` | `initialize()` 一次性创建；缺失 = 该变体不可用 |
| 帧环 `frames_[6]` | `GpuBackend` | cmd + fence + 描述符集按帧归属；环槽复用前等 fence |
| `GpuBuffer` | `shared_ptr`，被 `GpuTensorT` 共享 | 析构可能**延迟**到引用它的帧完成（`defer_buffer_destroy`） |
| `GpuTensorT<P>` | `shared_ptr`，被 `Tensor` 类型擦除持有 | 零拷贝 `with_shape` / 别名共享同一 `GpuBuffer` |

两条容易踩的规则：

1. **batch 录制期创建的张量不能立即销毁**。已录制的 descriptor set 仍引用
   buffer；`GpuBuffer::~GpuBuffer` 会向 `GpuBackend` 登记延迟销毁。
2. **写版本跟着 buffer 走，不跟着句柄走**。`GpuBuffer::write_epoch()` 是
   BitLinear 量化缓存的失效依据；任何新的直写路径必须补 `mark_written`。

## 4. 允许的依赖方向

```text
L0  core_config / core_errors / core_threadpool
     ▲
L1  algebra_matrix / algebra_span / algebra_ops
     ▲        （只应看到 GPU 存储类型，不应看到 Vulkan 后端）
L2  compute_engine ──► compute_gpu_engine ──► vk backend
     ▲                      │                    │
     │                      └──► gpu storage ◄───┘
L3+ model / domain / cli / app
```

**当前偏差**：`algebra_matrix.hpp` 为了 `GpuTensorT` 的声明直接
`#include "backend/compute_vk_backend.hpp"`，于是所有包含 `Tensor` 的 TU 都
被迫解析整个 Vulkan 后端（含 SPIR-V 嵌入头与融合注册表）。

目标：把 `GpuBuffer` / `GpuTensorT<P>` 下沉到 `backend/compute_gpu_storage.hpp`
（只依赖 `compute_memory_pool.hpp`），`algebra_matrix.hpp` 只包含该存储头；
需要完整 `GpuBackend` 的 `from_matrix` / `to_matrix` 实现移到后端层的
`compute_gpu_tensor_impl.hpp`，由 `compute_vk_backend.hpp` 末尾包含。

## 5. 后端内部的五块职责

改动前先判断属于哪一块，别再往 `GpuBackend` 里加"顺手"的成员：

| 职责 | 现有位置 | 目标类型 |
|---|---|---|
| 设备与能力 | `compute_vk_device.hpp` | `vk::Device`（已有 `VulkanDevice`） |
| Pipeline 创建/查找 | `destroy.inc` 的 `initialize()` | `vk::PipelineCatalog` + `PipelineId` |
| 描述符分配/绑定 | `pipelines.inc` + 各算子 | `vk::DescriptorAllocator` + `vk::DescriptorWriter` |
| 命令录制/提交 | `frame.inc` + `pipelines.inc` | `vk::CommandRecorder` + `vk::FrameRing` |
| 算子语义 | `ops.inc` + `frame.inc` 的 matmul | `gpu::ops::*`（接收 `OpContext`） |

### 5.1 已收敛与待收敛

已完成（P2-7 第 2–3 刀）：

- 44 个 pipeline 成员 + 40 个 `has_*_pipeline()` 手工转发 → `PipelineCatalog`
  + 一个 `has(PipelineId)` 查询入口；
- `initialize()` 中 40 段"取 SPIR-V → 判空 → 创建 → 移动" → 一张
  `PipelineSpec` 表 + 一个循环（`compute_vk_pipeline_specs.hpp`）；
- 融合 kernel 的 9 张并行 map → 一个 `FusedKernel` 记录（注册时一次写定）。

仍待收敛：

- 每个算子重复"分配描述符 → 写 4 个 binding → bind → push → dispatch →
  barrier → submit" → 目标是一个 `DispatchRequest` + `CommandRecorder`；
- `GpuBackend` 仍同时承担设备、存储、帧环、描述符、剖析、算子六件事 →
  目标是 `vk::` 协作者 + `gpu::ops::*` 薄算子。

## 6. 怎么加一个原语算子

1. `shaders/<name>.comp` 写 kernel；在 `compute_vk_backend.hpp` 文件头加
   `__has_include` 探测块，并把 token 加进 `NN_SPV_GETTER_LIST`（RV-08 单一
   注册表）。
2. `compute_engine.hpp` 加 op-level 接口（NVI 入口 + 默认 "unsupported"）。
3. `GpuEngine` 加 `*_impl`：只做形状/精度决策，把"怎么录"交给后端。
4. 后端加一个 `*_gpu`：绑定 → push constants → dispatch → barrier。
5. `shaders/scan_exprs` 不需要登记（op-level 原语不进 AOT 闭合世界）。

## 7. 怎么加一个 pipeline 变体

当前：在 44 个成员里加一个、在 `initialize()` 里抄一段创建、在 `has_*` 区抄
一个转发、在算子里加一个三元选择——**四处**。

目标：在 `PipelineSpec` 表加一行（id / 名字 / SPIR-V getter / binding 数 /
push 字节数 / 能力要求），算子侧用 `PipelineId` 选路——**两处**。

## 8. 验收口径

任何一步结构改动都按下面的门槛走（编译时间是观测量，不是验收项）：

- `ctest --test-dir build-gpu`（GPU + CPU）全绿；
- `NN_BIND_DEBUG=1 ctest` 全绿；
- 字节锚不变：`gpu_stability_probe --steps 20`、
  `layer_bench --op matmul,matmul_q,matmul_q_bw,scan_prefix,scan_suffix --f16 --hash`、
  `rapt_test --gpu` / `--f16 --gpu`、`scan_bitwise_test`；
- 融合产物不变：`expr_fuse_test`、`fusion_custom_layer_example`；
- 默认构建**不引入 LTO**；头文件布局不因"想少编译几个 TU"而妥协。

## 9. 阅读顺序建议

1. `compute_vk_device.hpp`：设备与 pipeline 的 RAII；
2. `compute_vk_backend_frame.inc` 的 `acquire_cmd` / `submit_and_wait`：同步模型；
3. `compute_vk_backend_ops.inc` 的 `elementwise_v2_gpu`：最短的完整算子；
4. `compute_vk_backend_ops.inc` 的 `run_fused_gpu`：融合元数据如何参与选路；
5. `compute_gpu_engine.hpp` 的 `matmul_impl` / `execute_fused_graph`：上层决策。
