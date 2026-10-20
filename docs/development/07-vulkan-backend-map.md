# Vulkan 后端结构地图（P2-7）

> 面向"第一次读后端代码"的人：先建立调用链与所有权模型，再动手改。
> 本文只描述**当前代码**与**已确定的重构方向**；每一步的提交信息（`git log`）
> 是该刀的范围与验收记录。

## 1. 为什么需要这张图

后端拆分过一轮（RV-08），但那是**物理拆分**：四个 `.inc` 片段被
`class GpuBackend { ... }` 在类体内按序 `#include`。文件数变多了，读者面对的
仍然是"一个类、四段成员定义、彼此共享全部私有状态"。

P2-7 的前几刀先把**数据与表**从类里移出去（存储层、pipeline 目录、融合
kernel 记录、描述符绑定），当前规模：

| 位置 | 行数 | 读者实际看到的内容 |
|---|---:|---|
| `backend/compute_gpu_storage.hpp` | 255 | `GpuBuffer` / `GpuTensorT<P>`（只依赖 Vulkan 句柄 + MemoryPool） |
| `backend/compute_vk_pipeline_catalog.hpp` | 176 | `PipelineId` / `PipelineSpec` / `PipelineCatalog` |
| `backend/compute_vk_pipeline_specs.hpp` | 146 | 44 行 pipeline 布局表（唯一真源） |
| `backend/compute_vk_backend.hpp` | 814 | 设备 + 目录 + 帧环/池/计时状态 |
| `backend/compute_vk_backend_destroy.inc` | 550 | 延迟销毁 + `initialize()`（一条创建序列） |
| `backend/compute_vk_backend_frame.inc` | 772 | 帧环、batch、上传下载、matmul |
| `backend/compute_vk_backend_pipelines.inc` | 721 | 扫描原语 + 描述符/屏障/solo fence 辅助 |
| `backend/compute_vk_backend_ops.inc` | 2415 | 15 个原语 dispatch + 融合执行 |
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
  │              ├─ alloc_desc_set_for(layout, buffers)
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

当前这两件事在 `GpuEngine::*_impl` 里仍混有 `cast_if_f16` /
`has(PipelineId)` 之类的后端能力判断；重构方向是把"能力查询"收进一个显式的
capability 对象，引擎只读事实、不读 pipeline 句柄。

## 3. 所有权与生命周期

| 对象 | 拥有者 | 生命周期要点 |
|---|---|---|
| `VulkanDevice` | `GpuBackend::device_` | `GpuBackend::shared()` 进程共享；最后一个强引用释放时 RAII 析构 |
| `MemoryPool` ×2（持久/瞬态） | `GpuBackend` | 所有 `GpuBuffer` 的子分配都来自其一 |
| `StagingRing` | `GpuBackend` | 阻塞上传/下载路径；时间线信号量可选 |
| `VulkanPipeline` ×44 | `GpuBackend` | `initialize()` 一次性创建；缺失 = 该变体不可用 |
| 帧环 `frames_[6]` | `GpuBackend` | cmd + fence + 描述符集按帧归属；环槽复用前等 fence |
| `GpuBuffer` | `shared_ptr`，被 `GpuTensorT` 共享；另持 `shared_ptr<GpuBackend>` keepalive | 析构可能**延迟**到引用它的帧完成（`defer_buffer_destroy`） |
| `GpuTensorT<P>` | `shared_ptr`，被 `Tensor` 类型擦除持有 | 零拷贝 `with_shape` / 别名共享同一 `GpuBuffer` |

两条容易踩的规则：

1. **batch 录制期创建的张量不能立即销毁**。已录制的 descriptor set 仍引用
   buffer；`GpuBuffer::~GpuBuffer` 经自己持有的 backend keepalive 登记延迟销毁，
   不再调用可能已销毁的全局 `instance()`。
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
| 描述符分配/绑定 | `pipelines.inc` 的 `alloc_desc_set_for` | `vk::DescriptorAllocator` + `vk::DescriptorWriter` |
| 命令录制/提交 | `frame.inc` + `pipelines.inc` | `vk::CommandRecorder` + `vk::FrameRing` |
| 算子语义 | `ops.inc` + `frame.inc` 的 matmul | `gpu::ops::*`（接收 `OpContext`） |

### 5.1 已收敛与待收敛

已完成（P2-7 第 2–5 刀）：

- 44 个 pipeline 成员 + 40 个 `has_*_pipeline()` 手工转发 → `PipelineCatalog`
  + 一个 `has(PipelineId)` 查询入口；
- `initialize()` 中 40 段"取 SPIR-V → 判空 → 创建 → 移动" → 一张
  `PipelineSpec` 表 + 一个循环（`compute_vk_pipeline_specs.hpp`）；
- 融合 kernel 的 9 张并行 map → 一个 `FusedKernel` 记录（注册时一次写定）。
- 15 处算子各自重复的描述符写入样板 → `alloc_desc_set_for(layout, buffers)`。

仍待收敛：

- 每个算子仍在重复"取 cmd → bind → push → dispatch → barrier → submit"
  → 目标是一个 `DispatchRequest` + `CommandRecorder`；
- `GpuBackend` 仍同时承担设备、存储、帧环、描述符、剖析、算子六件事 →
  目标是 `vk::` 协作者 + `gpu::ops::*` 薄算子。

## 6. 怎么加一个原语算子

1. `shaders/<name>.comp` 写 kernel；在 `compute_vk_backend.hpp` 文件头加
   `__has_include` 探测块，并把 token 加进 `NN_SPV_GETTER_LIST`（RV-08 单一
   注册表）。
2. `compute_engine.hpp` 加 op-level 接口（NVI 入口 + 默认 "unsupported"）。
3. `GpuEngine` 加 `*_impl`：只做形状/精度决策，把"怎么录"交给后端。
4. 后端加一个 `*_gpu`：`alloc_desc_set_for` → push constants → dispatch →
   barrier（不再手写 `VkWriteDescriptorSet`）。
5. `shaders/scan_exprs` 不需要登记（op-level 原语不进 AOT 闭合世界）。

## 7. 怎么加一个 pipeline 变体

当前：`PipelineId` 加一个名字，`pipeline_specs()` 加一行（SPIR-V getter /
binding 数 / push 字节数 / 能力要求），算子侧用 `PipelineId` 选路——**两处**。
`has()`、`get()`、创建循环都不需要改。

## 8. 验收口径

任何一步结构改动都按下面的门槛走（编译时间是观测量，不是验收项）：

- `ctest --test-dir build-gpu`（GPU + CPU）全绿；
- 设备域检查默认生效（无独立开关；跨域操作数 = 明确 `Result` 错误）；
- 字节锚不变：`gpu_stability_probe --steps 20`、
  `layer_bench --op matmul,matmul_q,matmul_q_bw,scan_prefix,scan_suffix --f16 --hash`、
  `rapt_test --gpu` / `--f16 --gpu`、`scan_bitwise_test`；
- 融合产物不变：`expr_fuse_test`、`fusion_custom_layer_example`；
- 默认构建**不引入 LTO**；头文件布局不因"想少编译几个 TU"而妥协。

### 8.1 为什么不用 LTO 换结构

本次重构的收益必须**在默认构建（无 LTO）下成立**。LTO 是单线程串行的
链接期优化，成本远高于并行编译；把它当作"跨文件内联的补偿"会让库外消费者
被迫接受一次不可并行的长尾。

因此：

- 头文件内的小访问器、转发函数保持 `inline` 定义（不产生跨 TU 调用）；
- 算子边界（per-dispatch）允许一次普通调用——GPU 提交本身就是微秒级，
  多一层函数调用不改变可观测性能；
- 任何"必须 LTO 才能回到原性能"的拆分方案直接否决。

### 8.2 当前会话的落地顺序

1. 本文件（结构地图）；
2. `compute_gpu_storage.hpp`：存储层下沉，切断 L1 → L2 反向依赖；
3. `PipelineCatalog` + `PipelineSpec`：44 个 pipeline 收敛为目录 + 一张表；
4. `FusedKernel`：9 张并行元数据表合并为一条记录；
5. `alloc_desc_set_for`：15 处描述符写入样板收敛。

每一步都是独立提交，且以 `gpu_test` / `expr_fuse_test` /
`scan_bitwise_test` 的通过为界。

## 9. 阅读顺序建议

1. `compute_vk_device.hpp`：设备与 pipeline 的 RAII；
2. `compute_vk_backend_frame.inc` 的 `acquire_cmd` / `submit_and_wait`：同步模型；
3. `compute_vk_backend_ops.inc` 的 `elementwise_v2_gpu`：最短的完整算子；
4. `compute_vk_backend_ops.inc` 的 `run_fused_gpu`：融合元数据如何参与选路；
5. `compute_gpu_engine.hpp` 的 `matmul_impl` / `execute_fused_graph`：上层决策。
