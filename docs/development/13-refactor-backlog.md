# 重构与性能机会清单（2026-09-25 审查）

> 来源：v1.4.2 全量代码审查（issue 跟进轮）。**本文档只记录方案与前提，不实施**。
> 正确性缺陷已在同轮修复（threadpool 确定性归约、require_same_shape Release 守卫、
> mnist optimizer batch 包裹、submit() 死代码删除、zipt abort 诊断、python 裸 except、
> 过时注释清理）；修复详情见 git log。重新立项时先读本文件 + 对应源码现状。

## 1. 线程池归约策略统一（P1 审查项的后续）

**现状（2026-09-25 已修）**：`parallel_transform_reduce` 两个重载已改为确定性分段
（边界只由 `total` 决定，见 `core_threadpool.hpp::reduce_chunk_count`），1-worker /
8-worker / 串行参考逐字节一致，性能 6.6×（9M 元素 1.32ms vs 8.7ms）。

**遗留重构机会**：
- `parallel_transform`（elementwise 版）与 `parallel_for_*` 仍用 `chunk_count()`
  （依赖 `workers_.size()`）。逐元素操作结果与分块无关，**无确定性问题**，但
  分块策略两套并存，概念上有统一空间。
- `row_reduce`/`col_reduce`（algebra_matrix）各有独立的行块并行实现 + 与
  `std::execution` fallback 并存。若要"全库唯一归约语义"，可复用
  `reduce_chunk_count` 的边界公式做行块划分（行内顺序不动，天然逐字节一致）。
- **注意**：PyTorch ATen::parallel_reduce 反例（部分和按 `results[tid]` 存放、
  边界依赖 `get_num_threads()`）在头注释有记录，重设计时别走回去。

## 2. 超大 header 拆分（编译时间）

现状行数（2026-09-25）：`compute_vk_backend.hpp` ≈4700 行、`compute_cpu_engine.hpp`
≈2543 行、`expr_glsl_gen.hpp` ≈2310 行。单文件多职责（backend 混 fence/staging/
dispatch/算子）。项目几乎全 header-only，增量编译是主要开发成本。

**方向**：backend 拆 fence 管理 / staging ring / dispatch / 算子为独立 TU
（`compute_staging_ring.hpp` 已先行拆出，证明可行）。**前提**：先补一组编译期/
链接期冒烟测试，拆分期间保证 ODR 与 include 次序不敏感。

## 3. text_train.cpp 模块化（1860 行）

CLI 解析、训练循环、device-lost 恢复、checkpoint 逻辑混杂。
**方向**：抽 `train_loop` / `recovery` 模块；退出码语义修正
（checkpoint 保存失败应 `exit(1)` 而非成功退出）。与 TDR 恢复路径纠缠的
`exit(0)` 类问题（历史 P1-33）就是这种纠缠的产物。

## 4. gui.py 五 Tab 声明式字段表（1636 行）

五个 Tab 的 `collect_args` 全是 `_int/_float/_str(entry, key)` 手工罗列
（:1471-1500）。**方向**：dataclass + 字段元数据 → 自动生成 UI 行 + 收集函数，
一处改动全局生效，消除漏参风险。

## 5. 性能机会（未实施）

### 5.1 F16C 半精度转换快路径
`precision.hpp::float_to_half_bits` 是纯软件位操作；已有 x86 pause 原语先例
（`NN_CPU_PAUSE`）。**方向**：`_cvtss_sh`（F16C）快路径 + 批量转换，
`from_matrix/to_matrix/cast` 的全量标量循环可提到 8-wide。
**前提**：F16C 需 `target = f16c` 或运行时 CPUID 探测——项目现用 `-march=native`
（NN_ENABLE_NATIVE=ON 默认），需处理 ON/OFF 两态与 MSVC `/arch:` 差异；
改前必须跑 `f16_cpu_probe` 全量对拍（历史 float_to_half_bits 次正规 UB 教训，
见 docs/development/05 §12.12）。

### 5.2 CPU 引擎 f32→f16→f32 冗余双趟
`compute_cpu_engine.hpp` to_matrix 的 "tensor=f32, 请求 P=F16" 分支做
f32→f16→f32 双趟 cast 再返回 f32（量化模拟语义）。热路径上多一次全量转换。
**方向**：引擎层缓存该组合，或 API 层显式区分"量化模拟"与"直接返回"。
**注意**：这是量化模拟语义，直接删除会改变行为——需先确认无调用方依赖。

### 5.3 GUI 绘制节流
- `gui.py` 画布 `_on_draw_move`（:1127）每次 motion 全 buffer 降采样 + 预览重绘，
  快速拖动卡顿。**方向**：`after(16)` 按帧节流（图表已有 `_RENDER_MS=250` 节流
  先例，:213/:383）。
- `ChartWidget._render_data` 定时全量重绘不查数据更新。**方向**：dirty 标志跳过空帧。

## 6. 已核对为误报/已修复的审查项（勿重复处理）

| 审查项 | 核对结论（2026-09-25 读码） |
|---|---|
| BPE vocab_size 下溢（P1） | **已修复**：`domain_tokenizer_bpe.hpp:84` 有运行时守卫（`< 258` 返回 Error），注释 :81-83 即该守卫的说明，审查误读为"未加校验" |
| 反序列化无长度上限（P1-S3） | **已修复**：`model_serialization.hpp` `kMaxSerializedStringBytes`=64MiB，read_spec_header/read_tokenizer 双处校验；本轮同步了过时 TODO 注释 |
| registry bin 丢 batch（P1-23/25） | **已修复**：`expr_registry.hpp:157/:198/:296/:342` 读写均含 batch（bin v5+）；`gen_fused.cpp:125/:150` 显式生成 |
| GPT generate 未包 begin_batch（P1-7/29） | **已修复**：`compute_layer_gpt.hpp:984-1099` fill_cache_ 与逐步生成均已包裹 |
| forward_sparse 越界 label 静默置 0（P2） | **非缺陷**：越界列 mask 同步置 0（`compute_loss.hpp:359/:383/:394`），loss/梯度/num_valid 全部排除该列 = 忽略语义；`ce_fusion_test.cpp:122-128` 有专项测试 |
| restart_on_device_lost exit(0)（P1-33） | 当前代码无此函数（`text_train.cpp:427` 的 exit(0) 是 `--help` 正常退出） |
| checkpoint 不存 Adam m/v（P1-34） | **属实但为功能缺口非缺陷**：`.bin` 无优化器状态段；需 MODEL_VERSION 升版 + 格式设计（见 §7） |
| ZiPT stored_tokens 无门控（P1-10） | **低优先**：clone 是 token IDs（(batch,seq)，量级 KB-MB），非激活张量；GPT/RAPT 同模式，门控需训练/推理模式标志，收益小 |

## 7. Checkpoint 优化器状态（P1-34 功能缺口，重新立项前提）

`.bin` 现只存权重 + extra_state，不存 Adam m/v（momentum）。TDR 重启后动量归零。
**设计要点**：MODEL_VERSION 升版；可选段（老文件缺段 = 冷启动语义）；
`Optimizer` 基类加 `save_state/load_state`；text_train 恢复路径接线。
**依赖**：先完成 §3（text_train 模块化），否则恢复逻辑继续膨胀。

## 8. CI 冒烟（issue #13 审查建议，未实施）

- MSVC Debug 冒烟 + 权重 MD5 变化断言（防"loss 不动"类静默失败）。
  现有 `f16_writeback_probe` 已覆盖 f16 写回子集，可扩展为通用训练 N 步
  参数变化校验。
- MSVC Debug 大 TU（text_train.cpp）可能需 `/bigobj`（C1128），未复现前不加。

## 9. 已执行：代码缩减轮（2026-09-26）

> 依据：全仓"悬空设计"审计（A/B/C 三档）+ activation offload 只读评估。逐项均已 build + ctest 验证（**20 → 19 个目标，19/19 全绿**）。

| 项 | 内容 | 验证 |
|---|---|---|
| A 档死码 | `elementwise_select_scalar_cond`、`axpy_inplace`、`broadcast_row_inplace`、`nn::one_hot`、`text_train::one_hot_labels`、`Matrix::multiply_transposed_add_to`、`nn::sigmoid`·`relu`、`ops::Sigmoid`·`ReLU` 全删 | ctest 19/19 |
| 测试结构 | 新增 `src/test_common.hpp`：`CHECK`/`make_tensor`/`check_close`/`approx`/`dot`/`close_to` 收敛唯一副本；聚合器不再逐符号 `#define` 重命名。**保留聚合编译**（`#define main`），子测试不再各自复制工具 | ctest 19/19 |
| RAPT→DSL | `compute_layer_rapt.hpp` `forward_step` 的 `num/den` 除法改 `dsl::compute(engine, leaf/(leaf+Scalar{1e-4}), rows, cols)`，与 `forward` 表达式**同构** → 复用已注册 AOT 键 | rapt_test / rapt_offload_test |
| 算子收敛 | 删 `elementwise_unary/binary/binary_scalar`、`broadcast_col_inplace`（+`broadcast.comp`/`broadcast_gpu`/pipeline/成员/查询全链）、引擎 `row_reduce_max`、`offload_store/load`、`UnaryOp/BinaryOp/CompareOp`；**引擎 virtual 58 → 49** | ctest 19/19 |
| AST 移除 | `algebra_expr.hpp` + `algebra_compute.hpp` 删除；`Expression`/`BoolExpression` 迁入 `expr_dsl.hpp`；`Matrix::detail::*` 死函数删除 → **CPU 求值只剩两套** | ctest 19/19 |
| offload 评估 | 结论：**保留 A 组生产链，删除 B 组**（`offload_store/load` + `offload_primitive_test` + 重复聚合器）；`gpt_offload_test` 补 `begin_batch/end_batch` 覆盖训练录制路径；CPU/ZiPT 由静默 no-op 改显式警告；帮助文本"互斥"更正为"可混合" | gpt_offload_test / rapt_offload_test |
| 死 override | `TransformerEncoderLayer` / `ZiPTBlock` 的 `activation_cache()`（20 行，无调用者）删除 | ctest 19/19 |

**本轮教训（进 §10 高频坑候选）**
1. `dsl::compute` **没有 2 参重载**——签名是 `(engine, expr, rows, cols[, P])`；旧文档写的"`dsl::compute(engine, expr)` 最常用"是错的（已在 AGENTS §7 更正）。
2. **eager → DSL 迁移前必须确认目标表达式结构已被 `scan_exprs` 覆盖**：否则 GPU 闭合世界运行期硬报错。本轮通过"写成与已扫描表达式同构"（RAPT 除法加 ε 位置对齐）零成本复用键。
3. 改测试公共头后，**聚合器的 `#undef CHECK` 必须同步删除**（pragma once 之下 `#undef` 会让后续子文件失去宏定义）。
4. **未收集**：`max_abs_diff`（9 份副本，其中 conv2d 版多形状守卫语义）——统一前需逐点确认语义，留待单独立项。
5. **未收集**：`gpt_test` 偶发失败（本轮 62 次直跑仅 1 次失败、日志被覆盖未定位；疑似资源/竞争，非本轮改动引入，登记观察）。

