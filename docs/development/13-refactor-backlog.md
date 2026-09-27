# 重构与性能机会清单（2026-09-25 审查）

> 来源：v1.4.2 全量代码审查（issue 跟进轮）。**本文档只记录方案与前提，不实施**。
> 重新立项时先读本文件 + 对应源码现状。
> 被核对为误报/已修复的审查项、已执行的代码缩减轮等历史记录见
> `docs/history.md`（含原 §6 对照表与原 §9 执行表）。

## 1. 线程池归约策略统一

**现状**：`parallel_transform_reduce` 两个重载是确定性分段（边界只由 `total` 决定，
见 `core_threadpool.hpp::reduce_chunk_count`），1-worker / 8-worker / 串行参考逐字节
一致；性能 6.6×（9M 元素 1.32ms vs 8.7ms）。

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
`exit(0)` 类问题（审查项 P1-33，核对结论见 `docs/history.md`）就是这种纠缠的产物。

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
改前必须跑 `f16_cpu_probe` 全量对拍（`float_to_half_bits` 的次正规移位 UB 教训，
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

**本节对照表已移入 `docs/history.md`**（8 项：BPE vocab_size 下溢、
反序列化长度上限、registry bin 丢 batch、GPT generate 未包 begin_batch、
forward_sparse 越界 label、restart_on_device_lost exit(0)、checkpoint 缺 Adam m/v、
ZiPT stored_tokens 门控——含各自核对结论与代码位置）。**重复立项前先读该表。**

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

> 执行记录（全仓"悬空设计"审计 A/B/C 三档逐项、ctest 20 → 19、算子收敛 58 → 49、
> AST 移除、offload A/B 组处置明细）已移入 `docs/history.md`。
> 本节保留仍然生效的工程教训：

1. `dsl::compute` **没有 2 参重载**——签名是 `(engine, expr, rows, cols[, P])`；文档或注释
   里"`dsl::compute(engine, expr)` 最常用"的说法是错的（AGENTS §7 已按此写）。
2. **eager → DSL 迁移前必须确认目标表达式结构已被 `scan_exprs` 覆盖**：否则 GPU 闭合世界
   运行期硬报错。低成本做法：把新表达式写成与已扫描表达式**同构**（如 RAPT 除法的 ε 位置
   对齐 `forward`），即可零成本复用键。
3. 改测试公共头后，**聚合器的 `#undef CHECK` 必须同步删除**（pragma once 之下 `#undef`
   会让后续子文件失去宏定义）。
4. **测试公共头未完全收敛**：`max_abs_diff` 仍有 9 份副本（其中 conv2d 版多形状守卫语义）
   ——统一前需逐点确认语义，留待单独立项。
5. **`gpt_test` 偶发失败待观察**：62 次直跑 1 次失败、日志被覆盖未定位；疑似资源/竞争。

