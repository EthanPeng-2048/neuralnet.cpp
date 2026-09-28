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

## 10. Tensor 绑定引擎 + 存储多态（提案，2026-09-28 设计讨论）

> 状态：**方案未实施**。选型已定（见 §10.8），重新立项时先读本节 + 源码现状。
> 范围（2026-09-28 复审扩充）：① 绑定与传参解耦（P1-P4）② 跨设备互传协议 `import`（§10.6）
> ③ 存储多态与访问器（P5-P6，兑现"简化后期开发新后端"）。
> **不采用**：`Tensor` 抽象基类 + `CPUTensor/GPUTensor` 句柄指针化（理由 §10.8）；
> `Tensor::host` / `.to(engine)` 两步创建（唯一入口收敛，§10.3 第 1 条）；
> `to_matrix` / `from_matrix` / `copy_from` 作为日常 API（降级为宿主中转特例，§10.6 表）。

### 10.1 现状与动机（坐标可复现）

仓库已有"绑定后免传 engine"的先例，本条目是把该模式下沉到 `Tensor`/`Layer`，消除内部不一致：

| 对象 | 现状 | 坐标 |
|---|---|---|
| `Model` | 持 `observer_ptr<ComputeEngine> engine_`，`forward/backward` **不带** engine，未绑定时 `Result` 报错 | `model_container.hpp:36` / `:185` |
| `Optimizer` | 持 `ComputeEngine& engine_`，`step()` 不带 engine | `compute_optimizer.hpp:46` |
| `Layer` | `init(ComputeEngine&)` 是**现成绑定位**（`Model::add` 调用），但 `forward/backward/zero_grad` 每次现传 | `compute_layer_base.hpp:105` / `:69` / `:80` |
| `Tensor` | 无绑定 → 上层被迫到处传 engine | `compute_tensor.hpp` |

耦合代价（数字复现方式写在括号内）：

1. **序列化必须从 Model 拿引擎**：`save_model` / `load_model` 需 `model.engine()` 才能下载/写入参数（`model_serialization.hpp:539` / `:639`）——L3 本可只依赖张量自身。
2. **GPU 原语入口防御性兜底**：`GpuEngine::ensure_gpu()` 有 **43 个调用点**（`compute_gpu_engine.hpp:455-1641`；定义 :1707，注释 :1706 自述"纯 GPU 架构下所有 Tensor 应已是 GPU，此方法为防御性兜底"）；另有 `copy_from:409-413` 的句柄重绑定 `dst = Tensor::from_matrix(...)`。绑定后"入参不是本引擎的张量"变成构造期不变量。
3. **每调用现传的规模**：`include/` 头文件中 `ComputeEngine&` **175 处**（含应保留的绑定位与应删除的每调用形参；复现：在 `include/` 下 grep `ComputeEngine&`）。对照：`Result<Tensor>` 全仓头文件 **203 处**——这是"子类化方案"必须指针化的量，本方案**一处不动**。
4. **存储层耦合（加后端要改容器本体）**：`compute_tensor.hpp` 有 **7 处 `#ifdef NN_HAS_VULKAN`**（行 28/65/99/161/205/245/304），并 include `backend/compute_vk_backend.hpp`（≈4700 行，见本文件 §2）——新后端 = 改 `compute_tensor.hpp` 加 variant 槽 + 加 `#ifdef` + 拖进新后端头。

### 10.2 目标用法（选型依据，代码为准）

```cpp
// ── 创建：唯一入口 = 引擎方法，engine + precision 出生一次到位，返回值类型 ──
auto t = engine.create_tensor(128, 64, nn::Precision::F16);
static_assert(std::is_same_v<decltype(t), nn::Tensor>);   // 仍按值，shared_ptr 存储
auto u = engine.create_tensor(32, 8);                     // 无 P 版：默认 F32（现状语义）
auto v = engine.from_host(host_matrix, nn::Precision::F16);  // 宿主数据 → 成绑张量（原 from_matrix）
// 未绑定态只有 nn::Tensor{}：空槽，valid()==false，禁止参与计算
// 不存在 "host-only 张量"：宿主侧数据用 Matrix 表示（L1 即宿主表示），要张量时已经在引擎上

// ── 查询与宿主动词（不再到处传 engine）──
t.bound();  t.engine();  t.precision();  t.valid();       // valid() = bound() && storage
auto host = t.to_host(nn::Precision::F32);                // 下载（原 to_matrix；宿主中转特例，§10.6）
t.upload(file_matrix, file_prec);                         // 写既有存储（原 copy_from；同为特例）
auto t16  = t.cast(nn::Precision::F16);
t.cpu_matrix<nn::Precision::F16>();                       // 既有访问器不变

// ── 跨设备互传：唯一对外入口 = 目标侧 import（协议见 §10.6）──
auto g = gpu.import(t);   // 同引擎 = 零拷贝别名 / cast；无 A<->B 快速通道 = 默认经宿主（CPU）中转
// 旧 ensure_gpu（43 处隐式防御）在 P3 收敛为显式 import

// ── Layer：init 是绑定位，forward/backward 删 engine 形参 ──
class Linear final : public nn::Layer {
    nn::observer_ptr<nn::ComputeEngine> engine_;          // init 时绑定
    nn::Tensor w_, b_, grad_w_, grad_b_, input_cache_;    // 成员类型不变
public:
    nn::Result<void> init(nn::ComputeEngine& e) override {
        engine_ = nn::make_observer(e);
        w_ = engine_->from_host(w_cpu, p_.param);         // 出生即绑定（原 from_matrix）；精度显式（§8.5）
        return {};
    }
    nn::Result<nn::Tensor> forward(const nn::Tensor& input) override {
        if (input.engine() != engine_.get())              // 跨引擎混用 → 硬错误
            return std::unexpected(nn::Error{"forward: tensor bound to another engine"});
        return nn::dsl::compute(*engine_, dsl::matmul(w_, input) + dsl::row_broadcast(b_),
                                w_.rows(), input.cols(), p_.compute);
    }
    void clear_cache() override { input_cache_ = nn::Tensor{}; }   // 空槽语义不变
};

// ── 序列化：L3 不再依赖 model.engine() ──
for (auto& p_tensor : params) {
    auto m = p_tensor.get().to_host(nn::Precision::F32);  // 原：engine.to_matrix(p_tensor, F32)
    detail::write_matrix_v5(ofs, *m, p_tensor.get().precision());
}
// load 侧：p_tensor.get().upload(file_matrix, file_prec)（in-place 写既有参数；精度契约由 P4 修复）
```

### 10.3 设计要点（八条不变量）

1. **张量只能由绑定引擎产生（唯一创建入口）**：`engine.create_tensor(rows, cols[, P])`、`engine.from_host(matrix, P)`（原 `from_matrix`）、算子输出；`reshape`/`clone` 保绑定。**不提供 `bind()/unbind()`**，也**不提供 `Tensor::host` / `.to(engine)` 两步创建**——宿主数据用 `Matrix` 表达，需要张量时它已经在某个引擎上。非绑定态仅剩 `nn::Tensor{}`（空槽，`valid()==false`，禁止参与计算）。晚绑定 = "未绑定 → 已绑定 → 换引擎"状态机，正是本条目要消灭的东西。
2. **值类型不变**：`Tensor` 仍是值句柄（shared_ptr 存储）；`Result<Tensor>`（203 处）、`std::span<const Tensor>`、`std::vector<Tensor>`、层成员类型**全部不改**。
3. **生命周期**：用 `observer_ptr`（`core_observer_ptr.hpp`，与 `Model` 同惯例）。**绑定不延长引擎寿命**——契约写明"engine 必须比它创建的 Tensor 活得久"；一期不加析构探测，若实际出现悬垂事故再评估 `weak_ptr<EngineToken>` 方案（先量化再决策）。
4. **跨引擎一致性**：双张量操作入口检查 `a.engine() == b.engine()`，不等返回 `Result` 硬错误；P1 阶段先加检查（零行为变化），P3 升级报错语义。设备间数据移动**只走 `import`**（§10.6），不走"把张量喂给别的引擎的算子"。
5. **精度出生即定、此后不可变**（§8.3 存储精度不变量的显式化）：`create_tensor` 无 P 版默认 **F32**（与现状签名一致，不引入 engine 侧 profile 第二真相源）；层内显式传 `p_.param`（§8.5）。
6. **头文件环（P1 唯一结构性前置）**：`compute_engine.hpp` include `compute_tensor.hpp`，故 `compute_tensor.hpp` 只**前向声明** `class ComputeEngine;` 并声明委托动词，`inline` 定义放在 `compute_engine.hpp` 尾部（调用这些动词的 TU 本就 include 它）。
7. **存储由引擎工厂造（P5 前置，否则循环依赖）**：`TensorStorage` 抽象基类定义在 `compute_tensor.hpp`（对后端 storage 只前向声明）；`GpuStorage` 这类**需要 backend 引用**的 storage 只能由对应 `XxxEngine::create_tensor` 内 `make_shared`，`Tensor` 只收 `shared_ptr`——若 storage 自己找 backend，就会 storage → backend → 引擎头 → Tensor 头成环。
8. **`reshape` 下沉为 virtual**（`reshape_view(new_rows, new_cols)`）：GPU 零拷贝共享 buffer、CPU 复制数据（坑 #6 的语义差异）由各 storage 自己实现，差异留在实现内、不外泄到 `Tensor`。

### 10.4 与既有精度决策的关系（本条目不改）

- **§8.5（P 显式）**：`create_tensor(rows, cols, P)` 显式重载保留；**无 P 版默认 = F32**（与现状签名 `P = Precision::F32` 完全一致，不引入 engine 侧 profile 真相源），层内按 `p_.param` 显式传。"profile 收归 engine 作默认"若要做，属下一条的另立项范围。
- **§9.2 D7（`Layer::p_` 注入链）**：profile 归属**不动**，不引入 engine 作为第二真相源；"profile 收归 engine" 是另一选题，牵动 f16 数值路径（`f16_precision_test` / `f16_writeback_probe` / 已知 `stable=f16` 冻结故障），须单独立项评估。
- **DSL 入口保留 engine 形参**（`expr_dsl.hpp:1340` `dsl::compute(engine, expr, rows, cols[, P])`）：它是输出分配上下文，纯标量表达式无叶子可推导；绑定后 Layer 传 `*engine_` 即可，签名零改动。

### 10.5 分期与验收（每期构建全绿；失败即停，不带坏状态继续）

| 期 | 内容 | 验收 |
|---|---|---|
| **P0** 调研基线（0 改动） | ① 43 处 `ensure_gpu` 按入参来源分类（真防御 vs 承载真实路径）② 175 处 `ComputeEngine&` 打标（绑定位保留 / 每调用可删）③ 同 seed 训练 N 步 loss 序列 + 权重 MD5 基线 ④ `expr_specs.bin` / `fused_registry.hpp` hash 基线 ⑤ `from_matrix`/`to_matrix`/`copy_from` 调用方盘点（P2 改名准备） | 数字入档，供 P1-P6 对照 |
| **P1** 加绑定，不删参数 | `observer_ptr` + `bound()/engine()` + 跨引擎一致性检查（只加不改） | build 全绿 + ctest 全绿（`ctest -N` 实测 20 个用例，含 `cnn_test_gpu`）+ **字节基线逐字节一致** + scan 产物 hash 零 diff（不一致 = 暴露了暗中跨引擎路径，本身就是发现 bug 的手段） |
| **P2** 删形参 + 宿主动词改名 | 按 P0 打标：Layer → Loss → 辅助函数 → CLI；编译器报错驱动，漏不掉；DSL 入口不改。同步改名（§10.6 表）：`to_matrix→to_host`、`from_matrix→from_host`、`copy_from→upload`，旧名全仓消失 | 同 P1 + `layer_bench` 无回退（警戒线：`Tensor::cpu_get_ptr` 注释实测 8-70×、`expr_dsl` `CpuViewCache` 1.5-4×） |
| **P3** 收紧语义 | 跨引擎混用升 `Result` 硬错误；43 处 `ensure_gpu` 按 P0 分类改为显式 `import`；序列化改 `p_tensor.get().to_host()`，L3 去 `model.engine()` 依赖 | 同 P1 + GPU 用例结果如实记录（退出码 77 = skip） |
| **P4** 正交轨道，**另案勿混入** | `Matrix` 宿主格式契约修复：`model_serialization.hpp:651-674` 同精度路径丢弃 `from_matrix` 结果（每个参数白做一次全量上传）、跨精度路径 5 次设备操作；`CpuEngine::copy_from`（`compute_cpu_engine.hpp:210`）`dst.cpu_matrix()` 硬编码 F32 槽（f16 dst 在 `NDEBUG` 下空指针，现靠 CLI 层"非全 f32 才包 PrecisionEngine"兜底：`text_train.cpp:949-959` / `mnist_train.cpp:605-609`）；`ComputeEngine::matmul_with_bias`（`compute_engine.hpp:327-352`）host 往返兜底为死码（三引擎均 override） | 单独立项、单独验收；绑定解决"谁执行传输"，不解决"传输格式契约" |
| **P5** 存储多态（P1-P4 之后独立做） | `compute_tensor.hpp` 引入 `TensorStorage` 抽象 + `CpuStorage`/`GpuStorage`；**storage 由引擎工厂造**（§10.3 第 7 条）；`reshape` 下沉 virtual（第 8 条）；消 7 处 `#ifdef NN_HAS_VULKAN` 与 backend include | build 全绿 + ctest 全绿 + **字节基线逐字节一致** + scan 产物 hash 零 diff（改的是容器内部，靠字基线验收，不靠编译器） |
| **P6** 访问器（纯加法） | `as_cpu()/as_gpu()` 检查式 downcast 返回子类引用（非虚，句柄仍值类型） | 纯加法：ctest 全绿即可 |

### 10.6 跨设备互传协议（`import` = 唯一对外入口）

**事实前提**：现有数据流全程单引擎驻留（AGENTS §4.2：`Matrix → from_matrix → 设备上算 → evaluate 时 to_host`），生产路径**没有**跨设备互传调用方；今天的"互传"只有 `GpuEngine::ensure_gpu()` 的 43 处隐式防御。故 **P1 只定义接口 + 同引擎 fast path**，跨设备实现等 P0 分类结果（先量化，不为不存在的路径写代码）。

```cpp
class ComputeEngine {
    // 唯一互传入口（目标侧拉取：分配与执行都在目标引擎——只有它知道自己的存储怎么造）
    // 基类默认实现 = 经宿主（CPU）中转，即"没有 A<->B 快速互转时的兜底"：
    //     auto m  = src.to_host(src.precision());   // 源下载，只搬原始字节
    //     auto m2 = cast_on_host(m, P);             // 精度转换一律在宿主侧做
    //     return this->from_host(m2, P);            // 目标上传
    // 同引擎 fast path（在默认实现内）：同 P = 零拷贝别名，异 P = cast
    // 快速通道 = 引擎 override：同设备直拷、将来 peer copy —— 调用方零改动
    [[nodiscard]] virtual Result<Tensor> import(const Tensor& src, Precision P /* 默认 = src 精度 */);
};
```

**`to_matrix` / `from_matrix` / `copy_from` 退出计算路径**，改名收敛为三个**宿主中转特例**（宿主表示就是 `Matrix`，IO/初始化绕不开）。**红线：`forward`/`backward`/算子实现内部不得直接使用宿主中转**——那正是今天 PCIe 往返散落的根源；只允许出现在 IO、init、`import` 回退三类场景：

| 新名 | 原名 | 允许出现的场景（计算路径禁用） |
|---|---|---|
| `Tensor::to_host(P)` | `to_matrix` | 序列化下载；评估/推理结果回读；`import` 默认回退的源侧 |
| `engine.from_host(m, P)` | `from_matrix` | 层 init 权重上传；序列化上传（产新张量）；`import` 默认回退的目标侧 |
| `dst.upload(m[, P])` | `copy_from` | in-place 写既有张量存储（序列化 load / 参数写回；契约见 `cast_into` 注释：不可替换 dst 对象，否则持有同一句柄的缓存静默失联） |

- 改名放 **P2**：编译器穷尽；旧名全仓消失后，**grep 旧名即可审计"计算路径是否还在绕过 `import` 直接用宿主中转"**（命名即防线）。
- `ensure_gpu()` 43 处 → P3 逐处改 `import`；`GpuEngine::copy_from:409-413` 的句柄重绑定防御分支随之删除。
- **精度规则**：转换一律发生在宿主侧，设备只搬原始字节——现状 `GpuEngine::to_matrix`（`compute_gpu_engine.hpp:245-257`：先下载 f16、再在 host 升 f32）正是这么做，本协议把它写成明文；与 P4 协同可消灭 `load_model` 跨精度 5 次往返。
- GPU→GPU（异设备）：当前单后端，走宿主中转；接口留位，将来 peer copy 只改 `import` 内部，调用方零改动。

### 10.7 新后端开发 checklist（本条目的兑现目标）

| 步骤 | 现状要改哪里 | P1-P6 之后 |
|---|---|---|
| 1. 实现引擎接口 | `ComputeEngine` **49 个 virtual 成员**（`compute_engine.hpp` 共 50 处 `virtual`，:69 是析构——文档"49"与审计脚本"50"的差异来源） | 不变；传输面收敛为 `import` + 两个宿主特例 |
| 2. 提供存储类型 | **改 `compute_tensor.hpp`**：加 variant 槽 + `#ifdef`（现有 7 处再加 1）+ include 新后端头 | **`compute_tensor.hpp` 零改动**：只写 `XxxStorage : TensorStorage`（P5） |
| 3. 创建入口 | 各自实现 `create_tensor` | 同左，返回值**出生即绑定**自己（P1） |
| 4. 条件编译 | `NN_HAS_XXX` 渗入容器头 | 只出现在新后端自己的头 + CMake（P5） |
| 5. AOT 代码生成 | — | `expr_emitter.hpp` 注册 emitter（现仅 `glsl`）；CPU 式后端走 DSL 模板路径可豁免；表达式结构必须被 `scan_exprs` 覆盖（铁律 7）——**新后端真正的大头，P1-P6 不解决** |
| 6. 测试 | — | 需 Vulkan 类后端的用例退出码 77 = skip，如实记录 |

### 10.8 选型记录（供 `docs/history.md` 归档引用）

2026-09-28 两轮讨论结论：

- **采用：出生绑定**（P1-P3，原"方案 B"）。与仓库先例同构（`Model` / `Optimizer` / `Layer::init`）；改动形态是**删形参**——漏改处全部编译报错、可被编译器穷尽，而改返回类型需逐处人工判断值/指针语义；值语义零改动。
- **采用：存储多态 + 访问器**（P5-P6，即原"A 的两项子诉求"）。消 `compute_tensor.hpp` 7 处 `#ifdef` 与 backend include，新后端不改容器头；子类式访问人体工学用 `as_cpu()/as_gpu()` 检查式 downcast（非虚）获得。
- **不采用：`Tensor` 抽象基类 + `CPUTensor/GPUTensor` 句柄指针化**（原"方案 A"）。它要解决的 device 互斥，收益与存储多态重叠，却要 `Result<Tensor>` **203 处**指针化 + `span/vector` 元素类型连锁 + `GpuEngine::copy_from:412` 句柄重绑定重写；且虚函数不能模板化，精度维度 variant 依旧要留（收益只覆盖一半），更关键的是**不解决 engine 传参耦合**（`forward(engine,...)` 一个都跑不掉）。
- **收敛（复审）**：单一创建入口——删除原稿的 `Tensor::host` / `.to(engine)` 两步创建；宿主数据一律用 `Matrix` 表达。
- **收敛（复审）**：`to_matrix/from_matrix/copy_from` 退出日常 API，仅作宿主中转特例（§10.6 表）；互传统一走 `import`，**没有 A<->B 快速互转时默认经宿主（CPU）中转**。

