# 代码审查整改立项台账（2026-10-11）

> **读者**：项目开发者（本仓内部工作文档）。本分册定位见 [`README.md`](README.md)；历史归档在 [`../history/history.md`](../history/history.md)。
>
> **来源**：2026-10-11 的一份外部代码审查报告（范围 = `include/neuralnet.cpp/` 约 4.6 万行 header-only 库 + `src/` + 构建系统）。
> **本文作用**：把审查报告里的问题**逐条立项**——先复核（哪些成立、哪些表述有误），再给每项定优先级、证据、
> 根因、方案、验收口径。**接手整改前先读本文**；未完成的"下一步该做什么"仍以 [`18-roadmap.md`](18-roadmap.md) 为准，
> 本文是它的**执行细节来源**，两者不重复立项（见 §5）。

## 0. 怎么用本文

**状态机**：`Open`（已立项，未动手）→ `Doing` → `Fixed`（**必须带验收证据**）→ `Archived`（结论进 `../history/history.md`）。

**每条的字段**：编号 · 优先级 · 状态 · 来源 · 现象与证据（符号名 + 2026-10-11 实测行号）· 根因 · 方案（分步）· 验收口径 · 影响面/风险 · 依赖。

**写作纪律**（与全仓一致）：行号取自 2026-10-11 实测、漂移后**以符号名为准**；只写"当前事实 + 决定"，历史叙事进 `../history/history.md`。

**通用验收基线**（本仓任何改动都要过，下文不再重复）：
`cmake --build build` 零告警 → `ctest --test-dir build` 全绿 →（涉及数值/默认结构时）字节锚逐位不变 →（涉及表达式/融合时）
`scan_exprs` 双 hash 不变 → `pwsh bench/doc_inventory.ps1` 与 `bench/doc_align_audit.ps1` exit 0。GPU 改动按 AGENTS §2 在**提权环境 + 显式选卡**跑。

---

## 1. 先读：审查结论的复核结果

**下表是"审查主张 → 本仓复核结论"**。先看这张表再往下读立项卡——它标出了 3 处表述有误/需修正的地方，避免照着错误结论改代码。

| 审查主张 | 复核结论 | 依据 |
|---|---|---|
| **P2**：`NN_ASSERT` 在 Release 为空展开 → 断言后继续解引用 = UB | ✅ **成立**（且本仓已有"修法先例"，见 RV-01） | `core_assert.hpp` 的 `NDEBUG` 分支 = `((void)0)`；8 类"断言后继续用"点，见 RV-01 |
| **P1**：`create_tensor` 失败返回**空 Tensor + stderr**，不是 `Result<Tensor>` | ✅ **成立**（代码注释自陈这是契约；同族 `from_matrix` 已是 `Result`） | `compute_engine.hpp::ComputeEngine::create_tensor`；调用面 **115** 处 |
| **P3**：`GpuBackend` 故意泄漏单例、缺显式 `shutdown()` | ✅ **已修复（2026-10-18，引擎所有权 v2）**：`shared_ptr` keepalive 取代故意泄漏；`GpuBuffer` 持 backend 强引用，`reduce_partial_` 改裸 `VkBuffer` 切断环；进程退出时静态 `shared_ptr` 正常析构并执行 `vkDestroy*` | `backend/compute_vk_backend.hpp`、`backend/compute_gpu_storage.hpp`、`src/layer_lifetime_test.cpp`；history「引擎所有权 v2」 |
| **P3**：`Tensor` 析构不探测引擎存活 → 悬垂观察者 | ⚠ **部分成立**：当前应用栈安全（引擎严于 Tensor 析构），嵌入场景是风险；`observer_ptr` 无 Debug 校验 | `core_observer_ptr.hpp`（明确"不提供 RAII"） |
| **P4**：绑定不变量靠人肉维护、`NN_BIND_DEBUG` 默认关不可靠 | ✅ 作为"不变量强度"问题**成立**；但**"`NN_BIND_DEBUG` 未走统一探针门 = 违例"标签不当**——AGENTS §2 明确把它列入探针门的**例外**（初始化期读一次并缓存的**行为开关**） | AGENTS §2；`compute_engine.hpp::bind_check_` |
| AdamW 的 decay / update **顺序依赖**（无数据依赖边） | ✅ **成立**（代码注释自陈"同目标两次 `compute_into`、无数据依赖边"） | `compute_optimizer.hpp::AdamW::step` |
| 稀疏 CE 原地覆写依赖派发顺序 | ✅ **成立**（注释自陈"派发序：lse → loss_sum → grad"） | `compute_loss.hpp::fused_forward_sparse_` |
| BitLinear 量化"用 round-half-away 而非 RN-E" | ❌ **不成立**：实现是**阈值 select**（严格 `>`，恰好 `0.5τ` 归 0），既不是 RN-E 也不是 round-half-away | `compute_layer_bitlinear.hpp::quantize_weights_`、`ternary_pack.hpp` |
| 优化器白名单重复 4 处 | ✅ **成立**，且实测是 **5–6 处** | 见 RV-04 |
| `compute_vk_backend.hpp` 6397 行 / `expr_glsl_gen.hpp` 3562 行 / `compute_cpu_engine.hpp` 2797 行 | ✅ 行数与实测**完全一致** | `wc -l`（2026-10-11） |
| "36 个 SPIR-V getter 同构样板" | ⚠ **数量有出入**：实测 **42** 个 `get_*_spirv()` | `rg -c 'static const std::vector<uint32_t>& get_[a-z0-9_]*spirv'` |
| CPU 引擎 `ExprOp` switch 分三处重复 | ✅ **成立**（标量求值 + 寄存器融合 ×2）；且 GLSL emitter 另有 **5 处** | 见 RV-07 |
| 环境变量读取样板 ≥6 处 | ✅ **成立**（库内 14 处 `getenv`、≥6 份各自 helper） | 见 RV-09 |
| VK Descriptor Set 回收需专项复审 | ✅ 保留为**只审计**项（RV-14） | `descriptor*` 出现 199 处 |
| 训练数值正确性（softmax/LayerNorm 稳定、CE 梯度缩放、Adam 偏差修正时机、AdamWQuant 尺度上界、Muon NS 系数） | ✅ 与本仓既有验收一致 → **不立项**（§4） | 见 §4 |
| Loss 直接构造 IR 底层 Spec（绕 DSL 门面）→ 第二条耦合通道；`FoldAnchor` 时间耦合 | ✅ **成立**（登记为 RV-12，**先设计后动手**） | `compute_loss.hpp` 顶部 + `fused_forward_sparse_` |

---

## 2. 立项清单

### RV-01 — Release（NDEBUG）下"断言后继续解引用" = UB；把编程错误统一为**无条件** fail-fast

- **优先级**：P0（必修） · **状态**：`Fixed`（2026-10-11） · **来源**：审查 P2 · **依赖**：无

**现象与证据**（`NN_ASSERT` 在 `NDEBUG` 下 = `((void)0)`，以下点位断言后**仍继续解引用/索引**）：

| 位置（符号） | 后果（Release） |
|---|---|
| `algebra_matrix.hpp::MatrixT::at` / `set_value` | 越界后 `ptr()[index(...)]` → OOB 读写 |
| `compute_tensor.hpp::Tensor::engine()` | `return *engine_` → 空解引用 |
| `compute_tensor.hpp::Tensor::{cpu_matrix,gpu_matrix,cpu_shared,gpu_shared}`（4 处） | 取视图后解引用空指针 |
| `compute_tensor.hpp::Tensor::reshape`（元素数校验） | 后续按错误形状索引 |
| `compute_layer_base.hpp::Layer::engine_ref()` | **已打印调用点**，但仍 `return *engine_` → 空解引用 |
| `compute_layer_base.hpp::sample_next_token_`、`model_container.hpp::Model::{engine_ref,layer_at}` | 空指针 / 越界索引 |
| `compute_layer_{attention,rapt,transformer}.hpp` 的 init 期整除校验、`{gpt,rapt}.hpp` 的 `seg < checkpoint_inputs_.size()` | 除零 / 越界索引 |
| `expr_dsl.hpp`（`data != nullptr` 后取视图）、`compute_position_encoding.hpp`（slope 缓存）、`compute_gpu_engine.hpp`（`:1779` 注释**自陈"Release 空解引用 = UB"**） | 空解引用 |

**先例（关键）**：`algebra_matrix.hpp::MatrixT::require_same_shape` **已经**是本仓认可的修法——
`NN_ASSERT(false, msg)` **之后**紧跟 `fprintf(stderr, "FATAL: …") + std::abort()`，注释写明
「Release（`NN_ASSERT` 展开为空）：**绝不静默 OOB**……此处 fail-fast 中止，代价为两次 `size_t` 比较」。
同一头文件里 `multiply_to_span` / `transpose_multiply_to_span` / `transpose_multiply_to_span` 等**没有**跟 abort
（旁边还留着 `(void)b_rows; // NN_ASSERT 在 Release 模式下展开为空` 的痕迹）——**修法已有，只是没铺开**。

**根因**：`NN_ASSERT` 语义是"可选诊断"，却被用在"前置条件校验"的位置——校验失败后**没有**终止路径。

**方案**（三步，可分期）：

1. `core_assert.hpp` 增补 `NN_ASSERT_ALWAYS(cond, msg)`（`NDEBUG` 下**也** `fprintf(file:line) + std::abort()`；
   Debug 下与 `NN_ASSERT` 同形），并把与 `NN_FATAL(msg)`（无条件终止）配套使用。
2. 逐点替换：**凡是断言后继续用该条件的点位**一律换 `NN_ASSERT_ALWAYS`；纯诊断（不影响后续正确性）保留 `NN_ASSERT`。
   顺手把 `require_same_shape` 的三段式收敛成宏，避免再抄一遍。
3. 加**负例回归**（新测试文件或扩 `error_macro_test`）：越界 `at`、未绑定 `engine()`、未 `init` 的 `engine_ref()`
   在 `-DNDEBUG` 构建下**必须**非零码终止并打印 `file:line`。

**验收**：`-DNDEBUG` 构建下 3 个负例均 abort（非静默 / 非 UB）；`ctest` 全绿；把"断言后仍解引用"的清单**清零**
（可加进 `bench/doc_inventory.ps1` 做可选门禁：抓 `NN_ASSERT` 后 3 行内无 `abort|return|NN_FAIL` 的点位）。

**影响面/风险**：只碰错误路径，热路径零成本（与 `require_same_shape` 同价：两次 `size_t` 比较）。

**验收证据（RV-01，2026-10-11）**：
① `core_assert.hpp` 新增 `NN_ASSERT_ALWAYS(cond, msg)` 与 `NN_FATAL(msg)`（两者
**无条件**终止，格式统一为 `<tag> failed at <file>:<line>: <msg>`，终止走既有
`detail::fail_abort`——Windows 下 CRT 弹窗/WER 已静音）；`require_same_shape` 的三段式
收敛成 `snprintf` + `NN_FATAL`（顺带修掉 `std::string_view::data()` 当 NUL 结尾串的隐患）。
② 全仓 `NN_ASSERT(...)` 调用点**清零**（`rg -n 'NN_ASSERT\(' include src tools tests examples`
只剩宏定义两行）：逐点改判为 `NN_ASSERT_ALWAYS` / `NN_FATAL`，覆盖 `algebra_matrix`
（`at`/`set_value`/`require_same_shape`/`multiply_to_span` 族/自引用）、`compute_tensor`
（`engine()`/4 个存储访问器/`reshape`）、`compute_layer_base`（`engine_ref`/`sample_next_token_`）、
`model_container`、`compute_layer_{attention,rapt,transformer,gpt}`、`expr_dsl`、
`compute_position_encoding`，以及 `src/` 测试侧同类点位（`test_common.hpp` 等）。
`NN_ASSERT` 语义（可选诊断）保留，供将来纯诊断使用。
③ 新增自举负例测试 `src/assert_always_test.cpp`（注册进 ctest）：`MatrixT::at` 越界 /
未绑定 `Tensor::engine()` / `sample_next_token_` 空 logits / 未 init 的层 `forward`
（→ `engine_ref`）四个子进程负例，断言"非零退出 + `NN_ASSERT_ALWAYS|NN_FATAL failed at`
+ `file:line` + 该点位消息"，并打印 `NDEBUG` 取值证明 Release 下同样终止。
本机 Release（`-O3 -DNDEBUG`）实测 `assert_always_test: 0 failure(s)`、`NDEBUG=1`。

### RV-02 — `create_tensor` 错误协议：空 Tensor + stderr → `Result<Tensor>`

- **优先级**：P1（必修，与 RV-01 同批） · **状态**：`Fixed`（2026-10-11） · **来源**：审查 P1 · **依赖**：RV-01（错误路径宏）

**现象与证据**：

- `compute_engine.hpp::ComputeEngine::create_tensor`（3 参重载 + `InitSpec` 重载）：`check_precision_supported` 失败时
  `fprintf(stderr, "[engine] create_tensor 拒绝非存储精度…"); return Tensor{};`，注释自陈"契约沿用失败返回空 Tensor（调用方查 `valid()`）"。
- **同族已是 `Result`**：`from_matrix` / `to_matrix` / `cast` / `read` / `write` / `import` 全部返回 `Result<T>`。
- **更隐蔽的一层**：`create_tensor_impl` 里 GPU 分配失败也 `if (!r) return Tensor();`——**分配失败的原因被丢弃**（连 stderr 都没有）。
- 调用面：`include/ + src/` 共 **115** 处 `create_tensor(`（2026-10-11 实测）。

**根因**：工厂早于 `Result` 协议定型；"空张量 + stderr"无法把错误结构化地交给调用方（库内"每调用 engine 形参"清理已收口，工厂是最后一块）。

**方案**（**保留旧签名做转发**，审查建议采纳）：

1. 新增 `Result<Tensor> try_create_tensor(rows, cols, P)` 与带 `InitSpec` 的重载（新协议：形状合法性 + 精度支持 + 分配失败都进 `Result`）；
2. `create_tensor_impl` 提升为 `Result<Tensor>`，把 GPU 分配失败原因带上（`GpuTensor::create_empty` 的错误）；
3. 旧 `create_tensor` 保留为 `Result` 版转发 + `NN_UNWRAP` 解包（或标注 deprecated，下一版删）；
4. 分批迁移调用点：批 A = `include/` 库内（`compute_layer_*` 最多），批 B = `src/` + 测试。

**验收**：工厂返回处不再有"空张量 + 查 `valid()`"兜底；新增负例（`T1_58`/`BF16` 请求 → 返回错误、**不打印** stderr、无空张量）；
`ctest` 全绿。**风险**：115 处调用迁移 → 靠编译器穷尽驱动 + `-Werror`，旧签名保留可分期。

**验收证据（RV-02，2026-10-11）**：
① `ComputeEngine::create_tensor_impl` 提升为 `Result<Tensor>`（CPU 侧恒成功、GPU 侧经
`NN_TRY` 把 `GpuTensor::create_empty` 的原因上抛——旧形态 `if (!r) return Tensor();`
连 stderr 都没有）；② 新增 `try_create_tensor(rows, cols, P)` 与带 `InitSpec` 的重载：
覆盖"非存储精度（T1_58/BF16/F64）拒绝 + 形状溢出（rows*cols 超 size_t）拒绝 + 分配失败
原因上抛"，**不再打印 stderr、不再返回空张量**；③ 旧签名 `create_tensor`（两个重载）
保留为 `NN_UNWRAP(try_create_tensor(...))` 转发——115 处调用点零改动即可编译，
失败从"静默空张量"变成 fail-fast（分期迁移到 `try_*`）；④ 库内原先靠 `valid()` 兜底的
4 处（`compute_layer_rapt::{make_dummy_,rms_norm_forward_,rms_norm_backward_}`、
`detail::upload_span`、录制占位创建）改为 `NN_TRY` 直接上抛。
⑤ 负例落在 `src/tensor_precision_test.cpp [5]`：`T1_58`/`BF16` → 返回错误且原因非空、
形状溢出 → 拒绝、`try_create_tensor(InitSpec)` 同协议；本机 Release ctest 全绿。

### RV-03 — 顺序依赖加固：AdamW `decay→update`、稀疏 CE 的原地覆写

- **优先级**：高 · **状态**：`Fixed`（2026-10-11） · **来源**：审查（训练计算"潜在风险点"） · **依赖**：`02-operator-fusion.md`、`03-ir-optimization.md`

**现象与证据**：

- `compute_optimizer.hpp::AdamW::step` 注释自陈：decay（`p = (1−lr·wd)·p`）与 `adam_update_` 是**同目标 p 的两次 `compute_into`、无数据依赖边**；
  当前正确性靠"执行序 = 节点序（稳定拓扑序保持列表序）"。
- `compute_loss.hpp::fused_forward_sparse_` 注释自陈："派发序：lse → loss_sum → grad（`grad_reuse` 原地覆写 logits，前两者须先读完）"。

**根因**：算法正确性依赖**隐式**顺序（列表序恰好 = 拓扑序），未编码成图上的依赖边。一旦 IR-C 图分析开始重排独立节点 / 批量派发（P3 已在做同结构分组），语义会**静默**改变。

**方案**：

1. 在 IR-C 图（`expr_graph.hpp`）显式支持**写后读/写后写（WAR/WAW）顺序边**：同目标存储的多次 `compute_into` 建边，且该边**不得**被重排/分组拆掉；
2. 若图层不便建边，退一步：在"同 dst 的 `compute_into` 必须串行且保持调用序"上写一条**明文契约**，并让分组器显式尊重它；
3. 补**最小回归**：① AdamW（`wd≠0`）单步与参考实现逐位对拍；② 稀疏 CE `grad_reuse` 路径与"不 reuse"路径数值一致；
   ③ **负例**：故意把两个节点反序，门禁必须报错/产生不同结果（证明回归真的能抓）。

**验收**：新增回归全过；`NN_IRC_TRACE=1` 可见依赖边生效（或顺序保证的显式记录）；`ctest` + 字节锚不变（除非算法本就要改）。

**验收证据（RV-03，2026-10-11）**：
① **显式建边**：`ExprGraphNode` 新增 `dst_id`（= `Tensor::storage_id()`，目标传递节点
才有；新占位存储为 0——新增的 `Tensor::storage_id()` 只返回不透明整数，M1 收口不破），
`ExprGraph::add_node` 接收它（`compute_engine.hpp::record_graph_into_node_` 与
`expr_dsl.hpp` 的 scan 分支各自传入 dst）；新增纯函数 `dst_order_constraints(g)`：
**WAW**（同 target 的两次写）+ **WAR**（上次写之后的读者 vs 本次写）+ **写后读**
（读到的外部张量此前被本图写过）三类关系抽成显式边 `(i, j) = "i 必须先于 j"`。
② **融合侧尊重**：`fuse_expr_graph` 用该边集算 `has_hazard`，P2 并查集**拒绝**把隐患
两端并入同一分量、`try_append`（P1 链）与 `comp_append`（P2 分量）**拒绝**把隐患节点
并入同一 kernel；`graph_cache_key` 同时把写目标的**共享结构**混入（同 dst 的节点按
首次出现顺序归组号，**不**直接用指针值——否则 Muon 那类"每步新建临时目标"的图每步
都会换 key、白白丢掉 P2-12 的图级计划缓存），避免"结构相同但写目标共享关系不同"
错误命中缓存的合并计划。
③ **执行期兜底**：新增 `validate_dst_order(g, ordered_kernels)`，在
`GpuEngine::execute_fused_graph` 的稳定拓扑序之后 fail-fast——任何未来的分组/重排
若破坏顺序，**报错而非静默算错**。
④ 回归（新增 `src/optimizer_test.cpp`，已注册 ctest）：AdamW（`wd≠0`）单步与手写参考
**逐位一致**（`max_abs=0.000e+00`）；顺序对调（先更新后衰减）差值 `2.500e-04` ≠ 0
（证明顺序真的影响数值，故显式边非形式主义）；同目标两节点的图分析断言"建出边 (n0,n1)
+ 不合组 + 正序通过、**反序必须报错**"（即审查建议的负例③）。稀疏 CE 的
`grad_reuse` 与不 reuse 路径逐位一致既有覆盖在 `ce_fusion_test`（其派发序是**同步**
提交序、不经 IR-C 图，故本项不涉及新机制，已在 `fused_forward_sparse_` 的注释中点明）。

### RV-04 — 优化器名单单一真源（新增一个优化器不再要改 5~6 处）

- **优先级**：高 · **状态**：`Fixed`（2026-10-11） · **来源**：审查（"重复出现四次"） · **依赖**：无

**现象与证据**（实测 6 处）：

1. `compute_optimizer.hpp::create_optimizer`（工厂 = 唯一真源）；
2. `src/text_train.cpp:463-465` 硬编码白名单；
3. `src/mnist_train.cpp` 帮助文本 + 校验；
4. `gui.py:52` `OPTIMIZER_OPTIONS`；
5. `tools/scan_exprs.cpp:179-180` 优化器 dry-run 列表；
6. `src/mem_probe.cpp:441`（`sgd_momentum || muon` 分支）。

AGENTS §3 已把"新增优化器别忘了这几处"写成铁律级提醒——**提醒本身说明设计有问题**。

**方案**：在 `compute_optimizer.hpp` 导出**元数据表**（`name` / 显示名 / 是否动量 / 需要的状态缓冲数 / 支持的精度槽），
`create_optimizer`、CLI 校验、`scan_exprs` dry-run、`mem_probe` 全部查表；GUI 的选项列表改由 CLI `--help` 动态取
（`bench/gui_cli_audit.py` 已有 GUI→CLI 差集审计），并在该脚本加一条"优化器名单一致性"断言。

**验收**：`rg 'adamwquant'` 在白名单位置只剩元数据表 1 处；`bench/gui_cli_audit.py` PASS；`doc_inventory` / `doc_align_audit` PASS。

**验收证据（RV-04，2026-10-11）**：
① 唯一真源 = `compute_optimizer.hpp` 的 `OptimizerMeta kOptimizerTable[]`（name /
display / beta / state_bufs / half_state）+ `optimizer_table()` / `find_optimizer(name)` /
`optimizer_names(sep)`；② 四处消费点全部查表：`src/text_train.cpp`（白名单 → `find_optimizer`，
帮助文本 → `optimizer_names("/")`，β 打印 → 表的 `beta` 字段）、`src/mnist_train.cpp`（帮助）、
`tools/scan_exprs.cpp`（dry-run 遍历 `optimizer_table()`）、`src/mem_probe.cpp`
（状态缓冲账 → `state_bufs`/`half_state`；其帮助原先漏了 `adamwquant`，查表后自动补齐）；
`create_optimizer` 仍是唯一构造点（构造器无法表驱动，已注释说明"新增优化器 = 表 1 行 + 工厂 1 分支"）。
③ GUI：`gui.py` 的 `OPTIMIZER_OPTIONS` 常量**删除**，改为 `optimizer_options(exe)` ——
运行时执行对应 CLI 的 `--help` 解析（按可执行名缓存；解析口径 = 先去掉全部空白再匹配
`优化器:<a/b/c>`，对帮助排版器的折行免疫）；仅在**取不到可执行文件**时退回
`_OPTIMIZER_FALLBACK`。`bench/gui_cli_audit.py` 新增第 [4] 节断言 CLI 声明 == GUI 兜底名单
（逐项、含顺序），故"改了表忘了改 GUI"会在门禁暴露——**这是对原方案的一处落地偏差**：
原方案要求 GUI 完全动态取，实际保留了兜底（GUI 常在未构建时可打开），改由审计断言收口。
④ 回归：新增 `src/optimizer_test.cpp` 断言"表内每个 name 都能被工厂构造 / 查表命中 /
名单文本与表一致 / 未知 name → nullptr"。

### RV-05 — GPU 后端生命周期：显式 `shutdown()` / 引用计数

- **优先级**：中 · **状态**：`Fixed`（2026-10-18，引擎所有权 v2） · **来源**：审查 P3 · **依赖**：GPU 提权环境（AGENTS §2）

**现象与证据**：`backend/compute_vk_backend.hpp::GpuBackend::instance` 注释 + **`TODO(1.1, L1)`**：
「故意泄漏（进程退出时由 OS 回收）：避免静态析构顺序问题……**若作为库被长驻进程 embed，此泄漏会持续累积**
（`vkDestroyDevice`/`vkDestroyInstance`/`vkFreeMemory` 永不调用）。**恢复方案：显式 `shutdown()` + 引用计数，或进程级一次清理**。」

**方案**：落地它自己写好的 TODO——`nn::vulkan::shutdown()`（幂等）+ `GpuEngine` 析构递减引用计数
（计数归零才 `vkDestroy*`）；Debug 下给 `Tensor` 析构加"引擎存活探测"（或给 `observer_ptr` 加 Debug 校验）。
注意保持铁律 6（录制期引用必须活到 `end_batch`）。

**验收**：`--gpu` 冒烟后显式 shutdown → 无 VVL 报错、无未销毁对象（VVL leak 检查或 `vkDestroy*` 计数）；
`examples/downstream/` 跑通"创建引擎 → 销毁 → 再创建"（作为库嵌入的长驻场景）。

**落地（2026-10-18）**：没有引入单独的 `nn::vulkan::shutdown()`，改用**共享所有权**直接消除故意泄漏：
`GpuBackend::shared()` 返回 `shared_ptr`，`GpuEngine` 与 `GpuBuffer` 都持 backend keepalive，
`GpuBackend` 析构时正常执行 `vkDestroy*`；`reduce_partial_` 改为 backend 自有裸 `VkBuffer`，
切断 `backend → GpuTensor → GpuBuffer(shared_ptr<GpuBackend>)` 的引用环；`Tensor` 只保留 8B
`EngineId`（不复用、无指针），Tensor 晚于引擎析构也安全。验收：CPU ctest **53/53**、GPU ctest
**55/55**（850M）；`layer_lifetime_test` 覆盖 Tensor/Layer/Model/Optimizer 保活与跨引擎拒绝，
`NN_GPU_LIFETIME_FULL=1` 额外验证 `GpuBuffer` 晚于 `GpuEngine`/backend 析构；设计与迁移清单见
[`../development/09-engine-ownership.md`](../development/09-engine-ownership.md)。

### RV-06 — 绑定不变量与 `NN_BIND_DEBUG`：把调试门升为 CI 门禁

- **优先级**：中 · **状态**：`Fixed`（2026-10-12） · **来源**：审查 P4（**标签已修正**，见 §1） · **依赖**：RV-05（若走 `EngineHandle` 方案）

**现象与证据**：`compute_engine.hpp::bind_check_`：**双方都绑定且指针不同 → 硬错误**；**单侧未绑定默认放行**，
只有 `NN_BIND_DEBUG=1` 才升级为错误（用于抓库内漏网与库外迁移清单）。`observer_ptr` 明确"非拥有、无 RAII"，
绑定随拷贝/reshape 传播靠各点自觉。

**方案**：

1. **先评估再定**：把"未绑定输入"默认升级为错误的库外迁移成本（消费方 = `docs/usage/05` + `examples/downstream/`）——成本低就默认开；
2. 若要更强不变量：给绑定加"代数（generation）"或引入 `EngineHandle`（带代号的观察者），init 后统一校验；
3. **不论走哪条**：把 `NN_BIND_DEBUG=1 ctest` 从"手工跑"升为**默认 CI 门禁**（它已是可执行门禁，见 `17-unified-tensor-engine.md` §5 M6）。
   `NN_BIND_DEBUG` 本身**保持**为探针门的既有例外（AGENTS §2），不动它的语义。

**验收**：调试门进入 CI（或在 `18-roadmap.md` 门禁清单登记为必跑）；库内"漏网中转"由审计脚本定期复扫清零。

**验收证据（RV-06，2026-10-12）**：
① **前置修复（库内 stamp 漏网）**：`NN_BIND_DEBUG=1` 门禁原为红——`t1_58_test` / `ternary_gemm_test` /
`model_roles_test` 三个三值用例在 `cast` 处报 "unbound input into engine"。根因 = `matmul_q_impl`（CPU 与
GPU 两份）在"边界 cast 回退"路径里：CPU 递归调用 `matmul_q_impl(...f32)` 后对**未 stamp** 的 impl 产物
直接 `cast(*r, P)`；GPU 对静态工厂产物 `Tensor::from_gpu(...)` 直接 `cast(res, P)`。两处改为先
`adopt(...)`（基类 protected 的库内 stamp 通道）再进公共 `cast`——与 M6 的 `to_prec` 修法同款。
② **CI 门禁**：`.github/workflows/cmake-single-platform.yml` 在 Linux/Windows 各加一轮
`NN_BIND_DEBUG=1 ctest --test-dir build --output-on-failure`（`NN_BIND_DEBUG` 是行为开关，默认构建即生效）。
③ **实测（CPU）**：本机 Release（clang++ 24，`-DNN_ENABLE_TESTS=ON -DNN_ENABLE_GPU=OFF -DNN_ENABLE_NATIVE=OFF`）
`NN_BIND_DEBUG=1 ctest` **40/40**（GPU 用例按 77 skip）；不带开关同样 40/40。
**实测（GPU，NVIDIA CMP 40HX @ `ethan@192.168.1.102`，Windows + clang++，Debug，GPU 启用）**：
`ctest` **42/42**、`NN_BIND_DEBUG=1 ctest` **42/42**（全部 GPU 用例真跑、零 skip，含 `ternary_gemm_test_gpu`
/ `t1_58_test_gpu` / `f16_writeback_probe`）——GPU `matmul_q_impl` 的同一修法得证。
⑤ **顺带修的 Windows 可移植性缺陷**：`src/assert_always_test.cpp::read_log` 用裸 `std::fopen`，
在 clang++ + MSVC STL + `-Werror` 下因 `-Wdeprecated-declarations` **直接编译失败**（这正是 GitHub CI
的 Windows 配置；此前 CI 不编测试故未暴露，本轮在 102 上编全量测试时撞到）→ 改为 `_MSC_VER` 下走
`fopen_s`（与 `core_profile.hpp` 同款）。修后 102 的 clang++ 全量测试构建通过。
④ 未做（方案里"先评估再定"的两条条件项）：把"未绑定输入"默认升级为错误、或引入 `EngineHandle`
代数——当前保持 `NN_BIND_DEBUG` 的探针门既有例外语义（AGENTS §2），不升级默认行为。

### RV-07 — CPU 侧 `ExprOp` 分派收敛（三处 switch → 表驱动）

- **优先级**：中 · **状态**：`Fixed`（2026-10-12） · **来源**：审查（复杂度热点） · **依赖**：无

**现象与证据**（新增一个算子要动的点，实测）：CPU 引擎 **3 处** `ExprOp` 分派——标量求值（`compute_cpu_engine.hpp` ≈1979）、
寄存器融合 J 路径（≈2518）、寄存器融合 `ins` 路径（≈2625）；再加 `expr_spec.hpp`（语义校验）、`expr_opt.hpp`（常量折叠）、
`expr_glsl_gen.hpp`（**5 处** emitter 分支）→ **一个算子 ≈ 9 处**。

**方案**：以 `algebra_ops.hpp`（唯一算子语义源：`nn::ops::*::apply`）为基，建 `constexpr` 算子元数据表；
CPU 三处、`expr_opt` 折叠、GLSL emitter 六处全部查表。

**验收**：新增一个算子只改 1–2 处；`expr_opt_test` / `expr_graph_test` / `fused_gpu_test` 全绿；**字节锚不变**（纯重构）。

**验收证据（RV-07，2026-10-12）**：
① 新增 `expr_op_table.hpp`：`nn::detail::{ExprOpInfo, kExprOpTable, expr_op_info, expr_alu_scalar}` +
  `ExprEmitKind`（Infix/Func2/Compare/Neg/Func1/Select）。每算子**一行**元数据 =
  元数 / 归约标志 / 可折叠 / CPU 一元/二元标量函数指针（直接取 `nn::ops::*::apply`；比较类经
  `detail::cmp::*` 桥接 bool→1.0/0.0）/ GLSL 发射形态 + 符号 + 向量比较内建名。
  末尾 `consteval expr_op_table_consistent()` + `static_assert` 强制本表与
  `expr_spec.hpp` 的 `expr_instr_num_operands` / `expr_op_is_reduce` / `expr_op_reduces_cols`
  逐项一致（写漏同步 → 编译失败）。`expr_instr_num_operands` 加 `constexpr`。
② CPU 引擎（`compute_cpu_engine.hpp`）三处 switch → 查表：fold 标量求值（`exec_alu` 转发
  `detail::expr_alu_scalar`）、寄存器融合 J 路径、寄存器融合 `ins` 路径（后两处按 `info.arity`
  分一元/二元/三元，保持 Select 短路）。
③ `expr_opt.hpp::expr_eval_const_fold` switch → 查表（`info.foldable` 门 + 一元/二元/Select 分流）。
④ GLSL emitter（`expr_glsl_gen.hpp`）`glsl_binary_op`/`glsl_unary_op` 两个符号表函数**删除**，
  改为 `glsl_alu_expr(info, a, b_fn, c_fn, vec4, native16, wrap)` 单一发射器；6 处指令发射块
  （标量主链 / `emit_assign` / 内联 `expr_inl` / 折叠段 `emit_assign` / `emit_scalar_chain` /
  vec4 快路径 / `emit_instrs`）全部改调它（`b`/`c` 以惰性回调传入，保持不使用的操作数不被求值）。
  **新增一个逐元素算子只需改 `expr_op_table.hpp` 一行**（元数非默认 2 时再改 `expr_spec.hpp`
  的 `expr_instr_num_operands`，由 static_assert 兜底）。
⑤ 实测：`build-rv`（Release/clang 24/CPU）`ctest` **40/40**（GPU 用例按配置 skip）。
⑥ **字节锚不变（强口径）**：用 `build/`（Debug/GPU/glslc 1.4.357.1）分别从「RV-07 前」与
  「RV-07 后」两侧源码编 `scan_exprs` 并生成融合产物，`diff -rq` → **805 个文件（fused_registry.hpp
  + 402 组 .comp/.spv）零差异**。即融合 shader 的 GLSL 源码与 SPIR-V 逐字节不变。
⑦ 全量 GPU 构建与 `ctest`（含 `NN_BIND_DEBUG=1`）在 40HX 侧复核（见 RV-06 口径）。

### RV-08 — `compute_vk_backend.hpp`（6397 行）拆分 + SPIR-V getter 表驱动

- **优先级**：中 · **状态**：`Fixed`（2026-10-12） · **来源**：审查（复杂度热点） · **依赖**：无（纯结构）

**现象与证据**：`backend/compute_vk_backend.hpp` **6397 行**；SPIR-V getter 型函数 **42 个**（审查说 36）；
pipeline 缓存、帧环 / solo fence、`pending_destroys_` 延迟销毁队列、各原语 dispatch 同处一文件。

**方案**（先做零风险的两件）：

1. **表驱动**：42 个 `get_*_spirv()` → `{name → embedded header}` 表 + 一次注册；
2. **物理拆分**（纯移动、零行为变化）：`*_pipeline.hpp`（pipeline 缓存/编译）、`*_frame.hpp`（帧环 / solo fence / 命令缓冲）、
   `*_destroy.hpp`（延迟销毁队列决策）、`*_ops.hpp`（原语 dispatch）；
3. 每步拆分后跑三件套（ctest + 字节锚 + scan 双 hash）——**必须不变**。

**验收**：每步拆分后三件套不变；单文件行数目标 < 3000。

**验收证据（RV-08，2026-10-12）**：
① **SPIR-V getter 表驱动（单一注册表）**：42 个 `get_*_spirv()` 的**手写函数体**
  （每个 8 行 `#ifdef NN_X_SPV_EMBEDDED / return nn_x_spirv_bytecode() / #else static empty
  / #endif`）与 42 个 `#define NN_X_SPV_EMBEDDED` 全部删除。文件头的 42 个
  `#if __has_include("<name>_spv.hpp")` 探测块改为把字节码函数绑到 `NN_SPV_FN_<name>`
  （有嵌入头 = 真实 `nn_<name>_spirv_bytecode`；纯 CPU 构建 = `NN_SPV_FN_MISSING` =
  `detail::spirv_missing()` 占位空向量）。`GpuBackend` 内改用 X 宏列表
  `NN_SPV_GETTER_LIST(NN_SPV_GETTER)` 一次性展开出全部 getter。**新增一个原语 shader
  只改两处**：文件头加一个探测块 + 在 `NN_SPV_GETTER_LIST` 加一个 token
  （`get_<name>_spirv()` 名称/签名不变，既有调用**零改动**）。
② **物理拆分（纯移动、零行为变化）**：`GpuBackend`（成员体 ~5375 行）拆为 4 个**片段**
  文件，由 `compute_vk_backend.hpp` 在 `class GpuBackend { ... }` 体内**按序** `#include`
  （片段不是独立头，带 banner 说明）：
  `compute_vk_backend_destroy.inc`（延迟销毁 / reap 单帧 / pipeline 句柄 / 收帧 / 显存回收，948 行）、
  `compute_vk_backend_frame.inc`（帧环 / batch / 阻塞上传下载 / dispatch / matmul，779 行）、
  `compute_vk_backend_pipelines.inc`（RLA 扫描 + 描述符集 / 跨 submit 依赖 / solo fence / 屏障，642 行）、
  `compute_vk_backend_ops.inc`（各原语 dispatch，2681 行）。
  主文件 **6397 → 1055 行**（< 3000），四个片段均 < 3000。
  正确性由**重构等价性检查**背书：脚本比对「主文件 `[:1000]` + 4 片段 + 主文件尾」与拆分前
  文件 → **逐行完全相同**（纯移动）。
③ 实测：本地 `build/`（Debug/GPU/glslc 1.4.357.1）编译 `mnist_train` / `text_train` 通过；
  融合 shader 未重生成（无 shader 相关改动）。102（40HX）全量构建 + `ctest` **42/42**、
  `NN_BIND_DEBUG=1 ctest` **42/42**（全部 GPU 用例真跑）；`bench/doc_inventory.ps1` 与
  `bench/doc_align_audit.ps1` 均 exit 0。
④ 脚注：片段内 `#include` 是本仓首次使用的「单类跨文件」手法（因 `GpuBackend` 是单一
  类，纯移动只能走此路）；若日后倾向「拆成多个协作类」，须单独立项（非纯移动）。

### RV-09 — 环境变量读取样板 → 单一 `core_env.hpp`

- **优先级**：中 · **状态**：`Fixed`（2026-10-12） · **来源**：审查（"≥6 处复制"） · **依赖**：无

**现象与证据**（实测）：库内 **14** 处 `getenv`，分散在 ≥6 份各自 helper：
`expr_dsl.hpp::env_flag`、`core_profile.hpp::env_flag`、`compute_engine.hpp::prec_env_flag`（+ `NN_BIND_DEBUG` 直读）、
`backend/compute_vk_device.hpp::get_env`、`backend/compute_vk_backend.hpp`（复用 device 版）、`compute_layer_base.hpp`（经 dsl 版）。
MSVC 的 `_dupenv_s` 分支在多处重复（`-Werror` 下 `getenv` 已弃用）。

**方案**：新增 `core_env.hpp`：`env_flag(name)`、`env_string(name)`（含 MSVC `_dupenv_s`）、`env_int`；
**保留 AGENTS §2 的两层语义**——探针门下的开关（`NN_F16_DEBUG` / `NN_PREC_TRACE` / …）走"门关**恒 false 且不求值** `getenv`"；
行为开关（`NN_BIND_DEBUG` / `NN_VULKAN_DEVICE` / `NN_POOL_*` / …）走"**读一次 + 缓存**"。

**验收**：`getenv` 只在 `core_env.hpp` 出现；`NN_ENABLE_PROBES=OFF` 的默认构建里探针开关仍恒 false 且**不求值**（双构建 + 反汇编/`strings` 抽查，或加单测）；`ctest` 全绿。

**验收证据（RV-09，2026-10-12）**：
① 新增 `core_env.hpp`：`nn::detail::{env_flag_raw,env_string_raw,env_int_raw}`（原始读取，行为开关用）
  `{probe_flag_raw,probe_string_raw}`（探针门下的读取，`NN_PROBES_ENABLED=0` 时恒 false/空串且**不求值
getenv**）。MSVC `_dupenv_s` 分支只此一份。
② 四处 helper 全部改为转发：`expr_dsl.hpp::dsl::env_flag`、`compute_engine.hpp::{prec_env_flag,
prec_trace_enabled_,bind_debug_enabled_}`、`core_profile.hpp` 的全局开关（`detail::env_flag/env_string`
删除）、`backend/compute_vk_device.hpp::VulkanDevice::get_env`。
③ 口径统一：raw flag = "存在、非空、首字符 != '0'"（`dsl::env_flag` 原为"任意非空即真"，一并对齐；
二者都是探针开关、默认不设，行为不变）。
④ 实测：`rg 'getenv|_dupenv_s' include/neuralnet.cpp` 只剩 `core_env.hpp`（其余为注释）；
mnist_train / text_train 在 `NN_ENABLE_PROBES` 关与开两种编译下均 `-fsyntax-only` 零告警通过。

### RV-10 — BitLinear 量化舍入语义：把**真实**语义写清楚（审查表述有误）

- **优先级**：低 · **状态**：`Fixed`（2026-10-12） · **来源**：审查（"round-half-away"） · **依赖**：无

**现象与证据**：`compute_layer_bitlinear.hpp::quantize_weights_` 用 `select(w > 0.5τ → 1, w < −0.5τ → −1, 0)`；
`ternary_pack.hpp` 同判据 `|w| > 0.5τ`。即**严格大于**、恰好 `|w| == 0.5τ` 归 **0**（半值向零）。
现有注释自称"在 [−1,1] 区间内与 `RoundClip(W/τ,−1,1)` 等价"——**在测度零的等值点上有差异**。
审查说"使用 round-half-away 而非 RN-E"**不成立**（既非 RN-E 也非 round-half-away）。

**方案**：把注释改成精确描述（严格 `>` / 半值归零 / 与 `RoundClip` 的唯一差异是等值点）；**不改算法**。

**验收**：`compute_layer_bitlinear.hpp` 与 `ternary_pack.hpp` 的判据表述一致；`t1_58_test` / `ternary_gemm_test` 不变。

**验收证据（RV-10，2026-10-12）**：
`compute_layer_bitlinear.hpp`（算法块注释 + `quantize_weights_` 内的实现注释）与
`ternary_pack.hpp::code_of` 的判据表述统一为："判据 = **严格** `|W| > 0.5τ`；恰好 `|W| == 0.5τ` 归 0
（半值向零）；与 `RoundClip(W/τ,-1,1)` 的唯一差异在这个测度零的等值点"。**不改算法**（纯注释）。
`t1_58_test` / `ternary_gemm_test` 数值不变（本机 ctest 全绿）。

### RV-11 — `Precision` 枚举 ↔ 序列化 tag / 变体索引：加编译期护栏

- **优先级**：低 · **状态**：`Fixed`（2026-10-12） · **来源**：审查（"隐式耦合缺 static_assert"） · **依赖**：无

**现象与证据**：`precision.hpp` 里有**两套编号**——枚举值（`F16=0, F32=1, BF16=2, F64=3, T1_58=4`）与
序列化 tag（`f32=0, f64=1, f16=2, bf16=3, t1_58=4`），二者**刻意不同**（保持 v4 文件兼容），目前无 `static_assert` 冻结关系。

**方案**：加 `static_assert` 冻结两张映射（枚举值 + tag 表 + `precision_from_tag(precision_tag(p))==p` 全覆盖）；
GPU 侧用 `static_cast<int>(P)` 拼精度签名/报错文本的地方补一致性断言。

**验收**：`precision_type_test` / `precision_test` 补断言；刻意改一处枚举即**编译失败**（负例验证）。

**验收证据（RV-11，2026-10-12）**：
`precision.hpp` 在 `precision_from_tag` 之后加了两组 `static_assert`：① **枚举值**冻结
（`F16=0,F32=1,BF16=2,F64=3,T1_58=4`；GPU 侧用 `static_cast<int>(P)` 拼精度签名依赖它）；
② **序列化 tag** 冻结（`f32=0,f64=1,f16=2,bf16=3,t1_58=4`，沿用 v4 数值）+ 每个精度的
`precision_from_tag(precision_tag(p)) == p` 往返。**负例**：只改枚举值或 tag 表任一处即**编译失败**
（static_assert 触发），强制同步两张映射。

### RV-12 — Loss 直构 IR spec 与 `FoldAnchor` 的时间耦合（**先设计**）

- **优先级**：低（设计评估优先） · **状态**：`Open` · **来源**：审查（耦合度） · **依赖**：`02`、`03`

**现象与证据**：`compute_loss.hpp` 顶部经 `expr_dsl.hpp::FoldAnchor` 在**静态初始化期**自登记 fold 结构；
稀疏 CE 的 `fused_forward_sparse_` 直接构造 IR 层 spec，**绕 `dsl::` 门面** → 与 Layer 的"表达式录制"通道并存（第二条耦合通道 + 时间耦合）。

**方案（两步，先只做设计）**：

1. **设计评估（不改行为）**：① CE 的 fold 能否走与 Layer 相同的 `dsl::` 门面（保持"单一真源"）？
   ② `FoldAnchor` 的登记时机能否由"编译期锚点 + per-layer dry-run/模型 pass"覆盖，从而**删掉**静态初始化期登记？
2. 评估结论写回本文；确有收益才立项改造。

**验收**：给出"做 / 不做 + 理由"；若不做，给出"时间耦合在当前架构下无害"的论证 + 一个回归（重复构造 Loss 不重复登记）。

**补充证据（2026-10-15，P4-4 ① 顺带实测，供问题 ② 用）**：**dry-run / 模型 pass
一条 fold 结构都产不出**——注意力 fold 13 条从 `scan_exprs` 显式块移到 `FoldAnchor`
自登记后，`NN_SCAN_DUMP_SOURCES=1` 显示 14 条 fold 结构**全部**落在锚点侧
（`dry-run 独有` 里 fold = 0）。根因：fold 的结构由**构造实参**（掩码/偏置/视图）决定，
类型层面推不出来。⇒ "删掉静态初始化期登记、改由编译期锚点 + dry-run 覆盖"这条
对本族 fold **不成立**（详见 `docs/history/history.md`「fold 结构全部自登记」条）。

### RV-13 — `Expression` 概念位置（L1 Span 附近）

- **优先级**：低（纯组织） · **状态**：`Fixed`（2026-10-12） · **来源**：审查（布局） · **依赖**：无

**现象与证据**：`Expression` / `BoolExpression` 概念定义在 `expr_dsl.hpp`，但它描述的是 **L1 代数层**（`algebra_span.hpp`）的表达式原语。

**方案**：把概念定义移到 L1（`algebra_span.hpp` 或邻近头），`expr_dsl.hpp` 引用；纯移动 + clean build 验证头文件自包含。

**验收**：`ctest` 全绿；`nn.hpp` 聚合入口不受影响。

**验收证据（RV-13，2026-10-12）**：
`nn::Expression` / `nn::BoolExpression` 从 `expr_dsl.hpp` 移到 L1 的 `algebra_span.hpp`
（`#include <concepts>` 已补）；`expr_dsl.hpp` 经 `algebra_matrix.hpp → algebra_span.hpp` 传递可见，
本头删去定义、留指针注释。`expr_dsl.hpp` 内 12 处 `nn::Expression` / `nn::BoolExpression` 约束零改动即编译。
实测：`mnist_train.cpp`（含 `nn.hpp` 全量）零告警 `-fsyntax-only` 通过；本机 ctest 40/40。

### RV-14 — VK Descriptor Set 回收：专项复审（**只审计，不改代码**）

- **优先级**：中（审计项） · **状态**：`Fixed`（2026-10-15，只审计） · **来源**：审查（"待验证"） · **依赖**：无（代码级审计，不需要 GPU 运行）

**现象与证据**：`compute_vk_backend.hpp` 中 `descriptor*` 出现 **199** 处；`pending_destroys_` 只管 **buffer**
（延迟销毁），descriptor set / pool 的归还路径（如 `:3595` 附近"只归还描述符集；solo fence/cmd 留待下次复用"）
未在文档里闭环，`todo/12` 的盘点也未覆盖。

**方案**：一次**只读**专项审计——列表（分配点 / 归还点 / 池重置点），核对"录制期引用"与"归还时机"是否满足**铁律 6**；
产出结论（表格：点位 / 是否可能 use-after-free / 证据）写回本文；发现真缺陷则另立 RV 编号。

**验收**：审计表格落进本文（或 `../history/history.md`）；结论明确（"闭环 / 有窗口 + 触发条件"）。

**审计结论（RV-14，2026-10-15）= 闭环，唯一窗口见 RV-15**

审计范围 = `backend/compute_vk_backend.hpp` + 四个片段（`_destroy/_frame/_pipelines/_ops.inc`，
共 6151 行）。计数实测：`vkFreeDescriptorSets` **27 处**（destroy 1 / frame 1 / pipelines 5 / ops 20，
`rg -n "vkFreeDescriptorSets" include/neuralnet.cpp/backend/*.inc`）；分配点**唯一**
（`alloc_desc_set`，`_pipelines.inc` 内 `vkAllocateDescriptorSets`）；**池重置点 = 0**
（全仓无 `vkResetDescriptorPool`），池只在 `~GpuBackend` 里 `vkDestroyDescriptorPool`，且其前一行是
`vkDeviceWaitIdle` ⇒ 析构期不存在"池先销毁、在飞帧仍引用集合"的窗口。

生命周期（两条模式，逐点核对）：

| 模式 | 分配 | 归属 | 归还时机 | 可能 use-after-free？ |
|---|---|---|---|---|
| batch（`batch_mode_`，录制期） | `alloc_desc_set` 成功即 `frames_[batch_frame_].desc_sets.push_back` | **帧** | `reap_frame(i)`：① 等/确认该帧 fence → ② 销毁该帧延迟 buffer → ③ 整批 `vkFreeDescriptorSets` | **否**：唯一释放点在该帧 fence 之后 |
| solo（`batch_mode_ == false`） | `alloc_desc_set`（不登记） | 调用点 | `submit_and_wait` 等完 fence 后释放；或"提交前"的错误分支释放 | **否**：前者在 fence 后，后者命令尚未提交 |

27 处释放点的分类（上下文逐点核对，无遗漏）：

- **帧回收 1 处**：`_destroy.inc::reap_frame` 内的批量释放（fence 之后）。
- **solo 权威点 5 处**：`_pipelines.inc::submit_and_wait` 内 4 个错误分支 + 结尾正常释放；该函数只被
  `owns_cmd == true`（⇒ solo）的调用点使用。
- **solo 错误分支 19 处**：`if (!cmd_r) { free(...) }` ×17 与 `if (owns_cmd) {...}` 块内 ×2。
  依据：`acquire_cmd()` 的 **batch 分支不可失败**（只置 `batch_has_ops_` / `ts_cur_` 后返回预分配的帧
  命令缓冲），故这些分支只在 solo 模式可达，而 solo 模式不登记 ⇒ 单次释放，正确。
- **1 处异常**：`_ops.inc` 的 `matmul_q_impl`（`#ks`/`#kr` 路径）在 `if (!ds2_r)` 分支即时释放
  `ds_ks`——**batch 模式下该集合已登记**，且该分支在 batch 模式**可达** ⇒ 双重释放风险，
  另立 **RV-15**。

帧槽复用与等待的闭合性（"录制期引用"如何被保证）：
`start_frame()` 在复用槽 `i` 前先 `reap_frame(i, wait=true)`；`reap_frame` 的顺序是
**等 fence → 销毁该帧延迟 buffer（`pending_destroys_` 按 `pd.frame == i` 过滤）→ 释放该帧描述符集
→ `in_flight = false`**，`vkResetCommandBuffer` 只在 `reap_frame` 之后发生。因此不存在
"集合仍被在飞帧引用、其引用的 buffer 已被销毁"或"命令缓冲已被 reset、集合已被释放"的窗口。
buffer 侧：`needs_deferred_destroy()` = batch 录制中恒真 / 非 batch 时看最近活跃帧是否在飞；
`defer_buffer_destroy` 打**当前录制帧**（batch）或最近活跃帧（solo）标签——标签恒 ≥ 任何可能引用
该 buffer 的帧（后录制的帧不可能引用已析构的 buffer）⇒ 不存在"释放早于在飞引用"的路径。

未提交帧 / 空帧：ring 槽内 `desc_sets` 非空而 `in_flight == false`（如 `end_batch` 的
`vkEndCommandBuffer` 失败路径）时不会被 `reap_completed_frames`/`wait_in_flight` 触及，但**该槽下次
复用时 `reap_frame` 照样释放**（有界，非泄漏）；进程退出由池销毁兜底。（`desc_sets` 也不会跨帧共享：
每个 dispatch 各分配一份。）

池预算：`maxSets = 262144` × `STORAGE_BUFFER 4/set`（`FREE_DESCRIPTOR_SET_BIT`）。耗尽时
`vkAllocateDescriptorSets` 经 `vk_check` 返回错误（**不静默**）；batch 模式下该错误的处理路径即 RV-15。

证据与复现：
`rg -n "vkFreeDescriptorSets" include/neuralnet.cpp/backend/*.inc`（27 处，逐点上下文已分类）；
`rg -n "vkAllocateDescriptorSets|vkResetDescriptorPool|vkDestroyDescriptorPool" include/neuralnet.cpp/backend/`
（分配 1 / 重置 0 / 销毁 1）；`rg -n "reap_frame|start_frame|defer_buffer_destroy" …`
（闭合性三段顺序见 `_destroy.inc` 的 `reap_frame` 与 `_frame.inc` 的 `start_frame`）。
**未做（口径说明）**：本机**无 Vulkan 校验层**（`/usr/share/vulkan/explicit_layer.d/` 只有
MESA/INTEL 层，无 `VK_LAYER_KHRONOS_validation`），故结论是**代码级生命周期分析**，
不是"校验层零告警"对拍；后者需要装有校验层的主机。

### RV-15 — batch 模式下 descriptor set 双重释放（RV-14 审计发现）

- **优先级**：低（条件可达） · **状态**：`Fixed`（2026-10-15） · **来源**：RV-14 审计 · **依赖**：无

**现象与证据**：`backend/compute_vk_backend_ops.inc` 的 `matmul_q_impl` K 切分路径：
`alloc_desc_set(#ks)` 成功后，`alloc_desc_set(#kr)` 失败时**即时**释放 `ds_ks`。
但 `alloc_desc_set` 在 `batch_mode_` 下已把 `ds_ks` 登记进 `frames_[batch_frame_].desc_sets`
（RV-14 表第 1 行）⇒ 该帧被提交并 `reap_frame` 时会**再次**释放同一句柄
（`vkFreeDescriptorSets` 对已释放句柄 = 校验层 VUID 违规 / 驱动 UB）。

**触发条件（三条同时）**：① 处于 batch 录制中；② 池分配在两次 `alloc_desc_set` 之间失败
（`maxSets = 262144` 耗尽或分配错误）；③ 调用方**从该错误恢复并继续**运行（库内训练路径会终止，
故当前实践不可达；受影响的是"记错误、继续跑"的嵌入方）。

**方案**（收敛成 1 个 helper，语义与 `alloc_desc_set` 的登记规则同源）：
新增 `release_desc_set_now(VkDescriptorSet)` = `if (!batch_mode_) vkFreeDescriptorSets(...)`，
把"提交前错误分支"的即时释放全部走它——batch 下改为"留在帧表里由 `reap_frame` 统一回收"
（该帧命令并未绑定这些集合，留着无害）；solo 下行为逐字节不变。

**验收**：`rg` 审计显示"`!cmd_r`/`!ds2_r` 错误分支不再直接调用 `vkFreeDescriptorSets`"；
GPU ctest（本机 850M）全绿；solo 路径行为不变（三值/融合用例覆盖 `matmul_q`）。

**验收证据（RV-15，2026-10-15）**：`compute_vk_backend_pipelines.inc` 新增
`release_desc_set_now(VkDescriptorSet)`（`VK_NULL_HANDLE` / batch 模式直接返回 ⇒ 交给
`reap_frame`）；`_ops.inc` 17 处 + `_frame.inc` 1 处（`dispatch_compute`）的"提交前错误分支"
全部改走该 helper，含 `matmul_q_impl` 的 `!ds2_r` 分支（原缺陷点）与两处一次释放两个集合的
写法（`{ds1, ds2}` 数组 → 两次 helper 调用）。`owns_cmd` 块内的两处直接释放保留并加注释
（该块只在 solo 可达，集合未登记 ⇒ 直接释放正确）。
审计复现：`rg -n "vkFreeDescriptorSets" include/neuralnet.cpp/backend/*.inc` ⇒ 只剩
helper 自身 1 处 + `submit_and_wait` 5 处 + `reap_frame` 1 处 + `owns_cmd` 2 处（共 9 处），
**错误分支 0 处**。验证：本机 850M（Vulkan 路径真的被编译到）全量重建零告警 + `ctest` **49/49**；
102 WSL clang24+libc++ 与 g++15.2+libstdc++ 各 **47/47**（无 Vulkan 树不含该 TU，属旁证）。

---

## 3. 建议批次与顺序

| 批次 | 内容 | 说明 |
|---|---|---|
| **批 A（必修）** | RV-01、RV-02 | 同一错误处理主题：先定"无条件 fail-fast"宏，再统一工厂错误协议。这两项有 Release 下**UB / 静默**风险，优先级最高 |
| **批 B（高）** | RV-03、RV-04 | RV-03 是**未来会静默错**的隐患（趁图优化还没重排）；RV-04 是低风险高收益的收敛 |
| **批 C（中）** | RV-05、RV-06、RV-07、RV-08、RV-09、RV-14（+ 审计派生的 RV-15） | 相对独立、可并行；RV-05 需 GPU 提权环境（RV-14 已完成——代码级审计不需跑 GPU）；RV-08 建议"表驱动 → 物理拆分"分多次提交 |
| **批 D（低）** | RV-10、RV-11、RV-12、RV-13 | RV-10/11/13 是小改（注释 / 断言 / 移定义）；RV-12 只做设计评估 |

**2026-10-11 进度**：批 A（RV-01、RV-02）与批 B（RV-03、RV-04）已 **Fixed**——每条的
验收证据见对应 RV 卡尾部的"验收证据"段。批 C / 批 D 未动。

**2026-10-12 进度**：批 D 的小改三条 **RV-10 / RV-11 / RV-13** 已 Fixed（注释精确化 / 编译期护栏 /
概念移到 L1），批 C 的 **RV-09**（`core_env.hpp` 环境变量单一入口）与 **RV-06**（绑定不变量 CI 门禁）
已 Fixed——RV-06 的前置是修掉 `matmul_q_impl`（CPU/GPU）"impl 产物进公共 `cast` 前未 stamp"的库内漏网
（三个三值用例在 `NN_BIND_DEBUG=1` 下原为红）。**当时仍 Open**：RV-05、RV-07、RV-08、RV-12、RV-14
（RV-05/RV-14 需 GPU 提权环境；RV-07/RV-08 是需单独提交并逐步过三件套的重构；RV-12 只做设计评估）。
**四组「编译器 × 标准库」实测（`ethan@192.168.1.102`，全部全新构建 + ctest + 审计门禁）**：

| 配置 | 编译器 / 标准库 | 结果 |
|---|---|---|
| WSL (Ubuntu 26.04) | clang++ 24.0.0 + **libc++** | 构建 0 告警；ctest **40/40**（GPU 用例跳过：WSL 侧无 Vulkan） |
| WSL (Ubuntu 26.04) | g++ 15.2 + **libstdc++** | 构建 0 告警；ctest **40/40** |
| Windows 102 | clang-cl 23.1.0 + **MSVC STL**（Vulkan GPU 启用） | 构建通过；ctest **42/42**（含全部 GPU 用例） |
| Windows 102 | cl.exe **19.51** (VS18) + **MSVC STL** | 构建通过；ctest **42/42** |

**2026-10-15 进度**：批 C 的 **RV-14**（VK descriptor set 回收专项复审）已 **Fixed** ——
**只审计**：27 个释放点逐点分类 ⇒ **闭环，无 use-after-free、无泄漏**；唯一窗口是 batch 模式下
"错后仍继续运行"的调用方，另立 **RV-15**（Open，收敛成 1 个 helper 即可）。RV-14 结论为
代码级生命周期分析，不再需要 GPU 提权环境（本机无校验层的口径已在卡内注明）。
**2026-10-18 进度**：**RV-05** 已 **Fixed**——引擎所有权 v2 用 `EngineId` + `shared_ptr`
keepalive 取代 `GpuBackend` 的故意泄漏，Vulkan 资源随最后一个强引用析构；GPU ctest 55/55。
**当前仍然 Open**：RV-12（RV-07 / RV-08 / RV-15 均已 Fixed，证据见各自卡片）。

门禁：`bench/doc_inventory.ps1` → `L2-VIOLATIONS: 0`、`HANDWRITTEN-RESIDUE: 0`、
`virtual methods: 52 declarations / 52 unique`（与本文 §4.3 口径一致）；
`bench/doc_align_audit.ps1` → exit 0；`bench/gui_cli_audit.py` → **PASS 可行动问题=0**
（含本轮新增的第 [4] 节）。两组 WSL 构建为纯 CPU（该环境无 Vulkan），与本仓既有口径
（g++ 15.2 侧历来只验 CPU-only）一致；GPU 路径由 102 的 Windows 两组覆盖。

**顺带修的构建阻塞（不在原立项内）**：`core_profile.hpp::Phase` 原先用
`std::optional<Scope>`，GCC 15 在 `~optional → _M_destroy → ~Scope → stop()` 的内联链上
误报 `-Wmaybe-uninitialized`（libstdc++ 的 `optional::emplace` 先置 engaged 后构造载荷），
`g++ -Werror` 下 `text_train.cpp` 直接编译失败（WSL 的 gcc+libstdc++ 配置阻塞项）。
改法：`Phase` 不再持 optional，状态内联为两个 POD 成员 + 抽出 `detail::close_frame`
供 `Scope::stop` / `Phase::stop` 共用（语义等价，stop 仍幂等）。
**另一处构建阻塞（MSVC cl）**：目录级 MSVC 选项写的是 `/source-charset:utf-8
/execution-charset:utf-8`，而 `neuralnet_cpp` 的 **INTERFACE** 选项又给链接它的目标
追加 `/utf-8` → 同一条命令行同时出现两者，cl 直接报
`D8016: "/source-charset:utf-8"和"/utf-8"命令行选项不兼容`（MSVC 19.51 / VS18 实测：
**凡链接本库的目标全数编译失败**，msvc+MSVC STL 配置此前不可用）。改法：目录级与接口级
统一用 `/utf-8`（实测 `cl /utf-8 /utf-8` 合法，只有"混用两种写法"才冲突）。

**建议**：批 C 里的 RV-07 / RV-08 / RV-09 是三项**重构**，务必单独提交、每步过三件套（ctest + 字节锚 + scan hash），
避免与功能改动混在一起——否则一旦锚动，无法区分是重构还是行为变化。

---

## 4. 不立项 / 正面结论（不要重复处理）

- **训练数值正确性**：审查给的正面结论（softmax/LayerNorm 数值稳定、CE 梯度缩放、Adam 偏差修正时机、AdamWQuant 尺度上界自洽、
  Muon Newton-Schulz 系数）与本仓既有验收一致（`precision_test` / `t1_58_test` / `ternary_gemm_test` / 各 `*_gradcheck` / `ce_fusion_test` 等）。**不立项**。
- **"无结构性内存泄漏" vs RV-05 的"单例故意泄漏"**：二者不矛盾——训练/推理路径无泄漏；`GpuBackend` 单例曾是**已知设计取舍**，
  且代码里已有 `TODO(1.1, L1)` 与恢复方案；RV-05 已于 2026-10-18 由引擎所有权 v2 落地（`EngineId` + `shared_ptr` keepalive）。
- **CUDA**：本仓**不支持 CUDA**（后端只有 CPU / Vulkan，AGENTS §2）——整改中**不得**引入任何 CUDA 相关改动或文档表述。
- **审查建议"旧签名保留为转发"**：采纳，已进 RV-02 方案 2。

---

## 5. 与 `18-roadmap.md` / `history/history.md` 的关系

- **立项的唯一入口仍是 [`18-roadmap.md`](18-roadmap.md)**（"下一步该做什么" + 做/条件触发/不做 的裁定）。
  本文是**它的执行细节来源**：整改某条时读本文对应 RV 卡，**不要**在 18 号里再抄一份（避免两处漂移）。
- **完成一条**：把结论与验收证据（数字、hash、日志路径）归档到 [`../history/history.md`](../history/history.md)，
  并在本文把该 RV 状态改成 `Fixed`（带证据坐标）；`18-roadmap.md` 只在"是否仍然要做"改变时更新一行指针。
- **仓库纪律**：任何一条的验收都走本文 §0 的通用基线；GPU 相关按 AGENTS §2 在提权环境跑。
