# 仓库历史记录（唯一归档）

> 本文档是全仓**唯一**允许记录历史状态的地方：代码注释与其余文档只描述
> 「当前是什么、为什么这样实现」；被移出的演进过程、已修复勘误、被否决方案、
> 性能 A/B 过程一律归档到这里。新增历史内容请追加到对应分组，**不要**回流到代码注释。

按来源分组；每条注明**整改前的原位置**（`文件:行` 或章节），行号以整改前版本为准。
类型标注：bug 根因 / 否决方案 / 性能 A/B / 删除清单 / 演进记录 / 覆盖缺口。

> **2026-10-11 文档整理说明**：`docs/development/` 的定位改为**面向调用者**（库使用者 /
> 扩展者）的开发文档，原其中的内部设计 / 路线图 / 盘点 / 规范类文档已迁至 `docs/todo/`；
> 其中 `15-computeengine-refresh.md`、`16-computeengine-p0-inventory.md` 两篇**整篇并入本文件**
>（见文末「docs/development 15 · 16」归档节），`13-refactor-backlog.md`（早先已降级为重定向 stub）
> 已删除——其内容本就在本文件与 `docs/todo/18-roadmap.md`。本文中出现的
> 「原位置 docs/development/NN…」字样均为**迁移前**路径，勿当现行路径。

## 目录

- [表达式 DSL / 融合 IR（expr_*.hpp）](#表达式 DSL / 融合 IR（expr_*.hpp）)（18 条）
- [GPU 后端与 GPU 引擎（backend/*, compute_gpu_engine.hpp）](#GPU 后端与 GPU 引擎（backend/*, compute_gpu_engine.hpp）)（17 条）
- [注意力 / ZiPT / Transformer / FeedForward 层](#注意力 / ZiPT / Transformer / FeedForward 层)（16 条）
- [RAPT / CNN / MLP / Softmax / GPT 层](#RAPT / CNN / MLP / Softmax / GPT 层)（13 条）
- [引擎接口 / CPU 引擎 / 精度 / 张量 / 损失 / 优化器](#引擎接口 / CPU 引擎 / 精度 / 张量 / 损失 / 优化器)（23 条）
- [代数层 / 基础设施 / 模型容器与序列化](#代数层 / 基础设施 / 模型容器与序列化)（15 条）
- [领域模型工厂与 CLI 公共头](#领域模型工厂与 CLI 公共头)（3 条）
- [src 应用入口（text/mnist/tokenizer/bench）](#src 应用入口（text/mnist/tokenizer/bench）)（5 条）
- [GPU 手写原语 shader（shaders/*.comp）](#GPU 手写原语 shader（shaders/*.comp）)（12 条）
- [构建期工具 / 基准脚本 / 对拍脚本](#构建期工具 / 基准脚本 / 对拍脚本)（14 条）
- [表达式与融合测试](#表达式与融合测试)（18 条）
- [层 / 模型测试](#层 / 模型测试)（14 条）
- [精度 / RAPT / ZiPT / 分词器测试与探针](#精度 / RAPT / ZiPT / 分词器测试与探针)（11 条）
- [docs/development 12 · 13 · 08](#docs/development 12 · 13 · 08)（17 条）
- [docs/development 02 · 03 · 05](#docs/development 02 · 03 · 05)（19 条）
- [docs/development 01 · 04 · 06 · 07 · 10 · 14](#docs/development 01 · 04 · 06 · 07 · 10 · 14)（25 条）
- [docs/development 15 · 16（ComputeEngine Refresh 设计与 P0 盘点，归档）](#docs/development 15 · 16（ComputeEngine Refresh：设计与 P0 盘点，2026-10-11 归档）)（2 条）
- [docs/introduction · usage · benchmarks](#docs/introduction · usage · benchmarks)（46 条）
- [构建系统 / CI（CMakeLists.txt、.github/workflows）](#构建系统 / CI（CMakeLists.txt、.github/workflows）)（8 条）
- [AGENTS.md](#AGENTS.md)（24 条）
- [ZiPT（AttnZip）移除](#ZiPT（AttnZip）移除（2026-10-01）)（1 条）
- [GUI / CLI 参数一致性清理](#GUI / CLI 参数一致性清理（2026-10-01）)（1 条）
- [分支归档与 legacy 命名空间整理](#分支归档与 legacy 命名空间整理（2026-10-02，v1.6.0 发布时）)（1 条）
- [统一数据集格式与加载（19 号设计落地）](#统一数据集格式与加载（19 号设计落地，2026-10-05/06）)（1 条）
- [训练 step 剖析器（NN_PROFILE）](#训练 step 剖析器（NN_PROFILE，2026-10-06）)（1 条）
- [matmul 细 tile 变体：P0 wgrad 派发饥饿落地](#matmul 细 tile 变体：P0 wgrad 派发饥饿落地（2026-10-07）)（1 条）
- [设备表 per-key 归因（NN_PROFILE 新增表 5）](#设备表 per-key 归因（NN_PROFILE 新增表 5，2026-10-07）)（1 条）
- [fold v2 恒等三改（P2-1）与注意力 fold 相位消融](#fold v2 恒等三改（P2-1）与注意力 fold 相位消融（2026-10-07）)（1 条）
- [fold v2 vecacc 双路径改写（P2-2）](#fold v2 vecacc 双路径改写（P2-2）：j 分片跨线程 + 标量累加器（2026-10-07）)（1 条）
- [matmul K 切分二次归约（`#ks`/`#kr`）](#matmul K 切分二次归约（`#ks`/`#kr`）：wgrad 派发饥饿第二刀（2026-10-07）)（1 条）
- [录制窗口排空隐藏 + SpecBuilder 输入槽去重（项目 4/5）](#录制窗口排空隐藏 + SpecBuilder 输入槽去重（项目 4/5，2026-10-08）)（1 条）
- [代码审查整改 批 A/B（RV-01..RV-04，2026-10-11）](#代码审查整改 批 A/B（RV-01..RV-04，2026-10-11）)（1 条）
- [仓库历史整理（legacy 接入 + 大文件剔除，2026-10-18）](#仓库历史整理（legacy 接入 + 大文件剔除，2026-10-18）)（1 条）
- [生命周期 v3 与 Session 层（v1.10.0，2026-10-20/21）](#生命周期 v3 与 Session 层（v1.10.0，2026-10-20/21）)（2 条）

---

# 表达式 DSL / 融合 IR（expr_*.hpp）

表达式 DSL / 融合 IR 家族头文件（expr_dsl / expr_emitter / expr_fold / expr_glsl_gen /
expr_opt / expr_registry / expr_spec）「历史状态类注释」摘录。整改原则：注释只记录当前实现的
原理、契约与理由；本文件保存被移出的历史叙事（bug 根因、被否决方案、性能 A/B 过程、
删除清单），供追溯用。

## fold 结构全部自登记：注意力组合与 v1 样例改由定义方挂锚（P4-4 ①，2026-10-15）

- 类型：演进记录 / 覆盖缺口收口（"手写登记块漏项 = GPU 闭合世界硬报错"的最后一个已知点）
- **原状**：注意力 fold 的 5 个组合（3 掩码 × 2 位置偏置 − 1）× {MHA, GQA} 由
  `tools/scan_exprs.cpp` 的显式登记块覆盖（连同 `expr_fold.hpp` 的 3 个 v1 样例，共 13 条结构）。
  层 forward 直调 `engine.eval_expr(make_fold_attn_o(...))`、不经 DSL 钩子，构造点没有登记
  通道 ⇒ 新增掩码/组合若忘了改那个块，只会在 GPU 运行期以闭合世界硬报错暴露
  （历史先例：P-C2 漏 Doc/AlibiDoc → GPU doc 训练硬报错）。
- **改动**：把登记搬到**定义方**（与 `FoldAnchor` 既有约定一致，`compute_loss.hpp` 的
  CE logsumexp 即此形态）：
  ① `compute_layer_attention.hpp`：掩码枚举改 **X 宏单一真源** `NN_ATTN_MASK_KIND_LIST`
     （枚举成员与登记表由同一清单展开）；新增 `namespace attn_fold_anchor` 的
     `register_all()`，在**静态初始化期**经 `FoldAnchor` 登记「清单里的每个掩码 ×
     {无偏置, 位置偏置} × {kv_ratio=1, 2}」——"Plain + 偏置"（本仓不构造）跳过，共 10 条。
     形状取代表值 `seq=64/dk=2/bh=2`（k/dk/bh 与 kv_ratio 的具体值都不进 key，见 `expr_spec_key`）；
  ② `expr_fold.hpp` 末尾同款自登记它的 3 个 v1 样例（rowmax/rowsum/softmax_denom）——
     它们不被 Layer 使用、没有别的消费方可挂锚；`expr_dsl.hpp`（`FoldAnchor` 所在）只在
     `NN_EXPR_SCAN` 段内包含，普通构建仍只依赖 `expr_spec.hpp`；
  ③ `scan_exprs.cpp`：fold 显式登记块**整体删除**（现在只剩 matmul-trans 与测试覆盖两类显式块），
     台账行的 "fold 显式 +N" 分段随之删除。新增的两个自登记点都受 `-DNN_SCAN_NO_ANCHOR` 门控
     （`compute_loss.hpp` 的 CE logsumexp 锚点早于该开关，不受其门控——证据③里残留的正是它）。
- **证据**（同树 A/B，本机 Debug+GPU 构建）：
  ① **结构集合逐条零差异**：改动前后各跑一遍 `NN_SCAN_DUMP_SOURCES=1`，126 条结构的
     key + 指令摘要逐条相同；来源标签的迁移**恰好**是 13 条（`[dry-only]` → `[anchor-only]`，
     全部 `fold=1`），其余 113 条标签不变。台账从
     `dry-run 侧 103（…fold 显式 +13）+ 锚点 79（新增 23）` 变为
     `dry-run 侧 90（per-layer+显式 58、模型 pass +32）+ 锚点 92（新增 36）`，合计仍 **126**。
  ② **生成物只差发射顺序**：两份 `fused_registry.hpp` 行多重集**逐行相同**（19001511 B；
     发射顺序随插入顺序变化）；同一二进制连跑两次**逐字节一致** ⇒ 扫描本身是确定性的。
  ③ **锚点是唯一来源**（反证）：单独把 `scan_exprs.cpp` 以 `-DNN_SCAN_NO_ANCHOR` 编译并链接，
     结构 126 → **91**，上述 13 条 fold 结构**全部消失**（fold 只剩 CE logsumexp 的 1 条，
     它的锚点在 `compute_loss.hpp`、不随该开关关闭）。
  ④ 新增回归 `src/expr_fold_test.cpp` §8「锚点代表值契约」：同一组合跨形状
     （`{64,2,2}` vs `{512,64,8}`、GQA 比 2 vs 4）key 必须相同、10 个组合两两不同
     ——锁住"登记一组代表值即覆盖所有形状"这一前提（11 条 PASS）。
  ⑤ 本机 clang + libstdc++（Release，`build-ab`，GPU 关）：全量 `ctest` **46/46 通过**
     （含 §8 新回归；GPU 标签用例在该构建下 skip）。
  ⑥ 本机 GCC 15.2 + Vulkan（Release，`build-gpu`，设备 = **NVIDIA GTX 850M**，自动选卡）：
     全量 `ctest` **48/48 通过**（含 `-Werror` 下的全新构建）；其中 `fused_gpu_test` 的
     **5 组合 × 5 形状**注意力 fold 用例、通用 fold v1 用例与"未扫描表达式 → GPU 硬报错"
     用例全过 —— 注意**服务形状 (bh,seq,dk) 与登记代表值 (64,2,2) 不同**，这同时证明
     "一组代表值覆盖任意形状"在真实注册表上成立。
- **边界**：登记表仍由"掩码清单"驱动（C++ 无反射）——新增掩码种类要在 X 宏加一项，但枚举与
  登记表**同源展开**，因此"加了枚举忘了登记"这一漏项形态已不可能出现；GQA 比本身是运行期
  vp 参数（不进 key），登记 kv_ratio=2 一条即覆盖任意 `n_rep`（docs/todo/22 §3.2 P2）。

## `dsl::compute` / `compute_reduce` 无形状重载（P4-4 ②，2026-10-15）

- 类型：演进记录 / API 摩擦点收敛
- **原状**：`dsl::compute(engine, expr, rows, cols[, P])` 必带输出形状，调用方反复手写
  `x.rows(), x.cols()`；`AGENTS.md` §7 曾把它列为"仍然生效的工程教训"（本文件另有
  「教训：`dsl::compute` 无 2 参重载」条——早期一次 eager→DSL 迁移正是在这里写错形态）。
- **改动**：新增 `dsl::compute(engine, expr[, P])` 与 `dsl::compute_reduce(engine, expr[, P])`；
  输出网格由表达式推导（`expr_dsl.hpp` 的 `GridInfer`）：各"网格贡献者"按**广播规则**
  （逐维相等或其一为 1）合并 —— 张量叶/形状保持视图给张量形状；归约视图与归约指令给
  **输入整网格**（`compute` 下它们是广播读；`compute_reduce` 要的也正是输入网格）；
  `grouped_reduce(t, R)` 给 `(rows/R, cols)`；matmul 段给 `(batch·M, N)`（GQA 的
  `batch/ratio` 行块与 `to_spec` 的 `k` 同口径；A/B 本身不参与合并）；标量、网格索引叶
  （`row()`/`col()`/`batch()`）与稀疏/批视图（RowGather/BatchMod/BatchCol）中性。
  **形状冲突、无法推导、全无贡献者 = 返回错误**并提示改用显式形式（不猜形状）。
- **库内接线**：6 处调用点改为无形状形式（`Linear` fwd/bwd、`ReLU` fwd/bwd、`Softmax` fwd/bwd、
  `PlainScoreMask::masked_scores` 的 Q·Kᵀ——后者形参 `seq` 随之写成 `/*seq*/`），其余调用点
  保持显式形式。表达式文本与形状取值逐字不变 ⇒ **AOT 键不变**（锚点 `FusedAnchor<Expr>`
  经 4 参重载照常 odr-use）。
- **证据**：
  ① 新增回归 `src/dsl_shape_infer_test.cpp`（推导 == 显式形状、逐位值一致、3 参 P 透传、
     负例：形状冲突 / 无网格贡献者报错）。
  ② **结构集合逐行比对不变**：未改动的 v1.8.2 树 vs 本改动树，`NN_SCAN_DUMP_SOURCES=1`
     的两份 dump 归一化树目录名后**除输出路径外零差异**（各 126 条结构；来源分项
     dry-run 103 + 锚点 79 亦相同）。
  ③ 本机 clang + libstdc++：CPU ctest 全绿。
  ④ 40HX（clang + MSVC STL，Windows）：全量 ctest 47/48 —— 唯一失败的
     `cli_precision_warning_test` 在**未改动的基线上同样失败**（`std::system` 启动 app 子进程
     处即打 "cannot find the path specified"，Windows 构建布局下的既有问题，与本次改动无关）。
- **未做（当时仍属 P4-4 ①）**：注意力 fold 的 5 组合自登记 —— 随后由本文件上一条
  「fold 结构全部自登记」收口。

## step 内固定 I/O 开销归因：实测推翻"固定开销掩盖"（2026-10-14，P2-9 / 26 号 S0 归档）

- 类型：性能 A/B / 否决方案（**已评估-不做**）
- **被测前提**：`NN_PROFILE` 显示每步约 20% 墙钟在 `copy_from`、`flush_batch`+`begin_batch`+`copy_from`
  ≈88% host 墙钟 ⇒ 判断为"可消除的固定开销，且会把计算侧收益吞掉"（26 号 S0 曾因此被定为先决项）。
- **实测（850M，`d64/h4/L4/ff256/seq256/b16`，15 步）**：
  ① 把 staging ring `64MB×4` → `8MB×16`（消除 `StagingRing::acquire` 的 fence 阻塞）：
     `copy_from` **392 → 3 ms**，但 `flush_batch` **955 → 1213 ms**，两者之和恒 ≈1347 ms；
     墙钟 3 轮 2200/2059/2053 vs 原 2044（噪声内）→ **改动已回退**。
  ② `--flush-interval 1 → 4`（每步 flush 105 → 20 次）：墙钟 **2044 → 2096 ms（+2.5%）**。
  ③ 设备忙率 **95.9%** ⇒ host 的时间几乎全在等 GPU，等待点可互相搬家。
- **结论**：`copy_from` 不是额外成本、减少帧数反而更慢；真天花板 = `墙钟 − 设备 kernel`
  ≈ **86 ms/15 步 ≈ 4.2%**（f32/f16 同幅）。**另一条推论**：26 号"S0 未完成前任何计算侧收益
  都不可信"的前提同时作废——S1s 在 S0 未做时已实测拿到 `--f16` +10.6% → +0.9%。
- **保留证据**：`--flush-interval` 默认值的"1 更快"注释由此获得独立复现。

## f16 网格粒度变体叠加到精度特化签名（2026-10-14，分支 `dev/f16_fix`）

- 类型：性能 A/B / 演进记录（P2-13）
- **背景（缺口）**：`#f`（细 tile）与 `#ks`/`#kr`（K 切分）**只对 V0 全 f32 发射**
  （`generate_glsl_matmul` 对 `k_split && sig != 0` 直接拒绝）。于是**网格饥饿形状**
  （大 batch 的 wgrad）在 f16 下没有 K 切分可派发 → 掉回 64-tile。
- **实测（40HX，b64）**：`71733007a831714a@fused_mm@bwd.blk#*` f32 `#ks` **25 ms/块**、
  f16 掉回 64-tile **79 ms/块**（4 块 +368 ms = 该机 f16 的全部残差，其余结构 f16 更快）。
  配对旁证 `NN_MM_NO_KSPLIT=1` → 细 tile **84 ms/块** ⇒ 细 tile 不够，**只有 K 切分能救**。
- **修法**：sig 穿进三条生成路径，为每个**静态签名变体**各发一份网格变体（键
  `key#10003#ks`）。**部分和缓冲恒 f32**（后端 `use_ks` 分支按 f32 分配）→ `#ks` 的输出
  声明钉成 `float` 并写原始部分和，注入 sig **只决定 A/B 读取类型**；`#kr` 跨段升序求和后
  再按 sig 落盘（含 RNE 预舍入）。**两条易错点**：① 后缀分层 ⇒ 后端"网格变体无需 16 位
  存储"的豁免必须收窄到**纯 V0 键**（`key#10003#ks` 里仍有 `float16_t` 缓冲，误豁免会在
  无 16 位存储的设备上建出非法 pipeline）；② `#kr` 里 `glsl_f16_store` 依赖 `nn_f16_rne`
  声明，漏发即 glslc 编译失败（已补）。运行期 `#x` 哨兵**不**叠加网格变体。
- **验收**：40HX 该 wgrad **f16 16.0 ms/块 < f32 25.1 ms/块**；端到端 40HX `--f16`
  **1013 vs f32 1027 ms（−1.4%）**、850M（b16）**2005 vs 2038 ms（−1.6%）**；loss 对拍
  （40HX 三点）f32 8.6757/8.2646/7.8743 vs f16 8.6676/8.2569/7.8643（<0.15%，容差内）；
  `ctest` GPU 43/43。变体数：细 tile 14→**46**、K 切分 14→**46**（fused shader 617→713）。
- **K 切分的数值口径不变**：跨段累加序仍 ≠ 逐 k 串联 → 仍属"移动 GPU 字节锚"的既有裁定
  范围（torch 同样不保证等价浮点计算逐位一致）；本次**未**新增逐位口径变化（部分和仍 f32）。

## 精度特化回归：扫描期签名 → 编译期特化 shader（2026-10-13，分支 `dev/f16_fix`）

- 类型：性能 A/B / 演进记录（**部分回退** 2026-10-01 那条"签名从身份降级为参数"的取舍）
- **背景（被本条目修正的判断）**：2026-10-01 的取舍把"每个 (结构, 签名) 一份 shader"改成"一个
  `#x` 覆盖任意签名"，理由是构建期不必枚举签名、显存不涨。代价当时**未量**：`#x` 的热循环里每次
  读写都带 `((prec >> i & 1u) ? float(bNh[…]) : bN[…])` 三元，编译器无法化简成单条路径。
- **实测（850M，`d64/h4/L4/ff256/seq256/b16`，15 步 `NN_PROFILE=1`；只换 shader、不改任何数值口径）**：
  CE 融合注意力 `f7e6c2bd8e35af36` **146.1 → 120.0 ms**（f32 基线 117.2）；LM head
  `c2aca8a29eb4617f` 154.4 → 137.4（f32 137.1）；wgrad `71733007a831714a` 143.9 → 138.8
  （f32 140.6）。端到端：`--f16` step 墙钟 **2260 → 2055 ms**（f32 2037；即 **+10.6% → +0.9%**），
  峰值显存 **818 → 606 MiB（−26%）**。
- **落地**：生成期追加两类**静态签名**变体——`key#s`（全输入 f16 + 输出 f16，无条件发射）与
  `key#xxxx`（扫描期**真实观察到的签名**：dry-run **与模型 pass 各跑 f32 + f16 两遍**，
  `scan_register` + CPU 引擎 `eval_expr*` 入口按真实张量精度 + 真实 P 记入 `ExprRegistry::variants`）。
  运行期查找序 = `#a` → `#xxxx` → `#s` → `#x` → 边界 cast。观察 110 条签名 → 发射 88 条
  （22 条全 f16 与 `#s` 同 shader 跳过）；精度变体总数 289 → 377。
- **与 2026-10-01 的差别（为什么不冲突）**：签名**仍不是身份**（`#x` 保留为任意签名的兜底，
  未枚举到的签名照旧命中它），特化集合只是**性能优化清单**。因此旧的"无需回填清单 / 端到端
  `[prec][miss]=0` / 新增调用点零改动"结论全部继续成立。
- **已知缺口（立项，不在本条修复范围）**：`#f`（细 tile）与 `#ks`/`#kr`（K 切分）按设计只发 V0
  全 f32 → 网格饥饿形状（大 batch 的 wgrad）在 f16 下掉回 64-tile：40HX 实测 25 → 79 ms/块（×4）。
  **40HX 因此仍是 `--f16` +30%**（其余结构 f16 反而更快，净 −50 ms）→ P2-13；优化器批量派发
  `#b` 同样只发 f32 → P2-12。
- **回归**：`ctest` CPU 41/41 + GPU 43/43（`f16_precision_test` / `ce_fusion_test` / `expr_fuse_test`
  全绿）。

## 精度签名从构建期枚举 → 运行期参数（2026-10-01，分支 `feat/fused-runtime-prec-dispatch`）

- 类型：演进记录 / 删除清单 / 性能 A/B
- **背景（被取代的机制）**：精度签名（哪些输入/输出是 f16，`ExprPrecSig`）由运行期张量精度决定、类型层面不可见，因此"每个 (结构, 签名) 一份 shader"必须靠**执行**发现。旧链路 = `FusedAnchor` 只登记结构 + dry-run `profile_f32`/`profile_f16` **两遍** + 模型级配置矩阵两个 profile + `NN_PREC_TRACE=1` 的 `[prec][miss]` → 手工写 `tools/prec_backfill.txt` → 重新构建。清单曾有两批：注意力 fold 文档掩码 5 输入形态（`74a6eaacc0e5d766 10007`）、MNIST/CNN f16 训练 8 条"f32 入 + f16 出"混合签名（`59e079001367d11b 10000`、`fe93c0d99c16113f 10000/10002`、`cb3d830f25b84895 10001`、`b7e6d363af1963ac 10000/10002`、`8a29d02213a0e196 10003`、`27c7edd7a0a4f808 10007`）。旧注册表 84 结构 + 86 签名变体 = 170 条 / 3.43 MB。
- **新机制**：V1「运行期精度分派」shader（键 `key#x`）对每个输入/输出各声明 f32 + `float16_t` **双视图**（同一 `VkBuffer` 绑两次，只解引用与真实元素类型一致的那一个），加载/存储按 push constant `uint prec` 走 uniform 分支；**索引数学逐字不变** ⇒ 与旧带类型变体数值等价。native16 改按**结构谓词**生成（键 `key#a`，代入"全输入 f16 + 输出 f16"常量签名）。新注册表 84 结构 + 84 分派 + 35 native16 = 203 条 / 4.36 MB。
- **删除清单**：`tools/prec_backfill.txt`（整文件）、`scan_exprs` 的 f16 dry-run 遍与 f16 模型 pass、"没有带类型变体 / 用 NN_PREC_TRACE 回填"提示、`supports_expr_precision_variant` 里的 (key,sig) 探测、旧 `key#<sighex>` 变体发射与 run-only 回填分支。
- **性能 A/B（决定 A 而非 B 的依据）**：曾实测「同质化 + 边界 cast」（方案 B，数值与混合变体等价）在 GPT f16 1 epoch / `tinystories_smoke` / 40HX 上为 4.1s vs 变体路径 4.0/3.9s（**≈+4%**，38 个签名落回 cast）；但 B 会给含 f16 输入的表达式造 f32 临时量、推高 f16 峰值显存，故最终选 A（运行期分派，显存不涨）。
- **实测验收**：build 零告警；ctest 19/19；CPU 锚 `6f8849f14da23110` / GPU dev2 锚 `8ef51b2927253c50` 逐位不变；GPT f16 训练 `NN_PREC_TRACE=1` → `[prec][miss]=0`、边界 cast 归因表为空、`[prec][alu-hit]` 6670；**f16 扫描遍对结构贡献为 0**（`NN_SCAN_F32_ONLY` 探针：跳过它签名 66 → 0、结构恒 84）；**端到端**：Layer 头里新写一个 dry-run 覆盖不到的结构 → 锚点 59→60、结构 84→85、注册表 +`key`/`key#x`/`key#a`，**其他文件零改动**。
- **未做（另案）**：把"收集 + 生成"做成库自带 CMake 能力 `nn_enable_gpu_fusion(target)`（下游自定义层）。**结构侧的关键约束已测明**：84 个结构里只有 13 个来自锚点，71 个来自 dry-run/模型 pass ⇒ 该收集器**必须保留模型 pass**，做不到"30 行纯锚点收集器"。

## 库自带的融合能力 `nn_enable_gpu_fusion`（2026-10-01，A2）

- 类型：演进记录 / 新增能力
- **背景（缺口）**：本库 header-only，但 `fused_registry.hpp` 的生成**绑死在本仓库**（`CMakeLists.txt` 硬编码 `tools/scan_exprs.cpp` + `tools/prec_backfill.txt`，且 `nn_core` 无 install/导出）——库外使用者写自己的 `dsl::compute` 后**没有任何合法途径**产出自己的注册表，只能 fork。这是"导入头文件即可用"真正卡住的地方。
- **落地**：CMake 函数 `nn_enable_gpu_fusion(<target> MAIN <含 main 的源> [OUT_DIR <目录>])`。收集器 = **库内收集逻辑**（`tools/scan_exprs.cpp`）+ **本目标的全部 TU**（`-DNN_EXPR_SCAN` 编译，令使用者调用点经 `FusedAnchor` 自登记）链接成一个可执行文件，产物目录只作该目标的私有 include 目录。样例 `examples/fusion_custom_layer{,.hpp,_main.cpp}`（"库外使用者"形态：自定义层 + 一行 CMake）。
- **实测**：样例目标的注册表 = 库内 203 条 + 自研层 `94a2cc052598fe09` / `94a2cc052598fe09#x`（该表达式含 rparams → native16 谓词不通过，无 `#a`，符合预期）；样例可执行文件运行结果与期望值 `0.6712035` 逐位一致。库内文件零改动。
- **顺带修的 header-only 缺陷**：`backend/compute_vk_backend.hpp` 的 `GpuBuffer::~GpuBuffer()` 类外定义**缺 `inline`** → 任何 ≥2 个 TU 的程序链接期 `duplicate symbol`（样例两 TU 复现）。已加 `inline`。这条此前从未暴露，因为库自身的可执行目标都是单 TU。
- **GPU 端到端验收（本轮补齐）**：样例原先只验 CPU——而 CPU 走模板求值、与注册表无关，等于**没验到本 PR 的核心机制**。补上 GPU 后实测：CPU f32 `0.6712035`（差 0）；GPU f32（V0）`0.6712036`（差 5.96e-08）；**GPU f16（V1 运行期精度分派）`0.6708984`（差 3.05e-04，f16 舍入内）**。该表达式含 rparams ⇒ `expr_prec_sig_native16` 谓词不通过 ⇒ f16 那一跑命中的必然是 `key#x` 分派 shader；注册表若漏它就是闭合世界硬报错，"跑通"本身即断言。
- **CMake include 顺序有语义（踩坑）**：`nn_enable_gpu_fusion` 起初只把 `${NNF_OUT_DIR}` 加进目标，样例 GPU 初始化报 `matmul SPIR-V bytecode not embedded`——手写原语 shader 的嵌入头在 `${CMAKE_BINARY_DIR}/generated`；但该目录**同时含库自身的 `fused_registry.hpp`**。最终把两者都加、且 `${NNF_OUT_DIR}` **排在前面**（`__has_include` 取第一个匹配），否则使用者的自定义结构全部闭合世界报错。已写进 AGENTS §7。
- **已注册为常驻 ctest（round 17）**：样例的目标原先只由默认构建覆盖编译（运行时断言需手工跑），会静默腐化。现已 `add_test(fusion_custom_layer_example)` + `SKIP_RETURN_CODE 77`（无 Vulkan 设备时跳过）——它是**库外使用者形态的 AOT 融合端到端门禁**。ctest 总数 19 → **20**（18 个测试目标 + `cnn_test_gpu` + 本样例；`AGENTS.md` §12 与 `docs/todo/12` 的当前数字已同步）。

## AOT 融合分支代码审核与本轮修复（2026-10-01，round 20）

- 类型：代码审核（简化 / 规范化 / 指针与内存 / 逻辑审查）+ 已修复缺陷
- **范围**：`feat/fused-runtime-prec-dispatch` 全部 29 个 commit 相对 `main` 的源码、工具、CMake、测试与文档。
- **本轮修掉的真缺陷（不只是风格）**：
  1. **native16 谓词与 key 的错配（静默错值）**：`expr_prec_sig_native16` 含**常量值**判据（`0 < |c| < 6e-8` 在 f16 下 flush-to-0），而步骤①让常量值**退出 key** ⇒ 生成期用"生成那一份时的常量"判定、运行期却按**位图**选 `#a` —— 同结构换一组常量就会被套上会 flush 的 native16 shader。修复：选 `#a` 前用**真实实例的 spec 重跑同一谓词**（`find_prec_variant_` 改为收 `const ExprSpec&`）。
  2. **`[prec][alu-hit]` 先打后查 + 双重打点**：见上一条节（12040 → 6022）。
  3. **push constant 固定头长度两侧各推一遍**：创建侧用 `fs.spec.fold/has_matmul/reduce_axis`、写入侧用 `is_fold/matmul_k/has_mm/raxis` —— 注释里记着"漂移 = 超 range 被驱动丢弃 = 假 PASS"，本次把它降为**注册时算一次 + 存 map**（`fused_pc_base_fixed_`），写入侧只读。
  4. **`in_rd` 用子串替换拼 f16 视图名**：找不到前缀时静默保留 f32 视图名 ⇒ 把 2B 缓冲当 float 读（编译期不报错）。改为显式拼 `b<i>[...]` / `b<i>h[...]`。
  5. **`nn_enable_gpu_fusion` 的 include 顺序不成立**：普通 `target_include_directories` 对重复项保留**首次出现位置**，库内 GPU 循环先加了 `${CMAKE_BINARY_DIR}/generated` ⇒ 追加的 `${NNF_OUT_DIR}` 反而排在后面。改 **`BEFORE`**。
  6. **收集器不继承目标的编译条件**：`NN_HAS_VULKAN`/私有 include/自定义宏缺失 ⇒ 使用者的调用点静默消失（而后只在 GPU 运行期闭合世界报错）。改为把目标的 `INCLUDE_DIRECTORIES`/`COMPILE_DEFINITIONS`/`COMPILE_OPTIONS` 用生成器表达式传给收集器。
  7. **模型 pass 失败只 warning**：它贡献 25/84 条结构，静默降级 = 注册表不完整。改为**失败即构建失败**（与 dry-run 的 fail-fast 一致）。
  8. **两条"永远通过"的测试**：`fold tri_skip 进 key` 用 Causal vs Plain 比 key（两者 body 本就不同，删掉该字段的 feed 也照样过）→ 改为**只翻该字段**；`expr_fused_key_test` 里与 `expr_matmul_test` 逐字重复的 mm_trans 位编码断言删掉一份。
  9. **`read_spv` 不校验 `tellg()`**：失败返回 -1 → `size_t(-1)/4` 请求 ~4GB；截断读也被忽略。改为校验 `sz > 0 && sz % 4 == 0` 与 `read_pod_span` 返回值。
- **删除的死物**：`prec_manifest` 形参（回填工作流残留）、`ExprRegistry::contains`（无调用点）、`fusion_anchor_touch`（**实测删除后库外样例的 2 条自定义结构仍被锚点登记**——registry 84 → 86 不变，说明该 TU 靠"包含头文件"即可）、`std::exit(1)`（改 `return false`，与同函数其余错误路径一致）、`scan_exprs` 里重复 4 次的 matmul trans 登记（转置不进 key 后 4 份折叠成 1 条）、AGENTS 里 4 处有争议的 `ctest 19/19`。
- **简化/去重**：`glsl_vp_slot()` 取代 6 处逐字重复的"按视图序累计 vp 槽"（漂移 = 读写错位 push constant）；`scan_register<E>()` 取代 3 个 dsl 入口各写一遍的锚点+折叠+校验+登记；`is_fusable`/`fused_vec_width_for` 取代生成/发射/元数据三循环各写一遍的判据；`expr_variant_out_f16()` 取代 4 处"分派变体按运行期签名、其余按变体身份"；f16 双 `float16_t()` 冗余转换删除；`EXPR_PREC_ALU_SUFFIX` 常量取代 3 处硬编码 `"#a"`；`tool_env_flag()` 取代两份 `_dupenv_s` 样板。
- **检测口径收紧**：`expr_prec_sig_in_f16` 先与 `EXPR_PREC_SIG_INPUT_MASK` 相与（否则哨兵 `0x00020000` 的 bit17 会被读成"第 17 个输入是 f16"，改 `EXPR_MAX_INPUTS` 即踩雷）；分发查询改用 `reg.keys` 而非重建集合；`--list-backends` 排序（铁律 8）。
- **文档纠偏**（本轮统一）：AGENTS §7/§12、`docs/todo/02/05/18`、`docs/usage/03` 中"运行期永不 miss""`[prec][miss]` 已删除""bin v9 变体段""`gen_fused` 产物""ctest 19""12040 次分派"等表述全部改写为当前事实；`docs/development/05` §12.7 改为**历史机制**（明确标注已删除），§12.10 ③ 标注为 V0 时代口径。
- **验收（本机 Debug+Ninja+clang）**：build 零告警（`-Werror`）；**ctest 21/21**；`NN_BIND_DEBUG=1 ctest` 全绿；**CPU 字节锚 `6f8849f14da23110` / GPU dev2 锚 `8ef51b2927253c50` 逐位不变**（GLSL 文本改动全部数值等价）；库外样例端到端值不变（CPU f32 `0.6712035`、GPU f32 `0.6712036`、GPU f16 `0.6708984`、归一化 f16 `1.2246094`）；`scan_exprs` 结构数 **84**（例 86）不变；`NN_PREC_TRACE` 分派日志 12040 → **6022**、distinct 仍 24（20 `#x` + 4 `#a`）；`L2-VIOLATIONS: 0`、`doc_align_audit` A/D/E/F 可行动项 0。
- **教训**：这一轮 9 条真缺陷里有 4 条的形态是同一个——**同一份知识被写在两个地方、靠注释维持同步**（谓词 vs key、创建侧 vs 写入侧 PC 长度、f16 视图名 vs 子串替换、目标编译条件 vs 收集器）。"能简化的简化"在本仓的直接收益就是**消掉这些同步点**；剩下两条（trace 先打后查、测试假通过）则是"**测量/断言必须先证明它能失败**"的同一个教训的延续（round 15/19 已各记一次）。

## 融合回归测试 `fused_gpu_test` 曾是孤儿（2026-10-01，round 18）

- 类型：已修复缺陷（测试覆盖缺口 / 构建配置遗漏）
- **发现**：`src/fused_gpu_test.cpp`（44 KB）**未被任何 CMake 目标引用**——既不编译也不注册 ctest。`git log -S'fused_gpu_test' -- CMakeLists.txt` 最后一次改动是 `516d781`（Release v1.2.0），此后再无引用。
- **它是什么**：融合 shader **逐形态** GPU 对拍——rope / swiglu / gelu / softmax / matmul / matmul+列归约 / norm / reduce_consts / **fold v1**（rowmax/rowsum/softmax_denom）/ **fold attn**（plain/causal/alibi/doc/alibidoc × 5 种形状）/ **未扫描表达式硬报错**。正是 AOT 融合最核心的回归面。
- **处置**：接入构建（`nn_add_executable` + `list(APPEND NN_TEST_TARGETS ...)`，CMakeLists 紧邻 `expr_gpu_test`）。
- **实测**：编译零告警、输出 `ALL PASS`、退出码 0；接入后 **ctest 20 → 21**（19 个测试目标 + `cnn_test_gpu` + `fusion_custom_layer_example`），21/21 全绿。
- **该文件的边界（round 19 实测更正）**：它 **0 处 f16 引用** ⇒ 覆盖的是 **V0（全 f32）** 逐形态路径。
  - ⚠ **曾据此误判**为"V1 无覆盖、无逐形态数值断言"——**该结论是错的**，见下条：V1 由 `f16_precision_test` 覆盖（20 条结构、四大生成器族、`CHECK_NEAR` 容差断言）。本文件缺的是"**本套件的** rope/swiglu/gelu/softmax/fold-attn 形状的 f16 变体"，不是"V1 无覆盖"。

## `NN_PREC_TRACE` 对 V1 是盲的（2026-10-01，round 19）

- 类型：已修复缺陷（诊断不对称 → 直接导致一次错误判断）
- **发现过程**：想量"V1 逐形态覆盖"，用 `NN_PREC_TRACE=1` 跑 6 个 GPU 测试，得到 **0 条 `#x`**、只有 4 条 `#a`。做**对照实验**（跑已知必然走 V1 的库外样例——它含 rparam ⇒ native16 结构谓词不通过）——**样例也一行都不打** ⇒ 说明是**测量工具**瞎，不是"V1 没被用"。
- **根因**：`compute_gpu_engine.hpp::find_prec_variant_` 的 `[prec][alu-hit]` 打印**只写在 V2（`#a`）分支**（原 1391-1392 行），V1（`#x`）分支与 V0 分支都**静默返回**。
- **处置**：V1 分支补同款日志（V0 默认路径**仍不记**——f32 热路径逐调用打日志会淹没输出）。现规则：**变体选择事件（V1/V2）一律打日志**。
- **round 20 复核（本次代码审核）**：上面那次"补日志"本身留了两个尾巴，本次一并修掉——① **先打后查**：V1 分支在 `find_fused` 返回前就打印，`#x` 缺失时也报 `alu-hit`（与刚被修掉的"V1 静默"是同一类错误的反面）→ 改为**确认命中后才打**，miss 时打 `[prec][fallback]`；② **双重打点**：`supports_expr_precision_variant`（每个 op 的 NVI 前置查询）也走同一函数并打日志 ⇒ 计数翻倍——**12040 行其实是 6020 次真实分派 × 2**。改为只有真正 dispatch 的三个入口传 `log_hit=true`。
- **修正后的实测（`f16_precision_test` 单测即可）**：**6022 条命中日志 / 24 条 distinct 结构 = 20 条 V1（`#x`）+ 4 条 V2（`#a`）**（修复双重打点后的真实分派数）；V1 的 20 条**覆盖全部四个生成器族**：元素 9 / 归约 6 / matmul 4 / fold 1。该测试用 `CHECK_NEAR` 容差断言 ⇒ **V1 有真实的逐结构数值覆盖**，"V1 无覆盖"不成立。
- **顺带**：`scan_exprs` 的来源 dump 新增 `[scan][both]`（锚点∩dry-run 交集，原 84 条里 38 条落在这里）——否则"某结构由哪个生成器负责、运行时是否按 V1 分派"无法从 dump 回答（V1 覆盖的 20 条里 18 条在交集中）。
- **教训（本会话第二次）**：一个覆盖/一致性结论在写进文档前，**必须先有一个已知必然命中/必然不命中的对照样本**。上一次是"镜像对"（粗摘要相同但指令序列不同），这一次是"trace 对 V1 盲"。

## 「锚点 vs dry-run」边界实测刻画（2026-10-01，round 15）

- 类型：覆盖缺口 / 否决方案（"删掉 dry-run"）
- **动机**：⑥ 的目标是"删除扫描"。此前只知"dry-run 贡献 71/84"，但**不知道那 71 条为什么锚点看不到**——没有这个答案就无法判断"能否再进一步"。为此给 `scan_exprs` 加了 env 门控诊断 `NN_SCAN_DUMP_SOURCES=1`：逐条打印 `[scan][dry-only]` / `[scan][anchor-only]` 加结构摘要（raxis / mm / fold / nreg / ninstr / nview / viewkinds）。
- **实测（84 条结构）**：per-layer dry-run 38（含模型 pass +25、显式登记 +8）、锚点 59，**重叠 46 → 锚点独有 13、dry-run 独有 25**。
- **dry-run 独有 25 条的分类**：
  - **fold 8 条** = 显式登记的掩码×偏置组合与通用 fold 样例（`fold=1`，viewkinds `[0]`/`[0,0,0]`/`[0,0,0,10]`/`[0,0,0,7,11]`/`[0,0,0,10,7,11]`）；
  - **含 matmul 段 7 条**（`mm=1`：注意力反向 / CE / 优化器步）；
  - **RowAccess(12) 4 条**（共享内存 SwiGLU 的半偏移行切分）；
  - **RowGather(9)/ColBroadcast(8) 若干**（稀疏 CE 的 5 视图形态 `[0,8,8,8,8]` 等）。
  - 共同点：**结构由运行期配置决定**（视图种类 / 项数 / 掩码组合），类型层面推不出来；符号实例 `Expr{}` 与真实实例折叠出**不同 key**。最直接的证据是**镜像对**——dry-only `784781c7b82fa76c` 与 anchor-only `28cdf7c6d5df0e1d` 的摘要逐字段相同（`raxis=-1 mm=0 fold=0 nreg=2 ninstr=2 nview=2 viewkinds=[0,0]`），只有 key 不同。
- **锚点独有 13 条**是镜像面：dry-run/模型 pass 没跑到的编译期可见调用点。最著名的是 `9ca81b4967cdfa20`——关掉锚点（`-DNN_SCAN_NO_ANCHOR`）后 `text_infer`（KV-cache 增量解码）在 GPU 上闭合世界硬报错。
- **结论（否决"删掉 dry-run"）**：锚点与 dry-run **互补、不冗余**，单独删任何一方都会丢结构（dry-only 25 / anchor-only 13）。dry-run + 模型 pass 是结构的**主要**来源，`scan_exprs` 这一**构建步骤本身无法删除**；能删的只有签名维（已完成，扫描工作量减半）。若将来真要清零，路径是"把运行期配置维度类型化"（让视图种类/项数进入表达式类型），使符号实例等价于真实实例——那是另一次架构变更，不属于本阶段。
- **推论与实测反驳（round 16）**：既然 25 条"只有执行才拿得到"，那**库外自定义层（没有 dry-run）是否就不可靠？** 直接用实验回答：给 `examples/fusion_custom_layer*` 加了第二种自研形态——**归一化形态** `rsqrt(col_reduce_sum(x)·inv + ε)`，与库内 LayerNorm/RMSNorm 的 `std_inv` 同形（`raxis=1`、`viewkinds=[0,8]`，正落在 dry-un-only 的形态类别里），**只用锚点**在 GPU 上以 f16 跑通（`1.2246094`，走 `key#x` 分派 shader）。⇒ **锚点对常见自定义层可靠**，A2 机制成立。
- **同时排除了一个错误假设**：曾怀疑"符号实例的常量默认值（全 0）触发 CSE 合并出不同结构"。查 `expr_opt.hpp` 的 CSE key = `op(8)+a.kind/idx+b.kind/idx+c.kind/idx`，**不含常量值** ⇒ 与值无关，假设排除。（此前的"镜像对"是错的：粗摘要相同但指令序列完全不同——`nconst=2 instrs=[17(1/0,1/1);19(0/0,2/0,2/1)]` vs `nconst=1 instrs=[1(2/0,0/0);10(1/0,0/0)]`，是两个不同表达式，不是同一表达式的两个 key。）
- **仍记录的残余不确定性**：25 条 divergence 的**根因未定位**（已知不是 CSE-值依赖、不是视图种类-运行期选择——`ExprViewKind` 全部由类型决定）。若将来使用者的自定义层遇到闭合世界报错，下一步是给 `nn_enable_gpu_fusion` 加 `COLLECT <file>` 钩子（使用者提供"跑一遍自己的层"的收集函数，等价于库内模型 pass）。诊断工具已就位：`NN_SCAN_DUMP_SOURCES=1` 打印逐条来源 + 完整指令序列。

## 融合 matmul BK 取值 A/B 流水（原位置 include/neuralnet.cpp/expr_glsl_gen.hpp:262-269、303-310）
- 类型：性能 A/B / 否决方案
- 内容：40HX 变体 trade-off 记录（% = vs 单缓冲耗时，负=更快）：BK=32 单缓冲(16KB) 为深网格基线、
  浅网格无流水；BK=32 双缓冲(32KB) barrier 节奏同原版但占用率砍半，浅点收益（linear 1024³ -1.9%、
  batch512 -4.9%）深点 ±0，**曾判"最终采用"**；**2026-09-24 OP/融合统一 A/B 复测推翻上条**（同参数
  四点对照，AGENTS §12 ⑥）：融合侧 BK32→16 反超，浅 -24%、深 fwd -24%、深 train -17%
  （8.21→6.81ms）→ 最终 BK=16，与 OP 级同参。历史口径备注：早期"深网格 BK16 -2%"是对单缓冲
  旧基线的口径，非同参双缓冲对比。整改后注释只留当前结论：BK=16 双缓冲、四点全点最优。

## 融合 matmul 共享布局与 32KB 设备要求（原位置 include/neuralnet.cpp/expr_glsl_gen.hpp:370-381）
- 类型：性能 A/B / 否决方案
- 内容：vec4 沿 k + 16 dot 的旧布局 ≈5× 指令发射，是融合 matmul 落后 op 级 ~2× 的主因
  （2026-09-24 由 fused_49fa 与 op 级同形状对拍 7.2 vs 3.15ms 定位）→ 改 vec4 沿 m/n。
  旧注释按 BK=32 描述：[2][32][16]vec4×2 = 32KB（布局改向后总量不变，旧 [2][8][64]vec4 同为
  32KB；BK16 变体曾 A/B 深网格 -2% 不取）；> 单缓冲 16KB → blocks/SM 砍半、深网格靠 WG 余量补、
  浅网格靠流水补；需设备 maxComputeSharedMemorySize ≥ 32KB（Vulkan 规范下限仅 16KB，合规低端
  设备 pipeline 创建失败）。代码现为 BK=16 双缓冲（Ash/Bsh 各 [2][16][16]vec4 = 8KB，合计 16KB，
  恰在规范下限内），原注释与代码不符、已按当前实现改写。

## TARGET_CHUNK=65536 的实测记录（原位置 include/neuralnet.cpp/expr_dsl.hpp:264-272）
- 类型：性能 A/B
- 内容：probe_par_break_even / probe_chunk_at_shape（本机 32 逻辑核）一次实测记录，结论"保持不变"：
  并行区固定开销 ~100µs（32 线程）/ ~3~16µs（≤16 线程）；隔离内核上计算型 body（exp）n=393216
  时 nch=48 比 nch=6 快 1.4x、带宽型 body（x-mean）任何 nch 都不如串行；但 TARGET_CHUNK 改
  4096/16384（→96/24 块）后层级别反而变差：layernorm 2.8→3.2ms、rmsnorm 1.7→1.9ms
  （softmax 1.99→1.7 改善）、transformer/gpt_block 合计下降——隔离内核收益不传递到层。
  故维持 65536，再动必须先拿层级别证据。

## cpu_preparable_leaf_v 默认值教训（原位置 include/neuralnet.cpp/expr_dsl.hpp:1060-1061）
- 类型：bug 根因
- 内容：白名单默认值曾被设为 true，导致 SwiGLU::backward 的 select(row()<d_ff,…) 在模板路径下
  row() 恒为 0、两半梯度选错（layer_gradcheck 的 fc1.w 全红）。由此确立 fail-safe 默认：
  只有 has_reduction_v 为假才可预绑定，含归约节点必须逐个显式加白名单。

## MatmulRef::prepare_cpu 精度-存储不匹配（原位置 include/neuralnet.cpp/expr_dsl.hpp:556-561）
- 类型：bug 根因
- 内容：C 固定以 P=F32 物化，而操作数可能带 f16 存储：内层 GEMM 的 f32 路径按 f32 存储直接读，
  cpu_matrix<F32>() 对 f16 存储返回空指针 → Release 读空 UB、Debug 断言；实测 CPU f16 训练
  step0 forward 即 NaN、text_train 0xC0000005。修复 = 先把非 f32 操作数抬到 f32 再物化
  （docs/development/05-mixed-precision.md §12.12 的双根因之一）。整改后注释只留当前契约与 UB 后果。

## 归约 max 恒等元 −inf−−inf=NaN（原位置 include/neuralnet.cpp/expr_glsl_gen.hpp:1285-1288、2136-2139 等 4 处）
- 类型：bug 根因
- 内容：max 恒等元取 -inf 时，全 -inf 输入（doc 掩码整块屏蔽）下 m_old=blk_m=-inf →
  dm = −inf−−inf = NaN 污染 l/O，是 2026-09 训练 -nan 的根因。全库归约（fold v1/v2、归约
  视图、归约指令、reduce 原语）统一 numeric_limits::lowest()（0xFF7FFFFF）根除该风险类。
  整改后注释只留 NaN 机制与统一 lowest 的现状。

## EXPR_FOLD_BLOCK 取值阶梯（原位置 include/neuralnet.cpp/expr_spec.hpp:825-827）
- 类型：性能 A/B
- 内容："历次调整均以交错 bench 实测裁决：32→64（每块协议减半，fwd −6%）；64→128（协议再减半 +
  链/归约活跃线程翻倍（tid<BLOCK）；shared 6.75→~8.9KB、驻留 8→7 WG 的占用代价被收益盖过——
  mha fwd 5.77→5.41，18/18 绿）"。整改后注释只留 128 的取值依据（相对 64 的结论与代价数字）。

## vecacc 4 路软件流水 A/B（原位置 include/neuralnet.cpp/expr_glsl_gen.hpp:1353-1355）
- 类型：性能 A/B
- 内容：旧版单发串行读 V（GPU 无硬件预取 → 64 级 L2 延迟链完全暴露），探针实测占 fold 55%
  （5694→2567µs）。4 路流水每拍发出 j..j+3 四个读使延迟重叠；四条独立累加语句保持 j 升序求和
  与旧版逐位一致。整改后注释只留当前结构的延迟重叠原理 + 55% 占比依据，删 A/B 对数值与
  "与旧版逐位一致"表述。

## 归约蝶式替换串行扫描（原位置 include/neuralnet.cpp/expr_glsl_gen.hpp:1271-1281、931-934）
- 类型：否决方案 / 性能 A/B
- 内容：旧实现 256 线程各串行扫 valid（64 级 shared 依赖链 ×256 冗余，窗口探针量化占 fold
  ~11%）；旧头注释"免 subgroup 防混邻 WG lane"的顾虑源自跨 WG 混 lane 的担心，对合规驱动不成立
  （规范保证 subgroup ⊆ workgroup，AMD wave64 经 gl_SubgroupSize 泛化）→ 采 subgroup
  shuffleDown 蝶式。整改后注释保留"串行扫描占 ~11%"的选型依据，删"旧实现/旧头注释/故此改"叙事。

## finalize 写后读跨迭代污染的实测症状（原位置 include/neuralnet.cpp/expr_spec.hpp:1093-1096）
- 类型：bug 根因
- 内容：dst 复用被读状态 l 时首列输出覆盖除数、次列起全错，实测症状值 0.4=10/25。
  静态校验（向量域多列循环下 finalize 被写寄存器不得再被读）即由此而来。

## vec_state_len 不进 key 的实证（原位置 include/neuralnet.cpp/expr_spec.hpp:645-648、expr_glsl_gen.hpp:973-974）
- 类型：bug 根因 / 否决方案
- 内容：veclen 曾进 expr_spec_key，导致每个 dk 一个 shader、闭合世界永远缺登记
  （offload/attn 测试按 dk miss 实证）→ 改为运行时经 PC 的 vector_out 槽填充（形状无关融合）。
  整改后注释改为"若进 key 则必然缺登记"的现在式警告。

## bin 格式版本流水（原位置 include/neuralnet.cpp/expr_registry.hpp:97-110）
- 类型：删除清单 / 版本流水
- 内容：原注释含逐版编年：v2 起支持 matmul 段（v1 无 matmul，读 v1 等价 has=0）；v3 rparams；
  v4 views.param2（RowAccess offset）；v5 MatmulSpec.batch；v6 FoldSpec 段；v7 FoldSpec 双域字段
  （vec_state_len/matmul/vecacc）；v8 tri_skip；v9 精度变体段。布局行原写作 version (u8=8)（陈旧，
  实际 kExprBinVersion=9）。read_registry 严格要求 ver == kExprBinVersion，旧版本一律拒绝，
  编年对当前代码无约束力 → 整改后只留当前字段布局 + 读写对称警告，version 行改指常量。

## DslExpr 概念的消歧由来（原位置 include/neuralnet.cpp/expr_dsl.hpp:748-750）
- 类型：删除清单
- 内容：原注释：DslExpr 约束偏序设计是"让 DSL 运算符严格优先于旧代数 nn::operator* 等"
  （旧代数与 DSL 共用 nn::Expression，不区分则二者对 DSL 叶子同为候选）。旧代数 AST
  （algebra_expr.hpp/algebra_compute.hpp）已于 2026-09 整体移除，该竞争者不复存在
  → 整改后改写为当前理由：DSL 运算符只接受可折叠节点（带 to_spec），非折叠的
  nn::Expression 不成为候选、误用编译期暴露。

## 带类型变体支持范围的过期表述（原位置 include/neuralnet.cpp/expr_glsl_gen.hpp:1391-1392、1408-1410）
- 类型：删除清单 / 版本流水
- 内容：原注释称"目前只支持纯逐元素形态；reduce / matmul / fold 拿到 sig != 0 时返回空串"
  与"已支持：纯逐元素 + matmul 段 + fold 段"自相矛盾（in-kernel f16 分期推进后的过期残留）。
  按代码现状订正：generate_glsl 支持纯逐元素 / 前置 matmul 段 / fold 段（v1/v2）的带类型生成，
  归约 kernel 由 generate_glsl_reduce 独立支持（sign_f16 + rd/wr 读写点转换）；gen_fused 按
  raxis 路由两者，仍返回空串的只有归约混合轴（== -2）与归约槽过多（>8）。

## 其余一句话级过时描述（直接删除，未摘录）
- 类型：删除清单
- 内容：expr_emitter.hpp「历史上的 CpuEmitter / CudaEmitter 均已随各自后端删除」；expr_fold.hpp
  「对拍 engine.row_reduce_max/row_reduce_sum」（引擎 row_reduce_max 算子已删，测试现对拍独立
  标量参考）；expr_dsl.hpp「此前只能靠引擎原地原语 add_inplace/scale_inplace」「DSL 取代
  elementwise_* 时性能不倒退」「与迁移前逐字节/逐位一致」等迁移等价性表述；expr_glsl_gen.hpp
  「GLSL 与迁移前逐字节相同（零回归）」「原行为」「语义与旧版一致」「此前出现的运行时参数视图
  个数（改写为按视图序累计）」；expr_glsl_gen.hpp 断链引用 matmul-opt.md（文件不存在，已删）；
  expr_dsl.hpp「未来运行时 JIT」表述（与 AOT 闭合世界铁律冲突，改写为当前执行策略）。

---

# GPU 后端与 GPU 引擎（backend/*, compute_gpu_engine.hpp）

GPU 后端 / GPU 引擎头文件「历史状态类注释」摘录。整改原则：注释只记录当前实现的原理、契约与理由；
本文件保存被移出的历史叙事（bug 根因、被否决方案、性能 A/B 过程、删除清单），供追溯用。

## solo fence / command buffer 复用的由来（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:12）
- 类型：删除清单
- 内容：旧实现每个算子各自 vkCreateFence/vkDestroyFence + vkAllocateCommandBuffers/vkFreeCommandBuffers，
  是逐元素算子固定开销的主要构成之一；现改为 initialize() 预分配 solo fence + solo cmd，submit 前 reset 复用。

## 单帧阻塞提交模型（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:559-562）
- 类型：否决方案 / 删除清单
- 内容：旧实现是「单一 batch_cmd_ + 单 fence」，end_batch/flush_batch 每次 vkQueueSubmit 后立即
  vkWaitForFences —— GPU 执行 step N 时 host 被 fence 卡死、无法录制 step N+1。原注释引用的
  《显存 & 负载不均衡分析报告》§2.1 在仓库中不存在（断链，整改时删除）。现为 PIPELINE_FRAMES=6
  的 (cmd + fence) 帧环，提交不等待、wait_in_flight 才阻塞。环深度 3→6 的加深记录：3 槽时
  host 录制与 GPU 执行重叠度不足。

## 逐元素算子固定开销（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:617）
- 类型：性能 A/B
- 内容：改造前逐元素算子固定开销实测 ≈0.16ms/次（kernel 本身 4096² 已达 370GB/s，瓶颈全在 host 侧）；
  solo fence/cmd 复用后降至 ≈0.143ms（AGENTS §12 同一改动的口径）。整改后注释只保留当前值 ≈0.14ms。

## 帧环预分配（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:1304-1305）
- 类型：删除清单
- 内容：旧实现每步（step 边界）vkAllocateCommandBuffers + vkCreateFence；现为 initialize() 预分配
  PIPELINE_FRAMES 个 cmd + fence，全程零分配。

## run_fused_gpu 固定头长度 bug（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:3495-3498）
- 类型：bug 根因
- 内容：「归约但无 matmul」形态曾按 5 个 uint 计算固定头（真实头部只有 4 个）→ push constant 常量池
  整体后移一个 uint → shader 从错位处读常量，实测把 select(cond,1,0) 的常量读成垃圾，
  **GPU 上带常量的归约静默错值、CPU 正常**（表现为 col_reduce_sum(...) 恒返回 kk-1 而非真实并列数）。
  教训：这类「只有 GPU 错」的问题先打印生成的 GLSL/IR 再猜成因。

## fold PC 漏分支 → 未定义残留（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:1727-1728）
- 类型：bug 根因
- 内容：pipeline 创建侧（pc_base 计算）漏 fold 分支时 range 少算，vkCmdPushConstants 超 range 部分被
  驱动丢弃 → fold_k 读未定义残留，**曾致滑窗式错值**，且残留随前序 op 漂移、表现为时对时错的假 PASS
  （对拍全绿）。整改后注释改为直接陈述该失败模式与「创建侧/写入侧必须同改」的契约。

## matmul dispatch 曾用 WORKGROUP_SIZE=16 统一计算（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:2560-2562）
- 类型：bug 根因 / 性能 A/B
- 内容：曾按 WORKGROUP_SIZE=16 统一计算工作组数 → tiled 版（64×64 块）多派 16 倍冗余工作组
  （每 16×16 一个组），GPU 做 16 倍无效计算，实测 matmul 峰值只剩 ~4%（0.7/15.7 TFLOPS）。

## transpose 曾误派发 16 宽砖（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:4602-4603）
- 类型：bug 根因（issue #13 P0-①）
- 内容：shader 按 8 宽砖解算 tile_c，dispatch 曾给 x=16 → tile_c 跨两砖且越界 tile 只早退一半，
  行>512 且列>512 时静默只写前 512 行。

## 物理设备选择：「第一个 DISCRETE_GPU」（原位置 include/neuralnet.cpp/backend/compute_vk_device.hpp:184）
- 类型：否决方案
- 内容：旧规则「取第一个 VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU」在本机枚举顺序下会选中 [0]
  AMD Radeon R5 240（老专有驱动，maxComputeSharedMemorySize 仅 32768 < matmul 分块所需 34560，
  校验层报 VUID-RuntimeSpirv-Workgroup-06530；且对跨 submit 信号量重复 wait 直接死锁）。
  现为「设备类型权重 + apiVersion 打分、同分取先枚举者、D3D12 转译层降权」。

## 上传 region 专属 cmd 与「submit 后立即 free」（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:2104-2110、compute_staging_ring.hpp:80-83）
- 类型：bug 根因 / 删除清单
- 内容：旧代码每次上传 vkAllocateCommandBuffers、submit 后立即 free/复用，违反
  VUID-vkFreeCommandBuffers-pCommandBuffers-00058（禁止释放 pending 的 command buffer）；
  实测导致驱动通道排序失效、fence 提前 signal、跨 submit 乱序执行。

## staging ring 初版预分配（原位置 include/neuralnet.cpp/backend/compute_staging_ring.hpp:9-13、47）
- 类型：性能 A/B / 删除清单
- 内容：旧默认 4×256MB=1GB 预分配过大；V100 上按「整机 host heap 1/16」规则曾达 2×2GB=4GB host 预算。
  现默认 4 个 region、每 region 取 max(用户请求, clamp(host_visible/16, 32MB, 256MB))，
  且单次 PCIe 传输量通常 << 64MB（最大 token_emb 上传约 5MB）。
  注：文件头原写「默认 2 个 region / 2 regions 共 512MB」，与 DEFAULT_NUM_REGIONS=4 不符，一并订正。

## 列归约两段式的 A/B 方法论（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:3713-3717）
- 类型：性能 A/B
- 内容：同负载窗交错 A/B 定案方法：TP/SP 两二进制逐轮换序、warmup 30、best-of-10、同码对照噪声地板
  ±5%；结果 5244² TP 4/5 胜（中位 0.405 vs 0.430，−6%）、4096² SP 5/5 胜 ~2%（0.277 vs 0.283）、
  512² TP −16%（对 ~10µs kernel 第二遍+屏障净亏）→ 512K 元素尺寸护栏只排除微型形状。
  整改后注释只保留三点实测数字与护栏结论。

## eval_expr_into 的 output_override 曾写死 F32（原位置 include/neuralnet.cpp/compute_gpu_engine.hpp:1639-1645）
- 类型：bug 根因（issue #13 P0-②）
- 内容：曾写死默认 F32 的 gpu_tensor()：f16 目标 → gpu_get<F32>() 空 shared_ptr → Debug NN_ASSERT 引爆；
  Release 空解引用 = UB：MSVC 得到空 override → 结果落临时 buffer 被丢弃 → **参数静默冻结
  （loss 恒 ln V）**；clang 当时「正常」只是 UB 代码生成运气。修复 = f16 目标经 f16_view 包 f16 buffer。

## 显存底材 128MB → 12MB 调参（原位置 include/neuralnet.cpp/backend/compute_memory_pool.hpp:42-52）
- 类型：性能 A/B
- 内容：2026-09 探针实测调参：峰值对块尺寸在 8–32MB 是平台区、最优 ≈12MB 且与模型规模无关
  （跨负载表保留在注释中）；「最小 2 的幂」阶梯类（NN_POOL_LADDER_MAX_MB）实测 d128 上
  ladder16=4497 比 fixed16=4457 差 40MB（配对 3/3），故阶梯默认关闭。

## 真原地语义与 cpu_fallback 的删除（原位置 include/neuralnet.cpp/compute_gpu_engine.hpp:25、466）
- 类型：删除清单
- 内容：2026-09 起引擎内 add_inplace/scale_inplace 等全部改为真原地（此前 copy-on-write 描述已作废）；
  GpuEngine::insert_rows 的 cpu_fallback（CPU 目标回退路径）已删除，CPU 目标直接报错。
  整改后注释只陈述当前契约（真原地的 read-before-write 论证、dst 必须为 GPU Tensor）。

## 收割点提前的探针数据（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:1850-1854）
- 类型：性能 A/B
- 内容：原先 reap_completed_frames 只在 step 边界调用 → 步内已析构中间张量的延迟销毁挂在
  「已完成但未被收割」的帧上不还池；bench 配置实测 transient live 3837MB 里 pending 高达 3000MB，
  步内峰值 ≈ 稳态 2.8×。现收割提前到每个帧提交点（submit_frame_no_wait）。
  整改后注释保留数字与「收割必须紧随提交点」的结论，去掉「原先/2026-09 起」的演进叙事。

## 精度归因编号（原位置 include/neuralnet.cpp/compute_gpu_engine.hpp:530、989、1272）
- 类型：性能 A/B
- 内容：曾用探针归因编号描述 f16 带类型变体的收益来源：scatter_add 的 (8208,64)+(8192,64)、
  add_inplace 的 inplace2_ 大头、reduce 的 (64,8192)×38（Linear::backward grad_bias）。
  这些编号依赖一次性探针输出，整改后改为直接陈述「免去 f32 副本物化」这一当前设计收益。

---

# 注意力 / ZiPT / Transformer / FeedForward 层

> 摘自注意力 / ZiPT / Transformer / FeedForward / 层基类头文件的历史状态类注释（本轮清理）。
> 每条记录原位置、类型与内容；琐碎的一句话描述直接删除、未收录。

## fold 注意力构造从 expr_fold.hpp 迁出（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:161）
- 类型：迁移叙事 / 删除清单
- 内容：`make_fold_attn_o` 与 `FoldAttnMask` 5 种掩码变体原定义在 `expr_fold.hpp`，fold 头因此带上了 attn 命名与掩码设计；2026-09-25 前按 AOT"表达式文本只出现在 Layer"原则迁入 `compute_layer_attention.hpp`，`expr_fold.hpp` 只保留 rowmax/rowsum/softmax_denom 等与注意力无关的通用样例。头文件注释改为现在式：本文件是该构造与掩码变体的定义来源。

## 掩码物化路径与两趟 forward 分支删除（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:490-504、697、394、1128、1196）
- 类型：删除清单
- 内容：P-C2-7（2026-09-23）把注意力 forward 收敛为单 fold kernel 后，`apply_mask_` 物化掩码钩子、`two_pass_mask_` 决策钩子（返回 `{use_two_pass, bias}` 组合式 `AttnBias` 描述子）与旧物化/两趟 forward 分支整体删除；掩码现恒在 fold body / `recompute_W_` 掩码树内表达，绝不物化 (BH·seq, seq) 掩码矩阵，只保留增量推理的 `apply_mask_step_`。原注释中的"与迁移前 apply_mask_ 默认 no-op 同义""原 two_pass_mask_ 的构建体原样保留、职责纯化""与旧 apply_mask_ override 语义同构"等对照论证随之删除。

## P-C2-7 / P0-5 计划项标签摘除（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:392、440、697、742、791、810）
- 类型：修复叙事
- 内容：原注释多处带"（P-C2-7 正确性修复）""（P0-5：从 Q/K 重算 W…）""P-C2-7 起 m/l 也不缓存""（P-C2-7：旧 softmax.backward 分支不保留）"等计划项/修复标签。P0-5 在 docs 中无对应条目（悬空引用）；P-C2-7 的演进叙事已完整记录于 `docs/todo/02-operator-fusion.md`（§关键算法状态横幅、§注意力 M6→S7→P-C2-7）。头文件注释改为直接陈述当前行为：forward 恒单 fold 路径、W/m/l/attn 均不缓存、`recompute_W_` 内部 softmax 单表达式归一化、反向走 R/X 路径。

## recompute_W_ 裸 matmul 的 dry-run 覆盖（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:458）
- 类型：bug 根因
- 内容：MHA（Plain）分支的裸 matmul 结构历史上从未被 scan dry-run 覆盖，AOT 闭合世界下会运行期硬报错；补 MHA dry-run 后由 `scan_exprs` 登记。注释改为契约表述"该结构必须有 dry-run 覆盖"。

## ALiBi 斜率表形状约束（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:1086、1138）
- 类型：bug 根因
- 内容：斜率表原为 (1, num_heads)，而 fold 的 `batch_mod(BH)` 按网格下标 b*H+h 直读 → batch≥2 时 b≥1 越界读、ALiBi 静默错；batch=1 时下标恒 <H 恰好掩蔽，历史单样本测试未暴露该缺陷。修复为 (1, batch*H)、按 (b,h) 块重复 slopes_[h]。旧 recompute 的 `batch_mod(%H)` 与增量推理按 h∈[0,H) 读均落首块，语义不变。头文件保留"表长必须 = batch*H 及其后果"的约束理由，删除"旧…表"实现史。

## doc_ids 缓存布局演进（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:1157）
- 类型：删除清单
- 内容：`doc_ids_cache_` 布局从 M5 时代 AttnBias 的 (1, batch*seq) 改为 S7 的 (1, BH*seq)（BatchCol 视图的 batch 下标 = BH 网格下标，按 (b,h) 块重复 doc_ids[b*seq..]），旧布局 S7 起不再适用。注释只保留"为什么必须 (1, BH*seq)"。

## 精度 profile 下传的历史缺陷（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:540、compute_layer_feedforward.hpp:75）
- 类型：bug 根因
- 内容：Phase 2 精度接线时 AttentionBase/FeedForward 等复合层最初无人把 PrecisionProfile 下传给子层，f16 配置下子层静默停在 f32（参数按 F32 创建、DSL 求值退回 F32）。注释改写为契约："复合层必须把 profile 下传给子层，否则子层静默停在 f32、f16 配置下静默失效"。

## scale 原地缩放写法迁移（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:687）
- 类型：迁移叙事 / 性能 A/B
- 内容：2026-09-27 `engine.scale_inplace` 改为 `dsl::compute_into`（零分配原地写）。旧注释"DSL 表达式多分配一整块缓冲更慢"针对的是 `dsl::compute` **分配版**，与 compute_into 无关。头文件改为直接写"用 compute_into（零分配原地写）；dsl::compute 是分配版，另分配一整块缓冲"。

## 三路梯度累加的等价性论证（原位置 include/neuralnet.cpp/compute_layer_attention.hpp:914）
- 类型：性能 A/B
- 内容：原注释以"与原实现一致（(giq+gik)+giv）→ 逐字节等价"论证 clone + 两次 `add_inplace` 改为单次 `dsl::compute_into` 的正确性。改为对当前实现的直接陈述："结合顺序固定为 (giq+gik)+giv —— 求和顺序确定，结果可复现"。

## DSL 迁移标注（"迁 dsl::…"）（原位置 attention:796/831/845/852/994/996/1015、zipt:132/142/477/484/494/555、transformer:391/397/418/423）
- 类型：迁移叙事
- 内容：计算原语全量迁 DSL（2026-09-27）后，注释普遍带"迁 dsl::matmul(batch)（原 engine.xxx）"式迁移说明，其中"与原 shader 写出时乘等价""原 alpha 写出时乘，等价"等与旧实现的等价性论证删除。保留的是当前契约：结构 key 与哪个 dry-run 同 key、rparam 值不进 `expr_spec_key`、闭合世界注册来源。

## ZiPT 联合注意力反向镜像标注（原位置 include/neuralnet.cpp/compute_layer_zipt.hpp:547）
- 类型：删除清单
- 内容：ZiPTBlock 联合注意力反向原注释标"（materialized 路径，镜像 AttentionBase 旧路径）"；AttentionBase 的旧物化反向路径已删，现 ZiPT 自成一条 materialized 路径（softmax `output_cache()` 持有 A + batched matmul）。删去镜像标注。

## ZiPT W/L 双模式命名（原位置 include/neuralnet.cpp/compute_layer_zipt.hpp:656、657、659、675、678、766、922、989、1035、1036）
- 类型：删除清单
- 内容："旧行为/新行为"命名对（W=L 无压缩 vs W<L 历史/窗口分离）改为当前态命名"兼容模式 / split（W<L）模式"；`W==0 或 W>seq_len → 回退 W=L` 是当前仍执行的兼容分支，改述为"缺省/越界时无压缩"。

## Transformer 尾部遗留的 CausalSelfAttention 类头注释（原位置 include/neuralnet.cpp/compute_layer_transformer.hpp:585-610）
- 类型：删除清单
- 内容：文件尾部遗留整段 CSA 类头注释（类本体定义在 `compute_layer_attention.hpp`，同名注释在该处保留改写版），且其中"在 scale 之后施加预计算掩码 S += mask (batch*H*seq, seq)"描述的物化掩码路径已不存在——整段删除。attention 内同段改为"掩码恒在 fold body / recompute_W_ 掩码树内逐块表达，绝不物化"。

## TransformerEncoder 池化反向旧写法（原位置 include/neuralnet.cpp/compute_layer_transformer.hpp:413）
- 类型：删除清单
- 内容：反向池化注释原列有"`scale(inv_n) → matmul(ones_row_, result_T)` … 直接：`matmul(grad_col_vec, ones_row_)`"的新旧两套写法，旧写法删除，只留现行管线（rearrange → 原地 scale(inv_n) → matmul(grad_col_vec, ones_row_) → rearrange 回）。

## 注意力变体拆分 + 位置编码多态统一（2026-10-01，原位置 include/neuralnet.cpp/compute_layer_attention.hpp 整文件结构、compute_layer_gpt.hpp PositionEncoder 家族、compute_layer_rapt.hpp:60/253/270/335/423/733/813/1143、compute_layer_zipt.hpp:806-817）
- 类型：演进记录
- 内容：原 `AttentionBase` 一个类里同时用 `use_rope_` 布尔 + `use_alibi_`/`has_doc_ids_` 布尔 + `fold_mask_variant_()` 虚钩子 + 4 个 `masked_*_` 模板承载"多种注意力"，且 `recompute_W_` 每次反向都重跑 `if (fold_mask_variant_() == Plain) … if (use_slopes && use_doc) … if (use_slopes) … if (use_doc) …` 组合链（forward 每调用也重算变体，另有三处 `if (use_rope_)`）。整改为**两个正交策略对象、构造期定型**：
  ① `AttnScoreMask` 族（`PlainScoreMask` / `CausalScoreMask` / `CausalDocScoreMask`，**只做掩码**）——每种掩码语义一个类，各自实现类级常量 `mask_kind()`、`prepare()`、`append_fold_inputs()`、`masked_scores()`（反向 DSL 文本）；由 `AttentionBase::make_score_mask_()` 在**配置期**（构造 / `set_doc_ids`）工厂化选定，forward/backward 各只做一次虚调用。文档掩码的"有/无"是运行期数据，其裁定落在 `set_doc_ids` 这一**配置调用**里（模式不变时零开销），不再是每步 forward 的判断。
  **后续同一轮内又把 ALiBi 偏置从掩码策略里摘出去**：原 `CausalAlibiScoreMask` / `CausalAlibiDocScoreMask` 两个类把"掩码"和"位置偏置"融成同一条 select，掩码工厂因此必须查询 `pos_->has_score_bias()`。现在掩码族只保留三个纯掩码类，位置偏置由 `PositionEncoder::apply_score_bias()` 在 backward 的"掩码之后、softmax 之前"独立叠加（forward 仍融在同一个 fold kernel 内，由 `make_fold_attn_o` 的正交第二入参 `bool score_bias` 表达）。`expr::FoldAttnMask`（5 值融合枚举）随之改为 `expr::AttnMaskKind{Plain,Causal,CausalDoc}` + `bool score_bias`，`scan_exprs` 登记块改为 3×2 组合循环（跳过不存在的 "Plain + 偏置"），掩码策略类由 5 个降为 3 个。
  ② `PositionEncoder` 族（新文件 `compute_position_encoding.hpp`）——`RotaryEmbedding` 从 attention 头迁出、由 `RopePositionEncoder` 持有；`use_rope_` 布尔消失，Q/K 旋转改走策略对象（非 RoPE 策略 = 恒等 no-op）。`PositionEncoder` 基类按**注入点**分三组（嵌入侧 `apply/apply_step/backward`、Q/K 侧 `apply_qk/apply_qk_step/set_position_offset`、分数侧 `has_score_bias/prepare_score_bias/apply_score_bias/apply_bias_step`），`Learned/Sinusoidal` 走嵌入侧、`RoPE` 走 Q/K 侧、`ALiBi` 走分数侧、`NoPositionEncoder` 恒等——各子类只覆写自己那组，其余继承基类 no-op。
  ③ **⚠ 拆开时踩到的坑（`row()` 的批内语义）**：ALiBi 的偏置项原来是 `batch_mod(slopes, H) * (col − row)`，`row()` 是"批内行号"、其分解来自**同一 ExprSpec 里的 matmul 段**（`MatmulSpec.batch`）——旧实现把它融在掩码表达式里，那个表达式恰好含 `dsl::matmul(..., BH)`，所以正确。拆成独立一步后 spec 里没有 matmul → batch 退化为 1 → `row()` 变**全局行号** → 偏置静默错值（探针实测 |偏置| 达 27，理论界 (seq−1)·m_0≈7），`attn_test` 的 alibi+doc gradcheck 立刻失败。修法：把 `(col − row)` 换成**`(rows,1)` 行表 + `dsl::row_broadcast`**（按全局行号直读，与网格分解无关）——`AlibiPositionEncoder::prepare_score_bias` 建两张 O(B·H·seq) 行表（每行斜率 `m_{h(row)}` 与批内位置 `row % seq`），`apply_score_bias` 用 `leaf(S) + rb(slope_row) * (col() − rb(pos_row))`。该坑已写进 `AGENTS.md` §7 的告警与 `docs/todo/02` §IR 扩展。
  ④ **所有权按注入点划分（谁拥有 = 谁负责）**：`PosEncodingType` 的分发收敛为两个工厂，同一策略类型只在一个工厂里实做、在另一个里映射为恒等——
     · 模型侧 `make_embedding_position_encoder(type, d_model, seq_len)`（Learned/Sinusoidal 实做），由 `GPTModel`/`ZiPTModel`/`RAPTModel` 持有并在 `apply`/`apply_step` 里施加；
     · 注意力侧 `make_attention_position_encoder(type, d_k, num_heads)`（RoPE/ALiBi 实做），由 `CausalSelfAttention`/`ReLULinearAttention` **自持**（`AttentionBase::install_position_encoder`），在 `apply_qk`/`apply_score_bias` 上取数。
     这样层间**没有任何位置编码对象传递**：删掉了 `AttentionBase::set_position_encoder(PositionEncoder&)`、`GPTBlock::set_position_encoder`、`RAPTBlock::set_position_encoder` 与两个模型的注入循环，也一并去掉了"非拥有指针指向模型对象的生命周期约束"（原先为规避层对象移动导致成员地址失效，兜底对象还得特意放堆上——现在 `pos_` 就是层自己的 `unique_ptr` 成员）。代价是 **RoPE 的 cos/sin 表随层构建**：每层 `2·d_k·seq·4B`，本仓默认配置（`d_model=128`/`H=4`/`layers=4`/`seq=256`）≈ 64KB/层、全模型 256KB，三角函数只在 `seq` 变化时算一次；层数极多的配置才需要回头考虑共享。
     另外，`AttentionBase::init_impl` 改为**调用** `pos_->init(engine)`（三组策略的 init 都是 no-op）——符合 M6 段 C 的不变量"复合层 `init_impl` 必须 init 全部子对象"；此前"不 init"是"注入的模型级对象由模型自己 init"这一设计的副产物。`max_len` 参数保留仅为签名兼容，当前实现不使用。AOT 闭合世界不受影响：`make_fold_attn_o` 与 5 个组合的登记仍在 `scan_exprs` 的显式块中。
- 验收：build 零告警（`-Werror`）；ctest 20/20（含 `NN_BIND_DEBUG=1` 一轮）；CPU 锚 `6f8849f14da23110` 与 GPU dev2 锚 `8ef51b2927253c50` 逐位不变；`--init-hash` 六模型与 M2 锚全同；`--io-roundtrip` CPU/GPU 全过；L2 审计 `L2-VIOLATIONS: 0`（`bench/doc_inventory.ps1` 的 L2 文件清单已补入 `compute_position_encoding.hpp`，否则新文件不在门禁内）；引擎 virtual 方法仍 49、Layer 直调算子仍 21。
  **scan 双 hash 在本轮内变过两次**（`bdc3a442…`/`7a10412c…` → `74e02702…`/`bb03e5e2…` → `45edce4a…`/`15899056…`）：位置编码**所有权**调整那次（注入 → 自持）表达式文本未动、hash 逐字节不变；而把 ALiBi 偏置从掩码表达式里拆成独立一步那次**必然改变**backward 的表达式树（`scores + select(blocked,-inf,alibi)` → `(scores + select(blocked,-inf,0)) + bias`），故 hash 变化是设计结果、不是漂移。两次都做到了**数值逐位不变**（CPU/GPU 字节锚不动）——这正是"改结构不改数"的判据：hash 说明结构变了，字节锚说明值没变。

## ActivationOffloader slab 容量校验（原位置 include/neuralnet.cpp/compute_layer_base.hpp:196）
- 类型：bug 根因
- 内容：slab 原先只在 `!slab_.valid()` 时按当时的激活总量分配、之后永不增长；后续 step 激活总量变大（批大小/序列长度变化、--resume 后续训、最后一个不满 batch 之后的 step）时 `offload_save` 按新 offset 越界写 slab → 缓冲区破坏/设备丢失。修复为每次导出按当前总量校验、不足即重建。头文件只保留现行不变量与其后果。

---

# RAPT / CNN / MLP / Softmax / GPT 层

> 摘自 RAPT/RLA、CNN 卷积池化、MLP、Softmax、GPT 模型层头文件的历史状态类注释（本轮清理）。
> 每条记录原位置、类型与内容；琐碎的一句话描述直接删除、未收录。

## Layer 层四处收敛：参数收集 / Pre-Norm 残差反向 / 解码采样 / 错误传播（2026-10-01，原位置 compute_layer_{gpt,rapt,zipt,transformer,attention,feedforward,base}.hpp、core_errors.hpp）
- 类型：演进记录
- 内容：
  ① **参数收集**：`parameters()` / `param_gradients()` 里 `p.insert(p.end(), x.begin(), x.end())` 的级联在三个模型 + 四个块层共 15 处同构重复 → `Layer::collect_refs(...)`（变参拼接：实参可以是单个 `Tensor` 或子层返回的 `std::vector<TensorRef>`）与 `Layer::collect_block_refs_(blocks, &Type::fn)`（块容器逐块取参）。拼接顺序即实参顺序，不改变优化器看到的参数序。
  ② **Pre-Norm 残差**：`GPTBlock` / `TransformerEncoderLayer` / `RAPTBlock` 的两处残差分流（正/反向）逐字同构 → `Layer::prenorm_residual_forward_` / `Layer::prenorm_residual_backward_`。**残差相加的表达式输出精度显式传入**：GPT/Transformer 传 `p.compute`，RAPT 历史上未传 profile（恒 F32），故传 `Precision::F32`——"顺手统一成 p.compute"会改变 f16 路径的数值行为，因此不做，只把差异显式化。
  ③ **解码采样**：`GPTModel/RAPTModel/ZiPTModel::generate()` 里"temperature 缩放 → 数值稳定 softmax → 随机采样 / 贪心"三段逐字相同 → `sample_next_token_()`（`compute_layer_base.hpp`）。RNG 消耗序（每步恰好一次 `dist(rng)`，仅采样分支）保持不变。**解码循环骨架刻意未合并**：三者运行态本质不同（GPT = KV cache 增量 + 滑窗重建、RAPT = RLA 运行态、ZiPT = 无状态整窗前向），用标志位强行统一会得到更难读的"上帝函数"，与"保持 Layer 简洁"的目标相反。
  ④ **错误传播**：`auto r = f(); if (!r) return std::unexpected(r.error());` 在 L2 层重复 494 处、占 Layer 代码近 9% → `NN_TRY(decl, expr)` / `NN_TRY_CHECK(x)` 宏（`core_errors.hpp`）。展开与手写形态**逐字等价**（同样的错误消息、返回类型、语句数），不含控制流或所有权语义；实现用普通语句而非 `do{}while(0)`，以便在 `if/else` 里保持与原 `if` 语句一致的悬垂-else 行为。机械改写脚本 `tools/apply_nn_try.ps1`（带 `-DryRun` 计数，只改模式完全匹配的行）可复现。
- 验收：build 零告警（`-Werror`）；ctest 20/20（含 `NN_BIND_DEBUG=1` 一轮）；scan 双 hash 与 CPU/GPU 字节锚逐位不变；`--init-hash` 六模型与 M2 锚全同；`--io-roundtrip` CPU/GPU 全过；L2 审计 `L2-VIOLATIONS: 0`。

## 缓存前置校验的修复过程叙述（原位置 include/neuralnet.cpp/compute_layer_conv.hpp:215-216、include/neuralnet.cpp/compute_layer_rapt.hpp:540-542）
- 类型：bug 根因
- 内容：Conv2D 原注释「不校验会拿空/上一 batch 的陈旧 im2col 静默算出错误梯度」、RLA-2 原注释「旧行为会把问题推到某个 matmul 里甚至算出垃圾梯度；这里立刻返回明确错误」——均为 2026-09-19 CNN 缓存契约加固与 RAPT backward 缓存校验的修复过程叙述（旧行为 = 空/陈旧缓存静默进 matmul）。改写为现行契约：backward 必须校验缓存命中且形状匹配、checkpoint/offload 模式下缓存为空必须立刻返回明确错误；错误文案（"checkpoint 模式需先 forward_recompute…"）原样保留。

## MaxPool2D 分组归约的逐通道循环与迁移标注（原位置 include/neuralnet.cpp/compute_layer_conv.hpp:367-369）
- 类型：删除清单 / 迁移叙事
- 内容：原注释「迁 DSL grouped_reduce_max 视图（缺口 A，2026-09-27）：单表达式单 dispatch（此前引擎原语；更早版本按通道循环 C 次 dispatch）」（HEAD 版本另作「早期版本按通道循环 slice_rows + col_reduce_max + insert_rows，C 次 dispatch」）。演进链：按通道 3 类原语循环 C 次 dispatch → `engine.grouped_reduce_max` 单原语（2026-09-20，连带消除逐通道循环）→ `dsl::grouped_reduce_max` 视图（2026-09-27，R 进 expr_spec_key）。改为当前陈述：「单表达式单 dispatch 完成全通道分组归约（grouped_reduce_max 视图，R = 窗口面积 kk）」。

## MaxPool2D 并列梯度的 argmax-first 等价性论证与头部日期流水（原位置 include/neuralnet.cpp/compute_layer_conv.hpp:265-273、418-420）
- 类型：等价性论证 / 日期流水
- 内容：原注释以「无并列最大值时与原 argmax-first 实现逐位一致」「无并列时与 argmax 散射逐位一致」论证与**已删除**的 argmax-first 散射实现等价，头部带日期「实现（全引擎化，2026-09-20）」，且 forward 一步仍写已不存在的「每通道 col_reduce_max」（现行 grouped_reduce_max）、backward 写「每通道 mask」（现为整张量单表达式）。改为当前语义直接陈述：无并列（cnt=1）时梯度全部落在唯一最大值位置、并列按 1/cnt 均分（总梯度守恒）；PyTorch 首个 argmax / TensorFlow 全量并列的对照保留为并列语义差异说明（删「历史上」）。

## 已不存在的 CPU to_matrix/from_matrix 路径描述（原位置 include/neuralnet.cpp/compute_layer_mlp.hpp:820-822、include/neuralnet.cpp/compute_layer_gpt.hpp:506-508）
- 类型：删除清单
- 内容：mlp.hpp 尾部 Conv2D 段落的说明「im2col/col2im 涉及复杂重排，沿用 PatchEmbedding 的先例在 CPU 端完成（to_matrix/from_matrix），GEMM 仍复用引擎 matmul 内核；MNIST 尺度下 CPU↔设备往返开销可忽略；GPU 融合卷积内核留作后续优化」描述的是 2026-09-20 CNN 全引擎化之前的 CPU im2col 路径（现 im2col/col2im 为设备原语、`col_cache_` 为设备张量，conv.hpp 同步注释「无 to_matrix/from_matrix 往返、无 CPU 标量循环」），整段删除（Conv2D 类段落本身留在原处，其布局/算法说明仍为当前实现）。gpt.hpp 的「token embedding 查表 + 位置 embedding 相加…此处用 to_matrix/from_matrix 在 CPU 端完成（batch 边界，PCIe 传输符合纯 GPU 架构约定）」同为已不存在的路径（现行 forward 全程 transpose/gather_rows/逐元素加，backward 用 scatter_add_rows），改为「查表与位置相加全程由引擎原语在设备端完成、无 CPU 中间拷贝」。

## Linear forward/backward 的 DSL 迁移标注与零回归论证（原位置 include/neuralnet.cpp/compute_layer_mlp.hpp:59、108-113、128-129、142、154-157）
- 类型：迁移叙事 / 删除清单
- 内容：2026-09-27 计算原语迁 DSL 时留下的标注——「迁移到 DSL（2026-09-27）：原 engine.matmul_with_bias（CPU/GPU 引擎内部实现本就是本表达式…）改为 Layer 直写」「（原 engine.matmul；scan 的 Linear backward dry-run 自动登记新结构）」「取代 matmul + accumulate 两次 dispatch」「累加步迁移为 dsl::compute_into（…取代 engine.accumulate）」「归约步迁移为 dsl::compute_reduce（原 engine.row_reduce_sum…）；累加步见下」「全 F32 配置下与迁移前逐字节一致（零回归）」全部删除/改写。保留当前契约：融合单 kernel（与 `engine.matmul_with_bias` 的 DSL 融合结构一致）、dry-run 自动登记该结构、归约与累加无法并入同一表达式的语义冲突理由、f16 经 PrecisionEngine 内部 f32 计算输出按目标精度舍入。

## SwiGLU 的 slice_rows/insert_rows 拆分描述（原位置 include/neuralnet.cpp/compute_layer_mlp.hpp:295-296、304-307、334、351）
- 类型：删除清单
- 内容：原注释写 gate/up 由 `slice_rows(x, 0, d_ff)` / `slice_rows(x, d_ff, d_ff)` 切出、「split/merge 用 slice_rows / insert_rows」「不再 slice_rows 物化半张量 → 消去 2 次 D2D 拷贝」「backward: 单 kernel 融合，消去 create+zero+2×insert_rows」——现行实现为 `dsl::row_access` 行视图（零拷贝）+ `select(Row() < d_ff, …)` 单表达式写出，不物化半张量也不经 insert_rows 拼装；改写为「行视图零拷贝、不物化半张量（无 D2D 拷贝）」「gate/up/s 经 RowAccess 行视图定位，两半由 select 一次写出」「单表达式融合，直接写出整张 grad_input (2*d_ff, batch)」。

## RLA-2 RMSNorm 反向缓存漏在 activation_cache 之外（原位置 include/neuralnet.cpp/compute_layer_rapt.hpp:352-353）
- 类型：bug 根因
- 内容：原注释「此前漏在 activation_cache 之外 → offload 无法覆盖它们；补齐后 offload 覆盖 backward 所需的全部激活」——RLA-2 RMSNorm 的 4 项反向缓存（Q/K normed、Q/K rms_inv）最初未登记进 `activation_cache()`，activation offload 因此换出不完整、backward 缺缓存。改写为现行契约：「契约：activation_cache 必须覆盖 backward 所需的全部激活，offload 才能完整换出并恢复」。

## GPT backward 逐块释放激活缓存的旧行为叙事（原位置 include/neuralnet.cpp/compute_layer_gpt.hpp:802-807）
- 类型：性能 A/B
- 内容：原注释「原先只在 checkpoint / offload 模式清理（默认路径不清理），于是每块激活一直驻留、累积到整个 backward 结束才被下一轮 forward 覆盖——探针实测这是 backward 段峰值主项（torch 的等价行为是"用完即释放"）」。删「原先…默认路径不清理」的旧行为描述，保留当前行为与理由：backward 后立即 `clear_cache()`，否则块激活驻留到整个 backward 结束（探针实测为 backward 段峰值主项），释放后内存可被后续块 backward 临时量复用、下一轮 forward 重新填充。

## 位置编码多态抽取叙事与不存在的标志位（原位置 include/neuralnet.cpp/compute_layer_gpt.hpp:262-264、36、426、941）
- 类型：删除清单
- 内容：PositionEncoder 段落原写「把 GPTModel 中**原本**按 PosEncodingType 散落的 if-else 位置编码逻辑**抽离**为多态层次…**消除 use_pos_emb_ / pos_emb_learnable_ 等标志位**分支的耦合」——两个标志位在代码中已不存在，抽取过程属于重构史，改为现在式「按 PosEncodingType 把位置编码逻辑组织为多态层次，GPTModel 通过基类指针使用，不依赖标志位分支」。同批琐碎项：「实现已抽到 compute_layer_base.hpp 的 ActivationOffloader」→「实现见 …」；「与 token_emb_ 共享同一 rng 序列（保持与旧实现完全一致的可复现性）」→「共享同一 rng 序列，保证跨 run 可复现」；「历文 K/V 不再重复投影」→「历史 K/V 复用缓存，不重复投影」（滑动窗口「丢弃最旧 token」「复用历史 K/V」等时序语义原样保留）。

## 「文档 22」悬空引用（原位置 include/neuralnet.cpp/compute_layer_rapt.hpp:35、411）
- 类型：删除清单（指向不存在文档的引用）
- 内容：注释引用「与原版 RLA 的关键差异（§3 文档 22）」「稳定数值分布…（文档 22 §4.3）」，但仓库内无"文档 22"编号；两处改指现存的 `docs/development/06-rapt-algorithm.md`（§3 家族演进即 RLA vs RLA-2 差异表，防神经元死亡 / RMSNorm 前置一并记录在该节）。RLA-2 与原版 RLA 的差异列表本身是当前算法定义，保留。

## "迁 dsl::…（原 engine.xxx）"与 dispatch 计数对照（原位置 conv.hpp:191、229、235、244、427；mlp.hpp:142；rapt.hpp:118、174、238-239、683、688-689、713-714、831-832、845、878-880）
- 类型：迁移叙事 / 性能 A/B
- 内容：2026-09-26~27 计算原语迁 DSL 后注释里残留的演进标注——「归约步迁 dsl::compute_reduce（原 engine.row_reduce_sum）」「纯 matmul 迁 DSL 直写（原 engine.matmul）」「（迁 DSL grouped_reduce_sum 视图，同 forward）」「原地累加迁 dsl::compute_into」「迁 dsl::matmul(batch)：…」「（GPU 上 2 → 1）/（3 → 1 次 dispatch）/（原 zero + broadcast_row_inplace + elementwise Add 三次 dispatch → 1 次）」「取代 matmul + accumulate 两次 dispatch」；rapt 另有「（原为 term/diff/gx 三个独立 kernel + 一个逐元素原语）」「原"乘 x² → 列归约 → 乘 1/dk → 加 eps → rsqrt"五步压成两步 DSL」「不再吞错后让…」（ensure_ones_ 的错误传播）「同一个 AOT 键…（此前的 eager elementwise_binary 已退役）」。删除后保留的当前陈述：单次 dispatch / 不物化中间张量的融合事实、该表达式与哪个 dry-run 同 `expr_spec_key`（闭合世界注册来源）、分两步执行的原因（归约融合 shader 不支持归约后逐元素后处理，GPU 侧显式拒绝）、错误传播避免根因被 "ensure_gpu: invalid tensor" 掩盖。

## BatchNorm 落地（2026-10-03，roadmap P0-3「BatchNorm 禁止静默回落」）
- 类型：演进记录（特性落地 + 兼容性变化）
- 背景：`NormType::BatchNorm` 枚举 / `mnist_train --norm batchnorm` / `gui.py` NORM_OPTIONS / README 声明早已存在，但 `make_norm_layer`（`compute_layer_mlp.hpp:782`）一直**静默回落到 LayerNorm**（roadmap C3 / P0-3）。P0-3 裁定「显式报错 或 真实现，二选一」，本轮选**真实现**。
- 实施：
  - `compute_layer_mlp.hpp` 新增 `BatchNorm` 层：训练态用 batch 统计（`row_reduce_sum` 沿 batch，与 LayerNorm 沿特征互为方向翻转）+ EMA 更新 `running_mean/running_var`；推理态用 running 统计（batch=1 安全）；γ/β 为参数，running 统计经 `extra_state()` 序列化（不进 `parameters()`，优化器不碰）；`set_training` 双态（**默认训练态**）、`forward_recompute` 抑制 EMA 重复更新（否则检查点场景双倍计数）；推理态 backward = `grad_x = gy·γ·inv_std`（无均值修正项，γ/β 归约项与训练态同形）。`make_norm_layer` 增加真分支。
  - **口径裁定**：running_var 的 EMA 用**有偏**方差（1/B，与归一口径一致；PyTorch 用无偏）——B=1 无除零、结果确定（铁律 #8）；`momentum=0.1` = 新 batch 权重（PyTorch 同口径）。
  - `tools/scan_exprs.cpp` 模型 pass：每个模型先训练态 fwd+bwd、**再推理态 fwd+bwd**——推理态是不同表达式结构（无归约链），只跑训练态则 GPU 推理闭合世界硬报错；对无双态层 `set_training` 是 no-op，多跑一轮只重复登记已知结构。
  - 测试：新增 `src/batchnorm_test.cpp`（工厂不回落断言 / 双态 forward vs 宿主参考 / running EMA 多批对拍 + 推理态不更新 / 训练态 gradcheck（γ/β/输入）/ 推理态 backward vs 宿主 / Model::set_training 转发 + extra_state + save-load 逐位往返），CMake 注册 `batchnorm_test` 与 `batchnorm_test_gpu`（77=skip 约定）→ **ctest 21 → 23**。
- 兼容性变化（**需随下个版本写入 release-notes**）：旧的 `norm_type=BatchNorm` 规格模型文件是当年按 LayerNorm 实际保存的（params 之后**没有** extra 张量），新代码加载时会在 extra 状态解析处报错（非静默错值）——这类文件规格与内容本就不符，预期失败；重新训练即可。
- 验收：build 零告警；ctest 23 用例中本机 21 过 + `gpt_offload_test`/`rapt_offload_test` 失败 = **docs/development/16 登记的 Mali 已知非确定基线**（`NN_VULKAN_DEVICE=1` Lavapipe 下两用例全过，与本改动无关）；scan 结构 **84 → 91**（库内口径：模型 pass 25 → 32、锚点 59 → 66，锚点独有仍 13、dry-run 独有仍 25；样例收集器 = 91 + 样例自定义 2 = 93）；端到端冒烟：`mnist_train --norm batchnorm` CPU 1 epoch test_acc 92.76%、`--resume` 往返续训正常、`--gpu` 训练+评估正常；`batchnorm_test --gpu` 全过（= 推理态/EMA 融合结构在 Mali 上闭合世界命中）。

---

## CNN / ViT 接入归一化（NormPlace，2026-10-03）
- 类型：演进记录（特性落地 + 兼容性口径）
- 背景：`mnist_train --norm` 与 GUI「归一化层」此前**只对 `--arch mlp` 生效**——`--arch transformer`（ViT）/ `--arch cnn` 两条路径既不读 `cfg.norm_type`，CNN 更是完全没有归一化层；ViT 的 pre-norm 类型在 `TransformerEncoderLayer` 里硬编码为 `LayerNorm norm1_/norm2_`。
- 实施：
  - 新增 `NormPlace{None,Conv,Head,Both,Final}`（`model_spec.hpp`）+ `ModelSpec::norm_place`，把「类型」（`--norm`）与「挂载位置」（`--norm-place`）拆成两个正交维度。CNN：`Conv` = 每个卷积(+池化)之后按 (C*H*W, batch) 每样本归一化、`Head` = 全连接头 `Linear→Norm→ReLU`（与 MLP 同款，末层 logits 不挂）、`Both`；ViT：`Final` = 编码器末端、全局平均池化之前的 ln_f（原版 ViT 的 final norm）。
  - `TransformerEncoderLayer` 的两个 pre-norm 槽位由 `LayerNorm` 成员改为 `make_norm_layer(d_model, norm_type)`（`unique_ptr<Layer>`，**参数顺序不变**：self_attn → norm1 → ff → norm2）；`TransformerEncoder` 接收 `norm_type` + `final_norm`；并在 `TransformerEncoder` 与 `TransformerEncoderLayer` 两级补 `set_training` 下传（BatchNorm 进 pre-norm/final 槽位后才能拿到训练/推理开关；GPT/RAPT 复合层仍不转发，口径见 `compute_layer_mlp.hpp` 注释）。
  - 序列化：`spec_to_kv`/`spec_from_kv` 新增 `norm_place` 键。**旧文件没有该键 → 读回 `NormPlace::None` = 不加额外归一化 → 参数布局与旧实现逐位一致**，旧 checkpoint 仍可加载；版本默认值表补了说明行（无需回落动作）。`spec_matches` 把 `norm_type`/`norm_place` 纳入 MLP/Transformer/CNN 的关键维度（原 MLP 只比 `layer_dims`，norm 错配会被推迟成矩阵形状错位）。
  - CLI/GUI：`--norm` 从「MLP 专用」提升为全架构通用；新增 `--norm-place auto|none|conv|head|both|final`，两者都支持 `auto`（= 未指定，按架构取默认）；非法组合 fail-fast，如 `--arch cnn --norm-place final` / `--arch mlp --norm-place conv` 均 exit 1；`gui.py` 的归一化行移出 MLP 专属组、新增「归一化位置」行（仅 transformer/cnn 显示，默认均选 `auto`），`cli_controllers.py` 与 `bench/gui_cli_audit.py` 的 CASES 同步。
  - `scan_exprs` 模型 pass 补 6 例归一化变体（ViT final-ln / final-rms / pre-norm-bn，CNN both-ln / conv-rms / head-bn）——归一化类型与挂载位置都是「运行期配置决定的结构」，与 GPT 的 pe/act/norm 维同性质。
- **各架构默认（初版按“零字节变化”定为 None，随后裁定改为按架构取最合适）**：CNN = `BatchNorm@conv`、ViT = `LayerNorm@final`、MLP = `LayerNorm`（结构内置 Linear→Norm→激活）；CLI/GUI 用 `auto` 表达“按架构默认”。**4 epoch / 1024 样本 / `--shuffle-steps false` 同 seed A/B（本机 CPU）**：

  | 架构 | 配置 | epoch4 loss / test_acc |
  |---|---|---|
  | CNN | 无 norm（旧默认） | 0.3337 / 86.71% |
  | CNN | **batchnorm@conv（新默认）** | **0.1535 / 89.48%** |
  | CNN | batchnorm@head | 0.2530 / 89.81% |
  | CNN | batchnorm@both | 0.2260 / 89.39% |
  | CNN | layernorm@conv | 0.1851 / 89.20% |
  | CNN | layernorm@both | 0.2361 / 89.05% |
  | CNN | rmsnorm@conv | 0.2012 / 89.12% |
  | ViT | layernorm@none（旧默认） | 0.7580 / 69.00% |
  | ViT | **layernorm@final（新默认）** | 0.7342 / **77.50%** |
  | ViT | batchnorm@final | 0.5230 / 72.50% |
  | ViT | batchnorm@none | 0.5011 / 75.50% |

  取舍：CNN 上 BatchNorm 全面占优，`conv` 与 `head`/`both` 差距在噪声内（−0.3pp/+0.1pp）而 loss 最低 → 取最经典的 Conv→BN；ViT 加 final norm +8.5pp，而 BatchNorm 类型掉点（72.50% < 77.50%）→ 类型保持 LayerNorm。
- **旧文件兼容（与“默认值”正交）**：`spec_from_kv` 缺 `norm_place`/`norm_type` 键时保持 `ModelSpec` 默认（`None`/`LayerNorm`）——**缺键不回落架构默认**，旧 checkpoint 参数布局逐位一致、仍可加载；新文件写全两键，`--resume`/`mnist_infer` 按文件值重建（`model_spec_validation_test` 7.0 有断言锁）。
- **锚点与验收（实测）**：默认变更令 `gpu_stability_probe --init-hash` 的 **cnn / mnist_transformer / gpt / rapt 四个 hash 变化**（mlp 首建不变）——后两者是**连带位移**：探针在同一引擎上按序建 5 个模型、`InitSpec` seed 按创建序号混流，CNN/ViT 新增的归一化参数把后续模型的序号整体推移（与「ZiPT 移除令 rapt 锚变化」同机理）。本机新锚（跨进程逐次一致）：mnist_mlp `0148e453908158dc` / cnn `1aa083597005faa0` / mnist_transformer `0e928b0fc18fe184` / gpt `7ac3e9d97f9d903a` / rapt `6f58e076e290e24c`（锚点机器/stdlib 相关，变更前的历史锚见本文件 M2 条）。`build/generated/fused_registry.hpp` sha256 **不变**（`7eaff502…d7589`，结构数 91——新默认涉及的结构全部已在 dry-run 覆盖内）；ctest 除 `gpt_offload_test`/`rapt_offload_test` 外全绿，**这两项在本改动之前的基线上同样失败**（把本分支 10 个源文件回退到 HEAD 另建构建复现，逐次波动 3~63 条 FAIL，属另案的非确定问题）；`python bench/gui_cli_audit.py` 三节 PASS（本机需把 `.exe` 后缀换成无后缀可执行名）。
- 测试：`model_spec_validation_test` 第 7 节 = 7.0 缺键兼容 / **7.0b 默认值回归锁（`make_cnn_spec` 默认 = BatchNorm@Conv、`make_mnist_transformer_spec` 默认 = LayerNorm@Final）** / 7.1 CNN both 往返 + 错配拦截 / 7.2 ViT final forward+backward（batch=2）+ 往返；`cnn_smoke_test` 第 2 段锁「默认 = BatchNorm@conv 且归一化层数 = 卷积阶段数」、第 2b 段 RMSNorm@Both 层数与 fwd/bwd。ctest 计数不变（23）。
- 端到端实测：不传 norm 参数时 `--arch cnn` → `BatchNorm @ conv`、`--arch transformer` → `LayerNorm @ final`、`--arch mlp` → `LayerNorm`；CNN/ViT 默认配置 `--gpu` 训练通过（GPU 闭合世界无 miss）；`--resume` 从文件读回 `norm_place` 后按原架构重建并加载成功；`mnist_infer` 对 CNN/ViT/BatchNorm 三个 norm 模型推理正常；`--norm-place conv`（MLP）与 `--norm-place final`（CNN）均 exit 1。


# 引擎接口 / CPU 引擎 / 精度 / 张量 / 损失 / 优化器

- **2026-10-18 引擎所有权 v2（EngineId + shared_ptr 强引用）**（原位置：
  `core_engine_lease.hpp` / `compute_tensor.hpp` / `compute_engine.hpp` /
  `compute_vk_backend.hpp`）：旧方案用进程级只增不减的 `EngineLiveness` 令牌池 +
  8B `EngineLease`，`require()` 仍是 check-then-use；`GpuBackend::instance()`
  故意泄漏 Vulkan 资源，`GpuBuffer` 析构还依赖全局单例存活。v2 改为：
  ① `Tensor` 只保存 8B `EngineId`（进程内单调递增、永不复用，无指针/无原子
  拷贝）；② `Layer` / `Model` / `Optimizer` / `ExprGraph` / `ExprBlock` 持有
  `shared_ptr<ComputeEngine>`；③ `CpuEngine` / `GpuEngine` 私有构造 +
  `create()`，禁止栈上引擎；④ `GpuBackend` 改 `shared()` / `create()` +
  `enable_shared_from_this`，`GpuBuffer` 持 backend keepalive，`reduce_partial_`
  改 backend 自有裸 `VkBuffer` 以切断引用环；⑤ 修复 `~GpuBackend` 中
  `reduce_partial_` 晚于 memory pool 析构的顺序隐患。验收：CPU ctest 53/53；
  GPU ctest 55/55（`NN_VULKAN_DEVICE=850M`）；`layer_lifetime_test` 覆盖
  EngineId 8B、Tensor 晚于引擎析构、Layer/Model/Optimizer 保活、跨引擎拒绝；
  `NN_GPU_LIFETIME_FULL=1` 额外验证 GpuBuffer 晚于 GpuEngine/backend 析构。
  设计与迁移清单见 `docs/development/09-engine-ownership.md`。
- **2026-10-18 发布前补漏：GPU f16 回退边界 cast 的 EngineId stamp**：`NN_BIND_DEBUG=1`
  全量 GPU ctest 暴露 `f16_writeback_probe` / `f16_precision_test` / `rapt_test_f16_gpu`
  三例失败，首个点位是 `ComputeEngine::cast` 的未绑定输入。根因是 GPU 引擎在无原生 f16
  pipeline 时的 `gather` / `rearrange_3d` / `transpose` / `im2col` / `col2im` /
  `matmul` / `batched_matmul` 回退路径用静态工厂造出 f32 产物后，直接进公共 `cast`，
  此时尚未走到外层入口的 `stamp_`。7 处统一改为 `cast(adopt(std::move(out)), F16)`；
  修复后 Release+GPU 的 `NN_BIND_DEBUG=1` ctest **55/55**、CPU-only **53/53**。

## P2-12：复合 kernel 的运行期精度签名在 scan 侧无登记入口（2026-10-14；原位置 include/neuralnet.cpp/compute_cpu_engine.hpp:71）
- 类型：覆盖缺口 → 修复（构建期扫描补记）
- 内容：`scan_exprs` 模式下的 CPU `end_expr` 只做了 `global_registry().add(k.spec)`（登记**结构** V0），没有登记**融合 kernel 的运行期精度签名**——单节点结构的签名由 `dsl::compute` 的 `scan_register` 与引擎 `eval_expr*` 入口按真实张量精度登记，而**融合结构只在 GPU 的 `end_expr` 执行时才算出签名**。后果：`--f16` 下复合结构只有 V0/`#s`/`#x` 三种载体 ⇒ ① 拿不到 `key#xxxx` 静态特化（回落带 `prec` 三元的 `#x`）；② 拿不到 `key#xxxx#b` 批量变体（即便 `#b` 已对每个静态签名变体发射）。修法 = 在 scan 的 `end_expr` 里对 `members.size() > 1` 的 kernel 用**与运行期同源**的口径补记 `add_variant(k.spec, expr_prec_sig_make(输入张量 f16 位图, k.prec == F16))`（等价于运行期 `expr_prec_sig_of(d.ins, out.precision())`；内部依赖输入未绑定张量 → 该槽按 f32 记，纯逐元素复合的成员引用经寄存器直替、不进 views，故正常构图下每槽都是外部输入）。生效后 Adam 复合结构拿到 `#10004` + `#10004#b`；实测（40HX/f16/b64/20 步，配对 `329f8f1` vs `5bb6a51`）：`[ir-c][degrade]` 70 → 0/步、优化器派发 138.7 → 8.55 calls/步（−94%）、优化器 host 阶段 −34%，但端到端墙钟在 GPU-bound（忙率 95.5%）下无可见收益。**边界**：只覆盖 scan 真跑过的 profile（f32 + `--f16`），其它显式组合仍按设计回落降级（构建期 2^4 精度枚举不可取）。

## 代码审查整改批 RV-06/09/10/11/13 与 CI 真跑测试（2026-10-12；证据坐标 docs/todo/00-code-review-remediation.md）
- 类型：bug 根因 + 结构收敛 + 门禁
- 内容：① **RV-06 前置去红**：`NN_BIND_DEBUG=1 ctest` 原为红（`t1_58_test` / `ternary_gemm_test` / `model_roles_test` 三例在 `cast` 处报 "unbound input into engine"）——根因是 `matmul_q_impl`（CPU 与 GPU 两份）在边界 cast 回退里：CPU 递归调用 impl 后、GPU 用静态工厂 `Tensor::from_gpu` 造输出后，都直接调**公共** `cast`（内含 `bind_check_`）而产物尚未 stamp。两处改为 `cast(adopt(std::move(x)), P)`（`adopt` = 基类 protected 的库内 stamp 通道，与 M6 的 `to_prec` 修法同款）；本机 Release+CPU `NN_BIND_DEBUG=1 ctest` 由 37/40 转 **40/40**。CI 侧新增一轮 `NN_BIND_DEBUG=1 ctest` 作为默认门禁。② **RV-09**：环境变量读取收敛到新 `core_env.hpp`（`nn::detail::{env_flag_raw,env_string_raw,env_int_raw}` + 探针门包装；`getenv`/`_dupenv_s` 只此一处、探针门关时恒 false 且不求值）；`dsl::env_flag` 的"任意非空即真"一并对齐为"0 = 关"。③ **RV-10**：BitLinear `quantize_weights_` 与 `ternary_pack::code_of` 的判据注释精确化（严格 `|W|>0.5τ`、半值归零；与 `RoundClip` 的唯一差异在等值点），不改算法。④ **RV-11**：`precision.hpp` 加 `static_assert` 冻结"枚举值 ↔ 序列化 tag"两张映射 + `precision_from_tag(precision_tag(p))==p` 往返（改一处即编译失败）。⑤ **RV-13**：`nn::Expression` / `nn::BoolExpression` 概念从 `expr_dsl.hpp` 移到 L1 的 `algebra_span.hpp`（`expr_dsl` 经 `algebra_matrix → algebra_span` 传递可见）。另（roadmap P0-1）：CI 配置加 `-DNN_ENABLE_TESTS=ON` 并在 Linux/Windows 各跑 `ctest --output-on-failure`，打包补齐 `dataset_gen`/`dataset_convert`（应用入口 6 → 8）。**仍 Open**：RV-05、RV-07、RV-08、RV-12、RV-14（RV-05/14 需 GPU 提权环境；RV-07/08 为需单独提交并逐步过三件套的重构；RV-12 仅设计评估）。

- GPU 实测补充（2026-10-12，NVIDIA CMP 40HX @ ethan@192.168.1.102，Windows + clang++，Debug，GPU 启用）：`ctest` **42/42**、`NN_BIND_DEBUG=1 ctest` **42/42**（全部 GPU 用例真跑、零 skip）。同批带出的 Windows 可移植性缺陷：`src/assert_always_test.cpp` 的裸 `std::fopen` 在 clang++ + MSVC STL + `-Werror` 下编译失败（正是 CI 的 Windows 配置，此前不编测试故未暴露）→ 改 `_MSC_VER` 下 `fopen_s`。

## 代码审查整改批 RV-07/RV-08：算子分派收敛 + VK 后端拆分（2026-10-12；证据坐标 docs/todo/00-code-review-remediation.md）
- 类型：结构收敛（纯重构，零行为变化）
- 内容：① **RV-07**：同一个 `ExprOp` 的分派原先散在约 9 处（CPU 引擎 3 处 switch + `expr_spec` 语义校验 + `expr_opt` 常量折叠 + GLSL emitter 6 处）。新增 `expr_op_table.hpp` 作**唯一真源**——以 `nn::ops::*::apply` 为 CPU 语义，每算子一行元数据（元数 / 归约标志 / 可折叠 / 一元·二元标量函数指针 / GLSL 发射形态 + 符号 + 向量比较内建名），末尾 `consteval` 断言强制与 `expr_spec` 的 `expr_instr_num_operands`/`expr_op_is_reduce`/`expr_op_reduces_cols` 逐项一致。CPU 三处 switch、`expr_eval_const_fold` switch、emitter 的 `glsl_binary_op`/`glsl_unary_op` 与 6 处指令发射块全部改为查表（新增 `glsl_alu_expr` 单一发射器，`b`/`c` 惰性回调）。**新增一个逐元素算子只需表里一行**。② **RV-08**：42 个 `get_*_spirv()` 的手写 `#ifdef` 函数体 + 42 个 `#define NN_*_SPV_EMBEDDED` 删除，改为文件头探测块把字节码函数绑到 `NN_SPV_FN_<name>` + `NN_SPV_GETTER_LIST` X 宏一次展开（新增 shader 只改探测块与列表各一处，调用点零改）。`GpuBackend`（成员体 ~5375 行）拆为 4 个**片段**文件 `backend/compute_vk_backend_{destroy,frame,pipelines,ops}.inc`，由主文件在 class 体内**按序** `#include`（片段不可单独编译）——主文件 6397 → **1055 行**。
- 验收：CPU `ctest` 40/40；**融合 shader 字节锚**用「RV-07 前 / 后」两侧源码各编 `scan_exprs` 生成后 `diff -rq` → 805 个产物（`fused_registry.hpp` + 402 组 `.comp`/`.spv`）**零差异**；拆分的正确性用脚本比对「主文件头 + 4 片段 + 尾」与拆分前文件**逐行相同**；40HX（102）全量构建 + `ctest` **42/42**、`NN_BIND_DEBUG=1 ctest` **42/42**；`bench/doc_inventory.ps1`（`L2-VIOLATIONS: 0`/`HANDWRITTEN-RESIDUE: 0`）与 `bench/doc_align_audit.ps1` 均 exit 0。**仍 Open**：RV-05、RV-12、RV-14。
- 脚注：片段内 `#include` 是本仓首次使用的「单类跨文件」手法（`GpuBackend` 是单一类，纯移动只能如此）。若日后倾向「拆成多个协作类」须单独立项（非纯移动，会改访问边界）。

## f16 次正规转换守卫的移位 UB（原位置 include/neuralnet.cpp/precision.hpp:197，整改前行号）
- 类型：bug 根因
- 内容：`float_to_half_bits` 的次正规分支曾误写守卫 `exp <= -46`，与其自身公式矛盾（公式只对 exp ∈ [-25,-15]、shift 14..24 合法）。exp ∈ [-45,-33]（|v| ∈ 2.8e-14 ~ 1.2e-10）时 shift ≥ 32 → uint32 移位 UB（x86 按 5 位掩码 → d 取满 24 位尾数 → 指数字段回绕成垃圾 half：512/8192/11776/18432，随机落 e=31 即 NaN）。梯度恰在该量级、前向激活 ~0.1 从不落入，故损坏只见于 f16 梯度写回、若干步后更新爆炸；GPU 走硬件转换故"只有 CPU 错"，单算子测试输入 ≥1e-3 故全过。修复 = 守卫改 `exp <= -26`（exp ≤ -26 时 D < 0.5 恒 flush 为 0，无需移位）。整改后注释只保留当前守卫的正确性论证（合法 shift 区间 14..24 < 32、exp ≤ -26 恒 0），删掉"曾误写成"叙事。

## `--f16` CLI 语义与 profile 混淆（原位置 include/neuralnet.cpp/precision.hpp:477、compute_optimizer.hpp:52）
- 类型：语义修正（历史误会）
- 内容：`profile_master_weights()` 曾是 `--f16` 的语义，曾造成"传了 --f16 却是 f32 主权重混合精度"的误解；后修正为 `--f16` = `profile_f16()` = {param:F16, compute:F16, stable:F32, optimizer:F32}（docs 05 §9.4）。`compute_optimizer.hpp` 的状态精度注释也曾把 `profile_all_f16` 标注为"CLI --f16"。整改后两处只保留当前事实：三种 f16 相关 profile（f16 / master_weights / all_f16）各自的字段与区分，删除误会叙事。

## Adam m/v 状态更新的旧 clone 开销（原位置 include/neuralnet.cpp/compute_optimizer.hpp:308）
- 类型：性能 A/B / 删除清单
- 内容：旧实现对 m/v 各做一次 `clone_tensor`（整份模型尺寸的 vkCmdCopyBuffer，逐步 2×model_size 的 Copy）+ 多次逐元素原语 + 多个中间张量；融合改造后为三个 DSL 融合 kernel（K1/K2/K3），超参经 RParam 承载，无任何 clone、无 m_hat/v_hat/sqrt_v/denom/ratio 物化。整改后注释只描述当前三 kernel 结构与 RParam 理由，删掉与旧实现的对比。

## Newton-Schulz 母矩阵边选择（原位置 include/neuralnet.cpp/compute_optimizer.hpp:537）
- 类型：性能 A/B（OOM 根因）
- 内容：旧构造沿长边做 A/A²——对 50257×1024 词嵌入即 50257² ≈ 10GB/个 OOM（Muon 显存 > AdamW 的根因）。当前实现恒选短边：母矩阵 min(m,n)²（该例 1024² ≈ 4MB）。整改后注释保留"为何短边侧构造在数学与显存上都成立"的当前推理，删掉"旧实现…新实现降至"对比。

## Newton-Schulz 每步 dispatch 计数（原位置 include/neuralnet.cpp/compute_optimizer.hpp:549、561）
- 类型：性能 A/B
- 内容：`A = b·A + c·A²` 原为 scale(A²,c) + scale(A,b) + add(A,A²) 三次 dispatch（原实现合计 7 次/步），融合为一次原地目标传递后合计 5 次/步。整改后注释只保留当前"每步 5 次 dispatch"的构成与"单 kernel 原地目标传递"，删掉 7→5 演进叙事。

## 稠密 CE 旧列广播路径（原位置 include/neuralnet.cpp/compute_loss.hpp:180）
- 类型：性能 A/B / 删除清单
- 内容：旧路径 = clone 一份 (classes,batch) 整块 vkCmdCopyBuffer + 3 次就地广播 dispatch；现 shifted/softmax/log_softmax 全部为列广播表达式，各 1 个融合 kernel、零中间拷贝。整改后注释只陈述当前"无中间拷贝"的事实。

## CE 稀疏/稠密路径引用已删除原语（原位置 include/neuralnet.cpp/compute_loss.hpp:274、277、332）
- 类型：删除清单
- 内容：注释曾引用 `col_softmax_denom`、`col_softmax_sparse_forward`（M5 期 op-level 原语，已随 S7 删除，见 docs/todo/02 §"S7 删除 M4-M6"）以及"融合原语不可用（如 CUDA 未对齐返回未实现）"的 CUDA 例子（CUDA 后端已整体移除）。当前实现为 S7 IR 组合（col_max 原语 + denom/loss_vec/grad 表达式）；labels 以 (1,total) 浮点打包，前置条件 vocab_size ≤ 2^24，失败直接透传错误。整改后注释改为当前结构 + 当前前置条件。

## CpuEngine::matmul_with_bias 覆盖前的 GPU host 往返（原位置 include/neuralnet.cpp/compute_cpu_engine.hpp:673）
- 类型：bug 根因 / 性能 A/B
- 内容：此前 GPU Linear 每步做 to_matrix/from_matrix 的 CPU 往返（走基类默认"matmul + host 加 bias"），且该 DSL 结构从未被 scan_exprs 扫描 → "Linear 结构"融合测试 AOT 未命中。现 CPU/GPU 两引擎均以 DSL 表达该结构，scan dry-run 收集 → gen_fused 生成融合 kernel。整改后注释只保留"覆盖基类默认 + scan 收集该结构（否则闭合世界缺 key）"的当前理由。

## fold 视图白名单两处清单曾失同步（原位置 include/neuralnet.cpp/compute_cpu_engine.hpp:1406）
- 类型：bug 根因
- 内容：CPU 解释器（compute_cpu_engine）与 expr_spec 的 fold validate 两处白名单曾漏同步，致 RowBroadcast(doc_col) 被拒、forward 报错被 scan dry-run 的 `(void)` 吞掉、backward 拿空缓存触发 NN_ASSERT。整改后注释保留"两处清单必须同步维护"的当前契约与后果，删掉事故叙事（dry-run 吞错已另修）。

## fold 标签表只查 rows 的越界读（原位置 include/neuralnet.cpp/compute_cpu_engine.hpp:1421）
- 类型：bug 根因
- 内容：标签表（BatchMod/BatchCol）此前只校验 rows，形状违约时静默越界读；现守卫 cols（BatchMod: cols ≥ param；BatchCol: cols ≥ (batch-1)*param + K）。整改后注释保留当前守卫规则与"只查 rows 不足"的理由。

## fold 状态复位与 vecd 跨行残留（原位置 include/neuralnet.cpp/compute_cpu_engine.hpp:1589）
- 类型：bug 根因
- 内容：vecd 等临时/行向量态槽曾在循环外声明，跨行残留累加、err 随行数滚雪球；现每行开头统一复位状态初值 + 归约槽 + 行向量态。整改后注释保留当前"每行独立复位"的理由，删掉"曾漏在循环外声明"的修复叙事。

## CPU 解释器拒收 f16 输入（原位置 include/neuralnet.cpp/compute_cpu_engine.hpp:1711）
- 类型：bug 根因
- 内容：解释器按 f32 存储读取 span，f16 输入直读 = 空指针 UB，曾是 CPU f16 训练 NaN/AV 家族成员；现为响亮报错（"CPU 解释器仅支持 f32 输入"）。整改后注释保留"直读即 UB，故响亮报错"的当前理由，删掉历史故障家族标注。

## PrecisionEngine 头部"零 shader 变体"论述已被推翻（原位置 include/neuralnet.cpp/compute_precision_engine.hpp:19-24）
- 类型：否决方案 / 性能 A/B
- 内容：头注释原主张"不做引擎内逐原语 f16 分支（§11.1 Phase 2 才做）、边界 cast 换零 shader 变体/零注册表膨胀、收益仅在带宽（显存占用由存储决定）"。后续实测（docs 05 §12.8-§12.11）证明逐算子边界 cast 使 transient 膨胀约 2.4×、f16 峰值反超 f32；带类型 in-kernel f16 变体（逐元素/归约/matmul 段，gen_fused 54 变体）落地后峰值显著下降。整改后头注释改为当前结构：**变体优先（f16 直读直写、零边界临时量）→ 无变体时回退边界 cast（任何 (结构,签名) 都有正确路径，代价是带宽 + f32 临时量）**，并保留"为何不做逐原语手写 f16 分支"的现行理由；同时 in-place 原语清单删去已不存在的 `axpy`/`broadcast_*`。

## compute_engine 接口注释中的 CUDA 残留（原位置 include/neuralnet.cpp/compute_engine.hpp:89、93）
- 类型：删除清单
- 内容：`release_idle_pool_blocks` / `pool_stats` 注释曾写"CPU/CUDA 引擎"——CUDA 后端已整体移除（见 AGENTS §2），现仅 CPU（no-op/返回空）与 GPU。已改为当前引擎清单。（**M5 2026-09-30 后**：CPU 不再返回空串，改报宿主直配账本，见下方"统一总纲 M5"条。）

## Tensor reshape 头注释与实现不符（原位置 include/neuralnet.cpp/compute_tensor.hpp:291）
- 类型：错误注释（文档失真）
- 内容：头注释原称"零拷贝 reshape（共享底层 buffer）——CPU/Vulkan 均适用"，但 CPU 分支并无零拷贝视图能力、实际复制数据（与 AGENTS §10.6 记载一致）。整改后改为如实记录当前语义：GPU 共享 buffer（零拷贝）、CPU 复制到新形状 Matrix。

## ComputeEngine Refresh P-1：PrecisionEngine 装饰器删除与 NVI 下沉（原位置 include/neuralnet.cpp/compute_precision_engine.hpp 整文件（948 行，46 override）；compute_engine.hpp 类结构）
- 类型：演进记录 / 性能 A/B
- 内容：P-1（2026-09-29，方案见 docs/development/15 §4.1）把 f16 边界 cast 装饰器 `PrecisionEngine`（每次现问内层 `inner_` 要不要 cast、49 方法全量委托）下沉为 `ComputeEngine` 基类 NVI：公有非虚入口承载边界 cast 逻辑（变体优先 / 全 f32 快速直通 / 抬 f32→算→按 P 落回），protected 虚 `*_impl` 只管算；CPU 33 / GPU 29 处 override 机械改名 `*_impl`（基类 virtual 总数仍 49，仅公有虚变非虚 + 新增 `_impl`）；`text_train` / `mnist_train` / `mem_probe` / `f16_*` / `scan_exprs` 等 7 处使用方迁移（`optional<PrecisionEngine>` 舞步与 `dump_temp_stats` 静态调用点改指基类），`compute_precision_engine.hpp` 整文件删除。验收：dev1 ctest 20/20（含 f16_precision/gpu_f16/writeback 4 个 f16 用例）、CPU 稳定性探针与 P0 基线逐字节、scan 产物（expr_specs.bin / fused_registry.hpp）hash 不变。**性能 A/B**：layer_bench 单轮小算子 ±30% 摆动且方向不一（linear/layernorm +25% vs mha -7%）→ 交错 4 轮复测方向翻转（layernorm +25%→-7%）→ 判定手机 DVFS/调度噪声主导、未测得系统性回退；精确数字留待桌面平台复测。整改后代码/注释只保留"基类 NVI 边界 cast 入口"的当前结构，适配层叙事仅存于本文与 docs 15/05。

## ComputeEngine Refresh P1：张量出生绑定与跨引擎检查（原位置 compute_tensor.hpp 类结构、compute_engine.hpp 公共入口集、expr_dsl.hpp compute/compute_reduce 出口）
- 类型：演进记录 / 性能 A/B
- 内容：P1（2026-09-29，方案见 docs/development/15 §3.1/§4.2/§4.3/§7.2）按"只加检查、不删形参"落地：① `Tensor` 加 8B `observer_ptr<ComputeEngine> engine_`（friend ComputeEngine）+ `bound()`/`engine()`（未绑定 `NN_ASSERT`）；拷贝/`reshape` 传播绑定，`Tensor{}` 空槽与库外直构保持不绑定。② 工厂与单操作数虚入口 NVI 化——`create_tensor`/`from_matrix`/`cast`/`create_offload_buffer`/`offload_restore` 改为公共非虚包装（尾部 `stamp_`），CpuEngine/GpuEngine 各 3+5 处 override 改名 `*_impl`（基类 virtual 总数仍 49，与 P-1 口径一致）。③ 20 个返回 Tensor 的公共入口逐个包 `return stamp_(...)`（含 `to_prec` 出口统一行替换）：未绑定才补、已绑定沿用 src 绑定（传播规则）、无效空槽不 stamp。④ 跨引擎检查 `bind_check_` 按 D3 字面：操作数指针判等，**双方都 bound 且不同 → Result 硬错误**；单侧未绑定按库外豁免放行（P1 零行为变化，ctest 不红）；`NN_BIND_DEBUG=1` 时"未绑定输入进引擎"也报错（带 source_location 与张量形状）；插入约 28 个公共入口顶部。⑤ 库内 stamp 通道 `ComputeEngine::adopt(Tensor&&)`：friend 限定给 `nn::dsl::compute`/`compute_reduce`（compute_engine.hpp 前置声明 dsl 模板——**默认实参必须写在前置声明处，C++ 不允许在后续声明追加**），覆盖 `eval_cpu` 出口、扫描占位张量、`reduce_vector_tensor` 归约向量出口（15 §4.2 片段 A/B）。验收（本机 Debug+Ninja+clang）：ctest 20/20 两轮全绿；scan 产物 hash 不变（`expr_specs.bin`=bdc3a442…a58360 与 15 锚点一致、`fused_registry.hpp`=7a10412c…；后者含 SPIR-V 字节，仅本机 pre/post 可比）；`gpu_stability_probe` CPU 两轮输出与 P0 基线**逐字节**；**性能 A/B（stash 往返重建）**：layer_bench linear/layernorm/mha/transformer 全部 ±6% 内（linear 反而 -6%，其余 ±1%，噪声内无回退），`text_train` 9 步 244.7s（基线）vs 245.0s（P1）持平。`NN_BIND_DEBUG=1` 诊断：mnist MLP（CPU/GPU）/CNN/Transformer、text_train gpt/rapt、zipt_test 全部零漏网（库内产物出生即绑定成立）。顺带修复既有构建失败：`src/gpu_stability_probe.cpp` 的 `std::getenv` 在 MSVC CRT `-Wdeprecated-declarations -Werror` 下编译不过（照仓库 `_dupenv_s` 模式改写）。观察到的既有问题（基线同现、非 P1 引入，未修）：`gpu_stability_probe --gpu` 首个 `vkCreateBuffer` 即崩（16 §7-2 未决）；`text_train --model zipt` 启动即 `set_checkpoint_every(0)` 无条件 abort 且 abort 后挂死（compute_layer_zipt.hpp:850 未判 stride==0）。整改后叙事收敛为当前结构：绑定字段/`stamp_`/`bind_check_`/`adopt` 各就其位，实施过程仅存于本文与 15。

## 统一总纲 M1：Tensor 访问收口——存储私有化 + reshape 引擎化 + 库外全迁（2026-09-30，原位置 compute_tensor.hpp 公共面、compute_engine.hpp 新增 reshape、expr_dsl.hpp 求值出口、model_serialization.hpp save f16 分支、src 9 测试文件 + text_train.cpp:756 + tools/scan_exprs.cpp）
- 类型：演进记录 / 性能 A/B
- 内容：M1（方案见 docs/todo/17 §4.1/§5 M1 行；commits `3d76478` 库外全迁、`c8f308f` 访问收口）分两段落地。① **私有化**：`Tensor` 的静态直构工厂（`from_matrix`×2/`cpu`×3/`cpu_uninitialized`/`from_gpu`×2）、存储构造、访问器（`cpu_matrix`×2/`cpu_shared`/`gpu_tensor`×2/`gpu_shared`）、`reshape` 全部转 private；friend 集 = `ComputeEngine/CpuEngine/GpuEngine` + 新增 `detail::TensorAccess`（前向声明于文件头、定义于尾部；`nn::dsl` 内 `using TensorAccess` 别名，eval_cpu 出口/扫描占位/求值叶子共 **19 处**改道）。公共面收敛为形状/精度/设备查询、`bound()/engine()`、`valid()`、`shape_str()`——绕过引擎 = 编译错误（铁律 #11 新立于 AGENTS §5）。② **`ComputeEngine::reshape`（D10）**：`bind_check_` + `stamp_` + 元素数不匹配返回 Result 错误（原 NN_ASSERT 升级）；`Tensor::reshape` 实现保留为私有成员供引擎调用（逻辑零搬移）。③ **库外全迁（D7，不留过渡期）**：`test_common.hpp` 新增 `upload/download` 助手（Result 失败 NN_ASSERT 兜底），`make_tensor`（48 调用点）/`check_close`（18 调用点）签名加 engine 首参；9 个测试文件读路径一律 download 值接收（防 `auto&` 悬垂）、写路径一律"本地 Matrix 写好 → upload 回传"（rng 消耗顺序逐位保持，断言/容差/打印原字节）；`fused_gpu_test` 双引擎对拍 32 个输入张量 cpu/gpu 各上传一份（出生绑定检查下跨引擎喂食 = 硬错误，逼出拆分）；`text_train.cpp:756` CPU/GPU 分支合一走 `engine.to_matrix`；`scan_exprs.cpp` 的 `scan_tensor` 收编为 dry_run lambda 内 `engine.create_tensor(rows,cols,g_scan_prof.compute)`（60 处调用点零改动，出生绑定与 dry-run 引擎一致）。④ **model_serialization save_model f16 分支**：CPU 直读 `cpu_matrix<F16>` 删除，CPU/GPU 统一经 `engine.to_matrix` 下载再转 f16 落盘（f16→f32→f16 往返精确，v5 字节不变）。⑤ `tensor_precision_test` 按引擎 I/O 重写：`cpu_shared` 别名断言随私有化取消（无公共等价物）、新增 reshape 元素数不匹配拒绝用例。**教训**：旧审计 pattern `cpu_matrix\s*\(` 漏掉**模板形态** `cpu_matrix<Precision::F16>()`——`model_serialization.hpp:550` 即此形态漏网，审计须用裸词 `\.cpu_matrix`。**验收**（本机 Debug+Ninja+clang）：build 122/122；ctest 20/20；`gpu_stability_probe --steps 20` hash=`6f8849f14da23110` 与 M1 前基线**逐位一致**；scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…）；库外 grep 旧 API 零命中（编译期强制成立）。**性能 A/B（配对裁决）**：首轮 --all 单点对拍显 feedforward fwd/train +6~7%——但基线采集期并发了子代理语法编译且为单样本；按配对方法重建 pre-M1 对照二进制（git worktree @ HEAD~2 独立构建 layer_bench），feedforward/swiglu/linear 各 4 轮 pre/post 交错：**分布重叠、方向混合**（feedforward train 中位 3229.7ms vs 3228.3ms，swiglu train 均值 10.28 vs 10.46，linear 落在彼此散布内），无系统性回退；附带实测本机 Debug layer_bench 单点 run-to-run 摆动可达 ±25%（linear fwd 0.537~0.860ms），单点不可裁决——机制上 M1 只动冷路径与调用点层（`cpu_get_ptr` 热循环零变化）。四子代理并行迁移 + 主线统一构建的分工可行（子代理仅做 `-fsyntax-only` 与 %TEMP% 单 TU 运行，不碰 cmake）。

## 统一总纲 M2：声明式初始化——InitSpec + Layer::init 迁移 + RNG 收编（2026-09-30，原位置 compute_engine.hpp 张量工厂区、compute_layer_{mlp,conv,gpt,zipt,rapt,transformer}.hpp 的 init 函数、src/gpu_stability_probe.cpp）
- 类型：演进记录 / 性能 A/B
- 内容：M2（方案见 docs/todo/17 §4.4/§5 M2 行）分三段落地（commits `8bd6bd5` 核心、`4492098` 层迁移、文档段随后）。① **`InitSpec` + 四参 `create_tensor`**（compute_engine.hpp）：`Kind{Uninitialized,Zero,Constant,Uniform,Normal}` + 工厂（分布类 **seed 必填** = U1 裁定，层传 `kInitSeed=42`）；四参 `create_tensor(rows,cols,P,spec)` 为公共非虚 NVI——Zero/Uninitialized 走既有 `create_tensor_impl`（Zero 另补 `zero()`），分布类 = 引擎内 host 生成后 `from_matrix` 上传（D6 策略调用方不可见）：`fill_host_` 用 mt19937_64，种子 = splitmix64(spec.seed × 引擎内创建序号 `init_seq_` 混流)——**同 seed 不同张量不撞流**（否则 GPT 同形状多层 Linear/注意力投影会互为镜像）；3 参 `create_tensor` 保持既有纯分配语义不动作（初值要什么显式走 InitSpec，零仍可显式 `zero`）。② **Layer::init 全迁**（6 头文件 25 处）：Linear/Conv2D 权重 `uniform(-limit,limit)`（层算 limit、引擎填数）、bias/梯度缓冲 `zero`、LayerNorm `gamma=1/beta=0`、RMSNorm `gamma=1`、GPT/ZiPT/RAPT token_emb `normal(0,0.02)`、LearnedPositionEncoder `normal(0,0.02)`（`AdditivePositionEncoder::init_` 入参 Matrix→Tensor，Sinusoidal 公式路径改走 `from_matrix`）、CrossAttention 记忆查询 P `uniform`、TransformerEncoder `ones_row_` `constant(1)`；mlp/conv 的 `thread_local rng_` 删除、init 侧 `random_device` 清零。**精度口径逐点保持**：conv/zipt/rapt 原 `from_matrix` 默认 F32 → create_tensor 显式 F32，gpt/norm 原 `p_.param` 照传——不借机改 profile。③ **裁定不迁（运行期随机性，17 §1 表已注）**：`generate()` 采样 RNG（gpt:954/zipt:1041/rapt:1461）、`text_train.cpp:1270` 数据洗牌 RNG——均非 init。④ **探针新增 `--init-hash` 模式**（gpu_stability_probe）：6 类模型（mnist_mlp/cnn/mnist_transformer/gpt/zipt/rapt）建后即哈希（不覆写），输出 `INIT1/INIT2` 固定前缀行供跨进程 diff；默认模式头注释更新——init 已确定性化后，覆写保留是为了**隔离 init 变量**、只测后端执行确定性。
- **计划内字节变化（17 §6 例外，新锚已立）**：原 random_device 播种站点与 seed-42 站点的 RNG 流（改引擎序号派生）字节全变。新锚（2026-09-30，两次独立进程逐字节）：`gpu_stability_probe --init-hash` 六模型 hash = mnist_mlp `a22e807ee05ec3ac` / cnn `4020958a14160bbd` / mnist_transformer `04a72d865624042a` / gpt `8efa936ac5c8c9b2` / zipt `f90bf8c8783c2f89` / rapt `b037632f75b7159c`；`mnist_train` 三架构（`--epochs 1 --max-samples 512 --batch-size 64 --shuffle-steps false`）双进程 loss 序列 + 模型文件逐字节一致（模型 SHA256：mlp `B596BB19…85A1`、cnn `19A29E0B…6FE7`、transformer `5D533039…2ADA`，仅墙钟行不同）。**反例定位**：`text_train` 双跑 loss 不等（9.0546 vs 9.0566）→ 根因 `text_train.cpp:1270` `mt19937_64{random_device{}()}` 数据洗牌——M2 前即存在、与 init 无关（init 已确定性后它成为 text_train 跨进程差异的唯一来源），范围外未修；要 text_train 全链路复现需另立 --seed 类开关。
- **验收**（本机 Debug+Ninja+clang）：build 122/122 零告警；ctest 20/20；`gpu_stability_probe --steps 20` hash=`6f8849f14da23110` **与 M1 锚点逐位一致**（该探针覆写 init，证明后端执行路径零变化）；scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…）；`--init-hash` 两进程 `INIT1` 行全同 + 进程内两轮 PASS；`NN_BIND_DEBUG=1` mnist 冒烟零诊断。**性能 A/B（配对 + 逆序复测）**：pre 基线 = git worktree @ M1 HEAD（735594d）独立构建 layer_bench，同窗交错；feedforward 正序 4 轮 fwd/train POST 中位 +3.0%/+2.9%，**逆序 4 轮（POST 先跑）复测差异依旧**（+3.1%/+3.5%）→ 排除运行顺序热偏置，为 Debug(-O0) 二进制重排/内联漂移级小差（M2 未改动任何热路径语义，init 仅建模时执行一次）；swiglu fwd/train ±2% 分布重叠；linear 分布交叉重叠（已知单点 ±25% 方差，M1 记录在案）。全部 < ±6% 验收带，**无系统性回退**。

## 统一总纲 M3：批量读写 API——read/write/get_index/set_index 落地 I/O 分组（2026-09-30，原位置 compute_engine.hpp I/O 分组区、src 10 个 gradcheck/一致性测试文件、src/gpu_stability_probe.cpp）
- 类型：演进记录 / 附带缺陷修复
- 内容：M3（方案见 docs/todo/17 §4.3 D3/D9、U2；commits `e0338d9` 核心、`4977a29` 探针、`3ddb13c` 测试迁移）分三段落地。① **`compute_engine.hpp` I/O 分组区**：`from_matrix/to_matrix/read/write/get_index/set_index` 六动词同组成文（D9：保留 `from_matrix/to_matrix` 原名，15 P2 的 `to_host/from_host` 改名随之取消）；新增 **基类非虚模板** `read(const Tensor&, span<T>)` / `write(Tensor&, span<T>)`（不进 49 virtual 骨架）——`read` 统一经 `to_matrix`（GPU 分支自带 `end_batch`→`wait_in_flight`→新帧，即"隐含 flush+同步"），`write` 宿主直写既有存储、GPU 走 `copy_from` 既有 drain 语义（F16 目标 = f32 上传 + `cast_into` 写回原存储），二者都**覆盖既有存储、不替换对象**（沿 `copy_into` 红线）；`get_index/set_index` 为语法糖（宿主直读写零往返，GPU = 整张批量往返，明确不给热循环承诺）。② **U2 精确匹配**：span 元素类型须与 `t.precision()` 严格一致（`float`↔F32、`nn::f16`↔F16）——元素类型非 float/f16 或 `read` 的 T 为 const = `static_assert` 编译期报错，精度错配/元素数不符/越界 = `Result` 运行期错误；`write` 形参用 `span<T>` 而非 `span<const T>`（模板实参推导中 `const T` 无法匹配非 const span，推导失败），静态断言按去 cv 后元素类型校验。`read` 对 f16 存储升 f32 后按 T 落回（RHE，往返位不变）。③ **调用约定成文**（17 §7-1/§7-4、铁律 #6）：read 隐含 flush、写入张量须存活到 `end_batch()` 之后；逐 step loss 回读继续走 `submit_scalar_readback` 快路。④ **测试迁移**（10 文件）：gradcheck 扰动-恢复、CPU→GPU 权重同步等 **f32 路径 `copy_from` → `write`**；f16 专项（`f16_cpu_probe`/`f16_precision_test`/`f16_writeback_probe`）**保留 `copy_from`**——其语义就是 f32 Matrix → f16 的转换填充（RHE），正是 U2 精确匹配下 `write` 不承担的转换入口，故两动词分工成文而非并吞。
- **附带缺陷修复（16 §7-2 根因，P0 遗留未决项收口）**：`gpu_stability_probe` GPU 模式首个 `vkCreateBuffer` 即 "Invalid device" 崩溃——根因 = `GpuBackend::instance()` 只是惰性单例，Vulkan 设备须显式 `backend.initialize()` 才建立，探针 `make_engine` 漏调（`cli_engine_factory` 与全部 GPU 测试入口都会调它，这正是 16 §7-2 当时记的"差异在探针路径"那一处），首个 buffer 拿到 VK_NULL_HANDLE 即崩。修复 = 探针建引擎前补 `initialize()`（空 selector 仍按 显式 > `NN_VULKAN_DEVICE` > 自动打分 选卡）。修复后 GPU 档位首次可测：Windows 本机 **dev2 `8ef51b2927253c50`、dev0 `47977edc71a21a20`、dev4 `01b7192bd15a59d9`** 进程内两轮 + 跨进程两次启动 loss/hash **逐字节一致** → D9 的 GPU 验收档位在本机坐实为**逐字节**；dev1（MESA/D3D12 转译）初始化即报 DXIL container 校验错误、exit=2，为设备级不可用（非探针/库问题，另记）。
- **探针新增 `--io-roundtrip` 模式**（M3 专属验收，GPU 模式即 staging 验收）：三用例 = `f32_roundtrip`（写后读回 + `read` vs `to_matrix` 交叉对拍）、`f16_roundtrip`（f16 精确匹配与 RHE 往返）、`batch_window`（铁律 #6：录制窗口内 write → read → `end_batch` → 复读值不变），另覆盖索引往返、越界、U2 错配错误路径；输出 `RESULT io_cases=3 io_failures=0 verdict=PASS`。
- **验收**（本机 Debug+Ninja+clang）：build 全绿零告警；ctest 20/20；`--io-roundtrip` CPU + GPU（dev2）三用例全 PASS；`gpu_stability_probe --steps 20` hash=`6f8849f14da23110` **与 M1/M2 锚点逐位一致**（I/O 组为纯新增、未动任何计算路径）；scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…）。

## 统一总纲 M4：Matrix 降级收口——L2+ 禁用 + span 宿主桥 + 分层审计（2026-09-30，原位置 compute_layer_{attention,base,conv,gpt,rapt,transformer,zipt}.hpp、compute_loss.hpp、compute_optimizer.hpp、compute_engine.hpp 尾部、bench/doc_inventory.ps1）
- 类型：演进记录 / 新增裁定（D11、U5）
- 内容：M4（方案见 docs/todo/17 §4.6、§5 M4 行）分四段落地。
  ① **先立审计再迁移（编译器/grep 驱动）**：`bench/doc_inventory.ps1` 新增第 [4] 节——它本就枚举 L2 文件集（`compute_layer*.hpp` + `compute_loss.hpp` + `compute_optimizer.hpp` + `model_container.hpp`），逐行剥离 `//` 与块注释（行内状态机处理跨行注释）后匹配三类：`Matrix`/`MatrixT` 类型（**硬**）、`from_matrix/to_matrix/copy_from`（**硬**，签名即含 Matrix）、宿主桥动词（**披露**，只计数不判违规）；末行输出 `L2-VIOLATIONS: N`（验收口径 0）。**首跑违规模基线 = 77 条（32 处 `Matrix` 类型 + 45 处 Matrix 型 I/O 动词），分布在 10 个头文件**。
  ② **D11 裁定（新增到 17 §3，10 项 → 11 项）**：17 §4.3 原字面"from_matrix/to_matrix/read/write 只准出现在 I/O/测试代码"在 L2 上**不可执行**——层确实需要生成索引/位置/掩码类辅助数据，而引擎没有 arange / RoPE 表 / 斜率表这类生成原语（要严格执行得先加原语，超 M4 范围）。裁定把禁令本体收窄为**`Matrix` 类型**（数据格式统一的可审计判据）：Matrix 型三动词随之在 L2 全禁，span/标量级作为受控宿主桥保留（判据 = 不引入 Matrix 即不破坏统一），大批量 I/O 仍归 I/O 层。U5 同时裁定：审计落点 = 扩展 `doc_inventory.ps1`，不另建脚本。
  ③ **宿主桥三个自由函数**（`compute_engine.hpp` 尾部 `namespace detail`，0 虚表项、0 新原语）：`upload_span(engine, rows, cols, P, span<const Scalar>) -> Result<Tensor>`（= `from_matrix` 的 span 形态；f16 目标按 `f16{float}` RHE，与 CPU `from_matrix_impl` 的 `dst[i] = src[i]` 同一条舍入路径）、`download_span(engine, t, span<Scalar>)`（= `to_matrix(t, F32)`：f16 存储先升 f32 再落，升 cast 精确无损）、`download_vector(engine, t)`（取值便捷形态）。三者内部只调 M3 的批量 `write/read`（GPU staging、隐含 flush 照旧），不进 49 virtual 骨架。
  ④ **10 个头文件全量迁移（45 处 I/O + 32 处类型）**，逐类：RoPE `fill_pos_column_` 签名 `Matrix&`→`span<Scalar>`+cols（`compute_layer_attention.hpp`）；正弦/可学习位置索引与采样 1×1 词元 id（`compute_layer_gpt.hpp`，`pe` 公式数据改 `upload_span(..., p_.param)`——f16 param 转换与原 `from_matrix(pe, p_.param)` 逐位同）；ALiBi 斜率表/偏置、doc_ids 的 `(1,BH*seq)` 与 `(BH*seq,1)` 双布局（attention）；RAPT 的 `V_ones` 改 `InitSpec::constant(1)`、`e_0` 稀疏表、逐头 RMSNorm 的 1/rms 缓存（下载→行缓冲→上传）、文档边界、decode 步 `den = q'·z` 主机点积（`to_matrix`×2 → `download_vector`×2，算术顺序不变）；ViT patch 提取与梯度散射（`compute_layer_transformer.hpp`，`at_unchecked(row,col)` → 行主序 flat 下标，**布局等价**）；ZiPT 联合掩码/滑窗输入/logits 列读取；CE 稀疏路径 labels+mask 打包与三处 loss 标量下载；优化器两处 (1,1) 范数下载；`nn_dbg_scan` 诊断下载；conv 的置换索引与分组广播索引。零梯度返回张量（gpt/zipt/rapt 的 `grad_input`）改 `create_tensor(..., InitSpec::zero())`，省一次主机上传。
- **教训（审计工具）**：给 `compute_engine.hpp` 的**注释**里写"49 virtual 骨架"会让 `bench/doc_align_audit.ps1` 的 `virtual_in_engine` 实测数 50→51、与文档 49/50 口径打架——注释里用中文"虚函数"即可；数字型断言的复现命令只看裸 token，不看语境。
- **验收**（本机 Debug+Ninja+clang）：① **审计零违规**：`L2-VIOLATIONS: 0`，`host_bridge_uses: 41`（披露）；② build 122/122 零告警；③ ctest 20/20；④ **字节零变化**（迁移是纯载体替换：`Matrix(rows,cols)` 与 `std::vector(n)` 同为零填充、行主序 flat 下标 = `at_unchecked(r,c)`、f16 转换同一条 RHE 路径）——CPU `--steps 20` hash=`6f8849f14da23110`、GPU dev2 `8ef51b2927253c50` **均与迁移前逐位一致**；⑤ scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…）——`scan_exprs` 的 dry-run 走 Layer forward，表达式文本未动；⑥ **layer_bench 配对 A/B**（pre = `git worktree add` @ 48189d4 独立构建 layer_bench，feedforward/swiglu/linear 各 4 轮 PRE/POST 交错，`--iter 4`）：feedforward fwd/train **-1.2%/-1.6%**、linear +1.4%/-4.2%、swiglu fwd -6.2%/train +0.2%——swiglu fwd 为 0.38–0.47ms 亚毫秒项、PRE 自身极差即 ±10%，全部落在 ±6% 噪声带且方向混合，**无系统性回退**；⑦ `doc_align_audit.ps1` A/D/E/F 可行动项全 0、`virtual_in_engine` 仍 50。
- **连带数字更新**：doc 12 的"Layer 直调 23 → **21**"（`from_matrix`/`to_matrix` 退出 L2；宿主桥是自由函数，不进 `engine.<op>(` 统计口径）与"其余 26 → 28"，§2.2 顺带修正 P-1 遗留的陈旧行 `cast_into/copy_into 仅 compute_precision_engine.hpp 内部`（该文件已删，实际调用方 = 基类 NVI 边界 cast + 引擎互调 + `write` 的 f16 目标）。AGENTS.md 新立**铁律 #12**（L2+ 禁用 Matrix + 宿主桥 + 审计口径），§3 审计表登记 `doc_inventory.ps1` 第 [4] 节。

## 统一总纲 M5：内存池契约统一——无池引擎 pool_stats 补齐（2026-09-30，原位置 algebra_matrix.hpp 顶部账本与 MatrixT 分配/释放路径、compute_engine.hpp 基类 pool_stats/release_idle_pool_blocks 注释、docs/todo/04-memory-optimization.md）
- 类型：演进记录
- 内容：M5（方案见 17 §4.5 / §5 M5 行）按"统一的是**接口与语义**、不是实现（D5）"落地，两段：
  ① **宿主直配账本 `nn::HostAllocLedger`**（`algebra_matrix.hpp` 顶部，L1）：`live_blocks/live_bytes/peak_bytes/total_blocks/total_bytes` 五个 `std::atomic<std::uint64_t>`，全程 `memory_order_relaxed`，**只在分配/释放路径记账，元素热循环零参与**；峰值用 load + CAS 循环（只在变大时写）。全进程唯一实例由 `inline` 函数内静态对象提供（header-only，跨翻译单元同一地址）。
  ② **`MatrixT` 三处释放出口收口**：分配在 `allocate_()` 成功后 `on_alloc(n * sizeof(element))`；释放统一走新增私有 `release_storage_()`（记账 + `data_.reset()` + `size_=0`），**析构 / 移动赋值 / `resize()`** 三处调用——移动赋值与 `resize` 原本是"直接覆盖 `data_`"的隐式释放，若只挂析构会漏记（账本越走越偏）。移动构造把源 `size_` 置 0（原实现已有），不重复记账。
  ③ **`ComputeEngine::pool_stats()` 基类默认实现**：由返回空串改为返回 `direct{ blocks=<活分配数> live_bytes=<> peak_bytes=<> total_blocks=<> total_bytes=<> }`（单位字节，key 显式带 `_bytes` 避免与 GPU 的 `total=..MB` 混淆）；`GpuEngine` 的 `persist{…} transient{…} pending=…` override 不动。`release_idle_pool_blocks()` 保持 no-op，但把理由（"直配无整块可归还，语义成立"）写进 `compute_engine.hpp` 注释与 `04-memory-optimization.md`——M4 之前那句"CPU 为 no-op/返回空"的口径一并更正（history 的同名旧条目已加后注）。
- **未做（保持 U3）**：CPU 真内存池是独立优化立项，需先分配剖析量化 DSL 临时张量热点；本期只补契约不改分配策略，`Matrix` 仍是 `std::unique_ptr<element[]>` 直配。
- **验收**（本机 Debug+Ninja+clang）：① **两引擎 `pool_stats` 非空且口径已文档化**（17 §4.5 + `04-memory-optimization.md` + `01-compute-engine-development.md`）——`NN_MEM_STATS=1 text_train` 实测抓取：CPU `engine-init` 全 0 → `model-built` `direct{ blocks=44 live_bytes=9168512 peak_bytes=9168512 total_blocks=56 … }` → `optimizer-created` `blocks=88 live_bytes=18337024 …`（随生命周期单调反映），GPU dev2 `persist{blocks=5 allocs=12 total=1028MB …} transient{…} pending=0MB` 照旧；② build 122/122 零告警；③ ctest 20/20；④ CPU `--steps 20` hash=`6f8849f14da23110` **与 M1–M4 锚点逐位一致**（账本只读计数，不进任何数值路径）；⑤ scan 双 hash 不变；⑥ `--io-roundtrip` CPU + GPU(dev2) 三用例全 PASS；⑦ L2 审计仍 `L2-VIOLATIONS: 0`；⑧ **layer_bench 配对 A/B**（pre = `git worktree add` @ d28c899 独立构建，feedforward/swiglu/linear 各 4 轮 PRE/POST 交错 `--iter 4`）：feedforward fwd/train **+0.3%/+1.1%**（分配最密集的算例，说明 relaxed 原子开销不可测）、swiglu fwd -5.7%/train +0.1%、linear fwd -3.0%/train **-13.0%**（POST 更快；linear 单点方差 M1/M2 已记录在案 ±25%，且两组分布交叉），**方向混合、无系统性回退**。

## 统一总纲 M6 段 A/B：15 D6/D7/D8 落地 + NN_BIND_DEBUG 门禁 + ensure_gpu→import（2026-09-30，原位置 model_serialization.hpp v5 加载分支、model_container.hpp set_engine、compute_engine.hpp（matmul_with_bias_impl 默认体 / to_prec / bind_error_ / 新增 import 与 import_impl）、expr_dsl.hpp 归约向量 F16 分支、compute_gpu_engine.hpp ensure_gpu）
- 类型：演进记录 / 新增引擎契约 / 门禁化
- 内容：17 §5 M6 分三段，本轮交付 **A、B**；**C（删每调用 engine 形参）未实施**，交接写在 17 §8。
  ① **段 A = 15 D6/D7/D8 裁定落地 + 门禁**（commit `8ecf1bd`）：
  - **D6-1（真缺陷，修）**：`load_model` v5 两支都白做——同精度支 `from_matrix(file_matrix, target_p)` 上传后**丢弃结果**再 `copy_from`（每参数一次全量上传白做），跨精度支"上传 → `cast` → `to_matrix` 下载 → 再 `copy_from`"（多两个 PCIe 往返）。因为 `file_matrix` 恒为 f32 宿主值（f16 文件数据读入时已精确升 f32，f16→f32→f16 位不变），四种 `file_prec × target` 组合与原路径**逐位一致**，收敛为 `engine.copy_from(p_tensor, file_matrix)` 一行。正确性实证：`mnist_train --save A` → `--resume A --lr 0 --save B` → **A/B SHA256 完全相同**（`7ad4c468…` f32 模型、`0e6c26b2…` f16-param 模型两条）。
  - **D6-2（复核为已修）**：15 §4.6-2 说 `CpuEngine::copy_from` 硬编码 F32 槽、f16 dst 在 NDEBUG 下解引用空指针——实际基类 `copy_from` 入口已拦 F16 目标（`from_matrix(src, F32)` + `cast_into`），`copy_from_impl` 的 `cpu_matrix()`（默认 P=F32）只在 F32 目标下可达。**裁定：现状即修复，无代码改动**，把该前提写进注释。
  - **D6-3（删死码）**：`matmul_with_bias_impl` 的默认实现是 `to_matrix → 主机逐列加 bias → from_matrix` 的 PCIe 兜底，CpuEngine/GpuEngine **均已 override**（全仓仅两个 `ComputeEngine` 子类）→ 改为 `= 0` 纯虚：删死码，且第三后端必须给出真现值（编译期强制），不再可能静默退化成主机往返。
  - **D7**：`Model::set_engine` 零调用方 + "绑定后换引擎"与"层已 init 绑定"冲突 → 删除；绑定只经 `Model(ComputeEngine&)`。
  - **D8**：基类 `create_offload_buffer_impl` / `offload_restore_impl` 默认体的 `Tensor::cpu(1,1)`（静态直构）→ `create_tensor(1,1)`（出生经引擎入口，P1 一致）。
  - **`NN_BIND_DEBUG=1` ctest 门禁化**（17 §5 M6 验收项）：首跑 **18/20**（`f16_precision_test` 13 处 FAIL、`f16_writeback_probe` 1 处）。**先补诊断**——`bind_error_` 原先把 file:line+形状只塞进 `Result`，测试的 `CHECK` 不打印 → 门禁失败无从定位；改为 NN_BIND_DEBUG 下同步 `fprintf(stderr, "[NN_BIND_DEBUG] …")`。定位出**两处库内中转漏网**：(a) `to_prec(t, P)` 的入参常是 `*_impl` 产物（`unary_/binary_/move_` 的 `fn(*a)` 返回值尚未走出调用方 `stamp_`），进 `cast`（内部有 `bind_check_`）前未绑定 → `to_prec` 内先 `t = stamp_(std::move(t))`；(b) `expr_dsl.hpp` 归约向量出口 `reduce_vector_tensor` 的 **F16 分支直接 `eng.cast(v, P)`**（F32 分支本就 `eng.adopt`）→ 改 `eng.cast(eng.adopt(std::move(v)), P)`。修完 **ctest 20/20（带与不带开关各一轮全绿）**，门禁成立。
  ② **段 B = `ensure_gpu` → `import`（15 P3 主体，commit 待本文档段）**：新增**引擎契约** `ComputeEngine::import(src[, P])`（公共 NVI：`bind_check_` + `stamp_`，无默认形参的单参重载委托到双参版）+ **新虚 `import_impl`（virtual 方法 48 → 49，`doc_inventory` 口径；`grep -c "\bvirtual\b"` 原始 51 处 = 49 方法 + 析构 1 + 注释里的 "Non-Virtual" 1——本轮唯一虚表变更）**。基类默认实现按 15 §3.2：同设备同精度 = 零拷贝别名、同设备异精度 = `cast`、跨设备 = `to_matrix`+`from_matrix` 宿主中转；`GpuEngine::import_impl` = 原 `ensure_gpu` 的宿主直传快路径（`GpuTensor::from_matrix(src.cpu_matrix(), backend_)`，f32→f32 零额外宿主拷贝），f16 源改走宿主中转分支——**顺带修掉 `ensure_gpu` 硬取 `cpu_matrix()`（F32 槽）对 f16 源取空指针的隐患**。调用点：`compute_gpu_engine.hpp` 全文 `ensure_gpu(` → `import(`（**48 处命中清零，符号 `ensure_gpu` 从代码中消失**），注释同步；`import` **返回新句柄、不改写 `src`**，B 类 6 处"调用方句柄重绑定"保持原形态但显式（`if (A.is_cpu()) A = std::move(*a_gpu);` 语义不变）。P3 第三项"序列化去 `model.engine()`"**裁定保留**：M1 后 save/load 的引擎来源只有"调用方显式传"与"模型回查"两条，改签名会破坏 `save_model(model, path)` 公开 API，收益为零（记入 15 §5 P3 行）。
- **连带数字更新（"49 virtual" 口径统一到 `doc_inventory` 脚本）**：AGENTS §4.3 注、doc 12（头行/§2 标题/§2.2 其余 28 + `import` 行/§2.15）、13 §10 表、17 §4.3 两条 bullet 与 import 行、`introduction/01` §L2 表 + 注、`introduction/03` 注、15 §5 P2/P3 行、16 §2 加"已迁完"横幅与失效的复现命令说明。**口径勘误（本轮发现）**：历史文档的"49 个 virtual / 共 50 处"实际把注释里的 `Non-Virtual` 也算进去了（真实方法数当时是 48）——现统一为 **virtual 方法 = `doc_inventory` 输出（现 49）**，并把三口径写明：`grep -c "\bvirtual\b"` = 51（49 方法 + 析构 1 + 注释词 1）、`doc_inventory` = 49、`doc_align_audit` 的 `virtual_in_engine` = 51。
- **验收**（本机 Debug+Ninja+clang）：① build 122/122 零告警；② ctest 20/20；③ **`NN_BIND_DEBUG=1 ctest` 20/20（门禁）**；④ CPU `--steps 20` hash=`6f8849f14da23110` 不变；⑤ **GPU dev2 hash=`8ef51b2927253c50` 不变**（import 快路径与 ensure_gpu 同一条，GPU 数值零变化）；⑥ scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…）；⑦ `--io-roundtrip` CPU + GPU(dev2) 三用例 PASS；⑧ L2 分层审计 `L2-VIOLATIONS: 0`；⑨ 序列化 round-trip SHA256 逐字节一致（f32/f16 param 两模型）；⑩ `doc_align_audit` A/D/E/F 可行动项 0。**未做 layer_bench A/B**：段 A/B 均为控制流/API 变更（无热路径语义改动），字节锚与 GPU 锚已逐位不变——如需补测，按 §6 口径配对复跑。

## 统一总纲 M6 段 C：删 Layer 每调用 engine 形参——init 绑定 + 四虚接口瘦身（2026-09-30，原位置 compute_layer_base.hpp 类结构、10 个 compute_layer_*.hpp、model_container.hpp、tools/scan_exprs.cpp、src 25 个测试/探针文件）
- 类型：演进记录 / API 变更
- 内容：15 P2 的主体（17 §5 M6 段 C），commit `402b213`。
  ① **绑定设计**：`Layer` 新增 `ComputeEngine* engine_`；`init` 从虚函数改成 **NVI**——公共 `init(engine)` 先绑定 `engine_` 再转 `init_impl(engine)`，10 个层头里 **17 处 `init(...) override` 全部改名 `init_impl`**（编译器穷尽驱动；`PositionEncoder` 家族的 `init` 是**另一套虚表**，刻意不改——它不是 Layer）。四个虚接口去形参：`forward/backward/zero_grad/forward_recompute`。
  ② **方法体零改写技巧**：每个改签名的方法在函数首行插 `ComputeEngine& engine = engine_ref();`（**54 处**）——方法体里原有的 `dsl::compute(engine, …)`、`engine.slice_rows(…)` 全部不动；DSL 入口按 15 §3.3 保留形参。**自动化过程与两处失误**：脚本按"签名行到首个 `{`"切区域做替换，(a) 对 `= 0;` 纯虚声明会一路扫到下一个函数的 `{`，在错误位置插了别名（`PositionEncoder::backward` 三处，手工回滚并恢复其 engine 形参与调用点）；(b) 只匹配 `.` 前缀调用，漏掉 `forward(engine, saved_input)` 这类裸调用与 `*engine_` 形参的 `Model` 调用点（编译器报 "too many arguments" 后逐个修）。
  ③ **调用点**：include 内 99 处 + src/tools 约 106 处（`engine/eng/cpu/gpu/ad/raw/*engine` 等形参名按白名单替换）；**Loss 的 forward 被误伤 7 处**（`ce.forward(engine, logits, target)` 是 3 参），按调用方变量名回滚。
  ④ **新不变量（本轮最有价值的发现）**：**复合层 `init_impl` 必须 init 全部子层**——包括 ReLU/GeLU/SwiGLU/Softmax 这类"无参数"子层，因为它们以前靠 forward 形参拿引擎。补齐 5 处：`FeedForward::gelu_/swiglu_`、`AttentionBase::softmax_`、`CrossAttention::softmax_`、`ZiPTBlock::softmax_`、`TransformerEncoder::pos_encoding_`。定位靠的是把 `engine_ref()` 的 NN_ASSERT 换成 **`std::source_location` 默认参 + `fprintf` 打调用点 file:line**（裸断言只报断言行，无法反查是哪个层），该诊断保留为长期工具。
  ⑤ **库外配套**：`tools/scan_exprs` 的 dry-run 是"构造层→直接 forward"（以前不需要 init）→ 9 处补 `init(engine)`；`fused_gpu_test`/`softmax_gradcheck`/`maxpool_gradcheck` 共 6 处同理。
- **保留形参（＝15 §3.3 "Layer 之后"的两档，本轮裁定不动，见 17 §8）**：`Loss::forward/forward_sparse`（**Loss 没有天然绑定时机**：默认构造、`zipt_consistency_test` 里同一作用域并存 `ce_cpu/ce_gpu` 两个引擎——要删须先裁定构造签名 `CrossEntropyLoss(engine)` 或 `Loss::bind`）；DSL 入口（15 明确保留）；`PositionEncoder`/`RotaryEmbedding`（`apply/apply_step/backward`）、`ActivationOffloader`、`nn_dbg_scan` 与层内私有 helper（由方法体的局部 `engine` 转传，签名不动）。
- **验收**（本机 Debug+Ninja+clang）：① build 122/122 零告警（`-Werror`）；② ctest 20/20；③ **`NN_BIND_DEBUG=1 ctest` 20/20（门禁）**；④ CPU `--steps 20` hash=`6f8849f14da23110` **逐位不变**；⑤ **GPU dev2 hash=`8ef51b2927253c50` 逐位不变**（`engine_ref()` 只是绑定读取，不进数值路径）；⑥ `--init-hash` 六模型与 M2 锚全同（`a22e807ee05ec3ac`/`4020958a14160bbd`/`04a72d865624042a`/`8efa936ac5c8c9b2`/`f90bf8c8783c2f89`/`b037632f75b7159c`）；⑦ scan 双 hash 不变（`expr_specs.bin`=bdc3a442…a58360、`fused_registry.hpp`=7a10412c…——**dry-run 收集的是表达式结构，与引擎绑定无关**）；⑧ `--io-roundtrip` CPU+GPU PASS；⑨ L2 分层审计 0。
- **性能 A/B（配对 + 逆序复测，M2 同款方法）**：pre = `git worktree add` @ `4ce5884` 独立构建 layer_bench，feedforward/swiglu/linear 各 4 轮 PRE/POST 交错：feedforward fwd/train **+4.1%/+4.5%**、swiglu -0.7%/+0.9%、linear fwd **+8.5%**/train +3.8%（linear fwd 为 0.8ms 亚毫秒项）。**逆序复测**（POST 先跑，3 轮）：feedforward **-0.7%/+0.4%**、linear +5.6%/-6.9% → 正序的 +4% 系**运行窗口/机器状态**（同窗 PRE 绝对值本身较上一轮高约 18%）与 -O0 二进制重排所致，**方向混合、无系统性回退**。

## CE 的 (1,1) loss 归约输出恒 F32——修 `stable=f16` 溢出（2026-10-13，原位置 include/neuralnet.cpp/compute_loss.hpp 的 `softmax_cols_`/`forward`(步骤4+5)/`fused_forward_sparse_`）
- 类型：bug 根因 / 已修复勘误（[`docs/todo/24`](../todo/24-f16-training-defects.md) §2.1/§3.1/§5 的 F2/F3；[`docs/todo/14`](../todo/14-f16-stable-gpu-loss-frozen.md)）
- 内容：`CrossEntropyLoss` 的两条链都把最终 **(1,1) loss 和**按 `p_.stable` 落盘。该标量 = 跨 `batch×seq` 的 log_prob 求和（≈ `-log(vocab)·token 数`），**大 token 数下必然超 f16 上限 65504**：部分硬件溢出为 ±inf，部分饱和为 f16 max。**这就解释了 14 号台账的 GPU `stable=f16`「loss 恒 4.0137」**——`4.0137 ≈ 65504/num_valid`，饱和值与权重/数据无关，故「跨 epoch/数据集/进程逐位相同」；原报告归因的「loss 回读链（`cast`/异步槽位读陈旧）」**被否定**（`ce_fusion_test` 的「异步标量回读 vs 同步下载」用例、迭代 24 轮 > 槽位 8，持续通过）。**修法**：两条链的 (1,1) 归约 `Precision::F32`（归约内部本就恒 f32 累加，输出改 F32 即得正确值；调用方随后本就 `cast` 到 F32 回读，等于顺手去掉一次 cast）。**触发范围 = 大 token 数**（本机 GPU 实测：`seq128×b8`=1024 token 健康；`seq256×b32`=8192 token 时 `stable=f16` 打印 `inf`、`stable=f32` 正常 9.0→8.7；修复后 `stable=f16` 9.0056→8.6922 正常下降）。
- **不变量**：f16 存储的**聚合标量**（跨 N 个元素求和、量级 ∝ N）不得按 f16 落盘；f16 只适合量级有界的逐元素/归一化中间量（`col_sum`/`log_softmax`/`lse` 保持 `p_.stable` 不变）。`optimizer=f16` 的爆炸（Adam 二阶矩下溢）与四字段全 f16 的数值性冻结**不属本修**——仍由 24 §2.3 的启动期告警覆盖（F1），根治留给 F5。
- **验收**：本机 CPU（build-ab，Release+tests）ctest **41/41**；本机 GPU（build/gpu，Release+tests+probes+850M/Intel）ctest **43/43**（含全部 `*_gpu` 变体与 `ce_fusion_test`）。新增回归：`f16_precision_test` 的 profile 组合矩阵（补 `all=f16` 行、提为默认门禁、CPU+GPU 都跑）+ `test_ce_loss_stable_f16_no_overflow`（`C=100,N=20000`，CPU/GPU 实测 `f32 5.1876 = stable=f16 5.1876`）。

---

# 代数层 / 基础设施 / 模型容器与序列化

## 线程池删除通用单任务 submit（原位置 include/neuralnet.cpp/core_threadpool.hpp:77）
- 类型：删除清单
- 内容：通用单任务提交（submit）已删除：全库无调用方（审查 P1-3），且每次调用 make_shared<packaged_task> 堆分配，违背本池"零分配 latch"设计。需要 future 语义时应在调用方分块后用 parallel_* 系列原语。

## 线程池 wait_for_latch 等待策略演进（原位置 include/neuralnet.cpp/core_threadpool.hpp:118）
- 类型：性能 A/B
- 内容：优化依据为性能审查报告。旧实现 spin 64 次 + yield，64 次 spin 中反复原子读取，CPU 占用率显示 100% 但实际有效计算比例低（调用者空转）。新实现为三阶段：1) 短自旋 16 次 pause；2) 自旋失败后 work-steal 从队列取任务执行；3) 队列为空阻塞 cv 等待，由 finish_chunk 唤醒。

## 线程池归约分段依赖 worker 数导致字节不一致（原位置 include/neuralnet.cpp/core_threadpool.hpp:541）
- 类型：bug 根因
- 内容：旧实现 n_chunks = chunk_count(total) 依赖 workers_.size()，同一输入在 1-worker 与 N-worker 下走不同折叠结构 → 浮点非结合律导致字节不一致（跨机也不一致）。修复后边界只由 total 决定，1-worker/N-worker、任何机器走完全相同分段。

## 归约 init 重复累加（原位置 include/neuralnet.cpp/core_threadpool.hpp:555）
- 类型：bug 根因
- 内容：旧实现每块都加 init、合并时再加一次，init≠0 时数学错误。现行约定：块 0 以 init 为种子、块 c>0 以块内首元素为种子，init 恰好计入一次。

## 线程池其它阈值/引用调整（原位置 include/neuralnet.cpp/core_threadpool.hpp:85、:106）
- 类型：删除清单
- 内容：MIN_CHUNK 阈值曾从 4096 降至 1024（使 MNIST 小隐藏层 64×batch 也能触发多核并行）；lost-wakeup 压测引用的 build/perfprobe/probe_pool2.cpp 已不存在（压测数据本身保留为持锁通知的实证）。

## GEMM 微内核 vs 旧标量内核 A/B（原位置 include/neuralnet.cpp/algebra_matrix.hpp:278-288）
- 类型：性能 A/B
- 内容：旧内核每个 (i,j) 一个标量累加器、每次 FMA 2 次 load（受 load port 限制），编译日志显示内层循环未被向量化（-Wpass-failed=transform-warning）。微内核（4×8 AVX2 列块）实测 GFLOPS（3072x768x512 / 768x3072x512 / 1024^3 / 512^3）：140→401 / 155→407 / 163→354 / 82→281；与旧内核逐位一致（对拍 max_abs_diff = 0）。

## row_reduce 并行门控曾恒定串行（原位置 include/neuralnet.cpp/algebra_matrix.hpp:744）
- 类型：bug 根因
- 内容：旧实现用 nn::for_each(row_indices)，把"行数"当元素数与 PARALLEL_THRESHOLD 比较 → 行数永远达不到 512K → 恒定串行（原 docs/development/11 §R3，该诊断文档已删除，结论并入 12-compute-engine-inventory/03 §5.3）。改为按元素数（R*C）门控 + parallel_for_samples 按行分片；col_reduce 行数门槛由 1024 降至 256。

## MatrixT F32 零回归验证（原位置 include/neuralnet.cpp/algebra_matrix.hpp:44）
- 类型：性能 A/B
- 内容：模板化改造时 F32 实例与旧非模板 Matrix 逐字节一致（测试项 T1 零回归）；f32 便捷别名保持 Matrix 以保证既有代码零改动。

## 旧代数 AST 移除（原位置 include/neuralnet.cpp/algebra_span.hpp:8、nn.hpp:8）
- 类型：删除清单
- 内容：旧代数 AST（自由运算符 + compute::apply 入口）已随逐元素算子移除；algebra_expr.hpp / algebra_compute.hpp 删除，Expression/BoolExpression 概念迁入 expr_dsl.hpp。Span 不再有"运算符构建 AST"的旧路径；ConstSpan 原注释引用的 expr.hpp 自由函数模板（operator+/-/* 等基于 Expression 概念统一处理，使 ConstSpan 与 Span/Val 自然组合）已不存在，现 DSL 运算符只接受 DslExpr 可折叠节点。

## 序列化审查待办 S3/M1/M2（原位置 include/neuralnet.cpp/model_serialization.hpp:54）
- 类型：bug 根因 / 删除清单
- 内容：S3（已修复）：read_spec_header / read_tokenizer 曾无长度上限，现均经 kMaxSerializedStringBytes（64 MiB）校验后才预分配，损坏/恶意文件返回 Error 而非 bad_alloc → terminate。M1（未修）：.bin 全文件无校验和（.nnpkg 有 sha256），内容损坏会被静默载入错误权重且无感知，曾建议升 MODEL_VERSION 加整文件校验和与尾部完整性标记。M2（审查记录）：extra_state 的注释称"旧文件读到 EOF 保持默认（running_mean=0 等）"但实现直接返回错误，注释与实现不符，需统一为按版本回退默认值。

## v1/v2/v3 偏移量格式移除支持（原位置 include/neuralnet.cpp/model_serialization.hpp:46、:430）
- 类型：删除清单
- 内容：v1/v2/v3 为旧的偏移量定长格式，已移除支持（无有意义的旧模型）；现仅接受自描述格式 v4+，旧格式在 read_and_validate_header 拒绝。Tokenizer JSON 读写曾标注"V3 新增"（版本流水，随 v1-v3 拒绝一并失去意义）。

## 失效文档链接清理（原位置 core_file.hpp:13/:42、model_serialization.hpp:24/:82、algebra_matrix.hpp:41/:43/:746、core_config.hpp:47）
- 类型：删除清单
- 内容：docs/17-pointer-audit.md、docs/23-mixed-precision.md、docs/development/11-cpu-performance-diagnosis.md、DEVELOPMENT_STANDARDS.md、src/bench_thresholds.cpp 均已删除或改名；对应链接改指 docs/todo/10-development-standards.md、docs/development/05-mixed-precision.md，或去掉文件名只留实测结论（blocked 优于 naive、并行阈值 524288 标定表等实测数据保留）。

## 错误处理宏族落地：NN_UNWRAP / NN_EXIT / NN_TRY_MSG / NN_FAIL（2026-10-05，原位置 include/neuralnet.cpp/core_errors.hpp 的 NN_TRY 段、AGENTS.md §5 铁律 #1、CMakeLists.txt 测试块、docs/todo/12 / 18、docs/usage/05 的 ctest 计数行）
- 类型：演进记录
- **背景（被替代的手写形态与实测量）**：全仓 `return std::unexpected(...)` 共 **1205** 处 —— 其中**纯传播** `return std::unexpected(r.error())` **628** 处（`NN_TRY`/`NN_TRY_CHECK` 此前只覆盖 **392** 处：L2 已收敛，其余散在 backend/engine/src/tools）、**错误源头** `return std::unexpected(Error{...})` **518** 处、`src/` 里 `if (!r) { std::cerr << ...; return 1/false; }` 形态 **374** 处（**全仓唯一没有任何宏覆盖的语义**）。原宏族只解决"传播"，解包与失败退出两类全靠手写，也正是"作为库易用性"评估里排第一的摩擦源。
- **新宏族（三种语义 + 一个糖）**：`NN_TRY` / `NN_TRY_CHECK`（传播，展开与手写逐字等价）；`NN_TRY_MSG(decl, "语境", expr)`（传播并把语境前缀进 `Error.message`）；`NN_UNWRAP(expr[, "语境"])`（解包 + 终止，别名 `NN_UNWRAP_1`，接受 `Result<T>` / `Result<void>` / `bool` 三形态）；`NN_EXIT(expr, code[, "语境"])`（解包 + `std::exit(code)`，CLI 参数错误保住退出码）；`NN_FAIL("msg")`（错误源头糖）。
- **实现约束（改动前必读）**：终止只有 abort/exit，**任何路径不出现 `throw`** —— 故未做"可抛异常"变体（铁律 #1 + `-fno-exceptions`）；变参分发走实参计数 `NN_DETAIL_PICK2/3` + 末尾哨兵，**不用 `__VA_OPT__`**（MSVC 传统预处理器不可靠）；解值用 `std::move(r).value()`（`expected<void>` 的 `operator*` 返回 void，`std::move(*r)` 编译不过）；非 void 重载带 `[[nodiscard]]`、void 重载**不带**（否则 `NN_UNWRAP(f());` 纯副作用调用误告警）；`check_message` / `contextualize` 单独成函数 = 可单测。**已知边界**：被检查表达式含**顶层逗号**（多参数模板列表、花括号初始化）会让实参计数出错 → 选错分支 → **编译期报错（响亮，不会静默错值）**，加一层括号即可（已写进宏注释与铁律 #1）。
- **本轮范围**：只交付宏族 + 测试 + 文档；存量 628 处传播与 374 处 `src/` 退出样板分 A/B 两批随后迁移（验收口径 = 两批残留 0），518 处错误源 `NN_FAIL` 化裁定为**不强制**（纯美观，风险 > 收益）。数据管道 `nn::data::`、`nn::generate`、易用层聚合头同期**裁定推迟**（数据格式待定型；`nn.hpp` 保持唯一聚合头，改为提供按需薄转发头）。
- **验收（本机 Release + Ninja + clang）**：build 零告警（`-Werror`）；ctest **23 → 24**（新增 `error_macro_test`：解值三形态 / `check_message` 打印格式 / `NN_TRY_MSG` 语境前缀 / 终止与退出码经 `std::system` 子进程 `--expect-abort`、`--expect-exit <n>` 断言）；带与不带 `NN_BIND_DEBUG=1` 各一轮 **24/24**；`gpu_stability_probe --init-hash` **五锚与改动前 A/B 逐位一致**（实测法见下条"过程坑"）：mnist_mlp `3e982d018c00d94a` / cnn `d8c8457f0053c4f9` / mnist_transformer `70c4c085052268b2` / gpt `f8c2d892f5834239` / rapt `7002b986b84f679f`；scan 结构仍 **91**（样例收集器 93）、`fused_registry.hpp` 内容未变；`L2-VIOLATIONS: 0`；`gui_cli_audit` PASS；`doc_align_audit` EXIT 0。
- ⚠ **锚点口径**：AGENTS.md §12 记录的 `0148e453908158dc` 等是**作者机器**的值（锚标注"机器/stdlib 相关"），本机 clang+MSVC STL 实测即为上列另一组 —— 判"锚有没有动"只能用**同机 A/B**，不能对文档数值。
- **过程坑（记录供后效）**：做 A/B 时用 `Copy-Item` 回退 `core_errors.hpp` / `CMakeLists.txt`，**源文件时间戳被一并复制**，恢复后 mtime 反而早于刚生成的 PCH → ninja 不重建 PCH，而 Clang 的 PCH 内容校验报 `file '…core_errors.hpp' has been modified since the precompiled header … was built`（`note: size changed from expected 4355 to 11119`）整片 fatal；同一原因令 `CMakeLists.txt` mtime 回退 → **未重新 configure** → `ctest -N` 少列一项（23 而非 24，容易误判成"测试没注册"）。**处置 = 把恢复文件的 `LastWriteTime` 推到当前再构建**。

## 错误处理宏族两批迁移完成（2026-10-05，A/B 批：src/ 退出样板 + 全仓传播/错误源）
- 类型：演进记录（上一条"错误处理宏族落地"的迁移收尾）
- **B 批（`src/` 退出样板 → `NN_EXIT`，320 处）**：两种形态 —— ①`if (!r) { std::cerr << …; return 1; }` 单行 304 处（`tools/apply_nn_exit.ps1`）、②`{ auto r = f(); if (!r) { …; return N; } }` 花括号 16 处（`tools/apply_nn_exit_braced.ps1`）。`NN_EXIT` 语义 = 打印后 `std::exit(code)`，`main()` 内与 `return code` 等价，打印内容是原手写形态的超集（ctx + error.message + 表达式原文 + file:line）。**不迁**：测试 helper 里的 `return false`（28 处，聚合语义）与 `return 77`（ctest SKIP 约定）。
- **A 批（全仓传播/错误源 → `NN_TRY`/`NN_TRY_CHECK`/`NN_FAIL`，1209 → 36 处，96%）**：分七种形态脚本化迁移（均 `tools/apply_*.ps1`、带 `-DryRun`）：
  - 守卫传播 `if (!x) return std::unexpected(x.error());`（单行/跨行）→ `NN_TRY_CHECK(x);`（宏 = 该守卫逐字等价）
  - init-statement 守卫 `if (auto r = f(); !r) return …;`（单行/跨行/花括号作用域）→ `NN_TRY(r, f());`
  - 同行声明+守卫 `auto k = import(K); if (!k) return …;` → `NN_TRY(k, import(K));`
  - 错误源 `return std::unexpected(Error{"…"});`（单行/多行/拼接）→ `NN_FAIL("…");`
  - 守卫+错误源 `if (!w_.valid()) return std::unexpected(Error{…});` → `if (!w_.valid()) NN_FAIL("…");`（`NN_FAIL` 含 return，接在 `if()` 后等价）
  - **残留 36 处需人工审**：`core_assert.hpp` 宏定义体与 `error_macro_test.cpp` 测试夹具（豁免）、`*init_error_` 解引用、`return std::unexpected(\n…)` 跨行构造（`Error` 不在行首）、嵌套在 `if`/`for` 体内多语句块、尾随 `// comment` 的守卫等。
- **踩坑（迁移中的作用域重定义）**：`NN_TRY(r, …)` 把声明**提升到所在块**，而原 `if (auto r = …; !r)` 把 `r` 限定在 if 的 init 里 —— 同块内连续多个 `NN_TRY(r, …)` 会 **redefinition**（实测 `compute_optimizer`/`model_serialization`/`gpu_stability_probe` 等约 12 处）。**处置 = 逐处改名 `r2`/`r3`**（编译器是 ground truth，`--clean-first` 全量重编确认零错误；脚本化改名因 C++ 真实作用域难精确判定而放弃）。另：首轮把 `return` 错误地加在 `NN_FAIL(...)`/`NN_TRY_CHECK(...)` 前（两宏**自带 return**）→ `return return …` 编译失败，已回退重做。
- **验收（本机 Release + Ninja + clang）**：`cmake --build --clean-first` 全量零告警（`-Werror`）；ctest **24/24**、带与不带 `NN_BIND_DEBUG=1` 各一轮 **24/24**；`gpu_stability_probe --init-hash` **五锚与迁移前逐位一致**（`3e982d018c00d94a`/`d8c8457f0053c4f9`/`70c4c085052268b2`/`f8c2d892f5834239`/`7002b986b84f679f`）；scan 结构仍 **91**（+130 变体 = 221 注册表条目）、`fused_registry.hpp` 内容未变；`L2-VIOLATIONS: 0`；`gui_cli_audit` PASS；`doc_align_audit` EXIT 0。
- **新门禁（`bench/doc_inventory.ps1` 第 [5] 节）**：**手写 `return std::unexpected` 残留计数** —— 剥注释/字符串、豁免 `core_assert.hpp` 宏定义体、`core_errors.hpp` 的 `#define` 宏体、`error_macro_test.cpp` 夹具；**验收 = 残留 ≤ 基线 45**（当前实测 36），超基线即 `exit 1`。用意：新增错误检查一律走宏族，不应再出现裸 `unexpected`，防止迁移成果回潮。
- **本轮未动**：518 处错误源 `NN_FAIL` 化中已迁的部分并入 A 批；剩余错误源裁定**不强制**（纯美观）。数据管道 `nn::data::`、`nn::generate` 仍**裁定推迟**。

## 错误处理宏族 2 阶段：36 处边缘形态人工清零，门禁基线 45 → 0（2026-10-05）
- 类型：演进记录（上一条"两批迁移完成"的收尾；原裁定"残留需人工审"的最终处置，剩余错误源同批并入）
- **范围**：上轮门禁放行的 **36 处**手写 `return std::unexpected`（= 全仓剩余站点，宏定义体/注释/测试夹具豁免不计）逐一人工改宏，全部为**传播**与**错误源**两类，无行为变化：
  - `compute_vk_backend.hpp` **22 处** = 主形态「失败分支先清理资源再传播」（`vkDestroyBuffer` / `vkFreeDescriptorSets` / `batch_mode_ = false` 后返回）→ 清理语句原样保留、return 行改 `NN_TRY_CHECK(x);`（宏在失败块内重测恒假 → 立即同错返回，展开与原手写逐字等价）；
  - `compute_layer_gpt.hpp` **5 处**：`init_impl` 内联单行与滑窗 `engine.zero` 循环 → `{ NN_TRY(r, …); }`（声明型宏必须留花括号，否则 if/for 控制两条语句编译失败）；`fill_cache_` 内 end_batch 收尾后传播 → `NN_TRY_CHECK(r)`；
  - `compute_layer_rapt.hpp` **4 处**：`dsl::compute` 结果传播（含尾随 `//` 注释两处）→ `NN_TRY_CHECK`；
  - `compute_engine.hpp` **2 处**：跨行 `return std::unexpected(\n bind_error_(…))` 错误源 → `NN_FAIL(bind_error_(…))`（`Error{Error}` 拷贝构造，消息与 `NN_BIND_DEBUG` 打印逐字不变）；
  - `compute_optimizer.hpp` **1 处**：`*init_error_` 解引用 → `if (init_error_) NN_FAIL(*init_error_);`
  - `cli_mnist_io.hpp` **1 处**（end_batch 收尾后传播）、`gpu_stability_probe.cpp` **1 处**（init-statement 守卫 → `NN_TRY(r3, …)`，同块序号连带 r3/r4/r5 顺移 = 纯变量改名）。
- **裁定保留（表达式形态，非 return 站点、门禁本就不计）**：`gpu_test.cpp:291` 的三元链 `Result<T>{std::unexpected(r.error())}` —— 错误沿表达式传给本循环自己的 `if (!out_r)` 处理（打印 + `++failures` + `continue`，不是函数返回），宏是语句、不能置于表达式内；改写需复制失败处理逻辑，风险 > 收益。
- **门禁基线 45 → 0**（`bench/doc_inventory.ps1` 第 [5] 节）：`HANDWRITTEN-RESIDUE: 0`；此后新增任何裸 `return std::unexpected` 直接 `exit 1`（宏定义体/注释/夹具三类豁免不变）。
- **验收（本机 Release + Ninja + clang）**：build 75 目标零告警（`-Werror`）；ctest **24/24**、带 `NN_BIND_DEBUG=1` 一轮 **24/24**；`gpu_stability_probe --init-hash` 五锚与迁移前**逐位一致**（`3e982d018c00d94a`/`d8c8457f0053c4f9`/`70c4c085052268b2`/`f8c2d892f5834239`/`7002b986b84f679f`）；`L2-VIOLATIONS: 0`、`HANDWRITTEN-RESIDUE: 0 (baseline 0)`；`gui_cli_audit` PASS；`doc_align_audit` EXIT 0。

---

# 领域模型工厂与 CLI 公共头

## BPE encode 每 chunk 的 4 次堆分配（原位置 include/neuralnet.cpp/domain_tokenizer_base.hpp bpe_merge_impl_）
- 类型：性能 A/B
- 内容：旧实现每处理一个 chunk 分配 4 次（`ll_prev` / `ll_next` / `std::vector<bool> alive` / `std::priority_queue` 内部 vector），输入 `ids` 与返回结果各再分配一次。现改为 thread_local `BpeMergeScratch` 复用容量、结果**追加**到调用方 `out`、堆用 `std::push_heap/pop_heap` 手工维护（与 `std::priority_queue` 同算法，弹出顺序逐位相同）。实测 34MB 语料 `tokenizer_infer --encode-file` 单线程 18.4s → 9.2s（本机 32 核，2026-10-01；旧二进制基线取自本会话）。输出逐字节不变：`text_train` 的 `.tokcache` 重新生成后 SHA256 = `9E1F6DD4CC6B00F057BB04654E62E6CD4665410ED76E2BAEF578639341CC326E`（与旧实现产出的缓存完全相同）。

## BPE encode 保序并行（原位置 include/neuralnet.cpp/domain_tokenizer_{base,bpe,charbpe}.hpp 的 encode）
- 类型：演进记录 / 性能 A/B
- 内容：`encode` 从「单线程顺序处理整段文本」改为 `encode_segments_`：按空白串起点（`find_safe_splits`）分段 → 各段经全局线程池并发编码 → **按段下标升序拼接**。正确性依据（已作为当前契约写进代码注释）：安全切分点对 BPE/CharBPE 的预分词**必然是 chunk 边界**、标记不含空白故不会被切断、预分词与合并都是 chunk 局部操作，因此任意并行度与顺序执行逐字节一致（铁律 #8）。文本 < 256 KiB（`PARALLEL_ENCODE_MIN_BYTES`）或找不到切分点时回退顺序路径；新增 `Tokenizer::set_encode_threads`（0=自动/1=顺序/>1=指定）。实测 8 MiB 多样文本 `encode` 0.93s → 0.134s（32 线程，7.0x；本机 32 核，2026-10-01）。此前「BPE 不能并行」的结论只适用于训练期合并循环（见 `docs/todo/08` §4.2）。

## evaluate_mnist 原始签名否决（cli/cli_mnist_io.hpp:16-18 头注释与 evaluate_mnist 函数注释，整改前行号）
- 类型：否决方案
- 内容：曾按「用户原始签名 evaluate_mnist(nn::Layer&, ...)」设计，实际不可行——nn::Model 不是 nn::Layer 的派生类，且 Model::forward 自带 engine 绑定（签名不同于 Layer::forward(engine, ...)）；最终改用 nn::Model&，原注释还附带「与原签名行为完全等价」的等价性论证。技术理由已改写为当前注释（签名为何取 nn::Model&），原始签名叙事与等价性论证移至本条。

---

# src 应用入口（text/mnist/tokenizer/bench）

## base 模式 BOS/EOS 插入的删除（原位置 src/text_train.cpp:4-5 头注释、src/text_train.cpp:98、src/text_train.cpp:808，整改前行号）
- 类型：删除清单
- 内容：早期 text_train 把每行编码为 `[BOS]+tokens+[EOS]` 再拼接，行边界靠 EOS 编码进 token 流；base 模式改为纯拼接、行间无分隔符后，文档边界改由 parallel_tokenize 产出的 doc_ids（块对角掩码）表达。随迁的等价性叙事：text_infer 曾以「训练时每行以 BOS 开头 / 与训练时每行格式一致」论证 prompt 加 BOS 的合理性（src/text_infer.cpp:171、:193、:362 整改前），该前提在纯拼接方案下不成立（`scripts/download_everyday_conversations.py` 生成的对话行也不含 BOS）。现注释只陈述两侧各自的现行行为。

## host 取 loss 的同步旧路径（原位置 src/text_train.cpp:1339，整改前行号）
- 类型：性能 A/B
- 内容：旧路径每步为取 loss 做 `end_batch + wait_in_flight`，drain 整条流水线 → GPU 在 host 录制期间空转，表现为占用率锯齿。现改为「异步回读槽位 + 非阻塞收割 harvest_loss + 槽位将满/epoch 收尾时阻塞兜底 drain_loss」；注释保留现行机制与「为何 host 不打断录制帧」的理由，删除旧路径描述。

## NaN 跳步的移除（原位置 src/text_train.cpp:1509，整改前行号）
- 类型：删除清单
- 内容：训练循环曾在 host 侧读回 loss 判定 NaN 并跳过该步；现 host 不判定 loss（loss 只在 device 侧异步回读），数值稳定性由 `--max-norm` 梯度裁剪 + 观察 loss 曲线负责。

## layer_bench transB/transA 变体与 plain 共用操作数（原位置 src/layer_bench.cpp:263-264，整改前行号）
- 类型：bug 根因
- 内容：旧版 setup 让 transB/transA 变体与 plain 共用 `(k,n)` 操作数：`k==n` 时方阵形状掩盖错配，`k≠n` 时 backend 返回 K mismatch 而 run 内 `*expected` 不查错 → UB，测出 0.000ms 垃圾时长 / 超物理 GFLOPS。已改为 per-variant 独立 setup + run 查 Result 失败即打印并中止；注释保留现行契约（每变体自建操作数、run 必须查错）与形状错配的坑。

## text_train::parallel_tokenize 换用全局线程池（原位置 src/text_train.cpp parallel_tokenize）
- 类型：演进记录
- 内容：旧实现用 `std::async(std::launch::async)` 每块起一个 future、按块顺序 `get()` 后拼接（保序但每次调用有 future 堆分配与共享状态）。现改为按块下标写 `parts[c]` + `nn::parallel_for_samples` + 顺序拼接；块边界仍是 `(n + n_threads - 1) / n_threads`，token 流与 doc_ids **逐字节不变**（`.tokcache` 重新生成 SHA256 一致）。同批 `tokenizer_infer` 新增 `--threads`（单次 encode 并行度）。

---

# GPU 手写原语 shader（shaders/*.comp）

## matmul_tiled vec4 优化策略的旧版对比（原位置 shaders/matmul_tiled.comp:10-14，整改前行号）
- 类型：性能 A/B / 差异对比
- 内容：原注释标题为「优化策略（对比旧标量 4×4 版：内层每 k 8 次标量 LDS + 16 FMA，2:1）」，并在正文写「消除旧版标量读的 bank 冲突」——即当前 vec4 配方的 8:1 计算:加载比是相对「旧标量 4×4 版每 k 8 次标量 LDS + 16 FMA（2:1）」表述的。整改后注释只陈述当前算法（vec4 LDS + 16 FMA、8:1、vec4 读沿 lane 连续消除 bank 冲突）。

## matmul_tiled v2/v3 版本标签与 barrier 演进叙事（原位置 shaders/matmul_tiled.comp:16-23、256）
- 类型：演进叙事
- 内容：原注释以「v2（双缓冲流水，BK=16）」「v3（vec4 全局加载/写出…）」标版本，v2 段写「每 tile 的 barrier 由 2 次减为 1 次」「BK=32 双缓冲需 32KB，会砍半 blocks/SM 抵消流水收益（实测净 0）」；流水循环内写「每 tile 仅 1 次 barrier（旧版 2 次）」。整改后改为现在式小节标题（双缓冲流水 / vec4 全局加载写出）与「每 tile 仅 1 次 barrier」。

## matmul_tiled BK=32 否决方案 A/B（原位置 shaders/matmul_tiled.comp:85-89）
- 类型：否决方案 / 性能 A/B
- 内容：原文「BK=16 + 双缓冲（2026-09-24 OP/融合统一 A/B 定案）：…曾试与融合侧同参 BK=32 (32KB)：OP 级四点**全回退**（512³ +9%、1024³ +26%、4096³ +29%、batched b8 +29%，3 样本 best）——32KB 每 SM 只驻 2 个 WG，浅深网格都被占用率卡死；OP 级恒取 16（下轮再统一须四点复测）」。四点实测数字与占用率结论已保留在现注释，删除的只是日期与「曾试与融合侧同参」的对比流水。

## batched_matmul 旧标量版与移植叙事（原位置 shaders/batched_matmul.comp:15-25）
- 类型：演进叙事 / 性能 A/B
- 内容：原文「与 matmul_tiled.comp 同配方（v3 移植，2026-09-24）：本 shader 长期停留在最初的标量 4×4 版（每 k 8 次标量 LDS + 每 tile 2 次 barrier + 标量全局访问带逐元素 /,% 与边界分支），而 matmul_tiled 已演进到 v3 —— 同形状下 batched 比 matmul 慢 1.85×（1024³ 实测 1.51 vs 0.81ms）。移植的三重配方（逐项与旧版的差异）：① vec4 共享（8:1，旧版标量 2:1）② BK=16 双缓冲（barrier 2→1 次）③ vec4 全局快路径（旧版每元素 1 次标量 ldg/stg + /,% + 边界判断）」。三条配方本身保留为现注释，删去旧版描述、1.85× 对比数据与移植流水。

## batched_matmul BK 注释的日期决策流水（原位置 shaders/batched_matmul.comp:93-94）
- 类型：演进叙事
- 内容：原文「分块维度常量（与 matmul_tiled 保持同步；2026-09-24 OP/融合统一 A/B：OP 级 BK=32 四点全回退 → 定案 BK=16 双缓冲 16KB）」。整改后只留当前事实（与 matmul_tiled 同步、BK=16 双缓冲 16KB 保占用率），数据指向 matmul_tiled 的 BK 常量注释。

## reduce 差异分解整段（原位置 shaders/reduce.comp:39-55）
- 类型：差异对比 / 性能 A/B
- 内容：原注释标题「2026-09-24：与 DSL 生成归约（expr_glsl_gen generate_glsl_reduce）同步」，下接「差异分解（此前两条路算法不一致，OP 级是慢的那条）」：① 旧版列归约「WG=单列、256 线程按行号跨步」——lane 间地址步进 cols×4B 完全不合并（每 warp 32 笔独立事务），4096² 实测 159 GB/s（同形状 row_reduce 482、add 412）；② 旧版 max 恒等元 -inf（0xff800000），DSL/CPU 侧 2026-09 已改；③ 旧版行归约 shared 串行树（8 次 barrier）。整改后三条只保留当前设计的现在式陈述（按列合并访存、lowest() 恒等元及 NaN 理由、subgroup 蝶式 3 次 barrier），旧版描述与 159 GB/s 对比数据移此。

## gather 旧版逐元素除模与「逐位同构」（原位置 shaders/gather.comp:21-24、96）
- 类型：演进叙事
- 内容：原头注释「vec4 + 按组求商（2026-09-24，与 DSL 侧…契约同步）：旧版每线程 1 元素、每元素一次 i=idx/D、d=idx%D 两笔整数除模 + 4 次标量表读。现：…」；回退路径原写「回退：标量（每元素求 i/d；与旧版逐位同构）」。整改后改为组级求商的现在式描述，回退路径改为「与 vec4 快路径逐位一致」（等价性指的是当前两条路径之间）。

## elementwise_v2 的旧标量 OP 级描述（原位置 shaders/elementwise_v2.comp:51-54）
- 类型：演进叙事
- 内容：原文「vec4 kernel（2026-09-24，与 DSL 融合逐元素 vec4 路径同步）：DSL 侧（expr_glsl_gen glsl_vec4_eligible + vec4 kernel）早已按『每线程 4 个相邻元素』发射…；OP 级此前是标量每线程 1 元素。本 shader 现镜像同一契约」。整改后只保留「与 DSL 侧同契约 + 本 shader 镜像同一契约」的现在式。

## group_reduce 恒等元注释的日期与旧值叙事（原位置 shaders/group_reduce.comp:59-61）
- 类型：演进叙事
- 内容：原文「max 恒等元 = numeric_limits::lowest()（0xFF7FFFFF，2026-09-24 与 reduce.comp / DSL / CPU 全库统一；旧 -inf 在全 -inf 输入下与减法组合会出 −inf−−inf=NaN）」。数值、全库统一契约与 NaN 理由保留，删除日期与「旧 -inf」叙事（改述为「-inf 不能作恒等元」）。

## transpose 消融链、轮次与 32×32 对比（原位置 shaders/transpose.comp:6-31、49）
- 类型：性能 A/B / 演进叙事
- 内容：原头注释「64×64 分块 + 8×8 WG 砖块化 dispatch（2026-09-24 二轮定稿）：一轮（32×32 tile / 线性 dispatch）5244² 实测 0.958ms；二轮经三步消融定案：消融链（5244²，wall，warmup 30 best-of-10）scale 纯拷贝 0.736 / 过 shared+barrier 恒等拷贝 0.754（shared/barrier 仅 ~2%）/ 真转置 + 线性 dispatch 0.942（+25% 来自写侧足迹散布，136 个并发 WG）/ 真转置 + 8×8 砖块化 0.798（本版本，−15%）；结论『瓶颈不是段宽（64×64 vs 32×32 仅 2%）』；其余定案（A/B 均已验证）：64×64 tile（vs 32×32）段宽 128B→256B、每线程 8→16 元素摊薄 barrier、共享落位 A/B 持平；f16 变体注释另含「（spike 验证 shared float16_t 合法）」。整改后保留：足迹聚集度结论 + 砖块化 0.942→0.798ms（−15%）+ 16×8 砖持平 8×8 + 全部设计要点；移出：轮次流水、完整四点消融链、32×32 对比数据、spike 备注。原文 dispatch 网格「(16→8, 8, n_bricks)」含 16→8 演进标记，已改为 (8, 8, n_bricks)（与文件末尾 dispatch 契约一致）。

## scan_prefix_outer_gen「不再驻 shared」措辞（原位置 shaders/scan_prefix_outer_gen.comp:3）
- 类型：措辞（历史残留）
- 内容：原句「A/B 运行态不再驻 shared（shared 是 O(dk²) 且受 48KB 预算限制），改驻全局显存 scratch」暗示曾驻 shared；整改后为「A/B 运行态驻全局显存 scratch（B*H*2*dk*dk）——shared 是 O(dk²) 且受 48KB 预算限制，无法驻留」。

## 精度变体注释的「逐字未改 / 零回归」参照措辞（原位置 shaders/matmul_tiled.comp:38、batched_matmul.comp:48、reduce.comp:69-70）
- 类型：措辞（历史残留）
- 内容：原文「f32 分支文本逐字未改（零回归）」「f32 分支展开与原文逐字同形 → SPIR-V 字节零回归」以改动前状态为参照；整改后改为当前机制陈述「f32 分支宏展开与裸写 f32 代码逐字同形 → SPIR-V 字节零差异」。

---

# 构建期工具 / 基准脚本 / 对拍脚本

> 本文件摘录 `tools/`、`bench/`、`compare_with_torch/` 注释整改中移出的历史状态记录（原位置为整改前行号）。

## 文档一致性收口 + `doc_align_audit` [A] 扩展到 markdown 链接目标（P0-2/E8，2026-10-15）

- 类型：演进记录 / 审计缺口收口（P0-2 的 ③⑤ 两项 + 台账漂移回正）
- **原状**：① `docs/release-notes/` 缺 `v1.5.1.md`（E3，自 v1.5.1 起就缺）；
  ② `bench/doc_align_audit.ps1` 的 [A] 只扫**反引号里的文件名**，markdown 链接目标不查 —— 实测
  `todo/25-release-requirements.md` 里 `../CONTRIBUTING.md` 指向 `docs/CONTRIBUTING.md`（不存在）
  这类死链因此长期漏检（E8）；③ ctest 计数在 12 号/AGENTS/18 号三处口径不一（47 vs 48），
  §6 触发清单仍把**已落地**的 IR-C 列为待触发项，26 号进度总览停在"S4 未做 / ≤ f32 未达成"。
- **改动**：① 补写 `docs/release-notes/v1.5.1.md`（依据 `git log/show v1.5.0..v1.5.1` 三个提交与
  history 的两条归档，未引入新测量；含迁移指南表）；② `bench/doc_align_audit.ps1` 的 [A] 增加
  "markdown 链接目标可达性"一节（相对路径按文档所在目录解析，外部 URL / 锚点 / `codex:` 豁免，
  **行内代码段先剥离** —— 否则文档里作为示例写的链接形态会误报，实测就撞到过一次）；
  ③ 台账回正：ctest 计数统一为 **48（GPU 树）/ 46（无 Vulkan 树）**（18 号 §2/§8.1/§8.2、AGENTS §11
  索引行），§6 的 IR-C 行改为"已落地、不再是触发项"，§4.4 P3 由 8 项改 7 项，26 号进度总览改写为
  "已全部收口"（S4 已完成 / S2 与 S5 已评估-不做 / 验收锚已达成），C9 标注已兑现，
  新增 **C10**（GitHub issue #15：RAPT GPU 训练比同配置 GPT 慢 ≈16×，此前只在 issue 里跟踪、
  未进任何台账）。
- **证据**：`pwsh -File bench/doc_align_audit.ps1` → [A] 可行动 0（反引号文件名 0 + 链接目标 0，
  检查 294 个链接目标）、[D]/[E]/[F] 各 0；`pwsh -File bench/doc_inventory.ps1` → `L2-VIOLATIONS: 0`、
  `HANDWRITTEN-RESIDUE: 0`、virtual 52 / 宿主桥 41（与文档一致）；`ctest -N` 实测 48（GPU 树）/
  46（CPU-only 树）。
- **边界**：链接目标检查只覆盖**仓库内**相对路径（外部 URL 按设计不联网校验）；P0-2 的遗留项
  `install()`/`find_package` 仍按用户裁定"本轮不做"（属 P0-4 剩余部分，不属本文档项）。

## P2-12 残差：优化器 dry-run 未透传 precision profile（2026-10-14；原位置 tools/scan_exprs.cpp:184）
- 类型：bug 根因（构建期 scan 与运行期不同源）
- 内容：`--f16` 下 Adam 的每参数复合 kernel `{K3,K4}` 运行期命中不到 AOT shader → 落"逐成员降级派发"（70 参数 = 140 dispatch/步），既不融合也不批量（P2-12 残差）。根因是 **scan 的优化器 dry-run 调 `create_optimizer` 时未传当前 pass 的 precision profile**（默认 `PrecisionProfile{}` = 全 f32），而真实路径 `text_train` 是透传 `cfg.precision` 的。于是 f16 pass 里 `upd_prec = p_.param` 仍取 f32 ⇒ 融合出的复合结构与运行期不同 ⇒ `composite_kernel_available_` 查不到 ⇒ 降级。修法 = 透传 `scan_prof`（f32 pass 与本改动等价，仅 f16 pass 生效）。**教训**：scan 的"结构/签名来源"必须与运行期**逐项同源**；dry-run 里任何"省略一个参数"的简化都会在运行期表现为降级（可观测但静默丢性能）。

## push-constant 固定头长度按 5 个 uint 打包的错误（原位置 tools/scan_exprs.cpp:572）
- 类型：bug 根因
- 内容：历史 bug 曾把"归约但无 matmul"形态的 push-constant 固定头按 5 个 uint 打包（应为 4）→ 常量池整体后移一个 uint → GPU 上"带常量池的归约"静默错值而 CPU 正常（表现为 `col_reduce_sum(select(x == col_broadcast(max), 1, 0))` 恒返回 kk-1 而非真实并列数）。回归用例 `fused_gpu_test::run_reduce_consts`。教训：这类"只有 GPU 错"的问题先打印生成的 GLSL/IR 再猜成因。

## dry-run 用 (void) 吞错导致真实路径从未登记（原位置 tools/scan_exprs.cpp:254；同型 :442）
- 类型：bug 根因
- 内容：旧 dry-run 用 `(void)` 吞掉 forward 错误（ZiPTBlock 的 dry-run 调单参 `forward(x)`，实际返回"用双参"错误）→ 该 dry-run 从未执行真实路径，存量结构恰好处处有 key 才长期绿，块内新表达式迁移后立刻闭合世界硬报错（2026-09-27 教训）。已修为：双参真实签名 + 检查返回值 + 失败带栈 abort；CSA 的 `run_csa` 同型（旧行为是吞错后 backward 在 batched_matmul 读空张量上 NN_ASSERT，栈无上下文难定位）。**dry-run 长期绿 ≠ 覆盖存在。**

## MHA 裸 S 表达式此前只存在于 masked 分支（原位置 tools/scan_exprs.cpp:491）
- 类型：覆盖演进
- 内容：backward recompute 的无掩码 `dsl::compute(matmul)`（裸 S 表达式）此前只在 masked 分支里出现过；MHA（Plain 双向无掩码）dry-run 加入后才成为该结构的闭合世界注册来源。

## fold 掩码 Doc/AlibiDoc 曾漏登记（原位置 tools/scan_exprs.cpp:617）
- 类型：bug 根因
- 内容：P-C2 attention fold 的 5 个掩码变体曾漏登记 Doc/AlibiDoc → GPU doc 训练闭合世界硬报错；由 fused 对拍补充 Doc/AlibiDoc 用例后暴露。该块是 fold spec 唯一注册来源（层 forward 直调 `engine.eval_expr(make_fold_attn_o)`，不经 `dsl::compute` 的 NN_EXPR_SCAN 钩子）。

## canonicalize 曾静默丢 fold 段（原位置 tools/gen_fused.cpp:249）
- 类型：bug 根因
- 内容：canonicalize 类变换曾静默丢掉 fold 段，产出"空指令表且无 matmul/fold"的结构；逐元素生成器对其 `back()` 触发 UB（实测 0xC00000FD 栈崩溃）。现生成器对此防御性跳过 + 告警；正常管线到不了该分支。

## gen_fused 对 matmul+归约/列归约形态的跳过逻辑（原位置 tools/gen_fused.cpp:248、259）
- 类型：删除清单
- 内容：`gen_fused` 曾对 matmul+列归约（S5 列方向，如 `col_max(matmul)`）与 matmul+归约（注意力结构）整体跳过生成；随生成器补齐（列方向按元素分解 batch `batch = row/m_per`、归约遍历全部 rows；`generate_glsl_reduce` 支持 Matmul 操作数内联点积）后，跳过分支删除，两形态均正常生成。

## 精度变体支持范围的过期声明（原位置 tools/gen_fused.cpp:309）
- 类型：演进状态
- 内容：变体发射注释曾称"reduce/matmul/fold 的带类型变体尚未实现"；现状为纯逐元素、matmul 段、fold 段（Phase D2）与归约 kernel（`generate_glsl_reduce(name,spec,sig)`）均已支持带类型生成，`gen_fused` 实测 0 skip。注：`include/neuralnet.cpp/expr_glsl_gen.hpp:1391` 存在同一过期表述（"目前只支持纯逐元素形态；reduce/matmul/fold 拿到 sig != 0 时返回空串"，与同文件 :1406 矛盾），不在本次 tools 整改范围内，需另行处理。

## edit_ranges 花括号护栏的事故来源（原位置 tools/edit_ranges.ps1:61）
- 类型：bug 根因
- 内容：结构性不变量（区间内 `{}` 自平衡）源于 2026-09-26 gpu_test 事故：边界串匹配全过但删掉了 `if (…)` 的起始行、留下其闭合 `}`，文本断言全过而结构已损坏。整改后注释只保留"断言全过 ≠ 结构对"的理由，去掉事故日期叙事。

## text_train.py 样本采样口径演进（原位置 compare_with_torch/text_train.py:139、268、293）
- 类型：演进叙事
- 内容：该脚本早期每 step 独立随机采样，后改为"epoch 开头 shuffle 索引队列 + 每 step 顺序切片"，保证每个样本每 epoch 恰被访问一次；C++ `src/text_train.cpp` 现为同一口径（`std::shuffle` 索引队列 + 顺序切片）。整改后注释只记录当前口径（"构造约定"/"每个样本每 epoch 恰被访问一次"），去掉"不再/改造点（相对 C++ 版）"叙事。

## "旧行为"表述与旧实现等价性论证（原位置 tools/scan_exprs.cpp:60、84、586；compare_with_torch/model.py:286；compare_with_torch/tokenizer.py:3）
- 类型：等价性论证（措辞）
- 内容：曾用"旧行为"描述 f32 pass（sig=0）与无 mask loss 路径，并论证 f32 pass 输出"逐字节旧行为 / bin 不变"；tokenizer.py 头注释曾写"不再手写 BPE 算法"。整改后改为现在式事实：f32 pass 收集 sig=0 基础结构（结构表）；无 mask 路径 `reduction="mean"` 逐位置等权；本模块直接使用 HuggingFace tokenizers 库、不手写 BPE 算法。

## Conv/Pool 引擎化与优化器/Linear 的"迁移"叙事（原位置 tools/scan_exprs.cpp:555、308、336、338）
- 类型：演进叙事
- 内容：MaxPool2D dry-run 注释曾写"这是 Conv/Pool 引擎化后新增的结构"；优化器与 Linear 注释曾写"已迁移为 DSL 融合表达式 / dsl::compute(...)”。整改后改为当前事实表述："该表达式未被扫描覆盖时 GPU 闭合世界硬报错"、"实现为 dsl::compute(...)"，表达式形态与 dry-run 理由原样保留。

## 悬空文档引用（原位置 tools/scan_exprs.cpp:352、389、359）
- 类型：失效链接
- 内容：注释曾引用不存在的 `docs/14`（S1-S3/S5 章节）与已无对应文档的条目号 `P2-13`。前者改指现存的 `docs/todo/02-operator-fusion.md`（二期 S1-S7 主线）；后者改为直接描述现行机制"同 RowMod/RotateHalf 的视图参数处理：不进 key，运行时填充"。`include/neuralnet.cpp/expr_spec.hpp:429` 亦有同款 `P2-13` 悬空引用，不在本次范围内，需另行处理。

---

# 表达式与融合测试

## ADL 作用域说明中的旧代数运算符（原位置 src/expr_dsl_test.cpp:27、src/expr_opt_test.cpp:28、src/expr_reduce_test.cpp:28、src/expr_matmul_test.cpp:30，整改前）
- 类型：演进记录
- 内容：四文件同一句「测试写在全局作用域（非 namespace nn），避免与旧代数运算符的 ADL 歧义。」——「旧代数运算符」指 algebra_expr.hpp / algebra_compute.hpp 的 Expression / BoolExpression（已于 2026-09 移除，概念迁入 expr_dsl.hpp），已不存在。整改后改为当前理由「避免 ADL 把匹配拉进 nn 命名空间」。

## matmul+列归约组合曾被 gen_fused 跳过（原位置 src/expr_matmul_test.cpp:269-270、src/fused_gpu_test.cpp:454-455，整改前）
- 类型：覆盖缺口 / 演进记录
- 内容：expr_matmul 原文「该结构曾被 gen_fused 跳过（"matmul+列归约组合暂不支持"），生成器补齐按元素 batch 分解后，此处锁死其语义：列归约遍历全部 rows（= batch*M），与独立标量参考…一致。」fused_gpu 原文「该组合曾被 gen_fused 跳过（"matmul+列归约组合暂不支持"），生成器补齐按元素 batch 分解后，此对拍锁死 GPU 与 CPU 语义一致（列归约遍历全部 rows，含所有 batch）。」整改后删除「曾被跳过/补齐」的实现进度流水，只留当前覆盖陈述（覆盖 matmul+列归约组合、列归约遍历全部 rows、GPU 与 CPU 语义一致）。

## 注意力 fold 构造的归位流水（原位置 src/expr_fold_test.cpp:33，整改前）
- 类型：演进记录
- 内容：原文「#include <neuralnet.cpp/compute_layer_attention.hpp> // 注意力 fold 构造（已从 expr_fold.hpp 归位到 Layer）」——fold 构造与 5 掩码变体于 2026-09-25 从 expr_fold.hpp 迁至 compute_layer_attention.hpp。整改后改为「（定义于 Layer：表达式文本只写在 Layer）」，只表达当前 AOT 分工。

## fold 对拍 NaN 守卫的红验证流水（原位置 src/expr_fold_test.cpp:46-47，整改前）
- 类型：bug 根因 / 演进记录
- 内容：原文「IEEE fmax(err, NaN) = err 会静默吞掉 NaN diff——2026-09 fold doc 掩码 GPU -nan 回归曾因此在对拍中漏抓（红验证实证：revert 后仍 PASS）。」整改后保留机制与当前理由（若无此守卫，被测端输出 NaN 时对拍仍可能 PASS，故直接把非有限 diff 记为 inf），删除回归日期与红验证/实证流水。

## doc 分段边界：2026-09 GPU -nan 触发形态与旧边界漏抓（原位置 src/expr_fold_test.cpp:316-321，整改前）
- 类型：bug 根因 / 覆盖缺口
- 内容：原文「…被文档掩码**全部**屏蔽——这正是 2026-09 GPU 训练 -nan 的触发形态（max init=-inf 时 m_old=blk_m=-inf → dm=−inf−−inf=NaN）。旧固定 seq/2=66<128 永远让首块留有有效项 → ctest 全绿漏抓。seq ≤ 128 时保持原二分段。」整改后保留当前覆盖理由（边界须越过 128 才能让首块被全屏蔽、触发 dm=−inf−−inf=NaN 分支；边界落在 128 之内则首块永留有效项、该分支测不到），删除 2026-09 日期与「旧固定 seq/2=66 / ctest 全绿漏抓」流水。

## GPU 侧 doc 边界的钉根因与旧边界漏抓（原位置 src/fused_gpu_test.cpp:732-734，整改前）
- 类型：bug 根因 / 覆盖缺口
- 内容：原文「让 i≥129 的行首 fold 块被 doc 掩码**全屏蔽**——钉住 2026-09 GPU -nan 根因（max init=-inf → dm=−inf−−inf=NaN）；旧固定 seq/2=66<128 首块永留有效项 → 对拍漏抓该 bug。」整改后改为当前分支触发说明（全屏蔽触发 NaN 分支；边界落在 128 之内测不到），删除「钉住…根因」与旧边界漏抓流水。

## NaN 守卫的双层漏抓出处（原位置 src/fused_gpu_test.cpp:789-790，整改前）
- 类型：演进记录
- 内容：原文「…会静默吞掉 → err 保持正常值照样 PASS（2026-09 fold doc -nan 回归双层漏抓之一，红验证实证）→ 记 inf 必超容差」。整改后保留机制与当前结论（记 inf 必超容差、NaN 输出必须硬失败），删除回归出处与红验证流水。

## gpu_test 头注中的逐元素职责（原位置 src/gpu_test.cpp:6-8，整改前）
- 类型：删除清单
- 内容：原文「验证 CpuEngine 与 GpuEngine 在 matmul / 转置 matmul / 逐元素 / 归约 / roundtrip 等原语上的一致性，并对比 GPU vs CPU matmul 性能。」——逐元素算子已随 2026-09 收敛删除，元素级表达式对拍职责转到 expr_gpu_test。整改后头注去掉「逐元素」并加指路「（元素级表达式对拍见 expr_gpu_test）」。

## 逐元素算子移除的删除清单碎片（原位置 src/gpu_test.cpp:354、466、503，整改前）
- 类型：删除清单
- 内容：原文「4b. CPU/GPU 张量准备（逐元素算子已移除；元素级 DSL 对拍见 expr_gpu_test）」「链式: matmul(A, B) → col_reduce_sum（Exp 级已随逐元素算子移除）」「── Step 2: 直接复用 matmul 输出（Exp 级已随逐元素算子移除）──」。整改后删除括号内的移除流水，只留指路与当前步骤名。另 :555 的输出字符串「（Exp 级已随逐元素算子移除）」是打印字面量，按规则未改动。

## transpose 砖块派发 bug 与旧形状漏检（原位置 src/gpu_test.cpp:271-276，整改前）
- 类型：bug 根因 / 覆盖缺口
- 内容：原文「── 3d. transpose 逐元素对拍（issue #13 P0-① 回归）──」「历史 bug：backend 派发 (16,8,n_bricks) 而 transpose.comp 按 8 宽砖解算（假定 gl_WorkGroupID.x ∈ [0,8)）→ 行>512 且 列>512 时静默只写前 512 行。旧测试形状最大 64×256，永远单边 ≤512，故漏检。」「形状表覆盖：单边 ≤512（旧代码 PASS）/ 双边 >512（旧代码 FAIL 50%）/ 奇数边界…」。整改后改为当前契约（backend 必须按 (8,8,n_bricks) 派发、与 transpose.comp 的砖解算不同源时双边 >512 静默只写前 512 行；单边 ≤512 测不出），删除 issue #13 编号、「历史 bug」叙述、「旧测试形状 64×256 漏检」与「旧代码 PASS/FAIL」对照。

## row_max/denom 与旧手写内核的等价性论证（原位置 src/fused_gpu_test.cpp:397-398，整改前）
- 类型：与旧实现的等价性论证
- 内容：原文「row_max(QK^T)（bmm_reduce Max 等价）与 denom = row_sum(exp(QK^T - rm))（bmm_denom 等价）均为"matmul 段 + 归约指令"单表达式，GPU 经 generate_glsl_reduce 的 matmul 支持单 kernel 完成。」——bmm_reduce / bmm_denom 旧实现已删除。整改后去掉两处「（…等价）」括注，保留结构归类与 GPU 单 kernel 完成的当前陈述。

## push-constant 固定头长度的既往错档 bug（原位置 src/fused_gpu_test.cpp:556-561，整改前）
- 类型：bug 根因
- 内容：原文「── 归约表达式内联常量：CPU vs GPU（push-constant 头长度回归）──…该结构带**常量池**。GPU 侧 push-constant 固定头长度必须按形态算；历史上"归约但无 matmul"曾多算 1 个 uint（5 vs 4）→ 常量池整体后移一个 uint → shader 读错常量 → **GPU 静默错值而 CPU 正常**。本用例锁死该回归。」整改后给出当前契约表（逐元素 2 / 逐元素+matmul 5 / 归约 4 / 归约+matmul 6），把「历史上曾多算」改写为「按错档则常量池整体错位 → shader 读错常量 → GPU 静默错值而 CPU 正常」，删除既往 bug 流水。

## fold K 族注释中的常量升位流水（原位置 src/fused_gpu_test.cpp:616，整改前；本轮补改）
- 类型：演进记录
- 内容：原文「K 族覆盖单列 / 非块整除 / 整除 / 尾块（EXPR_FOLD_BLOCK=128 边界两侧——256/260 补多块+尾块；BLOCK 升 128 后 K≤100 会静默退化单块）。」本轮改写为「…256/260 补多块+尾块；K≤100 全落 128 以内的单块区间，测不到多块/尾块）」，保留 K 族选值理由，去掉常量升位的历史叙述。

## 列式 softmax 手写原语的删除清单（原位置 src/ce_fusion_test.cpp:5-6，整改前）
- 类型：删除清单
- 内容：原文「- col_softmax_denom：denom[c] = Σ_r exp(logits[r][c] - col_max[c]) → 已删除（IR 融合替代）」「- col_softmax_sparse_forward：单 kernel 稠密梯度 + 标签位置 loss_vec → 已删除（IR 融合替代）（不物化全 softmax），含 mask / 越界标签处理」。整改后改为当前覆盖清单：IR denom 由 S7 组合 col_reduce_sum(exp(logits − cb(col_max))) 对照手写参考；loss_vec/grad 由 RowGather + Row 操作数组合（不物化全 softmax）；另新增一条「归约 + 后处理（reduce(...) * k + c）：CPU 正确求值、GPU 必须硬报错」。

## check_matrix 返回值曾被忽略（原位置 src/ce_fusion_test.cpp:38-39，整改前）
- 类型：bug 根因
- 内容：原文「返回是否通过：调用方必须把 false 计入失败数（此前只累加 g_fail 而不影响 run_case 返回值/进程退出码，导致 [FAIL] 行被 "ALL PASS" 掩盖 —— 修）。」整改后保留调用契约与后果（只累加 g_fail 而不影响退出码时，[FAIL] 行会被 "ALL PASS" 掩盖），删除「此前…—— 修」的修复流水。

## 归约+后处理的两段背景（原位置 src/ce_fusion_test.cpp:169-178，整改前）
- 类型：bug 根因 / 演进记录
- 内容：原文「── 回归：带"归约 + 后处理"的归约向量表达式（reduce(...) * k + c）──…背景一（CPU）：eval_expr_reduce 的"输出沿归约轴恒定"前置校验原本按**寄存器号**传播所需指令（needed[reg]），而寄存器分配器按 liveness 复用逐元素寄存器号 —— 后处理指令的 dst 与归约前的 Mul 同号时，归约前的定义被误判为输出链的一部分（Linear 访问），该类表达式被错误拒绝。现改为按**指令下标**反向切片，CPU 正确求值。背景二（GPU）：归约融合 shader 尚未正确实现"归约后仍有逐元素后处理"的形态（实测静默错值），故 GPU 必须**硬报错**而非返回错值；本用例断言这一点，防止将来退化为静默错误。」整改后改为当前实现/契约陈述（前置校验按**指令下标**反向切片；GPU 不支持该形态只会静默错值，故必须硬报错），删除「原本 / 现改为 / 尚未正确实现」的修复与实现进度流水；寄存器号复用导致误拒的机制作为当前设计理由保留。

## inv_num_valid 的旧接线方式（原位置 src/ce_fusion_test.cpp:308，整改前）
- 类型：演进记录
- 内容：原文「注：inv_num_valid 由 RParam 承载（运行时标量，不进 expr_spec_key）→ 与整条逐元素链融合为单 kernel（原先在表达式后补一次 scale_inplace）。」整改后删除括号内旧接线（原先补 scale_inplace），保留 RParam 不进 key、整链单 kernel 的当前陈述。

## 算子 f16 变体对拍的覆盖缺口与先例引用（原位置 src/gpu_f16_test.cpp:69-70，整改前）
- 类型：覆盖缺口
- 内容：原文「…的 f16 存储变体（这些路径此前只有 f32 覆盖，f16 变体一旦类型/索引出错是静默错值——对拍是唯一护栏，同 gather vec4 改写先例）」。整改后保留当前风险理由（f16 变体的类型/索引一旦出错即静默错值，对拍是唯一护栏），删除「此前只有 f32 覆盖」缺口叙述与 gather vec4 先例引用。

---

# 层 / 模型测试

## MaxPool2D 测试覆盖缺口（原位置 src/maxpool_gradcheck.cpp:3）
- 类型：覆盖缺口
- 内容：原文「目的：MaxPool2D 此前**没有任何测试覆盖**（Conv2D 由 conv2d_gradcheck 覆盖）。」整改后只保留当前覆盖陈述（本片段把层实现与独立参考逐元素比对）。

## clear_cache 后 backward 的旧行为（原位置 src/maxpool_gradcheck.cpp:8-9）
- 类型：bug 根因
- 内容：原文「① clear_cache() 后直接 backward 必须报错 —— 旧行为是越界读空 vector（UB；因 vector::clear() 保留容量，表现为"静默沿用陈旧索引"不报错）」。整改后改写为当前契约「clear_cache() 后 backward 必须报错——缓存为空时不得读取」，并把「vector::clear() 保留容量 → 越界读未必崩溃」保留为一般性陷阱警告（不再叙述「旧行为」）。

## Conv2D 测试覆盖缺口（原位置 src/conv2d_gradcheck.cpp:3）
- 类型：覆盖缺口
- 内容：原文「目的：Conv2D 此前**没有任何测试覆盖**。本片段把层实现与一份独立写法的直接卷积…」。整改后删除覆盖缺口叙述，只留「本片段把层实现与独立写的直接卷积参考逐位比对」。

## CNN 无端到端覆盖的背景（原位置 src/cnn_smoke_test.cpp:3-5）
- 类型：覆盖缺口
- 内容：原文「目的：CNN 此前只有单层 Conv2D 参考比对（conv2d_gradcheck），**没有端到端覆盖**——最大池化的反向、以及整链 forward/backward/optimizer 只能靠手工跑 `mnist_train --arch cnn` 验证。本片段补上：」。整改后改写为当前分工陈述：单层参考比对不覆盖规格序列化/层组成/整链训练，本片段验证这些。

## 合并测试里的覆盖缺口标签（原位置 src/cnn_test.cpp:10、src/layer_gradcheck_test.cpp:9）
- 类型：覆盖缺口
- 内容：两行原为「── maxpool_gradcheck（独立参考实现比对；MaxPool2D 此前无任何覆盖）──」与「── conv2d_gradcheck（独立参考实现比对；Conv2D 此前无任何覆盖）──」。整改后删除括号里的「此前无任何覆盖」，只留「独立参考实现比对」。

## doc_mask_test / build_attention_mask 删除清单（原位置 src/doc_attn_test.cpp:4-6）
- 类型：删除清单
- 内容：原文「（原 doc_mask_test 已随物化掩码构建函数 build_attention_mask 一并删除——掩码语义现由 fold body 表达：spec 级覆盖见 expr_fold_test 三掩码，doc 端到端覆盖见下方 e2e）」。整改后删除「原 doc_mask_test 已…删除」的删除清单叙述，保留当前覆盖分工（掩码语义由 fold body 表达，spec 级覆盖见 expr_fold_test，端到端见本 e2e）。

## MHA 掩码默认值的回归背景（原位置 src/attn_test.cpp:24-26）
- 类型：bug 根因
- 内容：原文「回归背景：fold 迁移曾把 fmask 默认树设为 Causal → MultiHeadAttention（本应双向无掩码，旧 apply_mask_ 默认 no-op）被静默因果遮蔽；层级此前只测 CSA，未被抓住。」整改后改写为当前约束「⚠ MHA 默认掩码必须为 Plain（双向）；默认树设为 Causal 会静默因果遮蔽双向注意力」，删除 fold 迁移流水、旧 apply_mask_ 行为与「层级此前只测 CSA」的覆盖缺口叙述。

## Doc/AlibiDoc backward 分支的零执行覆盖叙述（原位置 src/attn_test.cpp:194-196）
- 类型：覆盖缺口
- 内容：原文「Doc / AlibiDoc backward 覆盖（masked_doc_ / masked_alibi_doc_ 分支此前零执行——forward 对、梯度串文档抓不住；默认 learned+doc 与 alibi+doc 各跑一遍数值梯度）」。整改后改写为当前覆盖理由「本段专门跑 masked_doc_/masked_alibi_doc_ 分支的梯度（仅 forward 通过不足以验证——梯度跨文档串扰 forward 抓不住）」。

## GPU 融合 matmul 尾链 batch 行号 bug 的日期流水（原位置 src/attn_w_batch_test.cpp:3-5）
- 类型：bug 根因 / 日期流水
- 内容：原文「背景：GPU 融合 matmul 尾链曾按 **batch 内行号**读取 (rows,1) 全网格输入（注意力的 m/l），导致 batch>1 或多头（BH>1）时 m/l 读错 → GPU 前向错误（2026-08-27 审查发现，P0-1）。」整改后改写为当前契约（尾链必须按全局行号读取，按 batch 内行号读会读错 m/l），删除日期与 P0-1 审查流水。同文件 :107 的「复现原 batch 内行号 bug」改为「暴露 batch 内行号读错」。

## forward/forward_step 一致性与旧模型权重的结论（原位置 src/attn_consistency_test.cpp:4-5）
- 类型：历史状态
- 内容：原文「…若一致则证明 Attention 语义修复正确，生成乱码是旧模型权重的问题（用 buggy 代码训练得到），需重训。」整改后只保留当前语义（两者一致即证明两条推理路径语义等价），删除「语义修复正确 / buggy 代码训练的旧权重 / 需重训」的历史结论。

## GPT 整链 gradcheck 的覆盖缺口与用户现象（原位置 src/gpt_gradcheck.cpp:5-6）
- 类型：覆盖缺口 / bug 根因
- 内容：原文「单层 gradcheck（rmsnorm/swiglu）已验证层内数学，但多层堆叠 + learned 位置编码 + token_emb scatter 的整链路径从未验证。用户现象：浅层 GPT 正常、深层 GPT loss 卡平台 → 疑似链路级 bug。」整改后改写为当前覆盖陈述（整链路径由本测试专门验证，链路级错误只有整链梯度对拍才抓得住），删除「从未验证」缺口叙述与用户现象流水。

## attn_gradcheck 的定位流水（原位置 src/attn_gradcheck.cpp:3-5）
- 类型：调试流水 / 覆盖缺口
- 内容：原文「目的：GPT 整链 gradcheck 显示 wo OK 但 wq/wk/wv FAIL，定位 attention 内部是否真的有反向 bug（batched_matmul / softmax backward / 掩码）。batch=1 时绕过 rearrange_3d，若仍 FAIL 则是 attention 内部问题。」整改后保留分层定位的设计意图，删除「显示 … FAIL」的既往调试结果。同文件 :142-144「该分支此前零执行覆盖（forward 对、梯度串文档抓不住）」改写为「本用例专门验证该分支的梯度（仅 forward 通过不足以验证）」。

## softmax_gradcheck 的既往失败流水（原位置 src/softmax_gradcheck.cpp:3-5）
- 类型：调试流水
- 内容：原文「目的：attention gradcheck 显示 grad_Q/grad_K FAIL 但 grad_V OK，共同差异是 grad_S = softmax.backward(grad_A)。本测试单独验证 Softmax 的 forward/backward 是否与中心差分一致。」整改后改为当前意图陈述（单独隔离验证 Softmax；grad_Q/grad_K 由 grad_S 派生、grad_V 不经 Softmax，故单独跑便于分层定位），删除既往 FAIL 结果叙述。

## 跨样本隔离用例的「历史 bug」表述（原位置 src/doc_attn_e2e_test.cpp:119）
- 类型：bug 根因
- 内容：原文「其 logits 只可能因跨样本串扰而变（position-major 类历史 bug 在 batch=1 下不可见）」。整改后保留理由、删去「历史 bug」：「batch>1 是必须的：batch=1 时 position-major 与 batch-major 布局重合，串扰不可见」。

---

# 精度 / RAPT / ZiPT / 分词器测试与探针

> 本轮「精度 / RAPT / ZiPT / 分词器 / 杂项测试与探针」注释整改中摘出的实质性历史。
> 对应改动文件：src/test_common.hpp、src/precision_profile_test.cpp、
> src/precision_type_test.cpp、src/f16_precision_test.cpp、src/f16_compute_test.cpp、
> src/tensor_precision_test.cpp、src/f16_cpu_probe.cpp、src/mem_probe.cpp、
> src/rapt_offload_test.cpp、src/tokenizer_consistency_test.cpp。

## test_common.hpp 聚合结构的由来（原位置 src/test_common.hpp:3-8）
- 类型：结构演进说明
- 内容：历史上每个子测试都自带一份 CHECK / make_tensor / check_close / approx /
  dot / close_to，聚合器（`#define main test_xxx + #include 子测试.cpp` 模式）
  被迫用一长串 `#define` 逐符号重命名来规避 ODR 冲突（漏一个就是重定义或静默绑错）。
  2026-09 收敛为 test_common.hpp 单份后，聚合器只需重命名 main、失败计数器与
  各测试独有的函数名。另：「刻意不收编」决定（2026-09 审查）——max_abs_diff 因
  9 份副本语义有分叉（conv2d 版多形状守卫返回 1e9）暂不收编，另案处理。

## ref_ulp 曾误写成 2^(e-10)（原位置 src/precision_type_test.cpp:55-57）
- 类型：bug 根因
- 内容：ref_ulp 曾误写成 2^(e-10)（正确为 2^(e-25)，大 2^15 倍）→ RHE 中点
  断言的容差比被测值本身还大 → float_to_half_bits 次正规 UB 窗口
  （exp ∈ [-45,-33]）产出的垃圾 half（如 0x4000=2.0）在该容差下永远测不出来。
  修复 = ref_ulp 改回 2^(e-25)，容差恢复到真实 ulp 量级。

## 次正规移位 UB 窗口与两个垃圾代表值（原位置 src/precision_type_test.cpp:195-198, 214）
- 类型：bug 根因
- 内容：precision.hpp 的 `float_to_half_bits` 次正规分支守卫曾写 `exp <= -46`，
  与其自身公式矛盾 → exp ∈ [-45,-33]（|v| ≈ 2.8e-14 ~ 1.2e-10）走 shift ≥ 32
  的移位 UB → 指数字段回绕产出垃圾 half（0x4000=2.0、0xCCCD… 等）→ CPU f16
  训练梯度被写成 512/8192/11776/18432/NaN，若干步后更新爆炸（梯度正是这个量级、
  前向激活 ~0.1 从不落入，故单算子测试全过）。修复 = 守卫改 `exp <= -26`
  （≤-26 按 D<0.5 恒 flush 0）。定点断言的两个代表值 1.13687e-12 与 8.44011e-11
  正是修复窗口上沿，修复前分别产出 0x4000=2.0 与 0x3333。

## `--f16` 曾等价于 profile_master_weights()（原位置 src/precision_profile_test.cpp:4-7）
- 类型：语义变更记录
- 内容：`--f16` 曾等价于 profile_master_weights()（F32 主权重 + f16 计算），
  用户以为开了全 f16、实际参数仍是 F32。2026-09 修订为 profile_f16()
  = {param:F16, compute:F16, stable:F32, optimizer:F32}（本测试把该语义钉死）。
  stable/optimizer 不取 F16 的实测证据：optimizer=F16 → Adam 的 v≈g²~1e-10
  f16 下溢、更新爆炸（loss 7.9→3.6e4）；stable=F16 → CE 链 ~200 步 NaN；
  四字段全 F16 → loss 恒定（更新被 f16 舍入吃光）。

## CPU 侧 f16 训练发散的双根因（原位置 src/f16_precision_test.cpp:569-573）
- 类型：bug 根因
- 内容：该处原为「历史"已知问题"（CPU 侧 f16 训练发散）已修复并转为硬失败」的
  根因注释：根因1 = DSL 预绑定把 f16 操作数喂给 f32 GEMM（cpu_matrix<F32>()
  拿到空指针 UB）；根因2 = float_to_half_bits 次正规分支 exp<=-46 的移位 UB
  （|v| ∈ [2.8e-14, 1.2e-10] 的梯度被写成垃圾 half）。完整记录见
  docs/development/05-mixed-precision.md §12.12。修复后 CPU f16 三组 profile
  6 步轨迹与 f32 逐位一致，本处改为硬失败断言。

## 逐元素原语已整体移除（原位置 src/f16_precision_test.cpp:93-94）
- 类型：删除清单
- 内容：原注释「（逐元素原语已整体移除；逐元素路径的 f32 零回归见
  test_f32_zero_regression_dsl）」——引擎级 elementwise_unary/binary/binary_scalar
  等算子于 2026-09 收敛时删除（逐元素一律走表达式 DSL），故 f32 零回归用例
  的代表原语选归约/搬运两类，逐元素路径的对拍由 DSL 版用例承担。

## rapt_offload：RLA backward 缓存曾不在 activation_cache() 内（原位置 src/rapt_offload_test.cpp:11-13）
- 类型：覆盖缺口
- 内容：RLA 的 backward 缓存（RMSNorm 后 Q/K、逐头 1/rms）此前不在
  activation_cache() 导出集合内，activation offload 会漏导出/恢复这些缓存。
  2026-09-19 RAPT 补齐 set_activation_offload 接线与 activation_cache() 覆盖后，
  本测试对拍 offload 开/关两侧的导出恢复集合完整性（缺一项即 backward 报缓存
  缺失或数值不一致）。

## 参考实现曾称「完整旧版 pre_tokenize」（原位置 src/tokenizer_consistency_test.cpp:35-36）
- 类型：措辞修正（与已删除实现的等价性论证）
- 内容：该参考函数原注释自述「完整旧版 pre_tokenize……逐字复刻优化前
  domain_tokenizer_bpe.hpp 的实现」。它当前是手写状态机（pre_match_len）的
  对拍基准，应表述为独立朴素实现（正则切分），不称「旧版」。

## docs/23 陈旧文档引用（原位置 src/precision_type_test.cpp:1、src/f16_compute_test.cpp:2、src/tensor_precision_test.cpp:2）
- 类型：失效链接
- 内容：三处「docs/23 §13 / §12.2」指向已不存在的旧编号设计文档；同内容现行
  位置为 docs/development/05-mixed-precision.md（§13 测试计划 T1–T11、
  §12.2 Phase 1 验收标准）。precision_type_test 的 printf 输出字符串里仍保留
  字面量「docs/23 T2」（字符串字面量不改动）。

## mem_probe 默认参数的 5.7GB 峰值锚点（原位置 src/mem_probe.cpp:22-23, 281）
- 类型：过期基准值
- 内容：原注释称默认参数「与已知"峰值 5.7GB"场景对齐」。该数字是 09-25 的
  f32 旧峰值（5743 MiB）；随后的显存优化已把它降到 ~3069 MiB，且
  docs/benchmarks/2026-09-25-vulkan-vs-cuda.md §"峰值已强相关于精度" 明确
  「09-25 ~5.7GB 与精度无关」的观察不再成立（f16 现为 6315/5738）。故摘除
  5.7GB 锚点，只保留「默认参数 = 该 bench 配置（vocab 8208 / d64 / h4 / L4 /
  ff256 / seq256 / batch64）」的测量口径陈述。

## f16_cpu_probe 曾自述为「临时诊断」（原位置 src/f16_cpu_probe.cpp:1-2）
- 类型：状态措辞
- 内容：探针最初为定位 CPU f16 训练发散的临时诊断工具；现已是注册的组件级
  探针（阶段标记 + 非有限值扫描），表头改为当前用途陈述。

---

# docs/development 12 · 13 · 08

> 本文件汇集 docs/todo/12-compute-engine-inventory.md、13-refactor-backlog.md、
> 08-pitfalls-and-lessons.md 整改中移出的历史叙事（演进流水 / 已修复勘误 / 否决方案 /
> 过程性验证数字）。正文只保留当前状态与仍然适用的教训。

## 2026-09-27 计算类原语全量迁 DSL 横幅（原位置 docs/todo/12-compute-engine-inventory.md 顶部横幅）
- 类型：演进记录
- 内容：迁移范围——inplace 族 `add_inplace`/`scale_inplace`/`accumulate` → `dsl::compute_into`（10 处：attention/mlp/conv/rapt/transformer/zipt）；单 matmul `matmul`/`matmul_with_bias` → `dsl::matmul`（Linear fwd/bwd、Conv gcol、Transformer 池化反传、Muon NS 6 处）；归约族 `row_reduce_sum`/`col_reduce_sum`/`col_reduce_max` → `dsl::compute_reduce`（mlp/conv/transformer/loss×4/optimizer×3/rapt）；批量 matmul `batched_matmul` 21 处 → `dsl::matmul(..., batch)`（attention backward+forward_step、CrossAttention、ZiPTBlock、rapt forward_step），alpha 经 `dsl::rparam(scale)` 尾链（值不进 key）；缺口 A：`grouped_reduce_sum/max` → 新 `ExprViewKind::GroupedReduceSum/Max` 视图（param=R 进 key），MaxPool2D 迁移。结果：Layer 直调 **34 → 23**，10 个计算类原语全部退出 Layer 直调（`bench/doc_inventory.ps1` 复测）；ctest 19/19。随迁的三条方法论教训（dry-run 禁止吞错 / alpha 用 rparam / 新 ExprViewKind 落点清单）保留在 12 §2.1 正文（现在式）。

## 2026-09-19 IR-C 删除横幅 + 2026-09-25 接口数字复核流水（原位置 docs/todo/12-compute-engine-inventory.md 顶部横幅）
- 类型：演进记录
- 内容：① 2026-09-19 横幅：IR-C 部分执行完毕——`expr_graph.hpp`、`ComputeEngine::begin_expr/end_expr`、`dsl::start_expr/end_expr`、`FusedChainLayer`、`Tensor::virtual_tag_` 均已从代码移除；§4.2/§4.3/§5/§8 曾保留为"决策前证据快照"。② 2026-09-25 复核：**58 个 virtual（Phase 2 加入 `cast_into`/`copy_into`/`supports_*`/`eval_expr_*` 等）、Layer/Loss/Optimizer 直调 35 个**；随后 2026-09-26 收敛更新为 **49 virtual / Layer 直调 32**，2026-09-27 迁移后为 **49 virtual / Layer 直调 23**（当前值）。

## 2026-09-26 §5 遗留物清单收敛记录横幅（原位置 docs/todo/12-compute-engine-inventory.md 顶部横幅）
- 类型：演进记录
- 内容：① 旧代数 AST 整体移除：`algebra_expr.hpp`、`algebra_compute.hpp`（`nn::compute::apply`）删除，`Expression`/`BoolExpression` 概念迁入 `expr_dsl.hpp`，`Matrix::detail::{apply,binary_apply,binary_apply_inplace}` 死函数删除；② 无根/测试-only 算子全部删除：`axpy_inplace`、`broadcast_row_inplace`、`broadcast_col_inplace`（连带 `shaders/broadcast.comp` 全链）、`elementwise_unary/binary/binary_scalar`、`elementwise_select_scalar_cond`、引擎 `row_reduce_max`、offload B 组 `offload_store/offload_load`、`UnaryOp/BinaryOp/CompareOp` 枚举；③ 数字：引擎 virtual **58 → 49**；CPU 求值机制 **4 → 2**（DSL 模板路径 + IR 解释器）；④ 随行清理：`src/test_common.hpp`、死 `activation_cache()` override、重复 `offload_test` 聚合目标（ctest 20 → 19）、`gpt_offload_test` 补 batch 录制包裹。

## §1 CUDA 移除范围清单（原位置 docs/todo/12-compute-engine-inventory.md §1）
- 类型：删除清单
- 内容：原先约 100 KB 死代码 + 10 处条件分支——删除 `cuda/`（`CMakeLists.txt`/`cuda_kernels.h`/`cuda_kernels.cu`）、`compute_cuda_engine.hpp`、`backend/compute_cuda_backend.hpp`；清除 `compute_tensor.hpp`（6 处 `#ifdef NN_HAS_CUDA`）、`nn.hpp`、`cli/cli_engine_factory.hpp`（`EngineConfig::use_cuda`）、`cli/cli_train_common.hpp`（`--cuda`）、`CMakeLists.txt` 停用段、15 个 `src/*.cpp` 的 `--cuda`/`use_cuda`/`cuda_enabled`、`cli_controllers.py` 的 `--cuda` 传参。历史快照见 git 分支 `legacy/cuda`。

## §2.2 失效的 2026-09-25 调用方快照行（原位置 docs/todo/12-compute-engine-inventory.md §2.2）
- 类型：已修复勘误（算子已删，行随之失效）
- 内容：`offload_store`/`offload_load`（仅 offload_primitive_test；Layer 用 `offload_save/offload_restore`）、`elementwise_unary`/`elementwise_binary_scalar`/`elementwise_binary`（含 `compute_layer_rapt.hpp:866` Div 生产调用）、`broadcast_row_inplace`（无根）、`broadcast_col_inplace`（仅 f16 测试/layer_bench）、`axpy_inplace`（无根）、`row_reduce_max` 引擎算子（仅适配层转发；Layer 走 `dsl::row_reduce_max` DSL 叶子）、`elementwise_select_scalar_cond`（全仓无调用点）——以上算子已于 2026-09-26 收敛轮从接口删除，当前 49 个 virtual 中不再存在。

## §3「四套机制」旧表与死/重复叙事（原位置 docs/todo/12-compute-engine-inventory.md §3）
- 类型：演进记录
- 内容：机制 1「旧代数 AST `compute::apply(span, expr)`」（`algebra_compute.hpp`/`algebra_expr.hpp`/`Matrix::detail::*`）已于 2026-09-26 整体删除；`eval_cpu` 串行模板分支此前已被并行化+向量化的 `eval_into_span` 取代。原判定："机制 2 与机制 3 是 DSL 在 CPU 上的两条路（一条内联、一条解释）——收敛后的全部 CPU 求值路径"；机制 3（IR 解释器）逐元素 switch 分派 + 归约前缀重放，实测比等价原语慢 1.5–3.8 倍（2026-09-18 诊断实测，原诊断报告已删除，结论数字保留）。

## §4.2–§4.3 IR-C 盘点原文（原位置 docs/todo/12-compute-engine-inventory.md §4.2、§4.2.1–§4.2.3、§4.3）
- 类型：否决方案 / 演进记录
- 内容：① §4.2 盘点：接口 `compute_engine.hpp:110-111`、录制图 `expr_graph.hpp`（22.5 KB，`ExprGraph`/`recording_graph_owner`）；生产调用方无——唯二调用者 `FusedChainLayer`（`compute_layer_mlp.hpp:762-800`，自述"IR-C 演示"层，不被模型工厂使用）与 `tools/scan_exprs.cpp:136-143` + `src/expr_fuse_test.cpp`。② §4.2.1 代数关系：IR-C 是 DSL 同一流水线的阶段 C（IR-A/B `expr_opt.hpp`、IR-C `expr_graph.hpp`、IR-D `expr_emitter.hpp`）；原始开发分支时间线 `bf86af2`(08-21 DSL) → `662cb08`(08-22 begin_expr/end_expr) → `36caba4`(08-24 IR-C/D)，DSL 先出现、IR-C/D 是同波加入。③ §4.2.2 数据流图：`dsl::compute → ExprSpec → IR-A/B → eval_expr → IR-D`，`begin_expr` 段走 `ExprGraph → fuse_expr_graph`；CPU `eval_expr_impl` 与 GPU `eval_expr` 都先 `canonicalize_expr_spec`。④ §4.2.3 非对称：GPU 真录制 + `execute_fused_graph`，CPU 仅 `NN_EXPR_SCAN` 下动作（普通运行 no-op）→ 接 IR-C 对 CPU 收益为零。⑤ §4.3：`dsl::start_expr/end_expr`（`ExprBlock`）仅被测试使用，随 IR-C 删除。
- 结论去向：完整取舍与重新立项前提见 `docs/todo/03-ir-optimization.md` §5.3。

## §5 遗留物清单历史行（原位置 docs/todo/12-compute-engine-inventory.md §5）
- 类型：演进记录（已删除/已修复条目）
- 内容：~~CUDA 引擎+backend+`cuda/`+10 处 `#ifdef`~~（~100 KB，已移除）；旧代数 AST `algebra_expr.hpp`+`algebra_ops.hpp`+`algebra_compute.hpp`（27.6 KB，已删除——注意 `algebra_ops.hpp` 仍在，判定原文含误列）；IR-D `CpuEmitter` 残留注释（2026-09-25 已清理，`cpu_emitter.hpp` 已删）；`FusedChainLayer`（~60 行，已移除）；`begin_expr`/`end_expr`+`expr_graph.hpp`+`Tensor::virtual_tag_`（~24 KB，已移除）；`eval_cpu` 串行模板路径（已由并行化 `eval_into_span` 取代）；`axpy_inplace`/`broadcast_row_inplace`（原判无根，2026-09-26 已删）；`broadcast_col_inplace`（原仅测试/bench，已删）；`nn::one_hot`（无调用点且与铁律 9 冲突，已删）；`Matrix::multiply_transposed_add_to`（自述死代码，已删）；`elementwise_select_scalar_cond`（无调用点，已删）；`offload_store`/`offload_load`（原仅测试，已删）；`row_reduce_max` 引擎算子（无 Layer 调用点，已删）。

## §6 差距表旧数字（原位置 docs/todo/12-compute-engine-inventory.md §6）
- 类型：演进记录
- 内容：A. Layer 直调算子 ~~35 个~~ → 23（2026-09-27 计算类已全量迁移）；B. 求值机制原为 4 套（含旧代数 AST），2026-09-26 起 CPU 只剩 2 套；C. 融合世代：IR-C 已于 2026-09-19 移除（设计无收益点，见 §8.9 与 03 §5.3）；D. 死代码：CUDA 全链 ~100 KB、IR-C ~24 KB 已清。

## §7 演进段（原位置 docs/todo/12-compute-engine-inventory.md §7.2/§7.4/§7.5/§7.6）
- 类型：演进记录
- 内容：① §7.2 类别①原列的 `elementwise_unary/binary/binary_scalar`、`broadcast_row/col_inplace`、`axpy_inplace` 已于 2026-09-26 删除（`add_inplace`/`scale_inplace`/`accumulate`/`zero` 仍在接口）。② §7.4 原文："保留与 DSL 重复的逐元素路径 → 就是今天 CPU 上'旧代数 AST / DSL 模板 / 解释器'三套并存、层直调 35 个算子的直接来源。"③ §7.5 原"缺三样"中的前两样已完成/已处置：`dsl::compute_into(eng, expr, dst)` 已落地（optimizer/残差/LayerNorm 梯度/CE 梯度已迁移）；跨表达式融合（IR-C）判定无收益并移除。④ §7.6 务实路径原文："先把 §7.2 ① 标记为 legacy 并逐层迁移（LayerNorm/Softmax/Linear/激活先行，optimizer/RLA/offload 放后面）"——该迁移已执行（2026-09-27，见顶部横幅条目）。

## §8 方案（决策前草案）：把 IR-C 吸收进 dsl::compute——全节（原位置 docs/todo/12-compute-engine-inventory.md §8.1–§8.10）
- 类型：否决方案
- 内容（精简，保留关键数字与结论）：2026-09-18 候选方案，最终未采用；2026-09-19 决定直接删除 IR-C。要点：
  - §8.1 用户入口 3 → 1：`dsl::compute/compute_reduce/compute_into`（保留）、`dsl::start_expr…end_expr(ExprBlock)`（仅 expr_dsl_test 用）、`engine.begin_expr…end_expr`（仅 FusedChainLayer + scan 用）——后两者删除；收敛不是靠"自动融合"，而是靠删除跨表达式融合能力。
  - §8.2 逃逸语义由用户负责的反例：LayerNorm（`compute_layer_mlp.hpp:499-510`）`normalized_cache_` 持有节点输出，自动内联会使 backward 静默算错 → 必须把逃逸检测变成框架不变量。
  - §8.3 可行性：`Tensor` 内持 `shared_ptr<MatrixT<P>>`（`compute_tensor.hpp:67-68`），拷贝构造时 `note_external_copy(tag)`，`end_expr` 时图外拷贝数 > 0 的节点 pin（禁止内联）。
  - §8.4 作用域方案：(a) `Layer::forward/backward` 加非虚壳（26+26 处，机械）建议；(b) `Model` 层 RAII（易漏）。
  - §8.5 自动 flush 点：非 DSL 引擎算子约 22 个 + 逃逸 + 作用域结束；`create_tensor`/`device` 不 flush。
  - §8.6 移除 `begin_expr/end_expr`、`FusedChainLayer`、`dsl::start/end_expr`；保留 `expr_graph.hpp` 为内部机制。
  - §8.7 代价：构建期 spec 数上升、未命中 AOT 硬报错、CPU no-op。
  - §8.8 第 0 步（删显式 API、保留 expr_graph.hpp）/ 第 1 步（自动作用域+逃逸检测，有真实需求再做）。
  - §8.9 接线可行性实测：融合条件（`expr_graph.hpp:180-211`：两节点无归约、同形状、非 vector_out、B 不含 matmul、tail 恰一个消费者且作 Linear 视图）逐层扫描，全部被三类阻挡——① 归约是硬边界（LayerNorm mean/var 两次 col_reduce、RMSNorm、Softmax、Attention m/l/W）；② eager 算子交错（RLA `rms_norm_backward_` 等，录制段内 eager 读到未写入的虚拟 buffer）；③ 中间量逃逸到 backward 缓存（`normalized_cache_`/`normed_cache_`/`residual2_cache_`/Attention `W_re`——缓存就是图外第二个消费者）。能融的地方早已写成单个表达式。三点结论：接线需 ~22 个 flush 点 + Tensor 拷贝钩子 + 26+26 非虚壳；当前无任何"安全且可融"的多表达式链；CPU 零收益。
  - §8.10 决策已执行（2026-09-19）：连 `expr_graph.hpp` 一起删（删显式 API 后它零调用方，保留即"悬空设计"）。删除清单：`expr_graph.hpp`、`ComputeEngine::begin_expr/end_expr`（接口+CPU/GPU 实现）、`GpuEngine::execute_fused_graph`、`dsl::start_expr/end_expr`、`FusedChainLayer`、`Tensor::virtual_tag_`、`src/expr_graph_test.cpp`/`src/expr_fuse_test.cpp`、`scan_exprs` 的 FusedChainLayer dry-run 段。保留 `run_fused_gpu` 的 `output_override`（现服务 `dsl::compute_into`）。验证：重建通过，ctest 15/15。重新立项前提：出现"纯逐元素、同形状、中间量不必为 backward 保留、且长到单个 dsl::compute 写不下"的链，按逃逸检测+自动作用域+自动 flush 实现，不复活显式录制 API。

## 13 文首「正确性缺陷已在同轮修复」清单（原位置 docs/development/13-refactor-backlog.md 文首）
- 类型：已修复勘误
- 内容：v1.4.2 审查同轮修复：threadpool 确定性归约、`require_same_shape` Release 守卫、mnist optimizer batch 包裹、`submit()` 死代码删除、zipt abort 诊断、python 裸 except、过时注释清理（修复详情见 git log）。

## 13 §6 已核对为误报/已修复的审查项对照表（原位置 docs/development/13-refactor-backlog.md §6）
- 类型：否决方案 / 已核对误报
- 内容（整块搬移，重复立项前先读）：

| 审查项 | 核对结论（2026-09-25 读码） |
|---|---|
| BPE vocab_size 下溢（P1） | **已修复**：`domain_tokenizer_bpe.hpp:84` 有运行时守卫（`< 258` 返回 Error），注释 :81-83 即该守卫的说明，审查误读为"未加校验" |
| 反序列化无长度上限（P1-S3） | **已修复**：`model_serialization.hpp` `kMaxSerializedStringBytes`=64MiB，read_spec_header/read_tokenizer 双处校验；同步了过时 TODO 注释 |
| registry bin 丢 batch（P1-23/25） | **已修复**：`expr_registry.hpp:157/:198/:296/:342` 读写均含 batch（bin v5+）；`gen_fused.cpp:125/:150` 显式生成 |
| GPT generate 未包 begin_batch（P1-7/29） | **已修复**：`compute_layer_gpt.hpp:984-1099` fill_cache_ 与逐步生成均已包裹 |
| forward_sparse 越界 label 静默置 0（P2） | **非缺陷**：越界列 mask 同步置 0（`compute_loss.hpp:359/:383/:394`），loss/梯度/num_valid 全部排除该列 = 忽略语义；`ce_fusion_test.cpp:122-128` 有专项测试 |
| restart_on_device_lost exit(0)（P1-33） | 当前代码无此函数（`text_train.cpp:427` 的 exit(0) 是 `--help` 正常退出） |
| checkpoint 不存 Adam m/v（P1-34） | **属实但为功能缺口非缺陷**：`.bin` 无优化器状态段；需 MODEL_VERSION 升版 + 格式设计（见 §7） |
| ZiPT stored_tokens 无门控（P1-10） | **低优先**：clone 是 token IDs（(batch,seq)，量级 KB-MB），非激活张量；GPT/RAPT 同模式，门控需训练/推理模式标志，收益小 |

## 13 §9 已执行：代码缩减轮执行表（原位置 docs/development/13-refactor-backlog.md §9）
- 类型：演进记录
- 内容：依据全仓"悬空设计"审计（A/B/C 三档）+ activation offload 只读评估，逐项 build + ctest 验证（**20 → 19 个目标，19/19 全绿**）：① A 档死码全删（`elementwise_select_scalar_cond`、`axpy_inplace`、`broadcast_row_inplace`、`nn::one_hot`、`text_train::one_hot_labels`、`Matrix::multiply_transposed_add_to`、`nn::sigmoid·relu`、`ops::Sigmoid·ReLU`）；② 新增 `src/test_common.hpp`（`CHECK`/`make_tensor`/`check_close`/`approx`/`dot`/`close_to` 收敛唯一副本，保留聚合编译 `#define main`）；③ RAPT `forward_step` 的 `num/den` 除法改 `dsl::compute(engine, leaf/(leaf+Scalar{1e-4}), rows, cols)`，与 `forward` 同构复用已注册 AOT 键；④ 算子收敛（删 `elementwise_unary/binary/binary_scalar`、`broadcast_col_inplace`+`broadcast.comp` 全链、引擎 `row_reduce_max`、`offload_store/load`、`UnaryOp/BinaryOp/CompareOp`；引擎 virtual 58 → 49）；⑤ AST 移除（`algebra_expr.hpp`+`algebra_compute.hpp` 删除，`Expression`/`BoolExpression` 迁入 `expr_dsl.hpp`，CPU 求值只剩两套）；⑥ offload 评估（保留 A 组生产链、删 B 组 `offload_store/load`+`offload_primitive_test`+重复聚合器；`gpt_offload_test` 补 `begin_batch/end_batch`；CPU/ZiPT 由静默 no-op 改显式警告；帮助文本"互斥"更正为"可混合"）；⑦ 死 override：`TransformerEncoderLayer`/`ZiPTBlock` 的 `activation_cache()`（20 行）删除。随附五条教训保留在 13 §9 正文（现在式）。

## 08 §3.5 CUDA obj 不随 Debug/Release 重编（原位置 docs/todo/08-pitfalls-and-lessons.md §3.5）
- 类型：已失效条目（CUDA 后端已整体移除，坑不再可能触发）
- 内容：Debug↔Release 切换后 `cuda_kernels.obj` 的 DEPENDS 只有 .cu，不重编 → `_ITERATOR_DEBUG_LEVEL` 不匹配（clang-cl=2 vs nvcc obj=0）→ lld-link 失败。修复：`-Xcompiler /D_ITERATOR_DEBUG_LEVEL=N` 对齐；切换配置后删 `build\cuda\cuda_kernels.obj` 再重编。教训（若恢复多编译器混合构建仍适用）：调试级宏必须在各编译器侧显式对齐。

## 08 §5 nvcc 行与 §7 CUDA 清单项（原位置 docs/todo/08-pitfalls-and-lessons.md §5 表格、§7 修改构建/工具链）
- 类型：已失效条目
- 内容：① §5 表行"nvcc 12.8 与 MSVC 2026 (v14.51) 不兼容，cudafe++ crash | 构建（CUDA 已停用）| 用 VS 2022 BuildTools；`-allow-unsupported-compiler`"。② §7 "修改构建/工具链"两条：`[x] Debug↔Release 切换清 build/cuda/cuda_kernels.obj`（CUDA 移除后作废）、`[ ] nvcc 编译器版本匹配（12.8 ↔ VS 2022 BuildTools）`。

## 08 各条目的过程性验证/定位叙事（原位置 docs/todo/08-pitfalls-and-lessons.md §2.5/§3.7/§4.10/§4.12/§6 模式 H）
- 类型：已修复勘误 / 过程叙事（教训本身保留在正文，现在式）
- 内容：
  - §2.5 验证流水：修复前 3 类 VUID 报错 → 强制校验层跑完整 `gpu_test`（256×256/10 迭代）0 条 VUID；`--gpu=AMD` 从"10s 超时失败"变 2s 通过；时间线 / `NN_VULKAN_NO_TIMELINE=1` 回退两路径 + ctest 15/15 全绿。
  - §3.7 修复验证：60000 训练 + 全量评估一轮跑通（`train_acc=89.95% test_acc=90.45%`，7.2s）。
  - §4.10 定位过程：最初怀疑 `generate_glsl_reduce` 内联 `col_broadcast` 下标算错，写了错误注释与工作区规避，多花一轮；打印生成的 GLSL 后证明 shader 正确，真因在 `run_fused_gpu` 的 `pc_base`（`raxis>=0 || has_mm ? 5 : 2` 把"归约无 matmul"算成 5，实际固定头 4）。
  - §4.10 复发叙事（2026-09-23，P-C1 fold 形态）：pc_base 有两处——写入侧 `run_fused_gpu` 与创建侧 `VulkanPipeline::create_generic` 的 `push_constant_size`；新增 fold（5 uint 头）只改写入侧 → `vkCmdPushConstants` 超 range 被驱动丢弃 → `fold_k` 读未定义残留，表现为"时对时错"（残留恰=K 造成假 PASS）；定案方法：K=1 打印全部行 CPU/GPU 实值。
  - §4.12 修复与红验证：恒等元 `0xff800000` → `0xFF7FFFFF`（对齐 CPU `lowest()`）；fold 测试 doc 分段边界改 `seq>EXPR_FOLD_BLOCK ? BLOCK+1 : seq/2`；对拍加 NaN 守卫；红验证——revert 修复后补边界仍绿，加 NaN 守卫才转红（`[FAIL] attn-fold doc seq=133 err=inf`）。
  - §6 模式 H 案例清单（2026-09 修复，均已完成）：`RAPTModel::clear_cache()` 曾清空 `token_emb_`；`RAPTModel` 三个模式开关曾是静默 no-op；`MaxPool2D::backward()` 曾在 `clear_cache()` 后越界读空 vector；`RAPTBlock` 曾用基类默认 `forward_recompute` → stride=2 静默用陈旧缓存；`RAPTModel::forward` 曾无法关闭文档感知。

---

# docs/development 02 · 03 · 05

`docs/development/` 三个文档（02-operator-fusion / 03-ir-optimization / 05-mixed-precision）
整改时移出的历史记录。整改原则：正文只记录当前状态、当前机制与仍然适用的结论；本文件保存
被移出的演进流水、分期验收记录、被否决方案与性能 A/B 过程，供追溯用。

## 两期主线与分期状态横幅（原位置 docs/todo/02-operator-fusion.md 篇首导语）
- 类型：演进记录
- 内容：原文把两期工作串成流水：「一期（M1-M7）用手写 op 级融合原语解决 GPT+Vulkan 训练显存/开销问题；
  二期（S1-S7）把 matmul 纳入 IR 融合、实现跨 kernel 自动融合（原规划的图级缓存落地方案已随 IR-C 于
  2026-09-19 删除），最终删除一期手写融合原语。状态：一期已实施完成（M4-M6 手写原语已由二期 IR 融合取代）；
  二期 S1-S5、S7 已实施，S6 自动窗口因用户决策搁置（其替代方案 P2-12 图级缓存亦随 IR-C 删除）；
  P-C2-7 起注意力 forward 进一步改为单 fold kernel。」正文导语已改为现在式（M4-M6 不存在、S6 不接线、
  forward = 单 fold kernel）。

## 表达式录制 API（begin_expr/end_expr）的设计与否证（原位置 02 §表达式录制与融合边界（已移除））
- 类型：被否决方案 / 取舍记录
- 内容：一期曾用 `begin_expr/end_expr` 作"计算级融合"入口（`begin_batch/end_batch` 是提交级）：
  `begin_expr` 进入录制、`end_expr` 做融合分析（构建虚拟寄存器 DAG + 判定融合边界）——小中间量留寄存器、
  逐元素链并入同一 kernel、大张量 spill 成下一 kernel 输入，CPU 端为 no-op。**最终不用它**（逐条经代码核对）：
  ① 融合条件要求"两节点均无归约、同形状、tail 恰好一个消费者"，而 LayerNorm/RMSNorm/Softmax/Attention
  的骨架是"归约 → 逐元素 → 归约"，归约处即中断；② 需要融的层都要为 backward 缓存中间量
  （`normalized_cache_`/`normed_cache_`/`residual2_cache_`/`W_re`）——缓存就是图外的第二个消费者，
  录制图看不见它，补逃逸检测（Tensor 拷贝钩子 + 自动作用域 + ~22 个 flush 点）远超收益；
  ③ 能融的长逐元素链本来就能写成**一个** `dsl::compute`；④ 唯一使用方是演示层 `FusedChainLayer`，无生产调用方。
  **删除项**：`expr_graph.hpp`、`ComputeEngine::begin_expr/end_expr`、`dsl::start_expr/end_expr`（`ExprBlock`）、
  `FusedChainLayer`、`Tensor::virtual_tag_`；**保留** `run_fused_gpu` 的 `output_override`
  （现服务于 `dsl::compute_into` 原地目标传递，与图 IR 无关）。

## 两趟式注意力 forward 三原语与 M4/M6 实施坑（原位置 02 §关键算法「两趟式注意力（Forward）」）
- 类型：演进记录 / 已删实现
- 内容：一期用三个 op 级融合原语（M4/M6，二期 S7 后删除）实现不物化的两趟 forward：
  Pass1a `engine.batched_matmul_reduce(Q,K,BH,ReduceOp::Max,transA=true,…,reduce_cols=true)` 求 m；
  Pass1b `engine.batched_matmul_softmax_denom(Q,K,m,…)` 求 l = Σ exp(alpha·QᵀK − m)；
  Pass2 `engine.batched_matmul_softmax_apply(Q,K,V,m,l,…)` 逐 tile 累加 O。显存收益：`scores`/`masked`/
  `attn_cache_` 三份 `BH·seq×seq` 消失，只剩 `m`/`l`（`BH·seq`）与 `O`；代价是 QᵀK 算两遍（2× FLOPs），
  该代价在 fold 单 kernel 后消失。掩码描述子：`apply_mask_` 钩子与 `two_pass_mask_` 决策钩子返回
  `{use_two_pass, bias}` 组合式 `AttnBias`（已删除）。M4/M6 实施坑：
  - 掩码用共享 (M,N) 张量（`-inf` 屏蔽 + 有限偏置），不在引擎硬编码因果/ALiBi；`bmm_apply` 共享内存存
    W 行（N≤4096），Phase1 算 W、Phase2 累加 O。
  - `dispatch_bmm_generic` 输出约定：返回张量永远最后一个 binding、out 参数倒数第二
    （曾写反导致 GPU grad_K 与 grad_V 互换）。
  - kv_backward 的 `A_b[:,i]` 索引：transA 时 `A_b[k][i] = flat[b*K*M + k*M + i]`；!transA 时
    `A_b[i][k] = flat[b*M*K + i*K + k]`（CPU 参考曾把两分支写反，以 shader/前向 dot_ab 约定为准）。
  - V/G 布局转换：`transpose → rearrange_3d(seq, BH, d_k, false)` 实现按 batch 转置。
  - S7 曾用 `batched_matmul_softmax_backward_q/kv` 两融合原语做 backward（已随 S7 删除）。

## 稀疏 CE 手写原语期（原位置 02 §关键算法「稀疏交叉熵（M5）」）
- 类型：演进记录 / 已删实现
- 内容：一期用 `col_softmax_denom` + `col_softmax_sparse_forward` 两个原语，单 kernel 同时算稠密梯度与
  标签位置 loss_vec；M5 期的坑：labels 以 `(1,N)` float 上传（vocab_size ≤ 2^24 精确表示，kernel 内
  `uint(labels[i])` 读取，越界/被 mask 列整列置 0）；`dispatch_bmm_generic` 的 binding 顺序；
  融合路径与回退路径的 num_valid 判定必须一致。二期 S7 后改由 IR 表达（`col_reduce_max + exp + row_gather`）；
  labels/loss_mask 以 `(1,total)` 浮点打包这一条**仍是当前机制**（见 compute_loss.hpp）。

## 跨 kernel 自动融合：S6 自动窗口与 P2-12 图级缓存（原位置 02 §跨 kernel 自动融合）
- 类型：被否决方案
- 内容：一期曾规划用 `begin_expr/end_expr` 做跨表达式融合（仅演示 Layer `FusedChainLayer` 使用；真实
  Layer 每表达式独立 dispatch）。二期 S6（P2-10 自动窗口）方案：`GpuEngine` 维护线程局部
  `std::optional<ExprGraph>` 窗口，`eval_expr/eval_expr_reduce` 调用时尝试并入、不兼容时 flush——该方案搁置；
  随后落地的 P2-12 图级缓存（`graph_cache_key`/`plan_from_kernel`/`instantiate_plan`，跨 step 复用融合分析）
  也随 IR-C 于 2026-09-19 删除。当前结论不变：不接线，能融的写成单个 `dsl::compute`。

## 二期 S1-S7 的依赖关系与删除顺序（原位置 02 §二期 S1-S7 表尾）
- 类型：演进记录
- 内容：依赖关系：S1→S2→S3 串行（IR → CPU → GPU 生成）；S4 依赖 S3；S5 依赖 S3；S7 依赖 S5。
  删除顺序：先删 GPU shader（`bmm_*.comp` / `col_softmax_*.comp`），再删 `ComputeEngine` 虚接口，
  最后删 CPU/CUDA 实现与 `vk_backend` dispatch；每步删除前对应 IR 融合路径已在测试覆盖。
  S6 行原记「搁置（用户决策）；替代方案 P2-12 图级缓存亦随 IR-C 于 2026-09-19 删除」。
  S7 行原记删除 7 个原语（`batched_matmul_reduce/softmax_denom/softmax_apply`、
  `batched_matmul_softmax_backward_q/kv`、`col_softmax_denom/col_softmax_sparse_forward`）
  的接口 + CPU/GPU 实现 + 7 个手写 shader + vk pipeline + 测试改造。

## 一期 M1-M7 里程碑表与当时验收（原位置 02 §里程碑与实施记录 一期 M1-M7）
- 类型：演进记录
- 内容（含当时测试名，现均已并入聚合目标）：
  | 里程碑 | 内容 | 验证（当时目标名） |
  |---|---|---|
  | M1 | `ExprSpec` 加归约视图/指令 + CPU `eval_expr` 扩展 | `expr_reduce_test` + `expr_dsl_test` |
  | M2 | `begin_expr/end_expr` 录制框架 + CPU no-op | 2026-09-19 随 IR-C 删除 |
  | M3 | Softmax/LayerNorm/RMSNorm fwd/bwd 改 DSL 归约表达式 + GPU 归约融合 shader | `fused_gpu_test` + gradcheck |
  | M4 | 三个 matmul 融合原语（bmm_reduce/denom/apply） | `matmul_fusion_test`（CPU err=0 / GPU err≤1.9e-6） |
  | M5 | CrossEntropyLoss 稀疏融合 | `ce_fusion_test`（CPU err=0 / GPU err≤4.8e-7） |
  | M6 | Attention 两趟式（forward + 反向重算 W） | `matmul_fusion_test` 8 用例 + gradcheck + 训练 |
  | M7 | 文档补"原语可专、不叫算法名"约定 | 全套测试 |
  测试名归并：`matmul_fusion_test` 已无同名目标（用例在 `expr_cpu_test`/`expr_gpu_test`）；
  `expr_dsl_test`/`expr_reduce_test`/`expr_matmul_test`/`expr_opt_test` 并入 `expr_cpu_test`；
  `fused_gpu_test`/`tensor_expr_test` 并入 `expr_gpu_test`。
  形状无关融合的原始效果记录：扫描表达式 24 → 18 条（RoPE 4×dk 去重）。

## 教训条目「dispatch_compute 误删后从调用点重建」（原位置 02 §二期关键教训 7）
- 类型：演进记录（修复叙事）
- 内容：一次把 `compute_vk_backend.hpp` 的 `dispatch_compute` 误删后，按调用点重建。
  正文已把该条改写为现在式提醒（改签名/删除时按调用点逐一核对）。

## 教训 1 的旧示例写法（原位置 02 §二期关键教训 1）
- 类型：演进记录
- 内容：原文为「`scale` 折进 Q（forward `scale_inplace` + backward 补乘）、`inv_num_valid` 后置
  `scale_inplace`」。核对当前代码：**scale 折进 Q + backward 补乘仍是当前做法**，但载体已从引擎
  `scale_inplace` 改为 `dsl::compute_into(engine, leaf(Q) * dsl::rparam(scale_), Q)`
  （`compute_layer_attention.hpp`，rparam 值不进 key）；`inv_num_valid` 也改为 CE 表达式内的
  `dsl::rparam(inv_num_valid)` 尾链（`compute_loss.hpp`），不再后置 `scale_inplace`。
  正文教训 1 已按当前写法改写。另：形状无关融合的原始效果记录（扫描条目 24 → 18，RoPE 4×dk 去重）
  与教训 5 的「74 条 spec 全生成」计数已从正文移除（计数以构建输出为准）。

## IR-C（图 IR + begin_expr/end_expr 融合分析）取舍记录（原位置 docs/todo/03-ir-optimization.md §5.3）
- 类型：取舍记录 / 被否决方案
- 内容：阶段 C 的落地形态 = `expr_graph.hpp`（`ExprGraph`/`fuse_expr_graph`/`recording_graph_owner`/图级缓存）
  + `ComputeEngine::begin_expr/end_expr` 显式录制 API + `dsl::start_expr/end_expr`（`ExprBlock`）
  + 演示层 `FusedChainLayer`，基础版 2026-08-24 实施、08-26 增补中间张量消除。
  **2026-09-19 决定：不接线，整体删除。** 依据（逐条经代码核对）：
  1. **归约是硬边界**。`fuse_expr_graph` 只拼接纯逐元素链（两节点均无归约、同形状、tail 恰好一个消费者
     且以 Linear 视图消费）；LayerNorm/RMSNorm 前向是 `col_reduce → 逐元素 → col_reduce`、Softmax 是
     `row_max/row_sum → 逐元素`、Attention 的 m/l/W 同理——需要融合的层恰好以归约为骨架。
  2. **backward 缓存逃逸**。融合要求"tail 只有一个消费者"，但要融的层都要为 backward 缓存中间量
     （`LayerNorm::normalized_cache_`、`RMSNorm::normed_cache_`、`GPTBlock::residual2_cache_`、
     Attention `W_re`）——缓存是图外第二个消费者，录制图看不见它；补逃逸检测需 Tensor 拷贝钩子 +
     自动作用域 + ~22 个 flush 点 + `Layer::forward` 非虚壳（26+26 处），成本远高于收益。
  3. **能融的地方早已写成单个表达式**。GeLU/SwiGLU/ReLU/Softmax/RoPE/掩码各是一个 `dsl::compute`；
     当前层集合里不存在"写不进一行"的纯逐元素链。
  4. **CPU 零收益**。`CpuEngine` 的 begin/end 本身是 no-op。
  5. **无生产调用方 → 悬空设计**。唯一使用方是演示层 `FusedChainLayer` + scan dry-run + 两个测试。
  **删除项**：`expr_graph.hpp`、`ComputeEngine::begin_expr/end_expr` 及 CPU/GPU 实现、
  `dsl::start_expr/end_expr`（`ExprBlock`）、`FusedChainLayer`、`Tensor::virtual_tag_`、
  `GpuEngine::execute_fused_graph` 与图级计划缓存、`expr_graph_test`/`expr_fuse_test`
  与 scan 的 `FusedChainLayer` dry-run 段。
  **保留项**：`run_fused_gpu` 的 `output_override`（服务 `dsl::compute_into`，与图 IR 无关）。
  **重新立项的前提**：出现或写出这样一条链——纯逐元素、同形状、中间量不必为 backward 保留、
  且长到单个 `dsl::compute` 写不下；届时按"逃逸检测 + 自动作用域 + 自动 flush"实现，
  而不是复活显式录制 API。
  （原正文另有指针：本记录亦是 AGENTS/多份文档所指"03 §5.3 末段"的重新立项前提。）

## IR-A/B 实施记录的日期与当时验收（原位置 03 §实施记录（2026-08-23，IR-A + IR-B 已完成））
- 类型：演进记录
- 内容：落地日期 2026-08-23；当时验证清单：新增 `src/expr_opt_test`；`expr_dsl_test`/`expr_reduce_test`/
  `tensor_expr_test`/`fused_gpu_test`/`matmul_fusion_test`/`ce_fusion_test`/gradcheck 系列/gpt_checkpoint_test
  全绿；MNIST + text_train（CPU/GPU、含 checkpoint-every）端到端正常（这些目标名现已并入
  `expr_cpu_test`/`expr_gpu_test` 聚合目标）。顺带修复的与 IR 无关 bug：`text_train --save-interval 0`
  触发 `(step+1) % 0` 整数除零崩溃。
  （实现要点、关键不变量、关键坑 ①② 已作为当前事实保留在正文。）

## CpuEmitter 的存在与删除（原位置 03 §后端 emitter 抽象「注意」块 + §当前状态小结表行）
- 类型：演进记录 / 已删实现
- 内容：历史上存在 `cpu_emitter.hpp`（生成可编译 C++ 直线代码），其产物从不参与编译
  （gen_fused 硬编码 glsl）、缺陷全隐性，随文件一并删除；当时 `expr_emitter.hpp:16` 注释仍提到
  `cpu_emitter.hpp`（该注释残留后来也已清理）。早期"Phase 2 前必须先修 CpuEmitter 缺陷"的前置条件随之作废。
  工作量小结原表（含计划期估计）：IR-A 小（1–2 天）/低；IR-B 中（3–5 天）/中；IR-C 大（1–2 周）/高；
  IR-D 中（3–5 天）/中。

## 05 §12.5 边界 cast 期的实测 A/B 与探针数据（原位置 docs/development/05-mixed-precision.md §12.5）
- 类型：性能 A/B
- 内容（40HX，GPT d64/h4/L4/ff256、vocab 8208、seq 256、adam lr 1e-3）：
  | 配置 | batch | 峰值 | 耗时 | avg_loss |
  |---|---|---|---|---|
  | f32 基线 | 64 | 3069 MiB | 4.7s | 7.32 |
  | `--f16`（profile_f16，边界 cast 期） | 64 | 2831 MiB 后 OOM | — | — |
  | f32 基线 | 32 | 1588 MiB | 4.7s | 6.6726 |
  | `--f16` | 32 | 4124 MiB（2.6×） | 6.2s（+32%） | 6.7086 |
  （batch32 轨迹一致：avg_loss 6.6726 vs 6.7086，偏差 0.5%，数值正确；但峰值 2.6× → 边界 cast
  在训练峰值上净亏。）探针逐阶段（transient live / pending）：
  step0/forward 1027/230 vs 1574/1100；step0/loss-fwd 1541/230 vs 2088/1100；
  step0/backward 2795/2300 vs **6687/7000**。
  根因：每个算子都要把 f16 入参抬 f32 副本——被 k 个算子读取的张量要 k 份副本；
  `(vocab, batch·seq)` 的 logits（512MB）被 loss 链读 4~6 次 → transient 桶 6×512MB（3.1GB）。
  已缓解：LM head 计算精度固定 `stable`（F32）→ loss 链零 cast（峰值 3069→2831 MiB）；
  隐藏层激活每次 cast 仍在（2.4× 膨胀）→ batch64 依旧 OOM。

## 05 §12.6 第一期计划与"下一步"（原位置 05 §12.6 尾部）
- 类型：演进记录（计划项，均已落地）
- 内容：原「下一步（in-kernel f16 第一期）」清单：`scan_exprs` 增加 f16 profile 的 dry-run pass
  （发现与运行时同源的变体）→ bin v9 增变体段 → `GlslEmitter` 按签名对每个输入/输出选 `float16_t`
  或 `float`（`GL_EXT_shader_16bit_storage` + 显式转换）→ 设备启用 `storageBuffer16BitAccess`
  （不支持则回退边界 cast）→ 适配层按签名选 pipeline。原「结论 2」：逐元素/归约占 84%（32/38），
  第一期只做逐元素 + 归约，matmul 段与 fold 留第二期。（该 38 变体空间实测表与"签名必须逐输入"
  结论作为当前事实保留在正文。）

## 05 §12.7 第一期落地的实测与"剩余缺口"（原位置 05 §12.7）
- 类型：性能 A/B / 演进记录
- 内容（40HX，GPT d64/h4/L4/ff256、vocab 8208、seq 256、batch 32、1 epoch）：
  | 配置 | 峰值 | avg_loss | 耗时 |
  |---|---|---|---|
  | f32 基线 | 1595 MiB | 6.6969 | 4.7s |
  | f16 in-kernel（纯逐元素变体） | 3457 MiB | 6.7059（差 0.13%） | 5.9s |
  | f16 边界 cast（NN_VULKAN_NO_16BIT_STORAGE=1） | 4124 MiB | 6.7238 | 6.3s |
  收益：相对边界 cast −16%、耗时 −6%，但仍高于 f32 基线 2.2×；`f16_precision_test` GPT 轨迹对拍
  max_rel 3.2%。当时 gen_fused：54 变体 → 注册 31（其余 23 = 10 matmul 段 + 13 含归约）。
  剩余缺口（下一期）：① matmul 段（Linear 族，10 变体）；② 含归约指令的逐元素（13 变体，
  LayerNorm/RMSNorm 统计量链）；③ fold（注意力）——后由 §12.10 三类变体与 Phase D2/D3 补齐。

## 05 §12.8 探针逐阶段归因数字（原位置 05 §12.8）
- 类型：性能 A/B
- 内容：f32 vs f16(in-kernel) 同配置逐阶段：步末 released 272.5 → 288.5 MB（基本持平 → 多出来全是
  transient）；backward 峰值 1397 → 2869 MB，10–100MB 中块 16 → 52 项、<10MB 小块 234 → 587 项；
  同窗交错 3 样本 f32 1588/1588/1588（稳定）vs f16 4514/4582/3448（双峰，贵 2.2~2.9×），
  f16 transient 分配次数 119 vs 50 块 → 池回收/复用时序成为峰值决定因素。
  `supports_native_data_move()`（clone/slice_rows/insert_rows/zero 对 f16 放行）探针复核逐项相同
  → GPT 路径峰值中性（数据量小），保留但非主攻方向。

## 05 §12.9 池底材粒度 A/B（被复测否定的单点）（原位置 05 §12.9）
- 类型：性能 A/B（方法论反面教材）
- 内容：原记录「4MB / 阶梯 ≤16MB → 2705 MiB、耗时 11.3s」不可复现——同窗交错 3 轮复测三档配置
  （默认 12MB / `NN_POOL_LADDER_MAX_MB=16` / `NN_POOL_BLOCK_MB=4`）峰值 3513/3509/3141~3509 MiB、
  耗时全部 ≈5.7s，f32 同码对照也从 1588 漂到 1753（~10%）。结论：跨会话单点 + 不同时段对照组
  会被系统漂移骗；机制之争必须用形状/尺寸归因表裁决，不要用总量推理。
  仍有效副产物：`NN_PREC_TRACE=1` 的 `[prec][miss]` 打印（38 个非零签名只 6 个未命中 = eval_expr
  路径 84% 已原生）——该工具结论保留在正文。

## 05 §12.10 的轮次实测与旧"下一步"（原位置 05 §12.10 ①②④⑤）
- 类型：性能 A/B / 演进记录
- 内容：
  - ① 池粒度复测表（同窗交错 3 轮，默认 12MB 3513/3513/3513 @6.03s；ladder16 3509/3509/3141 @5.74s；
    block4 3509/3249/3457 @5.69s）→ 均值仅 −3.6%，"−21% + 2× 耗时"不成立；池账本 1 万次 allocate
    只扫 ~5 万 block / ~2 万 region、`vkAllocateMemory` 累计 ~150ms → 池记账不是瓶颈。
  - ② 形状级归因表原始输出（batch32 f16，1 step）：
    `->f32 副本 (32768,256) x36 1152.0MB ← W/grad_A`；`(64,8192) x135 270.0MB`；
    `(256,8192) x20 160.0MB`；`->按P落回 (32768,256) x8 128.0MB`；`(2048,256) x56 112.0MB`。
    `(32768,256) = (batch·H·seq, seq)` 即 AttentionBase::backward 物化的 W 与 grad_A（各 32MB f32），
    被 `batched_matmul`/`compute_reduce`/`compute_into` 反复抬 f32，每个副本活到帧末。
    教学点：机制之争必须用形状/尺寸归因表裁决。
  - ④ 实测（同窗 3 轮，batch32/steps2/no-kv）：f32 1753（3/3 一致）、f16 2025/2025/2273
    （1.15~1.30×；此前 3513 = 2.2×，cast 路径 −35%~−41%）；耗时 f32 5.33s / f16 5.40s（无回退）；
    backward transient live 2878 → 1767MB、32MB 级块 44 → 16、pending 2914 → 1674MB；
    batch64 f16 不再 OOM（4475 vs f32 3411）；gen_fused 由「74 结构 + 41 变体（14 skip）」
    → 「74 + 54 变体（0 skip）」（纯逐元素 31 + matmul 段 10 + 含归约 13）。
  - ⑤ 当时的 `[prec][miss]` 7 条清单（fold 1 / matmul 段 run-only sig 3 / 含归约 `[f32,f16]→f32` 2 /
    逐元素 2）与「下一步」：让 gen_fused 发射补集变体、op-level f16 GEMM（3 个手写 GEMM 宏参数化
    编第二份 SPIR-V）、fold 带类型变体——三者分别由 `tools/prec_backfill.txt`（Phase D3）、
    `-DNN_SHADER_F16`（§12.11）与 Phase D2 落地。

## 05 §12.11 的零回归字节数与失败叙事（原位置 05 §12.11 ①②）
- 类型：演进记录
- 内容：
  - 零回归证明（当时）：`matmul_tiled_spv.hpp` / `batched_matmul_spv.hpp` 字节数改动前后完全一致
    （68670 / 74408）→ 宏展开后的 f32 GLSL 编出同一份 SPIR-V。CMake 接线当时只有两行：
    `nn_embed_shader(batched_matmul_f16 … -DNN_SHADER_F16=1)`、`nn_embed_shader(matmul_tiled_f16 … -DNN_SHADER_F16=1)`
    （后续已扩展到 reduce/transpose/gather/rearrange_3d/scan_*/outer_col/group_reduce/scatter_add/
    im2col/col2im/elementwise_v2/matmul_gemv）。
  - 失败叙事：适配层第一版漏了「把 `P != F32` 直接下传内层」——归因表逐项不变、峰值纹丝不动
    （内层永远看不到 f16 → 原生 GEMM 永远命中不了）；`CpuEngine::matmul_with_bias` 曾 `(void)P`，
    导致 scan 的 f16 dry-run 只登记全 f32 签名 → matmul 段变体永远发现不到（补后变体数 54 → 56）。
  - 当时的剩余项（现已处理/部分过期）：仍走 cast 的 op-level 原语约 550MB
    （`(2048,256)×80`、`(64,8192)×72`、`(256,8192)×10`、`(32768,16)×24`、`(8208,64)×10`）；
    7 条运行时签名；fold 带类型变体未做。
- 备注：`gen_fused` 变体数 54→56、以及演进全景数字（f16 边界 cast 4124 → in-kernel 首期 3513 →
  +matmul/归约/目标传递 2025~2286 → +op-level f16 GEMM 1570，f32 基线 ~1588~1754）
  属分阶段 A/B 流水，最终数字以正文 §12.11 ③ 为准。

---

# docs/development 01 · 04 · 06 · 07 · 10 · 14

> 本文件汇集 docs/development/01-compute-engine-development.md、04-memory-optimization.md、
> 06-rapt-algorithm.md、10-development-standards.md、14-f16-stable-gpu-loss-frozen.md
> 整改中移出的历史叙事（演进流水 / 已修复勘误 / 删除清单 / 性能 A/B / 排查过程）。
> 正文只保留当前状态、当前机制与仍然适用的教训；07-zipt-algorithm.md 为纯算法文档，无移出条目。

## 01 §5 引擎 row_reduce_max 删除注脚（原位置 docs/development/01-compute-engine-development.md §5）
- 类型：删除清单
- 内容：原文「⚠️ 2026-09 收敛：`row_reduce_max`（引擎算子）已删除——按行求最大走 DSL 归约叶子 `dsl::row_reduce_max(...)`（详见 `expr_dsl.hpp`；AGENTS §4.3 有完整删除清单）」。当前做法（按行求最大用 DSL 叶子、归约表达式走 `dsl::compute_reduce`）已保留为正文提示。

## 01 §6 逐元素/广播算子删除清单与理由（原位置 docs/development/01-compute-engine-development.md §6）
- 类型：删除清单
- 内容：`broadcast_row_inplace`、`broadcast_col_inplace`、`elementwise_unary`、`elementwise_binary`、`elementwise_binary_scalar`、`elementwise_select_scalar_cond`（连同 `UnaryOp`/`BinaryOp`/`CompareOp` 枚举）已全部删除（2026-09 收敛轮）。理由：这些算子在 DSL 落地后已无生产调用方，且与表达式能力完全重复（详见 `todo/12-compute-engine-inventory.md` 顶部收敛横幅与 `13-refactor-backlog.md` §9）。

## 01 性能优化节旧 dsl::compute 示例（原位置 docs/development/01-compute-engine-development.md 性能优化 §2）
- 类型：已修复勘误（API 形式过期）
- 内容：原文用 lambda + 张量列表形式 `dsl::compute(engine, [](auto a, auto b, auto c){...}, {tensor_a, tensor_b, tensor_c}, rows, cols)`——该重载在代码中不存在。现 API 为 `dsl::compute(eng, expr, rows, cols[, precision])`（expr_dsl.hpp:1317），已按 `dsl::leaf(*a) + dsl::leaf(*b) * dsl::leaf(*c)` 形式改写。

## 01 Q3 旧答案：通过 UnaryOp/BinaryOp 枚举加激活函数（原位置 docs/development/01-compute-engine-development.md 常见问题 Q3）
- 类型：已修复勘误
- 内容：原文步骤为「1. 在 `UnaryOp` 或 `BinaryOp` 中添加枚举值；2. 在 `CpuEngine` 中实现逐元素运算；3. 在 `GpuEngine` 中实现 shader」——枚举与逐元素引擎算子已删除，该路径不存在。现行答案：在 Layer 内用 DSL 组合表达（`ReLU::forward` = `dsl::max(dsl::leaf(x), Scalar{0})`），引擎/ shader 不认识算法名（铁律 3）。

## 04 实施进度日期标注与 RAPT 接入旧叙事（原位置 docs/todo/04-memory-optimization.md 背景与目标 §实施进度）
- 类型：演进记录 / 已修复勘误
- 内容：① 原节标题「实施进度（2026-08-22）」及各条目的日期戳（L1 续：RLA/RAPT 接入 2026-09-19）。② 「此前 `set_checkpoint_every`/`set_activation_offload`/`set_flush_interval` 对 RAPT 是静默 no-op，而 CLI 会打印"已启用"」——该缺陷已于 2026-09-19 修复，当前 RAPT 三开关真实生效。③ 「`activation_cache()` 补齐此前遗漏的 4 项（否则 offload 覆盖不全）」——当前已列出全部 7 项层内缓存 + 4 个子层缓存。④ L2 条目中「（CPU no-op；CUDA 已停用）」的 CUDA 提法——CUDA 后端已整体移除（96a3675），当前只写 CPU no-op。⑤ 「新增 rapt_offload_test」「text_train 新增 --checkpoint-every」等"新增"措辞。

## 04 基准显存 27G vs 29G 的融合 A/B（原位置 docs/todo/04-memory-optimization.md 背景与目标 §基准配置与峰值显存）
- 类型：性能 A/B
- 内容：原文「该配置下训练峰值显存 ≈ 27 GB（已含 M5/M6 融合收益），此前 29 GB（融合前）」；「融合（M4/M5/M6）已把非线性象限（seq² 注意力、vocab×seq 全 softmax）从 ~3× 砍到几近为 0」。现正文只保留 27GB 当前口径与"非线性象限不物化 ≈0"的当前状态。

## 04 f16 红线修订注脚（原位置 docs/todo/04-memory-optimization.md 现状与根因 §数值精度）
- 类型：演进记录
- 内容：原文「（立项时口径……）（历史红线"不引入 f16"已随混合精度修订。）」。当前事实：本文档数字为 fp32 基线；v1.2.0 起 f16 混合精度见 05-mixed-precision，本文档显存账不适用于 f16 路径。

## 04 L2 旧问题叙述：内存池永不归还（原位置 docs/todo/04-memory-optimization.md 现状与根因 §内存池碎片化）
- 类型：已修复勘误
- 内容：原文「但：从未将整个空 Block 归还 GPU（`blocks_.clear()` 仅在析构时触发），block 底材按需 128MB（或超尺寸单块）申请后不回收 ⇒ 峰值生命周期等于整个进程/测试生命周期」；「算子融合文档将"内存池 first-fit 碎片化 + 永不归还"列为独立跟踪项、不随融合解决」。该问题已由 L2 的 `release_idle_blocks()` / `release_idle_pool_blocks()` 解决（step 边界整块归还）；残余风险（step 内仍累积）保留为正文。

## 04 注意力两趟式旧实现描述（原位置 docs/todo/04-memory-optimization.md 现状与根因 §注意力形态）
- 类型：演进记录
- 内容：原文「原 `batched_matmul_reduce/max → denom → apply` 两趟式 forward 与 `batched_matmul_softmax_backward_q/kv` backward 重算 W 方案，已被 IR 融合替代（算子融合文档 S7；2026-09-23 起 forward 进一步换单 fold kernel——FoldSpec 分块流式……）」。当前状态（单 fold kernel、S 不落显存、backward 为 R/X 表达式 + batched_matmul）已保留为正文。

## 04 「已核对（2026-08-24）」审查记录框（原位置 docs/todo/04-memory-optimization.md L2 节）
- 类型：演进记录（结论仍适用，已改写为常设约定）
- 内容：原文以「已核对（2026-08-24）：中间 Tensor 的归还路径是安全的，无需修改即可维持正确性」开头的审查记录。四条安全约定（延迟销毁 + 录制期全内存屏障、复用前必写满、新增原语必须插 output barrier、期望为 0 必须显式 `zero()`）全部保留为正文「内存池复用安全约定」。

## 04 旧落地顺序与过期约束（原位置 docs/todo/04-memory-optimization.md 落地顺序建议 / 与既有约束的关系）
- 类型：演进记录 / 已修复勘误
- 内容：① 原落地顺序「1. 先 L1……2. 再 L2……3. 若仍不足 → 立项 L3……」（L1/L2 已完成，现正文改为"L1/L2 已就位，下一步立项 L3"）。② 「用训练 step 耗时采样替代（bench 工具已移除）」的 bench 删除提法。③ 原约束「不引入 f16（bf16/fp16）低精度训练」——已过时（f16 混合精度已落地），改为「L1/L2 与精度设置正交，本文档账目为 fp32 口径」。

## 06 §4 引擎化改造流水（原位置 docs/development/06-rapt-algorithm.md §4 开头）
- 类型：演进记录
- 内容：原文「2026-09-04 的引擎化改造删掉了 Layer 内的 `scan_forward_/scan_backward_` 纯 CPU 标量循环（PCIe 往返），改为 3 个 op-level 扫描原语，由 Layer 用原语 + 逐元素原语组合表达算法（铁律 3：shader 永不含算法）」。当前状态保留为正文。

## 06 §4.3 坑 3/坑 4 的历史措辞（原位置 docs/development/06-rapt-algorithm.md §4.3）
- 类型：已修复勘误
- 内容：① 坑 3 原文「写错则 doc-aware gradcheck 才会暴露（历史已踩过）」。② 坑 4 原文「不能让空张量流进 matmul（旧行为 = 静默算出垃圾梯度）」——旧行为已修复，现行规则（缺缓存立刻硬报错）保留为正文。

## 06 §4.4 验证基线的演进措辞与失效交叉引用（原位置 docs/development/06-rapt-algorithm.md §4.4）
- 类型：已修复勘误 / 断链修正
- 内容：① 原标题「验证基线（2026-09-04，GTX 850M）」与表内「max_err 与改造前基线一致」的对照措辞（现直接列 max_err 基线数值）。② 原注「6 处 Vulkan 接线细节沉淀在 `01-compute-engine-development.md`」——01 号文档并无该清单，属失效交叉引用，改为直接指向 `compute_gpu_engine.hpp` 与 `backend/compute_vk_backend.hpp`。

## 06 §5 历史：两趟式注意力（已删除）——全节（原位置 docs/development/06-rapt-algorithm.md §5）
- 类型：删除清单 / 演进记录
- 内容：S7 的"两趟式多 kernel 注意力"（forward 3 个融合 kernel m→l→W × 4 种掩码变体、backward 3 个，W 物化供复用）已删除——2026-09-24 起注意力 forward 为单 fold kernel 分块流式（QKᵀ/掩码/online softmax/ΣwV 逐 `EXPR_FOLD_BLOCK=128` 块完成，S 矩阵绝不物化）。与传统 Flash Attention 的对照分析（kernel 数量、IO 复杂度、掩码扩展方式）原注「见 git 历史中本节原文」。保留的结论（当前事实）已改写为 §5 正文：本项目不存在单一 flash-attn kernel、fold 流式实现 FlashAttention 核心思想、掩码变体经 FoldSpec 登记漏登记硬报错。

## 06 §6 RAPT 开关与已修前置缺陷（原位置 docs/development/06-rapt-algorithm.md §6）
- 类型：已修复勘误
- 内容：① 「（2026-09-19 补齐；此前 RAPT 侧是静默 no-op 而 CLI 会打印"已启用"）」。② 「已修的两个前置缺陷（不修则静默毁模型）：1. `RAPTModel::clear_cache()` 曾清空 `token_emb_`（模型参数）——检查点每块 backward 后都调它，等于毁掉词嵌入。2. `RAPTModel::forward` 曾只在 `doc_ids` 非空时下发文档 id → `set_doc_ids({})` 关不掉文档感知，跨 step 残留边界重置（跨样本串扰）」。③ 「`activation_cache()` ……（Q_normed/K_normed/Q_rms_inv/K_rms_inv 曾漏列 → offload 漏搬，已补）」。④ §3 表格「分数定义：推理/训练曾不一致」措辞（改为"原版缺陷"）。

## 06 §4.2 forward_step 旧原语调用（原位置 docs/development/06-rapt-algorithm.md §4.2）
- 类型：已修复勘误（API 过期）
- 内容：原文「先更新状态（`batched_matmul(V, Kp, transB)` → `add_inplace(B_state)`；`add_inplace(z_state, Kp)`）→ num = B·q'（batched_matmul）」——2026-09-27 计算类原语迁 DSL 后，Layer 不再直调 `batched_matmul`/`add_inplace`：现为 `dsl::matmul(V, Kp, false, true, H)` + `dsl::compute_into` 原地累加（compute_layer_rapt.hpp:832-848），已按代码现状订正。

## 10 compute::apply / 代数 AST 移除注脚（原位置 docs/todo/10-development-standards.md §2.6、§4.3.1、§4.3.2）
- 类型：已修复勘误
- 内容：三处注脚原文——「（旧 `compute::apply(span, expr)` 代数 AST 已于 2026-09 整体移除）」「注：旧 compute::apply 代数 AST 与广播/逐元素算子在 2026-09 已移除」「（旧 compute::apply 代数 AST 已于 2026-09 移除）」。旧规则表表述「Layer 通过 Matrix 语义 API 或 `compute::apply(span, expr)` 表达算法」「上层通过 Matrix API 或 compute::apply 间接享受并行加速」已改为表达式 DSL。

## 10 目录树中的已删文件行（原位置 docs/todo/10-development-standards.md §5.1）
- 类型：删除清单
- 内容：目录树原有一行「├── （algebra_expr.hpp / algebra_compute.hpp 已于 2026-09 移除）」——文件树只列当前存在的文件，该行已删（`algebra_expr.hpp`/`algebra_compute.hpp` 确已不存在；`Expression`/`BoolExpression` 概念迁入 `expr_dsl.hpp`）。

## 10 §6.2.4 SmartPolicy 历史说明（原位置 docs/todo/10-development-standards.md §6.2.4）
- 类型：已修复勘误
- 内容：原文「历史说明：早期文档把该策略命名为「SmartPolicy」并给出用户可替换 policy 的扩展示例，该类型在代码中从不存在（只有上述自由函数 + 阈值门控），示例已按实际 API 更正」。当前事实（无 policy 类型，只有 `parallel_for_samples`/`parallel_for_blocks` + 阈值门控）保留在代码注释形式的正文中。

## 10 过期 API 示例批量订正（原位置 docs/todo/10-development-standards.md §2.5、§3.2.4、§4.2.2、§4.3.1、§4.3.4、§5.3.1、§5.3.2、§5.4、§10.2）
- 类型：已修复勘误（API 现状订正；原内容与代码现状不符，非单纯演进注脚）
- 内容：① §2.5 「遵循隔离」示例 `matrix.apply_relu()`——`Matrix` 无此方法且违反 §2.6 自身铁律（Matrix 不写算法），改为 `add_inplace`/`multiply_to`。② §3.2.4 「避免使用已废弃的 `data_ptr()`」——接口只有 `span()`（algebra_matrix.hpp:194 注明 span 替代所有 data_ptr 场景）。③ §4.2.2 正确示例原为 `compute::apply(x, max(x, Scalar{0}))` 与 `p.binary_apply_inplace(g, lambda)`——分别改为 `dsl::compute` 与 `dsl::compute_into`（与 compute_layer_mlp.hpp ReLU::forward、compute_optimizer.hpp SGD::step 一致）。④ §4.3.1 Matrix API 列表原含 `apply`/`binary_apply`/`binary_apply_inplace` 模板——三者已不存在（`Matrix::detail::apply/binary_apply/binary_apply_inplace` 死函数删除），只留 `reduce/row_reduce/col_reduce/span`。⑤ §4.3.4 「Adam::step 通过 binary_apply_inplace 表达」——现为三个 DSL kernel（K1/K2 dsl::compute + K3 compute_into，RParam 承载偏置校正，compute_optimizer.hpp:304-353）。⑥ §5.3.1 链式 fluent 示例 `model.add<Linear>(784,64).add<ReLU>()...add<CrossEntropyLoss>()`——`Model::add` 实际返回 `Result<void>`（不可链式），`CrossEntropyLoss` 不是 Layer 不可 add；§5.3.2 原工厂示例返回 `Model&`——均改为实际实现。⑦ §5.4 检查清单「是否支持链式调用？」同步改为 Result 传播检查项。⑧ §10.2 Doxygen 示例 `@throws std::invalid_argument`——与全项目禁异常冲突，改为 Error 返回说明。

## 10 §7.2 C++ 特性表过期行（原位置 docs/todo/10-development-standards.md §7.2）
- 类型：已修复勘误
- 内容：删除三行未使用特性——`std::ranges`（原注「如 ranges::generate Xavier 初始化」，全仓无 ranges::generate）已改为实际用途（ranges::distance/advance 切分并行区间、ranges::sort）；`std::views::zip`（原注「优化器参数更新」，全仓无 views::zip——优化器用 dsl::compute_into 逐参数处理）；`std::execution::par_unseq`（原注「并行执行策略（矩阵运算）」，全仓无 std::execution，core_config.hpp:64 明言替代 std::execution::par）。`std::expected`/`std::span`/`std::views::iota` 三行保留（经核实仍在使用）。

## 10 不存在的头文件引用（原位置 docs/todo/10-development-standards.md §8.3、§9.1）
- 类型：已修复勘误（断链修正）
- 内容：① §9.1 标题「基础类型定义（nn_core_config.hpp）」——仓库无 `nn_core_config.hpp`，`Error`/`Result` 定义在 `core_errors.hpp:15-20`。② §8.3 包含顺序示例 `#include <neuralnet.cpp/nn_core_config.hpp>`（同上）与第三方库 `#include <fmt/format.h>`（全仓无 fmt 依赖）。③ §5.2.1 注释中「layer.hpp - 层定义」——现名 `compute_layer.hpp`。

## 14 §4 回归窗口与历史脉络（原位置 docs/todo/14-f16-stable-gpu-loss-frozen.md §4）
- 类型：演进记录（排查过程流水）
- 内容：① 09-24 16:54 bench 原始数据（早于 75255da）同配置行为「epoch1 loss 活（7.40）→ epoch2 发散 -nan」（该配置历史上就不可训练，§12.5 stable=F16 时 CE 链 ~200 步 NaN，但 loss 至少是活的）；② 09-26 HEAD `4a12876` 症状变为「epoch1 起即静默冻结」，「冻结比 NaN 更危险」的评述；③ 嫌疑提交清单（`git log 75255da..HEAD`：`eba744d` CPU f16 修复、`200955f` transpose 派发、`3c2b9e8` f16 MSVC 修复、`4c8c58c` 审查修复、`4a12876` 技术债清理；基线若追到 09-24 则 `b4ee2df`/`4cd0616` 也在窗口内）。与 issue #13 P0-② 的对照表为当前有效的事实区分，保留在 §4 正文。

## 14 证据行号订正（原位置 docs/todo/14-f16-stable-gpu-loss-frozen.md §3、§5、§6）
- 类型：已修复勘误（数字订正）
- 内容：报告引用的源码行号整体偏移 1–7 行，已按 HEAD `4a12876` 工作区实况订正：`text_train.cpp` 1513→1514（forward_sparse_sum）、1522→1523（cast）、1571→1572（submit_scalar_readback）、1366→1367（harvest 打印）、582→583（实验性注释）；`compute_gpu_engine.hpp` 150→146（submit_scalar_readback 定义）；`f16_precision_test.cpp` 642-649→639-646（profile 单字段矩阵）、698-702→691-700（main() CPU 块 + NN_F16_DEBUG 门）。缺陷本身未修（HEAD 仍为 4a12876，无修复提交）。

---

# docs/introduction · usage · benchmarks

> 本文件收录 docs/introduction/、docs/usage/、docs/benchmarks/ 整改中移出的
> 「旧状态 / 演进过程 / 已修复勘误 / 删除清单」内容（2026-09-27 摘录）。
> 这些内容只作历史存档，不再是这些文档的正文。

---

## CUDA 后端移除备注（原位置 docs/introduction/01-architecture.md 篇末「备注：CUDA 后端已移除」整节）
- 类型：演进记录 / 删除清单
- 内容：CUDA 后端已整体移除——`cuda/` 目录（`cuda_kernels.cu` / `.h` / `CMakeLists.txt`）、`compute_cuda_engine.hpp`、`backend/compute_cuda_backend.hpp`，以及全库 `NN_HAS_CUDA` 条件分支（`compute_tensor.hpp` 6 处、`nn.hpp`、`cli/cli_engine_factory.hpp`、`cli/cli_train_common.hpp`、`CMakeLists.txt`）均已删除；移除原因是融合原语（M4/M5/M6）与 DSL 表达式在 CUDA 上未实现且无真实回退，属「文档声称支持但实际损坏」的死代码；CLI 不再接受 `--cuda`（传入报「未知参数」）；带 CUDA 的历史快照在 git 分支 `legacy/cuda`。当前事实（正文保留）：引擎只支持 `CpuEngine` 与 Vulkan `GpuEngine` 两个后端。

## 旧代数 AST 删除行（原位置 docs/introduction/01-architecture.md L1 模块表）
- 类型：删除清单
- 内容：`~~algebra_expr.hpp~~ / ~~algebra_compute.hpp~~ —— 🗑️ 2026-09 已移除：旧代数 AST 与 `compute::apply`；`Expression`/`BoolExpression` 概念迁入 `expr_dsl.hpp`（另见头文件依赖图括注「旧 algebra_expr / algebra_compute 已于 2026-09 移除」）。

## 引擎算子收敛横幅（原位置 docs/introduction/01-architecture.md L2 表下 blockquote）
- 类型：删除清单
- 内容：**2026-09 收敛**：`axpy_inplace`、`broadcast_row_inplace/col_inplace`、`elementwise_unary/binary/binary_scalar`、`elementwise_select_scalar_cond`、引擎 `row_reduce_max`、`offload_store/load` 与 `UnaryOp/BinaryOp/CompareOp` 枚举已全部删除（逐元素/广播/条件选择一律走表达式 DSL），引擎 virtual 58 → 49；逐项清单见 `todo/12-compute-engine-inventory.md` 顶部收敛横幅。

## IR-C 块式融合移除括注（原位置 docs/introduction/01-architecture.md 表达式入口表 `expr_dsl.hpp` 行）
- 类型：演进记录
- 内容：「（`start_expr/end_expr` 块式融合已于 2026-09-19 随 IR-C 移除）」。

## 「不再有 forward_gpu / backward_gpu」表述（原位置 docs/introduction/01-architecture.md §1 引擎化架构）
- 类型：演进记录
- 内容：原文「Layer 的 forward/backward 只写一次，通过 ComputeEngine 参数自动适配 CPU/GPU。不再有 forward_gpu / backward_gpu。」——已改写为当前事实「每个 Layer 只有一份与后端无关的 forward/backward 实现」。

## SmartPolicy 旧称（原位置 docs/introduction/02-performance.md §1 标题）
- 类型：演进记录
- 内容：标题曾为「自适应并行策略（旧文档称「SmartPolicy」）」。

## 线程池 latch 的旧版问题叙事（原位置 docs/introduction/02-performance.md §2）
- 类型：演进记录
- 内容：「旧版问题」：每次 `submit` 需要构造 `shared_ptr<packaged_task>`（堆分配）、获取 `future`（同步开销）、加锁入队（互斥开销）；「新版优化 / 性能收益」对比：旧版 N 分块 = N 次堆分配 + N 次加锁 + N 个 future 同步，新版 N 分块 = 1 次加锁 + 0 次堆分配 + 1 个原子计数器。正文已改为「设计目标 / 当前实现 / 开销构成」三段当前事实。

## MSVC /fp:fast 早期标志记录（原位置 docs/introduction/02-performance.md §4 编译标志注）
- 类型：已修复勘误
- 内容：原文记「MSVC 分支为 `/O2 /fp:precise`——早期为 `/fp:fast`，但其禁用 NaN 语义与本铁律冲突，实测 `precision_type_test` 的 NaN 断言在 `/fp:fast` 下被编译器折叠为假失败，故对齐为 precise」；同时权威行号由 `CMakeLists.txt:74/76` 更正为 `77/79`。正文保留的理由（`/fp:fast` 与 NaN 铁律冲突）已改为当前事实陈述。

## 已删除 eager 算子的示例代码（原位置 docs/introduction/02-performance.md §5.1）
- 类型：删除清单
- 内容：批量提交示例中的 `engine.elementwise_binary(...); // 不立即执行`——`elementwise_binary` 已不在引擎接口中，示例已改为 `engine.matmul` + `engine.transpose`。

## 单算子级广播/融合原语删除条目（原位置 docs/introduction/02-performance.md §5.3 第二条）
- 类型：删除清单
- 内容：「~~`broadcast_row_inplace` / `broadcast_col_inplace`、`axpy_inplace`、`elementwise_select_scalar_cond`~~：2026-09 已全部删除——单算子级广播/融合被表达式级融合取代（`dsl::row_broadcast` / `dsl::col_broadcast` / `dsl::select` / `dsl::compute_into`）」。

## 注意力批量化的历史写法（原位置 docs/introduction/02-performance.md §8）
- 类型：演进记录
- 内容：小节标题「旧版：per-head 循环」与括注「（P-C2-7 起 forward 进一步收敛为 `eval_expr(fold)` 单 dispatch……）」——正文已改为「朴素写法（不批量化时的形态）/ 当前做法（forward 单 fold kernel + backward 批量 dispatch）」。

## 因果掩码缓存历史设计（原位置 docs/introduction/02-performance.md §9 整节）
- 类型：演进记录 / 删除清单
- 内容：「因果掩码缓存【历史：已随 fold 迁移删除】」全节——状态横幅「P-C2-7，2026-09-23：掩码物化与缓存路径整体删除，现行单 fold kernel 在 body 内以 select 链表达掩码（`tri_skip` 把被屏蔽块整块钳成空转），掩码矩阵从不存在、无需缓存；示例中 `(batch << 16) | seq_len` 位打包键正是 `08-pitfalls-and-lessons.md` §3.3 记录的溢出缺陷写法，该缺陷随整段删除一并消失」；「问题（历史）」GPT 每个 forward 需创建因果掩码矩阵 `(batch*H*seq, seq)`；「优化（历史）」曾以 `mask_cache_` + `ensure_mask(engine, batch, seq_len)` 按 `(batch, seq_len)` 键缓存掩码矩阵（完整代码见 git 历史 `compute_layer_attention.hpp`）；「收益（历史）」相同 `(batch, seq_len)` 只构造一次掩码。正文现只保留当前事实「掩码不物化、无缓存」。

## 旧代数 AST 括注（原位置 docs/introduction/02-performance.md §12 标题）
- 类型：删除清单
- 内容：标题「表达式模板（DSL；旧代数 AST 已于 2026-09 移除）」。

## 优化效果总结表中的演进条目（原位置 docs/introduction/02-performance.md 末表）
- 类型：演进记录
- 内容：「掩码/编码缓存 | GPT 训练 | 避免重复构造（掩码物化缓存已随 fold 删除，见 §9）」与「GPU 注意力 fold 流式（2026-09-24） | GPU mha/causal | fwd 7.9→5.41ms（−31.5%）、causal −40%、train −13%（40HX 交错 bench）」——正文现为「位置编码缓存（掩码不物化、无需缓存，见 §9）」与绝对值「fwd 5.41ms（mha）/ 4.75ms（causal）、train 17.5ms」。

## dsl::compute 缺 rows/cols 的示例（原位置 docs/introduction/03-algorithm-reference.md §2 ReLU）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：原文为 `dsl::compute(engine, dsl::max(dsl::leaf(x), 0))` 与 `dsl::compute(engine, dsl::select(dsl::leaf(x) > 0, dsl::leaf(grad_out), 0))`（2 参重载不存在）。核对结果：`dsl::compute(ComputeEngine&, const E&, std::size_t rows, std::size_t cols, Precision P = F32)`（`expr_dsl.hpp:1318-1320`），必带输出形状。已补 `rows, cols` 并把字面量标量改为 `Scalar{0}`（与 `compute_layer_mlp.hpp:194/209` 的真实写法一致）。

## LayerNorm 典型用途归约方向（原位置 docs/usage/03-compute-engine-usage.md 归约节「典型用途」）
- 类型：已修复勘误（与代码不符，已订正）
- 内容：原文「LayerNorm: 归一化每行 → `row_reduce_sum(*x)` + `1.0f / x->cols()`」。核对结果：LayerNorm 沿特征维（行）求均值、每列一个均值，真实实现用 `col_reduce_sum` 且除以 `F = rows`（`compute_layer_mlp.hpp:492-511`）。已改为 `col_reduce_sum(*x)` + `1.0f / x->rows()`。

## Linear 实现的旧写法（原位置 docs/introduction/03-algorithm-reference.md §1）
- 类型：演进记录（示例与代码不符，已订正）
- 内容：forward 曾记为「一行原语：`out = engine.matmul_with_bias(W, x, b)`」；backward 曾记为「`grad_input = engine.matmul(W, grad_out, /*transA=*/true)`；`grad_W ← dsl::compute_into(leaf(grad_W) + matmul(grad_out, x, transB))`；`grad_b += engine.accumulate(engine.row_reduce_sum(grad_out))`」。当前代码（`compute_layer_mlp.hpp:99-166`）为：forward = `dsl::compute(engine, dsl::matmul(w_, input) + dsl::row_broadcast(b_), rows, cols, p_.compute)`；backward = `dsl::compute(dsl::matmul(w_, grad_output, true, false))` + `dsl::compute_into(leaf(grad_w_) + dsl::matmul(grad_output, input_cache_, false, true))` + `dsl::compute_reduce(row_reduce_sum)` 后 `compute_into` 累加。

## MHA 历史路径括注（原位置 docs/introduction/03-algorithm-reference.md §6）
- 类型：演进记录
- 内容：「（历史路径为 `rearrange_3d` + `batched_matmul` 把 H 个 per-head matmul 融合为 1 次 batch dispatch，该结构现仅存于 backward。）」与小节标签「Forward（P-C2-7 单 fold 路径……）」「S7 起折进 Q 免独立 pass」——正文已改为当前事实（backward 使用该批量结构；缩放随表达式融合）。

## 掩码缓存删除括注（原位置 docs/introduction/03-algorithm-reference.md §12 CausalSelfAttention）
- 类型：删除清单
- 内容：「——没有 `S += mask` 矩阵，**也没有掩码缓存**（历史物化式掩码缓存已随 fold 迁移删除）」。

## 优化器篇 eager 原语删除横幅（原位置 docs/introduction/03-algorithm-reference.md Optimizer 篇开头）
- 类型：删除清单
- 内容：「不再直调 `axpy_inplace`/`elementwise_*` 这类 eager 原语（这些原语已于 2026-09 整体删除，引擎现有 49 个 virtual）」。

## Muon 旧算法描述与旧 DSL 写法（原位置 docs/introduction/03-algorithm-reference.md §5）
- 类型：演进记录（示例与代码不符，已订正）
- 内容：原文步骤为「1. SGD-Momentum: v ← μ·v + g；2. update = NS_5(v)；3. p ← p − η·update」，DSL 用 `engine.matmul(A, A)`、`compute_into(leaf(A)*rparam(b)+...)`（缺 engine/dst 实参）。当前代码（`compute_optimizer.hpp:658-717`）含 Nesterov 方向（`update = g + μ·v`）、形状缩放 `p ← p − η·0.2·√max(m,n)·NS(update)`（NorMuon），且 NS 母矩阵取短边（m≤n 行正交化 / m>n 列正交化）；示例已按 `dsl::compute` / `dsl::compute_into` 完整签名订正。

## 总览表演进措辞（原位置 docs/introduction/04-innovative-designs.md §0）
- 类型：演进记录
- 内容：「不物化 `O(seq²)` 分数矩阵（原两趟式已演进）」「把显存峰值从 29GB 压到 27GB（并持续下探）」——正文现为「不物化 O(seq²) 分数矩阵」与「GPT 训练峰值显存压至 ~27GB 量级」。

## CUDA 停用括注（原位置 docs/introduction/04-innovative-designs.md §1.1）
- 类型：演进记录
- 内容：「自动适配 CPU / GPU 双后端（CUDA 已停用）」。

## key 确定性铁律的事故叙事（原位置 docs/introduction/04-innovative-designs.md §2.3）
- 类型：已修复勘误
- 内容：「这条铁律来自一次真实事故：C++ 实参求值顺序未指定曾导致 key 跨编译器不稳定。」——正文改为当前警告「C++ 实参求值顺序未指定是这类漂移的常见来源，属必须规避的坑」。

## 三个 matmul 融合原语整节（原位置 docs/introduction/04-innovative-designs.md §3.3）
- 类型：删除清单
- 内容：「三个 matmul 融合原语【历史：已随 S7 删除】」全节——为承载当年两趟注意力而新增的通用原语：`batched_matmul_reduce`（matmul 后沿输出维度归约，不物化 `A·B`）、`batched_matmul_softmax_denom`（减行 max → exp → 按列求和）、`batched_matmul_softmax_apply`（行 softmax 归一化后与 V 相乘累加、逐 tile 流式）；反向曾有 `..._softmax_backward_q` / `..._softmax_backward_kv`（kernel 内重算权重矩阵）。正文 §3.3 已改写为当前事实「matmul 参与表达式融合（`dsl::matmul` 折叠为 ExprSpec 前置 matmul 段）」。

## 原两趟式注意力方案整节（原位置 docs/introduction/04-innovative-designs.md §4.3 与 §4.4 前半）
- 类型：演进记录 / 删除清单
- 内容：「【历史】原两趟式方案（已删除）」——Forward：`m = max of QᵀK`（bmm_reduce）→ `l = Σ exp(QᵀK − m)`（bmm_denom）→ `O = W·V` 逐 tile（bmm_apply），三个原语已随 S7 删除；Backward：反向重算 W（`bmm_softmax_backward_q/kv`，已删）；§4.4 旧句「fold 化后连原两趟式『用计算换显存』的 2× QKᵀ 代价也消失——forward 的 QKᵀ 只算一遍」；以及 §4 标题「（fold 单遍流式；原两趟式）」与 §4.2「现行方案（P-C2-7，2026-09-23）」标签。

## 稀疏 CE 旧原语括注（原位置 docs/introduction/04-innovative-designs.md §5.2）
- 类型：删除清单
- 内容：「原为两个 op-level 原语（`col_softmax_denom` / `col_softmax_sparse_forward`，**已随 S7 删除**），现行由 IR 结构表达……」。

## IR 优化动机的「无 pass 时代」叙述（原位置 docs/introduction/04-innovative-designs.md §6.1）
- 类型：演进记录
- 内容：「`ExprSpec` 已是事实上的轻量 IR……但没有优化 pass，直接导致真实问题：子表达式 `grad*gamma` 重复 3 次 → 超 `EXPR_MAX_INPUTS=8`，**被迫**手工拆表达式」——正文改为当前事实（不做优化会带来这些问题，IR-A/IR-B pass 解决）。

## IR-C 移除条目（原位置 docs/introduction/04-innovative-designs.md §6.2）
- 类型：演进记录
- 内容：「**IR-C 图 IR + 融合分析**：~~`begin_expr/end_expr` 录制虚拟寄存器 DAG；逐元素链拼接~~ → **已评估并整体移除（2026-09-19）**……」——正文改写为「不采用」+ 指向 `docs/todo/03-ir-optimization.md` §5.3 的当前表述。

## f32-only 红线修订记录（原位置 docs/introduction/04-innovative-designs.md §7「设计红线（立项时基线，v1.2.0 起修订）」）
- 类型：演进记录
- 内容：「~~全程**不引入 f16/bf16**（数值统一 fp32）~~——**已修订**：v1.2.0 起引入 f16 混合精度（`Precision` 类型系统 + `PrecisionProfile`，compute/stable 档可保 fp32；BF16 仍为『使用即报错』的保留值）。fp32-only 是立项时的取舍记录，不再是现行红线。」正文现为「精度红线（当前）」段。

## 自适应并行与线程池的旧版对比（原位置 docs/introduction/04-innovative-designs.md §8.1 / §8.2）
- 类型：演进记录
- 内容：「实测在 ~512K 元素处首次稳定 >1.5x，**此前串行更优**」；「**旧版**每任务 = N 次堆分配 + N 次加锁 + N 个 future 同步；**新版** = 1 次加锁 + 0 次堆分配 + 1 个原子计数器」。

## 算子收敛与掩码缓存删除条目（原位置 docs/introduction/04-innovative-designs.md §8.4）
- 类型：删除清单
- 内容：「单算子级融合原语（`axpy_inplace` = clone+scale+add 三步并一步、`elementwise_select_scalar_cond` = 条件选择）——**2026-09 已随算子收敛整体删除**」；「多头注意力批量化……（历史：`rearrange_3d → 单次 batched_matmul → 转回` 把 H 次融为 1 次，该结构现仅存于 backward）」；「因果掩码物化缓存**已随 fold 迁移删除**……位置编码缓存保留」；以及 §4.2 的「`tri_skip`（原名 `causal_skip`）」改名记录。

## 收益一览表的演进条目（原位置 docs/introduction/04-innovative-designs.md §12）
- 类型：演进记录
- 内容：「fold 流式注意力（**原两趟**）」「算子融合 + 显存体系 | GPT 训练峰值 **~29GB → ~27GB**（并持续下探）」「注意力批量化 | H 次 matmul → 1 次 batched_matmul（**现为** fold 单 kernel 单 dispatch）」「融合 axpy | 每 step **减少** ~600 次 GPU buffer 分配」。

## add_linear / add_relu 链式构建示例（原位置 docs/usage/01-quickstart-model.md 三件套、方式一、方式三、完整前向示例、陷阱 #2；docs/introduction/01-architecture.md 理解路线图第四步）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：原文用 `model.add_linear(784, 256).add_relu().add_linear(256, 128)` 链式构建，并称「方式一：链式构建（推荐）」「方式三：模板 add（自定义层）——`add<nn::Linear>` 等价于 `add_linear`」。核对结果：`Model` 中**不存在** `add_linear` / `add_relu`（全仓 grep 仅命中 AGENTS.md 与文档）；唯一构建入口是 `template <typename LayerType, typename... Args> Result<void> add(Args&&...)`（`model_container.hpp:71-80`），返回 `Result<void>`、**不返回 `*this`，不可点链调用**，构造后自动 `Layer::init(engine)`。示例已全部订正为 `model.add<nn::Linear>(784, 256);` 逐行写法。

## 自定义 Layer 旧示例签名（原位置 docs/usage/01-quickstart-model.md「自定义 Layer」）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：原文为「构造函数 `MyLayer(ComputeEngine& engine, in_dim, out_dim)` 内初始化权重」+ `nn::Matrix w_cpu = nn::Matrix::random(out_dim, in_dim); // 假设有此方法` + `std::vector<nn::Tensor*> parameters() override { return {&weight_}; }`。核对结果：`Layer::parameters()` / `param_gradients()` 返回 `std::vector<TensorRef>`（`TensorRef = std::reference_wrapper<Tensor>`，`compute_tensor.hpp:338`）；`Matrix::random` **不存在**；权重初始化契约是 override `init(ComputeEngine&)`（`Linear` 即如此，`compute_layer_mlp.hpp:46-76`）。示例已按当前契约重写。

## 陷阱 #3「to_matrix 错误用法」（原位置 docs/usage/01-quickstart-model.md 常见陷阱）
- 类型：已修复勘误（原描述与代码不符）
- 内容：原文「❌ GPU 张量调用 `to_matrix` → 错误：tensor is not GPU；✅ 先下载到 CPU → `auto m = engine.to_matrix(t)`（错误信息会告诉你）」——实际 `to_matrix(const Tensor&, Precision)` 就是统一下载入口（`compute_engine.hpp:177`），不存在该报错；真正的跨设备错误是绕过 engine 直接取 `tensor.cpu_matrix()`（`NN_ASSERT: tensor has no CPU storage`）。示例已订正。

## 陷阱 #4 旧调用签名（原位置 docs/usage/01-quickstart-model.md 常见陷阱）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：原文为 `model.backward(engine, grad)` 与 `model.zero_grad(engine)`、`optimizer.step()`（非指针）。核对结果：`Model::forward/backward/zero_grad` 均**不带 engine 参数**（`model_container.hpp:178/196/249`），`Optimizer::step/zero_grad` 也不带（`compute_optimizer.hpp:117/182`）；`Loss::forward` 需要 engine。

## 训练流程时序图中的 engine 实参（原位置 docs/introduction/01-architecture.md「完整训练流程」mermaid 图）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：原图写 `U->>M: zero_grad(engine)`、`U->>M: backward(engine, grad)`、`U->>M: forward(engine, test_input)`、`O->>E: scale_inplace / add_inplace / ...`。核对结果：`Model::zero_grad()` / `Model::forward(const Tensor&)` / `Model::backward(const Tensor&)` 均不带 engine（`model_container.hpp:178/196/249`，engine 在 Model 构造时绑定；只有下推到 `Layer::forward(engine, tensor)` 时才带）；优化器步进实为 `dsl::compute_into(engine_, ...)`。已按当前签名订正（Mermaid 结构保留）。

## mnist_infer 的 --image / --interactive 示例（原位置 docs/usage/02-quickstart-train-infer.md MNIST 推理）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：原文「`./build/mnist_infer --model mnist_model.bin --image datasets/...`」「`./build/mnist_infer --interactive`」。核对结果（`src/mnist_infer.cpp:34-113`）：用法是 `mnist_infer <image.csv|目录> [选项]`（输入为**位置参数**），选项只有 `--model/--topk/--show-pixels/--gpu/--help`；`--image` 与 `--interactive` 均不存在（`--interactive` 只属于 `text_infer` / `tokenizer_infer`）。示例已订正。

## 训练 C++ 示例中的 engine 实参（原位置 docs/usage/02-quickstart-train-infer.md 两个训练示例）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：`model.backward(engine, *grad);` ×2 → `model.backward(*grad);`；`nn::save_model("gpt_model.bin", model, model.spec(), tokenizer_json)` → `*model.spec()`（`Model::spec()` 返回 `const std::optional<ModelSpec>&`，`model_container.hpp:115`）。

## GUI 步骤中的「CUDA 已停用」（原位置 docs/usage/02-quickstart-train-infer.md MNIST 训练 Tab / GPT 推理 Tab 可选项）
- 类型：演进记录
- 内容：「✅ GPU 加速 — 支持 Vulkan（下拉选择；CUDA 已停用）」两处——已改为「支持 Vulkan（下拉选择设备）」。

## 模型保存格式旧表述（原位置 docs/usage/02-quickstart-train-infer.md「模型保存格式」）
- 类型：已修复勘误（与代码不符，已订正）
- 内容：原文「V4 = 当前自描述格式 / V1-V3 = 已移除支持 / `save_model` 统一写入 v4」。核对结果：`MODEL_VERSION = 5`（v5：per-tensor precision tag，`model_serialization.hpp:53/151/500`），`save_model` 写 v5、`load_model` 接受 v≥4、v1–v3 在 header 校验即拒绝（报「旧格式…请用当前版本重新训练/保存」）。正文已改为 v5/v4/v1-v3 三行的当前事实。

## 引擎使用文档中的已删 API 段落（原位置 docs/usage/03-compute-engine-usage.md 多处）
- 类型：删除清单（示例与代码不符，已订正）
- 内容：
  - 「### 融合 axpy（2026-09 已移除该原语）」——`engine.axpy_inplace` 已删除；
  - 归约节「按行求最大值：引擎算子已于 2026-09 删除」（引擎 `row_reduce_max` 不存在，`dsl::row_reduce_max` 是 DSL 叶子）；
  - 「### 广播操作（2026-09 已移除两个原语）」——`broadcast_row_inplace` / `broadcast_col_inplace` 已删除；
  - 逐元素四小节标题「（2026-09：原语已删除，改用表达式）」×4；
  - 「### ~~块式融合~~（已移除）」整节——`engine.begin_expr()` / `engine.end_expr()` 已随 IR-C 删除（2026-09-19）；
  - 性能建议 1 的 ❌ 示例 `engine.elementwise_unary(UnaryOp::Exp, *x)` / `engine.elementwise_binary(BinaryOp::Add, ...)`（`UnaryOp`/`BinaryOp` 枚举不存在）；
  - 示例 3 的「旧式多原语逐步写」对照块 `engine.broadcast_col_inplace(*centered, *mean, BinaryOp::Sub)`（API 不存在）；
  - 逐元素节导语「单步调试/小工具用下面的 eager 原语即可」（eager 逐元素原语不存在）；
  - 页脚「*最后更新：2026-09-25（DSL 签名与示例对齐当前代码…）*」。

## 示例 1 / 示例 3 / 示例 4 的旧实现口径（原位置 docs/usage/03-compute-engine-usage.md）
- 类型：演进记录（示例与代码不符，已订正）
- 内容：示例 1 曾记 Linear forward = `engine.matmul_with_bias(...)`（现为 DSL 融合表达式，`compute_layer_mlp.hpp:112-114`）；示例 3 曾记 LayerNorm「拆成 4 条 DSL 表达式」（现为 7 步：mean_raw→mean→diff→var_raw→std_inv→normalized→out，`compute_layer_mlp.hpp:486-544`）；示例 4 横幅「**真实实现已演进**：注意力 forward 现为单 fold kernel……不再是物化链」（改为当前事实陈述）。

## 训练包文档中的 CUDA 状态句（原位置 docs/usage/04-train-package.md 配置模板注释与 train 命令注释）
- 类型：演进记录
- 内容：`"device": "cpu", // cpu | gpu（CUDA 已停用；train 时可用 --device 覆盖）`；`python train_pkg.py train … --device gpu # 按设备覆盖（CUDA 已停用）`。当前事实（正文新增）：`--device` 取值 `cpu`/`gpu`，CLI 同时接受 `cuda` 但引擎无 CUDA 后端、该值不启用 GPU（`train_pkg.py:525` choices、`cli_controllers.py` 只处理 `gpu`）。

## 基准报告中的「旧→新」演进叙述（原位置 docs/benchmarks/2026-09-25-vulkan-vs-cuda.md）
- 类型：演进记录（已改写为报告口径；**数据全部保留**）
- 内容（原文措辞 → 现口径措辞）：
  - 头部「**2026-09-26 更新**：nn 侧已在 HEAD `4a12876` 上全量重跑；torch 侧沿用 09-25 数据（**本轮未动 torch**）」→「数据版本：nn 侧采集于 HEAD `4a12876`（2026-09-26 会话）；torch 侧采集于 2026-09-25」。
  - §0「nn 快 2.3×（**09-25 的 3.7× 系 P0 虚报**，见 §3.3‡）」→「（本表口径，见 §3.3‡）」。
  - §0「**f32 5 epoch 健康收敛（09-25 的 NaN 发散已修复）**」→「（09-26 采集；09-25 采集为 NaN 发散，见 §2.5）」。
  - §0 发现③「f32 显存 **5743→3069 MiB（−47%）**，09-25『峰值与精度无关、由池粒度主导』的观察**已不成立**」→「3069 MiB（09-25 采集 5743，差 −47%）——峰值强相关于精度，见 §2.3」。
  - §2.1「torch 吞吐 2.13×（**09-25 为 2.44×——nn f32 提速 13%，为 09-25 之后的优化收益**）」→「（按 09-25 采集的 116.3 ms/step 计为 2.44×；本表口径 101.0）」；「09-25 **旧值**」→「09-25 采集值」。
  - §2.2「09-25 同类长测 epoch 5 发散 -nan 的问题**已不复现**」→「09-25 采集的同类长测在 epoch 5 出现 -nan（见 §2.5）」。
  - §2.3「**09-25『峰值 ~5.7 GB 与精度无关、由 Vulkan 内存池粒度主导』的观察不再成立**」→「09-25 采集值为 f32 5743 / f16 5772（两者接近）；差值 f32 −47%、f16 短测 +9%」。
  - §2.5 标题「（**09-26 重写**；如实记录）」→「（按采集日期如实记录）」；「在 f32 上**已不复现**（修复落在 09-25 之后的提交窗口，未逐个二分）」→「09-25 采集记录为……；两者的差异归于 09-25 之后的提交窗口，未逐个二分」；「nn 四字段全 f16 的**历史**：09-25 之前的行为是……09-26 **变为**……症状**由 NaN 变为冻结**，风险等级上升」→「缺陷表现（按采集日期）」；「optimizer=f16 语义（**09-26 修正，替换 09-25 旧论断**）……**现已过时**」→「（本报告口径，以代码为准）……09-25 采集条目与当前代码不符，以本条为准」。
  - §3 口径「nn 列为 2026-09-26 **重跑**」「GB/s …（归约 = N + 输出，**不虚记 2N**）」→「nn 列采集于 2026-09-26」「（归约 = N + 输出）精确字节计」。
  - §3 脚注 †「已随 2026-09-26 技术债清理……从 layer_bench 算子表移除，**本轮无法复现，保留 09-25 旧值**（nn 侧标注为历史数据）」→「本行数值为 09-25 采集口径；当前算子表不含这两项……沿用 09-25 读数、不再复现」（数值 0.145/0.204/… 全保留）。
  - §3 脚注 ‡「transpose 行为 09-26 **修复后首次正确测量**：09-25 报告的是 P0 缺陷 `200955f`……**虚报值**……优势**由 3.7× 修正为 2.3×**」→「transpose 记账口径：本表数值按『派发网格与 shader 行界契约同源』口径采集（缺陷记录 `200955f`）；09-25 采集的 4096² 0.342 / 8192² 0.940 ms 为契约不同源状态下的读数……按本表口径优势 = 2.3×（09-25 采集口径为 3.7×）」（全部数值保留）。
  - §3 观察 3「09-25 报告的 3.7×/2× 系 P0 静默半写虚报，**本轮为修复后正确值**」→「口径与早先采集值（3.7×/2×）的差异见 ‡」；观察 5「（**此前** 2N 口径会虚高 ~2×）」→「按 N+out 精确字节口径」。
  - §4 观察 3 标题「（**09-26 更新**）」与「09-26 的 e2e **较 09-25 提速 13%**」→「按两个采集日的差异（在 ±15% 会话漂移内）」。
  - §5.3「（**09-26 修正**）……09-25 版本此处记……**已随代码演进过时**」→「（本报告口径，以代码为准）……09-25 采集条目与当前代码不符，以本条为准」。
  - §5.10「**09-26 重跑口径**……为真实变化」→「**采集批次口径**……反映真实差异」。
  - §6 产物索引与复现命令注释「09-26 **新增** / **重跑**」「09-26 **起**主对比」「torch 训练（09-25 数据，**本轮未重跑**）」→「09-26 采集 / 主对比 / 09-25 采集口径」。
- 说明：报告的日期化结构、全部表格与实测数字（101.0 / 47.5 ms、3069/5743 MiB、0.342/0.940 ms、3.7×/2.3× 等）**一条未删**。

---

# 构建系统 / CI（CMakeLists.txt、.github/workflows）

## 干净构建三缺陷：注册表常驻掩盖了编译失败与依赖环（2026-10-02，`f093261`）

- 类型：bug 根因 / 教训
- 内容：v1.6.0 发布前首次把仓库放进**注册表缺失的干净构建目录**编译，立即暴露三缺陷：
  ① `compute_vk_backend.hpp` 直接使用 `expr_spec.hpp` 的 6 个符号却只靠生成的 `fused_registry.hpp` 传递带入——注册表缺失时 **clang 与 MSVC 在完全相同的符号列表上报错**（`ExprPrecSig`/`EXPR_PREC_DISPATCH_SUFFIX`/`EXPR_MAX_INPUTS`/`EXPR_MAX_CONSTS`/`EXPR_MATMUL_BLOCK`/`EXPR_FOLD_ROWS_PER_WG`），显式 include 修复；
  ② `nn_enable_gpu_fusion` 收集器继承了目标 `INCLUDE_DIRECTORIES`（含其自身产物目录）→ 注册表存在时 depfile 记录 `fused_registry.hpp → collect.exe → .obj → 该头` = **ninja 依赖环**、图加载失败 → 立不变量 `NN_EXPR_SCAN ⇒ 不嵌入 fused_registry`（收集器是生成者、绝不消费自己的产物）；
  ③ `compute_gpu_engine.hpp` 的 `psig` 声明在 `#ifdef NN_FUSED_REGISTRY_EMBEDDED` 内、使用在无 ifdef 处 + 连带 unused-parameter `-Werror` → 改无条件计算（无注册表时 NVI 抬 f32 → 恒 0 → 正确落闭合世界错误）。
- 为什么此前没暴露：开发机 `build/generated/fused_registry.hpp` 长期常驻，M1–M6 与融合分支的全部验收都在**脏构建目录**上进行；app 目标靠 `add_dependencies(compile_gpu_shaders)` 兜底，只有收集器两头都不占。
- 教训：**验收口径必须含一次干净构建**；两个编译器报出完全相同的符号列表 = 根因在源码、不在编译器（本轮 MSVC 首跑即被误判为“MSVC 特有问题”的反例）。
- 验收（v1.6.0）：clang 203/203 零告警 + ctest 21/21；MSVC 0 error + ctest 21/21；ASan+UBSan ctest 21/21；二次增量构建无依赖环、收集器 depfile 干净；结构数 84/86 与样例值全部不变。

## 收集器依赖环复发：`#if __has_include(...) && !defined(NN_EXPR_SCAN)` 在 clang 21 下防不住 depfile（2026-10-03）

- 类型：bug 根因（f093261 ② 的守卫形式不充分）
- 内容：BatchNorm 落地后的增量重建中 ninja 报 `fused_registry.hpp → collect → .obj → fused_registry.hpp` 依赖环、**图加载即失败**（任何构建目标都拒绝执行）。根因：f093261 的守卫把两个条件写在**同一行 `#if`**（`__has_include("fused_registry.hpp") && !defined(NN_EXPR_SCAN)`），而 **clang 21 对 `#if` 条件里的 `__has_include` 无论是否被短路求值、都记进 `-MD` depfile**（实测：原序与反序 `!defined && __has_include` 都记录；收集器 NNF_OUT_DIR 在 include 首位 → 命中自身产物）。修复 = 守卫改**嵌套条件编译**：`#ifndef NN_EXPR_SCAN` 包住 `#if __has_include(...)` + include——被跳过的组不出词法，0 探测记录（实测对照：NN_EXPR_SCAN 下 depfile 0 条 fused_registry、普通 TU 仍 1 条、行为不变）。
- 触发条件：收集器 TU 在**自身产物注册表已存在**时重编译（改 `tools/scan_exprs.cpp` 或其头链即触发）→ 依赖记录入库 → 下次 ninja 图加载报环；首次干净构建不会触发（当时注册表尚不存在），所以 f093261 当时的“二次增量构建无依赖环”验证漏掉了这一形态。
- 修复位置：`include/neuralnet.cpp/backend/compute_vk_backend.hpp` 的 fused_registry 嵌入守卫（注释同步记录“为何不能用 && 同行守卫”）。
- 验收：`ninja -t deps` 中 `fusion_custom_layer_example_nnfusion_collect` 的 TU 不再出现 `nn_fusion/.../fused_registry.hpp`（应用 TU 对注册表的合法持有不变）；全量干净重建 + ctest 23 用例（21 过 + 2 个 Mali 已知非确定 offload 用例，`NN_VULKAN_DEVICE=1` Lavapipe 下全过）。

## 构建提速：关模块扫描 / 共享 PCH / glslc 并行 + 注册表内容稳定写入（2026-10-04）

- 类型：性能 A/B / 演进记录
- 背景：Debug + clang + Ninja + `NN_ENABLE_TESTS=ON` 下全量净构建 **125.5s**，单 TU **17.5s**，改任意公共头即接近全量重编。
- 实测诊断（`.ninja_log` 逐边计时 + 手工 A/B）：
  - 单 TU 编译 17.5s（`-O0`，成本在前端 parse 而非 codegen）；
  - CMake 为**每个** TU 追加一次 `clang-scan-deps` 模块依赖扫描 ~2.1s（`*.obj.ddi` / `CXX.dd` / `*.obj.modmap` 三条边），而本项目零 `import`/`export module`；
  - `scan_exprs` 生成阶段 **55.7s ≈ 221 次串行 glslc**（91 结构 + 130 精度变体，单次 ~250ms）；
  - `fusion_custom_layer_example` 的收集器另跑一遍 ~279 次 glslc（60–77s）；
  - `fused_registry.hpp`（4.7MB）**无条件 ofstream 覆写** → 每次重跑都改 mtime → 下游全部 TU 重编（该自定义命令本来就带 `restat = 1`，白白浪费）。
- 整改（四处）：
  1. `CMakeLists.txt`：`set(CMAKE_CXX_SCAN_FOR_MODULES OFF)`——零模块 → 扫描纯浪费，`.ddi`/`.modmap` 边整体消失。
  2. `CMakeLists.txt`：`NN_ENABLE_PCH`（默认 ON）——`target_precompile_headers` 把 `nn.hpp` 预编译一次，31 个编译条件相同的目标 `REUSE_FROM nn_pch` **共享 1 份**（而非 31 份 × 79MB / ~780s CPU）；收集器目标**不套** PCH（`scan_exprs.cpp` 在源文件内、include 之前自行 `#define NN_EXPR_SCAN`，PCH 会先于该宏展开 `nn.hpp` → DSL 收集钩子失效、注册表空）。
  3. `tools/fused_generate.hpp`：glslc 改线程池并行（默认 `hardware_concurrency`，`NN_SCAN_JOBS=<n>` 覆盖，`=1` 即旧串行）；注册表改 `write_if_different` 内容稳定写入。
  4. `CMakeLists.txt`：编译器缓存探测扩到 ccache/**sccache**（Windows 上 ccache 常缺席）+ `NN_COMPILER_LAUNCHER` 覆盖；调用方已给 `-DCMAKE_CXX_COMPILER_LAUNCHER`（CI 即如此）时不再叠 `RULE_LAUNCH_COMPILE`，避免 `ccache ccache clang++`。
- **⚠ 共享 PCH 必须 `add_dependencies(nn_pch compile_gpu_shaders)`**（本轮唯一踩到的坑）：`nn.hpp` 里 36 个 `#if __has_include("..._spv.hpp")` 与融合注册表嵌入，是在**编 PCH 那一刻**求值的——早于生成完成就整块落空，且因为"没 include 就没 depfile 记录"**永远不会自愈**。本轮症状：6 个 GPU 用例运行期报 `matmul SPIR-V bytecode not embedded`（脏目录上重建一次就看不出来）。逐目标 PCH 天然带 `cmake_object_order_depends_target_*`，共享 PCH 没有，必须显式补。
- 实测（本机 Debug / clang 23 / Xeon E5-2697A v4 · 32 线程 / 64GB）：全量净构建 **125.5s → 72.7s（-42%）**；改单个源文件 **43.9s → 10.4s**；`scan_exprs` **55.7s → 6.4s**（`NN_SCAN_JOBS=1` 仍 55.9s，证明并行是唯一变量）；无改动构建 0.1s（restat 生效）。
- 字节/行为零变化验收：`fused_registry.hpp` SHA256 与整改前**逐字节相同**（`7445E642DB8CEF6833D9DA6796AD8E5EA44D3394F31AFB063F325DFCC59960EF`，注意写回仍是**文本模式**以免把生成头从 CRLF 变 LF）；ctest **23/23** 全过（含 6 个 GPU 用例）；`-DNN_ENABLE_PCH=OFF` 与 `-DNN_ENABLE_TESTS=OFF` 两种配置均配置+构建通过。
- 教训：`__has_include` 在 PCH 里是**一次性**求值；生成物参与 PCH 时，PCH 的编译边必须显式依赖生成边，否则 depfile 里没有记录、构建系统不会自愈。

## 作为库被消费（P0-4 嵌入路径，2026-10-04）

- 类型：演进记录 / bug 根因（roadmap E2「无 install/export、库无法被下游消费」的**嵌入路径**部分；install/find_package 与 CMakePresets 本轮按用户裁定**不做**、留 P0-4 剩余部分）
- 背景：用户诉求 = 「把库放进 3rd_party/，CMake 两行 + 一个 include 即可用，且能随意控制 NN_HAS_VULKAN 这类参数」。改造前实测缺口：① 无门面目标（消费方要自己拼 include 目录/线程/标准）；② 根 CMakeLists 用 `CMAKE_SOURCE_DIR`/`CMAKE_BINARY_DIR` 共 ~30 处——嵌入时全部指到**消费方**工程根（路径全错）；③ 无条件创建 9 个 app/探针目标 + `CMAKE_BUILD_TYPE` FORCE + ccache `RULE_LAUNCH_*`（GLOBAL 属性）泄漏进消费方；④ `nn_enable_gpu_fusion` 函数体在**调用方作用域**求值，`SCAN_SRC`/`GLSLC`/`CMAKE_SOURCE_DIR` 等在树外作用域全错——树外工程根本无法调用（AGENTS §7 已注明的缺口）。
- 整改（`CMakeLists.txt` 重构 + 新增 `examples/downstream/`、`docs/usage/05-consume-as-library.md`）：
  1. **门面目标 `neuralnet::nn`**（ALIAS `neuralnet_cpp`）：`add_subdirectory` 后一行 `target_link_libraries(app PRIVATE neuralnet::nn)` 即拿到 include/`target_compile_features`（**实测 CMake 4.3：MSVC cl 不识别 `cxx_std_26`**（报 "not known to CXX compiler"）、其 `cxx_std_23` 映射即 `/std:c++latest`；Clang/GNU 认 `cxx_std_26` → 按 `CMAKE_CXX_COMPILE_FEATURES` 实测列表分流）/Threads/stacktrace；GPU 时再补 `NN_HAS_VULKAN` + `generated/` 嵌入头目录 + `Vulkan::Vulkan` + `add_dependencies(compile_gpu_shaders)`（**INTERFACE 库的 add_dependencies 实测挂到消费方目标的编译边** `cmake_object_order_depends_target_*`，保证 SPIR-V 头先于消费方 TU 编译）。
  2. **`NN_BUILD_APPS`（默认 = `PROJECT_IS_TOP_LEVEL`）**：嵌入时默认 OFF，消费方不再被迫编 9 个 app；`f16_writeback_probe` 因双身份（app + ctest）按「任一开关」单独创建。
  3. **`NN_ENABLE_GPU=AUTO|ON|OFF`**（默认 AUTO = 历史行为）：ON = 缺 Vulkan/glslc 配置期 fail-fast（含 glslc 单独校验）；OFF = 完全跳过探测/shader/`scan_exprs`（纯 CPU 最快路径）。控制权交给消费方（用户原始诉求）。
  4. **路径全改 `CMAKE_CURRENT_*` + 函数作用域修复**：`nn_enable_gpu_fusion` 体内改用 `NN_LIB_SOURCE_DIR`/`NN_LIB_BINARY_DIR`（`CACHE INTERNAL`，每次 configure 刷新、全局可见）——因为**函数体在调用方作用域求值**，`CMAKE_SOURCE_DIR` 在树外 = 消费方根（实测）。GPU 不可用时定义 **stub 函数**给指向根因的 FATAL_ERROR（否则是难定位的 "Unknown CMake command"）。
  5. **泄漏收口**：`CMAKE_BUILD_TYPE` FORCE、ccache `RULE_LAUNCH_*`（GLOBAL）自动探测均仅顶层工程生效；GNU stacktrace 探测失败的 FATAL_ERROR 降级为警告（嵌入方编头文件用不到它——**全仓只有 `tools/scan_exprs.cpp` 用 `std::stacktrace`**，grep 确认）。
  6. **消费方编译选项随门面传递**（实测缺口：下游 Release 构建库头文件出 6 个 `-Wpass-failed` 警告——`#pragma clang loop` 向量化未达成，库内靠 `-Wno-pass-failed` 压住，**消费方开 `-Werror` 即挂**）：Clang 传 `-fexperimental-library` + `-Wno-pass-failed`（`$<COMPILE_LANG_AND_ID:CXX,...>` 限定），MSVC 传 `/utf-8`（库头文件是 UTF-8 + 中文注释）；`-fno-exceptions` **不传**（库头文件不用异常，异常策略留给消费方）。
  7. **收集器改链门面**：`nn_enable_gpu_fusion` 的收集器原链 `nn_core` + 直连 `Vulkan::Vulkan`——树外实测 **imported 目标不可跨目录引用**（`target was not found`，探针复现）；且消费方目标的 `NN_HAS_VULKAN` 经 INTERFACE 传递、`$<TARGET_PROPERTY:...,COMPILE_DEFINITIONS>` **取不到**（只查目标自身属性）→ 收集器会漏编 `#ifdef NN_HAS_VULKAN` 内的层调用点 = 闭合世界硬报错。改链 `neuralnet::nn` 一并解决（imported 目标在**定义侧**解析，探针验证可用）。
- 验收：① 本仓顶层配置+全量构建回归（首次重配后 ninja "no work to do" = 编译命令行零变化）；② 下游样例 **CPU 路径**（`NN_ENABLE_GPU=OFF`）构建+运行过、警告归零；③ 下游 **GPU 路径**（`NN_ENABLE_GPU=ON`）构建（shader 生成 + `scan_exprs` 收集 **91 条 = 库内口径**）+ 运行过（CPU/GPU forward 都 OK）；④ fusion 树外调用（`-DNN_DOWNSTREAM_FUSION=ON`，从消费方 CMakeLists 调 `nn_enable_gpu_fusion`）；⑤ 本仓 `NN_ENABLE_TESTS=ON` 全量 ctest。
- 未做（按用户裁定留 P0-4 剩余）：`install()`/`find_package`/`CMakePresets.json`；`nn_enable_gpu_fusion` 的 `COLLECT <file>` 钩子（25 条 dry-run 独有结构的残余不确定性，见 §7 融合条）。
- 探针实测记录（改造前用 6 个最小 CMake 工程验证的 4 个行为，均影响方案写法）：INTERFACE 库携带 imported target（Threads/Vulkan）可跨目录传递 ✓；INTERFACE 库的 `add_dependencies` 传递到消费方**编译边** ✓；MSVC cl 不识别 `cxx_std_26` ✗（Clang 认）；ALIAS/普通目标跨目录可见 ✓、**imported 目标直连不可见** ✗；目标自身 include 目录恒先于 INTERFACE 目录（决定 `NNF_OUT_DIR` 排序语义）✓；空列表 `foreach` 安全 ✓。

## 编译器与标准库一律默认（2026-09-27，原位置 .github/workflows/cmake-single-platform.yml 原 2/53-60/71-73/182-215/281 行、AGENTS.md:22、src/expr_dsl_test.cpp:13、src/expr_opt_test.cpp:13）

- 类型：bug 根因 / 演进记录
- 内容：Linux 云端构建复现历史链接失败（`tokenizer_train` FAILED，大量 `std::__cxx11::*`/`std::__throw_*`/`std::thread` undefined reference）：链接命令带两处 `-stdlib=libc++`（分处 CMAKE_CXX_FLAGS 与 CMAKE_EXE_LINKER_FLAGS 槽位），而 .o 是 libstdc++ ABI——clang 带 `-stdlib=libc++` 但系统无 libc++ 头时**静默回退 libstdc++ 头**，链接却切到 libc++ → ABI 分叉。演进三段：① 旧 workflow 曾传 `-DCMAKE_CXX_FLAGS/-DCMAKE_EXE_LINKER_FLAGS=-stdlib=libc++`，与 CMakeLists 当时的 `NN_STDLIB`（Linux+Clang 默认 libstdc++，`add_compile_options/add_link_options(-stdlib=...)`）冲突（68565b0 曾统一 libstdc++ 缓解）；② 1b415bf 从 CMakeLists 移除 NN_STDLIB 及「PATH 有 clang++ 即设为默认」块，仓库自此无任何编译器/标准库指定；③ 本轮云端**在干净源码上仍复现** → 残留载体：workflow build 目录缓存（`restore-keys` 跨 sha 恢复旧 `CMakeCache.txt`，configure 不清除未重传的 `-D` 缓存变量，历史 `-stdlib` 因此永久驻留）或调用方环境（`CXXFLAGS`/`LDFLAGS`/旧 configure 命令/旧 tag）。整改（2026-09-27，两轮）：轮1——workflow 删 build 目录缓存步骤（ccache 保留，注释记明原因）、删 `-DCMAKE_CXX_COMPILER/-DCMAKE_C_COMPILER` 与 `env CC/CXX`、删与 CMakeLists 重复的 `-DCMAKE_CXX_STANDARD*`，NN_STDLIB 陈旧注释改写；AGENTS.md:22 撤回「PATH 有 clang++ 会自动优先选」（1b415bf 已删该机制）；src/expr_{dsl,opt}_test.cpp:13 手动编译建议去掉 `-stdlib=libc++`。轮1 后云端回退默认 `/usr/bin/c++`（GCC 15）→ `-Werror=format-truncation` 失败（`expr_spec.hpp:715` 与 `gen_fused.cpp:341` 的 `snprintf` 缓冲区按 `%04x` 最坏 8 位十六进制不够；GCC 独有警告、clang 不报——为何此前 WSL g++ 15.2「全量构建 exit 0」未捕获待查，疑与该环境未开 Vulkan、不编译 tools/ 有关）；轮2——按用户要求**恢复 CI 指定 clang++**（双平台 `-DCMAKE_CXX_COMPILER=clang++`/`-DCMAKE_C_COMPILER=clang` + `env CC/CXX`，名称 `CMake (Clang, C++26)`、release 说明「Clang」）。当前事实：**CI 指定 clang++，CMakeLists 与标准库一律不指定**（标准库跟随编译器默认，编译/链接两侧必须一致）；上述两处 format-truncation 编码在 GCC 本地构建（开 Vulkan 时含 tools/）仍会触发（未修，CI 走 clang 不受影响）；云端若仍复现 -stdlib 问题，先查环境变量与删除陈旧 `build/CMakeCache.txt`。

## ccache × 共享 PCH 不兼容：42 个目标的编译全部不可缓存（2026-10-11）

- 类型：性能 A/B / bug 根因（构建系统）
- 背景：`NN_ENABLE_PCH`（默认 ON）落地后，`ccache -s -v` 把"用了 PCH 的 TU"整类记进
  `Could not use precompiled header`（756 条不可缓存里 125 条是该原因）。而本仓 40+ 个
  app/test 目标的编译**全部**走 PCH ⇒ ccache 对这些编译形同虚设，且不报错、只是慢——
  "装了 ccache 却感觉没加速"的隐藏根因。
- 根因（ccache 手册《Precompiled headers》，Clang 侧三条件必须**同时**满足）：① `sloppiness`
  含 `pch_defines,time_macros`；② 以 `-include-pch` 使用 PCH（CMake 天然满足）；③ **编译带
  `-fno-pch-timestamp`**。缺 ①/③ 即整类不可缓存。
- 整改（`CMakeLists.txt`，三处）：① 探测到 ccache 时用 `cmake -E env` 把
  `CCACHE_SLOPPINESS=pch_defines,time_macros` 钉进 `RULE_LAUNCH_COMPILE` 前缀——**不改用户全局
  ccache 配置**（"换机器/换人就悄悄失效"正是这次暴露的问题）；② `NN_ENABLE_PCH` 路径给 PCH 生成方
  `nn_pch` 与**全部** PCH 使用方加 `-Xclang -fno-pch-timestamp`（cc1 开关，驱动不认该拼写；ccache
  是逐 token 匹配字面量，`-Xclang` 形式同样满足——实测命中）；③ 删除 `RULE_LAUNCH_LINK`（本项目不做
  LTO，ccache 包链接只有进程开销与 "Called for linking" 统计噪声，全量一次 337 次）。
- 两项 sloppiness 的安全性（沿手册风险面逐条核对）：全仓无 `__DATE__`/`__TIME__`/`__TIMESTAMP__`
  （已 grep）；`pch_defines` 只在"用 PCH + 源内 `#define` 影响 PCH 内容"时有风险，而 PCH 是**强制前置
  包含**、源内 `#define` 影响不到它（`scan_exprs.cpp` 的 `NN_EXPR_SCAN` 本就**不套 PCH**，见前面条目）。
- 实测（本机 8 核 / Release + `NN_ENABLE_TESTS=ON` + GPU）：同一 TU（`mnist_train.cpp`，PCH 就绪）
  **30.2s → 0.11s**（`ccache -s`：Cacheable 2/2、Hits 1）；全量冷构建 **13m26s**（189 条构建边；
  单 TU ≈30s、PCH ≈54s），`ninja -t clean` 后**同目录**重建 **58.8s**（13m26s → 58.8s；`ccache -s`：55/55 次编译命中——51 direct + 4 preprocessed；剩余时间是 shader 生成与链接）。
  ⚠ 跨目录（换 `binaryDir`）不命中：PCH 的**绝对路径**在命令行里、参与哈希。
- 顺带归档的耗时分摊（`-ftime-trace`，Release 单 TU）：总计 33.1s = **后端 29.1s**（优化器 16.6s，
  其中 ModuleInlinerWrapper 10.5s / DevirtSCCRepeated 8.6s；代码生成 12.5s）+ **前端 4.0s**；
  **无 PCH 时**前端 16.8s、总计 45.8s。⇒ PCH 之后的大头已是 LLVM 优化/代码生成，继续压这一侧要
  减少每 TU 的代码量（拆目标/拆 TU），不是再加 PCH 花活。
- 验收：全新配置目录 `-DCMAKE_BUILD_TYPE=Release -DNN_ENABLE_TESTS=ON`（Vulkan SDK 在 PATH）→
  `Compiler launcher: /usr/bin/cmake -E env CCACHE_SLOPPINESS=pch_defines,time_macros /usr/bin/ccache`；
  `build.ninja` 的 `FLAGS` 含 `-Xclang -fno-pch-timestamp`（54 处）；全量构建通过。

## `CMakePresets.json` 落地（2026-10-11，E2/P0-4「剩余部分」的一半）

- 类型：演进记录（**撤销一次旧裁定**）
- 内容：E2/P0-4 曾记「`install()`/`find_package` 与 `CMakePresets.json` 本轮按用户裁定**不做**」。
  2026-10-11 用户要求补上 preset 部分：新增 `CMakePresets.json`（v6）6 个配置 preset
  （`release`/`dev`/`debug`/`portable`/`cpu`/`probes`，`binaryDir = build/<preset名>`）+ 对应
  build/test preset + 2 个 workflow preset（`cmake --workflow --preset dev` 一条命令走完
  配置/构建/测试）。**preset 只描述构建口径、不改变 CMakeLists 默认值**：不带 preset 的
  `cmake -B build -G Ninja` 行为逐字不变（避免"引入 preset 反而改了默认构建"）。
- 仍未做：`install()`/`find_package(neuralnet.cpp)`（P0-4 的另一半）——树外消费仍走
  `add_subdirectory`/FetchContent（见 `docs/usage/05-consume-as-library.md`）。
- 验收：`cmake --list-presets`（configure/build/test/workflow 四类均列出）；`cpu` 与 `portable`
  两个 preset 各跑通一次真实配置（`build/cpu`、`build/portable`）。

## 代码风格可执行化：`.clang-format` / `.editorconfig` / `.gitattributes`（2026-10-11）

- 类型：演进记录 / 否决方案（"整仓 clang-format 重排"被否决）
- 内容：风格规则原本只在 `docs/todo/10-development-standards.md` §8（散文），新增机器可读的
  `.clang-format`。取值是**实测拟合**出来的（对 `include/` 71 个文件跑 `clang-format` 后统计与
  原文的行级差异）：Allman（函数/类型/控制语句大括号一律另起一行，仓内主流）+ `NamespaceIndentation:
  None` + `ColumnLimit: 100`（全仓 99% 的行 ≤100 列）+ `ReflowComments: false` +
  `AlignTrailingComments: Leave`（仓内大量按**显示宽度**手工对齐的中文注释块，clang-format 按码点
  计算宽度，接管即打乱）+ `SortIncludes: false`（包含顺序在 §8.3 有语义约定）。逐项对照见该文件
  头注释。
- **被否决的方案**：整仓 `clang-format -i`。实测**即使是最佳拟合配置**，对全仓 150 个文件整体格式化
  仍会改 **41%** 的行（主因：续行折行策略、`if (c) stmt;` 单行/两行两种写法在仓内各约一半、
  以及少数历史文件带 namespace 缩进）。因此定位为**目标风格 + 增量工具**：推荐 `git clang-format`
  只格式化自己的改动行，并在 CONTRIBUTING/文件头明确写"不要在功能 PR 里顺手格式化无关文件"。
- 同批：`.editorconfig`（缩进/行尾/编码/尾随空格）、`.gitattributes`（**全仓 LF**：`scan_exprs` 的
  AOT 键与 KVRecord/日志解析对 `\r` 敏感，见 `docs/todo/08-pitfalls-and-lessons.md` 的 CRLF 条）。
- 验收：`.clang-format` 能被 clang-format 23 解析（`--style=file` 退出码 0）；`.gitattributes`
  落地后 `git status` 无重规范化噪声（工作树本来就全 LF）。

---

# AGENTS.md

> 来源：AGENTS.md（AI 开发速览）整改摘录（2026-09-27）。AGENTS.md 自此只记录当前状态；演进记录、已修复勘误、被否决方案与性能 A/B 过程统一归档于本文档。

## 版本交付内容（原位置 AGENTS.md 头部版本行 + §12「已交付能力」v1.1.0/v1.2.0）

- 类型：演进记录
- 内容：v1.1.0 = GPT 训练错误修复、算子融合优化、ZiPT/RAPT 掩码。v1.2.0（Release）= 混合精度（`precision.hpp` + CPU/GPU f16 路径）、RLA-2/RAPT、防 NaN 跳步、CPU 优化、头文件保护更换、编译器兼容（GCC/MSVC）、测试补充。头部"当前版本 v1.2.0 + 后续提交（Vulkan 设备选择、activation offload）"表述同轮移除；当前能力清单见 AGENTS §12。

## 计算类原语全量迁 DSL（2026-09-27，原位置 AGENTS.md §12「后续提交」第 1 条）

- 类型：演进记录 + 教训
- 内容：Layer 直调 **34 → 23**（`bench/doc_inventory.ps1` 复测），10 类计算原语全部退出 Layer 直调——`add_inplace`/`scale_inplace`/`accumulate`→`dsl::compute_into`（10 处）、`matmul`/`matmul_with_bias`→`dsl::matmul`（Linear/Conv/Transformer/Muon）、`row/col_reduce_*`→`dsl::compute_reduce`（11 处）、`batched_matmul` 21 处→`dsl::matmul(..., batch)`（alpha 用 `*dsl::rparam(scale)` 尾链，累加后乘三端等价）；缺口 A：新增 `ExprViewKind::GroupedReduceSum/Max`（`dsl::grouped_reduce_sum/max(t, R)`，R 进 key、glsl_gen 编译期展开、CPU 模板路径独立归约），MaxPool2D 迁移。保留 23 个 = 基础设施/数据搬运（transpose/slice/gather/im2col…）/状态扫描（scan_*/outer_col）/fold 显式登记，与 `docs/todo/12` §7.3 形态一致。ctest 19/19 全绿。
- 教训①：scan dry-run **禁止 `(void)` 吞错**——ZiPTBlock 的 dry-run 调单参 `forward(x)`（实际返回"用双参"错误）被吞 → **该 dry-run 从未执行真实路径**，存量结构恰好处处有 key 才长期绿，迁移新结构立刻闭合世界硬报错（已修为双参调用 + 失败 abort）；**dry-run 长期绿 ≠ 覆盖存在**。
- 教训②：新 `ExprViewKind` 落点清单：枚举+validate（`expr_spec.hpp`）→ SpecBuilder helper+叶子+自由函数+`has_reduction_v=false` 显式特化（`expr_dsl.hpp`）→ `glsl_view_read` case（`expr_glsl_gen.hpp`，R 编译期展开）→ 解释器两处 switch（`compute_cpu_engine.hpp` 形状校验+读取）；GPU 侧无 per-view 分派，vec4 白名单/uses_row 排除集自动正确。

## Vulkan 设备选择 / activation offload / CPU 逐元素优化的提交流水（原位置 AGENTS.md §12「后续提交」）

- 类型：演进记录
- 内容：Vulkan 设备选择（15eb731）：`--gpu` 参数指定设备。activation offload（初现 `42c0883`/v0.2.0、GPT 接线 `26d8eb4`/v1.1.0、`ActivationOffloader` 抽取 `8f2990f`）：相较梯度检查点更省时（文档记载 18s vs 26s / 5 step，**仓库内无原始计时**）；API 为 `GPTModel::set_activation_offload`（`set_offload_enabled` 不存在）+ `ComputeEngine::create_offload_buffer/offload_save/offload_restore`（`offload_store/load` 于 2026-09 删除）；checkpoint 与 offload **可混合**（此前帮助文本"互斥"是错的）。CPU 逐元素优化 + `dsl::compute_into`（c0d3298）：DSL 模板路径向量化/并行、`Tensor::cpu_get_ptr`、零分配原地目标传递（optimizer/layer/loss 已迁移）。

## 代码缩减重构：算子收敛 + 测试结构 + AST 移除（2026-09-26，原位置 AGENTS.md §12「后续提交」）

- 类型：删除清单
- 内容：① 删 A 档死码（`elementwise_select_scalar_cond` / `axpy_inplace` / `broadcast_row_inplace` / `nn::one_hot` / `one_hot_labels` / `Matrix::multiply_transposed_add_to` / `nn::sigmoid`·`relu` / `ops::Sigmoid`·`ReLU`）；② 新增测试公共头 `src/test_common.hpp`（`CHECK`/`make_tensor`/`check_close`/`approx`/`dot`/`close_to` 收敛一份；`max_abs_diff` 因 9 份副本语义有分叉**暂不收编**）；③ RAPT `forward_step` 的 `num/den` 除法迁 DSL，写成与 `forward:491` 同构的 `leaf/(leaf+Scalar{1e-4})` → 复用已注册 AOT 键（无需改 scan_exprs）；④ 删 `elementwise_unary/binary/binary_scalar`、`broadcast_col_inplace`（连带 `broadcast.comp` / `broadcast_gpu` / pipeline / `has_*` / 成员全链）、引擎 `row_reduce_max`、offload B 组 `offload_store/load`、`UnaryOp/BinaryOp/CompareOp` 枚举（**引擎 virtual 58 → 49**）；⑤ **旧代数 AST 整体移除**：`algebra_expr.hpp`/`algebra_compute.hpp` 删除，`Expression`/`BoolExpression` 概念迁入 `expr_dsl.hpp`，`Matrix::detail::{apply,binary_apply,binary_apply_inplace}` 死函数删除 → **CPU 求值只剩「DSL 模板路径 + IR 解释器」两套**；⑥ 删死 `activation_cache()` override（ViT/ZiPT，20 行）与重复的 `offload_test` 聚合目标（**ctest 20 → 19**），`gpt_offload_test` 补 `begin_batch/end_batch`（此前无 offload save/restore 训练真实录制路径覆盖）；`--activation-offload` 对 CPU/ZiPT 改为显式警告（原为静默 no-op 却打印"已启用"）。
- 教训：`dsl::compute` **无 2 参重载**（必带 `rows, cols`）；eager→DSL 迁移**先确认目标表达式结构已被 `scan_exprs` 覆盖**，否则 GPU 闭合世界运行期硬报错。

## CUDA 后端整体移除（96a3675，原位置 AGENTS.md §2 + §12）

- 类型：删除清单
- 内容：删除 `cuda/`、`compute_cuda_engine.hpp`、`backend/compute_cuda_backend.hpp`、全库 `NN_HAS_CUDA` 分支与 CLI `--cuda`；带 CUDA 的历史快照在 git 分支 `legacy/cuda`（c0d3298）。删除依据：40HX Vulkan 端到端 2890 ≈ torch 2640 tok/s 但算子级仅 cuBLAS 1/3、batch8 OOM。当前结论保留在 AGENTS §2：**本项目不支持 CUDA**（后端仅 CPU/Vulkan，CLI 无 `--cuda`）。

## IR-C（图 IR / 跨表达式融合）整体移除（原位置 AGENTS.md §12 + §7）

- 类型：被否决方案
- 内容：删除 `expr_graph.hpp`、`compute_engine` 的 `begin_expr/end_expr`、`dsl::start_expr/end_expr`、`FusedChainLayer`、`Tensor::virtual_tag_`。取舍记录正文（五条否证依据、删除项/保留项清单、重新立项前提）见 `docs/todo/03-ir-optimization.md` §5.3（归档正文在 `docs/history.md`）。当前结论保留在 AGENTS §7：**IR-C 未采用，当前没有任何跨表达式录制机制**。

## RLA/RAPT 激活重计算 + offload 接线（2026-09-19，原位置 AGENTS.md §12）

- 类型：已修复勘误
- 内容：`RAPTModel/RAPTBlock` 补齐 `set_flush_interval / set_checkpoint_every / set_activation_offload`（此前对 RAPT 是静默 no-op，而 CLI 会打印"已启用"）；`ReLULinearAttention` 响应 `checkpoint_mode_`（跳过 7 项 backward 缓存）、`activation_cache()` 补齐遗漏的 4 项、backward 缺缓存硬报错；抽出通用 `ActivationOffloader`（`compute_layer_base.hpp`，GPT/RAPT 共用）；修 `RAPTModel::clear_cache()` 清空 `token_emb_`（参数被毁）与 `doc_ids` 无法关闭两个缺陷。新增 `rapt_checkpoint_test`（并入 `rapt_test`）与 `rapt_offload_test`，此前**从未被编译**的 `gpt_offload_test` 转正（ctest 15 → 18）。

## CNN 缓存契约加固 + 测试补齐（2026-09-19，原位置 AGENTS.md §12）

- 类型：已修复勘误
- 内容：`MaxPool2D::backward` 补 argmax 缓存/形状校验（旧行为：`clear_cache()` 后**越界读空 vector 且不报错**——`vector::clear()` 保留容量，表现为静默用陈旧索引）；`Conv2D::backward` 补 im2col 缓存/形状校验（旧行为：checkpoint 模式或 batch 变化后**静默用陈旧 im2col**）；两层 `forward` 在 checkpoint 模式下**显式清空**缓存（只跳过赋值会因 size 相同而静默沿用旧数据）；两层补 `recompute_supported()`。新增 `cnn_test`（`maxpool_gradcheck` 独立参考比对 3 组配置 + 缓存/checkpoint 契约 + `cnn_smoke_test`），CPU 与 `--gpu` 逐位一致（此前 Conv2D/MaxPool2D **无任何 GPU 覆盖**、MaxPool2D **无任何测试**）。

## CNN 全引擎化（2026-09-20，原位置 AGENTS.md §12）

- 类型：演进记录
- 内容：新增 `im2col`/`col2im` 两个 op-level 数据搬运原语（接口 + CPU + Vulkan shader + backend + GpuEngine 包装）；`Conv2D`/`MaxPool2D` 的 forward/backward 改写为纯「引擎原语 + DSL」组合——消除 `to_matrix/from_matrix` PCIe 往返与 CPU 标量 im2col 循环，`col_cache_` 由 CPU `Matrix` 改为设备张量。布局 `(C,B*P) ↔ (C*P,B)` 用 `rearrange_3d` + `gather_rows/scatter_add_rows` + 缓存置换索引实现（`rearrange_3d` 单独做不到该三维转置）。池化反向改为「窗口内并列最大值**均分**梯度」（总梯度守恒；无并列时与 argmax-first 逐位一致）。`scan_exprs` 补 MaxPool2D dry-run（69 → 72 条融合表达式）。

## 分组归约原语 + 融合 push-constant 缺陷修复（2026-09-20，原位置 AGENTS.md §12）

- 类型：已修复勘误
- 内容：新增 `grouped_reduce_sum/max`（沿行方向按固定长度 R 分组归约），MaxPool2D 因此去掉逐通道 C 次 dispatch → 单次原语；新增 `fused_gpu_test::run_reduce_consts` 回归用例。根因：`run_fused_gpu` 的 push-constant **固定头长度必须逐形态**与生成器 PC 声明一致（逐元素 2 / 逐元素+matmul 5 / 归约 4 / 归约+matmul 6）；bug 把"归约但无 matmul"按 **5** 算 → 常量池整体后移一个 uint → **GPU 上"带常量的归约"静默错值而 CPU 正常**（表现为 `col_reduce_sum(select(x == col_broadcast(max), 1, 0))` 恒返回 kk-1 而非真实并列数）。教训：这类"只有 GPU 错"的问题要**先打印生成的 GLSL/IR 再猜成因**。详见 `docs/todo/08-pitfalls-and-lessons.md` §4.10。

# 训练 step 剖析器（NN_PROFILE，2026-10-06）

## 训练 step 剖析器落地（2026-10-06，全新设施 `core_profile.hpp`）

- 类型：演进记录
- 内容：目标 = 用工具回答"一个训练 step 里各阶段/各原语的开销构成、GPU 利用率多少"，让性能优化方向由数据决定。四层插桩：① **`core_profile.hpp` 聚合器**（帧栈 inclusive/exclusive 双口径——stage 表打 inclusive 墙钟、op 表打 exclusive 净归属（跨线程可加总、嵌套不重复计数）；`stage::op` 两级聚合；env 门控 `NN_PROFILE=1` 聚合打表、`NN_PROFILE_TRACE=<path>` 写 Chrome trace JSON；`device_add` 供设备侧回填；`interned()` 动态名驻留；默认关闭零时钟调用）；② **引擎/DSL 插桩**：`compute_engine.hpp` 42 个公共 NVI 入口首行 RAII 计时（脚本注入 + DryRun 逐条审阅）+ `expr_dsl.hpp` 三入口（`dsl::compute/compute_into/compute_reduce`——**CPU 模板路径不经引擎公共入口**，漏掉这三处会丢 optimizer/激活/残差的主体耗时）+ 两引擎 `to_matrix/copy_into/cast_into/release_idle_pool_blocks` 与 GPU 批次生命周期 override（public virtual 被 override 时基类计时不触发）；③ **phase/层级打点**：`text_train` step 循环 phase（data/forward/submit-fwd/backward/sync/optimizer/ckpt，`nn::prof::Phase` 连续阶段句柄——各阶段共享局部变量、不能用花括号块隔开）+ Model/GPT/RAPT/TransformerEncoder 逐层/逐块 scope；④ **GPU 设备侧 timestamp query**（`compute_vk_backend.hpp`）：query pool 按帧分块 + solo 块（`TsCursor`），`ts_dispatch` 括住全部 15 个 `vkCmdDispatch` 调用点（`dispatch_compute` 加 `ts_label` 形参按调用方标 matmul/batched_matmul/scan/outer_col），帧块在 `reap_frame`（fence 已等）、solo 块在 `submit_and_wait` resolve；`compute_vk_device.hpp` 保存所选队列族 `timestampValidBits`/`timestampPeriod`（**validBits=0 的队列整体降级 host 计时**——R5 240 有族=0，按所选族判门）。口径注记：begin/end 均 `ALL_COMMANDS`，每对间隔 ≈ 该 kernel 的 GPU 执行时间、dispatch 间屏障空隙不计入任何标号（= "kernel 时间 vs 空隙"口径）；GPU 内传输（fill/copy/slice/insert）暂未标号，host 侧 op 计时已覆盖 PCIe 归因。
- 验收：CPU `gpu_stability_probe --steps 20` 剖析关/开 hash 逐位一致（`49ded8cf81d9f8f4`）、GPU dev2 同（`5e992f668b4209c8`）——插桩零数值影响；Debug（文档锚口径）下 CPU 锚 `6f8849f14da23110` 与文档逐字一致；scan 91 结构 / 130 精度变体不变。实测样本（GPT d128/L4/H4/seq256/B8，TinyStories 2k doc）：**CPU 20 步**——forward:backward ≈ 72:28，`eval_expr` 净归属占墙钟 **63.7%**（注意力 fold 在 CPU 走 IR 解释器，avg 194ms/call/块），`batched_matmul` 11.2%；**GPU 20 步**（dev2）——**GPU busy 66.4%**（设备 kernel 384.7ms/墙钟 579ms：`fused_mm` 241ms/1900 次 + `fused` 132ms/10360 次），host 侧 `copy_from`（staging 上传）净归属 **30.6%**、optimizer 阶段 21.5%（其中 `import` 3.4 万次/20 步的 NVI 摊薄开销）、`release_idle_pool_blocks` 8.7%。
- 附注（既有问题，非本次引入）：**Release 构建下字节锚与 init-hash 均偏离 AGENTS §12 文档锚**（CPU 训练锚 `49ded8cf81d9f8f4` ≠ `6f8849f14da23110`、GPU `5e992f668b4209c8` ≠ `8ef51b2927253c50`、init-hash gpt `daec9e935f9b4c68` ≠ `7ac3e9d97f9d903a`）；**Debug 构建下训练锚复现 `6f8849f14da23110`** ⇒ 文档锚是 Debug 口径、Release 的 FP 优化（`-O3 -funroll-loops -march=native` 等）改变归约/初始化数值；init-hash 在 Debug 下也不复现文档值（跨进程稳定、非 flaky）——疑为 2026-10-03 后探针/构造序变动（0c53f48/d1926f0 触碰过 `gpu_stability_probe.cpp`）或锚未随最后变更刷新，**待办：重刷 §12 init 锚并注明构建口径**。

## GPU 训练 staging 上传收敛（NN_PROFILE，2026-10-06）

- 类型：性能 A/B
- 内容：首轮 profile 显示 GPT d128/L4/H4/seq256/B8 的 GPU 墙钟 579ms/20 steps、GPU busy 66.4%，`copy_from` staging 30.6%，其中文档感知注意力掩码每步把同一份 `doc_ids` 宿主缓冲按 `(1,BH·seq)` 和 `(BH·seq,1)` **重复上传**。`CausalDocScoreMask::prepare` 改为先上传一次，再在 GPU 张量上调用 `transpose` 生成列形状；CPU 路径保留原双宿主构造，避免 CPU 引入额外遍历。
- A/B（同一 Release 构建、同一 dev2、20 steps）：墙钟 **579.19→542.03ms（−6.4%）**；设备 kernel **384.71→390.45ms**；GPU busy **66.4→72.0%（+5.6 个百分点）**；GPU kernel 调度总量基本不变（`fused_mm` 1900→1883、`fused` 10360→10322）。profile 中 `copy_from` 调用数 **201→121**，说明重复 staging 路径已消除；其净毫秒受 batch drain/设备负载影响，不作为单独收益口径。
- 验收：Release `cmake --build build` 通过；CPU `gpu_stability_probe --steps 20` hash=`49ded8cf81d9f8f4`，GPU dev2 hash=`5e992f668b4209c8`，两者进程内双轮 loss/hash 均 PASS；`bench/doc_inventory.ps1` = virtual 49、`L2-VIOLATIONS: 0`、`HANDWRITTEN-RESIDUE: 0`。CPU 画像保持原瓶颈：`eval_expr` 约 64% 墙钟，下一阶段应转向 CPU fold 解释器快路；GPU 下一优先级是 optimizer 的 3.4 万次 `import`/20 steps 与 `release_idle_pool_blocks`，而不是继续重复上传。

## GPU optimizer import 快路径（NN_PROFILE，2026-10-06）

- 类型：性能 A/B
- 内容：profile 发现 optimizer/DSL 每 20 steps 约 3.4 万次 `import`，其中绝大多数是同一引擎、同一精度、已经在 GPU 上的 Tensor。`ComputeEngine::import` 增加安全快路：仅当 Tensor 有效、已绑定到当前引擎且 precision 一致时直接返回共享句柄；跨引擎、跨精度、未绑定和无效 Tensor 仍走原 `bind_check_` + `import_impl` 路径。快路不改变数据或所有权语义，只省去重复的绑定检查与虚分派。
- A/B（同一 Release 构建、同一 dev2、20 steps，设备时钟存在正常运行间抖动）：`import` 总计 **12.26→11.55ms（−5.8%）**，optimizer 内 `import` **4.83→4.55ms（−5.8%）**，调用数维持 68128（其中 optimizer 30800），证明优化的是 host/NVI 固定开销而非减少计算；单次总墙钟受 GPU 频率与 staging 排队影响（本轮 542~558ms）不作单点收益结论。
- 验收：`cmake --build build` 通过；CPU hash=`49ded8cf81d9f8f4`、GPU dev2 hash=`5e992f668b4209c8`，两轮 loss/hash 均 PASS；`bench/doc_inventory.ps1` = virtual 49、`L2-VIOLATIONS: 0`、`HANDWRITTEN-RESIDUE: 0`。后续若要继续压 GPU host 开销，应处理 `release_idle_pool_blocks` 的每步固定成本；optimizer import 已从首要低风险项中移出。

## GPU 内存池 idle block 释放单次扫描（NN_PROFILE，2026-10-06）

- 类型：性能 A/B
- 内容：`release_idle_pool_blocks` 每 step 调用 `release_idle_blocks`。原实现无论 `retain_free_bytes_` 是否为默认 0，都先遍历全部 `free_regions` 计算 `total_free`，再遍历 `blocks_` 判断并释放完全空闲 block；默认保留阈值始终为 0，因此第一遍统计没有决策价值。新增 `retain_free_bytes_ == 0` 快路径，单次扫描直接按原完全空闲条件释放；非零保留阈值继续使用原双扫描精确逻辑，避免改变配置语义或内存安全契约。
- A/B（同一 Release 构建、同一 dev2、20 steps）：`release_idle_pool_blocks` **39.88→34.76ms（−12.8%）**，单次平均约 1.99→1.74ms；总墙钟 **542.03→527.42ms（−2.7%）**。GPU busy 受设备时钟/排队影响为 72.0%→71.1%，不将其视为该改动的收益指标；本轮直接收益来自 host 侧释放扫描减少。
- 验收：`cmake --build build` 通过；CPU hash=`49ded8cf81d9f8f4`、GPU dev2 hash=`5e992f668b4209c8`，两轮 loss/hash 均 PASS；`bench/doc_inventory.ps1` = virtual 49、`L2-VIOLATIONS: 0`、`HANDWRITTEN-RESIDUE: 0`。profile 产物为 `build/prof_gpu_pool_opt.txt` 与 `build/trace_gpu_pool_opt.json`。

## nn vs torch 端到端训练基准重测（2026-10-06，新报告 docs/benchmarks/2026-10-06-vulkan-vs-torch.md）

- 类型：性能 A/B / 演进记录
- 内容：上一份报告（`2026-09-25-vulkan-vs-cuda.md`）所用 4000 行语料未入库，且其 nn 复现命令已因 `text_train` 改收 `.nndataset` 而失效 → 按同口径重建语料（`tinystories_small.txt` 前 4000 非空行，`dataset_gen` 记录 `source.sha256=584422376434996e…`），**重新生成一份基准报告**，nn/torch 两侧同日、同卡、同语料采集以规避 ±15% 跨会话漂移。主对比 f32：nn **107.1 ms/step / 152,917 tok/s / 3141 MiB** vs torch **50.2 ms/step / 326,593 tok/s / 3313 MiB** → **torch 吞吐 2.14×、nn 显存省 5%**，两侧 loss 曲线几乎重合（7.335→5.171 / 7.383→5.342，负载等价的交叉验证）。可用低精度：nn `--f16` 118.4 ms/step（健康，2113 MiB）vs torch fp16 30.4 ms/step → 3.89×；nn 全 f16 本轮 epoch1 起即 `-nan`（14 号缺陷的第三种表现，前两种为发散/打印冻结），计时与显存（1526 MiB）有效、训练无效。
- **关键差异：torch fp16"病态慢"本轮不复现**——同脚本、同 torch 2.14.0+cu132、同卡，fp16 30.4 ms/step 比 fp32 **快 1.65×**（旧值 150.3，慢 3.16×）；单步 probe：forward 47.64→**8.55**ms、fwd+loss+bwd 150.02→**26.604**ms，而 host 构 batch / H2D / Adam / zero_grad 各段不变（= 脚本与管线未变）⇒ 差异在 GPU kernel 执行本身，疑驱动 / 设备时钟 / 运行时环境；旧环境不可复现、**未二分**，标为待查项，不判旧报告为错误。f32 侧与显存则复现：吞吐比值 2.13→**2.14**、torch 显存 3313/1981 MiB **逐字一致**、两侧各 +5.7%/+6.0% 落在 ±15% 漂移内。
- step 剖析（**d64/B64，与 e2e 同配置**，20 步，`NN_PROFILE=1`）：墙钟 2197.68ms（109.9 ms/step，观测开销 +2.6%）、**GPU busy 89.8%**（设备 kernel 1972.59ms）；阶段 forward 85.4% / backward 5.4% / optimizer 5.0% / sync 3.7% / data 0.3%；op 表 `copy_from` **78.3%**（121 次、avg 14.2ms）、`release_idle_pool_blocks` 3.5%、`import` 0.5%（68128 次）；设备侧 `fused_mm` 1211ms/1830 次 + `fused` 704ms/10197 次 = 每步 95.8ms 的 **87%**。读数：`copy_from` 的 78.3% **含阻塞等待、不可读作 PCIe 传输**——`fwd.blk#0` 单次 57ms 而该处每步上传仅 KB 级，物理上不可能是传输时间，实为承接上一步设备排队；**瓶颈在设备 kernel 效率与每步约 602 次 dispatch**，三项 host 侧优化落地后继续压 host 固定开销难以改变 2.14× 差距，下一杠杆应转向 `fused_mm`/`fused`。
- 产物：`docs/benchmarks/2026-10-06-vulkan-vs-torch.md`；`build/{nn_short_f32,nn_short_f16rec,nn_short_f16,torch_short_fp32,torch_short_fp16}_20261006.log`（+`.vram`/`.peak`）、`build/torch_train_fp{32,16}_20261006.json`、`build/prof_bench_20261006.txt`、`build/trace_bench_20261006.json`、`build/{bench_bpe_8192.nnvocab,tinystories_bench40.nndataset}`。

## 评估分块（CNN 全量评估 OOM 修复，2026-09-20，原位置 AGENTS.md §12）

- 类型：已修复勘误
- 内容：`evaluate_mnist` 新增 `eval_batch`（默认 1000）分块前向 + 每块 `release_idle_pool_blocks()`；`mnist_train` 传 `cfg.batch_size`。根因：CNN/MLP 走 `N = x.cols()` 全量单次 forward，CNN 的 im2col 是 k²·C_in 倍 → 60000 样本需 ~6.4 GB → `vkAllocateMemory failed: -2`（训练步其实只多 130 MB）。见 `docs/todo/08-pitfalls-and-lessons.md` §3.7。

## 复合层 checkpoint 模式坑（原位置 AGENTS.md §12「坑」）

- 类型：已修复勘误
- 内容：复合层 override `forward_recompute` 必须调用**虚函数** `set_checkpoint_mode` 关闭子层；基类默认实现只改本块标志位 → 子层缓存不重建（stride>1 时被上一轮陈旧缓存掩盖，表现为部分 stride 通过）。模式开关（checkpoint/offload/doc-mask）的缓存契约见 `docs/todo/08-pitfalls-and-lessons.md` 模式 H。

## GPU 算子性能优化（2026-09-23；其 BK trade-off 表于 2026-09-24 被同参 A/B 推翻，原位置 AGENTS.md §12）

- 类型：性能 A/B（含被推翻结论）
- 内容：① `matmul_tiled.comp` 改 BK=16+双缓冲（共享保持 16KB 占用率不变），op 级 matmul +11~20%（教训：BK=32 双缓冲要 32KB → blocks/SM 砍半，流水收益被占用率损失抵消净 0）。② 融合生成器 `generate_glsl_matmul` 当时采用 BK=32+双缓冲（浅网格 linear 1024³ **-1.9%**、batch512 **-4.9%**，深网格 **±0**）——其"BK16 深网格真回退 -2%"的 trade-off 表后被 09-24 同参双缓冲 A/B **推翻**（见下条全算子压榨 ⑥，最终两侧同参 BK=16）。③ `submit_and_wait` 改 solo fence/cmd 复用 + `NN_GPU_PROFILE=1` stderr 分段计时，逐元素固定开销 0.16→0.143ms；余量 submit≈70µs + wait≈75µs 属驱动/唤醒延迟，kernel 本身 4096² 已达 370-440GB/s（峰值 448）**无优化空间**。④ bench 方法论：`layer_bench` 短跑必须 `--warmup ≥20` 等时钟爬坡（空闲 300MHz→1470→2070MHz，best-of 短跑双峰如 batched 4.57/5.9ms 即爬坡所致）；torch 用 CUDA event 只测 kernel、`layer_bench` 是 wall-clock 含提交开销；40HX 理论峰值 FP32 9.14 TFLOPS / 448 GB/s（34 SM×64×2×2.1GHz）。⑤ **batch=32 超线性异常（09-24 解决）**：train 侧残留 B² 项根因 = 融合 matmul 后端 dispatch `wg_y` 按**总行** rows 派发、z=batch 又数第二遍 → 工作量 ∝ BH²（生成器契约是 y 只覆盖批内行 `m_per = rows/mm_batch`，多派的 WG 被写回守卫整块丢弃——**结果正确，对拍永远测不出，只有计时能暴露**）。修复后 train batch=32 **2999.9→227.7ms（−92%）**、b16→b32 翻倍比 3.74→1.95 线性，batch=1 亦 −16%。教训：**dispatch 网格必须与生成器行界契约同源核对；"全绿"只证明算得对，不证明没空转**。

## fold 分块流式求值 + GPU 注意力全链优化（2026-09-24，7cdb41c+fd68573，原位置 AGENTS.md §12）

- 类型：性能 A/B
- 内容：注意力 forward 改单 fold kernel（`FoldSpec`：QKᵀ/掩码/online softmax/ΣwV 逐 `EXPR_FOLD_BLOCK=128` 块完成，S 绝不物化；`tri_skip`（当时字段名 `causal_skip`）整块跳过被屏蔽区；NR=2 每 WG 两行；subgroup 蝶式归约 + vecacc 4 路软件流水）；旧物化/两趟 forward 与 `build_attention_mask`/`mask_cache_`/`m/l/attn_cache_` 等死码删除。**实测 vs main 基线 7.9/20.3：mha fwd 5.41（−31.5%）、causal fwd 4.75（−40%）、train 17.5~17.7（−13%）**，18/18 绿。质量门禁轮：Doc/AlibiDoc fold 注册与测试矩阵全覆盖、ALiBi slopes 表 `(1, batch·H)` 越界修复、GEMV subgroup≥4 门禁、tri_skip bin v8 往返断言。

## 融合 matmul vec4 内核移植（2026-09-24，原位置 AGENTS.md §12）

- 类型：性能 A/B + 坑
- 内容：融合 matmul 同形状比 op 级 `matmul_tiled` 慢 2.3×（768×768×8192 fwd 6.18 vs 3.15ms——**"融合 vs op 级同形状对拍"是发现内核代差的 X 光**）；三重根因：内层 `acc+=dot(a[i],b[j])` 标量链（每 k ≈5× 指令发射）、全局 4×标量 ldg+16×标量 stg、共享 vec4 沿 k 布局。移植 op 级 v3 配方（共享沿 m/n + 每 k 2 LDS+4 VFMA.128 + vec4 别名快路径；trans 进 key → 分组生成期定死；BK=32/barrier 节奏/尾链不动）后：**linear 浅/深/投影三点 −33%/−48%/−47%（投影 6.18→3.25ms，达 op 级 97%）、mha fwd 4.59→3.63、mha train b1 13.0→10.8、b32 train 227.7→173.1（vs dispatch 修复前累计 2999.9→173 = −94%）**，18/18 绿。坑（rapt_offload 抓出）：共享索引必须用 tile 内**局部 k**，全局 k（`t*BK+local`）只进缓冲地址/守卫——混用则 t≥1（mm_k>32）越界共享 = 非确定性错值，num_tiles=1 的小形状对拍全测不出（全仓唯 rapt_offload d_model=64→2 tiles 有对照）；"对拍全绿但数值每跑不同"→ 先同二进制跑两次 diff。

## 全算子压榨 + OP/DSL 同步轮（2026-09-24，原位置 AGENTS.md §12）

- 类型：性能 A/B（含推翻 09-23 trade-off 表）+ bench 坑
- 内容：① `batched_matmul.comp` 移植 matmul_tiled v3（自 0.2.0 起未优化，同形 1024³ 比 matmul 慢 1.85×）：**b1 1.66→0.876ms（−47%）、b8 8.12→4.08ms（2116→4214 GFLOPS）**。② `transpose.comp` 重写 32×8 WG / 32×32 共享分块（tile[32][33]，读写双向 128B 合并、零 bank 冲突），backend dispatch 改 (ceil(C/32), ceil(R/32))：**4096² 1.575→0.492ms（−69%）**。③ `reduce.comp` 列归约重写（lane=列/warp=行块/32 列 tile + shared 跨块合并 + 4 路 ILP），max 恒等元 −inf→`lowest()`、行归约→subgroup 蝶式：**4096² 0.843→0.273（−68%）**；reduce 用 subgroup → CMake 需单独 `--target-env=vulkan1.2`（同 matmul_gemv）。④ DSL 列归约同结构重构（`generate_glsl_reduce`：dispatch cols/256→cols/32——CE 训练热路径旧 WG 并行度极低；合并值覆写回 s_red 与他 lane 读同址，**必须 3 barrier：partial 完成→读完→写完，否则 sum 双计**）。⑤ `elementwise_v2`/`broadcast`/`gather` → vec4（base=gid*4），三处 dispatch ÷4，broadcast 4096² −17%。⑥ **OP/融合 matmul BK 统一 A/B（推翻 09-23 结论②的 trade-off 表）**：OP 级 BK16→32 四点全回退（512³+9%、1024³+26%、4096³+29%、batched b8+29%）→ 维持 16；融合侧 BK32→16 反超 linear 浅 −24% / 深 fwd −24% / 深 train −17%（8.21→6.81ms）→ **两侧同参 BK=16**（09-23"深网格 BK16 −2%"是对单缓冲旧基线的口径，非同参双缓冲对比）。⑧ **bench 坑**：`layer_bench` transB/at 操作数曾恒建 (k,n)（应 (n,k)/(k,m)），k≠n → backend K mismatch → run 内 `*expected` 不查错 = UB → **0.000ms / 1.7e11 GFLOPS 超物理垃圾值**（k=n 正方形长期掩盖）；已改 per-variant setup + 错误检查 abort。教训：**bench 出现超物理数值先怀疑测法本身**。gather 快路径曾漏读 `indices[]`（把索引下标当表行号），zipt CPU/GPU 对拍抓出——vec4 改写后对拍必跑。最终 @1024：batched 0.871、matmul 0.853、col_reduce 0.123、transpose 0.138、broadcast 0.119；mha fwd 3.449 / causal fwd 2.819 / linear b4096 train 6.901ms。18/18 绿。

## transpose 砖块化 + reduce 两段式 + bench 记账修复（2026-09-24 二轮，原位置 AGENTS.md §12）

- 类型：性能 A/B + 方法论
- 内容：① **bench 虚高记账修复（"部分算子超出 40HX 理论"的根因）**：`bytes_reduce` 曾按 2N 记归约（输出仅向量级，实际 ≈N）→ 5244² 虚报 440 GB/s、4096² 虚报 491（超 448 理论峰值，物理不可能）；改精确 (N+out) 后真实 217-263 → **reduce 其实有一倍头寸**。② `transpose.comp` 定稿：64×64 tile（段宽 128→256B）+ 共享落位交换 + **8×8 WG 砖块化 dispatch**：5244² 0.99→0.75ms（−24%）、4096² −17~28%。消融链定根因：纯拷贝 0.556 → 过 shared+barrier 0.665 → 线性 dispatch 0.850，**真凶 = 并发 WG 写侧足迹散布整块**（转置输出行=输入列：线性 dispatch 下 136 个并发 WG 每行只写 256B、行距 21KB ≈180MB 散布）——砖块化让连续 64 WG 填满一砖、足迹收敛 ~1MB 聚集；16×8 砖与 8×8 持平。③ `reduce.comp`：行归约 vec4 快路径（cols%4==0）；列归约**两段式 partials**（ggml `rms_norm_partials` 模式：pass1 行向切 nchunk → partials(nchunk,cols)，pass2 复用整表合并；同 cmd 双 dispatch 单次提交 → 固定提交开销不翻倍；scratch 走成员 `reduce_partial_` 防录制期 UAF）：5244² 0.50→0.38-0.42ms；push 扩 5×uint=20B（`chunk_rows`），**pipeline layout push range 同步 4→5 个 uint——漏改即静默错值**；512K 元素护栏排除微型形状（512² 两段 −16% 净亏）。④ `broadcast.comp` vec 行单 vec4 装载 5244² −13~16%。⑤ **方法论（调参/判退前必读）**：同码对照两 exe 同窗交错实测 **噪声地板 ±5%、跨会话系统漂移 ±15%**（round2 全面慢 5-10%）——两段式曾被跨会话单点误判"4096² 净亏 20%"，交错 A/B 实测只亏 2%（SP 5/5 配对胜）、5244² 反赢 6-9%（TP 4/5）才定案保留；**调参/判退一律同窗交错 + 配对统计，禁止对着跨会话单点改参数**。18/18 绿。

## 多精度 Phase 2：边界 cast → in-kernel f16 → op-level f16 GEMM（2026-09-25，原位置 AGENTS.md §12 第 ⑭ 大条，docs 05 §12.5-§12.11）

- 类型：性能 A/B + 演进记录
- 内容（按阶段保留关键数字与结论）：
  - **接线**：引擎运算类原语加 `Precision P = F32`、新增 `cast_into`/`copy_into`；新增 `PrecisionEngine` 适配层（f16 边界 cast 集中一处，全 f32 = 纯直通逐字节一致）；`dsl::compute/compute_reduce` 加 P；Layer/Loss/Optimizer/工厂/CLI 全链接线。教训：**复合层必须把 profile 下传给子层**（`AttentionBase`/`FeedForward`/`TransformerEncoderLayer`——不下传则子层静默停 f32；`Model::add` 用未设默认值覆盖 GPTModel 构造器已注入的 profile 曾导致 token_emb 退回 f32）。
  - **`--f16` 语义** = `profile_f16()` = {param:F16, compute:F16, stable:F32, optimizer:F32}；**四字段全 f16 不可训练**（optimizer=F16：Adam 的 v≈g²~1e-10 下溢→更新爆炸 loss 7.9→3.6e4；stable=F16：CE 链 ~200 步 NaN；两者同 f16：loss 恒定）。
  - **边界 cast 结论**（40HX，GPT d64/h4/L4/ff256、vocab 8208、seq256）：f32 峰值 3069MiB；f16 边界 cast 峰值反而升高（2831MiB 后 backward OOM，探针 transient live 2795→6687MB）→ **边界 cast 只能拿"存储减半"，拿不到"峰值下降"；须做 in-kernel f16**。
  - **in-kernel 第一期**：`ExprPrecSig`（bit i = 第 i 个输入 f16、bit16 = 输出 f16，全 0 = 旧行为）+ `ExprRegistry::variants/add(spec,sig)` + bin v9 只存 `{sig, 基础结构下标}`；`scan_exprs` 双 pass（**f16 pass 必须经适配层——原生 CpuEngine 喂 f16 = heap corruption 0xC0000374**）；`GlslEmitter::generate(name,spec,sig)`（`RotateHalf` 取负 / `RowGather` 的 `uint()` 必须在叶子处转，否则 glslc 报错）；`run_fused_gpu` 输入 `FusedInputs{owners,bufs}` **必须同时持有 owner**（只存裸指针 = 录制中途释放上传缓冲，实测 `A+=B err=0.5`）；f16 输出按 2B/元素分配并重贴 `GpuTensorF16`（漏了 → loss=NaN）。实测 batch32：f32 1595 / f16 in-kernel 3457 / 边界 cast 4124MiB（f16 仍 2.2×）。**签名必须逐输入**：真实 GPT f16 变体空间 38 个 = 逐元素/归约 32 + matmul 段 5 + fold 1，存在混合签名。
  - **归因演进（教学点）**：⑪ 判"边界 cast 临时块数量膨胀"→ ⑫ 判"池底材粒度"（阶梯≤16MB −21% 但 2× 耗时）→ ⑬ 同窗复测**推翻 ⑫**（峰值 3513/3513/3513 持平、耗时全 ≈5.7s，"2× 时间"是跨会话单点产物，池小分配快路径工程项作废——池账本 1 万次 allocate 只扫 ~5 万 block、`vkAllocateMemory` 累计 ~150ms，本就不是瓶颈）+ 新增形状级归因表（`dump_temp_stats`，`NN_PREC_TRACE=1`）定罪 **op-level cast**：`(32768,256)` 注意力反向 W/grad_A 占 **1152MB**，其次 `(64,8192)` 270、`(256,8192)` 160、`(2048,256)` 112MB。**机制之争必须用形状/尺寸归因表裁决，别用总量推理。** 据此补三类带类型变体（matmul 段 `float16_t`+`uvec2`+`unpackHalf2x16` / 归约 kernel `rd()/wr()` 读写点转换 / `eval_expr_into|eval_expr_reduce` 加 `(key,sig)` 匹配）→ `gen_fused` 由 **41 变体/14 skip → 54 变体/0 skip**；实测同窗 3 轮：f32 1753（3/3 一致）、f16 2025/2025/2273 = 1.15~1.30×（此前 3513，即 cast 路径 **−35%~−41%**），耗时无回退，batch64 f16 不再 OOM。
  - **op-level f16 GEMM（第 ⑭ 阶段）**：手法 = 一份 .comp 用 `glslc -DNN_SHADER_F16=1` 编第二份 SPIR-V（`NN_ETYPE=float16_t`/别名槽 `NN_V4=uvec2`/`NN_LOADV4=unpackHalf2x16`，**f32 分支留在 `#else` 逐字未动 = 零回归证明：头文件字节数不变**（`matmul_tiled_spv.hpp` 68670 / `batched_matmul_spv.hpp` 74408））；后端 pipeline 只在 `has_16bit_storage()` 时创建；**适配层必须把 `P != F32` 直接下传**（第一版漏了这步 → 归因表逐项不变、峰值纹丝不动）；`CpuEngine::matmul_with_bias` 补下传 `P`（否则 scan 的 f16 dry-run 只登记全 f32 签名；变体 54→56）。实测同窗 3 轮（batch32/steps2）：f32 1754×3、f16 1570×3 = **−10.5% 显存且略快**（5.05 vs 5.11s）；batch64 亦 −10.5%；`text_train --f16` avg_loss 6.7493 vs f32 6.6979（0.77%）。**演进全景：f16 边界 cast 4124 → in-kernel 首期 3513 → +matmul段/归约/目标传递 2025~2286 → +op-level GEMM 1570**；剩余约 550MB cast（`(2048,256)×80`、`(64,8192)×72` 等）+ 7 条 run-only 签名 + fold 变体，收益已递减。

## CPU 侧 f16 训练发散双根因修复（2026-09-25，原位置 AGENTS.md §12 末条，docs 05 §12.12）

- 类型：已修复勘误（根因 + 修法 + 测试盲区）
- 内容：§12.5"已知问题 1"（CPU f16 5~7 步 NaN、未定位）实为**双缺陷叠加**。
  ① **DSL 预绑定喂错精度**：`MatmulRef::prepare_cpu` 固定以 P=F32 物化 C，而 `CpuEngine::matmul` f32 路径按 f32 存储直读 f16 张量 → `cpu_matrix<F32>()` 空指针 UB（Debug 断言 `tensor has no P-precision CPU storage`；Release 即 `text_train --f16` CPU 0xC0000005）；同族：`ReduceViewRef::prepare` 直读、适配层 `matmul/batched_matmul` 把 `P!=F32` 无条件下传、CPU 解释器 `eval_expr_impl` 无精度校验。修复 = prepare 先抬 f32 再物化 / f16 输入一次性镜像 / `CpuEngine::matmul|batched_matmul` 精度与存储不匹配时回退 f32 空间计算 + 按 P 舍入（两端同 f16 才走原生 f16 GEMM）/ 解释器加精度 `NN_REQUIRE`。
  ② **`float_to_half_bits` 次正规分支移位 UB（真正的 NaN 制造机）**：守卫 `exp <= -46` 与其自身公式矛盾（只对 exp∈[-25,-15]/shift 14..24 成立）→ **exp∈[-45,-33]（|v|∈2.8e-14~1.2e-10）shift≥32 移位 UB** → 小值被转成垃圾 half（0x4000=2.0 / 512 / 8192 / 11776 / 18432，随机落 e=31 即 NaN）；**梯度正是这个量级、前向激活 ~0.1 从不落入** → 损坏只见于 f16 梯度写回；GPU 走硬件转换故"只有 CPU 错"；单算子测试输入 ≥1e-3 故"全过"。修复 = 守卫改 `exp <= -26`（≤-26 按 D<0.5 恒 flush 0）。
  ③ **测试盲区**：`ref_ulp` 写成 `2^(e-10)`（应 `2^(e-25)`，**大 2^15 倍**）→ RHE 容差比被测值本身还大、垃圾值全放行；修复 ref_ulp + 补 exp∈[-60,-26] 全网格"必须恒 0"断言 + 两个历史垃圾代表值定点断言。
  ④ **方法论**：新组件探针 `f16_cpu_probe`（阶段 A–I）+ `NN_F16_DEBUG=1` 层内逐中间量扫描——把故障钉到 `accumulate: pre_dst=0, pre_src=8.7e-13, post_dst=18432` 单步即锁定转换函数。修复后 CPU f16 三组 profile 6 步轨迹与 f32 **逐位一致**；`f16_precision_test` 容忍分支改**硬失败**；profile 单字段矩阵仅 optimizer=F16 仍发散（§12.5 数值性限制，与本缺陷无关）。

## 基准报告更新与 f32 发散勘误（2026-09-25/09-26，原位置 AGENTS.md §12「已知问题」）

- 类型：已修复勘误 / 性能 A/B
- 内容：**f32 训练发散 NaN 已修复**——09-26 实测短测 3 epoch + 长测 5 epoch 健康收敛，09-25 基准记录的"f16/f32 均发散"在 f32 上不复现。**09-25 基准报告 nn 侧全量重跑**（`docs/benchmarks/2026-09-25-vulkan-vs-cuda.md`）：f32 优先主对比 101.0 ms/step；transpose 3.7×→2.3×（P0 静默半写虚报修正）；`elementwise_exp`/`broadcast_col` 已删标注；f32 显存 5743→3069 MiB。

## S7 关键教训中的过程叙述（原位置 AGENTS.md §12「S7 关键教训」第 5、6 条）

- 类型：演进记录（约束本体保留在 AGENTS §12「融合二期状态与 IR 编码约束」）
- 内容：matmul + 列归约于 **2026-09-26 补齐**（`generate_glsl_reduce` 列分支按元素分解 batch、`gen_fused` 三处跳过删除）。注意力 forward 曾为 m/l/W 表达式 + bm(W,V_t) 的 S7 IR 链结构（**已删**），现为单 fold kernel（`FoldSpec`，5 掩码变体经 `fold_mask_variant_`）；bwd = R/X 表达式 + 3 个 `batched_matmul`。

## 「已过时的历史说明」整节（原位置 AGENTS.md §12 末节）

- 类型：删除清单
- 内容：五条——① **全库代码审查报告（2026-09-04）**：一次性审查文档已删除（git 历史可查）；`--tdr-retry`/`--max-tdr-retries` 选项已从代码移除（当年"被解析但从未使用"的问题不复存在——当前结论保留于 AGENTS §12）。② **CPU 性能诊断报告与路线选择建议（2026-09-18）**：临时诊断/决策文档已删除，结论并入 `12-compute-engine-inventory.md`（CPU 侧求值机制 2026-09-26 起只剩两套：DSL 模板路径 + IR 解释器）与 `03-ir-optimization.md` §5.3（IR-C 取舍）。③ **CpuEmitter**：`cpu_emitter.hpp` 与实现已删除，早期"待修 CpuEmitter 隐性缺陷"问题随之消失勿引用（当前表述：AGENTS §7 仅 `glsl` 后端注册，无 CpuEmitter/CudaEmitter）。④ **融合三期 S6**：未列入当前计划；替代方案 P2-12 图级缓存随 IR-C 于 2026-09-19 删除。⑤ **IR-C**：已评估并整体移除（删除项见上文 IR-C 条目）。

## 其余移除叙述与过期事实订正（原位置 AGENTS.md §2/§3/§4.3/§5/§7/§8/§10/§11）

- 类型：删除清单 / 已修复勘误
- 内容：
  - **§4.3「2026-09 收敛」块**：`axpy_inplace`、`broadcast_row/col_inplace`、`elementwise_unary/binary/binary_scalar`、`elementwise_select_scalar_cond`、`row_reduce_max`（引擎算子）、`offload_store/offload_load` 以及 `UnaryOp/BinaryOp/CompareOp` 枚举已全部删除（逐元素/广播一律走表达式 DSL；`dsl::row_reduce_max` 是 DSL 叶子）；**引擎 virtual 58 → 49**。
  - **§5 铁律 5 历史叙述**：历史上 GPT 曾用 position-major（`i = t*batch + b`）导致跨样本串扰的灾难级 bug，现已统一为 batch-major。
  - **§10 坑 7 历史数字**：BPE"所有出现位置"类索引曾膨胀到 60-80GB；大词表 one-hot 曾 3.2GB。
  - **§7 移除叙述**：跨表达式融合（`start_expr/end_expr`、`begin_expr/end_expr`、`expr_graph.hpp`）2026-09-19 移除；fold 构造 2026-09-25 前在 `expr_fold.hpp`（后按 AOT 原则归位 `compute_layer_attention.hpp`）；`FoldSpec.tri_skip` 原名 `causal_skip`；`dsl::grouped_reduce_sum/max` 2026-09-27 新增；`broadcast.comp` 随 `broadcast_*_inplace` 于 2026-09 删除；`CpuEmitter`（已删除）/`CudaEmitter`（随 CUDA 移除）勿引用。
  - **§3 目录表括注**：旧代数 AST（`algebra_expr.hpp`/`algebra_compute.hpp`）2026-09 移除、`Expression`/`BoolExpression` 迁入 `expr_dsl.hpp`；`expr_graph.hpp`/IR-C 2026-09-19 移除。
  - **§8**：RAPT "自 2026-09-19 起"与 GPT 同档（当前：RAPT 与 GPT 同档）；链式构建伪 API 见下条。
  - **§11 索引过期事实订正**：`development/12` 描述由"49 个 virtual（原 58，删 9 个）、**Layer 直调 32 个**"订正为实测 **23**（`bench/doc_inventory.ps1`：virtual 49/49、layer-called 23、not layer-called 26）；"后附 2026-09 收敛记录"→ 演进记录已移出（本文件/w4-dev1）；`development/13` "含误报/已修复对照表（§6）"→ 对照表已移入 `docs/history.md`；`development/03` "§5.3 是取舍记录"→ §5.3 现只指向 `docs/history.md`；`introduction/01` "CUDA 已移除备注见篇末"→ 该文档已无此备注；`development/04` "已实施"、`development/02` "删手写原语"等状态词与历史词改为当前形态描述；新增 §11 总档行 `docs/history.md`。

## §8 链式构建伪 API（`add_linear`/`add_relu`，原位置 AGENTS.md §8）

- 类型：已修复勘误
- 内容：原文写 `model.add_linear(784,256).add_relu().add_linear(256,10)`——**全仓不存在 `add_linear`/`add_relu`**（grep 仅命中 AGENTS.md 自身）。`Model::add` 是模板方法 `template <typename LayerType, typename... Args> Result<void> add(Args&&...)`（`include/neuralnet.cpp/model_container.hpp:71-72`），无链式 API，层构造后自动 `init(engine)`。已改为逐层写法 `model.add<nn::Linear>(784,256); model.add<nn::ReLU>(); model.add<nn::Linear>(256,10);`，与 `docs/usage/01-quickstart-model.md:19-21` 一致。

---

# ZiPT（AttnZip）移除（2026-10-01）

> 本文是全仓唯一归档处；`AGENTS.md` §12、`docs/introduction/04` §10、`docs/development/15` §7.3
> 现只保留「已移除 + 指针」。**代码本体完整保留在 `legacy/zipt` 分支**（`git show legacy/zipt:<path>`）。

## ZiPT 移除裁决与恢复前提（2026-10-01）

- 类型：删除清单 / 否决方案（附**恢复前提**）
- 决定：**AttnZip（ZiPT）整体移出主线**（`main`），代码与文档归档到 `legacy/zipt` 分支。
- 依据（代码事实，非文档声明）：
  1. **默认路径严格劣于既有 GPT 路径**：`ZiPTConfig.window = 0` → `W = L` → `split_ = false`
     （`domain_zipt.hpp:58`、`compute_layer_zipt.hpp:740-743`），此时块输入就是整条序列，
     块内仍做 `W = L` 的联合注意力（**O(L²)**），并额外物化并缓存一张
     `BH·W·(M+W)` 的 F32 掩码（`mask_cache_`，`compute_layer_zipt.hpp:286-316`）。
     同样的计算量下，比 fold 流式注意力多一张显式掩码 + 物化 scores + 一个压缩器 pass。
  2. **压缩器只有模型级一份、且作用在嵌入上**（`compressor_.forward(x)`，x = embedding + pos，
     `compute_layer_zipt.hpp:850-861`）；`07-zipt-algorithm.md`（已删）§6.2 承诺的
     "每隔几层重复压缩" 从未实现。
  3. **算法层未闭环**：缺省模式下压缩器压缩**整条含未来 token 的序列**（非因果）；
     项目自己的文档承认 "严格因果需压缩 prefix-only（待做）"（`04-innovative-designs.md` §10.4，已删）。
  4. **训练显存开关全缺**：无 `forward_recompute` → 不支持梯度检查点（override 直接
     `std::abort()`，`compute_layer_zipt.hpp:785-795`）；activation offload 未接线。
     而 "长上下文" 恰恰是它唯一的卖点——最需要这两项的场景它一项都没有。
  5. **零质量/成本数据**：全仓（`bench/`、`docs/benchmarks/`）没有任何 ZiPT 的
     loss / perplexity / 显存 / 吞吐对比；`04` 的"里程碑通过"仅为冒烟级"loss 下降"。
  6. **CLI 不可用且无人使用**：`text_train --model zipt` 自 2026-09-29 起启动即 abort
     （`Model::set_checkpoint_every(0)` 无条件下发 → ZiPT override 不判 `stride==0`），
     长期无人发现 = 无实际使用者。
  7. **持续重构税**：M1–M6、位置编码策略化、`NN_TRY` 收敛每一轮都把 ziPT 纳入迁移范围
     （约 1200 行 + 33 个文件的引用）；`compute_layer_zipt.hpp` 1026 行 + `domain_zipt.hpp` 121 行
     + 测试 812 行。
- 生态位判断：ZiPT 想占的是 "窗口内精确 + 全局压缩记忆" 这一点，**该生态位真实存在**
  （NSA / MoBA / gist-token 一族），但当前实现离卖点差三件关键事（因果压缩、有信息量的
  压缩层级、可训练长上下文）；对照之下 fold 流式注意力（精确、O(L) 显存）与
  RLA-2/RAPT（O(L) 时间、已实测）都已成立并接入。
- **恢复前提（两条同时满足才允许恢复）**：
  1. **算法层**：压缩向量必须**因果**——压缩器只能看到当前位置之前的内容；
     并明确压缩发生在哪一层（嵌入层 vs 逐层）及其依据。
  2. **代码层**：实现质量对齐 RAPT/GPT——`forward_recompute`（梯度检查点）+ activation offload
     接线、CLI/`ModelSpec`/序列化/ctest 覆盖齐备、且**有与 GPT/RAPT 同参同等墙钟预算的
     对比数据**（否则仍是不可裁决的设计）。
- 删除清单：
  - 文件：`include/neuralnet.cpp/compute_layer_zipt.hpp`、`include/neuralnet.cpp/domain_zipt.hpp`、
    `src/zipt_{test,gradcheck,smoke_test,consistency_test,doc_test}.cpp`、
    `docs/development/07-zipt-algorithm.md`。
  - 符号：`CrossAttention`、`ZiPTBlock`、`ZiPTModel`、`build_zipt_model`、
    `build_zipt_model_from_spec`、`make_zipt_spec`、`ZiPTConfig`、`ZIPT_MEMORY_TOKENS`、
    `ModelSpec::is_zipt()`、`concat_cols`（仅 ZiPT 使用的死函数）。
  - 字段 / CLI：`ModelSpec::memory_tokens`、`ModelSpec::window` 及序列化中的两个可选键；
    `text_train` 的 `--window` / `--memory-tokens`（ZiPT 专用；滑动窗口步长是既有的 `--stride`）；
    `ModelType` 的 `zipt` 取值。
  - 测试/探针：ctest 目标 `zipt_test`（20 → 19 用例）；
    `gpu_stability_probe --init-hash` 六类模型 → **五类**（mlp/cnn/transformer/gpt/rapt）。
- 文件格式兼容：`ModelType` 的 6 号取值**保留为 `Reserved_ZiPT` 占位**（不删除枚举值），
  `spec_from_kv` 对 `type=6` 返回明确错误（"ZiPT 架构已移除…见 legacy/zipt"），
  而不是含糊的 `Unknown`；`spec_summary` 对该值输出 `ZiPT(removed)`。
  旧 `.bin`（含 `memory_tokens`/`window` 键）因此得到可读的拒绝信息而非静默误解。
- 复用与复用边界：`concat_cols` 已删除（唯一调用方是 ZiPT 的 K/V 拼接）；
  其实现方式（`transpose + insert_rows + transpose`）如需复用，可在 `legacy/zipt` 分支取回。
- 验收（本轮实测）：build 195/195 步、零告警（`-Werror`）；**ctest 19/19**（20 → 19，
  `zipt_test` 退出）；`gpu_stability_probe --steps 20` CPU 字节锚
  **`6f8849f14da23110` 逐位不变**；L2 分层审计 `L2-VIOLATIONS: 0`
  （`matrix_type_hits: 0 / io_verb_hits: 0 / host_bridge_uses: 40` → 41，ZiPT 的宿主桥用量消失）；
  引擎 virtual 仍 **49/49**、Layer 直调仍 **21**（删除的是层与领域代码，不动引擎接口）。
- **计划内锚点变化**：`gpu_stability_probe --init-hash` 由六类降为五类，且 **rapt 锚变化**
  `b037632f75b7159c` → **`d1998412d41ef2a0`**。原因不是 rapt 本身改了，而是探针**在同一引擎上
  顺序构建**模型、`InitSpec` 分布初始化按引擎内**创建序号**混流 seed
  （`17 §4.4` / M2 U1 裁定）——ZiPT 退出后 rapt 的序号前移。前四类
  （mnist_mlp `a22e807ee05ec3ac` / cnn `4020958a14160bbd` / mnist_transformer
  `04a72d865624042a` / gpt `8efa936ac5c8c9b2`）构建在 ZiPT 之前，**锚逐位不变**，
  与上述解释一致（若前四类也变了，说明改动越界）。真实用户单独构建 rapt 不受影响
  （序号从 1 起）。

## ZiPT 评测历史（供恢复时对照）

- **无实测数据可归档**：项目内从未记录过 ZiPT 与 GPT/RAPT 的同参对比（唯一记载是
  `04-innovative-designs.md` §10.3 的 "gradcheck + CPU/GPU 一致性 + 端到端 loss 下降，全部通过"，
  属冒烟级）。因此本次移除是"**未被证据支撑的设计退出**"，与 CUDA 后端、旧代数 AST、
  IR-C 图融合、`CpuEmitter` 同属一类处置。
- 与已修缺陷的关系：issue #13 P1-10（"ZiPT stored_tokens 无门控"，低优先）随之作废。

---

# GUI / CLI 参数一致性清理（2026-10-01）

> 触发：ZiPT 移除后 `gui.py` 仍在无条件发 `--window 0` / `--memory-tokens 32`，
> 而 `text_train` 对未知参数直接 `exit(1)` —— 即 **GPT 训练页彼时已整页不可用**。
> 借此把 GUI / `cli_controllers.py` / 各 CLI 的"缺失·多余·默认值"三类偏差一次收敛。
> **门禁**：`python bench/gui_cli_audit.py`（三节：GUI→CLI flag 差集 + 反向缺失清单、
> 幽灵选项=帮助声明 vs 解析分支、`gui.py` 的 `self.X` 赋值静态校验；退出码 0 = 无偏差）。

## 一、多余项（GUI/控制器发出、CLI 不存在）

- **CUDA 残留（CUDA 移除时漏网的一半）**：`gui.py` 的 `ENGINE_OPTIONS` 仍列 `"CUDA"`，
  4 处 `collect_args` 里有 `elif engine == "CUDA": args["cuda"] = True`，
  两个 `build_pack_config` 还会据此写 `device: "cuda"`；`train_pkg.py` 甚至有
  `--device cuda` 与 `hyper["cuda"] = True`。而 CUDA 后端早已整体移除
  （见上文"CUDA 后端移除备注"），CLI 传入 `--cuda` 即"未知参数"；
  控制器侧对 `cuda` 键也从不消费（静默丢弃）→ 用户选了 CUDA 实际按 CPU 跑且无任何提示。
  本轮：`ENGINE_OPTIONS` 去掉 CUDA、4 处分支删除、`train_pkg.py` 的 `--device`
  只留 `cpu/gpu`（旧包里的 `cuda` 字段改为**显式忽略**而非报错）。
- **ZiPT 残留**：`model_type` 选项列表含 `"zipt"`、`memory_tokens` / `window` 两个控件
  与 `_on_model_type_change` 显隐回调；`cli_controllers.py` 的 `--memory-tokens`/`--window`
  映射。其中 `window` 用了 `skip_vals=()` → **每次运行都无条件发 `--window 0`**，
  在 ZiPT 移除当刻就会让 `text_train` 报"未知参数"退出。
  本轮全数删除，`model_type` 收敛为 `["gpt", "rapt"]`（列表提为常量 `GPT_MODEL_OPTIONS`）。

## 二、幽灵选项（帮助声明了、解析分支根本不存在）

- **`mnist_train --osc-guard <on|off>` / `--osc-window <n>` / `--osc-threshold <f>`**：
  三个选项只在 `--help` 的打印链里存在，**解析分支为零**（`grep -n 'osc' src/ include/`
  只命中帮助三行 + `compute_optimizer.hpp:114` 一句"供 OscillationGuard 等自适应调度器使用"的注释）；
  实测 `mnist_train --osc-guard on` → `未知参数: --osc-guard` + exit(1)。
  即：自"振荡检测自动降 lr"这一功能被删除起，帮助文本与 `cli_controllers.py`
  （`MnistTrainController` 的 `osc_guard/osc_window/osc_threshold` 三条映射）就一直是悬空的。
  本轮按"文档不得声明不存在的选项"处理：**帮助文本 + 控制器映射 + GUI 控件三层一起删**
  （不补实现——与仓库近期"移除自适应启发式"的取向一致，如 NaN 跳步的移除）。
  复现判据：`build/cli_phantom_option_audit.py` 六入口全 OK。

## 三、缺失项（CLI 有、GUI 无）

| 入口 | 本轮补上的参数 |
|---|---|
| `mnist_train` | `--shuffle-steps`（勾选框，默认 true）、`--lr-per-epoch` |
| `text_train` | `--resume-epoch` / `--resume-step`（续训定位）、`--no-cache`、`--lr-per-epoch` |
| `tokenizer_train` | `--threads`（0=自动，对应 BPE 保序并行） |
| `tokenizer_infer` | `--top`、`--threads`、**位置参数"文本文件"= 统计每行 token 数模式** |
| `text_infer` | `--vocab`（模型未嵌入 tokenizer 时必填） |

- 对应地在 `cli_controllers.py` 新增 `resume_epoch` / `resume_step` / `no_cache`（GptTrain）、
  `threads`（TokenizerTrain）、`top` / `threads` / 位置参数 `text_file`（TokenizerInfer）映射；
  并把 GptTrain 的 `memory_tokens` / `window` 映射删除。
- **有意不暴露**：`text_infer --interactive` 与 `tokenizer_infer --interactive`——
  GUI 本身就是交互外壳，再套一个 stdin 交互循环没有意义（审计脚本将其列为预期缺失）。

## 四、默认值与 CLI 不一致（对齐 CLI / 实现）

| 页 | 项 | 原 GUI 默认 | 现默认（= CLI/实现） |
|---|---|---|---|
| MNIST 训练 | 架构 | transformer | **mlp** |
| MNIST 训练 | 批大小 / 权重衰减 | 32 / 0.0 | **64 / 0.01** |
| MNIST 训练 | d_model / num_layers / d_ff | 128 / 4 / 512 | **64 / 2 / 128** |
| MNIST 训练 | patch_size / eval_samples | 空 / 100 | **7 / 200** |
| MNIST 训练 | 每 epoch 打乱 batch | 未暴露 | **true**（新控件） |
| GPT 训练 | 批大小 | 4 | **32** |
| GPT 训练 | d_model / d_ff | 256 / 1024 | **128 / 512** |
| GPT 训练 | flush 间隔 | 0（=不间断） | **1**（实现默认；帮助文本同时由 2 订正为 1） |
| GPT 训练 | 精度预设 f16 的回填值 | param=f32/compute=f16（**master-weights 配方**） | **param=f16/compute=f16/stable=f32/opt=f32**（= `text_train --f16` 的真实语义 `profile_f16()`；原回填值与实际发出的 `--f16` 语义不符，属误导） |
| GPT 推理 | 温度 | 0.8 | **1.0** |

- **控制器侧顺带修复**：`--shuffle-steps` 是**带值**选项（CLI 接受 `true/false/1/0`），
  但 `CLIController._format_arg_value` 对 `bool` 返回空串（bool 一律当开关标志）→
  控制器把 `shuffle_steps=True` 渲染成 `--shuffle-steps ""`，必然触发
  `无效 --shuffle-steps`。现 `shuffle_steps` 分支对 bool 显式转 `"true"/"false"`，
  字符串原样透传（GUI 侧也改为直接传字面量）。
- **保留的有意差异**：MNIST 保存路径默认 `pretrained/mnist_model.bin`（CLI 为 `mnist_model.bin`），
  与 `mnist_infer` 的默认模型路径 `pretrained/model.bin` 一致，属 GUI 的有意选择。

## 五、本轮验收（全部实测）

- `python -m py_compile gui.py cli_controllers.py train_pkg.py` 通过。
- `python bench/gui_cli_audit.py`：三节全 PASS——11 个用例覆盖六个 Tab 的每条可选分支，
  **GUI 发出的 flag 全部存在于对应 CLI 的 `--help`**；反向缺失仅 `--interactive`（有意）；
  六入口**幽灵选项 0**；11 个类**未定义自读属性 0**。
- 真实运行（新参数逐个实跑，非仅 flag 名比对）：
  - `mnist_train` MLP（`--shuffle-steps true` + `--lr-schedule cosine` + `--warmup-epochs`）exit 0；
    Transformer（`--lr-per-epoch` + 全套 TF 参数）exit 0；
  - `text_train` GPT（`--no-cache --flush-interval 1 --grad-log`）exit 0；
    `--resume … --resume-epoch 0 --resume-step 0` exit 0；`--model rapt` exit 0；
  - `tokenizer_train --threads 0` exit 0；`tokenizer_infer --encode … --threads 0` exit 0；
    `tokenizer_infer <text-file> --top 3 --threads 0`（统计模式）exit 0；
  - `text_infer --vocab … --temperature 1.0` exit 0。
- 反证（说明修复必要）：旧 GUI 会发的 `--window 0` 现在报
  `未知参数: --window` + exit(1)。

## 六、未处理（另案）

- `13-refactor-backlog.md` §4 的"五 Tab 声明式字段表"（dataclass + 字段元数据自动生成 UI 与
  collect 函数）**仍未做**——本轮只做一致性修正，字段仍是手工罗列；
  但三个校验脚本已可作为该重构的回归护栏（改完跑一遍即可验证零漂移）。
- GUI 画布节流（13 §5.3）未动。

---

# docs/development 13（13 号清单合并归档，2026-10-01）

## 13 号清单合并归档（原 docs/development/13-refactor-backlog.md，2026-10-01）
- 类型：合并归档 / 演进记录 / 被否决方案
- 内容：

**合并说明**：`docs/development/13-refactor-backlog.md` 的未完成项与后续验证口径已迁入
`docs/todo/18-roadmap.md`（未来方向路线图）；13 号文件自 2026-10-01 起只是重定向 stub
（指向 18 与本节），**并于 2026-10-11 文档整理时删除**（内容已全部在 18 号与本文件，无信息丢失）。
本节归档它原有的两类内容——已完成 / 已被吸收的项，与 2026-10-01 裁定"不做"的项。
13 文首「正确性缺陷已在同轮修复」清单、原 §6 误报对照表、原 §9 代码缩减轮执行表**早已归档在本文件**
（三条既有条目见下），故本节只留指针、不复制正文。

**已完成 / 已被吸收（只记坐标，正文见本文件既有条目）**：

- 13 §10（Tensor 绑定引擎 + 存储多态提案）整体被 `docs/todo/17-unified-tensor-engine.md`
  吸收为 M1–M6，并于 2026-09-30 **全部实施**（访问收口 / InitSpec 声明式初始化 / 批量读写 /
  Matrix 降级 / 内存池契约统一 / `import` + Layer 删 engine 形参）；**M7 未立项**。
  逐期记录见本文件「统一总纲 M1…M6」诸条。
- 13 §9（代码缩减轮）执行表见本文件既有条目「13 §9 已执行：代码缩减轮执行表
  （原位置 docs/development/13-refactor-backlog.md §9）」。
- 13 §9b 已执行部分——位置编码多态统一（`compute_position_encoding.hpp`）、`AttnScoreMask` 族拆分 /
  掩码×位置偏置正交、`NN_TRY`/`NN_TRY_CHECK` 收敛——见本文件既有条目
  「注意力变体拆分 + 位置编码多态统一（2026-10-01…）」与「Layer 层四处收敛…（2026-10-01…）」。
- 13 文首「正确性缺陷已在同轮修复」清单、原 §6 误报对照表、原 §9 执行表分别对应本文件既有条目
  「13 文首…」「13 §6…」「13 §9…」（标题均自称"原位置 docs/development/13-refactor-backlog.md"，
  故 13 的 stub 必须继续声明这些归档条目的来源）。

**2026-10-01 裁定"不做"的项**：

1. **13 §1 线程池归约分块统一** —— 裁定：**不做**。`core_threadpool.hpp` 里 `reduce_chunk_count`
   （确定性分段：边界只由 total 决定）与 `chunk_count`（依赖 `workers_.size()`）两套并存；
   逐元素操作**结果与分块无关**，不存在确定性问题（铁律 #8 的逐字节一致性只约束归约）。
   纯概念一致性收益 < 改动并行内核的风险。若将来仍要做：**禁止**走 PyTorch `ATen::parallel_reduce`
   式"部分和按 `results[tid]` 存放、分块边界依赖线程数"的老路——那正是本项目
   `parallel_transform_reduce` 明确修正过的反例。

2. **13 §5.2 CPU 引擎 f32→f16→f32 冗余双趟** —— 裁定：**不做**。
   `compute_cpu_engine.hpp` 的 `to_matrix` 分支（tensor=f32、请求 P=F16）做 f32→f16→f32 两趟 cast
   再返回 f32，这是**量化模拟语义**（模拟落 f16 后的回读），直接删除会改变行为；收益
   （热路径省一次全量转换）低于语义风险，且当前没有热路径调用方依赖它被消除。
   触发条件：出现真实热路径调用方，且愿意承担语义变更（届时按 API 层显式区分"量化模拟 / 直接返回"）。

3. **13 §9b H3/H4/H5** —— 裁定：**不做**（三条均低成本低收益）。
   H3 解码循环骨架：仅剩三个 `generate()` 的循环体（采样段已合并），因运行态不同而保留；
   H4 精度/形状三元组收敛：`dsl::compute(engine, expr, rows, cols, p_.compute)` 尾部精度参数不宜机械
   改写——RAPT 残差相加用的是**刻意不同**的精度，是反例；H5 掩码/索引构造：注意力侧已随
   `CausalDocScoreMask::prepare` 收敛（两份数值相同的宿主缓冲并为一份），conv 置换 / transformer
   patch 提取仍是"CPU 循环 + `upload_span`"、未声明式化。H5 若将来做，随 header 拆分批次一起做。

4. **13 §9.4 测试公共头 `max_abs_diff` 多份副本统一** —— 裁定：**择机，不单独立项**。
   实测当前有 **7 份定义**（`attn_consistency_test` / `ce_fusion_test` / `conv2d_gradcheck` /
   `fused_gpu_test` / `gpu_test` / `maxpool_gradcheck` / `tensor_expr_test`），其中 conv2d 版带多形状
   守卫（越界返回 1e9），其余无守卫。统一前必须逐点确认每个调用点是否依赖该守卫语义，故
   "不机械统一"是唯一安全前提；留待下次动测试公共头时顺带处理。

5. **13 §10「Tensor 抽象基类 + `CPUTensor`/`GPUTensor` 句柄指针化」选型** —— 裁定：**维持否决**。
   由 `docs/todo/17-unified-tensor-engine.md` §4.2 与
   `docs/development/15-computeengine-refresh.md` §4 裁定：它要解决的 device 互斥收益与存储多态
   重叠，却要 `Result<Tensor>` 203 处指针化 + `span/vector` 元素类型连锁，且**不解决 engine 传参耦合**。

6. **13 §10.6 宿主中转动词改名（`to_matrix`/`from_matrix`/`copy_from` → `to_host`/`from_host`/`upload`）**
   —— 裁定：**方案作废，不再执行**。由 17 §3 D9 改为"**保留名称** + I/O 分组 + grep 审计"：
   `Matrix` 被裁定降级为宿主 I/O 载体（L2+ 禁用即铁律 #12），动词名不再承担"命名即防线"的职责，
   改由 `bench/doc_inventory.ps1` 第 [4] 节分层审计（L2 违规计数）兑现。

**仍未做的项（已迁出本节）**：13 §2（超大 header 拆分）、§3（text_train.cpp 模块化）、§4（gui.py
声明式字段表）、§5.1（F16C 半精度转换快路径）、§5.3（GUI 绘制节流）、§7（checkpoint 优化器状态）、
§8（CI 冒烟）、§9 第 5 条（18 号文档记作 §9.5；`gpt_test` 偶发失败待观察：62 次直跑 1 次失败、
日志被覆盖未定位，已列为 18 号文档 P1-4）、§9b 末尾未编号 bullet（`bench/doc_inventory.ps1` 的
`host_bridge_uses` 计数口径未复核，13 与 17 记 41、18 号文档实测 40），以及 §10 吸收后的 **M7**，
其行动项与触发条件见 `docs/todo/18-roadmap.md`。13 原文件中"仍然生效的工程教训"类正文
（§9 前四条约束、§9b 的 `dsl::compute` 无 2 参重载等）属当前状态说明，不在本节归档（避免与
AGENTS §7、`docs/development/01-compute-engine-development.md` 等现行文档重复）。

**「文档坐标已过期」事实记录（2026-10-01 实测，供后续纠偏）**：13 正文中的行数/份数是审查当时快照，
现已过期——`include/neuralnet.cpp/backend/compute_vk_backend.hpp` **5246** 行（13 写 ≈4700；
A1 段改动前为 5174）、`include/neuralnet.cpp/expr_glsl_gen.hpp` **2746** 行（写 2310；
A1 段改动前为 2424）、`include/neuralnet.cpp/compute_cpu_engine.hpp` **2353** 行（写 2543）、
`src/text_train.cpp` **1746** 行（写 1860）、`gui.py` **1646** 行（非空 1456；13 写 1636，
且所谓"五 Tab"实为 **7** 个 `collect_args` 定义）；`max_abs_diff` 定义实测 **7** 份
（13 与 `src/test_common.hpp:19` 注释均写 9，交接口径记 8——三处不一致，以本实测为准）。
**ctest 计数（口径已统一，2026-10-01 A4/A5 两轮 + 本次审核）**：审查当时实测 **19**
（= 18 个测试目标 + `cnn_test_gpu`），此后 A4 注册 `fusion_custom_layer_example`、A5 接回
`fused_gpu_test` → **20 → 21**（= **19** 个测试目标 + `cnn_test_gpu` + `fusion_custom_layer_example`）。
`AGENTS.md` §12 与 `docs/todo/18-roadmap.md` §2/§7 已统一写 **21**；
`docs/release-notes/v1.5.0.md` 与 `docs/todo/17-unified-tensor-engine.md` 记的 **20/20** 是
**2026-09-30 里程碑当时的计数**（历史值，不回改）。19 / 20 / 21 三个数并存过一阵，根因就是
"是否含 `cnn_test_gpu`"与"后来新增了两个目标"两件互不相关的事叠在一起——**这不构成矛盾**，
但必须写明口径（本次补记）。
---

# 分支归档与 legacy 命名空间整理（2026-10-02，v1.6.0 发布时）

## 两条开发分支迁入 legacy/dev/（本地与 origin 同步）

- 类型：演进记录 / 删除清单
- 内容：`feat/fused-self-register` → **`legacy/dev/fused-self-register`**、
  `feat/fused-runtime-prec-dispatch` → **`legacy/dev/fused-runtime-prec-dispatch`**（沿用
  `legacy/zipt` 惯例，`feat/` 换为 `legacy/dev/`；远端旧 `feat/*` 引用已删除，分支历史完整保留
  于新名字与 main 上的合并提交 PR #14 / #16）。
- 连带：origin 上的旧分支 **`legacy`**（顶端 `b368eea`「clang++ & libstdc++」，非 main 祖先的
  编译器兼容实验）占住了 `legacy` 这个名字，与 `legacy/dev/*` 构成 git **目录/文件引用冲突**
  （D/F conflict，远端拒绝推送）→ 按裁定改名为 **`legacy/compiler-compat`** 释放命名空间。
- 教训：`legacy/<dir>/*` 这类目录式分支命名，前提是 `legacy` 本身不能是分支名；归档前先查
  `git branch -a` 是否存在平铺同名分支。

---

# 统一数据集格式与加载（19 号设计落地，2026-10-05/06）

## 阶段一~三实施记录（KVRecord v2 / .nnvocab / .nndataset / text_train 切换 / 模型 v6）

- 类型：演进记录 / 删除清单 / 覆盖缺口
- **背景**：接入一个新数据集要重写一套读取+解析+缓存——六条互不相通的读取路径（MNIST CSV
  ×2、txt 直读、`.tokcache`、JSON 词表、`.nnpkg`）散落在三个文件里；词表/语料/掩码/缓存四个
  概念互相纠缠。设计与逐条裁决见 `docs/todo/19-unified-dataset.md`（2026-10-05 对话定稿，
  本轮按其 §9 分期落地，分段提交）。
- **阶段一（KVRecord v2 + `.nnvocab`）**：`model_keyvalue_record.hpp` 增 v2 分支——哨兵
  `kvrec_version` 自举 v1/v2 分派（无版本字段 = v1，有 = 按版本读）、字段只记 key/类型/文件绝对
  地址、数据块 `[value_len u64][value]` 按字段序连续排布、`Type::Record` 层层嵌套、未知类型按
  长度跳过。**实施解释一处**：解析器对数据块地址做**连续性校验**（`addr == 期望游标`），整体搬迁
  的字节（换基址复制）硬报错而非静默错值——这是 §10-1/§10-2"哨兵是唯一内联字段、绝对地址不可
  搬迁"两条风险的合流兑现。模型 spec 头仍走 v1 `serialize()`，**字节零变化**。词表 `.nnvocab`
  = `[label NNVC][词表 kvrec v2 @8]`（type/vocab blob/merges 三元组/markers 8 元组）；tokenizer
  `save/load` 切 `.nnvocab`（JSON 写出废弃、读取保留给模型 v5 与迁移器）；`tokenizer_train`
  默认输出 `bpe_vocab.nnvocab`；`text_*`/`tokenizer_*` 的 `--vocab` 只吃 `.nnvocab`，旧 JSON 报错
  指路 `dataset_convert vocab`（新 CLI）。
- **阶段二（`.nndataset` + `nn::Dataset` + text_train 切换 + 模型 v6）**：`.nndataset` =
  `[label NNDS][文件级 kvrec v2][doc 块 arena]`，companion 三层嵌套（vocab 词表 kvrec /
  source sha256（`core_sha256.hpp` 新增）/ license 纯文本 / gen 参数），train/test 子集
  `doc_index` 绝对偏移寻址；**`loss_scope=all` 不写 mask 段、加载端合成全 1**（裁决 #11 的体积
  实施解释）。`nn::Dataset` 只读类 + 生成期公共件（`read_text_docs` / `mark_assistant_span` /
  `encode_docs_parallel`——§4.3.2 三段一致性硬约束**逐字**迁自 text_train）；`dataset_gen` CLI
  （`--test`/`--loss-scope`/`--license`/`--source`）。`text_train` 位置参数改 `<dataset.nndataset>`，
  **删除清单**：`--vocab`/`--test-file`/`--loss-scope`/`--no-cache` 四选项 + `read_file_lines`/
  `parallel_tokenize`/`mark_assistant_spans`/tokcache 全部内联代码（`.tokcache` 机制随之消亡）；
  窗口化 `--seq-len`/`--stride` 保留（裁决 #3：窗口是训练策略）。**模型格式 v6**：内嵌词表改
  `.nnvocab` 字节，`MODEL_VERSION 5→6`，v5 及以下仍读内嵌 JSON（`load_model` 带出版本号 →
  `load_tokenizer_from_model_blob` 按版本分派，旧模型零破坏）。⚠ 版本号抢号提示已在
  `model_serialization.hpp` 记录（roadmap P1-2 也计划 5→6，谁先落地谁占 6）。
- **阶段三（Python 侧 + 文档同批）**：`gui.py` GPT Tab 的 `text_file` 改指 `.nndataset`、删除
  vocab/test_file/loss_scope/no_cache 四行（含 `collect_args`/`build_pack_config` 角色映射）；
  `cli_controllers.py` GptTrainController 删对应四个参数块；`train_pkg.py` GPT 数据角色收敛为
  单一 `train`（`.nndataset` 自带词表/掩码/test）；`bench/gui_cli_audit.py` 用例同批改；
  usage 02/04、AGENTS、12 号盘点、19 号状态行同步。
- **验收（本机 Release+Ninja+clang）**：① build 零告警（`-Werror`）；② **ctest 24/24 → 27/27**
  （新增 `kvrec_test`/`nnvocab_test`/`dataset_test` 全绿）；③ **§8 逐位一致口径**：`dataset_test`
  把 trim 规则、保序并行 encode、逐 doc 掩码、flow+doc_ids+mask 全链与**旧参考实现**（原
  `read_file_lines`/`mark_assistant_spans` 逐字复制进测试）对拍逐位一致（含空 token doc 行号
  保持）；端到端同语料同词表：`dataset_gen` 产出 **3525 tokens / assistant 604/3525** 与改动前
  旧路径基线（tokcache + mark_assistant_spans 实测留存 `build/ds_baseline/`）逐位一致，滑窗样本
  数 56、文档数 173 全同；④ **字节锚 `gpu_stability_probe --steps 20` 前后逐位一致**（本机
  Release 口径 `49ded8cf81d9f8f4`，pre/post worktree 双向对比；默认无 CMAKE_BUILD_TYPE 口径
  `6f8849f14da23110` = 文档锚，post 侧复现）；⑤ `gui_cli_audit.py` PASS（可行动问题 0）、
  `doc_inventory.ps1` L2-VIOLATIONS 0 / HANDWRITTEN-RESIDUE 0；⑥ v6 模型端到端：text_train 保存
  → text_infer 按版本分派加载内嵌词表并生成。
- **顺带查明的既有事实（非本轮引入）**：`text_train` CPU 训练**跨进程非确定**——同配置连续 4 次
  运行 loss 轨迹与保存模型 SHA256 两两不同（4.5077/4.4322/4.2834/4.3736 起步；首轮还观测到
  "优化器: adam 学习率: 0.001 vs 0.0"打印差异）。故 text_train 无法用"模型文件 hash"当字节锚，
  §8 的"训练字节锚"只能落在确定性的 `gpu_stability_probe` 上；数据等价改由④的逐位对拍承担。
  该非确定性在改动前的基线运行即存在（本轮基线取证时发现），根因未定位，如需立项另案。
- **未做（按裁决/分期）**：阶段四 tabular/csv（`dataset_convert csv` + `load_tabular` +
  `mnist_train` 切换）、DataLoader/流式（P2-4 后半）、`.nnpkg` 链路改造（用户裁定"先不管"，
  本轮只做了参数名级同步使其不红）。

---

# dataset_gen 编码进度条（2026-10-06）

## encode_docs_parallel 增可选进度条 + 进度条构件下沉 nn::detail（原位置
## include/neuralnet.cpp/domain_tokenizer_base.hpp 控制台进度条共享构件段、encode_docs_parallel；
## src/dataset_gen.cpp build_subset）

- 类型：演进记录
- 内容：① 控制台进度条构件（`render_progress_bar`/`finish_progress_bar`，`\r` 原地刷新 40 格
  条）从 `Tokenizer` protected static **下沉为 `nn::detail` 自由函数**——`encode_docs_parallel`
  是 namespace 级自由函数，够不着类内 protected 成员；类内 `render_progress_`/`finish_progress_`
  改为薄委托，tokenizer 训练（`bpe_train_impl_`/`parallel_pretokenize`）调用点零改动。②
  `encode_docs_parallel` 增可选参数 `show_progress`/`label`（默认关，`dataset_test` 等既有
  调用零变化）：并行/顺序两条路径都逐 doc `tick()`，原子完成数 + 锁内按 1% 节流渲染（多线程
  并发完成不撕裂 `\r` 行），收尾 `finish_progress_bar`；关闭时零开销（tick 首行即返回）。③
  `dataset_gen` 的 train/test 两个子集编码各开一条（label `Tokenize 训练集`/`Tokenize 测试集`），
  原「Tokenize: N docs ...」单行提示删除（进度条自身带百分比）。多线程编码本身**未改**——
  `encode_docs_parallel` 的分块并行（`nn::parallel_for_samples` 保序归位）原样保留，输出
  逐字节不变（`dataset_test` 回归锁定）。

---

# 训练 step「算子 vs 框架」开销归因（2026-10-06，接同日端到端基准重测）

## 设备侧 kernel 表按阶段归因（剖析器新增）+ torch 设备流探针 + GEMM 形状探针

- 类型：性能 A/B / 演进记录
- 背景：新报告（`docs/benchmarks/2026-10-06-vulkan-vs-torch.md`）主差距 torch 2.14×；
  回答「慢在算子还是框架、框架开销还能否降低、是否没吃满加速特性」。
- **测量 ①（torch 侧设备时间，新增 `bench/probe_torch_gpu.py`）**：CUDA event 口径——
  fp32 整步墙钟 50.70 ms/step、**设备流 50.61 ms/step → GPU busy 99.8%**（host 完全被掩盖）；
  分段（best-of-50）：H2D 0.507 / forward 15.762 / +CE 2.722 / zero+fwd+bwd 49.973
  （推得 backward 31.490）/ Adam 1.666 / zero_grad 0.108。**限制**：kineto 在 CMP 40HX 报
  `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`，torch 的 kernel 名表拿不到，torch 侧只有分段口径。
- **测量 ②（nn 设备时间按阶段归因，剖析器新增能力）**：timestamp resolve 在 fence 之后、
  此时 scope 栈已出栈，故在 `ts_dispatch` **录制期**抓 `nn::prof::current_stage()` 并以
  `interned_pair` 驻留成 `label@stage` 键（`core_profile.hpp` 新增两函数；
  `compute_vk_backend.hpp` ts_dispatch 改 4 行；**仅剖析开启分支内执行**，fused_registry
  hash 不变、默认路径零开销）。20 步实测（`build/prof_bench_stage_20261006.txt`）：
  设备 99.26 ms/step、墙钟 106.8、busy 93.0%。**按阶段**：backward 67.1（其中 fused_mm
  53.6）/ forward 31.6（fused_mm 7.3 + 融合表达式 23.2 含 CE≈5.6）/ optimizer 仅 0.57。
  **按家族**：GEMM 60.9（61%）、融合表达式 fold/elementwise 35.5（36%）、布局 2.9（3%）。
  全仓只有一处 `vkCmdDispatch`（均经 ts_dispatch）→ 表无遗漏。
- **测量 ③（host 框架账）**：host 未与设备重叠的空闲 = 106.8 − 99.3 = **7.5 ms/step
  （7.0%，torch 对应值 ≈0.1）**；copy_from 87.1 ms/step 是等上一步队列（非 PCIe）；
  release_idle_pool_blocks 2.4/step、sync(loss 回读) 2.6/step、data 0.4/step；
  **Adam host 提交 4.5 ms/step 却只有 266 个约 2µs 小 kernel（设备仅 0.57）**。
- **测量 ④（GEMM 形状探针，`layer_bench --gpu --op matmul --forward-only --m/--k/--n`）**：
  方阵 4096³ = 4040 GFLOPS（44% FP32 峰值）；fwd 形状 (16384,64)x(64,192) 1454、
  (16384,64)x(64,8208) 2965、dgrad (16384,192)x(192,64) 1527；**病态——wgrad.QKV
  (192,16384)x(16384,64) = 208、wgrad.ffn1 (256,16384)x(16384,64) = 273 GFLOPS**；
  wgrad.head (8208,16384)x(16384,64) 2941 正常。另测层微基准（含每迭代 I/O，仅供横向参照）：
  causal_attn fwd 7.14 / train 21.78、layernorm fwd 1.02、linear 64-256 fwd 0.145 /
  train 0.687、feedforward train 6.56 ms。
- **根因（已定位，未修）**：tiled matmul 输出块 64x64、`grid = ceil(N/64) x ceil(M/64)`
  （engine 路径与 fused S5 路径同用 `EXPR_MATMUL_BLOCK`）——反传 wgrad 形状
  `M=64..256, K=16384, N=64..256` 只派 **3-4 个工作组**，40HX 约 34 SM 大面积空转。
  按阶段表反推：每 block 反传 11.18 ms 中 4 个 wgrad 约 7.9 ms，x4 block =
  **约 31.6 ms/step ≈ 整步 30%、设备差距 48.7 ms 的约 65%**。
- **结论**：① 框架 host 开销只剩 ~7%（可压项见测量 ③，天花板远小于差距），
  **差距 87% 在设备侧算子**；② 时间饱和度并不低（nn busy 93% vs torch 99.8%），
  低的是**单位时间产出**（77 GFLOP/step：nn 0.78 TFLOPS vs torch 1.52）与
  **派发并行度**——「没完全利用加速特性」实锤在 wgrad 工作组饥饿；
  ③ 另一项未兑现：`--f16` 反而慢 11%（118.4 vs 107.1），torch fp16 快 1.65×。
- **修复方向与估算（未实施，供立项）**：P0 修 wgrad 派发（WG 持久循环扫多输出块 /
  K 切分二次归约 / 网格过小时切小 tile），约 -26~28 ms/step → 107→约 80（2.14×→约 1.6×）；
  P1 CE 多趟扫描 5.6 → torch 2.7；P1 Adam 每张量合并提交（4.5 ms host）；
  P2 host 7%（release_idle 降频、loss 回读重叠）；P3 f16 提速兑现。
  验收参照：`layer_bench --op matmul` 形状探针、`expr_gpu_test`、
  `gpu_stability_probe` 字节锚、ctest。
- **产物**：`build/prof_bench_stage_20261006.txt`、`build/torch_gpu_20261006.txt`、
  探针 `bench/probe_torch_gpu.py`；剖析器改动 `include/neuralnet.cpp/core_profile.hpp`、
  `include/neuralnet.cpp/backend/compute_vk_backend.hpp`。

---

# IR-C P2 机制落地：写穿物化 + 多输出 kernel（2026-10-06）

## fuse 双规则 + ExprSpec.extras + eval_expr_multi_into（Adam 金刚石 GPU 端到端 err≈2e-7）

- 类型：演进记录（「IR-C 定位修正与恢复立项」的 P2 机制部分；**Adam/CE 真实接线与
  NN_PROFILE 收益核验在下一阶段**）
- 内容：
  1. **IR：`ExprSpec.extras`**（P2 多输出，聚合第 8 字段）——主输出（instrs.back().dst）
     之外额外物化的寄存器列表，进 key / expr_spec_equal / validate（仅纯逐元素、
     无 matmul/fold/归约、寄存器互异且被定义、≤4）。**canonicalize 四个 pass 全部
     支持 extras**：simplify 的定义指令走输出指令豁免（不折叠/不 alias）、DCE 播种
     存活、CSE/renumber/allocate 重映射（allocate 的 liveness 保护到 kernel 末）。
  2. **fuse_expr_graph 双规则**：① P1 链语义（matmul 头链等非纯逐元素 F32 场景，
     单输出、中间量契约不变）；② **P2 分量融合**——纯逐元素 F32 同形状节点经依赖边
     连通成组（union-find，小根确定性），**多消费者/菱形不再是融合边界**（写穿全输出：
     成员序 [0..n-2] 进 extras、尾成员为主输出）；成员引用任意已并入成员经寄存器直替；
     预算不足分裂为多 kernel（跨 kernel 依赖经写穿输出读取）。执行侧补**稳定拓扑序**
     （分量分裂与 P1 kernel 交错时列表序不保证依赖序）。
  3. **执行链**：`ComputeEngine::eval_expr_multi_into`（新 NVI + virtual
     `eval_expr_multi_into_impl`，**virtual 51 → 52**）——GPU = V0 查表 +
     `run_fused_gpu` 追加 `extra_outs` 绑定（`output_override` 同款精度视图）；
     CPU = 解释器逐槽求值（extras 用 `-(-x)` 精确复制链提升为主输出，Neg∘Neg 逐位恒等）。
     `compute_into` 录制段**目标传递**：P1 的拒绝解除，节点输出 = dst 既有张量
     （Adam 的 p 更新直写参数存储，张量身份不替换）。单输出入口对 extras spec 防御性
     拒绝（防描述符绑定数错位）。
  4. **测试**：`expr_graph_test` 12/12（链/三节点/多消费者**写穿融合 1 kernel 3 输出**/
     边界/预算分裂/确定性/S5 全保留，多输出槽序与逐节点参考逐项对拍）；
     `expr_fuse_test` 11/11（新增 **Adam 金刚石**：m/v 状态写穿 + d 中间量 + p 目标
     传递 → 单 kernel 4 输出，GPU vs CPU err=2.4e-07 级；目标传递 dst 写穿用例）。
- **两个教训（坑录）**：① **聚合发射漏字段静默错值**——`fused_generate.hpp` 的
  ExprSpec 位置式发射停在旧字段表（fold 后无 extras），registry spec 丢 extras →
  管线布局少输出绑定 → 多输出 shader 绑定错位、**结果错但不报错**（tail 取到成员 A
  的值，err≈1.5）；结构化聚合初始化的字段清单必须随 struct 扩展同步（与 matmul
  batch 漏发同族）。② **分量簿记漏 push**——`comp_append` 成员下标簿记漏
  `mem.push_back(bi)`，3 节点以上深链的依赖查找落空退化为外部输入（2 节点/扇出
  形态恰好测不出）；融合簿记与成员表同源更新 + 深链用例才能抓到。
- **验收**：全量构建 84 目标零告警（-Werror）；**ctest 29/29 全绿**；CPU 字节锚
  `49ded8cf81d9f8f4`、GPU dev2 `5e992f668b4209c8` **逐位不变**（机制未接入训练路径，
  数值面零变化）；`doc_inventory`：virtual 52/52、L2-VIOLATIONS 0、HANDWRITTEN-RESIDUE 0；
  闭合世界自登记实测（expr_fuse_test 收集器）：91 → 100（+9 多输出/链结构）。

---

# IR-C P1 恢复完成（2026-10-06）

## expr_graph + begin_expr/end_expr + GPU 图执行 + 双测试恢复（ctest 27 → 29 全绿）

- 类型：演进记录（承接同日「IR-C 定位修正与恢复立项」的 P1）
- 内容（相对 git `8f2990f^` 的移植 + 两点加固）：
  1. **图核**：`expr_graph.hpp` 恢复（`ExprGraph`/`fuse_expr_graph`/`recording_graph_owner`
     + P2-12 图级计划缓存）+ `Tensor::virtual_tag_` 占位标记恢复（公共元数据访问器，
     存储仍守铁律 #11）。加固：**融合成员精度一致门禁**（旧版无精度维度，混精度链
     拼进单 kernel = 静默错值）+ `ExprGraph::owner` 跨引擎 fail-fast。
  2. **引擎 API**：`ComputeEngine::begin_expr/end_expr`（**virtual 49 → 51**，
     doc_inventory 口径；先例 = M6 的 import_impl 48→49）；`eval_expr`/
     `eval_expr_reduce` NVI 顶部录制拦截（占位 Tensor 经 `create_tensor` 出生即
     绑定、tag = 节点号+1；`record_graph_node_`）；`eval_expr_into` 拒绝录制段
     （目标传递待 P2 写穿，dsl 侧同款守卫双保险）。
  3. **引擎实现**：CpuEngine 普通运行 no-op（CPU 行为不变）；NN_EXPR_SCAN 下
     begin 开录制、end **融合并登记复合 spec**（闭合世界两端一致）。GpuEngine
     begin/end + `execute_fused_graph`（P2-12 缓存 + 占位保留集合 + 被融合中间
     占位即时释放归池）+ `run_graph_kernel_into_`（output_override 写回 tail，
     覆盖 fold/fold_k 形态、S5 matmul 段、vector_out、精度变体 `#a`/`#x`/V0；
     精度视图同 eval_expr_into_impl）。
  4. **dsl**：scan 分支录制感知（入图 + tag 占位；锚点 odr-use 两路保留，编译期
     可达覆盖不因录制分支丢失）+ `dsl::start_expr/end_expr`（ExprBlock 语法糖）恢复。
  5. **测试**：`expr_graph_test` **12/12**（纯 CPU：链融合/三节点/边界/上限/确定性/
     依赖识别/输入序/S5 两形态/emitter）+ `expr_fuse_test` **9/9**（GPU 端到端：
     3 节点链融合逐位 **err=0.00**、P2-12 缓存重复命中、独立分支 2 kernel、S5
     matmul 链、未录制回归、vec4 路径、compute_into 契约守卫）。**闭合世界自登记**：
     `src/expr_fuse_shapes.cpp`（收集器 TU）NN_EXPR_SCAN 静态初始化 dry-run 录制段，
     scan 端 end_expr 把复合 spec 登记进本目标专属 registry（`nn_enable_gpu_fusion`，
     实测收集 91 → 96 = +5 测试结构）；链定义集中在 `expr_fuse_shapes.hpp` 单一
     事实源，运行期/收集期同调一组函数、不会漂移。
  6. **用例语义**：录制段内同次求值的输入须同引擎（`bind_check_` 硬规则——混引擎
     fail-fast 是设计本意，旧测试靠当年的未绑定张量绕过）；测试外部输入先
     `import` 到 GPU（GPU 常驻训练的真实形态）。
- **验收**：全量构建零告警（-Werror）；**ctest 29/29 全绿**；CPU 字节锚
  `49ded8cf81d9f8f4`、GPU dev2 `5e992f668b4209c8` **逐位不变**；scan 库内 registry
  **91 结构不变**（"内容未变化，保持 mtime"）；`doc_inventory.ps1`：virtual 51/51、
  L2-VIOLATIONS 0、HANDWRITTEN-RESIDUE 0。
- 顺带修复：`652c53d`（HEAD「易用性审查」）误删整个 `examples/`（CMakeLists /
  AGENTS / 20 号文档仍在引用，20 号文档同提交还在称 `fusion_custom_layer*` 为
  "现有样例"）→ 从 `652c53d^` 恢复 5 个文件；`-DNN_ENABLE_TESTS=ON` 此前在
  configure 阶段即失败，现恢复可用。
- 待办（**已于 2026-10-06 随 P2 收官完成**）：`docs/todo/02、03、12`、
  `docs/usage/03`、`docs/introduction/04` 中"IR-C 不存在/未采用"的当前状态表述。

---

# IR-C 定位修正与恢复立项（2026-10-06）

## 「否决」修正为「暂时废弃、可恢复」，按 git `8f2990f^` 旧实现恢复 + 通用扩展

- 类型：演进记录 / 用户裁定（覆盖 18 号路线图判据）
- **裁定**：① IR-C 不是被否决——是当年实现不完整而**暂时废弃、可恢复**；② **设计原则：
  不为单一操作加专用原语**（Adam 多状态 foreach/arena、CE 专用 shader 之类），
  通用机制优先（图融合扩展 / fold 通用折叠）；③ 授权按旧分支实现合并回主干。
  18 号路线图「不得复活显式录制 API」判据被本裁定覆盖：**先按旧形态恢复**
  （`begin_expr/end_expr` 显式录制），自动作用域 + 逃逸检测作为后续演进
  （= 03 §5.3 重新立项前提的第 1 步路线）。AGENTS §3/§7/§11 措辞同批改。
- **旧实现盘点**（删除于 `8f2990f`，最后存在 = `8f2990f^`/`e936f4d`；研究副本
  `build/ir_c_study/`）：`expr_graph.hpp` 480 行（`ExprGraphNode`/`ExprGraph`/
  `fuse_expr_graph`/`recording_graph_owner` + P2-12 图级计划缓存
  `graph_cache_key`/`FusedKernelPlan`/`instantiate_plan`）+ `src/expr_graph_test.cpp`/
  `src/expr_fuse_test.cpp`；引擎 API `ComputeEngine::begin_expr/end_expr`、
  `dsl::start_expr/end_expr`（`ExprBlock`）、`GpuEngine::execute_fused_graph`、
  `Tensor::virtual_tag_`、演示层 `FusedChainLayer`。
- **能力边界（"做得不好"实指，`fuse_expr_graph` 判据 = expr_graph.hpp:180-211）**：
  只拼「纯逐元素、同形状、tail 恰一个消费者、B 以 Linear 视图引用 tail、双方无归约、
  B 无 matmul」的链（A 可带前置 matmul 段 = S5）→ ① 无写穿物化（中间量逃逸/多消费者
  直接放弃）；② 单输出（只有 tail）；③ 归约硬边界；④ 无跨链/跨张量批量派发
  （37 条独立链仍 37 个 kernel）；⑤ 显式录制、无自动作用域/逃逸检测；⑥ CPU 录制
  普通运行 no-op。2026-09-19 的五条否证（归约边界、eager 交错、缓存逃逸、22 flush 点、
  CPU 零收益）针对的正是这个能力面——**否证的是当年实现，不是图融合路线本身**。
- **对两个性能目标的映射（诚实口径）**：as-is 只能把 Adam K3(delta)+K4(p+=) 折成一个
  （4→3 kernel/张量；且这步不恢复 IR-C、用单条 `compute_into` 表达式同样能拿到）；
  K1/K2 的 m/v 状态**写穿给 K3**、以及**跨张量合并**都需要扩展——「写穿物化/多消费者 +
  多输出 kernel」（Adam 每张量 1 kernel，266→~37）与「跨链批量派发」（37→3，可选后置）。
  CE 的 max→denom 是归约数据依赖，IR-C 管不了：修法 = **online-softmax 单趟**，
  走既有 fold 通用折叠机制加一个 spec（(m,l) 运行态与注意力 fold 同款），不新增原语类。
- **分期**：P1 恢复移植（expr_graph + begin/end_expr + virtual_tag 通道 + GpuEngine 图
  执行 + 测试 + scan 集成，适配 M1-M6 后 Tensor/Result/NVI 基座）→ P2 通用扩展：写穿
  物化 + 多输出（Adam 每张量 1 kernel）→ P3 跨链批量派发（可选）→ P4 CE online-softmax
  fold。**P0（wgrad 派发饥饿 −26ms/step）不属 IR 范畴，独立推进**。
- **验收口径**：ctest 27 全绿 + 新 graph/fuse 测试恢复、`gpu_stability_probe` 字节锚、
  scan 双 hash、`bench/doc_inventory.ps1` 门禁、融合前后 loss 逐位一致（链融合不改
  数值语义）、optimizer 设备 kernel 数/耗时对照（NN_PROFILE 设备表）。

---

# IR-C P2 收官：Adam/AdamW 接线 + NN_PROFILE 收益核验 + 混精度闭合世界通用降级（2026-10-06）

## 承接「IR-C 定位修正与恢复立项」的 P2；并修复 f16 GPU 训练崩溃（0xC0000374）

- 类型：演进记录（性能收益核验 + 缺陷修复 + 文档校订）
- **1. Adam/AdamW 接线（P2 落地到训练路径）**：`Adam::step` / `AdamW::step` 用
  `engine.begin_expr()` + `ExprSegment`（RAII 收口，作用域退出/早退都兜底 `end_expr`）把
  **整步**录成一张图：每参数张量 K1（m 更新）/K2（v 更新）状态写穿 + K3（delta）+
  K4（`p += delta` 目标传递）在 f32 下融合成**每张量 1 个多输出 kernel**
  （`GpuEngine::execute_fused_graph` → `run_graph_kernel_into_` / `eval_expr_multi_into`）。
  CPU 普通运行 `begin_expr/end_expr` 仍是 no-op（数值面零变化）。
- **2. NN_PROFILE 收益核验（验收口径「optimizer 设备 kernel 数/耗时对照」）**：同配置
  （`text_train build/tinystories_bench40.nndataset --model gpt --gpu=40HX --optimizer adam
  --max-steps 20 --d-model 64 --num-layers 4 --num-heads 4 --d-ff 256 --seq-len 256
  --batch-size 64`）设备侧表：
  | 口径 | P2 前（`prof_bench_stage_20261006.txt`） | P2 后（`prof_p2_adam_20261006.txt` / `prof_p2_adam_final.txt`） |
  |---|---|---|
  | `fused@optimizer` 调用数 | 5320（**266/step**） | 1330（**66.5/step**） |
  | `fused@optimizer` 设备耗时 | 11.35 ms（**0.57 ms/step**） | 4.58 ms（**0.23 ms/step**） |
  即 **kernel 数 4.0×、设备耗时 2.5×**（单 kernel 从 ~2.1µs 小 kernel 变为 ~3.4µs
  多输出 kernel；host 提交次数同步下降）。⚠ 口径注记：device 侧数字与构建类型无关，
  host 侧数字在 Debug 构建下被显著放大（同配置 Debug 墙钟 3250ms / optimizer inclusive
  1012ms，Release 2135ms / 95.7ms）——对照只取设备表，host 侧数字须同构建类型比。
- **3. 缺陷 ①：`NN_TRY_CHECK` 重复求值 → f16 GPU 训练堆损坏（0xC0000374）**。旧展开
  `if (!(x)) return std::unexpected((x).error());` 把 `x` 求值**两遍**；对带副作用的实参
  （`Adam::step` 的 `NN_TRY_CHECK(seg.end())`）第二遍会重新进入 `end_expr`——此时录制
  owner 已被第一遍 move + reset，故第二遍**返回成功**，随后对 success 的 `std::expected`
  调 `.error()`：Debug STL 触发 `expected` 断言（MSVC），Release 读已析构的 Error 字符串
  → 堆损坏。**根因与"失败"无关**：底层失败（见 4）只是把第二遍求值这条路径打开。修法 =
  宏改单次求值 `if (auto&& r = (x); !r) return std::unexpected(r.error())`（`auto&&` 不
  引入拷贝，兼容不可拷贝 Result 载体），与 `core_errors.hpp` 文档承诺的"展开后与手写形态
  逐字等价"重回一致；回归锁 `error_macro_test::test_try_check_single_eval`（失败路径与
  成功路径各断言求值次数 == 1，并断言用的是第一次求值的错误）。
- **4. 缺陷 ②：混精度 profile 的闭合世界硬报错 → 通用降级（非单算子特例）**。
  `text_train --f16`（`profile_f16` = optimizer:F32 / param:F16）实测
  `GpuEngine::end_expr: 融合 kernel 未命中 AOT 融合 shader（闭合世界）… key=60bb980ae730649a`。
  根因：**融合分组本身依赖精度**（P2 写穿组要求成员全 F32；P1 链要求成员同精度），于是
  f16 下 Adam 的 K3/K4（F16）落成 P1 尾链 `{K3,K4}`——这个复合结构 f32 扫描期根本不存在
  （f32 下 K1..K4 属同一 P2 分量）。构建期按精度组合枚举是 2^4 爆炸，不可取。**通用机制**：
  `GpuEngine::execute_fused_graph` 对多成员复合 kernel 先查可用性
  （`composite_kernel_available_`：`(key, psig)` 变体选路 / 写穿形态要求全 f32 输入输出），
  不可用则拆回成员逐个派发（`dispatch_graph_member_`，成员 = 已登记的**单节点**结构）——
  闭合世界仍然成立（只跑 AOT 已登记 shader；**未登记的单节点结构照旧硬报错**），该段退化
  为 IR-C 之前的"1 节点 = 1 dispatch"。诊断开关 `NN_IRC_TRACE=1` 打 `[ir-c][degrade]`
  （与 `NN_PREC_TRACE` 同哲学：降级是优化取舍而非错误，但必须可观测）。实测
  `NN_IRC_TRACE=1 f16_precision_test`：**760 条 `复合 kernel(2 成员, extras=0) 未登记 →
  逐成员派发`**（正是 K3/K4 尾链）→ GPU f16 端到端恢复全绿。
- **5. 回归覆盖**：`f16_precision_test` 的 GPU f16 端到端即缺陷 ② 的回归锁（修复前堆损坏 /
  闭合世界报错，修复后 ALL PASSED，且 `NN_IRC_TRACE=1` 可观测降级发生）；`error_macro_test`
  新增单次求值用例锁缺陷 ①。
- **6. 验收（本机 clang，Debug 与 Release 双构建）**：
  | 项 | Debug（`build/`） | Release（`build-release/`） |
  |---|---|---|
  | ctest | **29/29 全绿** | 关键测试（f16_precision / expr_fuse / expr_graph）单跑全绿 |
  | CPU 字节锚 `--steps 20` | `6f8849f14da23110` | `49ded8cf81d9f8f4` |
  | GPU 40HX 字节锚 `--steps 20` | `8ef51b2927253c50` | `5e992f668b4209c8` |
  两个构建的各锚均与既有文档锚**逐位一致**（Release 锚 = 本仓性能档口径，见另条「Release
  构建下字节锚与 init-hash 均偏离 AGENTS §12 文档锚」的说明）⇒ **IR-C 接入训练路径后
  f32 数值面零变化**。scan 注册表 **92 结构不变**（构建日志 `内容未变化，保持 mtime`；
  库内口径）；`bench/doc_inventory.ps1`：virtual **52/52**、layer-called **22**、
  `L2-VIOLATIONS: 0`、`HANDWRITTEN-RESIDUE: 0`；`text_train --f16 --gpu=40HX` 3 步端到端
  跑通（loss 9.0279 → 8.9614 → 8.9176；f32 对照 9.0197 → 8.9681 → 8.9041，差在 f16 存储
  舍入内）；`expr_fuse_test` 11 → **12 用例全绿**（含 P2 写穿多输出、融合 vs 未融合逐位
  一致 diff=0）。
- **7. 文档校订（回收 P1 条的遗留待办）**：AGENTS §3/§7/§11/§12 与 `docs/development/`
  `02-operator-fusion.md`、`03-ir-optimization.md`、`12-compute-engine-inventory.md`、
  `docs/usage/03-compute-engine-usage.md`、`docs/introduction/04-innovative-designs.md` 中
  "IR-C 不存在 / 未采用 / 已移除"的**当前状态**表述全部改为现行（含 P1/P2 规则、GPU 真
  录制 / CPU no-op、写穿多输出、闭合世界降级与 `NN_IRC_TRACE`）；历史取舍正文仍留在本文件。
- **8. 剩余（未做，后置；2026-10-06 用户裁定本轮收口，作为后续独立立项）**：P3 跨链批量派发（66.5 → ~3 kernel/step，可选）、P4 CE
  online-softmax fold（归约数据依赖，IR-C 管不了，走 fold 通用机制）；P0（wgrad 派发饥饿
  −26ms/step）仍在 IR 范畴之外，独立推进。

---

## 勘误补记（同日，承接上条）
- 上条 §2 的 host 侧对照曾只取 Release 口径；Debug 构建的 host 侧数字（optimizer
  inclusive 1012ms、墙钟 3250ms）**不是回退**——device 侧表完全一致（1330 调用 /
  4.58ms），差异来自 Debug 未优化 STL 与 IR 处理的 host 耗时。跨构建类型比较一律用
  `NN_PROFILE` 的设备侧表。
- **构建期 `[skip] 分派变体 6b634571443ba779#x 的形态暂不支持生成` 的语义澄清
  （2026-10-06，此前多轮构建均出现但未记录）**：该 key 是 **P2 的 Adam 多输出
  （extras）复合结构**（6 输入 + extras={2,1,3} 写穿 m/v/delta、8 个 rparams），不是
  形态缺口——**多输出结构按设计只生成 V0 变体**（`expr_glsl_gen` 的 P2 约束：
  `#x`/`#a` 均跳过，f16 复合结构由 `execute_fused_graph` 的逐成员降级覆盖），但
  `fused_generate` 的发射器把"发不出 `#x`"一律打成"形态暂不支持生成"，把设计内跳过
  伪装成了报错文本。已修正为两类成因分打（extras → "[skip] 多输出(extras) 结构 …
  按设计不生成（仅 V0；混精度走逐成员降级）"；其余才是真形态缺口提示）。运行期
  影响为零（该结构本就只走 V0 全 f32 路径）。

---

# Windows CRT abort() 模态弹窗静音（2026-10-06，用户截图报障）

- 类型：缺陷修复（测试/脚本可用性）+ 工具链坑（已录入 `docs/todo/08` §5）
- **现象**：Windows Debug CRT 在 `abort()` 时弹**模态**"Microsoft Visual C++ Runtime
  Library — Debug Error! abort() has been called (Press Retry to debug the application)"
  对话框（用户截图实测，来源 `error_macro_test.exe` 的 abort 负例子进程）。两个要害：
  ① **GUI 弹窗不进 stdout/stderr**——测试输出/ctest 日志/审计 grep 全看不见它
  （"测试全绿但用户看到报错"）；② **模态阻塞**——子进程卡在弹窗等人点，`std::system`
  的父进程连带挂起，ctest 被卡死。此前未记录（报错不在任何日志流里）。
- **修法**（通用，非测试特例）：abort 唯一落点 `core_errors.hpp::fail_abort` 在
  `std::abort()` 前静音 CRT 弹窗与 WER 上报——`_set_abort_behavior(0,
  _WRITE_ABORT_MSG | _CALL_REPORTFAULT)` + `_set_error_mode(_OUT_TO_STDERR)`
  （Windows 宏守卫，其余平台 no-op）。错误信息本函数已打印到 stderr，可见性不变；
  abort 语义保持"打印后立即终止"。`fail_exit`（`std::exit`）无此问题。
- **验收**：`error_macro_test` abort 负例子进程直跑 <10s 内以 **exit code 3**
  终止（此前挂弹窗）、stderr 错误行完整；`error_macro_test` 全套 0 failures、
  0.5s 跑完（修复前 ctest 里 3.5s 的差额即等人点弹窗）。

---

# CE online-softmax 单趟 fold（P4）：Transpose 视图 + FoldAnchor 自登记 + fold 生成器通用优化（2026-10-06）

## 承接「IR-C P2 收官」§8 的 P4；用户裁定"表达式自登记、不为单一操作加专用原语"

- 类型：演进记录（性能 A/B + 数值验收 + 测量方法坑）+ 锚点重置
- **1. 定位与设计裁定**：CE 的 online-softmax 归约带跨块数据依赖（m/l 跨块进位），
  IR-C 图融合管不了；按用户裁定"不想加太多专门针对某一个操作设计的原语"，走
  **FoldSpec 通用折叠**（P-C1/C2 机制）——全程零 CE 专属 shader / 原语 / 视图。
  旧 5-kernel 链（col_max 原语 + denom/loss_vec/grad 表达式 + Σ）重写为 **3-kernel**
  （fold lse 单趟 + loss_sum 归约链 + grad），logits DRAM 趟数 **3 → 2**；grad 公式
  由 `exp(x−m)/denom − onehot` 等价改写为 `exp(x−lse) − onehot`（少一次除法与广播）。
- **2. 通用机制新增（全部是通用面，非 CE 特例）**：
  - `ExprViewKind::Transpose = 15`（`expr_spec.hpp`）：转置直读 `data[col*param + row]`，
    param = 行距走运行期视图参数 `vp`、**不进 key**（同结构不同 stride 共享 shader）；
    fold 与非 fold 双生成路径 + CPU 解释器两处 switch + 越界守卫
    （`(K-1)*stride + rows <= size`）；助手 `expr::trans(stride)`。
  - `make_fold_logsumexp(k, input_view)`（`expr_fold.hpp` 第 4 个通用 fold 样例）：
    body 与 `make_fold_softmax_denom` 同构（online m/l rescale），finalize 多指令链
    `lse = m + log(l)`（reg 10/11 链内临时）。
  - **`FoldAnchor` 自登记**（`expr_dsl.hpp`）：fold 值构造与 `FusedAnchor` 同机制入
    `anchor_registry()`——用户裁定"新表达式一律自登记，`tools/scan_exprs.cpp` 不得
    随新表达式改动"；本 P4 scan_exprs **零 diff**（此前误改显式登记块，已撤销）。
    注意力 fold 5 组合仍在 scan 显式块（存量不动）。
  - finalize 校验一般化（`expr_spec.hpp`）：多指令 finalize 链内临时允许 def-before-use
    （`fin_def`）+ 精确跨迭代污染规则（`fin_write_max`）；原"写出寄存器不得再读"的
    保守规则会拒绝这类合法 finalize（fold v1 状态自拷贝语义由新规则精确保留）。
  - fold v1 生成器两个**通用**优化（`expr_glsl_gen.hpp`）：① **归约输入内联**——只被
    归约消费的元素临时量在归约块循环内逐块重算，消 `e[n][128]` 私有数组物化；
    ② **循环剥离 + 预取软件流水**（`v=src(0u); for(kb=1..){ vn=src(kb); acc+=v; v=vn; }
    acc+=v`）消 64 级 L2 串行延迟链。累加序不变 ⇒ CPU/GPU 逐位同序（对拍口径不变）。
- **3. 数值验收（用户裁定「P4 另加逐位不变验收」：允许移锚 + 逐用例对拍 + gradcheck）**：
  `ce_fusion_test::run_old_new_parity` 新旧 CE 逐用例对拍 **8 形状**（K=129/300/1000/4096
  分块边界 + scale-40 压力值）× CPU/GPU，容差 `1e-6*(1+|ref|)`：**grad_err ~1e-8**、
  loss 相对误差 ~2e-7（scale-40 压力下绝对值 ≤1.1e-5）；3 个旧形 CE 单测改写为现行链形
  （lse fold + loss_sum + grad 逐段 vs 参考）；gradcheck 全绿；ce_fusion_test CPU+GPU
  ALL PASS。
- **4. 性能口径（同窗交错 / 同窗复测；测量坑见 §5）**：
  - kernel 数 **5 → 3/step**（设备表实测 60 calls/20 steps）；
  - 新侧设备耗时实测（GPU 40HX，logits (8208,16384)，同窗 5 次复测稳定）：
    `ce.fold` 2.95 + `ce.grad` 2.92 + `ce.lossum` 0.02 = **5.93 ms/step**；
  - 旧侧**分项标定**（剖析器晚于旧路径、无同码 A/B；`build/head_ab/` 留有
    git worktree @HEAD 的基线二进制供后续补测）：col_max ≈ 1.5 ms（`layer_bench` 同形
    `col_reduce_sum(8208,16384)` = 1.495 ms 标定）+ denom ≈ 2.9 ms（与 fold 同工作量的
    一趟 exp+归约）+ grad 2.92 ms（同构 kernel 实测）+ loss_vec/Σ ≈ 0.1 ms
    ≈ **7.4 ms/step** ⇒ P4 设备时间 **≈ −20%**，主体 = col_max+denom 两趟（~4.4 ms）
    融为 fold 一趟（2.95 ms）；即便保守按"持平"读，确凿收益仍是 kernel 5→3 与
    logits 趟数 3→2。
- **5. 测量方法坑（跨时段设备计时漂移；已录 `development/08` §5）**：同二进制同 shader
  （注册表 mtime/内容跨时段不变证实）的 CE 3 kernel 会话早期测 **13.2 ms/step**、晚期
  稳定 **5.93 ms/step**（2.2×），同时段非 CE kernel 一致 ⇒ 非时钟/热。已排除三个假设：
  模型形状混淆（`--d-model` 默认 128 与显式 64 的对照混入，同形化后仍 5.93）、精度变体
  分派（`--f16` / `--precision-stable f16` 实测 fold 仍 ~3 ms）、shader 版本（注册表
  内容一致）。**结论口径：性能数字只认同窗交错测量**；本条全部对照均为同窗
  （新旧二进制同窗交错 2+2 轮、新侧同窗 5 复测）。
- **6. 锚点重置（用户裁定：数值变更允许移锚并更新文档）**：
  | 锚 | 旧 | 新（P4 后） |
  |---|---|---|
  | CPU 字节锚（Debug） | `6f8849f14da23110` | **`8809f07142e29472`** |
  | GPU dev2 字节锚（Debug） | `8ef51b2927253c50` | **`c69a3378c7dcc037`** |
  | CPU 字节锚（Release） | `49ded8cf81d9f8f4` | **`f360f9bdbbfbe12c`** |
  | GPU dev2 字节锚（Release） | `5e992f668b4209c8` | **`4103194cb9e3e230`** |
  均 `gpu_stability_probe` 双轮 PASS。**init 锚逐位不变**（HEAD worktree 同探针对照：
  `a22e807ee05ec3ac`/`2d7ecb570d91fd37`/`27f50137087d09b5`/`daec9e935f9b4c68`/
  `d31c2e46bda6f3b5` 全同）⇒ P4 未动初值；顺带收口「Release 构建下字节锚与 init-hash
  均偏离 AGENTS §12 文档锚」条的待办——§12 init 锚按 Debug/Release 双口径重刷
  （1950 条留下的漂移，与 P4 无关；Release init 新值 `3e982d018c00d94a`/
  `d8c8457f0053c4f9`/`70c4c085052268b2`/`f8c2d892f5834239`/`7002b986b84f679f`）。
- **7. 收尾验收**：ctest **29/29**（Debug 全量）+ Release 全量构建与 ctest；
  `text_train --f16` 冒烟跑通；registry **93 结构**（92 + LSE trans fold；其 linear
  视图变体未被消费、按不消费不登记处理）；`doc_inventory` / `doc_align_audit` 门禁过。

# matmul 细 tile 变体：P0 wgrad 派发饥饿落地（2026-10-07）

## 承接「设备侧 kernel 表按阶段归因」条的修复方向①（P0）；键后缀 `#f`

- 类型：性能 A/B + 演进记录（含两条被否决设计）
- **1. 落地形态**：网格饥饿的纯 matmul 结构（spec.matmul、无 fold、raxis<0、
  仅 V0 f32）在构建期多发一份**细 tile 变体**（键 = `key#f`，
  `EXPR_MM_FINE_SUFFIX`；`fused_generate` 现打「N 条细 tile 变体」计数）。
  运行期 `run_fused_gpu` 在 **64-tile 网格工作组数（含 batch 维） <
  `EXPR_MATMUL_FINE_MIN_WGS`（32 ≈ 40HX SM 数）** 时改派 `key#f` pipeline
  （同一 spec 元数据/PC 布局/绑定数，仅网格粒度不同；未登记 = f16 精度变体
  键、设备未注册、或 `NN_MM_NO_FINE=1` → 留在 64-tile 路径，降级非错误）。
  细 tile：输出块 64×64 → 16×16、local_size 16² → 8²、每线程 4×4 → **2×2
  （R=2）**。OP 级同病同修：`matmul_tiled.comp` 用 `-DNN_MM_FINE=1` 编第二份
  （CMake `nn_embed_shader(matmul_tiled_fine …)`），`matmul_gpu` 按同一判据
  选 pipeline；`NN_MM_FINE_NAIVE=1` 另可强制走 matmul.comp 对照（A/B 用）。
  注册门禁的「带 `#` 需 16 位存储」判定豁免 `#f`（它是 f32 变体，与 `#b` 同列）。
- **2. 三轮设计 A/B（`layer_bench --op matmul` 形状探针，同窗交错 2 轮，
  wgrad.QKV (192,16384)×(16384,64) / ffn1 (256,16384)×(16384,64)）**：
  | 设计 | 结论（GFLOPS，QKV/ffn1） |
  |---|---|
  | 不修（64-tile，1~4 WG） | 205~206 / 274~276（基线） |
  | ① R=4 等比缩块（16×16 块 / 4×4 线程） | **169 / 222 —— 比不修还差** |
  | ② R=1（16×16 线程，线程数 = M×N） | 305~415（≈1.5×） |
  | matmul.comp naive 对照 | 323~440（≈1.6×，略优于②） |
  | **③ R=2（8×8 线程，定案）** | **302~405（≈1.5×）** |
  **教训（线程数不变量）**：输出张量固定时，R=4 每线程 16 输出 → 线程总数
  恒 = M×N/16（wgrad 仅 768）；把块缩成 16² **不会多出线程**，只是把同样的
  线程拆成更多半 warp 工作组，占用率反崩。R² 与网格是硬取舍：R=4 网格只有
  1~4 个（M×N/4096），R=1 网格 48 个但每输出指令/ILP 最差；**R=2 = 摊铺
  （48 WG ≥ SM 数）× 线程内 ILP=4 的甜点**。同 48 WG×256 线程对照（N=1024
  形状走 64-tile）实测 R=4 比 R=1 每线程快 ~7×——纯吞吐上 R=4 仍是理想，
  但网格摊不开，无解（保序前提下）。
- **3. e2e 收益（NN_PROFILE=1 text_train d64/L4/H4/ff256/seq256/B64/adam、
  20 步，off/fine 交错 2×2）**：
  | 口径 | off | fine | Δ |
  |---|---|---|---|
  | `fused_mm@bwd.blk#0`（设备，ms/20 步） | 222.5（avg） | 171.5（avg） | **−23%** |
  | step 墙钟（ms/20 步） | 2128.5（avg） | 1961.8（avg） | **−7.8%**（106.4→98.1 ms/step） |
  | 对照 `fused_mm@fwd.blk#0` / `fwd.layer#0` | 10.5 / 102~103 | 平 | ✓ 只有反传 GEMM 变 |
  形状探针 controls 不回退：1024³ 2410~2489、fwd.QKV 1525~1560、LM-head wgrad
  （8208 行网格 129 ≥32 走 64-tile）3189~3588 GFLOPS，与改前同窗持平。
  对 torch（50.7 ms/step）差距 **2.10× → 1.94×**。
- **4. 被否决（防重蹈）**：① **k+1 预取软件流水**（把下一轮 LDS 提到本轮
  FMA 前，曾在 fold 生成器奏效）——细 tile 实测 **1.33→2.13ms（+60%）**、
  64-tile 持平：该配方的 shared LDS 链**不能**照搬 fold 的私有数组预取，
  已回退并把结论写进 `matmul_tiled.comp` 注释；② R=4 等比缩块（见上表）。
  另记录一个**未定位的地板**：细 tile 三设计与 naive 全部在 ~1.2~1.35ms
  （≈300~440 GFLOPS）平台、耗时对 K 线性——每 k 迭代 ~80~134ns 的串行
  地板（LDS 链/调度）未能用预取打破，是残余疑点。
- **5. 残余与下一步**：wgrad 仍 ~1.6× 差距上限（保序设计已榨干）；history
  P0 原估 −26ms/step 的另一半需要 **K 切分二次归约**（改变累加序 →
  **动字节锚，需另行裁定**）或 f16 提速兑现（P3）。
- **6. 验收**：Release 构建（`build/`）字节锚**逐位不变**——CPU
  `f360f9bdbbfbe12c`、GPU dev2 `4103194cb9e3e230`（`gpu_stability_probe
  --steps 20`，含 `--gpu 2`）；**ctest 29/29**；`doc_inventory` 三门禁
  （virtual 52 / L2-VIOLATIONS 0 / HANDWRITTEN-RESIDUE 0）；库内结构 93 不变
  （注册表 288 行 = 93 + 132 精度 + 54 批量 + **9 细 tile**）；`text_train`
  交错 A/B 中对照项逐位平。
- **7. 关键文件**：`expr_spec.hpp`（`EXPR_MM_FINE_SUFFIX`/`EXPR_MATMUL_BLOCK_FINE`/
  `EXPR_MATMUL_TILE_FINE`/`EXPR_MATMUL_FINE_MIN_WGS`）、`expr_glsl_gen.hpp`
  （`generate_glsl_matmul` block 变参 + R 分支计算/写回 + `generate_glsl`
  mm_fine 入口）、`expr_emitter.hpp`（`generate_matmul_fine` 虚接口）、
  `tools/fused_generate.hpp`（#f 发射）、`backend/compute_vk_backend.hpp`
  （注册豁免 / run_fused_gpu 选择 / op 级 pipeline + 选择 / 两个逃生阀）、
  `shaders/matmul_tiled.comp`（`NN_MM_FINE` 参数化）、`CMakeLists.txt`。
- 产物：`build/prof_wgrad_{off,fine}_20261007.txt`（首轮 A/B）、交错 2×2 在
  会话内直读；形状探针复现 = 上表。

---

# 设备表 per-key 归因（NN_PROFILE 新增表 5，2026-10-07）

## 承接 wgrad P0 的下一步：定位"forward-fused 17.2 ms/step 桶"的构成

- 类型：演进记录 + 性能归因
- **背景（缺口）**：设备表（表 4）只按 `label@stage` 聚合——`fused@fwd.blk#*`
  四行合计 ~17 ms/step 这类大桶看不出由哪些**结构**贡献，无法定点。host 侧
  op 表也只有 `eval_expr` 汇总。per-key 归因 = 第 5 张表，按
  `key@label@stage`（运行期 shader key + 变体后缀 + 派发标签 + 阶段）细分。
- **实现**（零开销、纯观测，不进任何计算路径）：
  - `core_profile.hpp`：`State.device_keys` 第二张聚合表；`interned_detail()`
    把运行期 key（`expr_spec_key` 及 `#x/#b/#f` 后缀、原语变体名）拷进稳定池
    （命名约束：聚合表 key 必须静态存储）；`device_add(name, ns, detail=)`
    detail 非空时双写；`dump()` 打表 5（top 60 + 总条数）；`reset` 同步清空。
  - `compute_vk_backend.hpp`：`TsCursor.marks` 改 struct（原 pair →
    `{key, detail, pair}`）；`ts_dispatch` 增可选 `detail`（string_view）与
    `detail_suffix`（静态后缀，细 tile 变体 `#f` 调用点零分配）；录制期把
    `detail[suffix]@label@stage` 驻留，resolve 时随 timestamp 一起回填。
  - **接线点**：`run_fused_gpu` 两条派发（`fused_mm` 带 `use_fine?#f`、
    `fused` 带 shader key）、`run_fused_gpu_batch`（`key#b`）、
    `dispatch_compute` 加 `ts_detail` 形参（op 级 matmul 传变体名
    `gemv/gemv_f16/tiled/tiled_f16/fine/naive_fine/naive`、batched_matmul 传
    `f16/f32`）。其余原语 label 已足够细，不加。
  - **口径**：表 4 完全不变（A/B 可继续对历史报告）；表 5 合计 ≤ 表 4 合计
    （仅带 detail 的派发入表 5）；解析串用 `@` 分段且 stage 恒为最后一段
    （key 与 label 本身不含 `@`）。
- **归因结果（`build/prof_keyattr_20261007.txt`，GPT d64/L4/H4/ff256/seq256/B64
  adam、20 步、GPU 40HX、Release）**——此前悬置的三个桶全部定位：
  | 桶 | 归属 key/结构 | 设备耗时（ms/20 步） |
  |---|---|---|
  | forward-fused ~17 ms/step | **`f7e6c2bd8e35af36` = 注意力 forward fold（fold v2 双域，4 块各 78.6，占 `fused@fwd.blk` 桶 ~92%）** | 314.4 合计（15.7 ms/step） |
  | CE 5.9 vs torch 2.7 | `0fdcedce8f01e6ad` lse fold 59.9 + `26c8f729e11e40bd` grad 58.3 | 118.2 合计（5.9 ms/step） |
  | wgrad 桶（P0 后仍最大） | `71733007a831714a#f` 细 tile（blk 各 ~113）+ 同 key 64-tile 在 layer#0 96.0 + `6996c1ccdb7656fc` dgrad（blk 各 ~46 + layer#0 80.2） | blk 细 tile 451 合计（22.6 ms/step） |
  | 其余热点 | `c2aca8a29eb4617f` fwd matmul+bias（fwd.layer#0 = LM head 101.5）；`86c84dd0293c4913` bwd 归约（rowmax/exp 链，blk 各 ~21.7）；`907399e90fa111b3` bwd.layer#0 归约 24.7 | — |
  **读数**：① forward 的 17 ms/step **不是散布的小算子，而是单一注意力
  fold kernel**（每块 ~3.93 ms/call × 4 块 × 1 call/step）——优化对象明确为
  fold v2 生成器（doc-mask 分支 + 每 k0 块 3 barrier 的节奏），与 wgrad 一样
  属"单 kernel 吞吐"而非派发开销；② CE 两个 key 各占一半，lse fold 略贵于
  grad；③ 设备 busy 91.1%，`copy_from` 1471ms 仍是阻塞等待口径（非 PCIe）。
- **验收**：字节锚逐位不变（CPU `f360f9bdbbfbe12c` / GPU dev2
  `4103194cb9e3e230`，剖析纯观测）；**ctest 29/29**；`doc_inventory` 三门禁
  52 / 0 / 0（exit 0）；表 4 各行与改前同配置读数一致（同窗 ±噪声）。
- **关键文件**：`core_profile.hpp`（`device_keys`/`interned_detail`/表 5 打印）、
  `backend/compute_vk_backend.hpp`（`TsMark`、`ts_dispatch` detail 参数、
  `dispatch_compute` `ts_detail`、6 个接线点）、AGENTS §3/§12 两处描述。
- 产物：`build/prof_keyattr_20261007.txt`。

---

# fold v2 恒等三改（P2-1）与注意力 fold 相位消融（2026-10-07）

## 承接 per-key 归因：注意力 fwd fold 的第一刀 + P2-2 立项所需的相位硬数据

- 类型：性能 A/B + 演进记录
- **背景与分期（用户裁定）**：per-key 归因把 forward ~17 ms/step 定位到单
  kernel `f7e6c2bd8e35af36`（注意力 forward fold，3.93 ms/call × 4 块）。
  静态解剖（生成 GLSL 实读）后原计划 P0 恒等小改 / P1 vecacc 小改 / P2 全量
  重写。用户问"P2 会不会让 P0/P1 白做"——裁定：**P1 与 P2 重复 → 取消、并入
  P2-2**；**P0 以"P2 第一阶段（P2-1 恒等层）"身份执行**（三改在任何重写里
  都必须保留，不会被扔掉），同时充当周期模型的廉价证伪探针。
- **P2-1 三改（`expr_glsl_gen.hpp` 的 `generate_glsl_fold_v2`，全部逐位恒等）**：
  1. **整块 skip**：tri_skip 原先只把 valid 钳成 0，空块内 6 个 WG barrier、
     空归约、空链照跑（文档"行界整块跳过"名不副实）——现在 `valid == 0`
     直接 `continue`。恒等论证：空块必晚于 block0，block0 恒 valid≥1 且含
     未屏蔽对角 j=qt（causal 不遮对角、doc 同位置同 id）→ 进空块时 m 必
     有限 → 旧空块语义 `max(m, lowest)=m`、`α=exp(0)=1.0`（FP 精确）、
     `l=l·1.0+0`、`o×1.0` 全部逐位恒等；valid 只依赖 row/k0 → 全 WG 均匀。
  2. **链读集载入**：flush_chain 原对全部元素寄存器对称 round-trip——注意力
     chain1 实测 9 次 sreg 载入**全是死加载**（覆写前载入/本段未用）、
     chain2 9 中 8 死。改"首访为读才载入、链内被写才存回"（按 Reg/Fanout/
     Reduce 操作数收集读集；未用的 b/c 默认 (Reg,0) 而寄存器 0 恒为行标量态
     → 不会误判），出口 shared 逐位一致。
  3. **归约收 subgroup0**：sred 本来就只取 lane0（= subgroup0 蝶式归约结果），
     其余 subgroup 的 gather 是重复读、shuffle 结果全被丢弃——gather 与
     shuffle 都加 `tid < gl_SubgroupSize` 守卫（subgroup 内均匀无发散），
     subgroup0 各 lane 的 v 同线程同序 → sred 逐位不变。
- **同窗交错 A/B（3 对、方向交替；基线 = 改前 exe 快照 `build/text_train_base.exe`，
  `NN_PROFILE=1` 同配置 20 步，取 `f7e6c2bd@fused@fwd.blk#*` 四行 avg_us 均值）**：
  - new：3789 / 3815 / 3794（均 **3799 µs**）；base：3956 / 4033（均 **3995 µs**；
    首跑 base 5147 为时钟爬坡冷启动，弃——同方法论坑见 development/08 §5）。
  - 配对 −3.6% / −5.9%，净值 **≈ −4.9%**（墙钟 1895→1885 ms/20 步量级，busy 92% 持平）。
  - **低于立项预估 −15~25% → 分段周期模型被证伪**：空转块与死加载是小头，
    成本几乎全在未改动的完整块相位——这正是 P2-1 先行的校准价值。
- **相位消融探针（临时构建期开关，量完即删）**：在 `generate_glsl_fold_v2`
  临时加 `NN_PROBE_FOLD_NOVECACC`（省 vecacc V 累加、保留 α rescale 与屏障
  节奏）/ `NN_PROBE_FOLD_NOMM`（省 mm 的 K 读+Σ_d 收缩、保留 spart/smm/双
  barrier 结构）两个发射开关，每次改后重建 scan 产物、各跑 2 次取稳态：

  | 消融 | fold avg_us | 归因份额 |
  |---|---|---|
  | 全量（P2-1 后基线） | 3799 | — |
  | 省 vecacc V 累加 | **1646** | **≈57%** |
  | 省 mm K 读 + 收缩 | 3216 | ≈15% |
  | 残余（掩码链×2、归约×2、屏障、Ash、收尾） | — | ≈28% |

  **读数：vecacc 是唯一量级大头**（d_k=16 → `dd=tid` 映射塌缩到半个 warp、
  单累加器 128 FMA 依赖链 ×4 周期 + V 读延迟，且卡住其后全部屏障等待），
  与静态解剖一致；1646 µs = 除 vecacc 外的地板，即 P2-2 天花板参照。
- **探针过程坑（记）**：① `std::getenv` 在本机 clang + `-Werror` 挂
  `-Wdeprecated-declarations`（MSVC CRT 弃用）→ pragma 屏蔽；② **ninja 只看
  输入变化，env 变了不重跑生成**——探针 B 首次"构建"报 `no work to do`、
  跑出的其实是探针 A 二进制（靠核对生成 `.comp` 内容才发现，读数已弃），
  必须 touch 输入或删产物强制重生成；③ 消融读数必须验证生成物确实变了
  （`Select-String` 查关键代码行），否则"探针"测的是旧 shader。
- **P2-2 设计约束（供立项，未实施；执行需用户批准——移 GPU 字节锚）**：
  ① FP 左折和**不可并行保序**（任何 j 分片/多累加器的结合序都不同）→
  vecacc 展宽必移 GPU dev2 锚（同 wgrad K-split 类处置：重测 + 更新 AGENTS
  锚行 + 本档记录；CPU 锚不动）；② **veclen 不进 key**（闭合世界，同 key 服务
  任意 d_k）→ 线程映射不能按 dk 注册期选型，只能运行期 uniform 双路径
  （vector_out≥256 走现行 `dd=tid`、小 dk 走 j 分片 + 归约）；③ RowSum 归约
  改序同在移锚范围（RowMax 顺序无关）；④ 通用性：仍是 FoldSpec 通用折叠
  （CE logsumexp、5 掩码组合共用），**不立注意力专属原语**（用户既定原则）。
  预期：vecacc 57% → fold 3799 → 约 1.7~2.2 ms/call（P2-2 内再收 mm/屏障）。
- **验收（P2-1）**：字节锚逐位不变（Release CPU `f360f9bdbbfbe12c`、GPU dev2
  `4103194cb9e3e230`，`gpu_stability_probe --steps 20 [--gpu 2]` 均 PASS）；
  **ctest 29/29**；`doc_inventory` 52/0/0 exit 0；`doc_align_audit` exit 0；
  探针全部撤除后重建，注意力 `.comp` SHA256 回到 P2-1 态逐字节一致
  （`1D30EE1F…C10F11`）。
- **关键文件**：`include/neuralnet.cpp/expr_glsl_gen.hpp`（`generate_glsl_fold_v2`
  三处：tri_skip 分支 continue、flush_chain 读集、归约守卫）；AGENTS §7 fold 段。
- 产物：`build/ab/run{1..6}_{base,new}.txt`（A/B）、`build/ab/probe{A,B}_run*.txt`（消融）。

---

# fold v2 vecacc 双路径改写（P2-2）：j 分片跨线程 + 标量累加器（2026-10-07）

## 承接 P2-1：vecacc 展宽落地（用户批准移 GPU 字节锚）

- 类型：性能 A/B / 演进记录
- **立项（用户裁定）**："P2-2，可以批" = 批准 FP 求和序变更移 GPU dev2 字节锚
  （原 P1 已在 P2-1 节裁定取消并入本项；CPU 锚不动）。
- **设计（在"veclen 不进 key"约束下的运行期 uniform 双路径，一份 shader 服务
  任意 d_k；仍是 FoldSpec 通用折叠，零注意力专属原语）**：
  - **路径 B（0 < vector_out ≤ 256）**：j 分片跨线程——列 `c = tid % vs`、
    片号 `g = tid / vs`、本列线程数 `GB = ceil((256-c)/vs)`，片内地板划分
    `[jlo, jhi) = [⌊g·valid/GB⌋, ⌊(g+1)·valid/GB⌋)` 精确覆盖 `[0, valid)`
    （ΣΔ⌊·⌋ 恒等、不重不漏、片可为空）。旧映射 `dd=tid` 在 d_k=16 时只有
    16/256 线程参战（半个 warp 单累加器长链），本路径 256 线程全忙。
    **归约推迟到块循环之后每行一次**：列间部分和经每线程一槽的槽表
    （有 mm 段复用 `spart` 线性重解释、无 mm 段新增 `sva[256]`——注意力恒带
    mm → 零 shared 增长），g=0 列主求和后回写**自己的** `o[0][0]`，finalize
    从不做线程间读；`vs<256` 每行 +2 屏障（vs=256 → GB=1 免归约）。
  - **路径 A（vs > 256）**：保留 `dd` 步进分片（256 线程已全忙、V 读跨线程
    合并），累加器 1 → `VA_ACC=4` 拆单依赖链（尾组 `% VA_ACC`）。
  - 收尾段崩叠：路径 B 标量求和、路径 A 分片内 4 → `[0]`；finalize 只读
    `o[shard][0]`。行首初始化/缩放与两路径覆盖口径一致（见下"弯路"）。
  - FP 结合序异于旧左折和 → GPU 锚移动（已批）；CPU 左折不变。
- **首版弯路（关键教训：动态下标局部数组 = scratch 雷区）**：首版把累加器放
  `float o[4][4]` 数组、缩放/初始化按 16 槽全 tile 在 256 线程上执行——同窗
  A/B **反慢 26%**（4839.7 vs 3838.2 µs）。诊断：`o` 带动态下标
  （`(dd-tid)/256`、`(j-jlo)%4`、循环 `oi/an`）被驱动放进 local/scratch，
  每次累加 RMW + 256 线程 ×16 槽 ×每块 的缩放/初始化是 GB 级流量——与
  P2-1 消融"省 vecacc 省出 2153 µs ≈ 2.1 GB o-RMW @ ~1.5 TB/s"的算术互相
  印证（旧版 o[4] 同样动态下标，本就是 scratch 流量大头）。**改法**：路径 B
  热路径全部**标量累加器 `ob0..ob3`**（恒寄存器、零数组访问），`o[][]` 降级
  为 finalize 接口（每行各写一次 `o[0][0]`，路径 B 从不读未写槽 → 不初始化）；
  尾段按固定累加器序展开（主循环出口 `(j-jlo) ≡ 0 (mod VA_ACC)` → 步 0,1,2
  依次落 ob0/ob1/ob2，无动态下标）。路径 A 数组保留但只缩放可达分片（×4）。
- **同窗交错 A/B（3 对；基线 = 改前 exe 快照 `build/ab/text_train_p21.exe`，
  `NN_PROFILE=1` 同配置 20 步，取 `f7e6c2bd@fused@fwd.blk#*` 四行 avg_us 均值；
  两 binary 各弃一次冷启动首跑）**：

  | 版本 | fold avg_us（3 对） | 相对基线 |
  |---|---|---|
  | P2-1 基线 | 3838.9 / 3820.8 / 3854.8（均 **3838.2**） | — |
  | P2-2 首版（数组累加，回退弃） | 4845.4 / 4807.2 / 4866.5（均 **4839.7**） | **+26.1%** |
  | P2-2 终版（标量累加） | 2662.9 / 2634.9 / 2670.8（均 **2656.2**） | **−30.8%**（配对 −30.6/−31.0/−30.7%） |

  - E2E：step 墙钟 1908.27 → 1812.81 ms/20 步（95.4 → 90.6 ms/step，
    **−5.0%**，与 4×(3838−2656) ≈ 4.7 ms/step 折算吻合）；GPU busy 92.0 →
    91.8% 持平（copy_from 阻塞等待占比口径不变）。
  - 分解（按 P2-1 消融份额估算）：vecacc 2153 → **≈900 µs**（mm 583、
    残余 1170 基本不动）。**预估 1.7~2.2 ms/call 未达**：剩余 vecacc ≈900
    中 V 读 L2 地板（与 mm 同流量，mm 583）+ 每行归约串行 16 槽求和 + 尾段
    ——下一步空间 = 归约改同 subgroup 映射（`c = tid/G` 连续线程组 + 块状
    shuffle 免 shared/屏障）与 mm/掩码链（另案）。
  - 首版读数记录于本条（其 run 文件已被终版同名覆盖）；终版产物
    `build/ab/p22v2_{base,new}_run*.txt`。
- **锚移动（已获批，处置按先例）**：Release GPU dev2
  `4103194cb9e3e230` → **`e9d6c87973fc01b5`**、Debug GPU dev2
  `c69a3378c7dcc037` → **`b41790d04e0007a0`**（两口径
  `gpu_stability_probe --steps 20 --gpu 2` 均 PASS，in-process loss/hash
  匹配；GPU/CPU loss 差 ~5e-7 = 改序的预期量级）；Debug CPU
  `8809f07142e29472`、Release CPU `f360f9bdbbfbe12c` **逐位不变**。
- **验收**：ctest **29/29**（两轮）；`doc_inventory` 52/0/0 exit 0；
  `doc_align_audit` exit 0；生成 shader 双路径核对
  （`build/generated/fused_f7e6c2bd8e35af36.comp` SHA256
  `59EC92A41FB5322AB61ED6035E7638E65A50CDEF89090D504559AF3430E3F511`）；
  backward fold（无 vecacc）生成路径未动、设备表 71.9 µs 持平。
- **关键文件**：`include/neuralnet.cpp/expr_glsl_gen.hpp`
  （`generate_glsl_fold_v2`：`VA_ACC` 常量、双形态初始化/缩放、路径 B 标量
  累加 + 展开尾段、路径 A 多累加器、收尾崩叠 + 跨线程列归约、`sva` 条件
  shared、VecState 渲染 `o[shard][0]`）；AGENTS §7 fold 段与 §12 锚行。
- 产物：`build/ab/ab_p22.ps1`（A/B 脚本）、`build/ab/p22v2_*.txt`。

---

# matmul K 切分二次归约（`#ks`/`#kr`）：wgrad 派发饥饿第二刀（2026-10-07）

## 承接 P0 细 tile（保序设计已榨干）；跨段求和序变更 = 移 GPU 锚（已批）

- 类型：性能 A/B + 演进记录（含实施期三个 bug 的根因与可复现诊断法）。
- **背景与立项裁定**：P0 细 tile 后 wgrad 仍 ~1.6× 差距上限（保序前提下"网格摊铺
  × 线程内 ILP"已到甜点，history P0 条原估 −26 ms/step 的另一半就需要 **K 切分
  二次归约**——改变累加序 → 移字节锚，当时挂起待裁）。2026-10-07 用户裁定先查
  PyTorch 官方口径：[Reproducibility 文档](https://docs.pytorch.org/docs/main/notes/randomness.html)
  原文 **"Completely reproducible results are not guaranteed across PyTorch
  releases, individual commits, or different platforms"**，且数值精度文档明确
  数学等价的浮点计算不保证逐位一致、换后端归约序不同属正常行为 → **torch 不保证
  ⇒ 咱们也没必要保证**：顺序变化不构成债务，仅按既定流程重测移锚；run-to-run
  确定性（探针 run1==run2 逐位）照旧必须保。据此 K-split 获批推进。
- **1. 落地形态**（闭合世界、keys 不变、通用机制零注意力专属原语）：
  - 派发判据（run_fused_gpu）：`has_mm && raxis<0 && !fold && 64-tile 网格饥饿
    (wg64 < EXPR_MATMUL_FINE_MIN_WGS) && !is_disp && 无 extras && #ks+#kr 同时注册
    && NN_MM_NO_FINE/NN_MM_NO_KSPLIT 均未关`，且细 tile 网格仍 < **256**
    （EXPR_MATMUL_KSPLIT_MIN_WGS ≈ 8×40HX SM）。S = clamp(⌈256/细网格⌉, 2, 8)。
  - **`#ks`**（`EXPR_MM_KSPLIT_SUFFIX`）= 细块（16×16，R=2）matmul 的 K 切分版：
    段号 = wg.x/nx、**列块 = wg.x % nx**、段 [k_lo, min(k_lo+span, K))，span =
    ⌈K/S⌉ **上取 16 对齐**（非末段段界必是 16 倍数 → 段内 vec4 快路径 k 连续
    判据成立）；dispatch x = nx·V（**只派有效段**，V = ⌈K/span⌉，空段 WG 会早退
    且不写）；**写原始部分和**到 (S·rows, cols) 缓冲（不跑尾链——尾链非线性，
    须在求和后统一执行）；PC 7 槽 = matmul 头 + `ks_span`（注册侧 pc_base_fixed
    按键后缀特判 7）。全部 k 上界守卫从 mm_k 换段界 kend（非末段越界会把下段
    的 k 拉进本段部分和 = 静默错值）。
  - **`#kr`**（`EXPR_MM_KSPLIT_REDUCE_SUFFIX`）= 伴生归约：绑定 = 原输入 0..N-1
    + 部分和(binding N) + 输出(binding N+1)；注册行用「原 spec + 追加一个 Linear
    视图」的**合成 spec**（views.size() 决定 num_bindings = N+2，与生成器逐槽
    同源；owned 先 reserve 保指针稳定）；PC 6 槽与 matmul 同布局（has_matmul=1
    → 注册 6，第 4 槽复用填**有效段数 V**，mm_trans 占位填 0）；local_size 256、
    每线程一输出，段号升序左折 `acc = ((p0+p1)+p2)+…` 后跑同一条尾链——尾链
    从 `generate_glsl_matmul` 抽出共用 helper **`glsl_emit_matmul_tail_fn`**
    （两处同一实现，字节级同发射）。
  - 两段派发：两 desc set + 两份 PC + partial 写转读屏障（`record_output_barrier`
    src SHADER_WRITE→dst MEMORY_READ 覆盖 compute→compute）；partial 局部析构
    走池回收——批量录制下块只可能被**录制序更晚**的算子复用（执行序 = 录制序，
    #kr 的读在复用写之前），安全论证写在 run_fused_gpu 注释。仅 V0 f32（精度
    变体不产 #ks/#kr）。逃生阀：**`NN_MM_NO_KSPLIT=1`**（回落细 tile，A/B 对照）、
    **`NN_MM_KSPLIT_S=<n>`**（强制 S，调参）。
- **2. 实施期三个 bug（全修复；③ 是主根因）**：
  1. **尾空段垃圾读**：span 上取 16 对齐后 (S-1)·span 可 ≥ K → 空段 WG 早退不写
     部分和，#kr 按 S 全加 → 读池空闲块历史数据（内容随 run1/run2 分配史漂移
     → 探针 run1≠run2）→ 只派发/只加**有效段 V**。
  2. **vec4 快路径行基对齐**：段内判据 `(kend&3)==0` 只保证段内 k 连续，丢了原
     `(mm_k&3)==0` 的**行基对齐**（别名索引 = grow·mm_k + kAG，行跨度是 mm_k）
     → K 非 4 倍数（vocab=257 的 dgrad K=257）跨行错读 → 双判据并存（非切分时
     kend==mm_k，与原式等价）。
  3. **列块未取模（主根因）**：`slice = wg.x/nx` 但 `block_col = wg.x×B` 漏了
     `% nx` → slice≥1 的 WG 列块整体越出 cols、写回被 `cc<cols` 守卫**整块静默
     丢弃** → slice≥1 的部分和从未写出。V=1 的小形状（单测覆盖的全是 K=6/8 之类）
     恒不触发，V≥2 的 gpt/rapt offload 一致性与稳定性探针全挂。
  **可复现诊断法（vkCmdFillBuffer 二分，诊断码已删）**：部分和缓冲整体清零 →
  探针 PASS（垃圾变零 → run1==run2）；只清零 [V·rows·cols, S·rows·cols) → 仍
  FAIL ⇒ 缺口在"已写区"[0,V) 内 ⇒ 网格映射问题而非段数 PC/循环问题。
- **3. 同窗交错 A/B**（同一 binary、env 开关对照；text_train d64/L4/H4/ff256/
  seq256/B64、adam、20 步、GPU 40HX、3 对、弃冷启动；NN_PROFILE=1）：
  | 口径 | OFF（NN_MM_NO_KSPLIT=1，细 tile 基线） | ON（K 切分） | Δ |
  |---|---|---|---|
  | step 墙钟 ms/20 步 | 1806.6 / 1822.3 / 1825.7（均 **1818.2**） | 1502.0 / 1502.5 / 1511.8（均 **1505.4**） | −16.9 / −17.6 / −17.2 → **−17.2%**（90.9 → **75.3 ms/step**，−15.6 ms/step） |
  | wgrad per-key 桶 `71733007a831714a`（ms/20 步） | 458.2 / 462.1 / 464.0（均 **461.4**） | 138.2 / 139.9 / 139.3（均 **139.1**） | **−69.9%** |
  | 设备表其余全部行 | — | — | 无一行变化 >5 ms（零附带回退） |
  对 torch fp32 50.7 ms/step 差距 **1.79× → 1.48×**。`#kr` 本体 ≈1.3 ms/20 步
  （低于 per-key 表显示下限，聚合在 `fused_mm` 桶内账目自洽）。
  S 调参单窗扫描（`NN_MM_KSPLIT_S`）：S=2 → 1643.6、4 → 1507.5、6 → 1497.3、
  8 → 1498.0、auto 阈值 128（qkv S3/ffn S2）→ 1516.4 ms/20 步 ⇒ 6~8 平台期 ⇒
  **阈值 128 → 256**（qkv S=6 / ffn S=4）为定稿参数（取值依据写进 expr_spec 注释）。
- **4. 验收**：ctest **29/29**（两轮）；稳定性探针 run1==run2 逐位 PASS（四口径）；
  **CPU 字节锚两口径逐位不变**（Release `f360f9bdbbfbe12c` / Debug
  `8809f07142e29472`）；**GPU dev2 锚按批移动**：Release `e9d6c87973fc01b5` →
  **`f1f5de1c4420d565`**、Debug `b41790d04e0007a0` → **`d782e300e340edda`**
  （`--steps 20 --gpu 2` 均 verdict=PASS）；`doc_inventory` 三门禁
  （virtual 52 / L2-VIOLATIONS 0 / HANDWRITTEN-RESIDUE 0）exit 0、
  `doc_align_audit` exit 0；注册表新增 **9 对 `#ks`/`#kr`**（main 100 结构、
  9 细 tile、9 K 切分对；`expr_fuse_test` 收集器 10 对；`[gen]` 行新增
  「+N 条 K 切分变体(#ks+#kr)」）。
- **5. 关键文件**：`expr_spec.hpp`（两个后缀 + KSPLIT_MIN_WGS/MAX_S 与取值依据）、
  `expr_glsl_gen.hpp`（`glsl_emit_matmul_tail_fn` 抽取共用 / `generate_glsl_matmul`
  的 k_split 分支（段界守卫、%nx 列块、免尾链写回）/ 新生成器
  `generate_glsl_matmul_ksplit_reduce` / `generate_glsl` 开关）、`expr_emitter.hpp`
  （`generate_matmul_ksplit{,_reduce}` 两个虚接口——ExprEmitter 不计入
  doc_inventory 的 52 口径）、`tools/fused_generate.hpp`（配对发射 + 注册行 +
  合成 spec）、`backend/compute_vk_backend.hpp`（注册豁免/pc_base_fixed 特判、
  选择与两段派发、两个 env 阀、批量录制安全注释）。
- **6. 缓做**：OP 级 `matmul_tiled` 的同机制 K 切分（e2e 瓶颈在 fused 路径，
  OP 级暂无实测收益证据）。
- 产物：`build/ab/ab_pks.ps1`（A/B 脚本）、`build/ab/pks_{ks,off,warm}_run*.txt`。

---

# P3 跨链批量派发：`#b` 批量变体 + BDA 实例表 + 图内同签名分组（2026-10-06）

- **类型/机制**：通用机制（闭合世界、可降级、可观测），非单算子特例；数值面零变化
  （批量派发不改变任何逐元素计算与顺序）。生成期给每个**纯逐元素**结构
  （`expr_spec_batchable`：无归约/fold/matmul，**含 P2 多输出 extras 复合**，仅 V0
  全 f32）追加 `key#b` 批量变体（`EXPR_BATCH_SUFFIX`、`ExprEmitter::generate_batched`）：
  缓冲声明块改「实例表（uvec2，stride = 输入+主输出+extras+尾槽）+ buffer reference
  + 名字宏」、main 序言按 `gl_WorkGroupID.y` 取实例、强制标量；主体渲染与普通版
  逐字一致，PC 与 V0 完全同布局。运行期 `GpuBackend::run_fused_gpu_batch`：实例表经
  `vkCmdUpdateBuffer` 录制期内联写入小 device-local 缓冲（≤64KB），一次 dispatch
  `(ceil(max_count/256), n_instances, 1)` 跑完整组。**取舍：BDA（buffer device
  address）而非描述符数组**——描述符数组管道的注册/绑定面翻倍（逐实例写描述符 +
  数组池），BDA 只要一条实例表 + 一个 binding，代价是设备特性门控（不支持即降级）。
- **分组规则**（`execute_fused_graph`，稳定拓扑序 + keep 占位之后）：组键 = 结构 key
  + consts/rparams/vp **值位型序列**（批量共享一份 PC）；贪心按列表序收集、单组
  CAP 32；组内两两**互不依赖**（存储级：无任一成员消费/覆写另一成员的物化输出，
  覆盖 AdamW decay/update 同 dst 的无依赖边形态），且入组不得逆置任何既有顺序
  （与更早 kernel 有存储冲突者其派发位置必须更早）。决策只按列表序 + 键值。
- **实例记录布局**（生成器 `expr_glsl_gen` batched 分支与后端 `FusedBatchInst`
  注释同源写明）：uvec2 数组、每实例一段，**stride S = n_in + 1 + n_extras + 1**——
  槽 `[0,n_in)` 输入地址、`[n_in]` 主输出、`[n_in+1+k]` extras（成员序）、
  `[S-1] = (count, cols)` 尾槽 = 本实例元素数 + **本实例列数**。组键与 rows/cols
  无关（设计）⇒ 同组形状可不同，count/cols 必须逐实例携带（shader 侧 `count_l`
  与 `#define cols` 均读尾槽；PC 的 count/cols 两槽仅保持 V0 布局）。这是正确性
  守卫而非偏离：PC 只有一个 cols 槽，同组混形状时复用它会静默错值。
- **降级与开关**：`key#b` 未登记 / 设备无 buffer device address（`bda_available()`
  为假时注册期即跳过 `#b`）/ `NN_IRC_NO_BATCH=1`（每次 `execute_fused_graph` 读取，
  测试可同进程 A/B）→ 逐 kernel 走原路径；`NN_IRC_TRACE=1` 打 `[ir-c][batch] N 实例
  → 1 派发 <组键>` 与 `[ir-c][batch-degrade] …`（与 `[ir-c][degrade]` 同哲学）。
  未登记的单节点结构照旧硬报错（闭合世界不变）。
- **验收数字**（Debug）：
  - **kernel 对照（验收口径 = 融合派发合计；ts 标签保持 `fused_b`）**：
    `NN_PROFILE=1 text_train --seq-len 256 --batch-size 64 --max-steps 20
    --d-model 64 --gpu=40HX`：`fused@optimizer` **1330 calls（66.5/step）→ 归零**，
    派发计入批量行 **`fused_b@optimizer` 57 calls（2.85/step）**，即
    **合计 1330 → 57 calls/20 步**（≤120、≤6/step 达标）；`NN_IRC_TRACE` 逐组迹证
    每步 **70 实例（32+32+6）→ 3 次派发（3.0/step）**（设备表少 3 = 尾帧 marks
    打表后才 resolve 的既有剖析口径，2 步复测 3/6 同证）。**批量与非批量分列是
    有意为之（用户裁定）**：`fused@optimizer` 行保留 = 任何漏批量/降级回落的派发
    立即可见（降级必须可见）；两行合计才是融合派发总数。
  - **批量 vs 逐个逐字节一致**：`expr_fuse_test` 新用例（3 同签名单节点 + 3 个
    P2 extras 金刚石 + 1 条 P1 链）`NN_IRC_NO_BATCH=1` 与默认各跑一遍，
    **16 张量 memcmp 逐字节相等**；CPU 逐节点对照 max_err 2.38e-07（容差 1e-4）；
    并断言**批量派发恰 2 组**（`fused_batch_dispatches == 2`，无 BDA 时跳过）——
    该断言兼作**同组混形状回归锁**：两组的三个实例 (R,C)=(3,5)/(4,7)/(2,11)
    各不同（count/cols 均逐实例）仍各自成组一次派发。
  - **字节锚逐位不变**：`gpu_stability_probe --steps 20` = `8809f07142e29472`、
    `--gpu 2` = `c69a3378c7dcc037`（与 P4 后锚逐位一致）。
  - **ctest 29/29** 全绿（含新用例）；构建 `-Werror` 零告警。
- **代码现实两处**（最小改动 + 同源处理）：① GLSL 侧在设计的
  `GL_EXT_buffer_reference` 之外追加 `GL_EXT_buffer_reference_uvec2`——uvec2 槽位
  → reference 的构造 glslc 强制要求该扩展（缺则编译期报错）；② ts 标签/指标行
  口径见上「kernel 对照」条（用户裁定：`fused_b` 与 `fused@optimizer` 分列，
  合计核对）。

## 库级违规设计清理：模型文件角色 + 参数形态协议 + 引擎能力体系（2026-10-11，docs/todo/23）

**整改前的原位置**：`include/neuralnet.cpp/compute_engine.hpp`（6 个能力谓词 + 19 处基类
NVI 能力分支）、`compute_layer_base.hpp` 的 `Layer::sub_layers()`、`compute_layer_bitlinear.hpp`
（5 处能力查询 + 4 个部署态守卫）、`ternary_deploy.hpp`（整文件）、`model_serialization.hpp`
（v6 扁平矩阵序列）、`src/{text,mnist}_{train,infer}.cpp`、`gui.py`、`cli_controllers.py`、
`bench/gui_cli_audit.py`、`src/ternary_deploy_test.cpp`。

**类型**：删除清单 + 演进记录 + 覆盖缺口。

**删除清单**：① 基类 `supports_native_data_move` / `supports_native_f16_reduce` /
`supports_expr_precision_variant` / `supports_ternary_gemm` / `supports_int8_dot` /
`int8_dot_accelerated`（能力改由 `describe()` 的 `EngineInfo` 披露；引擎侧落成私有事实 +
`*_impl` 内部分支）；② `Layer::sub_layers()`（参数下标转发改走受保护的 `param_children()`）；
③ `include/neuralnet.cpp/ternary_deploy.hpp` + `src/ternary_deploy_test.cpp`（NNQT 独立
magic/版本/读写器全套）；④ CLI/GUI 的 `--export-deploy` / 「导出部署文件」行 /
`cli_controllers.py` 透传 / `gui_cli_audit.py` 用例 / 推理 CLI 的 `is_deployment_file`
分流分支。

**演进记录**：v6 → **v7**（文件头加 `role` 字节；参数区从"扁平矩阵序列"改为
`[kind][precision][rows][cols][aux_len][payload_len][aux][payload]` 的**参数级 kind**
条目）。角色：`.nnmodel` = 推理（三值层写打包形态，盘上 base-3 1.6 bpw + τ）、
`.nnckpt` = 训练断点（写 f32 latent）；`--save` 默认 `gpt_model.nnmodel`、
`--save-interval` 写 `.nnckpt`、`--resume` 只认 `.nnckpt`。

**被否决/改写的历史**：① 早先"v7 与引擎能力内敛"稿里的 **NNQT 独立格式 + 部署标签**被裁决
取消（改成单一 NNNN 格式 + 文件角色）；② 「按 `ExprSpec` 对象身份缓存 key」方案未采用 ——
F2 的实测瓶颈是**双份归因**（`supports_expr_precision_variant` 每次都算一遍 key/签名，
`eval_expr_impl` 再算一遍），把选路与边界回退下沉进 `*_impl` 后每次求值只归因一次即可；
③ 原计划的 `supports_*_` 私有化命名（尾随下划线）未沿用 —— 设备事实直接落成
`backend_.has_*_pipeline()` 查询，避免新增私有谓词层。

**覆盖缺口（未做）**：P3（激活 int8 + 整数点积全链路：`quantize_act_i8` / `matmul_q_i8` /
端到端对拍）仍是独立新功能，不在本轮清理范围内（能力探测位已就位，102/40HX 实测可用）；
"免 latent 构建"（载入 `.nnmodel` 时按角色跳过 latent 分配）也未做 —— 峰值显存因此尚未
下降（老 NNQT 那组"11.9× 更小 / 峰值持平"的实测数字仍然是本项的口径）。

**落地验收**：102（40HX / clang 23 / Release）全量构建 + ctest **37/37**；新增
`src/model_roles_test.cpp`（Model 角色只读态 + 前向逐位一致 / Checkpoint 角色 latent
逐位一致 + 可续训 / 推理文件小于断点文件 / `peek_model_role` / 只读态按断点角色导出报错）
取代原 `ternary_deploy_test` 的部署格式面。
---

# 录制窗口排空隐藏 + SpecBuilder 输入槽去重（项目 4/5，2026-10-08）

## copy_from fresh 上传免排空（项目 4）与输入槽去重（项目 5）

- 类型：性能 A/B + 演进记录（含闭合世界分叉的实测暴露与补登记）。
- **立项来源**：剖析画像定位的两个瓶颈——① `copy_from` 121 次/20 步、占墙钟
  71.8%（host 反复 `end_batch → wait_in_flight → begin_batch` 全排空，
  busy 已 90% ⇒ 收益在"把等待藏掉"）；② 反传注意力 `86c84dd0293c4913`
  （`recompute_W_` softmax）归约链 4 槽冗余 + `fused@bwd.blk` 家族 10.3 ms/step。

- **项目 5：SpecBuilder 输入槽去重（通用 IR 机制，非注意力专属）**：
  - 中心化 `SpecBuilder::add_input(t, view, dedup=true)`：同
    `Tensor::same_storage`（同设备/精度/存储指针/形状）+ 同 `ExprView` →
    复用已有槽、不新增绑定。全部 `add_input_*` 与 `BatchModRef/BatchColRef`
    的 `to_spec` 收口到它。
  - **两处调用方必须独占新槽**（返回值即下标，不能用 `inputs.size()` 预判）：
    `RowGatherRef`（登记后改写 `views[ti]` 为 RowGather，复用会污染原输入视图
    → `dedup=false`）；`MatmulNode` A/B（vec4 快路径以 `a_slot != b_slot` 为
    启用条件，同张量 `matmul(X,X)`（Muon Nesterov/NS）合并成单槽会**整体关闭
    快路径**、标量回退 → A/B 恒 `dedup=false`）。
  - `Tensor::same_storage`：新增公开**身份**比较（返回 bool，不暴露存储访问，
    符合铁律 #11）；**无效张量一律判不等**——锚点符号实例（占位张量）不参与
    去重，否则扫描期/运行期结构分叉（见下）。
  - 收益（`recompute_W_` 实测）：4 槽 → 2 槽，归约生成器的 RowReduceMax 全矩阵
    pass 减半 + 重复 `sub/exp` 链因输入下标一致被 CSE 合并。fold spec、
    `expr_graph` FusedKernelInput 不走 SpecBuilder，零影响。

- **实施期暴露的闭合世界分叉（已修）**：去重后**运行期真实张量结构 ≠ 锚点符号
  结构**，同一表达式两个 key。锚点侧登记旧结构、dry-run 侧（真实张量）登记新
  结构；只被锚点覆盖、dry-run 未执行到的调用点 → 运行期新 key 未登记 → 硬报错。
  实测中招：**`Softmax::backward`**（`expr_gpu_test`/`fused_gpu_test` 双挂，
  key=`f017f5936c6a76c1`）——softmax 层不在任何模型主干（注意力走 fold），
  此前只靠锚点。修法 = `scan_exprs` 补 **Softmax fwd+bwd 真实张量 dry-run**
  （按既有约定：dry-run 是结构主要来源）。冒烟扫残余 miss：gpt/rapt 训练、
  mnist mlp/transformer/cnn 训练、`--f16` 分派、text_infer 增量推理、
  checkpoint-every、Muon（`matmul(X,X)`）**全部无闭合世界报错**。
  结构数 91 → **99**（去重令部分结构锚点/双侧各一 + Softmax 补登记）。
  残余风险披露：锚点独有结构中含重复视图者（见 `NN_SCAN_DUMP_SOURCES=1`）若
  未来新增同张量调用点而无 dry-run 覆盖 → 闭合世界硬报错，届时补 dry-run 即可。

- **项目 4：copy_from fresh 上传免排空**：
  - `copy_from(dst, src, fresh_dst=false)` / `write(t, span, fresh_dst=false)`
    新增参数（`copy_from_impl` 虚签名同步；CPU 实现忽略）。
    `fresh_dst=true` = 调用方声明 dst 是刚创建、当前录制帧与在飞帧都不可能
    引用的全新张量 → GPU 侧**跳过整段 `end_batch/wait_in_flight/begin_batch`
    排空**，直接 `upload_blocking`。
  - 正确性论证：反向依赖（当前帧先前命令写 dst、被先执行的上传覆盖）不存在——
    全新 buffer 无任何已录命令引用（3 参 `create_tensor` = 纯分配不录命令；
    池化复用只在写出方帧 fence 信号、`flush_pending_destroys` 归还之后才可能）；
    正向依赖（后续帧读 dst）由 `upload_blocking` 的时间线信号量 + 各帧 submit
    处 `collect_staging_waits` 显式保证（与 `from_matrix` 批内上传同一条既有
    机制）。
  - **唯一承诺方 = `detail::upload_span`**（create→write 相邻形态），两处
    `write` 传 `true`；其余调用方（load_model / 测试直写既有张量）默认
    `false` 保留原排空语义 → **零回归**。
  - 逃生阀 **`NN_COPY_FROM_DRAIN=1`**（进程内静态缓存，fresh 也强制走旧排空
    路径，A/B 对照；与 `NN_MM_NO_KSPLIT` 同口径）。
  - 实施期另一发现：去重把 Muon 的 `matmul(X,X)` 合并成单槽后，K 切分生成器的
    vec4 别名声明被 `vec4_ok=false` 跳过但 `b0v4` 引用仍在 → `glslc` 12 错；
    matmul 排除去重后即消失（见上项目 5 两处独占槽）。

- **同窗交错 A/B**（text_train d64/L4/H4/ff256/seq256/B64、adam、20 步、
  GPU 40HX、3 对、弃冷启动、`NN_PROFILE=1`）：
  1. **项目 4 单项**（同一 binary，`NN_COPY_FROM_DRAIN` 开关）：
     | 口径 | OFF（旧排空） | ON（fresh） | Δ |
     |---|---|---|---|
     | step 墙钟 ms/20 步 | 1481.8 / 1462.5 / 1477.7（均 **1474.0**） | 1354.6 / 1363.2 / 1353.0（均 **1356.9**） | **−7.9%** |
     | `copy_from` 合计 ms | ~1076 | **~231** | **−78%** |
     | GPU busy% | 90.2 | **95.5** | +5.3pt |
  2. **项目 5 单项**（基线二进制 vs 新二进制，新侧关 fresh 阀 = 只差去重）：
     | 口径 | 基线 | 去重 | Δ |
     |---|---|---|---|
     | step 墙钟 ms/20 步 | 1497.2 / 1494.7 / 1468.0（均 **1486.6**） | 1450.4 / 1456.6 / 1465.2（均 **1457.4**） | **−2.0%** |
     | softmax 重算桶（`86c84dd`→`b58f7bf1`，4 块合计 ms/20 步） | 88.4（均） | **67.1**（均） | **−24.0%** |
  3. **总效果**（基线二进制 vs 两改动全开）：
     | 口径 | 基线 | 全开 | Δ |
     |---|---|---|---|
     | step 墙钟 ms/20 步 | 1505.6 / 1489.2 / 1495.0（均 **1496.6**） | 1380.5 / 1366.6 / 1360.9（均 **1369.3**） | **−127.3 ms（−8.5%）** |
     | E2E ms/step | 74.8 | **68.5** | −6.3 ms/step |
     | `copy_from` 合计 | ~1100 ms | **~234 ms** | −79% |
     | GPU busy% | 89.2~90.5 | **95.4~95.6** | +5.3pt |
     对 torch fp32 50.7 ms/step 差距 **1.48× → 1.35×**。

- **验收**：`cmake --build build` 干净（ninja: no work to do）；ctest
  **29/29**（去重中途两轮挂 `expr_gpu_test`/`fused_gpu_test` 的 Softmax
  闭合世界 miss，补 dry-run 后全绿）；**CPU 字节锚 `f360f9bdbbfbe12c` 与
  GPU dev2 字节锚 `f1f5de1c4420d565` 双双逐位不变**——两改动均数值中性
  （去重只消冗余同值计算、fresh 只改上传时序，求值序/求和序零变化 ⇒ **锚不
  移动**，无需按 K-split 先例重测移锚）；`doc_inventory` exit 0
  （virtual 52 / L2-VIOLATIONS 0 / HANDWRITTEN-RESIDUE 0）、
  `doc_align_audit` exit 0；多架构冒烟（gpt/rapt/mlp/transformer/cnn/f16/
  infer/checkpoint/muon）无闭合世界报错。
- **关键文件**：`expr_dsl.hpp`（`SpecBuilder::add_input` 去重中心 + `RowGather/
  Matmul` 独占槽）、`compute_tensor.hpp`（`same_storage`）、
  `compute_engine.hpp`（`copy_from/write` 的 `fresh_dst` 契约 + `upload_span`
  承诺方）、`compute_cpu_engine.hpp`/`compute_gpu_engine.hpp`（`copy_from_impl`
  新签名 + fresh 分支 + `NN_COPY_FROM_DRAIN` 阀）、`tools/scan_exprs.cpp`
  （Softmax dry-run 补登记）、`build/ab/ab_copy.ps1`/`ab_p5.ps1`（A/B 脚本）。

---

# docs/development 15 · 16（ComputeEngine Refresh：设计与 P0 盘点，2026-10-11 归档）

> **归档说明（2026-10-11 文档整理）**：`docs/development/15-computeengine-refresh.md`（详细设计与未决点）
> 与 `docs/development/16-computeengine-p0-inventory.md`（P0 盘点报告）的原正文随本次整理
> 从 `docs/development/` 撤下，全文并入本文；两份文档的**已落地成果**（P-1 PrecisionEngine 下沉、
> P1 张量出生绑定）与**未实施部分**（P2-P6）均已由现行总纲
> `docs/todo/17-unified-tensor-engine.md`（改期 M1-M7，M1-M6 已实施）承接，无信息丢失。
> 以下为原文（标题降一级；正文路径与行号按归档当时版本书写，行号漂移后以符号名为准）。

## ComputeEngine Refresh —— 详细设计与未决问题

> 本文是 `docs/development/13-refactor-backlog.md` §10 的展开版：补全设计细节、
> 列出模糊点并给出各方案利弊。**本文只记录方案，不实施。**
> 立项时读本文 + 13 §10 + 对应源码现状。历史归档见 `docs/history.md`。
> 修订：D1 已裁定（PrecisionEngine 下沉，§4.1）；D9 已裁定（分层验收，§4.9）；
> D2 补真实代码片段并修正结论（§4.2）。
> **实施进度：P0 完成（盘点见 16）；P-1 完成（验收四件套全过）；
> P1 完成（张量出生绑定 + 跨引擎检查，验收四件套 + NN_BIND_DEBUG
> 诊断零漏网，过程归档 `docs/history.md`）。P2-P6 未实施。**
> **未实施的 P2-P6 已被 `17-unified-tensor-engine.md`（统一底层架构总纲）
> 吸收改期——P2/P3 → M6、P5/P6 → M7、P4 维持正交另案；本文降级为历史引用，
> P-1/P1 成果继续有效。P2 的宿主动词改名（to_host/from_host）已由 17 §3 D9 取消。**

## 0. 一句话

把"张量出生即绑定引擎"下沉为构造期不变量，删掉 Layer/Loss/序列化的**每调用
engine 形参**，用 `import` 收敛跨设备互传，用存储多态消掉容器头的
`#ifdef NN_HAS_VULKAN`；**前置一步：PrecisionEngine 下沉进基类、该类整体删除**。

## 1. 现状事实（可复现坐标，实测核对）

| 事实 | 数字 | 复现 |
|---|---|---|
| `include/` 中 `ComputeEngine&` 出现 | 175 处 | `grep -rn "ComputeEngine&" include/ \| wc -l` |
| `Result<Tensor>`（全仓头文件） | 203 处 | 同上换 pattern（值语义零改动的依据） |
| `GpuEngine::ensure_gpu` 调用点 | 43 处 | `grep -n "= ensure_gpu(" compute_gpu_engine.hpp` |
| `compute_engine.hpp` virtual | 50（49 + 析构） | `grep -c "virtual"` |
| `compute_tensor.hpp` 的 `#ifdef NN_HAS_VULKAN` | 7 处 | 行 28/65/99/161/205/245/304 |
| 宿主中转调用（from/to_matrix/copy_from，排除引擎自身） | ~596 处 | `grep` 后过滤引擎头 |
| `Model::set_engine` | **零调用方 = 死码** | `grep -rn "set_engine(" src/` 空 |
| `PrecisionEngine` | 独立装饰器类（>900 行），49 方法全量委托 `inner_` | `compute_precision_engine.hpp` |
| 测试直构张量（`Tensor::from_matrix`/`Tensor::cpu`） | 98 处 / 8 文件 | `grep -c src/*_test.cpp` |
| 三处绑定先例 | Model(observer_ptr)、Optimizer(`ComputeEngine&`)、`Layer::init` | 各头文件 |

## 2. 目标与非目标

**目标**
1. **P-1：PrecisionEngine 下沉删除**（§4.1）——消解 D1 绑定身份问题。
2. P1-P3：张量出生即绑定 → 删每调用 engine 形参（编译器穷尽驱动）。
3. P3：跨引擎混用升 `Result` 硬错误；`ensure_gpu` → 显式 `import`。
4. P5-P6：存储多态，新后端不改 `compute_tensor.hpp`。
5. P4（正交另案）：宿主格式契约修复。

**非目标（明确不做）**
- `Tensor` 抽象基类/句柄指针化（203 处 `Result<Tensor>` 连锁，13 §10.8 已否决）。
- DSL 入口 `dsl::compute(engine, expr, rows, cols[, P])` 签名不动。
- **去掉的是 PrecisionEngine 这个类，不是"边界 cast"本身**——f16 无变体算子
  "抬 f32 算 → 按 P 落回"的语义照旧，只是换地方住（§4.1）。
- 不改 AOT 闭合世界、不改 batch-major 布局、不动 `PrecisionProfile` 归属。

## 3. 设计详解

### 3.1 绑定的存储位置与传播规则

```cpp
class Tensor {
    observer_ptr<ComputeEngine> engine_ = nullptr;   // 新增，+8B
public:
    [[nodiscard]] bool bound() const noexcept { return engine_ != nullptr; }
    [[nodiscard]] ComputeEngine& engine() const;      // 未绑定 → NN_ASSERT
};
```
- **出生点唯一**：`engine.create_tensor` / `engine.from_host` / 算子输出。
  实现上在引擎工厂函数尾部统一 stamp，避免每个 override 各自漏 stamp。
- **库内配套通道**（D2 修正的关键）：`dsl::compute` 出口、引擎默认实现等
  "有引擎在场但产物走静态工厂"的位置，统一经内部
  `ComputeEngine::adopt(Tensor&&)`（包内/friend，非公有 API）stamp。
- **传播**：`clone/cast/reshape/slice_rows/...` 沿用 `src` 绑定；
  `Tensor{}` 空槽不绑定，`valid()==false`。
- **生命周期契约**：engine 必须比它创建的 Tensor 活得久；一期不加析构探测，
  出悬垂事故再评估 `weak_ptr<EngineToken>`（13 §10.3 第 3 条）。

### 3.2 `import` 协议（唯一对外互传入口）

```cpp
// 目标侧拉取；默认实现 = 经宿主中转；同引擎 fast path 在默认实现内。
[[nodiscard]] virtual Result<Tensor> import(const Tensor& src,
                                            Precision P = /*src 精度*/);
```
- 同引擎 + 同 P = 零拷贝别名；同引擎 + 异 P = 引擎内 cast。
- 异引擎 = `src.to_host → cast_on_host → this->from_host`；引擎可 override
  为同设备直拷（将来 peer copy 调用方零改动）。
- 三个宿主中转特例改名（P2）：`to_matrix→to_host`、`from_matrix→from_host`、
  `copy_from→upload`；`forward/backward` 内禁用（命名即防线，grep 旧名可审计）。
- 别名约束见 D4：**只在同绑定张量之间别名**。

### 3.3 删形参的顺序（P2）

编译器报错驱动，顺序：Layer（`forward/backward/zero_grad/forward_recompute`）
→ Loss → 辅助函数（RoPE、`ActivationOffloader`、`nn_dbg_scan`…）→ CLI/测试。
层内统一用成员 `engine_`（`init` 绑定）。DSL 入口保留形参，层传 `*engine_`。

### 3.4 存储多态（P5）

```cpp
class TensorStorage {                    // 定义在 compute_tensor.hpp
    Device dev_; Precision p_; size_t rows_, cols_;
    virtual Result<void> reshape_view(size_t r, size_t c) = 0;  // 第 8 条
    virtual ~TensorStorage();
};
// CpuStorage<P> / GpuStorage<P>（GpuStorage 只能由 GpuEngine::create_tensor
// make_shared —— 第 7 条，防 storage→backend→engine→tensor 成环）
```
- `Tensor` 只持 `shared_ptr<TensorStorage>`；`#ifdef` 与 backend include 从容器头消失。
- `as_cpu()/as_gpu()`（P6）：检查式 downcast，非虚，句柄仍值类型。
- 逐元素热点（`cpu_get_ptr` 8-70× 教训）：**访问器机制不变**——storage 多态
  只动"容器头怎么拿到 storage"，取出 `MatrixT<P>*` 后的热循环零变化。
  P5 验收必须含 `layer_bench` 无回退。

### 3.5 验收基线（分层，见 D9 裁定）

- build 全绿 + ctest 20/20（GPU 用例退出码 77 = skip 如实记录）。
- **CPU 同设备 pre/post**：loss 序列 + 权重 MD5 **逐字节**。
- **GPU 同设备 pre/post**：P0 实测 run-to-run 稳定性；稳定 → 逐字节，
  不稳定 → 容差（loss ε + 每张量 max-abs-diff ε）。
- **跨设备（CPU↔GPU）**：永不逐字节，只比容差。
- **scan 产物 hash**（`expr_specs.bin` / `fused_registry.hpp`）：与设备无关，
  恒逐字节。

## 4. 设计点逐条裁定

### 4.1 D1【已裁定】PrecisionEngine 直接去掉——下沉为基类 NVI

**裁定：能去，且必须去。** 精度在张量出生时已由 `PrecisionProfile` + 显式 P
定死，"装饰器每次现问内层要不要 cast"本身就是错位——**能力查询已经长在
接口上**（`supports_native_data_move` / `supports_native_f16_reduce` /
`supports_expr_precision_variant` 就是给这个决策用的），说明正确的分层是
"基类问能力 → 基类决定 cast → 引擎只管算"：

```cpp
class ComputeEngine {
public:
    // 非虚入口（NVI）：原 PrecisionEngine 的边界 cast 逻辑住这里
    [[nodiscard]] Result<Tensor> matmul(const Tensor& A, const Tensor& B,
                                        bool tA, bool tB, Precision P)
    {
        if (/* 输入含 f16 且 !supports_f16_matmul(A,B) */)
        {
            auto a32 = cast(A, Precision::F32);          // 抬
            auto b32 = cast(B, Precision::F32);
            auto r   = matmul_impl(*a32, *b32, tA, tB, Precision::F32);
            if (!r) return std::unexpected(r.error());
            return cast(std::move(*r), P);               // 按 P 落回
        }
        return matmul_impl(A, B, tA, tB, P);
    }
protected:
    virtual Result<Tensor> matmul_impl(...) = 0;         // 引擎只实现这个
};
```

**收益**
1. **D1 消失**：没有装饰器 = 没有"stamp 谁"问题，张量出生绑定的就是唯一引擎。
2. `text_train.cpp:955-966` 的 `optional<PrecisionEngine>` + `static_cast` 舞步删除；
   `scan_exprs` 的 f16 dry-run pass 同理（它现在显式包一层跑 f16 登记）。
3. 49 个方法的全量委托样板（`begin/end_batch`、工厂、扫描……逐个转发 `inner_`）消失。
4. in-place 存储精度不变量（`cast_into` 写回原存储）、f16 临时量归因
   （`note_temp_`）各归一处，不再有"适配层管一半、引擎管一半"。

**成本与风险**
1. 49 个 virtual 改名 `*_impl` + 三引擎 override 跟随（编译器驱动，机械改动）。
2. f32-only 路径多一个可预测分支（今天全 f32 不包适配层）→ `layer_bench` 对拍。
3. 归因 instrumentation 挪家（建议基类入口内、环境变量门控）。
4. `compute_engine.hpp` 会变大——与 §2 "拆大头文件"方向不冲突（删的是另一个
   文件），但拆分顺序要排在下沉之后。

**插入时机**：新增 **P-1**，排在 P0（字节基线已建立）之后、P1（加绑定）之前——
先下沉，P1 就不用讨论 D1；用 P0 基线验收下沉"逐字节零回归"。

### 4.2 D2【修正后】"唯一创建入口" vs 满地静态工厂——真实代码片段

原稿说 13 §10.3 第 1 条"非绑定态仅剩 `Tensor{}`"**过强**，核对后修正：冲突点比
想象的窄，库内几乎都有引擎在场，缺的只是一个 stamp 通道。

**片段 A：`expr_dsl.hpp` eval_cpu（约 :1290-1302）——CPU 求值出口**

```cpp
template <typename E>
[[nodiscard]] Tensor eval_cpu(const E& e, std::size_t rows, std::size_t cols,
                              Precision P = Precision::F32)
{
    Matrix out = Matrix::make_uninitialized(rows, cols);
    eval_into_span(e, out.span(), cols);
    if (P == Precision::F16)
    {
        MatrixT<Precision::F16> h(rows, cols);
        ...
        return Tensor::from_matrix(std::move(h));   // ← 产物未绑定任何引擎
    }
    return Tensor::from_matrix(std::move(out));     // ← 同上
}
```
调用方 `dsl::compute(eng, ...)` **手里有 `eng`**，但 CPU 分支直接
`return eval_cpu(...)`。产物是"valid 但未绑定"的 CPU 张量，随后流进下一层
`forward`——若 P1 检查写成 `input.engine() != engine_ → 报错`，**这条正常
CPU 训练路径会被误杀**。→ 修法：`dsl::compute` 出口统一
`return eng.adopt(eval_cpu(...))`（内部通道，非公有 API）。

**片段 B：`expr_dsl.hpp` 扫描占位（:1344-1358）**

```cpp
#ifdef NN_EXPR_SCAN
    (void)eng;                                     // ← 引擎在场，被刻意丢弃
    auto [spec, inputs] = to_expr_spec(e);
    ...
    fused::global_registry().add(spec, sig);
    return P == Precision::F16 ? Tensor::cpu<Precision::F16>(rows, cols)  // 未绑定
                               : Tensor::cpu(rows, cols);                 // 未绑定
#endif
```
占位张量会流进 dry-run 下游层（缓存、切片、后续表达式叶子）。→ 修法同 A：
占位也 `eng.adopt(...)`（scan 模式 eng 是真引擎，CpuEngine/PrecisionEngine）。

**片段 C：`src/fused_gpu_test.cpp` :72-76——测试双引擎互喂**

```cpp
const Tensor qc = Tensor::from_matrix(Matrix(q));  // 宿主直构，未绑定
auto cr = rope_cpu.apply(cpu, qc, seq, backward);  // 喂 CPU
auto gr = rope_gpu.apply(gpu, qc, seq, backward);  // 同一个 qc 再喂 GPU
```
98 处测试直构张量是**库外**代码。若严格执行"非绑定不得参与计算"，全要改成
`cpu.from_host(q)` + `gpu.from_host(q)`（或 `gpu.import(qc)`）。

**片段 D：`compute_engine.hpp` :104/:117——基类默认实现**

```cpp
[[nodiscard]] virtual Result<Tensor> create_offload_buffer(std::size_t)
{
    return Tensor::cpu(1, 1);    // ← 看似无引擎，其实 this 就在——可 stamp
}
```
说明：**引擎成员函数里的静态工厂调用都能 stamp（有 `this`）**——绑定范围以此为准。

**修正结论（分库内/库外）**
- **库内不变量保持**（13 §10.3 第 1 条不作废）：所有库内产物出生即绑定——
  靠两个配套：① `ComputeEngine::adopt` 内部 stamp 通道（DSL 出口、默认实现、
  扫描占位）；② 工厂尾部统一 stamp。
- **库外（测试/示例）**：P2 起迁移到 `engine.from_host`；编译器抓不到
  （运行期才炸）→ P1 检查带 `NN_BIND_DEBUG=1` 开关，ctest 在该开关下把
  "未绑定输入进引擎"也报错，抓漏网；P3 视情况转为默认开。

### 4.3 D3【倾向定案】绑定判等粒度

指针判等；**P1 检查仅在"两操作数都 bound 且指针不同"时报错**，未绑定输入按
4.2 库外豁免放行（避免 P1 零行为变化阶段 ctest 变红）。双引擎对拍测试在 P3
改显式 `import`。同设备多 `GpuEngine` 实例（共享 `GpuBackend::instance()`）
按不同引擎处理——这正是"显式化"的目的。待 P0 测试盘点最终确认。

### 4.4 D4【待定案】`import` 别名语义

同引擎 fast path 的零拷贝别名**只允许发生在同绑定张量之间**；跨引擎产物必是
新 storage + 新绑定。否则"绑定后不可换引擎"不变量被共享 storage 打破。
倾向此约束，待写进 13 §10.6。

### 4.5 D5【待定案】`to_host/upload` 的空槽与 Const

未绑定张量调 `to_host` → `Result` 错误（"unbound tensor: use Matrix directly"）。
`upload` 与 `cast_into` 同一红线：**不替换 dst 对象**（否则共享句柄的缓存静默失联）。

### 4.6 D6【待定序】P4 三处宿主格式契约

1. `model_serialization.hpp:651-674` 同精度路径丢弃 `from_matrix` 结果
   （每参数白做一次全量上传）——P2 改名时要预先标注"计划内行为变化"。
2. `CpuEngine::copy_from` 硬编码 F32 槽（`compute_cpu_engine.hpp:210`，
   f16 dst 在 NDEBUG 下空指针）——**先修再搬**还是搬完再修需定序。
3. `matmul_with_bias` 默认实现 host 往返兜底是死码（三引擎均 override）——
   删前 grep 确认无第四个实现者。

### 4.7 D7【待定案】`Model::set_engine` 死码

零调用方；绑定后"换引擎"与"层已 init 绑定"冲突。倾向直接删除；保留则须
定义为"只能在 `add` 层之前调用"。

### 4.8 D8【待定案】dummy/占位张量

`make_dummy_`（RAPT）本就走 `engine.create_tensor(1,1)` → 自动 stamp，无事；
`create_offload_buffer` 等 CPU 默认实现从 `Tensor::cpu(1,1)` 改为
`create_tensor(1,1)`（同 4.2 片段 D）。P2 顺手处理。

### 4.9 D9【已裁定】字节基线分层——跨设备永不逐字节

**裁定依据**：CPU/GPU 即使精度相同，硬件差异（归约顺序/分块、FMA 收缩、
shader 内循环结构）也会造成小幅偏移；GPU 侧另有原子累加顺序（embedding 反向
`scatter_add_rows` 类路径）这类 run-to-run 非确定源。所以：

| 对比维度 | 验收标准 |
|---|---|
| CPU，改动前 vs 改动后（同 seed 同机） | **逐字节**（分块边界只由 total 决定，确定性有保证） |
| GPU，改动前 vs 改动后（同 seed 同设备同驱动） | **P0 实测两次裸跑**：稳定 → 逐字节；不稳定 → 容差（loss ε + 每张量 max-abs-diff ε） |
| CPU vs GPU（同 seed） | **永不逐字节**，只比容差（现状测试已如此，如 `err < 1e-4`） |
| scan 产物 hash | 恒逐字节（构建期产物，与设备无关） |

P0 的"字节基线"因此拆成两半：**同设备 pre/post 锚点**（不变）+
**GPU run-to-run 稳定性实测**（新增，决定 P1/P5 的 GPU 验收档位）。

## 5. 分期速查（含新增 P-1）

> P0 已完成，盘点结果见 `docs/development/16-computeengine-p0-inventory.md`
> （GPU 档位实测尚缺，见 16 §7-2 未决项）。

| 期 | 内容 | 关键点 |
|---|---|---|
| P0 | 0 改动调研：43 ensure_gpu 分类 / 175 打标 / 字节+scan 基线 / 宿主中转盘点 / **GPU run-to-run 稳定性实测** | D9 实测、D6 调用方 |
| **P-1** ✅ | **PrecisionEngine 下沉删除（NVI）**：基类非虚入口 + protected `*_impl`（CPU 33 / GPU 29 处机械改名），`compute_precision_engine.hpp` 删除，7 处使用方迁移 | 验收全过：dev1 ctest 20/20（含 4 个 f16 用例）、CPU 探针与 P0 基线逐字节、scan 产物 hash 不变、layer_bench 交错 4 轮无系统性回退（A/B 过程归档 history.md） |
| P1 | 加绑定 + 跨引擎检查（只加不改） | ✅ D2（adopt 通道）、D3（指针判等）已落地，见 §7.2 交接 |
| P2 | 删形参 + 宿主动词改名 + 测试 `from_host` 迁移 | **M6 段 C ✅（Layer 范围）**：`forward/backward/zero_grad/forward_recompute` 去 engine 形参 + `init` NVI 绑定；Loss/DSL/辅助对象保留形参（理由与清单见 17 §8）。**D6/D7/D8 已裁定（M6 段 A，17 §8 U4）**；宿主动词改名已由 17 D9 取消（保留 `from_matrix/to_matrix`） |
| P3 | 检查升硬错误 + 43 处 ensure_gpu → import + 序列化去 `model.engine()` | D4、D5 —— **M6 段 B 已完成前两项**：跨引擎检查自 P1 起默认硬错误，`ensure_gpu`→`import` 全量迁移（17 §4.3，虚表 49→50）；第三项**裁定保留 `model.engine()`**（M1 后 save/load 的引擎来源只有"调用方显式传"或"模型回查"两条，改签名会破坏 `save_model(model, path)` 公开 API，收益为零） |
| P4 | 宿主格式契约（**正交另案**） | — |
| P5 | 存储多态 + reshape virtual | `layer_bench` 热点回归 |
| P6 | as_cpu/as_gpu 访问器（纯加法） | — |

## 6. 与既有文档关系

- 方案出处与选型记录：`docs/development/13-refactor-backlog.md` §10（含 §10.8 否决记录；
  §10.3 第 1 条按本文 4.2 补"库内/库外"脚注）。
- 引擎接口盘点：`docs/todo/12-compute-engine-inventory.md`。
- 精度边界语义（P-1 下沉对象）：`docs/development/05-mixed-precision.md` §8/§12。
- 历史归档：`docs/history.md`。

## 7. 当前进度与交接

### 7.1 已完成

| 期 | 产出 | 验收（均可复现） |
|---|---|---|
| P0 | `16-computeengine-p0-inventory.md`（ensure_gpu 43 分类 / ComputeEngine& 175 打标 / 宿主中转 381 清单 / ctest 双基线）；`src/gpu_stability_probe.cpp`（app 目标，ctest 仍 20 个） | 见 16 §0 |
| P-1 | PrecisionEngine 删除 + NVI 下沉（本文 §4.1/§5）；7 处使用方迁移；history.md 归档条目 | ① `NN_VULKAN_DEVICE=1 ctest` = 20/20；② `build/gpu_stability_probe --steps 20` 与 P0 基线逐位一致（`grep -E '^(run1)'` diff 为空）；③ `sha256sum build/generated/expr_specs.bin` = `bdc3a442a5a47a4cf6c0b3ced83c05c870c6fc4ce503681a4a97aca5d6a58360`、`fused_registry.hpp` = `d751008c3ad6074ff444ae59a75ef7c0137de988428644843f37d6581b1df714`（scan 产物锚点，后续重构不得变动） |
| **P1** ✅ | **张量出生绑定 + 跨引擎检查**（本文 §3.1/§4.2/§4.3）：`Tensor +8B engine_`（`bound()`/`engine()`，拷贝/reshape 传播）；`create_tensor`/`from_matrix`/`cast`/`create_offload_buffer`/`offload_restore` NVI 化（CPU/GPU 共 8 处 override 改名 `*_impl`，基类 virtual 仍 49）；20 个返回 Tensor 的公共入口统一 `stamp_`；`bind_check_` 按 D3 字面（双方 bound 且不同 → 硬错误；单侧未绑定豁免；`NN_BIND_DEBUG=1` 未绑定也报错）插入约 28 个入口；`adopt` friend 通道给 `nn::dsl::compute/compute_reduce`（默认实参在前置声明处）。history.md 归档条目 | 本机（Debug+Ninja+clang，Windows）：① ctest 20/20 ×2；② CPU `gpu_stability_probe --steps 20` 与 P0 基线**逐字节**；③ `expr_specs.bin` hash = `bdc3a442…a58360` 不变（=15 锚点）、`fused_registry.hpp` = `7a10412c…` 不变（含 SPIR-V 字节，仅本机 pre/post 可比，Termux 锚点 d751… 不适用）；④ layer_bench stash 往返 A/B：linear/layernorm/mha/transformer 全 ±6% 噪声内无回退；`text_train` 9 步 244.7s vs 245.0s 持平；⑤ `NN_BIND_DEBUG=1` 诊断零漏网：mnist MLP（CPU+GPU）/CNN/Transformer、text_train gpt/rapt（CPU+GPU）、zipt_test |

### 7.2 下一步：P2（删形参 + 宿主动词改名 + 测试 `from_host` 迁移）

1. 先读：本文 §3.3（删形参顺序：Layer → Loss → 辅助函数 → CLI/测试，编译器
   报错驱动，层内统一用成员 `engine_`（`init` 绑定），DSL 入口保留形参）+
   16 §3（`ComputeEngine&` 175 处打标，REVIEW 7 处逐条判定）+ 16 §4（宿主中转
   381 处改名清单：`to_matrix→to_host`、`from_matrix→from_host`、
   `copy_from→upload`；测试 306 处顺手迁 `engine.from_host`）。
2. 改动面：编译器报错驱动删形参（P1 结束时形参都还在）；宿主动词改名后
   **grep 旧名即可审计"计算路径是否还在绕过 import 直接用宿主中转"**（命名即防线）。
3. 未决点（P2 立项时裁定）：D6 定序（`model_serialization.hpp` 白做上传 /
   `CpuEngine::copy_from` F32 槽硬编码的"先修再搬还是搬完再修"）、D7
   （`Model::set_engine` 死码删除）、D8（dummy/占位张量改 `create_tensor`）、
   16 §3 REVIEW 7 处。
4. 验收：同 P1 四件套 + 库外测试迁移后 `NN_BIND_DEBUG=1` ctest 可作为门禁
   （P1 时它只是诊断开关，库外直构输入仍会报错）。
5. 注意：engine 生命周期需比其创建的 Tensor 长（§3.1）；析构探测仍不加
   （13 §10.3 第 3 条留待出事故再评估）。

### 7.3 搜置与未决（后续期处理，均不阻塞 P2）

- **GPU 档位未定**：`gpu_stability_probe --gpu` 首个 `vkCreateBuffer` 即崩（16 §7-2，同设备同构建下既有测试全部正常 → 差异在探针路径，未定位；**Windows 本机同样复现**，P1 验收时记录）。修好后回填 16 §6 GPU 行，D9 的 GPU 验收档位（逐字节 vs 容差）才有结论。
- ~~**`text_train --model zipt` 既有故障（非 Refresh 范围，未修）**~~：**已随 ZiPT 整体移除
  而消失**（见 `docs/history.md`）——原故障 = 启动即调 `set_checkpoint_every(0)`
  → `compute_layer_zipt.hpp` 该 override 不判 `stride==0` 直接 abort，且 abort 后进程挂死。
- **Mali（GPU0）offload 非确定失败**（16 §5/§7-1）：设备级问题，另立 issue，不阻塞 refresh；**ctest 基线以 `NN_VULKAN_DEVICE=1`（Lavapipe 20/20）为准**（Termux 口径；Windows 本机 ctest 20/20 默认选卡即可）。
- P0 遗留：`ComputeEngine&` REVIEW 7 处待 P2 逐条判定（16 §3）；宿主中转口径勘误 596→381，P2 改名清单按 381（16 §4）。
- layer_bench 精确性能复测：P-1 时本机为手机（DVFS/调度噪声 ±30%）；Windows 桌面 P1 A/B 已完成（±6% 噪声内，见 §7.1），但为 Debug 构建口径，Release 精确数字仍待补。

### 7.4 环境注记（本机 Termux/Android，后续会话复用）

- Vulkan 两卡：GPU0 = Mali（apiVersion 1.1.177，自动选卡落点）、GPU1 = Lavapipe（1.4.354）；选卡 `--gpu` / `NN_VULKAN_DEVICE`（`cli/cli_gpu_option.hpp`）。
- 验收命令集：
  ```bash
  cmake -B build -G Ninja -DNN_ENABLE_TESTS=ON && cmake --build build -j8
  NN_VULKAN_DEVICE=1 ctest --test-dir build          # 门槛：20/20
  build/gpu_stability_probe --steps 20               # CPU 逐位（exit 0）
  sha256sum build/generated/expr_specs.bin build/generated/fused_registry.hpp  # 对 §7.1 锚点
  ```

---

## ComputeEngine Refresh P0 盘点报告

> 配套文档：方案见 `docs/development/15-computeengine-refresh.md`（分期 P0-P6；
> 未实施的 P2-P6 由 `17-unified-tensor-engine.md` §5 改期 M1-M7），
> 选型出处 `docs/development/13-refactor-backlog.md` §10。
> **本文是 P0（0 代码改动调研）的实测结果**；未决点与失败项只记录、不修复。
> ⚠ 探针构建前提（2026-10-10）：本文引用的 `gpu_stability_probe` 属**探针**，
> 只在 `-DNN_ENABLE_PROBES=ON` 配置的构建里创建目标（默认 `cmake -B build` 不再
> 产出它；见 AGENTS §2 探针统一编译门）。下述命令需先按该选项配置。
>
> 除新增探针 `src/gpu_stability_probe.cpp`（P0 工具，app 目标不注册 ctest）外，
> 未改动任何库代码。

## 0. 结论速览

| P0 项 | 状态 | 结果 |
|---|---|---|
| 43 处 `ensure_gpu` 分类 | ✅ | 23 普通上传 / 6 调用方句柄重绑定 / 14 Result 传播 |
| 175 处 `ComputeEngine&` 打标 | ✅ | KEEP 58 / DROP 110 / REVIEW 7 |
| 宿主中转调用盘点 | ✅ | 引擎外 381 处（include 71 + src 306 + 探针 4）；13 §10 的 596 为宽口径，见 §4 |
| CPU 字节基线可行性 | ✅ | 探针实测：进程内两轮 + 跨进程两次启动，loss 序列与参数校验和**逐位一致** |
| GPU run-to-run 稳定性（D9 档位） | ✅ | 崩溃根因 = 探针漏 `backend.initialize()`（§7-2，M3 期间定位并修复）；修复后 Windows 本机 dev0/dev2/dev4 进程内两轮 + 跨进程两次启动 loss 与 hash **逐字节一致**（§6） |
| ctest 基线 | ✅（带发现） | 默认（自动选卡=Mali）18/20；`NN_VULKAN_DEVICE=1`（Lavapipe）20/20；Mali offload 失败**非确定**（§5） |

## 1. 环境（本机实测基线）

- Clang 21.1.8 / CMake（Makefile 生成器）/ RelWithDebInfo / `-march=native=ON`。
- Vulkan 实例 1.4.363，两设备：
  - **GPU0**：vendorID `0x13b5`(ARM/Mali)，apiVersion **1.1.177**
  - **GPU1**：vendorID `0x10005`(Mesa/Lavapipe)，apiVersion **1.4.354**
- 自动选卡按（设备类型权重, apiVersion）打分 → **本机落到 GPU0（Mali）**。

复现：`vulkaninfo --summary`；选卡逻辑 `backend/compute_vk_device.hpp:188-258`。

## 2. `ensure_gpu` 43 处分类（P3 → `import` 的迁移清单）——**已全部迁至 `import`（M6 段 B）**

> **当前状态**：`ensure_gpu` 已从 `compute_gpu_engine.hpp` 删除，48 处命中（43 调用 + 定义/注释）
> 全部改为公共入口 `import(...)`（17 §4.3）。原分类表保留为迁移依据；**复现命令已失效**
> （`grep "= ensure_gpu("` = 0 处），改用 `grep -n "= import(" include/neuralnet.cpp/compute_gpu_engine.hpp`。
> 语义落点：`GpuEngine::import_impl` = 原 A/C 类的宿主直传快路径；**B 类 6 处的"调用方句柄重绑定"
> 保持原样但显式化**（`if (A.is_cpu()) A = std::move(*import(A));` 的形态不变，`import` 本身
> **返回新句柄、不改写 src**——与 `cast_into` 不替换红线一致）；D4/D5 由 17 §3 D11 之外的
> "同设备别名只在同绑定张量之间"由 `bind_check_` 承担。顺带修复：原 `ensure_gpu` 硬取
> `cpu_matrix()`（F32 槽）对 f16 源会取到空指针，现 f16 源走宿主中转分支。

复现（迁移前口径）：`grep -n "= ensure_gpu(" include/neuralnet.cpp/compute_gpu_engine.hpp`

| 类 | 数 | 形态 | P3 含义 |
|---|---|---|---|
| A 普通上传 | 23 | `auto x_gpu = ensure_gpu(x);`（原句柄不动） | 直接改 `engine.import(x)` |
| B **调用方句柄重绑定** | 6 | `if (A.is_cpu()) A = std::move(*a_gpu);`（行 1017/1055/1068/1091/1113/1126，in-place 型算子） | **语义敏感**：引擎改写调用方 `Tensor&` 对象——与 `cast_into` "不替换 dst" 红线同族，`import` 重绑定语义必须先定案（15 D4/D5） |
| C Result 传播 | 14 | `auto k = ensure_gpu(K); if (!k) return std::unexpected(...)`（行 1165-1171 / 1213-1216 / 1247-1249，多操作数算子） | 同 A，错误传播保持 |

附带：另有 4 处 `std::move(*c)` 重绑定**局部** f32/f16 提升临时量（行 517/897/1005/1030，
`i32/c32/b16/b32`），不涉及调用方句柄，P3 随边界 cast 下沉（P-1）一并处理。

## 3. 175 处 `ComputeEngine&` 打标（P2 删形参的工时预算）

复现：`grep -rn "ComputeEngine&" include/ --include=*.hpp`（清单存 `~/p0logs/ce_ref_full.txt`，
可按 §3 规则重跑）

| 桶 | 数 | 规则 | 构成 |
|---|---|---|---|
| **KEEP** | 58 | `init(`、`compute_optimizer.hpp`、`expr_dsl.hpp`、`model_container.hpp`、`domain_*.hpp` | 绑定入口 `init` ×22、Optimizer 构造/成员 ×9、DSL `compute/compute_into/compute_reduce`+`cpu_prepare` ×11、Model 容器 ×3、domain 工厂 ×13 |
| **DROP** | 110 | 其余（forward/backward/zero_grad/export/import 辅助、Loss 7 处、未用 `/*engine*/` 形参、`nn_dbg_scan` 等） | Layer 类占大头（gpt 19、rapt 15、zipt 12、mlp 12、transformer 10、conv 10、attention 9、base 8、softmax 2、feedforward 2…） |
| **REVIEW** | 7 | `generate` 推理入口 ×3、`load_image_tensor_from_csv_line` ×1、梯度裁剪辅助 ×1、多行签名挂起 ×2 | P2 立项时逐条人工判定（是否改走成员 `engine_`） |

> 结论：**DROP 是纯机械改动**（编译器穷尽驱动）；KEEP 集中在 5 类"绑定/创建"边界，
> 与 15 §3.3 的 P2 顺序一致。

## 4. 宿主中转调用（P2/P3 改名清单）

口径：`\.from_matrix(|\.to_matrix(|\.copy_from(`，**排除 4 个引擎实现头**（引擎内部实现不算调用方）。

- **引擎外合计 381** = include 71 + src 306 + 新增探针 4（P0 盘点时点 377）。
- **生产代码 71 处**（P2 改名 `to_host/from_host/upload` 的真实工作量）：

| 文件 | 数 | 备注 |
|---|---|---|
| `model_serialization.hpp` | 11 | **P3 主目标**（15 D6-1：同精度路径丢弃 `from_matrix` 结果的白做上传就在这里） |
| `compute_layer_rapt.hpp` | 13 | 层内常量/状态中转 |
| `compute_layer_attention.hpp` | 8 | mask/bias 构造 |
| `compute_layer_gpt.hpp` | 7 | PE/位置索引/token 表构造 |
| `compute_loss.hpp` | 5 | loss 标量回读 + labels/mask 上传 |
| `compute_layer_mlp/transformer/zipt/conv` | 5/6/6/4 | 权重与归一化常量 |
| `compute_optimizer.hpp` | 2 | 梯度范数回读 |
| `cli_mnist_io` / `domain_mnist` / `compute_layer_base` | 2/1/1 | CLI 与调试回读 |
- **测试 306 处**：P2 顺手迁移 `engine.from_host`；`NN_BIND_DEBUG=1` 兜底（15 §4.2）。
- **勘误**：13 §10 记 "596" 为宽口径（含 `Tensor::from_matrix` 静态工厂与引擎头内部）；
  改名清单按本节口径（381）做，避免把静态工厂误纳入 `engine.` 前缀改名范围。

## 5. ctest 基线（含设备差异发现）

| 配置 | 结果 | 细节 |
|---|---|---|
| 默认（自动选卡 → Mali GPU0） | **18/20** | `gpt_offload_test`、`rapt_offload_test` 失败 |
| `NN_VULKAN_DEVICE=1`（Lavapipe GPU1） | **20/20 全绿** | 52.7s |

**Mali 上 offload 失败是非确定的**（同二进制连跑两次）：
- run1：`grad[7] max_abs=0.0226 FAIL`（89 项 OK）
- run2：`grad[79] max_abs=0.372 FAIL`
- 两次失败下标/数值均不同 → **GPU0（Mali）执行非确定 + 结果错误**，不是固定 bug 复现。

**P0 处置（记录不修）**：
1. Refresh 期间的 ctest 基线**以 `NN_VULKAN_DEVICE=1`（20/20）为准**；
   Mali 非确定问题另立 issue，与 Refresh 解耦。
2. 该发现同时说明：D9 的"GPU 同设备逐字节档位"至少在 Mali 上**不可能达成**。

## 6. run-to-run 稳定性实测（D9，探针 `gpu_stability_probe`）

工具：`src/gpu_stability_probe.cpp`（新增，app 目标）。单进程内两轮完全相同的
微型 GPT 训练（init 后固定公式覆写全部参数——P0 时为绕开 `thread_local rng(random_device)`；
M2（17 §5）起 init 已收编为确定性 InitSpec，覆写保留以**隔离 init 变量**、只测后端执行确定性），
比对每步 loss（%.9g）与全参数 FNV-1a 校验和；退出码 0=逐位一致 / 1=非确定 / 2=错误。
M2 新增 `--init-hash` 模式：只建模型不训练，比 6 类模型（mlp/cnn/transformer/gpt/zipt/rapt）
的**初值**确定性（M2 专属验收，见 17 §5 M2 行）。

```bash
build/gpu_stability_probe --steps 20            # CPU
build/gpu_stability_probe --init-hash           # 初值确定性（M2；跨进程 grep '^INIT1' | diff）
build/gpu_stability_probe --io-roundtrip [--gpu] # 批量读写语义对拍（M3；GPU 即 staging 验收）
build/gpu_stability_probe --gpu 1 --steps 20    # 指定 GPU
## 跨进程比对：两次启动后 grep -E '^(CONFIG|run1)' | diff -
```

| 设备 | 进程内两轮 | 跨进程两次启动 | 结论 |
|---|---|---|---|
| **CPU** | ✅ loss 序列 + 参数 hash 逐位一致（exit=0，两轮独立进程均 PASS） | ✅ 逐位一致 | **CPU 逐字节基线可行**——15 §4.9 第 1 行坐实 |
| GPU（dev1 / dev0） | ⏸ 未测得 | ⏸ | **探针崩溃**（§7-2；已修复，见下一行） |
| GPU（Windows 本机，修复后） | ✅ dev2 `8ef51b2927253c50`、dev0 `47977edc71a21a20`、dev4 `01b7192bd15a59d9`（各两轮 PASS） | ✅ 三设备 `CONFIG\|run1` 行逐字节一致 | **GPU 逐字节档位在本机三设备坐实**；dev1（MESA/D3D12 转译）初始化即报 DXIL container 校验错误、exit=2——设备级不可用，非探针/库问题 |

## 7. 未决问题（只记录，均不阻塞 P-1/P1 立项）

1. **Mali（GPU0）offload 非确定失败**（§5）——设备级，另立 issue。
2. ~~**探针 GPU 首 buffer 即崩**~~ **已定位并修复**：根因
   = `GpuBackend::instance()` 只是惰性单例，Vulkan 设备须显式 `backend.initialize()`
   才建立——探针 `make_engine` 漏调（`cli_engine_factory` 与全部 GPU 测试入口都会
   调它，差异因此只出现在探针路径），首个 `vkCreateBuffer` 拿到
   VK_NULL_HANDLE 即 "Invalid device"。修复 = 探针建引擎前补 `initialize()`（空
   selector 仍按 显式 > `NN_VULKAN_DEVICE` 环境变量 > 自动打分 选卡）；§6 GPU 档位已回填（本机 dev0/dev2/dev4 逐字节一致；dev1 为
   MESA/D3D12 设备级 DXIL 校验失败、exit=2，另记）。
3. **REVIEW 7 处**（§3）：P2 立项时逐条判定。
4. **13 §10 数字口径**（§4）：596 → 381，P2 改名清单按 381。
5. **`Model::set_engine` 死码**（15 D7）：维持"删除"倾向，P2 定。

## 8. P0 产出清单

- 静态盘点 §2-§4（P1/P2/P3 工作量与迁移清单齐了）
- CPU 字节基线工具与实测（§6）
- ctest 双基线（§5：dev1 20/20 = Refresh 基线；dev0 记录在案）
- 新增 `src/gpu_stability_probe.cpp` + CMake app 目标（ctest 仍为 20，数量未变）

**P0 收尾**：§7-2 已修复（探针补 `backend.initialize()`），§6 已回填本机三设备实测——D9 的 GPU 档位在本机坐实为**逐字节**；
原机器（Mali）档位受 §5 Mali offload 非确定限制，维持"设备级另立 issue"。

---

# 代码审查整改 批 A/B（RV-01..RV-04，2026-10-11）

来源：`docs/todo/00-code-review-remediation.md`（2026-10-11 外部代码审查的立项台账）。本轮只做
**批 A（RV-01、RV-02）+ 批 B（RV-03、RV-04）**，批 C / 批 D 未动。每条的方案与验收口径见该文
对应 RV 卡；这里只留本文件该管的**历史信息**（被替换掉的旧形态、踩到的坑、环境结论）。

## 批 A/B 整改结论与本仓"四组编译器 × 标准库"验收（2026-10-11）

**RV-01（bug 根因）——"断言后继续解引用"的 Release UB**：整改前全仓 68 处 `NN_ASSERT(...)`，
而它在 `NDEBUG` 下展开为 `((void)0)`；其中"断言后**仍**解引用/索引被校验对象"的点位
（`MatrixT::at`/`set_value`、`Tensor::engine()` 与 4 个存储访问器、`Tensor::reshape`、
`Layer::engine_ref()`、`sample_next_token_`、`Model::{engine,layer_at}`、
注意力/位置编码/checkpoint 系列、`multiply_to_span` 族…）在 Release 是**静默 UB**。
`algebra_matrix.hpp::require_same_shape` 早有"assert + fprintf + abort"的三段式先例，本轮把
该修法收敛成宏并铺满：新增 `NN_ASSERT_ALWAYS(cond, msg)` / `NN_FATAL(msg)`（**无条件**终止、
统一打印 `file:line`），`NN_ASSERT` 退回"可选诊断"语义（现全仓零调用点）。
验收：`src/assert_always_test.cpp`（4 个子进程负例）。**踩坑**：负例判定最初取"tag 后第一个
冒号"当行号分隔符，在 Windows 路径（`D:\...`）上撞驱动器冒号 → 改为"第一个冒号+数字"。

**RV-02（演进记录）——工厂错误协议**：`create_tensor` 旧契约"失败返回空 Tensor + stderr"
（GPU 分配失败连 stderr 都没有），与同族 `from_matrix`/`to_matrix`/`cast`/`read`/`write`/
`import`（全是 `Result`）不一致。现 `create_tensor_impl` 为 `Result<Tensor>`，新增
`try_create_tensor`（含 `InitSpec` 重载）承载"非存储精度 / 形状溢出 / 分配失败"；旧签名保留为
`NN_UNWRAP(try_...)` 转发（115 处调用点零改动，失败从静默变 fail-fast）。库内 4 处 `valid()`
兜底（rapt 三处、`upload_span`/录制占位）改 `NN_TRY` 直抛。

**RV-03（bug 根因）——同目标写序的隐式依赖**：AdamW 的 `p=(1−lr·wd)·p` 与 `p+=delta` 是
**同目标两次写、图上无数据依赖边**（原注释自陈"靠执行序 = 节点序"）；稀疏 CE 的
`grad_reuse` 同理。现把这类隐患抽成显式边（`Tensor::storage_id()` + `ExprGraphNode::dst_id`
+ `dst_order_constraints()` 的 WAW/WAR/写后读三态），融合侧拒绝把隐患两端合进同一 kernel，
执行期用 `validate_dst_order` 兜底 fail-fast；图级计划缓存 key 用"同 dst 的**共享结构**"而非
指针值——否则 Muon 那类"每步新建临时目标"的图每步换 key、白丢 P2-12 的计划缓存。

**RV-04（删除清单）——优化器名单六份副本收敛为一份**：旧副本 = `create_optimizer` 工厂 +
`text_train` 硬编码白名单 + `mnist_train` 帮助 + `gui.py::OPTIMIZER_OPTIONS` + `scan_exprs`
dry-run 列表 + `mem_probe` 状态缓冲分支。现唯一真源 = `compute_optimizer.hpp` 的
`kOptimizerTable`；`mem_probe` 原帮助**漏了 `adamwquant`**，查表后自动补齐。GUI 选项改为
运行时从 CLI `--help` 派生（取不到可执行文件才退回兜底名单，由 `bench/gui_cli_audit.py`
新增的 [4] 节断言与 CLI 声明逐项一致）。

**顺带修的两处构建阻塞**（"四组验证"此前并不真的成立）：① GCC 15 + libstdc++：
`core_profile.hpp::Phase` 持 `std::optional<Scope>` → GCC 在
`~optional→_M_destroy→~Scope→stop()` 内联链上误报 `-Wmaybe-uninitialized`（libstdc++ 的
`optional::emplace` 先置 engaged 后构造载荷），`-Werror` 下 `text_train.cpp` 编译失败；改法 =
状态内联为两个 POD 成员 + 共用新的 `detail::close_frame`（语义等价、stop 幂等）。
② MSVC cl：目录级 `/source-charset:utf-8 /execution-charset:utf-8` 与 `neuralnet_cpp`
INTERFACE 的 `/utf-8` 同处一条命令行 → `cl` 报
`D8016: "/source-charset:utf-8"和"/utf-8"命令行选项不兼容`（MSVC 19.51 / VS18 实测：凡链接
本库的目标全数失败）；改法 = 目录级同样用 `/utf-8`（实测 `cl /utf-8 /utf-8` 合法，只有"混用
两种写法"才冲突）。

**验收（`ethan@192.168.1.102`，四组各自全新构建 + ctest）**：WSL clang++ 24 + libc++ **40/40**
（0 告警；该环境有两个 libstdc++ 安装，clang 驱动报 `-Wgcc-install-dir-libstdcxx`，需
`-Wno-` 压掉——环境性质、非代码问题）、WSL g++ 15.2 + libstdc++ **40/40**、Windows clang-cl
23.1 + MSVC STL（Vulkan 启用）**42/42**、Windows cl 19.51 + MSVC STL **42/42**；
`bench/doc_inventory.ps1` → `L2-VIOLATIONS: 0` / `HANDWRITTEN-RESIDUE: 0` /
`virtual methods 52 declarations, 52 unique`；`bench/doc_align_audit.ps1` → exit 0；
`bench/gui_cli_audit.py` → PASS（含新增 [4] 节）。新增测试目标：`assert_always_test`、
`optimizer_test`（后者含 AdamW `wd≠0` 单步与手写参考**逐位一致**、顺序对调差值 ≠ 0、以及
同目标写序的建边/不合组/反序 fail-fast 三断言）；本机 Linux（clang + libstdc++、CPU-only）
另跑 40/40。

# RV-14 专项审计：VK descriptor set 回收（2026-10-15，只审计）

**背景**：外部审查把"VK Descriptor Set 回收需专项复审"列为待验证项（`descriptor*` 在旧
`compute_vk_backend.hpp` 中出现 199 处；`pending_destroys_` 只管 buffer），立项为 RV-14。
本轮按"只审计、不改代码"执行，审计对象 = `backend/compute_vk_backend.hpp` + 四个片段
（`_destroy/_frame/_pipelines/_ops.inc`，6151 行）。

**做法**：机械清点三类点位（分配 / 归还 / 池重置）后逐点判"是否可能 use-after-free"。
计数实测：`vkFreeDescriptorSets` **27 处**（destroy 1 / frame 1 / pipelines 5 / ops 20），
分配点**唯一**（`alloc_desc_set`），**池重置 0 处**（无 `vkResetDescriptorPool`；池只在
`~GpuBackend` 由 `vkDestroyDescriptorPool` 销毁，且其前一行是 `vkDeviceWaitIdle`）。

**结论 = 闭环**：① batch 模式：`alloc_desc_set` 成功即把集合登记进
`frames_[batch_frame_].desc_sets`，唯一释放点是 `reap_frame`（顺序 = 等该帧 fence → 销毁该帧
延迟 buffer → 释放该帧描述符集 → `in_flight = false`），而 `start_frame` 复用槽前必先
`reap_frame(i, wait=true)` ⇒ 不存在"集合仍被在飞帧引用却被释放"或"命令缓冲已 reset、集合已
释放"的窗口；② solo 模式：`submit_and_wait` 等到 fence 后才释放，其余释放点全在**提交前**的
错误分支；③ 27 处释放点里 19 处是 `if (!cmd_r)`/`if (owns_cmd)` 分支——`acquire_cmd` 的 batch
分支**不可失败**（直接返回预分配的帧命令缓冲），故这些分支只在 solo 模式可达、集合未登记，
单次释放正确；④ buffer 侧：延迟销毁按"当前录制帧"（batch）/最近活跃帧（solo）打标签，标签恒
≥ 任何可能引用该 buffer 的帧 ⇒ 无提前释放；标签在池区间归还时一并释放，不会出现"区间被复用
而 buffer 仍被引用"。

**审计派生一条真缺陷（另立 RV-15，Open，未在本轮修）**：`matmul_q_impl` 的 `#ks`/`#kr` 路径
在 `alloc_desc_set(#kr)` 失败分支里**即时**释放 `ds_ks`——batch 模式下该集合已登记进帧表，帧
被 `reap_frame` 时会**再释放一次**（双重释放）。触发需三条同时成立：batch 录制中 + 池分配在
两次分配之间失败（`maxSets = 262144` 耗尽）+ 调用方从错误恢复继续跑（库内训练路径会终止，
故当前实践不可达）。方案 = 收敛一个 `release_desc_set_now()`（batch 下不即时释放、留给 reap）。

**口径说明**：结论是**代码级生命周期分析**，不是"校验层零告警"对拍——本机
`/usr/share/vulkan/explicit_layer.d/` 只有 MESA/INTEL 层，**无 `VK_LAYER_KHRONOS_validation`**。
复现命令见 `docs/todo/00-code-review-remediation.md` 的 RV-14 卡（审计表 + 分类依据 + 未做项）。

# P2-6「绘制节流」半边：画板拖动开销（2026-10-15）

**立项表述与实测不符（先纠偏）**：13 §5.3 / X5 记的是"`_on_draw_move` 每次 motion
**全量降采样重绘**"。实测当时代码并不做降采样——拖动期的每事件开销是
**逐事件新建画布项**（`canvas.create_line`，一笔长拖 = N 个 item）+ **逐事件 PIL 画线**
（`ImageDraw.Draw(...).line(...)`）。所以"复用 `_RENDER_MS` 节流先例"这条处方对不上病灶：
真正要封顶的是"每个 motion 事件的工作量"与"画布项数随事件数增长"。

**改动（`gui.py`）**：
1. 新增纯逻辑类 `StrokeTrack`（无 Tk/PIL 依赖 ⇒ 可单测）：一笔的点集原地增长，
   `extend()` 对"坐标未变的重复事件"返回 False（**空帧跳过**），`end()` 把折线推入
   待光栅化队列，`drain()` 一次性取走。
2. `_paint_live()`：**复用同一个画布项**（`canvas.coords`）——拖动期画布项数 = 笔数。
3. **视觉刷新按帧节流**（`_PAINT_MS = 16`）：数据侧永不节流（每个事件都进 `StrokeTrack`），
   只限制刷新频率；结束一笔时强制补最后一次刷新。
4. **PIL 光栅化延迟**：`_rasterize_pending()` 按批一次 `draw.line(折线)`，由
   `_downsample_buffer()`（读缓冲的唯一入口）触发 ⇒ 拖动期零 PIL 调用。

**实测（本机，合成事件；`python bench/gui_draw_runtime_audit.py`）**：2000 个 motion 事件
（一半是空帧）**16.6 ms vs 旧写法 37.9 ms（2.3×）**；画布项 **2058 → 59**（网格 58 + 笔迹 1），
即复杂度从 O(事件数) 降到 O(笔数)。功能闭环同测：一笔落下后 28×28 缓冲非零像素 100/784，
`collect_args` 输出 784 值 CSV 且非全零（证明"延迟光栅化"没丢笔迹）。

**门禁**：`bench/gui_draw_logic_audit.py`（`StrokeTrack` 语义 + AST 源码级不变量：拖动不建项 /
按帧节流 / 读缓冲前必先光栅化 / 旧属性零引用；**用桩替换 customtkinter，无需 GUI**）、
`bench/gui_draw_runtime_audit.py`（真机功能 + 开销对照；无显示时退出码 77 = SKIP）；
`bench/gui_cli_audit.py` 退出码 0。
**剩余**：P2-6 的字段表半边（dataclass 元数据自动生成 UI 行与 `collect_args`）未动。

# `gpt_test` 偶发失败（C6 / P1-4）复现结论：未复现 + 同机完全确定（2026-10-15）

**历史记录**：13 §9.5 记"62 次直跑 1 次失败（2026-10-01），日志被覆盖未定位"，列为 C6/P1-4。

**本轮复现（本机 Release + clang 24.0.0 / libstdc++ / 8 线程）**：

| 口径 | 次数 | 失败 | 日志指纹 |
|---|---|---|---|
| 串行直跑（`build-gpu/test/gpt_test`） | 200 | **0** | 200 份**同一 md5** `dc077445bf463a8f0018b91dc0581e3d` |
| 8 路并发直跑 | 8 | 0 | 与串行**同一 md5** |
| 串行直跑（加环境指纹行后） | 40 | 0 | 同一 md5 `8f4b82d6e8330980e3405cadbb754149` |
| `taskset -c 0,1`（**2 核超订 8 线程**，最大争用） | 30 | 0 | 同上，**同一 md5** |

⇒ **同机同码下该测试逐字节确定**（连打印出的 `max_err` / `max_abs` 都完全一致）：不存在
"数值比较随机漂移"的成分。因此历史那一次失败**不能归因于测试自身的随机性**，只剩两类可能：
① **环境差异**（编译器/标准库/构建类型/CPU 指令集——例如 FMA 收缩会改变浮点舍入，
而 gradcheck 是容差判据）；② **当次进程级异常**（分配/线程创建失败等）。两者在本机都不可复现。
统计口径：0/200 ⇒ 本机失败率 95% 置信上界 ≈ **1.5%**，与历史 1/62（≈1.6%）相容。

**处置（不加 retry / skip，理由显式）**：输出逐字节确定 ⇒ 任何失败都必然对应"环境变了或
代码变了"，重试/跳过只会掩盖真实回归。因此只做**可归因性**加固：
`src/gpt_test.cpp` 新增 `print_env_fingerprint()`，输出一行
`[env] <编译器> / <标准库> / <构建类型> / hw_threads=N / FMA=on|off`——历史失败的真正障碍是
**日志被覆盖**（P0-1 已让 CI 用 `ctest --output-on-failure` 保留日志），本行让任何一份失败日志
自带环境坐标，FMA 开/关尤其关键（它是唯一能让 gradcheck 数值在容差边界漂移的编译期因素）。

**复现命令**：
`for i in $(seq 1 200); do ./build-gpu/test/gpt_test > /tmp/r.log 2>&1 || echo FAIL $i; md5sum /tmp/r.log; done | sort -u | wc -l`
→ 1（= 无失败且日志单一指纹）；并发版把命令换成 `&` + `wait`，超订版加 `taskset -c 0,1`。

# C10 归因：RAPT GPU 训练为什么慢（2026-10-15，只定位不改代码）

**立项背景**：issue #15 记"同一命令只换 `--model rapt`，40HX 上 62.9 s vs 3.9 s（≈16×）"，
原怀疑方向（**未验证**）= 短序列下扫描原语的逐 step kernel 启动 + staging 常数。

**复现（本机 850M / Vulkan / d128·h4·L4·ff512 / seq128 / b8 / `--max-steps 3` / `NN_PROFILE=1`）**：
RAPT step **741.81 ms** vs GPT **134.84 ms**（3 步口径）= **5.5×**。（40HX 的 16× 是同一机制的
更大比值：GPT 在那台机器上更快，而往返成本是"每调用一次 drain"的固定时长，与算力无关。）

**结论一：原怀疑方向证伪**。host 侧扫描派发极小——`scan_prefix_outer` 0.80 ms、
`outer_col` 1.38 ms、`scan_suffix_outer` 0.40 ms（合计 ≈0.3% 墙钟）；扫描的问题在
**设备侧 kernel 时长**（见结论三），不在"启动 + staging 常数"。

**结论二：主因是 `rms_norm_forward_` 的逐头 host 往返**。exclusive 表：
`to_matrix` **573.82 ms / 768 次**（= 墙钟 **75%**，单次 747 µs），且全部落在
`fwd.blk#N`（每 block 192 次 / 3 步 = 64/step，恰等于 `2 × batch × heads` = 2×8×4）；
`begin_batch`/`end_batch` 各 **774 次**（258/step）——每次 `download_vector` 都会
`wait_in_flight` 抽干整条流水线，随后相邻算子再开一帧。GPT 同配置只有 6 次 begin_batch。

**A/B 验证（决定性）**：`--checkpoint-every 1` 时 forward **不收集**逐头 1/rms
（`rms_norm_forward_` 的 `rms_inv_out == nullptr` 分支明确"省掉 BH 次 GPU→CPU 下载"）
⇒ RAPT **forward 659.88 → 160.77 ms（4.1×，与 GPT forward 103.19 ms 同量级）**；
总墙钟反而升到 1022.81 ms，因为 backward 的重算路径把同样的下载又做了一遍（768 次 to_matrix
移到 `bwd.blk#N`）——即**下载成本随"哪条路径收集 rms_inv"迁移，不随数学变化**。

**结论三：设备侧最大单项是 `scan_prefix` kernel 本体**。设备 kernel 合计 530.40 ms
（busy 69.3%），其中 `scan_prefix@fwd|bwd.blk#N` = **340 ms（64%）**，单次 8–10 ms
（dk=32 / seq=128 / BH=32）——不是启动开销，是 kernel 本身慢；这是 P4-2 的逐 op 靶子。

**归口（未在本轮改代码）**：两项分别归 **P4-2**——(a) 逐头 1/rms 收集改设备侧拼装
（`insert_rows` 进 (BH,seq) 缓冲，去掉 download+upload 往返；**f16 注意**：现有往返会把值
落成 F32，设备侧改写要保持被消费处的精度一致）+ 逐头循环批量化（同循环还有 1060 次
`slice_rows`/step）；(b) `scan_prefix` tile/并行度调参。复现命令 =
`NN_VULKAN_DEVICE=850M NN_PROFILE=1 ./build-gpu/text_train datasets/tinystories_200mb.nndataset
--model rapt|gpt --d-model 128 --num-heads 4 --num-layers 4 --d-ff 512 --seq-len 128
--batch-size 8 --max-steps 3 --gpu`（对照加 `--checkpoint-every 1`）。

# P0-5 推理采样：top-k / top-p / 重复惩罚 / 种子（2026-10-15）

**改了什么**：
1. `compute_layer_base.hpp` 新增 `nn::SamplingConfig`（`temperature` / `top_k` / `top_p` /
   `repeat_penalty` / `repeat_window` / `seed` + `has_seed`）与扩展的共用采样器
   `sample_next_token_(logits, cfg, rng, dist, history)`；顺序固定 =
   **重复惩罚 → 温度 → top-k → top-p（核采样，保留集内重新归一化）→ 数值稳定 softmax →
   采样/贪心**。确定性口径：top-k 按（值降序, 下标升序）；逆 CDF 按下标升序累加；
   贪心并列取最小下标且**不消耗 RNG**。**全默认（top_k=0/top_p=1/repeat_penalty=1/无 history）
   与旧 temperature 路径逐位一致**（返回下标 + 概率向量 + RNG 状态三者相同），旧 4 参重载保留为转发。
2. `GPTModel` / `RAPTModel::generate` 新增 `SamplingConfig` 重载（旧 `Scalar temperature`
   重载 = 全默认语义）；`has_seed` 时用 `cfg.seed` 播种（否则仍 `random_device`）；
   `history` = prompt + 已生成 token。
3. `src/text_infer.cpp`：`--top-k`（0=关）/ `--top-p`（1.0=关）/ `--repeat-penalty`（1.0=关）/
   `--repeat-window`（0=全部上下文）/ `--seed`；CLI 校验 + 生成横幅回显；GUI「GPT 推理」页
   同步暴露同样 5 个字段（`cli_controllers.GptInferController` + `gui.py` 的 `GptInferTab`）。
4. 新测试目标 `sampling_test`（ctest 48 → **49**；无 Vulkan 树 46 → **47**）：默认配置 ≡ 旧路径
   的逐位等价性、top-k/top-p/重复惩罚/`repeat_window` 语义、贪心不消耗 RNG、模型级
   「同 seed 可复现 + 不同 seed 不同 + `top_k=1` ≡ 贪心」。

**端到端实测（本机 850M，`gpt_model.nnmodel` + `bpe_vocab_stories.nnvocab`）**：
`--max-tokens 16 --seed 7 --top-k 40 --top-p 0.9 --repeat-penalty 1.2` **连跑 3 次输出逐 token 相同**；
换 `--seed 8` 输出不同；`--temperature 0` 与 `--top-k 1` 输出逐 token 相同（贪心等价）。

**三组「编译器 × 标准库」验证（最终源状态）**：

| 环境 | 编译器 / 标准库 | 构建 | ctest |
|---|---|---|---|
| 本机（Vulkan 850M） | clang 24 + libstdc++ | 零告警 | **49/49** |
| `102` WSL（`~/nn-verify`，CPU-only） | clang 24 + **libc++** | 零告警 | **47/47**（GPU 用例 77=skip） |
| `102` WSL（同目录 `build-gcc`） | g++ 15.2 + libstdc++ | 零告警 | **47/47** |

**libc++ 抓到的真实移植性缺陷（已修）**：`sampling_test.cpp` 里用 `{1}` 直接当
`std::span<const std::size_t>` 实参——libstdc++ 接受，**libc++ 报 "no matching function"**
（span 无 `initializer_list` 构造）。改为显式数组 `const std::size_t hist_neg[1] = {1};`。
这正是 AGENTS §2「clang + libc++ 口径更严」那条纪律的又一例。

**未做（按 27 号计划的拆分）**：GUI 的多轮上下文闭环（连续 N 轮不丢上下文）留 P2-6 批次。

# RV-15（descriptor set 双重释放）+ P4-2 (a)（RAPT 逐头 host 往返）合并提交（2026-10-15）

## RV-15：batch 模式下 descriptor set 双重释放（RV-14 审计派生）

**改法**（`backend/compute_vk_backend_pipelines.inc` + `_ops.inc` + `_frame.inc`）：
新增 `release_desc_set_now(VkDescriptorSet)`——**语义与 `alloc_desc_set` 的登记规则严格对齐**：
`VK_NULL_HANDLE` 或 batch 模式（集合已在 `frames_[batch_frame_].desc_sets` 名下）直接返回，
交给 `reap_frame` 统一回收；solo 模式（未登记）立即 `vkFreeDescriptorSets`。
所有「拿到 cmd 之前的错误分支」全部改走它：`_ops.inc` 17 处 + `_frame.inc::dispatch_compute` 1 处，
含**原缺陷点** `matmul_q_impl` 的 `!ds2_r` 分支，以及两处一次释放两个集合的写法
（`{ds1, ds2}` 数组 → 两次 helper 调用）。`owns_cmd` 块内两处直接释放保留（该块只在 solo 可达 ⇒
集合未登记 ⇒ 直接释放正确），并加注释说明依据。

**审计复现**：`rg -n "vkFreeDescriptorSets" include/neuralnet.cpp/backend/*.inc` ⇒ 9 处 =
helper 自身 1 + `submit_and_wait` 5 + `reap_frame` 1 + `owns_cmd` 2，**错误分支 0 处**。
**验证**：本机 850M 全量重建零告警（该 TU 仅在有 Vulkan 时编译）+ ctest 49/49；
102 WSL clang24+libc++ 47/47、g++15.2+libstdc++ 47/47（无 Vulkan 旁证）。

## P4-2 (a)：RAPT 逐头 1/rms 的 host 往返 → 设备侧拼装

**背景**：C10 归因（同日）测得 RAPT 的 `to_matrix` 占墙钟 75%，来源是 `rms_norm_forward_`
为收集逐头 1/rms 缓存而「每头一次 download + 末尾一次 upload」（= 每个 norm 调用 BH 次
GPU→CPU→GPU 往返，每次 `to_matrix` 都会 `wait_in_flight` 抽干整条流水线）。

**改法**（`compute_layer_rapt.hpp::rms_norm_forward_`）：一次 `(BH, seq)` F32 张量 +
逐头 `cast_into(*ri, ri_f32)`（f16 时，f16→f32 精确）→ `insert_rows(rms_out, bh, ...)`，
全部在设备内完成。p_.compute == F32 时直接 `insert_rows`（连 cast 都不做）。
checkpoint 模式（`rms_inv_out == nullptr`）行为不变。

**A/B（850M / Vulkan / `d128·h4·L4 / seq128 / b8 / --max-steps 3` / `NN_PROFILE=1`，配对同窗）**：

| 指标 | before | after | Δ |
|---|---|---|---|
| step（3 步） | 741.81 ms | **521.68 ms** | **−29.7%** |
| forward / backward | 659.88 / 38.05 ms | 372.25 / 118.18 ms | −43.6% / +2.1×（backward 的 GPU 工作此前被 forward 的 drain「隐藏」） |
| `to_matrix` | 573.82 ms / **768 次**（75% 墙钟） | **0 次** | 消除 |
| `begin_batch` / `end_batch` | 774 / 774（258/step） | **6 / 6** | −99% |
| GPU 忙率 / 墙钟 | 69.3% / 765.50 ms | **84.0%** / 543.29 ms | 忙率 +14.7pt，墙钟 −29.0% |

**数值不变性（逐位证明）**：102 WSL 上把 (a) 三段代码单独回退后重建，同一 `rapt_test`
（CPU f32）与 `rapt_test --f16`（CPU f16）的**完整 stdout 逐字节一致**（含逐参数梯度
`max_err` 的 7+ 位有效数字、causal 与 bidirectional 两个 gradcheck、block 级检查）
⇒ `... | diff - && echo BIT_IDENTICAL`。GPU 侧为构造性等价（`cast_into` 的 f16→f32 是
文档化的精确升位 + `insert_rows` 为设备内行拷贝）+ 全量 GPU 套件通过（含 `rapt_test --f16 --gpu`）。

**测量方法论（本轮踩到的坑，写给下一次 A/B）**：
· `text_train` 每个 epoch 用 `random_device` shuffle 样本 ⇒ **训练 loss 轨迹不可复现**
  （实测同命令两次 `9.0271…` vs `9.0245…`，模型文件 md5 也不同）⇒ A/B 只能看墙钟/剖析；
· 数值 oracle 用**确定性测试二进制**：`rapt_test --gpu` / `rapt_test --f16 --gpu`（两次运行逐字节一致），
  或 `rapt_test`（CPU）。

## P4-2 (b)：`scan_prefix` 的延迟/占用改写（2026-10-15，**逐位不变**）

**改了什么**（`shaders/scan_prefix_outer.comp`，专用路径 dk ≤ 64；通用 dk>64 的 `_gen` 不动）：

1. **行归属（row ownership）**：线程 a 独占 A/B 的第 a 行 ⇒「前缀累加 → `B·q`/`A·q` 行读出」由同一
   线程完成、**不需要跨线程可见性** ⇒ per-t barrier 4(5) → **3**；
2. **R 行块 stage 进 shared**（旧写法 `[2)`/`[4)` 每 t 各读全局 R：前者 32×32 次、后者单线程 64 次）
   ⇒ 每 t 的全局往返 3 → **1**；
3. **`[4)` 标量改由 dk 个线程各算一遍（同值）后各写本行**，消掉"单线程串行 2·dk 次"的关键路径；
4. **三条点积与 A/B 行累加 4 路展开**："先并行取回 4 个操作数、再按原序累加" —— 只改 ILP，
   **不改任何元素的累加次序**；
5. **窄共享变体 `scan_prefix_outer_d32[_f16]`**：同一份 `.comp` 加 `-DNN_SCAN_DK_MAX=32` 再编一份
   （A/B 共享 33.3KB → **8.5KB**）⇒ Maxwell 每 SM 可驻留的 workgroup 从 **1** 提到 **6**。
   C++ 侧按 `dk ≤ 32` 选它，缺变体时自动回落宽变体（同数值、只是慢）。

**为什么是这些（消融实验，850M / BH=1 / seq=128 / dk=32；骨架 = 只留每 t 的 3 次 barrier + 一次常量写）**：

| 变体 | ms（128 个 t） |
|---|---|
| 骨架（仅 3×barrier/t） | 0.113 |
| 骨架 + 载入（K/V/P/R 行块） | 0.251 |
| 骨架 + 载入 + A/B 行累加 | 0.732 |
| 完整（改写前结构） | 1.375 |

⇒ **barrier 本身只占 8%**：真正的成本是"每个 t 串起来的共享访存循环、一迭代一次依赖访存"的延迟
暴露（4 路展开让同一循环从 3.75 µs/t 降到 <1 µs/t）。另一条腿是**占用率**：BH=1/2/4 的每 WG 时间
恒 ≈0.75 ms（4 个 WG 落在 4 个 SM 上互不干扰），BH=8 → 2.58 ms、BH=32 → 4.77 ms（32 WG / 5 SM =
6.4 个波次，线性于波次）⇒ 33 KB 共享把每 SM 限到 1 个 WG，窄变体正是打这一条。

**配对 A/B（`layer_bench --op scan_prefix`，同会话 back-to-back；形状 = RAPT d128/h4：BH=32、seq=128）**：

| 形状 | before | after | Δ | 位级哈希（`--hash`，前后必须相同） |
|---|---|---|---|---|
| dk=32 f32 | 10.87–10.94 ms | **1.81–1.87 ms** | **5.9×** | `0d831accfadc5791` |
| dk=32 f16 | 11.86 ms | **3.07 ms** | 3.9× | `3581a1a55dc53ff0` |
| dk=64 f32（宽变体） | 22.33–22.41 ms | **7.44–7.50 ms** | 3.0× | `a3c8515473d7a748` |
| dk=16 f32 | 6.35 ms | **1.44 ms** | 4.4× | `93551ae98ca84979` |

**端到端（RAPT `d128·h4·L4 / seq128 / b8 / --max-steps 3`；旧 kernel / 新 kernel 两个构建各自跑
`text_train` + `NN_PROFILE=1`，同会话配对）**：

| 指标 | 旧 kernel | 新 kernel | Δ |
|---|---|---|---|
| step（3 步） | 888.24 ms | **510.24 ms** | **−42.6%** |
| 墙钟 / 设备 kernel / 忙率 | 934.17 / 761.99 ms / 81.6% | 539.57 / 447.62 ms / 83.0% | −42% / −41% |
| `scan_prefix` 单次（`@fwd.blk#N`） | 9.81 ms | **1.64 ms** | −83% |
| `copy_from`（host staging 等待） | 595.62 ms | 355.67 ms | −40% |

两条口径要点：① **e2e 收益大于"设备 kernel 时间之差"**——host 侧 staging 等待（`copy_from`）与 GPU
忙时成正比，设备侧快了宿主等待随之缩短（与 P2-9 "host 等待总量不变、只是位置可移"同一机制）；
② **e2e 必须同会话配对测**：跨会话绝对墙钟差异可达 1.7×（同命令同 shader：前会话 step 521.68 ms、
本会话 888.24 ms，两会话 GPU 频率都是 810 MHz），只看跨会话数字会得出相反的结论。

**数值不变性（两道独立证明）**：

1. `rapt_test --gpu` 与 `rapt_test --f16 --gpu` 在**旧/新 kernel 两个构建**下的完整 stdout **md5 相同**
   （f32 `9f7722a24dc4f87c2a4bed3cc05aab41`、f16 `abb242917e36205050466fb7f77cfc40`）——含逐参数
   梯度 `max_err` 的 7+ 位有效数字、causal 与 bidirectional 两个 gradcheck、block 级检查；
2. 新增门禁 `scan_bitwise_test`（**CPU 参考 ↔ GPU 逐位比较**，61 个用例）：dk ∈ {16, 31, 32, 33, 64}
   （16/31/32 走窄变体、33/64 走宽变体；31 与 33 覆盖 4 路展开的尾循环）× causal × has_state ×
   has_bnd 全组合 + f16 存储路径 + suffix 全组合，**每个用例逐位一致（0 failure）**。它同时把
   "CPU 实现 ≡ GPU kernel 的累加次序"这一**锚**钉死——此前 `gpu_test` 只覆盖 dk=128 的通用路径且
   容差 2e-3，**专用路径完全没进回归网**。

**被否决的一半（suffix 试了两次都退回原样）**：`scan_suffix_outer` 与 prefix 同族，最初照
"行归属 + 窄变体"改；实测**反而慢**（同会话 dk=32/BH=32：原样 7.82–7.92 ms、行归属版 18.33 ms、
窄变体版 16.18 ms、同尺寸宽变体 13.17 ms），且 WG 越多越差（BH=4 → 1.03 ms、BH=32 → 16.2 ms，
超线性）⇒ 其主导成本是 **D `(dk², seq)` 的列式散布读**（每个元素独占一个 32B sector）：这是
**流量/争用**受限，提高占用率只会加剧争用。**结论：suffix 保持原样**（RAPT 每块 prefix 3 次、
suffix 1 次，不是热点）。

**测量工具（本轮新增，后续 kernel 配对 A/B 都用它）**：`layer_bench --op scan_prefix,scan_suffix
--dk <n> --hash`。`--hash` 在**计时之后**打印输出张量的位级 FNV-1a 64（逐位不变的改写必须哈希
相同）。**放在计时之后是硬要求**：`to_matrix` 会改变池/流水线状态，放计时前会污染 ms——同一命令
实测 `--hash` 前置 8.67 ms vs 后置 10.9 ms（差 25%），纯属测量伪影。

**成本口径（可复用的捷径）**：shader 迭代真的不需要全量重建——SPIR-V 嵌入头进 PCH 只意味着
**PCH + 用它的 TU** 要重编；`ninja -C build-gpu layer_bench` 只重编 PCH + 1 个 TU（≈2 min），
全量重建（8 核 ≈20 min）只在最后做一次。本轮 shader 迭代 8 轮全部按这个口径做。

# P2-6「字段表」半边：GUI 参数改成声明式表（2026-10-15）

**问题**：加一个 CLI 选项要手改两处——`__init__` 里建一行 + `collect_args` 里取值/转换/条件
发送。6 个 Tab、7 处手写 `collect_args`、约 110 个参数行，漏一处就是"界面有、命令没带"
（或反过来收集了不存在的键），而且**没人能静态审计**。

**改法**：`Field`（frozen dataclass）+ 紧凑构造器 `_e/_i/_fl/_o/_b/_file/_dir/_save/_sep/_v`
+ **纯函数 `collect_fields`**（参数语义的唯一实现：option 总是发、entry 空则不发、`skip_zero`
按旧口径跳过 0、check 按 `literal` 发 CLI 字面量或仅在改动时发、`group` + `_field_active`
决定条件发送）+ `ParamsMixin`（建行 / 取值 / 显隐 `_apply_groups`）。`TabBase` 继承 mixin；
**每个 Tab 只剩 `fields()` 一张表 + 少量显式后处理**（`engine → --gpu`、`precision_preset →
--f16/--t1_58`、`input_method → input`、画板导 CSV）。

**逐字段等价性验证（本次重构的核心证据）**：从 `git show HEAD:gui.py` 取出旧的 `collect_args`
方法体，用**同一个 fake self**（控件桩按场景给值）分别驱动旧函数与新表实现，22 个场景比对
kwargs：

| 结果 | 数量 | 说明 |
|---|---|---|
| 逐字段完全一致 | **17** | 含每 Tab 的默认场景 + 架构/调度/预设/引擎切换场景 |
| 有意收紧 | 4 | **fixed 调度下不再发被隐藏的 `min_lr`/`warmup_epochs`**（旧代码会把隐藏控件的默认值一起发出去：`--lr-schedule fixed --warmup-epochs 0 --min-lr 1e-06`；GptTrainTab 原本就是这个口径） |
| 命令行层等价 | 1 | `show_bytes=False`：旧版显式发 False、新版不发 —— 控制器侧 `if kwargs.get(...)` 两者**生成的命令都是空**（实测） |

**门禁**：`bench/gui_cli_audit.py` 新增第 **[5] 节（声明式字段表 ↔ Controller 闭包）**——
表里声明的每个 kwargs 键喂给对应 Controller 后必须真的产生影响（否则这一行表单是死的），
`_v()` 虚拟项（`gpu`/`f16`/`t1_58`/`input`…）一并纳入；`ui_only=True` 的字段（`engine`/
`precision_preset`/`input_method`）显式声明"由后处理映射成别的键"。实测 6 个 Tab 的
35/5/6/8/43/12 个键**全部有效**。

**重构过程中审计抓到的两个真 bug（都在提交前修掉）**：

1. `TabBase` 里遗留的 `collect_args` 桩（`return {}`）**覆盖了 mixin 的实现** ⇒
   Tokenizer 两页 / GPT 推理页 / MNIST 推理页会**一个参数都不发**（GUI 静默失效，
   逻辑上跑得通、只是命令永远是默认值）；
2. 我把 tokenizer 页的属性名写成 `show_bytes`，与旧代码的 `show_bytes_var` 不一致 ⇒
   等价性对照直接抛 `AttributeError`。

**顺带清理**：删掉从未被读的 `_mp_widgets` 死码；`PRECISION_PRESETS` 常量收敛两页重复的
预设字面量；旧的 5 组手写显隐清单（`_mlp_widgets`/`_tf_widgets`/`_cnn_widgets`/
`_norm_place_widgets`/`_lr_schedule_widgets`）由 `group` 声明 + `_apply_groups` 取代。

**复现**：`python bench/gui_cli_audit.py`（exit 0，五节）、`python bench/gui_draw_logic_audit.py`
（exit 0）、`python bench/gui_cli_audit.py` 的 [3] 节同时验"控件删了引用还在"这类只在运行期
才炸的错误。

# P1-1（第二刀）：`text_train` 模块化 + `std::exit` 收敛（2026-10-16）

**类型**：演进记录（结构重构）+ 回归验证（逐字节 A/B）。

**背景**：18 §4.2 P1-1 的后半（前半"退出码语义"见上一节）。要点：`src/text_train.cpp`
1808 行里 CLI 解析 / 训练循环 / TDR 恢复 / checkpoint 混杂，`std::exit` 约 40 处；
E5 记的"任一改动牵动全文件"是真实成本，也是 **P1-2（checkpoint 优化器状态）的前置阻塞**。
接缝盘点（含"TDR 重启续训段与循环状态耦合最紧、抽它要先定义状态结构"）写在上一节的
"未做"里，本轮按那份盘点执行。

**落地形态**（`src/text_train.cpp` 1830 → **396** 行，只留接线 + 唯一出口 `main`）：

| 模块（`include/neuralnet.cpp/cli/`） | 职责 | 行数 |
|---|---|---|
| `text_train_config.hpp` | `TrainConfig` + `--help` 排版 + argv 解析（**CLI 参数解析层 = 全流程唯一允许 `std::exit` 的地方**） | 522 |
| `text_train_loop.hpp` | epoch × step 循环：采样/前向/稀疏 CE/反向/累积/裁剪/优化器、非阻塞 loss 回读、测试评估、`--max-steps`、定期断点、`NN_NAN_TRACE`/`NN_MEM_STATS` 探针、梯度统计 | 988 |
| `text_train_recovery.hpp` | `--resume`：角色门禁 → 读规格 → 按规格重建（GPT/ALiBi-GPT/RAPT）→ 载参 | 127 |
| `text_train_checkpoint.hpp` | `.nnckpt` 路径派生 + 断点保存**记账**（失败只记账不中断）+ 收尾退出码 + `.nnmodel` 产出 | 83 |

**关键决定（为什么这么切）**：

1. **三个模块不终止进程**——错误一律 `Result` 上抛，`main` 是唯一出口（`error` → 打印 +
   退出码 1）。这是 P1-2 能落地的前提：恢复路径必须**可被测试直接调用**，而不是"调用即
   可能结束进程"。探针中止（`NN_NAN_TRACE` 命中）不是错误而是"诊断完成"，经
   `TextTrainLoopOutcome::abort_code = 9` 回报，调用方据此跳过收尾保存与剖析报告
   （与旧 `return 9` 行为逐字一致）。
2. **循环体逐字搬移**，只动三类地方：错误分支（`NN_EXIT` → `Result` 上抛）、少数名字/
   语义替换（`optimizer->` → 引用 `.`、`tokenizer->vocab_size()` → 传入的 `vocab_size`、
   `mem_mark(...)` → 探针对象的 lambda）、断点保存调用。循环里的**输出文案与顺序**是回归锚，
   搬迁期间不动它，让"改动前后 stdout/stderr 逐行相同"成为可机械验证的判据（见下）。
3. **状态归属**：循环需要而 src 原本持有的东西（`lr_sched_cfg`、`optimizer_current_lr`、
   `steps_per_epoch`、测试子集、`eff_seq`、`x/y/loss_mask/flat_targets` 缓冲）一律在 loop 内
   派生或经 `TextTrainLoopInput` 传入——不留"一半状态在 src、一半在 header"的缝合线。
4. **角色门禁的错误文案不加语境前缀**（用 `NN_TRY_CHECK` 而非 `NN_TRY_MSG`）：该文案是
   用户可见契约（docs/todo/23 §2.1），逐字保留。
5. **探针随循环进 loop 模块**：`NN_NAN_TRACE` 的四个 helper 与 `NN_MEM_STATS` 的
   `TextTrainMemProbe` 都只在循环里用；探针门语义不变（默认构建整块不编译、`on` 恒 false、
   不求值 `getenv`）。
6. **共同坑**：`nn::create_optimizer` 返回 `unique_ptr`（未知名字 = nullptr），**不是**
   `Result`——`NN_EXIT(optimizer, 1, …)` 能吃它是因为宏族对"上下文可判定类型"有重载；
   改成 `Result` 上抛时必须显式判空（本轮在 src 里改为
   `if (!optimizer_holder) NN_FAIL(...)`），否则 `NN_TRY_MSG` 会在编译期报
   "no member named 'error'"。

**验收（2026-10-16，`build-rv` = clang 24 + Release + `NN_ENABLE_NATIVE=OFF` + 无 Vulkan）**：

1. **关键路径逐字节一致**：21 个场景在改动前后对照（f32 / `--f16` / `--model rapt` /
   `--t1_58` / `--grad-log` / `--checkpoint-every 1` / `--activation-offload` /
   `--accum-steps 2 --max-steps 4` / `step_cosine` / `--max-norm` / 单窗口多 epoch /
   `--resume` 成功路径 / 错误路径 ×6 / `--help`）——**stdout 与 stderr 逐行相同**（仅
   `time=X.Xs` / `总耗时: X.Xs` 两个时间字段归一）、**退出码相同**、写出的
   `.nnmodel` / `.nnckpt` **md5 逐位相同**。用例数据 = 44 token 的 `.nndataset` +
   `seq_len 256`（**单窗口** ⇒ 无 shuffle 随机性，可逐字节比对；多窗口场景结构一致、
   数值随 `std::random_device` 变动，属既有语义）。
2. **错误路径文案只有两处有意变更**：`Error: X` → `推理模型保存失败: X`（补语境）、
   数据集打开错误去掉 `[nn] … (expr) at file:line`（改由 `NN_TRY_MSG` 语境透出）。
   角色门禁（`--resume` 认 `.nnmodel` 为断点）与断点失败记账文案**逐字未变**。
3. **新门禁 `src/text_train_modules_test.cpp`（注册进 ctest）**：断点路径派生（含
   `.nnckpt.nnckpt` / 裸 `.nnmodel` 退化输入）/ 角色门禁三态（Model 拒绝、Checkpoint 放行、
   不存在放行）/ **断点往返（按文件规格重建 + 载参后参数逐位一致 + 内嵌词表字节数一致）** /
   记账与退出码（写失败计数 1 + 收尾码 1；成功不计账、不打印）/ 推理模型角色可被
   `peek_model_role` 认出。全 CPU、秒级、临时目录自建自清。
4. **门禁与审计**：`ctest --test-dir build-rv` **49/49**（无 Vulkan 树；GPU 用例 77 = skip）；
   `pwsh bench/doc_inventory.ps1` → `L2-VIOLATIONS: 0`、`HANDWRITTEN-RESIDUE: 0`，
   **新增第 [6] 节**（`text_train` 的 loop/recovery/checkpoint 与入口不得出现
   `std::exit` / `NN_EXIT` / `NN_UNWRAP`）PASS；`bench/doc_align_audit.ps1` exit 0；
   `python bench/gui_cli_audit.py` PASS 可行动问题 0（选项名单与帮助/解析分支未漂移）。
5. **跨编译器**：`build-gpu`（g++ 15 + libstdc++ + Release + Vulkan 启用）编译
   `text_train` 与 `text_train_modules_test` 零告警，新测试在该构建下同样通过。

**顺手回正的文档漂移**（都不是本轮改动引入，属 27 号象限二的"索引/口径卫生"）：
`host_bridge_uses` 实测 **41 → 39**（P4-2 拆掉 RAPT 逐头 host 往返的效果，文档一直写着 41）；
Layer 直调算子 **27 → 28**（`cast_into` 早已进 L2 文件，12 号未同步）；`18 §8.2` 第 2 条
仍写"统一写实测 49/47"（应为 51/49）；各计数行（`include/` 78 hpp/43424 行、`src/`
79 cpp/26574 行、`docs/` 46 篇、`ctest` 51/49）按 2026-10-16 实测刷新。

**未做 / 后续**：P1-2（`.bin` 优化器状态段）现在**直接可开工**（前置已解锁）；
`--resume-epoch/--resume-step` 的续训语义本轮未动（只是搬进 loop 的同一段代码）；
`mnist_train` / `text_infer` 等的参数解析层仍在各自 `src/*.cpp`（本轮范围外）；
setup 路径上的错误不再带 `(expr) at file:line`（如需保留可另立）。

# P1-1（第一刀）：`text_train` 的退出码语义——断点写失败不得成功退出（2026-10-15）

**缺陷**（docs/todo/18 §4.2 P1-1 的"退出码语义修正"）：`--save-interval` 的 `.nnckpt`
写失败时旧代码只打印一行 `[ckpt] 保存失败: …` 就继续跑，最后照常 `return 0`——上层
（脚本 / CI / `train_pkg`）看到的是"这轮训练完好"，而实际上**断点没落盘**，`--resume`
会拿到旧断点或空文件。

**改法**（`src/text_train.cpp`，24 行）：

1. 循环内断点保存失败 → `++ckpt_save_failures`，错误行带上第几次（不中断训练：
   磁盘抖动不该毁掉整轮）；
2. 收尾：**只要有过断点保存失败就以 1 退出**，并打印明确错误行（推理模型文件仍写出）；
3. 最终推理模型保存失败 → 仍然是失败退出（原来是 `NN_EXIT(..., 1)`，语义不变，
   改写为显式 `if (!save_result) return 1;` 以便同处集中记账）。

**验证（实测，850M / 2 步 / `--save-interval 1`）**：

| 场景 | 结果 |
|---|---|
| 正常可写目录 | 退出码 **0**，`m.nnckpt` + `m.nnmodel` 均写出 |
| 把 ckpt 路径占成**目录**（`/tmp/nn_exit_bad2/m.nnckpt/`） | 退出码 **1**，日志含 `[ckpt] 保存失败（第 1/2 次）: Cannot open file for writing: …` 与 `[错误] 本次训练有 2 次 checkpoint(.nnckpt) 保存失败（推理模型已写出）⇒ 退出码 1`；`m.nnmodel` 仍写出 |

**未做（下一次接手 P1-1 的起点）**：`train_loop` / `recovery` / `checkpoint` 的**模块抽取**
与 `std::exit` 收敛。现状盘点（2026-10-15 实测）：`src/text_train.cpp` **1808 行**、
`std::exit` 约 **40 处**——绝大多数在 `parse_args`（属允许层），需要随模块化搬走的是
训练循环内的四处错误分支与 `main` 里的几个前置校验。可切分的接缝（按行号区间，行号会漂移、
以注释标题为准）：`--resume` 的角色校验 + 加载（`// 规格是加载时的权威来源` 起）、
循环内 `// ── 定期保存 checkpoint`、收尾 `// ── 保存模型`；TDR 重启续训段
（`// TDR 重启续训：resume 时本 epoch 内跳过已完成的前 start_step 步`）与循环状态
（`step`/`epoch`/loss 累加器/`stop_training`）耦合最紧，抽它需要先定义状态结构。

# P1-2 设计裁定：`.nnckpt` 训练状态段 + role 语义（2026-10-16）

**类型**：演进记录（裁定）+ 否决方案。**状态**：已裁定，未实施（落地见 18 §4.2 P1-2 行 + 门禁 §8.2 第 7 条；
设计正文见 `docs/todo/23` §2.1.1）。

**背景**：原 P1-2 只写"`.bin` 增加可选优化器状态段（Adam m/v）"。设计讨论中把范围扩到"断点 = 训练的可续点"，
并顺带查证出三处**静默错误**（下面的"依据"），它们决定了最终口径。

**裁定（三条）**

1. **不做独立格式、不独立版本号**：`.nnckpt` 继续是同一容器里的 **role 字节**（v7 的"一份格式两种角色"），
   v8 只追加 role 门控的**训练状态段**——计数器 / 优化器状态（Adam m/v **+ `t`**、momentum、Muon、
   AdamWQuant 的 f16 状态与逐行尺度）/ 训练配方（精度 profile 四槽、lr 调度配方与**曲线形状**、
   batch_size/accum_steps/max_norm/stride）/ 数据集身份（`.nndataset` sha256 + tokens/docs/windows/
   `steps_per_epoch`/loss_scope）。`MODEL_VERSION` **7 → 8**，与 P2-1（尾部 `[sha256][NNEND]`）同批升版。
2. **role 决定 CLI 的发言权**：`Checkpoint` ⇒ 除"完全不改变训练"的参数（保存路径/保存间隔/log 间隔/`--gpu`/
   激活卸载/`--max-steps`/探针环境变量/数据集**路径**）外，**任何显式给出的训练参数都报错**（按**旗标名**
   判定、**不看值**；`--epochs` 与 `--resume-epoch/--resume-step` 同在禁列）；`Model` ⇒ **CLI 全权**，
   并**放开 `--resume` 对 `.nnmodel` 的支持**（SFT warm start：优化器冷启动、step/lr 从 0）。
3. **数据集身份不匹配 = 硬报错，且不提供逃生阀**；合法出口是第 2 条的 `Model` 行（换数据集 = 新任务）。
   三值模型的"打包 → latent 反量化载入"**单独立项**（18 §4.2 P1-7 / §3.4 M6），不塞进 P1-2。

**依据（本轮实测锚，均为当前代码事实）**

- **精度 profile 完全不进文件**：`spec_to_kv` 只写形状与 `weight_quant`（`model_serialization.hpp:287`），
  而恢复重建用的是 **CLI 的 `cfg.precision`**（`cli/text_train_recovery.hpp:104`）。于是 `--f16` 训出的断点、
  resume 时忘带 `--f16`（或反之）→ 架构校验照样过（`spec_matches` 只比形状类字段，`model_spec.hpp:192`）、
  参数按"异精度经引擎 cast"静默转换（`compute_layer_base.hpp:424` 的 `import_into_`）⇒ **整轮训练静默换回 f32**。
  三值侧因为 `effective_weight_quant`（`compute_layer_base.hpp:187`）与层能力校验会当场报错，f16 侧毫无保护。
  ⇒ "配方必须随断点走"不是洁癖，是修真缺陷。
- **数据集/`--epochs`/`--batch-size` 三者都能改变 lr 曲线**：`lr_sched_cfg.total_epochs = cfg.epochs`
  （`cli/text_train_loop.hpp:365`）、`step_lr_cfg.total_steps = steps_per_epoch × cfg.epochs`（`:414`）、
  `global_step = epoch*steps_per_epoch + step`（`:538`）。数据集一改 ⇒ 窗口数变 ⇒ `steps_per_epoch` 变 ⇒
  曲线形状 + 步号语义 + epoch 边界**同时**变 ⇒ "同一断点 + 不同数据集"不存在正确的续训语义
  ⇒ 硬错是语义必需，逃生阀提供的是假续训。同理：`--epochs` 不是"跑多久"而是配方，故入禁列；
  "本次跑多久"由 C 类的 `--max-steps` 表达。
- **只存 m/v 不够**：Adam 偏置修正按 `t` 走（`compute_optimizer.hpp:299` 的 `t_` + `bias_correction_(t_next)`），
  丢了 `t` 重启后前几百步更新量偏小——P1-2 原验收项"loss 轨迹连续"就过不了。
- **三值续训确实没有入口**：`.nnmodel` 载荷是打包形态，`BitLinear::import_param` 走 `TernaryPacked2` 分支后进
  `packed_only_` 部署态，`backward()` 直接报错（`compute_layer_bitlinear.hpp:394`「部署态只读（没有 latent 可更新）」）
  ⇒ 三值模型的 SFT 需要反量化载入路径（新立 P1-7）。
- **版本号现实**：v7 已被"文件角色"改动占用（`model_serialization.hpp:62`）；P1-2 原行写的 `5 → 6` 是过期口径
  （v1.7.0/v1.8.x 时期的"抢号提示"）。本轮把 18 号与 `docs/todo/19` 的提示刷新为 **7 → 8**；
  release note 作为历史文档保留原样（它写的"谁后落地谁顺延"正是已发生的事实）。

**被否决的方案（含否决理由）**

| 方案 | 否决理由 |
|---|---|
| `.nnckpt` 独立格式 + 独立版本号 | v7 的"一份格式两种角色"是 2026-10-11 既定裁决（23 §0/§2.1），独立格式要把头部/spec/参数条目/词表读写复制一遍、让推理侧嗅探两个 magic；而 `.nnckpt` 还要能被 `text_infer` 直接读做推理自检 ⇒ 同容器 + role 字节是唯一自然表达 |
| "无差别忽略 CLI"（原提议：ckpt 恢复时忽略一切 CLI 输入） | 会把 `.nnmodel` warm start 一起堵死（SFT 没有配方可读）；且与"不静默降级"口径冲突 |
| 折中三档（文件为默认 + CLI 显式覆盖 + 冲突仅警告） | 用户裁定收紧为**显式给出即报错**：A/B 类参数在断点续训里没有合法用途，警告留给"值相同"的徒劳场景之外没有信息量 |
| 数据集不匹配仅警告 / 提供 `--allow-dataset-change` 逃生阀 | 见依据第 2 条：网格改变后"续训"无正确语义，逃生阀 = 假续训 |
| 保留 CLI 覆盖 batch_size（"OOM 后要改"） | 论据不成立：本训练路径稳态显存不膨胀、不存在"训到一半显存变大"；恢复主机显存不小时的出口是 CPU 引擎或从 `.nnmodel` 重开（写进 23 §2.1.1 作为已知取舍） |

**连带影响（后续落地时不得漏）**

- **反转 P1-1 落地的角色门禁**：`--resume` 由"只认 Checkpoint"改为"Model 也接受（CLI 全权）"。
  该门禁文案是 P1-1 验收里"**逐字未变**"的用户可见契约（`docs/todo/23` §2.1），
  落地时要同步 `text_train_modules_test` 的"Model 必须被拒"用例与 18 §5 的对应行。
- 禁列校验需要**旗标名**信息（解析层的默认值无法与用户输入区分）：`parse_text_train_args` 记录"本进程显式
  给出的 A/B 类旗标"名单，恢复路径发现非空即报错（+ 单测覆盖禁列）。
- GUI 的 `--resume-epoch/--resume-step` 字段随 role 禁用；`bench/gui_cli_audit.py` 纳入门禁。
- 只读 `peek_train_state(path)`：加载权重前即可显示断点的 epoch/step/lr 与数据集身份（也是数据集校验失败时的
  诊断面）。

# P1-2 / P2-1（第一刀）：v8 训练状态段骨架 + 整文件校验和（2026-10-16）

**类型**：演进记录（格式升版）。**范围**：只做**格式与读写**，不动训练路径（一次改太多的接口面
无法逐项验收）。裁定归档见上一节。

**落地**

- `MODEL_VERSION 7 → 8`；spec 之后新增 **role 门控状态段**
  `[state_len u64][TrainState kvrec][优化器状态张量条目 × N]`（Model 角色写 0 长度；
  老文件缺段 = 冷启动语义）。文件尾新增 `[sha256 32B]["NNEND" 5B]`（P2-1）。
- 校验和**边写边算**：`detail::HashingStreamBuf` 把哈希塞进 `streambuf`，写路径的流类型从
  `std::ofstream&` 放宽到 `std::ostream&` ⇒ 无需"写完再读回来算一遍"，也不需要把整文件拼在内存里。
  读侧 `verify_model_integrity` 分块（1 MiB）流式校验；`load_model` / `load_train_state` 先行校验，
  截断/损坏在解析参数之前就明确报错（旧缺陷 C5：静默载入错误权重）。
- 新头 `include/neuralnet.cpp/train_state.hpp`：`TrainState`（计数器 / 优化器 `t` 与 blob 数 /
  训练配方含**精度四槽与 lr 曲线形状** / 数据集身份）+ kvrec 编解码（浮点字段 `bit_cast` 到 u64
  落盘、缺键 = 默认值、保留精度值如 BF16 读回即报错）+ `dataset_mismatch_reason`（硬报错口径的
  差异说明：逐项列出变了什么）。
- 新 API：`save_model(..., const CheckpointState&)`（**role 门禁**：Model 角色携带状态段即报错；
  张量无状态也报错）、`peek_train_state`、`load_train_state`（张量物化为 f32 并记录**文件里的**
  每张量精度）、`verify_model_integrity`。
- L0 补件：`core_sha256.hpp` 增 `sha256_finalize`（增量收尾）与 `sha256_hex(摘要)` 重载；
  `core_file.hpp` 增 `sha256_file_prefix` / `sha256_file_hex`（流式，大文件不全读）。
- 顺带收口：`read_param_blob` 的"未信任长度防御"抽成 `ensure_blob_fits_`，新增 `skip_param_blob`
  —— 读状态段与读权重各需跳过对方时按条目头 seek 过去（载荷读进内存再丢掉，在大模型上一条
  embedding 就是几百 MB）。

**验证（实测）**

- 新门禁 `train_state_test`（8 组）：kvrec 往返全字段一致；Checkpoint 往返（状态逐项一致 +
  f32/f16 张量逐位 + 每张量精度记录）；role 门禁两条负例；Model 角色状态段**字节级**为 0；
  校验和（完好通过 / 翻一字节 / 截断各自明确报错，且 `load_model` 同步失败）；`load_model`
  跳过状态段后**参数逐位一致**；数据集身份差异说明；**把 v8 文件剥成 v7 后仍能 `load_model`**。
- `ctest --test-dir build-rv` **50/50**（GPU 用例 77 = skip）；`build-gpu` 注册 **52** 条（计数
  已同步 AGENTS / 18 / 12）。
- 端到端（本机 CPU 树）：`text_train` 写出 v8 文件（`state_len=0`、尾部 `NNEND`、role/version 字节
  经字节级核对）→ `text_infer` 读 `.nnmodel` / `.nnckpt` 两角色均正常 → `--resume` 该断点续训正常；
  **P1-1 期间产出的 v7 真实文件**（`/tmp/p11/ref_final/*`）仍可加载且生成结果与改动前一致
  （零破坏的现场证据，非仅单测）。
- 审计：`doc_inventory`（L2-VIOLATIONS 0 / HANDWRITTEN-RESIDUE 0 / TERMINATING-CALLS 0）与
  `doc_align_audit`（[A]0 [D]0 [E]0 [F]0）全绿。

**未做（后续刀）**：`Optimizer::save_state/load_state`（Adam m/v + `t`、SGD momentum、Muon、
AdamWQuant 的 f16 状态与逐行尺度）；训练路径接线（写状态段、按状态段还原精度/计数器/曲线形状）；
A/B 类**禁列校验**（显式给出即报错）；数据集身份门禁（硬错、无逃生阀）；`.nnmodel` 恢复放开
（SFT warm start）；GUI 的 `--resume-epoch/--resume-step` 随 role 禁用。

# P1-2（第二刀）：优化器断点状态 `save_state` / `load_state`（2026-10-16）

**类型**：演进记录（功能补齐）。上一刀见「v8 训练状态段骨架」。

**改法**

- `Optimizer` 基类新增断点状态 API：`save_state(std::vector<Tensor>&)`（导出**只读别名**，
  零拷贝）、`load_state(std::span<const Tensor>)`、`state_step()` / `set_state_step()`
  （偏置修正计数 `t`）。默认实现 = 无状态（SGD）；有状态者各自 override。
  **张量顺序随版本冻结**，契约写在基类 API 注释里（要改 = 升 `TRAIN_STATE_VERSION`）：
  `sgd_momentum` / `muon` = 每参数 1 个动量缓冲；`adam` / `adamw` = `m,v` 交替；
  `adamwquant` = `m16,v16,sm,sv`（f16 载荷 + f32 逐行尺度，缩放态本身就是状态，故往返即可）。
- `restore_state_tensor_`（受保护实现件）：形状不符 → 明确报错；精度不同经引擎 cast
  （与参数形态协议同口径：同精度零转换）。数量不符同样报错（**不静默接受部分状态**）。

**验收（实测）**

- 新门禁 `optimizer_state_test`（覆盖 = `nn::optimizer_table()` **全表**）：
  ① 状态存在性 + 张量数符合表口径（无状态优化器不得导出非空，有状态者不得导出空 —— 新优化器
  忘了实现 `save_state` 会在这里红）；② 重启后动量非零、Adam 家族 `state_step` 正确；
  ③ **中途存状态 → 优化器析构（模拟进程死）→ 新建 + 载入 → 续跑 = 不中断直跑（参数逐位一致）**
  （10 步、第 5 步打断、6 个优化器全过 —— 这是 18 §4.2"loss 轨迹连续"的可机械化形式）；
  ④ 负例：有状态优化器**冷启动**续跑必须与直跑**不同**（否则 ③ 不具判别力）；无状态优化器（sgd）
  冷启动续跑与直跑**相同**（如实断言：没有状态可丢，不是缺陷）；⑤ 数量偏少 / 形状不符 /
  sgd 收到状态 → 三种都给明确报错。
- `ctest --test-dir build-rv` **51/51**（GPU 用例 77 = skip）；`build-gpu` 注册 **53** 条
  （计数已同步 AGENTS / 18 §2·§8.2 / 12）。审计 `doc_inventory`（L2 0 / 手写残留 0 /
  终止调用 0）与 `doc_align_audit`（[A][D][E][F] 全 0）全绿。

**未做（后续刀）**：训练路径接线（`--save-interval` 写状态段、`--resume` 走
`load_train_state` + `load_state`、按状态段还原精度 profile / 计数器 / lr 曲线形状）；
A/B 类禁列校验；数据集身份门禁；`.nnmodel` 恢复放开；GUI 同步。

# P1-2（第三刀）：训练路径接线 + role 语义落地（2026-10-16）

**类型**：演进记录（行为变更）。前两刀见本文件上方两条。

**改法**

- **写侧**：`TrainConfig` 增 `explicit_train_flags`（解析层按**旗标名**登记本进程真正
  出现过的 A/B 类旗标 —— 默认值与用户输入无法区分，故必须在解析期记）；禁列表
  `kForbiddenResumeFlags` + `is_forbidden_resume_flag` / `forbidden_resume_flags_text`
  （报错文案与帮助共用一口径）。`TextTrainCheckpointLog::save` 收 `TrainState` + 优化器
  状态张量，新增 `note_failure`（状态导出失败 = 同一记账口径，P1-1 语义不变）。
  `text_train_loop.hpp` 增 `make_train_state`（counters = **下一步的位置**、epoch 边界
  归一化、`global_step` 与 step 级调度同定义、recipe = 本进程实际配方、dataset = 身份 +
  网格）与 `save_ckpt_with_state`；`src/text_train.cpp` 计算数据集身份
  （`sha256_file_hex` + tokens/docs/windows/steps_per_epoch/loss_scope）注入循环。
- **读侧**：`adopt_resume_state`（新）在**建模型/建优化器之前**把断点状态读出来并改写
  cfg —— 形状/架构按文件 spec（两种角色都适用：文件就是那个模型，不可能把 16 维断点
  warm start 成 128 维；顺带让配置横幅与真实模型一致，消掉一次被丢弃的建模），
  Checkpoint 再按状态段覆盖精度 profile / 曲线形状 / 网格 / 起点（A/B 类全量）；
  显式给出 A/B 旗标即报错。`load_optimizer_state`（新）在建优化器之后恢复张量与
  Adam 的 `t`。`verify_dataset_identity`（新）＝硬错、无逃生阀，报错列出逐项差异
  （sha256 / tokens / docs / windows / steps_per_epoch / loss_scope）。
- **role 语义落地（2026-10-16 裁定的执行）**：`check_resume_role` → `check_resume_target`：
  `.nnckpt` = 断点续训（状态段唯一权威）；`.nnmodel` = **warm start 放行**（SFT 入口，
  CLI 全权、优化器冷启动）；唯一仍拒绝 = **三值 `.nnmodel`**（打包形态无 latent ⇒
  `backward()` 报错），文案指向 P1-7 与 `.nnckpt`。**这是对 P1-1 落地门禁文案的有意反转**
  （P1-1 验收里那行"逐字未变"的用户可见契约在此修订，裁定见 d069583）。

**验收（实测）**

- **端到端四例（本机 CPU 树）**：① 首轮 1 步 → 断点；② 无 A/B 旗标续训 ⇒ 采纳状态段
  （横幅显示断点的 16/2/1/32 与 batch=2/epochs=3，证明形状与配方都来自断点）；
  ③ 显式 `--lr` 续训 ⇒ **退出码 1** + 禁列文案（列出被禁旗标与仍允许的 C 类）；
  ④ 换数据集续训 ⇒ **退出码 1** + 身份差异文案（sha256 / token 数 / 两侧路径）；
  ⑤ 三值 `.nnmodel` 续训 ⇒ **退出码 1** + 指向 P1-7。
- 单测：`text_train_modules_test` 新增 [6] 接线用例（`save_ckpt_with_state` → `peek_train_state`
  → 新建优化器 → `load_optimizer_state`；断言 epoch 边界归一化 / `t` 恢复 / blob 数 /
  配方与身份落盘 / 身份门禁接线一致）与 role 语义改写（dense `.nnmodel` 走通 warm start、
  三值被拒、禁列谓词与解析层登记）；`train_state_test` 新增 [9] **save → load → save
  逐字节一致**（f32 状态；P1-2 验收口径）。
- `ctest --test-dir build-rv` **51/51**；`build-gpu`（g++ 15 + Vulkan）四个受影响目标
  零告警编译通过。

**已知取舍（如实记录）**：f16 优化器状态张量经 `load_train_state` 会**物化为 f32**，
若把它原样再 `save` 会被提升成 f32 编码（**值逐位不变**）。生产路径不存在这种再写
（save 一律从活状态导出），故未在接口上引入精度回写提示；`save→load→save` 的逐字节验收
按 f32 状态（默认 profile 的常态）成立。

**未做（第四刀 · 收口）**：GUI 的 `--resume-epoch/--resume-step` 随 role 禁用与
`gui_cli_audit.py` 门禁同步（当前 GUI 训练页在 ckpt 续训时会撞上禁列报错——报错文案清晰，
但 GUI 应主动不发这些旗标）；`mnist_train` 的 `--resume` 尚未接同一套语义（它不写状态段，
本轮范围外）。

# P1-2（第四刀）：GUI 侧收口（断点续训禁列）（2026-10-16）

**类型**：演进记录（客户端对齐）。第三刀把 CLI 的禁列规则落地后，GUI 训练页仍会照发
形状/配方旗标 ⇒ ckpt 续训必然撞报错（文案清晰但不该由用户承担）。本刀让 GUI 按角色主动收敛。

**改法（`gui.py`）**

- `model_file_role(path)`：读模型文件头第 10 字节的 role（v7+；布局 `magic|version|precision|role`），
  返回 `model` / `checkpoint` / `None`。GUI 因此**按文件真实角色**决策，不靠扩展名猜。
- `CKPT_RESUME_FORBIDDEN_KEYS`（kwargs 键集合，唯一真源指向 CLI 的 `kForbiddenResumeFlags`）；
  `GptTrainTab.collect_args` 在断点续训模式下把这些键**全部弹出**（`.nnmodel` warm start 不受影响）。
- 表格里新增一行模式提示（`group="ckpt_resume"`，仅该模式下可见）：形状/配方/计数器来自断点，
  本页对应输入不发送。

**门禁（`bench/gui_cli_audit.py` 新增 [6] 节，三判据）**

① gui 的禁列集合 ↔ CLI 头文件 `kForbiddenResumeFlags` **逐项一致**（键→旗标用字段的 arg/name
映射，`--model`/`--norm`/`--t1_58` 三个例外显式登记）——两边任一漂移都会红；
② 造一个 v8 头指向 Checkpoint 角色 ⇒ `collect_args()` 禁列键 0 泄漏；
③ 指向 Model 角色 ⇒ 禁列键**照发**（防止把 SFT 的配方面一起砍掉），且 C 类（`save`/`log_interval`）
在任何模式下都保留。

**验证**：`python bench/gui_cli_audit.py` 退出码 0（六节全绿，`可行动问题=0`）；
`gui.py` 经 AST 语法检查通过（无 customtkinter 环境的导入由既有桩模块承担）。

**未做**：`mnist_train` 侧的 `--resume` 仍走旧语义（它不写状态段，故禁列不适用）；
`train_pkg.py` 的打包配置里仍会记录形状/配方（与"断点权威"并存时以 CLI 侧报错兜底）。

# P1-7：三值模型的续训入口（打包 → latent 反量化）（2026-10-16）

**类型**：演进记录（新能力 + 收掉最后一条"必须拒绝"）。背景（18 §3.4 M6）：三值
`.nnmodel`（role=Model）的载荷是打包形态，载入后 `BitLinear` 落在 `packed_only_` 部署只读态
⇒ `backward()` 明确报错 ⇒ 三值模型没有 SFT / QAT 续训入口（dense 不受影响）。前置 P1-2 的
role 语义已落地，本轮开工。

**改法**

- **载入意图与文件角色正交**：`compute_layer_base.hpp` 新增
  `WeightLoadIntent{Inference, Trainable}` —— 角色（`ModelRole`）回答"这份文件是什么"，意图回答
  "载入方要拿它做什么"。`load_model(...)` 追加**带默认值**的 `intent = Inference` ⇒ 既有调用方
  （`text_infer` / `mnist_infer` / 测试）零改动、逐位不变。
- **反量化在序列化层（宿主侧）**：`param_blob_to_entry(engine, blob, intent)` 遇到
  `TernaryPacked2/3` 且 `intent == Trainable` 时逐行解码 + 反量化成稠密 f32 latent，返回
  `TernaryLatent + aux(τ)`；`Inference` 分支逐字未动（部署只读态）。
- **反量化取值 = 量化器的不动点**（`ternary_pack.hpp` 新增 `latent_row_scale` / `dequant_row`）：
  `L = trit·(τ·K/n)`（n = 该行非零 trit 数）。**为什么不是字面的 `trit·τ`**：本仓的 τ 定义是**逐行
  absmean**，`trit·τ` 反代回去的逐行 absmean 是 `τ·n/K` ⇒ 第一次参数写回后重算 τ 就跳 `n/K` 倍
  ⇒ 有效权重 `wq·τ` 整体缩水，等于"warm start 一开步就换了个模型"。`trit·τ·K/n` 让两条不变量
  同时成立：① 行 L1 质量 = `K·τ`（⇒ `absmean(L) == τ`，与训练时 latent 同定义）；② 非零幅值
  `τ·K/n ≥ τ > 0.5τ`（⇒ 用同一个 τ 再量化**复现同一组 trit**）。
- **层侧只做"播种"**（`BitLinear::import_param` 的 `TernaryLatent` 分支）：`aux(τ)` 有效时用**文件 τ**
  重建打包派生物（`engine.trit_pack(latent, τ_file)`，trit 必与文件一致）、登记写版本、
  `packed_only_ = false` ⇒ 首次 forward 的 wq/τ **逐位来自文件**，不经"用自己的 absmean 重算 τ"
  那一步；`.nnckpt`（无 `aux`）路径逐字不变（缓存作废、首次 forward 自行量化）。
- **CLI**：`cli/text_train_recovery.hpp` **删除 `check_resume_target`**（P1-2 第三刀的"唯一拒绝"）
  并把载入改为 `Trainable`，`adopt_resume_state` / `resume_text_train` 各增一行三值提示；
  `mnist_train` 的 `--resume` 同样传 `Trainable`（其 role 门禁仍只收 Checkpoint，P1-2 遗留项）；
  `text_train --help` 的 `--resume` 文案重写（原文案仍写"`.nnmodel` 会被拒绝"，属 P1-2 之后的过时文案）。

**验收（实测）**

- 单测（本机 clang 24 + Release + `NN_ENABLE_NATIVE=OFF`、无 Vulkan 的 `build-rv`），`ctest` **51/51**
  （GPU 用例 77 = skip）：
  ① `model_roles_test` 新增 [6]（三值 MLP）：Trainable 载入后**不是** `packed_only_`；打包字（**位级**
  比较 —— 容器是"f32 张量承载原始字"，字里可能是 NaN 位型，浮点比较会误报）与 τ 与源模型**逐位一致**；
  首次 forward 与源模型（= 文件里的模型）**逐位一致**且 `quantize_calls() == 1`（播种缓存命中，不重算）；
  反量化公式逐位 + 逐行 `absmean(latent) == τ`；8 步 Adam（forward/CE/backward/step）后 latent
  **移动 512/512**、`quantize_calls() > 1`（写版本驱动重算）；dense 文件两种意图载入参数逐位一致。
  ② `ternary_gemm_test` 新增 [8]（纯宿主）：K ∈ {17,32,37}、含全零行 / 全非零行 / 近零行 ⇒
  公式逐位、同 τ 再打包**容器字逐位相同**、逐行 absmean == τ（实测相对偏差 worst **5.012e-08**）、
  n = 0 行整行 0（不产生 inf/NaN）。
  ③ `text_train_modules_test`：三值 GPT 走通 `resume_text_train`（**不再被拒**）、同精度口径下首次
  forward 与推理文件逐位一致、8 步 Adam 移动 **3728/4496** 个参数项；另覆盖"不给精度旗标也能载入
  （BitLinear 由文件规格确定）"。
- **端到端（本机 CPU 树，`/tmp/p11/tiny.nndataset`）**：① `--t1_58 … --max-steps 2 --save q1.nnmodel`
  （loss 9.0023 → 8.9979）；② `--resume q1.nnmodel --max-steps 2 --save q2.nnmodel --save-interval 1`
  （**不给** `--t1_58`）⇒ 横幅 + "三值（T1_58）推理模型：按文件 τ 反量化打包权重为 f32 latent ⇒ 可续训"
  + warm start 提示，loss **8.9346 → 8.9115**（无尺度跳变），写出 `q2.nnmodel`（1.49 MB）与
  `q2.nnckpt`（6.82 MB = latent + 状态段）；③ `--resume q2.nnckpt --max-steps 1` ⇒ 采纳状态段
  （epoch=3 / 44 个优化器张量 / 步计数 2），loss 8.8909 ⇒ **「三值预训练 → `.nnmodel` warm start（QAT）
  → `.nnckpt` 断点续训」一条链闭合**（此前第 ② 步必然报错并指向 P1-7）。
- 门禁：`bench/doc_inventory.ps1` 第 [4] 节 `L2-VIOLATIONS: 0`（`host_bridge_uses` 39 不变，反量化在
  L3 宿主侧，未在 L2 引入 Matrix/IO 动词）、第 [6] 节 PASS；`bench/gui_cli_audit.py` 退出码 0
  （六节；禁列 32 项一致，`.nnmodel` warm start"照发参数"符合角色语义）。

**已知取舍（如实记录）**

- **精度 profile 仍由命令行给出**（Model 角色 = CLI 全权；文件只记 `weight_quant`，这是 P1-2 已裁定的
  口径）。所以"首次 forward 逐位一致"是**同精度口径**下成立：`--resume` 一个 `--t1_58` 训出的模型而
  **不带**任何精度旗标 ⇒ BitLinear 仍由文件规格确定，但 compute/stable 槽走 CLI 默认（f32）⇒ 前向数值
  不同。测试两条口径都覆盖（带 `--t1_58` 逐位一致；不带也能载入并前向/反向）。
- 反量化在宿主侧（L3）：一次性 O(rows·K) 逐行解码 + 上传；**推理路径（默认意图）不付这份成本**。
- **GPU 路径本轮未在本机实测**（受限沙箱内 `/dev/nvidia*` 不可见 ⇒ GPU 用例 77 = skip）。改动本身
  引擎无关：反量化全在宿主，播种只经 `trit_pack`（两引擎都已实现；"引擎 `trit_pack` = 宿主 `pack_row`
  逐位一致"由 `ternary_gemm_test` [2] 在 GPU 树上把关）。

**未做**：`mnist_train` 的 role 语义仍只收 Checkpoint 角色（P1-2 遗留项，另一案；它的 `.nnmodel`
目前仍会被自己的门禁拒绝，与 text_train 不一致）；没有为"warm start 未给精度旗标"加告警（CLI 全权
是既定口径，见上）。

# docs 台账回正：进度盘点发现的三处偏差（2026-10-16）

**类型**：文档勘误（台账与实测不符）。起因：按 18 §4 + 00 号 RV 台账逐条盘点进度时发现三处偏差，
两处是**标记/计数滞后**，一处是**真缺口**（"接手时掉了"）。

**① P2-1 行标记滞后（已回正）**：`.bin` 整文件校验和 + 尾部完整性标记实际**已随 P1-2 同批交付**
（v8 尾部 `[sha256 32B]["NNEND" 5B]`、`verify_model_integrity` 在 `load_model` / `load_train_state`
先行校验、门禁 `train_state_test` [5] 三态 + §8.2 门禁 7 的 ④），但 §4.3 的 P2-1 行一直没打 ✅
（27 号象限（三）的 P1-2 行早已写"+ P2-1 合并"）。现补 ✅ + "已实施"证据指针。

**② §4.0 的 P3 项数 8 → 7（已回正）**：§4.4 实际列 7 项（S6 自动窗口 / M7 存储多态 / IR-D 第二
emitter / ZiPT 恢复 / LRLA / CUDA / 分布式）——IR-C 已落地故不在本档，§4.0 的 8 是移出前的旧计数。

**③ GUI 多轮对话闭环：真缺口，重新登记为独立待办**：P0-5 的"多轮对话闭环（连续 N 轮不丢上下文）"
当年按 27 号计划划给 P2-6 批次（证据：本文件「P0-5 采样器 + CLI 半边」条的"未做"段），但 P2-6 收口
（2026-10-15，「绘制节流」+「字段表」两侧）**未包含**它：`gui.py` 实测无任何上下文/历史处理
（`rg 'context|多轮|history'` 零命中）⇒ 该项实际未交付，而 P2-6 已被标 ✅ ⇒ 缺口被标记掩盖。
**处置**：18 §3.3 X1 行 / §4.1 P0-5 行 / §4.0 P0 行改为"未交付 + 重新登记"，并在 27 号 §3 象限（二）
（不重要但简单·顺手做）新增一行（含缺口描述与成本，前置 = 无）。**未实施**（仅登记）。

**盘点口径（顺带的结论，备查）**：剔除 P3 的 7 项择机项后，当前实质待办集 = **44 项**
（P0 5 + P1 7 + P2 13 + P4 4 + RV 15），已收口 **31/44 ≈ 70%**（其中 29 项真交付、2 项按裁定关闭 =
P0-4 的 `install()/find_package` 与 P2-9）；剩 13 项 = 部分完成 5（P0-5 / P2-4 / P2-10 / P4-2 / P4-4）
+ 未做 6（P1-5 / P2-2 / P2-7 / P2-5 条件触发 / P4-1 / P4-3）+ RV 待办 2（RV-05 / RV-12）。
**下一项主线 = P1-5**（P2-7 的前置），见 27 号 §4 象限（三）。

# P0-5 尾项：多轮对话闭环（库 + CLI + GUI，2026-10-16）

**类型**：功能交付（P0-5 的剩余半边）。P0-5 的采样器 + CLI 采样开关已于 2026-10-15 交付
（见本文件「P0-5 推理采样」条），剩余半边 = **"多轮对话闭环（连续 N 轮不丢上下文）"**，
由本文件「docs 台账回正」条 ③ 确认为**真缺口**而非标记问题：CLI 的 `--interactive`
每轮只把当轮输入拼进 prompt（第 2 轮起就看不到第 1 轮），`gui.py` 更是没有任何
上下文/历史处理（每轮从零起算）。

**改了什么（四层，唯一真源在最下层）**

1. **库层**（`domain_tokenizer_base.hpp`）：新增 `Tokenizer::ChatTurn`（`user` / `text` /
   `tokens`）与 `Tokenizer::chat_prompt(turns, system_prompt, max_tokens, *dropped_turns)` ——
   多轮 prompt 构建的**唯一实现**。对话词表套训练时的标记模板
   （`BOS<|system|>…<|end_of_system|><|user|>…<|end_of_user|><|assistant|>…`；末轮是 user 时补
   `<|assistant|>` 待生成，末轮是 assistant 时以 `<|end_of_assistant|>` 收尾）；无对话标记的词表
   退化为轮次文本按 `\n` 相接 + BOS。内容两种给法：`text` 走 `encode`，`tokens` **原样灌入**
   （上一轮生成的 token 回灌不必过 decode → encode，避免 BPE 往返漂移）。
   超 `max_tokens`（调用方传模型 `spec.seq_len`）时**从最旧的一问一答（轮次对）整轮丢弃**：
   system 块与末轮恒保留、丢弃后序列仍以 user 起头（标记结构不破）、丢弃轮数经出参回报；
   对话词表下若"system 块 + 末轮"本身超限则原样返回（宁愿超限也不切断标记结构，由 `generate`
   的滑动窗口兜底）；纯文本词表无结构约束，超限直接截尾。
2. **CLI**（`src/text_infer.cpp`）：① `--interactive` 修掉"每轮从零"——进程内累积 `history`，
   每轮把**整段历史**交给 `chat_prompt`，助手轮用生成的原 token 回灌，超限时打印
   `[上下文] 已达 seq_len=… 上限：丢弃最旧 N 轮`；② 新增 `--turn <role>:<text>`（可重复）
   + `--reply-out <path>` + `--system <text>`（**无状态多轮协议**：一次一问、历史由调用方自带、
   回复落盘），GUI/脚本据此接续上下文；③ 形状校验放在**参数解析期**（role 写错 / 首条不是 user /
   不交替 / 末条不是 user / `--turn` 与 `--prompt`·`--interactive` 并用 / `--reply-out` 无 `--turn`
   → 报错退出，不静默降级）；④ 单轮路径（`--prompt`）**逐字未动**（采样默认 ≡ 旧路径的等价性
   由 `sampling_test` 继续锁）。
3. **GUI**（`cli_controllers.py` + `gui.py`）：`ChatSession` 持有交替的 `(role, text)` 历史
   （不变量：user 起头、assistant 收尾；本轮输入**先拼进参数、回复到达后才落账** ⇒ 生成失败
   历史不变，不会留下 CLI 会拒的"连续两个 user 轮"）；`GptInferController._format_args` 把
   `turns` 翻成逐个 `--turn`、`reply_out` 翻成 `--reply-out`、`system` 翻成 `--system`。
   「GPT 推理」页新增「对话记录」面板 + **「💬 发送（多轮）」** + **「🧹 清空对话」**
   （字段表新增「系统提示（多轮对话）」→ `--system`）；`TabBase` 把回调装配 + worker 线程抽成
   `_run_args(args, on_done)`（单次运行与多轮共用），回复经临时文件 `--reply-out` 取回后
   在主线程落账并刷新面板。
4. **门禁**：新测试目标 `chat_prompt_test`（ctest **53 → 54**；无 Vulkan 树 **51 → 52**）+
   `bench/gui_cli_audit.py` 新增第 **[7]** 节。

**为什么是"每轮把整段历史交回 CLI"（而不是把 stdin 交互模式接进 GUI）**：stdin 交互的
历史只活在子进程里，GUI 拿不到；而 GUI 需要"每轮可改采样参数"（持久进程会把参数钉死在启动时）、
需要在进程崩掉后保留历史、还需要从输出流里认出"这一轮回复到哪结束"（stdout 非 tty 时是块缓冲，
`>>> ` 提示无换行 ⇒ 流式解析脆弱）。无状态协议把这些都消掉：代价只有每轮重编码整段历史
（本项目推理本来就不带跨轮 KV cache，一次 prefill 的量级）。CLI 侧仍保留 `--interactive`
给终端用户（历史留在进程内 + 生成 token 原样回灌）。

**实测（本机，CPU 树 `build/cpu` = Release + Tests）**

1. **上下文不丢（端到端，真模型 `gpt_model.nnmodel`）**：空系统提示下两轮
   `--turn user:"hi there" --turn assistant:"…" --turn user:"bye now"`，
   第 2 轮 prompt = `256, 8194, 104, 105, 401, 8195, 8196, 1259, 486, 7536, 4655, 2272, 3603, 8197, 8194, 1104, 960, 8195, 8196`
   —— **第 1 轮的 prompt（前 7 个 token）逐 token 原样是第 2 轮的前缀**，其后是第 1 轮生成的
   3 个 token + `<|end_of_assistant|>` + `<|user|>` + 新输入 + `<|end_of_user|>` + `<|assistant|>`。
   该链路走的是 **GUI 的真实通道**（`ChatSession.ask()` → `GptInferController._build_command` →
   `text_infer`），不是手工拼命令。
2. **`--interactive` 多轮**（`printf 'hi\nyo\nquit\n' | text_infer --interactive --show-tokens`）：
   第 2 轮的 prompt = 第 1 轮全部 + 第 1 轮生成 token（`4498, 3614, 855`，**原 token 回灌**）+
   `<|end_of_assistant|>` + `<|user|>` + `yo` + `<|end_of_user|>` + `<|assistant|>`。
3. **截断可见**：小模型 `seq_len=64` + 默认系统提示下三轮 `--turn` 会打印
   `多轮对话: 3 轮 (对话标记模板) → 进入上下文 1 轮（seq_len=64 上限，丢弃最旧 2 轮）`，
   随后 prompt 只剩 system 块 + 末轮（标记结构完整）。
4. **门禁**：`chat_prompt_test` 全绿（库级 5 节 + 真实 `text_infer` 子进程：CLI 的 prompt tokens
   与库侧 `chat_prompt` **逐 token 一致**、`--reply-out` 内容 == 生成 token 的解码、
   7 条负例各自非零退出且 stderr 说明原因）；`python bench/gui_cli_audit.py` 退出码 0（七节）；
   `bench/doc_inventory.ps1`（L2-VIOLATIONS 0 / HANDWRITTEN-RESIDUE 0 / TERMINATING-CALLS 0）与
   `bench/doc_align_audit.ps1`（[A][D][E][F] 全 0）PASS。
5. **三组「编译器 × 标准库」验证**（2026-10-16，全部含新用例、全部零告警 `-Werror`）：

   | 环境 | 编译器 / 标准库 | 构建 | ctest |
   |---|---|---|---|
   | 本机（Linux，`build/cpu`） | clang 24 + libstdc++（Release + CPU-only） | 零告警 | **52/52** |
   | `102` WSL（`~/nn-verify/build`） | clang 24 + **libc++** | 零告警 | **52/52** |
   | `102` WSL（同目录 `build-gcc`） | g++ 15.2 + libstdc++ | 零告警 | **52/52** |

   同一轮另跑过：本机 GPU 树（`build-gpu`）新用例构建 + 运行通过、`ctest -N` 注册 **54** 条
   ——**全量 GPU ctest 未跑**（本轮结论只依赖 CPU 树，与 P1-7 的 GPU 口径说明同类）。

**实现时踩到的坑（值得记）**：`chat_prompt` 首版把"恒保留前缀的块数（`head_blocks`）"与
"组装起点（`first`）"当成同一个量（`first = head_blocks`）——于是有 system 块时**整个 system 块
被静默跳过**（`--show-tokens` 一看就少 30 来个 token）。这正是"用 `--show-tokens` 做逐 token
断言"的价值：`chat_prompt_test` [1] 的模板逐 token 比对会立刻红，而不是等用户发现"系统提示
好像没生效"。修法 = 两个量分开（恒保留 `0..head_blocks`、丢弃只发生在 `keep_from` 之后）。

**已知取舍（如实记录）**

- GUI 路径每轮把整段历史重新 encode 一次并重放 prefill：没有跨轮 KV cache 复用（本项目
  推理侧本来就没有跨请求 cache）。**取**：每轮可改参数、历史在 GUI 手里、无 stdin 协议。
- GUI 的 assistant 轮以**文本**回灌（`--turn assistant:<text>`，由 CLI 侧重编码），
  不是 token —— 无状态协议只能传文本；`--interactive` 路径用原 token（有进程内状态）。
  BPE 下 decode → encode 不保证逐 id 往返，故两轮的上下文会有极小的重编码差异；对"上下文
  不丢"（前缀不变式）无影响（前缀断言在 CLI e2e 里用的是同一条 `--turn` 形状）。
- 超 `seq_len` 时按"一问一答"整轮丢弃（不是按 token 滑窗），故极端长单轮仍可能超限 ——
  交给 `generate` 的滑动窗口兜底（它保留尾部的 `<|assistant|>` 开标记）。
- 系统提示默认值仍在**两处**字面量：`src/text_infer.cpp` 的 `--system` 默认与
  `cli_controllers.py` 的 `DEFAULT_SYSTEM_PROMPT`（GUI 字段默认）。`chat_prompt_test` 的
  CLI 端到端显式传 `--system SYS`，避免把这份字面量复制进测试；两份默认值若漂移，GUI 与 CLI
  的默认行为差异只在"用户没改系统提示"时可见（GUI 总是显式发 `--system`）。

---

# P1-5：CI 扩 MSVC Debug + 权重变化断言（2026-10-18）

- 类型：演进记录 / 门禁补强
- **原状**：CI 的两个平台矩阵都显式用 `clang++` + Release，Debug 专有路径（STL 断言、
  迭代器调试、未初始化读、assert 分支）没有自动覆盖。权重变化断言只存在于
  `f16_writeback_probe`，而它在无 Vulkan 的树里整例返回 77（skip）⇒ CPU-only Debug
  job 覆盖不到。
- **改动**：
  ① `.github/workflows/cmake-single-platform.yml` 新增 `msvc-debug` job：
  `windows-latest` + **Visual Studio 2026 生成器**（直接得到 cl.exe，无需 vcvars 脚本），
  `NN_ENABLE_NATIVE=OFF` / `NN_ENABLE_GPU=OFF`（纯 CPU，避开 Vulkan/glslc 变量）/
  `NN_ENABLE_TESTS=ON`；`cmake --build`（Debug 配置）后 `ctest -C Debug`（除 C11 外全跑）。
  **2026-10-18 补正**：GitHub 的 `windows-latest` 已切到 VS 2026 镜像，
  `Visual Studio 17 2022` 生成器找不到实例；工作流改为 `Visual Studio 18 2026`。
  ② `src/optimizer_test.cpp` 新增 [4] 训练冒烟：小 MLP + Adam 跑 8 步，断言 loss 有限且下降、
  **全参数 L1 位移 > 1e-3**；同一个 `detect_movement()` 的负例（同一快照自比）必须判为
  "无移动"，证明断言不是恒真。该段纯 CPU f32，因此 MSVC Debug CPU-only job 会真实执行。
- **验收**：本机 `build/cpu`（Release + CPU-only）`optimizer_test` ALL PASS，
  实测 `loss 3.016254 → 0.155520（8 步），参数 L1 位移 41824.5`；workflow YAML 解析通过。
  另用本机 `build/p1-5-debug`（clang 24 / Debug / `NN_ENABLE_NATIVE=OFF` / CPU-only）
  跑通全量构建（**98 个 ninja 步骤**），`optimizer_test` 同样 ALL PASS（`3.016255 → 0.155522`）。
  Debug 全量 ctest **51/52**：唯一失败是既有 **C11**（`rapt_test_f16` 第 9 步 NaN）；
  按 CI 同款排除后 **51/51 通过**（165.28 s，18 条 GPU 用例按 77 skip）。
  该问题用户已排查，**怀疑是设备/后端自身问题；Release 下不触发只是概率**，不是"Debug 专有"，
  因此本轮不把它算作 P1-5 回归。已在 18 §3.1 登记，CI 的 Debug 测试步暂以
  `-E "^rapt_test_f16$"` 排除（根因定位后删除该排除项）。
  **MSVC 102 实机验证（2026-10-18，`ethan@192.168.1.102`，全新目录 `nn-p1-5-msvc`）**：
  VS 2026 18.10.4 / **cl 19.51.36260.0** / CMake 4.3.3，Debug + CPU-only。首次全量构建
  `text_train.cpp` 触发 **C1128**（节数超对象文件格式限制）⇒ 加 `/bigobj`（MSVC 目录级 +
  接口选项，消费方大 TU 同样受益）；随后 ctest 有 **9 个用例 0xC00000FD 栈溢出**
  （`t1_58` / `bitnet_struct` / `model_roles` / `batchnorm` / `model_spec_validation` /
  `text_train_modules` / `train_state` + `t1_58_test_gpu` / `bitnet_struct_test_gpu`）——
  MSVC 默认栈 1 MB，Linux 是 8 MB ⇒ 加 `/STACK:8388608` 后全部恢复（7 个直接运行 EXIT=0）。
  CI 口径（C11 排除）**51/51 通过 / 0 失败 / 180.00 s**（20 条 GPU/app 依赖用例 skip；
  `rapt_test_f16` 在该 MSVC Debug 树单独跑 **3/3 通过**，排除只是为了避免概率性红）；
  **另跑不排除的完整 ctest：52/52 通过 / 181.61 s**（`rapt_test_f16` 2.33 s 通过）；
  `optimizer_test` MSVC 实测 `loss 2.773293 → 0.170448（8 步），参数 L1 位移 41965.2`。
- **口径**：`/bigobj` 与 `/STACK:8388608` 已在 102 实测触发后加入，不是预先猜测；两者均写入
  `CMakeLists.txt` 的 MSVC 分支（`/bigobj` 同时随接口传给消费方）。
- **顺带清理**：`t1_58_test` 的 `--expect-post-init-abort` 与 `assert_always_test` 的四个
  crash 负例都是**故意 abort** 的子进程；原先每次 ctest 都会让 systemd-coredump 留下
  多条 SIGABRT 记录（容易误判为真故障）。现在子进程入口先 `setrlimit(RLIMIT_CORE, 0)`
  （POSIX；Windows 不适用），父进程的"非零退出 + stderr 断言"语义不变。

---

# 仓库历史整理（legacy 接入 + 大文件剔除，2026-10-18）

- 类型：演进记录 / 仓库维护（**所有提交哈希已变化**）
- **原状**：仓库存在两条互不相干的历史——`legacy`（2026-05-15 `Init` … 2026-08-31
  `clang++ & libstdc++`，209 个提交）与 `main`（2026-09-11 `Release v0.0.1` 起重起，
  开头 7 个提交是把 legacy 各阶段压成的快照）。两条线没有共同祖先；`main` 上的
  `v0.0.1` … `v1.1.0` 与 legacy 的历史内容重复。
- **整理目标**：把 legacy 的真实提交接回主线，去掉重复快照与错误提交的模型 / 词表 /
  数据集 / 构建产物，同时保留原始提交时间。
- **做法**：
  ① 以 `v1.2.0` 快照作为**内容桥接提交**（它的树 = 旧 v1.2.0 的树），父提交改为
  legacy 末端；`v0.0.1` … `v1.1.0` 六个快照不再位于 `main` 的祖先链，但仍以 tag
  形式保留为归档（tag 指向的提交也一并清除了大文件）。
  ② 全历史剔除：`gpt_model.bin` / `gpt_model.nnmodel` / `mnist_model.bin`、
  根目录各版 BPE / CharBPE 词表 JSON 与 `vocab_*.txt` / `dataset.txt`、
  `compare_with_torch/gpt_bpe_torch.json`，以及 `build2/` `build_fused/` `build_review/`
  `build_test/` `build_verify/` 下的 PDB、可执行文件、Ninja/CMake 生成物。
  ③ `--prune-empty` 移除剔除后变空的 2 个 legacy 提交（提交信息均为 `1`）。
  ④ 所有保留提交的 author date / committer date 原样保留（含 8 个 author≠committer
  的提交）；`v1.2.0` 及之后的 tag 随祖先改写重新指向，tag 日期不变。
- **结果**：`legacy` 仍是独立分支（207 个提交），并成为 `main` 的祖先；`main` 与
  `dev/api_opti` 的最终树与整理前**逐文件一致**，仅少了上述被剔除的产物。
  `dev/api_opti` 在 `main` 之后保留原有 41 个提交，随后合并进 `main`。
- **协作影响**：这是**不可快进的历史改写**，远端 `main` / `dev/api_opti` /
  `legacy/compiler-compat` 与全部版本 tag 均已 force 更新。旧克隆不要再向这些分支
  合并或推送，按 `git fetch --all --prune --force` + `git reset --hard origin/<branch>`
  重新对齐；改写前的完整备份 bundle 保存在维护者本机（不进入仓库）。

---

# 生命周期 v3 与 Session 层（v1.10.0，2026-10-20/21）

- 类型：演进记录 / 删除清单 / 设计裁决（**当前契约**以 `docs/development/09-engine-ownership.md`
  与 `docs/todo/28-lifecycle-v3.md`、`docs/todo/29-session-architecture.md` 为准；本条只归档
  「为什么删/为什么这样改」与已否决方案）。
- **原状（v1.9.0）**：`Tensor` 保存 `EngineId` provenance + `stamp_` / `adopt` / `bound` 出生绑定机制
  （`core_engine_id.hpp` 83 行；`stamp_` 35 处、`adopt` 18 处、`bind_check_` 42 处、
  `NN_BIND_DEBUG` 3 处 + CI 第二轮 ctest）。`bind_check_` 只让操作数**互相**比较（首个 bound 作基准），
  从不验证操作数属于当前引擎 ⇒ **D1**：单操作数入口可让外来 GPU 张量进入任意引擎（读外来
  `VkBuffer` = UB）。
- **裁定（用户）**：设备域从**存储**派生 —— CPU 存储 → 进程常量 host 域；GPU 存储 → 其
  `GpuBuffer` 的 backend 标识（buffer 持 backend keepalive）；空/无效 → 0（交给既有无效检查）。
  删除 `EngineId`、`stamp_`、`adopt`、`bound`、`NN_BIND_DEBUG` 与 `ComputeEngine::id()`；
  `import` 是唯一跨域入口；每个 public NVI 入口**逐操作数**比较设备域。机制点 **101 → 42**，
  `sizeof(Tensor)` **CPU 64→56 B / GPU 88→80 B**，净代码约 −120 行。
- **同批收口**：D1（`GpuEngine::to_matrix` 拒绝异 backend）；D3（`TrainingSession` 同时拥有
  Model + Optimizer + TrainState，结构性消除 optimizer 非拥有 `TensorRef` 的 UAF）；D4（offload
  状态机 fail-fast + `discard()` 作废通道 + 精度组合配置期拒绝）；D5（`GpuBackend::shared()` 改
  weak 缓存 + 删 `instance()`，最后使用者释放后确定性回收）；D6（`ExprSegment` 持 `shared_ptr`；
  GPU `end_expr` 无录制明确报错）；D9 洁癖项（`BindingState` 三态化等）留后续。
- **P2b（v1.10.0-rc0.1）**：数据集存储单位 = **doc**、训练单位 = **sample**（`seq_len` 窗口）、
  `step = ceil(样本数 / batch)` 是派生量；`TextTrainer` 把 epoch×step / 累积 / 裁剪 / lr 调度 /
  测试评估 / 断点状态装配上收库层；`TrainingSession` 持有 `TrainState`；断点身份键
  `dataset_windows` → `dataset_samples`（**单向**旧键兼容），`num_windows` → `num_samples`。
- **正式版（v1.10.0）收尾**：独立发布审查 P0 = 0 / P1 = 1 / P2 = 9；发布前修 P1-1
  （`TextTrainer::create` 校验配方 seq_len ↔ 模型 `spec.seq_len`）、P2-1（`--resume` 的
  `adopt_resume_state` 之后重新校验精度×offload 组合）、P2-2（空 test 子集恢复打印"样本数: 0"）；
  其余 P2 记录在 `docs/release-notes/v1.10.0.md` §2.5 / §9。四棵树矩阵与真实数据端到端验收
  见 `docs/todo/25-release-requirements.md` §7。
- **否决方案**：① 保留 provenance 但"按操作数两两比较 + 补 `this` 比较"（D1 漏点仍在、机制税照付）；
  ② 用 `dynamic_cast` 查具体层类型做形状校验（铁律 #14 禁止）；③ 把 offload slab 精度协议化
  （本轮选"配置期拒绝"的最短正确路径）；④ `Layer::sequence_length()` 全覆盖形状校验
  （留后续，本轮用 `Model::spec()` 校验带 spec 的模型）。
- **影响**：破坏性 API（迁移表见 `docs/release-notes/v1.10.0.md` §6）；模型文件格式 v8 不变；
  v1.9.0 二进制读不到 rc0.1+ 写出的 `dataset_samples` 键。

## TrainingSession 单步边界与 global/micro batch 语义（2026-10-22）

- 类型：演进记录 / 破坏性 API 修订 / 设计裁决
- **原状**：`TrainingSession` 名义上只拥有 Model + Optimizer，但同时持有包含 epoch、lr 曲线、
  batch/accum 与数据集身份的 `TrainState`；`TextTrainer` 与 `mnist_train` 各自手写
  zero_grad → forward/loss/backward → clip → step 管线。CLI 的 `batch_size` 是单次
  forward/backward 的 micro-batch，`accum_steps` 实际把 update batch 放大为
  `batch_size × accum_steps`。
- **裁定**：`TrainingSession` 接收已装配的 **micro-batch**，通过任务 Step 回调执行
  forward/loss/backward，内置梯度累积，并在 update 边界完成归一化、裁剪、step、zero_grad；
  不拥有数据、epoch、lr 曲线、评估或断点策略。`batch_size` 改为一次 optimizer update
  的理论 **global batch**，`micro_batch_size = batch_size / accum_steps`；step、lr 调度、
  checkpoint 与计数器统一按 optimizer update 计数。
- **归一化裁决**：禁止把各 micro 的均值简单乘 `1/accum_steps`。Step 回调报告已采用的
  `gradient_units`，Run Driver 传入整个 global update 的 `normalization_units`，Session
  按 `gradient_units / normalization_units` 缩放后再累积。文本 sparse-CE 以全 global
  batch 的有效 token 数为分母；`TextSample.loss_mask` 是 token 是否参与 loss 的最终样本语义。
- **状态拆分**：Session 只保留 lr/optimizer step/累积 microsteps/pending loss；运行态
  `TrainState` 由 `TextTrainer` 装配持有，`.nnckpt` 保存时与 Session/optimizer 状态合并。
  外层 v8 文件格式不变，训练状态段新增 `batch_size_semantics` 与 `micro_batch_size`。
- **兼容**：旧状态缺语义键时按 legacy micro-batch 解释，恢复为
  `global_batch_size = legacy_batch_size × accum_steps`，并把 micro step 计数折算成 update
  step。旧断点若落在累积中间，pending gradients 原本未落盘，只能从最近 update 边界继续。
- **证据**：远端 Windows clang + MSVC STL / Vulkan，全量构建 120/120；ctest 59 注册，
  58 pass + 1 个平台条件 skip（`scan_bitwise_test`）+ 0 failed。`doc_align_audit`、
  `doc_inventory`、`gui_cli_audit` 均 exit 0。

## 训练骨架 RunDriver 与开放规格 RFC 32（issue #39/#26，feature/run-driver-openspec）

- **动机**：`TrainingSession`（单步）与 `TextTrainer`（文本域驱动）之间没有领域无关
  骨架，自定义域从 `train_step` 往上全要自己写（go CLI `go_train_cli.cpp` 达 1456 行，
  其中 epoch 循环 / 断点节奏 / lr 接线 / 指标落盘全是与领域无关的重复）。
- **RFC**：`docs/todo/32-trainer-skeleton-and-openspec.md`，四扩展点统一心智 =
  「虚基类定义契约 + 主仓持基类引用 + 子类注入特化」（TrainState / ModelSpecBase /
  BatchSource / RunDriver）。
- **阶段 A**：`TrainState` 虚基类化（编解码骨架 + `state_kind`/`write_extra_kv`/
  `read_extra_kv`/`verify_extra` 钩子），`stride` + `DatasetIdentity` 下沉为
  `TextTrainState`；`verify_resume` + `ResumePolicy` 把 go CLI 手写的断点交叉校验
  （配方形状 / 计数器越界 / 数据集身份）上收为库口径。
- **阶段 B**：`BatchSource` Pull 契约 + `IndexedBatchSource` 预置适配器
  （单 RNG 流重放式 shuffle：恒等序起同一 RNG 重放 0..epoch 次，(seed, epoch)
  唯一决定顺序且幂等）；`steps_per_epoch=nullopt` = 步进驱动。
- **阶段 C**：`RunDriver`（epoch×step 循环、`compute_epoch_lr`/`compute_step_lr`
  双级接线、epoch 边界 + save_interval 断点节奏、计数器"下一步位置"归一化、
  探针中止 exit 9、--max-steps/--stop-after-epoch、续训起点 + rewind）。
  分层裁定：`train_step` 是 Batch 模型模板 ⇒ 取批 + micro 切分 + `train_step`
  归域（`RunHooks::do_update`），driver 只给 lr/步号/final 标记。
- **评审修复**：① `global_step` 原为进程本地计数（续训断点会从 0 重数），改为
  绝对步号 `epoch·steps_per_epoch + step`，断点计数器与 step 级调度统一；
  ② 适配器 shuffle 原为每 epoch 重播种（与 TextTrainer 单流语义冲突，同 seed
  轨迹不同），改单流重放。
- **注记迁移**：此前列在头文件注释里的版本注记（宿主桥公开 API、
  `param_children` 广播、`logits_transform` 钩子、`lr_per_epoch` 正式化的
  v1.14.0/issue 编号）系历史信息，已从注释移入本条——**注释只写行为契约，
  不写历史**。v1.14.0 从未定档：最新版仍是 `v1.13.0`，`dev` 从不定档版本号，
  版本只在经审核合入 `main` 时确定。
- **兼容口径**：仅向下兼容（新读旧）且不承诺——该能力是 v4 模型格式引入 kvrec
  的特性，不是新增保证；反向与跨版本逐位一致性不在承诺范围。
- **验证**：ctest 70/70（新增 `batch_source_test` 6 组、`run_driver_test` 9 组；
  `train_state_test` 含状态段逐字节重写一致校验）。

## 开放规格 ModelSpecBase（issue #26，feature/openspec-modelspec）

- **动机**：`ModelType` 是封闭枚举，`spec_matches` / `spec_summary` /
  `spec_to_kv` / `spec_from_kv` 四处 `switch(type)` 构成封闭世界——衍生项目
  加自定义模型必须改主仓枚举与所有分支。
- **设计**（RFC 32 §6）：`ModelSpec` 改为**公共字段外壳**（output_schema /
  output_packing + `unique_ptr<ModelSpecBase> arch`，深拷贝 clone 语义），
  类型参数与判别符（`type_name()`）随多态对象走；五个内建子类
  （MlpSpec/TransformerSpec/GptSpec/CnnSpec/RaptSpec）承载各自字段与
  `arch_compatible` / `summary` / `write_kv` / `read_kv` 虚实现。
- **文件格式**：`type` 键从数值枚举改为**类型名字符串**；读侧数值回落
  （`legacy_spec_type_name`，0-7 一一映射，4=ALiBi_GPT 归 GptSpec 同族，
  6=ZiPT 维持明确报错）。仅向下兼容（新读旧），反向不承诺——v4 模型格式
  引入 kvrec 的自描述特性本就只保证这个方向。
- **开放点**：自定义类型 = 写子类 + `ModelRegistry::instance().register_type`
  （内建类型直通 `make_spec_by_name`，未命中查注册表），加新模型不改主仓
  任何 switch；由 model_spec_validation_test §8 验收（注册/往返/虚分发/未注册报错）。
- **拍板**（RFC §10 #3）：`build()` 不进基类——build 需外壳公共字段 +
  运行期 precision + Model 类型，属域层自由函数；`model_spec.hpp` 保持纯数据头。
  封闭点已收敛为四个虚方法 + 工厂。
- **迁移**：约 180 处字段读取点类型化（`spec.as<T>()` / `emplace<T>()`），
  分布在 4 个 domain 头、序列化、recovery、4 个 CLI、11 个测试；
  `is_gpt()` 等判别调用点保留（实现改为 dynamic_cast），调用面无感。
- **踩点**：`CnnSpec.norm_type` 的结构默认必须是 LayerNorm（旧扁平布局的
  缺键回落值），builder 默认 BatchNorm@Conv 由 `make_cnn_spec` 显式传入并
  落盘——两者混淆会让"缺 norm_type 键的旧文件"读成 BatchNorm、参数布局错位。
- **验证**：全量构建 0 错误；ctest 70/70（含 spec 往返、旧数值 type 回落、
  ZiPT 拒绝、norm_place 错配拦截、三值权重往返、自定义注册表 §8）。
