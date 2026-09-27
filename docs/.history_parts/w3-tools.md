# w3-tools

> 本文件摘录 `tools/`、`bench/`、`compare_with_torch/` 注释整改中移出的历史状态记录（原位置为整改前行号）。

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
- 内容：注释曾引用不存在的 `docs/14`（S1-S3/S5 章节）与已无对应文档的条目号 `P2-13`。前者改指现存的 `docs/development/02-operator-fusion.md`（二期 S1-S7 主线）；后者改为直接描述现行机制"同 RowMod/RotateHalf 的视图参数处理：不进 key，运行时填充"。`include/neuralnet.cpp/expr_spec.hpp:429` 亦有同款 `P2-13` 悬空引用，不在本次范围内，需另行处理。
