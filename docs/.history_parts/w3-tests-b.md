# w3-tests-b

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
