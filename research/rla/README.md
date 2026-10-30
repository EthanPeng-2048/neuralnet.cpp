# RLA / RAPT 研究档案

> 研究基线：`9b5e7ec`（2026-10-24）。
> 本目录回答两个不同的问题：
>
> 1. **RLA 是什么？** 已有算法与实现基线，见 [`01-what-is-rla.md`](01-what-is-rla.md)。
> 2. **RLA 效果怎么样？** 实验尚未执行，只登记研究方案与验收口径，见
>    [`02-rla-effectiveness.md`](02-rla-effectiveness.md)。

## 1. 状态

| 文档 | 状态 | 可以怎么引用 |
|------|------|--------------|
| [`01-what-is-rla.md`](01-what-is-rla.md) | ✅ 定义基线 | 可引用为“本仓 `9b5e7ec` 的 RLA-2/RAPT 算法与实现说明”；不是质量结论 |
| [`02-rla-effectiveness.md`](02-rla-effectiveness.md) | 🚧 TODO，未执行 | **不可**引用为效果结论；可引用为计划中的实验矩阵与判定标准 |

“RLA 是线性注意力”是复杂度层面的机械事实；“RLA 比 softmax 注意力更好/更快/更可解释”
是效果主张，必须等 `02` 完成并有原始数据、样本量、设备和边界后才能写入可引用结论。

## 2. 命名边界

- **RLA**：ReLU Linear Attention，指用逐元素 ReLU 特征映射构造线性注意力的算法族。
- **RLA-2**：本仓当前运行的修正版：Sum 分母、Q/K 内置逐头 RMSNorm、无二次型 A 状态。
- **RAPT**：本仓的 decoder-only 模型与工程实现。它用 RLA-2 替换 softmax 自注意力，并配套
  Pre-Norm 残差块、FFN、LM head 与 RoPE。
- **LRLA**：旧概览里提到的可学习 ReLU 特征映射扩展；当前仓库只有设计描述，没有实现。

因此“RLA 的效果”至少应拆成品质量和运行效率两类；不能拿 `rapt_test` 通过来推导模型质量，
也不能拿理论复杂度直接推导实际吞吐。

## 3. 事实来源优先级

本文档以当前源码为算法事实源：

1. [`compute_layer_rapt.hpp`](../../include/neuralnet.cpp/compute_layer_rapt.hpp)：
   RLA-2、RAPTBlock、RAPTModel。
2. [`domain_rla.hpp`](../../include/neuralnet.cpp/domain_rla.hpp)：公开构建约束与配置。
3. [`compute_engine.hpp`](../../include/neuralnet.cpp/compute_engine.hpp)：扫描原语语义。
4. [`src/rapt_gradcheck.cpp`](../../src/rapt_gradcheck.cpp) 与
   [`src/rapt_smoke_test.cpp`](../../src/rapt_smoke_test.cpp)：正确性和接线证据。
5. [`docs/development/06-rapt-algorithm.md`](../../docs/development/06-rapt-algorithm.md)：
   面向使用者的概览。历史设计叙述若与源码冲突，以源码和 `01` 为准。

## 4. 当前不回答的问题

- RLA 是否在某一学术意义上“首创”；本仓目前没有文献综述或优先权证据。
- RLA 在语言建模、长上下文检索或真实业务任务上是否优于 GPT/softmax 注意力。
- RLA 在相同参数、token、算力和种子预算下是否更快、更省显存或更稳定。
- “可解释性最高”“最适合硬检索”等旧文表述是否成立。

这些问题进入 [`02-rla-effectiveness.md`](02-rla-effectiveness.md)，不得用设计直觉替代测量。
