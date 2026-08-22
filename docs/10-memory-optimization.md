# 显存优化计划（Memory Optimization）— GPT + Vulkan/CPU 训练显存下探

> 文档状态：规划（草案）
> 关联文档：[09-operator-fusion.md](./09-operator-fusion.md)（算子融合已完成部分）

## 0. 背景与目标

以实测配置为基准（滑动窗口，seq_len=1024，stride=1024，样本数 435736）：

```
词表 32782 · 模型维度 1024 · 注意力头 16 · Transformer 层数 16
FFN 维度 4096 · 序列长度 1024 · 优化器 adamw · 批大小 6 · GPU 启用
```

该配置下 **训练峰值显存 ≈ 27 GB**（已含 M5/M6 融合收益），此前 **29 GB**（融合前）。

- 融合（M4/M5/M6）已把 **非线性象限**（`seq²` 注意力、`vocab×seq` 全 softmax）从 ~3× 砍到几近为 0。
- 剩余 27 GB 的构成与融合无关，**集中在激活缓存、优化器状态、内存池碎片化三处**。

### 0.1 显存总账（估算，单位 GB）

| # | 项目 | 计算 | 约占 27 GB |
|---|---|---|---|
| ① | 参数 + 梯度 + Adam(m/v) | ~236M × 4B的4× ≈ 3.7 | ~14% |
| ② | 逐层激活缓存（16 层） | 每层 ~0.4 ≈ **6.4** | ~24% |
| ③ | seq² 注意力分数/p | M6 已清除，≈ 0 | — |
| ④ | logits（未经软银缩的 LM head 输出） | 6×1024×32782×4B ≈ 0.8 | ~3% |
| ⑤ | 反向临时 + 内存池碎片/不归还 | 其余 | ~59% |

> 注：①≈3.7、②≈6.4、④≈0.8 为**可推导的硬性下限**；⑤ 为反算残差，需实测采样确认真实构成（见 §4 仪器化）。

### 0.2 目标分级

| 级别 | 手段 | 目标降幅（相对 27G） | 触及架构红线？ |
|---|---|---|---|
| **L1 白拿** | 激活重计算（梯度检查点） | ② 6.4→0.6，最大省 ~6G | 否（Layer/Model 级）|
| **L2 回收** | 内存池复用/归还 + 生命周期边界 | ⑤ 中碎片部分，省 ~2-4G | 否（后端内存管理）|
| **L3 算法** | 自动融合优化（替代手写注意力）+ 优化器降内存 | ③+① 二次缩减 | 否（复用现有自动融合路径）|

优先级按此从 L1 做起，L1/L2 不影响数值正确性与分层铁律，风险最低、收益可立即验证。

---

## 1. 现状与根因（为什么是 27G）

### 1.1 数值精度
- [config.hpp](../../include/neuralnet.cpp/config.hpp) 中 `using Scalar = float`，全项目纯 fp32。
- 激活、梯度、权重、优化器态全部 4B。**不引入 f16（bf16/fp16）半精度训练**——数值统一 fp32，避免触及"单套 fp32"精度红线。

### 1.2 激活缓存策略（核心问题 L1）
- 层内 forward 为 backward 保留中间结果：如 [compute_layer.hpp](../../include/neuralnet.cpp/compute_layer.hpp) 的 `input_cache_`（Linear）、`sigmoid_cache_`（SiLU）、注意力保留的 Q/K/V 与 norm 输入等。
- `Model::forward` → `Model::backward`（[model_container.hpp](../../include/neuralnet.cpp/model_container.hpp)）是**逐层顺序**执行，层间不丢弃激活。
- 每层驻留约 8 个 `B·seq·d` 与 2 个 `B·seq·d_ff` 的 fp32 副本（d_ff=4096 每张 100MB 是重要大头），16 层累加 ~6.4G。**无梯度重计算（activation checkpointing）**。

### 1.3 内存池碎片化 + 不归还（问题 L2）
- [memory_pool.hpp](../../include/neuralnet.cpp/backend/memory_pool.hpp) 已实现 **first-fit 子分配 + 相邻 free region 自动前后合并**（O(log n)、O(1) 合并）。
- 但：**从未将整个空 Block 归还 GPU**（`blocks_.clear()` 仅在析构时触发），block 底材按需 128MB（或超尺寸单块）申请后不回收 ⇒ 峰值生命周期等于整个进程/测试生命周期，碎片与闲置块长期累积。
- 文档 09 §0 将"内存池 first-fit 碎片化 + 永不归还"列为**独立跟踪项、不随融合解决**。

### 1.4 注意力形态（L3，语义复杂）
- M6 已实现**两趟式**：forward 用 `batched_matmul_reduce/max → denom → apply` 不物化 `BH·seq²`；backward 默认**反向重算 `W`**（[09 §4 A 方案](09-operator-fusion.md)），省 `attn_cache_`，代价 2× FLOPs。
- 该路径已消除 `BH·seq²` 物化，剩余驻留为逐层 Q/K/V 激活集（属 L1 激活重计算可覆盖范围）。**不手写** flash-attention 类融合 kernel：沿用现有自动融合/算子生成路径（见 [09-operator-fusion.md](./09-operator-fusion.md)），由 L3 的自动融合优化统一推进。

### 1.5 分布式分片（超长序列，远期）
- 无张量/序列并行、无 FSDP/ZeRO 跨设备分片。真正进入几十万 token 时只能靠压缩 batch/seq。

---

## 2. 优先级与拆分（Roadmap）

按"不动分层、不动精度、收益先验证"排序：

### L1 — 激活重计算（梯度检查点）【提案，优先做】
- **目标**：把逐层激活从"全存"降到"只存 block 边界 input（或隔层存）"，backward 重算中间，使② 从 ~6.4G 降到 ~0.6G。
- **归属层**：`Model`（[model_container.hpp](../../include/neuralnet.cpp/model_container.hpp)）编排 + `Layer` 契约新增可选方法，`Layer::compute_backward_from_scratch`（默认走缓存；支持者重算）。**不动 ComputeEngine、不动精度**。
- **接口**：
  ```cpp
  // Model 层
  void set_checkpoint_every(std::size_t /* layer_stride */);
  // Layer 可选
  virtual bool recompute_supported() const { return false; }
  virtual Result<Tensor> forward_recompute(
      ComputeEngine&, const Tensor& /* saved_input */) { /* 重算 forward，只存必要 O(1) 状态 */ }
  ```
- **实现方式**：`Model::backward` 遇到 checkpoint 层时，先 `forward_recompute(saved_input)` 重建局部中间，再按层回传梯度。
- **验证时序**：逐层数 1/2/4/8 层梯度与现有全存结果数值一致（复用 `gpt_gradcheck.cpp` 渠道）。
- **风险**：约 +1 次前向 FLOPs（训练可接受）；需保证重算路径与原始 forward 数值一致。

### L2 — 内存池复用/归还 + 生命周期边界【提案，低风险】
- **目标**：回收 ⑤ 中碎片与闲置底材。
- **方案**：
  1. `MemoryPool` 支持**整块释放**：`allocation_count==0` 且全 region 空闲的 Block，调用 `vkFreeMemory` 并从 `blocks_` 移除（阈值控制，避免抖动）。
  2. 统计上报：`pool_debug_stats()`（总占用/空闲/block 数/碎片比），供 §4 采样确认真实构成。
  3. 评估按"训练-step 生命周期池 + 长生命周期参数池"分池，避免 step 间临时张量污染参数驻留区；或由 `staging`/`Tensor::destructor` 返回池而非直接 `vkFreeMemory`（现状中间 Tensor 的归还路径需核对）。
- **范围**：仅 `backend/memory_pool.hpp` 与 Tensor 分配/销毁路径，不涉及算法。
- **风险**：低；需 benchmark 分配总时长与碎片比变化。

### L3 — 自动融合优化 + 优化器降内存（远期）
- **自动融合（替代手写注意力 kernel）**：不手写任何融合算子；基于既有自动融合/算子生成路径（[09-operator-fusion.md](./09-operator-fusion.md)），将注意力与 softmax 相关子图自动合并为更省内存的单一计算，消除 Q/K/V 激活集物化。
- **优化器降内存**：采用 Adafactor 类**无 m/v** 优化器，去除每参数两份 moment 状态（对应 §0.1 ① 中约一半），不引入 fp16/fp8 低精度 moment。

---

## 3. 可量化收益（预期）

| 级别 | 手段 | 预期省显存 | 累计（估） |
|---|---|---|---|
| L1 | 激活重计算 | ~6G | 27 → ~21G |
| L2 | 池归还+复用（碎片部分） | ~2-4G | ~21 → ~17-19G |
| L3 | 自动融合 + 优化器降内存 | ~2-4G | ~17 → ~13-15G |

> 说明：数值为估算；⑤ 的真实构成需 §4 采样后修正。理想状态下（L1+L2+L3）可降至 ~13-15G，接近成熟框架同量级（成熟框架还会再叠在线注意力与分片；本项目在线注意力以自动融合路径推进，不手写 kernel）。

---

## 4. 验证与回归（Instrumentation）

1. **显存采样**：在训练 step 间记录 `pool_stats`（块数/已分配/空闲/碎片比）与 GPU 总占用曲线，标出 step 内峰值出现阶段（forward/backward/optimizer）。
2. **逐项归因**：跑一次 step，分别关闭激活缓存（假想）、池归还、自动融合，量化每项独立贡献——修正 §0.1 的⑤残差。
3. **数值回归**：`gpt_gradcheck`、`rmsnorm_gradcheck`、`swiglu_gradcheck`、`softmax_gradcheck`、`matmul_fusion_test`、`ce_fusion_test` 保持全绿；训练 loss 曲线与参考一致。
4. **性能回归**：`compute_bench`/`mnist_bench` 确认"省显存"未以显著耗时退化为代价（重计算倍率、池复用消耗）。

---

## 5. 风险与取舍

| 项 | 说明 | 应对 |
|---|---|---|
| 重计算精度一致性 | 重算路径需与 forward 一致 | 先小层数回归、复用 gradcheck |
| 池归还抖动 | 频繁整块释放/重建开销 | 阈值 + 池分级，实测调参 |
| 自动融合子图边界 | 融合子图切分需与自动融合器对齐 | 复用现有自动融合回归渠道 |
| 优化器降内存 | 无 m/v 优化器收敛特性差异 | 独立评测 loss 曲线 |
| 分布式分片 | 超出当前后端范围 | 远期另立 |

---

## 6. 落地顺序建议

1. **先 L1（激活重计算）**：纯 Layer/Model 级，不动分层与精度，收益最大、风险最低，立即用 `text_train` + `gpt_gradcheck` 验证。
2. **再 L2（内存池归还/复用）**：接入 §4 统计，确认 ⑤ 真实构成后回收碎片。
3. 若仍不足 → 立项 **L3（自动融合 + 优化器降内存）** 独立里程碑。
4. 分布式按需远期推进。

---

## 7. 与既有约束的关系

- **不破坏分层铁律**：L1/L2 只在 `Model`/`Layer` 契约与 `backend` 内存管理内实现，`ComputeEngine` 原语与 `expr_dsl` 不动。
- **不破坏闭合世界/AOT**：L1/L2 不引入新 shader、不删改 `expr_registry`；不在运行期进入"未预生成表达式"分支。
- **不违背确定性**：全部路径保持定点、可复现；不引入 f16（bf16/fp16）低精度训练。