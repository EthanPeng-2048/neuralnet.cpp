# RLA 效果怎么样

> **状态：TODO，尚未执行。**
> 本文不是结果报告；不包含可引用的质量、速度或显存结论。
> 前置定义见 [`01-what-is-rla.md`](01-what-is-rla.md)，源码基线为 `9b5e7ec`。

## 1. 当前结论

**现在还不能回答“RLA 效果怎么样”。**

仓库已有的 `rapt_gradcheck`、`rapt_smoke_test`、KV-cache 一致性和 checkpoint/offload 测试回答的是：

- 解析梯度与有限差分是否相符；
- 整链能否 forward/backward；
- 增量状态是否与整序列路径一致；
- 重计算/offload 是否保持数值一致。

它们**没有**回答：

- 同等参数、token、算力和 seed 预算下，RLA 的语言建模质量是否接近或超过 softmax GPT；
- 长度外推和长上下文检索是否受益于固定大小状态；
- GPU/CPU 实测是否更快，训练显存和解码状态是否更省；
- ReLU 门控是否真的形成稀疏、可解释、可复用的检索模式；
- f32/f16 下的稳定性边界在哪里。

## 2. 完成本 TODO 前必须先修的可复现性问题

当前 [`TextTrainer::fit`](../../include/neuralnet.cpp/text_trainer.hpp#L484) 用
`std::random_device` 初始化 shuffle RNG，CLI 配置没有数据顺序 seed。多 seed 实验若只在模型
初始化一致、数据顺序不一致，就无法建立配对比较。

先决条件：

1. 在训练配置与 CLI 增加显式 seed；
2. 同一 seed 同时控制参数初始化、batch 顺序与任何采样/增广；
3. 保存 seed 到训练配方/断点；
4. 用双进程同 seed 跑小步数，断言参数与 loss 轨迹逐位一致。

没有这一步，不启动正式质量对比。

## 3. 要回答的问题

### Q1. 质量

相同参数量、相同 token 数、相同优化器/学习率搜索预算下，RAPT 与 softmax GPT 的验证
loss/perplexity 和下游任务表现是否可比？

### Q2. 长上下文

随序列长度与干扰项数量增长，RLA 的检索精度、困惑度和外推衰减如何变化？固定状态是否导致
容量瓶颈或近期偏置？

### Q3. 效率

在 prefill、训练和逐 token decode 三种负载上，RLA 的实际延迟、吞吐、峰值内存和状态内存
各自如何变化？理论 `O(L)` 是否转化为真实优势？

### Q4. 机制

ReLU 正支持集交集是否产生可观测的稀疏门控？门控统计能否预测检索行为？

### Q5. 消融

质量或效率差异来自：

- ReLU 核；
- Sum 分母；
- Q/K 内置 RMSNorm；
- RoPE 注入点；
- 固定状态压缩；
- 还是仅来自实现开销/调参预算？

## 4. 实验阶梯

### E0：工程前置门禁

目标：保证后续每个数字可重跑。

必须记录：

- commit、构建 preset、编译器/版本、CPU/GPU 型号、驱动；
- 参数量、有效 token 数、update steps；
- 数据集 sha256、tokenizer sha256、序列网格；
- seed、数据顺序 seed、学习率调度；
- f32/f16 profile；
- 原始 CSV、命令、日志；权重与大日志不入仓。

通过条件：同 seed 双进程结果可复现；CPU/GPU 各自内部确定性成立。

### E1：合成检索（先做小实验）

目的：在语言建模噪声之上分离“状态检索能力”。

建议任务：

- key-value recall：序列前部给 key/value，末尾 query 某个 key；
- associative recall：多对 key/value 加同型干扰项；
- copying / induction head 小任务；
- 扫描长度 `L`、干扰项数 `D`、关键信息距离 `Δ`。

对照：

- softmax GPT；
- RLA-2；
- 去掉 ReLU 的线性核；
- 去掉 Q/K RMSNorm；
- Sum 分母与 L2 分母。

主指标：task accuracy / exact match。诊断指标：不同距离、不同干扰项下的准确率曲线，而非
只报一个总平均。

### E2：短上下文语言建模

建议先用 TinyStories，固定 tokenizer、全局 batch 与总 token 预算，避免数据集差异混入。

最低对照矩阵：

| 组 | 注意力 | 目的 |
|----|--------|------|
| A | softmax multi-head，同 d/H/L/FFN | 主基线 |
| B | RLA-2 | 当前实现 |
| C | 线性核但不做 ReLU | 分离 ReLU 门控贡献 |
| D | RLA-2 去掉 Q/K RMSNorm | 分离数值预处理贡献 |
| E | RLA-2 改 L2 分母 | 分离 RLA-2 分母贡献 |

每组至少 5 个 seed。若算力不允许一次做满，先做 A/B 配对 pilot；C/D/E 只在 A/B 出现稳定差异
后继续，避免无目标的超参堆积。

主指标：validation loss、perplexity、训练/验证 loss 曲线积分。辅助指标：梯度范数、参数更新
幅度、NaN/Inf 步数。

### E3：长上下文与外推

在训练长度之外测试多个长度档位，例如训练 `L` 的 `1×/2×/4×`，并区分：

- **质量外推**：held-out language modeling loss；
- **检索外推**：needle/key-value 距离增加后的准确率；
- **状态容量**：同长度下干扰项增加；
- **位置策略**：当前 RoPE 参数与位置偏移行为。

若外部数据集成本过高，先用可合成的多文档检索；不能只凭一条生成样例判断。

### E4：效率

按三种负载分别测，禁止混成一个“速度”：

1. **训练 step**：固定模型/序列/batch，记录 tokens/s、step time、峰值显存；
2. **prefill**：全长整段输入，扫 `L` 与 `d`；
3. **decode**：逐 token 生成，扫上下文起点与生成长度，记录 TTFT、tokens/s、状态 bytes。

同时报告理论模型和实测值：

- FLOPs/step；
- 实际 device busy / kernel time（若探针构建可用）；
- host 等待与传输；
- 峰值设备内存；
- RLA 状态 bytes vs softmax KV bytes；
- f16/f32 差异。

效率归因还需记录一个实现事实：RLA-2 的公式不消费 A 状态，但当前通用
`scan_prefix_outer` 仍内部计算 `A=Σk'k'ᵀ`。因此实测延迟里包含这部分当前无用的工作，
不能从理论公式直接推断 kernel 成本。

设备至少覆盖一个 CPU 配置和一个 Vulkan 配置；结论必须绑定设备，不能把单卡结果写成普遍事实。

### E5：门控与可解释性

仅测试“score 非负”不算效果。需要观察训练后的行为：

- positive support overlap 的比例；
- 精确零 score 的比例；
- 每个 query 的有效 key 数；
- 文档内/跨位置的 score 分布；
- 答案位置与高贡献 score 是否对齐；
- 不同 seed/层是否稳定。

若声称“可解释”，至少要给出一个预注册的读出门控实验，例如定位关键 key 后干预其
value/score，验证输出按预测变化；普通注意力热图本身不足以证明可解释性。

## 5. 指标口径

### 质量

| 指标 | 用途 |
|------|------|
| validation loss / perplexity | 主比较量，报告均值 ± 标准差与 seed 数 |
| task accuracy / exact match | 检索和合成任务 |
| valid-token count | 统一不同长度/有效 token 的 loss 汇总 |
| loss curve AUC | 同 token 预算下的训练效率 |
| NaN/Inf/failed steps | 稳定性，不能从最终 loss 掩盖 |

### 效率

| 指标 | 用途 |
|------|------|
| tokens/s | 训练/prefill/decode 分别报告 |
| step time / TTFT / inter-token latency | 用户可见延迟 |
| peak device memory | 训练与推理分开 |
| state bytes | RLA 固定状态 vs softmax KV |
| FLOPs | 与实测延迟并列，不能替代实测 |

### 统计

- 质量结论至少 `n=5` 个独立 seed；
- 报 `mean ± sample std`、最小/最大值与每个 seed；
- 速度测量先做 warmup，重复多次，报中位数与离散度；
- 配对比较固定数据集、token 数、序列长度和调参预算；
- 不把单轮差异或不同预算的“最佳点”当作架构结论。

## 6. 判定规则

正式报告写入可引用结论前，至少满足：

1. **可复现**：同 seed 重跑一致；命令、哈希、设备齐全。
2. **配对公平**：参数/token/算力预算明确；超参搜索预算相同或明确记录差异。
3. **有基线**：质量至少含同结构 softmax GPT；效率至少含同模型与相邻序列长度。
4. **有样本量**：质量 `n≥5`，不能只展示一条漂亮曲线。
5. **有边界**：分别写明短上下文、长上下文、CPU/GPU、f16/f32 的适用范围。
6. **消融可归因**：若声称收益来自 ReLU/RMSNorm/Sum 分母，必须有对应消融而非只靠直觉。
7. **异常诚实**：NaN、偶发失败、退化 seed 与失败配置全部保留。

建议的结论措辞：

- ✅ “在 X 数据、Y 参数/token、Z 设备、n=5 下，RAPT 的验证 loss 为 `a±b`，GPT 为 `c±d`。”
- ✅ “在 850M、seq=…、batch=… 的 decode 负载上，RLA 状态从 `x` 降到 `y`，吞吐从 `u` 到 `v`。”
- ❌ “RLA 更快/更强/更可解释。”——没有数据集、预算、设备和边界的单句不能入库。

## 7. 预期产物

完成后本目录应新增：

```text
research/rla/
├── README.md                     # 状态与索引
├── 01-what-is-rla.md             # 本文档的配套定义
├── 02-rla-effectiveness.md       # 从 TODO 转为结果报告
├── scripts/                      # 生成数据、训练、汇总的入口
└── results/
    ├── runs.csv                  # 每 run 一行
    ├── summary.md                # 结论表（含 mean/std/n）
    └── charts/                   # 必要的质量/效率图
```

原始权重、完整日志和大数据切片继续由 `.gitignore` 排除；重跑命令必须能从公开/可再生成数据
恢复。

## 8. 已知风险

- `TextTrainer` 当前数据 shuffle 无显式 seed，正式实验前必须解决。
- RAPT f16 曾有已知不稳定项（历史登记为 C11）；f16 结果必须单独列边界，不能从 f32 外推。
- GPU 结果受设备与驱动影响；850M 与 40HX 不能混在同一列比较。
- RLA 的参数预算看似相近，但内部 Q/K RMSNorm 与扫描 kernel 的计算开销不同，必须同时比参数量
  和实际算力。
- 当前旧概览中的“更长上下文更强”“可解释性最高”“零超参数”等是待验证命题，不是本研究的
  先验结论。

## 9. 完成定义

只有以下条件同时满足，才把本文状态从 TODO 改为结果报告：

- E0 可复现门禁通过；
- E1/E2 至少完成一组配对质量实验；
- E3 或 E4 至少完成一项能支撑“长上下文/效率”边界的测量；
- 消融能解释主要差异，或明确写出无法归因；
- 所有数字带 seed 数、设备、数据集和命令；
- 报告包含“本实测不能说明什么”。
