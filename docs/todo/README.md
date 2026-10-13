# todo/ —— 项目开发者内部工作区

> **本分册写给项目开发者**（改这个库的人），不是给调用者的。
> 调用者要看的开发文档在 [`../development/`](../development/)；历史归档在 [`../history/history.md`](../history/history.md)。

## 这里的文档怎么组织

- **一件事一篇**：设计稿（含未实施部分）、路线图与裁定、现状盘点、编码规范、踩坑记录、
  未修故障、审查/评审整改立项。
- **允许带进度状态**（✅ 已实施 / ⏳ 待做 / ❌ 已否决），但**不写历史叙事**——
  演进过程、被否决方案的理由、旧数字一律进 `../history/history.md`。
- 编号沿用历史编号（`02`…`23`），**编号缺号是正常的**（`05` `06` `20` 等留在
  `development/`；`13` 已删除；`15` `16` 已并入 `history/history.md`）。

## 分类与索引

### 立项 / 计划 / 台账（先看这些）

| 文档 | 何时读 |
|------|--------|
| [`00-code-review-remediation.md`](00-code-review-remediation.md) | **2026-10-11 代码审查整改立项台账**：审查发现的问题逐条立项（现象/证据/根因/方案/验收/优先级） |
| [`18-roadmap.md`](18-roadmap.md) | **"下一步该做什么"的唯一入口**：未来 6–12 个月方向 + 未完成项裁定台账（做 / 条件触发 / 不做） |
| [`04-memory-optimization.md`](04-memory-optimization.md) | 显存优化计划：L1 激活重计算（已实施）/ L2 内存池归还（已实施）/ L3 远期 |

### 现行设计 / 总纲（改对应模块前读）

| 文档 | 何时读 |
|------|--------|
| [`17-unified-tensor-engine.md`](17-unified-tensor-engine.md) | Tensor / ComputeEngine / MemoryPool 底层总纲（M1–M6 已实施；M7 未立项）：访问不变量、InitSpec、批量读写、Matrix 降级、内存契约、`import`、Layer 删 engine 形参 |
| [`02-operator-fusion.md`](02-operator-fusion.md) | 算子融合全篇（归约语义 IR → 表达式录制 → matmul 参与融合 → 跨 kernel 自动融合）：改融合/IR 前必读 |
| [`03-ir-optimization.md`](03-ir-optimization.md) | IR-A/B/C/D 优化链设计（IR-C = 图 IR + `begin_expr/end_expr` 融合分析） |
| [`19-unified-dataset.md`](19-unified-dataset.md) | 统一数据集（`.nndataset` / `.nnvocab` / KVRecord v2）格式与 `nn::Dataset` |
| [`21-quantized-weights.md`](21-quantized-weights.md) | 三值（1.58-bit）量化权重与 `BitLinear`：`Precision::T1_58`、按槽能力声明、打包 GEMM |
| [`22-bitnet-2b4t-architecture.md`](22-bitnet-2b4t-architecture.md) | BitNet b1.58 2B4T 的结构对齐：SubLN / GQA / tied embedding（与 21 号正交） |
| [`23-library-design-cleanup-and-model-file-roles.md`](23-library-design-cleanup-and-model-file-roles.md) | 库级违规设计清理：模型文件角色（`.nnmodel` / `.nnckpt`）+ 参数形态协议 + 引擎能力体系 |

### 现状盘点 / 规范 / 教训 / 故障

| 文档 | 何时读 |
|------|--------|
| [`12-compute-engine-inventory.md`](12-compute-engine-inventory.md) | 引擎接口现状盘点（复现：`bench/doc_inventory.ps1`）：virtual 方法数、Layer 直调分类、两套 CPU 求值机制、ctest 计数 |
| [`10-development-standards.md`](10-development-standards.md) | C++ 编码规范全文（内存管理 / 模块隔离 / 高性能 / 错误处理 / 注释） |
| [`08-pitfalls-and-lessons.md`](08-pitfalls-and-lessons.md) | **踩坑警示录，改代码前读**：分级 bug 与跨领域根因模式 |
| [`14-f16-stable-gpu-loss-frozen.md`](14-f16-stable-gpu-loss-frozen.md) | **未修故障报告**：GPU `--precision-stable f16` 训练 loss 打印冻结（权重不冻结）——触发矩阵、覆盖缺口、证据 |
| [`24-f16-training-defects.md`](24-f16-training-defects.md) | **f16 训练缺陷台账**：全 f16 NaN（`stable=f16` CE 溢出 + `optimizer=f16` 二阶矩下溢）、loss 冻结（GPU 回读 `14` + 数值泛化）、`--f16` 降速——含报告核对与立项 F1–F5 |
| [`25-release-requirements.md`](25-release-requirements.md) | **大版本发布要求**（`1.x.y → 1.(x+1).0`）：端到端模型训练测试 / 发布文档 / 完整 ctest + Debug 构建 / 多编译器 × 标准库矩阵 |

## 相关（相邻分册 / 总档）

| 想去哪 | 文档 |
|--------|------|
| 全仓历史归档（演进 / 勘误 / 否决 / A-B） | [`../history/history.md`](../history/history.md) |
| 总索引与分册定位 | [`../README.md`](../README.md) |
| 性能对比基准 | [`../benchmarks/`](../benchmarks/) |
