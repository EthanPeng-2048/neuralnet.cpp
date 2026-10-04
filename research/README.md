# research/ — 研究与排查归口

本目录下每个子目录是**一次独立研究/排查的自包含归口**（报告 + 脚本/探针 + 原始日志 + 产物），
互不混写：

| 子目录 | 主题 | 主报告 |
|---|---|---|
| [`mnist/`](mnist/README.md) | MNIST：模型尺寸 / 训练超参 → 准确率（743 KB CNN，5 轮均值 99.34%） | `mnist/README.md` |
| [`f16_weight_decay/`](f16_weight_decay/REPORT.md) | GPT `--precision-param f16` 训练 loss 中途回弹 → AdamW 权重衰减写回的 f16 舍入缺陷（GPU `OpFConvert` 向零截断，每 optimizer step 掉 1 个 f16 网格步） | `f16_weight_decay/REPORT.md` |

约定（沿用原 `research/README.md` 的口径）：

- 入库范围 = 报告（`.md`）、脚本/探针、汇总 CSV/表、SVG 图；`.gitignore` 全域忽略
  `*.bin`（权重）与 `*.log`（日志），这两类只留在本地，可按各报告里的命令从零重跑。
- MNIST 研究的脚本路径常量按 `research/mnist/` 解析（`ROOT` = 仓库根，三层 `dirname`）。
