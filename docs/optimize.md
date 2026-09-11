# 显存/速度优化实测记录

## 梯度检查点（--checkpoint-every N）实测

同一 batch、同一配置（6step/batch × 20batch 积累）：

| N | time | vram |
|---|---|---|
| 0 | 22s | 26GB |
| 1 | 31s | 9GB |
| 2 | 27s | 16.6GB |

结论：
- **要最快 → N=0**（前提显存够）。检查点引入的额外前向 FLOPs 会随 batch 放大，拉高 batch 无法抵消。
- **要省显存换更大 batch → N=2** 性价比最优（几乎减半显存、只慢 ~23%）。
- **N=1 只用于极限压显存**（9GB），但重计算代价最大（+1 整遍前向）。

---

## Activation Offload（--activation-offload，L1-offload）

**思路**：检查点（重算）用 FLOPs 换显存；offload 用 **PCIe 带宽**换显存——把每块内部激活搬 host-visible 内存，backward 拷回再反向，**不重算**（FLOPs 保持 1.0×）。

### 预期收益（PCIe 3.0 x16 ~7GB/s，激活 ~5GB）
- 显存：~9GB 级（同 N=1）
- 每 step：~23s（vs N=1 的 31s），**快 ~8s，无重算**

### 实现（已落地，阶段1+2+3）
- 引擎原语：`ComputeEngine::offload_store(activation)` / `offload_load(handle, rows, cols)`（CPU no-op；GPU 用**录制式** `vkCmdCopyBuffer(dev↔host_visible)`，batch 内不提交，避免打断 batch）。
- 后端：`GpuBuffer::create_host_visible` / `GpuTensor::create_host_visible_empty`（HOST_VISIBLE 内存池）；`copy_buffer_region_gpu`（带偏移拷贝）。
- 层：`Layer::activation_cache()` 枚举 backward 所需中间激活；GPTBlock `export_activations`（forward 后搬 host + 释放 GPU）/ `import_activations`（backward 前拷回，不重算）。掩码小而常驻不参与 offload。
- **slab 复用**：每块用一块持久 host-visible slab（大小=激活总字节，跨 step 复用），激活按 offset 写入/读出 → RAM = 激活实际体积，避免 128MB 块碎片膨胀。
- 接入：`text_train --activation-offload`（与 `--checkpoint-every` 互斥）、GUI/CLI。

### RAM 实测（batch 6）
- 早期（每 tensor 独立缓冲）：RAM **26GB**（128MB 块碎片膨胀）。
- slab 修复后：RAM ≈ 激活实际体积 **~7GB**。

### 验证
- `offload_test`：GPU 激活 → host → GPU 往返 **bit-exact**（非 batch + batch 路径）。
- `gpt_offload_test`：GPU 上 offload 与全存基线 forward+全部梯度 **bit-exact**。
- 全部 gradcheck/fusion 回归绿。

### 注意
- 需足够主机 RAM（激活随 batch 线性增长）与 PCIe 带宽；PCIe 3.0/x8 收益减半。
- CUDA 后端 offload 暂为 no-op（未接 pinned memory）。

## 显存实测诊断（doc_ids 场景）

**问题**：doc_ids 强制注意力走旧路径，物化 (BH×seq, seq)=384MB/块 注意力矩阵，且曾把 `attn_cache_`+`softmax.output_cache_` 存两份 → slab 实测 **22GB**（理论 ~11GB）。

**修复**：
1. **去重**：Softmax 单缓冲，旧路径只用 `softmax_.output_cache()` 一份 → slab 22→**16GB**。
2. **（M7，2026-08）`AttnBias` 组合偏置统一位置编码**：两趟式原语掩码契约升级为组合式描述子（causal + doc_ids 块对角 + ALiBi slopes，见 [09 §4](09-operator-fusion.md)），doc_ids / ALiBi 及其组合**不再回退旧路径**，全部在融合 kernel 内按 (b,h,i,j) 组合偏置。doc_ids 仅需 `O(batch·seq)` 小张量每步上传，不再物化 (BH·seq,seq) 注意力矩阵。因此 `--doc-aligned-windows`（丢弃跨文档窗口以退化为纯因果）已**删除**——所有窗口（含跨文档）均保留，块对角文档掩码在两趟式路径内生效。


