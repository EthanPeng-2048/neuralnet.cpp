# 14 — GPU `stable=f16` 训练 loss 回读冻结（2026-09-26 基准重跑发现）

> 状态：**未修复（Open）** · 发现日期：2026-09-26 · 环境：Windows + clang++ 22（Release `-O3`）+ NVIDIA CMP 40HX（Vulkan）· HEAD `4a12876`
> 发现途径：`docs/benchmarks/2026-09-25-vulkan-vs-cuda.md` 全量重跑（nn 侧）。本报告只记录事实与证据，不实施修复。
> 关联：issue #13 P0-②（MSVC f16 权重冻结，3c2b9e8 已修，**本缺陷与之不同**）；`docs/development/05-mixed-precision.md` §12.5（四字段全 f16 历史即不可训练）。

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 症状 | GPU + `--precision-stable f16` 时，`text_train` 打印的 loss **冻结为常数**（step 5–45 恒 `4.0137`），跨 epoch、跨数据集、跨进程逐位相同 |
| 触发条件 | **仅 `stable=f16`**（GPU）；与 `optimizer` 字段无关；**CPU 同配置健康**；推荐路径 `--f16`（stable=f32）**完全健康** |
| 权重是否冻结 | **没有**——同一次运行 epoch1-end vs epoch2-end 模型快照 63% 参数字节不同；冻结发生在 **loss 回读/报告链** |
| 训练质量 | **无法验证**（loss 打印不可信，无 ground truth）；库级测试全绿但**不覆盖该配置**（见 §3） |
| 与 bench 的关系 | §2「双方全 f16」主对比的 nn 侧 loss/稳定性数据失效（**计时数据仍有效**）；bench 改为 f32 优先，见基准文档 |

## 1. 复现

```powershell
# 冻结（49 步内 step 5–45 全部打印 4.0137，avg_loss 跨 epoch 逐位相同）
$env:NN_VULKAN_DEVICE='40HX'
.\build\text_train.exe datasets\tinystories_bench40.txt --vocab datasets\bench_bpe_8192.json `
  --d-model 64 --num-heads 4 --num-layers 4 --d-ff 256 --seq-len 256 --batch-size 64 `
  --epochs 1 --optimizer adam --lr 0.001 --gpu=40HX --save-interval 0 --log-interval 5 `
  --precision-param f16 --precision-compute f16 --precision-stable f16 --precision-optimizer f16

# 健康（同命令去掉 --precision-stable f16，或直接用推荐路径 --f16）
```

- 完整定性矩阵日志：`bench/raw/f16_stable_matrix.log`
- 冻结值 `4.0137` 出现在：短测（bench40）/长测（bench40K，521 步/epoch）/多次独立进程 —— **与权重、数据、batch 位置均无关**。

## 2. 症状与定性

### 2.1 loss 打印冻结（表：1 epoch / 49 步，逐步日志节选）

| 配置 | step 5 | step 25 | step 45 | 末步(49) | avg_loss | 判定 |
|---|---|---|---|---|---|---|
| b) `--f16`（param/compute=f16, stable/optimizer=**f32**） | 8.7040 | 7.2302 | 6.3470 | 6.2036 | 7.3827 | ✅ 健康下降 |
| c) stable=f32 + optimizer=**f16** | 1.84e6 | 1.14e7 | 1.72e7 | 1.80e7 | 1.05e7 | 💥 爆炸（= §12.5 已知数值限制，**非本回归**） |
| d) stable=**f16** + optimizer=f32 | 4.0137 | 4.0137 | 4.0137 | 7.4591 | 4.0422 | ❄️ **冻结** |
| e) 四字段全 f16（bench 原配置） | 4.0137 | 4.0137 | 4.0137 | 10.2751 | 4.0649 | ❄️ **冻结** |

观察：
1. **触发器 = `stable=f16`**（d、e 冻结；b 健康；c 的爆炸是 optimizer=f16 的已知问题，与冻结无关）。
2. `4.0137` 在**所有** stable=f16 运行中逐位相同（不同数据集、不同进程、不同随机初始化）→ 不是任何数据/权重的函数，指向**陈旧/未写入的缓冲区读数**或固定垃圾值。
3. 末步值行为分裂：e 配置末步 `10.2751` 跨 ≥3 个独立进程完全相同；d 配置末步随进程微变（7.4541 / 7.4591）。
4. 5 epoch 长测中 avg_loss 每个 epoch **逐位相同**（4.0182 ×5，`bench/raw/nn_train.log`）。

### 2.2 权重没有冻结（同进程内快照对比）

方法：单次 e 配置运行（2 epoch，`--save-interval 49 --save <path>`），在 epoch1 保存之后、epoch2 保存之前从外部复制文件，再与最终文件逐字节比对（避开「随机初始化不可跨进程比对」的问题）。

```
bench/raw/f16_ep1.bin  (2,971,754 B)  ← epoch1 结束时
bench/raw/f16_ep2.bin  (2,971,754 B)  ← epoch2 结束时
差异字节 = 1,873,255 (63.04%)，首个差异 offset=4652，末个=2552254
（offset 4652 起为参数区；2552255 之后的 tokenizer JSON 段零差异）
```

⇒ **优化器在更新权重**；冻结仅存在于 loss 的产生/回读/打印链。但权重更新是否数值健康**无法从外部验证**（见 §2.4、§7）。

### 2.3 CPU 对照：健康

`NN_F16_DEBUG=1 build\test\f16_precision_test.exe` 的 profile 单字段矩阵（CPU）：

```
[dbg f32        ] 3.4776 → 3.1803
[dbg param=f16  ] 3.4776 → 3.1803
[dbg compute=f16] 3.4776 → 3.1803
[dbg stable=f16 ] 3.4785 → 3.1797   ← CPU 上 stable=f16 健康
[dbg opt=f16    ] 3.4776 → 1.02e6   ← 已知数值限制
[dbg f16(CLI)   ] 3.4776 → 3.1803
```

⇒ 冻结是 **GPU 特有**（GPU 分支不跑该矩阵，见 §3）。

### 2.4 权重更新健康度：未知

- `stable=f16` 下 loss 不可信，且没有独立的 ground-truth 对拍工具覆盖该配置；
- 库级轨迹对拍（`f16_precision_test` GPU 段）只用 `profile_f16`（stable=f32），对拍通过（max_rel=6e-4）**不能外推**到 stable=f16；
- 因此报告结论止步于「权重有更新」，**不断言更新数值正确**。

## 3. 为什么 ctest 19/19 全绿与本缺陷并存（覆盖缺口）

| 工具 | 结果 | 为什么不覆盖 |
|---|---|---|
| `f16_writeback_probe --gpu` | ALL PASSED | 参数移动用例用 `profile_f16()`（stable=f32） |
| `f16_precision_test`（GPU 轨迹对拍） | ALL PASSED（max_rel 6e-4） | GPU 段只跑 `run_suite`；对拍 profile = `profile_f16()`（stable=f32） |
| `ctest`（19/19，34.2 s） | 全绿 | 见下行 |
| `f16_precision_test` 的 profile 单字段矩阵（`f16_precision_test.cpp:639-646`） | 覆盖 stable=f16，但仅含 `{F32,F32,F16,F32}` 单字段形态 | **① 只在 CPU 跑**（`main()` CPU 块，`f16_precision_test.cpp:691-700`）；**② 藏在 `NN_F16_DEBUG` 环境变量之后**，默认 ctest 不执行；**③ 无 param/compute=f16+stable=f16 组合形态** |

⇒ 缺口一句话：**「GPU + stable=f16（含全 f16）」没有任何自动化用例**——测试全绿与 bench 冻结因此可以并存。

## 4. 与已修缺陷的区分

- 与 **issue #13 P0-②**（MSVC f16 权重冻结，`3c2b9e8` 已修）**不是同一问题**：

| | P0-②（3c2b9e8 已修） | 本缺陷 |
|---|---|---|
| 编译器 | MSVC（clang 正常） | **clang（本机 clang++ 22）** |
| 精度字段 | param=f16 即触发 | 仅 stable=f16 触发 |
| 冻结对象 | **权重冻结**（loss 恒 = ln V） | **loss 回读冻结**（权重正常更新） |
| 工具现状 | `f16_writeback_probe` ALL PASSED | 同工具无法覆盖本配置 |

- 回归窗口比对与嫌疑提交排查（09-24 行为对照、`git log 75255da..HEAD` 嫌疑清单）已移入 `docs/history.md`。

## 5. 链路与嫌疑点（未定根因）

`src/text_train.cpp` 每步 loss 链：

```
1514: auto loss_sum_t  = ce_loss.forward_sparse_sum(...)      // 设备端 loss 求和（stable=f16 时走 f16）
1523: auto loss_sum_f32 = engine->cast(*loss_sum_t, F32)      // cast 到 f32（回读接口仅支持 F32）
1572: engine->submit_scalar_readback(pl.slot, pl.keepalive)   // 排队异步回读（compute_gpu_engine.hpp:146）
1367: loss = -sum * pl.inv_num_valid                          // harvest 打印
```

- `submit_scalar_readback` 对非 F32 直接报错（"调用方先 cast"），运行中无该错误输出 → 提交成功；
- `inv_num_valid` 由 host 每步计算（batch 满时恒 16384），与冻结值组合后 loss 完全恒定 ⟺ **`loss_sum` 读回值恒定**；
- `4.0137` 与权重/数据无关 → 嫌疑集中在 `forward_sparse_sum` 的 f16 归约输出、`cast` 读陈旧内容、或异步回读槽位读到未写入数据；
- 末步（epoch 收尾 `drain_loss` 阻塞路径）行为与常规步不同（§2.1 观察 3），可作为二分切入点。

## 6. 影响面

1. **`docs/benchmarks/2026-09-25-vulkan-vs-cuda.md` §2**：「双方全 f16」主对比中 nn 侧的 loss/稳定性数据不可信（**wall-clock 计时与显存采样仍有效**）；已改 f32 优先。
2. 所有 `--precision-stable f16`（GPU）用户：训练监控失效，**无告警**。
3. `--f16` 推荐路径（param/compute=f16, stable/optimizer=f32）不受影响，实测健康。
4. 四字段全 f16 本就是实验性配置（`text_train.cpp:583` 注释「想要四字段全 f16（实验性）：--f16 --precision-stable f16」），但 CLI 允许无告警组合出坏状态。

## 7. 建议下一步（只记录，不实施）

1. **最小复现收窄**：绕过 text_train，直接用小张量 + `PrecisionEngine(GpuEngine)` 复现 `forward_sparse_sum`（stable=f16）输出恒定——把嫌疑从 CLI 压到单个原语。
2. **补 GPU profile 矩阵用例**：把 `debug_profile_matrix` 的 stable=f16 形态搬到 GPU 分支并加入默认 ctest（或至少 `{F16,F16,F16,F16}` 端到端轨迹对拍），关闭 §3 的覆盖缺口。
3. **回读二分**：在 `harvest_loss` 打印 `sum` 原始值 + 槽位号，验证「提交前 vs 读回」哪一侧为常数（`NN_F16_DEBUG` 风格开关，默认零开销）。
4. 若确认权重更新健康，考虑给 `--precision-stable f16` 组合加启动期告警（与 `--activation-offload` 的 CPU 警告同风格）。

## 8. 证据索引

| 文件 | 内容 |
|---|---|
| `bench/raw/f16_stable_matrix.log` | 4 组定性矩阵逐步日志（b/c/d/e） |
| `bench/raw/f16_ep1.bin` / `f16_ep2.bin` | 同进程跨 epoch 模型快照（哈希 `64DB920E…` vs `5E5A114E…`，63% 字节不同） |
| `bench/raw/f16_probe_gpu.log` | `f16_writeback_probe --gpu` 输出（ALL PASSED） |
| `bench/raw/f16_precision_gpu.log` | `f16_precision_test` 输出（ALL PASSED） |
| `bench/raw/nn_short.log` / `nn_train.log` | e 配置短测/长测原始训练日志（冻结） |
| `bench/raw/nn_short_f32.log` / `nn_train_f32.log` | f32 对照（健康收敛） |
| `bench/raw/layers_nn.log` / `ops_nn.log` | 本次重跑的层/算子基准（f32，与本缺陷无关） |
