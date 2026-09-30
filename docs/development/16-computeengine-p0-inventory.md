# ComputeEngine Refresh P0 盘点报告（2026-09-28）

> 配套文档：方案见 `docs/development/15-computeengine-refresh.md`（分期 P0-P6；
> 2026-09-30 起未实施的 P2-P6 由 `17-unified-tensor-engine.md` §5 改期 M1-M7），
> 选型出处 `docs/development/13-refactor-backlog.md` §10。
> **本文是 P0（0 代码改动调研）的实测结果**；未决点与失败项只记录、不修复。
> 除新增探针 `src/gpu_stability_probe.cpp`（P0 工具，app 目标不注册 ctest）外，
> 未改动任何库代码。

## 0. 结论速览

| P0 项 | 状态 | 结果 |
|---|---|---|
| 43 处 `ensure_gpu` 分类 | ✅ | 23 普通上传 / 6 调用方句柄重绑定 / 14 Result 传播 |
| 175 处 `ComputeEngine&` 打标 | ✅ | KEEP 58 / DROP 110 / REVIEW 7 |
| 宿主中转调用盘点 | ✅ | 引擎外 381 处（include 71 + src 306 + 探针 4）；13 §10 的 596 为宽口径，见 §4 |
| CPU 字节基线可行性 | ✅ | 探针实测：进程内两轮 + 跨进程两次启动，loss 序列与参数校验和**逐位一致** |
| GPU run-to-run 稳定性（D9 档位） | ✅（2026-09-30 补测） | 崩溃根因 = 探针漏 `backend.initialize()`（§7-2，M3 期间定位并修复）；修复后 Windows 本机 dev0/dev2/dev4 进程内两轮 + 跨进程两次启动 loss 与 hash **逐字节一致**（§6） |
| ctest 基线 | ✅（带发现） | 默认（自动选卡=Mali）18/20；`NN_VULKAN_DEVICE=1`（Lavapipe）20/20；Mali offload 失败**非确定**（§5） |

## 1. 环境（本机实测基线，2026-09-28）

- Clang 21.1.8 / CMake（Makefile 生成器）/ RelWithDebInfo / `-march=native=ON`。
- Vulkan 实例 1.4.363，两设备：
  - **GPU0**：vendorID `0x13b5`(ARM/Mali)，apiVersion **1.1.177**
  - **GPU1**：vendorID `0x10005`(Mesa/Lavapipe)，apiVersion **1.4.354**
- 自动选卡按（设备类型权重, apiVersion）打分 → **本机落到 GPU0（Mali）**。

复现：`vulkaninfo --summary`；选卡逻辑 `backend/compute_vk_device.hpp:188-258`。

## 2. `ensure_gpu` 43 处分类（P3 → `import` 的迁移清单）

复现：`grep -n "= ensure_gpu(" include/neuralnet.cpp/compute_gpu_engine.hpp`

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
# 跨进程比对：两次启动后 grep -E '^(CONFIG|run1)' | diff -
```

| 设备 | 进程内两轮 | 跨进程两次启动 | 结论 |
|---|---|---|---|
| **CPU** | ✅ loss 序列 + 参数 hash 逐位一致（exit=0，两轮独立进程均 PASS） | ✅ 逐位一致 | **CPU 逐字节基线可行**——15 §4.9 第 1 行坐实 |
| GPU（dev1 / dev0） | ⏸ 未测得 | ⏸ | **探针崩溃**（§7-2；2026-09-30 已修复，见下一行） |
| GPU（2026-09-30 Windows 本机，修复后） | ✅ dev2 `8ef51b2927253c50`、dev0 `47977edc71a21a20`、dev4 `01b7192bd15a59d9`（各两轮 PASS） | ✅ 三设备 `CONFIG\|run1` 行逐字节一致 | **GPU 逐字节档位在本机三设备坐实**；dev1（MESA/D3D12 转译）初始化即报 DXIL container 校验错误、exit=2——设备级不可用，非探针/库问题 |

## 7. 未决问题（只记录，均不阻塞 P-1/P1 立项）

1. **Mali（GPU0）offload 非确定失败**（§5）——设备级，另立 issue。
2. ~~**探针 GPU 首 buffer 即崩**~~ **已定位并修复（2026-09-30，M3 期间）**：根因
   = `GpuBackend::instance()` 只是惰性单例，Vulkan 设备须显式 `backend.initialize()`
   才建立——探针 `make_engine` 漏调（`cli_engine_factory` 与全部 GPU 测试入口都会
   调它，这正是当时"差异在探针路径"的那一处），首个 `vkCreateBuffer` 拿到
   VK_NULL_HANDLE 即 "Invalid device"。修复 = 探针建引擎前补 `initialize()`（空
   selector 仍按 显式 > `NN_VULKAN_DEVICE` 环境变量 > 自动打分 选卡），commit
   `4977a29`；§6 GPU 档位已回填（本机 dev0/dev2/dev4 逐字节一致；dev1 为
   MESA/D3D12 设备级 DXIL 校验失败、exit=2，另记）。
3. **REVIEW 7 处**（§3）：P2 立项时逐条判定。
4. **13 §10 数字口径**（§4）：596 → 381，P2 改名清单按 381。
5. **`Model::set_engine` 死码**（15 D7）：维持"删除"倾向，P2 定。

## 8. P0 产出清单

- 静态盘点 §2-§4（P1/P2/P3 工作量与迁移清单齐了）
- CPU 字节基线工具与实测（§6）
- ctest 双基线（§5：dev1 20/20 = Refresh 基线；dev0 记录在案）
- 新增 `src/gpu_stability_probe.cpp` + CMake app 目标（ctest 仍为 20，数量未变）

**P0 收尾（2026-09-30）**：§7-2 已修复（探针补 `backend.initialize()`，commit
`4977a29`），§6 已回填本机三设备实测——D9 的 GPU 档位在本机坐实为**逐字节**；
原机器（Mali）档位受 §5 Mali offload 非确定限制，维持"设备级另立 issue"。
