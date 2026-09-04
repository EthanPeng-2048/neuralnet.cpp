# g13 训练 CLI 代码审查

## 模块概览

g13 组负责训练/推理 CLI 入口，共 6 个文件：

| 文件 | 行数 | 职责 |
|------|------|------|
| `src/text_train.cpp` | 1748 | GPT 文本生成训练全流程：BPE→训练→保存，checkpoint 保存/恢复，TDR 处理，断点续训 |
| `src/mnist_train.cpp` | 784 | MNIST 训练 CLI（MLP/Transformer/CNN 三架构） |
| `src/mnist_infer.cpp` | 353 | MNIST 推理 CLI |
| `include/neuralnet.cpp/cli/cli_train_common.hpp` | 168 | 公共训练参数解析（epochs/lr/batch-size/optimizer 等） |
| `include/neuralnet.cpp/cli/cli_mnist_io.hpp` | 240 | MNIST 数据加载 + 评估 |
| `include/neuralnet.cpp/cli/cli_lr_scheduler.hpp` | 120 | 学习率调度（epoch 级/step 级 cosine + warmup） |

架构关系：
- `mnist_train` 委托 `cli_train_common` 解析通用参数，`text_train` 自行解析全部参数（代码重复）
- 两者都使用 `cli_lr_scheduler` 的 `compute_epoch_lr` / `compute_step_lr`
- MNIST 数据 I/O 通过 `cli_mnist_io` 共享

## 发现

### P1

**P1-1 `src/text_train.cpp:750-761` — 死代码 `one_hot_labels` 函数违反铁律 9 维护精神**

```cpp
nn::Matrix one_hot_labels(const std::vector<std::size_t> &tokens, std::size_t vocab_size)
{
    const std::size_t n = tokens.size();
    nn::Matrix result(vocab_size, n);
    result.zero();
    for (std::size_t i = 0; i < n; ++i)
    {
        if (tokens[i] < vocab_size)
            result.set_value_unchecked(tokens[i], i, 1.0);
    }
    return result;
}
```

分析：该函数在整个项目中无任何调用点（grep 确认仅定义处命中）。它是铁律 9（大词表禁 one-hot）实施前的遗留代码。虽然当前不被调用，但作为死代码留在训练入口文件中，可能误导后续开发者在需要 one-hot 时直接调用它，绕过 `forward_sparse` 稀疏路径。

修复建议：删除该函数。

---

**P1-2 `src/text_train.cpp:1574-1578` — 优化器 end_batch 缺失 VK_ERROR_DEVICE_LOST 处理**

```cpp
// L1574-1578
auto end_r = engine->end_batch();
if (!end_r) {
    std::cerr << "end_batch (optimizer) failed: " << end_r.error().message << '\n';
    return 1;  // ← 直接退出，未检查 DEVICE_LOST
}
```

对比同文件其他 4 处 end_batch/flush_batch 调用（L1412-1421、L1440-1448、L1461-1472、L1501-1512）均检测了 `VK_ERROR_DEVICE_LOST` 并调用 `restart_on_device_lost`。唯独优化器 batch 的 end_batch 缺失此路径。

分析：`vkQueueSubmit`（end_batch 内部调用）可返回 `VK_ERROR_DEVICE_LOST`。若 GPU 在优化器 step 提交时丢失，程序直接退出而不保存 checkpoint、不重启，丢失自上次 checkpoint 以来的所有训练进度。

修复建议：在 L1574-1578 增加与 L1501-1512 相同的 DEVICE_LOST 检测与 restart 逻辑。

---

**P1-3 `src/text_train.cpp:504-556` — 关键模型超参数未校验 > 0**

```cpp
// L504-508: --seq-len
auto v = nn::parse_number<std::size_t>(argv[++i]);
if (!v) { std::cerr << "无效 --seq-len: " << v.error().message << "\n"; std::exit(1); }
cfg.seq_len = *v;  // ← 可为 0

// L534-538: --d-model
auto v = nn::parse_number<std::size_t>(argv[++i]);
if (!v) { std::cerr << "无效 --d-model: " << v.error().message << "\n"; std::exit(1); }
cfg.d_model = *v;  // ← 可为 0

// 同样 --num-heads (L540), --num-layers (L546), --d-ff (L552), --stride (L510)
```

分析：`std::size_t` 可接受 0。`seq_len=0` 会导致滑动窗口为空、除零、空张量崩溃；`d_model=0` 或 `num_heads=0` 会导致注意力维度为零。对比 `--batch-size`（L494 检查 `*v == 0`）和 `--accum-steps`（L501 检查 `*v == 0`）已有校验，此处不一致。

对比 `mnist_train.cpp`：`--patch-size`（L253 检查 `*v == 0 || nn::MNIST_IMG_SIZE % *v != 0`）已校验，说明项目有此惯例。

修复建议：为 `--seq-len`、`--d-model`、`--num-heads`、`--num-layers`、`--d-ff` 增加 `*v == 0` 检查。

---

**P1-4 `src/text_train.cpp:249-294` — `restart_on_device_lost` 失败时 exit(0)**

```cpp
[[noreturn]] void restart_on_device_lost(...)
{
    // L264-268: 保存 checkpoint（失败仅打印，不退出）
    auto save_r = nn::save_model(save_path, model, spec, tokenizer_json);
    if (save_r)
        std::cerr << "  [TDR] 模型已保存到: " << save_path << "\n";
    else
        std::cerr << "  [TDR] 保存失败: " << save_r.error().message << "\n";
    // ...
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Command failed with code " << ret << "\n";
    }
    std::exit(0);  // ← 无论保存是否成功、重启是否成功，均 exit(0)
}
```

分析：`[[noreturn]]` 函数始终 `exit(0)`。两个问题：
1. checkpoint 保存失败后仍尝试重启，重启后的 `--resume` 会因文件不存在/损坏而失败，形成死循环（每次 TDR 都重启→失败→TDR→重启…）
2. `std::system` 返回非零（命令执行失败）后仍 `exit(0)`，监控脚本/批调度系统无法检测失败

修复建议：save 失败时 `exit(1)` 不重启；system 返回非零时 `exit(1)` 或至少 `exit(ret)`。

---

**P1-5 checkpoint 不保存优化器状态（Adam momentum/variance 丢失）**

text_train.cpp 和 mnist_train.cpp 的 checkpoint 保存均只调用 `nn::save_model(save_path, model, spec, ...)`，不保存优化器的 momentum/state。

分析：TDR 重启后，Adam 的一阶矩 `m` 和二阶矩 `v` 被重置为零。训练恢复后前几步的梯度更新会因缺少动量历史而震荡，可能需要数百步才能恢复到中断前的训练轨迹。对于大模型训练（GPT），这是可观的训练质量损失。

修复建议：扩展 save/load 接口以持久化优化器状态，或在 TDR 重启时接受短暂的训练质量下降（当前行为的文档化折中）。

---

**P1-6 `src/mnist_train.cpp:720-730` — 优化器 step/zero_grad 未包裹在 begin_batch/end_batch 中**

```cpp
// L716: end_batch 提交 forward+loss+backward
auto eb = engine->end_batch();
if (!eb) { ... return 1; }

// L720-730: 优化器操作在 batch 外
auto step_result = optimizer->step();      // ← 使用 engine 原语（axpy_inplace 等）
if (!step_result) { ... return 1; }
auto zero_result = optimizer->zero_grad(); // ← 使用 engine 原语（zero）
if (!zero_result) { ... return 1; }
```

对比 `text_train.cpp` L1547-1578：优化器 step/zero_grad 包裹在独立的 `begin_batch/end_batch` 对中。

分析：`optimizer->step()` 内部调用 `engine_.axpy_inplace()` 等 GPU 原语（`compute_optimizer.hpp:161`），`optimizer->zero_grad()` 调用 `engine_.zero()`（`compute_optimizer.hpp:150`）。在 GPU 引擎中，这些操作需要在 batch 录制期内执行才能批量提交。MNIST 训练在 GPU 模式下，优化器操作会逐个提交（无 batching），造成额外的 `vkWaitForFences` 同步开销。

修复建议：参照 text_train 的模式，为 MNIST 优化器操作包裹独立的 `begin_batch/end_batch` 对。

---

### P2

**P2-1 `src/text_train.cpp:446-747` — 未使用 `cli_train_common.hpp`，参数解析完全重复**

text_train.cpp 自行解析所有通用训练参数（epochs/lr/batch-size/optimizer/weight-decay/min-lr/warmup-epochs/lr-per-epoch/lr-schedule/gpu/cuda），与 `cli_train_common.hpp` 提供的 `parse_train_common_args` 功能完全重叠。

对比 `mnist_train.cpp` L144-162：正确委托给 `nn::cli::parse_train_common_args`。

分析：代码重复导致两处校验逻辑可能不一致（如 text_train 的 `--lr-schedule` 接受 `step_cosine` 但 `cli_train_common.hpp` 不接受）。维护时容易遗漏一侧的修改。

修复建议：重构 text_train 的参数解析，通用参数委托给 `cli_train_common.hpp`，仅保留 text_train 专有参数（--seq-len, --stride, --model, --tdr-retry 等）。

---

**P2-2 `src/mnist_train.cpp` — 无周期性 checkpoint 保存**

mnist_train.cpp 仅在训练结束时保存模型（L775）。text_train.cpp 有 `--save-interval` 周期保存（L1583-1590）+ TDR 自动保存。

分析：MNIST 训练通常较短（几分钟），此问题影响有限。但若用于大型 CNN 或未来扩展，中断将丢失全部进度。

修复建议：可接受现状（P2 建议级别）。如需改进，添加 `--save-interval` 支持。

---

**P2-3 `src/text_train.cpp:1561-1564` — 梯度统计在优化器 batch 内调用 to_matrix，拆分提交**

```cpp
// L1547: begin_batch (optimizer batch)
// L1554: optimizer->step()
// L1561-1564:
if (cfg.grad_log && ...) {
    std::cout << "\n  [grad] step " << step + 1 << ":" << std::endl;
    log_gradient_stats(*engine, model.param_gradients());
    // log_gradient_stats 内部调用 engine.to_matrix() → end_batch() + begin_batch()
}
// L1567: zero_grad()
// L1574: end_batch()
```

分析：`log_gradient_stats` 内部对每个 GPU 梯度张量调用 `engine.to_matrix()`（L781），该函数内部会 `end_batch() + begin_batch()` 以执行同步下载。这会将优化器 batch 拆分为两次 GPU 提交：第一次提交 step 操作，第二次提交 zero_grad。额外的 GPU 提交增加同步开销。

修复建议：将梯度统计移到 `end_batch()` 之后（此时 batch 已提交，to_matrix 不会拆分 batch），或仅在 CPU 模式下启用梯度统计。

---

**P2-4 `include/neuralnet.cpp/cli/cli_train_common.hpp:121-127` — `--warmup-epochs` 接受负值**

```cpp
if (arg == "--warmup-epochs" && i + 1 < argc)
{
    auto v = nn::parse_number<int>(argv[++i]);
    if (!v) { std::cerr << "无效 --warmup-epochs: " << v.error().message << "\n"; std::exit(1); }
    cfg.warmup_epochs = *v;  // ← 可为负数
    return true;
}
```

分析：`warmup_epochs` 为 `int` 类型，`parse_number<int>` 不拒绝负值。负值在 cosine 调度中效果等同于 0（`epoch < warmup_epochs` 永假），不会崩溃，但用户意图不明确。`--epochs` 已有 `*v <= 0` 校验（L75），`--warmup-epochs` 应保持一致。

修复建议：增加 `*v < 0` 检查。

---

### P3

**P3-1 错误信息风格不一致**

各文件错误信息混用中英文：
- `mnist_train.cpp`: `"未知参数:"` / `"无效 --d-model:"` / `"层维度不能为 0"`
- `mnist_infer.cpp`: `"参数错误:"` / `"无法打开文件:"` / `"输入路径不存在:"`
- `text_train.cpp`: `"无效 --seq-len:"` / `"Error:"` / `"请指定训练文本文件"`
- `cli_train_common.hpp`: `"无效 --epochs:"` / `"--epochs 必须为正整数"`

分析：风格不一致影响用户体验和日志解析。项目整体倾向中文，但部分错误信息为英文。

修复建议：统一为中文（与多数 CLI 一致），或统一为英文（与 core_errors.hpp 一致）。

---

**P3-2 `src/mnist_infer.cpp:317-320` — 目录迭代未处理权限错误**

```cpp
for (auto &entry : fs::directory_iterator(input))
{
    if (entry.path().extension() == ".csv")
        csv_files.push_back(entry.path());
}
```

分析：`fs::directory_iterator` 在遇到权限错误时会抛异常（`fs::filesystem_error`），但本项目编译选项为 `-fno-exceptions`。在 `-fno-exceptions` 下，`directory_iterator` 的行为取决于实现（可能 `std::terminate`）。

修复建议：使用 `fs::directory_iterator(input, ec)` 的 error_code 重载。

---

## 已知项核对

### VK_TIMEOUT 重试现状

**状态：未实现。**

text_train.cpp 有完整的配置接口（`--tdr-retry on/off`，`--max-tdr-retries N`），`TrainConfig` 中定义了 `auto_tdr_retry`（L423）和 `max_tdr_retries`（L424），CLI 解析也正确（L694-711）。但在整个训练循环（L1254-1730）中，**没有任何代码读取或使用这两个配置值**。grep 确认 `auto_tdr_retry` 和 `max_tdr_retries` 仅在定义和解析处出现，训练逻辑中无引用。

对比文档描述（L356-357）：
```
--tdr-retry <on|off>  GPU 超时自动减小 batch 重试 (默认: on)
--max-tdr-retries <n> 最大重试次数，每次 batch 减半 (默认: 4)
```

用户看到此帮助信息会认为 TDR 重试已启用，但实际无效。这是**功能声明与实现不符**。

### DEVICE_LOST 路径

**状态：基本完善，有一处缺口。**

已覆盖的 DEVICE_LOST 检测点（4/5）：
| 位置 | 操作 | 检测 | restart |
|------|------|------|---------|
| L1412-1421 | end_batch (NaNSkip logits) | ✅ | ✅ |
| L1440-1448 | end_batch (NaNSkip loss) | ✅ | ✅ |
| L1461-1472 | flush_batch (forward+loss) | ✅ | ✅ |
| L1501-1512 | end_batch (backward) | ✅ | ✅ |

未覆盖（1 处）：
| 位置 | 操作 | 检测 | restart |
|------|------|------|---------|
| L1574-1578 | end_batch (optimizer) | ❌ | ❌ → 直接 return 1 |

`restart_on_device_lost` 函数本身（L249-294）功能完整：保存 checkpoint → 等待 5 秒 → 构造 `--resume --resume-epoch --resume-step --flush-interval` 命令 → `std::system` 重启。但 exit code 始终为 0（P1-4）。

### 已修 P1 校验项（确认仍生效）

| 校验项 | 文件 | 行号 | 状态 |
|--------|------|------|------|
| batch-size ≥ 1 | cli_train_common.hpp | L90 | ✅ 生效 |
| max-tokens ≥ 1 | text_train.cpp | L494 (`batch-size` ≥ 1) | ✅ 生效 |
| epochs > 0 | cli_train_common.hpp | L75 | ✅ 生效 |
| patch-size 整除 28 | mnist_train.cpp | L253 | ✅ 生效 |
| cnn-pool ≤ 28 | mnist_train.cpp | L305 | ✅ 生效 |
| accum-steps ≥ 1 | text_train.cpp | L501 | ✅ 生效 |
| epoch lr 钳制 | cli_lr_scheduler.hpp | L66-67 | ✅ 生效（progress 钳制到 [0,1]） |

## 待验证

1. **MNIST GPU 模式优化器正确性**：`mnist_train.cpp` 的 optimizer step/zero_grad 在 begin_batch/end_batch 外执行。需在 GPU 模式下验证优化器操作是否正常工作（CPU 模式下 begin_batch/end_batch 为 no-op，无影响）。
2. **`log_gradient_stats` 的 to_matrix 在 DEVICE_LOST 时的行为**：若 to_matrix 内部的 end_batch 触发 DEVICE_LOST，错误会被 `if (!m) continue` 静默忽略（L782），训练继续使用损坏的 GPU。需确认 to_matrix 的 DEVICE_LOST 是否会传播到调用方。
3. **text_train NaN skip 后梯度累积**：当 `accum_steps > 1` 且 NaN 发生在累积中间步时，跳过 backward 会导致该 epoch 的有效 batch 缺少该步的梯度贡献。影响程度取决于 NaN 频率（通常极低）。
