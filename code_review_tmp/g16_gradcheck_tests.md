# g16 gradcheck/一致性/基准测试代码审查

## 模块概览

本组 13 个测试文件覆盖四大类：

| 类别 | 文件 | 核心验证点 |
|------|------|-----------|
| 数值梯度（gradcheck） | attn_gradcheck / gpt_gradcheck / rapt_gradcheck / rmsnorm_gradcheck / softmax_gradcheck / swiglu_gradcheck | 中心差分 vs 解析梯度，容差/步长/覆盖 |
| 文档感知掩码 | doc_attn_test / doc_mask_test | set_doc_ids 块对角掩码正确性 |
| Activation offload | gpt_offload_test / offload_test | GPU→host slab 往返等价性 |
| 其他 | rapt_smoke_test / model_spec_validation_test / layer_bench | RAPT 端到端 / spec 校验 / 性能基准 |

所有 6 个 gradcheck 测试均采用相同的数值梯度框架：
- 中心差分 `(lp - lm) / (2 * eps)`，eps = 1e-3
- 逐元素全量遍历（非随机采样）
- 相对+绝对混合容差 `|num - ana| <= tol * (1 + |num| + |ana|)`
- `eval_loss()` 每次扰动调用完整 `forward()` 重建缓存

## 发现

### P0

无。

### P1

**P1-1 `softmax_gradcheck.cpp:132-135` — forward/backward/from_matrix 返回值未检查，失败即 UB**

```cpp
auto x = eng.from_matrix(x_m);      // line 132: 未检查
auto go = eng.from_matrix(go_m);    // line 133: 未检查
auto y_fwd = sm.forward(eng, *x);   // line 134: 未检查
auto gx = sm.backward(eng, *go);    // line 135: 未检查
```

若任一调用返回 `std::unexpected`（如 GPU 初始化失败、维度不匹配），后续解引用空 `expected` 导致 UB/crash。同文件 `eval_loss()`（line 37-42）也未检查 `forward()` 和 `to_matrix()` 返回值。其他 5 个 gradcheck 测试均有完整的 `if (!r) return 1` 错误检查。

**建议**：每行加 `if (!x) { ... return 1; }` 检查，与其余 5 个 gradcheck 保持一致。

---

**P1-2 `doc_attn_test.cpp:43` — batch=1 违反铁律 5，无法检测布局 bug**

```cpp
Matrix x_m(seq, 1);  // line 43: batch=1
```

`run_forward` 始终创建 batch=1 的输入。batch=1 时 position-major（`i = t*batch + b`）与 batch-major（`i = b*seq + t`）布局完全重合，历史上的跨样本串扰灾难级 bug 在此条件下不可复现。测试的不变性验证（doc A 变化不影响 doc B）在 batch=1 下仍然有效，但缺少 batch>1 覆盖。

**建议**：增加 batch=2+ 的测试段，或在现有测试中将 batch 参数化。doc_mask_test 作为纯 CPU 掩码数学测试不受此约束。

---

**P1-3 全部 gradcheck 默认容差过松：attn/gpt/rapt = 5e-2（5%），rmsnorm/swiglu = 2e-2**

| 测试 | 默认 tol | 文件:行 |
|------|---------|--------|
| attn_gradcheck | 5e-2 | attn_gradcheck.cpp:106 |
| gpt_gradcheck | 5e-2 | gpt_gradcheck.cpp:128 |
| rapt_gradcheck | 5e-2 | rapt_gradcheck.cpp:144 |
| rmsnorm_gradcheck | 2e-2 | rmsnorm_gradcheck.cpp:122 |
| swiglu_gradcheck | 2e-2 | swiglu_gradcheck.cpp:126 |
| softmax_gradcheck | 1e-2 | softmax_gradcheck.cpp:91 |

`approx()` 公式 `|num - ana| <= tol * (1 + |num| + |ana|)` 本身正确，但 tol=5e-2 意味着 gradient 值在 1.0 量级时允许 5% 误差。这会漏掉 scaling 类 bug（如已修 P0-1 grad_gamma 多乘 γ 的早期版本若误差在 3-4% 量级）。softmax_gradcheck 的 1e-2 是合理基准。

**建议**：将 attn/gpt/rapt 默认 tol 降至 1e-2 或 2e-2，与 softmax/rmsnorm 对齐。若 ReLU 拐点导致特定元素无法收敛到更紧容差，由 rapt_gradcheck 的 kink skip 机制处理。

### P2

**P2-1 `rapt_gradcheck.cpp:202` — bidirectional 段 batch=1，该路径未覆盖铁律 5**

```cpp
const std::size_t d_model = 8, heads = 2, seq = 4, batch = 1;  // line 202
```

causal 段（line 165, batch=2）和 doc-aware 段（line 240, batch=2）覆盖了 batch>1，但 bidirectional 路径（非因果双向注意力）仅在 batch=1 下测试。bidirectional 路径的 rearrange_3d / batched_matmul 维度组合与 causal 不同，batch=1 无法暴露布局问题。

**建议**：将 bidirectional 段 batch 提升至 2。

---

**P2-2 全部 gradcheck 测试维度均为 4 对齐（8/16/32），缺少非对齐 shape 覆盖**

| 测试 | d_model/rows | 其他维度 |
|------|-------------|---------|
| attn_gradcheck | 16 | seq=8, batch=2 |
| gpt_gradcheck | 16 | seq=8, batch=2, d_ff=32 |
| rapt_gradcheck | 8 | seq=4, heads=2 (d_k=4) |
| rmsnorm_gradcheck | 16 | batch=4 |
| softmax_gradcheck | rows=8, cols=8 | — |
| swiglu_gradcheck | 8 | d_ff=16, batch=4 |

BLOCK_SIZE=64 的 matmul 分块、vec4 SIMD 路径在非 4 对齐维度下的尾链处理是常见 bug 源。当前测试无法覆盖这些边界。

**建议**：至少在 softmax_gradcheck 和 rmsnorm_gradcheck 中增加非 4 对齐维度（如 features=13, rows=7）的测试段。

---

**P2-3 `layer_bench.cpp:374-382` — 仅报告 best 时间，无方差/标准差统计**

```cpp
double best = 1e30;
for (int i = 0; i < cfg.iter; ++i)
{
    auto t0 = std::chrono::steady_clock::now();
    spec.run(engine, cfg, ctx);
    best = std::min(best, ms_since(t0));
}
```

基准测试仅取最小值，无法评估测量噪声。对于 CI 回归检测，best 时间过于乐观且不稳定（依赖系统负载瞬时状态）。

**建议**：增加中位数 + stddev 或至少报告 min/median/max 三值。可通过收集所有迭代时间到 vector 后排序计算。

---

**P2-4 `rapt_smoke_test.cpp` — dk≤64 / d_k 偶数约束仅由 NN_ASSERT 保护，测试使用合规值不覆盖约束失效路径**

```cpp
cfg.d_model = 16;   // d_k = d_model/heads = 8（偶数，RoPE 约束）
```

已知 P2-3：这些约束在 `-DNDEBUG` 下消失。测试始终使用合规参数（d_k=8, dk≤64），不验证约束违反时的行为。虽然在 `-DNDEBUG` 下约束消失属于被测代码问题，但测试至少应验证 `NN_ASSERT` 在 debug 模式下能正确拦截。

---

**P2-5 `softmax_gradcheck.cpp` — 仅测试输入梯度，Softmax 无参数梯度可理解，但未在注释中说明**

Softmax 无 trainable 参数，测试只验证 `grad_x`。这不是 bug，但缺少注释说明为何不测参数梯度，可能让审查者困惑。

### P3

**P3-1 `model_spec_validation_test.cpp:101,124,152` — 临时文件使用相对路径**

```cpp
const std::string file = "arch_check_matching.bin";  // line 101
```

三个测试段分别创建 `arch_check_matching.bin`、`arch_check_mismatch.bin`、`arch_check_nospec.bin`。若 ctest 工作目录与源码目录不同，文件可能残留或冲突。`std::remove` 清理（line 117/145/169）依赖成功路径。

**建议**：使用 `std::filesystem::temp_directory_path()` 或 `std::tmpnam` 生成唯一路径。

---

**P3-2 `doc_mask_test.cpp:21` — 匿名命名空间中的全局 `int failures` 计数器**

```cpp
namespace {
int failures = 0;  // line 21
```

虽在匿名命名空间内不泄漏，但全局可变状态不符合现代 C++ 测试实践。若将来与其他测试合并编译可能冲突。

---

**P3-3 `offload_test.cpp:91` — 容差硬编码 `1e-6f`，不可配置**

```cpp
const bool ok = err < 1e-6f;  // line 91
```

对于 GPU 往返精度，1e-6 是合理默认值，但不可通过 CLI 调整。若 GPU 精度降级或浮点舍入变化，需要改代码才能调整阈值。

---

**P3-4 `gpt_offload_test.cpp` — GPU-only 测试无 warmup，首次 forward 包含初始化开销**

gpt_offload_test 直接 forward+backward 对比基线，无 warmup pass。虽然对比的是两次独立运行的绝对值（非计时），但 GPU 首次操作可能触发 lazy 初始化，影响数值精度。当前未因此失败，但属于潜在不稳定性来源。

---

## 坑 8 合规矩阵（6 个 gradcheck 测试 × 是否先 forward 再 backward）

| 测试文件 | 初始 forward 行号 | backward 行号 | eval_loss 内 forward | 合规 |
|---------|-------------------|---------------|---------------------|------|
| attn_gradcheck.cpp | 168 | 178 | line 44（每次扰动调用） | ✅ |
| gpt_gradcheck.cpp | 211 | 229 | line 55（每次扰动调用） | ✅ |
| rapt_gradcheck.cpp | 190 (causal) / 227 (bidir) / 272 (doc-aware) | 192 / 229 / 274 | lambda 内 line 183/220/265 | ✅ |
| rmsnorm_gradcheck.cpp | 175 | 186 | line 48（每次扰动调用） | ✅ |
| softmax_gradcheck.cpp | 134 | 135 | line 38（每次扰动调用） | ✅ |
| swiglu_gradcheck.cpp | 181 | 192 | line 52（每次扰动调用） | ✅ |

**结论**：全部 6 个 gradcheck 测试均在 backward 前调用了 forward 填充 input_cache_/softmax_cache_ 等内部缓存，坑 8 合规。数值梯度路径（eval_loss）每次扰动独立调用 forward 重建缓存，也合规。

## 疑似被测问题

**SP-1 `compute_layer_attention.hpp:587` — 旧物化路径仍存活**

`compute_layer_attention.hpp:587` 注释标注 "旧路径：物化得分矩阵（ALiBi/doc_ids 等共享掩码不适用时回退）"。虽然 S7 后 two_pass_mask_ 统一了所有掩码路径（line 1099-1151），doc_ids 走 two_pass（`use_two_pass=true`），但旧路径代码仍在 else 分支中。doc_attn_test 和 rapt_gradcheck 的 doc-aware 段已覆盖活跃的 doc_ids 路径（two_pass），但旧路径是否可达需确认——若不可达则为死代码，应清理。

---

**SP-2 `rapt_smoke_test.cpp:105-106` — loss 有效性检查仅验证非 NaN，未验证 loss > 0**

```cpp
if (!(first_loss > 0) || !(first_loss == first_loss) || !(last_loss == last_loss))
```

`first_loss > 0` 覆盖了负 loss（cross-entropy 不应为负），`== self` 覆盖 NaN。但对于 RAPT（非 softmax 输出），loss 是否应始终 > 0 取决于 loss 函数实现。当前检查合理但不够严格——未验证 loss 下降（`first_loss > last_loss`），仅依赖人工观察输出。
