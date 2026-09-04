# g17 ZiPT/一致性测试代码审查

## 模块概览

| 文件 | 测试目标 | 两条路径 | batch>1 |
|------|---------|---------|---------|
| `zipt_consistency_test.cpp` | ZiPT 全模型 CPU vs GPU 一致性 | CPU engine vs GPU engine（同构权重） | ❌ batch=1 |
| `zipt_doc_test.cpp` | ZiPTBlock 文档感知掩码不变性 | 有 doc_ids vs 无 doc_ids（负对照） | ❌ batch=1 |
| `zipt_gradcheck.cpp` | ZiPT 梯度数值检查 | CrossAttention + ZiPTBlock 各自 numerical vs analytical | ✅ batch=2（硬编码） |
| `zipt_smoke_test.cpp` | ZiPT 端到端冒烟 | 正常模式 / 压缩模式(L=W+C) / batch>1 | ✅ batch=2 |
| `tokenizer_consistency_test.cpp` | BPE pre_tokenize 对拍 | 手写状态机 vs std::regex 旧实现 | N/A |
| `attn_consistency_test.cpp` | Attention 两趟式一致性 | forward(整段) vs forward_step(KV cache 增量) | ❌ batch=1 |

## 发现

### P1

**P1-1 `tokenizer_consistency_test.cpp:221-236` — 缺失确定性回归保护（铁律 8）**

证据：`test_marker_roundtrip()` 只做了 `decode(encode(text)) == text`（往返一致性），但从未对同一 `text` 调用两次 `tok.encode(text)` 并断言两次输出字节级完全相同。铁律 8 要求"任何依赖容器迭代顺序的决策点（BPE 平局打破、ID 分配）必须显式排序"，确定性回归的核心保护——多次 encode 同输入结果一致——在此测试中完全缺失。

建议：在 `test_marker_roundtrip()` 末尾（或新建 `test_determinism()`）添加：
```cpp
const auto ids1 = tok.encode(text);
const auto ids2 = tok.encode(text);
CHECK(ids1 == ids2, "encode deterministic: two calls produce identical ids");
// 逐字节比较
for (std::size_t i = 0; i < ids1.size(); ++i)
    CHECK(ids1[i] == ids2[i], "byte-level id match");
```

---

**P1-2 全组缺失 BPE vocab=258 回归测试用例**

证据：项目 AGENTS.md §12 记录"BPE 曾有 vocab<258 崩溃（已修 P1）"。`tokenizer_consistency_test.cpp` 中唯一训练 BPE 的用例 `test_marker_roundtrip()` 使用 `vocab_size=320`（覆盖 258 边界但非精确边界）。无任何用例显式构造 `vocab_size=258` 并验证 encode/decode 不崩溃——这正是回归测试的定义。

建议：在 `test_marker_roundtrip()` 中（或新增 `test_vocab_boundary()`）添加：
```cpp
nn::BPETokenizer tok258;
nn::BPETokenizer::Config cfg258;
cfg258.vocab_size = 258;
cfg258.min_freq = 1;
auto tr258 = tok258.train("hello world test data for vocab 258 boundary", cfg258);
CHECK(tr258.has_value(), "vocab=258 train ok");
if (tr258) {
    auto ids = tok258.encode("hello");
    CHECK(!ids.empty(), "vocab=258 encode ok");
}
```

### P2

**P2-1 `zipt_doc_test.cpp` — 文档掩码测试缺失 batch>1（铁律 5）**

证据：`main()` (line 79-149) 始终以 `batch=1` 运行 ZiPTBlock forward（`kWindow=8, kMemory=2`）。铁律 5 明确要求"所有注意力/序列相关测试必须覆盖 batch>1（batch=1 时两种布局重合，测不出跨样本串扰）"。`build_mask_` 内部按 `b * window_ + jl` 索引 doc_ids（`compute_layer_zipt.hpp:288`），batch>1 时若索引逻辑有跨样本串扰 bug，batch=1 测试完全不可见。

建议：新增 batch=2 测试段，构造 `doc_ids` 长度 `batch*window`，验证两个 batch 的 doc B 输出互不干扰。

---

**P2-2 `attn_consistency_test.cpp` — forward vs forward_step 一致性缺失 batch>1 数据**

证据：`test_consistency()` (line 132) 始终构造 `Matrix tok_mat(seq_len, 1)`——batch 维度恒为 1。`use_batch` 参数（line 94）实际控制的是 GPU 命令录制（`begin_batch`/`end_batch`），不是数据 batch 维度。KV cache 增量推理与整段推理的一致性在 batch>1 时可能因 batch-major 布局差异而失效，当前测试无法捕获。

建议：在 `test_consistency()` 中增加 `batch_size` 参数，默认测试 batch=1 和 batch=2 两个配置。

---

**P2-3 `tokenizer_consistency_test.cpp` — 缺失 BPE vs CharBPE 一致性对拍**

证据：任务要求覆盖"BPE vs CharBPE 一致性"，但测试文件仅实现了 `BPETokenizer::pre_tokenize` 对拍 `std::regex` 旧实现（line 100-109）。无任何 CharBPE 相关测试代码。若 CharBPE 的 `pre_tokenize` 行为与 BPE 不一致（或共享实现中存在隐式分歧），当前测试无法发现。

建议：新增 `test_bpe_vs_charbpe_consistency()`，分别用 `BPETokenizer` 和 `CharBPETokenizer` 编码相同输入，验证往返一致性和确定性。

---

**P2-4 `zipt_smoke_test.cpp:193-249` — batch>1 冒烟缺少 loss 下降断言**

证据：batch>1 测试段（line 193-249）仅在 line 243-248 检查 `bfirst > 0 && bfirst == bfirst`（非 NaN），但不检查 `last_loss < first_loss`（loss 下降）。batch=1 段（line 113-118）和压缩模式段（line 183-188）均有 loss 下降断言。batch>1 的 loss 下降被遗漏，意味着 batch>1 路径中梯度可能无效（如 batch-major 布局错误导致梯度为零或噪声），但测试仍通过。

建议：在 line 242 后添加 `blasting` 追踪和 `last_loss` 断言，与 batch=1 段对齐。

---

**P2-5 `zipt_doc_test.cpp` — 文档切换位置未覆盖边界变体**

证据：所有测试用例固定 `doc_ids=[1,1,1,1,2,2,2,2]`（窗口正中切换，line 106）。未测试文档边界偏移场景（如 `[1,2,2,2,2,2,2,2]` 单 token 文档、`[1,1,1,2,2,2,2,2]` 1:3 比例等）。若 `build_mask_` 的 doc 边界逻辑在非对称分割时有 off-by-one，当前测试无法发现。

建议：增加 2-3 组不同 doc 切分比例的测试向量。

---

**P2-6 `zipt_consistency_test.cpp:82-89` — 输入构造 batch=1 且 token 类型单一**

证据：`Matrix in(cfg.seq_len, 1)` 始终 batch=1（line 82）。同时 token 均为 `uniform_int(0, vocab_size-1)` 随机整数，Cast 到 Scalar 作为输入。虽然此测试主责是 CPU vs GPU 一致性（batch 维度不是其核心关注），但 batch=1 仍使 GPU kernel 的 batch 维度路径不可见。

建议：可选增加 batch=2 配置的 CPU vs GPU 比较，优先级低于 P2-1/2-2。

### P3

**P3-1 `attn_consistency_test.cpp:241` — 容差偏宽**

证据：`const Scalar tolerance = 5e-2f;`（line 241）。这是 forward 与 forward_step 在**同一引擎**内的比较（非跨引擎），理论上两者数学等价、仅受 fp32 累积噪声影响。5e-2 的容差可能掩盖严重的数值偏差（如 attention weight 计算错误被宽松容差"通过"）。

建议：收紧至 `1e-3` 或 `1e-4`，或改为相对容差 `tol * (1 + |fwd| + |step|)`。当前项目无 `-ffast-math`（铁律 §9），fp32 累积误差通常在 1e-5~1e-4 量级。

---

**P3-2 `zipt_consistency_test.cpp` — CPU vs GPU 路径共享全部高层实现代码**

证据：两条路径均调用 `ZiPTModel::forward(engine, input)`（line 103-104），只是 engine 不同。所有 Layer 逻辑（`ZiPTBlock::forward`、`CrossAttention::forward`、`Softmax` 等）完全共享，差异仅在 `ComputeEngine` 底层原语（CPU vs GPU kernel）。这不是"自证预言"问题——CPU/GPU 一致性测试确实是检测 GPU kernel 实现 bug 的有效手段——但测试名称"双路径一致性"可能误导读者以为测试了两套独立算法实现。

建议：在文件头注释中明确说明"两条路径 = CPU engine vs GPU engine，共享 Layer 实现代码"。

---

## 已知项核对（set_doc_ids 覆盖现状、BPE 258 回归、确定性回归）

### set_doc_ids 覆盖现状

- `zipt_doc_test.cpp`：测试了 `ZiPTBlock` 级别的 `set_doc_ids`（line 56），验证文档掩码不变性。✅ 覆盖。
- `doc_attn_test.cpp`（同组外）：测试了 `GPTModel` 级别的 `set_doc_ids`（line 50），验证端到端文档掩码。✅ 覆盖。
- **缺失**：无测试验证 `set_doc_ids` 在 `batch>1` 时的文档隔离（P2-1）。`ZiPTModel::set_doc_ids`（line 790-794）按 `b * seq_len_ + t` 索引（line 887），batch>1 时若有 off-by-one，当前测试无法发现。

### BPE vocab=258 回归

- `tokenizer_consistency_test.cpp` 的 `test_marker_roundtrip()` 使用 `vocab_size=320`（line 202），覆盖了 258 以上范围，但无**精确边界** `vocab_size=258` 的回归用例（P1-2）。
- 其余 5 个测试文件不涉及 BPE 训练。

### 确定性回归

- `tokenizer_consistency_test.cpp` 未测试"同一输入多次 encode 输出字节级一致"（P1-1）。铁律 8 的回归保护在此测试组中完全缺失。
- 其余 5 个测试文件不涉及 tokenizer encode 确定性。

## 疑似被测问题

**待验证：`compute_layer_zipt.hpp:288` — build_mask_ 文档掩码索引可能在 batch>1 时跨样本串扰**

证据：`build_mask_` 中 `doc_ids_[b * window_ + jl]`（line 288）使用 `window_` 作为步长，但 `doc_ids_` 是从 `ZiPTModel::forward` 通过 `block_doc_ids[b * win_cols + t] = doc_ids_[b * seq_len_ + ...]` 切片得到的（line 883-887）。若 `ZiPTModel` 的 doc_ids 切片逻辑或 `ZiPTBlock` 的 `build_mask_` 索引在 batch>1 时有 off-by-one，当前 batch=1 测试无法发现。由于 `zipt_doc_test.cpp` 仅测 batch=1（P2-1），此疑似问题需通过增加 batch>1 测试验证或排除。
