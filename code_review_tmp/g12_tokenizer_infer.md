# g12 Tokenizer + 推理入口代码审查

## 模块概览

本次审查覆盖 8 个文件，分三层：

| 层级 | 文件 | 职责 |
|------|------|------|
| 库（L4 领域） | `domain_tokenizer_base.hpp` | Tokenizer 抽象基类：BPE 共享训练/合并算法、JSON 序列化、并行预分词、对话标记 |
| 库（L4 领域） | `domain_tokenizer_bpe.hpp` | BPETokenizer：字节级 BPE，GPT-2 风格预分词状态机 |
| 库（L4 领域） | `domain_tokenizer_charbpe.hpp` | CharBPETokenizer：字符级 BPE（UTF-8/CJK），工厂函数 |
| 库（L4 领域） | `domain_tokenizer.hpp` | 聚合头 |
| 库（CLI） | `cli/cli_engine_factory.hpp` | CPU/GPU/CUDA 引擎选择工厂 |
| 可执行（L5） | `src/tokenizer_train.cpp` | BPE 训练 CLI |
| 可执行（L5） | `src/tokenizer_infer.cpp` | 分词器编码/解码 CLI |
| 可执行（L5） | `src/text_infer.cpp` | GPT 文本生成推理 CLI |

**审查维度优先级**：正确性 > 生命周期/内存 > 并发 > 规范 > 设计

---

## 发现

### P0

无。

### P1

**P1-1 `domain_tokenizer_bpe.hpp:163` — BPETokenizer::train 缺少 vocab_size 内部校验，vocab_size < 258 触发无符号下溢**

代码摘录：
```cpp
// domain_tokenizer_bpe.hpp:163
merges_.reserve(config.vocab_size - BYTE_BASE - 2);
// BYTE_BASE = 256, BYTE_BASE + 2 = 258
// 当 config.vocab_size < 258 时，无符号减法下溢为极大值
```

分析：CLI `tokenizer_train.cpp:82-88` 有 `vocab_size >= 258` 的校验，但 `BPETokenizer::train` 作为库 API 缺乏防御。直接调用 `tok.train(text, {.vocab_size = 100})` 会导致 `merges_.reserve(~SIZE_MAX)` → 分配失败/abort。CharBPE 在 `domain_tokenizer_charbpe.hpp:195-202` 有正确的 `MIN_VOCAB_SIZE` 校验，BPE 缺失同类守卫。

修复建议：在 `BPETokenizer::train` 入口添加校验：
```cpp
if (config.vocab_size < BYTE_BASE + 2)  // 258
    return std::unexpected(Error{"BPE vocab_size must be >= 258 (256 bytes + BOS/EOS), got "
                                 + std::to_string(config.vocab_size)});
```

---

**P1-2 `compute_layer_gpt.hpp:967,1071` — GPTModel::generate 中 context 向量无界增长**

代码摘录：
```cpp
// compute_layer_gpt.hpp:967
std::vector<std::size_t> context(prompt);
// ... 在 generate 循环中:
// compute_layer_gpt.hpp:1071
context.push_back(next_token);
// 滑动窗口只重建 KV cache（line 1013-1022），不裁剪 context
```

分析：`generate()` 的 KV cache 有滑动窗口（`cur_len >= seq_len_` 时重建，保留最后 `seq_len_-1` 个 token），但 `context` 向量只追加不裁剪。每生成一个 token，`context` 增长 `sizeof(size_t)` 字节。对于 `max_tokens=200`（默认）无问题；但用户设置 `--max-tokens 100000` 时，`context` 占用 ~800KB（仍可接受），且滑动窗口重建时 `fill_cache_` 遍历 `context` 末尾 `seq_len_-1` 个元素效率不变。**实际风险低**，但设计上 KV cache 有界而 context 无界不一致。

修复建议（可选优化）：在滑动窗口重建时同步裁剪 `context`：
```cpp
if (cur_len >= seq_len_) {
    const std::size_t keep = seq_len_ - 1;
    if (context.size() > keep)
        context.erase(context.begin(), context.end() - keep);
    // ... 重建 KV cache
}
```

---

### P2

**P2-1 `tokenizer_infer.cpp:221` — longest_lines_mode 在非 main 函数中调用 std::exit(1)**

代码摘录：
```cpp
// tokenizer_infer.cpp:217-222
void longest_lines_mode(const nn::Tokenizer &tokenizer, const std::string &path, std::size_t top_n)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        std::cerr << "打开文件失败: " << path << '\n';
        std::exit(1);  // ← 库/工具函数中直接 exit
    }
```

分析：`longest_lines_mode` 是被 `main` 调用的工具函数，非 `main` 本身。`std::exit` 跳过局部对象析构（虽然此函数内无关键 RAII 对象）。更重要的是，如果此函数未来被其他入口复用，`exit(1)` 会终止整个进程而非传播错误。项目铁律 1（禁止 throw）下，应返回 `Result<void>` 或错误码。

修复建议：改为返回 `Result<void>` 或 `bool`，由调用方 `main` 决定退出码。

---

**P2-2 `tokenizer_infer.cpp:107-123` — parse_ids 静默丢弃无效 token ID**

代码摘录：
```cpp
// tokenizer_infer.cpp:120-122
auto v = nn::parse_number<std::size_t>(token);
if (v) ids.push_back(*v);
// 无效 token 被静默跳过，无警告
```

分析：用户输入 `--decode "1,abc,3"` 时，`abc` 被静默丢弃，输出只有 token 1 和 3 的解码结果。用户无法区分"静默跳过"和"输入被正确解析"。虽然 `parse_number` 不抛异常（`-fno-exceptions`），但可以在 stderr 输出警告。

修复建议：在 `!v` 分支添加 `std::cerr << "警告: 跳过无效 token ID: " << token << "\n";`。

---

**P2-3 `compute_layer_gpt.hpp:988-999` — 空 prompt + 无 BOS 时 generate 从未初始化的 logits 采样**

代码摘录：
```cpp
// compute_layer_gpt.hpp:988
Tensor last_logits_t;   // 默认构造，未初始化
// ...
// compute_layer_gpt.hpp:994-1001
auto fill_cache_ = [&](std::size_t start) -> Result<void>
{
    for (std::size_t i = start; i < context.size(); ++i)
    {
        // ... forward_step 写入 last_logits_t
        last_logits_t = *r;
        ++cur_len;
    }
    return {};
};
// 若 context 为空（prompt 无内容且 bos_id == npos），fill_cache_ 不执行
// last_logits_t 保持默认构造状态
```

分析：当 `prompt` 为空且 tokenizer 无 BOS token（`bos_id == npos`）时，`context` 为空向量，`fill_cache_` 不执行任何迭代，`last_logits_t` 保持默认构造的空 Tensor。随后 `generate` 循环的 `engine.to_matrix(last_logits_t)` 会访问未初始化张量，行为未定义。

修复建议：在 `generate()` 入口校验 `!context.empty()`，或在 fill_cache_ 后检查 `last_logits_t` 有效性：
```cpp
if (context.empty())
    return std::unexpected(Error{"generate: empty context (no prompt tokens and no BOS)"});
```

---

### P3

**P3-1 `domain_tokenizer_bpe.hpp:154-173, domain_tokenizer_charbpe.hpp:305-325` — 对话标记追加后实际 vocab_size 可超过 config.vocab_size**

代码摘录：
```cpp
// domain_tokenizer_bpe.hpp:169-173
bpe_train_impl_(vocab_, byte_chunks, target_merges, ...);
rebuild_merge_map_();
add_dialogue_markers_to_vocab_(vocab_);  // 追加 8 对话标记 + 保留 token
```

分析：`target_merges = vocab_size - 258` 保证 BPE 合并后 vocab 恰好到 `vocab_size`（=258 + target_merges）。但 `add_dialogue_markers_to_vocab_` 追加 8 个对话标记 + 8 个保留 token，使最终 `vocab_.size()` 超出 `config.vocab_size`。CharBPE 同理。这不是 bug（对话标记是必需的），但 API 用户可能对 `vocab_size()` 返回值大于请求值感到意外。

修复建议：文档明确说明最终词表会包含额外对话标记。或在 Config 中预留槽位。

---

**P3-2 `domain_tokenizer_base.hpp:320-331` — bpe_train_impl_ 增量更新 dec lambda 中 w 可能为负**

代码摘录：
```cpp
// domain_tokenizer_base.hpp:320-325
auto w = static_cast<std::ptrdiff_t>(weight);
auto dec = [&](std::uint64_t k) {
    auto it = pair_freq.find(k);
    if (it != pair_freq.end())
    {
        if (it->second <= static_cast<std::size_t>(w))
            pair_freq.erase(it);
        else
            it->second -= static_cast<std::size_t>(w);
    }
};
```

分析：`weight` 是 `std::size_t`（无符号），转为 `ptrdiff_t` 后 `w` 始终非负（因为 weight 来自词频计数，不可能为负）。`dec` 用 `static_cast<std::size_t>(w)` 还原为无符号值做减法。若 `weight` 为 0，`dec` 不会擦除任何条目（`it->second <= 0` 仅在 `it->second == 0` 时成立，但 pair_freq 中不会有 0 值条目因为只有 `+=` 操作插入/累加）。逻辑正确，但 `w` 的 signed 转换不必要——可直接用 `weight`。

修复建议（代码整洁）：将 `dec` 改为直接使用 `weight`（`std::size_t`），避免 signed/unsigned 混用。

---

**P3-3 `domain_tokenizer_base.hpp:100-103` — restore_dialogue_markers 线性扫描 8×vocab 查找标记**

代码摘录：
```cpp
// domain_tokenizer_base.hpp:100-108
for (int i = 0; i < 8; ++i)
{
    ids[i] = npos;
    for (std::size_t j = 0; j < vocab.size(); ++j)
    {
        if (vocab[j] == marker_strs[i])
        { ids[i] = j; break; }
    }
```

分析：8 个标记 × vocab_size 的线性扫描。对于大词表（如 50000+），这是 400000 次字符串比较。实际影响：此函数仅在 `load_from_string` 时调用一次，不在热路径上。但可优化为 O(vocab + 8) 的一次扫描。

修复建议（可选优化）：改为单次扫描 vocab，用 `std::string_view` 匹配 8 个标记字符串。

---

## 已知项核对

| 铁律 | 状态 | 说明 |
|------|------|------|
| 1. 禁 throw/try/catch | ✅ | 8 个文件无 throw/try/catch；错误走 `Result<T>` |
| 2. 禁 new/delete/裸指针所有权 | ✅ | 使用 `std::unique_ptr`/`std::vector`；`compute_vk_backend.hpp` 的 `new GpuBackend()` 在审查范围外 |
| 3. 分层职责单一 | ✅ | L4 领域层不写底层计算；L5 CLI 不含算法逻辑 |
| 4. 不穿透接口 | ✅ | CLI 通过 `model.layer_at(0)` + `dynamic_cast` 获取 GPTModel，未访问内部数据结构 |
| 5. 布局 batch-major | ✅ | `domain_tokenizer_bpe.hpp:699-701` 注释明确 `i = b*seq + t`；doc_ids 掩码 `domain_tokenizer_bpe.hpp:958-959` 使用 `b * seq_len + i` |
| 6. GPU 命令录制生命周期 | ✅ | `generate()` 中 `begin_batch/end_batch` 包裹 `forward_step`；KV cache 张量在循环外预分配，生命周期覆盖整个生成过程 |
| 7. AOT 闭合世界 | ✅ | 分词器/推理入口不涉及 GPU shader |
| 8. 确定性 | ✅ | **BPE merge 候选选择**：`bpe_train_impl_` line 282-289 用 `pair_key` 打破平局，注释明确"与 unordered_map 迭代顺序无关"；**CharBPE 字符 ID 分配**：`domain_tokenizer_charbpe.hpp:254-263` 按码点排序后分配，不依赖 `unordered_set` 迭代顺序；**并行预分词**：`parallel_pretokenize` 在安全切分点切分，map-reduce 归并，注释声明"结果与单线程完全一致" |
| 9. 大词表禁 one-hot | ✅ | 分词器不涉及 one-hot；text_infer 使用 `GPTModel::generate` 逐 token 解码 |

### 高频坑核对

| 坑 | 状态 | 说明 |
|----|------|------|
| BPE "所有出现位置"索引线性膨胀 | ✅ 已修 | `bpe_train_impl_` 使用去重 chunk + 增量 pair 频次更新，内存与唯一词数（非语料大小）成正比 |
| 大词表 one-hot 3.2GB | ✅ 不涉及 | 分词器/推理入口无 one-hot |
| GPU 录制期 use-after-free | ✅ | `generate()` 的 KV cache 张量在循环外预分配，生命周期覆盖整个生成过程 |
| 布局混用 | ✅ | batch-major 全局一致 |

---

## 待验证（如有）

1. **`compute_layer_gpt.hpp` 中 `forward_step` 对 KV cache 的写入偏移**：`forward_step` 接收 `pos` 参数（= `cur_len`），需确认 GPTBlock 内部 `self_attn_.forward_step` 使用 `pos` 作为 KV cache 列索引写入。已读 `forward_step` 签名（line 903-909），但 GPTBlock 内部实现未在本次审查文件中，交叉引用需查看 `compute_layer_feedforward.hpp` 或 `compute_layer_attention.hpp` 的 `forward_step` 方法。

2. **`text_infer.cpp` 交互模式下 prompt_tokens 构建与训练格式一致性**：对话模式下 `<|system|>...<|end_of_system|><|user|>...<|end_of_user|><|assistant|>` 的拼接顺序需与训练时 `text_train.cpp` 的预处理格式完全一致。已读 `text_infer.cpp:165-188` 的构建逻辑，但训练时的格式定义在 `text_train.cpp` 中，未在本次审查范围内完整交叉验证。

3. **`parallel_pretokenize` 在极端输入下的行为**：当文本全为单一字符（无空白切分点）时，`find_safe_splits` 返回空数组，自动回退顺序执行。逻辑正确，但未验证线程池 `parallel_for_samples` 在 `n=1` 时的行为（应退化为单次调用）。
