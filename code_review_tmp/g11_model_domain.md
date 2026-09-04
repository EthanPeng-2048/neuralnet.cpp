# g11 Model 容器/序列化/domain 工厂代码审查

## 模块概览
本组审查覆盖 neuralnet.cpp 的 L3 实现层（Model 容器、规格、序列化）和 L4 领域构建层（5 个域工厂）。
- **Model 容器** (`model_container.hpp`): 负责层组合、前向/反向传播委托及参数管理。
- **ModelSpec** (`model_spec.hpp`): 纯数据结构，记录模型架构参数。
- **序列化** (`model_serialization.hpp`): 实现 v4 自描述格式的模型存取。
- **域工厂** (`domain_*.hpp`): 针对 MNIST/GPT/CNN/RAPT/ZiPT 的参数预设与模型实例化。

## 发现

### P1

**P1-1 `model_serialization.hpp:325` — 恶意文件触发巨量内存分配**
```cpp
// detail::read_spec_header
const auto len = static_cast<std::size_t>(*len_r);
std::string bytes(len, '\0');  // 若 len 为 0xFFFFFFFFFFFFFFFF，此处触发 bad_alloc
```
**分析**：`len` 由文件输入，无上限校验。在 `-fno-exceptions` 环境下，分配失败导致 `std::terminate`。同见 `read_tokenizer` (line 406)。
**修复建议**：引入 `MAX_SPEC_LEN` 和 `MAX_TOKENIZER_LEN` 上限常量（如 64MB），在分配前校验 `len`。

**P1-2 `model_serialization.hpp:175` — 矩阵维度溢出风险**
```cpp
const auto rows = static_cast<std::size_t>(*rows_r); // rows_r 为 uint64_t
const auto cols = static_cast<std::size_t>(*cols_r);
```
**分析**：虽然 `read_matrix` 会校验 `rows != m.rows()`，但 `m` 是根据当前 Model 的 tensor 创建的。恶意文件可能在早期阶段通过巨大的 `rows/cols` 导致 `Matrix tmp` 构造函数分配失败。
**修复建议**：在 `read_matrix` 开头增加 `rows * cols * sizeof(Scalar) > MAX_MATRIX_BYTES` 的校验。

### P2

**P2-1 `model_spec.hpp:121-129` — GPT/RAPT 字段匹配不完整**
```cpp
// GPT family spec_matches
return a.vocab_size == b.vocab_size && a.d_model == b.d_model && ...
```
**分析**：虽然 ZiPT 已单独处理，但若未来 `ModelSpec` 为 GPT 类型新增字段（如新的 dropout 率），需手动同步 `spec_matches`。当前代码逻辑正确，但属于维护性风险。
**修复建议**：建议在 `spec_matches` 的 GPT 分支中也包含 `memory_tokens == 0 && window == 0` 的显式断言或校验，确保不同家族间不发生参数误植。

**P2-2 `nn.hpp:58` — 保留了 `one_hot` 工具函数**
```cpp
[[nodiscard]] inline Result<Matrix> one_hot(const std::vector<std::size_t> &true_i, std::size_t mat_size)
```
**分析**：铁律 9 要求“大词表禁 one-hot”。虽然工具函数本身不违规，但在头文件入口暴露此函数可能诱导用户在 GPT 场景误用。
**修复建议**：将 `one_hot` 移入 `internal` 命名空间或加 `[[deprecated("Use forward_sparse for large vocab")]]` 标注。

### P3

**P3-1 `model_keyvalue_record.hpp:181` — 未知类型跳过逻辑**
```cpp
if (type != Type::UInt && type != Type::Str && type != Type::UIntArray)
    continue;
```
**分析**：向前兼容设计良好，跳过未知字段而不报错，确保了旧版本读取新版本文件时的健壮性。

## 已知项核对
- **S3 待办项**：已在 P1-1 中重新评估。虽然原备注称“暂不处理”，但在网络传输或不可信来源场景下，此为 P0/P1 级安全风险。
- **架构一致性**：域工厂均正确调用 `Model::add` 且不直接访问 `layers_` 内部，符合铁律 4。
- **布局约定**：工厂中维度计算（如 CNN 的 `flattened = c * h * w`）遵循 batch-major。

## 待验证（如有）
- `compute_layer.hpp` 中 `GPTModel` / `ZiPTModel` 的构造函数参数顺序是否与域工厂完全匹配（已初步 grep 确认一致）。
