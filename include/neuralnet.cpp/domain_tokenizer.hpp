#ifndef NN_DOMAIN_TOKENIZER_HPP
#define NN_DOMAIN_TOKENIZER_HPP

// ── domain_tokenizer.hpp — 分词器（聚合头） ─────────────────────────────
//
// 按分词器类型拆分为多个子文件，此文件按依赖顺序聚合，保持对既有代码
// （nn.hpp / text_train / tokenizer_train 等）的单一入口兼容。
//
//   domain_tokenizer_base.hpp      Tokenizer 基类（共享训练/JSON/并行构件）
//   domain_tokenizer_bpe.hpp       字节级 BPE（BPETokenizer）
//   domain_tokenizer_charbpe.hpp   字符级 BPE（CharBPETokenizer）+ 工厂函数
// ─────────────────────────────────────────────────────────────────────────

#include "domain_tokenizer_base.hpp"
#include "domain_tokenizer_bpe.hpp"
#include "domain_tokenizer_charbpe.hpp"

#endif // NN_DOMAIN_TOKENIZER_HPP
