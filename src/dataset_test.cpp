// ── dataset_test — .nndataset 统一数据集（nn::Dataset）测试 ─────────────────
// 覆盖（docs/development/19-unified-dataset.md §8/§9 阶段二验收）：
//   1. 行 trim 规则与原 read_file_lines 逐字一致
//   2. 并行 tokenize 保序 = 顺序 encode（任意并行度逐字节一致，铁律 #8）
//   3. 逐 doc 掩码 = 原全局 mark_assistant_spans（doc 边界重置）逐位一致
//   4. 写 .nndataset → Dataset::load_text 与旧 tokcache 产物逐位一致
//      （token_flow / doc_ids / loss_mask；空 token doc 不丢行号）
//   5. train/test 双子集、loss_scope=all 全 1 合成、配套元数据（sha256 等）
//   6. 词表 kvrec 往返稳定（模型 v6 内嵌字节来源）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <neuralnet.cpp/dataset.hpp>
#include <neuralnet.cpp/domain_tokenizer.hpp>

namespace
{

int g_failures = 0;

void expect(bool ok, const std::string &name)
{
    if (!ok)
    {
        ++g_failures;
        std::printf("  [FAIL] %s\n", name.c_str());
    }
    else
        std::printf("  [ ok ] %s\n", name.c_str());
}

// ── 原 text_train read_file_lines 的 trim 语义（只读参考实现）──────────
std::vector<std::string> ref_read_docs(const std::string &buffer)
{
    std::vector<std::string> docs;
    std::size_t pos = 0;
    const std::size_t total = buffer.size();
    while (pos < total)
    {
        std::size_t line_start = pos;
        while (pos < total && buffer[pos] != '\n') ++pos;
        std::size_t line_end = pos;
        if (pos < total) ++pos;
        while (line_start < line_end &&
               (buffer[line_start] == ' ' || buffer[line_start] == '\t'
                || buffer[line_start] == '\r'))
            ++line_start;
        while (line_end > line_start &&
               (buffer[line_end - 1] == ' ' || buffer[line_end - 1] == '\t'
                || buffer[line_end - 1] == '\r'))
            --line_end;
        if (line_start < line_end)
            docs.emplace_back(buffer, line_start, line_end - line_start);
    }
    return docs;
}

// ── 原 text_train mark_assistant_spans（全局扫描 + doc 边界重置）────────
std::vector<unsigned char> ref_mark_assistant_spans(
    const std::vector<std::size_t> &flow, const std::vector<std::size_t> &doc_ids,
    std::size_t asst_begin, std::size_t asst_end, std::size_t &asst_count)
{
    std::vector<unsigned char> mask(flow.size(), 0);
    bool in_asst = false;
    asst_count = 0;
    for (std::size_t i = 0; i < flow.size(); ++i)
    {
        if (!doc_ids.empty() && i > 0 && doc_ids[i] != doc_ids[i - 1])
            in_asst = false;
        const std::size_t id = flow[i];
        if (id == asst_begin) in_asst = true;
        mask[i] = in_asst ? 1u : 0u;
        if (mask[i]) ++asst_count;
        if (id == asst_end) in_asst = false;
    }
    return mask;
}

void write_file(const std::string &path, const std::string &content)
{
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
}

std::string read_file(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
}

const std::string kCorpus =
    "the quick brown fox jumps over the lazy dog\r\n"
    "   padded line with spaces   \n"
    "\n"
    "\t tab padded \t\r\n"
    "<|user|>what is machine learning<|end_of_user|><|assistant|>it is a field of study<|end_of_assistant|>\n"
    "<|user|>hello there<|end_of_user|><|assistant|>hi friend<|end_of_assistant|>\n"
    "人工神经网络深度学习训练中。\n"
    "<|user|>unclosed assistant doc<|end_of_user|><|assistant|>tail without end marker\n"
    "final line of corpus\n";

}  // namespace

int main()
{
    std::printf("dataset_test:\n");
    const std::string corpus_path = "dataset_test_corpus.txt";
    write_file(corpus_path, kCorpus);

    // ── 1. 行 trim 规则逐字一致 ─────────────────────────────────
    auto docs_r = nn::read_text_docs(corpus_path);
    expect(static_cast<bool>(docs_r), "read_text_docs 成功");
    if (!docs_r) return 1;
    const std::vector<std::string> &docs = *docs_r;
    const std::vector<std::string> ref_docs = ref_read_docs(kCorpus);
    expect(docs == ref_docs, "行 trim 规则与原 read_file_lines 逐字一致");
    expect(docs.size() == 8, "doc 数正确（空行/全空白行被跳过）");

    // ── 2. 并行 tokenize 保序 = 顺序 encode ─────────────────────
    nn::BPETokenizer tok;
    nn::BPETokenizer::Config tcfg;
    tcfg.vocab_size = 320;
    tcfg.min_freq = 1;
    tcfg.log = [](std::string_view) {};
    auto tr = tok.train(kCorpus, tcfg);
    expect(static_cast<bool>(tr), "分词器训练成功");
    if (!tr) return 1;

    const auto per_doc = nn::encode_docs_parallel(tok, docs);
    bool same = per_doc.size() == docs.size();
    std::vector<std::size_t> ref_flow, ref_doc_ids;
    for (std::size_t i = 0; same && i < docs.size(); ++i)
    {
        const auto ref_toks = tok.encode(docs[i]);
        if (per_doc[i] != ref_toks) same = false;
        for (std::size_t t : ref_toks)
        {
            ref_flow.push_back(t);
            ref_doc_ids.push_back(i + 1);   // 原 parallel_tokenize：行号 + 1
        }
    }
    expect(same, "并行 tokenize 与顺序 encode 逐字节一致");

    // ── 3. 逐 doc 掩码 = 全局扫描（doc 边界重置）逐位一致 ─────────
    std::vector<unsigned char> ref_mask_flat;
    {
        std::size_t asst_count = 0;
        ref_mask_flat = ref_mark_assistant_spans(ref_flow, ref_doc_ids,
            tok.assistant_marker_id(), tok.end_assistant_marker_id(), asst_count);
        expect(asst_count > 0, "参考掩码含 assistant 目标");
    }
    std::vector<unsigned char> per_doc_mask_flat;
    std::vector<std::vector<unsigned char>> per_doc_masks;
    for (const auto &toks : per_doc)
    {
        per_doc_masks.push_back(nn::mark_assistant_span(
            toks, tok.assistant_marker_id(), tok.end_assistant_marker_id()));
        per_doc_mask_flat.insert(per_doc_mask_flat.end(),
                                 per_doc_masks.back().begin(),
                                 per_doc_masks.back().end());
    }
    expect(per_doc_mask_flat == ref_mask_flat,
           "逐 doc 掩码与全局 mark_assistant_spans 逐位一致");

    // ── 4/5/6. 写 .nndataset → Dataset 读回逐位一致 ──────────────
    for (int mode = 0; mode < 2; ++mode)
    {
        const bool assistant = (mode == 1);
        const std::string tag = assistant ? "assistant" : "all";
        nn::NndatasetBuild build;
        build.vocab_kvrec = tok.to_kvrec();
        build.source_path = corpus_path;
        build.source_sha256 = nn::sha256_hex(kCorpus);
        build.source_docs = docs.size();
        build.loss_scope = assistant ? "assistant" : "all";
        build.tool = "dataset_gen v1";
        build.created = 12345;
        // train：在第 3 个 doc 位置插一个空 token doc（不丢行号语义）
        build.train_docs = per_doc;
        build.train_docs.insert(build.train_docs.begin() + 3,
                                std::vector<std::size_t>{});
        if (assistant)
        {
            build.train_masks = per_doc_masks;
            build.train_masks.insert(build.train_masks.begin() + 3,
                                     std::vector<unsigned char>{});
        }
        // 期望产物（含空 doc：doc_id = 子集内序号 + 1，空 doc 不产生 token）
        std::vector<std::size_t> exp_flow, exp_ids;
        std::vector<unsigned char> exp_mask;
        for (std::size_t i = 0; i < build.train_docs.size(); ++i)
        {
            for (std::size_t t : build.train_docs[i])
            {
                exp_flow.push_back(t);
                exp_ids.push_back(i + 1);
            }
            if (assistant)
                exp_mask.insert(exp_mask.end(), build.train_masks[i].begin(),
                                build.train_masks[i].end());
            else
                exp_mask.insert(exp_mask.end(), build.train_docs[i].size(), 1u);
        }
        // test：前 3 个 doc
        for (std::size_t i = 0; i < 3; ++i)
        {
            build.test_docs.push_back(per_doc[i]);
            if (assistant) build.test_masks.push_back(per_doc_masks[i]);
        }

        std::ostringstream oss;
        auto wr = nn::write_nndataset(oss, build);
        expect(static_cast<bool>(wr), tag + ": write_nndataset 成功");
        if (!wr) continue;
        const std::string path = "dataset_test_" + tag + ".nndataset";
        write_file(path, nn::write_file_label(nn::NNDATASET_MAGIC,
                                              nn::NNDATASET_FILE_VERSION) + oss.str());

        auto ds_r = nn::Dataset::open(path);
        expect(static_cast<bool>(ds_r), tag + ": Dataset::open 成功");
        if (!ds_r) continue;
        const nn::Dataset &ds = *ds_r;
        expect(ds.info().kind == nn::DatasetKind::Text, tag + ": kind=text");
        expect(ds.info().num_docs == build.train_docs.size(), tag + ": num_docs（含空 doc）");
        expect(ds.info().test_docs == 3, tag + ": test 子集元数据");
        expect(ds.info().loss_scope == build.loss_scope, tag + ": loss_scope 一致");
        expect(ds.info().source_sha256 == nn::sha256_hex(kCorpus), tag + ": source.sha256 一致");

        // 词表 kvrec 往返稳定（模型 v6 内嵌字节来源）
        expect(nn::write_nnvocab(ds.vocab_kvrec()) == nn::write_nnvocab(build.vocab_kvrec),
               tag + ": 词表 kvrec 往返字节稳定");

        // train 子集逐位一致（空 doc 不产生 token、但不改后续 doc_id）
        auto corpus_r = ds.load_text(false);
        expect(static_cast<bool>(corpus_r), tag + ": load_text(train) 成功");
        if (!corpus_r) continue;
        expect(corpus_r->token_flow == exp_flow, tag + ": token_flow 逐位一致");
        expect(corpus_r->doc_ids == exp_ids, tag + ": doc_ids 逐位一致（空 doc 不丢行号）");
        expect(corpus_r->loss_mask == exp_mask,
               tag + (assistant ? ": loss_mask 逐位一致" : ": loss_scope=all 合成全 1 掩码"));

        // test 子集（前 3 个 doc）逐位一致
        auto test_r = ds.load_text(true);
        expect(static_cast<bool>(test_r), tag + ": load_text(test) 成功");
        if (!test_r) continue;
        std::vector<std::size_t> t_flow, t_ids;
        std::vector<unsigned char> t_mask;
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t t : per_doc[i])
            {
                t_flow.push_back(t);
                t_ids.push_back(i + 1);
            }
            if (assistant)
                t_mask.insert(t_mask.end(), per_doc_masks[i].begin(), per_doc_masks[i].end());
            else
                t_mask.insert(t_mask.end(), per_doc[i].size(), 1u);
        }
        expect(test_r->token_flow == t_flow && test_r->doc_ids == t_ids
               && test_r->loss_mask == t_mask, tag + ": test 子集逐位一致");
    }

    // ── 附加：.nnvocab 可 open（词表独立加载场景），load_text 明确报错 ──
    {
        const std::string vpath = "dataset_test_vocab.nnvocab";
        auto sv = tok.save(vpath);
        expect(static_cast<bool>(sv), "保存 .nnvocab");
        auto ds_r = nn::Dataset::open(vpath);
        expect(static_cast<bool>(ds_r), ".nnvocab 可被 Dataset::open 嗅探打开");
        if (ds_r)
        {
            auto bad = ds_r->load_text(false);
            expect(!bad, ".nnvocab 的 load_text 明确报错（不含数据集子集）");
            expect(nn::write_nnvocab(ds_r->vocab_kvrec()) == read_file(vpath),
                   ".nnvocab 词表 kvrec 逐字节稳定");
        }
    }

    std::error_code ec;
    for (const char *p : {"dataset_test_corpus.txt", "dataset_test_all.nndataset",
                          "dataset_test_assistant.nndataset", "dataset_test_vocab.nnvocab"})
        std::filesystem::remove(p, ec);

    std::printf("\ndataset_test: %d failure(s)\n", g_failures);
    return g_failures != 0 ? 1 : 0;
}
