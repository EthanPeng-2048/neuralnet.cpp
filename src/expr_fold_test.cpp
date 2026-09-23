// ───────────────────────────────────────────────────────────────────────────
//  expr_fold_test.cpp — P-C1 分块状态归约（FoldSpec）CPU 语义对拍
//
//  覆盖（全部走共享构造 expr_fold.hpp —— 与 scan_exprs AOT 收集同源，
//  保证 key/结构一致）：
//    1. rowmax fold        vs 独立逐行 max 参考
//    2. rowsum fold        vs 独立逐行 sum 参考
//    3. softmax_denom fold（online 双状态：m 跨块更新 + l = l·α + Σexp(x−m')）
//                          vs 独立 Σ_k exp(x − max_k x) 参考
//    4. K 边界族：{1, 7, 32, 33, 100, 1000}——单列 / 非块整除 / 整除 / 尾块
//    5. attention fold：5 掩码变体（含 Doc/AlibiDoc）× 4 形状族 vs 独立公式
//       参考；{2,133,4} 跨 EXPR_FOLD_BLOCK=128 → 多块流式 + causal 整块跳过
//    6. validate 负例：状态吃元素源 / fold+顶层 instrs / fold+matmul /
//       finalize 读输入 —— 四类违规必须被拒绝（静态拒而非静默错算）
//    7. registry bin roundtrip：v8 causal_skip 写读对称（key 含该位 →
//       往返 key 全等即证未丢；丢失=静默退化全量算、其余测试仍会绿）
//  纯 CPU；输出 (rows,1)（eval_expr 的 cols 参数 = 1 调用约定）。
// ───────────────────────────────────────────────────────────────────────────

#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <neuralnet.cpp/compute_tensor.hpp>
#include <neuralnet.cpp/compute_cpu_engine.hpp>
#include <neuralnet.cpp/expr_fold.hpp>

using nn::Scalar;

int test_expr_fold()
{
    int fail = 0;
    const auto check = [&fail](bool ok, const std::string& msg)
    {
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << msg << "\n";
        if (!ok) ++fail;
    };

    nn::CpuEngine engine;
    std::mt19937 rng(20260923);
    std::uniform_real_distribution<Scalar> dist(-2.0f, 2.0f);
    constexpr Scalar kTol = Scalar{1e-4};

    for (const std::size_t K : {std::size_t{1}, std::size_t{7}, std::size_t{32},
                                std::size_t{33}, std::size_t{100}, std::size_t{1000}})
    {
        const std::size_t rows = 5;
        nn::Tensor x = nn::Tensor::cpu(rows, K);
        {
            auto sp = x.cpu_matrix().span();
            for (auto& v : sp) v = dist(rng);
        }
        const std::array<nn::Tensor, 1> ins{x};

        // ── 1) rowmax ──
        {
            nn::ExprSpec spec = nn::expr::make_fold_rowmax(static_cast<std::uint32_t>(K));
            check(nn::validate_expr_spec(spec, 1).has_value(),
                  "rowmax validate K=" + std::to_string(K));
            auto out = engine.eval_expr(spec, ins, rows, 1);
            if (!out) { check(false, "rowmax eval K=" + std::to_string(K) + ": " + out.error().message); }
            else
            {
                const auto os = out->cpu_matrix().span();
                const auto xs = x.cpu_matrix().span();
                Scalar err = 0;
                for (std::size_t r = 0; r < rows; ++r)
                {
                    Scalar ref = -std::numeric_limits<Scalar>::infinity();
                    for (std::size_t k = 0; k < K; ++k)
                        ref = std::fmax(ref, xs[r * K + k]);
                    err = std::fmax(err, std::fabs(os[r] - ref));
                }
                check(err <= kTol, "rowmax vs ref K=" + std::to_string(K) +
                       " err=" + std::to_string(err));
            }
        }

        // ── 2) rowsum ──
        {
            nn::ExprSpec spec = nn::expr::make_fold_rowsum(static_cast<std::uint32_t>(K));
            check(nn::validate_expr_spec(spec, 1).has_value(),
                  "rowsum validate K=" + std::to_string(K));
            auto out = engine.eval_expr(spec, ins, rows, 1);
            if (!out) { check(false, "rowsum eval K=" + std::to_string(K) + ": " + out.error().message); }
            else
            {
                const auto os = out->cpu_matrix().span();
                const auto xs = x.cpu_matrix().span();
                Scalar err = 0;
                for (std::size_t r = 0; r < rows; ++r)
                {
                    Scalar ref = 0;
                    for (std::size_t k = 0; k < K; ++k) ref += xs[r * K + k];
                    err = std::fmax(err, std::fabs(os[r] - ref) / std::fmax(Scalar{1}, std::fabs(ref)));
                }
                check(err <= kTol, "rowsum vs ref K=" + std::to_string(K) +
                       " err=" + std::to_string(err));
            }
        }

        // ── 3) softmax 分母（online 双状态）──
        {
            nn::ExprSpec spec = nn::expr::make_fold_softmax_denom(static_cast<std::uint32_t>(K));
            check(nn::validate_expr_spec(spec, 1).has_value(),
                  "softmax_denom validate K=" + std::to_string(K));
            auto out = engine.eval_expr(spec, ins, rows, 1);
            if (!out) { check(false, "softmax_denom eval K=" + std::to_string(K) + ": " + out.error().message); }
            else
            {
                const auto os = out->cpu_matrix().span();
                const auto xs = x.cpu_matrix().span();
                Scalar err = 0;
                for (std::size_t r = 0; r < rows; ++r)
                {
                    Scalar m = -std::numeric_limits<Scalar>::infinity();
                    for (std::size_t k = 0; k < K; ++k)
                        m = std::fmax(m, xs[r * K + k]);
                    Scalar ref = 0;
                    for (std::size_t k = 0; k < K; ++k)
                        ref += std::exp(xs[r * K + k] - m);
                    err = std::fmax(err, std::fabs(os[r] - ref) / std::fmax(Scalar{1}, std::fabs(ref)));
                }
                check(err <= kTol, "softmax_denom vs ref K=" + std::to_string(K) +
                       " err=" + std::to_string(err));
            }
        }
    }

    // ── 6) validate 负例（静态拒绝）──
    {
        const auto rejects = [&](nn::ExprSpec spec, const char* msg)
        {
            const bool ok = !nn::validate_expr_spec(spec, spec.views.size()).has_value();
            check(ok, std::string("validate rejects ") + msg);
        };
        // (a) 状态吃元素类源（Input → 状态，kb 依赖覆盖）
        {
            nn::ExprSpec s = nn::expr::make_fold_rowsum(16);
            s.fold->body = {};
            nn::ExprInstr b;
            b.op  = static_cast<uint8_t>(nn::ExprOp::Add);
            b.dst = 0;
            b.a   = nn::expr::input(0);
            b.b   = nn::expr::cst(0);
            s.fold->body = { b };
            rejects(s, "state written from element source");
        }
        // (b) fold + 顶层 instrs 非空
        {
            nn::ExprSpec s = nn::expr::make_fold_rowsum(16);
            nn::ExprInstr t;
            t.op  = static_cast<uint8_t>(nn::ExprOp::Add);
            t.dst = 1;
            t.a   = nn::expr::reg(0);
            t.b   = nn::expr::cst(0);
            s.instrs = { t };
            rejects(s, "fold + non-empty top-level instrs");
        }
        // (c) fold + matmul 段互斥
        {
            nn::ExprSpec s = nn::expr::make_fold_rowsum(16);
            nn::MatmulSpec mm;
            mm.a_input = 0; mm.b_input = 0; mm.k = 16; mm.batch = 1;
            s.matmul = mm;
            rejects(s, "fold + matmul segment");
        }
        // (d) finalize 读输入
        {
            nn::ExprSpec s = nn::expr::make_fold_rowsum(16);
            s.fold->finalize = {};
            nn::ExprInstr t;
            t.op  = static_cast<uint8_t>(nn::ExprOp::Add);
            t.dst = 0;
            t.a   = nn::expr::reg(0);
            t.b   = nn::expr::input(0);
            s.fold->finalize = { t };
            rejects(s, "finalize reading inputs");
        }
    }

    // ── P-C2 attention fold 对拍（独立公式参考；5 掩码 × 4 形状族）────────
    // 确定性迷你 case：seq=dk=bh=1，Q=2 K=3 Vt=5 → s=6, softmax=1, O=5
    {
        nn::Tensor Q = nn::Tensor::cpu(1, 1), K = nn::Tensor::cpu(1, 1),
                   Vt = nn::Tensor::cpu(1, 1);
        Q.cpu_matrix().span()[0] = 2;
        K.cpu_matrix().span()[0] = 3;
        Vt.cpu_matrix().span()[0] = 5;
        nn::ExprSpec sp = nn::expr::make_fold_attn_o(1, 1, 1,
                            nn::expr::FoldAttnMask::Plain);
        const std::vector<nn::Tensor> ii{Q, K, Vt};
        auto o = engine.eval_expr(sp, ii, 1, 1);
        const Scalar got = o ? o->cpu_matrix().span()[0] : Scalar{-999};
        check(o.has_value() && std::fabs(got - Scalar{5}) < Scalar{1e-5},
              "attn fold mini(1x1) O=" + std::to_string(got) + " expect 5"
              + (o ? "" : (" err=" + o.error().message)));
        // 分辨"多 j"vs"多行"：
        //   A 单行多 j：bh=1 seq=4 dk=1；B 多行单 j：bh=2 seq=1 dk=1
        for (int ab = 0; ab < 2; ++ab)
        {
            const std::uint32_t bh2 = ab == 0 ? 1u : 2u;
            const std::uint32_t seq2 = ab == 0 ? 4u : 1u;
            nn::Tensor Q2 = nn::Tensor::cpu(static_cast<std::size_t>(bh2) * 1, seq2);
            nn::Tensor K2 = nn::Tensor::cpu(static_cast<std::size_t>(bh2) * 1, seq2);
            nn::Tensor V2 = nn::Tensor::cpu(static_cast<std::size_t>(bh2) * seq2, 1);
            for (std::size_t i = 0; i < Q2.cpu_matrix().span().size(); ++i)
                Q2.cpu_matrix().span()[i] = Scalar{2};
            for (std::size_t i = 0; i < K2.cpu_matrix().span().size(); ++i)
                K2.cpu_matrix().span()[i] = Scalar{3};
            for (std::size_t i = 0; i < V2.cpu_matrix().span().size(); ++i)
                V2.cpu_matrix().span()[i] = Scalar{5};
            nn::ExprSpec sp2 = nn::expr::make_fold_attn_o(seq2, 1, bh2,
                                 nn::expr::FoldAttnMask::Plain);
            const std::vector<nn::Tensor> i2{Q2, K2, V2};
            auto o2 = engine.eval_expr(sp2, i2, bh2 * seq2, 1);
            std::string vals = "?";
            if (o2)
            {
                vals.clear();
                const auto sp2o = o2->cpu_matrix().span();
                for (std::size_t i = 0; i < bh2 * seq2; ++i)
                    vals += " " + std::to_string(sp2o[i]);
            }
            // 常数 Q=2,K=3 → 每行 softmax=1 → O = V = 5（所有行）
            check(o2.has_value(), std::string("mini-") + (ab ? "B" : "A")
                  + " eval" + (o2 ? "" : (": " + o2.error().message)));
            if (o2)
            {
                bool okv = true;
                const auto sp2o = o2->cpu_matrix().span();
                for (std::size_t i = 0; i < bh2 * seq2; ++i)
                    if (std::fabs(sp2o[i] - Scalar{5}) > Scalar{1e-5}) okv = false;
                check(okv, std::string("mini-") + (ab ? "B(bh2seq1)" : "A(seq4)")
                      + " O=" + vals + " expect 5×" + std::to_string(bh2 * seq2));
            }
        }
        // C: 非均匀 Vt、Q=K=0（s 全 0 → p=1/4 均匀）→ 手算期望：
        //   d0 = 0.25·(10+20+30+40) = 25；d1 = 0.25·(1+2+3+4) = 2.5
        {
            nn::Tensor Q3 = nn::Tensor::cpu(2, 4), K3 = nn::Tensor::cpu(2, 4),
                       V3 = nn::Tensor::cpu(4, 2);   // Q/K: (bh·dk, seq)=(2,4)
            for (auto& v : Q3.cpu_matrix().span()) v = 0;
            for (auto& v : K3.cpu_matrix().span()) v = 0;
            const Scalar vv[8] = {10, 1, 20, 2, 30, 3, 40, 4};
            for (int i = 0; i < 8; ++i) V3.cpu_matrix().span()[i] = vv[i];
            nn::ExprSpec sp3 = nn::expr::make_fold_attn_o(4, 2, 1,
                                nn::expr::FoldAttnMask::Plain);
            const std::vector<nn::Tensor> i3{Q3, K3, V3};
            auto o3 = engine.eval_expr(sp3, i3, 4, 2);
            check(o3.has_value(), "mini-C eval"
                  + (o3 ? "" : (": " + o3.error().message)));
            if (o3)
            {
                const auto s3 = o3->cpu_matrix().span();
                std::string vals;
                for (std::size_t i = 0; i < 8; ++i)
                    vals += " " + std::to_string(s3[i]);
                // Q=K=0 → S 全 0 → p 均匀 1/4：全部 4 行输出相同
                //   d0 = (10+20+30+40)/4 = 25；d1 = (1+2+3+4)/4 = 2.5（全量比对）
                bool okc = true;
                for (std::size_t i = 0; i < 8; ++i)
                {
                    const Scalar want = (i % 2 == 0) ? Scalar{25} : Scalar{2.5};
                    if (std::fabs(s3[i] - want) >= Scalar{1e-4}) okc = false;
                }
                check(okc, "mini-C O=" + vals + " expect 4×(25 2.5)");
            }
        }
    }

    for (const auto mk : {nn::expr::FoldAttnMask::Plain,
                          nn::expr::FoldAttnMask::Causal,
                          nn::expr::FoldAttnMask::Alibi,
                          nn::expr::FoldAttnMask::Doc,
                          nn::expr::FoldAttnMask::AlibiDoc})
    {
        const char* mname = mk == nn::expr::FoldAttnMask::Plain    ? "plain"
                         : mk == nn::expr::FoldAttnMask::Causal    ? "causal"
                         : mk == nn::expr::FoldAttnMask::Alibi     ? "alibi"
                         : mk == nn::expr::FoldAttnMask::Doc       ? "doc"
                                                                 : "alibidoc";
        struct Sh { std::uint32_t bh, seq, dk; };
        // {2,133,4}：seq > EXPR_FOLD_BLOCK(128) → 跨块流式（m/l 进位 + vecacc
        //   rescale）与 causal 整块跳过分支（k0>qt 空块）首次被执行；
        //   其余形状覆盖单块与奇数行（clamp + row_ok）
        for (const Sh sh : {Sh{2, 9, 4}, Sh{1, 33, 8}, Sh{3, 5, 2},
                            Sh{2, 133, 4}})
        {
            const std::size_t rows_out = static_cast<std::size_t>(sh.bh) * sh.seq;
            nn::Tensor Q = nn::Tensor::cpu(static_cast<std::size_t>(sh.bh) * sh.dk, sh.seq);
            nn::Tensor K = nn::Tensor::cpu(static_cast<std::size_t>(sh.bh) * sh.dk, sh.seq);
            nn::Tensor Vt = nn::Tensor::cpu(rows_out, sh.dk);
            nn::Tensor slopes = nn::Tensor::cpu(1, sh.bh);
            // doc 输入（与 Layer 组包同序：Q,K,Vt,[slopes],doc_col,doc_ids）：
            //   doc_col (rows,1) = 行（查询位置）文档 id；doc_ids (1, bh*seq)
            //   按 (b,h) 块重复——BatchCol(seq) 读 [块*seq+j]，heads>1 时
            //   (1, batch*seq) 会越界（AGENTS S7 教训 #2）
            const std::uint32_t sseq = sh.seq;
            nn::Tensor doc_col = nn::Tensor::cpu(rows_out, 1);
            nn::Tensor doc_ids_t = nn::Tensor::cpu(
                1, static_cast<std::size_t>(sh.bh) * sh.seq);
            const auto doc_of = [sseq](std::uint32_t pos) -> Scalar
            { return pos < sseq / 2 ? Scalar{1} : Scalar{2}; };
            for (std::size_t r = 0; r < rows_out; ++r)
                doc_col.cpu_matrix().span()[r] =
                    doc_of(static_cast<std::uint32_t>(r % sh.seq));
            for (std::uint32_t blk = 0; blk < sh.bh; ++blk)
                for (std::uint32_t j = 0; j < sh.seq; ++j)
                    doc_ids_t.cpu_matrix().span()
                        [static_cast<std::size_t>(blk) * sh.seq + j] = doc_of(j);
            for (auto& v : Q.cpu_matrix().span()) v = dist(rng);
            for (auto& v : K.cpu_matrix().span()) v = dist(rng);
            for (auto& v : Vt.cpu_matrix().span()) v = dist(rng);
            for (auto& v : slopes.cpu_matrix().span()) v = dist(rng) * Scalar{0.1};
            const bool alibi = (mk == nn::expr::FoldAttnMask::Alibi ||
                                mk == nn::expr::FoldAttnMask::AlibiDoc);
            const bool docm  = (mk == nn::expr::FoldAttnMask::Doc ||
                                mk == nn::expr::FoldAttnMask::AlibiDoc);
            std::vector<nn::Tensor> ins{Q, K, Vt};
            if (alibi) ins.push_back(slopes);
            if (docm) { ins.push_back(doc_col); ins.push_back(doc_ids_t); }

            nn::ExprSpec spec = nn::expr::make_fold_attn_o(sh.seq, sh.dk, sh.bh, mk);
            if (auto v = nn::validate_expr_spec(spec, ins.size()); !v)
            {
                check(false, std::string("attn fold ") + mname + " validate: "
                      + v.error().message);
                continue;
            }
            auto out = engine.eval_expr(spec, ins, rows_out, sh.dk);
            if (!out)
            {
                check(false, std::string("attn fold ") + mname + " eval: "
                      + out.error().message);
                continue;
            }

            // 独立公式参考：per (b, i)：s_j = Σ_d Q[(b·dk+d),i]·K[(b·dk+d),j]
            //   [+掩码] → m → p=exp(s−m) → l → O[i,d] = Σ_j p_j·Vt[j,d] / l
            const auto qs = Q.cpu_matrix().span();
            const auto ks = K.cpu_matrix().span();
            const auto vs = Vt.cpu_matrix().span();
            const auto sl = slopes.cpu_matrix().span();
            const auto os = out->cpu_matrix().span();
            Scalar err = 0;
            std::vector<Scalar> srow(sh.seq), prow(sh.seq);
            std::vector<Scalar> ref_out(static_cast<std::size_t>(rows_out) * sh.dk, 0);
            for (std::uint32_t b = 0; b < sh.bh; ++b)
            {
                for (std::uint32_t i = 0; i < sh.seq; ++i)
                {
                    Scalar m = -std::numeric_limits<Scalar>::infinity();
                    for (std::uint32_t j = 0; j < sh.seq; ++j)
                    {
                        Scalar acc = 0;
                        for (std::uint32_t d = 0; d < sh.dk; ++d)
                            acc += qs[(b * sh.dk + d) * sh.seq + i] *
                                   ks[(b * sh.dk + d) * sh.seq + j];
                        if (mk != nn::expr::FoldAttnMask::Plain && j > i)
                            acc = -std::numeric_limits<Scalar>::infinity();
                        // 文档块对角：跨文档 -inf（fold body 内 causal→doc→alibi
                        //   链序；-inf 加有限斜率项不改值）
                        if (docm && ((i < sh.seq / 2) != (j < sh.seq / 2)))
                            acc = -std::numeric_limits<Scalar>::infinity();
                        if (alibi)
                            acc += sl[b] * static_cast<Scalar>(
                                static_cast<int>(j) - static_cast<int>(i));
                        srow[j] = acc;
                        m = std::fmax(m, acc);
                    }
                    Scalar l = 0;
                    for (std::uint32_t j = 0; j < sh.seq; ++j)
                    {
                        prow[j] = std::exp(srow[j] - m);
                        l += prow[j];
                    }
                    for (std::uint32_t d = 0; d < sh.dk; ++d)
                    {
                        Scalar acc = 0;
                        for (std::uint32_t j = 0; j < sh.seq; ++j)
                            acc += prow[j] * vs[(b * sh.seq + j) * sh.dk + d];
                        const Scalar ref = acc / l;
                        const std::size_t oi =
                            static_cast<std::size_t>(b) * sh.seq + i;
                        ref_out[oi * sh.dk + d] = ref;
                        err = std::fmax(err, std::fabs(os[oi * sh.dk + d] - ref) /
                                             std::fmax(Scalar{1}, std::fabs(ref)));
                    }
                }
            }
            check(err <= kTol,
                  std::string("attn fold ") + mname + " bh=" + std::to_string(sh.bh)
                  + " seq=" + std::to_string(sh.seq) + " dk=" + std::to_string(sh.dk)
                  + " err=" + std::to_string(err));
            if (err > kTol)
            {
                std::cout << "      out[0..5]:";
                for (std::size_t i = 0; i < 6 && i < os.size(); ++i)
                    std::cout << " " << os[i];
                std::cout << "\n      ref[0..5]:";
                for (std::size_t i = 0; i < 6 && i < ref_out.size(); ++i)
                    std::cout << " " << ref_out[i];
                std::cout << "\n";
            }
        }
    }

    // ── 7) bin 序列化 roundtrip（v8 causal_skip 写读对称）─────────────────
    {
        const std::string tmp = "nn_expr_fold_roundtrip.tmp.bin";
        nn::fused::ExprRegistry reg;
        reg.add(nn::expr::make_fold_attn_o(
            64, 4, 2, nn::expr::FoldAttnMask::Causal));   // causal_skip=true
        reg.add(nn::expr::make_fold_attn_o(
            64, 4, 2, nn::expr::FoldAttnMask::Plain));    // causal_skip=false
        reg.add(nn::expr::make_fold_rowsum(64));
        const bool wok = nn::fused::write_registry(tmp, reg);
        nn::fused::ExprRegistry back;
        const bool rok = wok && nn::fused::read_registry(tmp, back);
        std::remove(tmp.c_str());
        check(wok && rok && back.specs.size() == reg.specs.size(),
              "registry roundtrip 写读成功 n=" + std::to_string(back.specs.size()));
        if (wok && rok && back.specs.size() == reg.specs.size())
        {
            // key 含 causal_skip 位 → 往返 key 全等即证该位与整段结构未丢
            bool keys_ok = true;
            bool saw_skip = false;
            for (std::size_t i = 0; i < reg.specs.size(); ++i)
            {
                if (nn::expr_spec_key(back.specs[i]) !=
                    nn::expr_spec_key(reg.specs[i]))
                    keys_ok = false;
                if (back.specs[i].fold && back.specs[i].fold->causal_skip)
                    saw_skip = true;
            }
            check(keys_ok, "registry roundtrip key 全等（结构+causal_skip 未丢）");
            check(saw_skip, "registry roundtrip causal_skip=true 读回仍 true");
        }
    }

    return fail;
}
