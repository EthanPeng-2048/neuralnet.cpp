// ───────────────────────────────────────────────────────────────────────────
//  ce_fusion_test.cpp — M5：列式 softmax 融合原语 + 稀疏交叉熵端到端
//
//  验证：
//    - col_softmax_denom：denom[c] = Σ_r exp(logits[r][c] - col_max[c]) → 已删除（IR 融合替代）
//    - col_softmax_sparse_forward：单 kernel 稠密梯度 + 标签位置 loss_vec → 已删除（IR 融合替代）
//      （不物化全 softmax），含 mask / 越界标签处理
//    - CrossEntropyLoss::forward_sparse 端到端（loss + grad vs 参考）
//  CPU 引擎 vs 手写参考；GPU（Vulkan）融合 shader vs CPU 参考。
//
//  用法：ce_fusion_test
// ───────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

using nn::Scalar;

int g_fail = 0;

Scalar max_abs_diff(const nn::Matrix& a, const nn::Matrix& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return 1e30f;
    Scalar e = 0;
    for (std::size_t i = 0; i < a.span().size(); ++i)
        e = std::max(e, std::fabs(a.span()[i] - b.span()[i]));
    return e;
}

void check_matrix(const nn::Matrix& got, const nn::Matrix& ref, const char* msg)
{
    const Scalar err = max_abs_diff(got, ref);
    const bool ok = err < 1e-4f;
    std::printf("[%s] %s  err=%.3e\n", ok ? "PASS" : "FAIL", msg, err);
    if (!ok) ++g_fail;
}

// ── 参考实现 ──────────────────────────────────────────────────────────────

// col_softmax_denom 参考
nn::Matrix ref_denom(const nn::Matrix& logits, const nn::Matrix& col_max)
{
    const std::size_t C = logits.rows(), N = logits.cols();
    nn::Matrix out(1, N);
    for (std::size_t i = 0; i < N; ++i)
    {
        const Scalar mv = col_max.at_unchecked(0, i);
        Scalar acc = 0;
        for (std::size_t r = 0; r < C; ++r)
            acc += std::exp(logits.at_unchecked(r, i) - mv);
        out.set_value_unchecked(0, i, acc);
    }
    return out;
}

// col_softmax_sparse_forward 参考：grad 与 loss_vec
nn::Matrix ref_sparse_forward(
    const nn::Matrix& logits, const std::vector<std::size_t>& labels,
    const std::vector<Scalar>* loss_mask, std::size_t vocab_size,
    Scalar inv_num_valid, nn::Matrix& lv_out)
{
    const std::size_t C = logits.rows(), N = logits.cols();
    nn::Matrix grad(C, N);
    nn::Matrix lv(1, N);
    for (std::size_t i = 0; i < N; ++i)
    {
        const bool masked = loss_mask && (*loss_mask)[i] < Scalar{0.5};
        const std::size_t lbl = labels[i];
        const bool valid = !masked && lbl < vocab_size;
        if (!valid)
        {
            for (std::size_t r = 0; r < C; ++r)
                grad.set_value_unchecked(r, i, Scalar{0});
            lv.set_value_unchecked(0, i, Scalar{0});
            continue;
        }
        Scalar mv = -std::numeric_limits<Scalar>::infinity();
        for (std::size_t r = 0; r < C; ++r)
            mv = std::max(mv, logits.at_unchecked(r, i));
        Scalar denom = 0;
        for (std::size_t r = 0; r < C; ++r)
            denom += std::exp(logits.at_unchecked(r, i) - mv);
        for (std::size_t r = 0; r < C; ++r)
            grad.set_value_unchecked(r, i,
                inv_num_valid * std::exp(logits.at_unchecked(r, i) - mv) / denom);
        grad.set_value_unchecked(lbl, i, grad.at_unchecked(lbl, i) - inv_num_valid);
        lv.set_value_unchecked(0, i, logits.at_unchecked(lbl, i) - mv - std::log(denom));
    }
    lv_out = std::move(lv);
    return grad;
}

// ── 测试主体 ──────────────────────────────────────────────────────────────
int run_case(nn::ComputeEngine& eng, const char* tag)
{
    const std::size_t C = 8, N = 12;   // classes=8, batch=12
    std::mt19937 rng(2026);
    std::uniform_real_distribution<Scalar> dist(-1.5f, 1.5f);

    nn::Matrix logits(C, N);
    for (auto& x : logits.span()) x = dist(rng);
    nn::Matrix col_max(1, N);
    for (std::size_t i = 0; i < N; ++i)
    {
        Scalar mx = -std::numeric_limits<Scalar>::infinity();
        for (std::size_t r = 0; r < C; ++r)
            mx = std::max(mx, logits.at_unchecked(r, i));
        col_max.set_value_unchecked(0, i, mx);
    }

    // 标签：大多数有效，2 个越界（vocab_size=6，标签 6/7 越界）
    const std::size_t vocab_size = 6;
    std::vector<std::size_t> labels(N);
    for (std::size_t i = 0; i < N; ++i)
        labels[i] = static_cast<std::size_t>(dist(rng) * 3.0f + 3.0f);  // 0..5
    labels[2] = 6;  // 越界
    labels[9] = 7;  // 越界

    // mask：2 列被 mask（< 0.5）
    std::vector<Scalar> loss_mask(N, 1.0f);
    loss_mask[4] = 0.0f;
    loss_mask[10] = 0.3f;

    // num_valid：N - 越界2 - mask2 = 8
    std::size_t num_valid = 0;
    for (std::size_t i = 0; i < N; ++i)
    {
        const bool masked = loss_mask[i] < Scalar{0.5};
        if (!masked && labels[i] < vocab_size) ++num_valid;
    }
    const Scalar inv_num_valid = Scalar{1} / static_cast<Scalar>(num_valid);

    int fail = 0;

    const auto check = [&](const char* nm, nn::Result<nn::Tensor>&& r, const nn::Matrix& ref)
    {
        if (!r) { std::printf("[FAIL] %s: %s\n", nm, r.error().message.c_str()); ++fail; return; }
        auto dm = eng.to_matrix(*r);
        if (!dm) { std::printf("[FAIL] %s: 下载失败: %s\n", nm, dm.error().message.c_str()); ++fail; return; }
        check_matrix(*dm, ref, nm);
    };

    const nn::Tensor t_logits = nn::Tensor::from_matrix(nn::Matrix(logits));
    const nn::Tensor t_colmax = nn::Tensor::from_matrix(nn::Matrix(col_max));

    // ── S7 IR 组合：denom = col_sum(exp(logits - cb(col_max))) ──
    {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "%s IR denom", tag);
        const nn::Matrix ref = ref_denom(logits, col_max);
        auto r = nn::dsl::compute_reduce(eng,
            nn::dsl::col_reduce_sum(
                nn::dsl::exp(nn::dsl::leaf(t_logits) - nn::dsl::col_broadcast(t_colmax))),
            C, N);
        check(nm, std::move(r), ref);
    }

    // ── S7 IR 组合：loss_vec / grad（RowGather + Row 操作数）──
    //   loss_vec = (rg(logits) - cb(col_max) - log(denom)) * cb(mask)  （(1,N)）
    //   grad     = (exp(logits-cb(col_max))/cb(denom)
    //               - select(Row==cb(labels),1,0)) * cb(mask) * inv     （(C,N)）
    {
        nn::Matrix labels_m(1, N);
        for (std::size_t i = 0; i < N; ++i)
            labels_m.set_value_unchecked(0, i, static_cast<Scalar>(labels[i]));
        const nn::Tensor t_labels = nn::Tensor::from_matrix(std::move(labels_m));

        // 无 mask
        {
            char nm[128];
            nn::Matrix lv_ref;
            const nn::Matrix g_ref = ref_sparse_forward(
                logits, labels, nullptr, vocab_size, inv_num_valid, lv_ref);
            // S7 Layer 行为：mask = valid 修正（越界 label → 0）
            nn::Matrix mask_m(1, N);
            for (std::size_t i = 0; i < N; ++i)
                mask_m.set_value_unchecked(0, i,
                    labels[i] < vocab_size ? Scalar{1} : Scalar{0});
            const nn::Tensor t_mask = nn::Tensor::from_matrix(std::move(mask_m));
            auto denom = nn::dsl::compute_reduce(eng,
                nn::dsl::col_reduce_sum(nn::dsl::exp(
                    nn::dsl::leaf(t_logits) - nn::dsl::col_broadcast(t_colmax))),
                C, N);
            if (!denom) { std::printf("[FAIL] %s denom: %s\n", tag, denom.error().message.c_str()); ++fail; }
            else
            {
                auto lv = nn::dsl::compute(eng,
                    (nn::dsl::row_gather(t_logits, t_labels) - nn::dsl::col_broadcast(t_colmax)
                     - nn::dsl::log(nn::dsl::leaf(*denom))) * nn::dsl::col_broadcast(t_mask),
                    1, N);
                std::snprintf(nm, sizeof(nm), "%s IR loss_vec (no mask)", tag);
                check(nm, std::move(lv), lv_ref);
                auto g = nn::dsl::compute(eng,
                    (nn::dsl::exp(nn::dsl::leaf(t_logits) - nn::dsl::col_broadcast(t_colmax))
                        / nn::dsl::col_broadcast(*denom)
                     - nn::dsl::select(nn::dsl::row() == nn::dsl::col_broadcast(t_labels),
                                       Scalar{1}, Scalar{0}))
                    * nn::dsl::col_broadcast(t_mask),
                    C, N);
                { auto gs = eng.scale_inplace(*g, inv_num_valid); if (!gs) { std::printf("[FAIL] scale: %s\n", gs.error().message.c_str()); ++fail; } }
                std::snprintf(nm, sizeof(nm), "%s IR grad (no mask)", tag);
                check(nm, std::move(g), g_ref);
            }
        }
        // 有 mask
        {
            char nm[128];
            nn::Matrix mask_m(1, N);
            for (std::size_t i = 0; i < N; ++i)
                mask_m.set_value_unchecked(0, i,
                    (loss_mask[i] >= Scalar{0.5} && labels[i] < vocab_size)
                        ? Scalar{1} : Scalar{0});
            const nn::Tensor t_mask = nn::Tensor::from_matrix(std::move(mask_m));
            nn::Matrix lv_ref;
            const nn::Matrix g_ref = ref_sparse_forward(
                logits, labels, &loss_mask, vocab_size, inv_num_valid, lv_ref);
            auto denom = nn::dsl::compute_reduce(eng,
                nn::dsl::col_reduce_sum(nn::dsl::exp(
                    nn::dsl::leaf(t_logits) - nn::dsl::col_broadcast(t_colmax))),
                C, N);
            if (!denom) { std::printf("[FAIL] %s denom: %s\n", tag, denom.error().message.c_str()); ++fail; }
            else
            {
                auto lv = nn::dsl::compute(eng,
                    (nn::dsl::row_gather(t_logits, t_labels) - nn::dsl::col_broadcast(t_colmax)
                     - nn::dsl::log(nn::dsl::leaf(*denom))) * nn::dsl::col_broadcast(t_mask),
                    1, N);
                std::snprintf(nm, sizeof(nm), "%s IR loss_vec (mask)", tag);
                check(nm, std::move(lv), lv_ref);
                auto g = nn::dsl::compute(eng,
                    (nn::dsl::exp(nn::dsl::leaf(t_logits) - nn::dsl::col_broadcast(t_colmax))
                        / nn::dsl::col_broadcast(*denom)
                     - nn::dsl::select(nn::dsl::row() == nn::dsl::col_broadcast(t_labels),
                                       Scalar{1}, Scalar{0}))
                    * nn::dsl::col_broadcast(t_mask),
                    C, N);
                { auto gs = eng.scale_inplace(*g, inv_num_valid); if (!gs) { std::printf("[FAIL] scale: %s\n", gs.error().message.c_str()); ++fail; } }
                std::snprintf(nm, sizeof(nm), "%s IR grad (mask)", tag);
                check(nm, std::move(g), g_ref);
            }
        }
        // 全越界（num_valid=0）：mask 修正为 0 → grad 全 0、loss_vec 0
        {
            char nm[128];
            std::vector<std::size_t> bad(N, 99);
            nn::Matrix bad_m(1, N);
            for (std::size_t i = 0; i < N; ++i)
                bad_m.set_value_unchecked(0, i, static_cast<Scalar>(bad[i]));
            const nn::Tensor t_bad = nn::Tensor::from_matrix(std::move(bad_m));
            nn::Matrix lv_ref;
            const nn::Matrix g_ref = ref_sparse_forward(
                logits, bad, nullptr, vocab_size, Scalar{0}, lv_ref);
            // 越界 label 修正为 0（Layer 行为）+ mask 全 0（越界无效）
            nn::Matrix mask_m(1, N);
            for (std::size_t i = 0; i < N; ++i)
                mask_m.set_value_unchecked(0, i, Scalar{0});
            const nn::Tensor t_mask = nn::Tensor::from_matrix(std::move(mask_m));
            auto denom = nn::dsl::compute_reduce(eng,
                nn::dsl::col_reduce_sum(nn::dsl::exp(
                    nn::dsl::leaf(t_logits) - nn::dsl::col_broadcast(t_colmax))),
                C, N);
            if (!denom) { std::printf("[FAIL] %s denom: %s\n", tag, denom.error().message.c_str()); ++fail; }
            else
            {
                auto lv = nn::dsl::compute(eng,
                    (nn::dsl::row_gather(t_logits, t_bad) - nn::dsl::col_broadcast(t_colmax)
                     - nn::dsl::log(nn::dsl::leaf(*denom))) * nn::dsl::col_broadcast(t_mask),
                    1, N);
                std::snprintf(nm, sizeof(nm), "%s IR loss_vec (all invalid)", tag);
                check(nm, std::move(lv), lv_ref);
                auto g = nn::dsl::compute(eng,
                    (nn::dsl::exp(nn::dsl::leaf(t_logits) - nn::dsl::col_broadcast(t_colmax))
                        / nn::dsl::col_broadcast(*denom)
                     - nn::dsl::select(nn::dsl::row() == nn::dsl::col_broadcast(t_bad),
                                       Scalar{1}, Scalar{0}))
                    * nn::dsl::col_broadcast(t_mask),
                    C, N);
                { auto gs = eng.scale_inplace(*g, inv_num_valid); if (!gs) { std::printf("[FAIL] scale: %s\n", gs.error().message.c_str()); ++fail; } }
                std::snprintf(nm, sizeof(nm), "%s IR grad (all invalid)", tag);
                check(nm, std::move(g), g_ref);
            }
        }
    }

    // ── CrossEntropyLoss::forward_sparse 端到端（loss + grad vs 参考） ──
    {
        char nm[128];
        nn::CrossEntropyLoss loss_layer;
        auto loss_r = loss_layer.forward_sparse(eng, t_logits, labels, loss_mask, vocab_size);
        if (!loss_r) { std::printf("[FAIL] %s: %s\n", tag, loss_r.error().message.c_str()); ++fail; }
        else
        {
            // 参考 loss = -(1/num_valid) Σ log_softmax[labels[i],i]（有效位置）
            Scalar ref_loss = 0;
            for (std::size_t i = 0; i < N; ++i)
            {
                const bool masked = loss_mask[i] < Scalar{0.5};
                if (masked || labels[i] >= vocab_size) continue;
                Scalar mv = -std::numeric_limits<Scalar>::infinity();
                for (std::size_t r = 0; r < C; ++r)
                    mv = std::max(mv, logits.at_unchecked(r, i));
                Scalar denom = 0;
                for (std::size_t r = 0; r < C; ++r)
                    denom += std::exp(logits.at_unchecked(r, i) - mv);
                ref_loss -= logits.at_unchecked(labels[i], i) - mv - std::log(denom);
            }
            ref_loss /= static_cast<Scalar>(num_valid);
            const Scalar err = std::fabs(*loss_r - ref_loss);
            const bool ok = err < 1e-4f;
            std::printf("[%s] %s CE loss  got=%.6f ref=%.6f err=%.3e\n",
                        ok ? "PASS" : "FAIL", tag, *loss_r, ref_loss, err);
            if (!ok) ++fail;

            // grad vs 参考（softmax - one_hot）/num_valid
            auto grad_r = loss_layer.backward();
            auto dg = eng.to_matrix(*grad_r);
            if (!dg) { std::printf("[FAIL] %s CE grad: 下载失败\n", tag); ++fail; }
            else
            {
                nn::Matrix lv_ref;
                const nn::Matrix g_ref = ref_sparse_forward(
                    logits, labels, &loss_mask, vocab_size, inv_num_valid, lv_ref);
                std::snprintf(nm, sizeof(nm), "%s CE grad", tag);
                check_matrix(*dg, g_ref, nm);
            }
        }
    }

    return fail;
}

int main()
{
    std::cout << "========================================\n"
              << "  M5 列式 softmax 融合 + 稀疏交叉熵验证\n"
              << "========================================\n";

    nn::CpuEngine cpu_engine;
    int fail = run_case(cpu_engine, "CPU");

#ifndef NN_HAS_VULKAN
    std::cout << "[SKIP] 无 Vulkan，跳过 GPU 部分\n";
#else
    auto& backend = nn::GpuBackend::instance();
    auto init_r = backend.initialize();
    if (!init_r)
    {
        std::cerr << "GPU 初始化失败: " << init_r.error().message << "\n";
        return 1;
    }
    nn::GpuEngine gpu_engine(backend);
    fail += run_case(gpu_engine, "GPU");
#endif

    std::cout << (fail == 0 ? "\nALL PASS\n" : "\nFAILED\n");
    return fail == 0 ? 0 : 1;
}

