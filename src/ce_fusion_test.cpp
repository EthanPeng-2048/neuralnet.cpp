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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <span>
#include <thread>
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

// 返回是否通过：调用方必须把 false 计入失败数（此前只累加 g_fail 而不影响
// run_case 返回值/进程退出码，导致 [FAIL] 行被 "ALL PASS" 掩盖 —— 修）。
bool check_matrix(const nn::Matrix& got, const nn::Matrix& ref, const char* msg)
{
    const Scalar err = max_abs_diff(got, ref);
    const bool ok = err < 1e-4f;
    std::printf("[%s] %s  err=%.3e\n", ok ? "PASS" : "FAIL", msg, err);
    if (!ok) ++g_fail;
    return ok;
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
        if (!check_matrix(*dm, ref, nm)) ++fail;
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

    // ── 回归：带"归约 + 后处理"的归约向量表达式（reduce(...) * k + c）──
    //   rsqrt(col_sum(x²)/C + ε) → (1,N)
    // 背景一（CPU）：eval_expr_reduce 的"输出沿归约轴恒定"前置校验原本按
    // **寄存器号**传播所需指令（needed[reg]），而寄存器分配器按 liveness 复用
    // 逐元素寄存器号 —— 后处理指令的 dst 与归约前的 Mul 同号时，归约前的定义
    // 被误判为输出链的一部分（Linear 访问），该类表达式被错误拒绝。现改为按
    // **指令下标**反向切片，CPU 正确求值。
    // 背景二（GPU）：归约融合 shader 尚未正确实现"归约后仍有逐元素后处理"的
    // 形态（实测静默错值），故 GPU 必须**硬报错**而非返回错值；本用例断言
    // 这一点，防止将来退化为静默错误。
    {
        char nm[128];
        std::snprintf(nm, sizeof(nm), "%s IR 归约+后处理 (reduce*k+c)", tag);
        const Scalar inv_c = Scalar{1} / static_cast<Scalar>(C);
        const Scalar eps = Scalar{1e-5f};
        nn::Matrix ref(1, N);
        for (std::size_t i = 0; i < N; ++i)
        {
            Scalar s = 0;
            for (std::size_t r = 0; r < C; ++r)
                s += logits.at_unchecked(r, i) * logits.at_unchecked(r, i);
            ref.set_value_unchecked(0, i, Scalar{1} / std::sqrt(s * inv_c + eps));
        }
        auto r = nn::dsl::compute_reduce(eng,
            nn::dsl::rsqrt(nn::dsl::col_reduce_sum(
                               nn::dsl::leaf(t_logits) * nn::dsl::leaf(t_logits))
                               * nn::dsl::rparam(inv_c)
                           + nn::dsl::rparam(eps)),
            C, N);
        if (eng.device() == nn::Device::CPU)
        {
            check(nm, std::move(r), ref);
        }
        else
        {
            // GPU：必须显式拒绝（"静默错值"是更坏的失败模式）
            const bool rejected = !r;
            std::printf("[%s] %s（GPU 应硬报错而非给出错值）%s%s\n",
                        rejected ? "PASS" : "FAIL", nm,
                        rejected ? "  错误: " : "",
                        rejected ? r.error().message.c_str() : "");
            if (!rejected) ++fail;
        }
    }

    // ── 目标传递（dsl::compute_into）：CPU + GPU 两引擎的原地语义 ──
    //   dst = f(dst, ...)：dst 与输入同 buffer（含"唯一输入就是 dst"的最强别名
    //   情形）。必须与引擎原地原语逐字节一致，且不得越界写。
    {
        const std::size_t R = 24, C = 17;
        nn::Matrix am(R, C), bm(R, C);
        for (std::size_t i = 0; i < am.span().size(); ++i)
        {
            am.span()[i] = Scalar{0.25f} + Scalar(i % 11) * Scalar{0.125f};
            bm.span()[i] = Scalar{-0.5f} + Scalar(i % 7) * Scalar{0.0625f};
        }
        const Scalar s = 0.75f;

        // 参考：v += w / v *= s
        nn::Matrix ref_add = am, ref_scale = am;
        for (std::size_t i = 0; i < ref_add.span().size(); ++i)
        {
            ref_add.span()[i] = am.span()[i] + bm.span()[i];
            ref_scale.span()[i] = am.span()[i] * s;
        }

        // (1) dst += other（dst 同时是输入）
        {
            nn::Tensor dsl_t = nn::Tensor::from_matrix(nn::Matrix(am));
            nn::Tensor prim_t = nn::Tensor::from_matrix(nn::Matrix(am));
            nn::Tensor other = nn::Tensor::from_matrix(nn::Matrix(bm));
            auto r = nn::dsl::compute_into(eng,
                nn::dsl::leaf(dsl_t) + nn::dsl::leaf(other), dsl_t);
            char nm[128];
            if (!r)
            {
                std::printf("[FAIL] %s compute_into (A+=B): %s\n", tag,
                            r.error().message.c_str());
                ++fail;
            }
            else
            {
                auto dm = eng.to_matrix(dsl_t);
                std::snprintf(nm, sizeof(nm), "%s compute_into (A+=B)", tag);
                if (!dm) { std::printf("[FAIL] %s 下载失败\n", nm); ++fail; }
                else if (!check_matrix(*dm, ref_add, nm)) ++fail;
                // 与原地原语逐字节一致
                (void)eng.add_inplace(prim_t, other);
                auto pm = eng.to_matrix(prim_t);
                const bool same_as_prim = pm && max_abs_diff(*dm, *pm) == Scalar{0};
                std::printf("[%s] %s compute_into == add_inplace（逐字节）\n",
                            same_as_prim ? "PASS" : "FAIL", tag);
                if (!same_as_prim) ++fail;
            }
        }
        // (2) dst *= s（唯一输入即 dst：最强别名）
        {
            nn::Tensor dsl_t = nn::Tensor::from_matrix(nn::Matrix(am));
            nn::Tensor prim_t = nn::Tensor::from_matrix(nn::Matrix(am));
            auto r = nn::dsl::compute_into(eng,
                nn::dsl::leaf(dsl_t) * nn::dsl::rparam(s), dsl_t);
            char nm[128];
            if (!r)
            {
                std::printf("[FAIL] %s compute_into (A*=s): %s\n", tag,
                            r.error().message.c_str());
                ++fail;
            }
            else
            {
                auto dm = eng.to_matrix(dsl_t);
                std::snprintf(nm, sizeof(nm), "%s compute_into (A*=s)", tag);
                if (!dm) { std::printf("[FAIL] %s 下载失败\n", nm); ++fail; }
                else if (!check_matrix(*dm, ref_scale, nm)) ++fail;
                (void)eng.scale_inplace(prim_t, s);
                auto pm = eng.to_matrix(prim_t);
                const bool same_as_prim = pm && max_abs_diff(*dm, *pm) == Scalar{0};
                std::printf("[%s] %s compute_into == scale_inplace（逐字节）\n",
                            same_as_prim ? "PASS" : "FAIL", tag);
                if (!same_as_prim) ++fail;
            }
        }
        // (3) 归约表达式必须被拒绝（compute_into 只支持逐元素）
        {
            nn::Tensor dsl_t = nn::Tensor::from_matrix(nn::Matrix(am));
            auto r = nn::dsl::compute_into(eng,
                nn::dsl::col_reduce_sum(nn::dsl::leaf(dsl_t)), dsl_t);
            const bool rejected = !r;
            std::printf("[%s] %s compute_into 拒绝归约表达式\n",
                        rejected ? "PASS" : "FAIL", tag);
            if (!rejected) ++fail;
        }
    }

    // ── S7 IR 组合：loss_vec / grad（RowGather + Row 操作数）──
    //   loss_vec = (rg(logits) - cb(col_max) - log(denom)) * cb(mask)  （(1,N)）
    //   grad     = (exp(logits-cb(col_max))/cb(denom)
    //               - select(Row==cb(labels),1,0)) * cb(mask) * rp(inv)  （(C,N)）
    // 注：inv_num_valid 由 RParam 承载（运行时标量，不进 expr_spec_key）→ 与
    //     整条逐元素链融合为单 kernel（原先在表达式后补一次 scale_inplace）。
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
                    * nn::dsl::col_broadcast(t_mask)
                    * nn::dsl::rparam(inv_num_valid),
                    C, N);
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
                    * nn::dsl::col_broadcast(t_mask)
                    * nn::dsl::rparam(inv_num_valid),
                    C, N);
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
                    * nn::dsl::col_broadcast(t_mask)
                    * nn::dsl::rparam(inv_num_valid),
                    C, N);
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
                if (!check_matrix(*dg, g_ref, nm)) ++fail;
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

    // ── P0-2：异步标量回读（submit/poll）与同步下载必须逐值一致 ─────────
    // 同一组 logits 反复求 loss，按训练热循环的真实调用序：
    //   begin_batch → forward_sparse_sum（录制）→ cast(F32) → flush_batch（提交）
    //   → submit_scalar_readback（主帧提交后，不等待）→ poll 就绪后取回。
    // 与 forward_sparse（同步 to_matrix）逐值比较；迭代 24 轮 > 槽位数（8），
    // 覆盖槽位环形复用——若回读串槽/读到陈旧值，此处必现差异。
    {
        std::cout << "\n── P0-2 异步标量回读 vs 同步下载 ──\n";
        constexpr std::size_t C = 37, N = 64;
        nn::Matrix logits(C, N), lm(1, N);
        std::mt19937 rng(12345);
        std::uniform_real_distribution<Scalar> dist(-2.0f, 2.0f);
        for (std::size_t r = 0; r < C; ++r)
            for (std::size_t c = 0; c < N; ++c)
                logits.set_value_unchecked(r, c, dist(rng));
        std::vector<std::size_t> labels(N);
        for (std::size_t i = 0; i < N; ++i)
            labels[i] = static_cast<std::size_t>(rng() % C);
        for (std::size_t i = 0; i < N; ++i)
            lm.set_value_unchecked(0, i, (i % 5 == 0) ? Scalar{0} : Scalar{1});

        auto t_logits_r = gpu_engine.from_matrix(logits);
        if (!t_logits_r)
        {
            std::printf("[FAIL] async readback: from_matrix 失败\n");
            ++fail;
        }
        else
        {
            nn::CrossEntropyLoss ce;
            const std::span<const Scalar> mask_span(lm.span());
            const std::size_t slots =
                std::max<std::size_t>(gpu_engine.scalar_readback_slots(), 1);
            constexpr int ITERS = 24;
            Scalar max_diff = 0;
            int mismatches = 0, spins_total = 0;

            for (int it = 0; it < ITERS; ++it)
            {
                // 同步参考（to_matrix 下载）
                auto ref = ce.forward_sparse(gpu_engine, *t_logits_r, labels,
                                             mask_span, C);
                if (!ref)
                {
                    std::printf("[FAIL] async readback: forward_sparse: %s\n",
                                ref.error().message.c_str());
                    ++mismatches;
                    break;
                }

                // 异步路径（与 text_train 完全同序）
                auto bb = gpu_engine.begin_batch();
                std::size_t nv = 0;
                auto sum_r = ce.forward_sparse_sum(gpu_engine, *t_logits_r, labels,
                                                   mask_span, C, nv);
                if (bb && sum_r)
                {
                    auto f32_r = gpu_engine.cast(*sum_r, nn::Precision::F32);
                    if (f32_r)
                    {
                        auto fb = gpu_engine.flush_batch();   // 提交主帧（不等待）
                        const std::size_t slot =
                            static_cast<std::size_t>(it) % slots;
                        auto sr = gpu_engine.submit_scalar_readback(slot, *f32_r);
                        if (!fb || !sr)
                        {
                            std::printf("[FAIL] async readback: submit 失败\n");
                            ++mismatches;
                            break;
                        }
                        Scalar got = Scalar{0};
                        int spins = 0;
                        for (;;)
                        {
                            auto pr = gpu_engine.poll_scalar_readback(slot, got);
                            if (!pr)
                            {
                                std::printf("[FAIL] async readback: poll 失败: %s\n",
                                            pr.error().message.c_str());
                                ++mismatches;
                                break;
                            }
                            if (*pr) break;
                            ++spins;
                            std::this_thread::sleep_for(
                                std::chrono::microseconds(200));
                            if (spins > 20000)   // 4s 上限，防死等
                            {
                                std::printf("[FAIL] async readback: poll 超时\n");
                                ++mismatches;
                                break;
                            }
                        }
                        // 与 text_train 一致：每步 begin/end 配对（end 不等待）
                        auto eb = gpu_engine.end_batch();
                        if (!eb)
                        {
                            std::printf("[FAIL] async readback: end_batch: %s\n",
                                        eb.error().message.c_str());
                            ++mismatches;
                            break;
                        }
                        spins_total += spins;
                        const Scalar got_loss =
                            (nv > 0) ? -got / static_cast<Scalar>(nv) : Scalar{0};
                        const Scalar d = std::fabs(got_loss - *ref);
                        if (d > max_diff) max_diff = d;
                        if (d > 1e-5f) ++mismatches;
                    }
                    else
                    {
                        std::printf("[FAIL] async readback: cast 失败\n");
                        ++mismatches;
                        break;
                    }
                }
                else
                {
                    std::printf("[FAIL] async readback: begin_batch/sum 失败\n");
                    ++mismatches;
                    break;
                }
            }

            const bool ok = (mismatches == 0);
            std::printf("[%s] GPU 异步回读 vs 同步（%d 轮 / %zu 槽）  max_diff=%.3e"
                        "  spin_total=%d\n",
                        ok ? "PASS" : "FAIL", ITERS, slots, max_diff, spins_total);
            if (!ok) fail += mismatches;
        }
    }
#endif

    std::cout << (fail == 0 ? "\nALL PASS\n" : "\nFAILED\n");
    return fail == 0 ? 0 : 1;
}

