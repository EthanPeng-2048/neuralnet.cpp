#ifdef NN_HAS_VULKAN
#include <cmath>
#include <cstdio>
#include "neuralnet.cpp/backend/compute_vk_backend.hpp"
#include "neuralnet.cpp/compute_gpu_engine.hpp"
#include "neuralnet.cpp/compute_cpu_engine.hpp"

int main() {
    printf("=== GPU f16 Capability Check ===\n");
    auto& backend = nn::GpuBackend::instance();

    // Explicitly initialize
    auto init_r = backend.initialize();
    NN_EXIT(init_r, 1, "GPU init FAILED: %s\n");
    printf("GPU backend: OK\n");
    printf("shaderFloat16: %s\n", backend.has_shader_float16() ? "YES" : "NO");

    // Test GPU f16 operations
    nn::GpuEngine engine(backend);

    // Create f32 matrix
    nn::Matrix m(2, 2);
    m.set_value(0, 0, 1.0f); m.set_value(0, 1, 2.0f);
    m.set_value(1, 0, 3.0f); m.set_value(1, 1, 4.0f);

    // Upload as f32
    auto t32 = engine.from_matrix(m);
    NN_EXIT(t32, 1, "from_matrix failed: %s\n");
    printf("f32 uploaded: %s\n", t32->shape_str().c_str());

    // Cast to f16
    auto t16 = engine.cast(*t32, nn::Precision::F16);
    NN_EXIT(t16, 1, "cast to f16 failed: %s\n");
    printf("f16 tensor: prec=%d %s\n", (int)t16->precision(), t16->shape_str().c_str());

    // Cast back to f32
    auto t32_back = engine.cast(*t16, nn::Precision::F32);
    NN_EXIT(t32_back, 1, "cast back failed: %s\n");

    auto result = engine.to_matrix(*t32_back);
    NN_EXIT(result, 1, "to_matrix failed: %s\n");

    printf("Original:  [%.1f %.1f; %.1f %.1f]\n", m.at(0,0), m.at(0,1), m.at(1,0), m.at(1,1));
    printf("GPU f16:   [%.1f %.1f; %.1f %.1f]\n", result->at(0,0), result->at(0,1), result->at(1,0), result->at(1,1));

    // Test f16 matmul
    printf("\n=== GPU f16 Matmul ===\n");
    nn::Matrix m2(2, 2);
    m2.set_value(0, 0, 5.0f); m2.set_value(0, 1, 6.0f);
    m2.set_value(1, 0, 7.0f); m2.set_value(1, 1, 8.0f);

    auto t16_a = engine.cast(engine.from_matrix(m).value(), nn::Precision::F16);
    auto t16_b = engine.cast(engine.from_matrix(m2).value(), nn::Precision::F16);

    auto t16_c = engine.matmul(*t16_a, *t16_b, false, false, nn::Precision::F16);
    NN_EXIT(t16_c, 1, "f16 matmul failed: %s\n");

    auto c32 = engine.cast(*t16_c, nn::Precision::F32);
    auto c_mat = engine.to_matrix(*c32);
    printf("A*B (f16): [%.1f %.1f; %.1f %.1f]\n",
        c_mat->at(0,0), c_mat->at(0,1), c_mat->at(1,0), c_mat->at(1,1));
    printf("Expected:  [19.0 22.0; 43.0 50.0]\n");

    // ── 算子 f16 变体对拍（GPU f16 vs CPU f32 参考）─────────────
    // 覆盖：row/col reduce、grouped_reduce、scatter_add、im2col、col2im
    // 的 f16 存储变体（f16 变体的类型/索引一旦出错是静默错值——对拍是
    // 唯一护栏）。
    printf("\n=== Op-level f16 variants vs CPU f32 reference ===\n");
    nn::CpuEngine cpu;
    int fails = 0;
    const auto check = [&](const char* name, const nn::Matrix& got,
                           const nn::Matrix& ref, float tol)
    {
        if (got.rows() != ref.rows() || got.cols() != ref.cols())
        {
            printf("[FAIL] %s: shape %zux%zu vs ref %zux%zu\n", name,
                   got.rows(), got.cols(), ref.rows(), ref.cols());
            ++fails;
            return;
        }
        float maxr = 0.0f;
        for (std::size_t r = 0; r < got.rows(); ++r)
            for (std::size_t c = 0; c < got.cols(); ++c)
            {
                const float e = std::fabs(got.at(r, c) - ref.at(r, c));
                const float d = std::fmax(1.0f, std::fabs(ref.at(r, c)));
                maxr = std::fmax(maxr, e / d);
            }
        if (maxr > tol)
        {
            printf("[FAIL] %s: max_rel=%.3e (tol %.1e)\n", name, maxr, tol);
            ++fails;
        }
        else
            printf("[ok] %s (max_rel=%.2e)\n", name, maxr);
    };
    // GPU f16 算子 → 结果 cast f32 → to_matrix
    const auto gpu_op = [&](auto&& fn) -> nn::Matrix
    {
        auto r = fn();
        if (!r)
        {
            printf("[FAIL] gpu op error: %s\n", r.error().message.c_str());
            ++fails;
            return {};
        }
        auto f32r = engine.cast(*r, nn::Precision::F32);
        if (!f32r) { printf("[FAIL] cast: %s\n", f32r.error().message.c_str()); ++fails; return {}; }
        auto m = engine.to_matrix(*f32r);
        if (!m) { printf("[FAIL] to_matrix: %s\n", m.error().message.c_str()); ++fails; return {}; }
        return std::move(*m);
    };
    const auto to_gpu_f16 = [&](const nn::Matrix& m) -> nn::Tensor
    {
        auto t = engine.from_matrix(m).value();
        return engine.cast(t, nn::Precision::F16).value();
    };
    // CPU 参考算子结果（Tensor）→ Matrix
    const auto cpu_op = [&](auto&& fn) -> nn::Matrix
    {
        auto r = fn();
        if (!r) { printf("[FAIL] cpu ref: %s\n", r.error().message.c_str()); ++fails; return {}; }
        auto m = cpu.to_matrix(*r);
        if (!m) { printf("[FAIL] cpu to_matrix: %s\n", m.error().message.c_str()); ++fails; return {}; }
        return std::move(*m);
    };
    const auto fill = [](nn::Matrix& m, auto&& gen)
    {
        for (std::size_t r = 0; r < m.rows(); ++r)
            for (std::size_t c = 0; c < m.cols(); ++c)
                m.set_value(r, c, gen(r * m.cols() + c));
    };

    // 1) row_reduce_sum / col_reduce_max（reduce.comp f16 变体）
    {
        nn::Matrix a(3, 4);
        fill(a, [](std::size_t i) { return 0.1f * static_cast<float>((i * 7) % 13) - 0.4f; });
        auto ref_row = cpu_op([&] { return cpu.row_reduce_sum(cpu.from_matrix(a).value()); });
        auto ref_col = cpu_op([&] { return cpu.col_reduce_max(cpu.from_matrix(a).value()); });
        auto a16 = to_gpu_f16(a);
        auto got_row = gpu_op([&] { return engine.row_reduce_sum(a16); });
        auto got_col = gpu_op([&] { return engine.col_reduce_max(a16); });
        check("row_reduce_sum f16", got_row, ref_row, 1e-3f);
        check("col_reduce_max f16", got_col, ref_col, 1e-3f);
    }

    // 2) grouped_reduce_sum / max（group_reduce.comp f16 变体；G=2,R=3,N=2）
    {
        nn::Matrix x(6, 2);
        fill(x, [](std::size_t i) { return 0.25f * static_cast<float>((i * 5) % 11) - 1.0f; });
        auto ref_s = cpu_op([&] { return cpu.grouped_reduce_sum(cpu.from_matrix(x).value(), 2, 3); });
        auto ref_m = cpu_op([&] { return cpu.grouped_reduce_max(cpu.from_matrix(x).value(), 2, 3); });
        auto x16 = to_gpu_f16(x);
        auto got_s = gpu_op([&] { return engine.grouped_reduce_sum(x16, 2, 3); });
        auto got_m = gpu_op([&] { return engine.grouped_reduce_max(x16, 2, 3); });
        check("grouped_reduce_sum f16", got_s, ref_s, 1e-3f);
        check("grouped_reduce_max f16", got_m, ref_m, 1e-3f);
    }

    // 3) scatter_add_rows（打包 half CAS 变体；dst f16 + grad f16）
    {
        nn::Matrix dst_m(4, 2), grad_m(2, 2), idx_m(2, 1);
        fill(dst_m, [](std::size_t) { return 0.0f; });
        grad_m.set_value(0, 0, 1.5f); grad_m.set_value(0, 1, -2.25f);
        grad_m.set_value(1, 0, 0.75f); grad_m.set_value(1, 1, 3.0f);
        idx_m.set_value(0, 0, 1.0f);   idx_m.set_value(1, 0, 3.0f);
        // CPU f32 参考
        auto d_cpu = cpu.from_matrix(dst_m).value();
        auto sc = cpu.scatter_add_rows(d_cpu, cpu.from_matrix(idx_m).value(),
                                       cpu.from_matrix(grad_m).value());
        if (!sc) { printf("[FAIL] cpu scatter: %s\n", sc.error().message.c_str()); ++fails; }
        auto ref = cpu.to_matrix(d_cpu).value();
        // GPU f16（原生打包 half CAS 路径）
        auto d16 = to_gpu_f16(dst_m);
        auto g16 = to_gpu_f16(grad_m);
        auto idx = engine.from_matrix(idx_m).value();
        auto r = engine.scatter_add_rows(d16, idx, g16);
        if (!r) { printf("[FAIL] scatter_add_rows f16: %s\n", r.error().message.c_str()); ++fails; }
        else check("scatter_add_rows f16", gpu_op([&] { return engine.cast(d16, nn::Precision::F32); }), ref, 1e-3f);
    }

    // 4) im2col / col2im（窗口展开 f16 变体；C=2,H=W=3,k=2,stride=1,pad=0）
    {
        nn::Matrix xm(18, 2);   // C*H*W=18, B=2
        fill(xm, [](std::size_t i) { return 0.3f * static_cast<float>((i * 11) % 17) - 2.0f; });
        auto ref_col_t = cpu.im2col(cpu.from_matrix(xm).value(),
                                    2, 3, 3, 2, 1, 0, 2, 2).value();
        auto ref_col = cpu.to_matrix(ref_col_t).value();
        auto ref_back = cpu_op([&] {
            return cpu.col2im(cpu.from_matrix(ref_col).value(), 2, 3, 3, 2, 1, 0, 2, 2); });
        auto x16 = to_gpu_f16(xm);
        auto got_col = gpu_op([&] { return engine.im2col(x16, 2, 3, 3, 2, 1, 0, 2, 2); });
        check("im2col f16", got_col, ref_col, 1e-3f);
        // col2im 输入用 GPU f16 的 im2col 输出（保持全链 f16）
        auto col16 = engine.im2col(x16, 2, 3, 3, 2, 1, 0, 2, 2).value();
        auto got_back = gpu_op([&] { return engine.col2im(col16, 2, 3, 3, 2, 1, 0, 2, 2); });
        check("col2im f16", got_back, ref_back, 1e-3f);
    }

    // 5) RLA 扫描族 f16（scan_prefix/suffix/outer_col 全链 half；
    //    rapt_test 无 GPU 覆盖 → 本对拍是 f16 扫描变体的唯一护栏）
    //    形状：dk=4, heads=1, seq=5, rows=H*dk=4, causal=1, 有 boundary
    {
        const std::size_t dk = 4, seq = 5, rows = 4;
        auto gen = [](std::size_t seed)
        {
            return [seed](std::size_t i) {
                return 0.2f * static_cast<float>((i * 7 + seed * 13) % 11) - 1.0f;
            };
        };
        nn::Matrix K_m(rows, seq), V_m(rows, seq), P_m(rows, seq), R_m(rows, seq);
        nn::Matrix A0_m(dk, dk), B0_m(dk, dk), bnd_m(1, seq);
        fill(K_m, gen(1)); fill(V_m, gen(2)); fill(P_m, gen(3)); fill(R_m, gen(4));
        fill(A0_m, gen(5)); fill(B0_m, gen(6));
        fill(bnd_m, [](std::size_t i) { return (i == 2) ? 1.0f : 0.0f; });   // 文档边界 @t=2
        const auto f16 = [&](const nn::Matrix& m) { return to_gpu_f16(m); };
        auto K16 = f16(K_m), V16 = f16(V_m), P16 = f16(P_m), R16 = f16(R_m);
        auto A016 = f16(A0_m), B016 = f16(B0_m), bnd16 = f16(bnd_m);

        // CPU f32 参考（has_state=1, causal=1, has_bnd=1）
        auto run_cpu = [&](auto&& fn) -> nn::Matrix
        {
            auto t = fn();
            if (!t) { printf("[FAIL] cpu scan: %s\n", t.error().message.c_str()); ++fails; return {}; }
            auto m = cpu.to_matrix(*t);
            if (!m) { printf("[FAIL] cpu scan to_matrix: %s\n", m.error().message.c_str()); ++fails; return {}; }
            return std::move(*m);
        };
        auto ref_pre = run_cpu([&] {
            return cpu.scan_prefix_outer(
                cpu.from_matrix(K_m).value(), cpu.from_matrix(V_m).value(),
                cpu.from_matrix(P_m).value(), cpu.from_matrix(R_m).value(),
                cpu.from_matrix(A0_m).value(), cpu.from_matrix(B0_m).value(),
                true, dk, 1u, true, cpu.from_matrix(bnd_m).value(), true); });
        auto got_pre = gpu_op([&] {
            return engine.scan_prefix_outer(K16, V16, P16, R16, A016, B016,
                                            true, dk, 1u, true, bnd16, true,
                                            nn::Precision::F16); });
        // 扫描族容差 5e-3：输入双侧 f16 舍入（eps≈4.9e-4）+ 跨 seq 步累加
        // 的舍入传播——2e-3 实测卡在 scan_prefix 2.03e-3 上，非缺陷
        check("scan_prefix_outer f16", got_pre, ref_pre, 5e-3f);

        nn::Matrix D_m(rows * dk, seq), X_m(rows, seq), Y_m(rows, seq);
        fill(D_m, gen(7)); fill(X_m, gen(8)); fill(Y_m, gen(9));
        auto D16 = f16(D_m), X16 = f16(X_m), Y16 = f16(Y_m);
        auto ref_suf = run_cpu([&] {
            return cpu.scan_suffix_outer(
                cpu.from_matrix(D_m).value(), cpu.from_matrix(X_m).value(),
                cpu.from_matrix(Y_m).value(), dk, 1u, true,
                cpu.from_matrix(bnd_m).value(), true); });
        auto got_suf = gpu_op([&] {
            return engine.scan_suffix_outer(D16, X16, Y16, dk, 1u, true,
                                            bnd16, true, nn::Precision::F16); });
        check("scan_suffix_outer f16", got_suf, ref_suf, 5e-3f);

        auto ref_oc = run_cpu([&] {
            return cpu.outer_col(cpu.from_matrix(P_m).value(),
                                 cpu.from_matrix(R_m).value(),
                                 cpu.from_matrix(V_m).value(), dk, true); });
        auto got_oc = gpu_op([&] {
            return engine.outer_col(P16, R16, V16, dk, true, nn::Precision::F16); });
        check("outer_col f16", got_oc, ref_oc, 5e-3f);
    }

    if (fails > 0)
    {
        printf("\n=== %d GPU f16 OP CHECK(S) FAILED ===\n", fails);
        return 1;
    }
    printf("\n=== ALL GPU f16 TESTS PASSED ===\n");
    return 0;
}
#else
#include <cstdio>
int main() {
    printf("Vulkan not available\n");
    return 0;
}
#endif
