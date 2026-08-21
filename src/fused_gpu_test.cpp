// ── fused_gpu_test — AOT 融合 shader GPU 数值验证 ─────────────────────────
// 验证"表达式单一事实来源"架构的 GPU 端：
//   1. 直接入口：backend.run_fused_gpu(rope_*) vs CPU ExprSpec 参考
//   2. 集成路径：GpuEngine.eval_expr AOT 匹配 → 融合 shader vs CPU 参考
//   3. 回退路径：未命中实例表的 spec → eager lowering vs CPU 参考
//
// 用法：fused_gpu_test
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using nn::Scalar;

#ifndef NN_HAS_VULKAN
int main()
{
    std::cout << "[SKIP] 此程序需要 Vulkan SDK 支持，请使用 -DNN_HAS_VULKAN 编译。\n";
    return 0;
}
#else
#include <neuralnet.cpp/backend/vk_backend.hpp>
#include <neuralnet.cpp/fused_exprs.hpp>

using nn::Matrix;
using nn::Tensor;
using nn::CpuEngine;
using nn::GpuEngine;
using nn::GpuBackend;
using nn::ExprSpec;

// ── 辅助：最大绝对误差 ────────────────────────────────────────────────────
Scalar max_abs_diff(const Matrix& a, const Matrix& b)
{
    Scalar max_err = 0.0f;
    auto as = a.span();
    auto bs = b.span();
    for (std::size_t i = 0; i < as.size(); ++i)
    {
        const Scalar diff = std::fabs(as[i] - bs[i]);
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

int main()
{
    std::cout << "========================================\n"
              << "  AOT 融合 shader GPU 数值验证\n"
              << "========================================\n";

    // ── 初始化引擎 ────────────────────────────────────────────────────────
    auto cpu_engine = std::make_unique<CpuEngine>();
    auto& backend = GpuBackend::instance();
    auto init_r = backend.initialize();
    if (!init_r)
    {
        std::cerr << "GPU 初始化失败: " << init_r.error().message << "\n";
        return 1;
    }
    auto gpu_engine = std::make_unique<GpuEngine>(backend);
    std::cout << "[init] CpuEngine + GpuEngine 就绪\n";

    std::mt19937 rng(12345);
    std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);

    constexpr Scalar kTol = 1e-4f;  // float32 融合路径容差
    int fail = 0;

    // ── 对每个 RoPE 实例做数值验证 ────────────────────────────────────────
    for (const auto& inst : nn::fused::kGenInstances)
    {
        if (inst.kind != nn::fused::FusedKind::RoPE) continue;  // SwiGLU 单独测试
        const std::uint32_t d_k = inst.d_k;
        const char* dir = inst.backward ? "backward" : "forward";

        // 形状约定（与 RotaryEmbedding::apply 一致）：
        //   q: (batch*H*d_k, seq)，rows 为 d_k 的整数倍
        //   cos/sin: (d_k, seq) 短表（RowMod 视图广播）
        const std::size_t rows = 2 * static_cast<std::size_t>(d_k);
        const std::size_t cols = 17;  // 非 2 的幂，检验索引通用性

        Matrix q(rows, cols), cos_m(d_k, cols), sin_m(d_k, cols);
        for (auto& v : q.span()) v = dist(rng);
        for (auto& v : cos_m.span()) v = dist(rng);
        for (auto& v : sin_m.span()) v = dist(rng);

        const ExprSpec spec = nn::fused::make_rope(d_k, inst.backward);

        // CPU 参考（融合解释器）
        std::vector<Tensor> cpu_inputs = {
            Tensor::from_matrix(Matrix(q)), Tensor::from_matrix(Matrix(cos_m)),
            Tensor::from_matrix(Matrix(q)), Tensor::from_matrix(Matrix(sin_m)),
        };
        auto cpu_r = cpu_engine->eval_expr(spec, cpu_inputs, rows, cols);
        if (!cpu_r)
        {
            std::cerr << "[FAIL] " << inst.name << ": CPU 参考计算失败: "
                      << cpu_r.error().message << "\n";
            ++fail;
            continue;
        }

        // ── 测试 A：直接入口 run_fused_gpu（纯融合 shader 数值）──────────
        if (!backend.has_fused_shader(inst.name))
        {
            std::cerr << "[FAIL] " << inst.name << ": 融合 shader 未注册\n";
            ++fail;
            continue;
        }
        std::vector<nn::GpuTensor> gpu_inputs;
        bool upload_ok = true;
        for (const auto& m : {q, cos_m, q, sin_m})
        {
            auto g = nn::GpuTensor::from_matrix(m, backend);
            if (!g) { upload_ok = false; break; }
            gpu_inputs.push_back(std::move(*g));
        }
        if (!upload_ok)
        {
            std::cerr << "[FAIL] " << inst.name << ": GPU 上传失败\n";
            ++fail;
            continue;
        }
        auto direct_r = backend.run_fused_gpu(
            inst.name, gpu_inputs, spec.consts, rows, cols);
        if (!direct_r)
        {
            std::cerr << "[FAIL] " << inst.name << ": run_fused_gpu 失败: "
                      << direct_r.error().message << "\n";
            ++fail;
            continue;
        }
        auto direct_m = direct_r->to_matrix(backend);
        if (!direct_m)
        {
            std::cerr << "[FAIL] " << inst.name << ": 结果下载失败\n";
            ++fail;
            continue;
        }
        const Scalar err_direct = max_abs_diff(cpu_r->cpu_matrix(), *direct_m);

        // ── 测试 B：集成路径 GpuEngine.eval_expr（AOT 匹配）──────────────
        // 前置确认：spec 与实例表定义一致（匹配必然命中融合路径）
        if (!nn::expr_spec_equal(spec, nn::fused::make_fused(inst)))
        {
            std::cerr << "[FAIL] " << inst.name << ": expr_spec_equal 不一致\n";
            ++fail;
            continue;
        }
        std::vector<Tensor> gpu_t_inputs;
        for (const auto& m : {q, cos_m, q, sin_m})
        {
            auto t = gpu_engine->from_matrix(m);
            if (!t) { upload_ok = false; break; }
            gpu_t_inputs.push_back(std::move(*t));
        }
        auto integ_r = gpu_engine->eval_expr(spec, gpu_t_inputs, rows, cols);
        if (!integ_r)
        {
            std::cerr << "[FAIL] " << inst.name << ": GpuEngine.eval_expr 失败: "
                      << integ_r.error().message << "\n";
            ++fail;
            continue;
        }
        auto integ_m = gpu_engine->to_matrix(*integ_r);
        if (!integ_m)
        {
            std::cerr << "[FAIL] " << inst.name << ": 集成结果下载失败\n";
            ++fail;
            continue;
        }
        const Scalar err_integ = max_abs_diff(cpu_r->cpu_matrix(), *integ_m);

        const bool ok = (err_direct < kTol) && (err_integ < kTol);
        std::cout << "[" << (ok ? "PASS" : "FAIL") << "] rope_" << dir
                  << "_dk" << d_k
                  << "  direct=" << std::scientific << std::setprecision(2) << err_direct
                  << "  eval_expr=" << err_integ << "\n";
        if (!ok) ++fail;
    }

    // ── 测试 B2：SwiGLU backward 融合 shader（grad_gate / grad_up）──────
    {
        const std::size_t rows = 4, cols = 7;
        std::mt19937 rng3(999);
        Matrix g(rows, cols), up(rows, cols), s(rows, cols), gate(rows, cols);
        for (auto& v : g.span()) v = dist(rng3);
        for (auto& v : up.span()) v = dist(rng3);
        for (auto& v : s.span()) v = dist(rng3);
        for (auto& v : gate.span()) v = dist(rng3);

        const nn::fused::GenInstance insts[] = {
            {"swiglu_grad_gate", nn::fused::FusedKind::SwiGLUGradGate, 0, false},
            {"swiglu_grad_up",   nn::fused::FusedKind::SwiGLUGradUp,   0, false},
        };
        for (const auto& inst : insts)
        {
            const ExprSpec spec = nn::fused::make_fused(inst);
            // grad_gate 输入序 [g, up, s, gate, s]；grad_up [g, gate, s]
            const Matrix* srcs_gate[] = {&g, &up, &s, &gate, &s};
            const Matrix* srcs_up[]   = {&g, &gate, &s};
            const int n = (inst.kind == nn::fused::FusedKind::SwiGLUGradGate) ? 5 : 3;
            const Matrix* const* srcs =
                (inst.kind == nn::fused::FusedKind::SwiGLUGradGate) ? srcs_gate : srcs_up;

            std::vector<Tensor> cpu_inputs, gpu_inputs_t;
            bool up_ok = true;
            for (int k = 0; k < n; ++k)
            {
                auto gt = gpu_engine->from_matrix(*srcs[k]);
                if (!gt) { up_ok = false; break; }
                gpu_inputs_t.push_back(std::move(*gt));
                cpu_inputs.push_back(Tensor::from_matrix(Matrix(*srcs[k])));
            }
            if (!up_ok) { std::cerr << "[FAIL] " << inst.name << ": 上传失败\n"; ++fail; continue; }

            auto cpu_r = cpu_engine->eval_expr(spec, cpu_inputs, rows, cols);
            if (!cpu_r) { std::cerr << "[FAIL] " << inst.name << ": CPU 参考失败\n"; ++fail; continue; }
            auto integ_r = gpu_engine->eval_expr(spec, gpu_inputs_t, rows, cols);
            if (!integ_r) { std::cerr << "[FAIL] " << inst.name << ": GpuEngine.eval_expr 失败\n"; ++fail; continue; }
            auto integ_m = gpu_engine->to_matrix(*integ_r);
            if (!integ_m) { std::cerr << "[FAIL] " << inst.name << ": 下载失败\n"; ++fail; continue; }
            const Scalar err = max_abs_diff(cpu_r->cpu_matrix(), *integ_m);
            const bool ok = err < kTol;
            std::cout << "[" << (ok ? "PASS" : "FAIL") << "] " << inst.name
                      << "  eval_expr=" << std::scientific << std::setprecision(2) << err << "\n";
            if (!ok) ++fail;
        }
    }

    // ── 测试 C：闭合世界——未命中 AOT 实例表的 spec，GPU 必须硬报错 ────
    {
        const std::uint32_t d_k = 32;
        const std::size_t rows = 2 * d_k;
        const std::size_t cols = 9;

        std::mt19937 rng2(777);
        Matrix q(rows, cols), cos_m(d_k, cols), sin_m(d_k, cols);
        for (auto& v : q.span()) v = dist(rng2);
        for (auto& v : cos_m.span()) v = dist(rng2);
        for (auto& v : sin_m.span()) v = dist(rng2);

        // 篡改第一条指令（Mul → Sub），使 spec 与所有实例都不匹配（[Sub,Mul,Add]）
        ExprSpec spec = nn::fused::make_rope(d_k, /*backward=*/false);
        spec.instrs[0].op = static_cast<std::uint8_t>(nn::ExprOp::Sub);

        // CPU 解释器（参考）仍能求值该任意 spec
        std::vector<Tensor> cpu_inputs = {
            Tensor::from_matrix(Matrix(q)), Tensor::from_matrix(Matrix(cos_m)),
            Tensor::from_matrix(Matrix(q)), Tensor::from_matrix(Matrix(sin_m)),
        };
        auto cpu_r = cpu_engine->eval_expr(spec, cpu_inputs, rows, cols);
        if (!cpu_r) { std::cerr << "[FAIL] 闭合世界: CPU 参考失败\n"; ++fail; }

        // GPU 闭合世界：未命中任何 AOT 融合 shader → 必须硬报错（无 eager 回退）
        std::vector<Tensor> gpu_inputs;
        for (const auto& m : {q, cos_m, q, sin_m})
        {
            auto t = gpu_engine->from_matrix(m);
            if (!t) { std::cerr << "[FAIL] 闭合世界: 上传失败\n"; ++fail; break; }
            gpu_inputs.push_back(std::move(*t));
        }
        auto gpu_r = gpu_engine->eval_expr(spec, gpu_inputs, rows, cols);
        const bool closed_world_ok = !gpu_r;  // 期待报错（闭合世界，无 eager）
        std::cout << "[" << (closed_world_ok ? "PASS" : "FAIL")
                  << "] 闭合世界: 未命中 AOT → GPU 硬报错\n";
        if (!closed_world_ok) ++fail;
    }

    std::cout << "========================================\n"
              << (fail == 0 ? "全部通过 ✅\n" : "存在失败 ❌\n");
    return fail == 0 ? 0 : 1;
}
#endif // NN_HAS_VULKAN
