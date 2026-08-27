// ── attn_w_batch_test.cpp — GPU 注意力 W 表达式 batch 行号回归测试（ctest）──
//
// 背景：GPU 融合 matmul 尾链曾按 **batch 内行号**读取 (rows,1) 全网格输入
// （注意力的 m/l），导致 batch>1 或多头（BH>1）时 m/l 读错 → GPU 前向错误
// （2026-08-27 审查发现，P0-1）。本测试用同一随机权重、同一输入对比
// CausalSelfAttention 的 CPU/GPU forward，覆盖 4 组配置
// (batch, heads) = (1,1)(2,1)(1,2)(2,2)（BH 最大 4），
// 任一 max_abs > 1e-3 判失败。
//
// 用法：
//   attn_w_batch_test              依次跑 4 组配置（ctest 默认）
//   attn_w_batch_test <b> <h> <s>  单配置模式（调试）
//
// 退出码：0 = 全部一致；2 = CPU 引擎创建失败；77 = GPU 不可用（ctest SKIP）；
//         3 = CPU/GPU 不一致。
// ────────────────────────────────────────────────────────────────────────────
#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include <cmath>
#include <iostream>
#include <random>

using nn::Scalar;
using nn::Matrix;

int main(int argc, char** argv)
{
    auto cpu_res = nn::cli::create_engine(nn::cli::EngineConfig{}, std::cout);
    if (!cpu_res) { std::cerr << "CPU 引擎创建失败: " << cpu_res.error().message << "\n"; return 2; }
    nn::cli::EngineConfig gcfg;
    gcfg.use_gpu = true;
    auto gpu_res = nn::cli::create_engine(gcfg, std::cout);
    if (!gpu_res) { std::cout << "GPU 不可用，跳过: " << gpu_res.error().message << "\n"; return 77; }
    nn::ComputeEngine& cpu = **cpu_res;
    nn::ComputeEngine& gpu = **gpu_res;

    const std::size_t d_model = 16;
    std::size_t batch = 0, heads = 0, seq = 5;
    if (argc >= 4) { batch = std::stoul(argv[1]); heads = std::stoul(argv[2]); seq = std::stoul(argv[3]); }

    std::mt19937_64 rng(123);
    std::uniform_real_distribution<Scalar> dist(-1, 1);

    auto run_cfg = [&](std::size_t b, std::size_t h) -> bool
    {
        std::cout << "attn W batch 行号回归: d_model=" << d_model << " heads=" << h
                  << " seq=" << seq << " batch=" << b << " (BH=" << b * h << ")\n";
        nn::CausalSelfAttention attn_cpu(d_model, h, seq, seq, nn::PosEncodingType::Learned);
        nn::CausalSelfAttention attn_gpu(d_model, h, seq, seq, nn::PosEncodingType::Learned);
        { auto r = attn_cpu.init(cpu); if (!r) { std::cerr << "cpu init: " << r.error().message << "\n"; return false; } }
        { auto r = attn_gpu.init(gpu); if (!r) { std::cerr << "gpu init: " << r.error().message << "\n"; return false; } }

        // 权重同步：CPU 随机 → GPU
        {
            auto pc = attn_cpu.parameters();
            auto pg = attn_gpu.parameters();
            for (std::size_t i = 0; i < pc.size(); ++i)
            {
                auto cm = cpu.to_matrix(pc[i].get());
                if (!cm) { std::cerr << "to_matrix cpu failed\n"; return false; }
                auto cp = gpu.copy_from(pg[i].get(), *cm);
                if (!cp) { std::cerr << "copy_from gpu failed: " << cp.error().message << "\n"; return false; }
            }
        }

        const std::size_t total = seq * b;
        Matrix x_m(d_model, total);
        for (std::size_t i = 0; i < x_m.size(); ++i) x_m.span()[i] = dist(rng);

        auto xc = cpu.from_matrix(x_m);
        auto xg = gpu.from_matrix(x_m);
        if (!xc || !xg) { std::cerr << "from_matrix failed\n"; return false; }

        auto yc = attn_cpu.forward(cpu, *xc);
        auto yg = attn_gpu.forward(gpu, *xg);
        if (!yc) { std::cout << "CPU forward 失败: " << yc.error().message << "\n"; return false; }
        if (!yg) { std::cout << "GPU forward 失败: " << yg.error().message << "\n"; return false; }

        auto ycm = cpu.to_matrix(*yc);
        auto ygm = gpu.to_matrix(*yg);
        if (!ycm || !ygm) { std::cerr << "to_matrix failed\n"; return false; }

        Scalar max_abs = 0;
        std::size_t bad = 0;
        const auto sc = ycm->span();
        const auto sg = ygm->span();
        for (std::size_t i = 0; i < ycm->size(); ++i)
        {
            const Scalar d = std::fabs(sc[i] - sg[i]);
            if (d > max_abs) max_abs = d;
            if (d > 1e-3f) ++bad;
        }
        std::cout << "  max_abs_diff=" << max_abs
                  << "  bad_cells(>1e-3)=" << bad << " / " << ycm->size()
                  << "  " << (max_abs <= 1e-3f ? "✅ 一致" : "❌ 不一致") << "\n";
        return max_abs <= 1e-3f;
    };

    int failed = 0;
    if (batch > 0 && heads > 0)
    {
        if (!run_cfg(batch, heads)) ++failed;
    }
    else
    {
        // ctest 默认：4 组配置全跑（任一组 BH>1 即可复现原 batch 内行号 bug）
        const std::size_t cfgs[4][2] = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};
        for (auto& c : cfgs)
            if (!run_cfg(c[0], c[1])) ++failed;
    }
    return failed > 0 ? 3 : 0;
}
