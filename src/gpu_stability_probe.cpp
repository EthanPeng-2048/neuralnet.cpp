// ── gpu_stability_probe.cpp — CPU/GPU 训练 run-to-run 确定性探针 ─────────
//
// 用途（ComputeEngine Refresh P0，见 docs/development/15 §4.9 D9 实测）：
//   单进程内连续跑两轮完全相同的微型 GPT 训练（每轮独立建引擎、建模型），
//   比对每步 loss（%.9g，f32 9 位有效数字可往返）与训练后全部参数的
//   FNV-1a 校验和是否**逐位一致**；同时输出固定前缀行，便于跨进程
//   （两次启动二进制）用 grep + diff 比对。
//
// 退出码：0 = 两轮逐位一致；1 = 不一致（run-to-run 非确定）；2 = 运行错误。
// 用法：gpu_stability_probe [--steps N] [--gpu [索引|--gpu=<名称>]] [--help]
//   设备选择在首个 GpuBackend::instance() 之前注入 NN_VULKAN_DEVICE。
//
// 注意：Linear 等层 init 用 thread_local rng(std::random_device{})，权重
//   起点天然跨轮不同——本探针在 init 后用固定公式**覆写全部参数**，
//   保证两轮训练起点逐位一致，只测后端执行的确定性。
// ────────────────────────────────────────────────────────────────────────

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "neuralnet.cpp/cli/cli_gpu_option.hpp"
#include "neuralnet.cpp/nn.hpp"

namespace
{
    struct RunResult
    {
        std::vector<double> losses;
        std::uint64_t param_hash = 0;
    };

    std::uint64_t fnv1a(std::uint64_t h, const void *data, std::size_t n)
    {
        const auto *p = static_cast<const unsigned char *>(data);
        for (std::size_t i = 0; i < n; ++i)
        {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    // 固定公式覆写全部参数（init 的 rng 起点不可跨轮复现，覆写后才可比对）
    nn::Result<void> fix_weights(nn::ComputeEngine &eng, nn::Model &model)
    {
        std::size_t idx = 0;
        for (auto tr : model.parameters())
        {
            auto &p = tr.get();
            auto pm = eng.to_matrix(p);
            if (!pm) return std::unexpected(pm.error());
            auto sp = pm->span();
            for (std::size_t i = 0; i < sp.size(); ++i)
                sp[i] = 0.05f * std::sin(0.61803398875f * static_cast<float>(i + 1) +
                                         0.017f * static_cast<float>(idx));
            if (auto r = eng.copy_from(p, *pm); !r) return std::unexpected(r.error());
            ++idx;
        }
        return {};
    }

    // 全参数 FNV-1a 校验和（经 to_host 语义回读后按位散列）
    nn::Result<std::uint64_t> hash_params(nn::ComputeEngine &eng, nn::Model &model)
    {
        std::uint64_t h = 14695981039346656037ULL;
        for (auto tr : model.parameters())
        {
            auto pm = eng.to_matrix(tr.get());
            if (!pm) return std::unexpected(pm.error());
            auto sp = pm->span();
            h = fnv1a(h, sp.data(), sp.size() * sizeof(nn::Scalar));
        }
        return h;
    }

    // 一轮完整训练：建引擎 → 建模型（覆写权重）→ N 步 AdamW → 回读校验和
    nn::Result<RunResult> run_once(bool gpu, std::size_t steps)
    {
        RunResult out;

        std::unique_ptr<nn::ComputeEngine> eng;
        if (gpu)
        {
#ifdef NN_HAS_VULKAN
            eng = std::make_unique<nn::GpuEngine>(nn::GpuBackend::instance());
#else
            return std::unexpected(nn::Error{"built without Vulkan (NN_HAS_VULKAN)"});
#endif
        }
        else
        {
            eng = std::make_unique<nn::CpuEngine>();
        }
        nn::ComputeEngine &e = *eng;

        // 微型配置：两层 GPT，seq=16 batch=2，双轮各 40 次前反向，足够覆盖
        // matmul / attention / embedding / CE / 优化器全链路。
        const std::size_t vocab = 257, seq = 16, batch = 2;
        nn::GptConfig cfg{};
        cfg.vocab_size = vocab;
        cfg.d_model    = 32;
        cfg.seq_len    = seq;
        cfg.num_heads  = 4;
        cfg.d_ff       = 128;
        cfg.num_layers = 2;
        auto model_r = nn::build_gpt_model(e, cfg);
        if (!model_r) return std::unexpected(model_r.error());
        nn::Model model = std::move(*model_r);

        if (auto r = fix_weights(e, model); !r) return std::unexpected(r.error());
        if (auto r = model.zero_grad(); !r) return std::unexpected(r.error());

        auto opt = nn::create_optimizer("adamw", e, model.parameters(),
                                        model.param_gradients(), 1e-3, 0.0);
        nn::CrossEntropyLoss ce;

        nn::Matrix x_m(seq, batch);
        std::vector<std::size_t> labels(seq * batch);
        for (std::size_t s = 0; s < steps; ++s)
        {
            // 每步输入/标签都是确定性公式（不依赖任何 rng）
            for (std::size_t i = 0; i < x_m.size(); ++i)
                x_m.span()[i] = static_cast<nn::Scalar>((i * 7u + s * 13u + 1u) % vocab);
            for (std::size_t i = 0; i < labels.size(); ++i)
                labels[i] = (i * 11u + s * 17u + 3u) % vocab;

            if (auto r = model.zero_grad(); !r) return std::unexpected(r.error());
            auto x = e.from_matrix(x_m);
            if (!x) return std::unexpected(x.error());
            auto logits = model.forward(*x);
            if (!logits) return std::unexpected(logits.error());
            auto loss = ce.forward_sparse(e, *logits, labels, {}, vocab);
            if (!loss) return std::unexpected(loss.error());
            auto grad = ce.backward();
            if (!grad) return std::unexpected(grad.error());
            auto in_grad = model.backward(*grad);
            if (!in_grad) return std::unexpected(in_grad.error());
            if (auto r = opt->step(); !r) return std::unexpected(r.error());
            out.losses.push_back(static_cast<double>(*loss));
        }

        auto h = hash_params(e, model);
        if (!h) return std::unexpected(h.error());
        out.param_hash = *h;
        return out;
    }
} // namespace

int main(int argc, char **argv)
{
    std::size_t steps = 20;
    bool gpu = false;
    std::string device_desc = "cpu";

    for (int i = 1; i < argc;)
    {
        if (auto g = nn::cli::parse_gpu_option(argc, argv, i, true); g.has_value())
        {
            gpu = true;
            if (!g->empty())
            {
#ifdef _WIN32
                _putenv_s("NN_VULKAN_DEVICE", g->c_str());
#else
                setenv("NN_VULKAN_DEVICE", g->c_str(), 1);
#endif
                device_desc = "gpu:" + *g;
            }
            else
            {
                device_desc = "gpu:auto";
            }
            ++i;
        }
        else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
        {
            steps = static_cast<std::size_t>(std::strtoull(argv[i + 1], nullptr, 10));
            i += 2;
        }
        else if (std::strcmp(argv[i], "--help") == 0)
        {
            std::printf("用法: %s [--steps N] [--gpu [索引]]\n", argv[0]);
            return 0;
        }
        else
        {
            std::fprintf(stderr, "未知参数: %s（--help 查看用法）\n", argv[i]);
            return 2;
        }
    }
    if (steps == 0)
    {
        std::fprintf(stderr, "--steps 必须 >= 1\n");
        return 2;
    }
    if (gpu)
    {
        // MSVC CRT 弃用 getenv（-Werror）：按平台用 _dupenv_s / getenv。
        const char *env = nullptr;
#if defined(_MSC_VER)
        char *buf = nullptr;
        std::size_t len = 0;
        _dupenv_s(&buf, &len, "NN_VULKAN_DEVICE");
        env = buf;
#else
        env = std::getenv("NN_VULKAN_DEVICE");
#endif
        std::printf("CONFIG device=%s (NN_VULKAN_DEVICE=%s) steps=%zu\n",
                    device_desc.c_str(), env ? env : "<unset>", steps);
#if defined(_MSC_VER)
        std::free(buf);
#endif
    }
    else
    {
        std::printf("CONFIG device=cpu steps=%zu\n", steps);
    }

    auto r1 = run_once(gpu, steps);
    if (!r1)
    {
        std::fprintf(stderr, "run1 失败: %s\n", r1.error().message.c_str());
        return 2;
    }
    auto r2 = run_once(gpu, steps);
    if (!r2)
    {
        std::fprintf(stderr, "run2 失败: %s\n", r2.error().message.c_str());
        return 2;
    }

    // 固定前缀行：跨进程比对用 `probe ... | grep -E '^run[12]' | diff -`
    for (std::size_t s = 0; s < steps; ++s)
        std::printf("run1 step=%02zu loss=%.9g\n", s, r1->losses[s]);
    std::printf("run1 hash=%016" PRIx64 "\n", r1->param_hash);
    for (std::size_t s = 0; s < steps; ++s)
        std::printf("run2 step=%02zu loss=%.9g\n", s, r2->losses[s]);
    std::printf("run2 hash=%016" PRIx64 "\n", r2->param_hash);

    const bool loss_match = (r1->losses == r2->losses);
    const bool hash_match = (r1->param_hash == r2->param_hash);
    const bool pass = loss_match && hash_match;
    std::printf("RESULT in_process_loss_match=%d in_process_hash_match=%d verdict=%s\n",
                loss_match ? 1 : 0, hash_match ? 1 : 0, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
