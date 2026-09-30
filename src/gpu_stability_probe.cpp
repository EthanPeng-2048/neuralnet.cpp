// ── gpu_stability_probe.cpp — CPU/GPU 训练 run-to-run 确定性探针 ─────────
//
// 用途（ComputeEngine Refresh P0，见 docs/development/15 §4.9 D9 实测）：
//   单进程内连续跑两轮完全相同的微型 GPT 训练（每轮独立建引擎、建模型），
//   比对每步 loss（%.9g，f32 9 位有效数字可往返）与训练后全部参数的
//   FNV-1a 校验和是否**逐位一致**；同时输出固定前缀行，便于跨进程
//   （两次启动二进制）用 grep + diff 比对。
//
// 两种模式：
//   默认（--steps N）：如上两轮训练比对。init 已于 M2 收编为确定性
//     InitSpec（17 §4.4），此处**仍用固定公式覆写全部参数**——彻底隔离
//     init 变量，只测后端执行的确定性（输运路径不受初值影响）。
//   --init-hash：只建模型、不训练，比**初值**本身的确定性（M2 专属验收：
//     两次运行初值逐字节同）。覆盖 mlp/cnn/transformer/gpt/rapt 五类
//     模型的全部 init 路径；输出 INIT1/INIT2 固定前缀行，跨进程比对：
//     两次启动后 `grep '^INIT1' 各自输出 | diff`。
//   --io-roundtrip：M3 专属验收——批量 read/write/get_index/set_index 语义
//     对拍（f32/f16 × 写后读回 × read vs to_matrix × 索引往返 × 越界/U2
//     错配错误路径 × 录制窗口内 write/read）。GPU 上即 staging 批量语义
//     （上传 copy_from + 下载 to_matrix 隐含 flush/同步）。
//     输出 IO 固定前缀行。
//
// 退出码：0 = 两轮逐位一致（io：全部用例过）；1 = 不一致；2 = 运行错误。
// 用法：gpu_stability_probe [--steps N] [--init-hash] [--io-roundtrip]
//       [--gpu [索引|--gpu=<名称>]] [--help]
//   设备选择在首个 GpuBackend::instance() 之前注入 NN_VULKAN_DEVICE。
// ────────────────────────────────────────────────────────────────────────

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <utility>
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
            if (auto r = eng.write(p, pm->span()); !r) return std::unexpected(r.error());
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

    // 设备引擎（GPU 在首个 GpuBackend::instance() 前已注入 NN_VULKAN_DEVICE）
    nn::Result<std::unique_ptr<nn::ComputeEngine>> make_engine(bool gpu)
    {
        std::unique_ptr<nn::ComputeEngine> eng;
        if (gpu)
        {
#ifdef NN_HAS_VULKAN
            auto &backend = nn::GpuBackend::instance();
            // 修复（16 §7-2 根因）：instance() 只是惰性单例，Vulkan 设备要
            // initialize() 才建立——本探针原先漏调，首个 vkCreateBuffer 拿到
            // VK_NULL_HANDLE 即 "Invalid device" 崩溃。与 cli_engine_factory
            // 及全部 GPU 测试入口同法。无显式 selector 时 initialize() 内部
            // 仍按 "显式 > NN_VULKAN_DEVICE 环境变量 > 自动打分" 读设备选择。
            if (auto ir = backend.initialize(); !ir)
                return std::unexpected(ir.error());
            eng = std::make_unique<nn::GpuEngine>(backend);
#else
            return std::unexpected(nn::Error{"built without Vulkan (NN_HAS_VULKAN)"});
#endif
        }
        else
        {
            eng = std::make_unique<nn::CpuEngine>();
        }
        return eng;
    }

    // 一轮完整训练：建引擎 → 建模型（覆写权重）→ N 步 AdamW → 回读校验和
    nn::Result<RunResult> run_once(bool gpu, std::size_t steps)
    {
        RunResult out;

        auto eng_r = make_engine(gpu);
        if (!eng_r) return std::unexpected(eng_r.error());
        nn::ComputeEngine &e = **eng_r;

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

    // ── 初值确定性（M2 专属验收）：建模型后立刻哈希参数（不覆写）────────
    using InitHashes = std::vector<std::pair<std::string, std::uint64_t>>;

    nn::Result<void> hash_model(nn::ComputeEngine &e, const char *name,
                                nn::Result<nn::Model> &&mr, InitHashes &out)
    {
        if (!mr) return std::unexpected(mr.error());
        nn::Model model = std::move(*mr);
        auto h = hash_params(e, model);
        if (!h) return std::unexpected(h.error());
        out.emplace_back(name, *h);
        return {};
    }

    // 五类模型 = 全部 init 路径（Linear/Conv/Norm 常数/token_emb/位置编码/
    // ones_row_）。同一引擎顺序构建：创建序号（InitSpec 混流）按构造顺序推进，
    // 跨进程同序 → 初值逐字节确定。
    // （原第 6 类 ZiPT 已随 AttnZip 于 2026-10-01 移除；CrossAttention P 的
    //   uniform 初始化路径随之消失。锚点表见 AGENTS.md §12 / docs/history.md。）
    nn::Result<InitHashes> run_init_hash(bool gpu)
    {
        auto eng_r = make_engine(gpu);
        if (!eng_r) return std::unexpected(eng_r.error());
        nn::ComputeEngine &e = **eng_r;
        InitHashes out;

        if (auto r = hash_model(e, "mnist_mlp", nn::build_mnist_mlp_model(e), out); !r)
            return std::unexpected(r.error());

        nn::CnnConfig ccfg;
        ccfg.convs = nn::MNIST_CNN_CONVS;
        ccfg.fc_dims = nn::MNIST_CNN_FC;
        if (auto r = hash_model(e, "cnn", nn::build_cnn_model(e, ccfg), out); !r)
            return std::unexpected(r.error());

        if (auto r = hash_model(e, "mnist_transformer",
                                nn::build_mnist_transformer_model(e), out); !r)
            return std::unexpected(r.error());

        nn::GptConfig gcfg{};
        gcfg.vocab_size = 257;
        gcfg.d_model     = 32;
        gcfg.seq_len     = 16;
        gcfg.num_heads   = 4;
        gcfg.d_ff        = 128;
        gcfg.num_layers  = 2;
        if (auto r = hash_model(e, "gpt", nn::build_gpt_model(e, gcfg), out); !r)
            return std::unexpected(r.error());

        nn::RAPTConfig rcfg;
        rcfg.vocab_size = 257;
        rcfg.d_model    = 32;
        rcfg.seq_len    = 16;
        rcfg.num_heads  = 4;
        rcfg.d_ff       = 64;
        rcfg.num_layers = 2;
        if (auto r = hash_model(e, "rapt", nn::build_rapt_model(e, rcfg), out); !r)
            return std::unexpected(r.error());

        return out;
    }

    // ── M3 批量读写语义对拍（17 §5 M3 专属验收）──────────────────────────
    // 每个用例覆盖：write → read 逐元素往返（GPU = 上传 copy_from + 下载
    // to_matrix 的 staging 链路）、read 与 to_matrix 两条下载路径逐位一致、
    // get_index/set_index 往返、越界与 U2 精度错配的错误路径。
    constexpr std::size_t kIoRows = 6, kIoCols = 7;

    nn::Result<void> io_f32_case(nn::ComputeEngine &e, const char *tag)
    {
        auto t = e.create_tensor(kIoRows, kIoCols, nn::Precision::F32,
                                 nn::InitSpec::zero());
        if (!t.valid())
            return std::unexpected(nn::Error{"io: create_tensor 失败"});
        std::vector<nn::Scalar> pat(kIoRows * kIoCols);
        for (std::size_t i = 0; i < pat.size(); ++i)
            pat[i] = 0.001f * static_cast<nn::Scalar>((i * 37u) % 1000u) - 0.2f;

        if (auto r = e.write(t, std::span<const nn::Scalar>(pat)); !r)
            return std::unexpected(r.error());
        std::vector<nn::Scalar> back(pat.size());
        if (auto r = e.read(t, std::span<nn::Scalar>(back)); !r)
            return std::unexpected(r.error());
        if (back != pat)
            return std::unexpected(nn::Error{std::string(tag) + ": write/read 往返不一致"});

        // read vs to_matrix：两条下载路径逐位一致
        auto m = e.to_matrix(t, nn::Precision::F32);
        if (!m)
            return std::unexpected(m.error());
        if (m->span().size() != pat.size())
            return std::unexpected(nn::Error{std::string(tag) + ": to_matrix 尺寸不一致"});
        for (std::size_t i = 0; i < pat.size(); ++i)
            if (m->span()[i] != pat[i])
                return std::unexpected(
                    nn::Error{std::string(tag) + ": read 与 to_matrix 结果不一致"});

        // get_index 往返（含首尾角与中部）
        const std::pair<std::size_t, std::size_t> pts[] = {
            {0, 0}, {kIoRows - 1, kIoCols - 1}, {2, 3}};
        for (const auto &rc : pts)
        {
            auto v = e.get_index(t, rc.first, rc.second);
            if (!v)
                return std::unexpected(v.error());
            if (*v != pat[rc.first * kIoCols + rc.second])
                return std::unexpected(nn::Error{std::string(tag) + ": get_index 不一致"});
        }

        // set_index → 批量 read 复核
        if (auto r = e.set_index(t, 3, 4, -0.125f); !r)
            return std::unexpected(r.error());
        if (auto r = e.read(t, std::span<nn::Scalar>(back)); !r)
            return std::unexpected(r.error());
        if (back[3 * kIoCols + 4] != -0.125f)
            return std::unexpected(nn::Error{std::string(tag) + ": set_index 未落盘"});

        // 错误路径：越界 / U2 精度错配 / 尺寸错配必须报错
        if (e.get_index(t, kIoRows, 0))
            return std::unexpected(nn::Error{std::string(tag) + ": get_index 越界未报错"});
        if (e.set_index(t, 0, kIoCols, 1.0f))
            return std::unexpected(nn::Error{std::string(tag) + ": set_index 越界未报错"});
        std::vector<nn::f16> wrong16(pat.size());
        if (e.read(t, std::span<nn::f16>(wrong16)))
            return std::unexpected(nn::Error{std::string(tag) + ": read f16 错配未报错"});
        if (e.write(t, std::span<const nn::f16>(wrong16)))
            return std::unexpected(nn::Error{std::string(tag) + ": write f16 错配未报错"});
        std::vector<nn::Scalar> shortbuf(3);
        if (e.read(t, std::span<nn::Scalar>(shortbuf)))
            return std::unexpected(nn::Error{std::string(tag) + ": read 尺寸错配未报错"});
        return {};
    }

    nn::Result<void> io_f16_case(nn::ComputeEngine &e, const char *tag)
    {
        auto t = e.create_tensor(kIoRows, kIoCols, nn::Precision::F16,
                                 nn::InitSpec::zero());
        if (!t.valid())
            return std::unexpected(nn::Error{"io: create_tensor(f16) 失败"});
        std::vector<nn::f16> pat(kIoRows * kIoCols);
        for (std::size_t i = 0; i < pat.size(); ++i)
            pat[i] = nn::f16(0.01f * static_cast<float>((i * 13u) % 200u) - 1.f);

        if (auto r = e.write(t, std::span<const nn::f16>(pat)); !r)
            return std::unexpected(r.error());
        std::vector<nn::f16> back(pat.size());
        if (auto r = e.read(t, std::span<nn::f16>(back)); !r)
            return std::unexpected(r.error());
        for (std::size_t i = 0; i < pat.size(); ++i)
            if (static_cast<float>(back[i]) != static_cast<float>(pat[i]))
                return std::unexpected(
                    nn::Error{std::string(tag) + ": f16 write/read 往返不一致"});

        // to_matrix（升 f32 精确无损）与 read(f16) 交叉对拍
        auto m = e.to_matrix(t, nn::Precision::F32);
        if (!m)
            return std::unexpected(m.error());
        for (std::size_t i = 0; i < pat.size(); ++i)
            if (m->span()[i] != static_cast<float>(pat[i]))
                return std::unexpected(
                    nn::Error{std::string(tag) + ": f16 read 与 to_matrix 不一致"});

        // 索引往返：set_index 的 Scalar 经 RHE 落 f16（0.5 可精确表示）
        auto v = e.get_index(t, 1, 2);
        if (!v)
            return std::unexpected(v.error());
        if (*v != static_cast<float>(pat[1 * kIoCols + 2]))
            return std::unexpected(nn::Error{std::string(tag) + ": f16 get_index 不一致"});
        if (auto r = e.set_index(t, 5, 6, 0.5f); !r)
            return std::unexpected(r.error());
        auto v2 = e.get_index(t, 5, 6);
        if (!v2 || *v2 != 0.5f)
            return std::unexpected(nn::Error{std::string(tag) + ": f16 set_index 未落盘"});

        // U2：f32 span 配 f16 张量必须报错
        std::vector<nn::Scalar> wrong32(pat.size());
        if (e.read(t, std::span<nn::Scalar>(wrong32)))
            return std::unexpected(nn::Error{std::string(tag) + ": read f32 错配未报错"});
        if (e.write(t, std::span<const nn::Scalar>(wrong32)))
            return std::unexpected(nn::Error{std::string(tag) + ": write f32 错配未报错"});
        return {};
    }

    // 录制窗口子测（17 §7-4 调用约定）：begin_batch 内 write（覆盖既有存储、
    // drain）→ read（隐含 flush + 同步）→ end_batch → 复读值不变；张量全程
    // 存活到 end_batch 之后（铁律 #6）。
    nn::Result<void> io_batch_window_case(nn::ComputeEngine &e, const char *tag)
    {
        auto t = e.create_tensor(4, 5, nn::Precision::F32, nn::InitSpec::zero());
        if (!t.valid())
            return std::unexpected(nn::Error{"io: create_tensor 失败"});
        std::vector<nn::Scalar> pat(20);
        for (std::size_t i = 0; i < pat.size(); ++i)
            pat[i] = 0.5f + static_cast<nn::Scalar>(i);

        if (auto r = e.begin_batch(); !r)
            return std::unexpected(r.error());
        if (auto r = e.write(t, std::span<const nn::Scalar>(pat)); !r)
            return std::unexpected(r.error());
        std::vector<nn::Scalar> back(pat.size());
        if (auto r = e.read(t, std::span<nn::Scalar>(back)); !r)
            return std::unexpected(r.error());
        if (back != pat)
            return std::unexpected(
                nn::Error{std::string(tag) + ": 窗口内 write/read 不一致"});
        if (auto r = e.end_batch(); !r)
            return std::unexpected(r.error());
        if (auto r = e.read(t, std::span<nn::Scalar>(back)); !r)
            return std::unexpected(r.error());
        if (back != pat)
            return std::unexpected(
                nn::Error{std::string(tag) + ": end_batch 后复读值变化"});
        return {};
    }

    int run_io_roundtrip(bool gpu)
    {
        auto eng_r = make_engine(gpu);
        if (!eng_r)
        {
            std::fprintf(stderr, "io-roundtrip 建引擎失败: %s\n",
                         eng_r.error().message.c_str());
            return 2;
        }
        nn::ComputeEngine &e = **eng_r;
        using CaseFn = nn::Result<void> (*)(nn::ComputeEngine &, const char *);
        const struct
        {
            const char *name;
            CaseFn fn;
        } cases[] = {
            {"f32_roundtrip", io_f32_case},
            {"f16_roundtrip", io_f16_case},
            {"batch_window", io_batch_window_case},
        };

        int failures = 0;
        for (const auto &c : cases)
        {
            auto r = c.fn(e, c.name);
            if (r)
            {
                std::printf("IO %s ok=1\n", c.name);
            }
            else
            {
                std::printf("IO %s ok=0\n", c.name);
                std::fprintf(stderr, "IO %s 失败: %s\n", c.name,
                             r.error().message.c_str());
                ++failures;
            }
        }
        std::printf("RESULT io_cases=%zu io_failures=%d verdict=%s\n",
                    sizeof(cases) / sizeof(cases[0]), failures,
                    failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    }
} // namespace

int main(int argc, char **argv)
{
    std::size_t steps = 20;
    bool gpu = false;
    bool init_hash = false;
    bool io_roundtrip = false;
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
        else if (std::strcmp(argv[i], "--init-hash") == 0)
        {
            init_hash = true;
            ++i;
        }
        else if (std::strcmp(argv[i], "--io-roundtrip") == 0)
        {
            io_roundtrip = true;
            ++i;
        }
        else if (std::strcmp(argv[i], "--help") == 0)
        {
            std::printf("用法: %s [--steps N] [--init-hash] [--io-roundtrip] [--gpu [索引]]\n",
                        argv[0]);
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

    // ── --io-roundtrip 模式（M3）：批量读写语义对拍（GPU 即 staging 验收）──
    if (io_roundtrip)
        return run_io_roundtrip(gpu);

    // ── --init-hash 模式（M2）：只建模型比初值，不训练 ────────────────────
    if (init_hash)
    {
        auto a = run_init_hash(gpu);
        if (!a)
        {
            std::fprintf(stderr, "init-hash round1 失败: %s\n", a.error().message.c_str());
            return 2;
        }
        auto b = run_init_hash(gpu);
        if (!b)
        {
            std::fprintf(stderr, "init-hash round2 失败: %s\n", b.error().message.c_str());
            return 2;
        }
        // 固定前缀行：跨进程比对用 `grep '^INIT1' <(两次启动输出) | diff`
        for (const auto &kv : *a)
            std::printf("INIT1 %s hash=%016" PRIx64 "\n", kv.first.c_str(), kv.second);
        for (const auto &kv : *b)
            std::printf("INIT2 %s hash=%016" PRIx64 "\n", kv.first.c_str(), kv.second);
        const bool match = (*a == *b);
        std::printf("RESULT init_in_process_match=%d verdict=%s\n",
                    match ? 1 : 0, match ? "PASS" : "FAIL");
        return match ? 0 : 1;
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
