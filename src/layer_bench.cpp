// ───────────────────────────────────────────────────────────────────────────
//  layer_bench.cpp — 各 Layer 前向/反向吞吐量基准（CPU / Vulkan GPU）
//
//  用法：
//    layer_bench [--gpu] [--all | --layer <name>[,...]]
//                [--forward-only] [--iter <n>] [--warmup <n>]
//                [尺寸参数]
//
//  尺寸参数（默认值见 BenchConfig）：
//    --in/--out/--batch/--seq/--dmodel/--heads/--dff/--cin/--cout/--kernel/--hw/--rows/--cols
//
//  层名：linear swiglu layernorm rmsnorm softmax mha causal_attn
//        feedforward transformer gpt_block conv2d maxpool
//
//  编译/注册：CMake target layer_bench（Debug/Release 均可）。
// ───────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

using nn::Scalar;
using nn::Matrix;
using nn::Tensor;
using nn::ComputeEngine;
using nn::Model;

namespace
{

// ═══════════════════════════════════════════════════════════════════════════
//  BenchConfig — CLI 参数与默认尺寸
// ═══════════════════════════════════════════════════════════════════════════
struct BenchConfig
{
    bool gpu = false;
    bool all = false;
    bool forward_only = false;
    std::vector<std::string> layers;
    int iter = 10;
    int warmup = 2;

    // 默认尺寸（GPT 小规模，CPU 秒级可测）
    std::size_t in = 1024, out = 1024;
    std::size_t dmodel = 768, heads = 12, dff = 3072;
    std::size_t batch = 1, seq = 512;
    std::size_t cin = 3, cout = 64, kernel = 3, hw = 32;
    std::size_t rows = 768, cols = 512;

    // 算子级尺寸（matmul: (m,k)·(k,n)；逐元素/归约: (m,n)）
    std::size_t m = 1024, n = 1024, k = 1024;

    // 算子级模式：非空时跑算子（--op <name|all>）
    std::vector<std::string> ops;
};

// ═══════════════════════════════════════════════════════════════════════════
//  工具
// ═══════════════════════════════════════════════════════════════════════════
double ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// 随机填充 Matrix 并上传到引擎设备
Tensor make_input(ComputeEngine& e, std::size_t rows, std::size_t cols)
{
    Matrix m(rows, cols);
    std::mt19937 g(1234u);
    std::uniform_real_distribution<Scalar> d(-1.0f, 1.0f);
    auto s = m.span();
    for (std::size_t i = 0; i < s.size(); ++i)
        s[i] = d(g);
    auto t = e.from_matrix(std::move(m));
    if (!t)
        std::abort();
    return std::move(*t);
}

// ═══════════════════════════════════════════════════════════════════════════
//  LayerSpec — 单个层的构建/输入形状/FLOP 估算
// ═══════════════════════════════════════════════════════════════════════════
struct LayerSpec
{
    const char* name;
    // 构建单层模型；返回 false 表示构建失败
    std::function<bool(Model&, const BenchConfig&)> build;
    // 输入张量形状 (rows, cols)
    std::function<std::pair<std::size_t, std::size_t>(const BenchConfig&)> shape;
    // 前向 FLOP 估算（用于 GFLOPS）
    std::function<double(const BenchConfig&)> flops;
    // 每次迭代的 token 数（sequence 层 = batch*seq；MLP 类 = batch）
    std::function<std::size_t(const BenchConfig&)> tokens;
};

bool build_linear(Model& m, const BenchConfig& c)
{
    auto r = m.template add<nn::Linear>(c.in, c.out);
    return r.has_value();
}
std::pair<std::size_t, std::size_t> shape_linear(const BenchConfig& c) { return {c.in, c.batch}; }
double flops_linear(const BenchConfig& c) { return 2.0 * c.in * c.out * c.batch; }
std::size_t tok_linear(const BenchConfig& c) { return c.batch; }

bool build_swiglu(Model& m, const BenchConfig& c)
{
    auto r = m.template add<nn::SwiGLU>(c.dff);
    return r.has_value();
}
std::pair<std::size_t, std::size_t> shape_swiglu(const BenchConfig& c) { return {2 * c.dff, c.batch}; }
double flops_swiglu(const BenchConfig& c) { return 8.0 * 2 * c.dff * c.batch; }
std::size_t tok_swiglu(const BenchConfig& c) { return c.batch; }

bool build_layernorm(Model& m, const BenchConfig& c)
{
    auto r = m.template add<nn::LayerNorm>(c.dmodel);
    return r.has_value();
}
bool build_rmsnorm(Model& m, const BenchConfig& c)
{
    auto r = m.template add<nn::RMSNorm>(c.dmodel);
    return r.has_value();
}
std::pair<std::size_t, std::size_t> shape_norm(const BenchConfig& c) { return {c.dmodel, c.batch * c.seq}; }
double flops_norm(const BenchConfig& c) { return 6.0 * c.dmodel * c.batch * c.seq; }
std::size_t tok_norm(const BenchConfig& c) { return c.batch * c.seq; }

bool build_softmax(Model& m, [[maybe_unused]] const BenchConfig& c)
{
    auto r = m.template add<nn::Softmax>();
    return r.has_value();
}
std::pair<std::size_t, std::size_t> shape_softmax(const BenchConfig& c) { return {c.rows, c.cols}; }
double flops_softmax(const BenchConfig& c) { return 6.0 * c.rows * c.cols; }
std::size_t tok_softmax(const BenchConfig& c) { return c.rows > 0 ? 1u : 0u; }

bool build_mha(Model& m, const BenchConfig& c)
{
    auto r = m.template add<nn::MultiHeadAttention>(c.dmodel, c.heads, c.seq);
    return r.has_value();
}
bool build_causal(Model& m, const BenchConfig& c)
{
    auto r = m.template add<nn::CausalSelfAttention>(c.dmodel, c.heads, c.seq, c.seq);
    return r.has_value();
}
std::pair<std::size_t, std::size_t> shape_attn(const BenchConfig& c) { return {c.dmodel, c.batch * c.seq}; }
double flops_attn(const BenchConfig& c)
{
    const double T = static_cast<double>(c.batch) * c.seq;
    return 6.0 * c.dmodel * c.dmodel * T + 4.0 * T * T * c.dmodel;
}
std::size_t tok_attn(const BenchConfig& c) { return c.batch * c.seq; }

bool build_feedforward(Model& m, const BenchConfig& c)
{
    auto r = m.template add<nn::FeedForward>(c.dmodel, c.dff);
    return r.has_value();
}
double flops_feedforward(const BenchConfig& c)
{
    return 4.0 * c.dmodel * c.dff * c.batch * c.seq;
}

bool build_transformer(Model& m, const BenchConfig& c)
{
    auto r = m.template add<nn::TransformerEncoderLayer>(c.dmodel, c.heads, c.dff, c.seq);
    return r.has_value();
}
double flops_transformer(const BenchConfig& c)
{
    return flops_attn(c) + flops_feedforward(c) + flops_norm(c) * 2.0;
}

bool build_gpt_block(Model& m, const BenchConfig& c)
{
    auto r = m.template add<nn::GPTBlock>(c.dmodel, c.heads, c.dff, c.seq, c.seq);
    return r.has_value();
}

bool build_conv(Model& m, const BenchConfig& c)
{
    auto r = m.template add<nn::Conv2D>(c.cin, c.cout, c.kernel, /*stride=*/1, /*pad=*/c.kernel / 2, c.hw, c.hw);
    return r.has_value();
}
std::pair<std::size_t, std::size_t> shape_conv(const BenchConfig& c) { return {c.cin * c.hw * c.hw, c.batch}; }
double flops_conv(const BenchConfig& c)
{
    return 2.0 * c.cout * c.cin * c.kernel * c.kernel * c.hw * c.hw * c.batch;
}
std::size_t tok_conv(const BenchConfig& c) { return c.batch; }

bool build_maxpool(Model& m, const BenchConfig& c)
{
    auto r = m.template add<nn::MaxPool2D>(c.cin, c.hw, c.hw, /*pool=*/2, /*stride=*/2);
    return r.has_value();
}
std::pair<std::size_t, std::size_t> shape_maxpool(const BenchConfig& c) { return {c.cin * c.hw * c.hw, c.batch}; }
double flops_maxpool(const BenchConfig& c)
{
    const std::size_t oh = (c.hw - 2) / 2 + 1;
    const std::size_t ow = (c.hw - 2) / 2 + 1;
    return 1.0 * c.cin * oh * ow * c.batch;
}
std::size_t tok_maxpool(const BenchConfig& c) { return c.batch; }

// ── 层注册表 ──────────────────────────────────────────────────────────────
const std::vector<LayerSpec>& layer_registry()
{
    static const std::vector<LayerSpec> reg = {
        {"linear", build_linear, shape_linear, flops_linear, tok_linear},
        {"swiglu", build_swiglu, shape_swiglu, flops_swiglu, tok_swiglu},
        {"layernorm", build_layernorm, shape_norm, flops_norm, tok_norm},
        {"rmsnorm", build_rmsnorm, shape_norm, flops_norm, tok_norm},
        {"softmax", build_softmax, shape_softmax, flops_softmax, tok_softmax},
        {"mha", build_mha, shape_attn, flops_attn, tok_attn},
        {"causal_attn", build_causal, shape_attn, flops_attn, tok_attn},
        {"feedforward", build_feedforward, shape_norm, flops_feedforward, tok_norm},
        {"transformer", build_transformer, shape_norm, flops_transformer, tok_norm},
        {"gpt_block", build_gpt_block, shape_norm, flops_transformer, tok_norm},
        {"conv2d", build_conv, shape_conv, flops_conv, tok_conv},
        {"maxpool", build_maxpool, shape_maxpool, flops_maxpool, tok_maxpool},
    };
    return reg;
}

// ═══════════════════════════════════════════════════════════════════════════
//  算子级基准（基础原语：matmul 及其变体、逐元素、归约、广播、转置）
// ═══════════════════════════════════════════════════════════════════════════
struct OpCtx
{
    Tensor a, b, c;    // 主操作数/输出
    Tensor aux;        // 辅助（行/列向量等）
};

struct OpSpec
{
    const char* name;
    // setup: 一次性创建输入张量（不计时）
    std::function<void(ComputeEngine&, const BenchConfig&, OpCtx&)> setup;
    // run: 每次迭代执行算子（计时）
    std::function<void(ComputeEngine&, const BenchConfig&, OpCtx&)> run;
    // 工作量：is_flops 时 = FLOPs；否则 = 读写字节数
    std::function<double(const BenchConfig&)> work;
    bool is_flops;     // true → GFLOPS; false → GB/s
};

void setup_matmul(ComputeEngine& e, const BenchConfig& c, OpCtx& ctx)
{
    ctx.a = make_input(e, c.m, c.k);
    ctx.b = make_input(e, c.k, c.n);
}
void run_matmul(ComputeEngine& e, const BenchConfig&, OpCtx& ctx)
{
    ctx.c = *e.matmul(ctx.a, ctx.b);
}
void run_matmul_bt(ComputeEngine& e, const BenchConfig&, OpCtx& ctx)
{
    ctx.c = *e.matmul(ctx.a, ctx.b, false, true);
}
void run_matmul_at(ComputeEngine& e, const BenchConfig&, OpCtx& ctx)
{
    ctx.c = *e.matmul(ctx.a, ctx.b, true, false);
}
double work_matmul(const BenchConfig& c) { return 2.0 * c.m * c.n * c.k; }

void setup_batched(ComputeEngine& e, const BenchConfig& c, OpCtx& ctx)
{
    ctx.a = make_input(e, c.batch * c.m, c.k);
    ctx.b = make_input(e, c.batch * c.k, c.n);
}
void run_batched(ComputeEngine& e, const BenchConfig& c, OpCtx& ctx)
{
    ctx.c = *e.batched_matmul(ctx.a, ctx.b, c.batch);
}
double work_batched(const BenchConfig& c) { return 2.0 * c.batch * c.m * c.n * c.k; }

void setup_add(ComputeEngine& e, const BenchConfig& c, OpCtx& ctx)
{
    ctx.a = make_input(e, c.m, c.n);
    ctx.b = make_input(e, c.m, c.n);
}
void run_add(ComputeEngine& e, const BenchConfig&, OpCtx& ctx)
{
    (void)e.add_inplace(ctx.a, ctx.b);
}
double bytes_add(const BenchConfig& c) { return 3.0 * c.m * c.n * sizeof(Scalar); }

void setup_exp(ComputeEngine& e, const BenchConfig& c, OpCtx& ctx)
{
    ctx.a = make_input(e, c.m, c.n);
}
void run_exp(ComputeEngine& e, const BenchConfig&, OpCtx& ctx)
{
    ctx.c = *e.elementwise_unary(nn::UnaryOp::Exp, ctx.a);
}
double bytes_binary(const BenchConfig& c) { return 2.0 * c.m * c.n * sizeof(Scalar); }

void setup_bcast_col(ComputeEngine& e, const BenchConfig& c, OpCtx& ctx)
{
    ctx.a = make_input(e, c.m, c.n);
    ctx.aux = make_input(e, 1, c.n);
}
void run_bcast_col(ComputeEngine& e, const BenchConfig&, OpCtx& ctx)
{
    (void)e.broadcast_col_inplace(ctx.a, ctx.aux, nn::BinaryOp::Add);
}
double bytes_bcast_col(const BenchConfig& c)
{
    return (2.0 * c.m * c.n + c.n) * sizeof(Scalar);
}

void run_row_reduce(ComputeEngine& e, const BenchConfig&, OpCtx& ctx)
{
    ctx.c = *e.row_reduce_sum(ctx.a);
}
void run_col_reduce(ComputeEngine& e, const BenchConfig&, OpCtx& ctx)
{
    ctx.c = *e.col_reduce_sum(ctx.a);
}
double bytes_reduce(const BenchConfig& c) { return 2.0 * c.m * c.n * sizeof(Scalar); }

void run_transpose(ComputeEngine& e, const BenchConfig&, OpCtx& ctx)
{
    ctx.c = *e.transpose(ctx.a);
}
double bytes_transpose(const BenchConfig& c) { return 2.0 * c.m * c.n * sizeof(Scalar); }

void run_scale(ComputeEngine& e, const BenchConfig&, OpCtx& ctx)
{
    (void)e.scale_inplace(ctx.a, Scalar{0.5f});
}

const std::vector<OpSpec>& op_registry()
{
    static const std::vector<OpSpec> reg = {
        {"matmul", setup_matmul, run_matmul, work_matmul, true},
        {"matmul_bt", setup_matmul, run_matmul_bt, work_matmul, true},
        {"matmul_at", setup_matmul, run_matmul_at, work_matmul, true},
        {"batched_matmul", setup_batched, run_batched, work_batched, true},
        {"add_inplace", setup_add, run_add, bytes_add, false},
        {"elementwise_exp", setup_exp, run_exp, bytes_binary, false},
        {"broadcast_col", setup_bcast_col, run_bcast_col, bytes_bcast_col, false},
        {"row_reduce_sum", setup_exp, run_row_reduce, bytes_reduce, false},
        {"col_reduce_sum", setup_exp, run_col_reduce, bytes_reduce, false},
        {"transpose", setup_exp, run_transpose, bytes_transpose, false},
        {"scale_inplace", setup_exp, run_scale, bytes_binary, false},
    };
    return reg;
}

void bench_op(ComputeEngine& engine, const BenchConfig& cfg, const OpSpec& spec)
{
    OpCtx ctx;
    spec.setup(engine, cfg, ctx);
    const bool is_flops = spec.is_flops;
    const double work = spec.work(cfg);
    const std::string_view unit = is_flops ? "GFLOPS" : "GB/s";
    const double scale = is_flops ? 1e9 : 1e9;  // 除以秒，再除以 1e9 得 G-单位

    // warmup
    for (int i = 0; i < cfg.warmup; ++i)
        spec.run(engine, cfg, ctx);

    double best = 1e30;
    for (int i = 0; i < cfg.iter; ++i)
    {
        auto t0 = std::chrono::steady_clock::now();
        spec.run(engine, cfg, ctx);
        best = std::min(best, ms_since(t0));
    }
    const double g = (work / (best / 1000.0)) / scale;
    std::printf("  %-14s : %8.3f ms  %10.1f %s\n", spec.name, best, g, unit.data());
}

// ═══════════════════════════════════════════════════════════════════════════
//  基准单层
// ═══════════════════════════════════════════════════════════════════════════
void bench_layer(ComputeEngine& engine, const BenchConfig& cfg, const LayerSpec& spec)
{
    Model model(engine);
    if (!spec.build(model, cfg))
    {
        std::printf("  %-14s : build failed\n", spec.name);
        return;
    }
    model.set_training(!cfg.forward_only);

    const auto [r, c] = spec.shape(cfg);
    const Tensor x = make_input(engine, r, c);

    // 首次 forward 取输出形状
    auto first = model.forward(x);
    if (!first)
    {
        std::printf("  %-14s : forward failed: %s\n", spec.name, first.error().message.c_str());
        return;
    }
    Tensor grad = make_input(engine, first->rows(), first->cols());

    const double fwd_flops = spec.flops(cfg);
    const std::size_t tok = spec.tokens(cfg);

    // warmup
    for (int i = 0; i < cfg.warmup; ++i)
    {
        (void)model.forward(x);
        if (!cfg.forward_only)
            (void)model.backward(grad);
    }

    // 计时
    double best_fwd = 1e30, best_train = 1e30;
    for (int i = 0; i < cfg.iter; ++i)
    {
        auto t0 = std::chrono::steady_clock::now();
        auto o = model.forward(x);
        double dt = ms_since(t0);
        if (o) best_fwd = std::min(best_fwd, dt);

        if (!cfg.forward_only)
        {
            (void)model.zero_grad();
            t0 = std::chrono::steady_clock::now();
            (void)model.forward(x);
            (void)model.backward(grad);
            dt = ms_since(t0);
            best_train = std::min(best_train, dt);
        }
    }

    const double fwd_gflops = (fwd_flops / (best_fwd / 1000.0)) / 1e9;
    std::printf("  %-14s : fwd %8.3f ms  %9.1f GFLOPS", spec.name, best_fwd, fwd_gflops);
    if (!cfg.forward_only)
    {
        const double train_s = best_train / 1000.0;
        const double tok_s = (train_s > 0) ? static_cast<double>(tok) / train_s : 0.0;
        std::printf("  |  train %8.3f ms  %9.0f tok/s", best_train, tok_s);
    }
    std::printf("\n");
}

// ═══════════════════════════════════════════════════════════════════════════
//  CLI 解析
// ═══════════════════════════════════════════════════════════════════════════
void print_help(const char* prog)
{
    std::printf("用法: %s [--gpu] [--all | --layer <name>[,...] | --op <name|all>[,...]] [选项]\n", prog);
    std::printf("  --gpu            使用 Vulkan GPU（默认 CPU）\n");
    std::printf("  --all            测试全部层\n");
    std::printf("  --layer <n>      指定层（逗号分隔可多个）\n");
    std::printf("  --op <n>         指定算子（matmul 等；'all' 测试全部算子）\n");
    std::printf("  --forward-only   仅前向（默认前向+反向）\n");
    std::printf("  --iter <n>       迭代次数（默认 10）\n");
    std::printf("  --warmup <n>     预热次数（默认 2）\n");
    std::printf("  --in/--out       Linear 输入/输出维度\n");
    std::printf("  --dmodel/--heads/--dff   序列层维度\n");
    std::printf("  --batch/--seq    batch / 序列长度\n");
    std::printf("  --cin/--cout/--kernel/--hw  卷积尺寸\n");
    std::printf("  --rows/--cols    Softmax 形状\n");
    std::printf("  --m/--n/--k      算子尺寸（matmul: (m,k)·(k,n)；逐元素/归约: (m,n)）\n");
    std::printf("层名: linear swiglu layernorm rmsnorm softmax mha causal_attn\n");
    std::printf("      feedforward transformer gpt_block conv2d maxpool\n");
    std::printf("算子: matmul matmul_bt matmul_at batched_matmul add_inplace elementwise_exp\n");
    std::printf("      broadcast_col row_reduce_sum col_reduce_sum transpose scale_inplace\n");
}

size_t parse_size(const char* val)
{
    if (!val) return 0;
    return static_cast<size_t>(std::strtoull(val, nullptr, 10));
}

} // namespace

int main(int argc, char* argv[])
{
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--gpu") { cfg.gpu = true; }
        else if (a == "--all") { cfg.all = true; }
        else if (a == "--forward-only") { cfg.forward_only = true; }
        else if (a == "--layer" && i + 1 < argc)
        {
            std::string val = argv[++i];
            std::string cur;
            for (char ch : val)
            {
                if (ch == ',') { if (!cur.empty()) cfg.layers.push_back(cur); cur.clear(); }
                else cur.push_back(ch);
            }
            if (!cur.empty()) cfg.layers.push_back(cur);
        }
        else if (a == "--iter" && i + 1 < argc) { cfg.iter = static_cast<int>(parse_size(argv[++i])); }
        else if (a == "--warmup" && i + 1 < argc) { cfg.warmup = static_cast<int>(parse_size(argv[++i])); }
        else if (a == "--in") { cfg.in = parse_size(argv[++i]); }
        else if (a == "--out") { cfg.out = parse_size(argv[++i]); }
        else if (a == "--dmodel") { cfg.dmodel = parse_size(argv[++i]); }
        else if (a == "--heads") { cfg.heads = parse_size(argv[++i]); }
        else if (a == "--dff") { cfg.dff = parse_size(argv[++i]); }
        else if (a == "--batch") { cfg.batch = parse_size(argv[++i]); }
        else if (a == "--seq") { cfg.seq = parse_size(argv[++i]); }
        else if (a == "--cin") { cfg.cin = parse_size(argv[++i]); }
        else if (a == "--cout") { cfg.cout = parse_size(argv[++i]); }
        else if (a == "--kernel") { cfg.kernel = parse_size(argv[++i]); }
        else if (a == "--hw") { cfg.hw = parse_size(argv[++i]); }
        else if (a == "--rows") { cfg.rows = parse_size(argv[++i]); }
        else if (a == "--cols") { cfg.cols = parse_size(argv[++i]); }
        else if (a == "--op" && i + 1 < argc)
        {
            std::string val = argv[++i];
            std::string cur;
            for (char ch : val)
            {
                if (ch == ',') { if (!cur.empty()) cfg.ops.push_back(cur); cur.clear(); }
                else cur.push_back(ch);
            }
            if (!cur.empty()) cfg.ops.push_back(cur);
        }
        else if (a == "--m") { cfg.m = parse_size(argv[++i]); }
        else if (a == "--n") { cfg.n = parse_size(argv[++i]); }
        else if (a == "--k") { cfg.k = parse_size(argv[++i]); }
        else if (a == "--help" || a == "-h") { print_help(argv[0]); return 0; }
        else { std::printf("未知参数: %s（--help 查看用法）\n", a.c_str()); return 2; }
    }

    // 选择引擎（CPU/GPU），失败硬报错不回退
    auto engine_r = nn::cli::create_engine(nn::cli::EngineConfig{cfg.gpu, false});
    if (!engine_r)
    {
        std::printf("引擎创建失败: %s\n", engine_r.error().message.c_str());
        return 2;
    }
    ComputeEngine& engine = **engine_r;
    const bool op_mode = !cfg.ops.empty();
    std::printf("=== layer_bench (%s%s) ===\n",
                cfg.gpu ? "GPU" : "CPU",
                op_mode ? ", ops" : (cfg.forward_only ? ", forward-only" : ", fwd+bwd"));

    // 算子模式：--op <name|all>
    if (op_mode)
    {
        std::vector<const OpSpec*> to_run;
        const auto& oreg = op_registry();
        for (const auto& want : cfg.ops)
        {
            if (want == "all")
            {
                for (const auto& s : oreg) to_run.push_back(&s);
                continue;
            }
            bool found = false;
            for (const auto& s : oreg)
            {
                if (want == s.name) { to_run.push_back(&s); found = true; break; }
            }
            if (!found) std::printf("未知算子: %s\n", want.c_str());
        }
        if (to_run.empty())
        {
            std::printf("未指定有效算子；用 --op all 或 --op matmul 等\n");
            return 2;
        }
        for (const auto* spec : to_run)
            bench_op(engine, cfg, *spec);
        return 0;
    }

    // 确定要跑的层
    std::vector<const LayerSpec*> to_run;
    const auto& reg = layer_registry();
    if (cfg.all)
    {
        for (const auto& s : reg) to_run.push_back(&s);
    }
    else if (!cfg.layers.empty())
    {
        for (const auto& want : cfg.layers)
        {
            bool found = false;
            for (const auto& s : reg)
            {
                if (want == s.name) { to_run.push_back(&s); found = true; break; }
            }
            if (!found) std::printf("未知层: %s\n", want.c_str());
        }
        if (to_run.empty())
        {
            std::printf("未指定有效层；用 --all 或 --layer <name>\n");
            return 2;
        }
    }
    else
    {
        std::printf("请指定 --all / --layer <name> / --op <name>\n");
        return 2;
    }

    for (const auto* spec : to_run)
        bench_layer(engine, cfg, *spec);
    return 0;
}
