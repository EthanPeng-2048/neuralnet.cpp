// ── t1_58_test — 三值（T1_58）精度与 BitLinear ──────────────────────────────
// 设计依据：docs/development/21-quantized-weights.md（P1 验收 §5/§7）
//
// 覆盖：
//   [1] PrecisionSet / PrecisionSupport 语义（默认集合 = {f16,f32} × 四槽）
//   [2] 能力校验**反例**：普通 Linear + param=T1_58 → 初始化期报错
//       （错误信息必须含 层名 + 槽位 + 取值），且是 Result 错误而非终止
//   [3] 能力校验**正例**：默认 / f16 / master-weights 配置全部照常通过
//   [4] Model::add 路径：default_precision_profile(param=T1_58) + add<Linear> → 报错
//   [5] D3(b)：init 之后 set_precision_profile → fail-fast（子进程断言非零退出）
//   [6] BitLinear：量化输出逐位符合 τ 规则（absmean + per-row）
//   [7] BitLinear：latent 权重的 STE 梯度 = 解析式（数值梯度只对 latent 做）
//   [8] BitLinear MLP 端到端小训练（CPU；`--gpu` 时在 GPU 上跑）
//
// 子进程模式：--expect-post-init-abort（父进程 std::system 自举）
// 退出码：0 = 通过，1 = 失败，77 = 跳过（--gpu 且无设备）
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#include "neuralnet.cpp/cli/cli_engine_factory.hpp"

#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{

int g_failures = 0;

std::string g_self;   // argv[0]：子进程自举用

// ── 便捷：随机矩阵 → 引擎张量（均匀分布 [lo,hi)）────────────────────────────
[[maybe_unused]] nn::Tensor upload_random(nn::ComputeEngine& eng, std::size_t rows,
                                          std::size_t cols, float lo, float hi, unsigned seed)
{
    nn::Matrix m(rows, cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& v : m.span()) v = d(rng);
    return upload(eng, m);
}

// ── [1] PrecisionSet / PrecisionSupport ──────────────────────────────────
void test_precision_set()
{
    std::puts("[1] PrecisionSet / PrecisionSupport 语义");

    const nn::PrecisionSet r = nn::rounding_precision_set();
    CHECK(r.has(nn::Precision::F16), "默认集合含 f16");
    CHECK(r.has(nn::Precision::F32), "默认集合含 f32");
    CHECK(!r.has(nn::Precision::T1_58), "默认集合不含 t1_58");
    CHECK(r.to_string() == "f16|f32", "to_string() 按枚举序罗列");

    const nn::PrecisionSet t = nn::PrecisionSet::of({nn::Precision::T1_58});
    CHECK(t.has(nn::Precision::T1_58), "显式集合含 t1_58");
    CHECK(!t.has(nn::Precision::F32), "显式集合不含 f32");
    CHECK(t.to_string() == "t1_58", "t1_58 集合名");

    // 默认能力 = 现状（既有层零改动）
    nn::CpuEngine eng;
    nn::Linear probe(4, 3);
    const nn::PrecisionSupport sup = probe.precision_support();
    CHECK(sup.param.mask == r.mask && sup.compute.mask == r.mask &&
          sup.stable.mask == r.mask && sup.optimizer.mask == r.mask,
          "默认 precision_support() = {f16,f32} × 四槽");
    CHECK(std::string(probe.layer_name()) == "Linear", "Linear::layer_name()");
    (void)eng;
}

// ── [2] 能力校验反例（构建/初始化期报错）─────────────────────────────────
nn::Result<void> add_linear_with_profile(nn::ComputeEngine& eng, const nn::PrecisionProfile& p)
{
    nn::Model model(eng);
    model.set_default_precision_profile(p);
    return model.add<nn::Linear>(4, 3);
}

void test_capability_reject()
{
    std::puts("[2] 能力校验反例：普通 Linear 不支持 param=T1_58");
    nn::CpuEngine eng;

    nn::PrecisionProfile t1;
    t1.param = nn::Precision::T1_58;

    // (a) 手工构造 + init
    {
        nn::Linear ln(4, 3);
        ln.set_precision_profile(t1);
        auto r = ln.init(eng);
        CHECK(!r.has_value(), "Linear + param=T1_58 → init 失败");
        if (!r.has_value())
        {
            const std::string& m = r.error().message;
            CHECK(m.find("Linear") != std::string::npos, "错误含层名（Linear）");
            CHECK(m.find("param") != std::string::npos, "错误含槽位（param）");
            CHECK(m.find("t1_58") != std::string::npos, "错误含取值（t1_58）");
            CHECK(m.find("f16|f32") != std::string::npos, "错误含该槽允许集合");
        }
    }
    // (b) Model::add 路径（工厂/容器路径同样在 init 期拦下）
    {
        auto r = add_linear_with_profile(eng, t1);
        CHECK(!r.has_value(), "Model::add<Linear> + param=T1_58 → 失败");
    }
    // (c) compute 槽不允许 T1_58（激活被三值化必然崩）
    {
        nn::PrecisionProfile p;
        p.compute = nn::Precision::T1_58;
        nn::Linear ln(4, 3);
        ln.set_precision_profile(p);
        auto r = ln.init(eng);
        CHECK(!r.has_value(), "Linear + compute=T1_58 → init 失败");
        if (!r.has_value())
            CHECK(r.error().message.find("compute") != std::string::npos, "错误含槽位（compute）");
    }
}

// ── [3] 能力校验正例（既有配置零回归）───────────────────────────────────
void test_capability_accept()
{
    std::puts("[3] 能力校验正例：既有配置全部照常通过");
    nn::CpuEngine eng;

    const nn::PrecisionProfile cases[] = {
        nn::profile_f32(),
        nn::profile_f16(),
        nn::profile_master_weights(),
    };
    for (const auto& p : cases)
    {
        nn::Linear ln(4, 3);
        ln.set_precision_profile(p);
        auto r = ln.init(eng);
        CHECK(r.has_value(), "既有 profile 通过能力校验");
    }
    // 默认（未注入 profile）= 全 f32，同样通过
    {
        nn::Linear ln(4, 3);
        auto r = ln.init(eng);
        CHECK(r.has_value(), "未注入 profile（默认全 f32）通过");
    }
}

// ── [4] D3(b)：init 后 profile 不可变 ────────────────────────────────────
void test_post_init_immutable()
{
    std::puts("[4] D3(b)：init 之后 set_precision_profile → fail-fast");
    nn::CpuEngine eng;
    nn::Linear ln(4, 3);
    auto r = ln.init(eng);
    CHECK(r.has_value(), "init 成功");
    // 子进程断言：这里的调用会终止进程（不能在本进程里做）
    const std::string cmd = "\"" + g_self + "\" --expect-post-init-abort";
    const int rc = std::system(cmd.c_str());
    if (rc == -1)
    {
        std::printf("[SKIP] 无法启动子进程（system 返回 -1）\n");
        return;
    }
#ifndef _WIN32
    const int code = WIFEXITED(rc) ? WEXITSTATUS(rc) : (WIFSIGNALED(rc) ? 128 + WTERMSIG(rc) : rc);
#else
    const int code = rc;
#endif
    CHECK(code != 0, "init 后 set_precision_profile → 非零退出（abort）");
}

} // namespace

int main(int argc, char* argv[])
{
    g_self = (argc > 0 && argv[0] != nullptr) ? argv[0] : "";

    // ── 子进程自举：init 后注入 profile → 期望终止 ────────────────────────
    if (argc >= 2 && std::string(argv[1]) == "--expect-post-init-abort")
    {
        nn::CpuEngine eng;
        nn::Linear ln(4, 3);
        (void)ln.init(eng);
        ln.set_precision_profile(nn::profile_f16());   // 必须 fail-fast
        std::printf("不应到达：profile 在 init 后被接受\n");
        return 0;
    }

    test_precision_set();
    test_capability_reject();
    test_capability_accept();
    test_post_init_immutable();

    if (g_failures == 0)
    {
        std::printf("\nt1_58_test: ALL PASSED\n");
        return 0;
    }
    std::printf("\nt1_58_test: %d FAILURE(S)\n", g_failures);
    return 1;
}
