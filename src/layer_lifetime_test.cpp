// ── layer_lifetime_test — 引擎借用生命周期契约 ───────────────────────────
//
// 覆盖（core_engine_lease.hpp + Layer / Model / Tensor / ExprBlock）：
//   [1] EngineLease 三态：unbound / alive / expired；令牌地址跨引擎不复用
//   [2] Layer 状态机：unbound → ready；init 失败 → failed；失败后可重新
//       init、可注入 profile（旧实现把"指针非空"当已 init，改 profile 会 abort）
//   [3] 跨引擎重绑 → Result 错误（明确拒绝，不是 UB）
//   [4] 引擎先析构：binding_state 仍 ready、engine_alive()==false；
//       后续 forward 在子进程中以 [engine-lease] 消息终止（exit != 0）
//   [5] 正常顺序（engine 活到 layer 之后）：forward 正常
//   [6] Model：引擎析构后 forward 返回错误、engine() 子进程终止
//   [7] Tensor：引擎析构后 bound() 仍 true、engine_alive()==false、
//       engine() 子进程终止；EngineLease 仍是 8B
//
// 子进程模式：--expect-layer-abort / --expect-model-abort / --expect-tensor-abort
// 退出码：0 = 通过，1 = 失败。
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#include "neuralnet.cpp/nn.hpp"
#include "test_common.hpp"

int g_fail = 0;

namespace
{

constexpr int kSkip = -1;
std::string g_self;

int run_self(const char* args)
{
    const std::string cmd = "\"" + g_self + "\" " + args;
    const int rc = std::system(cmd.c_str());
    if (rc == -1)
    {
        std::printf("[SKIP] 无法启动子进程（system 返回 -1）\n");
        return kSkip;
    }
    return rc;
}

int normalize_rc(int rc)
{
#ifdef _WIN32
    return rc;
#else
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
    if (WIFSIGNALED(rc)) return 128 + WTERMSIG(rc);
    return rc;
#endif
}

// ── 测试用层：init_impl 首次失败，之后成功 ────────────────────────────────
class FlakyLayer : public nn::Layer
{
public:
    [[nodiscard]] nn::Result<nn::Tensor> forward(const nn::Tensor&) override
    {
        NN_FAIL("FlakyLayer::forward: 未实现");
    }
    [[nodiscard]] nn::Result<nn::Tensor> backward(const nn::Tensor&) override
    {
        NN_FAIL("FlakyLayer::backward: 未实现");
    }
    [[nodiscard]] const char* layer_name() const noexcept override { return "FlakyLayer"; }

    int attempts = 0;

protected:
    [[nodiscard]] nn::Result<void> init_impl(nn::ComputeEngine&) override
    {
        ++attempts;
        if (attempts == 1)
            NN_FAIL("first init fails（测试注入）");
        return {};
    }
};

// ── [1] EngineLease 三态 + 令牌不复用 ────────────────────────────────────
void test_lease_basics()
{
    nn::EngineLease unbound;
    CHECK(!unbound.bound(), "默认构造的 lease 未绑定");
    CHECK(!unbound.alive(), "默认构造的 lease 不 alive");
    CHECK(unbound.get() == nullptr, "默认构造的 lease get() == nullptr");

    const nn::EngineLiveness* id1 = nullptr;
    nn::EngineLease outer;
    {
        nn::CpuEngine engine;
        outer = engine.lease();
        CHECK(outer.bound() && outer.alive(), "引擎存活时 lease 有效");
        CHECK(outer.get() == &engine, "lease.get() 指向原引擎");
        CHECK(outer.identity() != nullptr, "lease 带令牌身份");
        id1 = outer.identity();
    }
    CHECK(outer.expired(), "引擎析构后 lease 变 expired");
    CHECK(!outer.alive(), "expired lease 不 alive");
    CHECK(outer.get() == nullptr, "expired lease get() 永不返回悬垂指针");
    CHECK(outer.identity() == id1, "expired 后身份仍稳定（同源比较可用）");

    const nn::EngineLiveness* id2 = nullptr;
    {
        nn::CpuEngine engine2;
        id2 = engine2.lease().identity();
    }
    CHECK(id1 != id2, "令牌地址跨引擎不复用（陈旧 lease 不会误认新引擎）");
    CHECK(sizeof(nn::EngineLease) == sizeof(void*), "EngineLease 保持 8B");
}

// ── [2] Layer 状态机 + init 失败回滚 ─────────────────────────────────────
void test_layer_state_machine()
{
    nn::CpuEngine engine;
    FlakyLayer layer;

    CHECK(layer.binding_state() == nn::Layer::BindingState::unbound,
          "新层状态 = unbound");
    CHECK(!layer.engine_alive(), "新层 engine_alive() == false");

    auto r1 = layer.init(engine);
    CHECK(!r1, "首次 init 按注入失败");
    CHECK(layer.binding_state() == nn::Layer::BindingState::failed,
          "init 失败后状态 = failed（不是 ready）");
    CHECK(layer.engine_alive(), "失败后仍保留可诊断的引擎绑定");

    // 旧实现：engine_ 非空 ⇒ set_precision_profile 误判"已 init"并 abort。
    layer.set_precision_profile(nn::profile_f16());
    CHECK(layer.precision_profile().compute == nn::Precision::F16,
          "failed 状态允许修正 profile（旧实现会 abort）");

    auto r2 = layer.init(engine);
    CHECK(r2, "失败后可重新 init");
    CHECK(layer.binding_state() == nn::Layer::BindingState::ready,
          "重新 init 成功后状态 = ready");
    CHECK(layer.engine_alive(), "ready 层 engine_alive() == true");
}

// ── [3] 跨引擎重绑拒绝 ───────────────────────────────────────────────────
void test_cross_engine_reinit_rejected()
{
    nn::CpuEngine e1;
    nn::CpuEngine e2;
    FlakyLayer layer;
    layer.attempts = 1;   // 让下一次 init_impl 直接成功（跳过注入的首次失败）
    auto r1 = layer.init(e1);
    CHECK(r1, "绑定 e1 成功");
    auto r2 = layer.init(e2);
    CHECK(!r2, "跨引擎重绑返回错误");
    CHECK(!r2 && r2.error().message.find("另一个引擎") != std::string::npos,
          "错误信息说明是跨引擎重绑");
    CHECK(layer.engine_alive(), "拒绝后仍绑定 e1（未被改写）");
}

// ── [4] 引擎先析构：状态可判定 ───────────────────────────────────────────
void test_engine_destroyed_before_layer()
{
    std::unique_ptr<nn::Linear> layer;
    nn::Tensor x;
    {
        nn::CpuEngine engine;
        // Linear 的输入布局是 (in_features, batch)
        auto xr = engine.try_create_tensor(4, 2, nn::Precision::F32);
        CHECK(xr, "创建输入张量");
        x = *xr;
        layer = std::make_unique<nn::Linear>(4, 3);
        auto ir = layer->init(engine);
        CHECK(ir, "Linear init");
    }

    CHECK(layer->binding_state() == nn::Layer::BindingState::ready,
          "引擎销毁后层状态仍是 ready（绑定事实不变）");
    CHECK(!layer->engine_alive(), "引擎销毁后 engine_alive() == false");

    const int rc = normalize_rc(run_self("--expect-layer-abort"));
    if (rc != kSkip)
        CHECK(rc != 0, "引擎已销毁时 forward 终止（而不是 UB）");
}

// ── [5] 正常顺序不回归 ───────────────────────────────────────────────────
void test_normal_order()
{
    nn::CpuEngine engine;
    auto xr = engine.try_create_tensor(4, 2, nn::Precision::F32);
    CHECK(xr, "创建输入张量");
    nn::Linear layer(4, 3);
    auto ir = layer.init(engine);
    CHECK(ir, "正常 init");
    auto y = layer.forward(*xr);
    CHECK(y, "引擎存活时 forward 正常");
    CHECK(y && y->rows() == 3 && y->cols() == 2, "forward 输出形状正确");
}

// ── [6] Model：引擎先析构 ────────────────────────────────────────────────
void test_model_engine_destroyed()
{
    std::unique_ptr<nn::Model> model;
    nn::Tensor x;
    {
        nn::CpuEngine engine;
        auto xr = engine.try_create_tensor(4, 2, nn::Precision::F32);
        CHECK(xr, "创建输入张量");
        x = *xr;
        model = std::make_unique<nn::Model>(engine);
        auto ar = model->add<nn::Linear>(4, 3);
        CHECK(ar, "Model::add");
    }
    auto y = model->forward(x);
    CHECK(!y, "引擎销毁后 Model::forward 返回错误");
    CHECK(!y && y.error().message.find("已销毁") != std::string::npos,
          "Model 错误信息说明引擎已销毁");

    const int rc = normalize_rc(run_self("--expect-model-abort"));
    if (rc != kSkip)
        CHECK(rc != 0, "引擎已销毁时 Model::engine() 终止");
}

// ── [7] Tensor：引擎先析构 ───────────────────────────────────────────────
void test_tensor_engine_destroyed()
{
    nn::Tensor t;
    {
        nn::CpuEngine engine;
        auto tr = engine.try_create_tensor(2, 2, nn::Precision::F32);
        CHECK(tr, "创建张量");
        t = *tr;
        CHECK(t.bound() && t.engine_alive(), "存活引擎上的张量 bound + alive");
    }
    CHECK(t.bound(), "引擎销毁后 bound() 仍表示出生绑定存在");
    CHECK(!t.engine_alive(), "引擎销毁后 engine_alive() == false");

    const int rc = normalize_rc(run_self("--expect-tensor-abort"));
    if (rc != kSkip)
        CHECK(rc != 0, "引擎已销毁时 Tensor::engine() 终止");
}

} // namespace

int main(int argc, char* argv[])
{
    g_self = (argc > 0 && argv[0] != nullptr) ? argv[0] : "";

    // ── 子进程模式（父进程 std::system 自举）─────────────────────────────
    if (argc >= 2 && std::string(argv[1]) == "--expect-layer-abort")
    {
        std::unique_ptr<nn::Linear> layer;
        nn::Tensor x;
        {
            nn::CpuEngine engine;
            x = engine.create_tensor(4, 2, nn::Precision::F32);
            layer = std::make_unique<nn::Linear>(4, 3);
            (void)layer->init(engine);
        }
        std::printf("子进程：引擎已销毁，调用 forward\n");
        (void)layer->forward(x);
        std::printf("不应到达\n");
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--expect-model-abort")
    {
        std::unique_ptr<nn::Model> model;
        {
            nn::CpuEngine engine;
            model = std::make_unique<nn::Model>(engine);
            (void)model->add<nn::Linear>(4, 3);
        }
        std::printf("子进程：引擎已销毁，调用 Model::engine()\n");
        (void)model->engine();
        std::printf("不应到达\n");
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--expect-tensor-abort")
    {
        nn::Tensor t;
        {
            nn::CpuEngine engine;
            t = engine.create_tensor(2, 2, nn::Precision::F32);
        }
        std::printf("子进程：引擎已销毁，调用 Tensor::engine()\n");
        (void)t.engine();
        std::printf("不应到达\n");
        return 0;
    }

    std::printf("── EngineLease / Layer / Model / Tensor 生命周期 ──\n");
    test_lease_basics();
    test_layer_state_machine();
    test_cross_engine_reinit_rejected();
    test_engine_destroyed_before_layer();
    test_normal_order();
    test_model_engine_destroyed();
    test_tensor_engine_destroyed();

    if (g_fail == 0)
        std::printf("layer_lifetime_test: ALL PASS\n");
    else
        std::printf("layer_lifetime_test: %d failure(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
