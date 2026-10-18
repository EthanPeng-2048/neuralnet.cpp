// ── layer_lifetime_test — EngineId + 强所有权生命周期 ─────────────────────
//
// v2 契约（docs/development/09-engine-ownership.md）：
//   [1] EngineId：8B、平凡复制、进程内不复用；引擎不可栈上构造
//   [2] Tensor 只保存 EngineId：引擎析构后仍可复制/析构，不存在悬垂指针
//   [3] Layer 持有 shared_ptr<ComputeEngine>：外部 shared_ptr 释放后仍可 forward
//   [4] Model 持有强引用：引擎随 Model 存活
//   [5] 跨引擎重绑返回 Result 错误
//   [6] Optimizer 持有强引用：外部引擎句柄释放后 step 仍安全
//   [7] Layer 的 init 失败回滚与 profile 注入
// ─────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "test_common.hpp"

int g_fail = 0;

namespace
{

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

void test_engine_id()
{
    static_assert(sizeof(nn::EngineId) == sizeof(void*),
                  "EngineId 必须保持 8B");
    static_assert(std::is_trivially_copyable_v<nn::EngineId>);
    static_assert(!std::is_default_constructible_v<nn::CpuEngine>,
                  "CpuEngine 不允许栈上构造（必须用 create()）");
#ifdef NN_HAS_VULKAN
    static_assert(!std::is_default_constructible_v<nn::GpuEngine>,
                  "GpuEngine 不允许栈上构造（必须用 create()）");
#endif

    nn::EngineId unbound;
    CHECK(!unbound.bound(), "默认 EngineId 未绑定");

    auto e1 = nn::CpuEngine::create();
    auto e2 = nn::CpuEngine::create();
    CHECK(e1->id().bound() && e2->id().bound(), "引擎构造后带非零身份");
    CHECK(e1->id() != e2->id(), "不同引擎身份不相等");

    const nn::EngineId a = e1->id();
    const nn::EngineId b = a;
    CHECK(a == b, "EngineId 复制保持相等");
}

void test_tensor_survives_engine()
{
    nn::Tensor t;
    nn::EngineId id;
    {
        auto engine = nn::CpuEngine::create();
        t = engine->create_tensor(2, 2, nn::Precision::F32);
        id = t.engine_id();
        CHECK(t.bound(), "引擎产物出生即带身份");
        CHECK(id == engine->id(), "Tensor::engine_id() == 创建引擎 id");
    }

    // 引擎已析构：Tensor 只是 8 字节身份 + 共享存储，复制/析构都安全。
    CHECK(t.bound(), "引擎析构后 Tensor 仍记录出生身份");
    CHECK(t.engine_id() == id, "身份不随引擎析构改变");
    nn::Tensor copy = t;
    CHECK(copy.engine_id() == id, "Tensor 拷贝传播 EngineId");
    CHECK(copy.rows() == 2 && copy.cols() == 2, "Tensor 元数据可安全读取");
}

void test_layer_keeps_engine_alive()
{
    std::unique_ptr<nn::Linear> layer;
    nn::Tensor x;
    {
        auto engine = nn::CpuEngine::create();
        x = engine->create_tensor(4, 2, nn::Precision::F32);
        layer = std::make_unique<nn::Linear>(4, 3);
        auto r = layer->init(*engine);
        CHECK(r, "Linear init");
        CHECK(layer->engine_alive(), "Layer 持有引擎强引用");
    }

    // 外部 engine shared_ptr 已释放；Layer 的强引用让引擎继续存活。
    CHECK(layer->engine_alive(), "外部句柄释放后 Layer 仍持有引擎");
    auto y = layer->forward(x);
    CHECK(y, "引擎由 Layer 保活，forward 正常");
    CHECK(y && y->rows() == 3 && y->cols() == 2, "forward 形状正确");
}

void test_model_keeps_engine_alive()
{
    std::unique_ptr<nn::Model> model;
    nn::Tensor x;
    {
        auto engine = nn::CpuEngine::create();
        x = engine->create_tensor(4, 2, nn::Precision::F32);
        model = std::make_unique<nn::Model>(*engine);
        auto r = model->add<nn::Linear>(4, 3);
        CHECK(r, "Model::add");
    }

    auto y = model->forward(x);
    CHECK(y, "Model 持有引擎强引用，forward 正常");
    CHECK(y && y->rows() == 3 && y->cols() == 2, "Model 输出形状正确");
}

void test_cross_engine_reinit_rejected()
{
    auto e1 = nn::CpuEngine::create();
    auto e2 = nn::CpuEngine::create();
    FlakyLayer layer;
    layer.attempts = 1;   // 下一次 init_impl 直接成功
    auto r1 = layer.init(*e1);
    CHECK(r1, "绑定 e1 成功");
    auto r2 = layer.init(*e2);
    CHECK(!r2, "跨引擎重绑返回错误");
    CHECK(!r2 && r2.error().message.find("另一个引擎") != std::string::npos,
          "错误信息说明是跨引擎重绑");
    CHECK(layer.engine_alive(), "拒绝后仍绑定 e1");
}

void test_optimizer_keeps_engine_alive()
{
    auto engine = nn::CpuEngine::create();
    nn::Tensor p = engine->create_tensor(
        2, 2, nn::Precision::F32, nn::InitSpec::constant(nn::Scalar{1}));
    nn::Tensor g = engine->create_tensor(
        2, 2, nn::Precision::F32, nn::InitSpec::constant(nn::Scalar{0.5}));
    std::vector<nn::TensorRef> params{std::ref(p)};
    std::vector<nn::TensorRef> grads{std::ref(g)};
    auto opt = nn::create_optimizer("sgd", *engine, params, grads, 0.1f, 0.0f);
    CHECK(opt != nullptr, "create_optimizer");

    // 丢掉外部强引用；Optimizer 内部 keepalive 必须让 step 继续安全。
    engine.reset();
    auto step = opt->step();
    CHECK(step, "Optimizer 持有引擎强引用，step 正常");
}

void test_gpu_buffer_outlives_engine()
{
#ifdef NN_HAS_VULKAN
    // 独立 backend 的完整初始化在部分老卡上要几十秒；默认只跑 CPU 生命周期。
    // 需要验证"最后一个 GpuBuffer 释放 → backend 正常析构"时手动设
    // NN_GPU_LIFETIME_FULL=1。
    if (!nn::detail::env_flag_raw("NN_GPU_LIFETIME_FULL"))
        return;

    nn::Tensor t;
    {
        // 独立 backend（不是进程共享实例），这样作用域结束时确实是最后一个
        // 强引用链的起点。初始化失败（CI 无 GPU）→ 本子用例跳过。
        auto backend = nn::GpuBackend::create();
        auto init = backend->initialize();
        if (!init)
        {
            std::printf("[SKIP] GpuBuffer 生命周期：%s\n",
                        init.error().message.c_str());
            return;
        }
        auto engine = nn::GpuEngine::create(backend);
        t = engine->create_tensor(16, 16, nn::Precision::F32);
        CHECK(t.is_gpu() && t.valid(), "创建 GPU 张量");
    }

    // engine / backend 的局部 shared_ptr 都已释放；GpuBuffer 自己持有
    // backend keepalive，因此这里的复制与析构都必须安全。
    CHECK(t.valid() && t.is_gpu(), "GpuEngine 析构后 GPU 张量仍有效");
    nn::Tensor copy = t;
    CHECK(copy.valid(), "GPU 张量可安全复制");
    t = nn::Tensor{};
    CHECK(copy.valid(), "原句柄释放后副本仍有效");
    // copy 在函数结束时析构：最后一个 buffer 释放 → backend 正常析构。
#endif
}

void test_layer_init_failure_rollback()
{
    auto engine = nn::CpuEngine::create();
    FlakyLayer layer;

    CHECK(layer.binding_state() == nn::Layer::BindingState::unbound,
          "新层状态 = unbound");
    CHECK(!layer.engine_alive(), "新层未持有引擎");

    auto r1 = layer.init(*engine);
    CHECK(!r1, "首次 init 按注入失败");
    CHECK(layer.binding_state() == nn::Layer::BindingState::failed,
          "init 失败后状态 = failed");
    CHECK(layer.engine_alive(), "失败后仍保留可诊断的引擎强引用");

    layer.set_precision_profile(nn::profile_f16());
    CHECK(layer.precision_profile().compute == nn::Precision::F16,
          "failed 状态允许修正 profile");

    auto r2 = layer.init(*engine);
    CHECK(r2, "失败后可重新 init");
    CHECK(layer.binding_state() == nn::Layer::BindingState::ready,
          "重新 init 成功后状态 = ready");
}

} // namespace

int main()
{
    std::printf("── EngineId / Layer / Model / Optimizer / Tensor 生命周期 ──\n");
    test_engine_id();
    test_tensor_survives_engine();
    test_layer_keeps_engine_alive();
    test_model_keeps_engine_alive();
    test_cross_engine_reinit_rejected();
    test_optimizer_keeps_engine_alive();
    test_gpu_buffer_outlives_engine();
    test_layer_init_failure_rollback();

    if (g_fail == 0)
        std::printf("layer_lifetime_test: ALL PASS\n");
    else
        std::printf("layer_lifetime_test: %d failure(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
