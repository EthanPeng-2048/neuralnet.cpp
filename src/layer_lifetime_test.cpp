// ── layer_lifetime_test — 生命周期 v3：设备域模型与强所有权 ────────────────
//
// v3 契约（docs/todo/28-lifecycle-v3.md §4）：
//   [1] 域从存储派生：CPU 张量 → kHostDomain；空/无效 = 0（唯一豁免）；
//       Tensor 不再保存任何引擎身份字段
//   [2] 引擎不可栈上构造（必须由 create() 的 shared_ptr 管理）
//   [3] Tensor 晚于引擎析构仍可复制/析构（只拥有存储）
//   [4] Layer / Model / Optimizer 持有 shared_ptr<ComputeEngine>：外部句柄释放
//       后仍可 forward/step
//   [5] 跨 CPU 引擎使用张量合法（同一宿主域）；Layer 跨引擎重绑仍拒绝
//   [6] 跨域 op 拒绝（CPU 张量进 GPU 引擎 / GPU 张量进 CPU 引擎 / 跨 backend），
//       import 是唯一 CPU→GPU 入口（GPU 构建 + 可用设备时）
//   [7] Layer init 失败回滚与 profile 注入
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

void test_domain_model()
{
    static_assert(std::is_same_v<nn::DomainId, std::uintptr_t>,
                  "DomainId = 不透明整数域标识");
    static_assert(nn::kHostDomain != 0, "宿主域非零（0 保留给空张量）");
    static_assert(!std::is_default_constructible_v<nn::CpuEngine>,
                  "CpuEngine 不允许栈上构造（必须用 create()）");
#ifdef NN_HAS_VULKAN
    static_assert(!std::is_default_constructible_v<nn::GpuEngine>,
                  "GpuEngine 不允许栈上构造（必须用 create()）");
#endif

    auto e1 = nn::CpuEngine::create();
    auto e2 = nn::CpuEngine::create();
    CHECK(e1->domain() == nn::kHostDomain, "CPU 引擎域 = kHostDomain");
    CHECK(e2->domain() == nn::kHostDomain, "所有 CPU 引擎共享宿主域");

    // 空张量 = 域 0（唯一豁免）；非空 CPU 张量属于宿主域。
    // device_domain_() 是引擎私有通道，库外不暴露：这里用"跨 CPU 引擎可用"
    // 间接证明两台 CPU 引擎确实同域。
    nn::Tensor t = e1->create_tensor(2, 2, nn::Precision::F32);
    CHECK(t.valid(), "e1 创建张量");
    auto c = e2->clone(t);
    CHECK(c, "跨 CPU 引擎使用张量合法（同一宿主域）");
}

void test_tensor_survives_engine()
{
    nn::Tensor t;
    {
        auto engine = nn::CpuEngine::create();
        t = engine->create_tensor(2, 2, nn::Precision::F32);
    }

    // 引擎已析构：Tensor 只拥有存储，复制/析构/读元数据都安全。
    CHECK(t.valid() && t.rows() == 2 && t.cols() == 2, "引擎析构后张量仍有效");
    nn::Tensor copy = t;
    CHECK(copy.valid(), "Tensor 可安全复制");
    t = nn::Tensor{};
    CHECK(copy.valid(), "原句柄释放后副本仍有效");
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

void test_backend_shared_weak_cache()
{
#ifdef NN_HAS_VULKAN
    // D5：shared() = weak 缓存。有活使用者时复用；全部释放后确定性回收；
    // 下一次 shared() 重新创建（不初始化设备，调用方仍需 initialize()）。
    std::weak_ptr<nn::GpuBackend> watch;
    {
        auto b = nn::GpuBackend::shared();
        CHECK(b != nullptr, "shared() 返回实例");
        watch = b;
        auto b2 = nn::GpuBackend::shared();
        CHECK(b2.get() == b.get(), "有活使用者时 shared() 复用同一实例");

        auto engine = nn::GpuEngine::create(b);
        b.reset();
        b2.reset();
        CHECK(!watch.expired(), "引擎句柄保活 backend");
        engine.reset();
    }
    CHECK(watch.expired(), "全部使用者释放后 backend 确定性回收");
    auto b3 = nn::GpuBackend::shared();
    CHECK(b3 != nullptr, "回收后 shared() 可重建");
#else
    std::printf("[SKIP] backend weak 缓存：构建未启用 Vulkan\n");
#endif
}

void test_cross_domain_rejected()
{
#ifdef NN_HAS_VULKAN
    // 独立 backend（不污染进程共享实例）；无设备/初始化失败 → 跳过
    auto backend = nn::GpuBackend::create();
    if (!backend->initialize())
    {
        std::printf("[SKIP] 跨域拒绝：本机无可用 Vulkan 设备\n");
        return;
    }
    auto gpu = nn::GpuEngine::create(backend);
    auto cpu = nn::CpuEngine::create();

    nn::Tensor host = cpu->create_tensor(2, 2, nn::Precision::F32);
    nn::Tensor dev = gpu->create_tensor(2, 2, nn::Precision::F32);

    // GPU 引擎拒绝 CPU 操作数（不再有逐 op 自动 import）
    auto bad_clone = gpu->clone(host);
    CHECK(!bad_clone, "CPU 张量进 GPU 引擎 : clone 返回错误");
    if (!bad_clone)
        CHECK(bad_clone.error().message.find("设备域") != std::string::npos,
              "错误信息点明设备域不匹配");

    // CPU 引擎拒绝 GPU 操作数（落宿主必须走 to_matrix）
    auto bad_cpu = cpu->clone(dev);
    CHECK(!bad_cpu, "GPU 张量进 CPU 引擎 : clone 返回错误");

    // import 是唯一 CPU→GPU 入口；导入后同域 op 正常
    auto imported = gpu->import(host);
    CHECK(imported, "import(CPU 张量) 成功");
    if (imported)
    {
        auto ok = gpu->clone(*imported);
        CHECK(ok, "import 后的 GPU 张量可参与 GPU 引擎 op");
    }

    // 跨 backend：另一个独立 backend 的引擎拒绝该张量（提示先落宿主）
    auto backend2 = nn::GpuBackend::create();
    if (backend2->initialize())
    {
        auto gpu2 = nn::GpuEngine::create(backend2);
        auto cross = gpu2->clone(dev);
        CHECK(!cross, "跨 GPU backend 直接使用张量返回错误");
        auto back = gpu->to_matrix(dev);
        CHECK(back, "GPU→宿主 to_matrix 正常");
    }
#else
    std::printf("[SKIP] 跨域拒绝：构建未启用 Vulkan\n");
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
    std::printf("── 生命周期 v3：设备域 / 强所有权 / 跨域拒绝 ──\n");
    test_domain_model();
    test_tensor_survives_engine();
    test_layer_keeps_engine_alive();
    test_model_keeps_engine_alive();
    test_cross_engine_reinit_rejected();
    test_optimizer_keeps_engine_alive();
    test_backend_shared_weak_cache();
    test_cross_domain_rejected();
    test_layer_init_failure_rollback();

    if (g_fail == 0)
        std::printf("layer_lifetime_test: ALL PASS\n");
    else
        std::printf("layer_lifetime_test: %d failure(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
