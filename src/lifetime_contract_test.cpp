// ── lifetime_contract_test — P0 生命周期契约（CPU 可跑的负例）───────────────
//
// 覆盖 docs/todo/28-lifecycle-v3.md §5 P0 的三处安全修复（不破坏 API）：
//   [1] ExprSegment 持 shared_ptr<ComputeEngine>：录制段内丢掉外部最后一个
//       引擎句柄，end() 仍然安全（原实现持裸引用 = 析构 UAF）
//   [2] ActivationOffloader 状态机：未 import 的导出不能被下一次 export 静默
//       覆盖；discard() 是 forward 入口的作废通道
//   [3] offload 精度门：slab 恒 f32，非 F32 激活导出必须 fail-fast
//   [4] optimizer 先于 Model 析构（契约要求的顺序）下 step 正常
// ─────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "test_common.hpp"

int g_fail = 0;

namespace
{

void test_expr_segment_holds_engine()
{
    auto engine = nn::CpuEngine::create();
    CHECK(engine != nullptr, "CpuEngine::create");

    auto begin = engine->begin_expr();
    CHECK(begin, "begin_expr");
    {
        nn::ExprSegment seg(engine);
        engine.reset();   // 外部最后一个句柄释放；守卫必须保活引擎
        auto end = seg.end();
        CHECK(end, "ExprSegment 强引用：外部句柄释放后 end() 仍安全");
    }
    CHECK(engine == nullptr, "外部引擎句柄确实已释放");
}

void test_offload_state_machine()
{
#ifdef NN_HAS_VULKAN
    auto backend = nn::GpuBackend::create();
    if (!backend->initialize())
    {
        std::printf("[SKIP] offload 状态机：本机无可用 Vulkan 设备\n");
        return;
    }
    auto engine = nn::GpuEngine::create(backend);
    nn::Tensor a = engine->create_tensor(2, 3, nn::Precision::F32);
    nn::Tensor b = engine->create_tensor(4, 1, nn::Precision::F32);
    CHECK(a.valid() && b.valid(), "创建 F32 激活张量");

    nn::ActivationOffloader off;
    off.set_enabled(true);

    auto r1 = off.export_activations(*engine, {a, b});
    CHECK(r1, "首次 export 成功");
    CHECK(off.offloaded(), "export 后状态 = Offloaded");

    // 未 import 就再次 export：必须报错。原实现静默 return {} —— 下一次
    // backward 会把上一轮 slab 当成这一轮的激活导回（静默错值）。
    auto r2 = off.export_activations(*engine, {a, b});
    CHECK(!r2, "未 import 的重复 export 必须 fail-fast");
    if (!r2)
        CHECK(r2.error().message.find("尚未 import") != std::string::npos,
              "错误信息点明未 import");

    // forward 入口的作废通道：discard 后状态复位，可重新导出
    off.discard();
    CHECK(!off.offloaded(), "discard 后状态 = Idle");
    a = engine->create_tensor(2, 3, nn::Precision::F32);
    b = engine->create_tensor(4, 1, nn::Precision::F32);
    auto r3 = off.export_activations(*engine, {a, b});
    CHECK(r3 && off.offloaded(), "discard 后可重新 export");

    auto im = off.import_activations(*engine);
    CHECK(im, "import 成功");
    CHECK(!off.offloaded(), "import 后状态 = Idle");

    // 关闭开关 = 作废未消费的导出（成员缓存已置空，不能静默保留 Offloaded）
    CHECK(off.export_activations(*engine, {a, b}), "再次 export");
    off.set_enabled(false);
    CHECK(!off.offloaded(), "set_enabled(false) 复位状态");
#else
    std::printf("[SKIP] offload 状态机：构建未启用 Vulkan\n");
#endif
}

void test_offload_rejects_cpu_engine()
{
    // CPU 引擎的 offload 原语是 no-op（restore 返回 1×1）→ export 必须拒绝，
    // 否则成员缓存被置空、backward 拿到 1×1 张量（CLI 与 Session 也各自拒绝）。
    auto engine = nn::CpuEngine::create();
    nn::Tensor a = engine->create_tensor(2, 2, nn::Precision::F32);
    nn::ActivationOffloader off;
    off.set_enabled(true);
    auto r = off.export_activations(*engine, {a});
    CHECK(!r, "CPU 引擎 export 必须 fail-fast");
    if (!r)
        CHECK(r.error().message.find("仅支持") != std::string::npos,
              "错误信息说明仅支持 GPU");
    CHECK(!off.offloaded(), "失败后状态保持 Idle");
    CHECK(a.valid(), "失败路径不得改动激活张量");
}

void test_offload_rejects_non_f32()
{
    auto engine = nn::CpuEngine::create();
    nn::Tensor h = engine->create_tensor(2, 2, nn::Precision::F16);
    CHECK(h.valid() && h.precision() == nn::Precision::F16, "创建 F16 激活张量");

    nn::ActivationOffloader off;
    off.set_enabled(true);
    auto r = off.export_activations(*engine, {h});
    CHECK(!r, "非 F32 激活导出必须 fail-fast");
    if (!r)
        CHECK(r.error().message.find("f16") != std::string::npos,
              "错误信息带出实际精度");
    CHECK(!off.offloaded(), "失败后状态保持 Idle");
    CHECK(h.valid(), "失败路径不得改动激活张量");
}

void test_optimizer_before_model_order()
{
    auto engine = nn::CpuEngine::create();
    nn::Model model(*engine);
    CHECK(model.add<nn::Linear>(4, 3), "add Linear(4,3)");
    CHECK(model.add<nn::ReLU>(), "add ReLU");
    CHECK(model.add<nn::Linear>(3, 2), "add Linear(3,2)");

    {
        // 契约要求：优化器必须在参数所有者（Model）之前销毁
        auto opt = nn::create_optimizer("adam", *engine, model.parameters(),
                                        model.param_gradients(), 1e-3f);
        CHECK(opt != nullptr, "create_optimizer");
        auto zero = model.zero_grad();
        CHECK(zero, "zero_grad");
        auto step = opt->step();
        CHECK(step, "optimizer 先于 Model 析构时 step 正常");
    }   // opt 先析构 → 随后 model 析构，无悬垂引用
}

} // namespace

int main()
{
    std::printf("── P0 生命周期契约（ExprSegment / offload / optimizer 顺序）──\n");
    test_expr_segment_holds_engine();
    test_offload_state_machine();
    test_offload_rejects_cpu_engine();
    test_offload_rejects_non_f32();
    test_optimizer_before_model_order();

    if (g_fail == 0)
        std::printf("lifetime_contract_test: ALL PASS\n");
    else
        std::printf("lifetime_contract_test: %d failure(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
