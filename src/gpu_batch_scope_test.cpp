// ── gpu_batch_scope_test — GPU 单次推理 batch 作用域（issue #28）──────────
// 覆盖：
//   [1] CPU：InferenceScope 透传（begin/end 均 no-op），active()==true
//       （batch_active()==false ⇒ 自开；CPU 引擎自身 no-op）
//   [2] GPU：Model::forward 不再要求调用方手动 begin_batch/end_batch
//       （多 Conv 模型逐原语路径曾 VK_ERROR_OUT_OF_DEVICE_MEMORY）
//   [3] GPU：外层已持 batch 时 InferenceScope 被动透传（不重复 begin）
//   [4] GPU：外层 begin_batch + InferenceScope + forward 组合不撞
//       "Already in batch mode"
// 无 Vulkan 构建 / 无 GPU 设备 → 返回 77 = ctest SKIP。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/cli/cli_engine_factory.hpp>

#include <iostream>
#include <memory>
#include <vector>

int g_fail = 0;
#define CHECK(cond, msg)                                                                  \
    do                                                                                    \
    {                                                                                     \
        if (!(cond))                                                                      \
        {                                                                                 \
            std::cout << "[FAIL] " << __FILE__ << ":" << __LINE__ << ": " << msg << "\n"; \
            ++g_fail;                                                                     \
        }                                                                                 \
    } while (0)

#ifdef NN_HAS_VULKAN
namespace
{

// 多 Conv 小模型：issue #28 的触发形态（第二次 Conv 分配处 OOM）
nn::Result<nn::Model> make_conv_model(nn::ComputeEngine& eng)
{
    nn::Model m(eng);
    // 位置参数：(in_channels, out_channels, kernel, stride, padding, in_h, in_w)
    NN_TRY(c1, m.add<nn::Conv2D>(1, 8, 3, 1, 0, 8, 8));
    NN_TRY(c2, m.add<nn::Conv2D>(8, 16, 3, 1, 0, 6, 6));
    return m;
}

} // namespace
#endif // NN_HAS_VULKAN

int main()
{
    // [1] CPU 透传
    {
        auto eng = nn::CpuEngine::create();
        CHECK(!eng->batch_active(), "CPU batch_active 恒 false");
        nn::InferenceScope scope(*eng);
        CHECK(scope.ok(), "CPU InferenceScope 应 ok");
        CHECK(scope.active(), "CPU：batch_active()==false ⇒ 自开（引擎侧 no-op）");
        CHECK(!eng->batch_active(), "CPU batch_active 仍 false");
    }

#ifdef NN_HAS_VULKAN
    auto engine_r = nn::cli::create_engine(nn::cli::EngineConfig{.use_gpu = true});
    if (!engine_r)
    {
        std::cout << "[SKIP] GPU 初始化失败：" << engine_r.error().message << "\n";
        return 77;
    }
    nn::ComputeEngine& gpu = **engine_r;

    // [2] GPU：不手动包 batch 直接 forward（issue #28 主诉路径）
    {
        auto model_r = make_conv_model(gpu);
        CHECK(model_r.has_value(), "建多 Conv 模型");
        if (model_r)
        {
            // 输入 (C·H·W, B) = (64, 1)：conv3→(288,1)→conv3→(256,1)
            std::vector<nn::Scalar> img(64, 0.5f);
            auto in = nn::upload_span(gpu, 64, 1, nn::Precision::F32, img);
            CHECK(in.has_value(), "upload 输入");
            if (in)
            {
                // 曾在此路径（无 begin_batch）第二次 Conv 分配报
                // VK_ERROR_OUT_OF_DEVICE_MEMORY；Model::forward 现自动作用域
                auto out = model_r->forward(*in);
                CHECK(out.has_value(), "GPU Model::forward 无手动 batch 应成功");
            }
        }
        CHECK(!gpu.batch_active(), "forward 返回后 batch 已收口");
    }

    // [3] 外层已持 batch：InferenceScope 被动透传
    {
        nn::InferenceScope outer(gpu);
        CHECK(outer.ok() && outer.active(), "外层作用域自开");
        nn::InferenceScope inner(gpu);
        CHECK(inner.ok(), "内层作用域 ok");
        CHECK(!inner.active(), "内层：外层已持 batch ⇒ 透传（不重复 begin）");
    }
    CHECK(!gpu.batch_active(), "外层作用域析构后 batch 已收口");

    // [4] 手动 begin_batch + InferenceScope 组合（模拟训练循环内评估）
    {
        auto br = gpu.begin_batch();
        CHECK(br.has_value(), "手动 begin_batch");
        nn::InferenceScope scope(gpu);
        CHECK(scope.ok() && !scope.active(), "手动 batch 下作用域透传");
        auto er = gpu.end_batch();
        CHECK(er.has_value(), "手动 end_batch");
    }

    std::cout << (g_fail == 0 ? "\n全部通过 ✅\n" : "\n存在失败 ❌\n");
    return g_fail == 0 ? 0 : 1;
#else
    std::cout << "[SKIP] 纯 CPU 构建（无 Vulkan）\n";
    return 77;
#endif
}
