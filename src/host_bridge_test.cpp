// ── host_bridge_test — 公开宿主桥 nn::upload_span / download_span / download_vector
//
// issue #30：宿主桥原在 nn::detail（无稳定性承诺），v1.14.0 升公开 API。
// 本测试锁定公开形态的语义契约：
//   [1] upload_span F32 往返逐位一致；
//   [2] upload_span F16：RHE 舍入（与 from_matrix 同口径）+ download 升 f32 无损；
//   [3] 形状/元素数不匹配报错；
//   [4] nn::detail:: 旧名过渡别名仍可用（向后兼容）。
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>
#include <neuralnet.cpp/compute_engine.hpp>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

int g_fail = 0;
#define CHECK(cond, msg)                                                  \
    do                                                                    \
    {                                                                     \
        if (!(cond))                                                      \
        {                                                                 \
            std::printf("[FAIL] %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

int main()
{
    auto engine = nn::CpuEngine::create();

    // [1] F32 往返逐位一致
    {
        std::vector<nn::Scalar> src{0.1f, -2.5f, 3.75f, 0.0f, 42.0f, -0.0f};
        auto t = nn::upload_span(*engine, 2, 3, nn::Precision::F32, src);
        CHECK(t.has_value(), "upload_span F32 应成功");
        if (t)
        {
            CHECK(t->rows() == 2 && t->cols() == 3, "upload_span F32 形状");
            std::vector<nn::Scalar> dst(src.size());
            auto r = nn::download_span(*engine, *t, dst);
            CHECK(r.has_value(), "download_span F32 应成功");
            for (std::size_t i = 0; i < src.size(); ++i)
                CHECK(src[i] == dst[i], "F32 往返应逐位一致");
        }
    }

    // [2] F16：RHE 舍入 + download 升 f32 无损（与 from_matrix 同口径）
    {
        std::vector<nn::Scalar> src{1.0f, 0.1f, 1.0f + 1.0f / 2048.0f, -3.5f};
        auto t = nn::upload_span(*engine, 2, 2, nn::Precision::F16, src);
        CHECK(t.has_value(), "upload_span F16 应成功");
        if (t)
        {
            CHECK(t->precision() == nn::Precision::F16, "F16 目标精度");
            // 对拍 from_matrix 同值（RHE 同路径）
            nn::Matrix m(2, 2);
            for (std::size_t i = 0; i < src.size(); ++i)
                m.span()[i] = src[i];
            auto ref = engine->from_matrix(m, nn::Precision::F16);
            CHECK(ref.has_value(), "from_matrix F16 应成功");
            if (ref)
            {
                auto v1 = nn::download_vector(*engine, *t);
                auto v2 = engine->to_matrix(*ref);
                CHECK(v1.has_value() && v2.has_value(), "download 应成功");
                if (v1 && v2)
                    for (std::size_t i = 0; i < src.size(); ++i)
                        CHECK((*v1)[i] == v2->span()[i],
                              "upload_span F16 应与 from_matrix 逐位同口径");
            }
        }
    }

    // [3] 元素数与形状不匹配 = 运行期错误
    {
        std::vector<nn::Scalar> src{1.0f, 2.0f};
        auto t = nn::upload_span(*engine, 2, 3, nn::Precision::F32, src);
        CHECK(!t.has_value(), "元素数不匹配应报错");
        auto t2 =
            nn::upload_span(*engine, 1, 2, nn::Precision::F32, std::vector<nn::Scalar>{1.0f, 2.0f});
        CHECK(t2.has_value(), "匹配尺寸应成功");
        if (t2)
        {
            std::vector<nn::Scalar> dst(3);
            auto r = nn::download_span(*engine, *t2, dst);
            CHECK(!r.has_value(), "download 元素数不匹配应报错");
        }
    }

    // [4] nn::detail:: 过渡别名可用（旧调用点不破坏）
    {
        std::vector<nn::Scalar> src{7.0f, 8.0f};
        auto t = nn::detail::upload_span(*engine, 1, 2, nn::Precision::F32, src);
        CHECK(t.has_value(), "detail::upload_span 别名应可用");
        if (t)
        {
            auto v = nn::detail::download_vector(*engine, *t);
            CHECK(v.has_value() && (*v)[0] == 7.0f && (*v)[1] == 8.0f,
                  "detail::download_vector 别名应可用");
        }
    }

    std::cout << (g_fail == 0 ? "\n全部通过 ✅\n" : "\n存在失败 ❌\n");
    return g_fail == 0 ? 0 : 1;
}
