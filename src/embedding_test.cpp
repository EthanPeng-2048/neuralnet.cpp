// ── embedding_test — Embedding 层（issue #32）─────────────────────────────
//
// 覆盖：
//   [1] forward 查表语义（out[i] = table[ids[i]]，对拍下载的表行）
//   [2] 输出布局 (d_model, N) 列布局 + batch>1（ids 任意形状 flat 遍历）
//   [3] backward 伴随正确性：重复 id 梯度累加；padding 行梯度恒零
//   [4] 索引无梯度（返回零张量）；ids_cache 缺失时 backward fail-fast
//   [5] forward_ids 整数入口：查表一致 + 越界 fail-fast
//   [6] padding 行初值恒零；参数协议（parameters 1 条、梯度惰性分配）
// ─────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

#include "neuralnet.cpp/nn.hpp"
#include "test_common.hpp"

int g_fail = 0;

namespace {

// 取表的某一行（宿主侧对拍基准）
std::vector<nn::Scalar> table_row(nn::ComputeEngine& eng, nn::Layer& emb,
                                  std::size_t id)
{
    auto& t = emb.parameters().at(0).get();
    auto m = download(eng, t);
    std::vector<nn::Scalar> out(m.cols());
    const auto sp = m.span();
    for (std::size_t c = 0; c < m.cols(); ++c)
        out[c] = sp[id * m.cols() + c];
    return out;
}

void test_forward_gather()
{
    auto engine = nn::CpuEngine::create();
    nn::Embedding emb(10, 4);
    CHECK(emb.init(*engine).has_value(), "init");
    // ids (2,3)：batch>1 覆盖（flat 遍历 = 列主序索引）
    const nn::Scalar ids_v[6] = {0, 3, 9, 3, 7, 0};
    nn::Matrix ids_m(2, 3);
    { auto s = ids_m.span(); for (int i = 0; i < 6; ++i) s[i] = ids_v[i]; }
    auto ids = upload(*engine, ids_m);

    auto out = emb.forward(ids);
    CHECK(out.has_value(), "[1] forward 成功");
    if (!out) return;
    CHECK(out->rows() == 4 && out->cols() == 6, "[2] 输出形状 (d_model, N) = (4,6)");
    auto om = download(*engine, *out);
    const auto osp = om.span();
    for (int i = 0; i < 6; ++i)
    {
        const auto row = table_row(*engine, emb, static_cast<std::size_t>(ids_v[i]));
        for (std::size_t c = 0; c < 4; ++c)
        {
            // 输出 (d_model=4, N=6)：列 i = 第 i 个索引的嵌入向量
            const float got = osp[c * 6 + static_cast<std::size_t>(i)];
            CHECK(std::fabs(got - row[c]) < 1e-6f, "[1] 查表行逐元素一致");
        }
    }
}

void test_backward_scatter()
{
    auto engine = nn::CpuEngine::create();
    nn::Embedding emb(6, 3);
    CHECK(emb.init(*engine).has_value(), "init");
    // ids 含重复：id 2 出现两次 → 梯度应为两次之和
    const nn::Scalar ids_v[4] = {2, 0, 2, 5};
    nn::Matrix ids_m(4, 1);
    { auto s = ids_m.span(); for (int i = 0; i < 4; ++i) s[i] = ids_v[i]; }
    auto ids = upload(*engine, ids_m);
    CHECK(emb.forward(ids).has_value(), "[3] forward");

    // grad_output (d_model=3, N=4)：第 i 列 = i+1
    nn::Matrix g(3, 4);
    { auto s = g.span(); for (int i = 0; i < 12; ++i) s[i] = static_cast<float>(i + 1); }
    auto grad = upload(*engine, g);
    auto gi = emb.backward(grad);
    CHECK(gi.has_value(), "[3] backward 成功");
    if (!gi) return;

    // 索引梯度 = 零
    auto gim = download(*engine, *gi);
    bool all_zero = true;
    for (auto v : gim.span()) if (v != 0.0f) all_zero = false;
    CHECK(all_zero, "[4] 索引无梯度（零张量）");

    // 表梯度：grad (3,4) 行主序 → 列0=(1,5,9) 列2=(3,7,11)；
    // id 2 出现在索引 0 与 2 → 行 = 列0+列2 = (4,12,20)
    auto& gt = emb.param_gradients().at(0).get();
    auto gtm = download(*engine, gt);
    const auto sp = gtm.span();
    const float expect[3] = {4.f, 12.f, 20.f};
    for (std::size_t c = 0; c < 3; ++c)
        CHECK(std::fabs(sp[2 * 3 + c] - expect[c]) < 1e-5f,
              "[3] 重复 id 梯度累加");
    // id 0 的行 = 列1 = (2,6,10)
    const float expect0[3] = {2.f, 6.f, 10.f};
    for (std::size_t c = 0; c < 3; ++c)
        CHECK(std::fabs(sp[0 * 3 + c] - expect0[c]) < 1e-5f, "[3] id0 梯度");
    // id 5 的行 = 列3 = (4,8,12)
    const float expect5[3] = {4.f, 8.f, 12.f};
    for (std::size_t c = 0; c < 3; ++c)
        CHECK(std::fabs(sp[5 * 3 + c] - expect5[c]) < 1e-5f, "[3] id5 梯度");
    // 未出现的 id 1,3,4 = 零
    for (std::size_t id : {std::size_t{1}, std::size_t{3}, std::size_t{4}})
        for (std::size_t c = 0; c < 3; ++c)
            CHECK(sp[id * 3 + c] == 0.0f, "[3] 未出现 id 梯度为零");
}

void test_padding()
{
    auto engine = nn::CpuEngine::create();
    nn::Embedding emb(5, 3, /*padding_idx=*/1);
    CHECK(emb.init(*engine).has_value(), "init");
    // [6] padding 行初值恒零
    const auto row1 = table_row(*engine, emb, 1);
    for (auto v : row1) CHECK(v == 0.0f, "[6] padding 行初值为零");

    // forward 含 padding id（输出列应为零向量）
    const nn::Scalar ids_v[3] = {1, 0, 1};
    nn::Matrix ids_m(3, 1);
    { auto s = ids_m.span(); for (int i = 0; i < 3; ++i) s[i] = ids_v[i]; }
    auto ids = upload(*engine, ids_m);
    auto out = emb.forward(ids);
    CHECK(out.has_value(), "[6] forward 含 padding id");
    if (!out) return;
    auto om = download(*engine, *out);
    const auto osp = om.span();
    for (std::size_t c = 0; c < 3; ++c)
    {
        CHECK(osp[c * 3 + 0] == 0.0f, "[6] padding 输出列为零");
        CHECK(osp[c * 3 + 2] == 0.0f, "[6] padding 输出列为零（第二处）");
    }

    // backward 后 padding 行梯度仍为零
    nn::Matrix g(3, 3);
    { auto s = g.span(); for (int i = 0; i < 9; ++i) s[i] = 1.0f; }
    auto grad = upload(*engine, g);
    CHECK(emb.backward(grad).has_value(), "[3] backward（含 padding id）");
    auto& gt = emb.param_gradients().at(0).get();
    auto gtm = download(*engine, gt);
    const auto sp = gtm.span();
    for (std::size_t c = 0; c < 3; ++c)
        CHECK(sp[1 * 3 + c] == 0.0f, "[3] padding 行梯度恒零");
}

void test_forward_ids()
{
    auto engine = nn::CpuEngine::create();
    nn::Embedding emb(8, 2);
    CHECK(emb.init(*engine).has_value(), "init");
    const std::size_t ids[3] = {7, 4, 0};
    auto out = emb.forward_ids(std::span(ids));
    CHECK(out.has_value(), "[5] forward_ids 成功");
    if (!out) return;
    CHECK(out->rows() == 2 && out->cols() == 3, "[5] forward_ids 输出形状");
    auto om = download(*engine, *out);
    const auto osp = om.span();
    for (int i = 0; i < 3; ++i)
    {
        const auto row = table_row(*engine, emb, ids[i]);
        for (std::size_t c = 0; c < 2; ++c)
            CHECK(std::fabs(osp[c * 3 + static_cast<std::size_t>(i)] - row[c]) < 1e-6f,
                  "[5] forward_ids 查表一致");
    }
    // 越界 fail-fast
    const std::size_t bad[1] = {8};
    auto r = emb.forward_ids(std::span(bad));
    CHECK(!r.has_value(), "[5] 越界索引 fail-fast");
}

void test_param_protocol()
{
    auto engine = nn::CpuEngine::create();
    nn::Embedding emb(4, 2);
    CHECK(!emb.grad_buffers_allocated(), "[6] init 前无梯度缓冲（F2）");
    CHECK(emb.init(*engine).has_value(), "[6] init");
    CHECK(emb.parameters().size() == 1, "[6] parameters 恰 1 条（嵌入表）");
    CHECK(!emb.grad_buffers_allocated(), "[6] init 后仍未分配梯度（惰性）");
    CHECK(emb.param_gradients().size() == 1, "[6] param_gradients 恰 1 条");
    CHECK(emb.grad_buffers_allocated(), "[6] 取梯度后已分配");
    CHECK(emb.parameters().at(0).get().rows() == 4 &&
          emb.parameters().at(0).get().cols() == 2,
          "[6] 表形状 (vocab, d_model)");

    // backward 前无索引 → fail-fast
    nn::Matrix g(2, 1);
    { auto s = g.span(); s[0] = 1.f; s[1] = 1.f; }
    auto grad = upload(*engine, g);
    auto r = emb.backward(grad);
    CHECK(!r.has_value(), "[4] 无缓存索引时 backward fail-fast");
}

} // namespace

int main()
{
    test_forward_gather();
    test_backward_scatter();
    test_padding();
    test_forward_ids();
    test_param_protocol();
    if (g_fail == 0) std::printf("embedding_test: ALL PASS\n");
    else std::printf("embedding_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
