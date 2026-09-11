// ── f16_compute_test.cpp — D6：f16 计算验证（CPU matmul + cast 往返）───────
// 验收标准（docs/23 §13）：
//   1. f16 matmul 结果与 f32 参考在容差内一致（§7.2：f32 累加 + f16 舍入）
//   2. cast 往返：f32 → f16 → f32 的精度损失在 f16 ulp 内
//   3. f16 线性模型训练：3 步后 loss 下降（端到端正确性）
//   4. f16 batched_matmul 与 f32 参考一致
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "neuralnet.cpp/precision.hpp"
#include "neuralnet.cpp/compute_tensor.hpp"
#include "neuralnet.cpp/compute_cpu_engine.hpp"

namespace
{

int g_failures = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "  FAIL line %d: %s\n", __LINE__, (msg)); \
            ++g_failures; \
        } \
    } while (0)

// ── T1：f16 matmul vs f32 参考 ─────────────────────────────────────────
void test_f16_matmul_vs_f32()
{
    std::printf("  [1] f16 matmul vs f32 reference...");
    nn::CpuEngine engine;

    // 创建 f32 矩阵 A(4,8) B(8,3)，用已知值
    nn::Matrix a32(4, 8);
    nn::Matrix b32(8, 3);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : a32.span()) v = dist(rng);
    for (auto& v : b32.span()) v = dist(rng);

    // f32 matmul 参考结果
    nn::Matrix c32_ref(4, 3);
    a32.multiply_to(c32_ref, b32);

    // f16 matmul：先 cast A/B 到 f16，再做 f16 matmul
    auto t_a32 = nn::Tensor::from_matrix(a32);
    auto t_b32 = nn::Tensor::from_matrix(b32);

    auto t_a16_r = engine.cast(t_a32, nn::Precision::F16);
    CHECK(t_a16_r.has_value(), "cast A to f16");
    auto t_b16_r = engine.cast(t_b32, nn::Precision::F16);
    CHECK(t_b16_r.has_value(), "cast B to f16");

    auto t_c16_r = engine.matmul(*t_a16_r, *t_b16_r, false, false, nn::Precision::F16);
    CHECK(t_c16_r.has_value(), "f16 matmul");

    // cast 结果回 f32 比较
    auto t_c32_r = engine.cast(*t_c16_r, nn::Precision::F32);
    CHECK(t_c32_r.has_value(), "cast result to f32");
    const auto& c32 = t_c32_r->cpu_matrix();

    // 比较：f16 matmul 结果与 f32 参考的相对误差
    // §7.2：f16 matmul = f32 参考 + f16 舍入；误差来源 = f16 输入精度 + f16 输出舍入
    // 容差：|f16_result - f32_ref| <= 数倍 f16 ulp（经验值：~10 倍 ulp 足够）
    float max_rel_err = 0.0f;
    for (std::size_t i = 0; i < 4 * 3; ++i)
    {
        float v32 = c32.span()[i];
        float vref = c32_ref.span()[i];
        float diff = std::fabs(v32 - vref);
        float ref_abs = std::fabs(vref);
        float rel_err = (ref_abs > 1e-6f) ? diff / ref_abs : diff;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }
    // f16 相对精度 ~2^-10 ≈ 0.001；matmul 累积误差约 10 倍
    CHECK(max_rel_err < 0.05f, "f16 matmul relative error within 5%");

    std::printf(" done (max_rel_err=%.6f)\n", max_rel_err);
}

// ── T2：cast 往返精度 ─────────────────────────────────────────────────
void test_cast_roundtrip()
{
    std::printf("  [2] cast roundtrip precision...");
    nn::CpuEngine engine;

    nn::Matrix m32(3, 4);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : m32.span()) v = dist(rng);

    auto t32 = nn::Tensor::from_matrix(m32);

    // f32 → f16 → f32
    auto t16_r = engine.cast(t32, nn::Precision::F16);
    CHECK(t16_r.has_value(), "cast to f16");
    auto t32_back_r = engine.cast(*t16_r, nn::Precision::F32);
    CHECK(t32_back_r.has_value(), "cast back to f32");

    const auto& result = t32_back_r->cpu_matrix();
    float max_err = 0.0f;
    for (std::size_t i = 0; i < 3 * 4; ++i)
    {
        float orig = m32.span()[i];
        float back = result.span()[i];
        float err = std::fabs(orig - back);
        if (err > max_err) max_err = err;
    }
    // f16 精度：尾数 10 位 → 对于 |v| < 1 的值，绝对误差 < 2^-10 ≈ 0.001
    // 对于 |v| ~ 10，绝对误差 < 10 * 2^-10 ≈ 0.01
    CHECK(max_err < 0.1f, "cast roundtrip error within f16 precision");

    std::printf(" done (max_err=%.6f)\n", max_err);
}

// ── T3：f16 线性模型训练 — loss 下降 ──────────────────────────────────
void test_f16_linear_training()
{
    std::printf("  [3] f16 linear training (loss decrease)...");
    nn::CpuEngine engine;

    // 简单线性回归：y = 2*x + 1，用 f16 matmul 训练
    const std::size_t N = 32;  // batch size
    const std::size_t D = 4;   // feature dim
    const int STEPS = 5;

    // 生成数据：X ~ N(0,1), y = X @ w_true + noise
    std::mt19937 rng(42);
    std::normal_distribution<float> ndist(0.0f, 0.5f);

    nn::Matrix x_data(D, N);
    nn::Matrix w_true(D, 1);
    for (auto& v : x_data.span()) v = ndist(rng);
    w_true.set_value(0, 0, 2.0f);
    w_true.set_value(1, 0, -1.0f);
    w_true.set_value(2, 0, 0.5f);
    w_true.set_value(3, 0, -0.3f);

    // y = X^T @ w_true → (N, 1)
    nn::Matrix y_data(N, 1);
    for (std::size_t i = 0; i < N; ++i)
    {
        float sum = 0.0f;
        for (std::size_t d = 0; d < D; ++d)
            sum += x_data.at(d, i) * w_true.at(d, 0);
        y_data.set_value(i, 0, sum);
    }

    // 初始化权重 w（随机）
    nn::Matrix w(D, 1);
    for (auto& v : w.span()) v = ndist(rng) * 0.1f;

    float lr = 0.01f;
    float prev_loss = 1e10f;

    for (int step = 0; step < STEPS; ++step)
    {
        // Cast 到 f16
        auto t_x16 = engine.cast(nn::Tensor::from_matrix(x_data), nn::Precision::F16);
        auto t_w16 = engine.cast(nn::Tensor::from_matrix(w), nn::Precision::F16);
        auto t_y32 = nn::Tensor::from_matrix(y_data);

        // forward: pred = X @ w  (f16 matmul)
        auto t_pred_r = engine.matmul(*t_x16, *t_w16, true, false, nn::Precision::F16);
        CHECK(t_pred_r.has_value(), "f16 matmul forward");

        // 边界 cast：f16 → f32 进入融合世界（§11.1 Phase 1 策略）
        auto t_pred32_r = engine.cast(*t_pred_r, nn::Precision::F32);
        CHECK(t_pred32_r.has_value(), "cast pred to f32");
        CHECK(t_pred32_r->precision() == nn::Precision::F32, "cast result is F32");
        CHECK(t_pred32_r->is_cpu(), "cast result is CPU");

        // loss = mean((pred - y)^2) — 在 f32 融合世界中计算
        auto t_diff_r = engine.elementwise_binary(
            nn::BinaryOp::Sub, *t_pred32_r, t_y32);
        CHECK(t_diff_r.has_value(), "sub");
        auto t_loss_r = engine.elementwise_binary(
            nn::BinaryOp::Mul, *t_diff_r, *t_diff_r);
        CHECK(t_loss_r.has_value(), "sqr");
        // 简化：取所有元素的平均作为 loss（通过 to_matrix 读回 CPU）
        auto loss_mat = engine.to_matrix(*t_loss_r);
        CHECK(loss_mat.has_value(), "to_matrix for loss");
        float loss = 0.0f;
        for (auto& v : loss_mat->span()) loss += v;
        loss /= static_cast<float>(N);

        // 反向：grad_w = 2/N * X @ (pred - y)  (f16 matmul)
        // scale 在 f32 融合世界完成，然后 cast 回 f16 做 matmul
        auto t_scale_r = engine.elementwise_binary_scalar(
            nn::BinaryOp::Mul, *t_diff_r, 2.0f / static_cast<float>(N), false);
        CHECK(t_scale_r.has_value(), "scale grad");
        // grad_w = X @ scaled_diff: X(D,N)^T * scaled_diff(N,1) = (D,1)
        auto t_scale16_r = engine.cast(*t_scale_r, nn::Precision::F16);
        CHECK(t_scale16_r.has_value(), "cast scale to f16");
        auto t_grad_r = engine.matmul(*t_x16, *t_scale16_r, false, false, nn::Precision::F16);
        CHECK(t_grad_r.has_value(), "f16 matmul backward");

        // cast grad 回 f32 更新权重
        auto t_grad32_r = engine.cast(*t_grad_r, nn::Precision::F32);
        CHECK(t_grad32_r.has_value(), "cast grad to f32");
        auto grad_mat = engine.to_matrix(*t_grad32_r);
        CHECK(grad_mat.has_value(), "to_matrix for grad");

        // w -= lr * grad
        for (std::size_t d = 0; d < D; ++d)
            w.set_value(d, 0, w.at(d, 0) - lr * grad_mat->at(d, 0));

        CHECK(loss < prev_loss, "loss decreases");
        prev_loss = loss;
    }

    std::printf(" done (final_loss=%.6f)\n", prev_loss);
}

// ── T4：f16 batched_matmul ─────────────────────────────────────────────
void test_f16_batched_matmul()
{
    std::printf("  [4] f16 batched_matmul...");
    nn::CpuEngine engine;

    const std::size_t batch = 2;
    const std::size_t M = 3, K = 4, N = 2;

    // A: (batch*M, K), B: (batch*K, N) — 注意 batched_matmul 的输入格式
    // 实际上 batched_matmul 的 A/B 是按 batch 块排列的
    // A: (2*3, 4) = (6, 4), B: (2*4, 2) = (8, 2) — 但 B 的 b_rows_per=4, 不匹配 K=4
    // 正确：A_b: (3, 4), B_b: (4, 2) → K=4 匹配

    nn::Matrix a32(batch * M, K);
    nn::Matrix b32(batch * K, N);  // 注意：batched_matmul 要求 b_rows_per = K
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : a32.span()) v = dist(rng);
    for (auto& v : b32.span()) v = dist(rng);

    // f32 batched_matmul 参考
    auto t_a32 = nn::Tensor::from_matrix(a32);
    auto t_b32 = nn::Tensor::from_matrix(b32);
    auto t_c32_ref_r = engine.batched_matmul(t_a32, t_b32, batch, false, false, 1.0f);
    CHECK(t_c32_ref_r.has_value(), "f32 batched_matmul");

    // f16 batched_matmul
    auto t_a16 = engine.cast(t_a32, nn::Precision::F16);
    CHECK(t_a16.has_value(), "cast A to f16");
    auto t_b16 = engine.cast(t_b32, nn::Precision::F16);
    CHECK(t_b16.has_value(), "cast B to f16");

    auto t_c16_r = engine.batched_matmul(*t_a16, *t_b16, batch, false, false, 1.0f,
                                          nn::Precision::F16);
    CHECK(t_c16_r.has_value(), "f16 batched_matmul");

    // cast 回 f32 比较
    auto t_c32_r = engine.cast(*t_c16_r, nn::Precision::F32);
    CHECK(t_c32_r.has_value(), "cast result to f32");

    const auto& c32 = t_c32_r->cpu_matrix();
    const auto& c32_ref = t_c32_ref_r->cpu_matrix();
    float max_rel_err = 0.0f;
    for (std::size_t i = 0; i < c32.size(); ++i)
    {
        float v32 = c32.span()[i];
        float vref = c32_ref.span()[i];
        float diff = std::fabs(v32 - vref);
        float ref_abs = std::fabs(vref);
        float rel_err = (ref_abs > 1e-6f) ? diff / ref_abs : diff;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }
    CHECK(max_rel_err < 0.05f, "f16 batched_matmul relative error within 5%");

    std::printf(" done (max_rel_err=%.6f)\n", max_rel_err);
}

// ── T5：错误路径 — BF16/F64 报错 ──────────────────────────────────────
void test_error_paths()
{
    std::printf("  [5] error paths (BF16/F64)...");

    // check_precision_supported
    auto r1 = nn::check_precision_supported(nn::Precision::BF16);
    CHECK(!r1.has_value(), "BF16 → error");
    auto r2 = nn::check_precision_supported(nn::Precision::F64);
    CHECK(!r2.has_value(), "F64 → error");
    auto r3 = nn::check_precision_supported(nn::Precision::F16);
    CHECK(r3.has_value(), "F16 → ok");
    auto r4 = nn::check_precision_supported(nn::Precision::F32);
    CHECK(r4.has_value(), "F32 → ok");

    std::printf(" done\n");
}

} // namespace

int main()
{
    std::printf("f16_compute_test (D6: f16 计算验证)\n");
    test_f16_matmul_vs_f32();
    test_cast_roundtrip();
    test_f16_linear_training();
    test_f16_batched_matmul();
    test_error_paths();
    if (g_failures == 0)
    {
        std::printf("f16_compute_test: ALL PASSED\n");
        return 0;
    }
    std::printf("f16_compute_test: %d FAILURES\n", g_failures);
    return 1;
}
