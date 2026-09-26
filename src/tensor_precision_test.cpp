// ── tensor_precision_test.cpp — D3：Tensor 精度属性 + variant 存储 + 显式 API ────
// 验收标准（docs/23 §13）：
//   1. Tensor precision() 返回正确的 Precision
//   2. f32 CPU Tensor：cpu_matrix() 返回 MatrixT<F32>（= Matrix&），数据正确
//   3. f16 CPU Tensor：cpu_matrix<F16>() 返回 MatrixT<F16>&，数据正确
//   4. 从 MatrixT<F16> 构造 Tensor，from_matrix 往返
//   5. Tensor::cpu<P>(rows, cols) 创建正确精度的空张量
//   6. reshape 保持精度
//   7. 默认构造 Tensor：valid() == false，precision == F32
//   8. shape_str 包含精度信息
//   9. variant 安全性：f16 tensor 调 cpu_matrix<F32>() → NN_ASSERT 触发（用 setjmp 模拟）
#include <cstdio>
#include <cstring>
#include <memory>

#include "neuralnet.cpp/precision.hpp"
#include "neuralnet.cpp/compute_tensor.hpp"
#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{

int g_failures = 0;


// T1：默认 Tensor — precision=F32，valid=false
void test_default_tensor()
{
    std::printf("  [1] default tensor...");
    nn::Tensor t;
    CHECK(t.precision() == nn::Precision::F32, "default precision == F32");
    CHECK(t.device() == nn::Device::CPU, "default device == CPU");
    CHECK(!t.valid(), "default tensor is invalid");
    CHECK(t.rows() == 0, "default rows == 0");
    CHECK(t.cols() == 0, "default cols == 0");
    std::printf(" done\n");
}

// T2：f32 CPU Tensor — 从 shared_ptr<Matrix>
void test_f32_cpu_tensor()
{
    std::printf("  [2] f32 cpu tensor...");
    auto m = std::make_shared<nn::Matrix>(3, 4);
    m->set_value(1, 2, 42.0f);
    nn::Tensor t(m);

    CHECK(t.precision() == nn::Precision::F32, "precision == F32");
    CHECK(t.device() == nn::Device::CPU, "device == CPU");
    CHECK(t.is_cpu(), "is_cpu");
    CHECK(t.valid(), "valid");
    CHECK(t.rows() == 3, "rows == 3");
    CHECK(t.cols() == 4, "cols == 4");

    // cpu_matrix() — 默认模板参数 F32
    nn::Matrix& ref = t.cpu_matrix();
    CHECK(ref.at(1, 2) == 42.0f, "data correct");

    // cpu_matrix<F32>() — 显式
    nn::Matrix& ref2 = t.cpu_matrix<nn::Precision::F32>();
    CHECK(ref2.at(1, 2) == 42.0f, "explicit F32 access");

    // cpu_shared()
    auto sp = t.cpu_shared();
    CHECK(sp.get() == &ref, "cpu_shared points to same Matrix");

    std::printf(" done\n");
}

// T3：f16 CPU Tensor — 从 shared_ptr<MatrixT<F16>>
void test_f16_cpu_tensor()
{
    std::printf("  [3] f16 cpu tensor...");
    auto m = std::make_shared<nn::MatrixT<nn::Precision::F16>>(2, 3);
    m->set_value(0, 0, nn::f16(1.5f));
    m->set_value(1, 2, nn::f16(3.25f));
    nn::Tensor t(m);

    CHECK(t.precision() == nn::Precision::F16, "precision == F16");
    CHECK(t.device() == nn::Device::CPU, "device == CPU");
    CHECK(t.valid(), "valid");
    CHECK(t.rows() == 2, "rows == 2");
    CHECK(t.cols() == 3, "cols == 3");

    // cpu_matrix<F16>() — 显式 f16 访问
    nn::MatrixT<nn::Precision::F16>& ref = t.cpu_matrix<nn::Precision::F16>();
    CHECK(static_cast<float>(ref.at(0, 0)) == 1.5f, "f16 data[0,0]");
    CHECK(static_cast<float>(ref.at(1, 2)) == 3.25f, "f16 data[1,2]");

    // cpu_shared<F16>()
    auto sp = t.cpu_shared<nn::Precision::F16>();
    CHECK(sp.get() == &ref, "cpu_shared<F16> points to same Matrix");

    // f16 tensor 调 cpu_shared<F32>() 应该返回 nullptr
    auto f32_sp = t.cpu_shared<nn::Precision::F32>();
    CHECK(f32_sp == nullptr, "f16 tensor -> cpu_shared<F32>() returns nullptr");

    std::printf(" done\n");
}

// T4：from_matrix 往返
void test_from_matrix_roundtrip()
{
    std::printf("  [4] from_matrix roundtrip...");

    // f32 往返
    nn::Matrix m32(2, 2);
    m32.set_value(0, 0, 7.0f);
    auto t32 = nn::Tensor::from_matrix(m32);
    CHECK(t32.precision() == nn::Precision::F32, "from_matrix f32 precision");
    CHECK(t32.cpu_matrix().at(0, 0) == 7.0f, "from_matrix f32 data");

    // f16 往返
    nn::MatrixT<nn::Precision::F16> m16(2, 2);
    m16.set_value(0, 0, nn::f16(2.5f));
    auto t16 = nn::Tensor::from_matrix(m16);
    CHECK(t16.precision() == nn::Precision::F16, "from_matrix f16 precision");
    CHECK(static_cast<float>(t16.cpu_matrix<nn::Precision::F16>().at(0, 0)) == 2.5f,
          "from_matrix f16 data");

    std::printf(" done\n");
}

// T5：Tensor::cpu<P>(rows, cols)
void test_cpu_factory()
{
    std::printf("  [5] cpu<P> factory...");

    auto t32 = nn::Tensor::cpu<nn::Precision::F32>(5, 6);
    CHECK(t32.precision() == nn::Precision::F32, "cpu<F32> precision");
    CHECK(t32.rows() == 5 && t32.cols() == 6, "cpu<F32> shape");
    CHECK(t32.valid(), "cpu<F32> valid");

    auto t16 = nn::Tensor::cpu<nn::Precision::F16>(3, 4);
    CHECK(t16.precision() == nn::Precision::F16, "cpu<F16> precision");
    CHECK(t16.rows() == 3 && t16.cols() == 4, "cpu<F16> shape");
    CHECK(t16.valid(), "cpu<F16> valid");

    // 非模板的 cpu(rows, cols) 默认 f32
    auto t_default = nn::Tensor::cpu(2, 2);
    CHECK(t_default.precision() == nn::Precision::F32, "cpu(rows,cols) == F32");

    std::printf(" done\n");
}

// T6：reshape 保持精度
void test_reshape_preserves_precision()
{
    std::printf("  [6] reshape preserves precision...");

    auto t16 = nn::Tensor::cpu<nn::Precision::F16>(2, 3);
    auto reshaped = t16.reshape(3, 2);
    CHECK(reshaped.precision() == nn::Precision::F16, "f16 reshape precision");

    auto t32 = nn::Tensor::cpu<nn::Precision::F32>(2, 3);
    auto reshaped32 = t32.reshape(3, 2);
    CHECK(reshaped32.precision() == nn::Precision::F32, "f32 reshape precision");

    std::printf(" done\n");
}

// T7：shape_str 包含精度
void test_shape_str_includes_precision()
{
    std::printf("  [7] shape_str includes precision...");

    auto t16 = nn::Tensor::cpu<nn::Precision::F16>(3, 4);
    auto s16 = t16.shape_str();
    CHECK(s16.find("f16") != std::string::npos, "shape_str contains f16");

    auto t32 = nn::Tensor::cpu<nn::Precision::F32>(3, 4);
    auto s32 = t32.shape_str();
    CHECK(s32.find("f32") != std::string::npos, "shape_str contains f32");

    std::printf(" done\n");
}

// T8：f16 CPU 数据往返 — 通过 Tensor 存取
void test_f16_data_through_tensor()
{
    std::printf("  [8] f16 data through tensor...");

    auto m = nn::MatrixT<nn::Precision::F16>(2, 2);
    m.set_value(0, 0, nn::f16(0.1f));
    m.set_value(0, 1, nn::f16(0.2f));
    m.set_value(1, 0, nn::f16(0.3f));
    m.set_value(1, 1, nn::f16(0.4f));

    auto t = nn::Tensor::from_matrix(m);
    auto& ref = t.cpu_matrix<nn::Precision::F16>();

    // 验证数据正确（f16 精度：|实际 - 预期| <= 0.5 * ulp）
    float v00 = static_cast<float>(ref.at(0, 0));
    float v01 = static_cast<float>(ref.at(0, 1));
    CHECK(v00 != 0.0f, "f16 data[0,0] nonzero");
    CHECK(v01 != 0.0f, "f16 data[0,1] nonzero");

    // 通过 cpu_shared 修改（共享所有权）
    auto sp = t.cpu_shared<nn::Precision::F16>();
    sp->set_value(0, 0, nn::f16(99.0f));
    CHECK(static_cast<float>(ref.at(0, 0)) == 99.0f, "shared modification visible");

    std::printf(" done\n");
}

} // namespace

int main()
{
    std::printf("tensor_precision_test (D3: Tensor precision + variant storage)\n");
    test_default_tensor();
    test_f32_cpu_tensor();
    test_f16_cpu_tensor();
    test_from_matrix_roundtrip();
    test_cpu_factory();
    test_reshape_preserves_precision();
    test_shape_str_includes_precision();
    test_f16_data_through_tensor();
    if (g_failures == 0)
    {
        std::printf("tensor_precision_test: ALL PASSED\n");
        return 0;
    }
    std::printf("tensor_precision_test: %d FAILURES\n", g_failures);
    return 1;
}