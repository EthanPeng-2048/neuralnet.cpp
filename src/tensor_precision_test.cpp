// ── tensor_precision_test.cpp — D3：Tensor 精度属性 + 引擎 I/O 往返 ──────────
// M1 访问收口（docs/development/17 §4.1）后，Tensor 存储访问器与静态直构
// 工厂已私有：本测试全部经 ComputeEngine 创建/读写（upload/download/create_tensor/
// reshape），验收口径同步修订（原 docs/development/05 §13 中依赖 cpu_matrix/
// cpu_shared 直读的条目改为"经引擎往返"等价验证）：
//   1. 默认构造 Tensor：precision==F32、valid()==false、形状 0（公共面不变）
//   2. f32 张量：经引擎上传后 precision/device/形状/valid 正确，下载数据正确
//   3. f16 张量：precision==F16，f16 宿主上传 + 下载往返精确升位
//   4. from_matrix 往返（engine.from_matrix + to_matrix）f32/f16 双精度
//   5. engine.create_tensor(rows, cols, P) 创建正确精度的空张量（零填充）
//   6. engine.reshape 保持精度与数据（D10：reshape 移入引擎；元素数不匹配报错）
//   7. shape_str 包含精度信息
//   8. f16 存储数据经下载升位后与宿主参考逐元素一致（f16→f32 精确无损）
#include <cstdio>
#include <cstring>
#include <memory>

#include "neuralnet.cpp/precision.hpp"
#include "neuralnet.cpp/compute_tensor.hpp"
#include "neuralnet.cpp/compute_cpu_engine.hpp"
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

// T2：f32 CPU Tensor — 经引擎上传 + 下载往返
void test_f32_cpu_tensor(nn::ComputeEngine& eng)
{
    std::printf("  [2] f32 cpu tensor...");
    nn::Matrix m(3, 4);
    m.set_value(1, 2, 42.0f);
    nn::Tensor t = upload(eng, m);

    CHECK(t.precision() == nn::Precision::F32, "precision == F32");
    CHECK(t.device() == nn::Device::CPU, "device == CPU");
    CHECK(t.is_cpu(), "is_cpu");
    CHECK(t.valid(), "valid");
    CHECK(t.rows() == 3, "rows == 3");
    CHECK(t.cols() == 4, "cols == 4");

    const nn::Matrix back = download(eng, t);
    CHECK(back.at(1, 2) == 42.0f, "data correct");

    std::printf(" done\n");
}

// T3：f16 CPU Tensor — f16 宿主上传 + 下载升位往返
void test_f16_cpu_tensor(nn::ComputeEngine& eng)
{
    std::printf("  [3] f16 cpu tensor...");
    nn::MatrixT<nn::Precision::F16> m(2, 3);
    m.set_value(0, 0, nn::f16(1.5f));
    m.set_value(1, 2, nn::f16(3.25f));
    nn::Tensor t = upload(eng, m);

    CHECK(t.precision() == nn::Precision::F16, "precision == F16");
    CHECK(t.device() == nn::Device::CPU, "device == CPU");
    CHECK(t.valid(), "valid");
    CHECK(t.rows() == 2, "rows == 2");
    CHECK(t.cols() == 3, "cols == 3");

    // f16 → f32 下载精确升位（1.5 / 3.25 在 f16 中可精确表示）
    const nn::Matrix back = download(eng, t);
    CHECK(back.at(0, 0) == 1.5f, "f16 data[0,0] roundtrip");
    CHECK(back.at(1, 2) == 3.25f, "f16 data[1,2] roundtrip");

    // 经 F16 槽位下载（降 cast 再升位）语义：仍是精确往返
    auto back16_r = eng.to_matrix(t, nn::Precision::F16);
    CHECK(static_cast<bool>(back16_r), "to_matrix P=F16 succeeds");
    if (back16_r)
        CHECK(back16_r->at(0, 0) == 1.5f, "f16 data[0,0] via P=F16 download");

    std::printf(" done\n");
}

// T4：from_matrix 往返（engine.from_matrix + download）
void test_from_matrix_roundtrip(nn::ComputeEngine& eng)
{
    std::printf("  [4] from_matrix roundtrip...");

    // f32 往返
    nn::Matrix m32(2, 2);
    m32.set_value(0, 0, 7.0f);
    auto t32 = upload(eng, m32);
    CHECK(t32.precision() == nn::Precision::F32, "from_matrix f32 precision");
    CHECK(download(eng, t32).at(0, 0) == 7.0f, "from_matrix f32 data");

    // f16 往返
    nn::MatrixT<nn::Precision::F16> m16(2, 2);
    m16.set_value(0, 0, nn::f16(2.5f));
    auto t16 = upload(eng, m16);
    CHECK(t16.precision() == nn::Precision::F16, "from_matrix f16 precision");
    CHECK(download(eng, t16).at(0, 0) == 2.5f, "from_matrix f16 data");

    std::printf(" done\n");
}

// T5：engine.create_tensor(rows, cols, P) — 原 Tensor::cpu<P> 工厂的引擎化
void test_create_tensor_factory(nn::ComputeEngine& eng)
{
    std::printf("  [5] create_tensor factory...");

    auto t32 = eng.create_tensor(5, 6, nn::Precision::F32);
    CHECK(t32.precision() == nn::Precision::F32, "create F32 precision");
    CHECK(t32.rows() == 5 && t32.cols() == 6, "create F32 shape");
    CHECK(t32.valid(), "create F32 valid");

    auto t16 = eng.create_tensor(3, 4, nn::Precision::F16);
    CHECK(t16.precision() == nn::Precision::F16, "create F16 precision");
    CHECK(t16.rows() == 3 && t16.cols() == 4, "create F16 shape");
    CHECK(t16.valid(), "create F16 valid");

    // 非模板默认 P == F32
    auto t_default = eng.create_tensor(2, 2);
    CHECK(t_default.precision() == nn::Precision::F32, "create(rows,cols) == F32");

    // 零填充语义（原 Tensor::cpu 的 Matrix(rows, cols) 零初始化）
    const nn::Matrix z = download(eng, t32);
    CHECK(z.at(0, 0) == 0.0f && z.at(4, 5) == 0.0f, "create zero-filled");

    std::printf(" done\n");
}

// T6：engine.reshape 保持精度（D10：reshape 移入引擎，错误走 Result）
void test_reshape_preserves_precision(nn::ComputeEngine& eng)
{
    std::printf("  [6] reshape preserves precision...");

    auto t16 = eng.create_tensor(2, 3, nn::Precision::F16);
    auto reshaped_r = eng.reshape(t16, 3, 2);
    CHECK(static_cast<bool>(reshaped_r), "f16 reshape ok");
    if (reshaped_r)
    {
        CHECK(reshaped_r->precision() == nn::Precision::F16, "f16 reshape precision");
        CHECK(reshaped_r->rows() == 3 && reshaped_r->cols() == 2, "f16 reshape shape");
    }

    auto t32 = eng.create_tensor(2, 3, nn::Precision::F32);
    auto reshaped32_r = eng.reshape(t32, 3, 2);
    CHECK(static_cast<bool>(reshaped32_r), "f32 reshape ok");
    if (reshaped32_r)
        CHECK(reshaped32_r->precision() == nn::Precision::F32, "f32 reshape precision");

    // 元素数不匹配 → Result 错误（原 NN_ASSERT 升级为可检查错误）
    auto bad_r = eng.reshape(t32, 5, 7);
    CHECK(!bad_r, "reshape element-count mismatch rejected");

    std::printf(" done\n");
}

// T7：shape_str 包含精度
void test_shape_str_includes_precision(nn::ComputeEngine& eng)
{
    std::printf("  [7] shape_str includes precision...");

    auto t16 = eng.create_tensor(3, 4, nn::Precision::F16);
    auto s16 = t16.shape_str();
    CHECK(s16.find("f16") != std::string::npos, "shape_str contains f16");

    auto t32 = eng.create_tensor(3, 4, nn::Precision::F32);
    auto s32 = t32.shape_str();
    CHECK(s32.find("f32") != std::string::npos, "shape_str contains f32");

    std::printf(" done\n");
}

// T8：f16 数据往返 — 宿主写入 → 上传 → 下载，与宿主参考逐元素一致
// （原"cpu_shared 直改存储"验证随存储私有化取消；写路径改为宿主侧 + 重上传）
void test_f16_data_roundtrip(nn::ComputeEngine& eng)
{
    std::printf("  [8] f16 data roundtrip...");

    nn::MatrixT<nn::Precision::F16> m(2, 2);
    m.set_value(0, 0, nn::f16(0.1f));
    m.set_value(0, 1, nn::f16(0.2f));
    m.set_value(1, 0, nn::f16(0.3f));
    m.set_value(1, 1, nn::f16(0.4f));

    auto t = upload(eng, m);
    const nn::Matrix back = download(eng, t);

    // 验证数据正确（f16 精度：非零 + 与宿主 f16 源升位后逐位一致）
    CHECK(back.at(0, 0) != 0.0f, "f16 data[0,0] nonzero");
    CHECK(back.at(0, 1) != 0.0f, "f16 data[0,1] nonzero");
    bool exact = true;
    for (std::size_t r = 0; r < 2 && exact; ++r)
        for (std::size_t c = 0; c < 2; ++c)
            if (back.at(r, c) != static_cast<float>(m.at(r, c)))
                exact = false;
    CHECK(exact, "f16 roundtrip bit-exact via f32 download");

    std::printf(" done\n");
}

} // namespace

int main()
{
    std::printf("tensor_precision_test (D3: Tensor precision + engine I/O roundtrip, M1)\n");
    nn::CpuEngine eng;
    test_default_tensor();
    test_f32_cpu_tensor(eng);
    test_f16_cpu_tensor(eng);
    test_from_matrix_roundtrip(eng);
    test_create_tensor_factory(eng);
    test_reshape_preserves_precision(eng);
    test_shape_str_includes_precision(eng);
    test_f16_data_roundtrip(eng);
    if (g_failures == 0)
    {
        std::printf("tensor_precision_test: ALL PASSED\n");
        return 0;
    }
    std::printf("tensor_precision_test: %d FAILURES\n", g_failures);
    return 1;
}
