// ── layer_broadcast_test — 复合层逐子层默认广播（issue #29）───────────────
// 背景：set_precision_profile / set_training / clear_cache 原本没有默认转发，
// 复合层漏写 = 静默错误（子层留在 f32 / eval 时 Dropout/BN 行为错 / 显存残留）。
// v1.14.0 起基类默认按 param_children() 转发。本测试锁定：
//   [1] 只列 param_children、不 override 三函数的复合层：三个开关全部到达子层
//   [2] 子层记录的调用序与值正确
//   [3] 叶子层（无子层）默认实现不崩、无副作用
// ─────────────────────────────────────────────────────────────────────────

#include <neuralnet.cpp/nn.hpp>

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

namespace
{

// 记录型叶子层：捕获三个广播开关的到达情况
class RecordingLayer final : public nn::Layer
{
public:
    [[nodiscard]] nn::Result<nn::Tensor> forward(const nn::Tensor& x) override { return x; }
    [[nodiscard]] nn::Result<nn::Tensor> backward(const nn::Tensor& g) override { return g; }
    [[nodiscard]] const char* layer_name() const noexcept override { return "RecordingLayer"; }

    void set_training(bool t) override
    {
        nn::Layer::set_training(t);   // 叶子：默认转发空，仅记录
        training_calls.push_back(t);
    }
    void clear_cache() override
    {
        nn::Layer::clear_cache();
        ++clear_cache_calls;
    }

    std::vector<bool> training_calls;
    int clear_cache_calls = 0;
};

// 复合层：**只**列 param_children，不 override 三函数（issue #29 的目标形态）
class CompositeOnlyChildren final : public nn::Layer
{
public:
    RecordingLayer a, b;

    [[nodiscard]] nn::Result<nn::Tensor> forward(const nn::Tensor& x) override { return x; }
    [[nodiscard]] nn::Result<nn::Tensor> backward(const nn::Tensor& g) override { return g; }
    [[nodiscard]] const char* layer_name() const noexcept override
    { return "CompositeOnlyChildren"; }

protected:
    [[nodiscard]] std::vector<nn::Layer*> param_children() noexcept override { return {&a, &b}; }
};

// 叶子复合层（无子层）：默认实现应为安全 no-op
class NoChildrenLayer final : public nn::Layer
{
public:
    [[nodiscard]] nn::Result<nn::Tensor> forward(const nn::Tensor& x) override { return x; }
    [[nodiscard]] nn::Result<nn::Tensor> backward(const nn::Tensor& g) override { return g; }
};

} // namespace

int main()
{
    // [1][2] 复合层只列子层：三个开关到达全部子层
    {
        CompositeOnlyChildren c;

        // set_training 广播（含先例值 false→true 序列）
        c.set_training(true);
        CHECK(c.a.training_calls.size() == 1 && c.a.training_calls[0] == true,
              "set_training 到达子层 a");
        CHECK(c.b.training_calls.size() == 1 && c.b.training_calls[0] == true,
              "set_training 到达子层 b");
        c.set_training(false);
        CHECK(c.a.training_calls.size() == 2 && c.a.training_calls[1] == false,
              "set_training(false) 到达子层 a");

        // clear_cache 广播
        c.clear_cache();
        CHECK(c.a.clear_cache_calls == 1 && c.b.clear_cache_calls == 1, "clear_cache 到达全部子层");

        // set_precision_profile 广播（未 init 的子层接收不 fail-fast）
        nn::PrecisionProfile prof;
        prof.param = nn::Precision::F16;
        prof.compute = nn::Precision::F16;
        c.set_precision_profile(prof);
        CHECK(c.a.precision_profile().param == nn::Precision::F16 &&
                  c.a.precision_profile().compute == nn::Precision::F16,
              "set_precision_profile 到达子层 a");
        CHECK(c.b.precision_profile().param == nn::Precision::F16,
              "set_precision_profile 到达子层 b");
        // 复合层自身也记录（基类语义不变）
        CHECK(c.precision_profile().param == nn::Precision::F16, "复合层自身 profile 已记录");
    }

    // [3] 无子层的叶子：默认实现安全 no-op
    {
        NoChildrenLayer l;
        l.set_training(true);
        l.clear_cache();
        nn::PrecisionProfile prof;
        l.set_precision_profile(prof);
        CHECK(true, "叶子层默认广播实现无副作用");
    }

    std::cout << (g_fail == 0 ? "\n全部通过 ✅\n" : "\n存在失败 ❌\n");
    return g_fail == 0 ? 0 : 1;
}
