// ── batch_source_test — RFC 32 §4：BatchSource Pull 契约 + Indexed 适配器 ──
//
// 覆盖：
//   [1] 步网格 = ceil(n/batch)，末批不满，epoch_done 标记
//   [2] 每 epoch shuffle：同 (seed, epoch) 复现同序；不同 epoch 不同序
//   [3] rewind 定位：start_step 越界报错；重排可复现
//   [4] next 越界报错；构造期参数校验（batch_size=0 / sample=0 / 空回调）
//   [5] 归一化单位由域回调给出；载荷 std::any 指针形式读取
//   [6] 无固定 epoch 的 Pull 源（steps_per_epoch = nullopt）骨架退化语义
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <neuralnet.cpp/batch_source.hpp>
#include <neuralnet.cpp/core_errors.hpp>

static int g_fail = 0;
#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::cout << "    [FAIL] " << (msg) << "  (" << #cond << ")\n";       \
            ++g_fail;                                                             \
        }                                                                         \
    } while (0)

namespace
{

// ── [1] 步网格 / 末批 / epoch_done ─────────────────────────────────────
void test_step_grid()
{
    std::cout << "\n── [1] 步网格 = ceil(n/batch)，末批不满 ──\n";
    auto src_r = nn::IndexedBatchSource::create(
        /*sample_count=*/10, /*batch_size=*/4, /*seed=*/7,
        [](std::span<const std::size_t> ix) -> nn::Result<std::pair<std::any, std::size_t>>
        {
            // 载荷 = 下标和（便于断言）；单位 = 下标个数
            std::size_t sum = 0;
            for (std::size_t i : ix) sum += i;
            return std::make_pair(std::any(sum), ix.size());
        });
    CHECK(src_r.has_value(), "create 合法参数成功");
    if (!src_r) return;
    nn::IndexedBatchSource& src = *src_r;

    CHECK(src.steps_per_epoch() == 3, "10/4 → 3 步");

    auto b0 = src.next(0);
    CHECK(b0 && b0->samples == 4 && b0->epoch_done == false, "步 0：4 样本，epoch 未完");
    auto b1 = src.next(1);
    CHECK(b1 && b1->samples == 4 && b1->epoch_done == false, "步 1：4 样本，epoch 未完");
    auto b2 = src.next(2);
    CHECK(b2 && b2->samples == 2 && b2->epoch_done == true, "步 2：末批 2 样本，epoch 完");
    auto bad = src.next(3);
    CHECK(!bad.has_value(), "step_hint=3 越界 → 报错");

    // 单位 = 下标个数
    CHECK(b0 && b0->units == 4 && b2 && b2->units == 2, "归一化单位随末批收缩");
}

// ── [2] 每 epoch shuffle 的确定性 ──────────────────────────────────────
void test_shuffle_determinism()
{
    std::cout << "\n── [2] 每 epoch shuffle：同 seed/epoch 复现，跨 epoch 变化 ──\n";
    auto assemble = [](std::span<const std::size_t> ix)
        -> nn::Result<std::pair<std::any, std::size_t>>
    { return std::make_pair(std::any(), ix.size()); };

    auto a_r = nn::IndexedBatchSource::create(32, 8, 42, assemble);
    auto b_r = nn::IndexedBatchSource::create(32, 8, 42, assemble);
    auto c_r = nn::IndexedBatchSource::create(32, 8, 43, assemble);   // 不同 seed
    CHECK(a_r && b_r && c_r, "三个源创建成功");
    if (!a_r || !b_r || !c_r) return;
    nn::IndexedBatchSource& a = *a_r;
    nn::IndexedBatchSource& b = *b_r;
    nn::IndexedBatchSource& c = *c_r;

    (void)a.rewind(0, 0);
    (void)b.rewind(0, 0);
    (void)c.rewind(0, 0);
    CHECK(std::vector<std::size_t>(a.order().begin(), a.order().end()) ==
              std::vector<std::size_t>(b.order().begin(), b.order().end()),
          "同 (seed, epoch) → 同顺序（跨续训可复现）");
    CHECK(std::vector<std::size_t>(a.order().begin(), a.order().end()) !=
              std::vector<std::size_t>(c.order().begin(), c.order().end()),
          "不同 seed → 不同顺序");

    // 同 seed 换 epoch → 顺序变化（每 epoch 重排）
    std::vector<std::size_t> e0(a.order().begin(), a.order().end());
    (void)a.rewind(1, 0);
    std::vector<std::size_t> e1(a.order().begin(), a.order().end());
    CHECK(e0 != e1, "同 seed 不同 epoch → 重排");

    // 回到 epoch 0 → 顺序与首次一致（确定性）
    (void)a.rewind(0, 0);
    CHECK(std::vector<std::size_t>(a.order().begin(), a.order().end()) == e0,
          "rewind 回 epoch 0 → 顺序复现");
}

// ── [3] rewind 定位与参数校验 ──────────────────────────────────────────
void test_rewind_validation()
{
    std::cout << "\n── [3] rewind：start_step 越界报错 ──\n";
    auto src_r = nn::IndexedBatchSource::create(
        10, 4, 1,
        [](std::span<const std::size_t> ix)
            -> nn::Result<std::pair<std::any, std::size_t>>
        { return std::make_pair(std::any(), ix.size()); });
    CHECK(src_r.has_value(), "create 成功");
    if (!src_r) return;
    nn::IndexedBatchSource& src = *src_r;
    CHECK(src.rewind(0, 3).has_value(), "start_step=3 == steps → 合法（本 epoch 末）");
    CHECK(!src.rewind(0, 4).has_value(), "start_step=4 > steps → 报错");
}

// ── [4] 构造期参数校验 ─────────────────────────────────────────────────
void test_ctor_validation()
{
    std::cout << "\n── [4] 构造期校验：batch_size=0 / sample=0 / 空回调 ──\n";
    auto assemble = [](std::span<const std::size_t> ix)
        -> nn::Result<std::pair<std::any, std::size_t>>
    { return std::make_pair(std::any(), ix.size()); };

    CHECK(!nn::IndexedBatchSource::create(10, 0, 1, assemble).has_value(),
          "batch_size=0 → 报错");
    CHECK(!nn::IndexedBatchSource::create(0, 4, 1, assemble).has_value(),
          "sample_count=0 → 报错");
    CHECK(!nn::IndexedBatchSource::create(10, 4, 1, nullptr).has_value(),
          "空装配回调 → 报错");

    auto ok = nn::IndexedBatchSource::create(5, 2, 0, assemble);
    CHECK(ok && ok->steps_per_epoch() == 3, "5/2 → 3 步");
}

// ── [5] 载荷 std::any 指针形式读取 ─────────────────────────────────────
void test_payload()
{
    std::cout << "\n── [5] 载荷：any 指针形式（-fno-exceptions 安全）──\n";
    auto src_r = nn::IndexedBatchSource::create(
        6, 3, 9,
        [](std::span<const std::size_t> ix)
            -> nn::Result<std::pair<std::any, std::size_t>>
        {
            std::string tag = "batch:" + std::to_string(ix.size());
            return std::make_pair(std::any(std::move(tag)), ix.size() * 10);
        });
    CHECK(src_r.has_value(), "create 成功");
    if (!src_r) return;
    auto b = src_r->next(0);
    CHECK(b.has_value(), "next 成功");
    if (b)
    {
        const auto* tag = std::any_cast<std::string>(&b->payload);
        CHECK(tag != nullptr && *tag == "batch:3", "指针形式读出域载荷");
        CHECK(std::any_cast<std::size_t>(&b->payload) == nullptr,
              "类型不符 → nullptr（不抛）");
        CHECK(b->units == 30, "域回调给出归一化单位");
    }
}

// ── [6] 无固定 epoch 的 Pull 源（骨架退化为步进驱动）───────────────────
class StepOnlySource : public nn::BatchSource
{
public:
    [[nodiscard]] nn::Result<nn::BatchAt> next(std::size_t hint) override
    {
        nn::BatchAt b;
        b.samples = 4;
        b.units = 4;
        b.epoch_done = false;      // 无 epoch 概念：永远不"完成一轮"
        b.payload = hint;
        return b;
    }
    // steps_per_epoch 不覆写 ⇒ nullopt（骨架退化为步进驱动）
};

void test_pull_source()
{
    std::cout << "\n── [6] Pull 源：steps_per_epoch = nullopt ──\n";
    StepOnlySource src;
    CHECK(!src.steps_per_epoch().has_value(), "无固定 epoch → nullopt");
    auto b = src.next(0);
    CHECK(b && !b->epoch_done && b->samples == 4, "步进驱动：无 epoch 完成语义");
    CHECK(b && std::any_cast<std::size_t>(&b->payload) && *std::any_cast<std::size_t>(&b->payload) == 0,
          "载荷可透传任意域数据");
    CHECK(src.rewind(0, 0).has_value(), "默认 rewind = 空实现（合法）");
}

} // namespace

int main()
{
    test_step_grid();
    test_shuffle_determinism();
    test_rewind_validation();
    test_ctor_validation();
    test_payload();
    test_pull_source();

    if (g_fail == 0)
    {
        std::cout << "\nbatch_source_test: ALL PASS\n";
        return 0;
    }
    std::cout << "\nbatch_source_test: " << g_fail << " FAILURE(S)\n";
    return 1;
}
