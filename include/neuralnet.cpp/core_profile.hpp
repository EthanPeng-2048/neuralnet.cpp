// ─────────────────────────────────────────────────────────────────────────────
//  core_profile.hpp — 训练 step 剖析器（host 侧聚合计时 + Chrome trace 输出）
//
//  目的：回答"一个训练 step 里各阶段/各原语的开销构成、GPU 利用率是多少"，
//  让性能优化方向由数据决定而不是猜。剖析器本身**不进入任何计算路径**：
//
//  · 默认关闭（零开销：构造体只读一个 bool，不调时钟）；
//  · `NN_PROFILE=1` 开启聚合计时，进程结束前由入口程序调 `nn::prof::dump()`
//    打表（stage 表 = 各阶段墙钟构成、op 表 = 各原语净归属、GPU 表 = 设备侧
//    kernel 时间与 busy%）；
//  · `NN_PROFILE_TRACE=<path>` 额外记录逐事件时间轴，`dump()` 时写 Chrome
//    trace JSON（chrome://tracing / Perfetto 直接打开）。
//
//  计时语义（帧栈实现，见 Frame 注释）：
//  · 每个计时点 = 一"帧"，停止时算 inclusive（自身墙钟）与 exclusive
//    （inclusive − 直接子帧 inclusive 之和，即净归属时间）；
//  · stage 表打 inclusive（阶段占 step 墙钟多少），op 表打 exclusive
//    （原语净归属，跨线程求和 ≤ 墙钟，不重复计数）；
//  · op 的归属 stage = 停止时所在线程的**最内层** stage 帧（stage::op 两级聚合）。
//
//  线程模型：计时帧栈是 thread_local（parallel_for 工作线程各自独立），聚合表
//  由互斥锁保护（算子粒度调用，锁开销 ~30ns，相对算子耗时可忽略）。
//
//  设备侧（Vulkan timestamp query）不在本文件实现：后端把逐 kernel 耗时经
//  `device_add()` 回填，报告统一由 `dump()` 输出。
//
//  命名约束：所有 name/stage 参数必须是**字符串字面量或静态存储**——聚合表
//  以 string_view 引用其地址作为 key，不拷贝（传临时 std::string 会悬垂）。
//
//  确定性：纯观测，不改任何计算结果/字节锚（铁律 #8）；关闭时编译产物不含
//  任何时钟调用路径。
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nn::prof
{

// ── 环境变量（沿用 NN_MEM_STATS / NN_PREC_TRACE 的门控惯例）────────────────
//   NN_PROFILE=1              开启 host + 设备侧聚合计时
//   NN_PROFILE_TRACE=<path>   额外写 Chrome trace JSON 到 <path>（隐含开启）
//   NN_PROFILE_EVERY=<n>      每 n 次 dump() 打一次阶段表（默认 0 = 仅显式 dump）

namespace detail
{

inline bool env_flag(const char* name)
{
#if defined(_MSC_VER)
    char* buf = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&buf, &len, name) == 0 && buf != nullptr)
    {
        const bool on = len > 0 && buf[0] != '\0' && buf[0] != '0';
        std::free(buf);
        return on;
    }
    return false;
#else
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
#endif
}

inline std::string env_string(const char* name)
{
#if defined(_MSC_VER)
    char* buf = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&buf, &len, name) == 0 && buf != nullptr)
    {
        std::string s = buf != nullptr ? buf : "";
        std::free(buf);
        return s;
    }
    return {};
#else
    const char* v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string();
#endif
}

}   // namespace detail

// ── 全局开关（静态初始化期读环境变量，进程内恒定）─────────────────────────
inline const bool g_enabled = detail::env_flag("NN_PROFILE") ||
                              !detail::env_string("NN_PROFILE_TRACE").empty();
inline const std::string g_trace_path = detail::env_string("NN_PROFILE_TRACE");

[[nodiscard]] inline bool enabled() noexcept { return g_enabled; }

// ── 聚合桶 ──────────────────────────────────────────────────────────────────
struct Bucket
{
    std::uint64_t calls = 0;
    std::uint64_t incl_ns = 0;   // inclusive 总耗时（阶段墙钟）
    std::uint64_t excl_ns = 0;   // exclusive 总耗时（净归属，跨行可加总）
    std::uint64_t max_ns = 0;    // 单次 inclusive 最大值
};

// 设备侧桶（GPU kernel：无嵌套，inclusive == exclusive）
struct DeviceBucket
{
    std::uint64_t calls = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t max_ns = 0;
};

namespace detail
{

// key = 字符串字面量地址（见文件头"命名约束"）
using StageMap = std::unordered_map<std::string_view,
                    std::unordered_map<std::string_view, Bucket>>;

// Chrome trace 逐事件（仅 NN_PROFILE_TRACE 时记录）
struct Event
{
    const char* name;
    const char* cat;
    std::uint64_t ts_us;
    std::uint64_t dur_us;
    std::uint32_t tid;
};

struct State
{
    std::mutex mu;
    StageMap stages;                                        // stage → op → 桶（op=="" = stage 本身）
    std::unordered_map<std::string_view, DeviceBucket> device;
    std::vector<Event> trace;
};

inline State& state()
{
    static State s;
    return s;
}

// ── 计时帧栈（thread_local）────────────────────────────────────────────────
//   push 时记 start；stop 时 dur = now − start，把 dur 计入父帧 child_ns，
//   自身 exclusive = dur − child_ns。嵌套调用（如 dsl::compute 内部过引擎
//   入口）天然不重复计数。
struct Frame
{
    const char* stage;         // 静态字符串；op 帧沿用最内层 stage
    const char* op;            // nullptr = 纯 stage 帧
    std::uint64_t start_ns;
    std::uint64_t child_ns;
};

inline std::vector<Frame>& stack()
{
    thread_local std::vector<Frame> st;
    return st;
}

[[nodiscard]] inline std::uint64_t now_ns() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline void accumulate(const char* stage, const char* op,
                       std::uint64_t incl_ns, std::uint64_t excl_ns)
{
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    Bucket& b = s.stages[stage][op != nullptr ? std::string_view(op)
                                              : std::string_view()];
    ++b.calls;
    b.incl_ns += incl_ns;
    b.excl_ns += excl_ns;
    if (incl_ns > b.max_ns) b.max_ns = incl_ns;
}

// ── Chrome trace ────────────────────────────────────────────────────────────
inline std::uint32_t current_tid() noexcept
{
    thread_local const std::uint32_t hashed = static_cast<std::uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()));
    return hashed;
}

inline void trace_event(const char* name, const char* cat,
                        std::uint64_t ts_us, std::uint64_t dur_us)
{
    if (g_trace_path.empty()) return;
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.trace.push_back(Event{name, cat, ts_us, dur_us, current_tid()});
}

}   // namespace detail

// ── 动态名驻留（层号等非字面量名）──────────────────────────────────────────
//   返回 "prefix#idx" 的稳定 const char*（首次生成后驻留，之后零分配）；
//   供 Scope/Op 使用动态名的场景（如 Model 逐层归因）。有锁，仅每层每步一次。
[[nodiscard]] inline const char* interned(const char* prefix, std::size_t idx)
{
    static std::mutex mu;
    static std::deque<std::string> pool;                       // 地址稳定
    static std::unordered_map<std::string, const char*> seen;
    std::string key = std::string(prefix) + "#" + std::to_string(idx);
    std::lock_guard<std::mutex> lock(mu);
    auto it = seen.find(key);
    if (it != seen.end()) return it->second;
    pool.push_back(std::move(key));
    const char* p = pool.back().c_str();
    seen.emplace(pool.back(), p);
    return p;
}

// ── 设备侧归因：录制期 stage 取值 + "label@stage" 驻留 ─────────────────────
//   timestamp 在 fence 之后 resolve，彼时 scope 栈已出栈——dispatch 录制期
//   （vkCmdDispatch 前）必须先把当前 stage 抓下来并驻留，device_add 才能把
//   设备 kernel 时间归到 step 阶段（fwd.blk#0 / optimizer / …），与 host 侧
//   阶段表同维度对账。仅剖析开启时被调用（调用点在 ts_enabled_ 分支内）。
[[nodiscard]] inline const char* current_stage() noexcept
{
    if (!g_enabled) return "";
    const auto& st = detail::stack();
    return (st.empty() || st.back().stage == nullptr) ? "" : st.back().stage;
}

[[nodiscard]] inline const char* interned_pair(const char* a, const char* b)
{
    static std::mutex mu;
    static std::deque<std::string> pool;                       // 地址稳定
    static std::unordered_map<std::string, const char*> seen;
    std::string key = std::string(a) + "@" + b;
    std::lock_guard<std::mutex> lock(mu);
    auto it = seen.find(key);
    if (it != seen.end()) return it->second;
    pool.push_back(std::move(key));
    const char* p = pool.back().c_str();
    seen.emplace(pool.back(), p);
    return p;
}

// ── RAII 计时器 ─────────────────────────────────────────────────────────────

// Scope：阶段/层级计时（forward / backward / L3 …），name 必须静态存储。
// 析构自动停止；跨语句的大区域可用同名变量 + 显式 stop()（幂等）划分
// 连续阶段（训练 step 里各阶段共享局部变量、不能用花括号块隔开）。
class Scope
{
public:
    explicit Scope(const char* name) noexcept
    {
        if (!g_enabled) return;
        start_ = detail::now_ns();
        name_ = name;
        detail::stack().push_back(detail::Frame{name, nullptr, start_, 0});
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    ~Scope() { stop(); }

    void stop() noexcept
    {
        if (!g_enabled || name_ == nullptr) return;
        name_ = nullptr;
        auto& st = detail::stack();
        const detail::Frame f = st.back();
        st.pop_back();
        const std::uint64_t end = detail::now_ns();
        const std::uint64_t dur = end - f.start_ns;
        const std::uint64_t excl = dur >= f.child_ns ? dur - f.child_ns : 0;
        if (!st.empty()) st.back().child_ns += dur;
        detail::accumulate(f.stage, nullptr, dur, excl);
        detail::trace_event(f.stage, "stage", f.start_ns / 1000, dur / 1000);
    }

private:
    const char* name_ = nullptr;
    std::uint64_t start_ = 0;
};

// Phase：可重启的连续阶段句柄——start() 自动收掉上一段，供"各阶段共享局部
// 变量、不能用花括号块隔开"的大区域（训练 step 的 data/forward/backward/…）
// 划分连续阶段。析构自动收口。
class Phase
{
public:
    Phase() noexcept = default;
    Phase(const Phase&) = delete;
    Phase& operator=(const Phase&) = delete;
    ~Phase() { stop(); }

    void start(const char* name)
    {
        stop();
        s_.emplace(name);
    }
    void stop() { s_.reset(); }

private:
    std::optional<Scope> s_;
};

// Op：引擎原语计时，归属到最内层 stage（stage::op 聚合），name 必须静态存储。
class Op
{
public:
    explicit Op(const char* name) noexcept
    {
        if (!g_enabled) return;
        auto& st = detail::stack();
        stage_ = st.empty() ? "" : st.back().stage;
        op_ = name;
        start_ = detail::now_ns();
        st.push_back(detail::Frame{stage_, name, start_, 0});
    }
    Op(const Op&) = delete;
    Op& operator=(const Op&) = delete;
    ~Op()
    {
        if (!g_enabled || op_ == nullptr) return;
        auto& st = detail::stack();
        const detail::Frame f = st.back();
        st.pop_back();
        const std::uint64_t end = detail::now_ns();
        const std::uint64_t dur = end - f.start_ns;
        const std::uint64_t excl = dur >= f.child_ns ? dur - f.child_ns : 0;
        if (!st.empty()) st.back().child_ns += dur;
        detail::accumulate(f.stage, f.op, dur, excl);
        detail::trace_event(f.op, f.stage, f.start_ns / 1000, dur / 1000);
    }

private:
    const char* stage_ = "";
    const char* op_ = nullptr;
    std::uint64_t start_ = 0;
};

// ── 设备侧回填（Vulkan timestamp query 解析后调用）─────────────────────────
inline void device_add(const char* name, std::uint64_t ns)
{
    if (!g_enabled) return;
    detail::State& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mu);
    DeviceBucket& b = s.device[name];
    ++b.calls;
    b.total_ns += ns;
    if (ns > b.max_ns) b.max_ns = ns;
}

// ── 宏（__LINE__ 拼接出唯一变量名）──────────────────────────────────────────
#define NN_PROF_CONCAT_(a, b) a##b
#define NN_PROF_CONCAT(a, b) NN_PROF_CONCAT_(a, b)
#define NN_PROF_SCOPE(name) \
    ::nn::prof::Scope NN_PROF_CONCAT(nn_prof_scope_, __LINE__)(name)
#define NN_PROF_OP(name) \
    ::nn::prof::Op NN_PROF_CONCAT(nn_prof_op_, __LINE__)(name)

// ── 报告 ────────────────────────────────────────────────────────────────────
//   out        : 打表文件（默认 stderr）
//   wall_ns    : 本次剖析覆盖的墙钟（调用方传入，如 step 循环总时长）；>0 时
//                报告尾部给出 GPU busy% / host 构成占比
//   reset      : 打完清零（周期性打表用）；默认不清
void dump(std::FILE* out, std::uint64_t wall_ns = 0, bool reset = false);

//   把逐事件时间轴写成 Chrome trace JSON（dump() 在设置了 NN_PROFILE_TRACE
//   时也会自动调用它）。
void write_trace(const std::string& path);

}   // namespace nn::prof

// ── dump / write_trace 实现（放头尾部，header-only 需 inline）───────────────
namespace nn::prof
{

inline void write_trace(const std::string& path)
{
    if (path.empty()) return;
    detail::State& s = detail::state();
    std::vector<detail::Event> events;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        events = s.trace;
    }
#if defined(_MSC_VER)
    std::FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "wb") != 0) f = nullptr;
#else
    std::FILE* f = std::fopen(path.c_str(), "wb");
#endif
    if (f == nullptr)
    {
        std::fprintf(stderr, "[profile] 无法写 trace 文件: %s\n", path.c_str());
        return;
    }
    std::fprintf(f, "{\"traceEvents\":[");
    bool first = true;
    for (const auto& e : events)
    {
        std::fprintf(f, "%s{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\","
                        "\"ts\":%llu,\"dur\":%llu,\"pid\":0,\"tid\":%u}",
                     first ? "" : ",", e.name, e.cat,
                     static_cast<unsigned long long>(e.ts_us),
                     static_cast<unsigned long long>(e.dur_us), e.tid);
        first = false;
    }
    std::fprintf(f, "]}\n");
    std::fclose(f);
    std::fprintf(stderr, "[profile] trace 已写入 %s（%zu 事件）\n",
                 path.c_str(), events.size());
}

inline void dump(std::FILE* out, std::uint64_t wall_ns, bool reset)
{
    if (!g_enabled) return;
    if (out == nullptr) out = stderr;

    detail::State& s = detail::state();
    detail::StageMap stages;
    std::unordered_map<std::string_view, DeviceBucket> device;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        stages = s.stages;
        device = s.device;
    }

    std::fprintf(out, "\n[profile] ══════ 训练 step 剖析 ══════\n");

    // ── 表 1：stage（inclusive = 阶段墙钟；% 相对墙钟，嵌套阶段可读）──────
    std::fprintf(out, "[profile] ── 阶段构成（inclusive 墙钟）──\n");
    std::fprintf(out, "[profile] %-28s %10s %8s %12s\n",
                 "stage", "ms", "%wall", "calls");
    // 收集 stage 行（op 键为空串 = stage 本身），按耗时降序
    std::vector<std::pair<std::string_view, Bucket>> stage_rows;
    std::uint64_t stage_sum = 0;
    for (const auto& [st, ops] : stages)
    {
        auto it = ops.find(std::string_view());
        if (it == ops.end()) continue;
        stage_rows.emplace_back(st, it->second);
        stage_sum += it->second.incl_ns;
    }
    std::sort(stage_rows.begin(), stage_rows.end(),
              [](const auto& a, const auto& b)
              { return a.second.incl_ns > b.second.incl_ns; });
    for (const auto& [st, b] : stage_rows)
    {
        const double ms = static_cast<double>(b.incl_ns) / 1e6;
        // 分母 = 调用方给的墙钟；没给墙钟时退回 stage 合计
        const double denom = wall_ns > 0 ? static_cast<double>(wall_ns)
                                         : static_cast<double>(stage_sum);
        const double pct = denom > 0
            ? 100.0 * static_cast<double>(b.incl_ns) / denom : 0.0;
        std::fprintf(out, "[profile] %-28s %10.2f %7.1f%% %12llu\n",
                     std::string(st).c_str(), ms, pct,
                     static_cast<unsigned long long>(b.calls));
    }

    // ── 表 2：op 跨 stage 汇总（exclusive = 净归属）────────────────────────
    std::fprintf(out, "[profile] ── 原语净归属（exclusive，跨 stage 汇总）──\n");
    std::fprintf(out, "[profile] %-24s %10s %12s %10s %8s\n",
                 "op", "ms", "calls", "avg_us", "%wall");
    std::unordered_map<std::string_view, Bucket> op_rollup;
    for (const auto& [st, ops] : stages)
    {
        (void)st;
        for (const auto& [op, b] : ops)
        {
            if (op.empty()) continue;
            Bucket& r = op_rollup[op];
            r.calls += b.calls;
            r.incl_ns += b.incl_ns;
            r.excl_ns += b.excl_ns;
            if (b.max_ns > r.max_ns) r.max_ns = b.max_ns;
        }
    }
    std::vector<std::pair<std::string_view, Bucket>> op_rows(op_rollup.begin(),
                                                             op_rollup.end());
    std::sort(op_rows.begin(), op_rows.end(),
              [](const auto& a, const auto& b)
              { return a.second.excl_ns > b.second.excl_ns; });
    for (const auto& [op, b] : op_rows)
    {
        const double ms = static_cast<double>(b.excl_ns) / 1e6;
        const double avg_us = b.calls > 0
            ? static_cast<double>(b.excl_ns) / 1e3 / static_cast<double>(b.calls) : 0.0;
        const double pctw = wall_ns > 0
            ? 100.0 * static_cast<double>(b.excl_ns) / static_cast<double>(wall_ns) : 0.0;
        std::fprintf(out, "[profile] %-24s %10.2f %12llu %10.1f %7.1f%%\n",
                     std::string(op).c_str(), ms,
                     static_cast<unsigned long long>(b.calls), avg_us, pctw);
    }

    // ── 表 3：stage × op 明细（top 40，exclusive 降序）────────────────────
    std::fprintf(out, "[profile] ── stage × op 明细（exclusive，top 40）──\n");
    std::fprintf(out, "[profile] %-28s %-20s %10s %12s\n",
                 "stage", "op", "ms", "calls");
    std::vector<std::pair<std::string, Bucket>> mix_rows;
    for (const auto& [st, ops] : stages)
    {
        for (const auto& [op, b] : ops)
        {
            if (op.empty()) continue;
            mix_rows.emplace_back(std::string(st) + "::" + std::string(op), b);
        }
    }
    std::sort(mix_rows.begin(), mix_rows.end(),
              [](const auto& a, const auto& b)
              { return a.second.excl_ns > b.second.excl_ns; });
    for (std::size_t i = 0; i < mix_rows.size() && i < 40; ++i)
    {
        const auto& [key, b] = mix_rows[i];
        const auto pos = key.find("::");
        const std::string st = key.substr(0, pos);
        std::fprintf(out, "[profile] %-28s %-20s %10.2f %12llu\n",
                     st.empty() ? "(step 外)" : st.c_str(),
                     key.substr(pos + 2).c_str(),
                     static_cast<double>(b.excl_ns) / 1e6,
                     static_cast<unsigned long long>(b.calls));
    }

    // ── 表 4：GPU 设备侧 kernel ────────────────────────────────────────────
    if (!device.empty())
    {
        std::fprintf(out, "[profile] ── GPU 设备侧 kernel（timestamp query）──\n");
        std::fprintf(out, "[profile] %-28s %10s %12s %10s\n",
                     "kernel", "ms", "calls", "avg_us");
        std::uint64_t dev_sum = 0;
        std::vector<std::pair<std::string_view, DeviceBucket>> dev_rows;
        for (const auto& [name, b] : device)
        {
            dev_rows.emplace_back(name, b);
            dev_sum += b.total_ns;
        }
        std::sort(dev_rows.begin(), dev_rows.end(),
                  [](const auto& a, const auto& b)
                  { return a.second.total_ns > b.second.total_ns; });
        for (const auto& [name, b] : dev_rows)
        {
            std::fprintf(out, "[profile] %-28s %10.2f %12llu %10.1f\n",
                         std::string(name).c_str(),
                         static_cast<double>(b.total_ns) / 1e6,
                         static_cast<unsigned long long>(b.calls),
                         b.calls > 0 ? static_cast<double>(b.total_ns) / 1e3
                                         / static_cast<double>(b.calls) : 0.0);
        }
        if (wall_ns > 0)
        {
            std::fprintf(out, "[profile] GPU busy%% = %.1f%%（设备 kernel %.2f ms / 墙钟 %.2f ms）\n",
                         100.0 * static_cast<double>(dev_sum) / static_cast<double>(wall_ns),
                         static_cast<double>(dev_sum) / 1e6,
                         static_cast<double>(wall_ns) / 1e6);
        }
    }

    if (wall_ns > 0)
        std::fprintf(out, "[profile] 墙钟合计 %.2f ms\n",
                     static_cast<double>(wall_ns) / 1e6);
    std::fprintf(out, "[profile] ════════════════════════\n");
    std::fflush(out);

    // 设置了 NN_PROFILE_TRACE 时把逐事件时间轴落盘（Chrome trace JSON）；
    // 须在 reset 清空 trace 之前写出。
    if (!g_trace_path.empty()) write_trace(g_trace_path);

    if (reset)
    {
        std::lock_guard<std::mutex> lock(s.mu);
        s.stages.clear();
        s.device.clear();
        s.trace.clear();
    }
}

}   // namespace nn::prof
