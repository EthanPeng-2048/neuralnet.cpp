// ── assert_always_test — RV-01：前置条件校验在 Release（NDEBUG）下也必须终止 ──
//
// 背景：`NN_ASSERT` 是**可选诊断**（NDEBUG 下展开为空），把它用在"断言后仍继续
// 解引用/索引被校验对象"的位置 = Release 静默 UB（越界读写 / 空指针解引用）。
// 修法是把这类点位换成 `NN_ASSERT_ALWAYS` / `NN_FATAL`（两者**无条件**终止并打印
// `file:line`），本测试即该修法的负例回归：
//
// 覆盖（每个负例经子进程断言，本进程不终止；子进程 stderr 重定向到文件后校验）：
//   ① `MatrixT::at` 越界            → NN_FATAL（algebra_matrix.hpp）
//   ② 未绑定张量的 `Tensor::engine()`→ NN_ASSERT_ALWAYS（compute_tensor.hpp）
//   ③ `sample_next_token_` 空 logits → NN_ASSERT_ALWAYS（compute_layer_base.hpp）
//   ④ 未 init 的层调 forward（→ engine_ref）→ NN_FATAL（compute_layer_base.hpp）
//
// 断言口径（三条同时满足才算通过）：退出码非零 + 打印 `<tag> failed at` +
// 紧跟 `file:line`（冒号后是数字）+ 携带该点位自己的消息。tag 必须是新的
// `NN_ASSERT_ALWAYS` / `NN_FATAL`（而不是旧的 NN_ASSERT 路径）——后者在 NDEBUG
// 下根本不打印，本断言因此**在 Release 与 Debug 两种构建下都有效**。
//
// 子进程模式：--crash-matrix-at / --crash-tensor-engine / --crash-empty-logits
//             / --crash-layer-uninit
// （父进程 std::system 自举）。退出码：0 = 通过，1 = 失败。
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <cstring>   // std::strlen（tag 之后的扫描起点）
#include <random>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#include <neuralnet.cpp/nn.hpp>   // Matrix / Tensor / sample_next_token_

#include "test_common.hpp"

int g_fail = 0;

namespace {

constexpr int kSkip = -1;   // std::system 无法启动子进程（区别于任何真实退出码）

std::string g_self;                              // argv[0]：子进程自举用
const char* kLogPath = "nn_assert_always_child.log";   // 子进程 stderr 落地

const char* kTagAssert = "NN_ASSERT_ALWAYS failed at";
const char* kTagFatal  = "NN_FATAL failed at";

int run_self(const char* arg)
{
    // 子进程 stderr → 文件（父进程据此校验 file:line 打印；abort 消息在 stderr）
    const std::string cmd = "\"" + g_self + "\" " + arg + " 2> " + kLogPath;
    const int rc = std::system(cmd.c_str());
    if (rc == -1)
    {
        std::printf("[SKIP] 无法启动子进程（system 返回 -1）\n");
        return kSkip;
    }
    return rc;
}

int normalize_rc(int rc)
{
#ifdef _WIN32
    return rc;
#else
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
    if (WIFSIGNALED(rc)) return 128 + WTERMSIG(rc);
    return rc;
#endif
}

std::string read_log()
{
    std::string s;
    // MSVC CRT 把 fopen 标记 deprecated（-Werror 下编译失败）→ 走 fopen_s；
    // 与 core_profile.hpp 的落盘/读盘口径一致（clang++ 在 Windows 下也定义
    // _MSC_VER，故这里不能用"编译器是 clang ⇒ 走 POSIX"来判）。
    std::FILE* f = nullptr;
#if defined(_MSC_VER)
    if (fopen_s(&f, kLogPath, "rb") != 0) f = nullptr;
#else
    f = std::fopen(kLogPath, "rb");
#endif
    if (f == nullptr) return s;
    char buf[512];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
        s.append(buf, n);
    std::fclose(f);
    return s;
}

// "<tag> failed at <file>:<line>: <msg>" —— 三个要素都要在：tag 命中 → 其后
// **第一处"冒号 + 数字"**即 <file>:<line> 的分隔冒号 → 消息子串跟在它之后。
// ⚠ 不能取 tag 后的第一个冒号：Windows 路径自带驱动器冒号（`D:\nn-ab\...`），
// 取第一个会把 `D:` 当成行号分隔符（本测试在 102 的 clang-cl 配置下实测踩到）。
bool has_fatal_line(const std::string& log, const char* tag, const char* msg)
{
    const std::size_t p = log.find(tag);
    if (p == std::string::npos) return false;
    std::size_t colon = std::string::npos;
    for (std::size_t i = p + std::strlen(tag); i + 1 < log.size(); ++i)
    {
        if (log[i] == ':' && log[i + 1] >= '0' && log[i + 1] <= '9')
        {
            colon = i;
            break;
        }
    }
    if (colon == std::string::npos) return false;
    return log.find(msg, colon) != std::string::npos;
}

// 一个负例：非零退出 + 打印 tag/file:line/消息
void check_crash_case(const char* arg, const char* tag, const char* msg,
                      const char* what)
{
    const int rc = normalize_rc(run_self(arg));
    if (rc == kSkip) return;
    CHECK(rc != 0, what);

    const std::string log = read_log();
    const bool ok = has_fatal_line(log, tag, msg);
    CHECK(ok, (std::string(what) + "：stderr 含 tag/file:line/消息").c_str());
    if (!ok)
        std::printf("  子进程 stderr 捕获：<<<%s>>>\n", log.c_str());
}

} // namespace

int main(int argc, char* argv[])
{
    g_self = (argc > 0 && argv[0] != nullptr) ? argv[0] : "";

    // ── 子进程模式：只在父进程经 std::system 自举时进入（走到终止即成功）────
    if (argc >= 2)
    {
        const std::string mode = argv[1];
        if (mode == "--crash-matrix-at")
        {
            nn::Matrix m(2, 2);
            const nn::Scalar v = m.at(5, 5);   // 越界：断言后仍会 ptr()[index(5,5)]
            std::printf("不应到达 v=%g\n", static_cast<double>(v));
            return 0;
        }
        if (mode == "--crash-tensor-engine")
        {
            nn::Tensor t;                      // 未绑定引擎
            (void)t.engine();                  // 断言后仍 return *engine_
            std::printf("不应到达\n");
            return 0;
        }
        if (mode == "--crash-empty-logits")
        {
            std::vector<nn::Scalar> empty;
            std::mt19937_64 rng(1);
            std::uniform_real_distribution<nn::Scalar> dist(0.0, 1.0);
            const std::size_t v = nn::sample_next_token_(empty, 1.0, rng, dist);
            std::printf("不应到达 v=%zu\n", v);
            return 0;
        }
        if (mode == "--crash-layer-uninit")
        {
            // 未 init(engine) 的层 → engine_ref() → NN_FATAL（Release 下旧行为是
            // NN_ASSERT 为空、随后 `return *engine_` 空解引用）
            nn::Linear lin(3, 2);
            nn::CpuEngine eng;
            nn::Tensor x = eng.create_tensor(3, 1);
            const auto y = lin.forward(x);   // Result<Tensor>（失败即已终止）
            std::printf("不应到达 %zux%zu\n", y->rows(), y->cols());
            return 0;
        }
    }

    std::puts("=== assert_always (RV-01：前置条件校验在 NDEBUG 下也终止) ===");
    std::printf("[info] NDEBUG=%d（NN_ASSERT_ALWAYS/NN_FATAL 与构建类型无关）\n",
#ifdef NDEBUG
                1
#else
                0
#endif
    );

    // 正例：条件成立时宏不得有任何副作用
    NN_ASSERT_ALWAYS(2 + 2 == 4, "算术条件成立");
    CHECK(true, "NN_ASSERT_ALWAYS 条件成立时不终止");

    // 负例三件（详见文件头）
    check_crash_case("--crash-matrix-at", kTagFatal,
                     "MatrixT::at index out of range",
                     "MatrixT::at 越界 → 非零退出");
    check_crash_case("--crash-tensor-engine", kTagFatal,
                     "未绑定引擎（先调 init(engine)）",
                     "未绑定 Tensor::engine() → 非零退出");
    check_crash_case("--crash-empty-logits", kTagAssert,
                     "sample_next_token_: empty logits",
                     "sample_next_token_ 空 logits → 非零退出");
    check_crash_case("--crash-layer-uninit", kTagFatal,
                     "未绑定引擎（先调 init(engine)）",
                     "未 init 的层 forward → 非零退出");

    std::remove(kLogPath);
    std::printf("\nassert_always_test: %d failure(s)\n", g_fail);
    return g_fail != 0 ? 1 : 0;
}
