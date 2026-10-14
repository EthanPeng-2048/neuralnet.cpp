// ── cli_precision_warning_test — 危险精度组合启动期告警（仅警告，不拒绝）────
//
// 验收（docs/todo/24-f16-training-defects.md §2.3 / §5 F1）：
//   [1] 触发集合 = optimizer=f16（四字段全 f16 被其覆盖）；stable=f16 单独不触发。
//   [2] 危险组合打印含字段/取值 + 风险 + 文档指引的警告；`--f16` 与合法组合静默。
//   [3] 行为 = 仅 stderr 文本：**不改变退出码、不阻断训练、不静默降级**。
//
// [1][2] 直接单测头文件函数 `nn::cli::dangerous_precision_warning`（廉价、确定、进默认门禁）。
// [3] 以子进程跑真实 CLI（text_train / mnist_train），把 stderr 重定向到文件后校验：
//     危险组合的 stderr 含警告且退出码与无告警组合一致；合法组合的 stderr 不含警告。
//     二进制缺失（纯库 / NN_BUILD_APPS=OFF 构建）→ 打印 [skip] 返回 77（ctest 记 Skipped）。
// 退出码：0 = 通过，1 = 失败，77 = 跳过。
// ─────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

#include "neuralnet.cpp/cli/cli_precision_warning.hpp"
#define NN_TEST_COUNTER g_fail
#include "test_common.hpp"

#ifndef _WIN32
#include <sys/wait.h>   // WEXITSTATUS
#endif

int g_fail = 0;

namespace
{
constexpr int kSkip = 77;   // ctest 约定：跳过（二进制缺失时不阻塞门禁）

std::string g_self;   // argv[0]：定位同构建的 app 二进制

std::string dirname_of(const std::string& path)
{
    const auto pos = path.find_last_of("/\\");
    return pos == std::string::npos ? std::string(".") : path.substr(0, pos);
}

bool file_exists(const std::string& p)
{
    std::ifstream f(p);
    return f.good();
}

std::string read_file(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
}

// 运行 cmd，stderr 落到 err_file；返回进程退出码（-1 = 无法启动子进程）。
int run_capture_stderr(const std::string& cmd, const std::string& err_file)
{
    const std::string full = cmd + " 2> \"" + err_file + "\"";
    const int rc = std::system(full.c_str());
#ifdef _WIN32
    return rc;
#else
    if (rc == -1) return -1;
    return WEXITSTATUS(rc);
#endif
}

// 在测试二进制附近寻找同构建的 app（<构建根>/<name> 或 <构建根>/test/../<name>）。
std::string find_app(const std::string& name)
{
    const std::string dir = dirname_of(g_self);
    const std::string candidates[] = {
        dir + "/../" + name,
        dir + "/" + name,
        name,
    };
    for (const auto& c : candidates)
        if (file_exists(c)) return c;
#ifdef _WIN32
    for (const auto& c : candidates)
        if (file_exists(c + ".exe")) return c + ".exe";
#endif
    return {};
}

// ── [1][2] 头文件函数的组合矩阵 ─────────────────────────────────────────
void test_helper_matrix()
{
    using P = nn::Precision;
    const auto has = [](nn::PrecisionProfile p) {
        return !nn::cli::dangerous_precision_warning(p).empty();
    };

    // 静默：f32 / --f16（profile_f16）/ --t1_58（profile_t1_58）
    CHECK(nn::cli::dangerous_precision_warning(nn::profile_f32()).empty(),
          "全 f32 应静默");
    CHECK(nn::cli::dangerous_precision_warning(nn::profile_f16()).empty(),
          "--f16（profile_f16）应静默");
    CHECK(nn::cli::dangerous_precision_warning(nn::profile_t1_58()).empty(),
          "--t1_58（profile_t1_58）应静默");
    // stable=f16 单独出现不触发（24 §2.3 口径）
    CHECK(!has({P::F32, P::F32, P::F16, P::F32}),
          "stable=f16 单独出现不触发告警");
    // param=t1_58（非 F16 存储）+ compute=f16 不触发
    CHECK(!has({P::T1_58, P::F16, P::F32, P::F32}),
          "param=t1_58 / compute=f16（optimizer=f32）应静默");

    // 触发：optimizer=f16（单独）→ 含字段与风险，且不含"全 f16"的冻结行
    const std::string opt_only =
        nn::cli::dangerous_precision_warning({P::F32, P::F32, P::F32, P::F16});
    CHECK(!opt_only.empty(), "optimizer=f16 必须触发告警");
    CHECK(opt_only.find("optimizer=f16") != std::string::npos,
          "告警应含命中的字段与取值 optimizer=f16");
    CHECK(opt_only.find("docs/todo/24-f16-training-defects.md") != std::string::npos,
          "告警应指回 24 号台账 §2");
    CHECK(opt_only.find("数值性冻结") == std::string::npos,
          "非全 f16 不应出现'数值性冻结'（那是全 f16 的形态）");

    // 触发：四字段全 f16（★ F2 补进的缺口组合）→ 合并一条更完整的警告
    const std::string all16 =
        nn::cli::dangerous_precision_warning(nn::profile_all_f16());
    CHECK(!all16.empty(), "四字段全 f16 必须触发告警");
    CHECK(all16.find("optimizer=f16") != std::string::npos &&
              all16.find("stable=f16") != std::string::npos,
          "全 f16 告警应同时列出 optimizer=f16 与 stable=f16");
    CHECK(all16.find("数值性冻结") != std::string::npos,
          "全 f16 告警应含'数值性冻结'风险（更新被舍入吃光）");

    // 触发：t1_58 版全 f16（param=T1_58，其余三槽 f16）→ stable 行随 optimizer 一并给出
    const std::string t1_all =
        nn::cli::dangerous_precision_warning({P::T1_58, P::F16, P::F16, P::F16});
    CHECK(!t1_all.empty() && t1_all.find("stable=f16") != std::string::npos,
          "t1_58 版全 f16 应触发并列出 stable=f16");
}

// ── [3] 真实 CLI 子进程：警告出现在 stderr、退出码不变 ───────────────────
// 返回 false = 二进制缺失（调用方据此计 skip）。
void test_cli_stderr(const std::string& exe, const std::string& label,
                     const std::string& dataset_arg)
{
    const std::string err_plain = "nn_precwarn_plain.err";
    const std::string err_f16 = "nn_precwarn_f16.err";
    const std::string err_opt = "nn_precwarn_opt.err";
    const std::string err_all = "nn_precwarn_all.err";

    const int rc_plain = run_capture_stderr(
        "\"" + exe + "\" " + dataset_arg, err_plain);
    const int rc_f16 = run_capture_stderr(
        "\"" + exe + "\" --f16 " + dataset_arg, err_f16);
    const int rc_opt = run_capture_stderr(
        "\"" + exe + "\" --precision-stable f16 --precision-optimizer f16 " + dataset_arg,
        err_opt);
    const int rc_all = run_capture_stderr(
        "\"" + exe + "\" --f16 --precision-stable f16 --precision-optimizer f16 " + dataset_arg,
        err_all);

    if (rc_plain == -1)
    {
        std::printf("  [skip] %s：无法启动子进程\n", label.c_str());
        return;
    }

    const std::string out_plain = read_file(err_plain);
    const std::string out_f16 = read_file(err_f16);
    const std::string out_opt = read_file(err_opt);
    const std::string out_all = read_file(err_all);

    CHECK(out_plain.find("[精度警告]") == std::string::npos,
          (label + "：无精度选项不应打印告警").c_str());
    CHECK(out_f16.find("[精度警告]") == std::string::npos,
          (label + "：--f16 不应打印告警").c_str());
    CHECK(out_opt.find("[精度警告]") != std::string::npos &&
              out_opt.find("optimizer=f16") != std::string::npos,
          (label + "：optimizer=f16 应在 stderr 打印含字段的告警").c_str());
    CHECK(out_all.find("数值性冻结") != std::string::npos,
          (label + "：全 f16 应打印含'数值性冻结'的告警").c_str());

    // 行为：警告不改退出码 —— 危险组合与无告警组合在同一（数据集缺失）失败点上退出码一致。
    CHECK(rc_f16 == rc_plain && rc_opt == rc_plain && rc_all == rc_plain,
          (label + "：告警不得改变退出码（危险组合与普通组合应一致）").c_str());

    std::remove(err_plain.c_str());
    std::remove(err_f16.c_str());
    std::remove(err_opt.c_str());
    std::remove(err_all.c_str());
}
} // namespace

int main(int argc, char** argv)
{
    g_self = (argc > 0 && argv[0] != nullptr) ? argv[0] : "";
    std::printf("cli_precision_warning_test（F1：危险精度组合启动期告警）\n");

    test_helper_matrix();

    bool any_app = false;
    // text_train 数据集是位置参数；mnist_train 用 --dataset <目录>。
    struct AppCase { const char* name; const char* dataset_arg; };
    const AppCase apps[] = {
        {"text_train",  "./nn_does_not_exist_dir_xyz/missing.nndataset"},
        {"mnist_train", "--dataset ./nn_does_not_exist_dir_xyz"},
    };
    for (const auto& app : apps)
    {
        const std::string exe = find_app(app.name);
        if (exe.empty())
        {
            std::printf("  [skip] 未找到 %s（纯库/未编译 app 构建）\n", app.name);
            continue;
        }
        any_app = true;
        test_cli_stderr(exe, app.name, app.dataset_arg);
    }

    if (!any_app)
    {
        std::printf("  [结果] 无 CLI 二进制可测 → SKIP\n");
        return kSkip;
    }

    if (g_fail == 0)
    {
        std::printf("  [结果] 全部通过\n");
        return 0;
    }
    std::printf("  [结果] 失败 %d 项\n", g_fail);
    return 1;
}
