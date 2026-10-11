// ───────────────────────────────────────────────────────────────────────────
//  fused_generate.hpp — AOT 算子融合：构建期合成（scan_exprs 的生成阶段）
//
//  对注册表里的每条表达式，用 emitter 抽象（IR-D：expr_emitter.hpp）展开为
//  单个融合 kernel 源码，再经 glslc 编译成 SPIR-V，最后内联进单个生成头
//  fused_registry.hpp（key → {ExprSpec, SPIR-V}）。
//
//  **不再经由中间的 .bin 序列化**：本文件是头文件，由 `tools/scan_exprs.cpp`
//  直接 include 并在收集完注册表后原地调用 `nn::tool::generate_fused_registry`
//  —— 构建期因此只有一步（scan + generate），`expr_registry.hpp` 的
//  `write_registry/read_registry/kExprBinVersion` 随之删除。
//
//  IR-D 落地：经 nn::emitter_registry 选择后端（默认 "glsl" = GlslEmitter），
//  不直接绑定 GLSL 生成函数——同一份 canonical IR 可由其它后端展开
//  （`scan_exprs --list-backends` 查看已登记后端）。
//
//  表达式**文本只出现在 Layer**；这里只消费折叠后的结构（派生物）。
//  产物 fused_registry.hpp 供运行时（GpuEngine::eval_expr / vk_backend）
//  按 expr_spec_key 精确匹配 dispatch——闭合世界，未命中硬报错。
//
//  产物： <out_dir>/fused_registry.hpp（+ 调试用 .comp/.spv）
// ───────────────────────────────────────────────────────────────────────────

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core_file.hpp"      // nn::read_pod_span（读 glslc 产物 SPIR-V）
#include "expr_spec.hpp"
#include "expr_registry.hpp"
#include "expr_emitter.hpp"   // IR-D：emitter 抽象（后端选择）
#include "expr_glsl_gen.hpp"       // 注册 GlslEmitter（默认后端）

namespace
{

[[nodiscard]] bool run_glslc(const std::string& glslc,
                             const std::string& src, const std::string& dst)
{
    // 仅给 glslc 路径加引号；src/dst 为无空格路径（out_dir 由 CMake 控制），
    // 正斜杠相对路径不加引号，避免 cmd/system 解析问题。
    // --target-env=vulkan1.2 → SPIR-V 1.5：subgroup 算子（GL_KHR_shader_subgroup_*）
    // 需 SPIR-V ≥1.3；现基本找不到 <1.2 的显卡，直接定 vulkan1.2。
    const std::string cmd =
        "\"" + glslc + "\" -fshader-stage=compute --target-env=vulkan1.2 -o " + dst + " " + src;
    return std::system(cmd.c_str()) == 0;
}

[[nodiscard]] std::vector<std::uint32_t> read_spv(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    const std::streamsize sz = f.tellg();
    // tellg() 失败返回 -1：不判就 static_cast<size_t>(-1) 请求 ~4GB 分配。
    // SPIR-V 恒为 4 字节字的整数倍——不是即文件损坏（截断的 glslc 产物）。
    if (sz <= 0 || sz % static_cast<std::streamsize>(sizeof(std::uint32_t)) != 0)
        return {};
    f.seekg(0, std::ios::beg);
    std::vector<std::uint32_t> v(static_cast<std::size_t>(sz) / sizeof(std::uint32_t));
    if (!nn::read_pod_span(f, std::span(v)))
        return {};   // 截断读 → 视为失败（调用方报 [FAIL] 并中止构建）
    return v;
}

// ExprSpec → 生成头里的聚合初始化
// （ExprSpec{ instrs, views, consts, rparams, num_regs, matmul, fold, extras }——
//   字段必须与 ExprSpec 声明序逐一对应，否则 -Wmissing-field-initializers）
[[nodiscard]] std::string emit_spec(const nn::ExprSpec& spec)
{
    std::ostringstream o;
    const auto emit_seq = [&](const std::vector<nn::ExprInstr>& seq)
    {
        for (std::size_t i = 0; i < seq.size(); ++i)
        {
            const auto& in = seq[i];
            if (i) o << ",";
            o << " {" << static_cast<int>(in.op) << ", " << static_cast<int>(in.dst)
              << ", {" << static_cast<int>(in.a.kind) << ", " << static_cast<int>(in.a.idx) << "}"
              << ", {" << static_cast<int>(in.b.kind) << ", " << static_cast<int>(in.b.idx) << "}"
              << ", {" << static_cast<int>(in.c.kind) << ", " << static_cast<int>(in.c.idx) << "}}";
        }
    };
    const auto emit_scalar = [&](nn::Scalar v)
    {
        // ±inf（掩码屏蔽 / fold 状态初值）不能直接流输出（C++ 无 inf 字面量）
        if (std::isinf(v))
            o << (v < 0 ? " -std::numeric_limits<Scalar>::infinity()"
                        : " std::numeric_limits<Scalar>::infinity()");
        else
            o << " static_cast<Scalar>(" << v << ")";
    };
    o << "ExprSpec{ \n";
    o << "        std::vector<ExprInstr>{";
    emit_seq(spec.instrs);
    o << "},\n        std::vector<ExprView>{";
    for (std::size_t i = 0; i < spec.views.size(); ++i)
    {
        const auto& v = spec.views[i];
        if (i) o << ",";
        o << " {" << static_cast<int>(v.kind) << ", "
          << static_cast<int>(v.negate_first_half) << ", " << v.param << "u, "
          << v.param2 << "u}";
    }
    o << "},\n        std::vector<Scalar>{";
    for (std::size_t i = 0; i < spec.consts.size(); ++i)
    {
        if (i) o << ",";
        emit_scalar(spec.consts[i]);
    }
    o << "},\n        std::vector<Scalar>{";   // rparams（运行时标量参数，值本身不参与 key）
    for (std::size_t i = 0; i < spec.rparams.size(); ++i)
    {
        if (i) o << ",";
        o << " static_cast<Scalar>(" << spec.rparams[i] << ")";
    }
    o << "}, " << spec.num_regs << "u, \n        std::optional<MatmulSpec>{";
    if (spec.matmul)
    {
        o << "MatmulSpec{" << static_cast<int>(spec.matmul->a_input) << ", "
          << static_cast<int>(spec.matmul->b_input) << ", "
          << static_cast<int>(spec.matmul->transA) << ", "
          << static_cast<int>(spec.matmul->transB) << ", "
          << spec.matmul->k << "u, "
          << spec.matmul->batch << "u}";  // batch 必须显式生成（漏则退化为默认 1，registry spec 失真）
    }
    o << "}, \n        std::optional<FoldSpec>{";
    if (spec.fold)
    {
        const nn::FoldSpec& f = *spec.fold;
        o << "FoldSpec{ " << static_cast<int>(f.num_state) << ", " << f.k << "u, "
          << "std::vector<Scalar>{";
        for (std::size_t i = 0; i < f.inits.size(); ++i)
        {
            if (i) o << ",";
            emit_scalar(f.inits[i]);   // -inf 状态初值（rowmax 型）走 limits 字面量
        }
        o << "}, std::vector<ExprInstr>{";
        emit_seq(f.body);
        o << "}, std::vector<ExprInstr>{";
        emit_seq(f.finalize);
        // fold v2 双域字段（声明序：vec_state_len, matmul, vecacc）
        o << "}, " << f.vec_state_len << "u, std::optional<MatmulSpec>{";
        if (f.matmul)
        {
            o << "MatmulSpec{" << static_cast<int>(f.matmul->a_input) << ", "
              << static_cast<int>(f.matmul->b_input) << ", "
              << static_cast<int>(f.matmul->transA) << ", "
              << static_cast<int>(f.matmul->transB) << ", "
              << f.matmul->k << "u, " << f.matmul->batch << "u}";
        }
        o << "}, std::optional<VecAccSpec>{";
        if (f.vecacc)
        {
            o << "VecAccSpec{ " << static_cast<int>(f.vecacc->vec_state) << ", "
              << static_cast<int>(f.vecacc->weight_reg) << ", "
              << static_cast<int>(f.vecacc->b_input) << ", "
              << static_cast<int>(f.vecacc->scale_reg) << ", "
              << static_cast<int>(f.vecacc->has_scale) << " }";
        }
        // FoldSpec 声明序末位 = tri_skip（聚合初始化 positional，字段加结构尾）
        o << "} , " << (f.tri_skip ? "true" : "false") << " }";
    }
    o << "}";
    // P2 多输出（extras）：ExprSpec 聚合第 8 字段——漏发则 registry spec 丢
    // extras → 管线布局少输出绑定 → 多输出 shader 绑定错位、结果静默错值
    o << ", std::vector<std::uint8_t>{";
    for (std::size_t i = 0; i < spec.extras.size(); ++i)
    {
        if (i) o << ",";
        o << " static_cast<std::uint8_t>("
          << static_cast<unsigned>(spec.extras[i]) << ")";
    }
    o << "} }";
    return o.str();
}

} // namespace

namespace nn::tool
{

// 构建期工具的环境变量开关（非空且首字符非 '0'）；与库里的 nn::dsl::env_flag
// 同语义，但工具链不为一个开关拉进整个 DSL 头。两个构建期工具共用本函数。
[[nodiscard]] inline bool tool_env_flag(const char* name)
{
#if defined(_MSC_VER)
    char* b = nullptr; std::size_t n = 0;
    _dupenv_s(&b, &n, name);
    const bool v = (b != nullptr && b[0] != '\0' && b[0] != '0');
    std::free(b);
    return v;
#else
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
#endif
}

// 同上的整数版（缺省/空串/非正数 → fallback）。MSVC 下 std::getenv 是
// -Wdeprecated-declarations 且在 -Werror 下直接失败，故走 _dupenv_s。
[[nodiscard]] inline int tool_env_int(const char* name, int fallback)
{
    int v = fallback;
#if defined(_MSC_VER)
    char* b = nullptr; std::size_t n = 0;
    _dupenv_s(&b, &n, name);
    if (b != nullptr && b[0] != '\0') v = std::atoi(b);
    std::free(b);
#else
    const char* e = std::getenv(name);
    if (e != nullptr && e[0] != '\0') v = std::atoi(e);
#endif
    return v;
}

// 该结构是否可生成融合 shader（生成循环 / emit_one / 元数据循环三处共用同一
// 判据——此前三处各写一遍，靠注释"同上"维持同步，漏一处就是"元数据与 shader
// 集合不一致"）。raxis == -2（行+列混合归约）无法单 kernel 融合；
// 空指令表且无 matmul/fold 段 = 结构损坏（上游变换丢了 fold 段）。
[[nodiscard]] inline bool is_fusable(const nn::ExprSpec& spec)
{
    if (nn::expr_spec_reduce_axis(spec) == -2) return false;
    if (spec.instrs.empty() && !spec.matmul && !spec.fold) return false;
    return true;
}

// 该结构/变体的每线程元素数（1=标量, 4=vec4）：vec4 资格 = 非 fold、非归约、
// 且逐元素形态可向量化；native16（ALU）变体恒标量（生成器同源判定）。
[[nodiscard]] inline std::uint32_t fused_vec_width_for(const nn::ExprSpec& spec,
                                                       bool alu)
{
    if (alu || spec.fold) return 1u;
    if (nn::expr_spec_reduce_axis(spec) >= 0) return 1u;
    return nn::glsl_vec4_eligible(spec) ? 4u : 1u;
}

// ── 生成期并行执行（glslc 子进程是这一阶段的全部耗时）──────────────────────
// 实测：91 条结构 + 130 条精度变体 = 221 次 glslc，单次 ~250ms，**串行**合计
// ~55s —— 占满整个构建期单步（所有 C++ TU 都 `add_dependencies(
// compile_gpu_shaders)`，等于全量构建开头硬等 55s）。
// glslc 是无状态子进程，天然可并行：默认 hardware_concurrency 个线程，可用
// NN_SCAN_JOBS=<n> 覆盖（调试/限流用）。
struct GenJob
{
    std::string ident;      // 生成头里的标识符后缀：key / key_x / key_a
    std::string label;      // 诊断用键名：key / key#x / key#a
    std::string comp_path;
    std::string spv_path;
    std::string glsl;
    std::vector<std::uint32_t> spv;
    bool ok = false;
};

[[nodiscard]] inline unsigned gen_job_threads()
{
    const int override_n = tool_env_int("NN_SCAN_JOBS", 0);
    if (override_n > 0) return static_cast<unsigned>(override_n);
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1u : hw;
}

// 并行：写 .comp → glslc → 读 .spv，逐条独立。任一条失败只标记该 job（其余
// 继续跑完再统一失败），对构建的可见语义与串行版一致（同样中止构建、同样
// 打印 [FAIL]）；只是不再"首发失败即丢弃后面所有已排好的工作"。
inline void run_gen_jobs(const std::string& glslc, std::vector<GenJob>& jobs)
{
    if (jobs.empty()) return;
    const unsigned nthreads =
        std::min<unsigned>(gen_job_threads(), static_cast<unsigned>(jobs.size()));
    std::atomic<std::size_t> next{0};
    const auto worker = [&]()
    {
        for (;;)
        {
            const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= jobs.size()) return;
            GenJob& j = jobs[i];
            {
                std::ofstream f(j.comp_path, std::ios::binary);
                if (!f)
                {
                    std::fprintf(stderr, "[FAIL] 无法写入 %s\n", j.comp_path.c_str());
                    continue;
                }
                f << j.glsl;
            }
            if (!run_glslc(glslc, j.comp_path, j.spv_path))
            {
                std::fprintf(stderr, "[FAIL] glslc 编译 %s 失败\n", j.label.c_str());
                continue;
            }
            j.spv = read_spv(j.spv_path);
            if (j.spv.empty())
            {
                std::fprintf(stderr, "[FAIL] 读取 %s 失败\n", j.spv_path.c_str());
                continue;
            }
            j.ok = true;
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nthreads > 0 ? nthreads - 1 : 0);
    for (unsigned t = 1; t < nthreads; ++t) pool.emplace_back(worker);
    worker();                       // 调用线程也干活，省一个线程的启动/回收
    for (std::thread& th : pool) th.join();
}

// 磁盘上的文件是否与给定内容相同（不存在 = 不同）。
// **文本模式**（与旧的无条件覆写一致）：Windows 上写出 CRLF、读入时再翻译回
// LF，两边口径一致；改二进制模式会让生成头从 CRLF 变 LF，平白制造全文件 diff。
[[nodiscard]] inline bool file_content_equal(const std::string& path,
                                             const std::string& content)
{
    std::ifstream in(path);
    if (!in) return false;
    std::ostringstream cur;
    cur << in.rdbuf();
    return cur.str() == content;
}

// 只在**内容真的变了**时落盘。
// ninja 给这个自定义命令生成了 `restat = 1`：输出 mtime 不变 → 下游（全部
// C++ TU 都 include 这份注册表）**不重建**。改一个 Layer 头会重新跑
// scan_exprs，但多数时候折叠出的结构逐字节不变——"无条件 ofstream 覆写"
// 会把这种情况放大成全量重编。
[[nodiscard]] inline bool write_if_different(const std::string& path,
                                             const std::string& content)
{
    if (file_content_equal(path, content)) return true;   // 内容相同 → 不碰 mtime
    std::ofstream f(path);
    if (!f) return false;
    f << content;
    return true;
}

// 生成融合 shader 注册表（见文件头说明）。reg 由调用方（scan_exprs）收集好后
// 传入。
[[nodiscard]] inline bool generate_fused_registry(
    const std::string& out_dir,
    const std::string& glslc,
    const nn::fused::ExprRegistry& reg)
{
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    // A/B 逃生阀：跳过运行期精度分派变体（键 #x）的发射。
    const bool nn_no_dispatch = tool_env_flag("NN_SCAN_NO_DISPATCH");

    if (reg.specs.empty())
    {
        std::fprintf(stderr, "[FAIL] 表达式集合为空（scan_exprs 未覆盖任何路径）\n");
        return false;
    }

    std::ostringstream H;
    H << "// ═══════════════════════════════════════════════════════════════\n";
    H << "//  fused_registry.hpp — AUTO-GENERATED by tools/fused_generate.hpp，请勿手动编辑\n";
    H << "//  生成者 = scan_exprs（收集 + 合成一体，见 tools/scan_exprs.cpp）\n";
    H << "//  每个条目：key(=expr_spec_key[#x|#a]) → {ExprSpec 结构, 内联 SPIR-V}\n";
    H << "//    · key            V0 全 f32\n";
    H << "//    · key#x          V1 运行期精度分派（PC `prec`，覆盖任意签名）\n";
    H << "//    · key#a          V2 native16 原生 f16 算术（结构谓词判定）\n";
    H << "//    · key#b          P3 批量派发（实例表 + buffer device address，"
         "仅纯逐元素 V0）\n";
    H << "//  运行时按 key 精确匹配 dispatch（闭合世界）。\n";
    H << "// ═══════════════════════════════════════════════════════════════\n";
    H << "#ifndef NN_FUSED_REGISTRY_HPP\n#define NN_FUSED_REGISTRY_HPP\n";
    H << "#define NN_FUSED_REGISTRY_EMBEDDED\n";
    H << "#include <cstdint>\n#include <limits>\n#include <string>\n#include <vector>\n";
    H << "#include \"neuralnet.cpp/expr_spec.hpp\"\n";
    H << "namespace nn::fused {\n";
    H << "struct FusedShader {\n";
    H << "    const char* key;\n";
    H << "    ExprSpec    spec;\n";
    H << "    const std::uint32_t* spirv;\n";
    H << "    std::size_t spirv_words;\n";
    H << "    int         reduce_axis;   // -1=逐元素, 0=行归约, 1=列归约\n";
    H << "    int         has_matmul;     // 1=含前置 matmul 段（push constants 多 rows+mm_k）\n";
    H << "    std::uint32_t view_param_count;  // 运行时视图参数个数（RowMod/RotateHalf 的 push constant vp 槽）\n";
    H << "    std::uint32_t rparam_count;    // 运行时标量参数个数（优化器 lr/eps/β 等的 push constant rp 槽）\n";
    H << "    std::uint32_t vec_width;    // 每线程处理元素数（1=标量, 4=vec4）\n";
    H << "    std::uint32_t prec_sig;     // 变体签名：V0=0；V1=EXPR_PREC_SIG_DISPATCH\n";
    H << "                                //   （运行期按真实签名填 PC `prec`）；\n";
    H << "                                //   V2=\"全输入 f16 + 输出 f16\"位图\n";
    H << "};\n\n";

    // 生成期任务队列：顺序 = 生成头里 SPIR-V 数组的顺序（先全部基础结构，
    // 再按发射顺序的全部精度变体）。GLSL 在这一轮全部产出，glslc 在
    // run_gen_jobs 里并行跑完，最后统一按序回填 H——输出与串行版逐字节一致。
    std::vector<GenJob> jobs;

    for (const auto& spec : reg.specs)
    {
        if (!is_fusable(spec))
        {
            // 混合归约轴（行+列）无法单 kernel 融合 / 空指令表无段 = 结构损坏
            // （上游变换丢掉 fold 段即此形态）。跳过而非让生成器对空表 back()
            // 越界（表现为 0xC00000FD 栈崩溃）。正常运行期到不了这里。
            std::fprintf(stderr, "[skip] 不可融合结构: %s\n",
                         nn::expr_spec_key(spec).c_str());
            continue;
        }
        // matmul+列归约（列方向，如 col_max(matmul)）：生成器按
        // 元素分解 batch（batch = row/m_per），归约遍历全部 rows（含所有
        // batch），与 CPU 端 matmul_out 语义一致 → 正常生成，不跳过。
        // matmul+归约（注意力结构）：generate_glsl_reduce 支持 Matmul
        // 操作数（内联点积，不物化 (batch*M,N) 中间矩阵），同样不跳过。
        const int raxis = nn::expr_spec_reduce_axis(spec);
        const std::string key = nn::expr_spec_key(spec);

        // 经 emitter 抽象（IR-D）生成 kernel 源码：默认 GlslEmitter。
        // 同一份 canonical IR 可由其它后端展开（--list-backends）。
        auto emitter = nn::emitter_registry::make("glsl");
        if (!emitter)
        {
            std::fprintf(stderr, "[FAIL] 无法创建 GLSL emitter（IR-D 注册表异常）\n");
            return false;
        }
        // 含归约（raxis >= 0）→ 归约 kernel（workgroup 级共享内存归约）；否则逐元素
        const std::string glsl = (raxis >= 0)
            ? emitter->generate_reduce("fused_" + key, spec)
            : emitter->generate("fused_" + key, spec);
        if (glsl.empty())
        {
            std::fprintf(stderr, "[FAIL] 生成 %s 失败（归约结构不支持？）\n", key.c_str());
            return false;
        }
        jobs.push_back(GenJob{key, key,
                              out_dir + "/fused_" + key + ".comp",
                              out_dir + "/fused_" + key + ".spv",
                              glsl, {}, false});
    }

    // ── 结构变体：V1 运行期精度分派（键 = key#x）与 V2 native16（键 = key#a）──
    //
    // V1：**每个结构无条件发射一份**（除非 A/B 逃生阀关闭）——输入/输出各声明
    // f32 + float16_t 双视图，加载/存储按 push constant `uint prec` 走 uniform
    // 分支 → 一个 shader 覆盖全部 (输入精度位图, 输出精度)。这是"签名不再需要
    // 构建期枚举"的载体：构建期只需"结构"这一维（锚点 + dry-run 已自动）。
    //
    // V2：生成条件 = **结构谓词** expr_prec_sig_native16 代入"全输入 f16 +
    // 输出 f16"这一结构性常量签名 → 无 fold / 无 matmul 段 / 无归约 / 无 rparams
    // / 无常量微值。**该谓词含常量值判据，而常量值不进 key**：运行期选
    // `#a` 前必须用真实实例的 spec 重跑同一谓词（GpuEngine::find_prec_variant_）
    // ——否则"生成时那个实例的常量"会被套用到常量不同的同结构实例上（微值
    // 在 f16 下 flush-to-0 → 静默错值）。
    // ALU 变体恒为**标量 kernel**（generate_glsl native16 分支不走 vec4），
    // vec_width 写 1（与生成器同源，否则 dispatch 宽度失配 → 静默算 1/4）。
    //
    // A/B 逃生阀：NN_SCAN_NO_DISPATCH=1 时不发射 V1（退回旧集合）；
    // 只影响登记集合，不影响语义。
    struct VariantEmit
    {
        const nn::ExprSpec* spec;
        nn::ExprPrecSig sig;      // 变体签名：分派哨兵 / "全 f16" 位图
        std::string     vkey;     // "key#x" 或 "key#a"（注册表键）
        std::string     suffix;   // "key_x" / "key_a"（标识符与文件名，不含 '#'）
        bool            alu = false;   // native16 变体（vec_width 恒 1）
    };
    std::vector<VariantEmit> emitted_variants;
    // 去重：同一 vkey 只发射一次（重复会生成重复的 kSpirv_ 标识符 → 重定义错误）
    std::unordered_set<std::string> emitted_vkeys;
    // 单个 (spec, sig) 的发射：alu = native16（V2）；否则分派（V1）
    const auto emit_one = [&](const nn::ExprSpec& spec, nn::ExprPrecSig sig,
                              bool alu) -> bool {
        if (!is_fusable(spec))
            return false;
        if (alu && !nn::expr_prec_sig_native16(spec, sig))
            return false;   // 谓词不通过 → 不发 ALU 变体（V1 仍覆盖）
        const std::string key = nn::expr_spec_key(spec);
        // 只有两种结构变体（精度不再是身份）：
        //   V1 运行期精度分派 → 键 key#x
        //   V2 native16（结构谓词判定）→ 键 key#a
        const bool dispv = nn::expr_prec_sig_is_dispatch(sig);
        const std::string vkey = dispv ? (key + nn::EXPR_PREC_DISPATCH_SUFFIX)
                                       : (key + nn::EXPR_PREC_ALU_SUFFIX);
        const std::string suffix = dispv ? (key + "_x") : (key + "_a");
        if (!emitted_vkeys.insert(vkey).second)
            return false;   // 已发射 → 幂等跳过
        const std::string comp_path = out_dir + "/fused_" + suffix + ".comp";
        const std::string spv_path  = out_dir + "/fused_" + suffix + ".spv";
        auto emitter = nn::emitter_registry::make("glsl");
        if (!emitter)
        {
            std::fprintf(stderr, "[FAIL] 无法创建 GLSL emitter（IR-D 注册表异常）\n");
            return false;
        }
        const int raxis = nn::expr_spec_reduce_axis(spec);
        const std::string glsl = (raxis >= 0)
            ? emitter->generate_reduce("fused_" + suffix, spec, sig)
            : emitter->generate("fused_" + suffix, spec, sig, alu);
        if (glsl.empty())
        {
            if (alu)
                return false;   // 结构谓词通过但发射器不支持该形态 → 静默跳过
                                //（V0/V1 仍覆盖，只是拿不到原生 f16 算术）
            // 分派变体（key#x）是唯一的签名无关 shader：发不出来 = 该结构
            // 没有 f16 路径 → 必须可见。**两类成因分开打**（否则"设计内跳过"
            // 被误读成"形态缺失报错"——2026-10-06 勘误）：多输出（extras）
            // 结构按设计仅 V0（expr_glsl_gen P2 写穿约束），混精度走
            // execute_fused_graph 的逐成员降级；其余才是真形态缺口。
            if (!spec.extras.empty())
                std::fprintf(stderr,
                             "[skip] 多输出(extras) 结构 %s 按设计不生成 #x/#a 变体"
                             "（仅 V0；混精度走逐成员降级）\n", vkey.c_str());
            else
                std::fprintf(stderr, "[skip] 分派变体 %s 的形态暂不支持生成\n",
                             vkey.c_str());
            return false;
        }
        // 只登记任务：glslc 统一在 run_gen_jobs 里并行执行。
        jobs.push_back(GenJob{suffix, vkey, comp_path, spv_path, glsl, {}, false});
        emitted_variants.push_back(VariantEmit{&spec, sig, vkey, suffix, alu});
        return true;
    };

    for (const auto& spec : reg.specs)
    {
        if (!is_fusable(spec))
            continue;
        if (!nn_no_dispatch &&
            !emit_one(spec, nn::EXPR_PREC_SIG_DISPATCH, /*alu=*/false))
        {
            // V1 发不出来只是少了 f16 路径（见 emit_one），V2/V0 仍可能成功；
            // 真正的失败（生成 GLSL 失败）已在 emit_one 内报 [FAIL]/[skip]。
        }
        // V2：结构性"全输入 f16 + 输出 f16"签名（输入位掩码取满，
        // 谓词内部再按 views 数量与其余条件判定）
        emit_one(spec, nn::expr_prec_sig_make(nn::EXPR_PREC_SIG_INPUT_MASK,
                                              /*out_f16=*/true),
                 /*alu=*/true);
    }

    // ── P3 跨链批量派发变体（键 = key#b）──────────────────────────────────
    //
    // 资格与运行期分组同源（expr_spec_batchable）：纯逐元素（无归约/fold/
    // matmul），**含 P2 多输出 extras 复合结构**（optimizer 主力）；只发 V0
    // 全 f32（f16/混精度走既有逐 kernel 路径）。变体行元数据：reduce_axis=-1、
    // has_matmul=0、vec_width=1（批量恒标量，与生成器同源）、vp/rparam 槽位数
    // 与结构一致、prec_sig=0。`#b` 未登记或设备无 buffer device address 时
    // 运行期逐 kernel 降级（闭合世界降级，不是错误）。
    struct BatchedEmit
    {
        const nn::ExprSpec* spec;
        std::string vkey;     // "key#b"（注册表键）
        std::string suffix;   // "key_b"（标识符与文件名，不含 '#'）
    };
    std::vector<BatchedEmit> emitted_batched;
    std::unordered_set<std::string> emitted_bkeys;
    for (const auto& spec : reg.specs)
    {
        if (!is_fusable(spec) || !nn::expr_spec_batchable(spec))
            continue;
        const std::string key = nn::expr_spec_key(spec);
        const std::string vkey = key + nn::EXPR_BATCH_SUFFIX;
        const std::string suffix = key + "_b";
        if (!emitted_bkeys.insert(vkey).second)
            continue;   // 同 vkey 幂等（重复会生成重复 kSpirv_ 标识符）
        auto emitter = nn::emitter_registry::make("glsl");
        if (!emitter)
        {
            std::fprintf(stderr, "[FAIL] 无法创建 GLSL emitter（IR-D 注册表异常）\n");
            return false;
        }
        const std::string glsl = emitter->generate_batched("fused_" + suffix, spec);
        if (glsl.empty())
        {
            // 资格判据与生成器同一函数 → 正常不可达；保底可见（不静默）。
            std::fprintf(stderr, "[skip] 批量变体 %s 的形态暂不支持生成\n",
                         vkey.c_str());
            continue;
        }
        jobs.push_back(GenJob{suffix, vkey,
                              out_dir + "/fused_" + suffix + ".comp",
                              out_dir + "/fused_" + suffix + ".spv",
                              glsl, {}, false});
        emitted_batched.push_back(BatchedEmit{&spec, vkey, suffix});
    }

    // ── 并行 glslc：GLSL 已全部产出，这里一次性并发编译 ────────────────────
    run_gen_jobs(glslc, jobs);
    for (const GenJob& j : jobs)
    {
        if (!j.ok)
            return false;   // [FAIL] 明细已在 run_gen_jobs 内逐条打印
    }

    // SPIR-V 数组按 jobs 顺序回填（= 串行版的「基础结构全部在前、变体全部在后」）
    for (const GenJob& j : jobs)
    {
        H << "inline constexpr std::uint32_t kSpirv_" << j.ident << "[] = {";
        for (std::size_t i = 0; i < j.spv.size(); ++i)
        {
            if (i % 8 == 0) H << "\n    ";
            H << "0x" << std::hex << j.spv[i] << "u, ";
        }
        H << std::dec << "\n};\n\n";
    }

    H << "inline const FusedShader kFusedShaders[] = {\n";
    for (const auto& spec : reg.specs)
    {
        if (!is_fusable(spec))
            continue;  // 与上方生成循环同一判据（共享 is_fusable）
        const std::string key = nn::expr_spec_key(spec);
        const std::string spec_lit = emit_spec(spec);   // 三行共用，只生成一次
        H << "    { \"" << key << "\",\n        " << spec_lit << ",\n"
          << "        kSpirv_" << key
          << ", sizeof(kSpirv_" << key << ")/sizeof(std::uint32_t), "
          << nn::expr_spec_reduce_axis(spec) << ", " << (spec.matmul ? 1 : 0) << ", "
          << nn::expr_spec_runtime_view_param_count(spec) << ", "
          << nn::expr_spec_runtime_param_count(spec) << ", "
          << fused_vec_width_for(spec, /*alu=*/false)
          << ", 0u },\n";
    }
    // 变体行（键 = key#x 或 key#a；元数据与基础结构同源——精度不进结构）
    for (const auto& ve : emitted_variants)
    {
        const nn::ExprSpec& spec = *ve.spec;
        H << "    { \"" << ve.vkey << "\",\n        " << emit_spec(spec) << ",\n"
          << "        kSpirv_" << ve.suffix
          << ", sizeof(kSpirv_" << ve.suffix << ")/sizeof(std::uint32_t), "
          << nn::expr_spec_reduce_axis(spec) << ", " << (spec.matmul ? 1 : 0) << ", "
          << nn::expr_spec_runtime_view_param_count(spec) << ", "
          << nn::expr_spec_runtime_param_count(spec) << ", "
          << fused_vec_width_for(spec, ve.alu)
          << ", " << ve.sig << "u },\n";
    }
    // P3 批量变体行（键 = key#b）：元数据 = 逐元素形态（reduce_axis=-1、
    // has_matmul=0）+ vec_width=1（批量恒标量）+ vp/rparam 槽位数与结构一致。
    for (const auto& be : emitted_batched)
    {
        const nn::ExprSpec& spec = *be.spec;
        H << "    { \"" << be.vkey << "\",\n        " << emit_spec(spec) << ",\n"
          << "        kSpirv_" << be.suffix
          << ", sizeof(kSpirv_" << be.suffix << ")/sizeof(std::uint32_t), "
          << "-1, 0, "
          << nn::expr_spec_runtime_view_param_count(spec) << ", "
          << nn::expr_spec_runtime_param_count(spec) << ", "
          << "1u, 0u },\n";
    }
    H << "};\n";
    H << "inline constexpr std::size_t kFusedShaderCount =\n"
      << "    sizeof(kFusedShaders) / sizeof(kFusedShaders[0]);\n";
    H << "[[nodiscard]] inline const FusedShader* find_fused(const std::string& key)\n";
    H << "{\n";
    H << "    for (const auto& f : kFusedShaders)\n";
    H << "        if (key == f.key) return &f;\n";
    H << "    return nullptr;\n";
    H << "}\n";
    H << "} // namespace nn::fused\n";
    H << "#endif // NN_FUSED_REGISTRY_HPP\n";

    const std::string reg_path = out_dir + "/fused_registry.hpp";
    const std::string reg_text = H.str();
    // 内容不变则**不落盘**（保持 mtime）——ninja 的 restat=1 据此判定下游
    // 是否重建；无条件覆写会把"结构没变"的重新生成放大成全量重编。
    const bool reg_changed = !file_content_equal(reg_path, reg_text);
    if (!write_if_different(reg_path, reg_text))
    {
        std::fprintf(stderr, "[FAIL] 无法写入 %s\n", reg_path.c_str());
        return false;
    }
    std::printf("[gen] %zu 条融合表达式 + %zu 条精度变体 + %zu 条批量变体 -> %s%s\n",
                reg.specs.size(), emitted_variants.size(), emitted_batched.size(),
                reg_path.c_str(),
                reg_changed ? "" : "（内容未变化，保持 mtime）");
    return true;
}

} // namespace nn::tool
