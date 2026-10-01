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
    f.seekg(0, std::ios::beg);
    std::vector<std::uint32_t> v(static_cast<std::size_t>(sz) / sizeof(std::uint32_t));
    if (!v.empty())
        nn::read_pod_span(f, std::span(v));
    return v;
}

// ExprSpec → 生成头里的聚合初始化
// （ExprSpec{ instrs, views, consts, rparams, num_regs, matmul, fold }——
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
    o << "} }";
    return o.str();
}

} // namespace

namespace nn::tool
{

// 生成融合 shader 注册表（见文件头说明）。reg 由调用方（scan_exprs）收集好后
// 传入；prec_manifest 可为空（空 = 不做 run-only 签名回填）。
[[nodiscard]] inline bool generate_fused_registry(
    const std::string& out_dir,
    const std::string& glslc,
    const nn::fused::ExprRegistry& reg,
    const std::string& prec_manifest = {})
{
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    // A/B 逃生阀：跳过运行期精度分派变体（键 #x）的发射。
    const bool nn_no_dispatch = [] {
#if defined(_MSC_VER)
        char* b = nullptr; std::size_t n = 0;
        _dupenv_s(&b, &n, "NN_SCAN_NO_DISPATCH");
        const bool v = (b != nullptr && b[0] != '\0' && b[0] != '0');
        std::free(b);
        return v;
#else
        const char* v = std::getenv("NN_SCAN_NO_DISPATCH");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
#endif
    }();

    if (reg.specs.empty())
    {
        std::fprintf(stderr, "[FAIL] 表达式集合为空（scan_exprs 未覆盖任何路径）\n");
        return false;
    }

    std::ostringstream H;
    H << "// ═══════════════════════════════════════════════════════════════\n";
    H << "//  fused_registry.hpp — AUTO-GENERATED by gen_fused，请勿手动编辑\n";
    H << "//  构建期 scan_exprs 收集 + gen_fused 合成的融合 shader 注册表\n";
    H << "//  每个条目：key(=expr_spec_key) → {ExprSpec 结构, 内联 SPIR-V}\n";
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
    H << "    std::uint32_t prec_sig;     // 精度签名（Phase 2 in-kernel f16；0 = 全 f32）\n";
    H << "                                //   位 i = 第 i 个输入 f16，bit16 = 输出 f16\n";
    H << "};\n\n";

    for (const auto& spec : reg.specs)
    {
        const int raxis = nn::expr_spec_reduce_axis(spec);
        if (raxis == -2)
        {
            // 混合归约轴（行+列）无法单 kernel 融合：跳过，运行时闭合世界硬报错
            std::fprintf(stderr, "[skip] 混合归约轴结构不支持融合: %s\n",
                         nn::expr_spec_key(spec).c_str());
            continue;
        }
        // matmul+列归约（列方向，如 col_max(matmul)）：生成器按
        // 元素分解 batch（batch = row/m_per），归约遍历全部 rows（含所有
        // batch），与 CPU 端 matmul_out 语义一致 → 正常生成，不跳过。
        // 防御：空指令表且无 matmul/fold 段 = 结构损坏（上游变换丢掉 fold 段
        //   即此形态）→ 跳过而非让逐元素生成器对空表 back() UB 崩溃
        //   （表现为 0xC00000FD 栈崩溃）。正常管线到不了这里。
        if (spec.instrs.empty() && !spec.matmul && !spec.fold)
        {
            std::fprintf(stderr, "[skip] 空指令表且无 matmul/fold 段（结构损坏）: %s\n",
                         nn::expr_spec_key(spec).c_str());
            continue;
        }
        // matmul+归约（注意力结构）：generate_glsl_reduce 支持 Matmul
        // 操作数（内联点积，不物化 (batch*M,N) 中间矩阵），该形态正常生成（不跳过）。
        const std::string key = nn::expr_spec_key(spec);
        const std::string comp_path = out_dir + "/fused_" + key + ".comp";
        const std::string spv_path  = out_dir + "/fused_" + key + ".spv";

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
        {
            std::ofstream f(comp_path);
            if (!f) { std::fprintf(stderr, "[FAIL] 无法写入 %s\n", comp_path.c_str()); return false; }
            f << glsl;
        }
        if (!run_glslc(glslc, comp_path, spv_path))
        {
            std::fprintf(stderr, "[FAIL] glslc 编译 %s 失败\n", key.c_str());
            return false;
        }
        const auto spv = read_spv(spv_path);
        if (spv.empty())
        {
            std::fprintf(stderr, "[FAIL] 读取 %s 失败\n", spv_path.c_str());
            return false;
        }

        H << "inline constexpr std::uint32_t kSpirv_" << key << "[] = {";
        for (std::size_t i = 0; i < spv.size(); ++i)
        {
            if (i % 8 == 0) H << "\n    ";
            H << "0x" << std::hex << spv[i] << "u, ";
        }
        H << std::dec << "\n};\n\n";
    }

    // ── 精度变体（Phase 2 in-kernel f16）：同一结构 + 非零精度签名 → 独立 shader
    // 注册键 = key#sig（expr_prec_sig_key），文件名/标识符用 key_sighex（标识符
    // 不能含 '#'）。生成器不支持该形态（超出当前带类型生成能力）
    // 时返回空串 → **跳过而非失败**：运行时不命中即回退边界 cast，正确性不变。
    //
    // ALU 变体（native16 原生 f16 算术）：谓词 expr_prec_sig_native16 通过时
    // 额外发射一份（键 = key#sig#a，标识符后缀 _a）。与 f32 算术变体并存——
    // 后端按设备 shaderFloat16 能力创建 pipeline，运行时优先命中 #a 键。
    // ALU 变体恒为**标量 kernel**（generate_glsl native16 分支不走 vec4），
    // vec_width 必须写 1（与生成器同源，否则 dispatch 宽度失配 → 静默算 1/4）。
    struct VariantEmit
    {
        const nn::ExprSpec* spec;
        nn::ExprPrecSig sig;
        std::string     vkey;     // "key#sig" 或 "key#sig#a"（注册表键）
        std::string     suffix;   // "key_sighex" 或 "key_sighex_a"（标识符/文件名）
        bool            alu = false;   // native16 变体（vec_width 恒 1）
    };
    std::vector<VariantEmit> emitted_variants;
    // 去重：scan 预测变体与 run-only 回填清单可能重叠 —— 重复
    // 发射会生成重复的 kSpirv_ 标识符与注册行（重定义编译错误）。
    std::unordered_set<std::string> emitted_vkeys;
    // 单个 (spec, sig) 的发射：kind = "f32" | "a"（ALU）
    const auto emit_one = [&](const nn::ExprSpec& spec, nn::ExprPrecSig sig,
                              bool alu) -> bool {
        const int raxis = nn::expr_spec_reduce_axis(spec);
        if (raxis == -2)
            return false;   // 混合归约轴（与基础结构 skip 一致）
        if (spec.instrs.empty() && !spec.matmul && !spec.fold)
            return false;
        if (alu && !nn::expr_prec_sig_native16(spec, sig))
            return false;   // 谓词不通过 → 不发 ALU 变体（f32 算术变体仍覆盖）
        const std::string key = nn::expr_spec_key(spec);
        // 运行期精度分派（V1）：键 = key#x（无签名 hex——精度不再是身份）
        const bool dispv = nn::expr_prec_sig_is_dispatch(sig);
        char sigbuf[16];
        std::snprintf(sigbuf, sizeof(sigbuf), "%04x", static_cast<unsigned>(sig));
        const std::string vkey = dispv
            ? (key + nn::EXPR_PREC_DISPATCH_SUFFIX)
            : (alu ? (key + "#a")   // native16：结构谓词判定 → 键与签名无关
                   : nn::expr_prec_sig_key(key, sig));
        const std::string suffix = dispv
            ? (key + "_x")
            : (alu ? (key + "_a")
                   : (key + "_" + sigbuf));
        if (!emitted_vkeys.insert(vkey).second)
            return false;   // 已发射（scan 预测 ∩ 回填清单重叠）→ 幂等跳过
        const std::string comp_path = out_dir + "/fused_" + suffix + ".comp";
        const std::string spv_path  = out_dir + "/fused_" + suffix + ".spv";
        auto emitter = nn::emitter_registry::make("glsl");
        if (!emitter)
        {
            std::fprintf(stderr, "[FAIL] 无法创建 GLSL emitter（IR-D 注册表异常）\n");
            std::exit(1);
        }
        const std::string glsl = (raxis >= 0)
            ? emitter->generate_reduce("fused_" + suffix, spec, sig)
            : emitter->generate("fused_" + suffix, spec, sig, alu);
        if (glsl.empty())
        {
            if (alu)
                return false;   // ALU 谓词/形态不支持 → 静默跳过（f32 变体已覆盖）
            std::fprintf(stderr, "[skip] 精度变体 %s（sig=%s）暂不支持带类型生成\n",
                         key.c_str(), sigbuf);
            return false;
        }
        {
            std::ofstream f(comp_path);
            if (!f) { std::fprintf(stderr, "[FAIL] 无法写入 %s\n", comp_path.c_str()); std::exit(1); }
            f << glsl;
        }
        if (!run_glslc(glslc, comp_path, spv_path))
        {
            std::fprintf(stderr, "[FAIL] glslc 编译精度变体 %s 失败\n", vkey.c_str());
            std::exit(1);
        }
        const auto spv = read_spv(spv_path);
        if (spv.empty())
        {
            std::fprintf(stderr, "[FAIL] 读取 %s 失败\n", spv_path.c_str());
            std::exit(1);
        }
        H << "inline constexpr std::uint32_t kSpirv_" << suffix << "[] = {";
        for (std::size_t i = 0; i < spv.size(); ++i)
        {
            if (i % 8 == 0) H << "\n    ";
            H << "0x" << std::hex << spv[i] << "u, ";
        }
        H << std::dec << "\n};\n\n";
        emitted_variants.push_back(VariantEmit{&spec, sig, vkey, suffix, alu});
        return true;
    };

    // ── 运行期精度分派变体（V1，键 = key#x）──────────────────────────────
    // **每个结构无条件发射一份**：输入/输出各声明 f32 + float16_t 双视图，
    // 加载/存储处按 push constant `uint prec` 走 uniform 分支 → 一个 shader
    // 覆盖全部 (输入精度位图, 输出精度)。
    //
    // 这是"签名不再需要构建期枚举"的载体：运行时按真实张量精度填 prec，
    // 任何签名都有 shader → **永不 miss**，因此 dry-run / 模型 pass /
    // tools/prec_backfill.txt 全部不再参与生成（A2 起从 scan 移除）。
    //
    // ── native16 变体（V2，键 = key#a）─────────────────────────────────
    // 生成条件 = **结构谓词**（expr_prec_sig_native16 代入"全输入 f16 +
    // 输出 f16"这一结构性常量签名）：无 fold / 无 matmul 段 / 无归约 /
    // 无 rparams / 无常量微值。运行时仅在真实签名确为全 f16 时才选它 →
    // 生成期不需要知道签名，运行期也不需要发现。
    //
    // A/B 逃生阀：NN_SCAN_NO_DISPATCH=1 时不发射 V1（退回旧集合）；
    // 只影响登记集合，不影响语义。
    for (const auto& spec : reg.specs)
    {
        if (!nn_no_dispatch)
            emit_one(spec, nn::EXPR_PREC_SIG_DISPATCH, /*alu=*/false);
        const std::size_t n_in = spec.views.size();
        const std::uint32_t all_in_bits =
            (n_in >= 32) ? 0u : ((1u << n_in) - 1u);
        emit_one(spec, nn::expr_prec_sig_make(all_in_bits, /*out_f16=*/true),
                 /*alu=*/true);   // 谓词不通过时内部静默跳过
    }
    (void)prec_manifest;   // 签名不再回填（V1 覆盖任意签名）

    H << "inline const FusedShader kFusedShaders[] = {\n";
    for (const auto& spec : reg.specs)
    {
        const int raxis = nn::expr_spec_reduce_axis(spec);
        if (raxis == -2)
            continue;  // 与上方跳过保持一致（混合轴）
        if (spec.instrs.empty() && !spec.matmul && !spec.fold)
            continue;  // 结构损坏（同上：生成循环已 skip，元数据保持一致）
        const std::string key = nn::expr_spec_key(spec);
        const std::uint32_t vecw =
            (!spec.fold && raxis < 0 && nn::glsl_vec4_eligible(spec)) ? 4u : 1u;
        // fold 形态恒 vecw=1（每线程一行标量状态机，无 vec4 路径）；fold 判定
        // 走 FusedShader.spec.fold（后端同源判断，不加结构字段）。
        H << "    { \"" << key << "\",\n        " << emit_spec(spec) << ",\n"
          << "        kSpirv_" << key
          << ", sizeof(kSpirv_" << key << ")/sizeof(std::uint32_t), "
          << raxis << ", " << (spec.matmul ? 1 : 0) << ", "
          << nn::expr_spec_runtime_view_param_count(spec) << ", "
          << nn::expr_spec_runtime_param_count(spec) << ", " << vecw
          << ", 0u },\n";
    }
    // 精度变体行（键 = key#sig 或 key#sig#a；元数据与基础结构同源——精度不进结构）
    for (const auto& ve : emitted_variants)
    {
        const nn::ExprSpec& spec = *ve.spec;
        const int raxis = nn::expr_spec_reduce_axis(spec);
        // ALU（native16）变体恒标量 kernel → vec_width 必须 1（与 generate_glsl
        // 的 native16 分支同源）；f32 算术变体维持原 vec4 资格判定。
        const std::uint32_t vecw =
            (!ve.alu && !spec.fold && raxis < 0 && nn::glsl_vec4_eligible(spec))
                ? 4u : 1u;
        H << "    { \"" << ve.vkey << "\",\n        " << emit_spec(spec) << ",\n"
          << "        kSpirv_" << ve.suffix
          << ", sizeof(kSpirv_" << ve.suffix << ")/sizeof(std::uint32_t), "
          << raxis << ", " << (spec.matmul ? 1 : 0) << ", "
          << nn::expr_spec_runtime_view_param_count(spec) << ", "
          << nn::expr_spec_runtime_param_count(spec) << ", " << vecw
          << ", " << ve.sig << "u },\n";
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
    {
        std::ofstream f(reg_path);
        if (!f) { std::fprintf(stderr, "[FAIL] 无法写入 %s\n", reg_path.c_str()); return false; }
        f << H.str();
    }
    std::printf("[gen] %zu 条融合表达式 + %zu 条精度变体 -> %s\n",
                reg.specs.size(), emitted_variants.size(), reg_path.c_str());
    return true;
}

} // namespace nn::tool
