// ───────────────────────────────────────────────────────────────────────────
//  gen_fused.cpp — AOT 算子融合生成器（构建期运行）
//
//  遍历 fused_exprs.hpp 的 kGenInstances（表达式单一事实来源），用
//  glsl_gen.hpp 把每个 ExprSpec 展开为单个融合 .comp 文件。
//  之后由 CMake 走 glslc → .spv → embed_spirv.cmake 嵌入运行时。
//
//  用法： gen_fused <out_dir>
//  产物： <out_dir>/<name>.comp
// ───────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "fused_exprs.hpp"
#include "glsl_gen.hpp"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::fprintf(stderr, "用法: gen_fused <out_dir>\n");
        return 2;
    }
    const std::string out_dir = argv[1];
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

    int fail = 0;
    for (const auto& inst : nn::fused::kGenInstances)
    {
        const auto spec = nn::fused::make_fused(inst);
        const auto glsl = nn::generate_glsl(inst.name, spec);
        if (glsl.empty())
        {
            std::fprintf(stderr, "[FAIL] %s: 生成失败\n", inst.name);
            ++fail;
            continue;
        }
        const std::string path = out_dir + "/" + inst.name + ".comp";
        std::ofstream f(path);
        if (!f)
        {
            std::fprintf(stderr, "[FAIL] %s: 无法写入 %s\n", inst.name, path.c_str());
            ++fail;
            continue;
        }
        f << glsl;
        std::printf("[gen] %s\n", path.c_str());
    }
    return fail == 0 ? 0 : 1;
}
