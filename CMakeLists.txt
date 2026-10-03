# CMake 3.30+ required for C++26 standard support (stable in 3.30+)
cmake_minimum_required(VERSION 3.30)

# ── 编译器与标准库：一律使用默认，不由本文件指定 ──────────────────────────
# 编译器：用 CMake 默认探测结果，或调用方显式 -DCMAKE_CXX_COMPILER / CXX 环境变量。
# 标准库：用编译器默认（Linux clang/gcc 默认 libstdc++）。如需 libc++，由调用方
# 通过 -DCMAKE_CXX_FLAGS / -DCMAKE_EXE_LINKER_FLAGS 统一指定（编译与链接必须一致，
# 否则链接期会出现 std 符号 ABI 不匹配）。
# 本文件不再隐式挑选编译器或 stdlib，避免与调用方/CI 的设置冲突。

project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

# ── C++26 标准 ──────────────────────────────────────────────────────────
# CMake 4.3 尚未内置 MSVC 的 CXX26 方言标志映射（VS2026 cl 19.51 也不认
# /std:c++26，C++26 特性走 /std:c++latest），故对 MSVC 用 CMAKE_CXX_STANDARD=23
# 让方言探测通过，再显式传 /std:c++latest。Clang/GCC 走标准 CMAKE_CXX_STANDARD=26。
if(MSVC)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED OFF)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_compile_options(/std:c++latest)
else()
    set(CMAKE_CXX_STANDARD 26)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# ── 默认 Release 模式，可通过 -DCMAKE_BUILD_TYPE=Debug 切换 ────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── 公共编译标志 ─────────────────────────────────────────────────────────
# 项目规范：禁止异常、开启严格警告。按编译器选择对应标志：
# - Clang/GCC：-fno-exceptions -Wall -Wextra -Wpedantic -Werror
# - MSVC：     /W4 /permissive-（严格一致性）；保留 /EHsc（ppltasks.h 需要），
#              不强制 /WX（MSVC 与 clang 警告集不同，代码未按 MSVC 零警告编写）
if(MSVC)
    add_compile_options(/W4 /permissive- /source-charset:utf-8 /execution-charset:utf-8)
else()
    add_compile_options(-fno-exceptions -Wall -Wextra -Wpedantic -Werror)
endif()

# ── 编译器特有选项 ───────────────────────────────────────────────────────
# -fexperimental-library 仅 Clang 识别；GCC 不识别会产生警告
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(-fexperimental-library)
    # f16 矩阵乘法（MatrixT<F16>）无原生 x86 SIMD → 优化器无法向量化是预期行为
    # （§7.2 兼容路径：f32 参考计算 + f16 舍入），抑制 -Wpass-failed 避免 -Werror 下编译失败
    add_compile_options(-Wno-pass-failed)
endif()
# C++ 标准库由编译器默认决定（见文件开头的说明），此处不再显式指定 -stdlib。
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

# ── 按构建类型设置优化标志 ───────────────────────────────────────────────
# Release: -O3 -funroll-loops -march=native（最大性能）
# Debug:   -O0 -g（调试优先）
# 注意：不使用 -ffast-math，因为它会禁用 NaN/Infinity 支持，影响训练稳定性
# MSVC 对应 /fp:fast 同样禁用 NaN 语义（实测：precision_type_test 的
# "NaN != NaN" / "NaN 比较均为 false" 断言在 /fp:fast 下被编译器折叠为
# false → 假失败；/fp:precise 下全绿）→ 按同一铁律用 /fp:precise。
# Clang/GCC 用 -O3 + 向量化（自带 -fno-math-errno/-fno-trapping-math）。
#
# -march=native 会让二进制只能在编译机同代 CPU 上运行，分发/换机可能 SIGILL。
# 默认开启（本地追求最大性能）；发布/CI 构建请用 -DNN_ENABLE_NATIVE=OFF 生成
# 通用可移植基线（见 .github/workflows 的 Configure 步骤）。
option(NN_ENABLE_NATIVE "Enable -march=native (max perf, non-portable binary)" ON)
if(MSVC)
    set(RELEASE_FLAGS /O2 /fp:precise)
    set(DEBUG_FLAGS /Od /Zi)
else()
    if(NN_ENABLE_NATIVE)
        set(RELEASE_FLAGS -O3 -fno-math-errno -fno-trapping-math -funroll-loops -march=native)
    else()
        set(RELEASE_FLAGS -O3 -fno-math-errno -fno-trapping-math -funroll-loops)
    endif()
    set(DEBUG_FLAGS -O0 -g)
endif()

add_compile_options(
    "$<$<CONFIG:Release>:${RELEASE_FLAGS}>"
    "$<$<CONFIG:Debug>:${DEBUG_FLAGS}>"
)

# ── 线程库（Linux 需要 pthread） ─────────────────────────────────────────
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ── std::stacktrace 运行时库探测 (GCC & Clang) ──────────────────────────
# GCC 14+ 和 Clang (使用 libstdc++) 都需要链接实验性库才能使用 std::stacktrace。
# GCC 使用 stdc++exp，Clang 可能需要 stdc++exp 或 stdc++_libbacktrace。
# 探测一次，成功后通过 nn_core INTERFACE 传播给所有目标。

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # ── GCC 探测逻辑 ──────────────────────────────────────────────
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES stdc++exp)
    check_cxx_source_compiles("
        #include <stacktrace>
        int main() { return std::stacktrace::current().empty() ? 1 : 0; }
    " NN_HAS_STDCXXEXP)
    unset(CMAKE_REQUIRED_LIBRARIES)

    if(NN_HAS_STDCXXEXP)
        set(NN_STACKTRACE_LIB stdc++exp)
    else()
        set(CMAKE_REQUIRED_LIBRARIES stdc++_libbacktrace)
        check_cxx_source_compiles("
            #include <stacktrace>
            int main() { return std::stacktrace::current().empty() ? 1 : 0; }
        " NN_HAS_STDCXX_LIBBACKTRACE)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(NN_HAS_STDCXX_LIBBACKTRACE)
            set(NN_STACKTRACE_LIB stdc++_libbacktrace)
        else()
            message(FATAL_ERROR "std::stacktrace 需要 stdc++exp 或 stdc++_libbacktrace")
        endif()
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # ── Clang 探测逻辑 ─────────────────────────────────────────────
    # Clang 可能链接 libstdc++ 或 libc++。我们通过尝试链接来探测。
    # 如果 Clang 使用 libc++ 且其版本尚不支持 stacktrace，则此处会失败。
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES stdc++exp)
    check_cxx_source_compiles("
        #include <stacktrace>
        int main() { return std::stacktrace::current().empty() ? 1 : 0; }
    " NN_CLANG_HAS_STDCXXEXP)

    if(NN_CLANG_HAS_STDCXXEXP)
        set(NN_STACKTRACE_LIB stdc++exp)
    else()
        # 回退尝试旧版 libbacktrace 库
        set(CMAKE_REQUIRED_LIBRARIES stdc++_libbacktrace)
        check_cxx_source_compiles("
            #include <stacktrace>
            int main() { return std::stacktrace::current().empty() ? 1 : 0; }
        " NN_CLANG_HAS_STDCXX_LIBBACKTRACE)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(NN_CLANG_HAS_STDCXX_LIBBACKTRACE)
            set(NN_STACKTRACE_LIB stdc++_libbacktrace)
        else()
            # Clang 可能在使用 libc++，其 stacktrace 支持可能不完整。
            message(WARNING "Clang 无法链接 std::stacktrace。如果使用 libc++，请确保版本 >= 18。CI 构建可能会失败。")
            set(NN_STACKTRACE_LIB "")
        endif()
    endif()
endif()

if(NN_STACKTRACE_LIB)
    message(STATUS "std::stacktrace link library: ${NN_STACKTRACE_LIB}")
else()
    message(STATUS "std::stacktrace link library: (none found,可能链接失败)")
endif()

# ── 公共 INTERFACE 库：所有目标共享 include 目录和链接库 ──────────────
# 通过 INTERFACE 库消除 7 个目标的重复设置
add_library(nn_core INTERFACE)
target_include_directories(nn_core INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(nn_core INTERFACE Threads::Threads)
if(NN_STACKTRACE_LIB)
    target_link_libraries(nn_core INTERFACE ${NN_STACKTRACE_LIB})
endif()
target_include_directories(nn_core INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(nn_core INTERFACE Threads::Threads)

# ── 便捷函数：统一添加可执行目标 ────────────────────────────────────────
function(nn_add_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE nn_core)
endfunction()

# ── 测试/验证目标开关（默认关闭） ──────────────────────────────────────
# 默认只构建应用与基准工具目标，避免生成大量验证程序；
# 需要时用 -DNN_ENABLE_TESTS=ON 开启全部 gradcheck / *_test / *_probe / *_debug。
option(NN_ENABLE_TESTS "Build test/verification executables" OFF)

# 收集实际构建的目标，供 GPU 统一启用（避免引用未定义目标）
set(NN_APP_TARGETS "")
set(NN_TEST_TARGETS "")
set(NN_BENCH_TARGETS "")   # 性能基准已移除（2026 清理：bench 工具不注册 ctest 且无法在 -Werror 下编译）

# ── MNIST 目标（标准构建核心应用） ───────────────────────────────────────
nn_add_executable(mnist_train src/mnist_train.cpp)
nn_add_executable(mnist_infer src/mnist_infer.cpp)
list(APPEND NN_APP_TARGETS mnist_train mnist_infer)

# ── 文本目标（GPT 训练/推理 + BPE 分词器训练/推理，标准构建核心应用） ──
nn_add_executable(text_train src/text_train.cpp)
nn_add_executable(text_infer src/text_infer.cpp)
nn_add_executable(tokenizer_train src/tokenizer_train.cpp)
nn_add_executable(tokenizer_infer src/tokenizer_infer.cpp)
list(APPEND NN_APP_TARGETS text_train text_infer tokenizer_train tokenizer_infer)

# ── 层吞吐量基准（CPU/Vulkan GPU；--all 或 --layer <name> 指定） ────────
nn_add_executable(layer_bench src/layer_bench.cpp)
list(APPEND NN_APP_TARGETS layer_bench)

# ── 运行时显存分项探针（双池采样 + nvidia-smi + 逻辑分项归因） ─────────────
nn_add_executable(mem_probe src/mem_probe.cpp)
list(APPEND NN_APP_TARGETS mem_probe)

# ── CPU f16 组件级诊断探针（A-I 阶段定位发散；NN_F16_DEBUG=1 开逐中间量扫描）──
nn_add_executable(f16_cpu_probe src/f16_cpu_probe.cpp)
list(APPEND NN_APP_TARGETS f16_cpu_probe)

# ── GPU f16 参数写回探针（issue #13 P0-②：MSVC 下权重冻结定位）──────────────
nn_add_executable(f16_writeback_probe src/f16_writeback_probe.cpp)
list(APPEND NN_APP_TARGETS f16_writeback_probe)

# ── CPU/GPU 训练 run-to-run 确定性探针（ComputeEngine Refresh P0 / D9 实测；
#    手动运行不注册 ctest，退出码 0=逐位一致 1=非确定 2=运行错误）──────────
nn_add_executable(gpu_stability_probe src/gpu_stability_probe.cpp)
list(APPEND NN_APP_TARGETS gpu_stability_probe)

# ── 测试 / 验证目标（默认关闭，NN_ENABLE_TESTS=ON 时构建）─────────────
if(NN_ENABLE_TESTS)

    # ══════════════════════════════════════════════════════════════════════
    # 测试目标（19 个，按功能域分组合并）
    # 合并策略：wrapper .cpp 通过 #define main + #include 聚合原测试源文件
    # ══════════════════════════════════════════════════════════════════════

    # ── GPU 集成测试（需要 Vulkan）────────────────────────────────────────
    nn_add_executable(gpu_test src/gpu_test.cpp)
    list(APPEND NN_TEST_TARGETS gpu_test)

    # ── GPU f16 参数写回回归（issue #13 P0-②：MSVC 下权重冻结）───────────
    # 复用 app 目标 f16_writeback_probe（已在 NN_APP_TARGETS 构建）：
    # 断言 in-place f16 写回值正确 + 8 步 Adam 后参数必须移动（防"静默冻结"）
    list(APPEND NN_TEST_TARGETS f16_writeback_probe)

    # ── GPU f16 能力检查 + cast/matmul 验证 ──────────────────────────────
    nn_add_executable(gpu_f16_test src/gpu_f16_test.cpp)
    list(APPEND NN_TEST_TARGETS gpu_f16_test)

    # ── 多精度类型系统（f16 转换 / 舍入 / 张量精度 / f16 计算）──────────
    nn_add_executable(precision_test src/precision_test.cpp)
    list(APPEND NN_TEST_TARGETS precision_test)

    # ── 多精度适配层（f16 存储真正生效：PrecisionEngine + DSL + 端到端）──
    nn_add_executable(f16_precision_test src/f16_precision_test.cpp)
    list(APPEND NN_TEST_TARGETS f16_precision_test)

    # ── 分词器正确性（BPE pre_tokenize 对拍 + 标记往返）─────────────────
    nn_add_executable(tokenizer_test src/tokenizer_test.cpp)
    list(APPEND NN_TEST_TARGETS tokenizer_test)

    # ── Attention（梯度检查 + forward 一致性 + batch 行号回归）──────────
    nn_add_executable(attn_test src/attn_test.cpp)
    list(APPEND NN_TEST_TARGETS attn_test)

    # ── 基础层数值梯度检查（SwiGLU / RMSNorm / Softmax）─────────────────
    nn_add_executable(layer_gradcheck_test src/layer_gradcheck_test.cpp)
    list(APPEND NN_TEST_TARGETS layer_gradcheck_test)

    # ── BatchNorm（工厂不回落 + 训练/推理双态 fwd/bwd + running EMA
    #    + Model::set_training 转发 + save/load 往返逐位一致）────────────
    nn_add_executable(batchnorm_test src/batchnorm_test.cpp)
    list(APPEND NN_TEST_TARGETS batchnorm_test)

    # ── CNN（MaxPool2D 独立参考比对 + 缓存/checkpoint 契约 + 端到端冒烟）──
    nn_add_executable(cnn_test src/cnn_test.cpp)
    list(APPEND NN_TEST_TARGETS cnn_test)

    # ── GPT 模型（整链梯度检查 + 梯度检查点一致性）──────────────────────
    nn_add_executable(gpt_test src/gpt_test.cpp)
    list(APPEND NN_TEST_TARGETS gpt_test)

    # ── Activation Offload（原语往返 + GPT/RAPT 模型级一致性）──────────
    # 模型级 offload 一致性（GPU-only：无 Vulkan 构建时自身返回 77 = SKIP）
    nn_add_executable(gpt_offload_test src/gpt_offload_test.cpp)
    list(APPEND NN_TEST_TARGETS gpt_offload_test)
    nn_add_executable(rapt_offload_test src/rapt_offload_test.cpp)
    list(APPEND NN_TEST_TARGETS rapt_offload_test)

    # ── 文档感知注意力（端到端不变性 + 灵敏度/跨样本隔离对照；掩码构建
    #    单测已随 build_attention_mask 删除）────────────────────────────────
    nn_add_executable(doc_attn_test src/doc_attn_test.cpp)
    list(APPEND NN_TEST_TARGETS doc_attn_test)

    # ── ZiPT / AttnZip 已于 2026-10-01 移除（保留在 legacy/zipt 分支）──
    #     恢复前提见 docs/history.md「ZiPT 移除」条。

    # ── RAPT / ReLU 线性注意力（梯度 + 冒烟 + 梯度检查点）──────────────
    nn_add_executable(rapt_test src/rapt_test.cpp)
    list(APPEND NN_TEST_TARGETS rapt_test)

    # ── 表达式/IR 纯 CPU（DSL + 归约 + matmul + 优化 pass + 图 IR）────
    nn_add_executable(expr_cpu_test src/expr_cpu_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_cpu_test)

    # ── 表达式/IR GPU（图融合端到端 + AOT shader 验证 + 表达式求值）────
    nn_add_executable(expr_gpu_test src/expr_gpu_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_gpu_test)

    # ── 融合 shader **逐形态** GPU 对拍（rope/swiglu/gelu/softmax/matmul/
    #    matmul+reduce/norm/reduce_consts/fold v1/fold attn/回退）──────────
    # ⚠ 此前该文件**未被任何 CMake 目标引用**（未编译、未注册）——它是融合
    #    shader 逐形态的回归面，重新接入。
    # 覆盖口径：**V0（全 f32）**逐形态对拍（该文件无 f16/精度引用）。
    # V1（`key#x` 运行期精度分派）的覆盖在 `f16_precision_test`（NN_PREC_TRACE
    # 实测 20 条 #x / 4 条 #a）；本测试不覆盖 V1。
    nn_add_executable(fused_gpu_test src/fused_gpu_test.cpp)
    list(APPEND NN_TEST_TARGETS fused_gpu_test)

    # ── ModelSpec 架构校验（load_model 文件头 vs 模型规格比对）──────────
    nn_add_executable(model_spec_validation_test src/model_spec_validation_test.cpp)
    list(APPEND NN_TEST_TARGETS model_spec_validation_test)

    # ── 稀疏交叉熵融合（loss/grad vs 参考，CPU+GPU）─────────────────────
    nn_add_executable(ce_fusion_test src/ce_fusion_test.cpp)
    list(APPEND NN_TEST_TARGETS ce_fusion_test)

endif()

# ── 测试目标统一处理：输出到 <默认可执行目录>/test，并注册到 ctest ──────────
# 单一开关 NN_ENABLE_TESTS 同时负责「编译测试程序」与「注册 ctest 测试」：
#   编译：上面 if(NN_ENABLE_TESTS) 内的 nn_add_executable(...)
#   注册：这里对 NN_TEST_TARGETS 逐一 add_test
# 可执行文件位置不硬编码生成器差异，统一设为 ${CMAKE_CURRENT_BINARY_DIR}/test，
# 由 CMake 自动适配：
#   - 单配置生成器（Ninja / Make）：默认可执行目录为 build 根目录 → build/test
#   - 多配置生成器（MSVC）：CMake 自动追加配置子目录 → build/test/Release(或 Debug)
# add_test 直接传目标名，CTest 会按当前配置自动解析出真实可执行文件路径。
if(NN_ENABLE_TESTS)
    enable_testing()
    foreach(TARGET ${NN_TEST_TARGETS})
        set_target_properties(${TARGET} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test)
        add_test(NAME ${TARGET} COMMAND ${TARGET})
        # 退出码 77 = 依约定"跳过"（如纯 CPU 构建下需要 Vulkan 的测试）：
        # 记 Skipped 而非 FAILED，否则 CPU-only 构建/CI 出现假红。
        set_tests_properties(${TARGET} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()
    # CNN 的 GPU 变体：DSL 表达式校验只在 GPU/scan 路径发生（CPU 模板路径不校验），
    # 默认的纯 CPU cnn_test 测不出"只在 GPU 触发"的表达式结构回归
    # （如 MaxPool2D 裸视图根 → 空指令表）。无 Vulkan 构建下返回 77 = SKIP。
    if(TARGET cnn_test)
        add_test(NAME cnn_test_gpu COMMAND cnn_test --gpu)
        set_tests_properties(cnn_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # BatchNorm 的 GPU 变体：推理态/EMA 表达式的 ExprSpec 校验只在 GPU
    # 路径发生（CPU 模板路径不校验），且闭合世界 miss 只在 GPU 运行期暴露。
    # 无 Vulkan 构建下返回 77 = SKIP。
    if(TARGET batchnorm_test)
        add_test(NAME batchnorm_test_gpu COMMAND batchnorm_test --gpu)
        set_tests_properties(batchnorm_test_gpu PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # 基准工具已移除（2026 清理：无法在 -Werror 下编译，且无 ctest 覆盖）
    foreach(TARGET ${NN_BENCH_TARGETS})
        set_target_properties(${TARGET} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test)
    endforeach()
endif()

# ── Vulkan GPU 加速（可选） ────────────────────────────────────────────────
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    message(STATUS "Vulkan found: ${Vulkan_LIBRARY}")

    # 查找 SPIR-V 编译器（Vulkan SDK 自带）
    find_program(GLSLC glslc HINTS ${Vulkan_GLSLC_EXECUTABLE})

    # ── 铁律：原语 shader 永不含任何算法 ──────────────────────────────
    # 原语 shader 只做 op-level 计算（矩阵乘、逐元素、归约、广播、搬移）。
    # ReLU/GeLU/Softmax/LayerNorm/Attention/RoPE/SwiGLU 等算法一律来自
    # Layer（组合原语）或 DSL（nn::dsl，见下），绝不写进原语 shader。
    set(GPU_SHADERS
        matmul
        matmul_tiled
        batched_matmul
        rearrange_3d
        elementwise_v2
        transpose
        gather
        scatter_add
        im2col
        col2im
        group_reduce
        scan_prefix_outer
        scan_suffix_outer
        scan_prefix_outer_gen
        scan_suffix_outer_gen
        outer_col
        cast)
    # 融合 shader：构建期从 Layer 内联表达式自动合成（**单步**）──────────
    # 表达式**文本只出现在 Layer**（compute / compute_reduce / compute_into）。
    # scan_exprs 同进程内完成：收集折叠出的 ExprSpec 结构（锚点自登记 + dry-run
    # + 模型 pass + 显式登记，按 key 去重）→ glsl_gen → glslc → 内联 SPIR-V
    # → fused_registry.hpp（key=expr_spec_key[#x|#a] → 融合 shader）。
    # 不经 .bin 中间序列化（原独立工具 gen_fused 已并入 tools/fused_generate.hpp）。
    # 运行时按 key 精确匹配 dispatch；未命中硬报错（闭合世界）。
    set(GPU_SHADER_HPPS "")

    # 便捷函数：glslc → .spv → embed_spirv.cmake → C++ 头文件
    # 可选附加 glslc 参数走 ARGN（如 --target-env=...；不传则保持默认，
    # 现有 shader 的编译参数完全不变）
    function(nn_embed_shader SHADER_NAME SRC_FILE)
        set(SPV ${CMAKE_BINARY_DIR}/${SHADER_NAME}.spv)
        set(HPP ${CMAKE_BINARY_DIR}/generated/${SHADER_NAME}_spv.hpp)
        set(FUNC "nn_${SHADER_NAME}_spirv_bytecode")
        set(GPU_SHADER_SPVS ${GPU_SHADER_SPVS} ${SPV} PARENT_SCOPE)
        set(GPU_SHADER_HPPS ${GPU_SHADER_HPPS} ${HPP} PARENT_SCOPE)

        # 编译 GLSL → SPIR-V
        add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${GLSLC} ${ARGN} -o ${SPV} ${SRC_FILE}
            DEPENDS ${SRC_FILE}
            COMMENT "[GPU] Compiling ${SHADER_NAME}.comp → ${SHADER_NAME}.spv"
        )

        # 嵌入 SPIR-V → C++ 头文件
        add_custom_command(
            OUTPUT ${HPP}
            COMMAND ${CMAKE_COMMAND}
                -DSPV_INPUT=${SPV}
                -DHPP_OUTPUT=${HPP}
                -DFUNC_NAME=${FUNC}
                -P ${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake
            DEPENDS ${SPV} ${CMAKE_SOURCE_DIR}/cmake/embed_spirv.cmake
            COMMENT "[GPU] Embedding ${SHADER_NAME} SPIR-V into C++ header"
        )
    endfunction()

    # 1. 普通（手写）纯原语 shader
    foreach(SHADER_NAME ${GPU_SHADERS})
        nn_embed_shader(${SHADER_NAME} ${CMAKE_SOURCE_DIR}/shaders/${SHADER_NAME}.comp)
    endforeach()

    # 1b. 小 N GEMV（matmul_gemv）：用 subgroup 归约（GL_KHR_shader_subgroup_*）
    #     需 SPIR-V ≥1.3 → 显式 --target-env=vulkan1.2（与 gen_fused 一致）；
    #     其余 shader 编译参数不变。
    nn_embed_shader(matmul_gemv ${CMAKE_SOURCE_DIR}/shaders/matmul_gemv.comp --target-env=vulkan1.2)

    # 1c. reduce：行列归约用 subgroup 蝶式（GL_KHR_shader_subgroup_*，与
    #     DSL 融合归约同配方）→ 同样需 SPIR-V ≥1.3 → --target-env=vulkan1.2
    nn_embed_shader(reduce ${CMAKE_SOURCE_DIR}/shaders/reduce.comp --target-env=vulkan1.2)

    # 1d. 精度变体（Phase 2 in-kernel f16）：**同一份原语 shader** 用
    #     -DNN_SHADER_F16=1 编出第二份 SPIR-V（缓冲声明 float16_t + 别名槽
    #     uvec2/unpackHalf2x16，f32 累加，f16 写出）。后端按操作数精度选
    #     pipeline；设备无 storageBuffer16BitAccess 时不创建（回退 f32 边界
    #     cast，正确性不变）。f32 分支文本逐字未改 → 零回归。
    nn_embed_shader(batched_matmul_f16 ${CMAKE_SOURCE_DIR}/shaders/batched_matmul.comp -DNN_SHADER_F16=1)
    nn_embed_shader(matmul_tiled_f16 ${CMAKE_SOURCE_DIR}/shaders/matmul_tiled.comp -DNN_SHADER_F16=1)
    # C2e：小 N GEMV 的 f16 变体（n_out≤8 推理热路径；subgroup → vulkan1.2）
    nn_embed_shader(matmul_gemv_f16 ${CMAKE_SOURCE_DIR}/shaders/matmul_gemv.comp --target-env=vulkan1.2 -DNN_SHADER_F16=1)
    # C2f：RLA 扫描原语的 f16 存储变体（输入输出均 half；输出 (·,seq) 全
    # 尺寸——消灭适配层 scan 的全尺寸 f32 副本；scratch St 仍 f32）。
    nn_embed_shader(scan_prefix_outer_f16 ${CMAKE_SOURCE_DIR}/shaders/scan_prefix_outer.comp -DNN_SHADER_F16=1)
    nn_embed_shader(scan_suffix_outer_f16 ${CMAKE_SOURCE_DIR}/shaders/scan_suffix_outer.comp -DNN_SHADER_F16=1)
    nn_embed_shader(scan_prefix_outer_gen_f16 ${CMAKE_SOURCE_DIR}/shaders/scan_prefix_outer_gen.comp -DNN_SHADER_F16=1)
    nn_embed_shader(scan_suffix_outer_gen_f16 ${CMAKE_SOURCE_DIR}/shaders/scan_suffix_outer_gen.comp -DNN_SHADER_F16=1)
    nn_embed_shader(outer_col_f16 ${CMAKE_SOURCE_DIR}/shaders/outer_col.comp -DNN_SHADER_F16=1)

    # 1e. Phase C1 op-level f16 变体（消灭边界 cast 的搬运大头）：
    #     transpose / rearrange_3d / gather 的 f16 存储版（纯位拷贝零算术）。
    #     - gather：indices 保持 float（行号整数值 > f16 精度，禁转）；
    #     - transpose：shared float16_t → SPIR-V Float16 capability →
    #       运行时要求 shaderFloat16（否则后端跳过该 pipeline，回退边界 cast）；
    #     - rearrange_3d：仅 StorageBuffer16BitAccess。
    #     与 matmul 同款：f32 分支逐字未改 → 零回归。
    nn_embed_shader(transpose_f16 ${CMAKE_SOURCE_DIR}/shaders/transpose.comp -DNN_SHADER_F16=1)
    nn_embed_shader(rearrange_3d_f16 ${CMAKE_SOURCE_DIR}/shaders/rearrange_3d.comp -DNN_SHADER_F16=1)
    nn_embed_shader(gather_f16 ${CMAKE_SOURCE_DIR}/shaders/gather.comp -DNN_SHADER_F16=1)
    # C1b：add_inplace/scale_inplace/accumulate 的 f16 存储变体
    # （归因表 (256,8192)/(64,8192)/(2048,256) inplace 大头；仅需16位存储）
    nn_embed_shader(elementwise_v2_f16 ${CMAKE_SOURCE_DIR}/shaders/elementwise_v2.comp -DNN_SHADER_F16=1)
    # C2：行列归约的 f16 存储变体（归因 L756 大头：Linear::backward grad_bias
    # 直调 row_reduce_sum 的 (64,8192)×38 + (256,8192)×8 = 140MB；输入直读
    # f16、输出归约向量保持 float）。subgroup → 同 reduce 需 vulkan1.2。
    nn_embed_shader(reduce_f16 ${CMAKE_SOURCE_DIR}/shaders/reduce.comp --target-env=vulkan1.2 -DNN_SHADER_F16=1)
    # C2b：embedding 梯度原子累加的 f16 存储变体（dst 打包 half CAS、
    # grad half 直读、indices 保持 float）——归因 L518/521 (8208,64)×2 +
    # (8192,64)×4；仅需 16bit 存储（CAS 对 uint word，无 f16 原子）。
    nn_embed_shader(scatter_add_f16 ${CMAKE_SOURCE_DIR}/shaders/scatter_add.comp -DNN_SHADER_F16=1)
    # C2c：分组归约的 f16 存储变体（CNN 池化路径：输入 half 直读、输出
    # float——调用方 P=F32；仅需 16bit 存储）
    nn_embed_shader(group_reduce_f16 ${CMAKE_SOURCE_DIR}/shaders/group_reduce.comp -DNN_SHADER_F16=1)
    # C2d：卷积窗口展开/伴随散射的 f16 存储变体（数据搬运，in/out 均 half）
    nn_embed_shader(im2col_f16 ${CMAKE_SOURCE_DIR}/shaders/im2col.comp -DNN_SHADER_F16=1)
    nn_embed_shader(col2im_f16 ${CMAKE_SOURCE_DIR}/shaders/col2im.comp -DNN_SHADER_F16=1)

    # 2. AOT 融合 shader：scan_exprs（**单步** = 收集 + 生成）
    #    收集：FusedAnchor 自登记（按表达式类型）+ dry-run + 模型 pass
    #    生成：expr_emitter（默认 GLSL）→ glslc → 内联 SPIR-V → fused_registry.hpp
    #    原 gen_fused 已并入 scan_exprs 的生成阶段（tools/fused_generate.hpp），
    #    不再经 expr_specs.bin 中间文件。
    set(NN_FUSED_REGISTRY_HPP ${CMAKE_BINARY_DIR}/generated/fused_registry.hpp)

    set(SCAN_SRC      ${CMAKE_SOURCE_DIR}/tools/scan_exprs.cpp)
    set(FUSED_GEN_SRC ${CMAKE_SOURCE_DIR}/tools/fused_generate.hpp)
    add_executable(scan_exprs ${SCAN_SRC})
    target_include_directories(scan_exprs PRIVATE ${CMAKE_SOURCE_DIR}/include/neuralnet.cpp)
    target_link_libraries(scan_exprs PRIVATE nn_core Vulkan::Vulkan)
    add_custom_command(
        OUTPUT ${NN_FUSED_REGISTRY_HPP}
        # scan_exprs 是原生可执行文件，直接写文件、不会自建目录。
        # Ninja 会自动创建自定义命令的输出目录，但 Makefiles 生成器不会，
        # 全新构建时 generated/ 尚未存在会写入失败（见 08-pitfalls）。
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
        COMMAND scan_exprs ${CMAKE_BINARY_DIR}/generated ${GLSLC}
        DEPENDS scan_exprs ${SCAN_SRC} ${FUSED_GEN_SRC}
        COMMENT "[GPU] Scan + synthesize fused shaders (scan_exprs) → fused_registry.hpp"
    )

    list(APPEND GPU_SHADER_HPPS ${NN_FUSED_REGISTRY_HPP})

    # ── 库自带的融合能力：为**任意目标**收集 + 生成它自己的 fused_registry.hpp ──
    # 用法：nn_enable_gpu_fusion(<target> MAIN <含 main 的源文件> [OUT_DIR <目录>])
    #
    # 语义（对应"写 dsl::compute 之后只跑构建即可"）：
    #   收集器 = 库内收集逻辑（`tools/scan_exprs.cpp`：锚点自登记 + dry-run +
    #   模型 pass + 显式登记）+ **本目标的全部 TU**（用 -DNN_EXPR_SCAN 编，
    #   让使用者的 `dsl::compute` 调用点在静态初始化期自登记）。
    #   产物目录只作为该目标的私有 include 目录——"`fused_registry.hpp` 在
    #   include path 上即生效"这一既有机制不变。
    #
    # ⚠ 收集器**必须保留库内收集逻辑**：实测 84 个结构里只有 13 个来自
    #   编译期锚点，71 个来自 dry-run/模型 pass（见 docs/history.md 与
    #   docs/development/02）。纯锚点收集器会漏掉库自身的结构。
    function(nn_enable_gpu_fusion target)
        cmake_parse_arguments(NNF "" "MAIN;OUT_DIR" "" ${ARGN})
        if(NOT TARGET ${target})
            message(FATAL_ERROR "nn_enable_gpu_fusion: 目标 ${target} 不存在")
        endif()
        if(TARGET ${target}_nnfusion_collect)
            message(FATAL_ERROR
                "nn_enable_gpu_fusion: 目标 ${target} 已调用过（${target}_nnfusion_collect 已存在）")
        endif()
        if(NOT NNF_OUT_DIR)
            set(NNF_OUT_DIR "${CMAKE_BINARY_DIR}/nn_fusion/${target}")
        endif()
        if(NNF_MAIN)
            get_filename_component(NNF_MAIN "${NNF_MAIN}" ABSOLUTE)
        endif()
        get_target_property(_nnf_srcs ${target} SOURCES)
        set(_nnf_inputs "")
        set(_nnf_main_found FALSE)
        foreach(_s IN LISTS _nnf_srcs)
            if(_s MATCHES "^\\$<")          # 生成器表达式源（跳过，罕见）
                continue()
            endif()
            get_filename_component(_abs "${_s}" ABSOLUTE)
            if(NNF_MAIN AND _abs STREQUAL NNF_MAIN)
                set(_nnf_main_found TRUE)
                continue()                 # 收集器自带 main（SCAN_SRC），排除目标的
            endif()
            list(APPEND _nnf_inputs "${_abs}")
        endforeach()
        # MAIN 拼写/路径不一致会让目标的 main 进入收集器 → 链接期 duplicate main。
        # 静默通过不了（链接会报），但报错点离根因很远，这里提前拦住。
        if(NNF_MAIN AND NOT _nnf_main_found)
            message(FATAL_ERROR
                "nn_enable_gpu_fusion: MAIN=${NNF_MAIN} 不在 ${target} 的源文件列表里")
        endif()
        set(_nnf_reg "${NNF_OUT_DIR}/fused_registry.hpp")
        add_executable(${target}_nnfusion_collect ${SCAN_SRC} ${_nnf_inputs})
        target_include_directories(${target}_nnfusion_collect
            PRIVATE ${CMAKE_SOURCE_DIR}/include/neuralnet.cpp)
        # 收集器必须按**与目标相同的编译条件**编译其 TU：使用者的层代码常在
        # `#ifdef NN_HAS_VULKAN` / 私有 include / 自定义宏之内，条件缺失会让那些
        # 调用点静默消失 → 收集不到 → GPU 运行期闭合世界硬报错（离根因很远）。
        target_include_directories(${target}_nnfusion_collect PRIVATE
            $<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>)
        target_compile_options(${target}_nnfusion_collect PRIVATE
            $<TARGET_PROPERTY:${target},COMPILE_OPTIONS>)
        target_compile_definitions(${target}_nnfusion_collect PRIVATE
            NN_EXPR_SCAN $<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>)
        target_link_libraries(${target}_nnfusion_collect PRIVATE nn_core Vulkan::Vulkan)
        add_custom_command(
            OUTPUT ${_nnf_reg}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${NNF_OUT_DIR}
            COMMAND ${target}_nnfusion_collect ${NNF_OUT_DIR} ${GLSLC}
            DEPENDS ${target}_nnfusion_collect ${SCAN_SRC} ${FUSED_GEN_SRC} ${_nnf_inputs}
            COMMENT "[nn] 为 ${target} 收集 + 合成融合 shader → ${_nnf_reg}")
        add_custom_target(${target}_nnfusion_registry DEPENDS ${_nnf_reg})
        # ⚠ include 顺序有意义：本目标自己的 `fused_registry.hpp` 必须排在
        #   `${CMAKE_BINARY_DIR}/generated` **之前**——后者含库自身的注册表
        #   （不含使用者的自定义结构）与手写原语 shader 的嵌入头。放在前面
        #   才能让 `__has_include("fused_registry.hpp")`（见
        #   backend/compute_vk_backend.hpp）解析到本目标专属的那一份。
        # **必须用 BEFORE**：CMake 的 INCLUDE_DIRECTORIES 属性对重复项保留
        # 首次出现的位置，若目标此前已（由 GPU 统一启用循环）拿到
        # `generated`，普通 append 会把它排在前面，顺序语义反而失效。
        target_include_directories(${target} BEFORE PRIVATE
            ${NNF_OUT_DIR} ${CMAKE_BINARY_DIR}/generated)
        add_dependencies(${target} ${target}_nnfusion_registry compile_gpu_shaders)
    endfunction()

    if(NOT GLSLC)
        message(WARNING "No SPIR-V compiler found (glslc). "
                        "GPU acceleration disabled. Install Vulkan SDK.")
        set(Vulkan_FOUND OFF)
    endif()

    if(Vulkan_FOUND)
        add_custom_target(compile_gpu_shaders DEPENDS ${GPU_SHADER_HPPS})

        # 为所有实际构建的目标启用 Vulkan（测试目标按 NN_ENABLE_TESTS 决定）
        set(NN_GPU_TARGETS ${NN_APP_TARGETS})
        if(NN_ENABLE_TESTS)
            list(APPEND NN_GPU_TARGETS ${NN_TEST_TARGETS})
        endif()

        foreach(TARGET ${NN_GPU_TARGETS})
            target_compile_definitions(${TARGET} PRIVATE NN_HAS_VULKAN)
            target_include_directories(${TARGET} PRIVATE ${CMAKE_BINARY_DIR}/generated)
            target_link_libraries(${TARGET} PRIVATE Vulkan::Vulkan)
            add_dependencies(${TARGET} compile_gpu_shaders)
        endforeach()

        # ── A2 验收样例：一个「库外使用者」形态的目标 ──────────────────────
        # 它有自己的自研融合层（examples/fusion_custom_layer.hpp），**不使用**
        # 库内 build/generated 的注册表，而是靠 nn_enable_gpu_fusion 生成自己
        # 的那一份 —— 证明"写 dsl::compute + 一行 CMake 即可"。
        # 只在 NN_ENABLE_TESTS 下构建：它的收集器会为 ~85 个库内结构各跑一遍
        # glslc（三份变体），是纯验收成本，不该加到默认构建上。
        if(NN_ENABLE_TESTS)
            nn_add_executable(fusion_custom_layer_example
                examples/fusion_custom_layer_main.cpp
                examples/fusion_custom_layer.cpp)
            target_compile_definitions(fusion_custom_layer_example PRIVATE NN_HAS_VULKAN)
            target_link_libraries(fusion_custom_layer_example PRIVATE Vulkan::Vulkan)
            nn_enable_gpu_fusion(fusion_custom_layer_example
                MAIN ${CMAKE_SOURCE_DIR}/examples/fusion_custom_layer_main.cpp)

            # 注册为**常驻 ctest**（注册循环在 Vulkan 探测之前跑，故此处手工注册，
            # 口径与循环一致：输出目录 + SKIP_RETURN_CODE 77 = 无设备时 skip）。
            # 这是"库外使用者写 dsl::compute + 一行 CMake → GPU 能跑"的唯一自动
            # 门禁（含 f16 运行期精度分派 `key#x` 路径与归一化形态）。
            set_target_properties(fusion_custom_layer_example PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test)
            add_test(NAME fusion_custom_layer_example
                     COMMAND fusion_custom_layer_example)
            set_tests_properties(fusion_custom_layer_example
                PROPERTIES SKIP_RETURN_CODE 77)
        endif()

        message(STATUS "GPU acceleration enabled (Vulkan compute)")
    endif()
else()
    message(STATUS "Vulkan not found. GPU acceleration disabled.")
endif()