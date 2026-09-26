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

# ── 公共 INTERFACE 库：所有目标共享 include 目录和链接库 ──────────────
# 通过 INTERFACE 库消除 7 个目标的重复设置
add_library(nn_core INTERFACE)
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

# ── 测试 / 验证目标（默认关闭，NN_ENABLE_TESTS=ON 时构建）─────────────
if(NN_ENABLE_TESTS)

    # ══════════════════════════════════════════════════════════════════════
    # 测试目标（18 个，按功能域分组合并；原 35 个独立目标已精简）
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

    # ── ZiPT / AttnZip（梯度 + 冒烟 + 一致性 + 文档掩码）────────────────
    nn_add_executable(zipt_test src/zipt_test.cpp)
    list(APPEND NN_TEST_TARGETS zipt_test)

    # ── RAPT / ReLU 线性注意力（梯度 + 冒烟 + 梯度检查点）──────────────
    nn_add_executable(rapt_test src/rapt_test.cpp)
    list(APPEND NN_TEST_TARGETS rapt_test)

    # ── 表达式/IR 纯 CPU（DSL + 归约 + matmul + 优化 pass + 图 IR）────
    nn_add_executable(expr_cpu_test src/expr_cpu_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_cpu_test)

    # ── 表达式/IR GPU（图融合端到端 + AOT shader 验证 + 表达式求值）────
    nn_add_executable(expr_gpu_test src/expr_gpu_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_gpu_test)

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
    # ── 融合 shader：构建期从 Layer 内联表达式自动合成 ───────────────
    # 表达式**文本只出现在 Layer**（compute / compute_reduce / compute_into）。
    # 构建期两步：
    #   ① scan_exprs：dry-run 跑 Layer 的 forward/backward，收集折叠出的
    #      ExprSpec 结构（去重）→ expr_specs.bin（派生物，非手写定义）
    #   ② gen_fused：读 bin → glsl_gen → glslc → 内联 SPIR-V
    #      → fused_registry.hpp（key=expr_spec_key → 融合 shader）
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

    # 2. AOT 融合 shader：scan_exprs + gen_fused
    set(NN_EXPR_SCAN_BIN     ${CMAKE_BINARY_DIR}/generated/expr_specs.bin)
    set(NN_FUSED_REGISTRY_HPP ${CMAKE_BINARY_DIR}/generated/fused_registry.hpp)

    # 2a. scan_exprs：dry-run Layer 收集内联表达式结构 → expr_specs.bin
    set(SCAN_SRC ${CMAKE_SOURCE_DIR}/tools/scan_exprs.cpp)
    add_executable(scan_exprs ${SCAN_SRC})
    target_include_directories(scan_exprs PRIVATE ${CMAKE_SOURCE_DIR}/include/neuralnet.cpp)
    target_link_libraries(scan_exprs PRIVATE nn_core Vulkan::Vulkan)
    add_custom_command(
        OUTPUT ${NN_EXPR_SCAN_BIN}
        # scan_exprs 是原生可执行文件，直接写文件、不会自建目录。
        # Ninja 会自动创建自定义命令的输出目录，但 Makefiles 生成器不会，
        # 全新构建时 generated/ 尚未存在会写入失败（见 08-pitfalls）。
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
        COMMAND scan_exprs ${NN_EXPR_SCAN_BIN}
        DEPENDS scan_exprs ${SCAN_SRC}
        COMMENT "[GPU] Scanning Layer expressions (scan_exprs) → expr_specs.bin"
    )

    # 2b. gen_fused：读 bin → glsl_gen → glslc → 内联 SPIR-V → fused_registry.hpp
    set(FUSED_GEN_SRC ${CMAKE_SOURCE_DIR}/tools/gen_fused.cpp)
    add_executable(gen_fused ${FUSED_GEN_SRC})
    target_include_directories(gen_fused PRIVATE ${CMAKE_SOURCE_DIR}/include/neuralnet.cpp)
    target_link_libraries(gen_fused PRIVATE nn_core)
    add_custom_command(
        OUTPUT ${NN_FUSED_REGISTRY_HPP}
        # 同 scan_exprs：gen_fused 也是原生可执行文件，需先确保输出目录存在
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/generated
        COMMAND gen_fused ${CMAKE_BINARY_DIR}/generated ${GLSLC} ${NN_EXPR_SCAN_BIN}
        DEPENDS gen_fused ${FUSED_GEN_SRC} ${NN_EXPR_SCAN_BIN}
        COMMENT "[GPU] Synthesizing fused shaders (gen_fused) → fused_registry.hpp"
    )

    list(APPEND GPU_SHADER_HPPS ${NN_FUSED_REGISTRY_HPP})

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

        message(STATUS "GPU acceleration enabled (Vulkan compute)")
    endif()
else()
    message(STATUS "Vulkan not found. GPU acceleration disabled.")
endif()