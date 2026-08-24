# CMake 3.30+ required for C++26 standard support (stable in 3.30+)
cmake_minimum_required(VERSION 3.30)
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
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-fno-exceptions -Wall -Wextra -Wpedantic -Werror)
endif()

# ── 编译器特有选项 ───────────────────────────────────────────────────────
# -fexperimental-library 仅 Clang 识别；GCC 不识别会产生警告
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(-fexperimental-library)
endif()

# ── 按构建类型设置优化标志 ───────────────────────────────────────────────
# Release: -O3 -funroll-loops -march=native（最大性能）
# Debug:   -O0 -g（调试优先）
# 注意：不使用 -ffast-math，因为它会禁用 NaN/Infinity 支持，影响训练稳定性
# MSVC 用 /O2 /fp:fast（近似 -O3 + 宽松浮点）；Clang/GCC 用 -O3 + 向量化。
#
# -march=native 会让二进制只能在编译机同代 CPU 上运行，分发/换机可能 SIGILL。
# 默认开启（本地追求最大性能）；发布/CI 构建请用 -DNN_ENABLE_NATIVE=OFF 生成
# 通用可移植基线（见 .github/workflows 的 Configure 步骤）。
option(NN_ENABLE_NATIVE "Enable -march=native (max perf, non-portable binary)" ON)
if(MSVC)
    set(RELEASE_FLAGS /O2 /fp:fast)
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

# 收集实际构建的目标，供 GPU/CUDA 统一启用（避免引用未定义目标）
set(NN_APP_TARGETS "")
set(NN_TEST_TARGETS "")

# 收集实际构建的目标，供 GPU/CUDA 统一启用（避免引用未定义目标）
set(NN_APP_TARGETS "")
set(NN_TEST_TARGETS "")
set(NN_BENCH_TARGETS "")   # 性能基准（仅测试构建，手工运行，不注册 ctest）

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

# ── 测试 / 验证目标（默认关闭，NN_ENABLE_TESTS=ON 时构建）─────────────
if(NN_ENABLE_TESTS)

    # GPU 测试（需要 Vulkan）
    nn_add_executable(gpu_test src/gpu_test.cpp)
    list(APPEND NN_TEST_TARGETS gpu_test)

    # ── Attention 一致性验证（forward vs forward_step）──────────────────────
    nn_add_executable(attn_consistency_test src/attn_consistency_test.cpp)
    list(APPEND NN_TEST_TARGETS attn_consistency_test)

    # ── SwiGLU 数值梯度检查（gradcheck，验证 split/merge 反向正确性）──────────
    nn_add_executable(swiglu_gradcheck src/swiglu_gradcheck.cpp)
    list(APPEND NN_TEST_TARGETS swiglu_gradcheck)

    # ── RMSNorm 数值梯度检查（gradcheck，验证 gamma/输入梯度） ───────────────
    nn_add_executable(rmsnorm_gradcheck src/rmsnorm_gradcheck.cpp)
    list(APPEND NN_TEST_TARGETS rmsnorm_gradcheck)

    # ── GPTModel 整链数值梯度检查（gradcheck，验证多层堆叠全部参数梯度） ──────
    nn_add_executable(gpt_gradcheck src/gpt_gradcheck.cpp)
    list(APPEND NN_TEST_TARGETS gpt_gradcheck)

    # ── GPT 梯度检查点一致性测试（验证激活重计算 L1 与全存结果一致） ───────
    nn_add_executable(gpt_checkpoint_test src/gpt_checkpoint_test.cpp)
    list(APPEND NN_TEST_TARGETS gpt_checkpoint_test)

    # ── activation offload 原语往返验证（GPU 激活 → host → GPU 一致） ──────
    nn_add_executable(offload_test src/offload_test.cpp)
    list(APPEND NN_TEST_TARGETS offload_test)

    # ── GPT activation offload 一致性测试（GPU：offload vs 全存基线） ──────
    nn_add_executable(gpt_offload_test src/gpt_offload_test.cpp)
    list(APPEND NN_TEST_TARGETS gpt_offload_test)

    # ── 文档感知注意力掩码构建单测（验证块对角∧因果∧ALiBi 组合） ───────────
    nn_add_executable(doc_mask_test src/doc_mask_test.cpp)
    list(APPEND NN_TEST_TARGETS doc_mask_test)

    # ── 文档感知掩码端到端正确性测试（不变性验证：跨文档屏蔽） ───────────────
    nn_add_executable(doc_attn_test src/doc_attn_test.cpp)
    list(APPEND NN_TEST_TARGETS doc_attn_test)

    # ── CausalSelfAttention 数值梯度检查（gradcheck，验证 attention 内部反向） ──
    nn_add_executable(attn_gradcheck src/attn_gradcheck.cpp)
    list(APPEND NN_TEST_TARGETS attn_gradcheck)

    # ── 融合 shader（AOT）GPU 数值验证（ExprSpec 单一事实来源 vs GPU 输出） ──
    nn_add_executable(fused_gpu_test src/fused_gpu_test.cpp)
    list(APPEND NN_TEST_TARGETS fused_gpu_test)

    # ── engine.compute(Mat_A + Mat_B) 表达式求值测试（CPU/GPU/CUDA 统一入口） ──
    nn_add_executable(tensor_expr_test src/tensor_expr_test.cpp)
    list(APPEND NN_TEST_TARGETS tensor_expr_test)

    # ── 统一表达式 DSL 单元测试（编译期融合 + AOT 折叠 + 防漂移断言，纯 CPU） ──
    nn_add_executable(expr_dsl_test src/expr_dsl_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_dsl_test)

    # ── ExprSpec 归约语义测试（M1：归约视图/归约指令 CPU 求值，纯 CPU） ──────
    nn_add_executable(expr_reduce_test src/expr_reduce_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_reduce_test)

    # ── IR 优化 pass 测试（IR-A canonicalize + IR-B CSE/寄存器分配，纯 CPU） ─
    nn_add_executable(expr_opt_test src/expr_opt_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_opt_test)

    # ── 图 IR + 融合分析测试（IR-C：expr_graph.hpp 纯函数，无 GPU） ─────
    nn_add_executable(expr_graph_test src/expr_graph_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_graph_test)

    # ── 图 IR 融合 GPU 端到端（IR-C：begin_expr/end_expr 融合执行） ─────
    nn_add_executable(expr_fuse_test src/expr_fuse_test.cpp)
    list(APPEND NN_TEST_TARGETS expr_fuse_test)

    # ── ModelSpec 架构校验一致性测试（load_model 文件头 vs 模型规格比对） ────
    nn_add_executable(model_spec_validation_test src/model_spec_validation_test.cpp)
    list(APPEND NN_TEST_TARGETS model_spec_validation_test)

    # ── Softmax 数值梯度检查（gradcheck，验证 softmax 反向） ───────────────────
    nn_add_executable(softmax_gradcheck src/softmax_gradcheck.cpp)
    list(APPEND NN_TEST_TARGETS softmax_gradcheck)

    # ── M4 matmul 融合原语测试（bmm_reduce/bmm_denom/bmm_apply，CPU+GPU） ──
    # ── M6 两趟式注意力反向原语测试（bmm_q_backward/bmm_kv_backward，CPU+GPU）
    # ── M5 列式 softmax 融合原语测试（col_softmax_denom/col_softmax_sparse_forward）
    nn_add_executable(matmul_fusion_test src/matmul_fusion_test.cpp)
    list(APPEND NN_TEST_TARGETS matmul_fusion_test)

    # ── M5 稀疏交叉熵融合测试（loss/grad vs 参考，CPU+GPU） ──
    nn_add_executable(ce_fusion_test src/ce_fusion_test.cpp)
    list(APPEND NN_TEST_TARGETS ce_fusion_test)

    # ── 性能基准工具（仅测试构建；手工运行，不注册 ctest） ──
    nn_add_executable(mnist_bench src/mnist_bench.cpp)
    nn_add_executable(compute_bench src/compute_bench.cpp)
    nn_add_executable(bench_thresholds src/bench_thresholds.cpp)
    nn_add_executable(reduce_bench src/reduce_bench.cpp)
    list(APPEND NN_BENCH_TARGETS mnist_bench compute_bench bench_thresholds reduce_bench)

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
    endforeach()
    # 基准工具输出到 test/，但不注册 ctest（手工运行）
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
        reduce
        broadcast
        transpose
        gather
        scatter_add
        bmm_reduce
        bmm_denom
        bmm_apply
        bmm_q_backward
        bmm_kv_backward
        col_softmax_denom
        col_softmax_sparse_forward)
    # ── 融合 shader：构建期从 Layer 内联表达式自动合成 ───────────────
    # 表达式**文本只出现在 Layer**（compute / start_expr..end_expr）。
    # 构建期两步：
    #   ① scan_exprs：dry-run 跑 Layer 的 forward/backward，收集折叠出的
    #      ExprSpec 结构（去重）→ expr_specs.bin（派生物，非手写定义）
    #   ② gen_fused：读 bin → glsl_gen → glslc → 内联 SPIR-V
    #      → fused_registry.hpp（key=expr_spec_key → 融合 shader）
    # 运行时按 key 精确匹配 dispatch；未命中硬报错（闭合世界）。
    set(GPU_SHADER_HPPS "")

    # 便捷函数：glslc → .spv → embed_spirv.cmake → C++ 头文件
    function(nn_embed_shader SHADER_NAME SRC_FILE)
        set(SPV ${CMAKE_BINARY_DIR}/${SHADER_NAME}.spv)
        set(HPP ${CMAKE_BINARY_DIR}/generated/${SHADER_NAME}_spv.hpp)
        set(FUNC "nn_${SHADER_NAME}_spirv_bytecode")
        set(GPU_SHADER_SPVS ${GPU_SHADER_SPVS} ${SPV} PARENT_SCOPE)
        set(GPU_SHADER_HPPS ${GPU_SHADER_HPPS} ${HPP} PARENT_SCOPE)

        # 编译 GLSL → SPIR-V
        add_custom_command(
            OUTPUT ${SPV}
            COMMAND ${GLSLC} -o ${SPV} ${SRC_FILE}
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
            list(APPEND NN_GPU_TARGETS ${NN_BENCH_TARGETS})
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

# ── CUDA GPU 加速（已停用）────────────────────────────────────────────────
# TODO(1.1): 恢复 CUDA 后端。
#   停用原因：M4/M5/M6 融合原语与 DSL 表达式在 CUDA 上仍未实现
#   （cuda_engine.hpp 直接返回 "未实现"），而 Layer 侧无条件接入融合路径且
#   无真实回退，导致 CUDA 上 GPT/MNIST 训练推理均无法运行。为避免"文档声称
#   支持但实际损坏"，1.0.0 起不提供该开关（不定义 NN_ENABLE_CUDA option）。
#   恢复前置条件：
#     1) 补齐 cuda/ 下的融合原语，或为各引擎加能力查询并在 Layer 侧做真实回退；
#     2) 解决 CUDA 内核 --use_fast_math 与主机侧"禁用 fast-math"的数值语义不一致；
#     3) 解除 cuda/CMakeLists.txt 中硬编码的 MSVC 版本路径（见其中 TODO）。
#   恢复时：恢复 option(NN_ENABLE_CUDA ...) 与下方 if(NN_ENABLE_CUDA) ... endif()
#   块，以及 add_subdirectory(cuda) 调用。
# CUDA 内核由 nvcc 编译为独立目标文件，通过 C-linkage 接口链接到主程序。
# - Windows：nvcc 以 MSVC 为 host 编译器，与 LLVM/Clang 通过 C ABI 兼容。
# - Linux：  nvcc 以 GCC/Clang 为 host 编译器，同样通过 C ABI 链接。
message(STATUS "CUDA backend disabled for 1.0.0 (fusion primitives unimplemented). See cuda/ TODO.")
