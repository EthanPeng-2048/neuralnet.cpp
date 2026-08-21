# CMake 3.30+ required for C++26 standard support (stable in 3.30+)
cmake_minimum_required(VERSION 3.30)
project(neuralnet.cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── 默认 Release 模式，可通过 -DCMAKE_BUILD_TYPE=Debug 切换 ────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── 公共编译标志 ─────────────────────────────────────────────────────────
# -fno-exceptions: 项目规范禁止异常，编译器层面强制执行
# -Wall -Wextra -Wpedantic -Werror: 严格警告
add_compile_options(-fno-exceptions -Wall -Wextra -Wpedantic -Werror)

# ── 编译器特有选项 ───────────────────────────────────────────────────────
# -fexperimental-library 仅 Clang 识别；GCC 不识别会产生警告
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(-fexperimental-library)
endif()

# ── 按构建类型设置优化标志 ───────────────────────────────────────────────
# Release: -O3 -funroll-loops -march=native（最大性能）
# Debug:   -O0 -g（调试优先）
# 注意：不使用 -ffast-math，因为它会禁用 NaN/Infinity 支持，影响训练稳定性
set(RELEASE_FLAGS -O3 -fno-math-errno -fno-trapping-math -funroll-loops -march=native)
set(DEBUG_FLAGS -O0 -g)

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

# ── MNIST 目标 ──────────────────────────────────────────────────────────────
nn_add_executable(mnist_train src/mnist_train.cpp)
nn_add_executable(mnist_infer src/mnist_infer.cpp)
nn_add_executable(mnist_bench src/mnist_bench.cpp)

# ── 文本目标（GPT 训练/推理 + BPE 分词器训练/推理） ───────────────────
nn_add_executable(text_train src/text_train.cpp)
nn_add_executable(text_infer src/text_infer.cpp)
nn_add_executable(tokenizer_train src/tokenizer_train.cpp)
nn_add_executable(tokenizer_infer src/tokenizer_infer.cpp)

# ── 计算性能基准测试 ─────────────────────────────────────────────────────
nn_add_executable(compute_bench src/compute_bench.cpp)
nn_add_executable(bench_thresholds src/bench_thresholds.cpp)

# ── GPU 测试（需要 Vulkan） ──────────────────────────────────────────────
nn_add_executable(gpu_test src/gpu_test.cpp)

# ── Attention 一致性验证（forward vs forward_step）──────────────────────
nn_add_executable(attn_consistency_test src/attn_consistency_test.cpp)

# ── SwiGLU 数值梯度检查（gradcheck，验证 split/merge 反向正确性）──────────
nn_add_executable(swiglu_gradcheck src/swiglu_gradcheck.cpp)

# ── RMSNorm 数值梯度检查（gradcheck，验证 gamma/输入梯度） ───────────────
nn_add_executable(rmsnorm_gradcheck src/rmsnorm_gradcheck.cpp)

# ── GPTModel 整链数值梯度检查（gradcheck，验证多层堆叠全部参数梯度） ──────
nn_add_executable(gpt_gradcheck src/gpt_gradcheck.cpp)

# ── 文档感知注意力掩码构建单测（验证块对角∧因果∧ALiBi 组合） ───────────
nn_add_executable(doc_mask_test src/doc_mask_test.cpp)

# ── 文档感知掩码端到端正确性测试（不变性验证：跨文档屏蔽） ───────────────
nn_add_executable(doc_attn_test src/doc_attn_test.cpp)

# ── CausalSelfAttention 数值梯度检查（gradcheck，验证 attention 内部反向） ──
nn_add_executable(attn_gradcheck src/attn_gradcheck.cpp)

# ── 融合 shader（AOT）GPU 数值验证（ExprSpec 单一事实来源 vs GPU 输出） ──
nn_add_executable(fused_gpu_test src/fused_gpu_test.cpp)

# ── engine.compute(Mat_A + Mat_B) 表达式求值测试（CPU/GPU/CUDA 统一入口） ──
nn_add_executable(tensor_expr_test src/tensor_expr_test.cpp)

# ── 统一表达式 DSL 单元测试（编译期融合 + AOT 折叠 + 防漂移断言，纯 CPU） ──
nn_add_executable(expr_dsl_test src/expr_dsl_test.cpp)

# ── Softmax 数值梯度检查（gradcheck，验证 softmax 反向） ───────────────────
nn_add_executable(softmax_gradcheck src/softmax_gradcheck.cpp)

# ── 注意力反向逐级调试探针（对比暴力 Matrix 参考与引擎原语） ────────────
nn_add_executable(attn_debug src/attn_debug.cpp)

# ── batched_matmul 原语探针（四种转置组合 vs 手工参考） ───────────────────
nn_add_executable(matmul_probe src/matmul_probe.cpp)

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
        scatter_add)
    # ── 融合 shader：算法在 C++（单一事实来源），构建期降级成 GPU ─────
    # GPU_FUSED_SHADERS 的每个条目都来自 include/neuralnet.cpp/fused_exprs.hpp
    # 的 kGenInstances：构建期 tools/gen_fused 用 dsl 折叠同一段表达式生成
    # .comp，再走 glslc → embed 管线。**算法唯一事实来源在 C++ 源码**，
    # 融合 shader 是自动生成的产物（请勿手写算法 GLSL）。
    # 一个算法允许拆成多个融合 kernel（如 SwiGLU backward = grad_gate +
    # grad_up 两个）；不强制"每算法单 kernel"。性能让位于清晰/可维护。
    set(GPU_FUSED_SHADERS
        rope_forward_dk32
        rope_forward_dk64
        rope_forward_dk128
        rope_backward_dk32
        rope_backward_dk64
        rope_backward_dk128
        swiglu_grad_gate
        swiglu_grad_up)
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

    # 2. AOT 融合 shader：构建期编译 tools/gen_fused（复用 fused_exprs.hpp /
    #    glsl_gen.hpp，单一事实来源），运行生成 fused/*.comp
    set(FUSED_GEN_SRC ${CMAKE_SOURCE_DIR}/tools/gen_fused.cpp)
    set(FUSED_OUT_DIR ${CMAKE_BINARY_DIR}/fused)
    add_executable(gen_fused ${FUSED_GEN_SRC})
    target_include_directories(gen_fused PRIVATE ${CMAKE_SOURCE_DIR}/include/neuralnet.cpp)

    set(FUSED_COMP_FILES "")
    foreach(SHADER_NAME ${GPU_FUSED_SHADERS})
        list(APPEND FUSED_COMP_FILES ${FUSED_OUT_DIR}/${SHADER_NAME}.comp)
    endforeach()

    add_custom_command(
        OUTPUT ${FUSED_COMP_FILES}
        COMMAND gen_fused ${FUSED_OUT_DIR}
        DEPENDS gen_fused ${FUSED_GEN_SRC}
        COMMENT "[GPU] Generating fused shaders from fused_exprs.hpp (gen_fused)"
    )

    foreach(SHADER_NAME ${GPU_FUSED_SHADERS})
        nn_embed_shader(${SHADER_NAME} ${FUSED_OUT_DIR}/${SHADER_NAME}.comp)
    endforeach()

    if(NOT GLSLC)
        message(WARNING "No SPIR-V compiler found (glslc). "
                        "GPU acceleration disabled. Install Vulkan SDK.")
        set(Vulkan_FOUND OFF)
    endif()

    if(Vulkan_FOUND)
        add_custom_target(compile_gpu_shaders DEPENDS ${GPU_SHADER_HPPS})

        # 为所有目标启用 Vulkan
        set(NN_GPU_TARGETS
            mnist_train mnist_infer mnist_bench
            text_train text_infer
            tokenizer_train tokenizer_infer
            compute_bench bench_thresholds gpu_test
            attn_consistency_test
            swiglu_gradcheck
            rmsnorm_gradcheck
            gpt_gradcheck
            attn_gradcheck
            fused_gpu_test
            tensor_expr_test
            softmax_gradcheck
            attn_debug
            matmul_probe
            doc_attn_test
        )

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

# ── CUDA GPU 加速（可选）──────────────────────────────────────────────────
# CUDA 内核由 nvcc 编译为独立目标文件，通过 C-linkage 接口链接到主程序。
# - Windows：nvcc 以 MSVC 为 host 编译器，与 LLVM/Clang 通过 C ABI 兼容。
# - Linux：  nvcc 以 GCC/Clang 为 host 编译器，同样通过 C ABI 链接。
option(NN_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)
if(NN_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA Toolkit not found. GPU acceleration disabled.")
    else()
        add_subdirectory(cuda)
        if(NN_CUDA_FOUND)
            set(NN_CUDA_TARGETS
                mnist_train mnist_infer mnist_bench
                text_train text_infer
                tokenizer_train tokenizer_infer
                compute_bench bench_thresholds gpu_test
                attn_consistency_test
                swiglu_gradcheck
                rmsnorm_gradcheck
                gpt_gradcheck
                attn_gradcheck
                softmax_gradcheck
                doc_attn_test
            )

            foreach(TARGET ${NN_CUDA_TARGETS})
                target_compile_definitions(${TARGET} PRIVATE NN_HAS_CUDA NN_HAS_CUBLAS)
                target_include_directories(${TARGET} PRIVATE
                    ${NN_CUDA_INCLUDE_DIR}
                    ${CUDAToolkit_INCLUDE_DIRS}
                )
                # nvcc 编译的 .obj + CUDA 运行时 + cuBLAS（matmul 加速）
                target_link_libraries(${TARGET} PRIVATE
                    ${NN_CUDA_KERNELS_OBJ}
                    CUDA::cudart
                    CUDA::cublas
                    CUDA::cublasLt
                )
                add_dependencies(${TARGET} compile_cuda_kernels)
            endforeach()

            message(STATUS "CUDA GPU acceleration enabled (nvcc → obj → link, cuBLAS enabled)")
        else()
            message(WARNING "CUDA not built. GPU acceleration disabled.")
        endif()
    endif()
else()
    message(STATUS "CUDA disabled. Use -DNN_ENABLE_CUDA=ON to enable.")
endif()
