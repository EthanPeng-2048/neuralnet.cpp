# ── cuda/CMakeLists.txt — CUDA 计算后端（独立 nvcc 编译）────────────────────
#
# 跨平台 nvcc 编译：
#   - Windows：nvcc 以 MSVC 为 host 编译器，需 vcvars64 环境 + MSVC CRT 标志，
#             输出 .obj，链接到主 Clang 项目（C ABI 兼容）。
#   - Linux：  nvcc 以 GCC/Clang 为 host 编译器，直接调用 nvcc 即可，
#             输出 .o，链接到主项目（同样 C ABI）。
#
# 构建流程：
#   1. nvcc -c cuda_kernels.cu → cuda_kernels.{obj|o}
#   2. 主项目编译 CudaBackend/CudaEngine（调用内核函数）
#   3. 链接内核目标 + cudart + cuBLAS
# ─────────────────────────────────────────────────────────────────────────

# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# 设置 SM 目标架构（V100 = sm_70）
if(NOT DEFINED NN_CUDA_ARCH)
    set(NN_CUDA_ARCH "70" CACHE STRING "CUDA SM architecture (e.g. 70, 80, 86, 89)")
endif()

if(WIN32)
    # ── Windows 分支：nvcc + MSVC（vcvars 环境）─────────────────────────
    # 由于 CMake 的 add_custom_command 无法直接设置 vcvars 环境，
    # 生成一个临时 .bat 脚本来包装编译命令。
    # Debug 模式使用 /MDd（动态调试 CRT），Release 模式使用 /MD（动态发布 CRT）
    # 同时对齐 _ITERATOR_DEBUG_LEVEL（主程序 Debug 为 2，Release 为 0），
    # 否则 lld-link 会报 /failifmismatch。
    set(NN_CUDA_HOST_CL "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe")
    set(NN_CUDA_HOST_VCVARS "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR (DEFINED CMAKE_CONFIGURATION_TYPES AND CMAKE_BUILD_TYPE MATCHES "Debug"))
        set(NN_CUDA_CRT_FLAG "/MDd")
        set(NN_CUDA_ITDL "2")
    else()
        set(NN_CUDA_CRT_FLAG "/MD")
        set(NN_CUDA_ITDL "0")
    endif()

    set(NN_CUDA_COMPILE_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/compile_cuda_kernels.bat")
    file(WRITE "${NN_CUDA_COMPILE_SCRIPT}"
        "@echo off\r\n"
        "call \"${NN_CUDA_HOST_VCVARS}\" >nul 2>&1\r\n"
        "\"${CUDAToolkit_BIN_DIR}/nvcc.exe\" -c -O3 --use_fast_math -allow-unsupported-compiler"
        " -Xcompiler ${NN_CUDA_CRT_FLAG}"
        " -Xcompiler /D_ITERATOR_DEBUG_LEVEL=${NN_CUDA_ITDL}"
        " -arch=sm_${NN_CUDA_ARCH} -DNN_HAS_CUBLAS"
        " -I\"${CMAKE_CURRENT_SOURCE_DIR}/../include\""
        " -I\"${CUDAToolkit_INCLUDE_DIRS}\""
        " \"%~1\" -o \"%~2\"\r\n"
        "exit /b %errorlevel%\r\n"
    )
    set(NN_CUDA_KERNELS_OBJ "${CMAKE_CURRENT_BINARY_DIR}/cuda_kernels.obj")
    set(NN_CUDA_COMPILE_CMD cmd /c "${NN_CUDA_COMPILE_SCRIPT}")
else()
    # ── Linux/UNIX 分支：nvcc + 系统 C++ 编译器 ──────────────────────────
    # 通过 -ccbin 复用主项目 CMake 选定的系统 C++ 编译器（CMAKE_CXX_COMPILER，
    # 可以是 clang 或 gcc，取决于配置），不写死任何编译器。
    # - -allow-unsupported-compiler：兼容较新的 GCC/Clang（nvcc 自带版本
    #   检查通常落后于发行版编译器，否则会报 unsupported GNU version）。
    # - 使用 -fPIC 以便目标可安全链接进可执行文件/共享库。
    # - -U_GNU_SOURCE：规避 glibc ≥ 2.39/2.41 与 CUDA 12.x 的已知冲突。
    #   clang（以及 gnu 方言的 gcc）会强制定义 _GNU_SOURCE，从而启用 glibc
    #   mathcalls.h 中 IEC_60559_FUNCS_EXT_C23 的声明（cospi/sinpi/cospif/
    #   sinpif/rsqrt/rsqrtf），它们带 noexcept(true)，与 CUDA math_functions.h
    #   无 noexcept 的声明冲突，报 "exception specification is incompatible"。
    #   撤销 _GNU_SOURCE 可禁用这些声明（无需修改系统 CUDA 头文件）。
    # - 注意：不要给 host 编译加 -fno-exceptions，否则同样会触发上述异常规范
    #   冲突。.cu 是独立 TU，通过 C ABI 链接，无需强制关闭异常。
    # - cuBLAS 宏在 nvcc 侧定义，使内核内部走 cuBLAS 加速路径。
    set(NN_CUDA_KERNELS_OBJ "${CMAKE_CURRENT_BINARY_DIR}/cuda_kernels.o")
    set(NN_CUDA_COMPILE_CMD
        "${CUDAToolkit_BIN_DIR}/nvcc" -c -O3 --use_fast_math
        -ccbin "${CMAKE_CXX_COMPILER}"
        -allow-unsupported-compiler
        -Xcompiler -fPIC
        -Xcompiler -U_GNU_SOURCE
        -arch=sm_${NN_CUDA_ARCH} -DNN_HAS_CUBLAS
        -I"${CMAKE_CURRENT_SOURCE_DIR}/../include"
        -I"${CUDAToolkit_INCLUDE_DIRS}"
    )
endif()

# ── CUDA 内核目标文件（.obj / .o） ─────────────────────────────────────
add_custom_command(
    OUTPUT ${NN_CUDA_KERNELS_OBJ}
    COMMAND ${NN_CUDA_COMPILE_CMD} "${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels.cu" "${NN_CUDA_KERNELS_OBJ}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels.cu
    COMMENT "[CUDA] Compiling cuda_kernels.cu with nvcc (sm_${NN_CUDA_ARCH})"
)

add_custom_target(compile_cuda_kernels DEPENDS ${NN_CUDA_KERNELS_OBJ})

# ── 导出变量给父项目 ────────────────────────────────────────────────────
set(NN_CUDA_KERNELS_OBJ ${NN_CUDA_KERNELS_OBJ} PARENT_SCOPE)
set(NN_CUDA_FOUND TRUE PARENT_SCOPE)
set(NN_CUDA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
