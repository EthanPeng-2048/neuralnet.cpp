# g15 GPU/层测试代码审查

## 模块概览

| 文件 | 职责 | 行数 | GPU 依赖 |
|------|------|------|----------|
| `src/gpu_test.cpp` | CPU/GPU 全原语交叉验证（matmul/batched_matmul/transpose/reduce/broadcast/elementwise/slice/insert/gather/scatter_add/rearrange_3d/clone）+ 性能对比 | 494 | 是（return 77） |
| `src/fused_gpu_test.cpp` | AOT 融合 shader GPU 数值验证（RoPE/SwiGLU/GeLU/Softmax/RMSNorm/LayerNorm/matmul 融合/matmul+归约/闭合世界回退） | 524 | 是（return 77） |
| `src/attn_w_batch_test.cpp` | 两趟式注意力 W 矩阵 batch 行号回归，4 配置 CPU/GPU 一致性 | 113 | 是（return 77） |
| `src/gpt_checkpoint_test.cpp` | GPT 梯度检查点分段 forward 等价性 + 全参数梯度一致性 | 178 | 否（纯 CPU） |

## 发现

### P1

**P1-1 `src/gpu_test.cpp:28`、`src/fused_gpu_test.cpp:35`、`src/attn_w_batch_test.cpp:34` — return 77 不被 ctest 识别为 SKIP，纯 CPU 构建时全部 GPU 测试报 FAILED**

证据：`CMakeLists.txt:292` 仅 `add_test(NAME ${TARGET} COMMAND ${TARGET})`，无 `WILL_FAIL` 或 `PASS_REGULAR_EXPRESSION` 属性。ctest 默认语义：exit 0 = Passed，非零 = Failed。exit 77 不是 ctest 的特殊值。

影响：纯 CPU 构建（`-DNN_ENABLE_TESTS=ON` 但无 Vulkan）时，gpu_test / fused_gpu_test / attn_w_batch_test 均返回 77，ctest 报 3 个 FAILED。CI 若不区分 "真失败" 与 "GPU 不可用"，会产生假红。

建议：在 `CMakeLists.txt` 中为 GPU 测试设置 `set_tests_properties(${TARGET} PROPERTIES PASS_REGULAR_EXPRESSION "\\[SKIP\\]")`，或改用 CMake 3.25+ 的 `SKIP_REGULAR_EXPRESSION`，或在测试注册前判断 `Vulkan_FOUND` 条件性跳过注册。

---

**P1-2 `src/attn_w_batch_test.cpp:40` — `std::stoul` 在 `-fno-exceptions` 下调用 `std::terminate`，违反铁律 1**

证据：`CMakeLists.txt:53` 加 `-fno-exceptions`；`attn_w_batch_test.cpp:40` 使用 `std::stoul(argv[1])` 等。`std::stoul` 在输入非法时抛 `std::invalid_argument`，在 `-fno-exceptions` 环境下直接 `std::terminate`，无错误信息输出。

对比：同组 `gpu_test.cpp:85-91` 已使用 `nn::parse_number<T>` 替代（注释明确写了"替代会抛异常的 std::stoi/stod"），说明项目已有共识。

建议：将 `std::stoul` 替换为 `nn::parse_number<std::size_t>` + 错误处理。

---

**P1-3 `src/gpt_checkpoint_test.cpp:38-39,156-157` — `NN_ASSERT` 在 Release 模式下为空操作，形状校验静默消失**

证据：`core_assert.hpp:12-13`：`#ifdef NDEBUG → #define NN_ASSERT(cond, msg) ((void)0)`。CMake 默认 Release（`CMakeLists.txt:42`），CMake Release 自动定义 `-DNDEBUG`。`gpt_checkpoint_test.cpp:38-39` 的 `NN_ASSERT(a.rows() == b.rows() && a.cols() == b.cols(), ...)` 和 `:156-157` 的 `NN_ASSERT(grads.size() == baseline_grads.size(), ...)` 在 Release 下完全消失。

影响：若梯度形状不匹配（如 checkpoint 重算路径出 bug），Release 构建的测试会静默通过（越界读取而非 abort），无法捕获回归。

建议：将 `NN_ASSERT` 替换为运行时校验（`if (...) { std::cerr << ...; return 1; }`），或 `NN_REQUIRE`（始终启用、返回 Error），或为测试构建单独设置 Debug 模式。

### P2

**P2-1 `src/fused_gpu_test.cpp:290-391` — 所有 matmul 融合测试矩阵尺寸过小（最大 6×9），未触达 BLOCK_SIZE=64 分块路径**

证据：`core_config.hpp:35`：`BLOCK_SIZE = 64`。`fused_gpu_test.cpp` 中最大 matmul 尺寸：`run_matmul` 的 M=5,K=7,N=4（line 290）和 M3=3,K3=6,N3=2（line 339），`run_matmul_reduce` 的 M=5,K=7,N=4（line 403）。所有维度均远小于 64，完全不进入 `algebra_matrix.hpp:183-200` 的分块路径（`i_blocks = (M+BLOCK_SIZE-1)/BLOCK_SIZE`，当 M<64 时 i_blocks=1，仅单块）。

影响：GPU 融合 shader 的 matmul 段（matmul spec dispatch）在小矩阵上走单次 dispatch，不触发分块累加、多 workgroup 调度。若分块路径存在 bug（如边界处理、部分和累加），fused_gpu_test 无法发现。

对比：`gpu_test.cpp` 默认 N=256 确实覆盖分块路径，但它测的是原语 matmul，不是融合 shader 路径。

建议：至少增加一组 M/K/N ≥ 128 的测试用例（如 M=128,K=256,N=128），验证融合 shader 在分块路径下的正确性。

---

**P2-2 `src/fused_gpu_test.cpp` — 无 batch>1 的融合 shader 测试**

证据：`fused_gpu_test.cpp` 所有测试函数（`run_rope`/`run_swiglu`/`run_gelu`/`run_softmax`/`run_norm`/`run_matmul`/`run_matmul_reduce`）均使用单 batch（`batch=1` 或隐含单样本）。`attn_w_batch_test.cpp` 的注释明确记载 batch 内行号 bug（P0-1），且铁律 5 要求"所有注意力/序列相关测试必须覆盖 batch>1"。

影响：融合 shader 中 batch 相关的索引计算（如 rearrange_3d、batched dispatch）在 batch=1 时退化为 trivial 路径，无法暴露 batch 索引错误。

建议：为至少 SwiGLU/GeLU/RoPE 增加 batch=2 或 batch=4 的测试配置。

### P3

**P3-1 `src/fused_gpu_test.cpp:157,169,127,206,223,258,275` — `gpu.to_matrix()` 返回值未检查即解引用**

证据：以 `run_gelu:157` 为例：`const Scalar err_f = max_abs_diff(fc->cpu_matrix(), *gpu.to_matrix(*fg));`。`to_matrix` 返回 `Result<Matrix>`（`std::expected`），若返回 `std::nullopt`（如 GPU buffer 无效），`*` 解引用为 UB。同样的模式出现在 `run_swiglu:127`、`run_softmax:206,223`、`run_norm:258,275`。

对比：`gpu_test.cpp` 和 `attn_w_batch_test.cpp` 中对 `to_matrix` 返回值都有检查。

建议：提取 `to_matrix_or_die` 辅助函数，或逐处加 `if (!gm) { ... return 1; }`。

---

**P3-2 `src/attn_w_batch_test.cpp:108` — 4 配置中最大 BH=4，seq=5 固定，d_k=4 较小**

证据：`cfgs[4][2] = {{1,1},{2,1},{1,2},{2,2}}`（line 108），`d_model=16`（line 38），`seq=5`（line 39）。BH 最大为 2×2=4，d_k=d_model/heads=16/2=8 或 16/1=16。

说明：batch>1 已覆盖（(2,1) 和 (2,2) 配置），这是该测试的核心目的（回归 batch 内行号 bug）。但 seq=5 固定不变，且 d_k 最大为 16（当 heads=1 时 d_k=16），对于多头注意力的 head 维度覆盖有限。

建议：可考虑增加一组 batch=4 或 seq=16 的配置以扩大覆盖（优先级低，当前配置已满足回归目的）。

---

**P3-3 `src/gpt_checkpoint_test.cpp` — 仅 CPU 路径，无 GPU 梯度检查点验证**

证据：`gpt_checkpoint_test.cpp:69`：`nn::CpuEngine eng;`，整个测试仅使用 CPU 引擎。GPU 路径的 checkpoint 重算（涉及 GPU 激活重计算、`pending_destroys_` 延迟队列）未被覆盖。

说明：这是测试范围选择问题，不是代码缺陷。GPU checkpoint 路径涉及显存管理，测试复杂度高，可作为后续测试扩展。

---

**P3-4 `src/gpu_test.cpp:179` — matmul 容差1e-2 对 256×256 矩阵合理，但缺少容差与矩阵规模的显式关联说明**

证据：`gpu_test.cpp:179`：`const bool matmul_ok = (matmul_err < 1e-2f) && (nrmse < 1e-3f);`。对于 N=256，float32 累加 256 项的 rounding error 上界约为 256×2^-23×|c_ij|≈0.008（最坏情况），1e-2 刚好覆盖。但当 `--size` 参数传入更大值（如1024），容差未相应缩放。

建议：注释说明容差与 N 的关系，或改为 `tol = 1e-2f * std::sqrt(static_cast<float>(N) / 256)`。

## 已知项核对（gpu_test 退出码/return 77）

| 检查项 | 结果 | 位置 |
|--------|------|------|
| 无 Vulkan 时 return 77 | ✅ 生效 | `gpu_test.cpp:24-29`：`#ifndef NN_HAS_VULKAN` 包裹整个 `main()`，return 77 |
| 失败计数进退出码 | ✅ 生效 | `gpu_test.cpp:117`：`std::size_t failures = 0;`；各检查点 `++failures`；`gpu_test.cpp:492`：`return failures > 0 ? 1 : 0;` |
| fused_gpu_test return 77 | ✅ 生效 | `fused_gpu_test.cpp:31-37`：`#ifndef NN_HAS_VULKAN` return 77 |
| attn_w_batch_test return 77 | ✅ 生效 | `attn_w_batch_test.cpp:34`：GPU 不可用时 return 77 |

## 待验证

1. **gpu_test matmul 容差边界**：`--size 1024` 时 matmul_err 是否仍 <1e-2。当前容差对 N=256 合理，但 N 增大后 GPU 累加顺序差异可能放大误差。建议实测确认或改为相对误差/按 N 缩放容差。
2. **fused_gpu_test SwiGLU 矩阵 d_ff=8**（line 99）：d_ff=8 远小于 BLOCK_SIZE=64，SwiGLU 内部涉及的 split/elementwise 操作在小尺寸下不走并行路径。是否需要扩大到 d_ff≥128 待评估。
3. **attn_w_batch_test 的 `std::stoul` 在 MSVC 下行为**：MSVC 未加 `-fno-exceptions`（`CMakeLists.txt:51`），`std::stoul` 抛异常后被 MSVC 的 EHsc 捕获 → `std::terminate` 或未定义行为。与 Clang/GCC 下的 `std::terminate` 行为不同，需确认是否影响 CI。
