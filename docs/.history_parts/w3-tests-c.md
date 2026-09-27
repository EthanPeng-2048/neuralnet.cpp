# w3-tests-c

> 本轮「精度 / RAPT / ZiPT / 分词器 / 杂项测试与探针」注释整改中摘出的实质性历史。
> 对应改动文件：src/test_common.hpp、src/precision_profile_test.cpp、
> src/precision_type_test.cpp、src/f16_precision_test.cpp、src/f16_compute_test.cpp、
> src/tensor_precision_test.cpp、src/f16_cpu_probe.cpp、src/mem_probe.cpp、
> src/rapt_offload_test.cpp、src/tokenizer_consistency_test.cpp。

## test_common.hpp 聚合结构的由来（原位置 src/test_common.hpp:3-8）
- 类型：结构演进说明
- 内容：历史上每个子测试都自带一份 CHECK / make_tensor / check_close / approx /
  dot / close_to，聚合器（`#define main test_xxx + #include 子测试.cpp` 模式）
  被迫用一长串 `#define` 逐符号重命名来规避 ODR 冲突（漏一个就是重定义或静默绑错）。
  2026-09 收敛为 test_common.hpp 单份后，聚合器只需重命名 main、失败计数器与
  各测试独有的函数名。另：「刻意不收编」决定（2026-09 审查）——max_abs_diff 因
  9 份副本语义有分叉（conv2d 版多形状守卫返回 1e9）暂不收编，另案处理。

## ref_ulp 曾误写成 2^(e-10)（原位置 src/precision_type_test.cpp:55-57）
- 类型：bug 根因
- 内容：ref_ulp 曾误写成 2^(e-10)（正确为 2^(e-25)，大 2^15 倍）→ RHE 中点
  断言的容差比被测值本身还大 → float_to_half_bits 次正规 UB 窗口
  （exp ∈ [-45,-33]）产出的垃圾 half（如 0x4000=2.0）在该容差下永远测不出来。
  修复 = ref_ulp 改回 2^(e-25)，容差恢复到真实 ulp 量级。

## 次正规移位 UB 窗口与两个垃圾代表值（原位置 src/precision_type_test.cpp:195-198, 214）
- 类型：bug 根因
- 内容：precision.hpp 的 `float_to_half_bits` 次正规分支守卫曾写 `exp <= -46`，
  与其自身公式矛盾 → exp ∈ [-45,-33]（|v| ≈ 2.8e-14 ~ 1.2e-10）走 shift ≥ 32
  的移位 UB → 指数字段回绕产出垃圾 half（0x4000=2.0、0xCCCD… 等）→ CPU f16
  训练梯度被写成 512/8192/11776/18432/NaN，若干步后更新爆炸（梯度正是这个量级、
  前向激活 ~0.1 从不落入，故单算子测试全过）。修复 = 守卫改 `exp <= -26`
  （≤-26 按 D<0.5 恒 flush 0）。定点断言的两个代表值 1.13687e-12 与 8.44011e-11
  正是修复窗口上沿，修复前分别产出 0x4000=2.0 与 0x3333。

## `--f16` 曾等价于 profile_master_weights()（原位置 src/precision_profile_test.cpp:4-7）
- 类型：语义变更记录
- 内容：`--f16` 曾等价于 profile_master_weights()（F32 主权重 + f16 计算），
  用户以为开了全 f16、实际参数仍是 F32。2026-09 修订为 profile_f16()
  = {param:F16, compute:F16, stable:F32, optimizer:F32}（本测试把该语义钉死）。
  stable/optimizer 不取 F16 的实测证据：optimizer=F16 → Adam 的 v≈g²~1e-10
  f16 下溢、更新爆炸（loss 7.9→3.6e4）；stable=F16 → CE 链 ~200 步 NaN；
  四字段全 F16 → loss 恒定（更新被 f16 舍入吃光）。

## CPU 侧 f16 训练发散的双根因（原位置 src/f16_precision_test.cpp:569-573）
- 类型：bug 根因
- 内容：该处原为「历史"已知问题"（CPU 侧 f16 训练发散）已修复并转为硬失败」的
  根因注释：根因1 = DSL 预绑定把 f16 操作数喂给 f32 GEMM（cpu_matrix<F32>()
  拿到空指针 UB）；根因2 = float_to_half_bits 次正规分支 exp<=-46 的移位 UB
  （|v| ∈ [2.8e-14, 1.2e-10] 的梯度被写成垃圾 half）。完整记录见
  docs/development/05-mixed-precision.md §12.12。修复后 CPU f16 三组 profile
  6 步轨迹与 f32 逐位一致，本处改为硬失败断言。

## 逐元素原语已整体移除（原位置 src/f16_precision_test.cpp:93-94）
- 类型：删除清单
- 内容：原注释「（逐元素原语已整体移除；逐元素路径的 f32 零回归见
  test_f32_zero_regression_dsl）」——引擎级 elementwise_unary/binary/binary_scalar
  等算子于 2026-09 收敛时删除（逐元素一律走表达式 DSL），故 f32 零回归用例
  的代表原语选归约/搬运两类，逐元素路径的对拍由 DSL 版用例承担。

## rapt_offload：RLA backward 缓存曾不在 activation_cache() 内（原位置 src/rapt_offload_test.cpp:11-13）
- 类型：覆盖缺口
- 内容：RLA 的 backward 缓存（RMSNorm 后 Q/K、逐头 1/rms）此前不在
  activation_cache() 导出集合内，activation offload 会漏导出/恢复这些缓存。
  2026-09-19 RAPT 补齐 set_activation_offload 接线与 activation_cache() 覆盖后，
  本测试对拍 offload 开/关两侧的导出恢复集合完整性（缺一项即 backward 报缓存
  缺失或数值不一致）。

## 参考实现曾称「完整旧版 pre_tokenize」（原位置 src/tokenizer_consistency_test.cpp:35-36）
- 类型：措辞修正（与已删除实现的等价性论证）
- 内容：该参考函数原注释自述「完整旧版 pre_tokenize……逐字复刻优化前
  domain_tokenizer_bpe.hpp 的实现」。它当前是手写状态机（pre_match_len）的
  对拍基准，应表述为独立朴素实现（正则切分），不称「旧版」。

## docs/23 陈旧文档引用（原位置 src/precision_type_test.cpp:1、src/f16_compute_test.cpp:2、src/tensor_precision_test.cpp:2）
- 类型：失效链接
- 内容：三处「docs/23 §13 / §12.2」指向已不存在的旧编号设计文档；同内容现行
  位置为 docs/development/05-mixed-precision.md（§13 测试计划 T1–T11、
  §12.2 Phase 1 验收标准）。precision_type_test 的 printf 输出字符串里仍保留
  字面量「docs/23 T2」（字符串字面量不改动）。

## mem_probe 默认参数的 5.7GB 峰值锚点（原位置 src/mem_probe.cpp:22-23, 281）
- 类型：过期基准值
- 内容：原注释称默认参数「与已知"峰值 5.7GB"场景对齐」。该数字是 09-25 的
  f32 旧峰值（5743 MiB）；随后的显存优化已把它降到 ~3069 MiB，且
  docs/benchmarks/2026-09-25-vulkan-vs-cuda.md §"峰值已强相关于精度" 明确
  「09-25 ~5.7GB 与精度无关」的观察不再成立（f16 现为 6315/5738）。故摘除
  5.7GB 锚点，只保留「默认参数 = 该 bench 配置（vocab 8208 / d64 / h4 / L4 /
  ff256 / seq256 / batch64）」的测量口径陈述。

## f16_cpu_probe 曾自述为「临时诊断」（原位置 src/f16_cpu_probe.cpp:1-2）
- 类型：状态措辞
- 内容：探针最初为定位 CPU f16 训练发散的临时诊断工具；现已是注册的组件级
  探针（阶段标记 + 非有限值扫描），表头改为当前用途陈述。
