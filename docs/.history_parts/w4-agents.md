# w4-agents

> 来源：AGENTS.md（AI 开发速览）整改摘录（2026-09-27）。AGENTS.md 自此只记录当前状态；演进记录、已修复勘误、被否决方案与性能 A/B 过程归档于此，合并目标 `docs/history.md`。

## 版本交付内容（原位置 AGENTS.md 头部版本行 + §12「已交付能力」v1.1.0/v1.2.0）

- 类型：演进记录
- 内容：v1.1.0 = GPT 训练错误修复、算子融合优化、ZiPT/RAPT 掩码。v1.2.0（Release）= 混合精度（`precision.hpp` + CPU/GPU f16 路径）、RLA-2/RAPT、防 NaN 跳步、CPU 优化、头文件保护更换、编译器兼容（GCC/MSVC）、测试补充。头部"当前版本 v1.2.0 + 后续提交（Vulkan 设备选择、activation offload）"表述同轮移除；当前能力清单见 AGENTS §12。

## 计算类原语全量迁 DSL（2026-09-27，原位置 AGENTS.md §12「后续提交」第 1 条）

- 类型：演进记录 + 教训
- 内容：Layer 直调 **34 → 23**（`bench/doc_inventory.ps1` 复测），10 类计算原语全部退出 Layer 直调——`add_inplace`/`scale_inplace`/`accumulate`→`dsl::compute_into`（10 处）、`matmul`/`matmul_with_bias`→`dsl::matmul`（Linear/Conv/Transformer/Muon）、`row/col_reduce_*`→`dsl::compute_reduce`（11 处）、`batched_matmul` 21 处→`dsl::matmul(..., batch)`（alpha 用 `*dsl::rparam(scale)` 尾链，累加后乘三端等价）；缺口 A：新增 `ExprViewKind::GroupedReduceSum/Max`（`dsl::grouped_reduce_sum/max(t, R)`，R 进 key、glsl_gen 编译期展开、CPU 模板路径独立归约），MaxPool2D 迁移。保留 23 个 = 基础设施/数据搬运（transpose/slice/gather/im2col…）/状态扫描（scan_*/outer_col）/fold 显式登记，与 `docs/development/12` §7.3 形态一致。ctest 19/19 全绿。
- 教训①：scan dry-run **禁止 `(void)` 吞错**——ZiPTBlock 的 dry-run 调单参 `forward(x)`（实际返回"用双参"错误）被吞 → **该 dry-run 从未执行真实路径**，存量结构恰好处处有 key 才长期绿，迁移新结构立刻闭合世界硬报错（已修为双参调用 + 失败 abort）；**dry-run 长期绿 ≠ 覆盖存在**。
- 教训②：新 `ExprViewKind` 落点清单：枚举+validate（`expr_spec.hpp`）→ SpecBuilder helper+叶子+自由函数+`has_reduction_v=false` 显式特化（`expr_dsl.hpp`）→ `glsl_view_read` case（`expr_glsl_gen.hpp`，R 编译期展开）→ 解释器两处 switch（`compute_cpu_engine.hpp` 形状校验+读取）；GPU 侧无 per-view 分派，vec4 白名单/uses_row 排除集自动正确。

## Vulkan 设备选择 / activation offload / CPU 逐元素优化的提交流水（原位置 AGENTS.md §12「后续提交」）

- 类型：演进记录
- 内容：Vulkan 设备选择（15eb731）：`--gpu` 参数指定设备。activation offload（初现 `42c0883`/v0.2.0、GPT 接线 `26d8eb4`/v1.1.0、`ActivationOffloader` 抽取 `8f2990f`）：相较梯度检查点更省时（文档记载 18s vs 26s / 5 step，**仓库内无原始计时**）；API 为 `GPTModel::set_activation_offload`（`set_offload_enabled` 不存在）+ `ComputeEngine::create_offload_buffer/offload_save/offload_restore`（`offload_store/load` 于 2026-09 删除）；checkpoint 与 offload **可混合**（此前帮助文本"互斥"是错的）。CPU 逐元素优化 + `dsl::compute_into`（c0d3298）：DSL 模板路径向量化/并行、`Tensor::cpu_get_ptr`、零分配原地目标传递（optimizer/layer/loss 已迁移）。

## 代码缩减重构：算子收敛 + 测试结构 + AST 移除（2026-09-26，原位置 AGENTS.md §12「后续提交」）

- 类型：删除清单
- 内容：① 删 A 档死码（`elementwise_select_scalar_cond` / `axpy_inplace` / `broadcast_row_inplace` / `nn::one_hot` / `one_hot_labels` / `Matrix::multiply_transposed_add_to` / `nn::sigmoid`·`relu` / `ops::Sigmoid`·`ReLU`）；② 新增测试公共头 `src/test_common.hpp`（`CHECK`/`make_tensor`/`check_close`/`approx`/`dot`/`close_to` 收敛一份；`max_abs_diff` 因 9 份副本语义有分叉**暂不收编**）；③ RAPT `forward_step` 的 `num/den` 除法迁 DSL，写成与 `forward:491` 同构的 `leaf/(leaf+Scalar{1e-4})` → 复用已注册 AOT 键（无需改 scan_exprs）；④ 删 `elementwise_unary/binary/binary_scalar`、`broadcast_col_inplace`（连带 `broadcast.comp` / `broadcast_gpu` / pipeline / `has_*` / 成员全链）、引擎 `row_reduce_max`、offload B 组 `offload_store/load`、`UnaryOp/BinaryOp/CompareOp` 枚举（**引擎 virtual 58 → 49**）；⑤ **旧代数 AST 整体移除**：`algebra_expr.hpp`/`algebra_compute.hpp` 删除，`Expression`/`BoolExpression` 概念迁入 `expr_dsl.hpp`，`Matrix::detail::{apply,binary_apply,binary_apply_inplace}` 死函数删除 → **CPU 求值只剩「DSL 模板路径 + IR 解释器」两套**；⑥ 删死 `activation_cache()` override（ViT/ZiPT，20 行）与重复的 `offload_test` 聚合目标（**ctest 20 → 19**），`gpt_offload_test` 补 `begin_batch/end_batch`（此前无 offload save/restore 训练真实录制路径覆盖）；`--activation-offload` 对 CPU/ZiPT 改为显式警告（原为静默 no-op 却打印"已启用"）。
- 教训：`dsl::compute` **无 2 参重载**（必带 `rows, cols`）；eager→DSL 迁移**先确认目标表达式结构已被 `scan_exprs` 覆盖**，否则 GPU 闭合世界运行期硬报错。

## CUDA 后端整体移除（96a3675，原位置 AGENTS.md §2 + §12）

- 类型：删除清单
- 内容：删除 `cuda/`、`compute_cuda_engine.hpp`、`backend/compute_cuda_backend.hpp`、全库 `NN_HAS_CUDA` 分支与 CLI `--cuda`；带 CUDA 的历史快照在 git 分支 `legacy/cuda`（c0d3298）。删除依据：40HX Vulkan 端到端 2890 ≈ torch 2640 tok/s 但算子级仅 cuBLAS 1/3、batch8 OOM。当前结论保留在 AGENTS §2：**本项目不支持 CUDA**（后端仅 CPU/Vulkan，CLI 无 `--cuda`）。

## IR-C（图 IR / 跨表达式融合）整体移除（原位置 AGENTS.md §12 + §7）

- 类型：被否决方案
- 内容：删除 `expr_graph.hpp`、`compute_engine` 的 `begin_expr/end_expr`、`dsl::start_expr/end_expr`、`FusedChainLayer`、`Tensor::virtual_tag_`。取舍记录正文（五条否证依据、删除项/保留项清单、重新立项前提）见 `docs/development/03-ir-optimization.md` §5.3（归档正文在 `docs/history.md`）。当前结论保留在 AGENTS §7：**IR-C 未采用，当前没有任何跨表达式录制机制**。

## RLA/RAPT 激活重计算 + offload 接线（2026-09-19，原位置 AGENTS.md §12）

- 类型：已修复勘误
- 内容：`RAPTModel/RAPTBlock` 补齐 `set_flush_interval / set_checkpoint_every / set_activation_offload`（此前对 RAPT 是静默 no-op，而 CLI 会打印"已启用"）；`ReLULinearAttention` 响应 `checkpoint_mode_`（跳过 7 项 backward 缓存）、`activation_cache()` 补齐遗漏的 4 项、backward 缺缓存硬报错；抽出通用 `ActivationOffloader`（`compute_layer_base.hpp`，GPT/RAPT 共用）；修 `RAPTModel::clear_cache()` 清空 `token_emb_`（参数被毁）与 `doc_ids` 无法关闭两个缺陷。新增 `rapt_checkpoint_test`（并入 `rapt_test`）与 `rapt_offload_test`，此前**从未被编译**的 `gpt_offload_test` 转正（ctest 15 → 18）。

## CNN 缓存契约加固 + 测试补齐（2026-09-19，原位置 AGENTS.md §12）

- 类型：已修复勘误
- 内容：`MaxPool2D::backward` 补 argmax 缓存/形状校验（旧行为：`clear_cache()` 后**越界读空 vector 且不报错**——`vector::clear()` 保留容量，表现为静默用陈旧索引）；`Conv2D::backward` 补 im2col 缓存/形状校验（旧行为：checkpoint 模式或 batch 变化后**静默用陈旧 im2col**）；两层 `forward` 在 checkpoint 模式下**显式清空**缓存（只跳过赋值会因 size 相同而静默沿用旧数据）；两层补 `recompute_supported()`。新增 `cnn_test`（`maxpool_gradcheck` 独立参考比对 3 组配置 + 缓存/checkpoint 契约 + `cnn_smoke_test`），CPU 与 `--gpu` 逐位一致（此前 Conv2D/MaxPool2D **无任何 GPU 覆盖**、MaxPool2D **无任何测试**）。

## CNN 全引擎化（2026-09-20，原位置 AGENTS.md §12）

- 类型：演进记录
- 内容：新增 `im2col`/`col2im` 两个 op-level 数据搬运原语（接口 + CPU + Vulkan shader + backend + GpuEngine 包装）；`Conv2D`/`MaxPool2D` 的 forward/backward 改写为纯「引擎原语 + DSL」组合——消除 `to_matrix/from_matrix` PCIe 往返与 CPU 标量 im2col 循环，`col_cache_` 由 CPU `Matrix` 改为设备张量。布局 `(C,B*P) ↔ (C*P,B)` 用 `rearrange_3d` + `gather_rows/scatter_add_rows` + 缓存置换索引实现（`rearrange_3d` 单独做不到该三维转置）。池化反向改为「窗口内并列最大值**均分**梯度」（总梯度守恒；无并列时与 argmax-first 逐位一致）。`scan_exprs` 补 MaxPool2D dry-run（69 → 72 条融合表达式）。

## 分组归约原语 + 融合 push-constant 缺陷修复（2026-09-20，原位置 AGENTS.md §12）

- 类型：已修复勘误
- 内容：新增 `grouped_reduce_sum/max`（沿行方向按固定长度 R 分组归约），MaxPool2D 因此去掉逐通道 C 次 dispatch → 单次原语；新增 `fused_gpu_test::run_reduce_consts` 回归用例。根因：`run_fused_gpu` 的 push-constant **固定头长度必须逐形态**与生成器 PC 声明一致（逐元素 2 / 逐元素+matmul 5 / 归约 4 / 归约+matmul 6）；bug 把"归约但无 matmul"按 **5** 算 → 常量池整体后移一个 uint → **GPU 上"带常量的归约"静默错值而 CPU 正常**（表现为 `col_reduce_sum(select(x == col_broadcast(max), 1, 0))` 恒返回 kk-1 而非真实并列数）。教训：这类"只有 GPU 错"的问题要**先打印生成的 GLSL/IR 再猜成因**。详见 `docs/development/08-pitfalls-and-lessons.md` §4.10。

## 评估分块（CNN 全量评估 OOM 修复，2026-09-20，原位置 AGENTS.md §12）

- 类型：已修复勘误
- 内容：`evaluate_mnist` 新增 `eval_batch`（默认 1000）分块前向 + 每块 `release_idle_pool_blocks()`；`mnist_train` 传 `cfg.batch_size`。根因：CNN/MLP 走 `N = x.cols()` 全量单次 forward，CNN 的 im2col 是 k²·C_in 倍 → 60000 样本需 ~6.4 GB → `vkAllocateMemory failed: -2`（训练步其实只多 130 MB）。见 `docs/development/08-pitfalls-and-lessons.md` §3.7。

## 复合层 checkpoint 模式坑（原位置 AGENTS.md §12「坑」）

- 类型：已修复勘误
- 内容：复合层 override `forward_recompute` 必须调用**虚函数** `set_checkpoint_mode` 关闭子层；基类默认实现只改本块标志位 → 子层缓存不重建（stride>1 时被上一轮陈旧缓存掩盖，表现为部分 stride 通过）。模式开关（checkpoint/offload/doc-mask）的缓存契约见 `docs/development/08-pitfalls-and-lessons.md` 模式 H。

## GPU 算子性能优化（2026-09-23；其 BK trade-off 表于 2026-09-24 被同参 A/B 推翻，原位置 AGENTS.md §12）

- 类型：性能 A/B（含被推翻结论）
- 内容：① `matmul_tiled.comp` 改 BK=16+双缓冲（共享保持 16KB 占用率不变），op 级 matmul +11~20%（教训：BK=32 双缓冲要 32KB → blocks/SM 砍半，流水收益被占用率损失抵消净 0）。② 融合生成器 `generate_glsl_matmul` 当时采用 BK=32+双缓冲（浅网格 linear 1024³ **-1.9%**、batch512 **-4.9%**，深网格 **±0**）——其"BK16 深网格真回退 -2%"的 trade-off 表后被 09-24 同参双缓冲 A/B **推翻**（见下条全算子压榨 ⑥，最终两侧同参 BK=16）。③ `submit_and_wait` 改 solo fence/cmd 复用 + `NN_GPU_PROFILE=1` stderr 分段计时，逐元素固定开销 0.16→0.143ms；余量 submit≈70µs + wait≈75µs 属驱动/唤醒延迟，kernel 本身 4096² 已达 370-440GB/s（峰值 448）**无优化空间**。④ bench 方法论：`layer_bench` 短跑必须 `--warmup ≥20` 等时钟爬坡（空闲 300MHz→1470→2070MHz，best-of 短跑双峰如 batched 4.57/5.9ms 即爬坡所致）；torch 用 CUDA event 只测 kernel、`layer_bench` 是 wall-clock 含提交开销；40HX 理论峰值 FP32 9.14 TFLOPS / 448 GB/s（34 SM×64×2×2.1GHz）。⑤ **batch=32 超线性异常（09-24 解决）**：train 侧残留 B² 项根因 = 融合 matmul 后端 dispatch `wg_y` 按**总行** rows 派发、z=batch 又数第二遍 → 工作量 ∝ BH²（生成器契约是 y 只覆盖批内行 `m_per = rows/mm_batch`，多派的 WG 被写回守卫整块丢弃——**结果正确，对拍永远测不出，只有计时能暴露**）。修复后 train batch=32 **2999.9→227.7ms（−92%）**、b16→b32 翻倍比 3.74→1.95 线性，batch=1 亦 −16%。教训：**dispatch 网格必须与生成器行界契约同源核对；"全绿"只证明算得对，不证明没空转**。

## fold 分块流式求值 + GPU 注意力全链优化（2026-09-24，7cdb41c+fd68573，原位置 AGENTS.md §12）

- 类型：性能 A/B
- 内容：注意力 forward 改单 fold kernel（`FoldSpec`：QKᵀ/掩码/online softmax/ΣwV 逐 `EXPR_FOLD_BLOCK=128` 块完成，S 绝不物化；`tri_skip`（当时字段名 `causal_skip`）整块跳过被屏蔽区；NR=2 每 WG 两行；subgroup 蝶式归约 + vecacc 4 路软件流水）；旧物化/两趟 forward 与 `build_attention_mask`/`mask_cache_`/`m/l/attn_cache_` 等死码删除。**实测 vs main 基线 7.9/20.3：mha fwd 5.41（−31.5%）、causal fwd 4.75（−40%）、train 17.5~17.7（−13%）**，18/18 绿。质量门禁轮：Doc/AlibiDoc fold 注册与测试矩阵全覆盖、ALiBi slopes 表 `(1, batch·H)` 越界修复、GEMV subgroup≥4 门禁、tri_skip bin v8 往返断言。

## 融合 matmul vec4 内核移植（2026-09-24，原位置 AGENTS.md §12）

- 类型：性能 A/B + 坑
- 内容：融合 matmul 同形状比 op 级 `matmul_tiled` 慢 2.3×（768×768×8192 fwd 6.18 vs 3.15ms——**"融合 vs op 级同形状对拍"是发现内核代差的 X 光**）；三重根因：内层 `acc+=dot(a[i],b[j])` 标量链（每 k ≈5× 指令发射）、全局 4×标量 ldg+16×标量 stg、共享 vec4 沿 k 布局。移植 op 级 v3 配方（共享沿 m/n + 每 k 2 LDS+4 VFMA.128 + vec4 别名快路径；trans 进 key → 分组生成期定死；BK=32/barrier 节奏/尾链不动）后：**linear 浅/深/投影三点 −33%/−48%/−47%（投影 6.18→3.25ms，达 op 级 97%）、mha fwd 4.59→3.63、mha train b1 13.0→10.8、b32 train 227.7→173.1（vs dispatch 修复前累计 2999.9→173 = −94%）**，18/18 绿。坑（rapt_offload 抓出）：共享索引必须用 tile 内**局部 k**，全局 k（`t*BK+local`）只进缓冲地址/守卫——混用则 t≥1（mm_k>32）越界共享 = 非确定性错值，num_tiles=1 的小形状对拍全测不出（全仓唯 rapt_offload d_model=64→2 tiles 有对照）；"对拍全绿但数值每跑不同"→ 先同二进制跑两次 diff。

## 全算子压榨 + OP/DSL 同步轮（2026-09-24，原位置 AGENTS.md §12）

- 类型：性能 A/B（含推翻 09-23 trade-off 表）+ bench 坑
- 内容：① `batched_matmul.comp` 移植 matmul_tiled v3（自 0.2.0 起未优化，同形 1024³ 比 matmul 慢 1.85×）：**b1 1.66→0.876ms（−47%）、b8 8.12→4.08ms（2116→4214 GFLOPS）**。② `transpose.comp` 重写 32×8 WG / 32×32 共享分块（tile[32][33]，读写双向 128B 合并、零 bank 冲突），backend dispatch 改 (ceil(C/32), ceil(R/32))：**4096² 1.575→0.492ms（−69%）**。③ `reduce.comp` 列归约重写（lane=列/warp=行块/32 列 tile + shared 跨块合并 + 4 路 ILP），max 恒等元 −inf→`lowest()`、行归约→subgroup 蝶式：**4096² 0.843→0.273（−68%）**；reduce 用 subgroup → CMake 需单独 `--target-env=vulkan1.2`（同 matmul_gemv）。④ DSL 列归约同结构重构（`generate_glsl_reduce`：dispatch cols/256→cols/32——CE 训练热路径旧 WG 并行度极低；合并值覆写回 s_red 与他 lane 读同址，**必须 3 barrier：partial 完成→读完→写完，否则 sum 双计**）。⑤ `elementwise_v2`/`broadcast`/`gather` → vec4（base=gid*4），三处 dispatch ÷4，broadcast 4096² −17%。⑥ **OP/融合 matmul BK 统一 A/B（推翻 09-23 结论②的 trade-off 表）**：OP 级 BK16→32 四点全回退（512³+9%、1024³+26%、4096³+29%、batched b8+29%）→ 维持 16；融合侧 BK32→16 反超 linear 浅 −24% / 深 fwd −24% / 深 train −17%（8.21→6.81ms）→ **两侧同参 BK=16**（09-23"深网格 BK16 −2%"是对单缓冲旧基线的口径，非同参双缓冲对比）。⑧ **bench 坑**：`layer_bench` transB/at 操作数曾恒建 (k,n)（应 (n,k)/(k,m)），k≠n → backend K mismatch → run 内 `*expected` 不查错 = UB → **0.000ms / 1.7e11 GFLOPS 超物理垃圾值**（k=n 正方形长期掩盖）；已改 per-variant setup + 错误检查 abort。教训：**bench 出现超物理数值先怀疑测法本身**。gather 快路径曾漏读 `indices[]`（把索引下标当表行号），zipt CPU/GPU 对拍抓出——vec4 改写后对拍必跑。最终 @1024：batched 0.871、matmul 0.853、col_reduce 0.123、transpose 0.138、broadcast 0.119；mha fwd 3.449 / causal fwd 2.819 / linear b4096 train 6.901ms。18/18 绿。

## transpose 砖块化 + reduce 两段式 + bench 记账修复（2026-09-24 二轮，原位置 AGENTS.md §12）

- 类型：性能 A/B + 方法论
- 内容：① **bench 虚高记账修复（"部分算子超出 40HX 理论"的根因）**：`bytes_reduce` 曾按 2N 记归约（输出仅向量级，实际 ≈N）→ 5244² 虚报 440 GB/s、4096² 虚报 491（超 448 理论峰值，物理不可能）；改精确 (N+out) 后真实 217-263 → **reduce 其实有一倍头寸**。② `transpose.comp` 定稿：64×64 tile（段宽 128→256B）+ 共享落位交换 + **8×8 WG 砖块化 dispatch**：5244² 0.99→0.75ms（−24%）、4096² −17~28%。消融链定根因：纯拷贝 0.556 → 过 shared+barrier 0.665 → 线性 dispatch 0.850，**真凶 = 并发 WG 写侧足迹散布整块**（转置输出行=输入列：线性 dispatch 下 136 个并发 WG 每行只写 256B、行距 21KB ≈180MB 散布）——砖块化让连续 64 WG 填满一砖、足迹收敛 ~1MB 聚集；16×8 砖与 8×8 持平。③ `reduce.comp`：行归约 vec4 快路径（cols%4==0）；列归约**两段式 partials**（ggml `rms_norm_partials` 模式：pass1 行向切 nchunk → partials(nchunk,cols)，pass2 复用整表合并；同 cmd 双 dispatch 单次提交 → 固定提交开销不翻倍；scratch 走成员 `reduce_partial_` 防录制期 UAF）：5244² 0.50→0.38-0.42ms；push 扩 5×uint=20B（`chunk_rows`），**pipeline layout push range 同步 4→5 个 uint——漏改即静默错值**；512K 元素护栏排除微型形状（512² 两段 −16% 净亏）。④ `broadcast.comp` vec 行单 vec4 装载 5244² −13~16%。⑤ **方法论（调参/判退前必读）**：同码对照两 exe 同窗交错实测 **噪声地板 ±5%、跨会话系统漂移 ±15%**（round2 全面慢 5-10%）——两段式曾被跨会话单点误判"4096² 净亏 20%"，交错 A/B 实测只亏 2%（SP 5/5 配对胜）、5244² 反赢 6-9%（TP 4/5）才定案保留；**调参/判退一律同窗交错 + 配对统计，禁止对着跨会话单点改参数**。18/18 绿。

## 多精度 Phase 2：边界 cast → in-kernel f16 → op-level f16 GEMM（2026-09-25，原位置 AGENTS.md §12 第 ⑭ 大条，docs 05 §12.5-§12.11）

- 类型：性能 A/B + 演进记录
- 内容（按阶段保留关键数字与结论）：
  - **接线**：引擎运算类原语加 `Precision P = F32`、新增 `cast_into`/`copy_into`；新增 `PrecisionEngine` 适配层（f16 边界 cast 集中一处，全 f32 = 纯直通逐字节一致）；`dsl::compute/compute_reduce` 加 P；Layer/Loss/Optimizer/工厂/CLI 全链接线。教训：**复合层必须把 profile 下传给子层**（`AttentionBase`/`FeedForward`/`TransformerEncoderLayer`——不下传则子层静默停 f32；`Model::add` 用未设默认值覆盖 GPTModel 构造器已注入的 profile 曾导致 token_emb 退回 f32）。
  - **`--f16` 语义** = `profile_f16()` = {param:F16, compute:F16, stable:F32, optimizer:F32}；**四字段全 f16 不可训练**（optimizer=F16：Adam 的 v≈g²~1e-10 下溢→更新爆炸 loss 7.9→3.6e4；stable=F16：CE 链 ~200 步 NaN；两者同 f16：loss 恒定）。
  - **边界 cast 结论**（40HX，GPT d64/h4/L4/ff256、vocab 8208、seq256）：f32 峰值 3069MiB；f16 边界 cast 峰值反而升高（2831MiB 后 backward OOM，探针 transient live 2795→6687MB）→ **边界 cast 只能拿"存储减半"，拿不到"峰值下降"；须做 in-kernel f16**。
  - **in-kernel 第一期**：`ExprPrecSig`（bit i = 第 i 个输入 f16、bit16 = 输出 f16，全 0 = 旧行为）+ `ExprRegistry::variants/add(spec,sig)` + bin v9 只存 `{sig, 基础结构下标}`；`scan_exprs` 双 pass（**f16 pass 必须经适配层——原生 CpuEngine 喂 f16 = heap corruption 0xC0000374**）；`GlslEmitter::generate(name,spec,sig)`（`RotateHalf` 取负 / `RowGather` 的 `uint()` 必须在叶子处转，否则 glslc 报错）；`run_fused_gpu` 输入 `FusedInputs{owners,bufs}` **必须同时持有 owner**（只存裸指针 = 录制中途释放上传缓冲，实测 `A+=B err=0.5`）；f16 输出按 2B/元素分配并重贴 `GpuTensorF16`（漏了 → loss=NaN）。实测 batch32：f32 1595 / f16 in-kernel 3457 / 边界 cast 4124MiB（f16 仍 2.2×）。**签名必须逐输入**：真实 GPT f16 变体空间 38 个 = 逐元素/归约 32 + matmul 段 5 + fold 1，存在混合签名。
  - **归因演进（教学点）**：⑪ 判"边界 cast 临时块数量膨胀"→ ⑫ 判"池底材粒度"（阶梯≤16MB −21% 但 2× 耗时）→ ⑬ 同窗复测**推翻 ⑫**（峰值 3513/3513/3513 持平、耗时全 ≈5.7s，"2× 时间"是跨会话单点产物，池小分配快路径工程项作废——池账本 1 万次 allocate 只扫 ~5 万 block、`vkAllocateMemory` 累计 ~150ms，本就不是瓶颈）+ 新增形状级归因表（`dump_temp_stats`，`NN_PREC_TRACE=1`）定罪 **op-level cast**：`(32768,256)` 注意力反向 W/grad_A 占 **1152MB**，其次 `(64,8192)` 270、`(256,8192)` 160、`(2048,256)` 112MB。**机制之争必须用形状/尺寸归因表裁决，别用总量推理。** 据此补三类带类型变体（matmul 段 `float16_t`+`uvec2`+`unpackHalf2x16` / 归约 kernel `rd()/wr()` 读写点转换 / `eval_expr_into|eval_expr_reduce` 加 `(key,sig)` 匹配）→ `gen_fused` 由 **41 变体/14 skip → 54 变体/0 skip**；实测同窗 3 轮：f32 1753（3/3 一致）、f16 2025/2025/2273 = 1.15~1.30×（此前 3513，即 cast 路径 **−35%~−41%**），耗时无回退，batch64 f16 不再 OOM。
  - **op-level f16 GEMM（第 ⑭ 阶段）**：手法 = 一份 .comp 用 `glslc -DNN_SHADER_F16=1` 编第二份 SPIR-V（`NN_ETYPE=float16_t`/别名槽 `NN_V4=uvec2`/`NN_LOADV4=unpackHalf2x16`，**f32 分支留在 `#else` 逐字未动 = 零回归证明：头文件字节数不变**（`matmul_tiled_spv.hpp` 68670 / `batched_matmul_spv.hpp` 74408））；后端 pipeline 只在 `has_16bit_storage()` 时创建；**适配层必须把 `P != F32` 直接下传**（第一版漏了这步 → 归因表逐项不变、峰值纹丝不动）；`CpuEngine::matmul_with_bias` 补下传 `P`（否则 scan 的 f16 dry-run 只登记全 f32 签名；变体 54→56）。实测同窗 3 轮（batch32/steps2）：f32 1754×3、f16 1570×3 = **−10.5% 显存且略快**（5.05 vs 5.11s）；batch64 亦 −10.5%；`text_train --f16` avg_loss 6.7493 vs f32 6.6979（0.77%）。**演进全景：f16 边界 cast 4124 → in-kernel 首期 3513 → +matmul段/归约/目标传递 2025~2286 → +op-level GEMM 1570**；剩余约 550MB cast（`(2048,256)×80`、`(64,8192)×72` 等）+ 7 条 run-only 签名 + fold 变体，收益已递减。

## CPU 侧 f16 训练发散双根因修复（2026-09-25，原位置 AGENTS.md §12 末条，docs 05 §12.12）

- 类型：已修复勘误（根因 + 修法 + 测试盲区）
- 内容：§12.5"已知问题 1"（CPU f16 5~7 步 NaN、未定位）实为**双缺陷叠加**。
  ① **DSL 预绑定喂错精度**：`MatmulRef::prepare_cpu` 固定以 P=F32 物化 C，而 `CpuEngine::matmul` f32 路径按 f32 存储直读 f16 张量 → `cpu_matrix<F32>()` 空指针 UB（Debug 断言 `tensor has no P-precision CPU storage`；Release 即 `text_train --f16` CPU 0xC0000005）；同族：`ReduceViewRef::prepare` 直读、适配层 `matmul/batched_matmul` 把 `P!=F32` 无条件下传、CPU 解释器 `eval_expr_impl` 无精度校验。修复 = prepare 先抬 f32 再物化 / f16 输入一次性镜像 / `CpuEngine::matmul|batched_matmul` 精度与存储不匹配时回退 f32 空间计算 + 按 P 舍入（两端同 f16 才走原生 f16 GEMM）/ 解释器加精度 `NN_REQUIRE`。
  ② **`float_to_half_bits` 次正规分支移位 UB（真正的 NaN 制造机）**：守卫 `exp <= -46` 与其自身公式矛盾（只对 exp∈[-25,-15]/shift 14..24 成立）→ **exp∈[-45,-33]（|v|∈2.8e-14~1.2e-10）shift≥32 移位 UB** → 小值被转成垃圾 half（0x4000=2.0 / 512 / 8192 / 11776 / 18432，随机落 e=31 即 NaN）；**梯度正是这个量级、前向激活 ~0.1 从不落入** → 损坏只见于 f16 梯度写回；GPU 走硬件转换故"只有 CPU 错"；单算子测试输入 ≥1e-3 故"全过"。修复 = 守卫改 `exp <= -26`（≤-26 按 D<0.5 恒 flush 0）。
  ③ **测试盲区**：`ref_ulp` 写成 `2^(e-10)`（应 `2^(e-25)`，**大 2^15 倍**）→ RHE 容差比被测值本身还大、垃圾值全放行；修复 ref_ulp + 补 exp∈[-60,-26] 全网格"必须恒 0"断言 + 两个历史垃圾代表值定点断言。
  ④ **方法论**：新组件探针 `f16_cpu_probe`（阶段 A–I）+ `NN_F16_DEBUG=1` 层内逐中间量扫描——把故障钉到 `accumulate: pre_dst=0, pre_src=8.7e-13, post_dst=18432` 单步即锁定转换函数。修复后 CPU f16 三组 profile 6 步轨迹与 f32 **逐位一致**；`f16_precision_test` 容忍分支改**硬失败**；profile 单字段矩阵仅 optimizer=F16 仍发散（§12.5 数值性限制，与本缺陷无关）。

## 基准报告更新与 f32 发散勘误（2026-09-25/09-26，原位置 AGENTS.md §12「已知问题」）

- 类型：已修复勘误 / 性能 A/B
- 内容：**f32 训练发散 NaN 已修复**——09-26 实测短测 3 epoch + 长测 5 epoch 健康收敛，09-25 基准记录的"f16/f32 均发散"在 f32 上不复现。**09-25 基准报告 nn 侧全量重跑**（`docs/benchmarks/2026-09-25-vulkan-vs-cuda.md`）：f32 优先主对比 101.0 ms/step；transpose 3.7×→2.3×（P0 静默半写虚报修正）；`elementwise_exp`/`broadcast_col` 已删标注；f32 显存 5743→3069 MiB。

## S7 关键教训中的过程叙述（原位置 AGENTS.md §12「S7 关键教训」第 5、6 条）

- 类型：演进记录（约束本体保留在 AGENTS §12「融合二期状态与 IR 编码约束」）
- 内容：matmul + 列归约于 **2026-09-26 补齐**（`generate_glsl_reduce` 列分支按元素分解 batch、`gen_fused` 三处跳过删除）。注意力 forward 曾为 m/l/W 表达式 + bm(W,V_t) 的 S7 IR 链结构（**已删**），现为单 fold kernel（`FoldSpec`，5 掩码变体经 `fold_mask_variant_`）；bwd = R/X 表达式 + 3 个 `batched_matmul`。

## 「已过时的历史说明」整节（原位置 AGENTS.md §12 末节）

- 类型：删除清单
- 内容：五条——① **全库代码审查报告（2026-09-04）**：一次性审查文档已删除（git 历史可查）；`--tdr-retry`/`--max-tdr-retries` 选项已从代码移除（当年"被解析但从未使用"的问题不复存在——当前结论保留于 AGENTS §12）。② **CPU 性能诊断报告与路线选择建议（2026-09-18）**：临时诊断/决策文档已删除，结论并入 `12-compute-engine-inventory.md`（CPU 侧求值机制 2026-09-26 起只剩两套：DSL 模板路径 + IR 解释器）与 `03-ir-optimization.md` §5.3（IR-C 取舍）。③ **CpuEmitter**：`cpu_emitter.hpp` 与实现已删除，早期"待修 CpuEmitter 隐性缺陷"问题随之消失勿引用（当前表述：AGENTS §7 仅 `glsl` 后端注册，无 CpuEmitter/CudaEmitter）。④ **融合三期 S6**：未列入当前计划；替代方案 P2-12 图级缓存随 IR-C 于 2026-09-19 删除。⑤ **IR-C**：已评估并整体移除（删除项见上文 IR-C 条目）。

## 其余移除叙述与过期事实订正（原位置 AGENTS.md §2/§3/§4.3/§5/§7/§8/§10/§11）

- 类型：删除清单 / 已修复勘误
- 内容：
  - **§4.3「2026-09 收敛」块**：`axpy_inplace`、`broadcast_row/col_inplace`、`elementwise_unary/binary/binary_scalar`、`elementwise_select_scalar_cond`、`row_reduce_max`（引擎算子）、`offload_store/offload_load` 以及 `UnaryOp/BinaryOp/CompareOp` 枚举已全部删除（逐元素/广播一律走表达式 DSL；`dsl::row_reduce_max` 是 DSL 叶子）；**引擎 virtual 58 → 49**。
  - **§5 铁律 5 历史叙述**：历史上 GPT 曾用 position-major（`i = t*batch + b`）导致跨样本串扰的灾难级 bug，现已统一为 batch-major。
  - **§10 坑 7 历史数字**：BPE"所有出现位置"类索引曾膨胀到 60-80GB；大词表 one-hot 曾 3.2GB。
  - **§7 移除叙述**：跨表达式融合（`start_expr/end_expr`、`begin_expr/end_expr`、`expr_graph.hpp`）2026-09-19 移除；fold 构造 2026-09-25 前在 `expr_fold.hpp`（后按 AOT 原则归位 `compute_layer_attention.hpp`）；`FoldSpec.tri_skip` 原名 `causal_skip`；`dsl::grouped_reduce_sum/max` 2026-09-27 新增；`broadcast.comp` 随 `broadcast_*_inplace` 于 2026-09 删除；`CpuEmitter`（已删除）/`CudaEmitter`（随 CUDA 移除）勿引用。
  - **§3 目录表括注**：旧代数 AST（`algebra_expr.hpp`/`algebra_compute.hpp`）2026-09 移除、`Expression`/`BoolExpression` 迁入 `expr_dsl.hpp`；`expr_graph.hpp`/IR-C 2026-09-19 移除。
  - **§8**：RAPT "自 2026-09-19 起"与 GPT 同档（当前：RAPT 与 GPT 同档）；链式构建伪 API 见下条。
  - **§11 索引过期事实订正**：`development/12` 描述由"49 个 virtual（原 58，删 9 个）、**Layer 直调 32 个**"订正为实测 **23**（`bench/doc_inventory.ps1`：virtual 49/49、layer-called 23、not layer-called 26）；"后附 2026-09 收敛记录"→ 演进记录已移出（本文件/w4-dev1）；`development/13` "含误报/已修复对照表（§6）"→ 对照表已移入 `docs/history.md`；`development/03` "§5.3 是取舍记录"→ §5.3 现只指向 `docs/history.md`；`introduction/01` "CUDA 已移除备注见篇末"→ 该文档已无此备注；`development/04` "已实施"、`development/02` "删手写原语"等状态词与历史词改为当前形态描述；新增 §11 总档行 `docs/history.md`。

## §8 链式构建伪 API（`add_linear`/`add_relu`，原位置 AGENTS.md §8）

- 类型：已修复勘误
- 内容：原文写 `model.add_linear(784,256).add_relu().add_linear(256,10)`——**全仓不存在 `add_linear`/`add_relu`**（grep 仅命中 AGENTS.md 自身）。`Model::add` 是模板方法 `template <typename LayerType, typename... Args> Result<void> add(Args&&...)`（`include/neuralnet.cpp/model_container.hpp:71-72`），无链式 API，层构造后自动 `init(engine)`。已改为逐层写法 `model.add<nn::Linear>(784,256); model.add<nn::ReLU>(); model.add<nn::Linear>(256,10);`，与 `docs/usage/01-quickstart-model.md:19-21` 一致。
