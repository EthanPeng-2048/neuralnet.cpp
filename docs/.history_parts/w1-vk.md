# w1-vk

GPU 后端 / GPU 引擎头文件「历史状态类注释」摘录。整改原则：注释只记录当前实现的原理、契约与理由；
本文件保存被移出的历史叙事（bug 根因、被否决方案、性能 A/B 过程、删除清单），供追溯用。

## solo fence / command buffer 复用的由来（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:12）
- 类型：删除清单
- 内容：旧实现每个算子各自 vkCreateFence/vkDestroyFence + vkAllocateCommandBuffers/vkFreeCommandBuffers，
  是逐元素算子固定开销的主要构成之一；现改为 initialize() 预分配 solo fence + solo cmd，submit 前 reset 复用。

## 单帧阻塞提交模型（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:559-562）
- 类型：否决方案 / 删除清单
- 内容：旧实现是「单一 batch_cmd_ + 单 fence」，end_batch/flush_batch 每次 vkQueueSubmit 后立即
  vkWaitForFences —— GPU 执行 step N 时 host 被 fence 卡死、无法录制 step N+1。原注释引用的
  《显存 & 负载不均衡分析报告》§2.1 在仓库中不存在（断链，整改时删除）。现为 PIPELINE_FRAMES=6
  的 (cmd + fence) 帧环，提交不等待、wait_in_flight 才阻塞。环深度 3→6 的加深记录：3 槽时
  host 录制与 GPU 执行重叠度不足。

## 逐元素算子固定开销（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:617）
- 类型：性能 A/B
- 内容：改造前逐元素算子固定开销实测 ≈0.16ms/次（kernel 本身 4096² 已达 370GB/s，瓶颈全在 host 侧）；
  solo fence/cmd 复用后降至 ≈0.143ms（AGENTS §12 同一改动的口径）。整改后注释只保留当前值 ≈0.14ms。

## 帧环预分配（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:1304-1305）
- 类型：删除清单
- 内容：旧实现每步（step 边界）vkAllocateCommandBuffers + vkCreateFence；现为 initialize() 预分配
  PIPELINE_FRAMES 个 cmd + fence，全程零分配。

## run_fused_gpu 固定头长度 bug（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:3495-3498）
- 类型：bug 根因
- 内容：「归约但无 matmul」形态曾按 5 个 uint 计算固定头（真实头部只有 4 个）→ push constant 常量池
  整体后移一个 uint → shader 从错位处读常量，实测把 select(cond,1,0) 的常量读成垃圾，
  **GPU 上带常量的归约静默错值、CPU 正常**（表现为 col_reduce_sum(...) 恒返回 kk-1 而非真实并列数）。
  教训：这类「只有 GPU 错」的问题先打印生成的 GLSL/IR 再猜成因。

## fold PC 漏分支 → 未定义残留（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:1727-1728）
- 类型：bug 根因
- 内容：pipeline 创建侧（pc_base 计算）漏 fold 分支时 range 少算，vkCmdPushConstants 超 range 部分被
  驱动丢弃 → fold_k 读未定义残留，**曾致滑窗式错值**，且残留随前序 op 漂移、表现为时对时错的假 PASS
  （对拍全绿）。整改后注释改为直接陈述该失败模式与「创建侧/写入侧必须同改」的契约。

## matmul dispatch 曾用 WORKGROUP_SIZE=16 统一计算（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:2560-2562）
- 类型：bug 根因 / 性能 A/B
- 内容：曾按 WORKGROUP_SIZE=16 统一计算工作组数 → tiled 版（64×64 块）多派 16 倍冗余工作组
  （每 16×16 一个组），GPU 做 16 倍无效计算，实测 matmul 峰值只剩 ~4%（0.7/15.7 TFLOPS）。

## transpose 曾误派发 16 宽砖（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:4602-4603）
- 类型：bug 根因（issue #13 P0-①）
- 内容：shader 按 8 宽砖解算 tile_c，dispatch 曾给 x=16 → tile_c 跨两砖且越界 tile 只早退一半，
  行>512 且列>512 时静默只写前 512 行。

## 物理设备选择：「第一个 DISCRETE_GPU」（原位置 include/neuralnet.cpp/backend/compute_vk_device.hpp:184）
- 类型：否决方案
- 内容：旧规则「取第一个 VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU」在本机枚举顺序下会选中 [0]
  AMD Radeon R5 240（老专有驱动，maxComputeSharedMemorySize 仅 32768 < matmul 分块所需 34560，
  校验层报 VUID-RuntimeSpirv-Workgroup-06530；且对跨 submit 信号量重复 wait 直接死锁）。
  现为「设备类型权重 + apiVersion 打分、同分取先枚举者、D3D12 转译层降权」。

## 上传 region 专属 cmd 与「submit 后立即 free」（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:2104-2110、compute_staging_ring.hpp:80-83）
- 类型：bug 根因 / 删除清单
- 内容：旧代码每次上传 vkAllocateCommandBuffers、submit 后立即 free/复用，违反
  VUID-vkFreeCommandBuffers-pCommandBuffers-00058（禁止释放 pending 的 command buffer）；
  实测导致驱动通道排序失效、fence 提前 signal、跨 submit 乱序执行。

## staging ring 初版预分配（原位置 include/neuralnet.cpp/backend/compute_staging_ring.hpp:9-13、47）
- 类型：性能 A/B / 删除清单
- 内容：旧默认 4×256MB=1GB 预分配过大；V100 上按「整机 host heap 1/16」规则曾达 2×2GB=4GB host 预算。
  现默认 4 个 region、每 region 取 max(用户请求, clamp(host_visible/16, 32MB, 256MB))，
  且单次 PCIe 传输量通常 << 64MB（最大 token_emb 上传约 5MB）。
  注：文件头原写「默认 2 个 region / 2 regions 共 512MB」，与 DEFAULT_NUM_REGIONS=4 不符，一并订正。

## 列归约两段式的 A/B 方法论（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:3713-3717）
- 类型：性能 A/B
- 内容：同负载窗交错 A/B 定案方法：TP/SP 两二进制逐轮换序、warmup 30、best-of-10、同码对照噪声地板
  ±5%；结果 5244² TP 4/5 胜（中位 0.405 vs 0.430，−6%）、4096² SP 5/5 胜 ~2%（0.277 vs 0.283）、
  512² TP −16%（对 ~10µs kernel 第二遍+屏障净亏）→ 512K 元素尺寸护栏只排除微型形状。
  整改后注释只保留三点实测数字与护栏结论。

## eval_expr_into 的 output_override 曾写死 F32（原位置 include/neuralnet.cpp/compute_gpu_engine.hpp:1639-1645）
- 类型：bug 根因（issue #13 P0-②）
- 内容：曾写死默认 F32 的 gpu_tensor()：f16 目标 → gpu_get<F32>() 空 shared_ptr → Debug NN_ASSERT 引爆；
  Release 空解引用 = UB：MSVC 得到空 override → 结果落临时 buffer 被丢弃 → **参数静默冻结
  （loss 恒 ln V）**；clang 当时「正常」只是 UB 代码生成运气。修复 = f16 目标经 f16_view 包 f16 buffer。

## 显存底材 128MB → 12MB 调参（原位置 include/neuralnet.cpp/backend/compute_memory_pool.hpp:42-52）
- 类型：性能 A/B
- 内容：2026-09 探针实测调参：峰值对块尺寸在 8–32MB 是平台区、最优 ≈12MB 且与模型规模无关
  （跨负载表保留在注释中）；「最小 2 的幂」阶梯类（NN_POOL_LADDER_MAX_MB）实测 d128 上
  ladder16=4497 比 fixed16=4457 差 40MB（配对 3/3），故阶梯默认关闭。

## 真原地语义与 cpu_fallback 的删除（原位置 include/neuralnet.cpp/compute_gpu_engine.hpp:25、466）
- 类型：删除清单
- 内容：2026-09 起引擎内 add_inplace/scale_inplace 等全部改为真原地（此前 copy-on-write 描述已作废）；
  GpuEngine::insert_rows 的 cpu_fallback（CPU 目标回退路径）已删除，CPU 目标直接报错。
  整改后注释只陈述当前契约（真原地的 read-before-write 论证、dst 必须为 GPU Tensor）。

## 收割点提前的探针数据（原位置 include/neuralnet.cpp/backend/compute_vk_backend.hpp:1850-1854）
- 类型：性能 A/B
- 内容：原先 reap_completed_frames 只在 step 边界调用 → 步内已析构中间张量的延迟销毁挂在
  「已完成但未被收割」的帧上不还池；bench 配置实测 transient live 3837MB 里 pending 高达 3000MB，
  步内峰值 ≈ 稳态 2.8×。现收割提前到每个帧提交点（submit_frame_no_wait）。
  整改后注释保留数字与「收割必须紧随提交点」的结论，去掉「原先/2026-09 起」的演进叙事。

## 精度归因编号（原位置 include/neuralnet.cpp/compute_gpu_engine.hpp:530、989、1272）
- 类型：性能 A/B
- 内容：曾用探针归因编号描述 f16 带类型变体的收益来源：scatter_add 的 (8208,64)+(8192,64)、
  add_inplace 的 inplace2_ 大头、reduce 的 (64,8192)×38（Linear::backward grad_bias）。
  这些编号依赖一次性探针输出，整改后改为直接陈述「免去 f32 副本物化」这一当前设计收益。
