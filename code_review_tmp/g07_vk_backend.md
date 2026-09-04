# Vulkan 后端（L0 硬件层）代码审查

## 模块概览

`compute_vk_device.hpp`（408 行）提供 `vk_check`/Scalar 转换辅助、`VulkanDevice`（实例/物理设备/队列 RAII）与 `VulkanPipeline`（shader module + descriptor/pipeline layout + compute pipeline RAII，含 `create_matmul`/`create_generic` 两个工厂）。`compute_vk_backend.hpp`（3132 行）是 GPU 后端的实际实现：`GpuBuffer`/`GpuTensor` RAII 封装、`GpuBackend` 单例（延迟泄漏，nn-allow 已知债）负责 pipeline 注册、descriptor pool（262144 set / 1M desc）、command pool、StagingRing、batch 模式（`begin_batch`/`end_batch`/`flush_batch` 三态命令录制）、14 个手写原语 + AOT 融合 shader 的 dispatch 路径、`pending_destroys_` 延迟销毁队列（D1 修复后内存归还与 buffer 销毁同延迟）、以及 `matmul_direct` 等 staging 路径。`GpuBuffer` 析构在 batch 模式下将 buffer 与区间一同推入 `pending_destroys_`，由 `end_batch`/`flush_batch` 提交完成后 `flush_pending_destroys()` 统一 `vkDestroyBuffer` + `pool->free`。

## 发现

### [P0] 正确性 / 崩溃 / UB / 设备丢失

（本组无新增 P0；已知未修项见"已知问题核对"节。）

### [P1] 生命周期 / 内存 / 并发 / 错误传播断裂 / 规范违反有实际影响

**P1-1** compute_vk_backend.hpp:933-1032（`end_batch`）— **`vkResetFences`/`vkQueueSubmit` 返回码被覆盖，错误传播断裂**。

```cpp
// 2. 提交
vkResetFences(device_.device(), 1, &batch_fence_);   // 返回码丢弃
...
r = detail::vk_check(vkQueueSubmit(...));              // 成功分支
// 3. 等待完成
VkResult wait_result = vkWaitForFences(...);           // 覆盖 r
...
return r;                                              // 1031：返回的是 vkQueueSubmit 的结果
```

`vkWaitForFences` 的结果赋给 `wait_result` 局部变量，`r` 自此停留在 `vkQueueSubmit` 的成功值；末尾 `return r`（line 1031）实际返回的是 submit 的结果而非 wait 的结果。VK_SUCCESS 路径下无影响，但 `vkWaitForFences` 在 wait 成功之外的路径（例如 `VK_ERROR_DEVICE_LOST` 之外的 `VK_INCOMPLETE`、负值错误）被"成功"地吞掉——`wait_result != VK_SUCCESS` 分支返回了 `std::unexpected`，但**若 `vkWaitForFences` 返回 `VK_TIMEOUT`（正值），会进入 `else if (wait_result != VK_SUCCESS)` 分支返回错误，但函数末尾的 `return r` 路径只在 `wait_result == VK_SUCCESS` 时可达**，因此实际语义正确但代码结构误导，且 `vkResetFences` 的失败（例如 fence 已 in-use）被静默丢弃——按铁律 1 应检查。建议：`vkResetFences` 走 `vk_check`；wait 成功后再 `return {}`，避免 `return r` 复用旧值。

**P1-2** compute_vk_backend.hpp:1178-1474（`upload_blocking`/`download_blocking` 分块路径）— **`in_flight` 标志从未被置位，staging region 环回绕可能覆盖在飞数据**。

`StagingRing::acquire()`（compute_staging_ring.hpp:180-194）依赖 `in_flight` 决定是否需要等待该 region 的 fence；`mark_in_flight()`（line 243）在整个代码库中**只被定义、从未被调用**。`upload_blocking`/`download_blocking` 在分块路径下每次循环 `staging_ring_->acquire()` 取 region，提交后**不标记** `in_flight = true`，下次环回同一 region 时 `acquire()` 因 `in_flight == false` 直接返回——若上一轮 `vkWaitForFences` 尚未真正完成（超时被忽略或设备重置场景），新数据会覆盖 GPU 仍在读取的旧 region。即使当前 10s 超时足够宽裕使实际窗口极小，仍是潜在 use-after-free 源（铁律 5 录制生命周期）。建议：提交成功后调用 `mark_in_flight(ri)`，并在 `acquire()` 等待 fence 后置回。

**P1-3** compute_vk_backend.hpp:1200-1251（`upload_blocking` 快速路径）/ 1345-1399（`download_blocking` 快速路径）— **`vkWaitForFences` 的 10 秒超时可返回 `VK_TIMEOUT`，被 `vk_check` 视为错误但实际 GPU 可能仍在工作**。

```cpp
r = detail::vk_check(
    vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, 10'000'000'000ULL),
    __FILE__, __LINE__);
```

`vk_check` 只放行 `VK_SUCCESS`，`VK_TIMEOUT`（正值，GPU 仍在跑）被当错误传播，但 GPU 端 copy 命令仍会完成——调用方收到错误后放弃数据，却数据仍可能在飞行中，staging region 被 `acquire` 复用（见 P1-2）会踩在飞数据。建议：`VK_TIMEOUT` 时延长等待或重试，或改为无限等待（staging 传输量小、TDR 风险低），或在错误路径上明确标记 region 为 dirty 并等待。

**P1-4** compute_vk_backend.hpp:1038-1176（`flush_batch`）— **错误路径上 `batch_desc_sets_` 被释放，但 `pending_destroys_` 未与 desc sets 同序处理；且新 command buffer 分配失败时直接 `batch_mode_ = false` 不返回错误给上层，破坏 batch 契约**。

```cpp
// 5. 分配新的 command buffer
r = detail::vk_check(vkAllocateCommandBuffers(...));
if (!r)
{
    batch_cmd_ = VK_NULL_HANDLE;
    batch_fence_ = VK_NULL_HANDLE;
    batch_mode_ = false;
    return std::unexpected(r.error());   // 正确：返回错误
}
```

错误路径本身返回了 `std::unexpected`，但 `batch_mode_` 已置 false——上层若未检查返回码，后续 `matmul_gpu` 等调用会以非 batch 模式独立提交（每 op 一次 fence wait），性能退化但不崩溃。真正的问题是 **`vkEndCommandBuffer` 失败分支（line 1045-1053）未释放 `batch_desc_sets_`**：录制中途失败时这些 descriptor set 永远留在 pool 中，pool 耗尽后 `alloc_desc_set` 返回 `VK_ERROR_OUT_OF_POOL_MEMORY`，后续 batch 全部失败。建议：`vkEndCommandBuffer` 失败路径补充 `vkFreeDescriptorSets`。

**P1-5** compute_vk_backend.hpp:1661-1706（`scan_prefix_outer_gpu`）— **`A0`/`B0`/`boundary` 在 `has_state=false`/`has_bnd=false` 时可能为 `GpuTensor()`（无效 buffer），但 `dispatch_compute` 无条件 `inputs[i].buffer().impl()` → 解引用空 `shared_ptr` UB**。

```cpp
std::vector<GpuTensor> inputs{K, V, P, R, A0, B0, boundary};
auto r = dispatch_compute(scan_prefix_outer_pipeline_, inputs, C, pc, 1u, BH, 1u);
```

`dispatch_compute`（line 1500）对每个 input 执行 `inputs[i].buffer().impl()`，而 `GpuTensor::buffer()` 返回 `*buffer_`（line 321），`buffer_` 为 `shared_ptr<GpuBuffer>`，**空 `GpuTensor` 的 `buffer_` 为 null，解引用 UB**。`scan_prefix_outer_gpu` 的签名接受 `const GpuTensor& A0/B0/boundary`，调用方在 `has_state=false` 时大概率传默认构造的 `GpuTensor()`。同样的问题存在于 `scan_suffix_outer_gpu`（boundary，line 1743）和 `outer_col_gpu`（S，line 1779）。建议：`dispatch_compute` 增加"无效 buffer 绑定 A 占位"逻辑（与 `elementwise_v2_gpu` 的占位策略一致，line 2005-2010），或要求调用方传入有效 dummy tensor 并在此处校验。

**P1-6** compute_vk_backend.hpp:2265（`run_fused_gpu` matmul 分支）— **`vkCmdDispatch(cmd, wg_x, wg_y, matmul_batch)` 未校验 `matmul_batch` 上限，`VK_WHOLE_SIZE` 描述符 + 大 z 值可能超出硬件 maxComputeWorkGroupCount 限制**。

GTX 850M 的 `maxComputeWorkGroupCount` 通常为 `{1024, 1024, 64}`（z 维上限 64）。`matmul_batch` 是 `uint32_t`，若调用方传入 `matmul_batch > 64`（例如 batch_size=128 的 GPT），`vkCmdDispatch` 返回 `VK_ERROR_VALIDATION_FAILED`（debug layer）或静默截断/UB（release）。建议：dispatch 前校验 `matmul_batch <= device props.maxComputeWorkGroupCount[2]`，超出则硬报错。

**P1-7** compute_vk_device.hpp:110-121（`VulkanDevice::initialize`）— **`vkCreateInstance` 与 `vkEnumeratePhysicalDevices` 失败路径泄漏 `instance_`；`vkEnumeratePhysicalDevices` 返回码未检查**。

```cpp
VkResult res = vkCreateInstance(&instance_info, nullptr, &instance_);
if (res != VK_SUCCESS)
    return std::unexpected(Error{...});          // instance_ 未设（OK）

uint32_t device_count = 0;
vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);   // 返回码丢弃
if (device_count == 0)
    return std::unexpected(Error{"No Vulkan devices found"});   // instance_ 泄漏

std::vector<VkPhysicalDevice> devices(device_count);
vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());   // 返回码丢弃
```

`vkEnumeratePhysicalDevices` 可能返回 `VK_INCOMPLETE`（正常，允许）或负值错误（设备不可用），均未检查。`device_count == 0` 路径下 `instance_` 已创建但函数直接返回错误，析构函数（line 78-84）不会执行（`instance_` 是成员，对象仍存活，但 `initialize()` 返回后调用方若不再使用该对象则泄漏到进程结束）。建议：检查 `vkEnumeratePhysicalDevices` 返回码；`device_count == 0` 路径调用 `vkDestroyInstance` 后再返回。

**P1-8** compute_vk_backend.hpp:2454-2455（`broadcast_gpu`）— **`rows * cols` 以 `uint32_t` 计算，无溢出检查**。

```cpp
const uint32_t total = rows * cols;
const uint32_t wg_count = (total + 255) / 256;
```

`rows`/`cols` 均为 `uint32_t`（line 2395-2396），`rows * cols` 在 32 位下溢出回绕。`run_fused_gpu` 有溢出检查（line 2122-2124），但手写原语路径（`broadcast_gpu`、`rearrange_3d_gpu` line 2485-2486、`gather_gpu` line 2960、`scatter_add_gpu` line 3064、`transpose_gpu` line 2878）均无。建议：统一用 `rows > 0 && cols > UINT32_MAX / rows` 检查，或改用 `uint64_t` 中间变量。

### [P2] 中等风险 / 设计缺陷（≤10 条）

**P2-1** compute_vk_backend.hpp:364（`gpu_tensor_pool_`）— **descriptor pool 固定 262144 sets / 1M descs，batch 模式下无回收机制**。`batch_desc_sets_` 在 `end_batch`/`flush_batch` 时 `vkFreeDescriptorSets` 释放，但 pool 本身不重建。若某 step 的 batch 恰好逼近上限（注释 line 660-661 估算 118K sets），下一 step 的 `alloc_desc_set` 将返回 `VK_ERROR_OUT_OF_POOL_MEMORY`。建议：监控 `batch_desc_sets_.size()`，逼近阈值时提前 `flush_batch`，或按 `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT`（已设，line 674）周期性重建 pool。

**P2-2** compute_vk_backend.hpp:543-560（`~GpuBackend`）— **析构中 `vkDeviceWaitIdle` 失败被忽略；`pending_destroys_` 在 wait 失败时仍被 flush，可能 destroy 尚未完成的 buffer**。

```cpp
vkDeviceWaitIdle(device_.device());   // 返回码丢弃
flush_pending_destroys();             // 无论 wait 成功与否
```

若 `vkDeviceWaitIdle` 返回 `VK_ERROR_DEVICE_LOST`，后续 `vkDestroyBuffer` 在死设备上调用是 UB。建议：检查 `vkDeviceWaitIdle` 返回码，失败时跳过 `flush_pending_destroys` 并标记 `device_lost_ = true`。

**P2-3** compute_vk_backend.hpp:882-931（`begin_batch`）— **`begin_batch` 未检查 `device_lost_`**。设备丢失后 `begin_batch` 仍会 `vkAllocateCommandBuffers`/`vkBeginCommandBuffer`/`vkCreateFence`，全部在死设备上调用，返回 `VK_ERROR_DEVICE_LOST` 但错误信息不含"设备已丢失"上下文。建议：`begin_batch` 入口检查 `device_lost_`，直接返回明确错误。

**P2-4** compute_vk_backend.hpp:3098-3128（`GpuBuffer::~GpuBuffer`）— **非 batch 模式下直接 `vkDestroyBuffer` + `pool->free`，但若 buffer 仍被上一个未提交的 command buffer 引用（`upload_blocking` 快速路径的 `vkWaitForFences` 超时后未真正完成），destroy 是 use-after-free**。当前 `upload_blocking` 在 wait 失败后返回错误但不释放 buffer，调用方若忽略错误则 buffer 析构时立即 destroy。建议：`upload_blocking`/`download_blocking` 失败路径确保 fence 已 signaled（例如无限等待或重试），或 `GpuBuffer` 析构时检查 `in_flight` 状态。

**P2-5** compute_vk_device.hpp:29-36（`vk_check`）— **错误信息不含 Vulkan 错误名称，只有数字**。`"Vulkan error 2 at file:line"` 调试时不直观。建议：加入 `vkToString(VkResult)` 或手动映射常见错误码（`VK_ERROR_DEVICE_LOST`/`VK_TIMEOUT`/`VK_ERROR_OUT_OF_POOL_MEMORY` 等）。

**P2-6** compute_vk_backend.hpp:1512/2021/2173/2338 等（`vkUpdateDescriptorSets` 调用）— **`vkUpdateDescriptorSets` 返回 `void`（Vulkan 规范），但调用方未处理其内部可能的 validation error**。Vulkan 1.0 中 `vkUpdateDescriptorSets` 返回 `void`，参数错误由 validation layer 报告。当前代码参数正确（binding 数与 layout 匹配），无实际风险，但缺乏静态断言确保未来修改不破坏 binding 对应关系。建议：增加 `static_assert` 或 runtime check 确保 `n_bindings` 与 pipeline 的 layout binding 数一致。

**P2-7** compute_vk_backend.hpp:354（`fused_pipelines_`）— **AOT 融合 shader 注册使用 `std::string` 作为 key，无哈希长度限制**。`expr_spec_key` 是构建期生成的字符串（见 `fused_registry.hpp`），若未来 key 过长（> 64 字节），`std::unordered_map<std::string, ...>` 的性能和缓存局部性下降。当前 key 长度合理，无实际风险。建议：监控 key 长度，超长时截断或改用 FNV-1a 64-bit 哈希。

**P2-8** compute_vk_backend.hpp:539-540（AOT 融合 shader 注册循环）— **`fused_pipelines_.emplace(fs.key, ...)` 在 `fp_r` 失败时静默跳过，后续 `run_fused_gpu` 报 "fused shader not registered" 错误信息不含 pipeline 创建失败原因**。若 `vkCreateComputePipelines` 失败（例如 SPIR-V 版本不兼容），用户看到的只是"未注册"而非"创建失败"。建议：`fp_r` 失败时记录警告（`std::cerr` 或日志），或返回 `Result` 让 `initialize()` 失败。

**P2-9** compute_vk_backend.hpp:1038-1176（`flush_batch` 新 command buffer 分配）— **`flush_batch` 成功路径创建新 fence + command buffer 但不重置 `batch_desc_sets_` 的 pool 配额**。每次 `flush_batch` 都消耗 pool 中的 descriptor set 配额（前一批的 sets 已释放，但 pool 的 `maxSets` 不变），频繁 `flush_batch`（例如每层一次）会导致 pool 碎片化。当前 `TENSOR_POOL_SETS = 262144` 足够大，无实际风险。建议：监控 pool 使用率，必要时重建。

**P2-10** compute_vk_device.hpp:248-322（`VulkanPipeline::create_matmul`）— **`create_matmul` 硬编码 3 个 storage buffer binding，但 `matmul.comp` 实际使用 2 个 binding（A, B）+ 1 个 output（C），`matmul_tiled.comp` 同样 3 个。若未来 shader 增加 binding（例如 bias），`create_matmul` 不会自动适配**。建议：统一使用 `create_generic`，`create_matmul` 作为 `create_generic(device, spirv, 3, 20)` 的薄包装。

### [P3] 风格 / 优化建议（≤10 条）

**P3-1** compute_vk_backend.hpp:1588（`matmul_gpu`）— **`std::vector<GpuTensor> inputs{A, B}` 每调用分配一次堆内存**。`dispatch_compute` 接受 `std::span<const GpuTensor>`，可改用栈上数组 + `std::span` 避免堆分配。

**P3-2** compute_vk_backend.hpp:1498-1499（`dispatch_compute`）— **`std::vector<VkDescriptorBufferInfo>` 和 `std::vector<VkWriteDescriptorSet>` 每调用分配**。binding 数 ≤ 9（`EXPR_MAX_INPUTS + 1`），可用 `std::array` 或栈上 VLA 替代。

**P3-3** compute_vk_backend.hpp:1920-1940（`record_input_barriers`）— **`std::vector<VkBufferMemoryBarrier>` 每次动态分配**。barrier 数 ≤ 9，可用栈上数组。

**P3-4** compute_vk_backend.hpp:1584-1586（`matmul_gpu` push constants）— **`std::vector<std::uint8_t> pc(sizeof(push_data))` 堆分配 20 字节**。可用 `std::array<std::uint8_t, 20>` 或 `alignas(16) uint8_t pc[20]`。

**P3-5** compute_vk_backend.hpp:1642-1643（`batched_matmul_gpu` push constants）— 同 P3-4，`sizeof(push)` = 24 字节堆分配，可栈上化。

**P3-6** compute_vk_backend.hpp:1696-1700（`scan_prefix_outer_gpu` push constants）— `sizeof(push)` = 28 字节堆分配，可栈上化。注意 `PushPrefix` 结构体对齐为 4（`uint32_t` 数组），`sizeof = 7 * 4 = 28`，Vulkan push constant 无需 16 字节对齐，栈上 `alignas(4) uint8_t pc[28]` 即可。

**P3-7** compute_vk_backend.hpp:2040-2045（`elementwise_v2_gpu` push constants）— 直接 `vkCmdPushConstants(cmd, ..., sizeof(push), &push)`，`push` 是局部结构体（32 字节），无堆分配，正确。但 `PushData` 结构体中 `float` 字段与 `uint32_t` 混排，`sizeof = 8 * 4 = 32`，对齐正确。建议注释说明对齐。

**P3-8** compute_vk_device.hpp:218-228（`VulkanPipeline` 移动构造）— **移动后源对象的 `device_` 被置 null，但 `operator=`（line 230-241）使用 `std::swap` 而非逐字段移动**。`std::swap` 在 `this == &o` 时安全（提前返回），但语义上移动赋值应是"源变无效"而非"交换"。当前实现中 `VulkanPipeline a; a = std::move(b);` 后 `b` 仍持有有效 handle（被 swap 到 `a` 的旧值，即 null），`b` 析构时不释放——正确。但 `VulkanPipeline a(Valid); a = std::move(b);` 后 `a` 的旧 handle 被 swap 到 `b`，`b` 析构时释放 `a` 的旧 handle——正确。实现无误，但风格上建议逐字段移动 + 置 null 以明确"源变无效"语义。

**P3-9** compute_vk_backend.hpp:579（`GpuBackend::instance()`）— **`new GpuBackend()` 裸 new，`nn-allow` 已知债（docs/17）**。TODO 注释完整，不重复上报。

**P3-10** compute_vk_backend.hpp:12（文件头注释）— **注释称"每个 matmul_gpu/elementwise_gpu 调用都创建 fence，提交后立即等待"，但实际代码在 batch 模式下不创建 fence、不等待**（`acquire_cmd` 返回 `batch_cmd_`，`owns_cmd = false`，跳过 `submit_and_wait`）。注释与实现不符，建议更新。

## 已知问题核对

**VK_TIMEOUT / VK_ERROR_DEVICE_LOST 错误路径未完善**（docs/08 高频坑 #4、AGENTS.md 铁律）：

- **`VK_ERROR_DEVICE_LOST`**：`end_batch`（line 977-996）和 `flush_batch`（line 1080-1097）均有分支，设置 `device_lost_ = true`，释放 fence/cmd/desc sets，flush pending destroys，返回明确错误信息（含 TDR 说明和 `--resume` 建议）。上游 `text_train.cpp:249-294`（`restart_on_device_lost`）实现 checkpoint 保存 + 自动重启（`std::system` 重建命令行带 `--resume`/`--resume-epoch`/`--resume-step`/`--flush-interval`）。**现状：已实现，闭环完整**。
- **`VK_TIMEOUT`**：`end_batch`（line 997-1014）和 `flush_batch`（line 1098-1113）将 `VK_TIMEOUT` 归入"其他错误"分支，返回错误信息但**未实现"减半 batch 重试"**。`text_train.cpp` 中无 `VK_TIMEOUT` 重试逻辑（grep 确认）。`submit_and_wait`（line 1897-1900）的 30s 超时同样将 `VK_TIMEOUT` 当错误传播，无重试。**现状：未实现，仍是已知未修项**。`upload_blocking`/`download_blocking` 的 10s 超时（line 1246/1248/1316/1391/1458）同样无 `VK_TIMEOUT` 重试。

**结论**：`VK_ERROR_DEVICE_LOST` 路径已完善（backend 设标志 + 上游 checkpoint + 自动重启）；`VK_TIMEOUT` 路径仍未实现"减半 batch 重试"，仍为已知未修项。

## 其他观察

1. **`WORKGROUP_SIZE = 16`**（line 404）是死常量：`matmul_gpu` 使用 `const uint32_t tile = use_tiled ? 64u : 16u`（line 1583），`WORKGROUP_SIZE` 从未被引用。建议删除。
2. **`GpuTensor::with_shape`**（line 324-327）零拷贝 reshape 正确（共享 `shared_ptr<GpuBuffer>`），但**无 `rows * cols` 守恒检查**——调用方传错形状（`new_rows * new_cols != rows_ * cols_`）会导致后续 dispatch 越界。建议增加 `if (new_rows * new_cols != rows_ * cols_)` 检查。
3. **`clone_gpu`**（line 2654-2656）使用 `sizeof(float)` 而非 `sizeof(Scalar)`。当前 `Scalar = float`（core_config.hpp），无实际 bug，但若未来 `Scalar` 变为 `double`/`half` 会静默错误。建议改为 `sizeof(Scalar)`。
4. **`batched_matmul_gpu`**（line 1639-1641）`PushDataBmm` 结构体中 `float alpha` 与 `uint32_t` 混排，`sizeof = 6 * 4 = 24`，对齐正确。但 `alpha` 是 `float`（line 1605），若未来 `Scalar` 变为 `double` 会 break。建议用 `Scalar` 类型。
5. **`VulkanDevice` 析构顺序**（line 78-84）：先 `vkDestroyDevice` 再 `vkDestroyInstance`，正确。但**未调用 `vkDeviceWaitIdle`**——若 `GpuBackend` 析构时 GPU 仍有在飞命令（例如 `device_lost_ = true` 后未 flush），`vkDestroyDevice` 可能返回 `VK_ERROR_DEVICE_LOST` 或 hang。`GpuBackend` 析构（line 547）已调用 `vkDeviceWaitIdle`，但 `VulkanDevice` 自身析构不依赖 `GpuBackend`（独立 RAII），若其他代码直接使用 `VulkanDevice` 则缺少 wait。建议 `VulkanDevice::~VulkanDevice` 中先 `vkDeviceWaitIdle`。
6. **`StagingRing` 析构**（line 158-171）：未调用 `vkDeviceWaitIdle`，若 region 的 fence 仍在 in-flight，`vkDestroyFence`/`vkDestroyBuffer` 在 GPU 仍在写 buffer 时执行是 UB。`GpuBackend` 析构先 `vkDeviceWaitIdle` 再 `staging_ring_.reset()`（line 547-551），顺序正确。但 `StagingRing` 作为独立 RAII 类缺少 wait。建议析构中加 `vkDeviceWaitIdle`。
7. **`MemoryPool` 析构**（line 209）：`blocks_.clear()` 触发 `Block` 析构（`vkFreeMemory`），但未 `vkDeviceWaitIdle`。`GpuBackend` 析构顺序：`vkDeviceWaitIdle` → `flush_pending_destroys` → `staging_ring_.reset()` → 销毁 desc pool/cmd pool → `memory_pool_.reset()`（line 547-559），`vkDeviceWaitIdle` 在最前，正确。
