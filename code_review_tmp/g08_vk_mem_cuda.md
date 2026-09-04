# Vulkan 内存管理 + GPU Tensor 实现 + CUDA 后端（已停用）代码审查

## 模块概览

本组 4 个文件构成 GPU 后端的内存与数据通路：`MemoryPool`（128MB block 子分配器，first-fit + 空闲区合并，带锁）；`StagingRing`（2×64-256MB HOST_VISIBLE 环形 staging，每 region 一个 fence）；`GpuTensor` 方法实现（device-local 单存储，CPU 仅经 `upload_blocking`/`download_blocking` 往返）；`CudaBackend`（`NN_HAS_CUDA` 永假，整体死代码）。

## 发现

### P0（无）

### P1

**P1-1 `compute_staging_ring.hpp:243-246` — `mark_in_flight` 全库无调用方，`in_flight` 标志恒 false，fence 复用保护是死代码**

```cpp
void mark_in_flight(std::size_t region_idx) noexcept { regions_[region_idx].in_flight = true; }
```
全库 grep 仅有定义与 `acquire()` 内部读写，无任何调用点。当前正确性依赖 `upload_blocking`/`download_blocking` 每次提交后立即 `vkWaitForFences`（vk_backend:1246/1316/1391/1458）串行化——fence 在 `acquire` 时恒为 signaled，`acquire` 的等待分支永不触发。任何未来调用方（若忘记阻塞等待）复用同一 region 时，`acquire` 不会阻塞，新写覆盖 GPU 尚未读完的 staging 数据 → 数据损坏。建议：要么删除 `in_flight`/`mark_in_flight` 并在 `acquire()` 注释中显式声明"调用方必须阻塞等待 fence"的契约，要么在 `upload_blocking`/`download_blocking` 提交后真正调用 `mark_in_flight`，让 fence 保护机制成为可执行代码。

**P1-2 `compute_memory_pool.hpp:129` — `std::vector<std::unique_ptr<Block>>` 冗余**

```cpp
std::vector<std::unique_ptr<Block>> blocks_;
```
`Block` 已有移动构造函数（L74-83）且 `release_idle_blocks` 的 erase 只析构不移位；vector 扩容时 unique_ptr 本身可平凡移动，完全不需要 indirection。L128 注释声称"避免触发 Block 的移动和析构"是错误理由（unique_ptr 元素移动不移动 Block 对象，vector erase 同样析构）。直接改 `std::vector<Block>` 更简洁、省一层间接。属规范/设计缺陷，有实际可读性影响但无行为 bug。

**P1-3 `compute_cuda_backend.hpp` 全文件 — 死代码（已停用）建议整体删除**

`NN_HAS_CUDA` 自 v1.0.0 起 CMake 永不定义（`docs/06-cuda-backend.md:5`、`AGENTS.md` §2），`nn.hpp:35` / `compute_tensor.hpp` / `cli_engine_factory.hpp` 的所有引用点均在 `#ifdef` 内不编译。`cuda/cuda_kernels.{h,cu}`、`cuda/CMakeLists.txt` 同属死代码。建议：删除 `backend/compute_cuda_backend.hpp` + `compute_cuda_engine.hpp` + `cuda/` 目录 + `nn.hpp:35`、`compute_tensor.hpp`、`cli_engine_factory.hpp:28,51` 的 `#ifdef NN_HAS_CUDA` 块，保留 `docs/06-cuda-backend.md` 作为恢复参考。当前不删也不影响正确性，但 `docs/17-pointer-audit.md:44` 已确认 6 处裸指针债集中在此，删除即清账。

### P2

**P2-1 `compute_staging_ring.hpp:160-170` — 析构未等待 in-flight fence，直接销毁 buffer/fence**

```cpp
for (auto& r : regions_) {
    if (r.fence != VK_NULL_HANDLE) vkDestroyFence(device_, r.fence, nullptr);
    if (r.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device_, r.buffer, nullptr);
```
若析构时某 region 的 fence 仍 in-flight（即 `acquire` 后未等到就退出），`vkDestroyFence` 在 in-flight fence 上是 UB。当前 `upload_blocking`/`download_blocking` 均阻塞等待，实际不会触发，但缺防御。建议析构前对每个 in-flight region `vkWaitForFences`。

**P2-2 `compute_memory_pool.hpp:233-244` — `allocate` first-fit 未校验 `effective_block_size` 与 `alignment` 的兼容性**

```cpp
for (auto& block_ptr : blocks_) {
    auto& block = *block_ptr;
    if (block.memory_type_index != *mem_type) continue;
    auto offset = find_free(block, alignment, alloc_size);
```
若 `alignment > block_size_`（极端情况下调用方传超大 alignment），新 block 用 `effective_block_size = max(alloc_size, block_size_)` 分配，但 `find_free` 在旧 block 中找不到合适区域后走到新 block 分支，新 block 的 `free_regions = {0, alloc_size}` 足够大，对齐仍能满足（因为 `alloc_size >= 对齐需求`）。实际无 bug，但代码未显式断言 `effective_block_size >= alignment`，未来改 `effective_block_size` 逻辑时易踩坑。建议加 `assert(alloc_size >= alignment)` 或注释。

**P2-3 `compute_staging_ring.hpp:101-107` — staging region 大小计算未考虑 `pool_->allocate` 的对齐 padding**

```cpp
VkDeviceSize calculated_size = max_host_visible / HOST_VISIBLE_FRACTION;
calculated_size = std::clamp(calculated_size, MIN_REGION_SIZE, MAX_REGION_SIZE);
region_size_ = std::max(static_cast<VkDeviceSize>(region_size), calculated_size);
```
`region_size_` 用作 `vkCreateBuffer` 的 `buf_info.size`，后续 `pool_->allocate(mem_reqs, ...)` 会用 `mem_reqs.alignment`（通常 4/8）做子分配对齐，pool 内部对齐后实际可用字节可能略小于 `region_size_`（padding 几字节）。`upload`/`download` 用 `offset + byte_size > region_size_` 做边界检查，若 `byte_size` 恰等于 `region_size_` 但 pool 对齐后实际可用略小，则 `memcpy` 越界 1-7 字节。概率极低（padding 通常 < 64 字节，且 `region_size_` 是 2^N 对齐的 64/256MB），但理论 UB。建议 `upload`/`download` 用 `r.alloc.size` 而非 `region_size_` 做边界检查。

**P2-4 `compute_gpu_tensor_impl.hpp:43-56` — `create_empty` 的 `rows * cols` 未做 64-bit 溢出保护**

```cpp
auto buf_res = GpuBuffer::create_device_local(..., rows * cols, TENSOR_BUFFER_USAGE);
```
`rows`/`cols` 是 `std::size_t`（64-bit），`rows * cols` 在 64-bit 上不会溢出（除非 rows=cols=2^32 这种不现实尺寸）。在 32-bit 平台（若未来支持）会溢出。当前 64-bit 平台无实际风险，标"疑似"。

**P2-5 `compute_staging_ring.hpp:182` — `acquire()` 的 `current_.fetch_add` 无溢出回绕保护**

```cpp
std::size_t idx = current_.fetch_add(1, std::memory_order_relaxed) % regions_.size();
```
`std::size_t` 溢出是 UB。`regions_.size()` 为 2，`current_` 需要 2^64 次 acquire 才溢出（约 2^53 年 @ 1MHz），实际无风险。标"疑似"。

### P3

**P3-1 `compute_memory_pool.hpp:109-118` — `SuballocKeyHash` 注释"Knuth 黄金比例"不准确**

```cpp
constexpr std::size_t KNUTH_GOLDEN = 0x9E3779B97F4A7C15ULL;
```
该常数实际是 FNV 质数变体 / 黄金比例 64-bit 表示，注释"Knuth 黄金比例散列常数"易误导。建议改注释为"黄金比例 64-bit 常量（0x9E3779B97F4A7C15 ≈ 2^64 * φ）"。

**P3-2 `compute_memory_pool.hpp:140` — `find_free` 对齐计算未用 64-bit 饱和**

```cpp
VkDeviceSize aligned = (it->offset + alignment - 1) & ~(alignment - 1);
```
`alignment` 为 1 时 `alignment - 1 = 0`，`& ~(0) = & 0 = 0`，`aligned = 0`，正确。`alignment > 1` 时正常。无 bug，但 `alignment` 若为 0（调用方传 0）会 `alignment - 1 = SIZE_MAX` 下溢。`allocate` 在 L228 做了 `alignment > 0 ? ... : 1` 保护，但 `find_free` 是 private static，若未来其他调用方绕过 `allocate` 直接调 `find_free` 会踩坑。建议 `find_free` 入口加 `assert(alignment > 0)`。

**P3-3 `compute_staging_ring.hpp:209/226` — `upload`/`download` 的 `memcpy` 未检查 `mapped_ptr` 非 null**

```cpp
std::memcpy(static_cast<char*>(r.mapped_ptr) + offset, data.data(), byte_size);
```
`mapped_ptr` 在 `initialize` 成功路径恒非 null（`vkMapMemory` 成功即写入），但 `acquire` 后若 region 未初始化（`device_ == VK_NULL_HANDLE`）则 `mapped_ptr` 为 null。当前 `acquire` 无前置检查，理论 UB。建议 `upload`/`download` 入口加 `if (!r.mapped_ptr) return std::unexpected(Error{"Region not initialized"});`。

**P3-4 `compute_gpu_tensor_impl.hpp:20-41` — `from_matrix` 在 `create_device_local` 失败后未释放已分配的 buffer**

```cpp
auto buf_res = GpuBuffer::create_device_local(...);
if (!buf_res) return std::unexpected(buf_res.error());
auto tensor = GpuTensor(std::make_shared<GpuBuffer>(std::move(*buf_res)), ...);
auto upload_res = backend.upload_blocking(tensor, cpu_mat.span());
if (!upload_res) return std::unexpected(upload_res.error());
```
`upload_blocking` 失败时 `tensor`（含 `shared_ptr<GpuBuffer>`）在函数返回时析构，`GpuBuffer::~GpuBuffer` 会调 `defer_buffer_destroy` 或 `vkDestroyBuffer + pool_->free`，正确释放。无泄漏。此条为审查确认项，非 bug。

**P3-5 `compute_cuda_backend.hpp:303-319` — `upload_blocking` 在 `in_batch_` 为 false 时仍先 `cudaMemcpyAsync` 再同步**

```cpp
auto err = cudaMemcpyAsync(dst.data(), cpu_data.data(), ..., cudaMemcpyHostToDevice, stream_);
if (!in_batch_) {
    auto sync_err = cudaStreamSynchronize(stream_);
```
顺序正确（先拷贝后同步），无 bug。CUDA 已停用，不深审。

## 已知问题核对

**vk backend 的 `pending_destroys_` 与 memory pool 协作**：`compute_vk_backend.hpp:587-608` 的 `defer_buffer_destroy` 在 batch 期间将 buffer + alloc + pool 打包入 `pending_destroys_`，`flush_pending_destroys`（L598-608）在 `end_batch` 提交完成后（L990/1009/1029/1093/1109/1122）依次 `vkDestroyBuffer` + `pool_->free`。本组 `MemoryPool::free` 在 `free` 中（L266-316）做空闲区合并，与 `pending_destroys_` 的延迟销毁时序正确——`free` 仅在 buffer 销毁后调用，不会出现两个存活 buffer 内存重叠。`StagingRing` 的 region 从 `MemoryPool` 分配（L129-134），析构时不显式 `pool_->free`（注释 L169"MemoryPool 会自动释放内存"），依赖 `MemoryPool` 析构时 `blocks_.clear()` 释放所有 block——但 `StagingRing` 析构（L158-171）早于 `GpuBackend` 析构（L549-559 `staging_ring_.reset()` 在 `memory_pool_.reset()` 之前），`StagingRing` 析构时 region 的 `alloc` 仍有效但未被 `pool_->free` 归还，直接依赖 pool 析构清理。无 bug（pool 析构释放所有 block），但 `StagingRing` 的 `alloc` 字段在析构时是"悬空"的（pool 还活着，但 region 已 destroy buffer，alloc 未归还）。若未来 `StagingRing` 析构后 `MemoryPool` 仍被其他组件使用（如 `release_idle_blocks`），region 占用的字节在 `free_regions` 中不可见（因为 `alloc` 未 `free`），`release_idle_blocks` 不会释放该 block（`allocation_count > 0`）。这是**设计预期**（staging 常驻），但注释应明确。

**无 P0 级 pool 与 `pending_destroys_` 协作缺口**。

## 其他观察

- `MemoryPool` 的 `release_idle_blocks`（L404-433）在 `end_batch` 完成后调用（vk_backend:866），与 `pending_destroys_` 的 flush 顺序正确（先 flush 再 release）。
- `StagingRing` 的 `pool_` 是 `observer_ptr<MemoryPool>`（L65），非拥有指针，与 `GpuBackend` 的 `unique_ptr<MemoryPool>` 生命周期一致。
- `GpuTensor` 的 CPU 侧无影子存储（`from_matrix` 上传后不保留 CPU 副本，`to_matrix` 每次重新下载），符合"GPU-resident 已禁用、走 staging"的设计。
- `compute_cuda_backend.hpp` 的 `CudaBuffer` 使用裸 `void*` + `cudaMalloc/cudaFree`（L68-79/109），是 `NN_HAS_CUDA` 内的 6 处裸指针债之一（`docs/17` 已确认），删除整个文件即清账。
