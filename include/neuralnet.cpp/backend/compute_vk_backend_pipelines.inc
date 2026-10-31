// ═══════════════════════════════════════════════════════════════════════════
//  compute_vk_backend_pipelines.inc — GpuBackend 成员定义片段（RV-08 物理拆分，**纯移动、零行为变化**）
//  内容：RLA 扫描原语 + 辅助（描述符集 / 跨 submit 依赖 / command buffer / solo fence / 屏障）
//  ⚠ 本文件不是独立头：仅由 backend/compute_vk_backend.hpp 在
//    `class GpuBackend { ... }` 体内**按序** #include（顺序即成员序）。
// ═══════════════════════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════════════
// RLA 扫描原语（手写原语，形状契约见 compute_engine.hpp 扫描级原语注释）
// 1 workgroup/头：dispatch (1, B*H, 1)；标量块在头块内逐行重复存放
// ══════════════════════════════════════════════════════════════════

// ── 前缀扫描 + matvec 读出：输出 (rows*5, seq) ─────────────────────
// f16_io（Phase C2f）：输入输出均 half（f16 pipeline）；scratch St 仍
// f32（引擎分配）。无 f16 pipeline 时硬报错，调用方走引擎内 cast 回退。
[[nodiscard]] Result<GpuTensor> scan_prefix_outer_gpu(const GpuTensor& K, const GpuTensor& V,
                                                      const GpuTensor& P, const GpuTensor& R,
                                                      const GpuTensor& A0, const GpuTensor& B0,
                                                      bool has_state, uint32_t dk, uint32_t heads,
                                                      bool causal, const GpuTensor& boundary,
                                                      bool has_bnd, bool f16_io = false)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (dk == 0u || heads == 0u) NN_FAIL("scan_prefix_outer_gpu: dk/heads must be > 0");
    const bool generic = dk > 64u; // 通用路径（状态驻全局 scratch，无 dk 上限）
    if (f16_io && !has_scan_f16_pipelines())
        NN_FAIL("scan_prefix_outer_gpu: f16 pipelines not available");
    // P4-2b：dk ≤ 32 时优先走窄共享变体（占用率高 6× ⇒ 32 个 WG 只排 ~1 个
    // 波次）；变体不可用（老 SPIR-V / 裁剪构建）→ 回落宽变体，数值逐位相同。
    const bool narrow = !generic && dk <= 32u;
    const VulkanPipeline& pipe =
        generic ? (f16_io ? pipelines_.get(PipelineId::scan_prefix_outer_gen_f16)
                          : pipelines_.get(PipelineId::scan_prefix_outer_gen))
                : (f16_io ? (narrow && has(PipelineId::scan_prefix_outer_d32_f16)
                                 ? pipelines_.get(PipelineId::scan_prefix_outer_d32_f16)
                                 : pipelines_.get(PipelineId::scan_prefix_outer_f16))
                          : (narrow && has(PipelineId::scan_prefix_outer_d32)
                                 ? pipelines_.get(PipelineId::scan_prefix_outer_d32)
                                 : pipelines_.get(PipelineId::scan_prefix_outer)));
    if (!generic && !has(PipelineId::scan_prefix_outer))
        NN_FAIL("scan_prefix_outer_gpu: pipeline not available");
    if (generic && !has(PipelineId::scan_prefix_outer_gen))
        NN_FAIL("scan_prefix_outer_gpu: generic pipeline not available");
    const auto rows = static_cast<uint32_t>(K.rows());
    if (rows % (dk * heads) != 0) NN_FAIL("scan_prefix_outer_gpu: rows not divisible by H*dk");
    if (V.rows() != rows || V.cols() != K.cols() || P.rows() != rows || P.cols() != K.cols() ||
        R.rows() != rows || R.cols() != K.cols())
        NN_FAIL("scan_prefix_outer_gpu: K/V/P/R shape mismatch");
    if (has_state &&
        (A0.rows() != heads * dk || A0.cols() != dk || B0.rows() != heads * dk || B0.cols() != dk))
        NN_FAIL("scan_prefix_outer_gpu: A0/B0 must be (H*dk, dk)");
    const auto seq = static_cast<uint32_t>(K.cols());
    if (has_bnd)
    {
        if (boundary.rows() != 1 || boundary.cols() != (rows / (dk * heads)) * seq)
            NN_FAIL("scan_prefix_outer_gpu: boundary must be (1, B*seq)");
    }
    const auto BH = rows / dk;
    const std::size_t out_rows = static_cast<std::size_t>(rows) * 5;
    GpuTensor C;
    if (f16_io)
    {
        auto f16r = GpuTensorF16::create_empty(out_rows, seq, *this);
        NN_TRY_CHECK(f16r);
        C = GpuTensor(f16r->shared_buffer(), out_rows, seq);
    }
    else
    {
        auto C_res = GpuTensor::create_empty(out_rows, seq, *this);
        NN_TRY_CHECK(C_res);
        C = std::move(*C_res);
    }
    struct PushPrefix
    { uint32_t dk, heads, seq, causal, has_state, has_bnd, rows; };
    PushPrefix push{dk, heads, seq, causal ? 1u : 0u, has_state ? 1u : 0u, has_bnd ? 1u : 0u, rows};
    std::vector<std::uint8_t> pc(sizeof(push));
    std::memcpy(pc.data(), &push, sizeof(push));
    if (generic)
    {
        // 通用 d_k > 64：状态驻全局 scratch（B*H*2*dk²），共享内存 O(1)，
        // 对任意 dk 安全。GpuTensor 析构走 pending_destroys 延迟归还，
        // batch 录制期被引用（descriptor set）也不会过早释放。
        auto St_res =
            GpuTensor::create_empty(static_cast<std::size_t>(BH) * 2u * dk * dk, 1u, *this);
        NN_TRY_CHECK(St_res);
        GpuTensor St = std::move(*St_res);
        std::vector<GpuTensor> inputs{K, V, P, R, A0, B0, boundary, St};
        auto r = dispatch_compute(pipe, inputs, C, pc, 1u, BH, 1u, "scan_prefix");
        NN_TRY_CHECK(r);
        return C;
    }
    std::vector<GpuTensor> inputs{K, V, P, R, A0, B0, boundary};
    auto r = dispatch_compute(pipe, inputs, C, pc, 1u, BH, 1u, "scan_prefix");
    NN_TRY_CHECK(r);
    return C;
}

// ── 后缀扫描 + matvec 读出：输出 (rows*3, seq) ─────────────────────
// f16_io 同 scan_prefix_outer_gpu（Phase C2c→C2f）。
[[nodiscard]] Result<GpuTensor> scan_suffix_outer_gpu(const GpuTensor& D, const GpuTensor& X,
                                                      const GpuTensor& Y, uint32_t dk,
                                                      uint32_t heads, bool causal,
                                                      const GpuTensor& boundary, bool has_bnd,
                                                      bool f16_io = false)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (dk == 0u || heads == 0u) NN_FAIL("scan_suffix_outer_gpu: dk/heads must be > 0");
    const bool generic = dk > 64u; // 通用路径（状态驻全局 scratch，无 dk 上限）
    if (f16_io && !has_scan_f16_pipelines())
        NN_FAIL("scan_suffix_outer_gpu: f16 pipelines not available");
    // 注意：**后缀扫描刻意不走窄共享变体**——实测它对 suffix 有害（D 的
    // dk²×seq 列式散布读使其为流量/争用受限：窄变体把每 SM 的驻留 WG 从 3
    // 提到 15，dk=32/BH=32 下 7.9 → 16.2 ms），故只给 scan_prefix 加变体。
    const VulkanPipeline& pipe =
        generic ? (f16_io ? pipelines_.get(PipelineId::scan_suffix_outer_gen_f16)
                          : pipelines_.get(PipelineId::scan_suffix_outer_gen))
                : (f16_io ? pipelines_.get(PipelineId::scan_suffix_outer_f16)
                          : pipelines_.get(PipelineId::scan_suffix_outer));
    if (!generic && !has(PipelineId::scan_suffix_outer))
        NN_FAIL("scan_suffix_outer_gpu: pipeline not available");
    if (generic && !has(PipelineId::scan_suffix_outer_gen))
        NN_FAIL("scan_suffix_outer_gpu: generic pipeline not available");
    const auto rows = static_cast<uint32_t>(X.rows());
    if (rows % (dk * heads) != 0) NN_FAIL("scan_suffix_outer_gpu: X rows not divisible by H*dk");
    if (D.rows() != rows * dk || D.cols() != X.cols() || Y.rows() != rows || Y.cols() != X.cols())
        NN_FAIL("scan_suffix_outer_gpu: D/X/Y shape mismatch");
    const auto seq = static_cast<uint32_t>(X.cols());
    if (has_bnd)
    {
        if (boundary.rows() != 1 || boundary.cols() != (rows / (dk * heads)) * seq)
            NN_FAIL("scan_suffix_outer_gpu: boundary must be (1, B*seq)");
    }
    const auto BH = rows / dk;
    const std::size_t out_rows = static_cast<std::size_t>(rows) * 3;
    GpuTensor C;
    if (f16_io)
    {
        auto f16r = GpuTensorF16::create_empty(out_rows, seq, *this);
        NN_TRY_CHECK(f16r);
        C = GpuTensor(f16r->shared_buffer(), out_rows, seq);
    }
    else
    {
        auto C_res = GpuTensor::create_empty(out_rows, seq, *this);
        NN_TRY_CHECK(C_res);
        C = std::move(*C_res);
    }
    struct PushSuffix
    { uint32_t dk, heads, seq, causal, has_bnd, rows; };
    PushSuffix push{dk, heads, seq, causal ? 1u : 0u, has_bnd ? 1u : 0u, rows};
    std::vector<std::uint8_t> pc(sizeof(push));
    std::memcpy(pc.data(), &push, sizeof(push));
    if (generic)
    {
        auto St_res = GpuTensor::create_empty(static_cast<std::size_t>(BH) * dk * dk, 1u, *this);
        NN_TRY_CHECK(St_res);
        GpuTensor St = std::move(*St_res);
        std::vector<GpuTensor> inputs{D, X, Y, boundary, St};
        auto r = dispatch_compute(pipe, inputs, C, pc, 1u, BH, 1u, "scan_suffix");
        NN_TRY_CHECK(r);
        return C;
    }
    std::vector<GpuTensor> inputs{D, X, Y, boundary};
    auto r = dispatch_compute(pipe, inputs, C, pc, 1u, BH, 1u);
    NN_TRY_CHECK(r);
    return C;
}

// ── 逐列外积：输出 (rows*dk, seq) ──────────────────────────────────
// f16_io 同 scan_prefix_outer_gpu（Phase C2f）。
[[nodiscard]] Result<GpuTensor> outer_col_gpu(const GpuTensor& P, const GpuTensor& R,
                                              const GpuTensor& S, uint32_t dk, bool has_scale,
                                              bool f16_io = false)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (!has(PipelineId::outer_col)) NN_FAIL("outer_col_gpu: pipeline not available");
    if (f16_io && pipelines_.get(PipelineId::outer_col_f16).handle() == VK_NULL_HANDLE)
        NN_FAIL("outer_col_gpu: f16 pipeline not available");
    const VulkanPipeline& pipe =
        f16_io ? pipelines_.get(PipelineId::outer_col_f16) : pipelines_.get(PipelineId::outer_col);
    if (dk == 0u) NN_FAIL("outer_col_gpu: dk must be > 0");
    const auto rows = static_cast<uint32_t>(P.rows());
    if (rows % dk != 0) NN_FAIL("outer_col_gpu: rows not divisible by dk");
    if (R.rows() != rows || R.cols() != P.cols()) NN_FAIL("outer_col_gpu: P/R shape mismatch");
    if (has_scale && (S.rows() != rows || S.cols() != P.cols()))
        NN_FAIL("outer_col_gpu: S must be (B*H*dk, seq)");
    const auto seq = static_cast<uint32_t>(P.cols());
    const std::size_t out_rows = static_cast<std::size_t>(rows) * dk;
    GpuTensor C;
    if (f16_io)
    {
        auto f16r = GpuTensorF16::create_empty(out_rows, seq, *this);
        NN_TRY_CHECK(f16r);
        C = GpuTensor(f16r->shared_buffer(), out_rows, seq);
    }
    else
    {
        auto C_res = GpuTensor::create_empty(out_rows, seq, *this);
        NN_TRY_CHECK(C_res);
        C = std::move(*C_res);
    }
    struct PushOuter
    { uint32_t dk, seq, rows, has_scale; };
    PushOuter push{dk, seq, rows, has_scale ? 1u : 0u};
    std::vector<std::uint8_t> pc(sizeof(push));
    std::memcpy(pc.data(), &push, sizeof(push));
    std::vector<GpuTensor> inputs{P, R, S};
    const auto total = static_cast<uint32_t>(rows) * dk * seq;
    auto r = dispatch_compute(pipe, inputs, C, pc, (total + 255u) / 256u, 1u, 1u);
    NN_TRY_CHECK(r);
    return C;
}

// ══════════════════════════════════════════════════════════════════
// 纯 GPU 原语方法（纯 GPU 架构核心）
//
// 所有方法遵循同一模式：
//   1. 分配输出 Tensor（如有输出）
//   2. 分配描述符集，写入 binding
//   3. 获取 command buffer（batch 共享 / 独立分配）
//   4. 录制：输入屏障 → bind pipeline → push constants → dispatch → 输出屏障
//   5. 独立模式：end → submit → wait → cleanup
// ══════════════════════════════════════════════════════════════════

// ── 辅助：分配描述符集 ──────────────────────────────────────────────
[[nodiscard]] Result<VkDescriptorSet> alloc_desc_set(VkDescriptorSetLayout layout)
{
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo desc_alloc{};
    desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    desc_alloc.descriptorPool = gpu_tensor_pool_;
    desc_alloc.descriptorSetCount = 1;
    desc_alloc.pSetLayouts = &layout;

    auto r = detail::vk_check(vkAllocateDescriptorSets(device_.device(), &desc_alloc, &desc_set),
                              __FILE__, __LINE__);
    NN_TRY_CHECK(r);
    // batch 模式：描述符集被当前帧的已录制命令引用，归属该帧，
    // 帧的 fence 信号后（reap_frame）才释放
    if (batch_mode_) frames_[batch_frame_].desc_sets.push_back(desc_set);
    return desc_set;
}

// ── 辅助：分配描述符集并把 buffers[i] 绑到 binding i ──────────────────
// 取代每个算子各写一遍的 "VkDescriptorBufferInfo[N] + VkWriteDescriptorSet[N]
// + vkUpdateDescriptorSets" 样板。失败路径自己归还集合（RV-15 语义），
// 调用方只需 NN_TRY_CHECK。
//
// 16 个以内走栈上数组（原语算子最多 4 个绑定，融合 extras 也在上限内）；
// 超过时退回堆——保持旧实现"支持任意 N"的行为不变。
[[nodiscard]] Result<VkDescriptorSet> alloc_desc_set_for(VkDescriptorSetLayout layout,
                                                         std::span<const VkBuffer> buffers)
{
    auto ds_r = alloc_desc_set(layout);
    if (!ds_r) return ds_r;
    const VkDescriptorSet desc_set = *ds_r;
    if (buffers.empty()) return desc_set;

    constexpr std::size_t kInline = 16;
    VkDescriptorBufferInfo inline_infos[kInline];
    VkWriteDescriptorSet inline_writes[kInline];
    std::vector<VkDescriptorBufferInfo> heap_infos;
    std::vector<VkWriteDescriptorSet> heap_writes;
    VkDescriptorBufferInfo* infos = inline_infos;
    VkWriteDescriptorSet* writes = inline_writes;
    if (buffers.size() > kInline)
    {
        heap_infos.resize(buffers.size());
        heap_writes.resize(buffers.size());
        infos = heap_infos.data();
        writes = heap_writes.data();
    }

    for (std::size_t i = 0; i < buffers.size(); ++i)
    {
        infos[i] = {buffers[i], 0, VK_WHOLE_SIZE};
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc_set;
        writes[i].dstBinding = static_cast<std::uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device_.device(), static_cast<std::uint32_t>(buffers.size()), writes, 0,
                           nullptr);
    return desc_set;
}

// ── 辅助：提交前释放描述符集（**错误分支专用**，RV-15）──────────────
// 语义与 alloc_desc_set 的登记规则严格对齐：
//   batch 模式：集合已在 frames_[batch_frame_].desc_sets 名下 → **此处不得
//     释放**（同一句柄被 vkFreeDescriptorSets 两次 = 校验层 VUID 违规 /
//     驱动 UB），交给 reap_frame 统一回收（该帧命令并未绑定它，留着无害）。
//   solo 模式：集合未登记 → 必须立即释放（否则池配额泄漏）。
// 所有"拿到 cmd 之前的错误分支"（`!cmd_r` / `!ds2_r` …）一律走本函数。
void release_desc_set_now(VkDescriptorSet set) noexcept
{
    if (set == VK_NULL_HANDLE || batch_mode_) return;
    vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &set);
}

// ── 辅助：跨 submit 数据依赖（多帧流水线核心）────────────────────────
// 背景：单队列 FIFO 只是执行顺序保证。实测本驱动（NVIDIA + Windows）
// 下，消费方 submit（download/matmul/batch 帧）紧跟上传 submit
// （<~2ms）时，消费方 GPU 操作会读到上传写入的旧值（零）；host 侧
// sleep ≥5ms 或 fence 等待可规避——即隐式跨 submit 数据依赖不可靠，
// 必须用队列级原语显式建立：
//   上传 submit:  pSignalSemaphores = {region 信号量} + value
//   消费 submit:  pWaitSemaphores   = {所有 in-flight region 信号量}
//                 pWaitSemaphoreValues = {各自的 value}
//                 pWaitDstStageMask  = {ALL_COMMANDS}（waitSemaphoreCount>0
//                                        时规范要求非空，见
//                                        VUID-VkSubmitInfo-pWaitDstStageMask）
// **必须用时间线信号量**：同一 value 可被任意多个 submit 等待且等待已达成
// 的 value 是 no-op。二进制信号量一次 signal 只能被一个 wait 消费，第二
// 个消费者会永久阻塞（VUID-vkQueueSubmit-pWaitSemaphores-03238），AMD
// 老驱动实测直接死锁。
// host 永不阻塞（多帧流水线收益保留），GPU 在队列内等待数据就绪。
// 设备不支持时间线信号量时不创建信号量，改由 drain_in_flight() 在 host
// 侧阻塞兜底（正确性优先）。
struct StagingWait
{
    std::vector<VkSemaphore> sems;
    std::vector<VkPipelineStageFlags> stages;
    std::vector<std::uint64_t> values;
    [[nodiscard]] bool any() const noexcept { return !sems.empty(); }
};
[[nodiscard]] StagingWait collect_staging_waits()
{
    StagingWait w;
    if (!staging_ring_) return w;
    if (!staging_ring_->timeline())
    {
        // 回退路径：host 等在飞上传完成（阻塞），后续 submit 无跨 submit
        // 依赖，无需信号量
        staging_ring_->drain_in_flight();
        return w;
    }
    for (std::size_t i = 0; i < staging_ring_->num_regions(); ++i)
    {
        if (!staging_ring_->in_flight(i)) continue;
        w.sems.push_back(staging_ring_->semaphore(i));
        w.stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        w.values.push_back(staging_ring_->signal_value(i));
    }
    return w;
}

// 把跨 submit 等待挂到 submit 上（pNext = VkTimelineSemaphoreSubmitInfo）
static void attach_staging_waits(VkSubmitInfo& si, const StagingWait& sw,
                                 VkTimelineSemaphoreSubmitInfo& tsi)
{
    if (!sw.any()) return;
    si.waitSemaphoreCount = static_cast<std::uint32_t>(sw.sems.size());
    si.pWaitSemaphores = sw.sems.data();
    si.pWaitDstStageMask = sw.stages.data();
    tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tsi.waitSemaphoreValueCount = static_cast<std::uint32_t>(sw.values.size());
    tsi.pWaitSemaphoreValues = sw.values.data();
    si.pNext = &tsi;
}

// 上传 submit 的信号挂载（时间线信号量模式）。
// 返回 false = 当前设备非时间线模式：不 signal，消费者走 drain 回退。
// 调用方提供的 sem_storage/value_storage 必须活到 vkQueueSubmit（
// pSignalSemaphores/pSignalSemaphoreValues 是借用指针）。
// 必须在 mark_in_flight() 之前调用：消费者只认"in-flight region 的
// signal_value"，value 必须先写好。
static bool attach_upload_signal(VkSubmitInfo& si, StagingRing& ring, std::size_t region_idx,
                                 VkTimelineSemaphoreSubmitInfo& tsi, VkSemaphore& sem_storage,
                                 std::uint64_t& value_storage)
{
    if (!ring.timeline()) return false;
    value_storage = ring.next_signal_value(region_idx);
    sem_storage = ring.semaphore(region_idx);
    tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tsi.signalSemaphoreValueCount = 1;
    tsi.pSignalSemaphoreValues = &value_storage;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &sem_storage;
    si.pNext = &tsi;
    return true;
}

// ── 辅助：获取 command buffer（batch 或独立）──────────────────────
// ── 算子 host 开销剖析开关（NN_GPU_PROFILE=1 时 stderr 打印分段耗时；
//    未设置时进程内仅查询一次，之后纯 bool 读取）────────────────────
[[nodiscard]] static bool profile_ops_enabled()
{
    static const bool on = []()
    {
        // 复用 VulkanDevice::get_env（MSVC 下 _dupenv_s，绕开 getenv 弃用）
        const std::string v = VulkanDevice::get_env("NN_GPU_PROFILE");
        return !v.empty() && v[0] != '0';
    }();
    return on;
}

// ── P0 细 tile（wgrad 网格饥饿）逃生阀：NN_MM_NO_FINE=1 关闭 ────────────
//    A/B 走双进程同窗交错（bench/run_ab_env.ps1 口径），故进程内静态
//    缓存即可——避免 run_fused_gpu 热路径每次派发都 getenv。
[[nodiscard]] static bool mm_fine_disabled()
{
    static const bool off = []()
    {
        const std::string v = VulkanDevice::get_env("NN_MM_NO_FINE");
        return !v.empty() && v[0] != '0';
    }();
    return off;
}

// ── P0 细粒度实验阀：NN_MM_FINE_NAIVE=1 → 饥饿形状改派 naive
//    （matmul.comp：16×16 线程、每线程 1 输出、无 shared/barrier——
//    线程数 = M×N（细 tiled 的 16×），靠 warp 切换吃满延迟）。
//    与 tiled 细变体同窗交错 A/B 定夺最终形态。
[[nodiscard]] static bool mm_fine_naive()
{
    static const bool on = []()
    {
        const std::string v = VulkanDevice::get_env("NN_MM_FINE_NAIVE");
        return !v.empty() && v[0] != '0';
    }();
    return on;
}

// ── K 切分逃生阀 + 调参（wgrad 二次归约）───────────────────────────────
// NN_MM_NO_KSPLIT=1 关闭 K 切分（回落细 tile / 64-tile 路径，A/B 基线）；
// NN_MM_KSPLIT_S=<n> 强制段数 S（1=auto，调参用，钳到 [2, MAX_S]）。
// 双进程同窗交错 A/B 口径下进程内静态缓存即可（与 mm_fine_disabled 同）。
[[nodiscard]] static bool mm_ksplit_disabled()
{
    static const bool off = []()
    {
        const std::string v = VulkanDevice::get_env("NN_MM_NO_KSPLIT");
        return !v.empty() && v[0] != '0';
    }();
    return off;
}

[[nodiscard]] static std::uint32_t mm_ksplit_forced_s()
{
    static const std::uint32_t s = []() -> std::uint32_t
    {
        const std::string v = VulkanDevice::get_env("NN_MM_KSPLIT_S");
        if (v.empty() || v[0] == '0') return 0u; // auto
        long n = 0;
        for (char c : v)
        {
            if (c < '0' || c > '9') break;
            n = n * 10 + (c - '0');
            if (n > 64) break;
        }
        return n >= 2 ? static_cast<std::uint32_t>(n) : 0u;
    }();
    return s;
}

// ── 设备侧计时 helper（口径见成员注释）────────────────────────────
// 录制块开始：reset 本块查询区间 + 游标复位（帧/solo 通用；上一块内容
// 必已在其 fence 等待点 resolve，见 reap_frame / submit_and_wait）
void ts_begin_block(VkCommandBuffer cmd, TsCursor& cur)
{
    cur.next = 0;
    cur.marks.clear();
    ts_cur_ = &cur;
    if (!ts_enabled_) return;
    vkCmdResetQueryPool(cmd, ts_pool_, cur.base, kTsPairsPerBlock * 2);
}

// dispatch 前后各写一个 timestamp（见成员注释的口径）
// detail：可选的 per-key 归因串（运行期 shader key / 原语变体名；传空跳过）
//   —— 录制期驻留进 prof 稳定池，resolve 时与 label@stage 一起回填。
// detail_suffix：拼在 detail 后的静态后缀（细 tile 变体 "#f"；调用点零分配）
void ts_dispatch(VkCommandBuffer cmd, const char* label, std::uint32_t gx, std::uint32_t gy,
                 std::uint32_t gz, std::string_view detail = {},
                 const char* detail_suffix = nullptr)
{
    const bool mark = ts_enabled_ && ts_cur_ != nullptr && ts_cur_->next < kTsPairsPerBlock;
    std::uint32_t q = 0;
    if (mark)
    {
        q = ts_cur_->base + ts_cur_->next * 2;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, ts_pool_, q);
        // 归因到录制期 stage（resolve 时 scope 已出栈）：key = "label@stage"，
        // 使设备 kernel 表可与 host 阶段表同维度对账（仅剖析开启时进入）
        const char* stage = nn::prof::current_stage();
        const char* key =
            (stage != nullptr && stage[0] != '\0') ? nn::prof::interned_pair(label, stage) : label;
        // per-key 细分（"detail[suffix]@label@stage"；detail 空 = 不入细分表）
        const char* d = detail.empty()
                            ? nullptr
                            : nn::prof::interned_detail(detail, detail_suffix, label, stage);
        ts_cur_->marks.push_back(TsMark{key, d, ts_cur_->next});
        ++ts_cur_->next;
    }
    vkCmdDispatch(cmd, gx, gy, gz);
    if (mark) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, ts_pool_, q + 1);
}

// resolve：调用点 fence 已等 → 读回本块全部 pair → 按标号聚合
void ts_resolve(TsCursor& cur)
{
    if (!ts_enabled_ || cur.next == 0 || cur.marks.empty())
    {
        cur.next = 0;
        cur.marks.clear();
        return;
    }
    std::vector<std::uint64_t> ts(cur.next * 2);
    const VkResult r = vkGetQueryPoolResults(
        device_.device(), ts_pool_, cur.base, cur.next * 2, ts.size() * sizeof(std::uint64_t),
        ts.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (r == VK_SUCCESS)
    {
        for (const auto& m : cur.marks)
        {
            const std::uint32_t p = m.pair;
            const std::uint64_t t0 = ts[static_cast<std::size_t>(p) * 2];
            const std::uint64_t t1 = ts[static_cast<std::size_t>(p) * 2 + 1];
            if (t1 >= t0)
                nn::prof::device_add(
                    m.key, static_cast<std::uint64_t>(static_cast<double>(t1 - t0) * ts_period_ns_),
                    m.detail);
        }
    }
    cur.next = 0;
    cur.marks.clear();
}

// 返回 (cmd, owns_cmd)。owns_cmd=true 时调用方需 submit_and_wait。
[[nodiscard]] Result<std::pair<VkCommandBuffer, bool>> acquire_cmd()
{
    if (batch_mode_)
    {
        batch_has_ops_ = true; // 当前帧已含 op（空帧不提交）
        ts_cur_ = &frames_[batch_frame_].ts; // 设备侧计时：归属当前帧块
        return std::make_pair(batch_cmd_, false);
    }

    // 独立模式复用 initialize() 预分配的 solo_cmd_：reset + begin，
    // 免去每算子 vkAllocateCommandBuffers/vkFreeCommandBuffers。
    // solo_cmd_ 仅在 submit_and_wait 等完 fence 后才会被再次取用，
    // 不存在 pending 期 reset；begin 失败时 buffer 留在 reset 后初态
    // （规范允许 reset 从错误态恢复），下次调用可重试。
    VkCommandBuffer cmd = solo_cmd_;
    if (cmd == VK_NULL_HANDLE) NN_FAIL("acquire_cmd: solo_cmd_ not initialized");
    auto r = detail::vk_check(vkResetCommandBuffer(cmd, 0), __FILE__, __LINE__);
    NN_TRY_CHECK(r);
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
    NN_TRY_CHECK(r);
    ts_begin_block(cmd, ts_solo_); // 设备侧计时：solo 块开始
    return std::make_pair(cmd, true);
}

// ── 辅助：独立模式提交+等待（solo fence/cmd 复用，仅归还描述符集）────
// cmd：acquire_cmd 返回的 solo_cmd_（非 batch 模式才会走到这里）
// desc_set 可为 VK_NULL_HANDLE（fill_zero/copy_buffer 无描述符集）
// NN_GPU_PROFILE=1：stderr 打印 host 分段耗时（µs）用于开销归因
[[nodiscard]] Result<void> submit_and_wait(VkCommandBuffer cmd, VkDescriptorSet desc_set)
{
    const bool prof = profile_ops_enabled();
    const auto t_begin = std::chrono::steady_clock::now();

    auto r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
    if (!r)
    {
        if (desc_set != VK_NULL_HANDLE)
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
        NN_TRY_CHECK(r);
    }
    const auto t_end = std::chrono::steady_clock::now();

    if (solo_fence_ == VK_NULL_HANDLE)
    {
        if (desc_set != VK_NULL_HANDLE)
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
        NN_FAIL("submit_and_wait: solo_fence_ not initialized");
    }
    // 上一次 submit 已 wait 到 signaled → 这里 reset 回 unsignaled 复用
    //（首次调用时是 init 创建的 fresh unsignaled，reset 为合法 no-op）
    r = detail::vk_check(vkResetFences(device_.device(), 1, &solo_fence_), __FILE__, __LINE__);
    if (!r)
    {
        if (desc_set != VK_NULL_HANDLE)
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
        NN_TRY_CHECK(r);
    }

    // 跨 submit 数据依赖：输入 buffer 可能刚由在飞上传的 copy 写入
    // （from_matrix 非阻塞提交后本原语立即执行）——等 in-flight region
    // 信号量。host 不阻塞；GPU 在队列内等数据就绪。
    const auto sw = collect_staging_waits();

    {
        std::lock_guard lock(queue_mutex_);
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd;
        VkTimelineSemaphoreSubmitInfo timeline_info{};
        attach_staging_waits(submit_info, sw, timeline_info);
        r = detail::vk_check(vkQueueSubmit(device_.compute_queue(), 1, &submit_info, solo_fence_),
                             __FILE__, __LINE__);
        if (!r)
        {
            if (desc_set != VK_NULL_HANDLE)
                vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
            NN_TRY_CHECK(r);
        }
    }
    const auto t_submit = std::chrono::steady_clock::now();

    // 30 秒超时（单个原语，比 batch 短）
    constexpr uint64_t kSingleOpTimeoutNs = 30'000'000'000ULL;
    r = detail::vk_check(
        vkWaitForFences(device_.device(), 1, &solo_fence_, VK_TRUE, kSingleOpTimeoutNs), __FILE__,
        __LINE__);
    const auto t_wait = std::chrono::steady_clock::now();

    ts_resolve(ts_solo_); // 设备侧计时：solo 块 resolve（fence 已等）

    // 只归还描述符集；solo fence/cmd 留待下次复用（不销毁/不释放）
    if (desc_set != VK_NULL_HANDLE)
        vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &desc_set);
    const auto t_done = std::chrono::steady_clock::now();

    if (prof)
    {
        const auto us = [](auto a, auto b)
        {
            return static_cast<long long>(
                std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
        };
        std::fprintf(stderr,
                     "[gpu-profile] solo_op: end=%lldus submit=%lldus wait=%lldus "
                     "cleanup=%lldus total=%lldus\n",
                     us(t_begin, t_end), us(t_end, t_submit), us(t_submit, t_wait),
                     us(t_wait, t_done), us(t_begin, t_done));
    }
    return r;
}

// ── 辅助：录制输入屏障（多个输入 buffer）──────────────────────────
void record_input_barriers(VkCommandBuffer cmd, std::initializer_list<VkBuffer> buffers)
{ record_input_barriers(cmd, std::span<const VkBuffer>(buffers.begin(), buffers.size())); }

void record_input_barriers(VkCommandBuffer cmd, std::span<const VkBuffer> buffers)
{
    std::vector<VkBufferMemoryBarrier> barriers;
    barriers.reserve(buffers.size());
    for (auto buf : buffers)
    {
        VkBufferMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = buf;
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        barriers.push_back(b);
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
}

// ── 辅助：录制输出屏障 ─────────────────────────────────────────────
void record_output_barrier(VkCommandBuffer cmd, VkBuffer output_buf)
{
    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = output_buf;
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 1, &b, 0, nullptr);
}
