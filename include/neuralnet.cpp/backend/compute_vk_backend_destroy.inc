// ═══════════════════════════════════════════════════════════════════════════
//  compute_vk_backend_destroy.inc — GpuBackend 成员定义片段（RV-08 物理拆分，**纯移动、零行为变化**）
//  内容：延迟销毁决策/入口、reap 单帧、f16 能力与 pipeline 句柄查询、收帧、显存回收
//  ⚠ 本文件不是独立头：仅由 backend/compute_vk_backend.hpp 在
//    `class GpuBackend { ... }` 体内**按序** #include（顺序即成员序）。
// ═══════════════════════════════════════════════════════════════════════════
// ── 延迟销毁决策（GpuBuffer 析构时调用）────────────────────────────
// 返回 true = 必须延迟（调用 defer_buffer_destroy）；false = 可立即销毁。
//   - batch 录制中：延迟，打当前帧标签（buffer 仍被本帧已录制命令引用）
//   - 非 batch：最近一次提交的帧若仍在飞行则延迟（buffer 可能被该帧
//     命令引用，帧 fence 信号即可证明安全）；否则立即销毁
// 内存归还与 buffer 销毁同批：buffer 销毁前不得把区间还给池。
[[nodiscard]] bool needs_deferred_destroy() const noexcept
{
    if (batch_mode_) return true;
    return last_active_frame_ < frames_.size() && frames_[last_active_frame_].in_flight;
}

// ── 延迟销毁入口（GpuBuffer 析构时调用）──────────────────────────
// 打上帧标签：batch 录制中 → 当前帧；非 batch → 最近活跃帧。
void defer_buffer_destroy(VkDevice device, VkBuffer buffer, const MemoryPool::Allocation& alloc,
                          observer_ptr<MemoryPool> pool)
{
    const std::size_t frame = batch_mode_ ? batch_frame_ : last_active_frame_;
    std::lock_guard lock(pending_mutex_);
    pending_destroys_.push_back({device, buffer, alloc, pool, frame});
}

// ── 统一执行延迟销毁（必须在对应帧的 fence 信号之后调用）─────────
// 先 vkDestroyBuffer 再归还内存到池：保证 buffer 对象销毁后
// 其区间才可被新分配复用，杜绝存活 buffer 间的内存重叠。
// flush 全部：用于"所有帧已完成"的 drain 点（wait_in_flight / 析构）。
void flush_pending_destroys()
{
    std::lock_guard lock(pending_mutex_);
    for (auto& pd : pending_destroys_)
    {
        if (pd.buffer != VK_NULL_HANDLE && pd.device != VK_NULL_HANDLE)
            vkDestroyBuffer(pd.device, pd.buffer, nullptr);
        if (pd.pool && pd.alloc.valid()) pd.pool->free(pd.alloc);
    }
    pending_destroys_.clear();
}

// ── reduce scratch 的生命周期 ────────────────────────────────────────
// scratch 由 backend 直接持有（不是 GpuBuffer），避免
// backend → GpuTensor → GpuBuffer(shared_ptr<GpuBackend>) 的引用环。

// 析构期：device 已 idle，直接销毁并归还 pool。
void destroy_reduce_partial_now_()
{
    if (reduce_partial_.buffer == VK_NULL_HANDLE) return;
    vkDestroyBuffer(device_.device(), reduce_partial_.buffer, nullptr);
    if (reduce_partial_.pool && reduce_partial_.alloc.valid())
        reduce_partial_.pool->free(reduce_partial_.alloc);
    reduce_partial_ = {};
}

// 运行期形状变化：batch 录制中 / 帧在飞时走延迟队列，其余立即释放。
void retire_reduce_partial_()
{
    if (reduce_partial_.buffer == VK_NULL_HANDLE) return;
    if (needs_deferred_destroy())
    {
        defer_buffer_destroy(device_.device(), reduce_partial_.buffer, reduce_partial_.alloc,
                             observer_ptr<MemoryPool>(reduce_partial_.pool));
    }
    else
    {
        vkDestroyBuffer(device_.device(), reduce_partial_.buffer, nullptr);
        if (reduce_partial_.pool && reduce_partial_.alloc.valid())
            reduce_partial_.pool->free(reduce_partial_.alloc);
    }
    reduce_partial_ = {};
}

[[nodiscard]] Result<void> ensure_reduce_partial_(std::uint32_t rows, std::uint32_t cols)
{
    if (reduce_partial_.buffer != VK_NULL_HANDLE && reduce_partial_.rows == rows &&
        reduce_partial_.cols == cols)
        return {};

    retire_reduce_partial_();

    MemoryPool& pool = alloc_pool();
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = static_cast<VkDeviceSize>(rows) * cols * sizeof(float);
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (pool.device_address_alloc()) info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    const VkResult cr = vkCreateBuffer(device_.device(), &info, nullptr, &buffer);
    if (cr != VK_SUCCESS)
        NN_FAIL("reduce scratch vkCreateBuffer failed: " + std::to_string(static_cast<int>(cr)));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_.device(), buffer, &req);
    auto alloc_r = pool.allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!alloc_r)
    {
        vkDestroyBuffer(device_.device(), buffer, nullptr);
        NN_TRY_CHECK(alloc_r);
    }
    const VkResult br =
        vkBindBufferMemory(device_.device(), buffer, alloc_r->memory, alloc_r->offset);
    if (br != VK_SUCCESS)
    {
        pool.free(*alloc_r);
        vkDestroyBuffer(device_.device(), buffer, nullptr);
        NN_FAIL("reduce scratch vkBindBufferMemory failed: " +
                std::to_string(static_cast<int>(br)));
    }

    reduce_partial_ = ReduceScratch{buffer, *alloc_r, &pool, rows, cols};
    return {};
}

// ── reap 单帧（多帧流水线）──────────────────────────────────
// 等待该帧 fence（wait=true；wait=false 要求调用方已用零超时
// vkWaitForFences 确认 VK_SUCCESS）→ 销毁该帧的延迟 buffer 并归还
// 内存 → 释放该帧描述符集（帧提交期间被已录制命令引用，fence 信号
// 前不得释放）→ in_flight=false
[[nodiscard]] Result<void> reap_frame(std::size_t i, bool wait)
{
    auto& f = frames_[i];
    if (f.in_flight && wait)
    {
        constexpr uint64_t kFrameTimeoutNs = 60'000'000'000ULL; // 60s
        const VkResult wr =
            vkWaitForFences(device_.device(), 1, &f.fence, VK_TRUE, kFrameTimeoutNs);
        if (wr == VK_ERROR_DEVICE_LOST)
        {
            // 设备丢失（TDR 触发）：GPU 已死亡，无法恢复
            device_lost_ = true;
            NN_FAIL("GPU 设备丢失 (VK_ERROR_DEVICE_LOST): Windows TDR 已重置 GPU 驱动。"
                    "\n模型已自动保存，请使用 --resume <save-path> 重启训练。"
                    "\n建议：减小 --batch-size 或 --seq-len，或增大 Windows TDR 超时"
                    " (注册表 TdrDelay)");
        }
        if (wr != VK_SUCCESS)
            NN_FAIL(std::string("GPU 帧等待失败: Vulkan error ") +
                    std::to_string(static_cast<int>(wr)) +
                    "\n建议：减小 --batch-size 或 --seq-len，或增大 Windows TDR 超时"
                    " (注册表 TdrDelay)");
    }
    ts_resolve(f.ts); // 设备侧计时：fence 已等（或调用方已确认），读回本帧 kernel 时间
    // 该帧的延迟销毁：先 vkDestroyBuffer 再归还内存
    {
        std::lock_guard lock(pending_mutex_);
        std::vector<PendingDestroy> rest;
        rest.reserve(pending_destroys_.size());
        for (auto& pd : pending_destroys_)
        {
            if (pd.frame == i)
            {
                if (pd.buffer != VK_NULL_HANDLE && pd.device != VK_NULL_HANDLE)
                    vkDestroyBuffer(pd.device, pd.buffer, nullptr);
                if (pd.pool && pd.alloc.valid()) pd.pool->free(pd.alloc);
            }
            else
            {
                rest.push_back(std::move(pd));
            }
        }
        pending_destroys_ = std::move(rest);
    }
    if (!f.desc_sets.empty())
        vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_,
                             static_cast<uint32_t>(f.desc_sets.size()), f.desc_sets.data());
    f.desc_sets.clear();
    f.in_flight = false;
    return {};
}

// 初始化
// device_selector：手动指定计算设备（索引 "2" 或名称子串 "40HX"/"NVIDIA"）。
// 空 = 自动选择（设备类型 + apiVersion 打分，见 VulkanDevice::initialize）。
[[nodiscard]] Result<void> initialize(std::string device_selector = {})
{
    std::lock_guard lock(init_mutex_);
    if (initialized_) return {};

    // 1. 检查 SPIR-V 是否可用
    const auto& spirv = get_matmul_spirv();
    if (spirv.empty()) NN_FAIL("matmul SPIR-V bytecode not embedded");
    // 2. 初始化 Vulkan 设备（先登记设备选择器）
    if (!device_selector.empty()) device_.set_device_selector(std::move(device_selector));
    auto dev_r = device_.initialize();
    if (!dev_r) return dev_r;

    // 显存峰值优化开关：默认在帧提交后非阻塞收割已完成帧，让步内已
    // 析构张量的内存尽早还池；NN_NO_EARLY_REAP=1 关闭（A/B 对照用）。
    early_reap_enabled_ = VulkanDevice::get_env("NN_NO_EARLY_REAP").empty();

    // 3. 按 pipeline_specs() 表创建全部原语 pipeline。
    //    缺失（SPIR-V 未嵌入 / 设备能力不足 / 创建失败）= 空句柄，
    //    由上层按 has_*() 回退；唯一硬要求是 matmul 本身。
    pipelines_.create_all(device_.device(), device_, pipeline_specs());
    if (!pipelines_.has(PipelineId::matmul)) NN_FAIL("matmul pipeline 创建失败");

    // 4. 创建 memory pool（持久 + 瞬态）
    //    底材粒度可用 NN_POOL_BLOCK_MB 覆盖（显存峰值调参/探针用）：
    //    128MB 大底材在混合尺寸+混合生命周期下会产生不可归还的内部碎片。
    VkDeviceSize block_bytes = MemoryPool::DEFAULT_BLOCK_SIZE;
    {
        const std::string v = VulkanDevice::get_env("NN_POOL_BLOCK_MB");
        if (!v.empty())
        {
            const unsigned long long mb = std::strtoull(v.c_str(), nullptr, 10);
            if (mb > 0) block_bytes = static_cast<VkDeviceSize>(mb) * 1024ull * 1024ull;
        }
    }
    // 自适应阶梯上限（NN_POOL_LADDER_MAX_MB>0 启用）：中等分配按最小
    // 2 的幂类别分池，兼顾小负载抗碎片与大负载少分配次数。
    VkDeviceSize ladder_max = 0;
    {
        const std::string v = VulkanDevice::get_env("NN_POOL_LADDER_MAX_MB");
        if (!v.empty())
        {
            const unsigned long long mb = std::strtoull(v.c_str(), nullptr, 10);
            if (mb > 0) ladder_max = static_cast<VkDeviceSize>(mb) * 1024ull * 1024ull;
        }
    }
    memory_pool_ =
        std::make_unique<MemoryPool>(device_.device(), device_.physical_device(), block_bytes,
                                     MemoryPool::DEFAULT_SMALL_BLOCK_SIZE, ladder_max);
    transient_pool_ =
        std::make_unique<MemoryPool>(device_.device(), device_.physical_device(), block_bytes,
                                     MemoryPool::DEFAULT_SMALL_BLOCK_SIZE, ladder_max);
    // P3（buffer device address）：设备支持即让**全部**池分配可取设备地址
    // （实例表引用张量缓冲；块内子分配同样覆盖），须在首次 allocate 前设置
    memory_pool_->set_device_address_alloc(device_.bda_available());
    transient_pool_->set_device_address_alloc(device_.bda_available());

    // 5. 创建 command pool
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = device_.queue_family_index();
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    auto r =
        detail::vk_check(vkCreateCommandPool(device_.device(), &pool_info, nullptr, &command_pool_),
                         __FILE__, __LINE__);
    if (!r) return r;

    // 6. 创建 staging ring
    //    use_timeline = 设备是否支持时间线信号量（跨 submit 依赖的正确
    //    原语；不支持则 staging 环走 host 等 fence 回退，见 StagingRing）
    staging_ring_ = std::make_unique<StagingRing>();
    auto st_r =
        staging_ring_->initialize(device_.device(), device_.physical_device(), command_pool_,
                                  *memory_pool_, device_.has_timeline_semaphores());
    if (!st_r) return st_r;

    // 6.5 异步标量回读槽：非阻塞取 loss 标量，避免每 step drain
    auto rb_r = init_scalar_readback_slots();
    if (!rb_r) return rb_r;

    // 7. 创建 descriptor pool for GPU-resident path
    // elementwise_v2 需要 4 个描述符/次，按最大值计算
    // batch 模式下描述符集延迟到 end_batch 释放，需足够大以容纳整个 batch
    //   ViT forward+backward ≈ 230 ops/样本，batch_size=512 时累积 ~118K sets
    //   增大到 262144 以留出余量，对应 desc 总量 1M
    constexpr std::size_t TENSOR_POOL_SETS = 262144;
    constexpr std::size_t TENSOR_POOL_DESCS = TENSOR_POOL_SETS * 4; // elementwise_v2 需要 4

    VkDescriptorPoolSize tensor_pool_size{};
    tensor_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    tensor_pool_size.descriptorCount = static_cast<uint32_t>(TENSOR_POOL_DESCS);

    VkDescriptorPoolCreateInfo tensor_pool_info{};
    tensor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    tensor_pool_info.maxSets = static_cast<uint32_t>(TENSOR_POOL_SETS);
    tensor_pool_info.poolSizeCount = 1;
    tensor_pool_info.pPoolSizes = &tensor_pool_size;
    tensor_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    r = detail::vk_check(
        vkCreateDescriptorPool(device_.device(), &tensor_pool_info, nullptr, &gpu_tensor_pool_),
        __FILE__, __LINE__);
    if (!r) return r;

    // 7b. 预分配流水线帧环：N 个 command buffer + N 个 fence，
    //     环内轮转复用、全程零分配——录制/提交路径不触碰
    //     vkAllocateCommandBuffers / vkCreateFence。
    frames_.resize(PIPELINE_FRAMES);
    {
        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = command_pool_;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = static_cast<uint32_t>(PIPELINE_FRAMES);
        std::vector<VkCommandBuffer> cmds(PIPELINE_FRAMES, VK_NULL_HANDLE);
        r = detail::vk_check(vkAllocateCommandBuffers(device_.device(), &cmd_alloc, cmds.data()),
                             __FILE__, __LINE__);
        if (!r) return r;
        for (std::size_t i = 0; i < PIPELINE_FRAMES; ++i)
            frames_[i].cmd = cmds[i];

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // 初始 unsignaled：vkQueueSubmit 要求 fence 未 signal
        for (std::size_t i = 0; i < PIPELINE_FRAMES; ++i)
        {
            r = detail::vk_check(
                vkCreateFence(device_.device(), &fence_info, nullptr, &frames_[i].fence), __FILE__,
                __LINE__);
            if (!r) return r;
        }
    }
    frame_next_ = 0;
    last_active_frame_ = 0;

    // 7c. 独立模式复用 fence + command buffer（见成员注释；省去
    //     每算子 create/destroy + alloc/free 的 host/driver 开销）
    {
        VkCommandBufferAllocateInfo cmd_alloc{};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = command_pool_;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;
        r = detail::vk_check(vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &solo_cmd_),
                             __FILE__, __LINE__);
        if (!r) return r;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // 初始 unsignaled：vkQueueSubmit 要求 fence 未 signal
        r = detail::vk_check(vkCreateFence(device_.device(), &fence_info, nullptr, &solo_fence_),
                             __FILE__, __LINE__);
        if (!r) return r;
    }

    // 7d. timestamp query pool（NN_PROFILE 设备侧计时；按帧分块 + solo 块）
    //     所选队列族 timestampValidBits=0（部分虚拟/软件设备）→ 整体
    //     降级 host 计时，ts_enabled_ 保持 false，零额外开销。
    ts_enabled_ = nn::prof::enabled() && device_.queue_timestamp_valid_bits() > 0;
    if (ts_enabled_)
    {
        ts_period_ns_ = device_.timestamp_period_ns();
        VkQueryPoolCreateInfo qp{};
        qp.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qp.queryCount = (PIPELINE_FRAMES + 1) * kTsPairsPerBlock * 2;
        const VkResult qres = vkCreateQueryPool(device_.device(), &qp, nullptr, &ts_pool_);
        if (qres != VK_SUCCESS)
        {
            ts_enabled_ = false;
            ts_pool_ = VK_NULL_HANDLE;
            std::fprintf(stderr, "[profile] timestamp query pool 创建失败(%d)，设备侧计时降级\n",
                         static_cast<int>(qres));
        }
    }
    for (std::size_t i = 0; i < PIPELINE_FRAMES; ++i)
        frames_[i].ts.base = static_cast<std::uint32_t>(i * kTsPairsPerBlock * 2);
    ts_solo_.base = static_cast<std::uint32_t>(PIPELINE_FRAMES * kTsPairsPerBlock * 2);

#ifdef NN_FUSED_REGISTRY_EMBEDDED
    // 16. 注册 AOT 融合 shader pipelines（构建期 scan_exprs 收集 +
    //     fused_generate.hpp 合成）。条目形态：
    //       · V0 / V2 / V1s / V1t（键 `key` / `key#a` / `key#s` /
    //         `key#xxxx` 特化签名）：N 输入 + 1 输出 binding；
    //       · V1 运行期精度分派（键 `key#x`）：2N + 2 binding（每个输入/输出
    //         各绑 f32 与 f16 两个视图）；PC 固定头末尾多一个 `prec` 槽。
    //     push constants = 固定头（count/cols/…）+ vp 槽 + 常量池 + rp 槽。
    for (const auto& fs : nn::fused::kFusedShaders)
    {
        if (!fs.spirv || fs.spirv_words == 0) continue; // 空 SPIR-V：跳过（构建配置缺失）
        // 运行期精度分派变体（键 "#x"）：输入/输出各双视图 + PC 多一个 prec 槽
        const bool is_disp = std::string_view(fs.key).ends_with(nn::EXPR_PREC_DISPATCH_SUFFIX);
        // P3 批量变体（键 "#b"）：单 binding（实例表），PC 与 V0 同布局
        const bool is_batch = std::string_view(fs.key).ends_with(nn::EXPR_BATCH_SUFFIX);
        // P0 细 tile 变体（键 "#f"）：V0 同布局（同绑定数、同 PC），
        // 只是另一套网格粒度（输出块 16×16）的 f32 shader。
        const bool is_fine = std::string_view(fs.key).ends_with(nn::EXPR_MM_FINE_SUFFIX);
        // K 切分变体（键 "#ks"，每 WG 一段 K → 部分和）与其伴生归约
        // （键 "#kr"）：f32 变体，PC/绑定见 EXPR_MM_KSPLIT_SUFFIX 注释。
        const bool is_ks = std::string_view(fs.key).ends_with(nn::EXPR_MM_KSPLIT_SUFFIX);
        const bool is_kr = std::string_view(fs.key).ends_with(nn::EXPR_MM_KSPLIT_REDUCE_SUFFIX);
        // 带 '#' 的 f16 变体（#x / #s / #xxxx 特化 / #a）需要 SSBO 16 位
        // 存储；设备未启用 → 跳过注册（运行时查不到 → 基类 NVI 入口回退
        // 边界 cast）。`#b` 与**纯 V0 的** `#f`/`#ks`/`#kr` 全 f32 无此
        // 要求（不得被这条误伤）。
        // ⚠ 后缀**可叠加**：`key#10003#ks` = f16 特化的 K 切分，其 shader
        //   里有 float16_t 缓冲 → 仍需要 16 位存储。判据 = 去掉网格粒度
        //   尾后缀（#kr/#ks/#f）之后是否仍含 '#'。
        const std::string_view key_sv(fs.key);
        std::string_view grid_stripped = key_sv;
        for (std::string_view suf : {std::string_view(nn::EXPR_BATCH_SUFFIX),
                                     std::string_view(nn::EXPR_MM_KSPLIT_REDUCE_SUFFIX),
                                     std::string_view(nn::EXPR_MM_KSPLIT_SUFFIX),
                                     std::string_view(nn::EXPR_MM_FINE_SUFFIX)})
        {
            if (grid_stripped.size() > suf.size() && grid_stripped.ends_with(suf))
            {
                grid_stripped.remove_suffix(suf.size());
                break;
            }
        }
        const bool needs_16bit_storage = key_sv.find('#') != std::string_view::npos &&
                                         !((is_batch || is_fine || is_ks || is_kr) &&
                                           grid_stripped.find('#') == std::string_view::npos);
        if (needs_16bit_storage && !device_.has_16bit_storage()) continue;
        // `#b` 批量变体需要 buffer device address（实例表按 64 位地址解引用
        // 各实例缓冲）；未启用 → 跳过注册 → 运行期逐 kernel 降级（可观测）。
        if (is_batch && !device_.bda_available()) continue;
        // ALU 变体（key 以 "#a" 结尾，native16 原生 f16 算术）额外需要
        // shaderFloat16 特性；未启用 → 跳过 → 运行时回退 `#x`（其 pipeline
        // 独立存在）或边界 cast——正确性不变。
        if (std::string_view(fs.key).ends_with(nn::EXPR_PREC_ALU_SUFFIX) &&
            !device_.has_shader_float16())
            continue;
        const std::uint32_t n_in = static_cast<std::uint32_t>(fs.spec.views.size());
        // 分派变体：输入 i 的 f32 视图 binding=i、f16 视图 binding=n_in+i；
        // 输出 f32/f16 视图 binding=2*n_in / 2*n_in+1。
        // P2 多输出（extras）：V0 追加 k 个输出 binding（bex<k>，主输出之后）；
        // `#x`/`#a` 变体不生成 extras（生成器跳过）⇒ 分派分支恒 2N+2。
        // `#b` 批量变体：只绑 binding=0 一个 storage buffer（实例表）。
        const std::uint32_t num_bindings =
            is_batch  ? 1u
            : is_disp ? (2u * n_in + 2u)
                      : (n_in + 1u + static_cast<std::uint32_t>(fs.spec.extras.size()));
        // push constant **固定头**槽数（uint）：逐形态与生成器的 PC 声明
        // 逐槽一致——
        //   逐元素 2（count, cols）/ +matmul 6（…, rows, mm_k, mm_batch,
        //   mm_trans）/ 归约 4（…, rows, vector_out）/ 归约+matmul 7 /
        //   fold 5（…, vector_out, fold_k）/ fold+mm 7（…, fold_k, mm_k,
        //   mm_batch）+ 分派变体 1 个 `prec`。
        // ⚠ 该值**只在此处计算一次**并存进 FusedKernel::pc_base_fixed，写入侧
        //   (run_fused_gpu) 只读：此前两侧各推导一遍，漂移会让
        //   vkCmdPushConstants 超 range 而被驱动丢弃 → shader 读到未定义
        //   残留（随前序 op 漂移，表现为时对时错的假 PASS，"全绿"验不出）。
        const std::uint32_t pc_base_fixed = is_ks ? 7u // K 切分：matmul 6 槽 + ks_span
                                            : (fs.spec.fold && fs.spec.fold->matmul)
                                                ? 7u // fold+mm 7 槽
                                            : (fs.spec.fold)                         ? 5u
                                            : (fs.reduce_axis >= 0 && fs.has_matmul) ? 7u
                                            : (fs.reduce_axis >= 0)                  ? 4u
                                            : (fs.has_matmul)                        ? 6u
                                                                                     : 2u;
        // 分派变体在固定头末尾多一个 uint prec（生成器同位置声明）
        const std::uint32_t pc_base = pc_base_fixed + (is_disp ? 1u : 0u);
        const std::uint32_t pc_uints = pc_base + fs.view_param_count;
        const std::uint32_t pc_size = static_cast<std::uint32_t>(
            pc_uints * sizeof(std::uint32_t) + sizeof(Scalar) * fs.spec.consts.size() +
            sizeof(Scalar) * fs.spec.rparams.size());
        auto fp_r = VulkanPipeline::create_generic(
            device_.device(), std::span<const std::uint32_t>(fs.spirv, fs.spirv_words),
            num_bindings, pc_size);
        if (fp_r)
        {
            FusedKernel kernel;
            kernel.pipeline = std::move(*fp_r);
            kernel.reduce_axis = fs.reduce_axis;
            kernel.has_matmul = fs.has_matmul != 0;
            kernel.fold = FoldMeta{
                fs.spec.fold ? (fs.spec.fold->vec_state_len > 0 ? fs.spec.fold->vec_state_len : 1u)
                             : 0u,
                // per_thread_row 判据与 generate_glsl 分派同源（v2 = 有
                // vec 态或 mm 段；v1 = 纯标量）
                static_cast<std::uint8_t>(
                    fs.spec.fold && !(fs.spec.fold->vec_state_len > 0 || fs.spec.fold->matmul))};
            kernel.view_param_count = fs.view_param_count;
            kernel.rparam_count = fs.rparam_count;
            kernel.vec_width = fs.vec_width;
            kernel.pc_base_fixed = pc_base_fixed;
            // P3 批量变体：实例表 stride（uvec2 槽数），run_fused_gpu_batch
            // 据此校验实例布局（生成器 S = 输入+主输出+extras+尾槽，同源）
            if (is_batch)
                kernel.batch_stride =
                    static_cast<std::uint32_t>(n_in + 1u + fs.spec.extras.size() + 1u);
            fused_kernels_.emplace(fs.key, std::move(kernel));
        }
    }
#endif

    // ── 查询 GPU f16 硬件能力（§7.1）─────────────────────────────
    // shaderFloat16：Vulkan 1.2 核心特性 / VK_KHR_shader_float16_int8 扩展。
    // 以**设备侧已启用**状态为准（VulkanDevice::initialize 查询 + 启用 +
    // NN_VULKAN_NO_16BIT_ALU 逃生阀）——只查物理特性不启用 = pipeline 创建
    // 时校验失败，故此处不独立查询。
    has_shader_float16_ = device_.has_shader_float16();

    initialized_ = true;
    return {};
}

[[nodiscard]] bool is_initialized() const noexcept
{ return initialized_; }
[[nodiscard]] bool gpu_available() const noexcept
{ return initialized_; }
// ── Pipeline 变体可用性 ─────────────────────────────────────────
// 唯一查询入口：id 见 compute_vk_pipeline_catalog.hpp；缺失 = 该变体
// 未创建（SPIR-V 未嵌入 / 设备能力不足 / 创建失败）。
[[nodiscard]] bool has(PipelineId id) const noexcept
{ return pipelines_.has(id); }

// 扫描族 f16 变体的整体可用性：四条路径齐备才算接通（缺任一 → 调用方
// 回退边界 cast，避免"部分直读、部分 cast"的混合语义）。
[[nodiscard]] bool has_scan_f16_pipelines() const noexcept
{
    return has(PipelineId::scan_prefix_outer_f16) && has(PipelineId::scan_suffix_outer_f16) &&
           has(PipelineId::scan_prefix_outer_gen_f16) && has(PipelineId::scan_suffix_outer_gen_f16);
}

// ── f16 硬件能力查询（§7.1）────────────────────────────────────
[[nodiscard]] bool has_shader_float16() const noexcept
{ return has_shader_float16_; }

[[nodiscard]] VulkanDevice& device() noexcept
{ return device_; }
[[nodiscard]] MemoryPool& memory_pool() noexcept
{ return *memory_pool_; }
[[nodiscard]] MemoryPool& transient_pool() noexcept
{ return *transient_pool_; }
// 延迟销毁队列总字节（探针归因用）：已析构但因帧未完成而尚未归还池的
// 内存。解释"Tensor 已置空但 live 不降"的账目缺口。
[[nodiscard]] VkDeviceSize pending_destroy_bytes()
{
    std::lock_guard lock(pending_mutex_);
    VkDeviceSize total = 0;
    for (const auto& pd : pending_destroys_)
        total += pd.alloc.size;
    return total;
}
// 瞬态/持久分池选择器：batch 录制期→瞬态池，否则→持久池（纯组织性）。
// 供 GpuTensorT 分配（create_empty / from_matrix / create_host_visible_empty）
// 使用，使批量内的临时/激活与构建期的参数/权重分池驻留。
[[nodiscard]] MemoryPool& alloc_pool() noexcept
{ return batch_mode_ ? *transient_pool_ : *memory_pool_; }
// 异步标量回读槽位数：供 GpuEngine 暴露给调用方做环形复用。
[[nodiscard]] static constexpr std::size_t scalar_readback_slot_count() noexcept
{ return SCALAR_READBACK_SLOTS; }
[[nodiscard]] StagingRing& staging_ring() noexcept
{ return *staging_ring_; }
[[nodiscard]] VkCommandPool command_pool() const noexcept
{ return command_pool_; }
[[nodiscard]] VkDescriptorPool gpu_tensor_pool() const noexcept
{ return gpu_tensor_pool_; }

// ── 非阻塞收割所有已完成帧（不含空闲块归还）────────────────────
// 显存峰值关键项：收割必须紧随每个帧提交点（submit_frame_no_wait）。
// 收割若推迟到 step 边界，步内已析构中间张量的延迟销毁会一直挂在
// "已完成但未被收割"的帧上、内存不还池 → backward 只能继续申请新底材
// （实测 bench 配置 transient live 3837MB 里 pending（已析构未还池）
//  高达 3000MB，步内峰值 ≈ 稳态 2.8×）。
[[nodiscard]] Result<void> reap_completed_frames()
{
    if (!initialized_) return {};
    for (std::size_t i = 0; i < frames_.size(); ++i)
    {
        if (!frames_[i].in_flight) continue;
        // 非阻塞查询：vkWaitForFences(timeout=0) 返回 VK_SUCCESS
        // （已就绪）或 VK_TIMEOUT（未就绪）。VK_NOT_READY 是
        // vkGetFenceStatus 的返回值，vkWaitForFences 不使用它。
        const VkResult rs = vkWaitForFences(device_.device(), 1, &frames_[i].fence, VK_TRUE, 0);
        if (rs == VK_ERROR_DEVICE_LOST)
        {
            // 设备丢失（TDR 触发）：GPU 已死亡，无法恢复
            device_lost_ = true;
            NN_FAIL("GPU 设备丢失 (VK_ERROR_DEVICE_LOST): Windows TDR 已重置 GPU 驱动。"
                    "\n模型已自动保存，请使用 --resume <save-path> 重启训练。"
                    "\n建议：减小 --batch-size 或 --seq-len，或增大 Windows TDR 超时"
                    " (注册表 TdrDelay)");
        }
        if (rs != VK_TIMEOUT && rs != VK_SUCCESS)
            NN_FAIL(std::string("GPU fence 非阻塞查询失败: Vulkan error ") +
                    std::to_string(static_cast<int>(rs)));
        if (rs == VK_SUCCESS)
        {
            auto rr = reap_frame(i, /*wait=*/false);
            NN_TRY_CHECK(rr);
        }
    }
    return {};
}

// ── 显存回收（L2）：先非阻塞收割已完成帧，再归还完全空闲的池底材 ──
// 调用时机：step 边界（end_batch 提交完成、延迟销毁已收割之后）。
// 未完成的帧留给环槽复用 / 下次 reap——不在 step 边界阻塞流水线。
[[nodiscard]] Result<void> release_idle_pool_blocks()
{
    auto r = reap_completed_frames();
    if (!r) return r;
    if (memory_pool_) memory_pool_->release_idle_blocks();
    if (transient_pool_) transient_pool_->release_idle_blocks();
    return {};
}
