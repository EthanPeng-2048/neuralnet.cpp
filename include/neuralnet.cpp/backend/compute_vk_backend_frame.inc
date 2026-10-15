// ═══════════════════════════════════════════════════════════════════════════
//  compute_vk_backend_frame.inc — GpuBackend 成员定义片段（RV-08 物理拆分，**纯移动、零行为变化**）
//  内容：帧环 / begin-end-batch / 阻塞上传下载 / 通用 dispatch / matmul / batched_matmul
//  ⚠ 本文件不是独立头：仅由 backend/compute_vk_backend.hpp 在
//    `class GpuBackend { ... }` 体内**按序** #include（顺序即成员序）。
// ═══════════════════════════════════════════════════════════════════════════
    // ══════════════════════════════════════════════════════════════════
    // Command Buffer Batching API（多帧流水线）
    // ══════════════════════════════════════════════════════════════════
    // 用法：
    //   backend.begin_batch();
    //   // ... 多次 matmul_gpu / elementwise_gpu / layernorm_gpu 调用 ...
    //   backend.end_batch();
    // 效果：所有操作录制到同一个 command buffer，一次提交（**不等待**）；
    // host 继续录制下一段，GPU 在队列上先行执行（host 录制与 GPU 执行
    // 重叠）。真正要读 GPU 数据时调 wait_in_flight()（to_matrix 内部）。
    // ══════════════════════════════════════════════════════════════════
    [[nodiscard]] bool in_batch() const noexcept { return batch_mode_; }

    // ── 取下一帧并开始录制（begin_batch / flush_batch 共用）──────────
    // 环槽复用前等待：该帧若仍在飞行，等其 fence（这是唯一的常规等待点；
    // 环深度 N 限制 host 至多超前 N 帧）。
    [[nodiscard]] Result<void> start_frame()
    {
        const std::size_t i = frame_next_;
        frame_next_ = (i + 1) % PIPELINE_FRAMES;

        auto rr = reap_frame(i, /*wait=*/true);
        if (!rr)
        {
            batch_mode_ = false;
            NN_TRY_CHECK(rr);
        }

        auto& f = frames_[i];
        auto r = detail::vk_check(vkResetCommandBuffer(f.cmd, 0), __FILE__, __LINE__);
        NN_TRY_CHECK(r);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = detail::vk_check(vkBeginCommandBuffer(f.cmd, &begin_info), __FILE__, __LINE__);
        NN_TRY_CHECK(r);
        batch_cmd_ = f.cmd;
        batch_frame_ = i;
        last_active_frame_ = i;
        batch_has_ops_ = false;
        batch_mode_ = true;
        ts_begin_block(f.cmd, f.ts);   // 设备侧计时：本帧块开始
        return {};
    }

    // ── 提交当前帧（不等待）────────────────────────────────────────────
    // 空帧不提交（无 op 的帧跳过，省一次 vkQueueSubmit + 一个环槽）。
    [[nodiscard]] Result<void> submit_frame_no_wait()
    {
        auto& f = frames_[batch_frame_];
        if (!batch_has_ops_)
            return {};  // 空帧：不提交

        // reap 已确保 fence 处于 unsignaled（或从未提交）状态
        auto r = detail::vk_check(
            vkResetFences(device_.device(), 1, &f.fence), __FILE__, __LINE__);
        NN_TRY_CHECK(r);
        // 跨 submit 数据依赖：帧命令的输入可能刚由 batch 内的 from_matrix
        // 独立上传提交写入——等 in-flight region 信号量（GPU 队列内等待，
        // host 不阻塞）
        const auto sw = collect_staging_waits();

        {
            std::lock_guard lock(queue_mutex_);
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &f.cmd;
            VkTimelineSemaphoreSubmitInfo timeline_info{};
            attach_staging_waits(submit_info, sw, timeline_info);
            r = detail::vk_check(
                vkQueueSubmit(device_.compute_queue(), 1, &submit_info, f.fence),
                __FILE__, __LINE__);
        }
        NN_TRY_CHECK(r);
        f.in_flight = true;
        last_active_frame_ = batch_frame_;
        // 提交后立即非阻塞收割其它已完成帧：让步内已析构张量的内存尽早
        // 还池，被后续分配复用（否则要等到环槽复用或 step 边界）。
        // NN_NO_EARLY_REAP=1 可关闭（A/B 对照用，见 bench/run_mem_ab.ps1）。
        if (early_reap_enabled_)
        {
            auto rr = reap_completed_frames();
            NN_TRY_CHECK(rr);
        }
        return {};
    }

    [[nodiscard]] Result<void> begin_batch()
    {
        if (!initialized_)
            NN_FAIL("GPU backend not initialized");
        if (batch_mode_)
            NN_FAIL("Already in batch mode");
        return start_frame();
    }

    // ── 结束 batch：提交当前帧，**不等待**─────────────────────
    // 帧在队列上执行；其延迟销毁 / 描述符集在 reap 时释放。
    // 设备丢失等错误在 wait_in_flight / 环槽复用 reap 时浮出。
    [[nodiscard]] Result<void> end_batch()
    {
        if (!batch_mode_)
            return {};  // 容错：非 batch 模式时 no-op

        // 1. 结束录制
        auto r = detail::vk_check(vkEndCommandBuffer(batch_cmd_), __FILE__, __LINE__);
        if (!r)
        {
            batch_mode_ = false;
            NN_TRY_CHECK(r);
        }

        // 2. 提交（不等待）
        r = submit_frame_no_wait();
        batch_mode_ = false;
        return r;
    }

    // ── 批处理中点刷新（防 TDR，提交不等待）────────────────────────
    // 提交当前帧（不等待），然后开始录制下一帧。用于拆分大 batch（如
    // forward 与 backward 之间），避免单次提交时间过长触发 Windows TDR。
    // 当前帧为空（无 op 录制）时 no-op——不提交、不换帧（避免浪费环槽
    // 与 submit，如 to_matrix 的 drain 已换过帧的场景）。
    [[nodiscard]] Result<void> flush_batch()
    {
        if (!batch_mode_)
            return {};  // 非 batch 模式时 no-op
        if (!batch_has_ops_)
            return {};  // 空帧：无需提交，继续在同一帧录制

        auto r = detail::vk_check(vkEndCommandBuffer(batch_cmd_), __FILE__, __LINE__);
        if (!r)
        {
            batch_mode_ = false;
            NN_TRY_CHECK(r);
        }

        r = submit_frame_no_wait();  // 不等待
        if (!r)
        {
            batch_mode_ = false;
            NN_TRY_CHECK(r);
        }

        // 仍处于 batch 模式：开始录制下一帧
        return start_frame();
    }

    // ── "真正要结果"的阻塞点───────────────────────────────────
    // 等待所有在飞帧完成并 reap（延迟销毁 + 描述符集释放）。
    // to_matrix / copy_from 读 GPU 内存前必须调用：要读的数据可能刚由
    // 在飞帧写入，队列 FIFO 只保证顺序、不保证完成。
    [[nodiscard]] Result<void> wait_in_flight()
    {
        for (std::size_t i = 0; i < frames_.size(); ++i)
        {
            if (frames_[i].in_flight)
            {
                auto r = reap_frame(i, /*wait=*/true);
                NN_TRY_CHECK(r);
            }
        }
        // 所有帧已完成：兜底清理剩余延迟销毁（正常应为空）
        flush_pending_destroys();
        return {};
    }

    // ── 阻塞式上传：CPU → GPU（支持分块传输大矩阵）─────────────────────
    // 当数据超过 staging region 大小时，自动分块上传。
    // 每块大小不超过 staging region，通过多次 memcpy + vkCmdCopyBuffer 完成。
    // 元素类型由 P 决定（f32=4B / f16=2B），memcpy 为字节级操作，§6.5。
    template <Precision P>
    [[nodiscard]] Result<void> upload_blocking(
        GpuTensorT<P>& dst, std::span<const elem<P>> cpu_data)
    {
        using ElemType = elem<P>;
        if (!initialized_)
            NN_FAIL("GPU backend not initialized");
        if (!dst.valid())
            NN_FAIL("Invalid destination GpuTensor");
        const std::size_t elem_count = dst.rows() * dst.cols();
        if (cpu_data.size() != elem_count)
            NN_FAIL("Upload size mismatch");
        const std::size_t total_bytes = elem_count * sizeof(ElemType);
        const std::size_t staging_cap = staging_ring_->region_size();

        // ── 小矩阵快速路径：单次传输 ──────────────────────────────────
        // region 专属 command buffer：上传 copy 录在 region 自己的
        // cmd 上，提交后既不 free 也不立刻复用——VUID-
        // vkFreeCommandBuffers-pCommandBuffers-00058 禁止释放 pending
        // （已提交、fence 未 signal）的 command buffer；"submit 后立即
        // free"是规范违规（实测导致驱动通道排序失效、fence 提前
        // signal、跨 submit 乱序）。acquire 已等在飞 fence → 该 cmd 的
        // 上一次使用完成（invalid 状态）→ vkResetCommandBuffer 复用合法。
        if (total_bytes <= staging_cap)
        {
            auto ri = staging_ring_->acquire();

            auto r = staging_ring_->upload(ri, cpu_data, 0);
            if (!r) return r;

            auto cmd = staging_ring_->command_buffer(ri);

            r = detail::vk_check(vkResetCommandBuffer(cmd, 0), __FILE__, __LINE__);
            if (!r) return r;

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
            if (!r) return r;

            VkBufferCopy cp{0, 0, total_bytes};
            vkCmdCopyBuffer(cmd, staging_ring_->buffer(ri), dst.buffer().impl(), 1, &cp);

            r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
            if (!r) return r;

            auto fence = staging_ring_->fence(ri);
            vkResetFences(device_.device(), 1, &fence);

            {
                std::lock_guard lock(queue_mutex_);
                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cmd;
                // 跨 submit 数据依赖：时间线信号量 signal 本次
                // 上传的 value。读取"本 copy 写入的 buffer"的后续 submit 必须
                // wait 该 value（见 collect_staging_waits）。
                VkTimelineSemaphoreSubmitInfo timeline_info{};
                VkSemaphore sem = VK_NULL_HANDLE;
                std::uint64_t signal_value = 0;
                attach_upload_signal(submit_info, *staging_ring_, ri,
                                     timeline_info, sem, signal_value);

                r = detail::vk_check(
                    vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence),
                    __FILE__, __LINE__);
            }
            if (!r) return r;

            // 提交后不等待——host 可立即继续录制。标记 region
            // in use：下次 acquire 到该 region 时等其 fence（staging 安全
            // + 等待后该 cmd 离开 pending，可 reset 复用）。**不释放 cmd**
            // （pending 状态，见上方 VUID 注释）。
            staging_ring_->mark_in_flight(ri);
            return {};
        }

        // ── 大矩阵分块上传 ───────────────────────────────────────────
        // 每块大小对齐到 sizeof(ElemType)，确保元素边界对齐
        const std::size_t chunk_bytes = (staging_cap / sizeof(ElemType)) * sizeof(ElemType);
        std::size_t dst_offset = 0;

        while (dst_offset < total_bytes)
        {
            const std::size_t this_chunk = std::min(chunk_bytes, total_bytes - dst_offset);
            const std::size_t this_elems = this_chunk / sizeof(ElemType);
            const std::size_t src_elem_offset = dst_offset / sizeof(ElemType);

            // 1. 获取 staging region 并上传当前块
            auto ri = staging_ring_->acquire();
            auto r = staging_ring_->upload(
                ri,
                std::span<const ElemType>(cpu_data.data() + src_elem_offset, this_elems),
                0);
            if (!r) return r;

            // 2. 录制 copy 命令（region 专属 cmd buffer：禁止 pending 释放，
            //    见小矩阵路径的 VUID 注释；acquire 已等在飞 fence，reset 合法）
            auto cmd = staging_ring_->command_buffer(ri);

            r = detail::vk_check(vkResetCommandBuffer(cmd, 0), __FILE__, __LINE__);
            if (!r) return r;

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
            if (!r) return r;

            VkBufferCopy cp{0, dst_offset, this_chunk};
            vkCmdCopyBuffer(cmd, staging_ring_->buffer(ri), dst.buffer().impl(), 1, &cp);

            r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
            if (!r) return r;

            // 3. 提交（不等待；region 标记 in use，下次 acquire 到该
            //    region 时等其 fence；不释放 cmd——pending 状态）+ 信号本
            //    region 信号量（跨 submit 数据依赖，见 collect_staging_waits）
            auto fence = staging_ring_->fence(ri);
            vkResetFences(device_.device(), 1, &fence);

            {
                std::lock_guard lock(queue_mutex_);
                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cmd;
                // 跨 submit 数据依赖：时间线信号量 signal 本次上传的 value
                // （读取本 copy 写入 buffer 的后续 submit 会 wait 该 value）
                VkTimelineSemaphoreSubmitInfo timeline_info{};
                VkSemaphore sem = VK_NULL_HANDLE;
                std::uint64_t signal_value = 0;
                attach_upload_signal(submit_info, *staging_ring_, ri,
                                     timeline_info, sem, signal_value);

                r = detail::vk_check(
                    vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence),
                    __FILE__, __LINE__);
            }
            if (!r) return r;

            staging_ring_->mark_in_flight(ri);
            dst_offset += this_chunk;
        }

        return {};
    }

    // ── 阻塞式下载：GPU → CPU（支持分块传输大矩阵）─────────────────────
    // 当数据超过 staging region 大小时，自动分块下载。
    // 元素类型由 P 决定（f32=4B / f16=2B），memcpy 为字节级操作，§6.5。
    template <Precision P>
    [[nodiscard]] Result<void> download_blocking(
        const GpuTensorT<P>& src, std::span<elem<P>> cpu_data)
    {
        using ElemType = elem<P>;
        if (!initialized_)
            NN_FAIL("GPU backend not initialized");
        if (!src.valid())
            NN_FAIL("Invalid source GpuTensor");
        const std::size_t elem_count = src.rows() * src.cols();
        if (cpu_data.size() != elem_count)
            NN_FAIL("Download size mismatch");
        const std::size_t total_bytes = elem_count * sizeof(ElemType);
        const std::size_t staging_cap = staging_ring_->region_size();

        // ── 小矩阵快速路径：单次传输 ──────────────────────────────────
        if (total_bytes <= staging_cap)
        {
            auto ri = staging_ring_->acquire();

            VkCommandBufferAllocateInfo cmd_alloc{};
            cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmd_alloc.commandPool = command_pool_;
            cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmd_alloc.commandBufferCount = 1;

            VkCommandBuffer cmd = VK_NULL_HANDLE;
            auto r = detail::vk_check(
                vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd),
                __FILE__, __LINE__);
            if (!r) return r;

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            VkBufferCopy cp{0, 0, total_bytes};
            vkCmdCopyBuffer(cmd, src.buffer().impl(), staging_ring_->buffer(ri), 1, &cp);

            r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            auto fence = staging_ring_->fence(ri);
            vkResetFences(device_.device(), 1, &fence);

            // 跨 submit 数据依赖：源 buffer 可能刚由在飞上传的 copy 写入
            // （from_matrix → to_matrix 无中间 op 即此场景）——等 in-flight
            // region 信号量（本下载自用的 region 已在 acquire 中等待并重置，
            // 不在 in-flight 集合内）。时间线信号量允许同一 value 被多个
            // submit 等待，故这里与前面的 matmul 帧重复等待也合法。
            const auto sw = collect_staging_waits();

            {
                std::lock_guard lock(queue_mutex_);
                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cmd;
                VkTimelineSemaphoreSubmitInfo timeline_info{};
                attach_staging_waits(submit_info, sw, timeline_info);

                r = detail::vk_check(
                    vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence),
                    __FILE__, __LINE__);
                if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }
            }

            r = detail::vk_check(
                vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, 10'000'000'000ULL),
                __FILE__, __LINE__);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            r = staging_ring_->download(ri, cpu_data, 0);

            vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
            return r;
        }

        // ── 大矩阵分块下载 ───────────────────────────────────────────
        const std::size_t chunk_bytes = (staging_cap / sizeof(ElemType)) * sizeof(ElemType);
        std::size_t src_offset = 0;

        while (src_offset < total_bytes)
        {
            const std::size_t this_chunk = std::min(chunk_bytes, total_bytes - src_offset);
            const std::size_t this_elems = this_chunk / sizeof(ElemType);
            const std::size_t dst_elem_offset = src_offset / sizeof(ElemType);

            // 1. 获取 staging region
            auto ri = staging_ring_->acquire();

            // 2. 录制 copy 命令：GPU → staging
            VkCommandBufferAllocateInfo cmd_alloc{};
            cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmd_alloc.commandPool = command_pool_;
            cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmd_alloc.commandBufferCount = 1;

            VkCommandBuffer cmd = VK_NULL_HANDLE;
            auto r = detail::vk_check(
                vkAllocateCommandBuffers(device_.device(), &cmd_alloc, &cmd),
                __FILE__, __LINE__);
            if (!r) return r;

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            r = detail::vk_check(vkBeginCommandBuffer(cmd, &begin_info), __FILE__, __LINE__);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            VkBufferCopy cp{src_offset, 0, this_chunk};
            vkCmdCopyBuffer(cmd, src.buffer().impl(), staging_ring_->buffer(ri), 1, &cp);

            r = detail::vk_check(vkEndCommandBuffer(cmd), __FILE__, __LINE__);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            // 3. 提交并等待（+ 跨 submit 数据依赖：等 in-flight region
            //    信号量，源 buffer 可能刚由在飞上传的 copy 写入）
            auto fence = staging_ring_->fence(ri);
            vkResetFences(device_.device(), 1, &fence);

            const auto sw = collect_staging_waits();

            {
                std::lock_guard lock(queue_mutex_);
                VkSubmitInfo submit_info{};
                submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit_info.commandBufferCount = 1;
                submit_info.pCommandBuffers = &cmd;
                VkTimelineSemaphoreSubmitInfo timeline_info{};
                attach_staging_waits(submit_info, sw, timeline_info);

                r = detail::vk_check(
                    vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence),
                    __FILE__, __LINE__);
                if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }
            }

            r = detail::vk_check(
                vkWaitForFences(device_.device(), 1, &fence, VK_TRUE, 10'000'000'000ULL),
                __FILE__, __LINE__);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            // 4. 从 staging 下载当前块到 CPU
            r = staging_ring_->download(
                ri,
                std::span<ElemType>(cpu_data.data() + dst_elem_offset, this_elems),
                0);
            if (!r) { vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd); return r; }

            vkFreeCommandBuffers(device_.device(), command_pool_, 1, &cmd);
            src_offset += this_chunk;
        }

        return {};
    }

    // ── 通用 compute dispatch（matmul 去重核心）──────────────────
    // 统一的 descriptor/cmd/barrier/dispatch/submit 样板，供 matmul /
    // batched_matmul / rearrange 等所有"N 输入 + 1 输出"的 compute kernel 复用。
    //
    // 参数：
    //   pipeline : 目标 compute pipeline
    //   inputs   : 输入张量（写入 binding 0..n-1）
    //   output   : 输出张量（写入最后一个 binding）
    //   pc       : push constants 字节流（按 pipeline_layout 布局）
    //   wg_x/y/z : dispatch 工作组数（三维）
    [[nodiscard]] Result<void> dispatch_compute(
        const VulkanPipeline& pipeline, std::span<const GpuTensor> inputs,
        const GpuTensor& output, const std::vector<std::uint8_t>& pc,
        std::uint32_t wg_x, std::uint32_t wg_y, std::uint32_t wg_z,
        const char* ts_label = "dispatch",   // 设备侧计时标号（NN_PROFILE）
        // per-key 归因 detail（NN_PROFILE；变体名如 "tiled"/"gemv"/"fine"，
        // 进表 5 的 "detail@label@stage" 行；空 = 只进表 4）
        std::string_view ts_detail = {})
    {
        if (!initialized_)
            NN_FAIL("GPU backend not initialized");
        const auto t_disp_start = std::chrono::steady_clock::now();
        auto ds_r = alloc_desc_set(pipeline.descriptor_layout());
        NN_TRY_CHECK(ds_r);
        VkDescriptorSet desc_set = *ds_r;

        const std::size_t n = inputs.size() + 1;
        std::vector<VkDescriptorBufferInfo> buf_infos(n);
        std::vector<VkWriteDescriptorSet> writes(n);
        for (std::size_t i = 0; i < inputs.size(); ++i)
            buf_infos[i] = {inputs[i].buffer().impl(), 0, VK_WHOLE_SIZE};
        buf_infos[n - 1] = {output.buffer().impl(), 0, VK_WHOLE_SIZE};
        for (std::size_t i = 0; i < n; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc_set;
            writes[i].dstBinding = static_cast<uint32_t>(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(device_.device(),
            static_cast<uint32_t>(n), writes.data(), 0, nullptr);

        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            release_desc_set_now(desc_set);
            NN_TRY_CHECK(cmd_r);
        }
        auto [cmd, owns_cmd] = *cmd_r;

        std::vector<VkBuffer> in_bufs;
        in_bufs.reserve(inputs.size());
        for (const auto& t : inputs)
            in_bufs.push_back(t.buffer().impl());
        record_input_barriers(cmd, in_bufs);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline_layout(), 0, 1, &desc_set, 0, nullptr);
        if (!pc.empty())
            vkCmdPushConstants(cmd, pipeline.pipeline_layout(),
                VK_SHADER_STAGE_COMPUTE_BIT, 0,
                static_cast<uint32_t>(pc.size()), pc.data());
        ts_dispatch(cmd, ts_label, wg_x, wg_y, wg_z, ts_detail);
        record_output_barrier(cmd, output.buffer().impl());

        if (owns_cmd)
        {
            const bool prof = profile_ops_enabled();
            const auto t_rec = std::chrono::steady_clock::now();
            auto r = submit_and_wait(cmd, desc_set);
            if (prof)
                std::fprintf(stderr,
                    "[gpu-profile] record+setup=%lldus（提交/等待分段见上一条 solo_op）\n",
                    static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(
                        t_rec - t_disp_start).count()));
            NN_TRY_CHECK(r);
        }
        return {};
    }

    // ── 纯 GPU 矩阵乘法（阻塞等待完成）──────────────────────────────
    // batch_mode_ 时：录制到 batch_cmd_，不提交不等待
    // transA: A 存储为 (K,M)，按 A^T 使用
    // transB: B 存储为 (N,K)，按 B^T 使用
    [[nodiscard]] Result<GpuTensor> matmul_gpu(
        const GpuTensor& A, const GpuTensor& B,
        uint32_t transA = 0, uint32_t transB = 0,
        bool f16_io = false)
    {
        if (!initialized_)
            NN_FAIL("GPU backend not initialized");
        // 计算有效维度（考虑转置）
        const auto M = static_cast<uint32_t>(transA ? A.cols() : A.rows());
        const auto K = static_cast<uint32_t>(transA ? A.rows() : A.cols());
        const auto N = static_cast<uint32_t>(transB ? B.rows() : B.cols());
        const auto K_B = static_cast<uint32_t>(transB ? B.cols() : B.rows());
        if (K != K_B)
            NN_FAIL("Dimension mismatch");
        // 1. 分配输出 Tensor（f16_io：按 2B/元素分配 + 纯绑定视图，见
        //    batched_matmul_gpu 的同款做法）
        const bool use_f16 = f16_io;
        // f16 可用性：N≤8 走 GEMV（C2e，仅需 gemv_f16），否则 tiled_f16
        if (use_f16 && N <= 8u && !has_matmul_tiled_f16_pipeline() &&
            has_gemv_f16_pipeline())
        {
            // gemv_f16 可用 → 下方分派走 GEMV，无需 tiled_f16
        }
        else if (use_f16 && !has_matmul_tiled_f16_pipeline())
            NN_FAIL("matmul_gpu: f16 pipeline 不可用（设备无 SSBO 16 位存储？）");
        std::optional<GpuTensor> owned;
        if (use_f16)
        {
            auto f16r = GpuTensorF16::create_empty(M, N, *this);
            NN_TRY_CHECK(f16r);
            owned.emplace(f16r->shared_buffer(), f16r->rows(), f16r->cols());
        }
        else
        {
            auto C_res = GpuTensor::create_empty(M, N, *this);
            NN_TRY_CHECK(C_res);
            owned.emplace(std::move(*C_res));
        }
        GpuTensor C = *owned;

        // 2. 选择 Pipeline（优先级：f16 存储版 > 小 N GEMV > 粗化分块 > naive）
        //   GEMV：N ≤ GEMV_MAX_N（shader ROWS=4 行/WG，grid.x=ceil(M/4)）。
        //   tiled 在小 N 下 64×64 块 (64-N)/64 列空转，GEMV 每线程只算
        //   真实 N 列（借鉴 ggml mul_mat_vec 分派）。
        constexpr uint32_t GEMV_MAX_N = 8;
        constexpr uint32_t GEMV_ROWS = 4;   // 与 matmul_gemv.comp 的 ROWS 一致
        // subgroup≥4 门禁：shader red[..][64] 的容量假设（256/4 = 64 槽）——
        //   更小 subgroup 会溢出 64 槽上限、部分和不落表 → 静默错值
        //   （"只有 GPU 错"一类，见 docs/todo/08-pitfalls-and-lessons.md
        //   §4.10；实测桌面卡恒 ≥8，门禁是保险丝）
        // Phase C2e：f16 也可走 GEMV（f16 变体 pipeline 存在时）——小 N
        //   （n_out≤8）f16 GEMM 直读直写；变体 pipeline 缺失时引擎回退
        //   边界 cast。
        const bool use_gemv = has_gemv_pipeline() && N <= GEMV_MAX_N
                              && device_.subgroup_size() >= 4u
                              && (!use_f16 || has_gemv_f16_pipeline());
        const bool use_tiled = (use_f16 || has_tiled_pipeline()) && !use_gemv;
        // P0 细 tile（wgrad 网格饥饿）：64-tile 网格只有
        // ceil(N/64)×ceil(M/64) 个工作组——反传 wgrad 形状（M,N=64..256 ×
        // K=16384）只派 1~4 个，40HX（34 SM）大面积空转（实测 208~278
        // GFLOPS vs 同卡方形 2489）。低于阈值换 16×16 输出块变体：网格细
        // 16×、每输出元素 K 累加序不变（逐位一致）。仅 f32（fine 变体无
        // f16 版），GEMV（N≤8）与 naive 兜底不受影响。
        bool use_fine = false;
        bool use_naive_fine = false;
        if (use_tiled && !use_f16 && !mm_fine_disabled())
        {
            const std::uint64_t wg64 =
                static_cast<std::uint64_t>((N + 63u) / 64u) *
                ((M + 63u) / 64u);
            if (wg64 < nn::EXPR_MATMUL_FINE_MIN_WGS)
            {
                // NN_MM_FINE_NAIVE=1 → naive（每线程 1 输出：线程数 = M×N，
                // 是细 tiled 的 16×；无 shared/barrier，靠 warp 切换吃延迟）
                use_naive_fine = mm_fine_naive() &&
                                 matmul_pipeline_.handle() != VK_NULL_HANDLE;
                use_fine = !use_naive_fine && has_tiled_fine_pipeline();
            }
        }
        auto& pipeline = use_gemv ? (use_f16 ? matmul_gemv_f16_pipeline_
                                             : matmul_gemv_pipeline_)
                       : use_f16 ? matmul_tiled_f16_pipeline_
                       : use_naive_fine ? matmul_pipeline_
                       : use_fine ? matmul_tiled_fine_pipeline_
                       : use_tiled ? matmul_tiled_pipeline_
                                   : matmul_pipeline_;

        // 3. 复用通用 dispatch：2 输入 + 1 输出，push {M,N,K,transA,transB}
        //   Dispatch：按 shader 输出块尺寸计算工作组数
        //     matmul.comp（naive）：16×16 线程网格 = 16×16 输出块
        //     matmul_tiled.comp：64×64 输出块（BM/BN）
        //     matmul_gemv.comp：每 WG ROWS=4 行 × N 列（单维 grid.x）
        //   ⚠ dispatch 必须按各 shader 的实际输出块尺寸计算（naive 16×16 /
        //   tiled 64×64 / GEMV ROWS=4 行），不可按 WORKGROUP_SIZE=16 统一算：
        //   那会让 tiled 版多派 16 倍冗余工作组（每 16×16 一个组而非 64×64），
        //   GPU 做 16 倍无效计算，实测 matmul 峰值只剩 ~4%（0.7/15.7 TFLOPS）。
        const uint32_t push_data[5] = {M, N, K, transA, transB};
        std::vector<std::uint8_t> pc(sizeof(push_data));
        std::memcpy(pc.data(), push_data, sizeof(push_data));

        std::vector<GpuTensor> inputs{A, B};
        // 设备侧计时 detail：把 matmul 的 pipeline 变体带进 per-key 表
        // （gemv / tiled / tiled_f16 / fine / naive_fine / naive）
        const char* mm_variant = use_gemv ? (use_f16 ? "gemv_f16" : "gemv")
            : use_naive_fine ? "naive_fine"
            : use_fine ? "fine"
            : use_tiled ? (use_f16 ? "tiled_f16" : "tiled")
            : "naive";
        if (use_gemv)
        {
            auto r = dispatch_compute(pipeline, inputs, C, pc,
                (M + GEMV_ROWS - 1u) / GEMV_ROWS, 1u, 1u, "matmul", mm_variant);
            NN_TRY_CHECK(r);
            return C;
        }
        const uint32_t tile = (use_fine || use_naive_fine) ? 16u
                             : use_tiled ? 64u : 16u;
        auto r = dispatch_compute(pipeline, inputs, C, pc,
            (N + tile - 1u) / tile, (M + tile - 1u) / tile, 1u,
            "matmul", mm_variant);
        NN_TRY_CHECK(r);
        return C;
    }

    // ── 纯 GPU 批量矩阵乘法 ──────────────────────────────────────────
    // 对每个 batch b 计算 C_b = alpha * op(A_b, B_b)，结果垂直堆叠为 (batch*M, N)
    // A: (batch * A_rows_per_batch, A_cols)，B: (batch * B_rows_per_batch, B_cols)
    // batch 步长由 shader 从 M*K / K*N / M*N 推导，无需额外传入
    // alpha: 输出缩放系数（cuBLAS sgemm 语义），在 shader 写出时一次完成
    //
    // f16_io（Phase 2 in-kernel f16）：A/B 是 **f16 缓冲**、C 也按 f16 分配
    //   （2B/元素）→ 用 f16 存储版 pipeline（f16 载入 + f32 累加 + f16 写出），
    //   彻底消除"每个操作数抬 f32 的整份副本"。此时 A/B 只作**缓冲视图**使用
    //   （GpuTensorT<F32> 包裹 f16 buffer，字节布局由 pipeline 决定 —— 与
    //   run_fused_gpu 的 out_f16 同一套做法）。
    [[nodiscard]] Result<GpuTensor> batched_matmul_gpu(
        const GpuTensor& A, const GpuTensor& B,
        uint32_t batch,
        uint32_t transA = 0, uint32_t transB = 0,
        float alpha = 1.0f,
        bool f16_io = false)
    {
        if (!initialized_)
            NN_FAIL("GPU backend not initialized");
        if (batch == 0)
            NN_FAIL("batched_matmul_gpu: batch must be > 0");
        const bool use_f16 = f16_io && has_batched_matmul_f16_pipeline();
        if (f16_io && !use_f16)
            NN_FAIL("batched_matmul_gpu: f16 pipeline 不可用（设备无 SSBO 16 位存储？）");
        if (!use_f16 && !has_batched_matmul_pipeline())
            NN_FAIL("batched_matmul_gpu: pipeline not available");
        // 校验：A.rows() 必须能被 batch 整除
        if (A.rows() % batch != 0 || B.rows() % batch != 0)
            NN_FAIL("batched_matmul_gpu: rows not divisible by batch");
        // 计算每个 batch 的逻辑维度
        const auto a_rows_per = static_cast<uint32_t>(A.rows() / batch);
        const auto b_rows_per = static_cast<uint32_t>(B.rows() / batch);
        const auto M = transA ? static_cast<uint32_t>(A.cols()) : a_rows_per;
        const auto K = transA ? a_rows_per : static_cast<uint32_t>(A.cols());
        const auto K_B = transB ? static_cast<uint32_t>(B.cols()) : b_rows_per;
        const auto N = transB ? b_rows_per : static_cast<uint32_t>(B.cols());
        if (K != K_B)
            NN_FAIL("batched_matmul_gpu: K dimension mismatch");
        // 1. 分配输出 Tensor: (batch * M, N)
        //    f16：按 2B/元素分配，再用 GpuTensor(shared_buffer,...) 包一层纯
        //    绑定视图交给 dispatch（调用方按 f16_io 重贴 GpuTensorF16）。
        std::optional<GpuTensor> owned;
        if (use_f16)
        {
            auto f16r = GpuTensorF16::create_empty(
                static_cast<std::size_t>(batch) * M, N, *this);
            NN_TRY_CHECK(f16r);
            owned.emplace(f16r->shared_buffer(), f16r->rows(), f16r->cols());
        }
        else
        {
            auto C_res = GpuTensor::create_empty(static_cast<std::size_t>(batch) * M, N, *this);
            NN_TRY_CHECK(C_res);
            owned.emplace(std::move(*C_res));
        }
        GpuTensor C = *owned;

        // 2. 复用通用 dispatch：2 输入 + 1 输出
        //   Push constants: M, N, K, transA, transB, alpha（24B，含输出缩放系数）
        //   shader 采用 64×64 寄存器分块（与 matmul_tiled 相同），
        //   每个 WorkGroup 计算 64×64 输出块，Z 维度索引 batch
        struct PushDataBmm {
            uint32_t M, N, K, transA, transB;
            float alpha;
        } push{M, N, K, transA, transB, alpha};
        std::vector<std::uint8_t> pc(sizeof(push));
        std::memcpy(pc.data(), &push, sizeof(push));

        // Dispatch: X/Y 覆盖 64×64 输出块，Z 维度 = batch
        constexpr uint32_t BM = 64, BN = 64;
        std::vector<GpuTensor> inputs{A, B};
        auto r = dispatch_compute(
            use_f16 ? batched_matmul_f16_pipeline_ : batched_matmul_pipeline_,
            inputs, C, pc, (N + BN - 1) / BN, (M + BM - 1) / BM, batch,
            "batched_matmul", use_f16 ? "f16" : "f32");
        NN_TRY_CHECK(r);
        return C;
    }

