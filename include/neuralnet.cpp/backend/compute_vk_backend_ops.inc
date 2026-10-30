// ═══════════════════════════════════════════════════════════════════════════
//  compute_vk_backend_ops.inc — GpuBackend 成员定义片段（RV-08 物理拆分，**纯移动、零行为变化**）
//  内容：各原语 dispatch（elementwise / reduce / transpose / im2col / cast / trit_pack / matmul_q …）
//  ⚠ 本文件不是独立头：仅由 backend/compute_vk_backend.hpp 在
//    `class GpuBackend { ... }` 体内**按序** #include（顺序即成员序）。
// ═══════════════════════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════════════
// elementwise_v2_gpu — 逐元素原语（unary/binary/select/axpy）
//
// Push Constants (32 bytes):
//   count, mode, op, cmp_op, flags, scalar_b, scalar_then, scalar_else
// Bindings: A(0), B(1), C(2), OUT(3)
//
// 对未使用的 binding（如 unary 模式的 B/C），绑定 A 的 buffer（无害占位）。
//
// out != nullptr 时为原地模式：OUT 直接绑定 out 的 buffer（通常即 A），
// 免分配 + 免全量写出。逐元素 kernel 每线程只读写自己下标一次，
// read-before-write 天然成立，别名安全。
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<GpuTensor>
elementwise_v2_gpu(const GpuTensor& A, const GpuTensor* B, const GpuTensor* C, uint32_t count,
                   uint32_t mode, uint32_t op, uint32_t cmp_op, uint32_t flags, float scalar_b,
                   float scalar_then, float scalar_else, const GpuTensor* out = nullptr,
                   bool f16_io = false)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (!has(PipelineId::elementwise_v2)) NN_FAIL("elementwise_v2 pipeline not available");
    // Phase C1b f16 变体（仅需 16 位存储门控）；请求但无 pipeline →
    // 报错由引擎层回退边界 cast
    if (f16_io && !has(PipelineId::elementwise_v2_f16))
        NN_FAIL("elementwise_v2_gpu: f16 pipeline 不可用（设备无16位存储）");
    const auto& pipeline = f16_io ? pipelines_.get(PipelineId::elementwise_v2_f16)
                                  : pipelines_.get(PipelineId::elementwise_v2);

    // 1. 输出 Tensor：原地模式直接复用 out 的 buffer，否则新分配
    //    （f16_io：按 2B/元素分配 + 纯绑定视图，同 matmul_gpu）
    GpuTensor output;
    if (out) { output = GpuTensor(*out); }
    else if (f16_io)
    {
        auto f16r = GpuTensorF16::create_empty(A.rows(), A.cols(), *this);
        NN_TRY_CHECK(f16r);
        output = GpuTensor(f16r->shared_buffer(), f16r->rows(), f16r->cols());
    }
    else
    {
        auto output_res = GpuTensor::create_empty(A.rows(), A.cols(), *this);
        NN_TRY_CHECK(output_res);
        output = std::move(*output_res);
    }

    // 2. 分配并写入描述符集（未使用的 binding 绑定 A 作为占位）
    VkBuffer a_buf = A.buffer().impl();
    const VkBuffer bindings[4]{
        a_buf,
        B ? B->buffer().impl() : a_buf,
        C ? C->buffer().impl() : a_buf,
        output.buffer().impl(),
    };
    auto ds_r = alloc_desc_set_for(pipeline.descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    // 3. 获取 command buffer
    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    // 4. 录制
    record_input_barriers(cmd, {a_buf});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    // Push constants: count, mode, op, cmp_op, flags (5×uint32) + scalar_b, scalar_then, scalar_else (3×float)
    struct PushData
    {
        uint32_t count, mode, op, cmp_op, flags;
        float scalar_b, scalar_then, scalar_else;
    } push{count, mode, op, cmp_op, flags, scalar_b, scalar_then, scalar_else};
    vkCmdPushConstants(cmd, pipeline.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push), &push);

    // vec4 kernel（与 DSL count/(256*vec_width) 同口径）：每线程 4 元素
    const uint32_t wg_count = (count + 256u * 4u - 1u) / (256u * 4u);
    ts_dispatch(cmd, "elementwise_v2", wg_count, 1, 1);
    record_output_barrier(cmd, output.buffer().impl());

    // 6. 独立模式提交
    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    return output;
}

// ══════════════════════════════════════════════════════════════════
// run_fused_gpu — AOT 融合 shader 通用执行入口
//
// 与 glsl_gen.hpp 生成的 shader 布局严格对应：
//   Bindings: 输入 0..N-1，输出 N（全部 storage buffer）
//   Push Constants（逐元素）: uint count, uint cols, float c0..（常量池）
//   Push Constants（归约）:   uint count, uint cols, uint rows, uint vector_out, ...
//   Push Constants（matmul 融合）: uint count, uint cols, uint rows, uint mm_k, ...
//   local_size_x = 256
//
// shader_name 必须是已注册的融合 shader 的 key（= expr_spec_key，见
// 构建期 fused_registry.hpp）。
// 所有输入同形状 (rows, cols)（matmul 段除外：A/B 按 matmul 形状解释，
// 但 Vulkan buffer 无形状概念，绑定即用）；同一 buffer 可绑定到多个输入。
// matmul_k：前置 matmul 段的求和维度（形状参数，运行时填充 mm_k 槽；
// 非 matmul shader 传 nullopt）。
// matmul_batch：matmul 段批量数（形状参数，填充 mm_batch 槽 + dispatch z）。
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<GpuTensor>
run_fused_gpu(const std::string& shader_name,
              // 输入按**底层 buffer** 绑定（元素类型由 shader 声明决定）→ 同一结构可在
              // f32 / f16 两种存储上运行（Phase 2 in-kernel f16 的带类型变体）。
              std::span<const GpuBuffer* const> inputs, std::span<const Scalar> consts,
              std::size_t rows, std::size_t cols, bool vector_out = false,
              std::span<const std::uint32_t> view_params = {}, std::span<const Scalar> rparams = {},
              GpuTensor* output_override = nullptr,
              std::optional<std::uint32_t> matmul_k = std::nullopt, std::uint32_t matmul_batch = 1,
              // matmul 段转置布局（bit0=transA，bit1=transB）：**运行期 operand layout，
              // 不进 expr_spec_key** —— 同一结构覆盖 4 种转置组合。仅 matmul 形态
              // （matmul / matmul+归约）使用；fold 形态的 mm 段转置仍在 key 里
              // （生成期定死），此处忽略。
              std::uint32_t matmul_trans = 0u,
              // fold 收缩轴长度（fold shader 传入填 fold_k 槽；非 fold 传 nullopt）
              std::optional<std::uint32_t> fold_k = std::nullopt,
              // 输出存储精度（Phase 2 in-kernel f16）：true → 分配 f16 字节布局的
              // 输出缓冲（返回的 GpuTensor 仅是**占位标签**，调用方按此标志重贴
              // GpuTensorF16 —— 见 GpuEngine::eval_expr）。
              bool out_f16 = false,
              // 运行期精度分派（键 "#x"）：真实 (输入精度位图, 输出精度)。仅在
              // 分派变体上写入 PC `prec` 槽；其余变体忽略（其精度已是身份）。
              nn::ExprPrecSig prec_sig = 0,
              // P2 多输出（extras，IR-C 写穿物化）：主输出之外的**额外输出存储**
              // （槽 1..k，与生成器的 `bex<k>` 绑定同序）。仅 V0（全 f32）逐元素
              // shader 支持（`#x`/typed/native16 变体不生成 extras，见 expr_glsl_gen）。
              std::span<GpuTensor* const> extra_outs = {})
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    const auto prof_t0 = std::chrono::steady_clock::now();
    const auto it = fused_kernels_.find(shader_name);
    if (it == fused_kernels_.end()) NN_FAIL("fused shader not registered: " + shader_name);
    const FusedKernel& kernel = it->second;
    // （pipeline 绑定推迟到细 tile 选择之后——基础键必须存在，细变体可选）
    // 运行期精度分派变体（键后缀 "#x"）：描述符 = 2N 输入视图 + 2 输出视图，
    // 且 PC 固定头末尾多一个 `prec` 槽。
    const bool is_disp = std::string_view(shader_name).ends_with(nn::EXPR_PREC_DISPATCH_SUFFIX);
    // PC 固定头槽数：**注册时算定，此处只读**（唯一来源 =
    // FusedKernel::pc_base_fixed；本地重推会让两侧漂移 →
    // push constant 超 range 被驱动丢弃 → 静默读未定义残留）。
    const std::uint32_t pc_base_fixed = kernel.pc_base_fixed;
    // 归约轴：-1=逐元素, 0=行归约, 1=列归约
    const int raxis = kernel.reduce_axis;
    // 是否含前置 matmul 段（push constants 多 rows + mm_k）
    const bool has_mm = kernel.has_matmul;
    // fold 形态（注册时从 FusedShader.spec.fold 取，与生成器同源）
    const FoldMeta fmeta = kernel.fold;
    const bool is_fold = kernel.is_fold();
    // 运行时视图参数个数（RowMod/RotateHalf 的 vp 槽）
    const std::uint32_t n_vp = kernel.view_param_count;
    // 运行时标量参数个数（优化器 lr/eps/β 的 rp 槽）
    const std::uint32_t n_rp = kernel.rparam_count;

    // ── P0 细 tile 选择（wgrad 网格饥饿）────────────────────────────────
    // 纯 matmul 形态（与下方 dispatch 分支同一判据）且 64-tile 网格的
    // 工作组数（含 batch 维）低于阈值 → 派发同结构的 16×16 输出块变体
    // （键 = shader_name + "#f"，注册表元数据/PC 布局/绑定数与 V0 完全
    // 同源，只是网格粒度不同）。`#f` 未登记（f16 精度变体键、设备未注册）
    // 或 NN_MM_NO_FINE=1 → 留在 64-tile 路径（降级，不是错误）。
    // 两变体逐输出元素的 K 累加序一致（BK=16 逐块升序 + 同一 vec4 点积
    // 序）→ 结果逐位一致，字节锚不移动。
    bool use_fine = false;
    // ── K 切分选择（wgrad 二次归约，键 #ks+#kr）────────────────────────
    // 细 tile 网格（m_per/16 × cols/16 × batch 个 WG）仍低于
    // EXPR_MATMUL_KSPLIT_MIN_WGS → 把 K 切成 S 段：#ks 按列块序号取段
    // [k_lo, k_lo+span)（dispatch x × S）写 (S·rows, cols) 部分和，
    // #kr 跨段升序求和 + 跑尾链落最终输出。S = clamp(ceil(阈值/网格),
    // 2, MAX_S)（NN_MM_KSPLIT_S 可强制，A/B 调参）；span 上取 16 对齐
    // （生成器 vec4 快路径的段界 k 连续判据依赖它）。
    // 门槛：V0（#ks/#kr 只注册基础键）、纯 matmul、无 extras、f32、
    // #ks+#kr 同时注册（配对）、NN_MM_NO_FINE 与 NN_MM_NO_KSPLIT 均未关。
    // ⚠ 跨段 K 累加序 ≠ 逐 k 串联 → **移动 GPU 字节锚**（已裁定放行，
    // 与 P2-2 同处置：重测 + 更新 §12 + history 记新旧值）。
    bool use_ks = false;
    std::uint32_t ks_s = 0; // 段数 S
    std::uint32_t ks_span = 0; // 段宽（16 对齐）
    if (has_mm && raxis < 0 && !is_fold && !mm_fine_disabled())
    {
        const std::uint32_t rows_u = static_cast<std::uint32_t>(rows);
        const std::uint32_t m_per = rows_u / matmul_batch; // rows 按 batch 整除（契约）
        constexpr std::uint32_t B64 = nn::EXPR_MATMUL_BLOCK;
        const std::uint64_t wg64 =
            static_cast<std::uint64_t>((static_cast<std::uint32_t>(cols) + B64 - 1u) / B64) *
            ((m_per + B64 - 1u) / B64) * matmul_batch;
        if (wg64 < nn::EXPR_MATMUL_FINE_MIN_WGS &&
            fused_kernels_.count(shader_name + nn::EXPR_MM_FINE_SUFFIX) != 0)
            use_fine = true;
        if (wg64 < nn::EXPR_MATMUL_FINE_MIN_WGS && !is_disp && !mm_ksplit_disabled() &&
            extra_outs.empty() && matmul_k && *matmul_k > 0 &&
            fused_kernels_.count(shader_name + nn::EXPR_MM_KSPLIT_SUFFIX) != 0 &&
            fused_kernels_.count(shader_name + nn::EXPR_MM_KSPLIT_REDUCE_SUFFIX) != 0)
        {
            constexpr std::uint32_t Bf = nn::EXPR_MATMUL_BLOCK_FINE;
            const std::uint64_t wg_fine =
                static_cast<std::uint64_t>((static_cast<std::uint32_t>(cols) + Bf - 1u) / Bf) *
                ((m_per + Bf - 1u) / Bf) * matmul_batch;
            std::uint32_t s = mm_ksplit_forced_s();
            if (s == 0u)
            {
                if (wg_fine == 0 || wg_fine >= nn::EXPR_MATMUL_KSPLIT_MIN_WGS)
                    s = 0u; // 网格够密 → 不切
                else
                    s = static_cast<std::uint32_t>((nn::EXPR_MATMUL_KSPLIT_MIN_WGS + wg_fine - 1u) /
                                                   wg_fine);
            }
            if (s >= 2u)
            {
                if (s > nn::EXPR_MATMUL_KSPLIT_MAX_S) s = nn::EXPR_MATMUL_KSPLIT_MAX_S;
                const std::uint32_t k = *matmul_k;
                const std::uint32_t span = ((k + s - 1u) / s + 15u) & ~15u; // 上取 16 对齐
                // 部分和缓冲 = S·rows·cols 个 float，shader 线性索引是 u32
                const std::uint64_t part =
                    static_cast<std::uint64_t>(s) * rows * static_cast<std::uint64_t>(cols);
                if (span > 0u && part <= 0xFFFFFFFFull)
                {
                    use_ks = true;
                    use_fine = false; // K 切分独占派发（不走基础/细 tile）
                    ks_s = s;
                    ks_span = span;
                }
            }
        }
    }
    const VulkanPipeline& pipeline =
        use_fine ? fused_kernels_.at(shader_name + nn::EXPR_MM_FINE_SUFFIX).pipeline
                 : kernel.pipeline;

    if (inputs.size() + 1 > EXPR_MAX_INPUTS + 1) NN_FAIL("run_fused_gpu: too many inputs");
    if (consts.size() > EXPR_MAX_CONSTS) NN_FAIL("run_fused_gpu: too many constants");
    if (view_params.size() != n_vp)
        NN_FAIL("run_fused_gpu: view_params count mismatch for " + shader_name);
    if (rparams.size() != n_rp) NN_FAIL("run_fused_gpu: rparams count mismatch for " + shader_name);
    if (has_mm && !matmul_k) NN_FAIL("run_fused_gpu: matmul shader 缺少 matmul_k（求和维度）");
    if (!has_mm && matmul_k && !is_fold) NN_FAIL("run_fused_gpu: 非 matmul shader 收到了 matmul_k");
    // fold 形态校验（与 matmul_k 同款对称防呆 + 调用约定）：
    //   fold 输出网格 = (rows, out_cols)，out_cols = vec_state_len（或 1），
    //   cols 参数必须等于它（GpuEngine/层调用约定）；PC cols 槽填 fold_k
    //   （视图/块轴列数=键长，见下方 cols32），不填输出列。
    //   fold+matmul 时 matmul_k/matmul_batch 同传（PC slot5/6）。
    if (is_fold && !fold_k) NN_FAIL("run_fused_gpu: fold shader 缺少 fold_k（收缩轴长度）");
    if (!is_fold && fold_k) NN_FAIL("run_fused_gpu: 非 fold shader 收到了 fold_k");
    if (is_fold && vector_out)
        NN_FAIL("run_fused_gpu: fold shader 以 vector_out 调度（应走 eval_expr）");
    if (is_fold && cols == 0)
        NN_FAIL("run_fused_gpu: fold 调用约定 cols（=输出列数 out_cols）必须 > 0");
    // （形状无关：out_cols 不做 shader 侧比对——veclen 不进 key，同一
    //   shader 服务任意 d_k，PC vector_out 槽按调用方 cols 运行时填充）
    // matmul+归约：raxis >= 0 时 mm_k 填入 PC 第 5 槽（见下方填充）

    // count 以 uint32 传入 shader（gl_GlobalInvocationID / push constant），
    // 必须保证 rows*cols 不溢出 uint32，否则分派与索引会静默截断。
    if (rows > 0 && cols > UINT32_MAX / rows)
        NN_FAIL("run_fused_gpu: rows*cols exceeds uint32 range");
    const std::uint32_t count = static_cast<std::uint32_t>(rows * cols);

    // 1. 分配输出 Tensor（vector_out：归约向量原生形状 (rows,1)/(1,cols)）
    //    若调用方指定 output_override（dsl::compute_into 的原地目标），
    //    则复用其 buffer（形状必须匹配），结果直接写入目标 → 不额外分配、
    //    不额外拷贝。
    const std::size_t out_rows = (vector_out && raxis == 0)   ? rows
                                 : (vector_out && raxis == 1) ? 1
                                                              : rows;
    const std::size_t out_cols = (vector_out && raxis == 1)   ? cols
                                 : (vector_out && raxis == 0) ? 1
                                                              : cols;
    std::optional<GpuTensor> owned_output;
    GpuTensor* output_ptr = output_override;
    if (output_override)
    {
        if (output_override->rows() != out_rows || output_override->cols() != out_cols)
            NN_FAIL("run_fused_gpu: output_override shape mismatch (" + std::to_string(out_rows) +
                    "x" + std::to_string(out_cols) + ")");
    }
    else if (out_f16)
    {
        // f16 输出：按 f16 字节布局分配（2B/元素，偶数槽位对齐见 create_f16_tensor）
        auto f16r = GpuTensorF16::create_empty(out_rows, out_cols, *this);
        NN_TRY_CHECK(f16r);
        owned_output.emplace(f16r->shared_buffer(), out_rows, out_cols);
        output_ptr = &*owned_output;
    }
    else
    {
        auto output_res = GpuTensor::create_empty(out_rows, out_cols, *this);
        NN_TRY_CHECK(output_res);
        owned_output.emplace(std::move(*output_res));
        output_ptr = &*owned_output;
    }
    GpuTensor output = *output_ptr;

    // P2 多输出（extras）：额外输出全部由调用方提供既有存储（写穿物化），
    // 主输出亦须是 output_override（全输出目标显式，无分配路径）。
    if (!extra_outs.empty())
    {
        if (is_disp) NN_FAIL("run_fused_gpu: 多输出（extras）仅支持 V0 变体");
        if (!output_override)
            NN_FAIL("run_fused_gpu: 多输出（extras）要求主输出为 output_override");
        for (const GpuTensor* eo : extra_outs)
            if (eo == nullptr) NN_FAIL("run_fused_gpu: 多输出（extras）含空输出指针");
    }

    // ── K 切分两段派发（use_ks，选择判据见上方）────────────────────────
    // #ks：细块网格 × S 段（x = 列块序号决定 K 段）→ (S·rows, cols)
    //      部分和；PC 7 槽 = matmul 头 + ks_span（pc_base_fixed=7）。
    // #kr：ceil(count/256) 每线程一输出，段升序求和 + 尾链 → 最终输出；
    //      PC 6 槽 = matmul 同布局（第 4 槽填段数 S，#kr 不读 mm_trans）。
    // 屏障：inputs 统一就绪 → #ks → partial 写转读（record_output_barrier
    // 的 src SHADER_WRITE→dst MEMORY_READ 覆盖 compute→compute）→ #kr →
    // 输出写转读。部分和缓冲局部析构 = 归还池；批量录制模式下块可能被
    // **录制序更晚**的算子复用——执行序 = 录制序，#kr 的读排在复用写
    // 之前（同一命令缓冲），故安全（与 owned_output 同一账目）。
    if (use_ks)
    {
        const VulkanPipeline& ks_pipe =
            fused_kernels_.at(shader_name + nn::EXPR_MM_KSPLIT_SUFFIX).pipeline;
        const VulkanPipeline& kr_pipe =
            fused_kernels_.at(shader_name + nn::EXPR_MM_KSPLIT_REDUCE_SUFFIX).pipeline;
        const std::uint32_t rows32 = static_cast<std::uint32_t>(rows);
        const std::uint32_t cols32 = static_cast<std::uint32_t>(cols);
        const std::uint32_t k32 = *matmul_k;

        // 1. 部分和缓冲 (S·rows, cols) f32（选择处已验证 u32 索引上界）
        auto part_r = GpuTensor::create_empty(static_cast<std::size_t>(ks_s) * rows, cols, *this);
        NN_TRY_CHECK(part_r);
        GpuTensor partial = std::move(*part_r);

        // 2. 两份 PC（固定头 + vp + consts + rparams，槽位与生成器逐字段同源）
        const std::size_t tail_bytes =
            sizeof(Scalar) * consts.size() + sizeof(Scalar) * rparams.size();
        std::vector<std::uint8_t> pc_ks((7u + n_vp) * sizeof(std::uint32_t) + tail_bytes);
        std::vector<std::uint8_t> pc_kr((6u + n_vp) * sizeof(std::uint32_t) + tail_bytes);
        const auto fill_head = [&](std::vector<std::uint8_t>& pc, int slots, std::uint32_t slot3,
                                   std::uint32_t slot4, std::uint32_t slot5)
        {
            std::memcpy(pc.data(), &count, sizeof(std::uint32_t));
            std::memcpy(pc.data() + 4, &cols32, sizeof(std::uint32_t));
            std::memcpy(pc.data() + 8, &rows32, sizeof(std::uint32_t));
            std::memcpy(pc.data() + 12, &slot3, sizeof(std::uint32_t));
            std::memcpy(pc.data() + 16, &slot4, sizeof(std::uint32_t));
            std::memcpy(pc.data() + 20, &slot5, sizeof(std::uint32_t));
            const std::size_t vp_off = static_cast<std::size_t>(slots) * sizeof(std::uint32_t);
            for (std::uint32_t i = 0; i < n_vp; ++i)
                std::memcpy(pc.data() + vp_off + i * sizeof(std::uint32_t), &view_params[i],
                            sizeof(std::uint32_t));
            if (!consts.empty())
                std::memcpy(pc.data() + vp_off + n_vp * sizeof(std::uint32_t), consts.data(),
                            sizeof(Scalar) * consts.size());
            if (!rparams.empty())
                std::memcpy(pc.data() + vp_off + n_vp * sizeof(std::uint32_t) +
                                sizeof(Scalar) * consts.size(),
                            rparams.data(), sizeof(Scalar) * rparams.size());
        };
        fill_head(pc_ks, 7, k32, matmul_batch, matmul_trans);
        // ks_span 在固定头第 7 槽（offset 24）
        std::memcpy(pc_ks.data() + 24, &ks_span, sizeof(std::uint32_t));
        // **有效段数** V = ceil(mm_k / span)（span 16 对齐上取 → 尾部可能
        // 存在 k_lo ≥ mm_k 的空段：#ks 早退不写部分和，#kr 必须只加这 V
        // 段，否则读到未初始化缓冲的垃圾——ctest 小形状全挂的教训）。
        const std::uint32_t ks_valid = (k32 + ks_span - 1u) / ks_span;
        fill_head(pc_kr, 6, ks_valid, matmul_batch, 0u);

        // 3. 描述符集：#ks = N 输入 + 1 部分和；#kr = N 输入 + 部分和 + 输出
        const std::size_t n_in = inputs.size();
        std::vector<VkBuffer> ks_bindings(n_in + 1);
        std::vector<VkBuffer> kr_bindings(n_in + 2);
        for (std::size_t i = 0; i < n_in; ++i)
        {
            ks_bindings[i] = inputs[i]->impl();
            kr_bindings[i] = inputs[i]->impl();
        }
        ks_bindings[n_in] = partial.buffer().impl();
        kr_bindings[n_in] = partial.buffer().impl();
        kr_bindings[n_in + 1] = output.buffer().impl();

        auto ds1_r = alloc_desc_set_for(ks_pipe.descriptor_layout(), ks_bindings);
        NN_TRY_CHECK(ds1_r);
        VkDescriptorSet ds_ks = *ds1_r;
        auto ds2_r = alloc_desc_set_for(kr_pipe.descriptor_layout(), kr_bindings);
        if (!ds2_r)
        {
            release_desc_set_now(ds_ks);
            NN_TRY_CHECK(ds2_r);
        }
        VkDescriptorSet ds_kr = *ds2_r;

        // 4. 录制：inputs 屏障 → #ks → partial 屏障 → #kr → 输出屏障
        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            release_desc_set_now(ds_kr);
            release_desc_set_now(ds_ks);
            NN_TRY_CHECK(cmd_r);
        }
        auto [cmd, owns_cmd] = *cmd_r;
        std::vector<VkBuffer> in_bufs;
        in_bufs.reserve(inputs.size());
        for (const auto* b : inputs)
            in_bufs.push_back(b->impl());
        record_input_barriers(cmd, in_bufs);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ks_pipe.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ks_pipe.pipeline_layout(), 0,
                                1, &ds_ks, 0, nullptr);
        vkCmdPushConstants(cmd, ks_pipe.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<std::uint32_t>(pc_ks.size()), pc_ks.data());
        {
            constexpr std::uint32_t Bf = nn::EXPR_MATMUL_BLOCK_FINE;
            const std::uint32_t m_per = rows32 / matmul_batch;
            const std::uint32_t ks_valid = (k32 + ks_span - 1u) / ks_span;
            // 只派发**有效段**（空段的 WG 会早退且不写部分和 → 派了也是
            // 白跑）；x = 列块序号决定段号，与生成器 slice = wg.x / nx 同源
            const std::uint32_t gx = ((cols32 + Bf - 1u) / Bf) * ks_valid;
            const std::uint32_t gy = (m_per + Bf - 1u) / Bf;
            // 段变体计时进同一 "fused_mm" 桶（#ks/#kr 两行与原单发同维对账）
            ts_dispatch(cmd, "fused_mm", gx, gy, matmul_batch, shader_name,
                        nn::EXPR_MM_KSPLIT_SUFFIX);
        }
        record_output_barrier(cmd, partial.buffer().impl());

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kr_pipe.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kr_pipe.pipeline_layout(), 0,
                                1, &ds_kr, 0, nullptr);
        vkCmdPushConstants(cmd, kr_pipe.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           static_cast<std::uint32_t>(pc_kr.size()), pc_kr.data());
        ts_dispatch(cmd, "fused_mm", (count + 256u - 1u) / 256u, 1u, 1u, shader_name,
                    nn::EXPR_MM_KSPLIT_REDUCE_SUFFIX);
        record_output_barrier(cmd, output.buffer().impl());

        // 5. 提交（独立模式等待后两个描述符集均可回收；批量模式由帧池
        //    reset 统一回收，与既有路径一致）——本块只在 `owns_cmd`
        //    （⇒ solo）下可达，故这里的直接释放不违反 RV-15 的登记规则
        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, ds_ks);
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &ds_kr);
            NN_TRY_CHECK(r);
        }
        if (profile_ops_enabled())
        {
            const long long dt = std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - prof_t0)
                                     .count();
            std::fprintf(stderr, "[gpu-profile] fused fold=0 out=%u total=%lldus key=%.16s ks\n",
                         static_cast<std::uint32_t>(cols), dt, shader_name.c_str());
        }
        return output;
    }

    // 2. 分配描述符集（N 输入 + 1 输出；分派变体 = 2N + 2；extras 追加 k 个输出）
    const std::size_t n_in = inputs.size();
    const std::size_t n_bindings = is_disp ? (2 * n_in + 2) : (n_in + 1 + extra_outs.size());
    std::vector<VkBuffer> bindings(n_bindings);
    if (is_disp)
    {
        // 同一 VkBuffer 绑两次（f32 视图 / f16 视图）；输入对是 readonly、
        // 输出对是 writeonly，shader 每个元素只解引用 PC `prec` 选中的那一个
        // → 另一个从不访问（无别名风险）。
        for (std::size_t i = 0; i < n_in; ++i)
        {
            bindings[i] = inputs[i]->impl();
            bindings[n_in + i] = inputs[i]->impl();
        }
        bindings[2 * n_in] = output.buffer().impl();
        bindings[2 * n_in + 1] = output.buffer().impl();
    }
    else
    {
        for (std::size_t i = 0; i < n_in; ++i)
            bindings[i] = inputs[i]->impl();
        bindings[n_in] = output.buffer().impl();
        // P2 多输出：bex<k> 绑定在主输出之后（与生成器 binding 同序）
        for (std::size_t k = 0; k < extra_outs.size(); ++k)
            bindings[n_in + 1 + k] = extra_outs[k]->buffer().impl();
    }
    auto ds_r = alloc_desc_set_for(pipeline.descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    // 3. 获取 command buffer
    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    // 4. 录制：输入屏障 → bind → push constants → dispatch → 输出屏障
    std::vector<VkBuffer> in_bufs;
    in_bufs.reserve(inputs.size());
    for (const auto* b : inputs)
        in_bufs.push_back(b->impl());
    record_input_barriers(cmd, in_bufs);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    // Push constants 布局与 glsl_gen.hpp 一致：
    //   逐元素: count, cols, [vp0..], c0..
    //   归约:   count, cols, rows, vector_out, [vp0..], c0..
    //   matmul: count, cols, rows, mm_k, mm_batch, mm_trans, [vp0..], c0..
    //   matmul+归约: count, cols, rows, vector_out, mm_k, mm_batch, mm_trans,
    //                [vp0..], c0..
    //
    // ⚠ 固定头长度**唯一来源 = 注册时的 pc_base_fixed**（本处只读，见上）。
    //   逐形态：逐元素 2 / 逐元素+matmul 6 / 归约 4 / 归约+matmul 7 /
    //   fold 5 / fold+matmul 7（+ 分派变体 1 个 prec）。
    //   错一个 uint 即常量池整体后移 → shader 从错位处读常量，GPU 上带
    //   常量的归约静默错值而 CPU 正常（实测把 select(cond,1,0) 的常量
    //   读成垃圾）。
    // 分派变体（键 "#x"）在固定头**末尾**多一个 uint prec：生成器同位置
    // 声明（各生成器的 PC 块都在固定槽之后、vp 之前发 `uint prec;`）。
    const std::uint32_t pc_uints = pc_base_fixed + (is_disp ? 1u : 0u);
    std::vector<std::uint8_t> pc((pc_uints + n_vp) * sizeof(std::uint32_t) +
                                 sizeof(Scalar) * consts.size() + sizeof(Scalar) * rparams.size());
    std::memcpy(pc.data(), &count, sizeof(std::uint32_t));
    // fold：PC cols 槽 = fold_k（生成器视图/块轴用输入列数 K 索引；
    //   输出网格列恒 1 不进 PC）。其余形态 = 调用方 cols。
    const std::uint32_t cols32 = is_fold ? *fold_k : static_cast<std::uint32_t>(cols);
    std::memcpy(pc.data() + sizeof(std::uint32_t), &cols32, sizeof(std::uint32_t));
    if (raxis >= 0)
    {
        const std::uint32_t rows32 = static_cast<std::uint32_t>(rows);
        std::memcpy(pc.data() + 2 * sizeof(std::uint32_t), &rows32, sizeof(std::uint32_t));
        const std::uint32_t vo = vector_out ? 1u : 0u;
        std::memcpy(pc.data() + 3 * sizeof(std::uint32_t), &vo, sizeof(std::uint32_t));
        if (has_mm)
        {
            // matmul+归约：mm_k + mm_batch + mm_trans（形状参数 + 运行期
            // operand layout，运行时填充）
            std::memcpy(pc.data() + 4 * sizeof(std::uint32_t), &*matmul_k, sizeof(std::uint32_t));
            std::memcpy(pc.data() + 5 * sizeof(std::uint32_t), &matmul_batch,
                        sizeof(std::uint32_t));
            std::memcpy(pc.data() + 6 * sizeof(std::uint32_t), &matmul_trans,
                        sizeof(std::uint32_t));
        }
    }
    else if (has_mm)
    {
        // matmul 融合：rows + mm_k + mm_batch + mm_trans（形状参数 + 运行期
        // operand layout，运行时按实际 spec 填充 → 同结构不同 K/batch/转置
        // 共享一个融合 shader）
        const std::uint32_t rows32 = static_cast<std::uint32_t>(rows);
        std::memcpy(pc.data() + 2 * sizeof(std::uint32_t), &rows32, sizeof(std::uint32_t));
        std::memcpy(pc.data() + 3 * sizeof(std::uint32_t), &*matmul_k, sizeof(std::uint32_t));
        std::memcpy(pc.data() + 4 * sizeof(std::uint32_t), &matmul_batch, sizeof(std::uint32_t));
        std::memcpy(pc.data() + 5 * sizeof(std::uint32_t), &matmul_trans, sizeof(std::uint32_t));
    }
    else if (is_fold)
    {
        // fold：rows + **vector_out = 调用方输出列数**
        //   （形状无关——veclen 不进 key，PC 运行时填充；v1 生成器
        //   不读此槽、填 1 无害）+ fold_k + [mm_k, mm_batch]（7 槽形态）。
        //   与生成器 PC 声明逐字段一致（4.10：创建侧 range 必须同改）
        const std::uint32_t rows32 = static_cast<std::uint32_t>(rows);
        const std::uint32_t vo = static_cast<std::uint32_t>(cols);
        std::memcpy(pc.data() + 2 * sizeof(std::uint32_t), &rows32, sizeof(std::uint32_t));
        std::memcpy(pc.data() + 3 * sizeof(std::uint32_t), &vo, sizeof(std::uint32_t));
        std::memcpy(pc.data() + 4 * sizeof(std::uint32_t), &*fold_k, sizeof(std::uint32_t));
        if (matmul_k)
        {
            std::memcpy(pc.data() + 5 * sizeof(std::uint32_t), &*matmul_k, sizeof(std::uint32_t));
            std::memcpy(pc.data() + 6 * sizeof(std::uint32_t), &matmul_batch,
                        sizeof(std::uint32_t));
        }
    }
    // 运行期精度分派：把真实 (输入精度位图, 输出精度) 写进固定头末尾的
    // prec 槽——shader 据此在每个加载/存储点走 uniform 分支。
    if (is_disp)
        std::memcpy(pc.data() + pc_base_fixed * sizeof(std::uint32_t), &prec_sig, sizeof(prec_sig));
    // 运行时视图参数（RowMod 周期 / RotateHalf 块大小），置于固定头之后、常量池之前
    for (std::uint32_t i = 0; i < n_vp; ++i)
        std::memcpy(pc.data() + (pc_uints + i) * sizeof(std::uint32_t), &view_params[i],
                    sizeof(std::uint32_t));
    if (!consts.empty())
        std::memcpy(pc.data() + (pc_uints + n_vp) * sizeof(std::uint32_t), consts.data(),
                    sizeof(Scalar) * consts.size());
    // 运行时标量参数（优化器 lr/eps/β 等 float rp 槽），置于常量池之后
    if (!rparams.empty())
        std::memcpy(pc.data() + (pc_uints + n_vp) * sizeof(std::uint32_t) +
                        sizeof(Scalar) * consts.size(),
                    rparams.data(), sizeof(Scalar) * rparams.size());
    vkCmdPushConstants(cmd, pipeline.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<std::uint32_t>(pc.size()), pc.data());

    // dispatch：逐元素 = ceil(count/(256*vec_width))；行归约 = rows 个工作组；
    // 列归约 = ceil(cols/32) 个工作组（每工作组 32 列 tile、warp=行块，
    // 与生成器 l/wb/col 结构及 OP 级 reduce.comp 契约同源）；
    // matmul 分块 = (ceil(cols/BLOCK), ceil(m_per/BLOCK), matmul_batch)，
    // BLOCK 与 glsl_gen 生成的输出块一致（EXPR_MATMUL_BLOCK=64：每工作组
    // 64×64 输出块、16×16 线程、每线程 4×4 寄存器分块；网格饥饿时的
    // 细 tile 变体 BLOCK=EXPR_MATMUL_BLOCK_FINE=16，见上方 use_fine）
    const std::uint32_t vec_width = kernel.vec_width;
    if (has_mm && raxis < 0 && !is_fold)
    {
        // P0 细 tile：use_fine 时输出块 64 → 16（同一 struct 的两个变体，
        // 只是网格粒度；生成器同源，见 EXPR_MATMUL_BLOCK_FINE）
        const std::uint32_t mm_block =
            use_fine ? nn::EXPR_MATMUL_BLOCK_FINE : nn::EXPR_MATMUL_BLOCK;
        const std::uint32_t wg_x = (static_cast<std::uint32_t>(cols) + mm_block - 1u) / mm_block;
        // batch：dispatch z = 批次，A/B 按 batch 垂直切分 → y 只覆盖
        // **批内**行 m_per = rows/mm_batch（生成器 main/load_tiles 均以 m_per
        // 为界、写回守卫 rr < m_per）。按总 rows 派 y 会把 batch 在 y/z 数
        // 两遍 → 工作量 ∝ batch²：多余 (BH−1)/BH 的 WG 跑完整条 mm_k 流水
        // 后整块丢弃——结果仍正确、对拍永远测不出，只有计时能发现。rows =
        // batch*M 按契约整除。
        const std::uint32_t rows_u = static_cast<std::uint32_t>(rows);
        const std::uint32_t m_per = rows_u / matmul_batch;
        const std::uint32_t wg_y = (m_per + mm_block - 1u) / mm_block;
        // batch：dispatch z = 批次，A/B 按 batch 垂直切分
        ts_dispatch(cmd, "fused_mm", wg_x, wg_y, matmul_batch, shader_name,
                    use_fine ? nn::EXPR_MM_FINE_SUFFIX : nullptr);
    }
    else
    {
        // fold 的 dispatch 分两种（FoldMeta，与生成器分派同源）：
        //   v1 标量 fold（每线程一行，row=gl_GlobalInvocationID.x）：
        //     count=rows×1 → ceil(count/256) 恰 = ceil(rows/256) ✓
        //   v2 双域 fold（每 WG NR 行，row=wg*NR+ri，EXPR_FOLD_ROWS_PER_WG
        //     与生成器行循环同源）：**wg = ceil(rows/NR)**
        //     （按 ceil(count/256)=ceil(rows×out/256) 派会丢行！）
        const std::uint32_t wg_count =
            (is_fold && !fmeta.per_thread_row)
                ? (static_cast<std::uint32_t>(rows) + nn::EXPR_FOLD_ROWS_PER_WG - 1u) /
                      nn::EXPR_FOLD_ROWS_PER_WG
            : (raxis == 0) ? static_cast<std::uint32_t>(rows)
            : (raxis == 1) ? (static_cast<std::uint32_t>(cols) + 31u) / 32u
                           : (count + 256u * vec_width - 1u) / (256u * vec_width);
        ts_dispatch(cmd, "fused", wg_count, 1, 1, shader_name);
    }
    record_output_barrier(cmd, output.buffer().impl());
    // P2 多输出：额外输出同样过写后屏障
    for (const GpuTensor* eo : extra_outs)
        record_output_barrier(cmd, eo->buffer().impl());

    // 5. 独立模式提交
    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    if (profile_ops_enabled())
    {
        const long long dt = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - prof_t0)
                                 .count();
        // fold 按形态如实打印（is_fold = 本形态是否 fold）；out= 运行时
        // 列数（veclen 不进 key，注册侧值可能与调用方实际 cols 不同——
        // 打印注册侧列数会误导归因）
        std::fprintf(stderr, "[gpu-profile] fused fold=%d out=%u total=%lldus key=%.16s\n",
                     static_cast<int>(is_fold), static_cast<std::uint32_t>(cols), dt,
                     shader_name.c_str());
    }
    return output;
}

// ══════════════════════════════════════════════════════════════════
// run_fused_gpu_batch — P3 跨链批量派发（键 = key#b）
//
// 同结构（同 key）、同参数值（consts/rparams/vp）、互不依赖的 N 个实例
// 合成**一次 dispatch**：实例表（uvec2 数组，见 FusedBatchInst 注释）经
// vkCmdUpdateBuffer 在录制期内联写入一个小 device-local 缓冲，shader 用
// buffer device address 按槽位解引用各实例的输入/输出缓冲。
//
// 与生成器（expr_glsl_gen 的 batched 分支）的契约：
//   Bindings: 仅 binding 0（实例表，storage buffer）
//   Push Constants = V0 逐元素同布局：count(=max_count), cols, [vp..],
//                    [c..], [rp..]（批量路径不读 count/cols 两槽——元素数
//                    与列数取每实例尾槽，见生成器 `#define cols`）
//   dispatch = (ceil(max_count/256), N, 1)，实例下标 = gl_WorkGroupID.y
//
// 前置（调用方 = execute_fused_graph 保证，本函数只防御性校验）：
// `key#b` 已注册、设备 bda_available、实例全 f32、组内互不依赖。
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<void> run_fused_gpu_batch(const std::string& shader_name,
                                               std::span<const FusedBatchInst> instances,
                                               std::span<const Scalar> consts,
                                               std::span<const std::uint32_t> view_params = {},
                                               std::span<const Scalar> rparams = {})
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    const auto prof_t0 = std::chrono::steady_clock::now();
    if (!device_.bda_available())
        NN_FAIL("run_fused_gpu_batch: 设备未启用 buffer device address"
                "（应先逐 kernel 降级，不该到此）");
    if (!std::string_view(shader_name).ends_with(nn::EXPR_BATCH_SUFFIX))
        NN_FAIL("run_fused_gpu_batch: 非批量变体键（应以 #b 结尾）：" + shader_name);
    const auto it = fused_kernels_.find(shader_name);
    if (it == fused_kernels_.end()) NN_FAIL("fused shader not registered: " + shader_name);
    const FusedKernel& kernel = it->second;
    const VulkanPipeline& pipeline = kernel.pipeline;
    if (instances.empty()) return {};
    // PC 槽位校验（与 run_fused_gpu 同源：vp/rp 计数来自注册表元数据）
    const std::uint32_t n_vp = kernel.view_param_count;
    const std::uint32_t n_rp = kernel.rparam_count;
    if (view_params.size() != n_vp)
        NN_FAIL("run_fused_gpu_batch: view_params count mismatch for " + shader_name);
    if (rparams.size() != n_rp)
        NN_FAIL("run_fused_gpu_batch: rparams count mismatch for " + shader_name);
    const std::uint32_t pc_base_fixed = kernel.pc_base_fixed;

    // 实例表布局：stride S = 输入 + 主输出 + extras + (count,cols) 尾槽
    const std::size_t n_in = instances[0].inputs.size();
    const std::size_t n_ex = instances[0].extra_outs.size();
    const std::size_t stride = n_in + 1 + n_ex + 1;
    if (kernel.batch_stride != 0 && kernel.batch_stride != static_cast<std::uint32_t>(stride))
        NN_FAIL("run_fused_gpu_batch: 实例槽数与注册结构不一致：" + shader_name);
    for (const FusedBatchInst& inst : instances)
    {
        if (inst.inputs.size() != n_in || inst.extra_outs.size() != n_ex)
            NN_FAIL("run_fused_gpu_batch: 组内实例结构不一致");
        if (inst.out == nullptr) NN_FAIL("run_fused_gpu_batch: 实例缺主输出缓冲");
        for (const GpuBuffer* b : inst.inputs)
            if (b == nullptr) NN_FAIL("run_fused_gpu_batch: 实例缺输入缓冲");
        for (const GpuBuffer* b : inst.extra_outs)
            if (b == nullptr) NN_FAIL("run_fused_gpu_batch: 实例缺 extras 输出缓冲");
    }
    // vkCmdUpdateBuffer 的硬上限：单次 ≤ 64KB、4 字节对齐
    if (instances.size() * stride * sizeof(std::uint64_t) > 65536u)
        NN_FAIL("run_fused_gpu_batch: 实例表超过 vkCmdUpdateBuffer 64KB 上限");

    // ── 实例表内容：uvec2 = (地址低 32 位, 高 32 位)；尾槽 = (count, cols) ──
    std::vector<std::uint32_t> table(instances.size() * stride * 2, 0u);
    std::uint32_t max_count = 0;
    std::uint32_t max_cols = 0;
    for (std::size_t i = 0; i < instances.size(); ++i)
    {
        const FusedBatchInst& inst = instances[i];
        const std::size_t base = i * stride * 2;
        bool addr_ok = true;
        const auto put_addr = [&](std::size_t slot, const GpuBuffer* b) -> bool
        {
            VkBufferDeviceAddressInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            ai.buffer = b->impl();
            const VkDeviceAddress addr = vkGetBufferDeviceAddress(device_.device(), &ai);
            table[base + slot * 2 + 0] = static_cast<std::uint32_t>(addr & 0xffffffffull);
            table[base + slot * 2 + 1] = static_cast<std::uint32_t>(addr >> 32);
            return addr != 0;
        };
        for (std::size_t j = 0; j < n_in; ++j)
            addr_ok = put_addr(j, inst.inputs[j]) && addr_ok;
        addr_ok = put_addr(n_in, inst.out) && addr_ok;
        for (std::size_t k = 0; k < n_ex; ++k)
            addr_ok = put_addr(n_in + 1 + k, inst.extra_outs[k]) && addr_ok;
        if (!addr_ok)
            NN_FAIL("run_fused_gpu_batch: vkGetBufferDeviceAddress 返回 0"
                    "（buffer 未带 SHADER_DEVICE_ADDRESS？）");
        table[base + (stride - 1) * 2 + 0] = inst.count;
        table[base + (stride - 1) * 2 + 1] = inst.cols;
        max_count = std::max(max_count, inst.count);
        max_cols = std::max(max_cols, inst.cols);
    }

    // 实例表缓冲：小 device-local（TRANSFER_DST 走 vkCmdUpdateBuffer）。
    // 数据内联进命令缓冲 → 录制生命周期安全；缓冲本体析构走 pending_destroys_
    // 延迟队列（铁律 6，与其余录制期资源同款）。
    auto inst_r = GpuBuffer::create_device_local(
        device_.device(), alloc_pool(), table.size() * sizeof(std::uint32_t),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, keepalive());
    NN_TRY_CHECK(inst_r);
    std::shared_ptr<GpuBuffer> inst_buf = std::make_shared<GpuBuffer>(std::move(*inst_r));

    // 描述符：key#b pipeline 只绑 binding 0（实例表）——复用 per-pipeline
    // 描述符机制
    // 批量变体只有一个 binding：实例表
    const VkBuffer instance_table = inst_buf->impl();
    auto ds_r = alloc_desc_set_for(pipeline.descriptor_layout(), {&instance_table, 1});
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    // command buffer
    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    // 录制：实例表写入 → 输入屏障（含实例表，去重）→ bind → PC → dispatch
    vkCmdUpdateBuffer(cmd, inst_buf->impl(), 0,
                      static_cast<std::uint32_t>(table.size() * sizeof(std::uint32_t)),
                      table.data());
    {
        // 屏障覆盖组内全部实例的输入 buffer + 实例表（vkCmdUpdateBuffer 是
        // TRANSFER 写，record_input_barriers 的 src 阶段含 TRANSFER）；
        // 同一 buffer 去重（组内互不依赖已由引擎保证，仅防重复屏障）。
        std::vector<VkBuffer> in_bufs;
        in_bufs.reserve(instances.size() * (n_in + 1) + 1);
        for (const FusedBatchInst& inst : instances)
            for (const GpuBuffer* b : inst.inputs)
                in_bufs.push_back(b->impl());
        in_bufs.push_back(inst_buf->impl());
        std::sort(in_bufs.begin(), in_bufs.end());
        in_bufs.erase(std::unique(in_bufs.begin(), in_bufs.end()), in_bufs.end());
        record_input_barriers(cmd, in_bufs);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    // Push constants 与 run_fused_gpu 的 V0 逐元素分支同源：
    //   count(=max_count), cols, [vp0..], [c0..], [rp0..]
    // 批量 shader 不读 count/cols 两槽（每实例取尾槽），两值仅保持 V0 布局。
    const std::uint32_t pc_uints = pc_base_fixed + n_vp;
    std::vector<std::uint8_t> pc((pc_uints) * sizeof(std::uint32_t) +
                                 sizeof(Scalar) * consts.size() + sizeof(Scalar) * rparams.size());
    std::memcpy(pc.data(), &max_count, sizeof(std::uint32_t));
    std::memcpy(pc.data() + sizeof(std::uint32_t), &max_cols, sizeof(std::uint32_t));
    for (std::uint32_t i = 0; i < n_vp; ++i)
        std::memcpy(pc.data() + (pc_base_fixed + i) * sizeof(std::uint32_t), &view_params[i],
                    sizeof(std::uint32_t));
    if (!consts.empty())
        std::memcpy(pc.data() + (pc_uints) * sizeof(std::uint32_t), consts.data(),
                    sizeof(Scalar) * consts.size());
    if (!rparams.empty())
        std::memcpy(pc.data() + (pc_uints) * sizeof(std::uint32_t) + sizeof(Scalar) * consts.size(),
                    rparams.data(), sizeof(Scalar) * rparams.size());
    vkCmdPushConstants(cmd, pipeline.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<std::uint32_t>(pc.size()), pc.data());

    // dispatch = (ceil(max_count/256), n_instances, 1)；批量恒标量
    // （vec_width=1，与生成器同源）
    const std::uint32_t wg_x = (max_count + 255u) / 256u;
    ts_dispatch(cmd, "fused_b", wg_x, static_cast<std::uint32_t>(instances.size()), 1, shader_name);

    // 输出屏障（主输出 + 全部 extras，去重）：组内输出互不重叠（引擎保证）
    {
        std::vector<VkBuffer> out_bufs;
        out_bufs.reserve(instances.size() * (1 + n_ex));
        for (const FusedBatchInst& inst : instances)
        {
            out_bufs.push_back(inst.out->impl());
            for (const GpuBuffer* b : inst.extra_outs)
                out_bufs.push_back(b->impl());
        }
        std::sort(out_bufs.begin(), out_bufs.end());
        out_bufs.erase(std::unique(out_bufs.begin(), out_bufs.end()), out_bufs.end());
        for (VkBuffer b : out_bufs)
            record_output_barrier(cmd, b);
    }

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    ++fused_batch_dispatches_;
    if (profile_ops_enabled())
    {
        const long long dt = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - prof_t0)
                                 .count();
        std::fprintf(stderr,
                     "[gpu-profile] fused_batch inst=%zu max_count=%u total=%lldus "
                     "key=%.16s\n",
                     instances.size(), max_count, dt, shader_name.c_str());
    }
    return {};
}

// 查询融合 shader 是否已注册（GpuEngine AOT 匹配前置判断）
[[nodiscard]] bool has_fused_shader(const std::string& shader_name) const noexcept
{ return fused_kernels_.find(shader_name) != fused_kernels_.end(); }

// P3 批量派发计数（测试/诊断：确认批量路径真的被走到，而非静默降级）
[[nodiscard]] std::uint64_t fused_batch_dispatches() const noexcept
{ return fused_batch_dispatches_; }

// ══════════════════════════════════════════════════════════════════
// reduce_gpu — 行/列归约原语（sum/max）
//
// Push Constants (20 bytes): rows, cols, mode, reduce_op, chunk_rows
// Bindings: In(0), Out(1)
// mode: 0=row_reduce → out(rows,1), 1=col_reduce → out(1,cols)
// reduce_op: 0=sum, 1=max
// 列归约大行数走两段式 partials（同 cmd 双 dispatch 单次提交，见分支内注释）
// f16_in（Phase C2）：输入缓冲按 float16_t 读（PipelineId::reduce_f16）；
// 输出/ partials 恒 f32 → 两段式 pass2 输入是 f32 partials，恒用 f32
// pipeline（ds/layout 各随所属 pipeline）。f16 pipeline 未创建时硬报错，
// 由调用方（GpuEngine）提前查 has(PipelineId::reduce_f16) 走引擎内 cast 回退。
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<GpuTensor> reduce_gpu(const GpuTensor& input, uint32_t mode,
                                           uint32_t reduce_op, bool f16_in = false)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (!has(PipelineId::reduce)) NN_FAIL("reduce pipeline not available");
    if (f16_in && !has(PipelineId::reduce_f16)) NN_FAIL("reduce f16 pipeline not available");
    // 输入侧 pipeline（pass1 / 单段）；输出侧与 pass2 恒 f32 pipeline
    const VulkanPipeline& in_pipe =
        f16_in ? pipelines_.get(PipelineId::reduce_f16) : pipelines_.get(PipelineId::reduce);

    const uint32_t rows = static_cast<uint32_t>(input.rows());
    const uint32_t cols = static_cast<uint32_t>(input.cols());

    // 1. 分配输出 Tensor
    std::size_t out_rows = (mode == 0) ? rows : 1;
    std::size_t out_cols = (mode == 0) ? 1 : cols;
    auto output_res = GpuTensor::create_empty(out_rows, out_cols, *this);
    NN_TRY_CHECK(output_res);
    GpuTensor output = std::move(*output_res);

    // ── 列归约大行数：两段式 partials（ggml rms_norm_partials 模式）──
    // 单段 dispatch = ceil(cols/32) 个 WG（5244² 仅 164 vs 136 并发槽 →
    // 1.2 波，末波只剩 ~20% 机器喂内存 → 带宽被波尾压约两成）。
    // pass1 把行向切 nchunk 块（dispatch y = nchunk）→ WG 数 ×nchunk，
    // 每 WG 归约一行块写 partials(nchunk, cols)；pass2 复用整表路径合并
    // （行数 = nchunk ≤ 8，成本可忽略）。两 pass 录进同一 cmd 单次提交
    // → 固定提交开销不翻倍。小行数保持单段（零开销）。
    uint32_t nchunk = 1u;
    if (mode == 1u)
    {
        const uint32_t tiles = (cols + 31u) / 32u;
        nchunk = (rows + 1023u) / 1024u;
        if (nchunk > 8u) nchunk = 8u;
        const uint32_t want = (512u + tiles - 1u) / tiles; // ≥512 WG ≈ 4 波
        if (want > nchunk) nchunk = want;
        if (nchunk > 8u) nchunk = 8u;
        // 小形状回落单段：512² 实测两段 −16%（第二遍+屏障对
        // ~10µs kernel 净亏），1024² 起两段持平或转优 → 边界 512K 元素
        if (rows < 256u || static_cast<std::uint64_t>(rows) * cols < 524'288ull) nchunk = 1u;
    }
    // 两段式启用条件（同窗交错 A/B 实测，同码对照噪声地板 ±5%）：
    //   5244² 两段 −6%（0.405 vs 0.430；单段 1.2 波尾欠喂被消除）
    //   4096² 两段 ~2%（0.277 vs 0.283；单波本已喂满，第二段纯付 ~6µs）
    //   512² 两段 −16%（对 ~10µs kernel 第二遍+屏障净亏）→ 尺寸护栏只排除它
    const bool two_pass = (nchunk > 1u);
    const uint32_t chunk_rows = two_pass ? (rows + nchunk - 1u) / nchunk : 0u;

    if (two_pass)
    {
        NN_TRY_CHECK(ensure_reduce_partial_(nchunk, cols));

        // pass1 读输入（f16_in → f16 pipeline，ds/layout 随之）；
        // pass2 读 f32 partials → 恒 f32 pipeline（f16 shader 读 f32
        // partials = 静默错值，必须分开）
        const VkBuffer pass1_bindings[2]{input.buffer().impl(), reduce_partial_.buffer};
        const VkBuffer pass2_bindings[2]{reduce_partial_.buffer, output.buffer().impl()};
        auto ds1_r = alloc_desc_set_for(in_pipe.descriptor_layout(), pass1_bindings);
        NN_TRY_CHECK(ds1_r);
        VkDescriptorSet ds1 = *ds1_r;
        auto ds2_r = alloc_desc_set_for(pipelines_.get(PipelineId::reduce).descriptor_layout(),
                                        pass2_bindings);
        NN_TRY_CHECK(ds2_r);
        VkDescriptorSet ds2 = *ds2_r;

        auto cmd_r = acquire_cmd();
        if (!cmd_r)
        {
            release_desc_set_now(ds1);
            release_desc_set_now(ds2);
            NN_TRY_CHECK(cmd_r);
        }
        auto [cmd, owns_cmd] = *cmd_r;

        // pass1：行块级部分归约 → partials（读输入 → in_pipe）
        record_input_barriers(cmd, {input.buffer().impl()});
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, in_pipe.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, in_pipe.pipeline_layout(), 0,
                                1, &ds1, 0, nullptr);
        const uint32_t push1[5] = {rows, cols, 1u, reduce_op, chunk_rows};
        vkCmdPushConstants(cmd, in_pipe.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push1), push1);
        ts_dispatch(cmd, "reduce", (cols + 31u) / 32u, nchunk, 1);

        // partials：写 → 读（同 cmd 内跨 dispatch 依赖）
        record_input_barriers(cmd, {reduce_partial_.buffer});

        // pass2：合并 partials（rows = nchunk ≤ 8，复用 mode1 整表路径；
        // 输入 = f32 partials → 恒 f32 pipeline）
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelines_.get(PipelineId::reduce).handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelines_.get(PipelineId::reduce).pipeline_layout(), 0, 1, &ds2, 0,
                                nullptr);
        const uint32_t push2[5] = {nchunk, cols, 1u, reduce_op, 0u};
        vkCmdPushConstants(cmd, pipelines_.get(PipelineId::reduce).pipeline_layout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push2), push2);
        ts_dispatch(cmd, "reduce", (cols + 31u) / 32u, 1, 1);
        record_output_barrier(cmd, output.buffer().impl());

        if (owns_cmd)
        {
            auto r = submit_and_wait(cmd, ds1); // 等 fence 后归还 ds1
            NN_TRY_CHECK(r);
            vkFreeDescriptorSets(device_.device(), gpu_tensor_pool_, 1, &ds2);
        }
        return output;
    }

    // 2. 分配描述符集（随 in_pipe：f16_in 时输入按 float16_t 读）
    const VkBuffer bindings[2]{input.buffer().impl(), output.buffer().impl()};
    auto ds_r = alloc_desc_set_for(in_pipe.descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    // 4. 获取 command buffer
    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    // 5. 录制
    record_input_barriers(cmd, {input.buffer().impl()});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, in_pipe.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, in_pipe.pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    const uint32_t push_data[5] = {rows, cols, mode, reduce_op, 0u};
    vkCmdPushConstants(cmd, in_pipe.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push_data), push_data);

    // 行：每 WG 256 线程协作归约一行 → rows 个 WG；
    // 列：每 WG 32 列 tile（lane=列、warp=行块，合并访问；与
    //     reduce.comp 的 WG 结构契约同源）→ (cols+31)/32 个 WG
    const uint32_t workgroups = (mode == 0) ? rows : (cols + 31u) / 32u;
    ts_dispatch(cmd, "reduce", workgroups, 1, 1);
    record_output_barrier(cmd, output.buffer().impl());

    // 6. 独立模式提交
    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    return output;
}

// ══════════════════════════════════════════════════════════════════
// rearrange_3d_gpu — 3D 维度转置 (M, B, N) ↔ (B, M, N)
//
// Push Constants (20 bytes): M, B, N, inverse, total
// Bindings: In(0), Out(1)
//   inverse=0: (M, B*N) → (B*M, N)
//   inverse=1: (B*M, N) → (M, B*N)
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<GpuTensor> rearrange_3d_gpu(const GpuTensor& input, uint32_t M, uint32_t B,
                                                 uint32_t N, uint32_t inverse, bool f16_io = false)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (!has(PipelineId::rearrange_3d)) NN_FAIL("rearrange_3d pipeline not available");
    if (f16_io && !has(PipelineId::rearrange_3d_f16))
        NN_FAIL("rearrange_3d_gpu: f16 pipeline 不可用（设备无16位存储）");
    const auto& pipeline = f16_io ? pipelines_.get(PipelineId::rearrange_3d_f16)
                                  : pipelines_.get(PipelineId::rearrange_3d);

    const uint32_t total = M * B * N;
    if (total != input.rows() * input.cols()) NN_FAIL("rearrange_3d: element count mismatch");
    // 1. 分配输出 Tensor（f16：2B/元素 + 纯绑定视图，同 matmul_gpu）
    const std::size_t out_rows = inverse ? M : (static_cast<std::size_t>(B) * M);
    const std::size_t out_cols = inverse ? (static_cast<std::size_t>(B) * N) : N;
    std::optional<GpuTensor> owned;
    if (f16_io)
    {
        auto f16r = GpuTensorF16::create_empty(out_rows, out_cols, *this);
        NN_TRY_CHECK(f16r);
        owned.emplace(f16r->shared_buffer(), f16r->rows(), f16r->cols());
    }
    else
    {
        auto output_res = GpuTensor::create_empty(out_rows, out_cols, *this);
        NN_TRY_CHECK(output_res);
        owned.emplace(std::move(*output_res));
    }
    GpuTensor output = *owned;

    // 2. 分配并写入描述符集
    const VkBuffer bindings[2]{input.buffer().impl(), output.buffer().impl()};
    auto ds_r = alloc_desc_set_for(pipeline.descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    // 4. 获取 command buffer
    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    // 5. 录制
    record_input_barriers(cmd, {input.buffer().impl()});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    const uint32_t push_data[5] = {M, B, N, inverse, total};
    vkCmdPushConstants(cmd, pipeline.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push_data), push_data);

    const uint32_t wg_count = (total + 255) / 256;
    ts_dispatch(cmd, "rearrange_3d", wg_count, 1, 1);
    record_output_barrier(cmd, output.buffer().impl());

    // 6. 独立模式提交
    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    return output;
}

// ══════════════════════════════════════════════════════════════════
// fill_zero_gpu — 清零 GPU buffer（使用 vkCmdFillBuffer，字节级操作，§6.3）
// ══════════════════════════════════════════════════════════════════
template <Precision P> [[nodiscard]] Result<void> fill_zero_gpu(GpuTensorT<P>& tensor)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    auto cmd_r = acquire_cmd();
    NN_TRY_CHECK(cmd_r);
    auto [cmd, owns_cmd] = *cmd_r;

    vkCmdFillBuffer(cmd, tensor.buffer().impl(), 0, VK_WHOLE_SIZE, 0);

    // 屏障确保 fill 完成后可见
    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = tensor.buffer().impl();
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 1, &b, 0, nullptr);

    if (owns_cmd)
    {
        // fill_zero 不使用描述符集，传 VK_NULL_HANDLE 跳过释放
        auto r = submit_and_wait(cmd, VK_NULL_HANDLE);
        NN_TRY_CHECK(r);
    }
    return {};
}

// ══════════════════════════════════════════════════════════════════
// copy_buffer_gpu — GPU 内 buffer 拷贝（使用 vkCmdCopyBuffer）
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<void> copy_buffer_gpu(VkBuffer src, VkBuffer dst, VkDeviceSize size)
{ return copy_buffer_region_gpu(src, 0, dst, 0, size); }

// ══════════════════════════════════════════════════════════════════
// copy_buffer_region_gpu — 带偏移的 buffer 拷贝（offload slab / 通用区域拷贝）
// 从 src[src_offset, src_offset+size) 拷到 dst[dst_offset, dst_offset+size)。
// batch 模式只录制不提交；非 batch 独立提交等待。
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<void> copy_buffer_region_gpu(VkBuffer src, VkDeviceSize src_offset,
                                                  VkBuffer dst, VkDeviceSize dst_offset,
                                                  VkDeviceSize size)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    auto cmd_r = acquire_cmd();
    NN_TRY_CHECK(cmd_r);
    auto [cmd, owns_cmd] = *cmd_r;

    VkBufferCopy cp{src_offset, dst_offset, size};
    vkCmdCopyBuffer(cmd, src, dst, 1, &cp);

    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = dst;
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 1, &b, 0, nullptr);

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, VK_NULL_HANDLE);
        NN_TRY_CHECK(r);
    }
    return {};
}

// ══════════════════════════════════════════════════════════════════
// 异步标量回读—— 语义见成员声明处
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<void> init_scalar_readback_slots()
{
    rb_slots_.resize(SCALAR_READBACK_SLOTS);
    for (auto& s : rb_slots_)
    {
        VkCommandBufferAllocateInfo ca{};
        ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool = command_pool_;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        auto r = detail::vk_check(vkAllocateCommandBuffers(device_.device(), &ca, &s.cmd), __FILE__,
                                  __LINE__);
        NN_TRY_CHECK(r);
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = detail::vk_check(vkCreateFence(device_.device(), &fi, nullptr, &s.fence), __FILE__,
                             __LINE__);
        NN_TRY_CHECK(r);
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = sizeof(float);
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        // P3：同其余缓冲创建点补 SHADER_DEVICE_ADDRESS（池分配侧配对）
        if (memory_pool().device_address_alloc())
            bi.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        r = detail::vk_check(vkCreateBuffer(device_.device(), &bi, nullptr, &s.host), __FILE__,
                             __LINE__);
        NN_TRY_CHECK(r);
        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(device_.device(), s.host, &mr);
        constexpr VkMemoryPropertyFlags kVis = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        constexpr VkMemoryPropertyFlags kCoh = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        auto ar = memory_pool().allocate(mr, kVis | kCoh, kVis);
        NN_TRY_CHECK(ar);
        s.alloc = *ar;
        r = detail::vk_check(
            vkBindBufferMemory(device_.device(), s.host, s.alloc.memory, s.alloc.offset), __FILE__,
            __LINE__);
        NN_TRY_CHECK(r);
        r = detail::vk_check(vkMapMemory(device_.device(), s.alloc.memory, s.alloc.offset,
                                         sizeof(float), 0, &s.mapped),
                             __FILE__, __LINE__);
        NN_TRY_CHECK(r);
    }
    return {};
}

void destroy_scalar_readback_slots()
{
    for (auto& s : rb_slots_)
    {
        if (s.mapped != nullptr && s.alloc.valid())
        {
            vkUnmapMemory(device_.device(), s.alloc.memory);
            s.mapped = nullptr;
        }
        if (s.host != VK_NULL_HANDLE) vkDestroyBuffer(device_.device(), s.host, nullptr);
        if (s.fence != VK_NULL_HANDLE) vkDestroyFence(device_.device(), s.fence, nullptr);
        if (s.alloc.valid()) memory_pool().free(s.alloc);
        s.host = VK_NULL_HANDLE;
        s.fence = VK_NULL_HANDLE;
        s.cmd = VK_NULL_HANDLE; // command buffer 随 command pool 释放
        s.pending = false;
    }
    rb_slots_.clear();
}

// 录制一次 4B D2H 拷贝并提交（**不等待**）。
// 调用约定：必须在产出该标量的主帧已提交之后调用（同队列 FIFO ⇒
// 拷贝排在生产者之后；若在主帧提交前提交，会读到上一轮的旧值）。
[[nodiscard]] Result<void> submit_scalar_readback(std::size_t slot, VkBuffer src)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (slot >= rb_slots_.size()) NN_FAIL("scalar readback: slot out of range");
    auto& s = rb_slots_[slot];

    // 上一轮的值尚未取走：等它就绪（防御路径；正常调用方先 poll）
    if (s.pending)
    {
        constexpr uint64_t kTimeoutNs = 60'000'000'000ULL;
        const VkResult wr = vkWaitForFences(device_.device(), 1, &s.fence, VK_TRUE, kTimeoutNs);
        if (wr == VK_ERROR_DEVICE_LOST)
        {
            device_lost_ = true;
            NN_FAIL("scalar readback: VK_ERROR_DEVICE_LOST（GPU 已被 TDR 重置）");
        }
        if (wr != VK_SUCCESS)
            NN_FAIL("scalar readback: fence wait failed " + std::to_string(static_cast<int>(wr)));
        s.pending = false;
    }

    auto r = detail::vk_check(vkResetCommandBuffer(s.cmd, 0), __FILE__, __LINE__);
    NN_TRY_CHECK(r);
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = detail::vk_check(vkBeginCommandBuffer(s.cmd, &cbi), __FILE__, __LINE__);
    NN_TRY_CHECK(r);
    VkBufferCopy cp{};
    cp.srcOffset = 0;
    cp.dstOffset = 0;
    cp.size = sizeof(float); // (1,1) f32 标量
    vkCmdCopyBuffer(s.cmd, src, s.host, 1, &cp);

    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = s.host;
    b.offset = 0;
    b.size = sizeof(float);
    vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,
                         nullptr, 1, &b, 0, nullptr);

    r = detail::vk_check(vkEndCommandBuffer(s.cmd), __FILE__, __LINE__);
    NN_TRY_CHECK(r);
    r = detail::vk_check(vkResetFences(device_.device(), 1, &s.fence), __FILE__, __LINE__);
    NN_TRY_CHECK(r);
    {
        std::lock_guard lock(queue_mutex_);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &s.cmd;
        r = detail::vk_check(vkQueueSubmit(device_.compute_queue(), 1, &si, s.fence), __FILE__,
                             __LINE__);
    }
    NN_TRY_CHECK(r);
    s.pending = true;
    return {};
}

// 非阻塞查询：就绪则写出 out 并返回 true；未就绪返回 false（不阻塞）。
[[nodiscard]] Result<bool> poll_scalar_readback(std::size_t slot, float& out)
{
    if (slot >= rb_slots_.size()) NN_FAIL("scalar readback: slot out of range");
    auto& s = rb_slots_[slot];
    if (!s.pending) return false;
    const VkResult st = vkGetFenceStatus(device_.device(), s.fence);
    if (st == VK_NOT_READY) return false;
    if (st == VK_ERROR_DEVICE_LOST)
    {
        device_lost_ = true;
        NN_FAIL("scalar readback: VK_ERROR_DEVICE_LOST（GPU 已被 TDR 重置）");
    }
    if (st != VK_SUCCESS)
        NN_FAIL("scalar readback: fence status " + std::to_string(static_cast<int>(st)));
    // size = VK_WHOLE_SIZE 时 offset 无对齐要求（VU 只约束非 WHOLE_SIZE）。
    VkMappedMemoryRange mr{};
    mr.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mr.memory = s.alloc.memory;
    mr.offset = s.alloc.offset;
    mr.size = VK_WHOLE_SIZE;
    if (vkInvalidateMappedMemoryRanges(device_.device(), 1, &mr) != VK_SUCCESS)
        NN_FAIL("scalar readback: invalidate mapped range failed");
    std::memcpy(&out, s.mapped, sizeof(float));
    s.pending = false;
    return true;
}

// ══════════════════════════════════════════════════════════════════
// clone_gpu — GPU 内深拷贝（分配新 buffer + copy，无 PCIe 传输）
// ══════════════════════════════════════════════════════════════════
template <Precision P> [[nodiscard]] Result<GpuTensorT<P>> clone_gpu(const GpuTensorT<P>& src)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    // 1. 分配同形状的新 buffer
    auto dst_res = GpuTensorT<P>::create_empty(src.rows(), src.cols(), *this);
    NN_TRY_CHECK(dst_res);
    GpuTensorT<P> dst = std::move(*dst_res);

    // 2. GPU 内拷贝
    const VkDeviceSize size = static_cast<VkDeviceSize>(src.rows() * src.cols() * sizeof(elem<P>));
    auto r = copy_buffer_gpu(src.buffer().impl(), dst.buffer().impl(), size);
    NN_TRY_CHECK(r);
    return dst;
}

// ── 通用精度转换原语（engine.cast 的 GPU 实现，无 PCIe 往返）────────
// 源/目标均为原始 32-bit word 缓冲；kind 分派精度对：
//   0 = f16→f32（src 2B/pair 打包，dst 4B/word）
//   1 = f32→f16（src 4B/word，dst 2B/pair 打包，round-half-to-even）
// 元素数保持不变（count = rows*cols），仅字节宽度差。后续精度对
// （BF16/F64/f8 等）扩 kind 枚举即可，绝不改引擎 cast API。
[[nodiscard]] Result<void> cast_gpu(const GpuBuffer& src, const GpuBuffer& dst, std::size_t count,
                                    std::uint32_t kind)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (!has(PipelineId::cast)) NN_FAIL("cast_gpu: cast pipeline not available");
    const VkBuffer bindings[2]{src.impl(), dst.impl()};
    auto ds_r = alloc_desc_set_for(pipelines_.get(PipelineId::cast).descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    std::vector<VkBuffer> in_bufs{src.impl()};
    record_input_barriers(cmd, in_bufs);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipelines_.get(PipelineId::cast).handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelines_.get(PipelineId::cast).pipeline_layout(), 0, 1, &desc_set, 0,
                            nullptr);

    struct Push
    {
        std::uint32_t count;
        std::uint32_t kind;
    };
    Push push{static_cast<std::uint32_t>(count), kind};
    std::vector<std::uint8_t> pc(sizeof(push));
    std::memcpy(pc.data(), &push, sizeof(push));
    vkCmdPushConstants(cmd, pipelines_.get(PipelineId::cast).pipeline_layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(pc.size()), pc.data());

    // 每线程处理 2 元素（一对一 half 或单词对）
    const auto pairs = (count + 1u) / 2u;
    const std::uint32_t wg = static_cast<std::uint32_t>((pairs + 255u) / 256u);
    ts_dispatch(cmd, "cast", wg < 1u ? 1u : wg, 1u, 1u);
    record_output_barrier(cmd, dst.impl());

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    return {};
}

// ── 三值打包（P2/F3）：W (rows,k) f32 + τ (rows) f32 → 打包字 (rows,wpr) ──
[[nodiscard]] Result<void> trit_pack_gpu(const GpuBuffer& w, const GpuBuffer& tau,
                                         const GpuBuffer& packed, std::uint32_t rows,
                                         std::uint32_t k, std::uint32_t wpr)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (!has(PipelineId::trit_pack)) NN_FAIL("trit_pack_gpu: trit_pack pipeline not available");
    const VkBuffer bindings[3]{w.impl(), tau.impl(), packed.impl()};
    auto ds_r =
        alloc_desc_set_for(pipelines_.get(PipelineId::trit_pack).descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    std::vector<VkBuffer> in_bufs{w.impl(), tau.impl()};
    record_input_barriers(cmd, in_bufs);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipelines_.get(PipelineId::trit_pack).handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelines_.get(PipelineId::trit_pack).pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    struct Push
    {
        std::uint32_t rows;
        std::uint32_t k;
        std::uint32_t wpr;
    };
    Push push{rows, k, wpr};
    std::vector<std::uint8_t> pc(sizeof(push));
    std::memcpy(pc.data(), &push, sizeof(push));
    vkCmdPushConstants(cmd, pipelines_.get(PipelineId::trit_pack).pipeline_layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(pc.size()), pc.data());

    const std::uint32_t total = rows * wpr;
    const std::uint32_t wg = (total + 255u) / 256u;
    vkCmdDispatch(cmd, wg < 1u ? 1u : wg, 1u, 1u);
    record_output_barrier(cmd, packed.impl());

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    return {};
}

// ── 量化 GEMM（P2/F3）：C = (τ∘A)^(T)·B（A 为打包三值）────────────────
// f16 = true 时用 f16 存储变体（B/C 均为 half），否则全 f32。
[[nodiscard]] Result<void> matmul_q_gpu(const GpuBuffer& packed, const GpuBuffer& tau,
                                        const GpuBuffer& b, const GpuBuffer& c, std::uint32_t M,
                                        std::uint32_t N, std::uint32_t K, std::uint32_t transA,
                                        std::uint32_t wpr, bool f16)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    // ── 选址（与稠密 matmul 同口径，实测驱动；docs 21 §4.12）──────────
    //   窄 N（≤ GEMV_MAX_N）→ **GEMV 内核**：分块版按 64×64 块算，n=1 时
    //   63/64 列空转（102/40HX 实测打包侧权重带宽只有 2.9 GB/s vs 稠密
    //   GEMV 97 GB/s）→ 解码路必须走"每线程沿打包字分片 + 只算真实列"。
    //   宽 N（训练）→ 分块内核（分块解码一次进 shared，计算受限）。
    //   GEMV 只在 transA=0（转发形态）实现。**布局契约**：shader 里每个
    //   subgroup 负责 RPS=4 行（`m0 = (WGID.x*gl_NumSubgroups + SubgroupID)*4`），
    //   故 grid.x 必须 = ceil(M / (4*(256/subgroup_size)))，且 256 必须被
    //   subgroup_size 整除（否则 gl_NumSubgroups 与 host 侧算出不一致）。
    //   设备侧 subgroup≥4 是红线（shader 的 subgroupAdd 归约语义）。
    constexpr std::uint32_t QGEMV_MAX_N = 8u;
    constexpr std::uint32_t QGEMV_THREADS = 256u; // 与 matmul_q_gemv.comp 的 local_size_x 一致
    constexpr std::uint32_t QGEMV_RPS = 4u; // 与 matmul_q_gemv.comp 的 RPS 一致
    const std::uint32_t sgs = device_.subgroup_size();
    const bool qgemv_layout_ok = sgs >= 4u && sgs <= QGEMV_THREADS && (QGEMV_THREADS % sgs) == 0u;
    const std::uint32_t qgemv_rows = qgemv_layout_ok ? (QGEMV_THREADS / sgs) * QGEMV_RPS : 0u;
    const bool use_gemv = (transA == 0u) && N <= QGEMV_MAX_N && qgemv_layout_ok &&
                          has(PipelineId::matmul_q_gemv) &&
                          (!f16 || has(PipelineId::matmul_q_gemv_f16));
    const VulkanPipeline* pipe = nullptr;
    if (use_gemv)
        pipe = f16 ? &pipelines_.get(PipelineId::matmul_q_gemv_f16)
                   : &pipelines_.get(PipelineId::matmul_q_gemv);
    else if (f16)
    {
        if (!has(PipelineId::matmul_q_f16))
            NN_FAIL("matmul_q_gpu: f16 pipeline not available（设备缺 16 位存储？）");
        pipe = &pipelines_.get(PipelineId::matmul_q_f16);
    }
    else
    {
        if (!has(PipelineId::matmul_q)) NN_FAIL("matmul_q_gpu: matmul_q pipeline not available");
        pipe = &pipelines_.get(PipelineId::matmul_q);
    }
    const VkBuffer bindings[4]{packed.impl(), tau.impl(), b.impl(), c.impl()};
    auto ds_r = alloc_desc_set_for(pipe->descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    std::vector<VkBuffer> in_bufs{packed.impl(), tau.impl(), b.impl()};
    record_input_barriers(cmd, in_bufs);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    struct Push
    { std::uint32_t M, N, K, transA, wpr; };
    Push push{M, N, K, transA, wpr};
    std::vector<std::uint8_t> pc(sizeof(push));
    std::memcpy(pc.data(), &push, sizeof(push));
    vkCmdPushConstants(cmd, pipe->pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<uint32_t>(pc.size()), pc.data());

    // 分块版 matmul_q.comp：16×16 线程 / 输出块 64×64（BM=BN=64）。
    // ⚠ 两条不变量（分块内核的"形状契约"，漏掉任一条都是**静默错值**）：
    //   1. 网格必须与 shader 的分块常量一致：写少了漏算、写多了重复算
    //      （正确但成倍变慢 —— 实测错误网格下宽 batch 训练慢 13×）。
    //   2. **轴向必须对齐**：shader 里 x → block_col（N 方向）、y →
    //      block_row（M 方向），与 matmul_tiled / batched_matmul 同一约定
    //      （两者 dispatch 均为 `(ceil(N/64), ceil(M/64), 1)`）。写成
    //      `(ceil(M/64), ceil(N/64))` 会让 M≠N 的形状只算到
    //      `ceil(N/64)*64` 行 / `ceil(M/64)*64` 列 —— 方形小形状
    //      （M=N≤64，单 workgroup）恰好正确，故此缺陷只在真实形状上暴露
    //      （2B4T 的 lm_head M=128256 只算到前 256 行）。
    if (use_gemv)
    {
        // 单维 grid.x = ceil(M/qgemv_rows)；qgemv_rows 见上面的布局契约注释
        const std::uint32_t gw = (M + qgemv_rows - 1u) / qgemv_rows;
        vkCmdDispatch(cmd, gw < 1u ? 1u : gw, 1u, 1u);
    }
    else
    {
        const std::uint32_t gx = (N + 63u) / 64u; // x ↔ 输出列（N）
        const std::uint32_t gy = (M + 63u) / 64u; // y ↔ 输出行（M）
        vkCmdDispatch(cmd, gx < 1u ? 1u : gx, gy < 1u ? 1u : gy, 1u);
    }
    record_output_barrier(cmd, c.impl());

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    return {};
}

// ── 创建逻辑形状 (rows,cols) 但存储为偶数槽位的 f16 张量 ───────────
// f32→f16 的 cast 每线程按整 word（两 half）写入；奇数元素 count 时最后一个
// word 会越界 2 字节。这里把底层缓冲按偶数元素分配（多 1 槽），逻辑形状不变，
// 写 word 永不越界；to_matrix 仍只读前 rows*cols 个元素。
[[nodiscard]] Result<GpuTensorF16> create_f16_tensor(std::size_t rows, std::size_t cols)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    const std::size_t cnt = rows * cols;
    const std::size_t cnt_pad = cnt + (cnt & 1u); // 偶数槽位
    auto b = GpuBuffer::create_device_local(device_.device(), *memory_pool_, cnt_pad * 2u,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                            keepalive());
    NN_TRY_CHECK(b);
    return GpuTensorF16(std::make_shared<GpuBuffer>(std::move(*b)), rows, cols);
}
template <Precision P>
[[nodiscard]] Result<GpuTensorT<P>> slice_rows_gpu(const GpuTensorT<P>& src, std::size_t start_row,
                                                   std::size_t count)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    const std::size_t cols = src.cols();
    if (start_row + count > src.rows()) NN_FAIL("slice_rows_gpu: range out of bounds");
    // 分配目标 buffer
    auto dst_res = GpuTensorT<P>::create_empty(count, cols, *this);
    NN_TRY_CHECK(dst_res);
    GpuTensorT<P> dst = std::move(*dst_res);

    // 行区间在行主序下连续：[start_row * cols, (start_row + count) * cols)
    const std::size_t elem_size = sizeof(elem<P>);
    const VkDeviceSize src_offset = static_cast<VkDeviceSize>(start_row * cols * elem_size);
    const VkDeviceSize size = static_cast<VkDeviceSize>(count * cols * elem_size);

    auto cmd_r = acquire_cmd();
    NN_TRY_CHECK(cmd_r);
    auto [cmd, owns_cmd] = *cmd_r;

    VkBufferCopy cp{src_offset, 0, size};
    vkCmdCopyBuffer(cmd, src.buffer().impl(), dst.buffer().impl(), 1, &cp);

    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = dst.buffer().impl();
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 1, &b, 0, nullptr);

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, VK_NULL_HANDLE);
        NN_TRY_CHECK(r);
    }
    return dst;
}

// ══════════════════════════════════════════════════════════════════
// insert_rows_gpu — 行插入（GPU 内就地写入，无 PCIe 传输）
// 将 src 的所有行写入 dst 的行 [dst_start_row, dst_start_row + src.rows())
// 真·就地修改（vkCmdCopyBuffer with dstOffset）。
// 注意：dst 必须以 TRANSFER_DST_BIT 创建（create_empty 已包含）。
// ══════════════════════════════════════════════════════════════════
template <Precision P>
[[nodiscard]] Result<void> insert_rows_gpu(GpuTensorT<P>& dst, std::size_t dst_start_row,
                                           const GpuTensorT<P>& src)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    const std::size_t cols = dst.cols();
    if (src.cols() != cols) NN_FAIL("insert_rows_gpu: column count mismatch");
    if (dst_start_row + src.rows() > dst.rows()) NN_FAIL("insert_rows_gpu: range out of bounds");
    const std::size_t elem_size = sizeof(elem<P>);
    const VkDeviceSize dst_offset = static_cast<VkDeviceSize>(dst_start_row * cols * elem_size);
    const VkDeviceSize size = static_cast<VkDeviceSize>(src.rows() * cols * elem_size);

    auto cmd_r = acquire_cmd();
    NN_TRY_CHECK(cmd_r);
    auto [cmd, owns_cmd] = *cmd_r;

    VkBufferCopy cp{0, dst_offset, size};
    vkCmdCopyBuffer(cmd, src.buffer().impl(), dst.buffer().impl(), 1, &cp);

    VkBufferMemoryBarrier b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = dst.buffer().impl();
    b.offset = 0;
    b.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 1, &b, 0, nullptr);

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, VK_NULL_HANDLE);
        NN_TRY_CHECK(r);
    }
    return {};
}

// ── Staging Path：CPU span → GPU → 计算 → GPU → CPU span ─────────
[[nodiscard]] Result<void> matmul_direct(std::span<const Scalar> a, std::span<const Scalar> b,
                                         std::span<Scalar> c, std::size_t M, std::size_t N,
                                         std::size_t K, uint32_t transA = 0, uint32_t transB = 0)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    // 1. 创建临时 GPU buffers（byte count = 元素数 × sizeof(float)，§6.3）
    auto a_buf = GpuBuffer::create_device_local(
        device_.device(), *memory_pool_, M * K * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, keepalive());
    NN_TRY_CHECK(a_buf);
    auto b_buf = GpuBuffer::create_device_local(
        device_.device(), *memory_pool_, K * N * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, keepalive());
    NN_TRY_CHECK(b_buf);
    // 2. 上传数据
    auto a_tensor = GpuTensor(std::make_shared<GpuBuffer>(std::move(*a_buf)), M, K);
    auto b_tensor = GpuTensor(std::make_shared<GpuBuffer>(std::move(*b_buf)), K, N);

    auto r = upload_blocking(a_tensor, a);
    if (!r) return r;

    r = upload_blocking(b_tensor, b);
    if (!r) return r;

    // 3. 执行 matmul
    auto c_tensor_res = matmul_gpu(a_tensor, b_tensor, transA, transB);
    NN_TRY_CHECK(c_tensor_res);
    // 4. 下载结果
    r = download_blocking(*c_tensor_res, c);
    if (!r) return r;

    return {};
}

// ══════════════════════════════════════════════════════════════════
// transpose_gpu — 矩阵转置 (R, C) → (C, R)
//
// Push Constants (8 bytes): rows, cols
// Bindings: In(0), Out(1)
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<GpuTensor> transpose_gpu(const GpuTensor& A, bool f16_io = false)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (!has(PipelineId::transpose)) NN_FAIL("transpose pipeline not available");
    // Phase C1 f16 变体：f16_io 请求但设备无 f16 pipeline（能力门控未过）
    // → 显式报错，引擎层回退边界 cast（不静默绑错类型）
    if (f16_io && !has(PipelineId::transpose_f16))
        NN_FAIL("transpose_gpu: f16 pipeline 不可用（需16位存储 + shaderFloat16）");
    const auto& pipeline =
        f16_io ? pipelines_.get(PipelineId::transpose_f16) : pipelines_.get(PipelineId::transpose);

    const uint32_t R = static_cast<uint32_t>(A.rows());
    const uint32_t C = static_cast<uint32_t>(A.cols());

    // 输出 (C, R)：f16 按 2B/元素分配 + 纯绑定视图（同 matmul_gpu 做法）
    std::optional<GpuTensor> owned;
    if (f16_io)
    {
        auto f16r = GpuTensorF16::create_empty(C, R, *this);
        NN_TRY_CHECK(f16r);
        owned.emplace(f16r->shared_buffer(), f16r->rows(), f16r->cols());
    }
    else
    {
        auto out_res = GpuTensor::create_empty(C, R, *this);
        NN_TRY_CHECK(out_res);
        owned.emplace(std::move(*out_res));
    }
    GpuTensor output = *owned;

    // 分配描述符集
    const VkBuffer bindings[2]{A.buffer().impl(), output.buffer().impl()};
    auto ds_r = alloc_desc_set_for(pipeline.descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    // 获取 command buffer
    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    // 录制
    record_input_barriers(cmd, {A.buffer().impl()});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    const uint32_t push_data[2] = {R, C};
    vkCmdPushConstants(cmd, pipeline.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push_data), push_data);

    // 砖块化 dispatch：(8, 8, n_bricks)——每 WG 覆盖 64×64 输出块，
    // 8×8 个 WG 组成一砖（并发足迹聚集，见 transpose.comp 头注释）。
    // 越界 tile 由 shader 早退（dispatch 固定 8×8，末砖可不满）。
    // ⚠️ x 必须是 8 而非 16：shader 按 8 宽砖解算 tile_c =
    // (bz%bx_count)*8 + gl_WorkGroupID.x，假定 gl_WorkGroupID.x ∈ [0,8)；
    // 派发 16 宽会让 tile_c 跨两砖、越界 tile 只早退一半，行>512 且列>512
    // 时静默只写前 512 行。
    const uint32_t tx = (C + 63u) / 64u;
    const uint32_t ty = (R + 63u) / 64u;
    const uint32_t bx = (tx + 7u) / 8u;
    const uint32_t by = (ty + 7u) / 8u;
    ts_dispatch(cmd, "transpose", 8, 8, bx * by);
    record_output_barrier(cmd, output.buffer().impl());

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    return output;
}

// ══════════════════════════════════════════════════════════════════
// ══════════════════════════════════════════════════════════════════
// im2col_gpu / col2im_gpu — 卷积/池化窗口展开（纯数据搬运，无算法）
//   Push Constants (36 bytes): C,H,W,k,stride,pad,OH,OW,B
//   Bindings: In(0), Out(1)
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<GpuTensor> im2col_gpu(const GpuTensor& x, std::size_t C, std::size_t H,
                                           std::size_t W, std::size_t k, std::size_t stride,
                                           std::size_t pad, std::size_t OH, std::size_t OW,
                                           bool f16_io = false)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (!has(PipelineId::im2col)) NN_FAIL("im2col pipeline not available");
    if (f16_io && !has(PipelineId::im2col_f16)) NN_FAIL("im2col f16 pipeline not available");
    if (C == 0 || H == 0 || W == 0 || k == 0 || stride == 0 || OH == 0 || OW == 0)
        NN_FAIL("im2col_gpu: C/H/W/k/stride/OH/OW must be > 0");
    if (x.rows() != C * H * W) NN_FAIL("im2col_gpu: x must be (C*H*W, B)");
    const VulkanPipeline& pipe =
        f16_io ? pipelines_.get(PipelineId::im2col_f16) : pipelines_.get(PipelineId::im2col);

    const std::size_t B = x.cols();
    const std::size_t rows = C * k * k;
    const std::size_t cols = B * OH * OW;

    // f16 输出按 2B/元素分配 + 纯绑定视图（同 matmul_gpu/elementwise f16_io）
    GpuTensor output;
    if (f16_io)
    {
        auto f16r = GpuTensorF16::create_empty(rows, cols, *this);
        NN_TRY_CHECK(f16r);
        output = GpuTensor(f16r->shared_buffer(), rows, cols);
    }
    else
    {
        auto out_res = GpuTensor::create_empty(rows, cols, *this);
        NN_TRY_CHECK(out_res);
        output = std::move(*out_res);
    }

    const VkBuffer bindings[2]{x.buffer().impl(), output.buffer().impl()};
    auto ds_r = alloc_desc_set_for(pipe.descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    record_input_barriers(cmd, {x.buffer().impl()});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    const uint32_t push_data[9] = {
        static_cast<uint32_t>(C),  static_cast<uint32_t>(H),      static_cast<uint32_t>(W),
        static_cast<uint32_t>(k),  static_cast<uint32_t>(stride), static_cast<uint32_t>(pad),
        static_cast<uint32_t>(OH), static_cast<uint32_t>(OW),     static_cast<uint32_t>(B)};
    vkCmdPushConstants(cmd, pipe.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push_data), push_data);

    const std::size_t total = rows * cols;
    const uint32_t wg_count = static_cast<uint32_t>((total + 255) / 256);
    ts_dispatch(cmd, "im2col", wg_count, 1, 1);
    record_output_barrier(cmd, output.buffer().impl());

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    return output;
}

[[nodiscard]] Result<GpuTensor> col2im_gpu(const GpuTensor& col, std::size_t C, std::size_t H,
                                           std::size_t W, std::size_t k, std::size_t stride,
                                           std::size_t pad, std::size_t OH, std::size_t OW,
                                           bool f16_io = false)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (!has(PipelineId::col2im)) NN_FAIL("col2im pipeline not available");
    if (f16_io && !has(PipelineId::col2im_f16)) NN_FAIL("col2im f16 pipeline not available");
    if (C == 0 || H == 0 || W == 0 || k == 0 || stride == 0 || OH == 0 || OW == 0)
        NN_FAIL("col2im_gpu: C/H/W/k/stride/OH/OW must be > 0");
    const std::size_t P = OH * OW;
    if (col.rows() != C * k * k || col.cols() == 0 || col.cols() % P != 0)
        NN_FAIL("col2im_gpu: col must be (C*k*k, B*OH*OW)");
    const VulkanPipeline& pipe =
        f16_io ? pipelines_.get(PipelineId::col2im_f16) : pipelines_.get(PipelineId::col2im);

    const std::size_t B = col.cols() / P;
    const std::size_t rows = C * H * W;
    const std::size_t cols = B;

    GpuTensor output;
    if (f16_io)
    {
        auto f16r = GpuTensorF16::create_empty(rows, cols, *this);
        NN_TRY_CHECK(f16r);
        output = GpuTensor(f16r->shared_buffer(), rows, cols);
    }
    else
    {
        auto out_res = GpuTensor::create_empty(rows, cols, *this);
        NN_TRY_CHECK(out_res);
        output = std::move(*out_res);
    }

    const VkBuffer bindings[2]{col.buffer().impl(), output.buffer().impl()};
    auto ds_r = alloc_desc_set_for(pipe.descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    record_input_barriers(cmd, {col.buffer().impl()});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    const uint32_t push_data[9] = {
        static_cast<uint32_t>(C),  static_cast<uint32_t>(H),      static_cast<uint32_t>(W),
        static_cast<uint32_t>(k),  static_cast<uint32_t>(stride), static_cast<uint32_t>(pad),
        static_cast<uint32_t>(OH), static_cast<uint32_t>(OW),     static_cast<uint32_t>(B)};
    vkCmdPushConstants(cmd, pipe.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push_data), push_data);

    const std::size_t total = rows * cols;
    const uint32_t wg_count = static_cast<uint32_t>((total + 255) / 256);
    ts_dispatch(cmd, "col2im", wg_count, 1, 1);
    record_output_barrier(cmd, output.buffer().impl());

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    return output;
}

// ══════════════════════════════════════════════════════════════════
// grouped_reduce_gpu — 分组归约（沿行方向按固定长度 R 分组求和/求最大）
//   x (G*R, N) → out (G, N)
//   Push Constants (16 bytes): G, R, N, reduce_op
//   Bindings: In(0), Out(1)；Dispatch: ceil(G*N / 256)
//   f16_in（Phase C2c）：x 按 half 读、输出恒 f32（CNN 调用 P=F32）；
//   pipeline 未创建时硬报错，调用方提前查 has(PipelineId::group_reduce_f16)
//   走引擎内 cast 回退。
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<GpuTensor> grouped_reduce_gpu(const GpuTensor& x, std::size_t G, std::size_t R,
                                                   bool is_max, bool f16_in = false)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (!has(PipelineId::group_reduce)) NN_FAIL("group_reduce pipeline not available");
    if (f16_in && !has(PipelineId::group_reduce_f16))
        NN_FAIL("group_reduce f16 pipeline not available");
    if (G == 0 || R == 0) NN_FAIL("grouped_reduce_gpu: G/R must be > 0");
    if (x.rows() != G * R) NN_FAIL("grouped_reduce_gpu: x must be (G*R, N)");
    const std::size_t N = x.cols();
    if (N == 0) NN_FAIL("grouped_reduce_gpu: N must be > 0");
    const VulkanPipeline& pipe = f16_in ? pipelines_.get(PipelineId::group_reduce_f16)
                                        : pipelines_.get(PipelineId::group_reduce);

    auto out_res = GpuTensor::create_empty(G, N, *this);
    NN_TRY_CHECK(out_res);
    GpuTensor output = std::move(*out_res);

    const VkBuffer bindings[2]{x.buffer().impl(), output.buffer().impl()};
    auto ds_r = alloc_desc_set_for(pipe.descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    record_input_barriers(cmd, {x.buffer().impl()});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    const uint32_t push_data[4] = {static_cast<uint32_t>(G), static_cast<uint32_t>(R),
                                   static_cast<uint32_t>(N), is_max ? 1u : 0u};
    vkCmdPushConstants(cmd, pipe.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push_data), push_data);

    const std::size_t total = G * N;
    const uint32_t wg_count = static_cast<uint32_t>((total + 255) / 256);
    ts_dispatch(cmd, "group_reduce", wg_count, 1, 1);
    record_output_barrier(cmd, output.buffer().impl());

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    return output;
}

// gather_gpu — 按行索引查表 (GPU-native)
//
// table: (vocab, D), indices: (num,) → output: (num, D)
// output[i] = table[indices[i]]，越界索引返回零行
//
// Push Constants (12 bytes): vocab, D, num
// Bindings: Table(0), Indices(1), Output(2)
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<GpuTensor> gather_gpu(const GpuTensor& table, const GpuTensor& indices,
                                           bool f16_io = false)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (!has(PipelineId::gather)) NN_FAIL("gather pipeline not available");
    // Phase C1 f16 变体：table/out f16；**indices 恒 f32**（行号整数值
    // 精度要求，shader 内 float indices 槽不变）
    if (f16_io && !has(PipelineId::gather_f16))
        NN_FAIL("gather_gpu: f16 pipeline 不可用（设备无16位存储）");
    const auto& pipeline =
        f16_io ? pipelines_.get(PipelineId::gather_f16) : pipelines_.get(PipelineId::gather);

    const uint32_t vocab = static_cast<uint32_t>(table.rows());
    const uint32_t D = static_cast<uint32_t>(table.cols());
    const uint32_t num = static_cast<uint32_t>(indices.rows() * indices.cols());

    // 输出 (num, D)（f16：2B/元素 + 纯绑定视图，同 matmul_gpu）
    std::optional<GpuTensor> owned;
    if (f16_io)
    {
        auto f16r = GpuTensorF16::create_empty(num, D, *this);
        NN_TRY_CHECK(f16r);
        owned.emplace(f16r->shared_buffer(), f16r->rows(), f16r->cols());
    }
    else
    {
        auto out_res = GpuTensor::create_empty(num, D, *this);
        NN_TRY_CHECK(out_res);
        owned.emplace(std::move(*out_res));
    }
    GpuTensor output = *owned;

    // 分配描述符集
    const VkBuffer bindings[3]{table.buffer().impl(), indices.buffer().impl(),
                               output.buffer().impl()};
    auto ds_r = alloc_desc_set_for(pipeline.descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    // 获取 command buffer
    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    // 录制
    record_input_barriers(cmd, {table.buffer().impl(), indices.buffer().impl()});
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    const uint32_t push_data[3] = {vocab, D, num};
    vkCmdPushConstants(cmd, pipeline.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push_data), push_data);

    const uint32_t total = num * D;
    // vec4 kernel（与 elementwise/broadcast/DSL 同口径）：每线程 4 元素
    const uint32_t wg_count = (total + 256u * 4u - 1u) / (256u * 4u);
    ts_dispatch(cmd, "gather", wg_count, 1, 1);
    record_output_barrier(cmd, output.buffer().impl());

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    return output;
}

// ══════════════════════════════════════════════════════════════════
// scatter_add_gpu — 按行索引原子累加梯度 (GPU-native)
//
// dst: (vocab, D) 原地修改，indices: (num,), grad: (num, D)
// dst[indices[i]][d] += grad[i][d]
// 使用 CAS 循环实现 float atomicAdd
//
// Push Constants (12 bytes): vocab, D, num
// Bindings: Dst(0), Indices(1), Grad(2)
//
// 注意：dst buffer 内容以 uint 视角访问用于 atomicCompSwap。
// dst 初始值必须为合法 float 位模式（如 0x00000000 = 0.0f）。
// f16_io（Phase C2b）：dst/grad 为 f16 存储——f16 shader 里 dst 每 uint
// word 打包 2 个 half（CAS 覆盖整 word）、grad half 直读；indices 恒
// float（行号 >2048 禁 f16）。pipeline 未创建时硬报错，调用方提前查
// has(PipelineId::scatter_add_f16) 走引擎内 cast 回退。
// ══════════════════════════════════════════════════════════════════
[[nodiscard]] Result<void> scatter_add_gpu(GpuTensor& dst, const GpuTensor& indices,
                                           const GpuTensor& grad, bool f16_io = false)
{
    if (!initialized_) NN_FAIL("GPU backend not initialized");
    if (!has(PipelineId::scatter_add)) NN_FAIL("scatter_add pipeline not available");
    const VulkanPipeline& pipe = f16_io ? pipelines_.get(PipelineId::scatter_add_f16)
                                        : pipelines_.get(PipelineId::scatter_add);
    if (f16_io && !has(PipelineId::scatter_add_f16))
        NN_FAIL("scatter_add f16 pipeline not available");
    const uint32_t vocab = static_cast<uint32_t>(dst.rows());
    const uint32_t D = static_cast<uint32_t>(dst.cols());
    const uint32_t num = static_cast<uint32_t>(indices.rows() * indices.cols());

    // 分配描述符集
    // dst 使用 uint 视图用于原子操作
    const VkBuffer bindings[3]{dst.buffer().impl(), indices.buffer().impl(), grad.buffer().impl()};
    auto ds_r = alloc_desc_set_for(pipe.descriptor_layout(), bindings);
    NN_TRY_CHECK(ds_r);
    VkDescriptorSet desc_set = *ds_r;

    // 获取 command buffer
    auto cmd_r = acquire_cmd();
    if (!cmd_r)
    {
        release_desc_set_now(desc_set);
        NN_TRY_CHECK(cmd_r);
    }
    auto [cmd, owns_cmd] = *cmd_r;

    // 录制（dst 需要读+写屏障）
    {
        std::vector<VkBufferMemoryBarrier> barriers;
        VkBufferMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        b.buffer = dst.buffer().impl();
        barriers.push_back(b);
        b.buffer = indices.buffer().impl();
        barriers.push_back(b);
        b.buffer = grad.buffer().impl();
        barriers.push_back(b);
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                             static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline_layout(), 0, 1,
                            &desc_set, 0, nullptr);

    const uint32_t push_data[3] = {vocab, D, num};
    vkCmdPushConstants(cmd, pipe.pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(push_data), push_data);

    const uint32_t total = num * D;
    const uint32_t wg_count = (total + 255) / 256;
    ts_dispatch(cmd, "scatter_add", wg_count, 1, 1);

    // 输出屏障（dst 被修改，需要 memory barrier）
    {
        VkBufferMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = dst.buffer().impl();
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &b, 0, nullptr);
    }

    if (owns_cmd)
    {
        auto r = submit_and_wait(cmd, desc_set);
        NN_TRY_CHECK(r);
    }
    return {};
}
