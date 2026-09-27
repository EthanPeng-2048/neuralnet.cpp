# w2-foundation

## 线程池删除通用单任务 submit（原位置 include/neuralnet.cpp/core_threadpool.hpp:77）
- 类型：删除清单
- 内容：通用单任务提交（submit）已删除：全库无调用方（审查 P1-3），且每次调用 make_shared<packaged_task> 堆分配，违背本池"零分配 latch"设计。需要 future 语义时应在调用方分块后用 parallel_* 系列原语。

## 线程池 wait_for_latch 等待策略演进（原位置 include/neuralnet.cpp/core_threadpool.hpp:118）
- 类型：性能 A/B
- 内容：优化依据为性能审查报告。旧实现 spin 64 次 + yield，64 次 spin 中反复原子读取，CPU 占用率显示 100% 但实际有效计算比例低（调用者空转）。新实现为三阶段：1) 短自旋 16 次 pause；2) 自旋失败后 work-steal 从队列取任务执行；3) 队列为空阻塞 cv 等待，由 finish_chunk 唤醒。

## 线程池归约分段依赖 worker 数导致字节不一致（原位置 include/neuralnet.cpp/core_threadpool.hpp:541）
- 类型：bug 根因
- 内容：旧实现 n_chunks = chunk_count(total) 依赖 workers_.size()，同一输入在 1-worker 与 N-worker 下走不同折叠结构 → 浮点非结合律导致字节不一致（跨机也不一致）。修复后边界只由 total 决定，1-worker/N-worker、任何机器走完全相同分段。

## 归约 init 重复累加（原位置 include/neuralnet.cpp/core_threadpool.hpp:555）
- 类型：bug 根因
- 内容：旧实现每块都加 init、合并时再加一次，init≠0 时数学错误。现行约定：块 0 以 init 为种子、块 c>0 以块内首元素为种子，init 恰好计入一次。

## 线程池其它阈值/引用调整（原位置 include/neuralnet.cpp/core_threadpool.hpp:85、:106）
- 类型：删除清单
- 内容：MIN_CHUNK 阈值曾从 4096 降至 1024（使 MNIST 小隐藏层 64×batch 也能触发多核并行）；lost-wakeup 压测引用的 build/perfprobe/probe_pool2.cpp 已不存在（压测数据本身保留为持锁通知的实证）。

## GEMM 微内核 vs 旧标量内核 A/B（原位置 include/neuralnet.cpp/algebra_matrix.hpp:278-288）
- 类型：性能 A/B
- 内容：旧内核每个 (i,j) 一个标量累加器、每次 FMA 2 次 load（受 load port 限制），编译日志显示内层循环未被向量化（-Wpass-failed=transform-warning）。微内核（4×8 AVX2 列块）实测 GFLOPS（3072x768x512 / 768x3072x512 / 1024^3 / 512^3）：140→401 / 155→407 / 163→354 / 82→281；与旧内核逐位一致（对拍 max_abs_diff = 0）。

## row_reduce 并行门控曾恒定串行（原位置 include/neuralnet.cpp/algebra_matrix.hpp:744）
- 类型：bug 根因
- 内容：旧实现用 nn::for_each(row_indices)，把"行数"当元素数与 PARALLEL_THRESHOLD 比较 → 行数永远达不到 512K → 恒定串行（原 docs/development/11 §R3，该诊断文档已删除，结论并入 12-compute-engine-inventory/03 §5.3）。改为按元素数（R*C）门控 + parallel_for_samples 按行分片；col_reduce 行数门槛由 1024 降至 256。

## MatrixT F32 零回归验证（原位置 include/neuralnet.cpp/algebra_matrix.hpp:44）
- 类型：性能 A/B
- 内容：模板化改造时 F32 实例与旧非模板 Matrix 逐字节一致（测试项 T1 零回归）；f32 便捷别名保持 Matrix 以保证既有代码零改动。

## 旧代数 AST 移除（原位置 include/neuralnet.cpp/algebra_span.hpp:8、nn.hpp:8）
- 类型：删除清单
- 内容：旧代数 AST（自由运算符 + compute::apply 入口）已随逐元素算子移除；algebra_expr.hpp / algebra_compute.hpp 删除，Expression/BoolExpression 概念迁入 expr_dsl.hpp。Span 不再有"运算符构建 AST"的旧路径；ConstSpan 原注释引用的 expr.hpp 自由函数模板（operator+/-/* 等基于 Expression 概念统一处理，使 ConstSpan 与 Span/Val 自然组合）已不存在，现 DSL 运算符只接受 DslExpr 可折叠节点。

## 序列化审查待办 S3/M1/M2（原位置 include/neuralnet.cpp/model_serialization.hpp:54）
- 类型：bug 根因 / 删除清单
- 内容：S3（已修复）：read_spec_header / read_tokenizer 曾无长度上限，现均经 kMaxSerializedStringBytes（64 MiB）校验后才预分配，损坏/恶意文件返回 Error 而非 bad_alloc → terminate。M1（未修）：.bin 全文件无校验和（.nnpkg 有 sha256），内容损坏会被静默载入错误权重且无感知，曾建议升 MODEL_VERSION 加整文件校验和与尾部完整性标记。M2（审查记录）：extra_state 的注释称"旧文件读到 EOF 保持默认（running_mean=0 等）"但实现直接返回错误，注释与实现不符，需统一为按版本回退默认值。

## v1/v2/v3 偏移量格式移除支持（原位置 include/neuralnet.cpp/model_serialization.hpp:46、:430）
- 类型：删除清单
- 内容：v1/v2/v3 为旧的偏移量定长格式，已移除支持（无有意义的旧模型）；现仅接受自描述格式 v4+，旧格式在 read_and_validate_header 拒绝。Tokenizer JSON 读写曾标注"V3 新增"（版本流水，随 v1-v3 拒绝一并失去意义）。

## 失效文档链接清理（原位置 core_file.hpp:13/:42、model_serialization.hpp:24/:82、algebra_matrix.hpp:41/:43/:746、core_config.hpp:47）
- 类型：删除清单
- 内容：docs/17-pointer-audit.md、docs/23-mixed-precision.md、docs/development/11-cpu-performance-diagnosis.md、DEVELOPMENT_STANDARDS.md、src/bench_thresholds.cpp 均已删除或改名；对应链接改指 docs/development/10-development-standards.md、docs/development/05-mixed-precision.md，或去掉文件名只留实测结论（blocked 优于 naive、并行阈值 524288 标定表等实测数据保留）。
