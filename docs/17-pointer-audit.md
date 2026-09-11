# 17 — 指针审查报告（改造难度 × 价值标注）

> 2026-08-30。工具：`scripts/audit_modern_style.py`（W1 零误报版）+ grep 盲补。
> 扫描范围：`include/` + `src/` + `tools/`，**100 个文件**。
> 基线：**ERROR 50 / WARN 52 / nn-allow 豁免 0**（豁免机制已建但从未使用）。
> **2026-08-29 第 1 轮执行完成**（约束：不用 nn-allow——能解的解，解不了的留基线）。
> 最新状态、新基线与审查器缺陷修复结果见 **§9**。
>
> 难度：低 = 机械改动/单点，中 = 需回归验证（全量 ctest 30/30），高 = 跨模块设计改动。
> 价值：按"消除铁律违反 / 消除死代码地雷 / CI 门禁可用性 / 纯形式"四档。

## 0. 一句话结论

项目指针使用**健康度良好**：无 delete、无 malloc、无泄漏（3 处 `new` 全部被
`unique_ptr` 或"故意泄漏"管理）。真正值得动的只有两类：
① **E4/E7 共 47 处是合法的二进制 I/O 与 C/Vulkan 互操作边界**，用 `nn-allow` 关账即可，
让审查器 FAIL→PASS、可挂 CI（最大性价比）；
② **emitter 工厂返回裸指针（3 处）**改成 `unique_ptr` 工厂，彻底消灭 `include/` 里的 `new`。
其余 WARN 多为审查器误报或平台 ABI 边界，修脚本比改代码更值。

## 1. P1 — new 操作符（E1 ×3）：唯一涉"所有权"的指针

| # | 位置 | 性质 | 改造方案 | 难度 | 价值 |
|---|------|------|----------|:---:|:---:|
| 1.1 | `expr_emitter.hpp:63` `Factory = ExprEmitter* (*)()` + `expr_glsl_gen.hpp:1313` + `expr_cpu_emitter.hpp:256` | 工厂返回裸指针；`make()`（`expr_emitter.hpp:86`）立即包 `unique_ptr`，**无泄漏** | `Factory` 改 `std::unique_ptr<ExprEmitter>(*)()`，两个 lambda 改 `return std::make_unique<…>()`。3 行改动，无调用方影响（`gen_fused.cpp:201` / `expr_graph_test.cpp:598-600` 都经 `make()`） | 低 | 中（消灭 `include/` 全部 `new`，铁律形式合规） |
| 1.2 | `compute_vk_backend.hpp:531` `static GpuBackend* backend = new GpuBackend();` | **故意泄漏**单例：注释已写明——全局 GpuTensor 析构可能晚于 backend，C++11 静态析构顺序问题；1.1 TODO 是显式 shutdown()+引用计数 | ① 保留 + `nn-allow: 故意泄漏，静态析构顺序（见 1.1 TODO）`（推荐）；② 1.1 再做 shutdown 重构 | 保留:低 / 重构:高 | 低（语义正确且有文档，关账即可；重构属 1.1 范围） |
| 1.3 | `expr_cpu_emitter.hpp` 全文件 | 死代码地雷（AGENTS.md §12：产物从不编译、缺陷全隐性）；且 `expr_graph_test.cpp:599,602` 测试 12 依赖 "cpu" 后端可创建 | ① 删注册行+测试改断言（推荐，M0 既定）；② 保留文件仅删死路径需逐条修 CpuEmitter 隐性缺陷（不推荐） | 低 | 中（E1×1 + 死代码地雷一次清掉） |

## 2. P2 — 二进制 I/O / 互操作边界（E4 ×39 + E7 ×8）：全部合法，建议关账

### 2.1 reinterpret_cast 字节 I/O（E4 ×39）

| 位置 | 处数 | 上下文 | 改造方案 | 难度 | 价值 |
|------|:---:|--------|----------|:---:|:---:|
| `expr_registry.hpp:78-170` | 28 | `expr_specs.bin` POD 读写（f.write/read + cast） | `nn-allow: 二进制文件 I/O 边界`（逐行）；或抽 `write_pod/read_pod` 助手收敛 cast（收益低，不推荐） | 低 | 中（ERROR 清零的主要部分） |
| `model_serialization.hpp:90,103,172,197` | 4 | v4 自描述格式字节流 | 同上 | 低 | 中 |
| `text_train.cpp:204-245` | 6 | 语料 .bin（hdr/flow/doc_ids） | 同上 | 低 | 中 |
| `tools/gen_fused.cpp:63` | 1 | 读 expr_specs.bin | 同上 | 低 | 中 |

### 2.2 void*（E7 ×8）

| 位置 | 性质 | 方案 | 难度 | 价值 |
|------|------|------|:---:|:---:|
| `compute_cuda_backend.hpp:69,75,109,118,259,261` | 整个文件 `#ifdef NN_HAS_CUDA`，**CUDA 后端 1.0.0 已停用**（AGENTS.md：勿依赖） | `nn-allow: 已停用 CUDA 后端，互操作边界`；或审查器整文件排除 | 低 | 低（死代码，清账即可，不值得改实现） |
| `compute_staging_ring.hpp:58` `void* mapped_ptr` | `vkMapMemory` 返回值，Vulkan 标准习语 | `nn-allow: vkMapMemory 互操作` | 低 | 低 |
| `expr_spec.hpp:359` `const void* p`（feed lambda） | FNV-1a 哈希喂字节；:361 的 `static_cast<const uint8_t*>`（W1）随之 | 改参数为 `const uint8_t*` 会把 cast 挪到 7 个调用点，**不减少** → `nn-allow: 哈希字节边界` 一行同时关 E7+W1 | 低 | 低 |

> P2 合计 47 行 `nn-allow` = 机械工作，预期 ERROR 50→0（1.1/1.2 另计），
> 审查器 `PASS`，可挂 `.github` CI（`--fail-on-warn` 暂不开）。**这是本清单性价比最高的一项。**

## 3. P3 — 非拥有视图裸指针（W1 ×4 + 审查器盲补 ×10）

铁律第 2 条禁的是裸指针**所有权**；以下全是"指向活容器内部的视图"，功能上无问题，
改造价值 = 形式合规 + 可读性，按性价比排序：

| # | 位置 | 现状 | 方案 | 难度 | 价值 |
|---|------|------|------|:---:|:---:|
| 3.1 | `algebra_matrix.hpp:886,912,915,930,952,977`（6 处，reduce/broadcast 并行热循环） | `const Scalar* row = self.data()+r*C`、`auto* acc = local_acc.data()+t*C` | `Scalar& base = local_acc[t*C]` + `base[c]`（引用替代指针算术，零成本）；`row` 同理用 `const Scalar&`。注意 912/930 在 lambda 内捕获，引用捕获语义需逐处核对 | **中**（热路径+并行，需 gradcheck 全量回归） | 低-中 |
| 3.2 | `algebra_span.hpp:30,88`（Span/ConstSpan 的 `data_` 成员） | 手写的 L1 视图容器，形状与 `std::span` 完全一致（前 std::span 普及产物） | 成员改存 `std::span`/`std::span<const Scalar>`（`data()` 转发，接口不变）。影响面 = 整个表达式模板路径，需全量 ctest | 中 | 低（行为完全等价，纯内部形态统一；风险>收益，可延后） |
| 3.3 | `core_errors.hpp:71`、`cli/cli_mnist_io.hpp:69` | `const auto* end = s.data()+s.size();` 喂 `std::from_chars` | ~~`std::string::end()` 即指向 '\0 的迭代器/指针~~ **原方案有误**：libc++ 的 `std::string::end()` 返回 `__wrap_iter` 迭代器，不能当 `const char*` 喂 `from_chars`（编译失败），仅 `std::string_view::end()` 适用。实际执行：string_view 处改 `s.end()`；std::string 处保留 `data()+size()` 并加注释（见 §9.1） | 低 | 低 |
| 3.4 | `core_errors.hpp:59` `char* p` | `strtoX` endptr，C 互操作（注释已说明不用 from_chars 的原因） | `nn-allow: C 互操作 endptr` | 低 | 低 |
| 3.5 | `text_train.cpp:68` `const char* buf = fc.buffer.data()` | 非拥有视图喂 fread/fwrite | 保持（非审查项）或改 `const char* buf = data` 内联 | 低 | 低 |
| 3.6 | `layer_bench.cpp:365`、`mnist_train.cpp:517` `const char* x = 字符串字面量` | 字面量天然是指针 | 改 `std::string_view`（可选） | 低 | 低（纯美观） |
| 3.7 | `expr_spec.hpp:361` | 与 2.2 同行，nn-allow 一并关闭 | 同 2.2 | 低 | 低 |

## 4. P4 — C 数组参数（W2 ×40）：主要是审查器噪音

| 组 | 处数 | 位置 | 判定 | 方案 | 难度 | 价值 |
|----|:---:|------|------|------|:---:|:---:|
| 4.1 | 22 | 各 `int main(int argc, char* argv[])`（19 个可执行入口 + tools 2 个） | 平台 ABI，**不可改** | 审查器豁免 `main` 签名中的 `char* argv` | 低 | 低（去噪 22） |
| 4.2 | 7 | `parse_args(int argc, char *argv[])`（mnist/text/tokenizer train+infer、cli_train_common:67） | CLI 边界，惯例保留 argv | 同 4.1 豁免（`argv` 参数名匹配） | 低 | 低（去噪 7） |
| 4.3 | 11 | 测试字符串表 `const char* tags[]/pnames[]/fixed[]/cont[]/pre[]/post[]`（attn_gradcheck:181, gpt_gradcheck:250, swiglu_gradcheck:197, tokenizer_consistency_test:113,164-166 等） | 真旧式，但都在测试/工具代码 | 改 `constexpr std::array<std::string_view, N>` | 低 | 低 |
| 4.4 | 4 | `domain_tokenizer_base.hpp:92,161,189`、`domain_tokenizer_bpe.hpp:386` `static constexpr std::string_view xxx[] = {…}` | **已是现代形式**（string_view！），`RE_ARR_PARAM` 误匹配数组声明语法 | 修审查器：W2 仅限函数参数上下文 | 低 | 低（去噪 4，纯误报） |

> 4.1-4.4 全做完，W2 从 40 → 11（4.3 改代码后归零，实际 → 0）。

## 5. P5 — C 风格函数指针（W3 ×8）

| # | 位置 | 判定 | 方案 | 难度 | 价值 |
|---|------|------|------|:---:|:---:|
| 5.1 | `layer_bench.cpp:94,96,98,100,247,249,251`（LayerSpec/OpSpec 函数指针表） | 真旧式，bench 工具代码 | 改 `std::function<…>` 成员（无热路径顾虑） | 低-中 | 低 |
| 5.2 | `core_threadpool.hpp:88` `tasks_.emplace([task]() { (*task)(); });` | **误报**：lambda 调用 `(*task)()`，`RE_FUNCPTR` 过宽 | 修 `RE_FUNCPTR`（要求 `(` 前为类型名而非 `*` 解引用上下文）或 `nn-allow` | 低 | 低（去噪 1） |

## 6. P6 — 审查器自身缺陷（过程项，影响后续基线可信度）

| # | 盲点 | 证据 | 修法 | 难度 | 价值 |
|---|------|------|------|:---:|:---:|
| 6.1 | W1 漏报"内置类型 *变量 = …"声明：左上下文以类型名结尾不命中 `_has_decl_ctx` | `core_errors.hpp:59 char* p`、`algebra_matrix.hpp:886/915/952/977 const Scalar* row` 均未进基线 | `_has_decl_ctx` 增加 `_BUILTIN_TYPES` 词尾匹配 | 低 | 中（否则基线系统性偏低，CI 门禁不可信） |
| 6.2 | W1 漏报类成员指针声明（行首有缩进，`^` 锚失效） | `algebra_span.hpp:30,88` | 左上下文 strip 后再匹配 | 低 | 中 |
| 6.3 | W2 误报 constexpr 数组初始化（非参数） | 4.4 | W2 加"必须处于参数列表内"约束 | 低 | 低 |
| 6.4 | W3 误报 `(*x)()` 解引用调用 | 5.2 | 见 5.2 | 低 | 低 |

## 7. 执行建议（按价值/难度比排序）

| 序 | 事项 | 改动量 | 效果 |
|:---:|------|--------|------|
| ① | P2 全部 `nn-allow`（47 行，含原因说明） | ~0.5h | ERROR 50→3，审查器接近可用 |
| ② | 1.1 Factory 改 unique_ptr 工厂（3 处） | ~10min | E1 3→1，`include/` 清零 `new` |
| ③ | 1.3 删 CpuEmitter 注册 + 改 expr_graph_test 测试 12 断言（需拍板） | ~20min | 死代码地雷出清（M0） |
| ④ | P6 审查器修 4 处（6.1-6.4）+ 重跑基线 | ~1h | 基线可信，WARN 大幅去噪 |
| ⑤ | 1.2 vk 单例 `nn-allow` + 3.3/3.4 小改 | ~15min | ERROR 3→0，**审查器 PASS，可挂 CI** |
| ⑥ | P4 字符串表 11 处 + P5 函数指针表 7 处（测试/工具代码，可择机） | ~1h | WARN → 0（或全部留 nn-allow） |
| ⑦ | 3.1/3.2 热循环与 Span 容器（需全量 ctest 回归） | ~半天 | 形式合规，无行为收益，**建议最后做或不做** |

全部做完（含⑥⑦）：ERROR 0、WARN 0、豁免 ~55（每行都有理由），审查器成为可挂 CI 的
真门禁。①-⑤ 即可达到"CI 门禁可用"，⑥⑦ 属锦上添花。

## 8. 明确不改的（及理由）

- **vk 单例的 `new`**（1.2）：故意泄漏有明确技术理由（静态析构顺序），`nn-allow` 关账；
  显式 shutdown 属 1.1 路线图，不在本次范围。
- **CUDA 后端 6 处 void***：已停用、仅作恢复参考，改实现是负收益。
- **`std::from_chars`/`strtoX` 的 C 指针参数**：标准库 API 设计如此，属互操作边界。
- **`memcpy(pc.data()+off, &x, n)` 家族**（`compute_vk_backend.hpp:1999-2034` 等）：
  `void*` 是 memcpy 的标准签名，非裸指针使用。

## 9. 2026-08-29 第 1 轮执行状态与最新基线

> 执行约束：**不使用 nn-allow**——能解的解掉，解不了的留基线。
> 与 §7 的差异：①（47 行 nn-allow）与 ⑤（1.2 的 nn-allow）**未做**，E 类留基线；
> 其余（②③④⑥⑦）全部完成，其中 ⑦ 只做 3.1、不做 3.2（风险>收益）。

### 9.1 已完成（代码均已过全量构建 + ctest 30/30）

| 项 | 实际做法 | 与原计划差异 |
|----|----------|--------------|
| 1.1 | `Factory` 改 `std::unique_ptr<ExprEmitter>(*)()`，注册 lambda 改 `std::make_unique` | 无 |
| 1.3 | 删 `expr_cpu_emitter.hpp`（文件+注册行），`expr_graph_test` 12 断言改为对 "cpu" 后端的硬报错断言 | 无 |
| 2.1（P2-E4） | **收敛**：`core_file.hpp` 新增 `write_pod/read_pod/write_pod_span/read_pod_span` 四个 POD 读写原语，4 文件 39 处 cast 全部改走它们 | 原计划逐行 nn-allow；现全库 `reinterpret_cast` 仅剩收敛点自身 4 处（`core_file.hpp:52,63,73,83`，文件头已注明"唯一收敛点"） |
| 3.1 | 6 处热循环行起点改 `std::span::subspan(r*C, C)`（reduce/broadcast 并行路径，`algebra_matrix.hpp:887,914,917,932,954,979`） | 原计划 `Scalar& base` 引用；span 更贴合铁律 2，同为 ptr+len 零成本 |
| 3.3 | `core_errors.hpp`（string_view）改 `s.end()`；`cli_mnist_io.hpp`（std::string）保留 `data()+size()` 并加注释 | 见 3.3 行的方案修正（libc++ `__wrap_iter` 不能喂 from_chars） |
| 3.6 | `layer_bench.cpp:366`、`mnist_train.cpp:518` 字面量改 `const std::string_view` | 无 |
| 4.3 | 11 处测试/工具字符串表改 `constexpr std::array<std::string_view, N>`（attn/gpt/swiglu gradcheck + tokenizer 4 张表 77 元素） | 无 |
| 5.1 | `layer_bench.cpp` 7 处函数指针成员改 `std::function` | 无 |
| 6.1/6.2 | `_has_decl_ctx` 补"左上下文以类型名（`_BUILTIN_TYPES` 或 `a::B` 限定标识符）词尾结束"判定；strip 后取最后 token，`::` 形式要求 `fullmatch`（防 `std::to_string(x * 2)` 乘法误报） | 无 |
| 6.3 | `RE_ARR_PARAM` 结尾字符类 `[,)=]` → `[),]`（`{`/`=` 是初始化非参数） | 无 |
| 6.4 | `RE_FUNCPTR` 命中后要求 `(` 前一 token 为类型（`_BUILTIN_TYPES`/`std`/`_is_type_token`），`(*task)()` 解引用调用不再误报 | 无 |
| argv 去噪 | `RE_ARR_PARAM` 命中且数组名为 `argv` 时豁免（main/parse_args 的 CLI 边界，现存 14 处 `char* argv[]`）；参数语境的 `char* argv` 同时被 `_known_ptr_params` 排除出 W1 | 审查器侧豁免实现 §4.1/4.2，**未用 nn-allow** |

**6.1-6.4 回测**（合成用例 + 实际基线双重验证）：`char* p`、`const Scalar* row`、
成员 `Scalar* data_`（`algebra_span.hpp:30,88`）均进基线；`(*task)()`、`char* argv[]`、
`constexpr std::string_view x[]` 不再误报；库内无 `qualified * qualified` 乘法模式，
6.1/6.2 新增 `::` 分支零误报。

### 9.2 最新基线（2026-08-29，`python3 scripts/audit_modern_style.py`）

**99 个文件：ERROR 13 / WARN 37（全 W1）/ W2 0 / W3 0 / 豁免 0**

| 类 | 数 | 构成 | 处置 |
|----|:--:|------|------|
| E1 | 1 | vk 单例故意泄漏（1.2，注释+§8 有技术理由） | 留基线（shutdown 重构属 1.1 范围） |
| E4 | 4 | `core_file.hpp:52,63,73,83` 收敛点自身 | 留基线（2.1 的"收敛而非消灭"已是终态） |
| E7 | 8 | CUDA 6（已停用）+ `compute_staging_ring.hpp:58`（vkMapMemory）+ `expr_spec.hpp:359`（FNV 哈希边界） | 留基线（改实现为负收益） |
| W1 | 37 | P3 剩余：`algebra_span.hpp:30,88`（3.2）、`core_errors.hpp:59`（3.4）、`text_train.cpp:47,68`（3.5）、`expr_spec.hpp:361`（3.7）、`expr_glsl_gen.hpp` 19 处 `const char* s`（GLSL 表）、`compute_gpu_engine.hpp:715,775,881`（FusedShader observer）、`cli_mnist_io.hpp:71,166`、`compute_vk_device.hpp:41,51`（__restrict 参数）、CUDA 3 处 | 择机改 span/引用或留基线 |
| W2/W3 | 0 | — | 6.3/6.4 + argv 豁免 + 4.3/5.1 改码后清零 |

### 9.3 未做（留基线，及理由）

- **① 47 行 nn-allow / ⑤ 1.2 的 nn-allow**：按执行约束未做。审查器当前 `FAIL`
  （13 ERROR，全部为 §8/§9.2 所列合法边界），**暂不挂 CI 硬门禁**；日常软检查
  （ERROR 计数不回升、新增裸指针进 W1 基线）。
- **3.2** `algebra_span.hpp` 成员改 `std::span`：风险>收益（§3 原判定），未动。
- **1.2** 单例 shutdown 重构：属 1.1 路线图，不在本次范围。

