# 🕳️ 踩坑警示录 (Pitfalls & Lessons)

> **定位**:本文档汇总项目开发至今踩过的所有坑,按严重度分级,提炼跨领域的根因模式,
> 并提供开发活动的防坑清单。**每次踩新坑,先补充到本文档,再修复代码。**
>
> 更新频率:随开发持续追加。最近更新:2026-08-19

---

## 📋 目录

1. [严重度分级总览](#-严重度分级总览)
2. [🔴 灾难级:正确性/设备死亡](#-灾难级正确性设备死亡)
3. [🟠 高危:内存爆炸/崩溃/断言](#-高危内存爆炸崩溃断言)
4. [🟡 中危:结果偏差/性能/确定性](#-中危结果偏差性能确定性)
5. [🟢 工具链与 UI 坑](#-工具链与-ui-坑)
6. [根因模式(跨领域总结)](#-根因模式跨领域总结)
7. [开发活动防坑清单](#-开发活动防坑清单)
8. [相关文档](#-相关文档)

---

## 🚦 严重度分级总览

| 级别 | 数量 | 特征 | 典型案例 |
|------|------|------|----------|
| 🔴 灾难级 | 4 | 数据损坏 / 设备永久失效 / 训练白费数小时 | GPT 布局混用、GPU-resident 10% 正确率 |
| 🟠 高危 | 6 | 内存爆炸 / 崩溃 / 断言失败 | BPE 60-80GB、reshape 视图断言 |
| 🟡 中危 | 11 | 结果偏差 / 性能退化 / 并行不确定 | BPE 迭代序泄漏、one-hot 3.2GB、scatter_add 非确定性 |
| 🟢 低危 | 15+ | 构建失败 / UI 异常 / 兼容性 | CRLF、MSVC 编译器版本、tkinter 保留属性 |

**一句话预警**:本项目最大的两类风险是 **① 布局/索引约定不一致** 和 **② GPU 资源生命周期**,
历史上所有灾难级 bug 都出自这两类。改代码前先读第 6 节的根因模式。

---

## 🔴 灾难级:正确性/设备死亡

### 1. GPT 展平列序混用:position-major vs batch-major(2026-08-17)

- **症状**:GPT 训练 loss 降到某平台后停滞,浅层梯度冻结;MNIST/ViT 正常。
- **根因**:`GPTModel` 展平列序为 **position-major**(`i = t*batch + b`),而注意力机制
  (`AttentionBase::rearrange_3d` + `batched_matmul` + 因果掩码)假定 **batch-major**(`i = b*seq + t`)。
  `rearrange_3d` 把 position-major 列解释为 block 内混入多个真实样本 → **跨样本串扰 + 因果链断裂**。
- **定位手段**:临时诊断程序对比 `batch=1` vs `batch=2`(样本 1 用不同 token)的输出:
  pos0 diff=0,pos1/2/3 diff≈O(1) → 跨样本泄漏坐实。
- **为什么测试没拦住**:`attn_consistency_test` 用 `batch=1`,两布局完全重合,测不出。
- **教训**:
  - 布局约定必须**全局统一**并在文档固化,不能一个文件一个样。
  - **所有注意力/序列相关测试必须覆盖 batch>1**。
  - "本地正确 ≠ 集成正确",集成层要加最小差异对比测试。

### 2. GPU-resident 多层层叠正确率 ~10%(2026-07, v6)

- **症状**:每个 GPU 算子单独验证正确,链式多层的网络正确率从 95% 暴跌到 ~10%。
- **根因**:`Matrix::gpu_shadow_ + cpu_dirty_` 影子一致性在多层链式调用中失同步
  (GPU 上修改后 CPU 侧不知道,反之亦然)。
- **结局**:整个 GPU-resident 路径被**禁用**,退回 staging 路径(每次算子往返 PCIe)。
- **教训**:
  - 双存储(CPU/GPU)是**分布式状态机**问题,比单存储难一个数量级。
  - 影子一致性必须有系统化的 invalidate 测试,不能靠"碰巧对"。
  - 谨慎评估"零层间 PCIe"的收益 vs 一致性维护成本 —— 本项目中 staging 路径的
    93.4% GPU 利用率已足够好。

### 3. TDR 设备死亡后重试:死循环(2026-07-26)

- **症状**:Windows TDR 在连续 GPU 工作 ~2s 后重置驱动;`VK_ERROR_DEVICE_LOST` 后
  所有后续 Vulkan 调用**全部失败**,旧代码仍尝试 `from_matrix()` → 永久失败循环。
- **关键区分**:
  - `VK_ERROR_DEVICE_LOST`(-4):GPU 已死,**不能重试**,保存 checkpoint 后优雅退出。
  - `VK_TIMEOUT`(正值):GPU 还活着,**可以减半 batch 重试**。
- **措施**:`device_lost_` 标记 + `flush_batch()` 把大提交拆成 forward+loss | backward 两段,
  降低单次提交超 TDR 的风险;定期 checkpoint。
- **教训**:设备错误必须检查 `VkResult` 原值,不能走统一 `vk_check` 吞掉区分信息。

### 4. batch 模式 use-after-free(2026-07 ~ 08)

- **症状**:`VK_ERROR_DEVICE_LOST` / 违反 `VUID-vkDestroyBuffer-buffer-00922`。
- **根因**:
  - 命令录制期引用的 GPU buffer 被提前析构(如 `forward_sparse` 中局部 `shifted`),
    提交时引用已释放内存。
  - copy-on-write 原语替换 Tensor 时,旧 `GpuBuffer` 析构 → `vkDestroyBuffer`,
    但已录制的 descriptor 仍引用它。
- **修复**:所有录制期引用的张量存活到 `end_batch()` 之后;销毁转入
  `pending_destroys_` 延迟队列,`end_batch/flush_batch` 释放 desc sets 后统一销毁。
  `GpuBackend::instance()` 改 new-leak 单例规避静态析构顺序问题。
- **教训**:"命令录制"与"命令执行"是两段时间线,对象生命周期必须覆盖到执行完成。

---

## 🟠 高危:内存爆炸/崩溃/断言

### 5. BPE pair_locations 内存爆炸:60-80GB(2026-08)

- **症状**:1.8GB 文本 → 18 亿 token → `unordered_map<uint64_t, vector<pair>>` 存全部
  出现位置 ≈ 60-80GB。
- **修复**:移除全位置索引,改用 `pair_freq`(频次) + `token_chunks`(token→chunk 集合),
  每轮合并取交集找候选 chunk。17.8MB 测试集 411s→115s,内存 ~15GB。
- **教训**:**"所有出现位置"类索引随输入线性膨胀,超大规模语料必须先做内存预算**。
  字符串层去重(unique 后仅 65K chunks,<1GB)是另一个关键前置优化。

### 6. Tensor::reshape CPU 视图与 Matrix shape 不一致(2026-08-17)

- **症状**:`add_inplace` 断言失败 `lhs=(seq,d_model) rhs=(seq*d_model,1)`。
- **根因**:`Tensor::reshape()` 是零拷贝视图:共享 `cpu_data_` 只改 Tensor 元数据 `rows_/cols_`,
  但 CPU 引擎的 `add_inplace` 用底层 `Matrix` 实际 shape 检查 → 不一致。
- **修复**:CPU 分支 reshape 复制数据到新 shape;GPU 分支保持零拷贝(buffer+count 无 shape)。
- **教训**:**任何"零拷贝视图 + 后端无关"的 Tensor 操作,CPU 与 GPU 语义天然不同**。
  排查 `add_inplace` 断言:先怀疑 reshape 后共享 Matrix 参与运算。

### 7. 缓存 key 冲突:哈希碰撞与位域溢出

- **MatmulKey 哈希碰撞**:`(512,600,784)` 与 `(128,600,256)` 撞车 → FNV-1a 修复。
  缓存 key 必须用强哈希,不要 naive 组合。
- **掩码/pos_indices 缓存 key**:`(batch<<16)|seq_len` 位域溢出冲突 → 改双字段
  `(batch, seq_len)`。
- **教训**:缓存 key 设计 = 一小块安全工程,值得单独 review。

### 8. gather_rows 用 idx.rows() 而非 idx.size()(2026-07-25)

- **症状**:batch>1 时 lookup 次数只有 `seq_len`,漏掉 batch 维 → 跨样本错位。
- **修复**:`gather_rows`/`scatter_add_rows` 用 `idx.size()`。
- **教训**:行数 ≠ 元素数,二维索引张量的遍历次数是 `rows*cols`。

### 9. CUDA obj 不随 Debug/Release 重编:链接失败(2026-08-07)

> ⚠️ **CUDA 后端已停用（v1.0.0）**，此坑仅作为历史参考，恢复 CUDA 时需注意。

- **症状**:Debug↔Release 切换后 `cuda_kernels.obj` 的 DEPENDS 只有 .cu,不重编 →
  `_ITERATOR_DEBUG_LEVEL` 不匹配(clang-cl=2 vs nvcc obj=0)→ lld-link 失败。
- **修复**:`-Xcompiler /D_ITERATOR_DEBUG_LEVEL=N` 对齐;切换配置后
  `Remove-Item build\cuda\cuda_kernels.obj -Force` 再重编。
- **教训**:多编译器混合构建,调试级宏必须在 nvcc 侧显式对齐。

### 10. 除零/越界防护缺失(2026-08-14 code review 批量发现)

- `steps=0`(mnist_train/mnist_bench/text_train)、`--topk>10`、tokenizer_infer 空输入、
  `--iters 0`、`evaluate_mnist` 空数据集、`mnist_io.hpp` 末行无换行符丢样本、
  SGF 坐标跳 i/嵌套括号。
- **教训**:CLI 入口的参数防护是一次性成本,review 时逐参数过一遍。

---

## 🟡 中危:结果偏差/性能/确定性

### 11. BPE 并行化破坏确定性(2026-08)

- **症状**:并行/顺序训练产出**不同词表**。
- **根因**:
  - CharBPE 非 ASCII 字符 ID 按 `word_freq`(unordered_map)迭代顺序分配。
  - `bpe_train_impl_` 找最优 pair 的平局打破依赖 `pair_freq` 迭代顺序。
- **修复**:字符先收集去重后**按码点排序**再分配 ID;平局按 `pair_key` 打破
  (`freq==best && key<best_key`)。
- **教训**:**任何并行化都必须先审计所有"依赖容器迭代顺序"的决策点**。
  正确性 = 结果必须与单线程逐字节一致(用词表逐字节比对验证)。

### 12. BPE 合并循环并行化失败(2026-08)

- `pair_freq` 增量更新,neighbor 变化影响下轮选择;delta 合并改变更新时序 → 结果偏差
  (207→194 tokens)。BPE 本质是**依赖链式迭代**,不是数据并行问题。
- **教训**:数据并行基础设施(parallel_for_samples)适合矩阵运算,不适合链式迭代算法;
  预分词(重 IO/正则)才适合并行化(34.2s→16.2s,~2.1x)。

### 13. one-hot 3.2GB 撑爆 staging(2026-07)

- vocab≈25k × batch 128 × seq 256 → y_onehot ≈ 3.2GB > staging region(默认 64MB)。
- **修复**:`CrossEntropyLoss::forward_sparse` 整数标签 + loss_mask,GPU 显存 25GB→10GB。
- **教训**:大词表场景禁止物化 one-hot;`upload_blocking/download_blocking` 支持
  按 region 容量自动分块。

### 14. GPU 影子一致性细节(2026-07, v6.1)

- `const span()` 必须走自动同步(不能读 `data_`);`transpose_to()` 写结果后
  `invalidate_gpu()`;`scale_inplace()`/`zero()`/`binary_apply_inplace()` CPU 写后
  reset GPU shadow。
- **教训**:影子规则要收敛到少数几个方法统一处理,靠每个调用点自觉必然漏。

### 15. gradcheck 必须先 forward 填充 input_cache_(2026-08-15)

- 直接 backward → 空缓存访问冲突 `0xC0000005`(崩溃而非报错)。
- **教训**:gradcheck 工具自身也要遵守"forward 先于 backward"的层协议。

### 16. Go AI 贪心坍缩 + 解析早退(2026-08-07)

- 温度 0.0 稳定输出 `<me>` tag;`--max-tokens 1` 输出 1 个 tag 就失败;
  `_parse_infer_output` 在 prompt 行找不到坐标就过早 return None → 永远无有效落子。
- **修复**:温度 0.8、max-tokens 16、解析器过滤日志行遍历整行找坐标。
- **教训**:AI 推理参数要有随机性 + 输出解析要容错;模型未训练好时 tag 坍缩要能识别。

### 17. 模型格式解析:UIntArray 循环边界(2026-08)

- 解析 UIntArray 时直接在条件里用 `value.size()`,`take_u64` 会缩短 value →
  只解析一半。
- **教训**:循环边界必须先 `count = value.size()/8` 固定;新格式字段维护点集中在
  `apply_spec_version_defaults`(字段缺失→按版本回落默认)。

### 18. 平台差异:MSVC std::optional<T&> 不支持

- MSVC(14.51)+ Clang-cl 在 Windows 上用 MSVC STL,`<optional>` 的 union 实现
  不支持引用成员(P2988R0 未实现)→ 用 `T*` 兜底。
- **教训**:新 C++26 特性先验证 MSVC STL 支持度再采用。

### 19. GPU scatter_add 的 CAS float atomicAdd 非确定性(2026-08-26,已文档化例外)

- **症状**:GPU 路径 `scatter_add_rows`(embedding 梯度累积)用 CAS 循环实现
  float atomicAdd,同一行多个梯度值的累加顺序取决于 GPU 调度 → 跨 run 有
  ~1e-7 相对误差的舍入抖动,违反铁律 8(逐字节确定性)。
- **为何从未被触发**:① 偏差仅为浮点舍入噪声,被 Adam/AdamW 的动量与学习率
  噪声完全吸收,loss 曲线与最终精度无可观测差异;② 项目没有"GPU vs CPU
  逐字节比对"的测试(只有容差比对),抖动落在容差内;③ 单行冲突率低
  (embedding 梯度中同一 token 在一个 batch 内重复次数少),多数行只累加
  一次,顺序无关。
- **处置**:**已文档化例外**(不修复)。理由:PyTorch 的 `index_add_` 同样
  非确定;改为"排序+顺序累加"需全局排序(开销大),改为"分桶+确定性归约"
  需额外显存与 kernel。偏差量级(1e-7)远小于训练噪声(1e-3),修复收益为负。
  CPU 参考路径(`CpuEngine::scatter_add_rows`)保持顺序累加、逐字节确定,
  作为 ground truth。
- **教训**:铁律 8 的"逐字节一致"对**原子累加类**算子应放宽为"容差内一致"
  (与 PyTorch 对齐);真正需要逐字节确定的是**决策类**逻辑(平局打破、ID
  分配、缓存 key),而非浮点累加顺序。新增原子算子时先评估冲突率与偏差量级,
  再决定"修复"还是"文档化例外"。

---

## 🟢 工具链与 UI 坑

| 坑 | 领域 | 教训 |
|----|------|------|
| Windows CLI 输出 CRLF(`\r\n`)与进度覆盖(`\r`)混淆,行全空 | GUI | 解析器必须区分:独立 `\r` 才是进度覆盖 |
| tkinter `self._w` 覆盖内部 widget 路径属性 → TclError | GUI | 子类属性禁止 `_w`/`_h`(tkinter 保留) |
| `_make_option_row` 返回 StringVar 而非控件,`grid_remove` 报错 | GUI | 显隐操作用 `getattr(w,"widget",w)` 解析真实控件 |
| QProcess `self._proc` 在 start() 后才赋值,启动瞬间输出丢失 | GUI | `self._proc` 必须在 `proc.start()` **前**赋值 |
| `_feed` 残留重复 `self._current += ch`,逐字符翻倍 | GUI | 编辑前先读一遍旧代码 |
| QSettings Windows 读回字符串需按参数 kind 强转 | GUI | `cli.coerce_values` 统一处理 |
| `gui.py infer_gpu_var` 字符串恒真,恒传 `--gpu` | GUI | 字符串判真 ≠ 布尔判真 |
| `_connect_deps` 只扫可见参数,隐藏参数依赖不连接 | GUI | 依赖扫描必须覆盖**全部**参数 |
| Gradio `gr.Sketchpad` 输入是 dict,PIL composite 是对象非路径 | GUI | `_to_gray()` 兼容 dict/PIL/路径/图层 |
| `torch.load` 反序列化漏洞 | Python | 一律 `weights_only=True` |
| Python 3.10 `tarfile.extractall(filter=...)` 不可用 | Python | `_safe_extract` 逐成员校验路径 |
| nvcc 12.8 与 MSVC 2026 (v14.51) 不兼容,cudafe++ crash | 构建(CUDA 已停用) | 用 VS 2022 BuildTools;`-allow-unsupported-compiler` |
| `train_bytebpe.py`:正则 `[ \t]+` 空格独立成段,空格占 44.8% token | 分词器 | GPT-2 风格正则 `\s*[CJK]+|\s*[a-zA-Z0-9]+|\s*[单标点]` |
| `byte_fallback=True` 不自动补 256 字节,中文→UNK | 分词器 | `BpeTrainer(initial_alphabet=ByteLevel.alphabet())` |
| C++ CharBPETokenizer 对中文仍是 UNK | 分词器 | 字节级 BPE 拆中文字节,须改 encode 或换 charbpe |
| tokenizer_infer 预览截断产生半个 UTF-8 字符(终端 `�`) | 分词器 | 按完整字符(首字节宽 1/2/3/4)截断 |
| 模型/词表等训练产物进 git(204MB 数据集) | 工程 | .gitignore 只留源码,产物一律忽略 |

---

## 🧬 根因模式(跨领域总结)

> 上面 30+ 个坑,收敛到 **7 个根因模式**。新代码评审时对照此表自查。

### 模式 A:同一数据多份表示,不变量未集中维护 🔴
GPU 影子(`gpu_shadow_`/`cpu_dirty_`)、Tensor reshape 视图、模型格式版本默认值、
`.tokcache` 缓存 —— 都因"两份数据 + 分散的同步点"出错。
**对策**:不变量收敛到单一方法/单一维护点;一致性测试必须系统化(见模式 G)。

### 模式 B:布局/索引约定散落,未固化 🔴
`i = t*batch + b` 出现在 GPTModel、`rearrange_3d` 假定 `i = b*seq + t`、
训练循环的 flat_targets/loss_mask 又一套 —— 三处各写各的 → 灾难 bug #1。
**对策**:布局公式写进文档 + 提供单一工具函数(禁止各处手写 flat 索引)+ batch>1 测试。

### 模式 C:依赖容器迭代序/哈希序 🟡
unordered_map 迭代顺序(字符 ID 分配、平局打破)、Naive 缓存 key 哈希。
**对策**:涉及确定性的决策点一律显式排序;缓存 key 用强哈希(FNV-1a 起步)。

### 模式 D:异步生命周期管理 ⚠️🔴
Vulkan "录制 vs 执行" 两段时间线(TDR、use-after-free、pending_destroys)、
静态析构顺序。
**对策**:资源所有权用 RAII 表达;销毁走延迟队列;设备错误区分 DEVICE_LOST/TIMEOUT。

### 模式 E:跨工具链语义差异 🟢
CRLF vs LF、MSVC/nvcc 版本、`_ITERATOR_DEBUG_LEVEL`、Python 3.10 API 差异。
**对策**:解析层统一处理换行;构建脚本兜底清理;新 API 先查最低支持版本。

### 模式 F:内存规模无预算 🟠
pair_locations(60-80GB)、字符串未去重(30GB)、one-hot(3.2GB)。
**对策**:大输入处理前先算内存预算(字节数 × 每 token 字节数 × 索引放大系数);
能稀疏化就稀疏化(forward_sparse);能先去重先去重(字符串层)。

### 模式 G:"本地正确 ≠ 集成正确" 🔴
GPU-resident 单算子对、链式错;attn batch=1 对、batch=2 错;gradcheck 对、训练错。
**对策**:集成层加**最小差异测试**(batch=1 vs batch=2 对比、单层 vs 多层对比),
这是定位 #1 和 #2 两个灾难 bug 的关键手段。

---

## 📋 开发活动防坑清单

### 添加新引擎原语(最高成本操作,5 处)
- [ ] `compute_engine.hpp` 接口 + 文档注释(shape/转置语义)
- [ ] `cpu_engine.hpp` 实现(注意 AVX2 可向量化,不要退化成裸指针循环)
- [ ] Vulkan:shader + SPIR-V 嵌入 + `vk_backend.hpp` 管线
- [ ] gradcheck / 一致性测试(forward+backward,CPU/GPU 双后端对比)
- [ ] 检查缓存 key(强哈希)、batch 模式生命周期(pending_destroys)

> 注:CUDA 已停用(v1.0.0),新增原语不再需要实现 CUDA 内核。

### 添加新 Layer
- [ ] 布局约定:flat 索引公式必须与 GPTModel/AttentionBase 一致(batch-major)
- [ ] forward 先于 backward 的协议(gradcheck 前先 forward 填缓存)
- [ ] 参数/梯度注册、`extra_state()`(BatchNorm 的 running 统计)
- [ ] 序列化:model_spec 新字段走 `apply_spec_version_defaults` 版本默认表
- [ ] 推理模式:`set_training(false)` 路径(batch=1 场景必须正确)

### 修改 GPU 相关
- [ ] batch 模式:录制期引用的张量存活到 end_batch 之后
- [ ] TDR:DEVICE_LOST 保存 checkpoint 退出,TIMEOUT 才减半 batch 重试
- [ ] 影子一致性:CPU 写后 invalidate,`const span()` 自动同步
- [ ] 内存预算:矩阵尺寸 × 4B × 张量个数,超 staging 走分块

### 修改 BPE/Tokenizer
- [ ] 确定性:所有 ID 分配显式排序,平局按 key 打破
- [ ] 内存:字符串层先去重,索引随输入不超线性
- [ ] 中文:UTF-8 字符语义(CharBPE)vs 字节语义(ByteBPE)勿混

### 修改 GUI
- [ ] CRLF/`\r` 区分;tkinter 保留属性 `_w`/`_h` 禁用
- [ ] QProcess 赋值时机;QSettings 类型强转;隐藏参数依赖
- [ ] CLI 输出格式变更 = 同步改 parser 正则(CLI 是 GUI 的唯一契约)

### 修改构建/工具链
- [ ] Debug↔Release 切换:清 `build/cuda/cuda_kernels.obj` 再编(CUDA 已停用,仅恢复时适用)
- [ ] nvcc 编译器版本匹配(12.8 ↔ VS 2022 BuildTools;CUDA 已停用)
- [ ] 新 C++26 特性先验证 MSVC STL 支持度

### 修改模型格式
- [ ] 新字段 → `apply_spec_version_defaults` 加默认值(唯一维护点)
- [ ] 旧格式处理:显式拒绝并提示重训,不静默读错

---

## 📚 相关文档

- `docs/DEVELOPMENT_STANDARDS.md` — 分层职责规范("每层只能负责每层的事")
- `docs/01-architecture.md` — 架构分层
- `docs/06-cuda-backend.md` — CUDA 构建与版本匹配（已停用，恢复参考）
- `docs/07-train-package.md` — 训练包格式
