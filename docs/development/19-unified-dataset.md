# 统一数据集格式与加载（.nndataset / .nnvocab / KVRecord v2）设计

> **状态**：设计定稿（2026-10-05 对话裁定）；**阶段一~三已实施（2026-10-05/06 落地）**——
> 阶段一 = KVRecord v2 + `.nnvocab` + `tokenizer_train` 切换 + `dataset_convert vocab`；
> 阶段二 = `.nndataset` + `dataset_gen` + `nn::Dataset` + `text_train` 切换 + 模型 v6 内嵌
> （`MODEL_VERSION 5→6`，v5 及以下仍读内嵌 JSON）；阶段三 = GUI / 控制器 / `train_pkg` /
> 审计脚本参数同步 + 文档。**阶段四（tabular/csv）未实施**（按裁决预留）。
> 落地验收与实测数字见 `docs/history.md`「统一数据集」条与 AGENTS §12。
> **关联**：`docs/development/18-roadmap.md` X2（无 Dataset 抽象）/ P2-4（Dataset/DataLoader + 流式）——
> 本文是其前半（格式统一 + 统一类）的详细设计；DataLoader/流式仍留在 P2-4 后半。
> `.nnpkg` 打包链路**本轮不在范围**（用户裁定），但 §6 标注了它的参数冲突依赖。

---

## 0. 裁决记录（对话定稿，逐条对应）

| # | 议题 | 裁决 |
|---|------|------|
| 1 | 范围 | 本轮**只写设计文档，不做代码**；`.nnpkg` 先不管 |
| 2 | 存储单元 | 早期讨论中的 "step" 即**一行文本 = 一篇文档（doc）**；最终**存 docs 不存 steps**（"step" 一词废弃，避免与训练步混淆） |
| 3 | 窗口化 | **保持原方法**：训练期把全部 doc 拼回全局 token 流再滑窗（`--seq-len`/`--stride` 仍是 text_train 参数）；**不**引入 `Dataset.resize()` 预切窗 |
| 4 | 文件格式 | 词表 `.nnvocab`、数据集 `.nndataset`（扩展名用户钦定） |
| 5 | KVRecord v2 | **字段只记录 key、数据类型、值的地址**（"v2 仅支持记录位移"）；版本号内嵌 kvrec 自身：**无版本字段 = v1，有 = 按版本读** |
| 6 | 嵌套结构 | 整文件 = 标签 + 一个文件级 KVRecord；配套文件、词表、train/test 子集**层层嵌套 KVRecord**（json 套 json，但格式是 kvrec）；**license 不是 kvrec，是纯文本文档** |
| 7 | 词表 | **全面 KVRecord 化**；独立词表文件**纯 kvrec**——旧 `.json` 词表**不直读**，需转换器；**模型按版本分派**：v5 及以下仍读其内嵌 JSON（模型格式的一部分），v6 起内嵌 kvrec，旧模型零破坏 |
| 8 | 分词器产出 | **复用 `tokenizer_train`，改成写 kvrec 格式**（`.nnvocab`） |
| 9 | 生成器词表入口 | `dataset_gen` 的 `--vocab` **必填**（tokenizer_train 先行，职责单一） |
| 10 | 测试集 | **两个文本输入**：生成器收 train 文本 + 可选 test 文本，分别写 train/test 子集 |
| 11 | loss 掩码 | **挪到生成器**：`dataset_gen --loss-scope`，每个 doc 顺带记录掩码；`text_train` 删除 `--loss-scope` 与扫描逻辑 |
| 12 | tokcache | **合入数据集**：生成期编码，**不存 txt 本身，只存编码和词表**；`.tokcache` 机制随之消亡 |
| 13 | txt 直读 | `text_train` **彻底移除**直读 `.txt` 路径（含内联 tokenize），只吃数据集文件 |
| 14 | token 宽度 | **文件存 u64**（与现状 `size_t` 逐位同构；体积影响见 §10-3） |
| 15 | MNIST/tabular | csv **同构设计**（一行 = 一个 doc），`kind` 字段区分；**本轮只写进文档，不做代码** |
| 16 | 词表类型现状 | 磁盘 JSON 无类型；内存全是 `std::size_t`（64 位 = u64）；merges 为 `{size_t,size_t,size_t}` 三元组 |

---

## 1. 背景：现状是六条互不相通的读取路径

| 数据 | 现有格式 | 读取代码 | 问题 |
|------|---------|---------|------|
| MNIST 训练/测试 | CSV（train.csv 320 MB） | `cli/cli_mnist_io.hpp` `load_mnist_csv` | 每次运行全量 from_chars 解析；只服务 MNIST |
| MNIST 单图推理 | CSV 单行 | `domain_mnist.hpp` `load_image_from_csv_line` | 与批量加载是两套解析 |
| 文本语料 | `.txt`（tinystories 1.9 GB） | `text_train.cpp` 内联 `read_file_lines` | 整段读进内存；逻辑埋在 1800+ 行 CLI 里不可复用 |
| tokenize 缓存 | `.tokcache`（TKCH v2 私有二进制） | `text_train.cpp` 内联 | 私有格式、失效 key 只比文件大小（同大小不同内容会骗过） |
| 分词器训练/推理 | 任意文本 / JSON 词表 | `core_file.hpp` `load_text_file`、`load_tokenizer_from_file` | 词表是 JSON，与数据集格式无关 |
| 打包分发 | `.nnpkg`（tar+manifest） | `train_pkg.py`（纯 Python） | 与 C++ 读取层零互通（**本轮不管**） |

**痛点**：新接一个数据集要重写一套读取+解析+缓存；训练 CLI 承担了数据加载职责；
词表/语料/掩码/缓存四个概念散落在三个文件里。

---

## 2. 总览

```
                    ┌──────────────────────────────┐
  文本语料 .txt ──► │ dataset_gen  (新 CLI 生成器)   │ ──► xxx.nndataset
  词表 .nnvocab ──► │  tokenize + 掩码 + 打包        │        │
                    └──────────────────────────────┘        │ 读取
  文本语料 .txt ──► tokenizer_train (改写 kvrec) ──► .nnvocab │
                                                             ▼
                    ┌──────────────────────────────┐
  xxx.nndataset ──► │ nn::Dataset  (统一只读类)      │ ──► tokens / doc_ids / loss_mask
                    └──────────────────────────────┘        │
                                                             ▼
                                            text_train（只剩：拼流 → 滑窗 → 训练）
```

**组件与职责边界**：

| 组件 | 职责 | 明确不做 |
|------|------|---------|
| `tokenizer_train` | 文本 → 词表，**输出改为 `.nnvocab`（kvrec）** | 不碰数据集 |
| `dataset_gen`（新） | 文本 + 词表 → `.nndataset`：编码、算掩码、写配套（来源/license/生成参数） | 不做训练相关任何事 |
| `dataset_convert`（新） | 旧 `.json` 词表 → `.nnvocab`；未来 csv → nndataset | 不做训练相关任何事 |
| `nn::Dataset`（新类） | 嗅探 + **只读**打开 `.nndataset`/`.nnvocab`；输出与现 tokcache 同构的 token 流 | 不写文件、不做隐式转换、不做窗口化 |
| `text_train` | 读数据集 → 拼流 → 滑窗 → 训练 | 不再含 tokenize/缓存/掩码扫描/词表参数 |

**术语**：本文用 **doc** = 语料中的一行（非空、已 trim）= 数据集的存储单元。
早期讨论中的 "step" 即此物，最终命名统一为 doc。

---

## 3. KVRecord v2 规范

### 3.1 v1（现状，不动）

```
[field_count u32]
field := [key_len u32][key bytes][type u8][value_len u32][value bytes]
type: 0=UInt(u64)  1=Str(utf8)  2=UIntArray(count×u64)
```

模型规格头（`model_serialization.hpp` 的 spec header）继续用 v1，**零改动**。

### 3.2 v2 布局

```
[field_count u32]                         // 含哨兵在内
field[0] := 哨兵，按 v1 布局编码:
            [key_len=13]["kvrec_version"][type=0][value_len=8][u64 version=2]
field[i>0] := [key_len u32][key bytes][type u8][addr u64]      ← 只记 key、类型、地址
addr       := 该值数据块在**文件内的绝对偏移**
数据块     := [value_len u64][value bytes]                      ← 每块自带长度
type: 0=UInt  1=Str  2=UIntArray  3=Record(嵌套 kvrec v2，value = 完整嵌套 kvrec 字节)
```

**版本自举（v1/v2 判别）**：解析器把 `field[0]` 按 v1 布局读出——
key == `"kvrec_version"` ⇒ 本记录为 v2，余下字段按 v2 布局解析；
key 不匹配 ⇒ 本记录为 v1（field[0] 已被正确按 v1 消费，继续 v1 解析）。
即：**无版本字段 = v1，有 = 按该版本读**（用户裁定原话）。
`"kvrec_version"` 为保留键名；空记录（field_count=0）按 v1 处理。

> ⚠ **哨兵是 v2 中唯一按内联编码的字段**——这是"版本内嵌且能自举"的最小代价：
> 判别版本必须先有一个两边都读得懂的字段。除此之外**所有值一律走地址**，满足"v2 仅记录位移"。

**未知类型跳过**：v2 字段本身不带长度，遇到未知 type 时 seek 到 `addr`、读 `value_len u64`、跳过——
沿用 v1 "未知字段按长度安全跳过" 的向前兼容承诺。

**嵌套**：type=3 的数据块内容是**完整的一个 v2 kvrec 字节**（含其自己的哨兵），
其内部字段的 addr 同样是**文件绝对偏移**。地址为绝对 ⇒ 嵌套 kvrec 不可整体搬迁到别的文件
（可接受；重定位工具如未来需要，按字段表重建 addr 即可，留作扩展）。

### 3.3 与 v1 的关系

- 两版本共享 `KeyValueRecord` 类，实现上增加 v2 分支（`parse` 按哨兵分派，`serialize_v2` 新增）；
- v1 记录里碰巧首个键叫 `kvrec_version` 会被误判——现实中不存在，写进注释作为保留键约束；
- **模型 spec header 不动**（v1），避免无谓的字节锚变动。

---

## 4. 文件格式

### 4.1 文件标签（两类文件通用）

```
[label] := [magic 4B][file_version u32]
  .nndataset: magic = 'N''N''D''S'   file_version = 1
  .nnvocab:   magic = 'N''N''V''C'   file_version = 1
```

`file_version`（文件布局演进）与 kvrec 内部 `kvrec_version`（记录编码演进）是**两条独立的演进轴**。
标签之后紧跟文件级 kvrec。**嵌套进别的文件的 kvrec 没有标签**（靠哨兵判版本）。

### 4.2 `.nnvocab` —— 词表文件（kvrec v2）

```
[label]
[kvrec v2]
  "type"    := Str  "bpe" | "char_bpe"        ← 沿用现 JSON 的 "type" 语义（工厂判别依据）
  "vocab"   := Str blob: [count u64][ [len u64][token utf8 bytes] ] × count
  "merges"  := UIntArray，展平三元组: (id_a, id_b, new_id) × n   ← 与现 JSON "merges" 一一对应
  "markers" := UIntArray(8)                    ← 可选；缺省时加载端按现 restore_dialogue_markers
                                                从 vocab 推导（与现行为一致）
```

- **纯 kvrec**：`.json` 词表**不直读**（加载报错并提示 `dataset_convert vocab`）——用户裁定；
- 字符级 charbpe 的字节级转义（现 `\xNN` 写法）原样进 blob 的 token 字符串，语义不变；
- vocab blob 是 Str 类型下的自描述结构（内部 length-prefixed），kvrec 层无需新类型。

### 4.3 `.nndataset` —— 数据集文件（标签 + 文件级 kvrec 嵌套全套）

```
[label: "NNDS" + file_version=1]
[kvrec v2 —— 文件级]
  "kind"      := UInt   0=text（本轮）| 1=tabular（预留，§4.4）
  "companion" := Record —— 配套文件 kvrec
      "vocab"     := Record —— 词表 kvrec（§4.2 的完整 kvrec，无标签；
                               即：文件kvrec → 配套kvrec → 词表kvrec 三层嵌套）
      "source"    := Record —— 来源:
                        "path"    := Str   源文本路径/名称
                        "sha256"  := Str   源文本内容摘要（非文件大小！）
                        "docs"    := UInt  源文档行数
      "license"   := Str —— 可选；**纯文本文档全文**，不是 kvrec（用户裁定）
      "gen"       := Record —— 生成参数:
                        "loss_scope" := Str "all" | "assistant"
                        "tool"       := Str  "dataset_gen vX.Y"
                        "created"    := UInt unix 时间戳
  "train"  := Record —— 训练子集（§4.3.1）
  "test"   := Record —— 测试子集（结构与 train 完全相同；可缺省）
```

#### 4.3.1 子集 kvrec（train/test 同构，内部按 doc 分配）

```
[kvrec v2 —— 子集]
  "num_docs"  := UInt   doc 数
  "doc_index" := UIntArray，长度 = num_docs + 1
                 doc_index[i] = 第 i 个 doc 块的文件绝对偏移；末项 = 末块结尾
  doc 块（连续字节，经 doc_index 寻址）:
      [token_count u64]
      [token_count × u64]   tokens      ← 每 token 一个 u64（裁决 #14）
      [token_count × u8]    loss_mask   ← 仅当 loss_scope=assistant 时存在（见下）
```

**掩码规则**（裁决 #11 + 体积优化）：

- `--loss-scope assistant`：生成期对每个 doc 独立跑 `mark_assistant_spans` 等价逻辑，
  **逐 doc 写入 mask 段**。逐 doc 独立计算与现全局扫描**结果逐位一致**
  （现逻辑在 doc 边界重置段状态，见 `text_train.cpp` `mark_assistant_spans` 注释）；
- `--loss-scope all`（默认）：**不写 mask 段**（全 1 是纯冗余），加载端按 `gen.loss_scope`
  合成全 1 向量；
- `text_train` 不再有 `--loss-scope`，语义选择发生在生成期，**数据集自带语义**。

**为什么掩码存在但 doc_ids 不存在**：doc_id = doc 在子集中的序号 + 1（现逻辑即行号 + 1），
加载端拼接时按 `doc_index` 顺序重建，**无需存储**；掩码无法从边界推导，故存储。

#### 4.3.2 加载输出与现 tokcache 的同构关系

`nn::Dataset::load_text()` 输出 `(token_flow, doc_ids, loss_mask)`，与 `text_train` 现状
（tokcache + `mark_assistant_spans` 的产物）**逐位同构** ⇒ 下游"拼流 → 滑窗 → 训练"代码
**一行不改**，训练字节锚得以保持。

**一致性硬约束**（迁移时逐字保留的三段逻辑）：

1. 行 trim 规则（首尾去空格/`\t`/`\r`、跳过空行）——`read_file_lines` 现逻辑，移入 `dataset_gen`；
2. 并行 tokenize 的保序拼接——按 chunk 下标升序合并（铁律 #8），移入 `dataset_gen`；
3. 掩码计算的 doc 边界重置语义——见上。

### 4.4 tabular 形态（预留规范，本轮不实现）

```
kind = 1（tabular）:  CSV 一行 = 一个 doc
  doc 块 := [label u64][feat_dim u64][feat_dim × f32 features]
  （无 mask 段；配套/子集结构与 text 完全相同）
```

- csv → nndataset 转换由 `dataset_convert csv`（阶段四）提供；
- `mnist_train` 切换到 `Dataset::load_tabular()` 同样在阶段四（本轮 csv 直读**不动**）；
- 数据集属 I/O 层，tabular 加载产物用 `Matrix` 不违反铁律 #12（该条禁止的是 L2+ 计算层）。

---

## 5. 统一类 `nn::Dataset`

新头 `include/neuralnet.cpp/dataset.hpp`（L3 I/O 层，随 `nn.hpp` 聚合入口暴露；
header-only ⇒ 全部 `inline`；错误一律 `Result<T>`；无隐式随机）。

```cpp
namespace nn {

enum class DatasetKind : std::uint64_t { Text = 0, Tabular = 1 };

struct DatasetInfo {
    DatasetKind kind;
    std::uint64_t num_docs;      // train 子集
    std::uint64_t test_docs;     // test 子集 doc 数（0 = 无 test 子集）
    std::string  loss_scope;     // "all" | "assistant"（来自配套 gen）
    std::string  source_sha256;  // 源文本摘要（诊断/复现用）
    // …配套只读视图按需扩展
};

struct TextCorpus {                  // 与现 tokcache + mark_assistant_spans 产物同构
    std::vector<std::size_t>  token_flow;
    std::vector<std::size_t>  doc_ids;     // = 拼接时按 doc_index 顺序重建（序号+1）
    std::vector<unsigned char> loss_mask;  // loss_scope=all 时为全 1
};

class Dataset {
public:
    [[nodiscard]] static Result<Dataset> open(const std::string &path);
        // 嗅探 label magic 分派 reader；.nndataset / .nnvocab 都能开；
        // 其它扩展/坏文件 → 明确错误（含"旧格式怎么办"的提示）

    [[nodiscard]] const DatasetInfo &info() const noexcept;

    [[nodiscard]] Result<TextCorpus> load_text() const;          // kind=Text
    // 阶段四:
    // [[nodiscard]] Result<std::pair<Matrix, Matrix>>
    //     load_tabular(std::size_t max_samples = 0) const;       // (features, labels)

private:
    // 内部 reader 策略: NndatasetReader（kvrec v2 解析）
};

} // namespace nn
```

**设计要点**：

- **只读、不转换、不窗口化**——窗口是训练策略，留在 `text_train`（裁决 #3）；
- `load_text` 一次性给出训练所需的全部三样（flow/doc_ids/mask），语料超大时
  分块迭代接口（`for_each_chunk`）留作 P2-4 流式后半的扩展点，本轮不设计细节；
- `.nnvocab` 也能 `open`（词表独立加载场景：`text_infer --vocab` 兜底、转换器等）。

---

## 6. 工具与 CLI 变更清单

### 6.1 新增

**`dataset_gen`**（文本数据集生成器）：

```
dataset_gen <text-file> --vocab <v.nnvocab> -o <out.nndataset>
             [--test <test-file>]              # 可选第二输入 → test 子集
             [--loss-scope all|assistant]      # 默认 all
             [--license <file>] [--source <name>]
```

- 读文本 → trim/过滤 → 并行 tokenize（复用现 `parallel_tokenize` 逻辑，移入库内）→
  按 §4.3 打包写 `.nndataset`；
- **`--vocab` 必填**（裁决 #9）：词表从配套嵌入，故必须先有词表；
- 失效判断基于配套里的 `source.sha256`（生成器重跑的依据），不再比文件大小。

**`dataset_convert`**（独立转换器）：

```
dataset_convert vocab <in.json> -o <out.nnvocab>    # 旧 JSON 词表迁移（必须品）
dataset_convert csv   <train.csv> [<test.csv>] -o <out.nndataset>   # 阶段四，预留
```

### 6.2 修改

| 入口 | 变更 |
|------|------|
| `tokenizer_train` | `save()` 写 `.nnvocab`（kvrec）；`--output` 默认 `bpe_vocab.json` → `bpe_vocab.nnvocab`；JSON 写出废弃 |
| `text_train` | 位置参数 `<text-file>` → `<dataset.nndataset>`；**移除** `--vocab`、`--test-file`、`--loss-scope`、`--no-cache` 四个选项及内联 `read_file_lines` / `parallel_tokenize` / tokcache / `mark_assistant_spans`；**保留** `--seq-len`、`--stride`（窗口化原样）；保存模型时内嵌词表 = 数据集配套词表（kvrec，v6，见 §7） |
| `text_infer` | `--vocab` 兜底改读 `.nnvocab`（旧 `.json` 报错指路 `dataset_convert`）；模型内嵌按 `MODEL_VERSION` 分派（§7） |
| `gui.py` + `cli_controllers.py` | GPT 训练 Tab：`text_file` 字段改指向 `.nndataset`；**删除** vocab / test_file / loss_scope / no_cache 四行（`gui.py:1366,1397,1447` 等）；导出训练包映射随之调整 |

### 6.3 不动

- `mnist_train` / `mnist_infer` / CSV 直读（阶段四再切）；
- `--seq-len` / `--stride` / 训练循环主体；
- `.nnpkg`（`train_pkg.py` 的 `text_file`/`vocab`/`test_file` 角色映射与新 CLI 冲突——
  **依赖提示**：阶段二落地时 `train_pkg.py` 与 `bench/gui_cli_audit.py` 必须同批跟进，
  否则 audit 必红；具体改造按用户裁定"先不管"，届时另行立项）。

---

## 7. 模型格式联动（v6 内嵌 kvrec）

- `text_train` 保存模型时把**数据集配套里的词表 kvrec 字节**直接嵌入模型
  （不再有 JSON 转写步骤）；`MODEL_VERSION 5 → 6`；
- 读取按版本分派：**v5 及以下读其内嵌 JSON**（现行为原样），**v6 读嵌入的 kvrec**——
  旧模型零破坏、不需要模型转换工具（裁决 #7）；
- `read_tokenizer`（`model_serialization.hpp`）加 v6 分支；
- ⚠ **版本号抢号提示**：roadmap P1-2（checkpoint 优化器状态）也计划 `MODEL_VERSION 5 → 6`。
  谁先落地谁占 6，另一项顺延 7；两项都碰版本段，**建议同批合并升版**（与 P2-1 校验和同理）。

---

## 8. 兼容与迁移

| 存量资产 | 处置 |
|---------|------|
| 旧 `.json` 词表 | **不直读**；`dataset_convert vocab` 迁移（报错信息必须给出这条命令） |
| 旧模型（v1–v5） | 照常加载（内嵌 JSON 是模型格式的一部分，版本分派） |
| 已有 `.txt` 语料 | 重新过一遍 `dataset_gen`（一次性成本）；下载脚本产出仍是 txt，入口不变 |
| 旧 `.tokcache` | **作废**（随 txt 直读路径一起从 text_train 移除），可直接删除 |
| MNIST csv | 本轮原样；阶段四提供转换 |

**验收基线（阶段二）**：同一语料 + 同一词表 → `dataset_gen` 产出的 tokens/mask 与旧
tokcache + `mark_assistant_spans` 输出**逐位一致** ⇒ `text_train` 训练数据逐位一致 ⇒
现有训练字节锚（CPU `--steps 20` hash、GPU dev2 hash）**不移动**。

---

## 9. 分期实施（供后续立项引用）

| 阶段 | 内容 | 成本 | 验收 |
|------|------|------|------|
| **一** | KVRecord v2（哨兵/地址/Record 类型）+ `.nnvocab` 读写 + `tokenizer_train` 切换 + `dataset_convert vocab` | 中 | kvrec v2 单测（v1 判别、嵌套、未知类型跳过）；json→nnvocab→加载 round-trip 语义一致；旧 json 明确拒绝 |
| **二** | `.nndataset` + `dataset_gen` + `nn::Dataset` + `text_train` 切换 + 模型 v6 内嵌 | 中–大 | §8 逐位一致口径 + 字节锚不移动；ctest 全绿 + 新增 `dataset_test`；txt 直读/tokcache/四选项符号清零 |
| **三** | `gui.py` / `cli_controllers.py` / `train_pkg.py` 参数同步 + 文档（usage、AGENTS 索引） | 小–中 | `bench/gui_cli_audit.py` 退出码 0；`doc_align_audit.ps1` 可行动项 0 |
| **四** | tabular：`dataset_convert csv` + `load_tabular` + `mnist_train` 切换；（此后才轮到 DataLoader/流式 = P2-4 后半） | 中 | csv→nndataset→加载逐位一致；MNIST 训练字节锚不移动 |

阶段一/二可合并为一次交付；阶段三必须与阶段二**同批**（否则 CLI 与 GUI/audit 断裂）。

---

## 10. 风险与开放问题

1. **哨兵是 v2 唯一内联字段**：严格说与"仅记录位移"字面有一处出入（§3.2 已论证必要性）——
   属于裁定 #5 的实施解释，如不认可需回到版本判别机制重议。
2. **绝对偏移的代价**：嵌套 kvrec 搬到别的文件会失效；本轮接受（写入方只在生成期写，无搬迁场景）。
3. **u64 token 的体积**：tinystories_full（1.9 GB 文本）量级估算 ~5 亿 token →
   tokens ≈ 4 GB + assistant mask ≈ 0.5 GB。裁决 #14 选定 u64（逐位同构优先）；
   若将来体积不可接受，`file_version=2` 升 u32 即可（读写两端都在自己手里，成本一次迁移）。
4. **MODEL_VERSION 6 与 P1-2 抢号**（§7）——立项时合并或错峰。
5. **阶段二与 train_pkg/gui_cli_audit 的耦合**（§6.3）——"先不管 .nnpkg"不等于可以不改：
   CLI 选项消失当天，依赖旧参数名的 Python 侧必须同步，否则审计红。
6. **loss_scope=all 不写 mask** 是对"每个 step 顺带记录掩码"的体积优化实施解释
   （全 1 向量不存储、加载合成）——语义等价，如要求物理存储再改回。

---

## 11. 关联索引

- 现状读取路径：`include/neuralnet.cpp/cli/cli_mnist_io.hpp`、`src/text_train.cpp`（§ 裁决涉及行：
  `read_file_lines:65`、`tokcache:203-267`、`mark_assistant_spans:179`、帮助行 `293-311`）
- KVRecord v1：`include/neuralnet.cpp/model_keyvalue_record.hpp`
- 模型格式与内嵌词表：`include/neuralnet.cpp/model_serialization.hpp`（`read_tokenizer:478`）
- 词表工厂与 JSON 解析：`domain_tokenizer{,_base,_bpe,_charbpe}.hpp`
- roadmap 对应条目：`docs/development/18-roadmap.md` X2 / P2-4
