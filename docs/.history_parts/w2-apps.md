# w2-apps

## base 模式 BOS/EOS 插入的删除（原位置 src/text_train.cpp:4-5 头注释、src/text_train.cpp:98、src/text_train.cpp:808，整改前行号）
- 类型：删除清单
- 内容：早期 text_train 把每行编码为 `[BOS]+tokens+[EOS]` 再拼接，行边界靠 EOS 编码进 token 流；base 模式改为纯拼接、行间无分隔符后，文档边界改由 parallel_tokenize 产出的 doc_ids（块对角掩码）表达。随迁的等价性叙事：text_infer 曾以「训练时每行以 BOS 开头 / 与训练时每行格式一致」论证 prompt 加 BOS 的合理性（src/text_infer.cpp:171、:193、:362 整改前），该前提在纯拼接方案下不成立（`scripts/download_everyday_conversations.py` 生成的对话行也不含 BOS）。现注释只陈述两侧各自的现行行为。

## host 取 loss 的同步旧路径（原位置 src/text_train.cpp:1339，整改前行号）
- 类型：性能 A/B
- 内容：旧路径每步为取 loss 做 `end_batch + wait_in_flight`，drain 整条流水线 → GPU 在 host 录制期间空转，表现为占用率锯齿。现改为「异步回读槽位 + 非阻塞收割 harvest_loss + 槽位将满/epoch 收尾时阻塞兜底 drain_loss」；注释保留现行机制与「为何 host 不打断录制帧」的理由，删除旧路径描述。

## NaN 跳步的移除（原位置 src/text_train.cpp:1509，整改前行号）
- 类型：删除清单
- 内容：训练循环曾在 host 侧读回 loss 判定 NaN 并跳过该步；现 host 不判定 loss（loss 只在 device 侧异步回读），数值稳定性由 `--max-norm` 梯度裁剪 + 观察 loss 曲线负责。

## layer_bench transB/transA 变体与 plain 共用操作数（原位置 src/layer_bench.cpp:263-264，整改前行号）
- 类型：bug 根因
- 内容：旧版 setup 让 transB/transA 变体与 plain 共用 `(k,n)` 操作数：`k==n` 时方阵形状掩盖错配，`k≠n` 时 backend 返回 K mismatch 而 run 内 `*expected` 不查错 → UB，测出 0.000ms 垃圾时长 / 超物理 GFLOPS。已改为 per-variant 独立 setup + run 查 Result 失败即打印并中止；注释保留现行契约（每变体自建操作数、run 必须查错）与形状错配的坑。
