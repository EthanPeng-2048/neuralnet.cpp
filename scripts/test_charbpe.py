"""测试 charbpe 分词器质量和压缩率。"""
import json
import sys

# 加载 JSON 词表
with open("openwebtext_charbpe_8k.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"分词器类型: {data['type']}")
print(f"vocab_size: {data['vocab_size']}")
print(f"合并规则数: {len(data['merges'])}")

# 测试样本
samples = [
    "former secretary of state hillary clinton meets voters at a campaign rally in st. louis on saturday.",
    "the quick brown fox jumps over the lazy dog.",
    "researchers have discovered a new species of dinosaur in argentina.",
    "the stock market rallied today as investors reacted to the latest economic data.",
    "the opinions expressed by columnists are their own and do not represent the views of townhall.com.",
]

# 从 vocab 构建 id → token 的查找表
# vocab 是 {id_str: hex_bytes} 格式
# 特殊 token (0-3): 直接文本
# 字节 token (4-259): 单字节
# 其他: hex 编码的 UTF-8 字节
id_to_token = {}
for sid, hex_val in data["vocab"].items():
    idx = int(sid)
    try:
        if idx < 4:
            # 特殊 token
            id_to_token[idx] = hex_val
        elif idx < 260:
            # 单字节
            b = int(hex_val, 16)
            id_to_token[idx] = bytes([b])
        else:
            # 多字节 UTF-8
            id_to_token[idx] = bytes.fromhex(hex_val)
    except:
        id_to_token[idx] = hex_val

# 构建合并规则: {id_a: {id_b: merged_id}}
merge_map = {}
for a, b, merged in data["merges"]:
    if a not in merge_map:
        merge_map[a] = {}
    merge_map[a][b] = merged

def encode(text):
    """简单编码器: 按字符拆分为字节 ID, 然后应用 BPE 合并."""
    # 将文本转为字节, 每个字节映射到 ID
    raw_bytes = text.encode("utf-8")
    ids = [b + 4 for b in raw_bytes]  # 字节 ID 从 4 开始
    
    # 应用合并规则
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(ids) - 1:
            a, b = ids[i], ids[i+1]
            if a in merge_map and b in merge_map[a]:
                ids[i] = merge_map[a][b]
                ids.pop(i+1)
                changed = True
                # 合并后可能可以和前面的继续合并
                if i > 0:
                    i -= 1
            else:
                i += 1
    
    return ids

def decode(ids):
    """简单解码器: ID → 字节 → 字符串."""
    result = b""
    for idx in ids:
        if idx in id_to_token:
            tok = id_to_token[idx]
            if isinstance(tok, str):
                result += tok.encode("utf-8")
            elif isinstance(tok, bytes):
                result += tok
    return result.decode("utf-8", errors="replace")

# 测试
print("\n" + "=" * 60)
print("分词器质量测试")
print("=" * 60)

for sample in samples:
    ids = encode(sample)
    decoded = decode(ids)
    ratio = len(sample.encode("utf-8")) / max(len(ids), 1)
    print(f"\n原文: {sample[:60]}...")
    print(f"编码: {len(ids)} tokens, 压缩率: {ratio:.2f} 字节/token")
    print(f"解码: {decoded[:60]}...")
    match = sample == decoded
    print(f"无损: {'✓' if match else '✗'}")
    
print("\n" + "=" * 60)
print("建议:")
print(f"  词表大小: {data['vocab_size']}")
print(f"  合并规则: {len(data['merges'])}")
print(f"  适合: 小写英文预训练")