import json
d = json.load(open('gpt_vocab.json', encoding='utf-8'))
v = d['vocab']
ascii_start = d['ascii_start']
print(f"词表大小: {len(v)}, ASCII起始ID: {ascii_start}")
print(f"'the' word ID: {v.get('the', 'N/A')}")
print(f"'a' word ID: {v.get('a', 'N/A')}")
print(f"'I' word ID: {v.get('I', 'N/A')}")
print(f"chr(97)='a' ID: {v.get(chr(97), 'N/A')}")
print(f"'a'是词? {v.get('a', 999) < ascii_start}")
print()

# 编码解码测试
text = "Alice was a very good girl, she said hello"
print(f"原文: {text}")

tokens = []
for word in text.split():
    if word in v:
        tokens.append(v[word])
    else:
        for c in word:
            tokens.append(v.get(c, 0))
print(f"Tokens: {tokens}")

# 解码
rev = {val: key for key, val in v.items()}
parts = []
for t in tokens:
    tok = rev.get(t, '<unk>')
    if t >= 3 and t < ascii_start:
        if parts: parts.append(' ')
        parts.append(tok)
    else:
        parts.append(tok)
result = ''.join(parts)
print(f"解码: {result}")
print(f"匹配: {result == text}")
