"""Test strip_markdown / clean_text function."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))

from hf_to_txt import clean_text

tests = [
    # (input, expected_substrings) — 检查核心 markdown 格式是否被移除
    ("### **角色基础信息** **姓名**：塞拉斯",
     ["角色基础信息", "姓名", "塞拉斯"]),
    ("**小瓶数量**：30盎司 = **5瓶** - **小瓶总价**：5瓶",
     ["小瓶数量", "30盎司", "5瓶", "小瓶总价"]),
    ("---### **1. 恋爱**- **学会**",
     ["1.", "恋爱", "学会"]),
    ("答案为：\\boxed{20}",
     ["答案为：20"]),
    ("\\[212 \\, \\text{字/分钟} - 40\\]",
     ["212", "字/分钟", "40"]),
    ("#### **第一幕：金雀花之血（童年）**",
     ["第一幕：金雀花之血（童年）"]),
    ("`print(42)` hello",
     ["print(42)", "hello"]),
    ("1. **学会** 2. **沟通**",
     ["学会", "沟通"]),
    ("\\( C \\) 正方体",
     ["C", "正方体"]),
]

all_pass = True
for i, (inp, substrings) in enumerate(tests):
    result = clean_text(inp)
    fails = [s for s in substrings if s not in result]
    # 检查没有残留的 markdown 标记
    md_residuals = ["**", "###", "####", "\\boxed", "\\text{", "---"]
    found_residuals = [r for r in md_residuals if r in result]

    if not fails and not found_residuals:
        print(f"  PASS #{i+1}: {result[:60]}")
    else:
        all_pass = False
        print(f"  FAIL #{i+1}")
        print(f"    IN:  {inp!r}")
        print(f"    OUT: {result!r}")
        if fails:
            print(f"    Missing: {fails}")
        if found_residuals:
            print(f"    Markdown residuals: {found_residuals}")

if all_pass:
    print("\nAll tests passed!")
else:
    print("\nSome tests failed!")
