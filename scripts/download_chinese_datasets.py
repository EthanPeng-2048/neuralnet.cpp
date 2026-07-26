"""从免费公开数据集下载中文语料。

数据源：
  1. HuggingFace 上的中文数据集（wikipedia_zh、mc4_zh 等）
  2. 中文新闻语料
  3. 中文百科语料

用法：
  python scripts/download_chinese_datasets.py

依赖：pip install datasets (可选，有 fallback)
"""

import urllib.request
import json
import os
import re
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "datasets")
MIN_SENT_LEN = 8
MAX_SENT_LEN = 200
SENT_SPLIT = re.compile(r"[。！？；\n]")


def is_valid_chinese(text: str) -> bool:
    text = text.strip()
    if len(text) < MIN_SENT_LEN or len(text) > MAX_SENT_LEN:
        return False
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if chinese < len(text) * 0.35:
        return False
    if text.startswith(("#", "|", "=", "-", ">", "http")):
        return False
    return True


def split_sentences(text: str) -> list[str]:
    text = text.replace("\r", "").replace("\n", " ").strip()
    parts = SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── 数据源 1：HuggingFace datasets 库 ────────────────────────────
def download_via_datasets_lib():
    """使用 HuggingFace datasets 库下载中文语料。"""
    print("\n📦 数据源 1: HuggingFace datasets 库")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ⚠ 未安装 datasets 库，跳过此数据源")
        print("  💡 安装方法: pip install datasets")
        return []

    # 尝试多个已知可用的中文数据集
    dataset_candidates = [
        ("tattsu/wikipedia_zh", "train"),
        ("CIAO-LM/ChineseWebText", "train"),
        ("donlee/wikitext-103-zh", "train"),
    ]

    sentences = []
    for ds_name, split in dataset_candidates:
        try:
            print(f"  尝试数据集: {ds_name}...")
            ds = load_dataset(ds_name, split=split, streaming=True)
            count = 0
            for item in ds:
                text = item.get("text", "") or item.get("content", "") or ""
                if not text:
                    continue
                for sent in split_sentences(text):
                    if is_valid_chinese(sent):
                        sentences.append(sent)
                        count += 1
                if count >= 15000:
                    break
                if count % 3000 == 0 and count > 0:
                    print(f"    进度: {count} 条")
            if count > 0:
                print(f"  ✓ {ds_name}: {count} 条")
                break
        except Exception as e:
            print(f"  ✗ {ds_name}: {e}")
            continue

    print(f"  ✅ HuggingFace: 共 {len(sentences)} 条")
    return sentences


# ── 数据源 2：GitHub 中文语料仓库 ────────────────────────────────
def download_github_chinese_corpus():
    """从 GitHub 上的公开中文语料仓库下载数据。"""
    print("\n📰 数据源 2: GitHub 中文语料")
    # 多个公开中文语料文件
    urls = [
        ("https://raw.githubusercontent.com/brightmart/nlp_chinese_corpus/master/data/zhidao_test.json", "jsonl"),
        ("https://raw.githubusercontent.com/MorvanZhou/nlp-tutorial/master/data/1.text-classification/cn_sample.txt", "txt"),
    ]

    sentences = []
    for url, fmt in urls:
        try:
            print(f"  下载: {url.split('/')[-1]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")

            if fmt == "jsonl":
                for line in raw.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            text = " ".join(str(v) for v in item.values() if isinstance(v, str))
                        elif isinstance(item, str):
                            text = item
                        else:
                            continue
                        for sent in split_sentences(text):
                            if is_valid_chinese(sent):
                                sentences.append(sent)
                    except json.JSONDecodeError:
                        continue
            else:  # txt
                for sent in split_sentences(raw):
                    if is_valid_chinese(sent):
                        sentences.append(sent)

            print(f"  ✓ {url.split('/')[-1]}: 完成")
        except Exception as e:
            print(f"  ✗ {url.split('/')[-1]}: {e}")
        time.sleep(0.5)

    print(f"  ✅ GitHub 语料: 共 {len(sentences)} 条")
    return sentences


# ── 数据源 3：精选中文段落（硬编码高质量语料）────────────────────
CURATED_PARAGRAPHS = [
    # 科学
    "科学研究表明，人类的大脑每天会产生约六万个想法，其中大部分是无意识的自动思维。正念冥想可以帮助人们觉察这些自动化的思维模式，从而获得更好的心理调控能力。",
    "地球的磁场是由外核中液态铁的对流运动产生的。这个磁场像一个巨大的保护罩，阻挡了来自太阳的高能粒子，保护了地球上的生命。",
    "光合作用是地球上最重要的化学反应之一。绿色植物利用阳光的能量将二氧化碳和水转化为葡萄糖和氧气，为整个食物链提供了能量基础。",
    "人类基因组包含约三十一亿个碱基对，其中只有不到百分之二的序列编码蛋白质。科学家们正在研究这些非编码区域的功能。",
    "量子纠缠现象被爱因斯坦称为幽灵般的超距作用。当两个粒子发生纠缠后，无论它们相距多远，测量其中一个粒子的状态会立即影响另一个。",
    # 历史文化
    "活字印刷术的发明极大地降低了书籍的成本，使知识不再是少数人的特权。毕昇用胶泥制作的活字可以反复使用，这一创新比欧洲的古腾堡早了四百年。",
    "茶马古道是古代中国西南地区最重要的贸易通道之一。马帮沿着这条道路将云南的茶叶运往西藏，同时带回藏区的马匹和药材。",
    "中国古代的科举制度虽然有其局限性，但它打破了贵族对权力的垄断，为平民子弟提供了一条通过读书改变命运的通道。",
    "敦煌壁画历经千年的风沙侵蚀，依然保持着绚丽的色彩。这些壁画不仅是艺术瑰宝，更是研究古代社会生活的珍贵资料。",
    "中国的二十四节气是古人通过长期观察太阳运行规律而总结出来的时间体系。它不仅指导着农业生产，还深刻影响着中国人的饮食起居。",
    # 日常生活
    "每天保持适量的运动对维持身体健康至关重要。世界卫生组织建议成年人每周至少进行一百五十分钟的中等强度有氧运动。",
    "睡眠不足会影响大脑的清除系统，导致代谢废物在脑内积累。长期睡眠不足与阿尔茨海默病的发病风险增加有关。",
    "色彩心理学研究发现，蓝色能让人感到平静和放松，红色则能激发兴奋和警觉。不同的颜色环境会影响人的情绪和行为表现。",
    "人类的味觉可以分辨五种基本味道：酸、甜、苦、咸、鲜。味蕾上的受体蛋白能够识别不同的化学物质，将信号传递给大脑进行处理。",
    "深呼吸能够激活副交感神经系统，降低心率和血压，帮助身体从压力状态中恢复过来。这也是为什么人们紧张时会不自觉地深呼吸。",
    # 地理自然
    "热带雨林被称为地球之肺，它们不仅产生大量氧气，还是数百万种动植物的家园。然而，全球每年有大片热带雨林被砍伐。",
    "珊瑚礁虽然只占海洋面积的不到百分之一，却养育了超过四分之一的海洋物种。海水温度上升导致的珊瑚白化正在威胁这些珍贵的生态系统。",
    "中国的黄土高原是世界上最大的黄土沉积区。这些黄土是数百万年来从沙漠地区吹来的细沙沉积而成的。",
    "北极光是太阳带电粒子与地球大气层中的原子碰撞时产生的自然现象。在高纬度地区，夜空中可以看到绿色、紫色和红色的光幕。",
    "河流的三角洲是河流入海时因流速减慢、泥沙沉积而形成的扇形地貌。长江三角洲是中国最富庶的地区之一。",
    # 数学与逻辑
    "质数在数学中扮演着重要角色。任何一个大于一的整数都可以唯一地分解为质数的乘积，这就是算术基本定理。",
    "斐波那契数列在自然界中随处可见。向日葵的种子排列、松果的鳞片分布、贝壳的螺旋结构都遵循着这个数学规律。",
    "概率论告诉我们，随机事件在大量重复实验中会呈现出稳定的统计规律。这就是大数定律的核心思想。",
    "二进制是计算机的基础语言。所有的文字、图片、音乐在计算机中都是以零和一的组合形式存储和处理的。",
    "博弈论研究的是理性决策者之间的策略互动。囚徒困境揭示了一个深刻的道理：个体的理性选择可能导致集体的非理性结果。",
]


def fetch_curated_paragraphs():
    """返回硬编码的精选语料。"""
    print("\n📖 数据源 3: 精选中文段落")
    sentences = []
    for para in CURATED_PARAGRAPHS:
        for sent in split_sentences(para):
            if is_valid_chinese(sent):
                sentences.append(sent)
    print(f"  ✅ 精选段落: 共 {len(sentences)} 条")
    return sentences


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_sentences = []
    seen = set()

    # 1. HuggingFace
    hf_sents = download_via_datasets_lib()
    for s in hf_sents:
        key = s[:40]
        if key not in seen:
            seen.add(key)
            all_sentences.append(s)

    # 2. GitHub 中文语料
    gh_sents = download_github_chinese_corpus()
    for s in gh_sents:
        key = s[:40]
        if key not in seen:
            seen.add(key)
            all_sentences.append(s)

    # 3. 精选语料
    curated = fetch_curated_paragraphs()
    for s in curated:
        key = s[:40]
        if key not in seen:
            seen.add(key)
            all_sentences.append(s)

    # 写入文件
    out_path = os.path.join(OUT_DIR, "chinese_datasets_corpus.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for sent in all_sentences:
            f.write(sent + "\n")

    print(f"\n{'='*60}")
    print(f"✅ 总计采集: {len(all_sentences)} 条")
    print(f"📝 输出文件: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
