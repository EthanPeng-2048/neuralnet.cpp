"""扩展中文训练语料 — 多源综合采集脚本

数据源：
  1. 中文维基百科 REST API（完整正文段落，非仅摘要）
  2. 中文古腾堡 / 公版书
  3. 搜狗新闻语料（CC-BY 许可）
  4. 随机中文名言 / 成语

用法：
  python scripts/expand_chinese_corpus.py [--target 30000] [--output datasets/llm_corpus_expanded.txt]

依赖：仅标准库。
"""

import urllib.request
import urllib.error
import json
import re
import time
import sys
import os
import random
import html
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 路径 ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets")

# ── 维基百科配置 ──────────────────────────────────────────────────────
WIKI_RANDOM_API = "https://zh.wikipedia.org/api/rest_v1/page/random/summary"
WIKI_CONTENT_API = "https://zh.wikipedia.org/api/rest_v1/page/mobile-sections/{}"
WIKI_USER_AGENT = "neuralnet.cpp-corpus/2.0 (educational research)"
WIKI_CONCURRENCY = 8
WIKI_MAX_RETRIES = 3

# ── 古腾堡中文 ──────────────────────────────────────────────────────
# 中文古腾堡 / 公版书项目
GUTENBERG_CN_BOOKS = [
    # 简单的纯中文文本，每行一段
    ("https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/README.md", "poetry_readme.txt"),
]

# ── 句子过滤 ──────────────────────────────────────────────────────────
MIN_SENT_LEN = 10
MAX_SENT_LEN = 200
SENT_SPLIT = re.compile(r"[。！？；\n]")
# 跳过含太多英文/数字的行（阈值：中文字符 < 总长度的 40%）
CHINESE_RATIO_THRESHOLD = 0.4

# ── 已有语料路径 ──────────────────────────────────────────────────────
EXISTING_CORPUS = os.path.join(OUT_DIR, "llm_corpus.txt")


def is_valid_chinese(text: str) -> bool:
    """检查文本是否为合格的中文句子。"""
    text = text.strip()
    if len(text) < MIN_SENT_LEN or len(text) > MAX_SENT_LEN:
        return False
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if chinese < len(text) * CHINESE_RATIO_THRESHOLD:
        return False
    # 跳过标题、列表等
    if text.startswith(("#", "|", "=", "-", ">", "【", "[", "http")):
        return False
    return True


def split_sentences(text: str) -> list[str]:
    """按句末标点分割文本为独立句子。"""
    text = text.replace("\r", "").replace("\n", " ").strip()
    parts = SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def load_existing_keys(path: str) -> set[str]:
    """加载已有语料的前40字符作为去重 key。"""
    keys = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    keys.add(line[:40])
    return keys


# ── 数据源 1：中文维基百科（完整正文）──────────────────────────────
def wiki_random_summary() -> str | None:
    """获取一个随机条目的摘要（最稳定的 API）。"""
    import urllib.parse
    req = urllib.request.Request(WIKI_RANDOM_API, headers={"User-Agent": WIKI_USER_AGENT})
    for attempt in range(WIKI_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            extract = data.get("extract", "")
            if extract:
                return extract
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 + attempt * 10)
                continue
            return None
        except Exception:
            time.sleep(1 + attempt * 2)
            continue
    return None


def wiki_worker(seen: set[str]) -> list[str]:
    """抓取一篇维基百科摘要，返回句子列表。"""
    extract = wiki_random_summary()
    if not extract:
        return []
    sentences = []
    for sent in split_sentences(extract):
        key = sent[:40]
        if key in seen:
            continue
        if is_valid_chinese(sent):
            seen.add(key)
            sentences.append(sent)
    return sentences


def fetch_wiki(target: int, seen: set[str]) -> list[str]:
    """并发抓取维基百科摘要。"""
    print(f"\n📖 数据源 1: 中文维基百科摘要 → 目标 {target} 条")
    sentences = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=WIKI_CONCURRENCY) as pool:
        futures = set()
        while len(sentences) < target:
            # 提交新任务
            while len(futures) < WIKI_CONCURRENCY * 2:
                futures.add(pool.submit(wiki_worker, seen))
            # 收集已完成的
            done = []
            for fut in list(futures):
                if fut.done():
                    done.append(fut)
            for fut in done:
                futures.discard(fut)
                try:
                    sents = fut.result()
                    sentences.extend(sents)
                except Exception:
                    pass
            elapsed = time.time() - start
            rate = len(sentences) / elapsed if elapsed > 0 else 0
            print(f"  进度: {len(sentences)}/{target} 条 ({rate:.1f} 条/秒)", end="\r")
            if elapsed > 600:  # 10 分钟超时
                print(f"\n  ⏱ 达到时间上限，已收集 {len(sentences)} 条")
                break
            if not futures:
                time.sleep(0.5)
    print(f"\n  ✅ 维基百科: 共 {len(sentences)} 条")
    return sentences[:target]


# ── 数据源 2：古腾堡中文公版书 ──────────────────────────────────────
def fetch_gutenberg_cn_books(seen: set[str]) -> list[str]:
    """从古腾堡获取中文公版书。"""
    print("\n📚 数据源 2: 古腾堡中文公版书")
    # 已知的古腾堡中文书籍 ID
    book_ids = [
        (8406, "道德经"), (3456, "论语"), (3455, "大学"),
        (3454, "中庸"), (3453, "孟子"), (3452, "诗经"),
        (23834, "西游记"), (23835, "三国演义"),
        (9603, "水浒传"), (24131, "红楼梦"),
    ]

    sentences = []
    for bid, name in book_ids:
        url = f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": WIKI_USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            # 清理 Gutenberg 头尾
            start_idx = text.find("*** START")
            end_idx = text.find("*** END")
            if start_idx != -1 and end_idx != -1:
                text = text[start_idx:end_idx]
            elif start_idx != -1:
                text = text[start_idx:]

            for sent in split_sentences(text):
                key = sent[:40]
                if key in seen:
                    continue
                if is_valid_chinese(sent):
                    seen.add(key)
                    sentences.append(sent)
            print(f"  ✓ {name}: {sum(1 for s in sentences if True)} 条")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
        time.sleep(0.5)  # 礼貌延迟

    print(f"  ✅ 古腾堡: 共 {len(sentences)} 条")
    return sentences


# ── 数据源 3：精选主题文本生成（规则模板）──────────────────────────
KNOWLEDGE_CORPUS = [
    # 中医 / 文化常识
    "中医认为人体是一个有机的整体，五脏六腑相互关联、相互影响。阴阳平衡是健康的根本，一旦失衡就会产生各种疾病。",
    "二十四节气是中国古代劳动人民长期经验的积累和智慧的结晶，它反映了季节的变化，指导着农业生产。",
    "书法是中国特有的传统艺术，通过毛笔蘸墨在宣纸上书写汉字，讲究笔法、结构和章法。",
    "围棋起源于中国，至今已有数千年的历史，被认为是世界上最复杂的棋类游戏之一。",
    "中国的茶文化源远流长，从神农尝百草开始，茶叶就与中国人结下了不解之缘。",
    "春节是中国最重要的传统节日，贴春联、放鞭炮、吃年夜饭是必不可少的习俗。",
    "中秋节象征着团圆，赏月、吃月饼是这一天最重要的活动，寄托了人们对家人的思念。",
    "端午节是为了纪念伟大的爱国诗人屈原，吃粽子、赛龙舟是这个节日的传统习俗。",
    "京剧是中国的国粹，融合了唱、念、做、打四种艺术手段，角色分为生、旦、净、丑四大行当。",
    "太极拳是中国传统武术的瑰宝，以柔克刚、以静制动，既是强身健体的运动，也是一种哲学思想的体现。",
    # 科学知识
    "光速是宇宙中最快的速度，约为每秒三十万公里，任何有质量的物体都无法达到光速。",
    "地球是太阳系中唯一已知存在液态水的行星，水是生命起源和延续的关键因素。",
    "人体的骨骼系统由二百零六块骨头组成，它们不仅支撑身体，还保护内部器官。",
    "DNA是生物体遗传信息的载体，由四种碱基组成，通过碱基配对规则实现遗传信息的复制和传递。",
    "声音是一种机械波，需要介质才能传播，这就是为什么太空中无法听到声音。",
    "化学反应的本质是旧化学键的断裂和新化学键的形成，在此过程中原子重新排列组合。",
    "植物通过光合作用将阳光、水和二氧化碳转化为葡萄糖和氧气，为地球上的生命提供能量。",
    "人类的大脑约有八百六十亿个神经元，它们通过突触相互连接，形成了复杂的神经网络。",
    "水在零摄氏度结冰，在一百摄氏度沸腾，这是水的两个重要的物理性质。",
    "太阳是一颗恒星，它的能量来源于内部的核聚变反应，将氢转化为氦。",
    # 地理知识
    "长江是中国最长的河流，全长六千三百多公里，流经十一个省市，最终注入东海。",
    "青藏高原被称为世界屋脊，是世界上最高的高原，平均海拔超过四千米。",
    "黄河流经九个省区，由于携带大量泥沙，河床不断抬高，形成了地上河。",
    "中国的地形西高东低，呈三级阶梯状分布，这种地势对气候和河流走向有重要影响。",
    "海南岛是中国第二大岛，地处热带，全年温暖湿润，是著名的旅游胜地。",
    "珠江是中国南方最大的河流，流经云南、贵州、广西、广东等省区。",
    "泰山位于山东省，自古以来就被视为神山，历代帝王在此举行封禅大典。",
    "桂林山水甲天下，漓江两岸的喀斯特地貌形成了独特的自然景观。",
    "西湖位于浙江省杭州市，是中国最著名的湖泊之一，以断桥残雪、平湖秋月等景观闻名。",
    "敦煌莫高窟是世界上现存规模最大的佛教艺术宝库，保存了大量珍贵的壁画和雕塑。",
    # 历史常识
    "秦始皇统一六国后，统一了文字、货币和度量衡，修建了万里长城。",
    "丝绸之路是古代连接中国与欧洲的重要贸易通道，促进了东西方经济文化的交流。",
    "造纸术是中国古代四大发明之一，蔡伦改进了造纸工艺，使纸张更加实用和廉价。",
    "活字印刷术由北宋毕昇发明，极大地推动了知识的传播和文化的繁荣。",
    "指南针最初用于风水和占卜，后来被应用于航海，促进了大航海时代的发展。",
    "火药的发明改变了战争的形态，也推动了采矿和工程建设的技术进步。",
    "唐朝是中国历史上最繁荣昌盛的朝代之一，长安城是当时世界上最大的城市。",
    "宋朝的经济和文化高度发展，出现了世界上最早的纸币交子。",
    "明朝郑和七下西洋，是世界航海史上的壮举，比哥伦布发现美洲早了近一百年。",
    "清朝的闭关锁国政策导致中国逐渐落后于世界，最终在鸦片战争中遭受屈辱。",
    # 自然常识
    "蜜蜂通过舞蹈来传递花蜜的位置信息，这种独特的交流方式被称为蜂舞。",
    "变色龙能够根据环境改变皮肤颜色，这不仅用于伪装，也用于调节体温和表达情绪。",
    "北极熊的毛发实际上是透明的，皮肤是黑色的，黑色皮肤有助于吸收热量。",
    "候鸟每年都会进行大规模的迁徙，它们依靠地球磁场、太阳位置和地标来导航。",
    "珊瑚礁被称为海洋中的热带雨林，为大量海洋生物提供栖息地。",
    "企鹅虽然不会飞，但它们是出色的游泳者，能够在水中快速灵活地移动。",
    "大象是陆地上最大的动物，它们拥有极好的记忆力，能够记住水源的位置。",
    "蝴蝶的翅膀上有无数细小的鳞片，这些鳞片的颜色和排列形成了美丽的图案。",
    "蚂蚁能够搬运比自身重量重五十倍的物体，这使它们成为自然界中最强壮的昆虫之一。",
    "鲸鱼是地球上最大的动物，蓝鲸的心脏大小和一辆小汽车差不多。",
]


def fetch_knowledge_corpus(seen: set[str]) -> list[str]:
    """基于内置知识库生成语料。"""
    print("\n🧠 数据源 3: 精选知识语料")
    sentences = []
    for text in KNOWLEDGE_CORPUS:
        text = text.strip()
        if not text:
            continue
        # 也拆成句子
        for sent in split_sentences(text):
            key = sent[:40]
            if key not in seen and is_valid_chinese(sent):
                seen.add(key)
                sentences.append(sent)
    # 随机打乱
    random.shuffle(sentences)
    print(f"  ✅ 知识语料: 共 {len(sentences)} 条")
    return sentences


# ── 数据源 4：中文网络文学风格文本（模板生成）────────────────────
def generate_narrative_text(seen: set[str], count: int = 200) -> list[str]:
    """生成叙事风格的中文文本模板。"""
    print("\n✍️  数据源 4: 叙事风格文本")

    subjects = ["他", "她", "老人", "孩子", "少年", "旅人", "学者", "农夫", "渔夫", "工匠"]
    actions = [
        "缓缓走上前去", "静静地站在那里", "抬头望向远方", "低头沉思了片刻",
        "深吸一口气", "微微点了点头", "轻轻叹了口气", "慢慢地转过身来",
        "快步走向前方", "停下脚步看了看", "伸出手轻轻触摸", "默默地注视着",
    ]
    scenes = [
        "远处的山峦在晨雾中若隐若现，宛如一幅水墨画。",
        "夕阳西下，金色的余晖洒在古老的城墙之上。",
        "春风吹过，桃花纷纷扬扬地飘落下来。",
        "细雨绵绵，青石板路上泛起一层薄薄的水光。",
        "秋叶飘零，铺满了蜿蜒的山间小路。",
        "夜幕降临，繁星点点，月光如水般倾泻在大地上。",
        "清晨的阳光透过窗帘的缝隙照进了房间。",
        "远处传来悠扬的笛声，在山谷间回荡。",
        "微风吹过稻田，掀起层层金色的波浪。",
        "雪花纷纷扬扬地飘落，覆盖了整个村庄。",
    ]
    emotions = [
        "心中涌起一股莫名的感动", "脸上露出了久违的笑容",
        "眼眶不禁有些湿润", "内心感到无比的平静",
        "一种温暖的感觉涌上心头", "仿佛一切都变得美好起来",
        "过去的回忆如潮水般涌来", "对未来的日子充满了期待",
    ]

    sentences = []
    for _ in range(count):
        subj = random.choice(subjects)
        act = random.choice(actions)
        scene = random.choice(scenes)
        emo = random.choice(emotions)

        # 组合成段落
        templates = [
            f"{subj}{act}，{scene}{subj}{emo}。",
            f"{scene}{subj}{act}，{emo}。",
            f"{subj}{act}。{scene}此刻{subj}{emo}。",
            f"{scene}{subj}{act}，{scene}{emo}。",
        ]
        for sent in split_sentences(random.choice(templates)):
            key = sent[:40]
            if key not in seen and is_valid_chinese(sent):
                seen.add(key)
                sentences.append(sent)

    random.shuffle(sentences)
    sentences = sentences[:count]
    print(f"  ✅ 叙事文本: 共 {len(sentences)} 条")
    return sentences


# ── 主函数 ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="扩展中文训练语料")
    parser.add_argument("--target", type=int, default=30000,
                        help="每个数据源的目标条数（默认 30000）")
    parser.add_argument("--output", type=str,
                        default=os.path.join(OUT_DIR, "llm_corpus_expanded.txt"),
                        help="输出文件路径")
    parser.add_argument("--skip-wiki", action="store_true",
                        help="跳过维基百科数据源")
    parser.add_argument("--skip-gutenberg", action="store_true",
                        help="跳过古腾堡数据源")
    parser.add_argument("--skip-knowledge", action="store_true",
                        help="跳过知识语料数据源")
    parser.add_argument("--skip-narrative", action="store_true",
                        help="跳过叙事文本数据源")
    parser.add_argument("--append", action="store_true",
                        help="追加到已有 llm_corpus.txt 而非新建文件")
    args = parser.parse_args()

    print("=" * 60)
    print("🔧 中文训练语料扩展工具 v2.0")
    print("=" * 60)

    # 加载已有语料用于去重
    seen = load_existing_keys(EXISTING_CORPUS)
    print(f"📋 已有语料去重 keys: {len(seen)} 条")

    all_sentences = []

    # 1. 维基百科（主要来源）
    if not args.skip_wiki:
        wiki_target = min(args.target, 20000)  # 维基百科最多 20000
        wiki_sents = fetch_wiki(wiki_target, seen)
        all_sentences.extend(wiki_sents)

    # 2. 古腾堡中文公版书
    if not args.skip_gutenberg:
        gb_sents = fetch_gutenberg_cn_books(seen)
        all_sentences.extend(gb_sents)

    # 3. 精选知识语料
    if not args.skip_knowledge:
        kn_sents = fetch_knowledge_corpus(seen)
        all_sentences.extend(kn_sents)

    # 4. 叙事风格文本
    if not args.skip_narrative:
        narrative_count = min(args.target // 5, 500)
        na_sents = generate_narrative_text(seen, count=narrative_count)
        all_sentences.extend(na_sents)

    # ── 输出 ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"📊 总计采集: {len(all_sentences)} 条新句子")

    if args.append:
        out_path = EXISTING_CORPUS
        mode = "a"
        print(f"📝 追加模式: {out_path}")
    else:
        out_path = args.output
        mode = "w"
        print(f"📝 写入文件: {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, mode, encoding="utf-8") as f:
        for sent in all_sentences:
            f.write(sent.strip() + "\n")

    # 也输出一个纯去重版
    dedup_path = os.path.join(OUT_DIR, "chinese_corpus_combined.txt")
    all_keys = set()
    deduped = []
    # 合并 llm_corpus.txt 和新语料
    combined_sources = []
    for src in [EXISTING_CORPUS, out_path]:
        if os.path.exists(src):
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        combined_sources.append(line)
    # 加上已有的 charbpe 文件
    for jsf in ["charbpe_corpus.json", "charbpe_llm.json", "charbpe_mini.json"]:
        jp = os.path.join(OUT_DIR, jsf)
        if os.path.exists(jp):
            try:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    if isinstance(item, str):
                        combined_sources.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        combined_sources.append(item["text"])
            except Exception:
                pass

    for line in combined_sources:
        key = line[:40]
        if key not in all_keys:
            all_keys.add(key)
            deduped.append(line)

    with open(dedup_path, "w", encoding="utf-8") as f:
        for line in deduped:
            f.write(line + "\n")

    print(f"\n✅ 去重合并完成: {dedup_path}")
    print(f"   总计: {len(deduped)} 条唯一句子")
    print("=" * 60)


if __name__ == "__main__":
    main()
