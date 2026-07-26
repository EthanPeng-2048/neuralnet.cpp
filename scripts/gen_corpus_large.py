"""用本地 llama-server (OpenAI 兼容接口) 生成多样化中文语料。

每行一段流畅的中文书面语，用于训练"说人话"的 GPT。
依赖: openai>=1.0
"""
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────
BASE_URL = "http://localhost:18080/v1"
API_KEY = "Ethan_2048"
MODEL = "local-model"  # llama-server 忽略具体值，但 openai 库要求必填
OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets",
    "llm_corpus.txt",
)
CONCURRENCY = 8
SAMPLES_PER_STYLE = 3  # 每个（主题×风格）组合生成 3 段
MIN_LEN = 10
MAX_LEN = 300
# 追加模式：不覆盖已有语料
APPEND = True

# ── 写作风格维度 ────────────────────────────────────────────────
WRITING_STYLES = [
    "像跟朋友聊天一样娓娓道来",
    "用讲道理的方式认真解释",
    "从一个亲身经历或见闻说起",
    "用打比方或类比让读者秒懂",
    "先抛出一个常见误解再纠正",
    "用简洁干练的语言直接说明",
    "带点幽默和调侃的语气",
    "引用一句俗语或老话然后展开",
]

# ── 中文句式结构 ──────────────────────────────────────────────────
SENTENCE_PATTERNS = [
    "因果关系（因为…所以…）",
    "转折关系（虽然…但是…）",
    "递进关系（不仅…而且…）",
    "假设关系（如果…就…）",
    "反问句（难道不是…吗？）",
    "感叹句（真是…啊！）",
    "对比句（和…相比，…更…）",
    "总分结构（总的来说…具体来说…）",
    "条件关系（只要…就…）",
    "并列关系（一边…一边…）",
]

# ── 生活常识主题（口语化、接地气）──────────────────────────────
TOPICS = [
    # ── 做饭与饮食 ──
    "怎么做红烧肉才好吃",
    "煮饺子什么时候加凉水",
    "炒菜时油温怎么判断",
    "大米怎么存放不容易生虫",
    "隔夜菜到底能不能吃",
    "微波炉热饭的正确方法",
    "什么时候该放盐才对",
    "切洋葱不流泪的小窍门",
    "怎么挑选新鲜的西瓜",
    "煮面条怎样才能不粘锅",
    "蒸鸡蛋羹怎么做才嫩滑",
    "豆腐怎么做才入味",
    "泡发干货的正确方法",
    "肉馅怎么调才好吃",
    "醋除了调味还能干什么",
    "厨房去油污的小妙招",
    "怎样泡茶才好喝",
    "喝牛奶的正确时间",
    "空腹不能吃什么",
    "水果到底饭前吃还是饭后吃",
    # ── 出行与交通 ──
    "下雨天开车要注意什么",
    "坐长途汽车怎么不晕车",
    "骑电动车安全注意事项",
    "怎么坐地铁最方便",
    "堵车的时候怎么办",
    "自驾游出发前要检查什么",
    "飞机上不能带什么东西",
    "打车遇到绕路怎么办",
    "走路快和走路慢哪个健康",
    "等红灯的时候该做什么",
    # ── 健康与养生 ──
    "感冒了应该多喝水还是多睡觉",
    "颈椎疼是怎么回事",
    "每天走多少步最合适",
    "熬夜对身体到底有多大伤害",
    "眼睛干涩怎么办",
    "嗓子疼吃什么好得快",
    "腰酸背痛怎么缓解",
    "失眠的人该怎么调整",
    "体检报告怎么看",
    "血压高的人要注意什么",
    "老年人怎么锻炼才安全",
    "小孩子发烧怎么处理",
    "跑步前后要做什么",
    "久坐对身体有什么危害",
    "怎么保护牙齿才对",
    "夏天怎么防止中暑",
    "冬天手脚冰凉怎么办",
    "头发掉得多是什么原因",
    "吃什么东西能提高免疫力",
    "心情不好的时候怎么调节",
    # ── 家居与生活 ──
    "新房子装修完多久能住",
    "家里总是有异味怎么办",
    "衣服上的各种污渍怎么洗",
    "洗衣机怎么清洗才干净",
    "空调多久清洗一次比较好",
    "冰箱里什么东西不能放",
    "厨房下水道堵了怎么通",
    "墙上有霉斑怎么处理",
    "木地板怎么保养",
    "家里太干燥怎么办",
    "WiFi信号不好怎么增强",
    "手机充电的正确方式",
    "电表怎么看用了多少电",
    "水龙头滴水怎么修",
    "暖气不热是什么原因",
    "窗帘怎么选才合适",
    "收纳整理有什么好方法",
    "搬家打包有什么技巧",
    "垃圾分类到底怎么分",
    "快递包装盒怎么处理",
    # ── 人际关系 ──
    "怎么跟不熟的人找话题聊天",
    "同事之间怎么相处",
    "吵架之后怎么和好",
    "怎么拒绝别人的请求",
    "朋友借钱不还怎么办",
    "遇到不讲理的人怎么办",
    "怎么跟父母沟通",
    "孩子不听话怎么教育",
    "夫妻之间怎么保持新鲜感",
    "怎么交到真正的朋友",
    "被误解了怎么解释",
    "怎么安慰伤心的人",
    "领导批评你的时候怎么办",
    "室友关系怎么处理",
    "邻里之间怎么和睦相处",
    # ── 花钱与理财 ──
    "怎么存钱最划算",
    "网购怎么避免被骗",
    "超市购物有什么省钱技巧",
    "租房子要注意什么",
    "怎么判断东西值不值得买",
    "月光族怎么开始理财",
    "信用卡怎么用才不会被坑",
    "打折促销的套路有哪些",
    "买菜什么时候最便宜",
    "怎么记账才能坚持下来",
    "保险到底有没有必要买",
    "公积金怎么用",
    "交水电费有什么方便的方式",
    "跳槽的时候怎么谈工资",
    "怎么规划旅游预算",
    # ── 四季与天气 ──
    "春天穿什么衣服合适",
    "夏天怎么防蚊子",
    "秋天干燥皮肤怎么办",
    "冬天取暖要注意安全",
    "梅雨天衣服怎么晾干",
    "回南天怎么除湿",
    "雾霾天出门要戴什么",
    "大风天出行要注意什么",
    "下雪天路面滑怎么开车",
    "台风来了要准备什么",
    # ── 节日与习俗 ──
    "过年为什么要贴春联",
    "中秋节除了吃月饼还能干什么",
    "端午节为什么要吃粽子",
    "元宵节和汤圆有什么关系",
    "清明节扫墓有什么讲究",
    "重阳节为什么要登高",
    "腊八节为什么要喝腊八粥",
    "压岁钱有什么寓意",
    "过年怎么拜年才得体",
    "结婚随份子钱给多少合适",
    # ── 养宠物 ──
    "第一次养狗要准备什么",
    "猫咪为什么喜欢磨爪子",
    "怎么教小狗定点上厕所",
    "鱼缸多久换一次水",
    "鸟叫声突然变少了怎么回事",
    "兔子能不能吃胡萝卜",
    "遛狗为什么要牵绳",
    "宠物掉毛怎么办",
    "小猫打疫苗的时间表",
    "养仓鼠需要注意什么",
    # ── 育儿与教育 ──
    "孩子几岁开始学说话算正常",
    "怎么陪孩子玩才有意义",
    "小孩不爱吃饭怎么办",
    "孩子怕黑怎么引导",
    "怎么给孩子讲睡前故事",
    "孩子撒谎了怎么教育",
    "小学生要不要上辅导班",
    "怎么培养孩子的阅读习惯",
    "青春期孩子叛逆怎么办",
    "孩子沉迷手机怎么限制",
    # ── 穿衣与打扮 ──
    "白衣服发黄怎么洗白",
    "皮鞋怎么保养才亮",
    "不同场合怎么穿衣服",
    "眼镜框怎么选才好看",
    "怎么叠衣服最节省空间",
    "围巾怎么搭配好看",
    "牛仔裤怎么洗不褪色",
    "运动鞋怎么清洗",
    "正式场合穿什么不犯错",
    "衣服起球了怎么处理",
    # ── 工具与修理 ──
    "扳手和钳子有什么区别",
    "家里的工具箱要备哪些东西",
    "水龙头漏水自己怎么修",
    "灯泡坏了怎么换",
    "马桶堵了怎么疏通",
    "墙面裂缝怎么修补",
    "门锁不好使怎么修",
    "抽油烟机怎么拆洗",
    "水管爆了怎么紧急处理",
    "电闸跳闸了怎么恢复",
    # ── 中文语法与表达 ──
    "的地得有什么区别",
    "为什么不能说\"我走了先\"",
    "量词的使用有什么规律",
    "怎么用\"虽然\"和\"但是\"造句",
    "什么时候用\"的\"什么时候用\"地\"",
    "一句话怎么加上合适的连接词",
    "怎么把长句子写通顺",
    "中文的\"了\"到底什么意思",
    "怎么用\"把\"字句和\"被\"字句",
    "为什么\"差不多\"和\"差很多\"意思相反",
    "语气词\"呢\"\"啊\"\"吧\"怎么用",
    "怎么用反问句加强语气",
    "中文数字的读法规则",
    "成语和四字词语有什么区别",
    "怎么用\"既…又…\"\"不但…还…\"",
    "书面语和口语的区别在哪里",
    "怎么用\"如果\"\"要是\"\"假如\"表达假设",
    "\"但是\"和\"可是\"有什么区别",
    "中文里\"才\"和\"就\"怎么用",
    "怎样写一段通顺的话",
    # ── 常识与道理 ──
    "为什么早上起来要喝水",
    "饭后百步走活到九十九是真的吗",
    "吃太辣的东西胃疼怎么办",
    "为什么有的人晕车有的人不晕",
    "打哈欠为什么会传染",
    "为什么人老了头发会变白",
    "热水和凉水哪个解渴",
    "为什么睡觉会做梦",
    "打嗝止不住怎么办",
    "为什么有些人怕痒有些人不怕",
    "蹲久了站起来为什么会头晕",
    "红药水和碘伏有什么区别",
    "蚊子为什么爱叮某些人",
    "天冷鼻涕为什么会流",
    "小孩和大人谁更容易感冒",
]

# ── 客户端 ────────────────────────────────────────────────
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120)


def make_prompt(topic: str, style_idx: int) -> str:
    style = WRITING_STYLES[style_idx % len(WRITING_STYLES)]
    return (
        f"请用生活化的语言描述「{topic}」，{style}。\n"
        "要求：\n"
        "1. 直接输出内容，不要带标题、编号、markdown 符号\n"
        "2. 内容必须是简体中文，自然流畅\n"
        "3. 长度控制在 30-150 字之间\n"
        "4. 使用自然的中文句式（因果、转折、递进、反问等），不要用模板化的表达\n"
        "5. 只输出这一段话，不要分多行，不要任何解释或前言"
    )


def clean_line(text: str) -> str:
    if not text:
        return ""
    # 合并换行
    line = re.sub(r"[\r\n]+", " ", text).strip()
    # 去引号包裹
    line = line.strip('"').strip("「」").strip()
    # 去思考标记
    line = re.sub(r"<think>[\s\S]*?</think>", "", line).strip()
    return line


def gen_one(topic: str, style_idx: int, sample_idx: int) -> str | None:
    prompt = make_prompt(topic, style_idx)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
            max_tokens=400,
            stream=False,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = resp.choices[0].message.content or ""
        return clean_line(text)
    except Exception as e:
        print(f"  [失败] {topic}[{style_idx},{sample_idx}]: {e}", file=sys.stderr)
        return None


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    # 二维组合：每个主题 × 每种风格 × 重复次数
    tasks = [
        (t, si, ri)
        for t in TOPICS
        for si in range(len(WRITING_STYLES))
        for ri in range(SAMPLES_PER_STYLE)
    ]
    random.shuffle(tasks)  # 随机打乱顺序，避免同一主题连续生成

    # 追加模式：加载已有语料做去重
    seen: set[str] = set()
    existing_lines = 0
    mode = "a" if APPEND and os.path.exists(OUTPUT) else "w"
    if mode == "a":
        with open(OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                seen.add(line[:40])
                existing_lines += 1
        print(f"追加模式: 已有 {existing_lines} 行, 将生成 {len(tasks)} 条新样本", flush=True)
    else:
        print(f"覆盖模式: 将生成 {len(tasks)} 条新样本", flush=True)

    total = 0
    skipped_dup = 0
    failed = 0
    start = time.time()

    with open(OUTPUT, mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = {pool.submit(gen_one, t, si, ri): (t, si, ri) for (t, si, ri) in tasks}
            for n_done, fut in enumerate(as_completed(futures), 1):
                line = fut.result()
                if line and len(line) >= MIN_LEN:
                    if len(line) > MAX_LEN:
                        line = line[:MAX_LEN]
                    key = line[:40]
                    if key in seen:
                        skipped_dup += 1
                        continue
                    seen.add(key)
                    f.write(line + "\n")
                    total += 1
                else:
                    failed += 1
                if n_done % 50 == 0 or n_done == len(tasks):
                    elapsed = time.time() - start
                    rate = total / elapsed if elapsed > 0 else 0
                    print(
                        f"  进度 {n_done}/{len(tasks)}, 新增 {total}, "
                        f"重复 {skipped_dup}, 失败 {failed}, {rate:.1f}条/秒, {elapsed:.0f}s",
                        flush=True,
                    )

    elapsed = time.time() - start
    size = os.path.getsize(OUTPUT)
    final_lines = existing_lines + total if mode == "a" else total
    print(f"\n完成! 本次新增 {total} 行, 重复跳过 {skipped_dup}, 失败 {failed}")
    print(f"文件总行数: {final_lines}, {size:,} 字节, 耗时 {elapsed:.0f}s")
    print(f"输出: {OUTPUT}")


if __name__ == "__main__":
    main()
