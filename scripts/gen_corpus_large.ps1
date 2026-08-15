# 用本地 llama-server 生成大量多样化中文语料，每行一段话
$ErrorActionPreference = "Stop"
$ApiUrl = "http://localhost:18080/v1/chat/completions"
$ApiKey = "Ethan_2048"
$OutputFile = "d:\Codes\neuralnet.cpp\datasets\llm_corpus.txt"

# 大量主题，覆盖各领域，确保语料多样
$topics = @(
    # 科技
    "人工智能", "机器学习", "深度学习", "神经网络", "自然语言处理",
    "计算机视觉", "数据科学", "云计算", "大数据", "区块链",
    "物联网", "量子计算", "5G技术", "边缘计算", "网络安全",
    "软件工程", "编程语言", "操作系统", "数据库", "分布式系统",
    # 自然
    "气候变化", "环境保护", "生物多样性", "森林生态", "海洋保护",
    "可再生能源", "太阳能", "风能", "水污染", "空气污染",
    "垃圾分类", "碳中和", "生态平衡", "自然灾害", "地震",
    # 教育
    "在线教育", "终身学习", "批判性思维", "阅读习惯", "学习方法",
    "教育公平", "素质教育", "STEM教育", "蒙台梭利", "翻转课堂",
    # 健康
    "健康饮食", "运动健身", "心理健康", "睡眠质量", "压力管理",
    "冥想", "瑜伽", "慢性病", "营养学", "中医养生",
    # 文化
    "传统文化", "节日习俗", "书法艺术", "茶文化", "京剧",
    "古诗词", "节日美食", "民间故事", "非物质文化遗产", "方言保护",
    # 城市
    "城市规划", "交通拥堵", "公共交通", "智慧城市", "绿色建筑",
    # 农业
    "现代农业", "有机农业", "粮食安全", "农药污染", "精准农业",
    # 艺术
    "电影艺术", "音乐欣赏", "绘画流派", "雕塑艺术", "舞蹈",
    "摄影艺术", "文学创作", "戏剧表演", "建筑设计", "工业设计",
    # 体育
    "足球", "篮球", "乒乓球", "游泳", "马拉松",
    "奥运会", "体育精神", "运动员训练", "体育产业", "电子竞技",
    # 天文
    "宇宙探索", "黑洞", "火星探测", "月球计划", "空间站",
    "望远镜", "行星科学", "银河系", "暗物质", "宇宙起源",
    # 心理
    "情绪管理", "人际交往", "自我认知", "幸福感", "焦虑应对",
    "抑郁症", "积极心理学", "认知偏差", "决策心理", "群体心理",
    # 经济
    "市场经济", "通货膨胀", "股票投资", "创业精神", "消费心理",
    "全球化", "供应链", "数字货币", "普惠金融", "经济周期",
    # 社会
    "社交媒体", "网络文化", "信息过载", "数字鸿沟", "隐私保护",
    "老龄化社会", "城市化", "人口流动", "社区治理", "志愿服务",
    # 哲学
    "人生意义", "伦理道德", "存在主义", "东西方哲学", "美学思考",
    "逻辑学", "认识论", "自由意志", "幸福哲学", "死亡哲学",
    # 医学
    "基因编辑", "精准医疗", "疫苗研发", "抗生素", "心理健康",
    "中医药", "临床试验", "流行病学", "免疫学", "衰老研究",
    # 数学
    "数学之美", "概率论", "统计学", "几何学", "数论",
    "密码学", "图论", "数学建模", "混沌理论", "分形几何",
    # 物理
    "相对论", "量子力学", "粒子物理", "凝聚态物理", "热力学",
    "光学", "声学", "电磁学", "纳米技术", "超导现象",
    # 化学
    "化学反应", "有机合成", "材料科学", "高分子", "催化",
    "电化学", "分析化学", "绿色化学", "晶体结构", "分子生物学",
    # 生物
    "进化论", "基因学", "细胞生物学", "生态学", "动物行为",
    "植物学", "微生物", "神经科学", "脑科学", "生物多样性"
)

$samplesPerTopic = 8  # 每主题 8 段
$totalLines = 0
$batchSize = 8
$batch = @()
$batchIdx = 0

Write-Host "生成中文语料到 $OutputFile ..."
Write-Host "主题数: $($topics.Count), 每主题 $samplesPerTopic 段, 预计 $($topics.Count * $samplesPerTopic) 行"

"" | Out-File -FilePath $OutputFile -Encoding UTF8

for ($i = 0; $i -lt $topics.Count; $i++) {
    $topic = $topics[$i]
    for ($j = 0; $j -lt $samplesPerTopic; $j++) {
        $variation = switch ($j % 4) {
            0 { "用一句完整的话阐述" }
            1 { "用两到三句话展开说明" }
            2 { "从一个具体例子切入描述" }
            3 { "用比喻或类比的方式说明" }
        }
        $prompt = "请就「$topic」这个话题，$variation。要求：
1. 直接输出内容，不要带标题、编号、markdown 符号
2. 内容必须是简体中文，自然流畅的书面语
3. 长度控制在 30-100 字之间
4. 只输出这一段话，不要分多行，不要任何解释或前言"

        $batch += [PSCustomObject]@{
            Prompt = $prompt
            Topic = $topic
            Idx = $j
        }

        if ($batch.Count -ge $batchSize -or ($i -eq $topics.Count - 1 -and $j -eq $samplesPerTopic - 1)) {
            Write-Host "  批次 $batchIdx : $($batch.Count) 个请求..."
            $batchIdx++

            $jobs = $batch | ForEach-Object -Parallel {
                $body = @{
                    messages = @(
                        @{ role = "user"; content = $_.Prompt }
                    )
                    temperature = 0.85
                    max_tokens = 400
                    stream = $false
                    chat_template_kwargs = @{ enable_thinking = $false }
                } | ConvertTo-Json -Depth 5

                try {
                    $resp = Invoke-RestMethod -Uri $using:ApiUrl `
                        -Method Post `
                        -Headers @{Authorization="Bearer $using:ApiKey"} `
                        -ContentType "application/json; charset=utf-8" `
                        -Body $body `
                        -TimeoutSec 120
                    $text = $resp.choices[0].message.content.Trim()
                    [PSCustomObject]@{ Topic = $_.Topic; Idx = $_.Idx; Text = $text; Ok = $true }
                } catch {
                    [PSCustomObject]@{ Topic = $_.Topic; Idx = $_.Idx; Text = "ERROR: $($_.Exception.Message)"; Ok = $false }
                }
            } -ThrottleLimit $batchSize

            foreach ($r in $jobs) {
                if ($r.Ok -and $r.Text.Length -gt 0) {
                    $line = $r.Text -replace "`r`n|`n|`r", " "
                    $line = $line.Trim()
                    $line = $line.Trim('"').Trim('「」').Trim()
                    # 去掉可能的思考标记
                    $line = $line -replace "<think>[\s\S]*?</think>", ""
                    $line = $line.Trim()
                    if ($line.Length -ge 10) {
                        Add-Content -Path $OutputFile -Value $line -Encoding UTF8
                        $totalLines++
                    }
                } else {
                    Write-Host "    失败: $($r.Topic)[$($r.Idx)] $($r.Text)"
                }
            }

            $batch = @()
        }
    }
}

Write-Host ""
Write-Host "完成! 共生成 $totalLines 行"
Write-Host "输出文件: $OutputFile"
