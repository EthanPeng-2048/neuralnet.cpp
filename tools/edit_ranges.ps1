# edit_ranges.ps1 — 带边界断言的行区间删除（重构批次用，防止行号漂移误删）
# 用法：pwsh -File tools/edit_ranges.ps1 <spec.json>
# spec.json: JSON 数组，每项：
#   { "file": "include/...hpp", "start": 100, "end": 120,
#     "expectStart": "起始行必须包含的子串", "expectEnd": "结束行必须包含的子串",
#     "note": "删除内容说明" }
# 约束：同一文件的多个区间按 start 降序应用（互不重叠）；任何断言失败 = 整体中止不写盘。
# -DryRun：只断言不写盘（先 dry-run 全量核对期望串，再真跑）。
# -AllowUnbalanced：放行"花括号不平衡"的区间（默认拒绝——防"删掉 if 外壳留孤立括号"）。
param([Parameter(Mandatory)][string]$SpecPath, [switch]$DryRun, [switch]$AllowUnbalanced)

$ErrorActionPreference = 'Stop'
$jobs = Get-Content $SpecPath -Raw | ConvertFrom-Json
if (-not $jobs) { throw "spec 解析为空: $SpecPath" }

# 按文件分组
$byFile = @{}
foreach ($j in $jobs) {
    if (-not $byFile.ContainsKey($j.file)) { $byFile[$j.file] = @() }
    $byFile[$j.file] += $j
}

$nlCache = @{}
foreach ($k in $byFile.Keys) {
    $path = if ([IO.Path]::IsPathRooted($k)) { $k } else { Join-Path (Get-Location) $k }
    if (-not (Test-Path $path)) { throw "文件不存在: $path" }

    $bytes = [IO.File]::ReadAllBytes($path)
    $hasBom = $bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF
    $raw = [IO.File]::ReadAllText($path)
    $nl = if ($raw -match "`r`n") { "`r`n" } else { "`n" }
    $lines = [IO.File]::ReadAllLines($path)

    # 降序排序，先断言全部区间，再一次性删除
    $ranges = $byFile[$k] | Sort-Object { -([int]$_.start) }
    foreach ($j in $ranges) {
        $s = [int]$j.start; $e = [int]$j.end
        if ($s -lt 1 -or $e -gt $lines.Count -or $s -gt $e) {
            throw "${k}: 区间 $s-$e 越界（共 $($lines.Count) 行）"
        }
        if ($j.expectStart -and -not $lines[$s - 1].Contains([string]$j.expectStart)) {
            throw "${k}:$s 起始行断言失败：期望含 '$($j.expectStart)'，实际 '$($lines[$s - 1])'"
        }
        if ($j.PSObject.Properties['expectStart2'] -and $j.expectStart2) {
            if ($s + 1 -gt $lines.Count -or -not $lines[$s].Contains([string]$j.expectStart2)) {
                $got = if ($s + 1 -le $lines.Count) { $lines[$s] } else { '<EOF>' }
                throw "${k}:$($s+1) 起始+1 行断言失败：期望含 '$($j.expectStart2)'，实际 '$got'"
            }
        }
        if ($j.expectEnd -and -not $lines[$e - 1].Contains([string]$j.expectEnd)) {
            throw "${k}:$e 结束行断言失败：期望含 '$($j.expectEnd)'，实际 '$($lines[$e - 1])'"
        }
        if ($j.PSObject.Properties['expectNext'] -and $j.expectNext) {
            if ($e + 1 -gt $lines.Count -or -not $lines[$e].Contains([string]$j.expectNext)) {
                $got = if ($e + 1 -le $lines.Count) { $lines[$e] } else { '<EOF>' }
                throw "${k}:$($e+1) 结束+1 行断言失败：期望含 '$($j.expectNext)'，实际 '$got'"
            }
        }
        # ── 结构性不变量：区间内 { } 必须自平衡 ──────────────────────────
        # 边界串匹配 ≠ 删对了：删掉 `if (…)` 的起始行却留下其闭合 `}`，
        # 文本断言全过、结构却已损坏（2026-09-26 gpu_test 事故的根因）。
        $ob = 0; $cb = 0
        for ($t = $s; $t -le $e; $t++) {
            $ob += ([regex]::Matches($lines[$t - 1], '\{')).Count
            $cb += ([regex]::Matches($lines[$t - 1], '\}')).Count
        }
        if ($ob -ne $cb) {
            $msg = "${k}:$s-$e 区间花括号不平衡（{=$ob, }=$cb）——整段删除必须自平衡；确需如此请显式加 -AllowUnbalanced"
            if (-not $AllowUnbalanced) { throw $msg }
            Write-Host ("[warn] " + $msg)
        }
        Write-Host ("[del] {0}:{1}-{2}  ({3})" -f $k, $s, $e, $j.note)
    }
    if ($DryRun) {
        Write-Host ("[dry] {0}: {1} 个区间断言全过，跳过写盘" -f $k, @($ranges).Count)
        continue
    }
    # 断言全过 → 删除（同文件按降序，区间互不重叠）
    $keep = New-Object System.Collections.Generic.List[string]
    $rangesSorted = $ranges | Sort-Object { [int]$_.start }
    $cursor = 1
    foreach ($j in $rangesSorted) {
        $s = [int]$j.start; $e = [int]$j.end
        for ($i = $cursor; $i -lt $s; $i++) { $keep.Add($lines[$i - 1]) }
        $cursor = $e + 1
    }
    for ($i = $cursor; $i -le $lines.Count; $i++) { $keep.Add($lines[$i - 1]) }

    $joined = [string]::Join($nl, $keep)
    [IO.File]::WriteAllText($path, $joined, [Text.UTF8Encoding]::new($hasBom))
    Write-Host ("[ok ] {0}: {1} → {2} 行" -f $k, $lines.Count, $keep.Count)
}
