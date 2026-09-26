# test_refactor.ps1 — 测试清理批量改写（删除重复定义 / 插入公共头 include / 正则替换）
# 用法：pwsh -File tools/test_refactor.ps1 <spec.json> [-DryRun]
# spec.json 结构：
# {
#   "delete":   [ { "file", "start", "expectStart", "note" } ... ]
#               # 从 start 行删到定义块结束：#define 行按反斜杠续行判定；
#               # 普通函数按大括号配对判定；单行语句删 1 行。
#   "replace":  [ { "file", "pattern", "replacement", "note" } ... ]
#               # .NET 正则全文替换；要求至少命中 1 次（否则报错中止）。
#   "include":  [ { "file", "text", "note" } ... ]
#               # 在文件最后一个 #include 行之后插入 text（多行原文）。
# }
# 所有阶段：先全量断言，任一失败 = 整体中止不写盘。-DryRun 只打印计划。
param([Parameter(Mandatory)][string]$SpecPath, [switch]$DryRun)

$ErrorActionPreference = 'Stop'
$spec = Get-Content $SpecPath -Raw | ConvertFrom-Json

function Get-Lines($path) { , [IO.File]::ReadAllLines($path) }
function Save-Lines($path, [string[]]$lines) {
    $bytes = [IO.File]::ReadAllBytes($path)
    $hasBom = $bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF
    $raw = [IO.File]::ReadAllText($path)
    $nl = if ($raw -match "`r`n") { "`r`n" } else { "`n" }
    [IO.File]::WriteAllText($path, [string]::Join($nl, $lines), [Text.UTF8Encoding]::new($hasBom))
}
function Resolve-Path2($f) { if ([IO.Path]::IsPathRooted($f)) { $f } else { Join-Path (Get-Location) $f } }

# ── 阶段 1：删除定义块 ──────────────────────────────────────────────────────
if ($spec.delete) {
    $byFile = @{}
    foreach ($d in $spec.delete) {
        $p = Resolve-Path2 $d.file
        if (-not $byFile.ContainsKey($p)) { $byFile[$p] = @() }
        $byFile[$p] += $d
    }
    foreach ($p in $byFile.Keys) {
        $lines = [IO.File]::ReadAllLines($p)
        $jobs = $byFile[$p] | Sort-Object { -([int]$_.start) }   # 降序：断言+删除都按原行号
        # 先断言全部
        $spans = @()
        foreach ($d in $jobs) {
            $s = [int]$d.start
            if ($s -lt 1 -or $s -gt $lines.Count) { throw "${p}: start $s 越界（共 $($lines.Count) 行）" }
            if (-not $lines[$s - 1].Contains([string]$d.expectStart)) {
                throw "${p}:$s 断言失败：期望含 '$($d.expectStart)'，实际 '$($lines[$s - 1])'"
            }
            # 计算块尾
            $end = $s
            if ($lines[$s - 1] -match '^\s*#(define|undef|endif|pragma)\b') {
                # 预处理指令：#define 按反斜杠续行判定，其余单行
                while ($end -lt $lines.Count -and $lines[$end - 1].TrimEnd().EndsWith('\')) { $end++ }
            } else {
                $depth = 0; $started = $false
                for ($j = $s; $j -le $lines.Count; $j++) {
                    $depth += ([regex]::Matches($lines[$j - 1], '\{')).Count
                    $depth -= ([regex]::Matches($lines[$j - 1], '\}')).Count
                    if ($depth -gt 0) { $started = $true }
                    if ($started -and $depth -eq 0) { $end = $j; break }
                    if (-not $started -and $depth -lt 0) { throw "${p}:$j 括号不配对（负深度）" }
                }
                if (-not $started) { throw "${p}:$s 起始块未找到配对大括号" }
            }
            $spans += [PSCustomObject]@{ s = $s; e = $end; note = $d.note }
            Write-Host ("[del] {0}:{1}-{2}  ({3})" -f (Split-Path $p -Leaf), $s, $end, $d.note)
        }
        # 同文件多个区间可能重叠？检查
        $sorted = $spans | Sort-Object { [int]$_.s }
        for ($i = 1; $i -lt $sorted.Count; $i++) {
            if ([int]$sorted[$i].s -le [int]$sorted[$i - 1].e) { throw "${p}: 区间重叠 $($sorted[$i-1].e)/$($sorted[$i].s)" }
        }
        if ($DryRun) { continue }
        # 升序游标法重建
        $keep = New-Object System.Collections.Generic.List[string]
        $cur = 1
        foreach ($sp in $sorted) {
            for ($i = $cur; $i -lt [int]$sp.s; $i++) { $keep.Add($lines[$i - 1]) }
            $cur = [int]$sp.e + 1
        }
        for ($i = $cur; $i -le $lines.Count; $i++) { $keep.Add($lines[$i - 1]) }
        Save-Lines $p $keep.ToArray()
        Write-Host ("[ok ] {0}: {1} → {2} 行" -f (Split-Path $p -Leaf), $lines.Count, $keep.Count)
    }
}

# ── 阶段 2：正则替换 ────────────────────────────────────────────────────────
if ($spec.replace) {
    foreach ($r in $spec.replace) {
        $p = Resolve-Path2 $r.file
        $raw = [IO.File]::ReadAllText($p)
        $cnt = ([regex]::Matches($raw, $r.pattern, 'Multiline')).Count
        if ($cnt -lt 1) { throw "${p}: 替换模式 0 命中：$($r.pattern)" }
        Write-Host ("[rep] {0}: {1} 处  ({2})" -f (Split-Path $p -Leaf), $cnt, $r.note)
        if ($DryRun) { continue }
        $new = [regex]::Replace($raw, $r.pattern, $r.replacement, 'Multiline')
        $bytes = [IO.File]::ReadAllBytes($p)
        $hasBom = $bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF
        [IO.File]::WriteAllText($p, $new, [Text.UTF8Encoding]::new($hasBom))
    }
}

# ── 阶段 3：插入 include（最后一个 #include 行之后）────────────────────────
if ($spec.include) {
    foreach ($inc in $spec.include) {
        $p = Resolve-Path2 $inc.file
        $lines = [IO.File]::ReadAllLines($p)
        if (($lines -join "`n").Contains('test_common.hpp')) { Write-Host ("[skip] {0}: 已含 test_common" -f (Split-Path $p -Leaf)); continue }
        $last = -1
        for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match '^\s*#include\s') { $last = $i } }
        if ($last -lt 0) { throw "${p}: 找不到任何 #include 行" }
        Write-Host ("[inc] {0}: 插入于第 {1} 行 #include 之后  ({2})" -f (Split-Path $p -Leaf), ($last + 1), $inc.note)
        if ($DryRun) { continue }
        $newLines = @()
        $newLines += $lines[0..$last]
        $newLines += ($inc.text -split "`n")
        if ($last + 1 -lt $lines.Count) { $newLines += $lines[($last + 1)..($lines.Count - 1)] }
        Save-Lines $p $newLines
    }
}

if ($DryRun) { Write-Host '=== DRY-RUN 完成：未写盘 ===' } else { Write-Host '=== 全部应用完成 ===' }
