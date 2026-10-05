# ── apply_nn_exit.ps1 — B 批迁移：src/ 手写「失败即退出」样板 → NN_EXIT ──────
#
# 目标形态（铁律 #1 宏族；见 include/neuralnet.cpp/core_errors.hpp）：
#     if (!x) { std::cerr << "ctx" << x.error().message << "\n"; return 1; }
#   → NN_EXIT(x, 1, "ctx")
#     if (!x) return 1;
#   → NN_EXIT(x, 1)
#
# 语义保持：NN_EXIT 打印后 std::exit(code) —— 在 main() 里与 `return code` 等价；
#          打印内容是**超集**（原 ctx + 自带的 <error.message> + 表达式原文 + file:line）。
#
# 只处理**一行内**的上述两种形态；条件必须是「标识符」或「标识符(无括号实参)」，
# 其余（解引用 / 成员访问 / 复杂表达式）一律跳过，不猜。
#
# 用法：
#     pwsh -File tools/apply_nn_exit.ps1 -DryRun          # 只统计（不改文件）
#     pwsh -File tools/apply_nn_exit.ps1 -Apply           # 改 src/ 全部
#     pwsh -File tools/apply_nn_exit.ps1 -File src/x.cpp  # 改单个文件
# ─────────────────────────────────────────────────────────────────────────────
param(
    [switch]$DryRun,
    [switch]$Apply,
    [string]$File
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$cond = '(?<cond>[A-Za-z_]\w*(?:\([^()]*\))?)'
$reBrace = [regex]("(?m)^([ \t]*)if\s*\(\s*!\s*$cond\s*\)\s*\{\s*(?<body>[^\r\n]*?)\s*return\s+(?<code>\d+)\s*;\s*\}")
$reBare  = [regex]("(?m)^([ \t]*)if\s*\(\s*!\s*$cond\s*\)\s*return\s+(?<code>\d+)\s*;")

function Get-Ctx([string]$body) {
    if ($body -notmatch 'cerr|cout|printf|fprintf|puts') { return $null }
    # 逐个字面量扫描：取**第一个含可读文字**的（跳过 "\n" / "  " 这类纯空白），
    # 避免把原代码的换行符当语境。
    foreach ($m in [regex]::Matches($body, '"((?:[^"\\]|\\.)*)"')) {
        $lit = $m.Groups[1].Value
        if ($lit -match '[A-Za-z0-9一-鿿]') { return $lit }
    }
    return $null
}

$targets = if ($File) { @(Join-Path (Get-Location) $File) } else {
    Get-ChildItem (Join-Path $root 'src') -Filter *.cpp | ForEach-Object FullName }

$totBrace = 0; $totBare = 0; $changed = 0

foreach ($path in $targets) {
    if (-not (Test-Path $path)) { Write-Warning "not found: $path"; continue }
    $orig = [System.IO.File]::ReadAllText($path)
    $text = $orig
    $script:fBrace = 0
    $script:fBare = 0

    $text = $reBrace.Replace($text, {
        param($m)
        $script:fBrace++
        $i = $m.Groups[1].Value; $c = $m.Groups['cond'].Value; $k = $m.Groups['code'].Value
        $ctx = Get-Ctx $m.Groups['body'].Value
        if ($ctx) { $i + 'NN_EXIT(' + $c + ', ' + $k + ', "' + $ctx + '");' }
        else      { $i + 'NN_EXIT(' + $c + ', ' + $k + ');' }
    })

    $text = $reBare.Replace($text, {
        param($m)
        $script:fBare++
        $i = $m.Groups[1].Value; $c = $m.Groups['cond'].Value; $k = $m.Groups['code'].Value
        $i + 'NN_EXIT(' + $c + ', ' + $k + ');'
    })

    $rel = $path.Substring($root.Length + 1)
    if ($text -ne $orig) {
        $changed++
        if ($DryRun -and -not $Apply) {
            Write-Output ("  {0,-34} brace={1,-3} bare={2,-3}" -f $rel, $script:fBrace, $script:fBare)
        } else {
            [System.IO.File]::WriteAllText($path, $text)
            Write-Output ("  applied {0}: brace={1} bare={2}" -f $rel, $script:fBrace, $script:fBare)
        }
    }
    $totBrace += $script:fBrace; $totBare += $script:fBare
}

Write-Output ("TOTAL brace={0} bare={1} sum={2} files_touched={3}" -f $totBrace, $totBare, ($totBrace + $totBare), $changed)
