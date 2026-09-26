# doc_rename.ps1 — 文档批量词法改名（带 dry-run 与命中数断言）
# 用法：pwsh -File tools/doc_rename.ps1 <spec.json> [-DryRun]
# spec.json: { "files": ["README.md","AGENTS.md","docs/introduction/*.md"],   # 支持 * 通配
#              "ops": [ { "pattern":"...", "replacement":"...", "minHits":1, "note":"..." } ] }
# 语义：ops 按顺序对每个文件全文应用（.NET 正则）；每个 op 的总命中数必须 >= minHits，
#       否则整体中止不写盘。保持原编码（BOM）与行尾（CRLF/LF）。
param([Parameter(Mandatory)][string]$SpecPath, [switch]$DryRun)

$ErrorActionPreference = 'Stop'
$spec = Get-Content $SpecPath -Raw | ConvertFrom-Json
$files = @()
foreach ($g in $spec.files) {
    $files += (Get-ChildItem -Path $g -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
}
$files = $files | Sort-Object -Unique
if (-not $files) { throw "未匹配到任何文件" }

# 先统计（全量断言，不写盘）——命中数按**模式全局**计，不按文件计
$plan = @{}
$opTotals = @()
foreach ($op in $spec.ops) { $opTotals += 0 }
$fi = 0
foreach ($f in $files) {
    $raw = [IO.File]::ReadAllText($f)
    $cur = $raw
    $rows = @()
    for ($k = 0; $k -lt $spec.ops.Count; $k++) {
        $op = $spec.ops[$k]
        $hit = ([regex]::Matches($cur, $op.pattern)).Count
        $opTotals[$k] += $hit
        $rows += [PSCustomObject]@{ Op = $op.note; Pattern = $op.pattern; Hits = $hit }
        if ($hit -gt 0) { $cur = [regex]::Replace($cur, $op.pattern, $op.replacement) }
    }
    $plan[$f] = @{ Before = $raw; After = $cur; Rows = $rows }
    $fi++
}

for ($k = 0; $k -lt $spec.ops.Count; $k++) {
    $op = $spec.ops[$k]
    $min = if ($op.PSObject.Properties['minHits'] -and $null -ne $op.minHits) { [int]$op.minHits } else { 1 }
    if ($opTotals[$k] -lt $min) {
        throw ("模式全局命中不足（{0} < {1}）pattern='{2}' note='{3}'" -f $opTotals[$k], $min, $op.pattern, $op.note)
    }
}

$totalHits = 0
foreach ($f in $plan.Keys) {
    $changed = ($plan[$f].Before -ne $plan[$f].After)
    foreach ($r in $plan[$f].Rows) { $totalHits += $r.Hits }
    if ($changed) {
        Write-Host ("[edit] {0}  {1}" -f (Resolve-Path -Relative $f), (($plan[$f].Rows | Where-Object { $_.Hits -gt 0 } | ForEach-Object { "{0}×{1}" -f $_.Op, $_.Hits }) -join ' + '))
    }
}
Write-Host ("总命中 {0} 处，涉及 {1} 个文件" -f $totalHits, ($plan.Keys | Where-Object { $plan[$_].Before -ne $plan[$_].After }).Count)

if ($DryRun) { Write-Host '=== DRY-RUN：未写盘 ==='; exit 0 }

foreach ($f in $plan.Keys) {
    if ($plan[$f].Before -eq $plan[$f].After) { continue }
    $bytes = [IO.File]::ReadAllBytes($f)
    $hasBom = $bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF
    [IO.File]::WriteAllText($f, $plan[$f].After, [Text.UTF8Encoding]::new($hasBom))
}
Write-Host '=== 全部应用完成 ==='
