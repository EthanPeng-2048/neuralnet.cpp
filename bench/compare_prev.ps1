# compare_prev.ps1 — 把本轮 nn 基准结果与 **torch 旧值**（不重跑 torch）及 **上一轮 nn 旧值**
# 联成一张对比表。只读工具，不跑任何基准。
#
# 数据源（默认）：
#   nn 本轮   : bench/raw/ops_nn.log            / bench/raw/layers_nn.log          （run_*_ps1 -Side nn 产出）
#   torch 旧值: bench/raw/ops_torch.json        / bench/raw/layers_torch_{model,large}.json
#   nn 上轮   : bench/raw/ops_nn_20260926.log   / bench/raw/layers_nn_20260926.log
#
# 用法:
#   ./bench/compare_prev.ps1 -Kind ops
#   ./bench/compare_prev.ps1 -Kind layers
param(
    [ValidateSet("ops", "layers")][string]$Kind = "ops",
    [string]$NnLog = "",
    [string]$NnPrevLog = "",
    [string]$TorchDir = "bench/raw"
)
$ErrorActionPreference = "Stop"
if (-not $NnLog)       { $NnLog       = if ($Kind -eq "ops") { "bench/raw/ops_nn.log" }       else { "bench/raw/layers_nn.log" } }
if (-not $NnPrevLog)   { $NnPrevLog   = if ($Kind -eq "ops") { "bench/raw/ops_nn_20260926.log" } else { "bench/raw/layers_nn_20260926.log" } }
foreach ($p in @($NnLog, $NnPrevLog)) { if (-not (Test-Path $p)) { throw "log not found: $p" } }

if ($Kind -eq "ops") {
    # ── nn 解析：best-of（最小 ms）────────────────────────────────────────
    function Get-NnBest([string]$path) {
        $best = @{}; $g = ""
        foreach ($line in Get-Content $path) {
            if ($line -match '^###\s+(\S+)\s+round(\d+)') { $g = $Matches[1]; continue }
            if ($line -match '^\s{2}(\w+)\s*:\s*([\d.]+)\s+ms') {
                $k = "$g/$($Matches[1])"; $ms = [double]$Matches[2]
                if (-not $best.ContainsKey($k) -or $ms -lt $best[$k]) { $best[$k] = $ms }
            }
        }
        return $best
    }
    # nn group/op → torch case 名（与 bench/run_ops.ps1、ops_torch.py 的形状表同源）
    function Get-TorchKey([string]$group, [string]$op) {
        switch -Regex ($group) {
            '^g[123]_(\d+)$' {
                $n = $Matches[1]
                if ($op -match '^(matmul|matmul_bt|matmul_at)$') { return "$op[${n}x${n}x${n}]" }
                return "$op[${n}x${n}]"
            }
            '^g4_(\d+)$'      { return "$op[$($Matches[1])x$($Matches[1])]" }
            '^g5_lmhead$'     { return "lm_head_$op[16384x8208x64]" }
            '^g6_batched_b64$'   { return "batched_matmul[B=64,256x256x256]" }
            '^g7_batched_b512$'  { return "batched_matmul[B=512,64x64x64]" }
            '^g8_batched_b4096$' { return "batched_matmul[B=4096,64x64x64]" }
        }
        return $null
    }

    $nn = Get-NnBest $NnLog
    $nnPrev = Get-NnBest $NnPrevLog
    $tj = Get-Content "$TorchDir/ops_torch.json" -Raw | ConvertFrom-Json
    $torch = @{}; foreach ($c in $tj.cases) { $torch[$c.op] = $c.best_ms }

    "## ops：本轮 nn vs torch(09-25) vs nn(09-26) — $NnLog"
    ""
    "| group/op | nn 本轮 (ms) | torch 09-25 (ms) | 比值 nn/torch | nn 09-26 (ms) | 本轮 vs 上轮 |"
    "|---|---:|---:|---:|---:|---:|"
    foreach ($k in ($nn.Keys | Sort-Object)) {
        $parts = $k -split '/', 2
        $tk = Get-TorchKey $parts[0] $parts[1]
        $nnMs = $nn[$k]
        $tMs = if ($tk -and $torch.ContainsKey($tk)) { $torch[$tk] } else { $null }
        $ratio = if ($tMs) { "{0:N2}" -f ($nnMs / $tMs) } else { "—" }
        $prev = if ($nnPrev.ContainsKey($k)) { $nnPrev[$k] } else { $null }
        $delta = if ($prev) { "{0:+0.0;-0.0;0.0}%" -f (($nnMs / $prev - 1) * 100) } else { "—" }
        $prevS = if ($prev) { "{0:N3}" -f $prev } else { "—" }
        "| $k | $nnMs | $(if ($tMs) { '{0:N3}' -f $tMs } else { '—' }) | $ratio | $prevS | $delta |"
    }
    ""
    "比值 >1 = torch 更快；torch 列为 09-25 采集（含跨会话漂移 ±15%）；elementwise_exp/broadcast_col 已从 layer_bench 移除，本轮无 nn 读数"
}
else {
    # ── layers：nn 解析 best-of（fwd/train 各自取最小）────────────────────
    function Get-NnBest([string]$path) {
        $best = @{}; $g = ""
        foreach ($line in Get-Content $path) {
            if ($line -match '^###\s+(\S+)\s+round(\d+)') { $g = $Matches[1]; continue }
            if ($line -match '^\s{2}(\w+)\s*:\s*fwd\s+([\d.]+)\s+ms.*\|\s*train\s+([\d.]+)\s+ms') {
                $k = "$g/$($Matches[1])"; $fwd = [double]$Matches[2]; $tr = [double]$Matches[3]
                if (-not $best.ContainsKey($k)) { $best[$k] = @{ fwd = $fwd; train = $tr } }
                else {
                    if ($fwd -lt $best[$k].fwd) { $best[$k].fwd = $fwd }
                    if ($tr  -lt $best[$k].train) { $best[$k].train = $tr }
                }
            }
        }
        return $best
    }
    $nn = Get-NnBest $NnLog
    $nnPrev = Get-NnBest $NnPrevLog
    $torch = @{}
    foreach ($g in @("model", "large")) {
        $p = "$TorchDir/layers_torch_$g.json"
        if (Test-Path $p) {
            $tj = Get-Content $p -Raw | ConvertFrom-Json
            foreach ($l in $tj.layers) { $torch["$g/$($l.layer)"] = $l }
        }
    }

    "## layers：本轮 nn vs torch(09-25) vs nn(09-26) — $NnLog"
    ""
    "| group/layer | nn fwd | torch fwd | 比值 | nn train | torch train | 比值 | nn 上轮 fwd/train |"
    "|---|---:|---:|---:|---:|---:|---:|---|"
    foreach ($k in ($nn.Keys | Sort-Object)) {
        $v = $nn[$k]; $t = $torch[$k]; $p = $nnPrev[$k]
        $tf = if ($t) { $t.fwd_ms } else { $null }
        $tt = if ($t) { $t.train_ms } else { $null }
        $rf = if ($tf) { "{0:N2}" -f ($v.fwd / $tf) } else { "—" }
        $rt = if ($tt) { "{0:N2}" -f ($v.train / $tt) } else { "—" }
        $prevS = if ($p) { "{0:N2} / {1:N2}" -f $p.fwd, $p.train } else { "—" }
        $tfS = if ($tf) { "{0:N3}" -f $tf } else { "—" }
        $ttS = if ($tt) { "{0:N3}" -f $tt } else { "—" }
        "| $k | $($v.fwd) | $tfS | $rf | $($v.train) | $ttS | $rt | $prevS |"
    }
    ""
    "比值 >1 = torch 更快；torch 列为 09-25 采集（含跨会话漂移 ±15%）"
}
