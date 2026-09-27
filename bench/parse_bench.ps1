# parse_bench.ps1 — 解析 layer_bench 日志（bench/run_ops.ps1、run_layers.ps1 的输出）
# 每个形状/层组取 3 轮 best-of（最小 ms），输出 markdown 表格行，供 docs/benchmarks 引用。
# 用法:
#   ./bench/parse_bench.ps1 -Kind ops    [-Log bench/raw/ops_nn.log]
#   ./bench/parse_bench.ps1 -Kind layers [-Log bench/raw/layers_nn.log]
param(
    [ValidateSet("ops", "layers")][string]$Kind = "ops",
    [string]$Log = ""
)
$ErrorActionPreference = "Stop"
if (-not $Log) { $Log = if ($Kind -eq "ops") { "bench/raw/ops_nn.log" } else { "bench/raw/layers_nn.log" } }
if (-not (Test-Path $Log)) { throw "log not found: $Log" }

$curGroup = ""
# best[group][op] = @{ ms; rate; unit; round }
$best = @{}
$round = 0
$unknown = [System.Collections.Generic.List[string]]::new()

foreach ($line in Get-Content $Log) {
    if ($line -match '^###\s+(\S+)\s+round(\d+)') {
        $curGroup = $Matches[1]; $round = [int]$Matches[2]; continue
    }
    if ($line -match '未知算子:\s*(\S+)') {
        $u = "$curGroup/$($Matches[1])"
        if (-not $unknown.Contains($u)) { $unknown.Add($u) }
        continue
    }
    # ops 行: "  matmul         :    0.853 ms      2518.5 GFLOPS"
    if ($Kind -eq "ops" -and $line -match '^\s{2}(\w+)\s*:\s*([\d.]+)\s+ms\s+([\d.]+)\s+(GFLOPS|GB/s)') {
        $op = $Matches[1]; $ms = [double]$Matches[2]; $rate = [double]$Matches[3]; $unit = $Matches[4]
        $key = "$curGroup/$op"
        if (-not $best.ContainsKey($key) -or $ms -lt $best[$key].ms) {
            $best[$key] = @{ ms = $ms; rate = $rate; unit = $unit; round = $round }
        }
        continue
    }
    # layers 行: "  mha            : fwd    7.538 ms  ... |  train   22.894 ms ..."
    if ($Kind -eq "layers" -and $line -match '^\s{2}(\w+)\s*:\s*fwd\s+([\d.]+)\s+ms.*\|\s*train\s+([\d.]+)\s+ms') {
        $lay = $Matches[1]; $fwd = [double]$Matches[2]; $tr = [double]$Matches[3]
        $key = "$curGroup/$lay"
        if (-not $best.ContainsKey($key)) { $best[$key] = @{ fwd = $fwd; train = $tr; round = $round } }
        else {
            if ($fwd -lt $best[$key].fwd) { $best[$key].fwd = $fwd }
            if ($tr -lt $best[$key].train) { $best[$key].train = $tr }
        }
        continue
    }
}

"## parsed from $Log (best-of-rounds)"
if ($Kind -eq "ops") {
    "" | Out-Null
    "| group/op | best ms | rate | unit | round |"
    "|---|---|---|---|---|"
    foreach ($k in ($best.Keys | Sort-Object)) {
        $v = $best[$k]
        "| $k | $($v.ms) | $($v.rate) | $($v.unit) | $($v.round) |"
    }
} else {
    "" | Out-Null
    "| group/layer | fwd ms | train ms |"
    "|---|---|---|"
    foreach ($k in ($best.Keys | Sort-Object)) {
        $v = $best[$k]
        "| $k | $($v.fwd) | $($v.train) |"
    }
}
if ($unknown.Count -gt 0) {
    ""
    "### 未知算子（已从 layer_bench 算子表移除，未能测得）"
    $unknown | Sort-Object | ForEach-Object { "- $_" }
}
