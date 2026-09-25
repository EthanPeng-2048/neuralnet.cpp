# run_ab_env.ps1 — 按环境变量做同窗交错 A/B（显存峰值 + 耗时）
#
# 为什么需要它：跨会话单点对比会被 ±15% 系统漂移骗（见 AGENTS §12 方法论）。
# 本脚本把 N 个配置**同窗交错**执行（A B C A B C ...），每个配置取 R 轮，
# 报峰值（nvidia-smi 200ms 采样）与墙钟耗时，并用配对统计给出胜负。
#
# 用法:
#   .\bench\run_ab_env.ps1 -Arguments @('--gpu=40HX','--f16','--steps','2','--no-kv','--batch','32')
#   .\bench\run_ab_env.ps1 -Rounds 3 -Variants @(
#       @{name='base';   env=@{}},
#       @{name='ladder16'; env=@{NN_POOL_LADDER_MAX_MB='16'}},
#       @{name='blk4';   env=@{NN_POOL_BLOCK_MB='4'}} )
param(
    [int]$Rounds = 3,
    [string]$Exe = ".\build\mem_probe.exe",
    [string[]]$Arguments = @(),
    [object[]]$Variants = @(
        @{ name = 'base';     env = @{} },
        @{ name = 'ladder16'; env = @{ NN_POOL_LADDER_MAX_MB = '16' } }
    ),
    [string]$Tag = "abenv"
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$results = @{}
foreach ($v in $Variants) { $results[$v.name] = @{ peaks = @(); secs = @() } }

for ($r = 1; $r -le $Rounds; $r++) {
    foreach ($v in $Variants) {
        foreach ($k in $v.env.Keys) { Set-Item -Path "Env:$k" -Value $v.env[$k] }
        $vargs = if ($v.ContainsKey('args')) { $v.args } else { $Arguments }
        $log = "build\${Tag}_$($v.name)_$r.log"
        $t = Measure-Command {
            & "$PSScriptRoot\run_with_vram.ps1" -Exe $Exe -Arguments $vargs -Log $log | Out-Null
        }
        $peak = [int](Get-Content "$log.peak")
        foreach ($k in $v.env.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
        $results[$v.name].peaks += $peak
        $results[$v.name].secs += $t.TotalSeconds
        Write-Host ("round {0} [{1}] peak={2} MiB  wall={3:N2}s" -f $r, $v.name, $peak, $t.TotalSeconds)
    }
}

Write-Host ""
foreach ($v in $Variants) {
    $p = $results[$v.name].peaks
    $s = $results[$v.name].secs
    Write-Host ("[{0,-12}] peak min={1} mean={2:N0} | wall mean={3:N2}s  ({4})" -f `
        $v.name, ($p | Measure-Object -Minimum).Minimum,
        ($p | Measure-Object -Average).Average,
        ($s | Measure-Object -Average).Average, ($p -join '/'))
}
# 配对比较：以第一个变体为基准
$baseName = $Variants[0].name
foreach ($v in $Variants | Select-Object -Skip 1) {
    $wins = 0
    for ($i = 0; $i -lt $Rounds; $i++) {
        if ($results[$v.name].peaks[$i] -lt $results[$baseName].peaks[$i]) { $wins++ }
    }
    Write-Host ("vs {0}: {1} 胜/平 {2}/{3}" -f $baseName, $v.name, $wins, $Rounds)
}
