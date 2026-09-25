# run_mem_ab.ps1 — 显存峰值 A/B 同窗交错配对实测
#   A = 默认（帧提交后立即非阻塞收割已完成帧）
#   B = NN_NO_EARLY_REAP=1（关闭该优化，等价旧行为）
# 每侧 N 轮，交错执行（ABAB...），避免跨时段漂移；输出配对差与胜负数。
# 用法: .\bench\run_mem_ab.ps1 [-Rounds 3] [-Exe .\build\text_train.exe] [-UseProbe]
param(
    [int]$Rounds = 3,
    [string]$Exe = ".\build\text_train.exe",
    [switch]$UseProbe
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if ($UseProbe) {
    $progArgs = @('--gpu=40HX','--steps','20','--no-kv')
    $tag = 'probe'
} else {
    $progArgs = @('datasets\tinystories_bench40.txt','--vocab','datasets\bench_bpe_8192.json',
              '--d-model','64','--num-heads','4','--num-layers','4','--d-ff','256',
              '--seq-len','256','--batch-size','64','--epochs','1','--optimizer','adam',
              '--lr','0.001','--gpu=40HX','--save-interval','0','--log-interval','100000')
    $tag = 'text_train'
}

function Run-One([string]$mode, [int]$i) {
    $log = "build\ab_${tag}_${mode}_$i.log"
    if ($mode -eq 'B') { $env:NN_NO_EARLY_REAP = '1' } else { Remove-Item Env:NN_NO_EARLY_REAP -ErrorAction SilentlyContinue }
    $out = & "$PSScriptRoot\run_with_vram.ps1" -Exe $Exe -Arguments $progArgs -Log $log 2>&1
    Remove-Item Env:NN_NO_EARLY_REAP -ErrorAction SilentlyContinue
    $peak = 0
    foreach ($l in $out) { if ($l -match 'peak=(\d+)') { $peak = [int]$Matches[1] } }
    return $peak
}

$a = @(); $b = @()
for ($i = 1; $i -le $Rounds; $i++) {
    $pa = Run-One 'A' $i; $a += $pa
    $pb = Run-One 'B' $i; $b += $pb
    Write-Host ("round {0}: A(early-reap)={1} MiB  B(off)={2} MiB  diff={3}" -f $i, $pa, $pb, ($pa - $pb))
}

$wins = 0
for ($i = 0; $i -lt $Rounds; $i++) { if ($a[$i] -lt $b[$i]) { $wins++ } }
Write-Host ("`n=== {0}  A mean={1:N0} min={2} max={3} | B mean={4:N0} min={5} max={6} | A wins {7}/{8} ===" -f `
    $tag, ($a | Measure-Object -Average).Average, ($a | Measure-Object -Minimum).Minimum, ($a | Measure-Object -Maximum).Maximum, `
    ($b | Measure-Object -Average).Average, ($b | Measure-Object -Minimum).Minimum, ($b | Measure-Object -Maximum).Maximum, $wins, $Rounds)

