# run_mem_variants.ps1 — 顺序跑 mem_probe 各变体，用同一把尺子（nvidia-smi 200ms）采样峰值
# 用法: .\bench\run_mem_variants.ps1
# 说明：一次只跑一个 GPU 任务（顺序执行），避免并行导致计时/显存互相污染。

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$variants = @(
    @{ name = 'baseline_f32';        args = @('--gpu=40HX','--steps','10','--no-kv') },
    @{ name = 'checkpoint1';         args = @('--gpu=40HX','--steps','10','--no-kv','--checkpoint-every','1') },
    @{ name = 'offload';             args = @('--gpu=40HX','--steps','10','--no-kv','--activation-offload') },
    @{ name = 'checkpoint1_offload'; args = @('--gpu=40HX','--steps','4','--no-kv','--checkpoint-every','1','--activation-offload') }
)

foreach ($v in $variants) {
    $log = "build\memvar_$($v.name).log"
    $out = & "$PSScriptRoot\run_with_vram.ps1" -Exe ".\build\mem_probe.exe" -Arguments $v.args -Log $log 2>&1
    $line = ($out | Select-String -Pattern 'peak=' | Select-Object -Last 1).Line
    Write-Host "[$($v.name)] $line"
}
