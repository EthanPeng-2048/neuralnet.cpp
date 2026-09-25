# run_layers.ps1 — Layer 对比：layer_bench --layer（neuralnet.cpp Vulkan）与 layers_torch（torch CUDA）
# 统一尺寸、每配置 3 轮、warmup=20 / iter=50；model 组 = 被测模型配置，large 组 = 大尺寸。
param(
    [ValidateSet("nn", "torch")][string]$Side = "nn"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:NN_VULKAN_DEVICE = "40HX"   # layer_bench 只接受 --gpu 布尔，设备用环境变量钉住

# 序列层列表（linear 单独跑：layer_bench 的 linear 输入列数=全局 --batch，
# 需要把 batch 设成 token 数才能与 torch 侧 (T,in) 对齐）
$seqLayers = "layernorm,softmax,mha,causal_attn,feedforward,gpt_block,transformer"

# main 组: batch/seq 用于序列层；rows/cols 用于 softmax（torch 侧为 (d, T)）
# linear 组: --batch = T = batch*seq，--in/--out = d/dff
$groups = @(
    @{ name = "model"; batch = 64; seq = 256; dmodel = 64; heads = 4; dff = 256;
       rows = 64; cols = 16384; linBatch = 16384; nin = 64; nout = 256 },
    @{ name = "large"; batch = 8;  seq = 1024; dmodel = 512; heads = 8; dff = 2048;
       rows = 512; cols = 8192; linBatch = 8192; nin = 512; nout = 2048 }
)

$rounds = 3
$outDir = "bench/raw"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if ($Side -eq "nn") {
    $log = "$outDir/layers_nn.log"
    "" | Set-Content $log
    foreach ($g in $groups) {
        for ($r = 1; $r -le $rounds; $r++) {
            $out = & .\build\layer_bench.exe --gpu --layer $seqLayers `
                --batch $g.batch --seq $g.seq --dmodel $g.dmodel --heads $g.heads --dff $g.dff `
                --rows $g.rows --cols $g.cols `
                --warmup 20 --iter 50 2>&1 | Out-String
            $outLin = & .\build\layer_bench.exe --gpu --layer linear `
                --batch $g.linBatch --in $g.nin --out $g.nout `
                --warmup 20 --iter 50 2>&1 | Out-String
            "### $($g.name) round$r`n$out$outLin" | Add-Content $log
            Write-Host "### $($g.name) round$r"
            Write-Host $out
            Write-Host $outLin
        }
    }
    Write-Host "log -> $log"
}
else {
    foreach ($g in @("model", "large")) {
        $log = "$outDir/layers_torch_$g.log"
        & .venv\Scripts\python.exe bench\layers_torch.py --group $g `
            --rounds $rounds --warmup 20 --iter 50 `
            --out "$outDir/layers_torch_$g.json" *> $log
        Get-Content $log | Write-Host
        Write-Host "log -> $log"
    }
}
