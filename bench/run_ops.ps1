# run_ops.ps1 — 算子对比：layer_bench --op（neuralnet.cpp Vulkan）与 ops_torch（torch CUDA）
# 统一尺寸、每配置 3 轮、warmup=20 / iter=50。
param(
    [ValidateSet("nn", "torch")][string]$Side = "nn"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:NN_VULKAN_DEVICE = "40HX"   # layer_bench 只接受 --gpu 布尔，设备用环境变量钉住

# 形状组：name, m, n, k, batch, op 列表
$groups = @(
    @{ name = "g1_1024";  m = 1024; n = 1024; k = 1024; batch = 1;
       ops = @("matmul", "matmul_bt", "matmul_at", "add_inplace", "elementwise_exp",
               "broadcast_col", "row_reduce_sum", "col_reduce_sum", "transpose", "scale_inplace") },
    @{ name = "g2_2048";  m = 2048; n = 2048; k = 2048; batch = 1;
       ops = @("matmul", "matmul_bt", "matmul_at", "add_inplace", "elementwise_exp",
               "broadcast_col", "row_reduce_sum", "col_reduce_sum", "transpose", "scale_inplace") },
    @{ name = "g3_4096";  m = 4096; n = 4096; k = 4096; batch = 1;
       ops = @("matmul", "matmul_bt", "matmul_at", "add_inplace", "elementwise_exp",
               "broadcast_col", "row_reduce_sum", "col_reduce_sum", "transpose", "scale_inplace") },
    @{ name = "g4_8192";  m = 8192; n = 8192; k = 8192; batch = 1;
       ops = @("add_inplace", "elementwise_exp", "broadcast_col",
               "row_reduce_sum", "col_reduce_sum", "transpose", "scale_inplace") },
    @{ name = "g5_lmhead"; m = 16384; n = 8208; k = 64; batch = 1;
       ops = @("matmul", "matmul_bt", "matmul_at") },
    @{ name = "g6_batched_b64";  m = 256; n = 256; k = 256; batch = 64;
       ops = @("batched_matmul") },
    @{ name = "g7_batched_b512"; m = 64; n = 64; k = 64; batch = 512;
       ops = @("batched_matmul") },
    @{ name = "g8_batched_b4096"; m = 64; n = 64; k = 64; batch = 4096;
       ops = @("batched_matmul") }
)

$rounds = 3
$outDir = "bench/raw"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if ($Side -eq "nn") {
    $log = "$outDir/ops_nn.log"
    "" | Set-Content $log
    foreach ($g in $groups) {
        $opList = ($g.ops -join ",")
        for ($r = 1; $r -le $rounds; $r++) {
            $out = & .\build\layer_bench.exe --gpu --op $opList `
                --m $g.m --n $g.n --k $g.k --batch $g.batch `
                --warmup 20 --iter 50 2>&1 | Out-String
            "### $($g.name) round$r`n$out" | Add-Content $log
            Write-Host "### $($g.name) round$r"
            Write-Host $out
        }
    }
    Write-Host "log -> $log"
}
else {
    $log = "$outDir/ops_torch.log"
    & .venv\Scripts\python.exe bench\ops_torch.py --rounds $rounds --warmup 20 --iter 50 `
        --out $outDir/ops_torch.json *> $log
    Get-Content $log | Write-Host
    Write-Host "log -> $log"
}
