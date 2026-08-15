# MNIST Transformer 优化器对比实验
# 对比 SGD, SGD+Momentum, Adam, AdamW 的训练效果

$ErrorActionPreference = "Continue"

$exe = ".\build\mnist_train.exe"
$arch = "transformer"
$epochs = 15
$batchSize = 64
$dModel = 64
$numHeads = 4
$numLayers = 2
$dFf = 128
$patchSize = 7

# 优化器配置：名称 -> "参数列表"
$optimizers = @{
    "sgd"          = @("--optimizer", "sgd", "--lr", "0.01")
    "sgd_momentum" = @("--optimizer", "sgd_momentum", "--lr", "0.01")
    "adam"         = @("--optimizer", "adam", "--lr", "0.001")
    "adamw"        = @("--optimizer", "adamw", "--lr", "0.001", "--weight-decay", "0.01")
    "muon"         = @("--optimizer", "muon", "--lr", "0.02")
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " MNIST Transformer 优化器对比实验" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "配置: arch=$arch, epochs=$epochs, batch_size=$batchSize"
Write-Host "      d_model=$dModel, num_heads=$numHeads, num_layers=$numLayers, d_ff=$dFf, patch_size=$patchSize"
Write-Host ""

$results = @{}

foreach ($name in $optimizers.Keys) {
    $optArgs = $optimizers[$name]
    $logFile = "benchmark_${name}.log"

    Write-Host "----------------------------------------" -ForegroundColor Yellow
    Write-Host "训练优化器: $name" -ForegroundColor Yellow
    Write-Host "----------------------------------------" -ForegroundColor Yellow

    $args = @(
        "--arch", $arch,
        "--epochs", $epochs,
        "--batch-size", $batchSize,
        "--d-model", $dModel,
        "--num-heads", $numHeads,
        "--num-layers", $numLayers,
        "--d-ff", $dFf,
        "--patch-size", $patchSize
    ) + $optArgs

    # 运行训练并保存输出
    $output = & $exe @args 2>&1
    $output | Out-File -FilePath $logFile -Encoding utf8

    # 提取最终准确率
    $lastAccLine = $output | Where-Object { $_ -match "准确率.*?(\d+\.\d+)%" } | Select-Object -Last 1
    if ($lastAccLine -match "(\d+\.\d+)%") {
        $acc = [double]$Matches[1]
        $results[$name] = $acc
        Write-Host "最终准确率: $acc%" -ForegroundColor Green
    } else {
        Write-Host "未能提取准确率" -ForegroundColor Red
        $results[$name] = 0
    }
    Write-Host ""
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " 结果汇总" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$results.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object {
    $bar = "█" * [math]::Floor($_.Value / 2)
    Write-Host ("{0,-15} {1,6:F2}% {2}" -f $_.Key, $_.Value, $bar) -ForegroundColor $(
        if ($_.Value -ge 98) { "Green" }
        elseif ($_.Value -ge 95) { "Yellow" }
        else { "Red" }
    )
}

Write-Host ""
Write-Host "详细日志保存在 benchmark_<optimizer>.log" -ForegroundColor Gray
