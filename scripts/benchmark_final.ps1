# MNIST Transformer 优化器对比 - 最终版
$exe = ".\build\mnist_train.exe"
$common = @(
    "--arch", "transformer",
    "--epochs", "5",
    "--batch-size", "128",
    "--d-model", "32",
    "--num-heads", "2",
    "--num-layers", "1",
    "--d-ff", "64",
    "--patch-size", "7",
    "--max-samples", "5000",
    "--eval-samples", "100"
)

$tests = @(
    @{ Name = "SGD+Momentum"; Args = @("--optimizer", "sgd_momentum", "--lr", "0.01") }
    @{ Name = "Adam";         Args = @("--optimizer", "adam",         "--lr", "0.001") }
    @{ Name = "AdamW";        Args = @("--optimizer", "adamw",        "--lr", "0.001", "--weight-decay", "0.01") }
    @{ Name = "Muon";         Args = @("--optimizer", "muon",         "--lr", "0.001") }
)

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " MNIST Transformer Optimizer Comparison" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Model: ViT d_model=32, heads=2, layers=1, d_ff=64"
Write-Host " Data:  5000 train, 5 epochs, batch=128"
Write-Host "================================================================"
Write-Host ""

$results = @()

foreach ($t in $tests) {
    Write-Host "[$($t.Name)]" -ForegroundColor Yellow
    $output = & $exe @common @($t.Args) 2>&1
    $lines = $output -join "`n"

    # 提取最终 loss 和准确率
    $lastEpoch = ($output | Where-Object { $_ -match "Epoch.*loss=" } | Select-Object -Last 1)
    if ($lastEpoch -match "loss=([\d.]+).*train_acc=([\d.]+)%.*test_acc=([\d.]+)%") {
        $loss = [double]$Matches[1]
        $trainAcc = [double]$Matches[2]
        $testAcc = [double]$Matches[3]
    }

    Write-Host "  Final: loss=$loss, train=$trainAcc%, test=$testAcc%"

    $results += [PSCustomObject]@{
        Name     = $t.Name
        Loss     = $loss
        TrainAcc = $trainAcc
        TestAcc  = $testAcc
    }
    Write-Host ""
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " RESULTS SUMMARY" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host ("{0,-15} {1,8} {2,10} {3,10}" -f "Optimizer", "Loss", "Train Acc", "Test Acc")
Write-Host ("{0,-15} {1,8} {2,10} {3,10}" -f "---------", "----", "---------", "--------")

$results | Sort-Object TestAcc -Descending | ForEach-Object {
    $color = if ($_.TestAcc -ge 60) { "Green" } elseif ($_.TestAcc -ge 40) { "Yellow" } else { "Red" }
    Write-Host ("{0,-15} {1,8:F4} {2,9:F1}% {3,9:F1}%" -f $_.Name, $_.Loss, $_.TrainAcc, $_.TestAcc) -ForegroundColor $color
}

Write-Host ""
Write-Host "Note: Each optimizer uses its own recommended lr." -ForegroundColor Gray
Write-Host "      Muon needs lower lr than paper default (0.02) for this small model." -ForegroundColor Gray
