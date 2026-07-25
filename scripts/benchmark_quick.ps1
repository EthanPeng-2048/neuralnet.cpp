# MNIST Transformer 优化器快速对比
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
    @{ Name = "SGD";          Args = @("--optimizer", "sgd",          "--lr", "0.05") }
    @{ Name = "SGD+Momentum"; Args = @("--optimizer", "sgd_momentum", "--lr", "0.01") }
    @{ Name = "Adam";         Args = @("--optimizer", "adam",         "--lr", "0.001") }
    @{ Name = "AdamW";        Args = @("--optimizer", "adamw",        "--lr", "0.001", "--weight-decay", "0.01") }
    @{ Name = "Muon";         Args = @("--optimizer", "muon",         "--lr", "0.02") }
)

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " MNIST Transformer Optimizer Benchmark" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Model: d_model=32, heads=2, layers=1, d_ff=64"
Write-Host " Data:  5000 train, 100 eval | 5 epochs, batch=128"
Write-Host "========================================================"
Write-Host ""

$results = @()

foreach ($t in $tests) {
    Write-Host "[$($t.Name)]" -ForegroundColor Yellow
    $output = & $exe @common @($t.Args) 2>&1
    $lines = $output -join "`n"

    $match = [regex]::Match($lines, "test_acc=(\d+\.\d+)%")
    $finalAcc = if ($match.Success) { [double]$match.Groups[1].Value } else { 0 }

    $epochLines = $output | Where-Object { $_ -match "Epoch" }
    foreach ($line in $epochLines) { Write-Host "  $line" }

    $results += [PSCustomObject]@{ Name = $t.Name; FinalAcc = $finalAcc }
    Write-Host ""
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " RESULTS" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$results | Sort-Object FinalAcc -Descending | ForEach-Object {
    $bar = [string]::new([char]0x2588, [math]::Floor($_.FinalAcc / 2))
    $color = if ($_.FinalAcc -ge 90) { "Green" } elseif ($_.FinalAcc -ge 70) { "Yellow" } else { "Red" }
    Write-Host ("  {0,-15} {1,6:F2}%  {2}" -f $_.Name, $_.FinalAcc, $bar) -ForegroundColor $color
}
Write-Host ""
