# run_with_vram.ps1 — 运行一个程序并用 nvidia-smi 采样整卡显存峰值
# 用法: ./bench/run_with_vram.ps1 -Exe <exe> -Arguments <args[]> -Log <程序日志>
# 输出: <Log>.vram (每 200ms 一个 MiB 采样) + 峰值打印到 stdout
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string[]]$Arguments,
    [Parameter(Mandatory = $true)][string]$Log
)

$vramFile = "$Log.vram"
$flagFile = "$Log.sampling"
Set-Content -Path $flagFile -Value ""
Set-Content -Path $vramFile -Value ""

$sampler = Start-Job -ArgumentList $flagFile, $vramFile -ScriptBlock {
    param($flag, $out)
    while (Test-Path $flag) {
        $v = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
        if ($v) { Add-Content -Path $out -Value $v.Trim() }
        Start-Sleep -Milliseconds 200
    }
}

& $Exe @Arguments *> $Log
$exitCode = $LASTEXITCODE

Remove-Item -Path $flagFile -Force
Wait-Job $sampler | Out-Null
Receive-Job $sampler | Out-Null
Remove-Job $sampler -Force

$samples = Get-Content $vramFile | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^\d+$' } | ForEach-Object { [int]$_ }
$peak = ($samples | Measure-Object -Maximum).Maximum
# 峰值落盘（同名 .peak）：调用方从文件读峰值，避免解析 Write-Host 输出
Set-Content -Path "$Log.peak" -Value $peak
Write-Host "=== $Exe exit=$exitCode  VRAM samples=$($samples.Count)  peak=${peak} MiB ==="
