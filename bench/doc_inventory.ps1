# 文档盘点探针：统计 ComputeEngine 接口规模与 Layer 直调算子集合。
# 用法：pwsh -File bench/doc_inventory.ps1
# 与 docs/development/12-compute-engine-inventory.md §2 配套（改接口后重跑核对数字）。
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$eng = Join-Path $root 'include\neuralnet.cpp\compute_engine.hpp'

# 1) 接口 virtual 方法名（去重）
$virts = Select-String -Path $eng -Pattern 'virtual\s+[\w:<>,\s\*&]+?\s+(\w+)\s*\(' -AllMatches |
  ForEach-Object { $_.Matches } | ForEach-Object { $_.Groups[1].Value }
$uv = $virts | Sort-Object -Unique
Write-Output "virtual methods: $($virts.Count) declarations, $($uv.Count) unique"

# 2) Layer/Loss/Optimizer 直调的算子
$layerFiles = Get-ChildItem (Join-Path $root 'include\neuralnet.cpp') -Filter 'compute_layer*.hpp'
$layerFiles += Get-Item (Join-Path $root 'include\neuralnet.cpp\compute_loss.hpp')
$layerFiles += Get-Item (Join-Path $root 'include\neuralnet.cpp\compute_optimizer.hpp')
$layerFiles += Get-Item (Join-Path $root 'include\neuralnet.cpp\model_container.hpp')
$calls = Select-String -Path $layerFiles.FullName -Pattern 'engine_?\.(\w+)\(' -AllMatches |
  ForEach-Object { $_.Matches } | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
$calls = $calls | Where-Object { $_ -ne 'reset' }   # model_container 的 engine_.reset(&engine) 非算子
Write-Output "layer-called ops: $($calls.Count)"
Write-Output ($calls -join ', ')

# 3) 接口中未被 Layer 直调的算子
$unused = $uv | Where-Object { $calls -notcontains $_ }
Write-Output "not layer-called: $($unused.Count)"
Write-Output ($unused -join ', ')
