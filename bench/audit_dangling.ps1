# audit_dangling.ps1 — 悬空设计扫描（只读，不改代码）
# 1) 头文件引用计数：每个 include/**.hpp 被 include/src/tools/cmake 引用了几次（排除自身）
# 2) 关键死码候选符号的调用点计数（排除定义/声明/文档）
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'

Write-Host "== [1] 头文件外部引用计数（0 = 疑似悬空头）=="
$hdrs = Get-ChildItem -Recurse -Path (Join-Path $Root 'include') -Filter *.hpp -File
$scanDirs = @('include', 'src', 'tools', 'tests', 'compare_with_torch') |
    ForEach-Object { Join-Path $Root $_ } | Where-Object { Test-Path $_ }
$files = $scanDirs | ForEach-Object { Get-ChildItem -Recurse -Path $_ -Include *.hpp, *.cpp, *.h, *.cu, *.py -File }
$cmake = Get-ChildItem -Path $Root -Filter *.txt -File -ErrorAction SilentlyContinue
$files += $cmake

$results = foreach ($h in $hdrs) {
    $hits = 0
    foreach ($f in $files) {
        if ($f.FullName -eq $h.FullName) { continue }
        $hits += (Select-String -Path $f.FullName -Pattern $h.Name -SimpleMatch -ErrorAction SilentlyContinue).Count
    }
    [PSCustomObject]@{ Header = $h.Name; ExtRefs = $hits }
}
$results | Sort-Object ExtRefs | Format-Table -AutoSize

Write-Host "== [2] 死码候选符号调用点（代码文件，排除定义行注释）=="
$symbols = @(
    # 2026-09 重构已整体删除（本清单兼作"不得复活"回归守卫，命中应为 0）：
    'axpy_inplace', 'broadcast_row_inplace', 'broadcast_col_inplace',
    'elementwise_select_scalar_cond', 'elementwise_unary', 'elementwise_binary_scalar',
    'multiply_transposed_add_to', 'offload_store', 'offload_load', 'one_hot',
    'algebra_expr.hpp', 'algebra_compute.hpp',
    # 仍存活（调用点应 >0）：
    'row_reduce_max(', 'pool_stats', 'release_idle_pool_blocks'
)
foreach ($s in $symbols) {
    $codeHits = @()
    foreach ($f in $files) {
        $m = Select-String -Path $f.FullName -Pattern $s -SimpleMatch -ErrorAction SilentlyContinue
        foreach ($x in $m) { $codeHits += ("{0}:{1}" -f (Resolve-Path -Relative $x.Path), $x.LineNumber) }
    }
    Write-Host ("-- {0}  ({1} 处)" -f $s, $codeHits.Count)
    $codeHits | ForEach-Object { Write-Host ("     " + $_) }
}
