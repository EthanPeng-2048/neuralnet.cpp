# doc_align_audit.ps1 — 文档↔代码对齐审计（只读）
# 检查三类不一致：
#   [A] 文档中反引号引用的 *.hpp/*.cpp/*.comp 文件是否真实存在
#   [B] 文档中是否仍引用已删除的符号
#   [C] 数字型断言（virtual 数 / ctest 目标数 / N/N 绿）是否与实测一致
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot),
    [int]$MaxPerFile = 12
)

$ErrorActionPreference = 'Stop'
Set-Location $Root

# ── 实测事实 ────────────────────────────────────────────────────────────
$fact = [ordered]@{}
$fact['virtual_in_engine'] = (Select-String -Path 'include\neuralnet.cpp\compute_engine.hpp' -Pattern '\bvirtual\b').Count
$fact['ctest_targets']     = (Select-String -Path 'CMakeLists.txt' -Pattern 'list\(APPEND NN_TEST_TARGETS').Count
$fact['headers']           = (Get-ChildItem -Recurse include -Filter *.hpp).Count
$fact['shaders']           = (Get-ChildItem shaders -Filter *.comp).Count
Write-Host '=== 实测事实 ==='
foreach ($k in $fact.Keys) { Write-Host ("  {0,-18} {1}" -f $k, $fact[$k]) }
Write-Host ''

# 全仓真实文件名索引（include/src/shaders/tools + build 生成物 + 各目录 python）
$realFiles = @{}
Get-ChildItem -Recurse -Include *.hpp,*.cpp,*.comp,*.py -File -Path include,src,shaders,tools,bench,scripts,compare_with_torch -ErrorAction SilentlyContinue |
    ForEach-Object { $realFiles[$_.Name] = $true }
Get-ChildItem -Recurse -Include *.hpp -File -Path build\generated -ErrorAction SilentlyContinue |
    ForEach-Object { $realFiles[$_.Name] = $true }
Get-ChildItem -File -Filter *.py | ForEach-Object { $realFiles[$_.Name] = $true }

# 历史白名单：这些文件**已被有意删除**，历史文档中提及属正常（勿再报警）
$historicalFiles = @('compute_cuda_engine.hpp','compute_cuda_backend.hpp','cuda_kernels.h','cuda_kernels.cu',
                     'expr_graph.hpp','cpu_emitter.hpp','algebra_expr.hpp','algebra_compute.hpp',
                     'perf_smoke.cpp','mnist_common.hpp','train_bytebpe.py',
                     'expr_fuse_test.cpp','expr_graph_test.cpp','expr_spec_test.cpp','matmul_fusion_test.cpp',
                     'offload_primitive_test.cpp','offload_test.cpp','broadcast.comp')

# ── [A] 文档引用的文件是否存在 ──────────────────────────────────────────
Write-Host '=== [A] 文档引用但仓库中不存在的文件 ==='
$docs = @('AGENTS.md','README.md') + (Get-ChildItem -Recurse docs -Filter *.md | ForEach-Object { $_.FullName })
$missCount = 0
$histCount = 0
foreach ($d in $docs) {
    $lines = Get-Content $d
    $hits = @()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        foreach ($m in [regex]::Matches($lines[$i], '`([A-Za-z0-9_]+\.(hpp|cpp|comp|py))`')) {
            $name = $m.Groups[1].Value
            if (-not $realFiles.ContainsKey($name)) {
                if ($historicalFiles -contains $name) { $histCount++; continue }
                $rel = if ($d -match '^[A-Z]:') { Resolve-Path -Relative $d } else { $d }
                $hits += ("    {0}:{1}  {2}" -f $rel, ($i + 1), $name)
            }
        }
    }
    if ($hits.Count -gt 0) {
        $missCount += $hits.Count
        Write-Host ("  -- {0}（{1} 处，示前 {2}）" -f $d, $hits.Count, [Math]::Min($MaxPerFile, $hits.Count))
        $hits | Select-Object -First $MaxPerFile | ForEach-Object { Write-Host $_ }
    }
}
Write-Host ("  [A] 可行动 {0} 处；历史白名单跳过 {1} 处（已删文件的历史提及）" -f $missCount, $histCount)
Write-Host ''

# ── [B] 已删除符号的文档残留 ────────────────────────────────────────────
Write-Host '=== [B] 已删除符号的文档残留 ==='
$dead = @('axpy_inplace','broadcast_row_inplace','broadcast_col_inplace','elementwise_unary',
          'elementwise_binary_scalar','elementwise_select_scalar_cond','offload_store','offload_load',
          'multiply_transposed_add_to','compute::apply','algebra_expr','algebra_compute',
          'set_offload_enabled','UnaryOp','CompareOp')
foreach ($s in $dead) {
    $all = Select-String -Path @($docs) -Pattern $s -SimpleMatch -ErrorAction SilentlyContinue
    if ($all.Count -gt 0) {
        Write-Host ("  -- {0}: {1} 处" -f $s, $all.Count)
        $all | Select-Object -First $MaxPerFile | ForEach-Object {
            Write-Host ("     {0}:{1}: {2}" -f (Split-Path $_.Path -Leaf), $_.LineNumber, $_.Line.Trim().Substring(0,[Math]::Min(78,$_.Line.Trim().Length)))
        }
    }
}
Write-Host ''

# ── [C] 数字型断言 ──────────────────────────────────────────────────────
Write-Host '=== [C] 数字型断言（需与实测核对）==='
Select-String -Path @($docs) -Pattern '\d+ 个 virtual|\d+ 个目标|ctest \d+|\d+/\d+ *(绿|全绿|通过|测试)|目标（\d+ 个）' -ErrorAction SilentlyContinue |
    ForEach-Object { "  {0}:{1}: {2}" -f (Split-Path $_.Path -Leaf), $_.LineNumber, $_.Line.Trim().Substring(0,[Math]::Min(88,$_.Line.Trim().Length)) }
Write-Host ''

# ── 代码侧符号索引（include + src + tools + python + CMake）──────────────
$codeText = ((Get-ChildItem -Recurse -Include *.hpp,*.cpp -File -Path include,src,tools |
             ForEach-Object { Get-Content $_.FullName -Raw }) -join "`n")
$pyText = ((Get-ChildItem -Recurse -Include *.py -File -Path . |
             Where-Object { $_.FullName -notmatch '\\build\\|\\.venv\\|node_modules' } |
             ForEach-Object { Get-Content $_.FullName -Raw }) -join "`n")
$codeText = $codeText + "`n" + $pyText
$cmText = Get-Content CMakeLists.txt -Raw

# 外部工具参数白名单（非本项目 CLI：ctest/cmake/glslc/torch 脚本等）
$externalFlags = @('--test-dir','--parallel','--target-env','--dtype','--adam-eps','--device',
                   '--task','--compress','--list','--image','--list-backends')
# 已在文档中明确标注"已移除"的历史参数
$removedFlags  = @('--cuda','--tdr-retry','--max-tdr-retries')

# ── [D] 文档里的 CLI 参数是否存在 ────────────────────────────────────────
Write-Host '=== [D] 文档引用但代码中不存在的 CLI 参数（--flag）==='
$flags = @{}
foreach ($d in $docs) {
    $lines = Get-Content $d
    for ($i = 0; $i -lt $lines.Count; $i++) {
        foreach ($m in [regex]::Matches($lines[$i], '(?<![\w-])--[a-z][a-z0-9-]{2,}')) {
            $f = $m.Value
            if (-not $flags.ContainsKey($f)) { $flags[$f] = @() }
            if ($flags[$f].Count -lt 4) {
                $flags[$f] += ("{0}:{1}" -f (Split-Path $d -Leaf), ($i + 1))
            }
        }
    }
}
$missFlags = 0
$skippedFlags = 0
foreach ($f in ($flags.Keys | Sort-Object)) {
    if ($codeText -notmatch [regex]::Escape($f)) {
        if ($externalFlags -contains $f -or $removedFlags -contains $f) { $skippedFlags++; continue }
        $missFlags++
        Write-Host ("  -- {0}   出现于: {1}" -f $f, ($flags[$f] -join ', '))
    }
}
Write-Host ("  [D] 找不到的 {0} 个；白名单跳过 {1} 个（外部工具参数/文档已标注移除）" -f $missFlags, $skippedFlags)
Write-Host ''

# ── [E] 文档里的 nn::Xxx 符号是否存在 ────────────────────────────────────
Write-Host '=== [E] 文档引用但 include/src 中不存在的 nn:: 符号 ==='
$syms = @{}
foreach ($d in $docs) {
    $lines = Get-Content $d
    for ($i = 0; $i -lt $lines.Count; $i++) {
        foreach ($m in [regex]::Matches($lines[$i], 'nn::([A-Za-z_][A-Za-z0-9_]*)')) {
            $s = $m.Groups[1].Value
            if (-not $syms.ContainsKey($s)) { $syms[$s] = @() }
            if ($syms[$s].Count -lt 3) { $syms[$s] += ("{0}:{1}" -f (Split-Path $d -Leaf), ($i + 1)) }
        }
    }
}
$missSyms = 0
foreach ($s in ($syms.Keys | Sort-Object)) {
    if ($codeText -notmatch ('\b' + [regex]::Escape($s) + '\b')) {
        $missSyms++
        Write-Host ("  -- nn::{0}   出现于: {1}" -f $s, ($syms[$s] -join ', '))
    }
}
Write-Host ("  [E] 合计 {0} 个符号找不到" -f $missSyms)
Write-Host ''

# ── [F] 文档里的测试目标/子测试文件名是否存在 ────────────────────────────
Write-Host '=== [F] 文档引用但既非 CMake 目标、也非 src 文件的测试名 ==='
$tests = @{}
foreach ($d in $docs) {
    $lines = Get-Content $d
    for ($i = 0; $i -lt $lines.Count; $i++) {
        foreach ($m in [regex]::Matches($lines[$i], '\b([a-z0-9_]{3,}_test|maxpool_gradcheck|conv2d_gradcheck|rmsnorm_gradcheck|softmax_gradcheck|swiglu_gradcheck|attn_gradcheck|gpt_gradcheck|rapt_gradcheck|zipt_gradcheck|f16_cpu_probe|f16_writeback_probe|mem_probe|layer_bench)\b')) {
            $t = $m.Groups[1].Value
            if (-not $tests.ContainsKey($t)) { $tests[$t] = @() }
            if ($tests[$t].Count -lt 3) { $tests[$t] += ("{0}:{1}" -f (Split-Path $d -Leaf), ($i + 1)) }
        }
    }
}
$missTests = 0
$histTests = 0
# 历史/已并入的测试名（历史交付表中的当时目标名，勿再报警）
$historicalTests = @('expr_fuse_test','expr_graph_test','expr_spec_test','matmul_fusion_test','offload_primitive_test')
foreach ($t in ($tests.Keys | Sort-Object)) {
    $inCm = $cmText -match [regex]::Escape($t)
    $inSrc = $realFiles.ContainsKey("$t.cpp")
    if (-not $inCm -and -not $inSrc) {
        if ($historicalTests -contains $t) { $histTests++; continue }
        $missTests++
        Write-Host ("  -- {0}   出现于: {1}" -f $t, ($tests[$t] -join ', '))
    }
}
Write-Host ("  [F] 可行动 {0} 个；历史/已并入白名单跳过 {1} 个" -f $missTests, $histTests)
