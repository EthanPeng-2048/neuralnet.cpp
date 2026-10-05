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
# compute_position_encoding.hpp 是 L2 辅助对象（位置编码策略族）→ 同受铁律 #12 约束
$layerFiles += Get-Item (Join-Path $root 'include\neuralnet.cpp\compute_position_encoding.hpp')
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

# 4) L2+ 分层审计：Matrix 降级（docs/development/17 §4.6 / §5 M4；AGENTS.md 铁律 #12）
#    规则：Layer / Loss / Optimizer / Model（L2 计算路径）头文件中——
#      [硬] 不得出现 Matrix / MatrixT 类型（不持有、不构造、不交换）
#      [硬] 不得出现 Matrix 型 I/O 动词 from_matrix / to_matrix / copy_from
#      [披露] 宿主桥 read/write/get_index/set_index 与 detail::upload_span /
#             download_span/download_vector 只计数
#             （层自算辅助数据的受控通道，17 §3 D11；数据集/权重仍走 I/O 层）
#    注释已剥离后才匹配（注释里讲规则/讲历史不算命中）。
#    验收口径：L2-VIOLATIONS 必须为 0。
$violType = @()
$violVerb = @()
$bridge = @()
foreach ($f in $layerFiles) {
    $inBlock = $false
    $lineNo = 0
    foreach ($raw in (Get-Content $f.FullName)) {
        $lineNo++
        $s = $raw
        if ($inBlock) {
            $end = $s.IndexOf('*/')
            if ($end -ge 0) { $s = $s.Substring($end + 2); $inBlock = $false } else { $s = '' }
        }
        while (-not $inBlock) {
            $open = $s.IndexOf('/*')
            if ($open -lt 0) { break }
            $close = $s.IndexOf('*/', $open + 2)
            if ($close -lt 0) { $s = $s.Substring(0, $open); $inBlock = $true; break }
            $s = $s.Substring(0, $open) + $s.Substring($close + 2)
        }
        $cmt = $s.IndexOf('//')
        if ($cmt -ge 0) { $s = $s.Substring(0, $cmt) }
        $leaf = Split-Path $f.FullName -Leaf
        if ($s -match '\bMatrixT?\b') {
            $violType += ("{0}:{1}: {2}" -f $leaf, $lineNo, $raw.Trim())
        }
        if ($s -match '\b(from_matrix|to_matrix|copy_from)\s*\(') {
            $violVerb += ("{0}:{1}: {2}" -f $leaf, $lineNo, $raw.Trim())
        }
        if ($s -match '\b(read|write|get_index|set_index|upload_span|download_span|download_vector)\s*\(') {
            $bridge += ("{0}:{1}" -f $leaf, $lineNo)
        }
    }
}
Write-Output ''
Write-Output '=== [4] L2+ 分层审计（Matrix 降级：17 §4.6/M4，铁律 #12）==='
foreach ($v in $violType) { Write-Output ("  [Matrix] {0}" -f $v) }
foreach ($v in $violVerb)  { Write-Output ("  [IO-VERB] {0}" -f $v) }
Write-Output ("  matrix_type_hits: {0}   io_verb_hits: {1}   host_bridge_uses: {2}" -f $violType.Count, $violVerb.Count, $bridge.Count)
if ($bridge.Count -gt 0) { Write-Output ("  bridge sites: " + ($bridge -join ', ')) }
Write-Output ("  L2-VIOLATIONS: {0}" -f ($violType.Count + $violVerb.Count))

# 5) 错误处理宏族手写残留审计（AGENTS.md 铁律 #1 / §12 错误处理宏族）
#    A 批（传播/错误源 → NN_TRY/NN_TRY_CHECK/NN_FAIL）与 B 批（src/ 退出样板 → NN_EXIT）
#    迁移后的**手写 `return std::unexpected` 残留**计数。
#    剔除：注释、字符串字面量内出现、以及 core_assert.hpp 的宏定义体（NN_ASSERT 实现，
#    其中 `return std::unexpected(nn::Error{(msg)})` 是宏展开目标，必须保留）。
#    验收口径：残留数不高于基线（基线 = 0，2026-10-05 2 阶段边缘形态清零），
#    且**不随新代码增长**——新增错误检查一律用宏族（NN_TRY/NN_EXIT/NN_FAIL），
#    不应再出现裸 unexpected。
#    当前基线：0（2026-10-05 两批迁移 + 2 阶段 36 处边缘形态清零后实测）。
$baseline = 0
$residue = @()
$scanDirs = @('include', 'src', 'tools', 'examples', 'bench')
foreach ($d in $scanDirs) {
    $dd = Join-Path $root $d
    if (-not (Test-Path $dd)) { continue }
    foreach ($f in (Get-ChildItem $dd -Recurse -Include *.hpp, *.cpp -File)) {
        if ($f.Name -eq 'core_assert.hpp') { continue }   # 宏定义体，豁免
        # core_errors.hpp 的 NN_TRY/NN_FAIL 宏定义体（`return std::unexpected(...)` 是
        # 展开目标，必须保留）与 error_macro_test.cpp 的测试夹具（故意构造 unexpected
        # 以断言宏行为）——逐行豁免见下。
        $isCoreErr = ($f.Name -eq 'core_errors.hpp')
        $isMacroFixture = ($f.Name -eq 'error_macro_test.cpp')
        $inDefine = $false   # core_errors.hpp：#define 宏体（续行以 \ 结尾，末行收尾）
        $lineNo = 0
        foreach ($raw in (Get-Content $f.FullName)) {
            $lineNo++
            $s = $raw.Trim()
            # core_errors.hpp：跟踪 #define 宏体（宏定义里的 return std::unexpected 是
            # 展开目标，必须保留）——以 "#define" 起，行尾 "\" 续行，末行收尾。
            if ($isCoreErr) {
                if ($s -match '^#define\s') { $inDefine = $true }
                if ($inDefine) {
                    $hadContinuation = ($s -match '\\\s*$')
                    if (-not $hadContinuation) { $inDefine = $false }   # 末行：退出宏体
                    continue   # 宏体行（含末行）一律豁免
                }
            }
            # 剥行注释与字符串字面量（只判代码部分）
            $code = [regex]::Replace($s, '"(?:[^"\\]|\\.)*"', '""')
            $cmt = $code.IndexOf('//')
            if ($cmt -ge 0) { $code = $code.Substring(0, $cmt) }
            if ($code -match 'return\s+std::unexpected') {
                # error_macro_test.cpp：豁免测试夹具（bad_int 等 helper 故意构造 unexpected）
                if ($isMacroFixture) { continue }
                $rel = $f.FullName.Substring($root.Length + 1)
                $residue += ("{0}:{1}" -f $rel, $lineNo)
            }
        }
    }
}
Write-Output ''
Write-Output '=== [5] 错误处理宏族手写残留（铁律 #1；NN_TRY/NN_EXIT/NN_FAIL 迁移后）==='
foreach ($r in $residue) { Write-Output ("  [residue] {0}" -f $r) }
Write-Output ("  HANDWRITTEN-RESIDUE: {0}   (baseline {1})" -f $residue.Count, $baseline)
if ($residue.Count -gt $baseline) {
    Write-Output ("  [GATE] FAIL — 残留 {0} 超过基线 {1}：新增错误检查须用宏族，勿写裸 unexpected。" -f $residue.Count, $baseline)
    exit 1
}
Write-Output "  [GATE] PASS — 残留未超基线（新增检查须用宏族）。"
