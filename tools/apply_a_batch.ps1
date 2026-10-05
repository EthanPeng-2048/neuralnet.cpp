# apply_a_batch.ps1 — A 批迁移：return std::unexpected(...) → 宏族（逐字等价）
#
#   [prop] return std::unexpected(x.error());        →  return NN_TRY_CHECK(x);
#         （NN_TRY_CHECK(x) 展开 = if(!(x)) return std::unexpected((x).error());
#           在「紧随 if(!x)」的语境里 x 恒为有值，故整式 ≡ return x.error()，
#           与原手写逐字等价；末尾 ; 与宏展开的最后一条语句拼成空语句，合法。）
#   [new]  return std::unexpected(nn::Error{"msg"}); →  return NN_FAIL("msg");
#         （NN_FAIL(msg) 定义即 return std::unexpected(nn::Error{(msg)})，逐字等价。）
#
# 只改**单行整行**形态；[other]（std::move / 多行构造 / 调用）一律不动，留人工审。
# 所有 [new] 字面量均不含 " 与 \，拼接安全。
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$reProp = [regex]'(?m)^(\s*)return\s+std::unexpected\(\s*(?:std::move\()?\s*([A-Za-z_]\w*)\s*\)?\s*\.error\(\)\s*\)(\s*;?)\s*$'
$reNew  = [regex]'(?m)^(\s*)return\s+std::unexpected\(\s*(?:nn::)?Error\s*\{\s*"((?:[^"\\]|\\.)*)"\s*\}\s*\)(\s*;?)\s*$'

$tp = 0; $tn = 0; $tf = 0
foreach ($r in @('include', 'src', 'tools', 'examples', 'bench')) {
    $dir = Join-Path $root $r
    if (-not (Test-Path $dir)) { continue }
    foreach ($f in (Get-ChildItem $dir -Recurse -Include *.hpp, *.cpp -File)) {
        $orig = [System.IO.File]::ReadAllText($f.FullName)
        $text = $orig
        $script:pn = 0; $script:nn = 0
        $text = $reProp.Replace($text, { param($m)
            $script:pn++
            $m.Groups[1].Value + 'return NN_TRY_CHECK(' + $m.Groups[2].Value + ');'
        })
        $text = $reNew.Replace($text, { param($m)
            $script:nn++
            # NN_FAIL 已含 return，前缀不再加 return
            $m.Groups[1].Value + 'NN_FAIL("' + $m.Groups[2].Value + '");'
        })
        if ($text -ne $orig) {
            $tf++
            $rel = $f.FullName.Substring($root.Length + 1)
            if ($DryRun) {
                Write-Output ("  {0,-52} prop={1,-4} new={2,-4}" -f $rel, $script:pn, $script:nn)
            } else {
                [System.IO.File]::WriteAllText($f.FullName, $text)
                Write-Output ("  applied {0}: prop={1} new={2}" -f $rel, $script:pn, $script:nn)
            }
            $tp += $script:pn; $tn += $script:nn
        }
    }
}
Write-Output ("TOTAL prop={0} new={1} files={2} (DryRun={3})" -f $tp, $tn, $tf, $DryRun)
