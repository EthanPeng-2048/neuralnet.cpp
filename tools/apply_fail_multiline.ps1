# apply_fail_multiline.ps1 — 多行 new-nonstd：return std::unexpected(Error{\n ... \n});
#   → NN_FAIL(<joined-content>);   内容拼成单行（原文是 + 拼接的字符串/表达式）。
# 护栏：若拼接内容含顶层逗号（不在字符串内、不在括号内）→ 跳过（人工审）。
# 简化护栏：直接检查拼接后是否含 ", " 且不在引号里 —— 用括号深度扫描判断顶层逗号。
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$reOpen  = [regex]'(?m)^(\s*)return\s+std::unexpected\(\s*(?:nn::)?Error\s*\{\s*$'
$tf = 0; $tot = 0; $skipped = 0
foreach ($r in @('include', 'src', 'tools', 'examples', 'bench')) {
    $dir = Join-Path $root $r
    if (-not (Test-Path $dir)) { continue }
    foreach ($f in (Get-ChildItem $dir -Recurse -Include *.hpp, *.cpp -File)) {
        $raw = [System.IO.File]::ReadAllText($f.FullName)
        if ($raw -notmatch 'return\s+std::unexpected\(\s*(?:nn::)?Error\s*\{\s*\r?\n') { continue }
        $lines = [System.IO.File]::ReadAllLines($f.FullName)
        $out = New-Object System.Collections.Generic.List[string]
        $changed = $false; $n = 0
        $i = 0
        while ($i -lt $lines.Count) {
            $ln = $lines[$i]
            if ($reOpen.IsMatch($ln)) {
                $ind = $reOpen.Match($ln).Groups[1].Value
                # 收集直到出现 `});` 收尾
                $buf = @(); $j = $i + 1; $closed = $false
                while ($j -lt $lines.Count) {
                    $cur = $lines[$j]
                    if ($cur -match '\}\s*\)\s*;\s*$') {
                        $part = $cur -replace '\}\s*\)\s*;\s*$',''
                        $buf += $part
                        $closed = $true; break
                    }
                    $buf += $cur
                    $j++
                }
                if ($closed) {
                    $inner = ($buf -join ' ').Trim()
                    # 顶层逗号护栏（忽略字符串内与括号内逗号）
                    $topComma = $false; $depth = 0; $inStr = $false; $esc = $false
                    for ($k = 0; $k -lt $inner.Length; $k++) {
                        $c = $inner[$k]
                        if ($esc) { $esc = $false; continue }
                        if ($c -eq '\') { $esc = $true; continue }
                        if ($c -eq '"') { $inStr = -not $inStr; continue }
                        if ($inStr) { continue }
                        if ($c -eq '(' -or $c -eq '{' -or $c -eq '[') { $depth++ }
                        elseif ($c -eq ')' -or $c -eq '}' -or $c -eq ']') { $depth-- }
                        elseif ($c -eq ',' -and $depth -le 0) { $topComma = $true; break }
                    }
                    if ($topComma) {
                        $out.Add($ln); $i++; $skipped++; continue   # 保留原样
                    }
                    $out.Add($ind + 'NN_FAIL(' + $inner + ');')
                    $changed = $true; $n++
                    $i = $j + 1; continue
                } else {
                    $out.Add($ln); $i++; continue
                }
            }
            $out.Add($ln); $i++
        }
        if ($changed) {
            $tf++; $tot += $n
            $rel = $f.FullName.Substring($root.Length + 1)
            if ($DryRun) { Write-Output ("  {0,-52} {1}" -f $rel, $n) }
            else { [System.IO.File]::WriteAllLines($f.FullName, $out); Write-Output ("  applied {0}: {1}" -f $rel, $n) }
        }
    }
}
Write-Output ("TOTAL multiline-Error -> NN_FAIL = {0} files={1} skipped(top-comma)={2} (DryRun={3})" -f $tot, $tf, $skipped, $DryRun)
