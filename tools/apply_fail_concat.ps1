# apply_fail_concat.ps1 — new-nonstd：return std::unexpected(Error{"..." + expr});
#   → NN_FAIL("..." + expr);
# NN_FAIL(msg) 展开 Error{(msg)}，msg 为任意表达式（+ 拼接无顶层逗号，安全）。
# 只匹配单行、Error{ 后是字符串字面量开头的构造（含 + 拼接 / to_string 等）。
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
# return std::unexpected( (nn::)?Error{ ... } );  —— 内容为单行、无顶层逗号歧义
$re = [regex]'(?m)^(\s*)return\s+std::unexpected\(\s*(?:nn::)?Error\s*\{(.+?)\}\s*\)\s*;\s*$'
$tot = 0; $tf = 0
foreach ($r in @('include', 'src', 'tools', 'examples', 'bench')) {
    $dir = Join-Path $root $r
    if (-not (Test-Path $dir)) { continue }
    foreach ($f in (Get-ChildItem $dir -Recurse -Include *.hpp, *.cpp -File)) {
        $orig = [System.IO.File]::ReadAllText($f.FullName)
        if ($orig -notmatch 'return\s+std::unexpected\(\s*(?:nn::)?Error\s*\{') { continue }
        $script:n = 0
        $text = $re.Replace($orig, { param($m)
            $script:n++
            $m.Groups[1].Value + 'NN_FAIL(' + $m.Groups[2].Value + ');'
        })
        if ($text -ne $orig) {
            $tf++
            $rel = $f.FullName.Substring($root.Length + 1)
            if ($DryRun) { Write-Output ("  {0,-52} {1}" -f $rel, $script:n) }
            else { [System.IO.File]::WriteAllText($f.FullName, $text); Write-Output ("  applied {0}: {1}" -f $rel, $script:n) }
            $tot += $script:n
        }
    }
}
Write-Output ("TOTAL concat-Error -> NN_FAIL = {0} files={1} (DryRun={2})" -f $tot, $tf, $DryRun)
