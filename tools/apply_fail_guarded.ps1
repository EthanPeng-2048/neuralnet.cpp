# apply_fail_guarded.ps1 — 守卫 + Error 源：if (<cond>) return std::unexpected(Error{"..."});
#   → if (<cond>) NN_FAIL("...");
# NN_FAIL 含 return，接在 if() 后等价。cond 保留原样（!w_.valid() 等）。
# 排除 core_assert.hpp（宏定义体，不能改）。
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$re = [regex]'(?m)^([ \t]*if\s*\(.*\)\s*)return\s+std::unexpected\(\s*(?:nn::)?Error\s*\{("(?:[^"\\]|\\.)*"(?:\s*\+\s*[^;}]+)?)\s*\}\s*\)\s*;\s*(//.*)?$'
$tot = 0; $tf = 0
foreach ($r in @('include', 'src', 'tools', 'examples', 'bench')) {
    $dir = Join-Path $root $r
    if (-not (Test-Path $dir)) { continue }
    foreach ($f in (Get-ChildItem $dir -Recurse -Include *.hpp, *.cpp -File)) {
        if ($f.Name -eq 'core_assert.hpp') { continue }   # 宏定义体，排除
        $orig = [System.IO.File]::ReadAllText($f.FullName)
        if ($orig -notmatch 'if\s*\(.*\)\s*return\s+std::unexpected\(\s*(?:nn::)?Error') { continue }
        $script:n = 0
        $text = $re.Replace($orig, { param($m)
            $script:n++
            $cmt = $m.Groups[4].Value
            $m.Groups[1].Value + 'NN_FAIL(' + $m.Groups[2].Value + ');' + $(if ($cmt) { ' ' + $cmt } else { '' })
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
Write-Output ("TOTAL guarded-Error -> NN_FAIL = {0} files={1} (DryRun={2})" -f $tot, $tf, $DryRun)
