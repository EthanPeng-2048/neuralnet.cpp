# apply_prop_guard.ps1 — A 批 prop 守卫形态：if (!x)\n return std::unexpected(x.error());
#   → NN_TRY_CHECK(x);   （跨两行合并；宏 = 该 guard 的逐字等价）
# 孤立 return（无守卫，前一行非 if）一律不动（需人工审：可能在 if 块内已知错）。
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
# 两行：if (!x)\n<ws>return std::unexpected(x.error());
$re = [regex]'(?m)^([ \t]*)if\s*\(\s*!\s*([A-Za-z_]\w*)\s*\)\s*\r?\n[ \t]*return\s+std::unexpected\(\s*(?:std::move\()?\s*\2\s*\)?\s*\.error\(\)\s*\)\s*;\s*$'
$tot = 0; $tf = 0
foreach ($r in @('include', 'src', 'tools', 'examples', 'bench')) {
    $dir = Join-Path $root $r
    if (-not (Test-Path $dir)) { continue }
    foreach ($f in (Get-ChildItem $dir -Recurse -Include *.hpp, *.cpp -File)) {
        $orig = [System.IO.File]::ReadAllText($f.FullName)
        if ($orig -notmatch 'if\s*\(\s*!\s*\w+\s*\)\s*\r?\n\s*return\s+std::unexpected') { continue }
        $script:n = 0
        $text = $re.Replace($orig, { param($m)
            $script:n++
            $m.Groups[1].Value + 'NN_TRY_CHECK(' + $m.Groups[2].Value + ');'
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
Write-Output ("TOTAL guarded-prop -> NN_TRY_CHECK = {0} files={1} (DryRun={2})" -f $tot, $tf, $DryRun)
