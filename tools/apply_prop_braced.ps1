# apply_prop_braced.ps1 — 花括号作用域守卫：
#   { auto r = <expr>; if (!r) return std::unexpected(r.error()); }
#   → { NN_TRY(r, <expr>); }
# 保留外层花括号（原本就是局部作用域）。同时覆盖 { if (!r) return ...; } ？
# 不覆盖——r 声明在别处时应走 NN_TRY_CHECK，但那种前面已处理。这里只处理
# 声明+守卫都在花括号内的单行形态。
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$re = [regex]'(?m)^([ \t]*)\{\s*auto\s+([A-Za-z_]\w*)\s*=\s*(.+?);\s*if\s*\(\s*!\s*\2\s*\)\s*return\s+std::unexpected\(\s*(?:std::move\()?\s*\2\s*\)?\s*\.error\(\)\s*\)\s*;\s*\}\s*$'
$tot = 0; $tf = 0
foreach ($r in @('include', 'src', 'tools', 'examples', 'bench')) {
    $dir = Join-Path $root $r
    if (-not (Test-Path $dir)) { continue }
    foreach ($f in (Get-ChildItem $dir -Recurse -Include *.hpp, *.cpp -File)) {
        $orig = [System.IO.File]::ReadAllText($f.FullName)
        if ($orig -notmatch '\{\s*auto\s+\w+\s*=.*;\s*if\s*\(\s*!\w+\s*\)\s*return\s+std::unexpected') { continue }
        $script:n = 0
        $text = $re.Replace($orig, { param($m)
            $script:n++
            $m.Groups[1].Value + '{ NN_TRY(' + $m.Groups[2].Value + ', ' + $m.Groups[3].Value + '); }'
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
Write-Output ("TOTAL braced-prop -> NN_TRY = {0} files={1} (DryRun={2})" -f $tot, $tf, $DryRun)
