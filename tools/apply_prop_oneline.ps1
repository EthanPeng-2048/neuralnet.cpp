# apply_prop_oneline.ps1 — A 批单行守卫：if (!x) return std::unexpected(x.error());
#   → NN_TRY_CHECK(x);
# 也覆盖单行 init 守卫：if (auto x = f(); !x) return std::unexpected(x.error()); → NN_TRY(x, f());
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
# 形态 A：if (!x) return std::unexpected(x.error());
$reA = [regex]'(?m)^([ \t]*)if\s*\(\s*!\s*([A-Za-z_]\w*)\s*\)\s*return\s+std::unexpected\(\s*(?:std::move\()?\s*\2\s*\)?\s*\.error\(\)\s*\)\s*;\s*$'
# 形态 B：if (auto x = <expr>; !x) return std::unexpected(x.error());
$reB = [regex]'(?m)^([ \t]*)if\s*\(\s*auto\s+([A-Za-z_]\w*)\s*=\s*(.+?);\s*!\s*\2\s*\)\s*return\s+std::unexpected\(\s*(?:std::move\()?\s*\2\s*\)?\s*\.error\(\)\s*\)\s*;\s*$'
$ta = 0; $tb = 0; $tf = 0
foreach ($r in @('include', 'src', 'tools', 'examples', 'bench')) {
    $dir = Join-Path $root $r
    if (-not (Test-Path $dir)) { continue }
    foreach ($f in (Get-ChildItem $dir -Recurse -Include *.hpp, *.cpp -File)) {
        $orig = [System.IO.File]::ReadAllText($f.FullName)
        if ($orig -notmatch 'if\s*\(.*\)\s*return\s+std::unexpected') { continue }
        $text = $orig
        $script:a = 0; $script:b = 0
        $text = $reB.Replace($text, { param($m)
            $script:b++
            $m.Groups[1].Value + 'NN_TRY(' + $m.Groups[2].Value + ', ' + $m.Groups[3].Value + ');'
        })
        $text = $reA.Replace($text, { param($m)
            $script:a++
            $m.Groups[1].Value + 'NN_TRY_CHECK(' + $m.Groups[2].Value + ');'
        })
        if ($text -ne $orig) {
            $tf++
            $rel = $f.FullName.Substring($root.Length + 1)
            if ($DryRun) { Write-Output ("  {0,-52} A={1,-3} B={2,-3}" -f $rel, $script:a, $script:b) }
            else { [System.IO.File]::WriteAllText($f.FullName, $text); Write-Output ("  applied {0}: A={1} B={2}" -f $rel, $script:a, $script:b) }
            $ta += $script:a; $tb += $script:b
        }
    }
}
Write-Output ("TOTAL guardA(NN_TRY_CHECK)={0} guardB(NN_TRY)={1} files={2} (DryRun={3})" -f $ta, $tb, $tf, $DryRun)
