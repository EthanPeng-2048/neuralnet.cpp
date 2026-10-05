# apply_decl_oneline.ps1 — 同行 声明+守卫：auto x = <expr>; if (!x) return std::unexpected(x.error());
#   → NN_TRY(x, <expr>);
# 两个语句在一行（分号分隔）。expr 不含分号（正则非贪婪到第一个 ; 后的 if）。
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
# auto <v> = <expr>; if (!<v>) return std::unexpected(<v>.error());
$re = [regex]'(?m)^([ \t]*)auto\s+([A-Za-z_]\w*)\s*=\s*(.+?);\s*if\s*\(\s*!\s*\2\s*\)\s*return\s+std::unexpected\(\s*(?:std::move\()?\s*\2\s*\)?\s*\.error\(\)\s*\)\s*;\s*$'
$tot = 0; $tf = 0
foreach ($r in @('include', 'src', 'tools', 'examples', 'bench')) {
    $dir = Join-Path $root $r
    if (-not (Test-Path $dir)) { continue }
    foreach ($f in (Get-ChildItem $dir -Recurse -Include *.hpp, *.cpp -File)) {
        $orig = [System.IO.File]::ReadAllText($f.FullName)
        if ($orig -notmatch 'auto\s+\w+\s*=.*;\s*if\s*\(\s*!\w+\s*\)\s*return\s+std::unexpected') { continue }
        $script:n = 0
        $text = $re.Replace($orig, { param($m)
            $script:n++
            $m.Groups[1].Value + 'NN_TRY(' + $m.Groups[2].Value + ', ' + $m.Groups[3].Value + ');'
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
Write-Output ("TOTAL decl-oneline -> NN_TRY = {0} files={1} (DryRun={2})" -f $tot, $tf, $DryRun)
