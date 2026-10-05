# apply_prop_init.ps1 — A 批 prop init-statement 守卫：
#   if (auto r = <expr>; !r)\n return std::unexpected(r.error());
#   → NN_TRY(r, <expr>);
# 仅当 return 的变量 == init 声明的变量时替换（否则不动）。
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
# if (auto <v> = <expr>; !<v>)\n<ws>return std::unexpected(<v>.error());
$re = [regex]'(?m)^([ \t]*)if\s*\(\s*auto\s+([A-Za-z_]\w*)\s*=\s*(.+?);\s*!\s*\2\s*\)\s*\r?\n[ \t]*return\s+std::unexpected\(\s*(?:std::move\()?\s*\2\s*\)?\s*\.error\(\)\s*\)\s*;\s*$'
$tot = 0; $tf = 0
foreach ($r in @('include', 'src', 'tools', 'examples', 'bench')) {
    $dir = Join-Path $root $r
    if (-not (Test-Path $dir)) { continue }
    foreach ($f in (Get-ChildItem $dir -Recurse -Include *.hpp, *.cpp -File)) {
        $orig = [System.IO.File]::ReadAllText($f.FullName)
        if ($orig -notmatch 'if\s*\(\s*auto\s+\w+\s*=.*;\s*!\w+\s*\)\s*\r?\n\s*return\s+std::unexpected') { continue }
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
Write-Output ("TOTAL init-guard -> NN_TRY = {0} files={1} (DryRun={2})" -f $tot, $tf, $DryRun)
