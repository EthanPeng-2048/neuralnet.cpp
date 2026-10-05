# ── apply_nn_exit.ps1 -Braced — B 批第二形态：{ auto r = f(); if (!r) {…return N;} } ──
#
#     { auto r = f(); if (!r) { std::cerr << "ctx" << r.error().message << "\n"; return 1; } }
#   → NN_EXIT(f(), 1, "ctx")
#
# 直接给表达式上 NN_EXIT（不需要中间变量 r），语境取打印里第一个含可读文字的
# 字面量；失败打印是**超集**（ctx + <error.message> + 表达式原文 + file:line）。
# ─────────────────────────────────────────────────────────────────────────────
param([switch]$DryRun)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

# 前导块 + 声明 + if 条件（条件变量必须 = 声明的变量）
$re = [regex](
    '(?m)^([ \t]*)\{\s*auto\s+(?<v>\w+)\s*=\s*(?<expr>[^;]+?);\s*' +
    'if\s*\(\s*!\s*\k<v>\s*\)\s*\{\s*(?<body>[^\r\n]*?)\s*return\s+(?<code>\d+)\s*;\s*\}\s*\}'
)

function Get-Ctx([string]$body) {
    if ($body -notmatch 'cerr|cout|printf|fprintf|puts') { return $null }
    foreach ($m in [regex]::Matches($body, '"((?:[^"\\]|\\.)*)"')) {
        $lit = $m.Groups[1].Value
        if ($lit -match '[A-Za-z0-9一-鿿]') { return $lit }
    }
    return $null
}

$tot = 0; $files = 0
foreach ($f in (Get-ChildItem (Join-Path $root 'src') -Filter *.cpp)) {
    $orig = [System.IO.File]::ReadAllText($f.FullName)
    $text = $orig
    $script:n = 0
    $text = $re.Replace($text, {
        param($m)
        $script:n++
        $i = $m.Groups[1].Value; $e = $m.Groups['expr'].Value; $k = $m.Groups['code'].Value
        $ctx = Get-Ctx $m.Groups['body'].Value
        if ($ctx) { $i + 'NN_EXIT(' + $e + ', ' + $k + ', "' + $ctx + '");' }
        else      { $i + 'NN_EXIT(' + $e + ', ' + $k + ');' }
    })
    if ($text -ne $orig) {
        $files++
        if ($DryRun) {
            Write-Output ("  {0,-34} {1}" -f $f.Name, $script:n)
        } else {
            [System.IO.File]::WriteAllText($f.FullName, $text)
            Write-Output ("  applied {0}: {1}" -f $f.Name, $script:n)
        }
    }
    $tot += $script:n
}
Write-Output ("TOTAL braced={0} files={1}" -f $tot, $files)
