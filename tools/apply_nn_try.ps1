# apply_nn_try.ps1 — 把 L2 层里 `auto X = f(); if (!X) return std::unexpected(X.error());`
# 形态机械收敛为 NN_TRY(X, f()); / NN_TRY_CHECK(X);
# 只做**逐字等价**改写：模式不匹配的行一律不动。
param(
    [switch]$DryRun
)

$files = @(
    'compute_layer_base.hpp',
    'compute_layer_mlp.hpp',
    'compute_layer_conv.hpp',
    'compute_layer_softmax.hpp',
    'compute_layer_attention.hpp',
    'compute_position_encoding.hpp',
    'compute_layer_feedforward.hpp',
    'compute_layer_transformer.hpp',
    'compute_layer_gpt.hpp',
    'compute_layer_zipt.hpp',
    'compute_layer_rapt.hpp'
)
$root = 'D:\Codes\neuralnet.cpp\include\neuralnet.cpp'

# Pass 1：跨两行 —— auto X = EXPR;\n<ws>if (!X) return X.error() 或 X;
$p1 = '(?m)^([ \t]*)auto[ ]*(?:&&[ ]*)?([A-Za-z_]\w*)[ ]*=[ ]*(.+?);[ \t]*\r?\n[ \t]*if[ ]*\(!\2\)[ ]*return[ ]*(?:std::unexpected\(\2\.error\(\)\)|\2);[ \t]*\r?$'
# Pass 2：同一行
$p2 = '(?m)^([ \t]*)auto[ ]*(?:&&[ ]*)?([A-Za-z_]\w*)[ ]*=[ ]*(.+?);[ ]*if[ ]*\(!\2\)[ ]*return[ ]*(?:std::unexpected\(\2\.error\(\)\)|\2);[ \t]*\r?$'
# Pass 3：已声明变量的独立检查
$p3 = '(?m)^([ \t]*)if[ ]*\(!([A-Za-z_]\w*)\)[ ]*return[ ]*(?:std::unexpected\(\2\.error\(\)\)|\2);[ \t]*\r?$'

$grand = 0
foreach ($f in $files)
{
    $path = Join-Path $root $f
    if (-not (Test-Path $path)) { Write-Host "SKIP (missing) $f"; continue }
    $raw = [System.IO.File]::ReadAllText($path)
    $n1 = ([regex]::Matches($raw, $p1)).Count
    $t  = [regex]::Replace($raw, $p1, '$1NN_TRY($2, $3);')
    $n2 = ([regex]::Matches($t, $p2)).Count
    $t  = [regex]::Replace($t, $p2, '$1NN_TRY($2, $3);')
    $n3 = ([regex]::Matches($t, $p3)).Count
    $t  = [regex]::Replace($t, $p3, '$1NN_TRY_CHECK($2);')
    $tot = $n1 + $n2 + $n3
    $grand += $tot
    Write-Host ("{0,-36} two-line={1,4}  one-line={2,4}  check={3,4}  total={4,4}" -f $f, $n1, $n2, $n3, $tot)
    if (-not $DryRun -and $tot -gt 0)
    {
        [System.IO.File]::WriteAllText($path, $t)
    }
}
Write-Host "TOTAL REWRITTEN = $grand  (DryRun=$DryRun)"
