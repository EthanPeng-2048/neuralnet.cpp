#!/usr/bin/env python3
"""Decode fused shader registry to find Flash Attention kernels."""
import re, sys

OPCODES = {0:'Neg',1:'Exp',2:'Log',3:'Sqrt',4:'Rsqrt',5:'Abs',6:'Tanh',
           7:'Add',8:'Sub',9:'Mul',10:'Div',11:'Max',12:'Min',
           13:'Lt',14:'Le',15:'Gt',16:'Ge',17:'Eq',18:'Ne',19:'Select',
           20:'ColSum',21:'ColMax',22:'RowSum',23:'RowMax'}
KINDS = {0:'Reg',1:'Input',2:'Const',3:'Fanout',4:'Reduce',5:'Matmul',6:'Row',7:'Col',8:'Batch'}

with open('build/generated/fused_registry.hpp') as f:
    c = f.read()

idx = c.find('kFusedShaders[]')
section = c[idx:]
entries = re.split(r'\{\s*"([0-9a-f]{16})"', section)

out = open('build/generated/fused_analysis.txt', 'w')
out.write("=== Fused Shader Analysis (has_matmul=1 only) ===\n\n")

count = 0
for i in range(1, len(entries), 2):
    key = entries[i]
    body = entries[i+1] if i+1 < len(entries) else ''

    m_instrs = re.search(r'ExprInstr>\{(.*?)\}', body, re.DOTALL)
    m_views = re.search(r'ExprView>\{(.*?)\}', body, re.DOTALL)
    m_consts = re.search(r'Scalar>\{(.*?)\}', body, re.DOTALL)
    m_mm = re.search(r'MatmulSpec>\{(.*?)\}', body)

    # Metadata is the LAST occurrence of ", N, N, N, N }" in the body
    # after "sizeof(std::uint32_t)"
    m_spirv = body.rfind('sizeof(std::uint32_t)')
    if m_spirv < 0: continue
    tail = body[m_spirv:]
    m_meta = re.search(r',\s*(-?\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\}', tail)
    if not m_meta: continue
    ra, hm, vp, vw = int(m_meta.group(1)), int(m_meta.group(2)), int(m_meta.group(3)), int(m_meta.group(4))
    if hm != 1: continue

    instrs_s = m_instrs.group(1) if m_instrs else ''
    instrs = re.findall(r'\{(\d+),\s*(\d+),\s*\{(\d+),\s*(\d+)\},\s*\{(\d+),\s*(\d+)\},\s*\{(\d+),\s*(\d+)\}\}', instrs_s)

    ops = [int(j[0]) for j in instrs]
    has_rmax = 23 in ops
    has_rsum = 22 in ops
    has_exp = 1 in ops
    tag = []
    if has_rmax: tag.append('RowMax')
    if has_rsum: tag.append('RowSum')
    if has_exp: tag.append('Exp')
    if not tag: tag.append('other')

    ra_s = {-1:'elem', 0:'row-red', 1:'col-red'}.get(ra, f'?{ra}')
    mm_s = m_mm.group(1).strip() if m_mm else 'none'
    views_s = m_views.group(1).strip() if m_views else ''
    consts_s = m_consts.group(1).strip() if m_consts else ''

    count += 1
    out.write(f"\n{'='*70}\n")
    out.write(f"key={key}  reduce={ra_s}  tags=[{' '.join(tag)}]  vec={vw}\n")
    out.write(f"  MatmulSpec: {mm_s}\n")
    out.write(f"  Views: [{views_s}]\n")
    if consts_s:
        out.write(f"  Consts: [{consts_s}]\n")
    out.write(f"  Instructions ({len(instrs)}):\n")
    for ins in instrs:
        op,dst,ak,ai,bk,bi,ck,ci = [int(x) for x in ins]
        on = OPCODES.get(op, f'Op{op}')
        an = KINDS.get(ak,'?')+'['+str(ai)+']'
        bn = KINDS.get(bk,'?')+'['+str(bi)+']'
        cn = KINDS.get(ck,'?')+'['+str(ci)+']'
        out.write(f"    r{dst} = {on}({an}, {bn}, {cn})\n")

out.write(f"\n\nTotal: has_matmul=1 shaders: {count}\n")
out.close()
print(f"Done! {count} entries written to build/generated/fused_analysis.txt")
