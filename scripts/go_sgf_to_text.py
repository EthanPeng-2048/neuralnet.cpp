#!/usr/bin/env python3
"""
Go SGF → Text converter + Tokenizer generator (胜利方视角版)
============================================================
核心设计：
  - 只从胜利方视角训练：模型学到的是"赢家怎么下"
  - <me> = 赢家，<enemy> = 输家
  - 推理时 <me> 即代表最强一方

每局棋只生成 1 条训练序列（从赢家视角）：
  <me> pd <enemy> dp <me> pp <enemy> qq ...

SGF 结果格式（RE 属性）：
  RE[B+R]  = 黑胜（认输）   RE[W+R]  = 白胜（认输）
  RE[B+1.5]= 黑胜 1.5 目    RE[W+0.5]= 白胜 0.5 目
  RE[Void] = 无结果（跳过）

用法：
  python go_sgf_to_text.py sgf_dir/ -o go_games.txt --tokenizer go_tokenizer.json

训练：
  text_train.exe go_games.txt --vocab go_tokenizer.json \\
      --d-model 128 --num-heads 4 --num-layers 4 --d-ff 512 \\
      --seq-len 600 --epochs 50 --lr 0.001 --batch-size 32 --cuda

推理：
  text_infer.exe --model go_model.bin \\
      --prompt "<me> pd <enemy> dp" --max-tokens 1 --temperature 0
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
#  SGF 解析
# ═══════════════════════════════════════════════════════════════════════════

def parse_sgf_result(sgf_text: str) -> str | None:
    """
    从 SGF 文本中提取对局结果。
    返回: "B" (黑胜) / "W" (白胜) / None (无结果)
    """
    # 匹配 RE[...] 属性
    m = re.search(r'RE\[(.*?)\]', sgf_text)
    if not m:
        return None
    result = m.group(1).strip()
    if result in ('Void', 'Void', '', '?'):
        return None
    # B+R, B+1.5, B+Resign, Black wins → 黑胜
    if result.upper().startswith('B') or 'BLACK' in result.upper():
        return 'B'
    # W+R, W+0.5, White wins → 白胜
    if result.upper().startswith('W') or 'WHITE' in result.upper():
        return 'W'
    return None


def convert_coord_system(moves: list[str]) -> list[str]:
    """
    统一 SGF 坐标体系。

    两种 19 路坐标体系：
      体系1 (标准): a-hj-t（19 个字符，跳过 i）
      体系2 (非标准): a-s（19 个字符，连续含 i）— CWI 部分文件使用

    判断：一局含 'i' 坐标且不含 't' 坐标（除 tt=PASS）→ 体系2，需转换。
    体系2 的第 n 个字符 = 体系1 的第 n 个字符，因此 i→j, j→k, ..., s→t。

    转换后坐标与体系1 完全一致（i 不再出现）。
    """
    # 检查是否含 i（非 tt 的坐标）
    has_i = any('i' in c for c in moves)
    if not has_i:
        return moves  # 体系1，无需转换

    # 检查是否含 t（除 tt=PASS 外的 t 坐标）
    has_t = any('t' in c and c != 'tt' for c in moves)
    if has_t:
        # 既含 i 又含 t：数据混用/异常，丢弃该局（返回 None 标记）
        return None

    # 体系2 → 体系1：i→j, j→k, k→l, ..., s→t
    trans = str.maketrans('ijklmnopqrs', 'jklmnopqrst')
    return [c.translate(trans) for c in moves]


def parse_sgf_games(sgf_text: str) -> list[tuple[list[str], str | None]]:
    """
    从 SGF 文本中提取所有对局的落子序列和结果。

    只处理 19×19 棋局（SZ[19] 或未标注默认 19）。
    9×9 / 13×13 等小棋盘（SZ[9]/SZ[13]）跳过。
    非标准坐标体系（a-s 连续含 i）自动转换为标准体系。

    返回: [([moves], "B"/"W"/None), ...]
    """
    results = []

    game_blocks = re.findall(r'\((.*?)\)', sgf_text, re.DOTALL)
    if not game_blocks:
        game_blocks = [sgf_text]

    for block in game_blocks:
        # 读取棋盘大小 SZ[...]，非 19 路跳过
        sz_m = re.search(r'SZ\[(\d+)\]', block)
        if sz_m:
            sz = int(sz_m.group(1))
            if sz != 19:
                continue

        moves = []
        # 先宽松提取坐标（a-z，含 tt=PASS），再统一坐标体系
        for m in re.finditer(r';([BW])\[([a-z]{2}|tt)\]', block):
            moves.append(m.group(2))

        if not moves:
            continue

        # 统一坐标体系（含 i 的体系2 → 标准体系1）；None = 数据异常丢弃
        converted = convert_coord_system(moves)
        if converted is None:
            continue

        # 提取本局结果
        winner = parse_sgf_result(block)
        results.append((converted, winner))

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  胜利方视角转换
# ═══════════════════════════════════════════════════════════════════════════

def game_to_winner_perspective(moves: list[str], winner: str | None) -> str | None:
    """
    将一局棋转换为胜利方视角的训练序列。

    winner="B": 黑方是 <me>，白方是 <enemy>
    winner="W": 白方是 <me>，黑方是 <enemy>
    winner=None: 返回 None（无结果，跳过）

    返回: "<me> pd <enemy> dp <me> pp ..." 或 None
    """
    if winner is None:
        return None

    tokens = []
    for i, coord in enumerate(moves):
        is_me = (i % 2 == 0 and winner == 'B') or \
                (i % 2 == 1 and winner == 'W')
        prefix = "<me>" if is_me else "<enemy>"
        tokens.append(f"{prefix} {coord}")

    return ' '.join(tokens)


# ═══════════════════════════════════════════════════════════════════════════
#  Tokenizer 生成
# ═══════════════════════════════════════════════════════════════════════════

def generate_go_tokenizer(output_path: str) -> int:
    """
    生成围棋专用的 SpaceTokenizer JSON（含 <me>/<enemy> 视角 token）。

    词表布局：
      ID 0     = <unk>
      ID 1     = <pad>
      ID 2     = <num>
      ID 3-258 = 256 个字节 token
      ID 259-620 = 362 个围棋坐标
      ID 621   = <me>
      ID 622   = <enemy>
      总词表大小 = 623
    """
    vocab = {}
    idx = 259

    # 361 个棋盘坐标
    for row in range(19):
        for col in range(19):
            coord = chr(ord('a') + col) + chr(ord('a') + row)
            hex_val = ''.join(f'{ord(c):02x}' for c in coord)
            vocab[str(idx)] = hex_val
            idx += 1

    # PASS
    hex_val = ''.join(f'{ord(c):02x}' for c in 'tt')
    vocab[str(idx)] = hex_val
    idx += 1

    # 视角 token
    for tag in ['<me>', '<enemy>']:
        hex_val = ''.join(f'{ord(c):02x}' for c in tag)
        vocab[str(idx)] = hex_val
        idx += 1

    tokenizer = {
        "type": "space_tokenizer",
        "vocab": vocab,
        "vocab_size": idx,
        "special_tokens": {
            "<unk>": 0,
            "<pad>": 1,
            "<num>": 2
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer, f, indent=2, ensure_ascii=False)

    return idx


# ═══════════════════════════════════════════════════════════════════════════
#  棋盘可视化
# ═══════════════════════════════════════════════════════════════════════════

def sgf_to_rc(coord: str) -> tuple[int, int]:
    if coord == 'tt':
        return (-1, -1)
    return (ord(coord[1]) - ord('a'), ord(coord[0]) - ord('a'))


def print_board(moves: list[str], last_n: int = 5):
    board = [[0] * 19 for _ in range(19)]
    last_positions = []
    for i, coord in enumerate(moves):
        r, c = sgf_to_rc(coord)
        if r < 0:
            continue
        color = 1 if i % 2 == 0 else 2
        board[r][c] = color
        last_positions.append((r, c, i))

    last_set = {(r, c) for r, c, _ in last_positions[-last_n:]}
    last_colors = {(r, c): (1 if i % 2 == 0 else 2)
                   for r, c, i in last_positions[-last_n:]}

    col_labels = '   ' + ' '.join('abcdefghjklmnopqrs')
    print(col_labels)
    print('  +' + '-' * 37 + '+')
    for r in range(19):
        row_str = f'{19 - r:2d}|'
        for c in range(19):
            if (r, c) in last_set:
                mark = 'X' if last_colors[(r, c)] == 1 else 'O'
                row_str += f'({mark})'
            elif board[r][c] == 1:
                row_str += ' X '
            elif board[r][c] == 2:
                row_str += ' O '
            else:
                row_str += ' . '
        row_str += f'|{19 - r}'
        print(row_str)
    print('  +' + '-' * 37 + '+')
    print(col_labels)


# ═══════════════════════════════════════════════════════════════════════════
#  Tokenizer 验证
# ═══════════════════════════════════════════════════════════════════════════

def verify_tokenizer(tokenizer_path: str, sample_text: str):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tok_data = json.load(f)

    id_to_token = {}
    for k, v in tok_data['vocab'].items():
        token_str = bytes.fromhex(v).decode('ascii')
        id_to_token[int(k)] = token_str
    token_to_id = {v: k for k, v in id_to_token.items()}

    def encode(text):
        tokens = []
        for word in text.split():
            if word in token_to_id:
                tokens.append(token_to_id[word])
            else:
                for b in word.encode('ascii'):
                    tokens.append(b + 3)
        return tokens

    def decode(ids):
        result = []
        for tid in ids:
            if tid in id_to_token:
                result.append(id_to_token[tid])
            else:
                result.append('<unk>')
        return ' '.join(result)

    encoded = encode(sample_text)
    decoded = decode(encoded)

    print(f"\n🔍 Tokenizer 验证:")
    print(f"  输入:  {sample_text}")
    print(f"  编码:  {encoded}")
    print(f"  解码:  {decoded}")
    print(f"  往返一致: {'✅' if decoded == sample_text else '❌ 不一致!'}")
    print(f"  词表大小: {tok_data['vocab_size']}")
    print(f"  Go token: 362 (ID 259~620)")
    print(f"  视角 token: <me>=621, <enemy>=622")


# ═══════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════

def collect_sgf_files(paths):
    files = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            if path.suffix.lower() == '.sgf':
                files.append(path)
        elif path.is_dir():
            found = sorted(path.rglob('*.sgf'))
            files.extend(found)
            print(f"📂 {path}: 找到 {len(found)} 个 SGF 文件")
        else:
            print(f"❌ 路径不存在: {path}", file=sys.stderr)
    return files


def main():
    parser = argparse.ArgumentParser(
        description='将 SGF 棋谱转换为 GPT 训练文本（胜利方视角）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
核心设计：只从胜利方视角训练。<me>=赢家，<enemy>=输家。
模型学到的是"赢家怎么下"。

示例：
  python go_sgf_to_text.py sgf_dir/ -o go_games.txt --tokenizer go_tokenizer.json
  python go_sgf_to_text.py sgf_dir/ --dry-run --max-games 10
""")
    parser.add_argument('input', nargs='+',
                        help='SGF 文件或包含 SGF 文件的目录')
    parser.add_argument('-o', '--output', default='go_games.txt',
                        help='输出文本文件 (默认: go_games.txt)')
    parser.add_argument('--tokenizer', default='go_tokenizer.json',
                        help='Tokenizer JSON (默认: go_tokenizer.json)')
    parser.add_argument('--min-moves', type=int, default=20,
                        help='最少落子数 (默认: 20)')
    parser.add_argument('--max-games', type=int, default=0,
                        help='最多处理局数, 0=不限 (默认: 0)')
    parser.add_argument('--dry-run', action='store_true',
                        help='只预览不写文件')
    parser.add_argument('--show-board', type=int, default=0, metavar='N',
                        help='显示前 N 局棋盘')
    args = parser.parse_args()

    # ── 收集文件 ──────────────────────────────────────
    sgf_files = collect_sgf_files(args.input)
    if not sgf_files:
        print("❌ 未找到任何 SGF 文件", file=sys.stderr)
        sys.exit(1)

    print(f"\n共 {len(sgf_files)} 个 SGF 文件待处理\n")

    # ── 解析 ──────────────────────────────────────────
    all_games = []      # [(moves, winner), ...]
    parse_errors = 0
    for sgf_file in sgf_files:
        try:
            text = sgf_file.read_text(encoding='utf-8', errors='replace')
            games = parse_sgf_games(text)
            for moves, winner in games:
                if len(moves) >= args.min_moves:
                    all_games.append((moves, winner))
        except Exception as e:
            parse_errors += 1
            print(f"⚠️  解析失败 {sgf_file.name}: {e}", file=sys.stderr)

    if not all_games:
        print(f"❌ 没有有效数据", file=sys.stderr)
        sys.exit(1)

    if args.max_games > 0:
        all_games = all_games[:args.max_games]

    # ── 统计结果 ──────────────────────────────────────
    total = len(all_games)
    has_result = sum(1 for _, w in all_games if w is not None)
    black_wins = sum(1 for _, w in all_games if w == 'B')
    white_wins = sum(1 for _, w in all_games if w == 'W')
    no_result = total - has_result

    print(f"📊 数据统计:")
    print(f"   总局数:     {total}")
    print(f"   有结果:     {has_result} (黑胜 {black_wins} / 白胜 {white_wins})")
    print(f"   无结果:     {no_result}（将跳过）")
    print(f"   训练序列:   {has_result} 条（仅胜利方视角）")

    total_moves = sum(len(m) for m, _ in all_games if _ is not None)
    avg_moves = total_moves / max(has_result, 1)
    print(f"   总落子数:   {total_moves}")
    print(f"   平均步数:   {avg_moves:.1f}")

    # ── 转换为训练序列 ────────────────────────────────
    all_sequences = []
    for moves, winner in all_games:
        seq = game_to_winner_perspective(moves, winner)
        if seq is not None:
            all_sequences.append(seq)

    # ── 预览 ──────────────────────────────────────────
    print(f"\n📝 胜利方视角预览（前 3 局有效棋）:")
    count = 0
    for moves, winner in all_games:
        if winner is None:
            continue
        seq = game_to_winner_perspective(moves, winner)
        if seq is None:
            continue
        winner_name = "黑方" if winner == 'B' else "白方"
        display = seq if len(seq) <= 120 else seq[:117] + '...'
        print(f"   局{count+1} (胜者={winner_name}): {display}")
        count += 1
        if count >= 3:
            break

    # ── 棋盘 ──────────────────────────────────────────
    if args.show_board > 0:
        count = 0
        for moves, winner in all_games:
            if winner is None:
                continue
            winner_name = "黑方" if winner == 'B' else "白方"
            print(f"\n棋盘 (胜者={winner_name}, 最后{min(5, len(moves))}手标记):")
            print_board(moves, last_n=5)
            count += 1
            if count >= args.show_board:
                break

    # ── Dry run ───────────────────────────────────────
    if args.dry_run:
        print(f"\n(--dry-run 模式)")
        vocab_size = generate_go_tokenizer(args.tokenizer)
        for moves, winner in all_games:
            if winner:
                sample = game_to_winner_perspective(moves, winner)
                verify_tokenizer(args.tokenizer, sample[:120])
                break
        return

    # ── 写文件 ────────────────────────────────────────
    with open(args.output, 'w', encoding='utf-8') as f:
        for seq in all_sequences:
            f.write(seq + '\n')
    print(f"\n✅ 文本文件: {args.output} ({len(all_sequences)} 条序列)")

    # ── Tokenizer ─────────────────────────────────────
    vocab_size = generate_go_tokenizer(args.tokenizer)
    print(f"✅ Tokenizer: {args.tokenizer} (vocab_size={vocab_size})")

    # ── 验证 ──────────────────────────────────────────
    if all_sequences:
        verify_tokenizer(args.tokenizer, all_sequences[0][:120])

    # ── 训练命令 ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🚀 训练命令:")
    print(f"  .\\build\\text_train.exe {args.output} \\")
    print(f"      --vocab {args.tokenizer} \\")
    print(f"      --d-model 128 --num-heads 4 --num-layers 4 --d-ff 512 \\")
    print(f"      --seq-len 600 --epochs 50 --lr 0.001 --batch-size 32 \\")
    print(f"      --save go_model.bin --cuda")
    print(f"\n  ⚠️ 视角 token 使序列长度翻倍，seq-len 需 ≥ 平均步数×2+2")
    print(f"     当前平均 {avg_moves:.0f} 步 → 建议 seq-len=600")
    print(f"\n🎯 推理:")
    print(f"  .\\build\\text_infer.exe --model go_model.bin \\")
    print(f"      --prompt \"<me> pd <enemy> dp\" --max-tokens 1 --temperature 0")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
