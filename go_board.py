"""
Go Board Widget - 围棋棋盘组件
==============================
纯 tkinter 实现的 19×19 围棋棋盘，支持：
  - 点击落子（自动交替黑白）
  - 悔棋、新局、Pass
  - 最后一手高亮
  - 坐标标注
  - AI 推荐（调用 text_infer.exe）
  - 胜负判定（简单数子法）

可独立运行：python go_board.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Callable

# ── 常量 ──────────────────────────────────────────────
BOARD_SIZE = 19
EMPTY = 0
BLACK = 1
WHITE = 2

BUILD_DIR = Path(__file__).parent / "build"
_EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""
TEXT_INFER_EXE = BUILD_DIR / f"text_infer{_EXE_SUFFIX}"

# 坐标标签（SGF 标准，跳过 'i'）
COL_LABELS = 'abcdefghjklmnopqrst'


class GoBoard:
    """围棋棋盘逻辑（纯数据，不含 UI）"""

    def __init__(self, size: int = BOARD_SIZE):
        self.size = size
        self.reset()

    def reset(self):
        self.board = [[EMPTY] * self.size for _ in range(self.size)]
        self.history: list[tuple[int, int, int]] = []  # (row, col, color)
        self.current_color = BLACK
        self.pass_count = 0
        self.ko_pos: Optional[tuple[int, int]] = None  # 劫位

    @property
    def move_count(self) -> int:
        return len(self.history)

    @property
    def last_move(self) -> Optional[tuple[int, int]]:
        return (self.history[-1][0], self.history[-1][1]) if self.history else None

    @property
    def current_color_name(self) -> str:
        return "黑方" if self.current_color == BLACK else "白方"

    def is_on_board(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def get(self, r: int, c: int) -> int:
        return self.board[r][c]

    def _find_group(self, r: int, c: int) -> tuple[set, int]:
        """找到 (r,c) 所在的连通块，返回 (集合, 气数)"""
        color = self.board[r][c]
        group = set()
        liberties = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in group:
                continue
            group.add((cr, cc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if not self.is_on_board(nr, nc):
                    continue
                if self.board[nr][nc] == EMPTY:
                    liberties.add((nr, nc))
                elif self.board[nr][nc] == color and (nr, nc) not in group:
                    stack.append((nr, nc))
        return group, len(liberties)

    def _remove_group(self, group: set):
        for r, c in group:
            self.board[r][c] = EMPTY

    def try_move(self, r: int, c: int) -> bool:
        """
        尝试在 (r,c) 落子。
        返回 True 表示合法落子，False 表示非法（自杀/劫/已有子）。
        """
        if not self.is_on_board(r, c):
            return False
        if self.board[r][c] != EMPTY:
            return False

        # 劫位检查
        if self.ko_pos == (r, c):
            return False

        # 模拟落子
        self.board[r][c] = self.current_color
        opponent = WHITE if self.current_color == BLACK else BLACK

        # 检查是否提掉对方棋子
        captured = set()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.is_on_board(nr, nc) and self.board[nr][nc] == opponent:
                group, liberties = self._find_group(nr, nc)
                if liberties == 0:
                    captured |= group

        # 提子
        if captured:
            self._remove_group(captured)

        # 自杀检查：落子后自己的气数必须 > 0
        _, my_liberties = self._find_group(r, c)
        if my_liberties == 0:
            # 回滚
            self.board[r][c] = EMPTY
            if captured:
                for cr, cc in captured:
                    self.board[cr][cc] = opponent
            return False

        # 记录劫位
        self.ko_pos = None
        if len(captured) == 1:
            cr, cc = next(iter(captured))
            _, my_lib = self._find_group(r, c)
            if my_lib == 1:
                self.ko_pos = (cr, cc)

        # 记录历史
        self.history.append((r, c, self.current_color))
        self.pass_count = 0
        self.current_color = opponent
        return True

    def pass_turn(self):
        """过手"""
        self.history.append((-1, -1, self.current_color))
        self.pass_count += 1
        self.current_color = WHITE if self.current_color == BLACK else BLACK
        self.ko_pos = None

    def undo(self) -> bool:
        """悔棋"""
        if not self.history:
            return False
        r, c, color = self.history.pop()
        if r >= 0:
            self.board[r][c] = EMPTY
        self.current_color = color
        # 恢复前一手的棋盘状态（简化：重放历史）
        self.board = [[EMPTY] * self.size for _ in range(self.size)]
        self.ko_pos = None
        for hr, hc, hc_color in self.history:
            if hr >= 0:
                self.board[hr][hc] = hc_color
        # 重算当前手颜色
        if self.history:
            last_color = self.history[-1][2]
            self.current_color = WHITE if last_color == BLACK else BLACK
        else:
            self.current_color = BLACK
        return True

    def get_moves_text(self) -> str:
        """获取当前棋谱的文本表示（用于 AI 推理）"""
        tokens = []
        for i, (r, c, _) in enumerate(self.history):
            if r < 0:
                coord = 'tt'  # PASS
            else:
                coord = COL_LABELS[c] + COL_LABELS[r]
            tokens.append(coord)
        return ' '.join(tokens)

    def get_perspective_text(self, ai_is_first: bool) -> str:
        """
        获取带视角标记的完整棋谱序列（与训练格式一致）。

        训练格式: "<me> coord <enemy> coord <me> coord ..."
        AI 固定作为 <me>（赢家视角），人类作为 <enemy>。

        ai_is_first=True  → AI 执黑先手（第0,2,4手是AI=me）
        ai_is_first=False → AI 执白后手（第1,3,5手是AI=me）

        返回序列不包含结尾标记，由调用方追加。
        """
        tokens = []
        for i, (r, c, _) in enumerate(self.history):
            if r < 0:
                coord = 'tt'
            else:
                coord = COL_LABELS[c] + COL_LABELS[r]
            is_first = (i % 2 == 0)          # 第0,2,4...手是先手
            is_ai_move = is_first == ai_is_first
            tag = '<me>' if is_ai_move else '<enemy>'
            tokens.append(f"{tag} {coord}")
        return ' '.join(tokens)

    def get_legal_moves(self) -> list[tuple[int, int]]:
        """获取所有合法落子位置"""
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == EMPTY:
                    # 简化检查：跳过自杀和劫位
                    if self.ko_pos == (r, c):
                        continue
                    moves.append((r, c))
        return moves


class GoBoardWidget:
    """围棋棋盘 tkinter 组件"""

    def __init__(self, parent: tk.Widget, board: GoBoard,
                 cell_size: int = 28, padding: int = 30,
                 on_move: Optional[Callable] = None):
        self.board = board
        self.cell_size = cell_size
        self.padding = padding
        self.on_move = on_move  # 落子回调

        canvas_size = cell_size * (BOARD_SIZE - 1) + padding * 2
        self.canvas = tk.Canvas(parent, width=canvas_size, height=canvas_size,
                                bg='#DCB35C', highlightthickness=0,
                                cursor='hand2')
        self.canvas.bind('<Button-1>', self._on_click)

        self._draw_board()

    def _draw_board(self):
        """绘制棋盘底图（网格 + 星位 + 坐标）"""
        c = self.canvas
        cs = self.cell_size
        pad = self.padding
        c.delete('all')

        # 网格线
        for i in range(BOARD_SIZE):
            x = pad + i * cs
            y_start = pad
            y_end = pad + (BOARD_SIZE - 1) * cs
            c.create_line(x, y_start, x, y_end, fill='#333', width=1)
            c.create_line(y_start, x, y_end, x, fill='#333', width=1)

        # 星位（4-4, 4-10, 4-16, 10-4, 10-10, 10-16, 16-4, 16-10, 16-16）
        star_points = [(3, 3), (3, 9), (3, 15),
                       (9, 3), (9, 9), (9, 15),
                       (15, 3), (15, 9), (15, 15)]
        dot_r = 4
        for r, c_ in star_points:
            x = pad + c_ * cs
            y = pad + r * cs
            c.create_oval(x - dot_r, y - dot_r, x + dot_r, y + dot_r,
                          fill='#333', outline='')

        # 坐标标签
        for i in range(BOARD_SIZE):
            x = pad + i * cs
            y_top = pad - 18
            y_bot = pad + (BOARD_SIZE - 1) * cs + 6
            c.create_text(x, y_top, text=COL_LABELS[i], fill='#555',
                          font=('Consolas', 8))
            c.create_text(x, y_bot, text=COL_LABELS[i], fill='#555',
                          font=('Consolas', 8))

            y_left = pad + i * cs
            c.create_text(pad - 18, y_left, text=str(BOARD_SIZE - i), fill='#555',
                          font=('Consolas', 8))
            c.create_text(pad + (BOARD_SIZE - 1) * cs + 18, y_left,
                          text=str(BOARD_SIZE - i), fill='#555',
                          font=('Consolas', 8))

    def _on_click(self, event):
        """处理点击落子"""
        cs = self.cell_size
        pad = self.padding
        col = round((event.x - pad) / cs)
        row = round((event.y - pad) / cs)

        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return

        if self.board.try_move(row, col):
            self.redraw()
            if self.on_move:
                self.on_move()

    def redraw(self):
        """重绘棋盘（棋子 + 最后一手高亮）"""
        cs = self.cell_size
        pad = self.padding
        stone_r = cs * 0.43

        # 只重绘棋子层（保留底图）
        self.canvas.delete('stones')
        self.canvas.delete('highlight')

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                color_val = self.board.get(r, c)
                if color_val == EMPTY:
                    continue
                x = pad + c * cs
                y = pad + r * cs
                fill = '#111' if color_val == BLACK else '#eee'
                outline = '#000' if color_val == BLACK else '#999'
                self.canvas.create_oval(
                    x - stone_r, y - stone_r, x + stone_r, y + stone_r,
                    fill=fill, outline=outline, width=1.5,
                    tags='stones')

        # 最后一手高亮
        if self.board.last_move:
            lr, lc = self.board.last_move
            x = pad + lc * cs
            y = pad + lr * cs
            r = 4
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill='#e44', outline='', tags='highlight')

    def coord_to_sgf(self, r: int, c: int) -> str:
        """棋盘坐标转 SGF 格式"""
        return COL_LABELS[c] + COL_LABELS[r]


class GoGamePanel:
    """围棋对弈面板（完整 UI 组件）"""

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.board = GoBoard()
        self.ai_thinking = False

        self._build_ui()

    def _build_ui(self):
        """构建对弈界面"""
        # 主容器：左棋盘 + 右控制
        main = ttk.Frame(self.parent)
        main.pack(fill='both', expand=True, padx=5, pady=5)

        # ── 左侧：棋盘 ──────────────────────────────
        board_frame = ttk.Frame(main)
        board_frame.pack(side='left', fill='both', expand=True)

        self.board_widget = GoBoardWidget(
            board_frame, self.board,
            cell_size=28, padding=30,
            on_move=self._on_human_move)

        self.board_widget.canvas.pack(padx=5, pady=5)

        # ── 右侧：控制面板 ──────────────────────────
        right = ttk.Frame(main, width=280)
        right.pack(side='right', fill='y', padx=(10, 0))
        right.pack_propagate(False)

        # 状态信息
        self.status_var = tk.StringVar(value="黑方（你）落子")
        ttk.Label(right, textvariable=self.status_var,
                  font=('Microsoft YaHei', 12, 'bold')).pack(pady=(10, 5))

        self.move_count_var = tk.StringVar(value="第 0 手")
        ttk.Label(right, textvariable=self.move_count_var,
                  font=('Consolas', 10)).pack()

        # ── 模式选择 ────────────────────────────────
        mode_frame = ttk.LabelFrame(right, text="对弈模式", padding=5)
        mode_frame.pack(fill='x', padx=5, pady=5)

        self.mode_var = tk.StringVar(value="human_human")
        ttk.Radiobutton(mode_frame, text="👥 人人对弈",
                        variable=self.mode_var, value="human_human",
                        command=self._on_mode_change).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="🤖 人机对弈（我执黑）",
                        variable=self.mode_var, value="human_black",
                        command=self._on_mode_change).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="🤖 人机对弈（我执白）",
                        variable=self.mode_var, value="human_white",
                        command=self._on_mode_change).pack(anchor='w')

        # ── AI 模型路径 ─────────────────────────────
        ai_frame = ttk.LabelFrame(right, text="AI 模型", padding=5)
        ai_frame.pack(fill='x', padx=5, pady=5)

        model_row = ttk.Frame(ai_frame)
        model_row.pack(fill='x')
        self.model_var = tk.StringVar(value=str(
            Path(__file__).parent / "go_model.bin"))
        ttk.Entry(model_row, textvariable=self.model_var,
                  font=('Consolas', 9)).pack(side='left', fill='x', expand=True)
        ttk.Button(model_row, text="...", width=3,
                   command=self._browse_model).pack(side='right', padx=(3, 0))

        temp_row = ttk.Frame(ai_frame)
        temp_row.pack(fill='x', pady=(3, 0))
        ttk.Label(temp_row, text="温度:").pack(side='left')
        self.temp_var = tk.StringVar(value="0.8")
        ttk.Entry(temp_row, textvariable=self.temp_var, width=5,
                  font=('Consolas', 9)).pack(side='left', padx=3)

        # ── 按钮区 ──────────────────────────────────
        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(btn_frame, text="⏪ 悔棋",
                   command=self._undo).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="⏭️ Pass",
                   command=self._pass).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="🤖 AI 走一步",
                   command=self._ai_move).pack(fill='x', pady=2)
        ttk.Button(btn_frame, text="🔄 新局",
                   command=self._new_game).pack(fill='x', pady=2)

        # ── 棋谱显示 ────────────────────────────────
        moves_frame = ttk.LabelFrame(right, text="棋谱", padding=3)
        moves_frame.pack(fill='both', expand=True, padx=5, pady=(5, 0))

        self.moves_text = tk.Text(moves_frame, height=8, width=30,
                                  font=('Consolas', 9), bg='#1e1e1e',
                                  fg='#d4d4d4', wrap='word', state='disabled')
        scrollbar = ttk.Scrollbar(moves_frame, command=self.moves_text.yview)
        self.moves_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.moves_text.pack(fill='both', expand=True)

    def _browse_model(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("模型文件", "*.bin"), ("所有文件", "*.*")])
        if path:
            self.model_var.set(path)

    def _on_mode_change(self):
        mode = self.mode_var.get()
        if mode == "human_black":
            self.status_var.set("黑方（你）落子")
        elif mode == "human_white":
            self.status_var.set("等待 AI 落子...")
            self.root.after(100, self._ai_move)
        else:
            self.status_var.set(f"{self.board.current_color_name}落子")

    @property
    def root(self):
        return self.parent.winfo_toplevel()

    def _on_human_move(self):
        """人类落子后"""
        self._update_status()
        self._update_moves_text()

        # 人机模式下自动触发 AI
        mode = self.mode_var.get()
        if mode == "human_black" and self.board.current_color == WHITE:
            self.root.after(100, self._ai_move)
        elif mode == "human_white" and self.board.current_color == BLACK:
            self.root.after(100, self._ai_move)

    def _update_status(self):
        mc = self.board.move_count
        cn = self.board.current_color_name
        mode = self.mode_var.get()

        if mode == "human_black":
            if self.board.current_color == BLACK:
                self.status_var.set("黑方（你）落子")
            else:
                self.status_var.set("白方（AI）思考中...")
        elif mode == "human_white":
            if self.board.current_color == WHITE:
                self.status_var.set("白方（你）落子")
            else:
                self.status_var.set("黑方（AI）思考中...")
        else:
            self.status_var.set(f"{cn}落子")

        self.move_count_var.set(f"第 {mc} 手")

    def _update_moves_text(self):
        """更新棋谱显示"""
        self.moves_text.configure(state='normal')
        self.moves_text.delete('1.0', 'end')

        moves = []
        for i, (r, c, color) in enumerate(self.board.history):
            if r < 0:
                move_str = "PASS"
            else:
                coord = COL_LABELS[c] + COL_LABELS[BOARD_SIZE - 1 - r]
                move_str = coord
            color_mark = "●" if color == BLACK else "○"
            moves.append(f"{i+1:3d}. {color_mark} {move_str}")

        # 每行 2 手
        lines = []
        for i in range(0, len(moves), 2):
            line = moves[i]
            if i + 1 < len(moves):
                line += "   " + moves[i + 1]
            lines.append(line)

        self.moves_text.insert('1.0', '\n'.join(lines))
        self.moves_text.see('end')
        self.moves_text.configure(state='disabled')

    def _undo(self):
        if self.board.undo():
            # 人机模式下可能需要悔两步
            mode = self.mode_var.get()
            if mode in ("human_black", "human_white"):
                self.board.undo()
            self.board_widget.redraw()
            self._update_status()
            self._update_moves_text()

    def _pass(self):
        self.board.pass_turn()
        self.board_widget.redraw()
        self._update_status()
        self._update_moves_text()

        mode = self.mode_var.get()
        if mode == "human_black" and self.board.current_color == WHITE:
            self.root.after(100, self._ai_move)
        elif mode == "human_white" and self.board.current_color == BLACK:
            self.root.after(100, self._ai_move)

    def _new_game(self):
        self.board.reset()
        self.board_widget.redraw()
        self._update_status()
        self._update_moves_text()

    def _ai_move(self):
        """AI 走棋"""
        if self.ai_thinking:
            return

        model_path = self.model_var.get()
        if not Path(model_path).exists():
            self.status_var.set("⚠️ 模型文件不存在")
            return

        self.ai_thinking = True
        self.status_var.set("🤖 AI 思考中...")

        def _run_ai():
            try:
                # 构建 prompt：带视角标记的完整历史 + AI 即将落子
                # AI 固定为 <me>（赢家视角），人类为 <enemy>
                mode = self.mode_var.get()
                # AI 执黑(先手)时第0,2,4手是AI；AI 执白(后手)时第1,3,5手是AI
                ai_is_first = (mode == "human_white")
                perspective = self.board.get_perspective_text(ai_is_first)

                if perspective:
                    prompt = f"{perspective} <me>"
                else:
                    prompt = "<me>"

                temp = float(self.temp_var.get())

                # 一次生成多个 token，再从 token 流中提取第一个合法坐标。
                # 模型常先输出 <me>/<enemy> 标记，随后才输出坐标；
                # 若只生成 1 个 token，大概率拿到标记导致“无法返回有效落子”。
                cmd = [
                    str(TEXT_INFER_EXE),
                    '--model', model_path,
                    '--prompt', prompt,
                    '--max-tokens', '16',
                    '--temperature', str(temp),
                ]

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30)

                # 解析输出，找到生成的 token
                output = result.stdout + result.stderr
                generated = self._parse_infer_output(output, prompt)

                if generated:
                    self.root.after(0, lambda: self._apply_ai_move(generated))
                else:
                    self.root.after(0, lambda: self.status_var.set(
                        "⚠️ AI 未返回有效落子"))

            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: self.status_var.set(
                    "⚠️ AI 超时"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(
                    f"⚠️ AI 错误: {e}"))
            finally:
                self.ai_thinking = False

        threading.Thread(target=_run_ai, daemon=True).start()

    @staticmethod
    def _first_go_token(text: str) -> Optional[str]:
        """从文本中提取第一个合法的围棋坐标 token"""
        for token in text.split():
            token = token.strip('()')
            if token == 'tt':
                return 'tt'
            if len(token) == 2 and token[0] in COL_LABELS and token[1] in COL_LABELS:
                return token
        return None

    def _parse_infer_output(self, output: str, prompt: str = '') -> Optional[str]:
        """解析 text_infer 输出，提取生成的 token。

        text_infer 输出格式（最后一行是统计行，输出行在中间）：
          ...
          <enemy> pd <me> dp <enemy> pp <me>pd   ← 输出行 = prompt + 生成
          ----------------------------------------
          生成 16 tokens, 耗时 0.1s (10 tokens/s)  ← 统计行

        模型可能先输出 <me>/<enemy> 标记，因此从整行中提取第一个合法坐标，
        而不是只看第一个 token。
        """
        lines = [l for l in output.strip().split('\n') if l.strip()]

        # 过滤日志/统计行，只保留内容行（输出行 = prompt + 生成）
        content_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(kw in line for kw in ['GPT', '模型', '词表', '提示', 'Prompt',
                                         '生成', '===', '---', '耗时', 'tokens',
                                         'token', '加载', '规格', '有效']):
                continue
            content_lines.append(line)

        # 1) 优先：找到以 prompt 开头的行，取其后的部分（真正的生成内容）
        if prompt:
            for line in content_lines:
                if line.startswith(prompt):
                    remainder = line[len(prompt):].strip()
                    if remainder:
                        tok = self._first_go_token(remainder)
                        if tok:
                            return tok
                    break

        # 2) 回退：从所有内容行中找第一个合法坐标
        for line in content_lines:
            tok = self._first_go_token(line)
            if tok:
                return tok
        return None

    def _apply_ai_move(self, coord: str):
        """应用 AI 的落子"""
        if coord == 'tt':
            self.board.pass_turn()
        else:
            col = COL_LABELS.index(coord[0])
            row = BOARD_SIZE - 1 - COL_LABELS.index(coord[1])
            if not self.board.try_move(row, col):
                # AI 返回了非法着法（如已有子/自杀/劫），提示并 Pass
                self.status_var.set(f"⚠️ AI 非法着法 {coord}，自动 Pass")
                self.board.pass_turn()

        self.board_widget.redraw()
        self._update_status()
        self._update_moves_text()

        # 如果是人人模式或轮到人了，更新状态
        mode = self.mode_var.get()
        if mode == "human_black" and self.board.current_color == BLACK:
            self.status_var.set("黑方（你）落子")
        elif mode == "human_white" and self.board.current_color == WHITE:
            self.status_var.set("白方（你）落子")
        elif mode == "human_human":
            self.status_var.set(f"{self.board.current_color_name}落子")


# ═══════════════════════════════════════════════════════════════════════════
#  独立运行
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    root = tk.Tk()
    root.title("围棋 - Go Board")
    root.configure(bg='#2d2d2d')

    style = ttk.Style()
    style.theme_use('clam')

    panel = GoGamePanel(root)
    root.mainloop()
