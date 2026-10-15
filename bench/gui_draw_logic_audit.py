# -*- coding: utf-8 -*-
"""GUI 画板绘制逻辑测试（P2-6「绘制节流」半边的可复现门禁）。

复现：`python bench/gui_draw_logic_audit.py`（无需 GUI / 无需 customtkinter：
本脚本用桩模块替换 `customtkinter` 后再导入 `gui.py`）。

背景：`gui.py` 的 MNIST 推理画板在拖动期曾"每个 motion 事件新建一个画布项
并写一次 PIL 缓冲"，快速拖动易卡。P2-6 改为：
  · 每笔只建**一个**画布项（原地 `coords` 刷新）——拖动期画布项数 = 笔数；
  · 空帧跳过（坐标未变的重复 motion 事件直接返回，不刷新）；
  · 视觉刷新**按帧节流**（`_PAINT_MS`；数据侧永不节流）；
  · PIL 光栅化**延迟**到边界（`_downsample_buffer` 是读缓冲的唯一入口）。

本测试两段：
  [A] `StrokeTrack` 纯逻辑语义（空帧 / 点累积 / 一次取出 / 单击成点 / 清空）；
  [B] 源码级不变量（拖动不做 create_line、缓冲读取前必先光栅化、结尾补一次刷新）
      —— 与 `bench/gui_cli_audit.py` 的 [3] 节同款"静态断言"风格。

退出码非 0 = 逻辑回归。
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Any:
    """万能桩：任何属性/调用都返回自身；作基类时解析为 object。"""

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()

    def __mro_entries__(self, bases):
        return (object,)


def _stub_customtkinter():
    mod = types.ModuleType("customtkinter")
    mod.__getattr__ = lambda name: _Any()   # PEP 562：模块级属性桩
    sys.modules["customtkinter"] = mod


def _load_gui():
    _stub_customtkinter()
    sys.path.insert(0, str(ROOT))
    import gui   # noqa: E402  （桩就位后再导入）
    return gui


def main() -> int:
    bad = 0
    gui = _load_gui()

    # ── [A] StrokeTrack 语义 ─────────────────────────────────────────
    print("=== [A] StrokeTrack 纯逻辑 ===")
    tr = gui.StrokeTrack()
    tr.begin(0, 0, 14, 280, 280)
    checks = []
    checks.append(("空帧跳过（同坐标重复事件不产生笔迹）",
                   tr.extend(0, 0) is False))
    checks.append(("新点返回 True（需要刷新画布）", tr.extend(5, 5) is True))
    checks.append(("点集按序累积", tr.live_points() == [(0, 0), (5, 5)]))
    checks.append(("结束一笔进入待画队列", tr.end() is True))
    drained = tr.drain()
    checks.append(("drain 一次性取走全部待画折线",
                   len(drained) == 1 and drained[0] == [(0, 0), (5, 5)]))
    checks.append(("drain 后再取为空（不重复光栅化）", tr.drain() == []))
    checks.append(("笔迹保留在 strokes()（供 _draw_grid 重放）",
                   len(tr.strokes()) == 1))
    checks.append(("结束后的笔迹宽度/画布尺寸已记录",
                   tr.strokes()[0]["width"] == 14 and tr.strokes()[0]["w"] == 280))

    tr.begin(7, 7, 10, 280, 280)          # 单击（无 motion）
    tr.end()
    click = tr.drain()
    checks.append(("单击成一笔（同坐标两点 + 圆头 = 一个点）",
                   len(click) == 1 and click[0] == [(7, 7), (7, 7)]))
    checks.append(("未开始一笔时 extend 不崩溃且为空帧",
                   tr.extend(1, 2) is False))
    tr.clear()
    checks.append(("clear 清空笔迹与待画队列",
                   tr.strokes() == [] and tr.drain() == []))
    for name, ok in checks:
        print(f"  [{'OK ' if ok else 'BAD'}] {name}")
        bad += 0 if ok else 1

    # ── [B] 源码级不变量（拖动路径复杂度封顶，AST 口径不受注释影响）──
    print("\n=== [B] 源码级不变量（拖动路径复杂度封顶）===")
    tree = ast.parse((ROOT / "gui.py").read_text(encoding="utf-8"))
    tab_cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == "MnistInferTab")
    methods = {n.name: n for n in tab_cls.body
               if isinstance(n, ast.FunctionDef)}

    def attrs_read(node):
        """该函数（不含嵌套函数）里 `self.<attr>` 的读取集合"""
        out = set()
        stack = list(node.body)
        while stack:
            cur = stack.pop()
            for sub in ast.iter_child_nodes(cur):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.Lambda)):
                    continue
                stack.append(sub)
            if isinstance(cur, ast.Attribute) and isinstance(cur.value, ast.Name) \
                    and cur.value.id == "self":
                out.add(cur.attr)
        return out

    def calls(node):
        """该函数（不含嵌套函数）里"被调用的属性名"集合"""
        out = set()
        stack = list(node.body)
        while stack:
            cur = stack.pop()
            for sub in ast.iter_child_nodes(cur):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.Lambda)):
                    continue
                stack.append(sub)
            if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
                out.add(cur.func.attr)
        return out

    move, start = methods["_on_draw_move"], methods["_on_draw_start"]
    end, ds = methods["_on_draw_end"], methods["_downsample_buffer"]
    rast, paint = methods["_rasterize_pending"], methods["_paint_live"]
    grid = methods["_draw_grid"]
    src_checks = [
        ("拖动不做 create_line/create_oval（不逐个 motion 新建画布项）",
         not ({"create_line", "create_oval"} & calls(move))),
        ("拖动先过空帧判定（_track.extend）", "_track" in attrs_read(move)),
        ("拖动刷新按帧节流（_PAINT_MS 判定 + 记录刷新时刻）",
         {"_PAINT_MS", "_last_paint"} <= attrs_read(move)),
        ("开始一笔不直接建项（交给 _paint_live）",
         not ({"create_line", "create_oval"} & calls(start))
         and "_paint_live" in calls(start)),
        ("结束一笔补一次视觉刷新 + 进入待光栅化",
         {"_paint_live", "_track"} <= attrs_read(end)),
        ("PIL 光栅化延迟：_paint_live 不建 ImageDraw",
         "ImageDraw" not in attrs_read(paint)),
        ("读缓冲前必先光栅化（唯一入口 _downsample_buffer）",
         "_rasterize_pending" in calls(ds)),
        ("光栅化按批取走折线（一次 drain + 一次 draw.line）",
         {"_track"} <= attrs_read(rast) and "drain" in calls(rast)
         and "line" in calls(rast)),
        ("重放走 _track.strokes()（不再读已删的笔迹属性）",
         "strokes" in calls(grid) and "create_line" in calls(grid)),
        ("画布项复用（_paint_live 原地 coords 刷新）",
         "coords" in calls(paint)),
    ]
    for name, ok in src_checks:
        print(f"  [{'OK ' if ok else 'BAD'}] {name}")
        bad += 0 if ok else 1

    # 附：确认删除的旧属性（MNIST 推理 Tab 上的 self._strokes）无人引用
    stale = sorted(
        n.lineno for n in ast.walk(tab_cls)
        if isinstance(n, ast.Attribute) and n.attr == "_strokes")
    print(f"  [{'OK ' if not stale else 'BAD'}] 旧 self._strokes 引用 = {stale or '无'}")
    bad += 0 if not stale else 1

    print(f"\n{'PASS' if bad == 0 else 'FAIL'}  可行动问题={bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
