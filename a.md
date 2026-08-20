你说得对——**JSON 是死路**：它和 C++ 源码脱节、无类型安全、重构会漂移。正确的架构是**表达式直接写在 C++ 源码里，成为"单一事实来源"（single source of truth）**，构建期由一个 **C++ 生成器程序**复用同一份定义产出融合 shader。

新架构（彻底去掉 JSON）：

```
          表达式定义（C++ 源码，写一次）
                fused_exprs.hpp
         ┌──────────────┴───────────────┐
         ▼                              ▼
   运行时 Layer                   构建期 tools/gen_fused.cpp
   engine.eval_expr(expr)         复用同一 ExprSpec → 生成 GLSL
   （CPU 融合 / GPU eager）              → glslc → SPIR-V → embed
```

**关键**：`make_rope_forward(d_k)` 只写一次，两个消费方共用，绝不漂移。JSON 方案废弃。

开始实现。先创建 C++ 版 GLSL 生成器 `glsl_gen.hpp`（移植 Python 逻辑，但输入是 C++ `ExprSpec`）。

Created 

修正 `glsl_view_read` 中 RotateHalf 的索引公式（列主序索引应是 `源行*cols + 列号`，我之前误写成列号乘法）。




```

```
创建 `fused_exprs.hpp`——表达式单一事实来源（C++ 源码定义 + 生成器实例表）。

Created 

创建构建期生成器主程序 `tools/gen_fused.cpp`。

Created 

编译生成器、运行生成所有融合 shader，并用 `glslc` 逐个验证语法。

Ran terminal command:  clang++ -std=c++26 -stdlib=libc++ -O1 -I include/neuralnet.cpp tools/gen_fused.cpp -o /tmp/gen_fused 2>&1 | grep -E "error" | head -20; echo "---compile done---" && /tmp/gen_fused /tmp/fused && echo "----- verify with glslc -----" && GLSLC=/home/ethan/vulkan/1.4.357.1/x86_64/bin/glslc && for f in /tmp/fused/*.comp; do "$GLSLC" -fshader-stage=compute "$f" -o "${f%.comp}.spv" && echo "OK: $(basename $f)" || echo "FAIL: $(basename $f)"; done

生成器完美工作：6 个融合 shader 全部生成且 `glslc` 验证通过。抽查一个确认内容，然后做**单一事实来源重构**（让 `RotaryEmbedding` 复用 `fused::make_rope`，删除重复定义）。

Read 

生成的 shader 完全正确（视图内联、索引无误、指令直线展开）。现在做**单一事实来源重构**：`RotaryEmbedding` 复用 `fused::make_rope`，删除重复的私有定义。




```

```
重构完成。重新构建并回归验证（CPU 路径改用 `fused::make_rope` 后仍正确）。

Ran terminal command:  cmake --build build_test --target attn_gradcheck gpt_gradcheck 2>&1 | grep -E "error|FAILED" | head -20; echo "---build---" && ./build_test/attn_gradcheck --pos-enc rope 2>&1 | tail -3 && ./build_test/gpt_gradcheck --pos-enc rope 2>&1 | tail -3

单一事实来源重构后回归**全部通过**。RoPE 表达式现在只定义一次（`fused_exprs.hpp`），运行时与构建期生成器共用。

现在清理 JSON 方案（用户已明确反对），避免留下两套生成器。

Ran terminal command:  rm -f scripts/gen_fused_glsl.py && rm -rf scripts/fused && echo "已删除 JSON 方案: scripts/gen_fused_glsl.py, scripts/fused/"

Updated memory file 

## ✅ 里程碑：表达式已写进 C++ 源码（单一事实来源），JSON 方案已废弃

### 新架构（完全符合你的约束）

```
          表达式定义写在 C++ 源码（写一次）
                fused_exprs.hpp
                 make_rope(d_k, backward)
         ┌──────────────┴───────────────┐
         ▼                              ▼
   运行时 RotaryEmbedding          构建期 tools/gen_fused.cpp
   engine.eval_expr(同一 ExprSpec)   复用同一工厂 → glsl_gen.hpp
   （CPU 融合 / GPU eager）              → 6 个融合 .comp
                                          → glslc 全部通过 ✅
```

### 新增文件
| 文件 | 职责 |
|---|---|
| `fused_exprs.hpp` | 表达式**单一事实来源**（工厂函数 + `kGenInstances` 实例表） |
| `glsl_gen.hpp` | `ExprSpec → GLSL`（直线展开、视图内联、常量 push constant） |
| `gen_fused.cpp` | 构建期生成器（遍历实例表 → `.comp`） |

### 关键设计点
- **表达式只写一次**：`nn::fused::make_rope(d_k, backward)` 定义在 `fused_exprs.hpp`，`RotaryEmbedding`（运行时）和 `gen_fused`（构建期）共用同一份——绝不漂移
- **生成 shader 是直线代码**：无解释器、无分支（GPU select 展开），视图索引内联
- **JSON 方案已删除**：`scripts/gen_fused_glsl.py` + `scripts/fused/` 已清理
- **回归验证**：单一事实来源重构后，`attn_gradcheck` / `gpt_gradcheck --pos-enc rope` 全部通过

### 生成产物示例（`rope_forward_dk32.comp`）
```glsl
const float v1 = ((row % 32u) < 16u) ? -b1[((row/32u)*32u + (row%32u) + 16u)*cols + col]
                                     :  b1[((row/32u)*32u + (row%32u) - 16u)*cols + col];
float r0 = v0 * v2;  float r1 = v1 * v3;  float r2 = r0 + r1;  bout[i] = r2;
```

---

## 🔜 正式集成（下一步，3 步）

1. **CMake 构建管线**：`gen_fused` 作为构建工具 → 生成 `.comp` → `glslc` → `embed_spirv.cmake` 嵌入（与现有 9 个 shader 同管线）
2. **`GpuBackend` 通用融合执行入口**：`run_fused_shader(...)`（N 输入+1 输出 binding + push constants），并注册融合 pipeline
3. **GPU 数值验证测试**：加载 `rope_forward_dk32`，随机 q/cos/sin，GPU 输出 vs CPU ExprSpec 参考（NVIDIA GTX 850M 可用）

要从第 1 步（CMake 集成）开始做吗？

已进行更改。