# 📋 理解检查清单

> 用于评估对项目的理解程度，识别知识盲区，制定学习计划。

---

## 🎯 使用方法

1. **自评打分**：每个问题打分（1-5分）
2. **识别盲区**：得分低于3分的领域需要重点学习
3. **制定计划**：优先学习得分最低的领域
4. **定期回顾**：每两周重新评估一次

---

## 📊 评估维度

### 1. 基础数据结构（L0-L1）

#### Matrix 和 Tensor
- [ ] 我能解释 `Matrix` 的存储方式（行主序、`data_[row*cols+col]`）
- [ ] 我能区分 `Matrix`（CPU）和 `Tensor`（跨设备）
- [ ] 我理解 `Tensor` 的 `shared_ptr` 实现（零拷贝传递）
- [ ] 我能解释 `Device::CPU` 和 `Device::GPU` 的区别
- [ ] 我知道如何创建 `Matrix` 和 `Tensor`

**自评分数**：____/5

#### 数据布局
- [ ] 我理解 batch-major 布局（`i = b*seq + t`）
- [ ] 我能解释为什么需要 batch-major（避免跨样本串扰）
- [ ] 我知道多头注意力的特殊布局（`(H*d_k, batch*seq)`）
- [ ] 我能解释 `rearrange_3d` 的作用
- [ ] 我理解位置编码如何与 batch-major 布局配合

**自评分数**：____/5

---

### 2. 计算引擎（L2）

#### ComputeEngine 接口
- [ ] 我能解释"引擎化"设计（同一个 Layer 适配 CPU/GPU）
- [ ] 我知道 `ComputeEngine` 提供哪些原语（matmul, add, exp 等）
- [ ] 我理解 `begin_batch/end_batch` 的作用（GPU command buffer）
- [ ] 我能解释 `engine.from_matrix()` 和 `engine.to_matrix()` 的区别
- [ ] 我知道如何选择 CPU 或 GPU 引擎

**自评分数**：____/5

#### 张量操作
- [ ] 我能解释 `Tensor` 的 `forward/backward` 流程
- [ ] 我理解梯度如何传播（`backward` 返回梯度）
- [ ] 我知道 `zero_grad()` 的作用
- [ ] 我能解释 `model.parameters()` 和 `model.param_gradients()`
- [ ] 我理解优化器如何更新参数（`optimizer.step()`）

**自评分数**：____/5

---

### 3. 神经网络层（L2）

#### Layer 基础
- [ ] 我能解释 `Layer` 的接口（`forward/backward`）
- [ ] 我理解 `Linear` 层的计算（矩阵乘法 + 偏置）
- [ ] 我能解释 `ReLU` 的实现（`max(x, 0)`）
- [ ] 我知道 `GeLU` 比 `ReLU` 复杂在哪里
- [ ] 我理解 `Softmax` 的数值稳定性实现

**自评分数**：____/5

#### 高级层
- [ ] 我能解释 `Attention` 层的工作原理
- [ ] 我理解多头注意力（`MultiHeadAttention`）
- [ ] 我知道 `Transformer` 层如何组合 Attention 和 FeedForward
- [ ] 我能解释 `GPTModel` 的特殊设计（因果掩码、位置编码）
- [ ] 我理解 `RAPT` 和 `ZiPT` 的创新点（可选）

**自评分数**：____/5

---

### 4. 模型容器（L3）

#### Model 类
- [ ] 我能解释 `Model` 如何管理多个 `Layer`
- [ ] 我理解链式构建（`model.add_linear().add_relu()`）
- [ ] 我知道 `ModelSpec` 的作用（架构描述）
- [ ] 我能解释模型序列化（`save_model/load_model`）
- [ ] 我理解 `.nnpkg` 训练包的结构

**自评分数**：____/5

#### 训练流程
- [ ] 我能解释完整的训练循环（zero_grad → forward → loss → backward → step）
- [ ] 我理解为什么需要 `loss.backward()` 和 `model.backward()`
- [ ] 我知道如何选择优化器（SGD/Adam/AdamW/Muon）
- [ ] 我能解释学习率调度（学习率衰减）
- [ ] 我理解梯度裁剪（gradient clipping）

**自评分数**：____/5

---

### 5. 表达式 DSL 与 AOT 融合（GPU 专用）

#### 表达式 DSL
- [ ] 我能解释 `nn::dsl` 的作用（统一表达式接口）
- [ ] 我理解 `start_expr/end_expr` 块式融合
- [ ] 我知道 CPU 和 GPU 如何处理表达式（编译期 vs AOT）
- [ ] 我能解释 `ExprSpec` 的作用（扁平 IR）
- [ ] 我理解 `expr_spec_key` 的作用（匹配预编译 shader）

**自评分数**：____/5

#### AOT 融合管线
- [ ] 我能解释 `scan_exprs` 的作用（收集表达式结构）
- [ ] 我理解 `gen_fused` 的作用（生成融合 shader）
- [ ] 我知道 `fused_registry.hpp` 的作用（运行时匹配）
- [ ] 我能解释"闭合世界"原则（未命中硬报错）
- [ ] 我理解为什么不能运行时编译（安全性、性能）

**自评分数**：____/5

---

### 6. GPU 后端（Vulkan）

#### Vulkan 基础
- [ ] 我能解释 Vulkan 与 OpenGL 的区别
- [ ] 我理解 command buffer 的作用
- [ ] 我知道 SPIR-V 是什么（GPU 中间表示）
- [ ] 我能解释 `glslc` 编译器的作用
- [ ] 我理解 GPU 内存管理（buffer、device memory）

**自评分数**：____/5

#### GPU 实现
- [ ] 我能解释 `GpuEngine` 如何实现 `ComputeEngine` 接口
- [ ] 我理解 `compute_vk_backend.hpp` 的作用
- [ ] 我知道 `shaders/*.comp` 文件的作用
- [ ] 我能解释 `VK_ERROR_DEVICE_LOST` 的含义
- [ ] 我理解 `pending_destroys_` 延迟销毁队列

**自评分数**：____/5

---

### 7. 构建与工具链

#### 构建系统
- [ ] 我能解释 CMakeLists.txt 的结构
- [ ] 我理解 Ninja 构建系统的优势
- [ ] 我知道如何启用/禁用 Vulkan 支持
- [ ] 我能解释测试系统（`NN_ENABLE_TESTS=ON`）
- [ ] 我理解编译选项（`-O3 -march=native -fno-exceptions`）

**自评分数**：____/5

#### 开发工具
- [ ] 我知道如何运行测试（`ctest --test-dir build`）
- [ ] 我能解释 `compile_commands.json` 的作用
- [ ] 我理解 `perf_smoke.cpp` 的作用（性能测试）
- [ ] 我知道如何调试 GPU 问题（`gpu_test`）
- [ ] 我能使用 `gui.py` 进行可视化

**自评分数**：____/5

---

## 📈 总分计算

| 领域 | 得分 | 权重 | 加权分 |
|------|------|------|--------|
| 基础数据结构 | ____/10 | 20% | ____ |
| 计算引擎 | ____/10 | 20% | ____ |
| 神经网络层 | ____/10 | 20% | ____ |
| 模型容器 | ____/10 | 15% | ____ |
| 表达式DSL | ____/10 | 10% | ____ |
| GPU后端 | ____/10 | 10% | ____ |
| 构建工具 | ____/10 | 5% | ____ |
| **总分** | | | **____/5** |

---

## 🎯 理解等级

### 5.0-4.5：专家级
- 理解所有核心概念
- 能独立修改和扩展功能
- 能指导他人学习
- **建议**：可以开始重构或设计新功能

### 4.4-4.0：高级
- 理解大部分核心概念
- 能修改现有功能
- 需要帮助处理复杂问题
- **建议**：专注于特定模块的深入学习

### 3.9-3.5：中级
- 理解基本概念
- 能使用现有功能
- 需要指导进行修改
- **建议**：先完成"理解路线图"的前4步

### 3.4-3.0：初级
- 理解部分概念
- 能运行示例代码
- 需要详细指导
- **建议**：从"快速理解"部分开始

### 2.9-0：入门
- 理解很少概念
- 需要从基础学起
- **建议**：先学习C++和深度学习基础

---

## 📚 学习资源

### 核心文档
1. **AGENTS.md**：AI开发速览（必读）
2. **docs/01-architecture.md**：架构设计（本文档）
3. **docs/08-pitfalls-and-lessons.md**：踩坑警示录（改代码前必读）

### 学习路径
1. **初学者**：AGENTS.md §1-§5 → docs/03-quickstart-model.md
2. **中级**：docs/05-algorithm-reference.md → docs/09-operator-fusion.md
3. **高级**：docs/11-ir-optimization.md → docs/14-operator-fusion-2.md

### 实践建议
1. **从简单开始**：先理解 MNIST MLP 模型
2. **逐步增加复杂度**：MNIST Transformer → GPT
3. **动手实验**：修改超参数，观察训练效果
4. **阅读测试**：测试代码是最好的文档

---

## 🔄 定期回顾

### 每周回顾
- [ ] 重新评估得分
- [ ] 识别新出现的盲区
- [ ] 调整学习计划

### 每月回顾
- [ ] 总结学习成果
- [ ] 更新理解等级
- [ ] 设定下月目标

### 项目里程碑回顾
- [ ] 完成新功能后重新评估
- [ ] 修复bug后重新评估
- [ ] 重构后重新评估

---

## 📝 个人笔记

### 我的优势领域
-


### 我的薄弱领域


### 我的学习目标


### 我的时间计划


---

*最后更新：2026-08-31*
*维护者：Ethan*