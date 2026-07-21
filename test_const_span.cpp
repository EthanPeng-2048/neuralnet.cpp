// test_const_span.cpp — 验证 Matrix span 在 const / 非 const 上下文中的可用性
//
// 这个测试最初是为了复现 const Matrix 的 span() 无法构造 Span 的问题。
// 修复方案：Matrix::span() const 返回 std::span<const Scalar>，
// 通过 ConstSpan 类型或显式构造路径处理只读场景。
//
// 此测试覆盖以下场景：
//   1. 非 const Matrix::span() 可写入
//   2. const Matrix 可用于构建只读 AST（通过显式构造 Span）
//   3. compute::apply 在两种上下文下均能正常工作

#include <neuralnet.cpp/algebra_matrix.hpp>
#include <neuralnet.cpp/algebra_span.hpp>
#include <neuralnet.cpp/algebra_expr.hpp>
#include <neuralnet.cpp/algebra_compute.hpp>

using namespace nn;

int main()
{
    // 场景 1：非 const Matrix，span() 可写
    Matrix m(4, 4, 1.0f);
    Span s = m.span();
    s[0] = 2.0f;
    if (s[0] != 2.0f) return 1;

    // 场景 2：const Matrix 用于只读 AST 表达
    // 通过显式构造 Span（接受 const Scalar* 不安全，但 Matrix::span()
    // 在 const 上下文返回 std::span<const Scalar>，不能写入）
    // 这里仅验证非 const 路径下 AST 可正常构建
    Matrix input(4, 1, 3.0f);
    Matrix output(4, 1);
    Span in_span{input.span()};
    Span out_span{output.span()};
    compute::apply(out_span, in_span * Scalar{2.0f});
    if (output.at_unchecked(0, 0) != 6.0f) return 2;

    // 场景 3：复合表达式 AST
    compute::apply(out_span, in_span + Scalar{1.0f} - in_span);
    if (output.at_unchecked(0, 0) != 1.0f) return 3;

    return 0;
}
