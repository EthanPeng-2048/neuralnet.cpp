// Test to verify const Matrix span issue
#include "algebra/matrix.hpp"
#include "algebra/span.hpp"
#include "algebra/expr.hpp"
#include "algebra/compute_dispatch.hpp"

using namespace nn;

int main() {
    Matrix m(4, 4, 1.0);
    const Matrix& cm = m;

    // This should fail to compile if my hypothesis is correct
    Span s = cm.span();  // cm is const, returns std::span<const Scalar>
    (void)s;
    return 0;
}
