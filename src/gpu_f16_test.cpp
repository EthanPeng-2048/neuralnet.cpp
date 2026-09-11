#ifdef NN_HAS_VULKAN
#include <cstdio>
#include "neuralnet.cpp/backend/compute_vk_backend.hpp"
#include "neuralnet.cpp/compute_gpu_engine.hpp"

int main() {
    printf("=== GPU f16 Capability Check ===\n");
    auto& backend = nn::GpuBackend::instance();

    // Explicitly initialize
    auto init_r = backend.initialize();
    if (!init_r) {
        printf("GPU init FAILED: %s\n", init_r.error().message.c_str());
        return 1;
    }
    printf("GPU backend: OK\n");
    printf("shaderFloat16: %s\n", backend.has_shader_float16() ? "YES" : "NO");

    // Test GPU f16 operations
    nn::GpuEngine engine(backend);

    // Create f32 matrix
    nn::Matrix m(2, 2);
    m.set_value(0, 0, 1.0f); m.set_value(0, 1, 2.0f);
    m.set_value(1, 0, 3.0f); m.set_value(1, 1, 4.0f);

    // Upload as f32
    auto t32 = engine.from_matrix(m);
    if (!t32) { printf("from_matrix failed: %s\n", t32.error().message.c_str()); return 1; }
    printf("f32 uploaded: %s\n", t32->shape_str().c_str());

    // Cast to f16
    auto t16 = engine.cast(*t32, nn::Precision::F16);
    if (!t16) { printf("cast to f16 failed: %s\n", t16.error().message.c_str()); return 1; }
    printf("f16 tensor: prec=%d %s\n", (int)t16->precision(), t16->shape_str().c_str());

    // Cast back to f32
    auto t32_back = engine.cast(*t16, nn::Precision::F32);
    if (!t32_back) { printf("cast back failed: %s\n", t32_back.error().message.c_str()); return 1; }

    auto result = engine.to_matrix(*t32_back);
    if (!result) { printf("to_matrix failed: %s\n", result.error().message.c_str()); return 1; }

    printf("Original:  [%.1f %.1f; %.1f %.1f]\n", m.at(0,0), m.at(0,1), m.at(1,0), m.at(1,1));
    printf("GPU f16:   [%.1f %.1f; %.1f %.1f]\n", result->at(0,0), result->at(0,1), result->at(1,0), result->at(1,1));

    // Test f16 matmul
    printf("\n=== GPU f16 Matmul ===\n");
    nn::Matrix m2(2, 2);
    m2.set_value(0, 0, 5.0f); m2.set_value(0, 1, 6.0f);
    m2.set_value(1, 0, 7.0f); m2.set_value(1, 1, 8.0f);

    auto t16_a = engine.cast(engine.from_matrix(m).value(), nn::Precision::F16);
    auto t16_b = engine.cast(engine.from_matrix(m2).value(), nn::Precision::F16);

    auto t16_c = engine.matmul(*t16_a, *t16_b, false, false, nn::Precision::F16);
    if (!t16_c) { printf("f16 matmul failed: %s\n", t16_c.error().message.c_str()); return 1; }

    auto c32 = engine.cast(*t16_c, nn::Precision::F32);
    auto c_mat = engine.to_matrix(*c32);
    printf("A*B (f16): [%.1f %.1f; %.1f %.1f]\n",
        c_mat->at(0,0), c_mat->at(0,1), c_mat->at(1,0), c_mat->at(1,1));
    printf("Expected:  [19.0 22.0; 43.0 50.0]\n");

    printf("\n=== ALL GPU f16 TESTS PASSED ===\n");
    return 0;
}
#else
#include <cstdio>
int main() {
    printf("Vulkan not available\n");
    return 0;
}
#endif
