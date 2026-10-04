// ───────────────────────────────────────────────────────────────────────────
//  nn_pch_stub.cpp — PCH 宿主 TU（仅供 CMake 的 `target_precompile_headers
//  REUSE_FROM` 使用，见 CMakeLists.txt 末尾「PCH 落地」块）。
//
//  本文件不做任何事：对象库 `nn_pch` 的唯一存在意义，是把 `nn.hpp` 预编译成
//  **一份** PCH，再由所有编译条件相同的应用/测试目标 `REUSE_FROM nn_pch`
//  共享（逐目标各编一份的话，本机 Debug 全量构建要为此花掉 ~780s CPU）。
//  它不参与任何可执行文件，也不导出任何符号。
//
//  注：这里不能写成空文件——`-Wpedantic -Werror` 下空翻译单元会直接编译失败。
// ───────────────────────────────────────────────────────────────────────────

namespace nn::detail::pch_stub
{

[[maybe_unused]] inline void anchor() noexcept {}

} // namespace nn::detail::pch_stub
