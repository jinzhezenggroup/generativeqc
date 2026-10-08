// Generated from tensor.scf through its retained weighted-Gram lowering.
#pragma once
#include <cmath>
namespace generativeqc::tensor::weighted_gram::generated {
struct ColumnScaleDgemmLp64 {};
using CpuExecution = ColumnScaleDgemmLp64;
inline bool energy_weight(double tensor_input_0, double tensor_input_1, double& tensor_output_0) noexcept {
  if (!std::isfinite(tensor_input_0) || !std::isfinite(tensor_input_1)) return false;
  const double v0 = tensor_input_1;
  const double v1 = tensor_input_0;
  const double v2 = v1 * v0;
  if (!std::isfinite(v2)) return false;
  if (!std::isfinite(v2)) return false;
  tensor_output_0 = v2;
  return true;
}
inline bool weighted_coefficient(double tensor_input_0, double tensor_input_1, double& tensor_output_0) noexcept {
  if (!std::isfinite(tensor_input_0) || !std::isfinite(tensor_input_1)) return false;
  const double v0 = tensor_input_1;
  const double v1 = tensor_input_0;
  const double v2 = v1 * v0;
  if (!std::isfinite(v2)) return false;
  if (!std::isfinite(v2)) return false;
  tensor_output_0 = v2;
  return true;
}
inline bool density_contribution(double tensor_input_0, double tensor_input_1, double& tensor_output_0) noexcept {
  if (!std::isfinite(tensor_input_0) || !std::isfinite(tensor_input_1)) return false;
  const double v0 = tensor_input_0;
  const double v1 = tensor_input_1;
  const double v2 = v0 * v1;
  if (!std::isfinite(v2)) return false;
  if (!std::isfinite(v2)) return false;
  tensor_output_0 = v2;
  return true;
}
inline bool density_update(double tensor_input_0, double tensor_input_1, double tensor_input_2, double& tensor_output_0) noexcept {
  if (!std::isfinite(tensor_input_0) || !std::isfinite(tensor_input_1) || !std::isfinite(tensor_input_2)) return false;
  const double v0 = tensor_input_2;
  const double v1 = tensor_input_0;
  const double v2 = tensor_input_1;
  double v4 = v0;
  v4 = std::fma(v1, v2, v4);
  if (!std::isfinite(v4)) return false;
  if (!std::isfinite(v4)) return false;
  if (!std::isfinite(v4)) return false;
  tensor_output_0 = v4;
  return true;
}
}  // namespace generativeqc::tensor::weighted_gram::generated
