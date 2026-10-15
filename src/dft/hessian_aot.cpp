#include "dft/hessian_aot.hpp"

#include <algorithm>
#include <limits>

namespace generativeqc::dft::hessian_aot {
namespace {

bool panel_fits(std::size_t rows, std::size_t columns) noexcept {
  return !rows || columns <= std::numeric_limits<std::size_t>::max() / sizeof(double) / rows;
}

}  // namespace

const Program* find(std::string_view name) noexcept {
  const auto inventory = programs();
  const auto found = std::lower_bound(
      inventory.begin(), inventory.end(), name,
      [](const Program& program, std::string_view key) { return program.name < key; });
  return found != inventory.end() && found->name == name ? &*found : nullptr;
}

cudaError_t launch_array(std::string_view name, const double* inputs, std::size_t lanes,
                         double* outputs, int* status, cudaStream_t stream) noexcept {
  const auto* program = find(name);
  if (!program || program->kind != Kind::array || !program->array) return cudaErrorNotSupported;
  if (!panel_fits(program->inputs.size(), lanes) || !panel_fits(program->outputs, lanes))
    return cudaErrorInvalidValue;
  if (!lanes) return cudaSuccess;
  if (!inputs || !outputs || !status) return cudaErrorInvalidValue;
  // CUDA grid.x is bounded even on devices accepting very large FP64 panels.
  if (lanes > static_cast<std::size_t>(std::numeric_limits<int>::max()) * 128)
    return cudaErrorInvalidValue;
  return static_cast<cudaError_t>(program->array(inputs, lanes, outputs, status, stream));
}

cudaError_t launch_ao(const double* basis, std::size_t atoms, std::size_t primitives,
                      std::size_t aos, const double* points, std::size_t point_count,
                      std::size_t jets, double* outputs, int* status,
                      cudaStream_t stream) noexcept {
  const auto* program = find("ao");
  if (!program || program->kind != Kind::ao || !program->ao) return cudaErrorNotSupported;
  if ((jets != 1 && jets != 4 && jets != 10 && jets != 20) || !atoms || !primitives || !aos ||
      !panel_fits(aos, point_count) || !panel_fits(aos * point_count, jets) ||
      !panel_fits(atoms, 3) || !panel_fits(primitives, 2) || !panel_fits(aos, 16))
    return cudaErrorInvalidValue;
  constexpr auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(double);
  if (3 * atoms > maximum - 2 * primitives || 16 * aos > maximum - 3 * atoms - 2 * primitives)
    return cudaErrorInvalidValue;
  if (!point_count) return cudaSuccess;
  if (jets * aos * point_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) * 128)
    return cudaErrorInvalidValue;
  if (!basis || !points || !outputs || !status) return cudaErrorInvalidValue;
  return static_cast<cudaError_t>(program->ao(basis, atoms, primitives, aos, points, point_count,
                                              jets, outputs, status, stream));
}

cudaError_t launch_integral(std::string_view name, const double* records, std::size_t count,
                            const long long* atoms, const double* direction, double* outputs,
                            int* status, cudaStream_t stream) noexcept {
  const auto* program = find(name);
  if (!program || program->kind != Kind::integral || !program->integral)
    return cudaErrorNotSupported;
  if (!count || !panel_fits(program->record_stride, count) || !records || !atoms || !direction ||
      !outputs || !status)
    return cudaErrorInvalidValue;
  return static_cast<cudaError_t>(
      program->integral(records, count, atoms, direction, outputs, status, stream));
}

}  // namespace generativeqc::dft::hessian_aot
