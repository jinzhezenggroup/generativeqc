#include "solver/cpu/prepared_spectral.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

#include "runtime/bounded_workspace.hpp"
#include "solver/cpu/generalized_eigen.hpp"

namespace generativeqc::solver::cpu {
namespace provider = ::generativeqc::tensor::cpu;

struct SpectralPlanData final {
  std::int64_t batch_size = 0;
  std::int64_t total_matrix_elements = 0;
  std::int64_t maximum_order = 0;
  double minimum_overlap_rcond = 0;
  SpectralStatusEncoding status_encoding;
  std::vector<std::int64_t> orbital_offsets;
  std::vector<std::int64_t> matrix_offsets;
  PreparedSymmetricEigen symmetric_eigen;
  std::size_t cache_size_bytes = 0;
  std::size_t factor_offset_bytes = 0;
  std::size_t generation_offset_bytes = 0;
  std::size_t status_offset_bytes = 0;
};

namespace {
using ::generativeqc::runtime::checked_add;
using ::generativeqc::runtime::checked_multiply;
struct Range {
  std::uintptr_t begin = 0;
  std::uintptr_t end = 0;
};
bool range(const void* pointer, std::size_t count, std::size_t element_size, std::size_t alignment,
           Range& output) noexcept {
  std::size_t bytes = 0;
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (!pointer || !count || begin % alignment || !checked_multiply(count, element_size, bytes) ||
      begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    return false;
  output = {begin, begin + bytes};
  return true;
}
template <class T>
bool span(const T* pointer, std::size_t count, Range& output) noexcept {
  return range(pointer, count, sizeof(T), alignof(T), output);
}
bool overlap(const Range& a, const Range& b) noexcept { return a.begin < b.end && b.begin < a.end; }
template <std::size_t N>
bool disjoint(const std::array<Range, N>& ranges) noexcept {
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = i + 1; j < N; ++j)
      if (overlap(ranges[i], ranges[j])) return false;
  return true;
}
template <std::size_t N, std::size_t C>
bool controls_disjoint(const PreparedSpectralPlan& plan, const std::array<Range, N>& ranges,
                       const std::array<Range, C>& controls) noexcept {
  for (const auto& r : ranges) {
    if (plan.overlaps_storage(reinterpret_cast<const void*>(r.begin), r.end - r.begin))
      return false;
    for (const auto& c : controls)
      if (overlap(r, c)) return false;
  }
  return true;
}
template <class T>
T* at(void* base, std::size_t offset) noexcept {
  return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(base) + offset);
}
const std::vector<std::int64_t>& empty_offsets() noexcept {
  static const std::vector<std::int64_t> empty;
  return empty;
}
bool valid_cache(const PreparedSpectralPlan& plan, const SpectralOverlapCache& cache,
                 Range& memory) noexcept {
  if (!plan.sealed() || cache.plan_identity != plan.identity() ||
      cache.workspace_size_bytes < plan.cache_size_bytes() ||
      !range(cache.workspace_base, plan.cache_size_bytes(), 1, kSpectralCacheAlignment, memory))
    return false;
  return cache.factors == at<double>(cache.workspace_base, plan.factor_offset_bytes()) &&
         cache.generations ==
             at<std::uint64_t>(cache.workspace_base, plan.generation_offset_bytes()) &&
         cache.statuses == at<std::int32_t>(cache.workspace_base, plan.status_offset_bytes());
}
bool symmetric_finite_row_major(const double* matrix, std::size_t n) {
  constexpr double multiplier = 64.0 * std::numeric_limits<double>::epsilon();
  for (std::size_t row = 0u; row < n; ++row) {
    for (std::size_t column = 0u; column < n; ++column) {
      const double value = matrix[row * n + column];
      if (!std::isfinite(value)) {
        return false;
      }
      if (column < row) {
        const double transpose = matrix[column * n + row];
        const double scale = std::max({1.0, std::abs(value), std::abs(transpose)});
        if (std::abs(value - transpose) > multiplier * scale) {
          return false;
        }
      }
    }
  }
  return true;
}

void copy_symmetric_row_to_column(const double* input, std::size_t n, double* output) {
  for (std::size_t row = 0u; row < n; ++row) {
    output[row + row * n] = input[row * n + row];
    for (std::size_t column = 0u; column < row; ++column) {
      const double value = 0.5 * (input[row * n + column] + input[column * n + row]);
      output[row + column * n] = value;
      output[column + row * n] = value;
    }
  }
}

double matrix_one_norm_column_major(const double* matrix, std::size_t n) {
  double maximum = 0.0;
  for (std::size_t column = 0u; column < n; ++column) {
    double sum = 0.0;
    for (std::size_t row = 0u; row < n; ++row) {
      sum += std::abs(matrix[row + column * n]);
    }
    maximum = std::max(maximum, sum);
  }
  return maximum;
}

bool finite_array(const double* values, std::size_t count) {
  for (std::size_t index = 0u; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool PreparedSpectralPlan::sealed() const noexcept { return data_ != nullptr; }
std::int64_t PreparedSpectralPlan::batch_size() const noexcept {
  return data_ ? data_->batch_size : 0;
}
std::int64_t PreparedSpectralPlan::total_matrix_elements() const noexcept {
  return data_ ? data_->total_matrix_elements : 0;
}
std::int64_t PreparedSpectralPlan::maximum_order() const noexcept {
  return data_ ? data_->maximum_order : 0;
}
double PreparedSpectralPlan::minimum_overlap_rcond() const noexcept {
  return data_ ? data_->minimum_overlap_rcond : 0;
}
std::size_t PreparedSpectralPlan::cache_size_bytes() const noexcept {
  return data_ ? data_->cache_size_bytes : 0;
}
std::size_t PreparedSpectralPlan::factor_offset_bytes() const noexcept {
  return data_ ? data_->factor_offset_bytes : 0;
}
std::size_t PreparedSpectralPlan::generation_offset_bytes() const noexcept {
  return data_ ? data_->generation_offset_bytes : 0;
}
std::size_t PreparedSpectralPlan::status_offset_bytes() const noexcept {
  return data_ ? data_->status_offset_bytes : 0;
}
std::size_t PreparedSpectralPlan::resident_bytes() const noexcept {
  return data_ ? sizeof(SpectralPlanData) +
                     sizeof(std::int64_t) *
                         (data_->orbital_offsets.capacity() + data_->matrix_offsets.capacity())
               : 0;
}
const std::vector<std::int64_t>& PreparedSpectralPlan::orbital_offsets() const noexcept {
  return data_ ? data_->orbital_offsets : empty_offsets();
}
const std::vector<std::int64_t>& PreparedSpectralPlan::matrix_offsets() const noexcept {
  return data_ ? data_->matrix_offsets : empty_offsets();
}
const PreparedSymmetricEigen& PreparedSpectralPlan::symmetric_eigen() const noexcept {
  static const PreparedSymmetricEigen empty;
  return data_ ? data_->symmetric_eigen : empty;
}
SpectralStatusEncoding PreparedSpectralPlan::status_encoding() const noexcept {
  return data_ ? data_->status_encoding : SpectralStatusEncoding{};
}
const SpectralPlanData* PreparedSpectralPlan::identity() const noexcept { return data_.get(); }
bool PreparedSpectralPlan::overlaps_storage(const void* pointer, std::size_t bytes) const noexcept {
  if (!bytes) return false;
  Range candidate;
  if (!range(pointer, bytes, 1, 1, candidate)) return true;
  if (!data_) return false;
  std::array<Range, 3> stored;
  if (!span(data_.get(), 1, stored[0]) ||
      !span(data_->orbital_offsets.data(), data_->orbital_offsets.capacity(), stored[1]) ||
      !span(data_->matrix_offsets.data(), data_->matrix_offsets.capacity(), stored[2]))
    return true;
  for (const auto& r : stored)
    if (overlap(candidate, r)) return true;
  return false;
}

SpectralResult prepare_spectral_plan(const std::int64_t* offsets, std::size_t offset_count,
                                     double minimum_rcond, SpectralStatusEncoding encoding,
                                     PreparedSpectralPlan& plan, std::string& error) {
  Range input;
  if (offset_count < 2 ||
      offset_count - 1 > std::size_t(std::numeric_limits<std::int64_t>::max()) ||
      !span(offsets, offset_count, input) || !std::isfinite(minimum_rcond) || minimum_rcond <= 0 ||
      minimum_rcond >= 1 || encoding.success == encoding.numerical_failure) {
    error = "invalid prepared spectral dimensions, overlap threshold, or status encoding";
    return SpectralResult::invalid_argument;
  }
  if (offsets[0] != 0) {
    error = "spectral orbital offsets must begin at zero";
    return SpectralResult::invalid_argument;
  }
  // Admit every difference before allocating metadata or subtracting signed offsets.
  std::int64_t total = 0, maximum = 0;
  for (std::size_t i = 0; i + 1 < offset_count; ++i) {
    if (offsets[i] < 0 || offsets[i + 1] <= offsets[i]) {
      error = "spectral orbital offsets must be strictly increasing";
      return SpectralResult::invalid_argument;
    }
    const auto n = offsets[i + 1] - offsets[i];
    if (n > std::numeric_limits<LapackInt>::max() ||
        n > std::numeric_limits<std::int64_t>::max() / n ||
        total > std::numeric_limits<std::int64_t>::max() - n * n) {
      error = "spectral dimensions exceed LP64 or index limits";
      return SpectralResult::invalid_argument;
    }
    total += n * n;
    maximum = std::max(maximum, n);
  }
  PreparedSymmetricEigen symmetric;
  if (!prepare_borrowed_symmetric_eigen(static_cast<std::size_t>(maximum), symmetric)) {
    error = "LAPACK eigensolver work dimensions exceed LP64 integer limits";
    return SpectralResult::invalid_argument;
  }
  try {
    auto candidate = std::make_shared<SpectralPlanData>();
    candidate->batch_size = static_cast<std::int64_t>(offset_count - 1);
    candidate->total_matrix_elements = total;
    candidate->maximum_order = maximum;
    candidate->minimum_overlap_rcond = minimum_rcond;
    candidate->status_encoding = encoding;
    candidate->symmetric_eigen = symmetric;
    candidate->orbital_offsets.assign(offsets, offsets + offset_count);
    candidate->matrix_offsets.resize(offset_count, 0);
    for (std::size_t i = 0; i + 1 < offset_count; ++i) {
      const auto n = offsets[i + 1] - offsets[i];
      candidate->matrix_offsets[i + 1] = candidate->matrix_offsets[i] + n * n;
    }
    std::size_t factor_bytes = 0, generation_bytes = 0, status_bytes = 0;
    ::generativeqc::runtime::WorkspaceLayout layout;
    if (!checked_multiply(static_cast<std::size_t>(total), sizeof(double), factor_bytes) ||
        !checked_multiply(offset_count - 1, sizeof(std::uint64_t), generation_bytes) ||
        !checked_multiply(offset_count - 1, sizeof(std::int32_t), status_bytes) ||
        !layout.append_bytes(factor_bytes, kSpectralCacheAlignment,
                             candidate->factor_offset_bytes) ||
        !layout.append_bytes(generation_bytes, kSpectralCacheAlignment,
                             candidate->generation_offset_bytes) ||
        !layout.append_bytes(status_bytes, kSpectralCacheAlignment,
                             candidate->status_offset_bytes) ||
        !::generativeqc::runtime::checked_align_up(layout.bytes(), kSpectralCacheAlignment,
                                                   candidate->cache_size_bytes)) {
      error = "eigensolver overlap cache packing overflows size_t";
      return SpectralResult::invalid_argument;
    }
    plan.data_ = std::move(candidate);
    error.clear();
    return SpectralResult::success;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate CPU eigensolver plan metadata";
    return SpectralResult::allocation_failure;
  } catch (const std::length_error&) {
    error = "CPU eigensolver plan metadata exceeds host container limits";
    return SpectralResult::allocation_failure;
  }
}

SpectralResult bind_spectral_overlap_cache(const PreparedSpectralPlan& plan, void* storage,
                                           std::size_t bytes, SpectralOverlapCache& cache,
                                           std::string& error) {
  std::array<Range, 1> active;
  std::array<Range, 3> controls;
  if (!plan.sealed() || bytes < plan.cache_size_bytes() ||
      !range(storage, plan.cache_size_bytes(), 1, kSpectralCacheAlignment, active[0]) ||
      !span(&plan, 1, controls[0]) || !span(&cache, 1, controls[1]) ||
      !span(&error, 1, controls[2]) || !controls_disjoint(plan, active, controls)) {
    error = "eigensolver overlap cache workspace is invalid or overlaps control storage";
    return SpectralResult::invalid_argument;
  }
  SpectralOverlapCache candidate{storage,
                                 bytes,
                                 at<double>(storage, plan.factor_offset_bytes()),
                                 at<std::uint64_t>(storage, plan.generation_offset_bytes()),
                                 at<std::int32_t>(storage, plan.status_offset_bytes()),
                                 plan.identity()};
  std::fill_n(candidate.factors, static_cast<std::size_t>(plan.total_matrix_elements()), 0.0);
  std::fill_n(candidate.generations, static_cast<std::size_t>(plan.batch_size()), 0u);
  std::fill_n(candidate.statuses, static_cast<std::size_t>(plan.batch_size()),
              plan.status_encoding().numerical_failure);
  cache = candidate;
  error.clear();
  return SpectralResult::success;
}

SpectralResult validate_spectral_overlap_cache(const PreparedSpectralPlan& plan,
                                               const SpectralOverlapCache& cache,
                                               std::string& error) {
  std::array<Range, 1> active;
  std::array<Range, 3> controls;
  if (!valid_cache(plan, cache, active[0]) || !span(&plan, 1, controls[0]) ||
      !span(&cache, 1, controls[1]) || !span(&error, 1, controls[2]) ||
      !controls_disjoint(plan, active, controls)) {
    error = "eigensolver overlap cache is not a canonical binding for this plan";
    return SpectralResult::invalid_argument;
  }
  return SpectralResult::success;
}

const double* AdmittedSpectralMatrices::matrix(const PreparedSpectralPlan& plan,
                                               std::int64_t system,
                                               std::int32_t spectrum) const noexcept {
  if (!plan.sealed() || plan_identity_ != plan.identity() || system < first_system_ ||
      static_cast<std::uint64_t>(system - first_system_) >= system_count_ || spectrum < 0)
    return nullptr;
  const auto index = static_cast<std::size_t>(system - first_system_);
  if (spectrum >= (counts_ ? counts_[index] : 1)) return nullptr;
  const auto n = plan.orbital_offsets()[system + 1] - plan.orbital_offsets()[system];
  return values_ + (offsets_[index] - offsets_[0]) + static_cast<std::int64_t>(spectrum) * n * n;
}
bool AdmittedSpectralMatrices::is_overlap_batch(const PreparedSpectralPlan& plan) const noexcept {
  return plan.sealed() && plan_identity_ == plan.identity() && first_system_ == 0 &&
         system_count_ == static_cast<std::size_t>(plan.batch_size()) && counts_ == nullptr &&
         offsets_ == plan.matrix_offsets().data();
}
bool AdmittedSpectralMatrices::overlaps_borrowed_storage(const void* pointer,
                                                         std::size_t bytes) const noexcept {
  Range candidate;
  if (!range(pointer, bytes, 1, 1, candidate)) return true;
  std::array<Range, 4> borrowed{};
  if (!span(this, 1, borrowed[0]) || !span(values_, value_count_, borrowed[1]) ||
      !span(offsets_, system_count_ + 1, borrowed[2]) ||
      (counts_ && !span(counts_, system_count_, borrowed[3])))
    return true;
  for (const auto& r : borrowed)
    if (overlap(candidate, r)) return true;
  return false;
}

SpectralResult admit_spectral_matrices(const PreparedSpectralPlan& plan, std::int64_t first_system,
                                       std::size_t system_count, const double* values,
                                       const std::int64_t* offsets, const std::int32_t* counts,
                                       AdmittedSpectralMatrices& admitted) {
  if (!plan.sealed() || first_system < 0 || first_system >= plan.batch_size() || !system_count ||
      system_count > static_cast<std::size_t>(plan.batch_size() - first_system) ||
      bool(offsets) != bool(counts))
    return SpectralResult::invalid_argument;
  if (!offsets) offsets = plan.matrix_offsets().data() + first_system;
  Range offset_range, count_range, values_range;
  if (!span(offsets, system_count + 1, offset_range) ||
      (counts && !span(counts, system_count, count_range)) || offsets[0] < 0)
    return SpectralResult::invalid_argument;
  for (std::size_t i = 0; i < system_count; ++i) {
    const auto system = first_system + static_cast<std::int64_t>(i);
    const auto n = plan.orbital_offsets()[system + 1] - plan.orbital_offsets()[system];
    const auto count = counts ? counts[i] : 1;
    if (count <= 0 || offsets[i] < 0 || offsets[i + 1] <= offsets[i] ||
        n * n > std::numeric_limits<std::int64_t>::max() / count ||
        offsets[i + 1] - offsets[i] != n * n * count)
      return SpectralResult::invalid_argument;
  }
  const auto total64 = static_cast<std::uint64_t>(offsets[system_count] - offsets[0]);
  if (total64 > std::numeric_limits<std::size_t>::max()) return SpectralResult::invalid_argument;
  const auto total = static_cast<std::size_t>(total64);
  Range output_range, plan_range;
  if (!span(values, total, values_range) || !span(&admitted, 1, output_range) ||
      !span(&plan, 1, plan_range) || overlap(output_range, values_range) ||
      overlap(output_range, offset_range) || (counts && overlap(output_range, count_range)) ||
      overlap(output_range, plan_range) || plan.overlaps_storage(&admitted, sizeof(admitted)))
    return SpectralResult::invalid_argument;
  for (std::size_t i = 0; i < system_count; ++i) {
    const auto system = first_system + static_cast<std::int64_t>(i);
    const auto n = static_cast<std::size_t>(plan.orbital_offsets()[system + 1] -
                                            plan.orbital_offsets()[system]);
    const auto count = counts ? counts[i] : 1;
    for (std::int32_t spectrum = 0; spectrum < count; ++spectrum)
      if (!symmetric_finite_row_major(
              values + (offsets[i] - offsets[0]) + std::size_t(spectrum) * n * n, n))
        return SpectralResult::invalid_argument;
  }
  AdmittedSpectralMatrices candidate;
  candidate.plan_identity_ = plan.identity();
  candidate.values_ = values;
  candidate.offsets_ = offsets;
  candidate.counts_ = counts;
  candidate.first_system_ = first_system;
  candidate.system_count_ = system_count;
  candidate.value_count_ = total;
  admitted = candidate;
  return SpectralResult::success;
}

namespace {
SpectralResult validate_factor_bindings(const PreparedSpectralPlan& plan,
                                        const double* overlap_input, std::uint64_t generation,
                                        const provider::CpuLinearAlgebraBackend& backend,
                                        const SpectralFactorWorkspace& work,
                                        const SpectralOverlapCache& cache, std::string& error,
                                        std::array<Range, 7>& active) {
  if (!plan.sealed()) {
    error = "eigensolver plan is default-constructed or moved-from";
    return SpectralResult::invalid_argument;
  }
  if (!backend.ready()) {
    error = "CPU eigensolver requires a verified LP64 backend";
    return SpectralResult::backend_unavailable;
  }
  const auto batch = static_cast<std::size_t>(plan.batch_size());
  const auto total = static_cast<std::size_t>(plan.total_matrix_elements());
  const auto required = plan.symmetric_eigen().required();
  std::array<Range, 5> controls;
  if (!generation || work.factor_capacity < total || work.generation_capacity < batch ||
      work.status_capacity < batch || work.lapack_work_capacity < std::size_t(required.doubles) ||
      work.lapack_integer_work_capacity < std::size_t(required.integers) ||
      !span(overlap_input, total, active[0]) || !valid_cache(plan, cache, active[1]) ||
      !span(work.factors, total, active[2]) || !span(work.generations, batch, active[3]) ||
      !span(work.statuses, batch, active[4]) ||
      !span(work.lapack_work, std::size_t(required.doubles), active[5]) ||
      !span(work.lapack_integer_work, std::size_t(required.integers), active[6]) ||
      !span(&plan, 1, controls[0]) || !span(&backend, 1, controls[1]) ||
      !span(&work, 1, controls[2]) || !span(&cache, 1, controls[3]) ||
      !span(&error, 1, controls[4]) || !disjoint(active) ||
      !controls_disjoint(plan, active, controls)) {
    error = "overlap input, cache, scratch, plan, and descriptors must not overlap";
    return SpectralResult::invalid_argument;
  }
  return SpectralResult::success;
}
SpectralResult validate_spectrum_bindings(const PreparedSpectralPlan& plan, std::int64_t system,
                                          const SpectralOverlapCache& cache,
                                          std::uint64_t generation, const double* hamiltonian,
                                          const provider::CpuLinearAlgebraBackend& backend,
                                          const SpectralWorkspace& work,
                                          std::array<Range, 6>& active) {
  if (!plan.sealed() || system < 0 || system >= plan.batch_size() || !generation)
    return SpectralResult::invalid_argument;
  if (!backend.ready()) return SpectralResult::backend_unavailable;
  const auto index = static_cast<std::size_t>(system);
  const auto n =
      static_cast<LapackInt>(plan.orbital_offsets()[index + 1] - plan.orbital_offsets()[index]);
  const auto dimension = static_cast<std::size_t>(n);
  const auto matrix_count = dimension * dimension;
  const auto required = plan.symmetric_eigen().required();
  std::array<Range, 4> controls;
  if (work.coefficient_capacity < matrix_count || work.eigenvalue_capacity < dimension ||
      work.lapack_work_capacity < std::size_t(required.doubles) ||
      work.lapack_integer_work_capacity < std::size_t(required.integers) ||
      !span(hamiltonian, matrix_count, active[0]) || !valid_cache(plan, cache, active[1]) ||
      !span(work.coefficients, matrix_count, active[2]) ||
      !span(work.eigenvalues, dimension, active[3]) ||
      !span(work.lapack_work, std::size_t(required.doubles), active[4]) ||
      !span(work.lapack_integer_work, std::size_t(required.integers), active[5]) ||
      !span(&plan, 1, controls[0]) || !span(&backend, 1, controls[1]) ||
      !span(&work, 1, controls[2]) || !span(&cache, 1, controls[3]) || !disjoint(active) ||
      !controls_disjoint(plan, active, controls))
    return SpectralResult::invalid_argument;
  if (cache.generations[index] != generation ||
      cache.statuses[index] != plan.status_encoding().success)
    return SpectralResult::numerical_failure;
  return SpectralResult::success;
}
}  // namespace

SpectralResult factor_spectral_overlaps(const PreparedSpectralPlan& plan,
                                        const double* overlap_input, std::uint64_t generation,
                                        const provider::CpuLinearAlgebraBackend& backend,
                                        const SpectralFactorWorkspace& work,
                                        const SpectralOverlapCache& cache, std::string& error) {
  std::array<Range, 7> active;
  const auto validation = validate_factor_bindings(plan, overlap_input, generation, backend, work,
                                                   cache, error, active);
  if (validation != SpectralResult::success) return validation;
  AdmittedSpectralMatrices admitted;
  const auto result = admit_spectral_matrices(plan, 0, static_cast<std::size_t>(plan.batch_size()),
                                              overlap_input, nullptr, nullptr, admitted);
  if (result != SpectralResult::success) {
    error = "overlap matrices must be finite and symmetric";
    return result;
  }
  return factor_admitted_spectral_overlaps(plan, admitted, generation, backend, work, cache, error);
}

SpectralResult factor_admitted_spectral_overlaps(
    const PreparedSpectralPlan& plan, const AdmittedSpectralMatrices& admitted,
    std::uint64_t generation, const provider::CpuLinearAlgebraBackend& backend,
    const SpectralFactorWorkspace& work, const SpectralOverlapCache& cache, std::string& error) {
  const double* overlap_input =
      admitted.is_overlap_batch(plan) ? admitted.matrix(plan, 0, 0) : nullptr;
  std::array<Range, 7> active;
  const auto validation = validate_factor_bindings(plan, overlap_input, generation, backend, work,
                                                   cache, error, active);
  if (validation != SpectralResult::success) return validation;
  const auto batch = static_cast<std::size_t>(plan.batch_size());
  for (std::size_t i = 1; i < active.size(); ++i) {
    if (admitted.overlaps_borrowed_storage(reinterpret_cast<const void*>(active[i].begin),
                                           active[i].end - active[i].begin)) {
      error = "overlap scratch and cache must not overlap admitted input metadata";
      return SpectralResult::invalid_argument;
    }
  }
  for (std::size_t system = 0; system < batch; ++system) {
    const auto n =
        static_cast<LapackInt>(plan.orbital_offsets()[system + 1] - plan.orbital_offsets()[system]);
    const auto dimension = static_cast<std::size_t>(n);
    const auto offset = static_cast<std::size_t>(plan.matrix_offsets()[system]);
    double* candidate = work.factors + offset;
    copy_symmetric_row_to_column(overlap_input + offset, dimension, candidate);
    const double one_norm = matrix_one_norm_column_major(candidate, dimension);
    LapackInt info = provider::cholesky_lower(backend, n, candidate);
    double reciprocal_condition = 0.0;
    if (info == 0) {
      info = provider::reciprocal_condition_lower(backend, n, candidate, one_norm,
                                                  &reciprocal_condition, work.lapack_work,
                                                  work.lapack_integer_work);
    }
    if (info < 0) {
      error = "LP64 LAPACK rejected an internal overlap-factorization argument";
      return SpectralResult::backend_failure;
    }
    const bool usable = info == 0 && std::isfinite(reciprocal_condition) &&
                        reciprocal_condition >= plan.minimum_overlap_rcond();
    if (usable) {
      for (std::size_t column = 0; column < dimension; ++column)
        for (std::size_t row = 0; row < column; ++row) candidate[row + column * dimension] = 0.0;
    }
    work.generations[system] = generation;
    work.statuses[system] =
        usable ? plan.status_encoding().success : plan.status_encoding().numerical_failure;
  }
  for (std::size_t system = 0; system < batch; ++system) {
    cache.generations[system] = work.generations[system];
    cache.statuses[system] = work.statuses[system];
    if (work.statuses[system] == plan.status_encoding().success) {
      const auto begin = static_cast<std::size_t>(plan.matrix_offsets()[system]);
      const auto end = static_cast<std::size_t>(plan.matrix_offsets()[system + 1]);
      std::copy_n(work.factors + begin, end - begin, cache.factors + begin);
    }
  }
  error.clear();
  return SpectralResult::success;
}

SpectralResult solve_prepared_spectrum(const PreparedSpectralPlan& plan, std::int64_t system,
                                       const SpectralOverlapCache& cache, std::uint64_t generation,
                                       const double* hamiltonian,
                                       const provider::CpuLinearAlgebraBackend& backend,
                                       const SpectralWorkspace& work) {
  std::array<Range, 6> active;
  const auto validation = validate_spectrum_bindings(plan, system, cache, generation, hamiltonian,
                                                     backend, work, active);
  if (validation != SpectralResult::success) return validation;
  AdmittedSpectralMatrices admitted;
  const auto result =
      admit_spectral_matrices(plan, system, 1, hamiltonian, nullptr, nullptr, admitted);
  if (result != SpectralResult::success) return result;
  return solve_admitted_spectrum(plan, system, 0, admitted, cache, generation, backend, work);
}

SpectralResult solve_admitted_spectrum(const PreparedSpectralPlan& plan, std::int64_t system,
                                       std::int32_t spectrum,
                                       const AdmittedSpectralMatrices& admitted,
                                       const SpectralOverlapCache& cache, std::uint64_t generation,
                                       const provider::CpuLinearAlgebraBackend& backend,
                                       const SpectralWorkspace& work) {
  const double* hamiltonian = admitted.matrix(plan, system, spectrum);
  std::array<Range, 6> active;
  const auto validation = validate_spectrum_bindings(plan, system, cache, generation, hamiltonian,
                                                     backend, work, active);
  if (validation != SpectralResult::success) return validation;
  const auto index = static_cast<std::size_t>(system);
  const auto dimension =
      static_cast<std::size_t>(plan.orbital_offsets()[index + 1] - plan.orbital_offsets()[index]);
  const auto matrix_count = dimension * dimension;
  const auto required = plan.symmetric_eigen().required();
  for (std::size_t i = 2; i < active.size(); ++i)
    if (admitted.overlaps_borrowed_storage(reinterpret_cast<const void*>(active[i].begin),
                                           active[i].end - active[i].begin))
      return SpectralResult::invalid_argument;
  const double* factor = cache.factors + plan.matrix_offsets()[index];
  double* coefficients = work.coefficients;
  double* eigenvalues = work.eigenvalues;
  copy_symmetric_row_to_column(hamiltonian, dimension, coefficients);
  const ::generativeqc::solver::GeneralizedEigenDomain domain{
      dimension, 1, 1, ::generativeqc::solver::GeneralizedEigenLayout::column_major};
  const ::generativeqc::solver::GeneralizedEigenMatrices matrices{
      coefficients, factor,       nullptr, coefficients, coefficients,
      matrix_count, matrix_count, 0,       matrix_count, matrix_count};
  const GeneralizedEigenLowering lowering{domain, matrices, backend};
  constexpr auto basis = ::generativeqc::solver::GeneralizedEigenBasis::lower_cholesky;
  if (::generativeqc::solver::reduce_generalized_eigen(basis, lowering) != 0)
    return SpectralResult::backend_failure;
  for (std::size_t column = 0; column < dimension; ++column) {
    for (std::size_t row = column + 1; row < dimension; ++row) {
      const double average =
          0.5 * (coefficients[row + column * dimension] + coefficients[column + row * dimension]);
      coefficients[row + column * dimension] = average;
      coefficients[column + row * dimension] = average;
    }
  }
  SymmetricEigenWorkBinding binding;
  if (!bind_symmetric_eigen_work(plan.symmetric_eigen(),
                                 {work.lapack_work, std::size_t(required.doubles),
                                  work.lapack_integer_work, std::size_t(required.integers)},
                                 binding))
    return SpectralResult::backend_failure;
  const auto result = execute_symmetric_eigen(
      plan.symmetric_eigen(), provider::bind_symmetric_eigen(backend),
      {dimension, coefficients, matrix_count, eigenvalues, dimension}, binding);
  if (!result.submitted() || result.info < 0) return SpectralResult::backend_failure;
  if (result.info > 0) return SpectralResult::numerical_failure;
  if (::generativeqc::solver::recover_generalized_eigen(basis, lowering) != 0)
    return SpectralResult::backend_failure;
  return finite_array(coefficients, matrix_count) && finite_array(eigenvalues, dimension)
             ? SpectralResult::success
             : SpectralResult::numerical_failure;
}

}  // namespace generativeqc::solver::cpu
