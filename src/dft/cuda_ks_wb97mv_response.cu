#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_model.hpp"
#include "dft/cuda_ks_wb97mv_response.hpp"
#include "dft/semilocal_family.hpp"
#include "molecule/basis.hpp"
#include "response/wb97mv_fock_response.cuh"
#include "scf/cuda_density_fitting_device.hpp"
#include "scf/cuda_direct_jk_device.hpp"
#include "scf/cuda_fock_execution.hpp"

namespace generativeqc::dft {
bool cuda_ks_wb97mv_model_matches(const CudaKsFinalStateToken& expected,
                                  const CudaKsResidentFockBinding& source) {
  if (!source) return false;
  namespace maps = generativeqc_wb97mv_fock_maps;
  const auto& model = expected.identity.model;
  const auto& strategy = source.primary->strategy();
  const auto& spec = strategy.spec;
  const auto occupation = source.spins == 1 ? 2.0 : 1.0;
  if (!(model.version == 1 &&
        model.scf_domain_version == semilocal_family_domain_version(SemilocalFamily::Wb97mv) &&
        model.functional == semilocal_family_code(SemilocalFamily::Wb97mv) &&
        model.spins == source.spins && model.device == source.device_id &&
        model.semilocal_exchange_scale == 1 && model.semilocal_correlation_scale == 1 &&
        model.nonlocal_correlation &&
        model.nonlocal_correlation->variant == nlc::Vv10Variant::vv10 &&
        model.nonlocal_correlation->b == maps::nonlocal_b &&
        model.nonlocal_correlation->c == maps::nonlocal_c &&
        model.nonlocal_correlation->coefficient == maps::nonlocal_coefficient &&
        model.nonlocal_density_domain == nlc::Vv10DensityDomain::MolecularV1))
    return false;
  if (!(strategy == expected.identity.determinant.model &&
        strategy.backend == scf::FockBackend::Cuda && spec.version == 1 &&
        spec.derivative_order == 0 &&
        spec.spin ==
            (source.spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted) &&
        spec.coulomb.present && spec.coulomb.coefficient == 1 &&
        spec.coulomb.op == scf::FockOperator::FullRange && spec.coulomb.omega == 0 &&
        spec.exchange.present && spec.exchange.op == scf::FockOperator::FullRange &&
        spec.exchange.omega == 0 &&
        spec.exchange.coefficient == -maps::short_exchange / occupation &&
        spec.coulomb.approximation == spec.exchange.approximation &&
        (spec.coulomb.approximation == scf::FockApproximation::Exact ||
         spec.coulomb.approximation == scf::FockApproximation::DensityFitted)))
    return false;
  if (!source.correction || !model.range_correction ||
      *source.correction != *model.range_correction)
    return false;
  const auto& correction = source.correction->spec;
  return source.correction->backend == scf::FockBackend::Cuda && correction.version == 1 &&
         correction.derivative_order == 0 && correction.spin == spec.spin &&
         !correction.coulomb.present && correction.exchange.present &&
         correction.exchange.op == scf::FockOperator::LongRange &&
         correction.exchange.approximation == scf::FockApproximation::Exact &&
         correction.exchange.omega == maps::omega &&
         correction.exchange.coefficient == -maps::range_correction / occupation;
}
namespace {
namespace maps = generativeqc_wb97mv_fock_maps;

void require(bool condition, const char* detail) {
  if (!condition) throw std::invalid_argument(detail);
}
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
std::size_t product(std::size_t first, std::size_t second) {
  require(!second || first <= std::numeric_limits<std::size_t>::max() / second,
          "physical Fock extent overflow");
  return first * second;
}
struct Layout {
  std::size_t bytes{};
  std::size_t add(std::size_t count) {
    const auto increment = product(count, sizeof(double));
    require(increment <= std::numeric_limits<std::size_t>::max() - 255 &&
                bytes <= std::numeric_limits<std::size_t>::max() - increment - 255,
            "physical Fock arena overflow");
    const auto offset = bytes;
    bytes = (bytes + increment + 255) / 256 * 256;
    return offset;
  }
};
bool overlap(const void* first, std::size_t first_bytes, const void* second,
             std::size_t second_bytes) {
  const auto left = reinterpret_cast<std::uintptr_t>(first);
  const auto right = reinterpret_cast<std::uintptr_t>(second);
  if (!first_bytes || !second_bytes) return false;
  return left <= right ? right - left < first_bytes : left - right < second_bytes;
}
__global__ void audit(const double* value, const double* reference, std::size_t nao, unsigned spins,
                      double absolute, double relative, int* status) {
  const auto matrix = nao * nao;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < spins * matrix;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto transposed = index / matrix * matrix + index % nao * nao + index / nao % nao;
    const auto current = value[index];
    const auto other = reference ? reference[index] : value[transposed];
    if (!isfinite(current) || !isfinite(other) ||
        fabs(current - other) > absolute + relative * fmax(fabs(current), fabs(other)))
      atomicExch(status, 1);
  }
}
}  // namespace

struct CudaKsWb97mvResponse::Impl {
  methods::PreparedBatch& batch;
  std::size_t item;
  CudaKsFinalStateToken expected;
  CudaKsGridReference reference;
  CudaKsResidentFockBinding source{};
  CudaKsResidentOrbitalFrameBinding frame{};
  scf::CudaDfReadonlySource fitted{};
  scf::CudaDirectJkPlan* direct{};
  scf::CudaDirectJkPlan* range_direct{};
  CudaWb97mvGridResponse grid;
  CudaKsWb97mvResponseOptions options;
  CudaKsWb97mvResponseWork receipt{};
  maps::Extents shape{};
  cudaStream_t stream{};
  void* arena{};
  std::size_t arena_bytes{}, matrix{}, elements{}, map_bytes{}, compensation_elements{};
  void *map_workspace{}, *grid_workspace{};
  std::size_t grid_bytes{};
  double *coulomb{}, *exchange{}, *long_range{}, *primary{}, *potential{}, *candidate{}, *zero{},
      *raw{}, *inverse{}, *seed{}, *compensation{};
  int* status{};
  int* direct_status{};
  std::uint64_t *census{}, *range_census{};
  bool ready{};

  Impl(methods::PreparedBatch& owner, std::size_t index, CudaKsFinalStateToken token,
       CudaKsWb97mvResponseOptions controls)
      : batch(owner),
        item(index),
        expected(std::move(token)),
        reference(batch, item, expected),
        options(controls) {}
  ~Impl() {
    if (stream) (void)cudaStreamSynchronize(stream);
    scf::destroy_cuda_direct_jk_plan(range_direct);
    scf::destroy_cuda_direct_jk_plan(direct);
    if (arena) (void)cudaFree(arena);
  }

  cudaError_t authenticate(cudaStream_t requested) {
    int device = -1;
    cudaStreamCaptureStatus capture{};
    if (requested != stream || cudaGetDevice(&device) != cudaSuccess || device != source.device_id)
      return cudaErrorInvalidResourceHandle;
    auto error = cudaStreamIsCapturing(stream, &capture);
    if (error != cudaSuccess) return error;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    CudaKsResidentFockBinding current;
    std::string detail;
    if (methods::detail::dft_cuda_resident_fock_sources(batch, item, expected, current, detail) !=
            GENERATIVEQC_STATUS_SUCCESS ||
        current != source)
      return cudaErrorInvalidResourceHandle;
    if (fitted &&
        scf::cuda_density_fitting_readonly_source(source.primary->cuda_fitted_source()) != fitted)
      return cudaErrorInvalidResourceHandle;
    const auto lease = reference.reference();
    return lease.authenticate ? lease.authenticate(lease.owner) : cudaErrorInvalidResourceHandle;
  }

  void validate_model() {
    require(cuda_ks_wb97mv_model_matches(expected, source),
            "physical response requires matched native WB97M-V grid, primary J/K and exact LR "
            "correction");
  }

  cudaError_t gate() {
    int failed[2]{};
    auto error = cudaMemcpyAsync(failed, status, sizeof(failed), cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return error;
    if ((error = cudaStreamSynchronize(stream)) != cudaSuccess) return error;
    return failed[0] || failed[1] ? cudaErrorInvalidValue : cudaSuccess;
  }

  generativeqc_status prepare_resident(std::string& detail) {
    const auto base_bytes = scf::cuda_direct_jk_plan_diagnostic(direct).device_bytes;
    auto remaining = std::min(options.max_resident_integral_bytes,
                              options.max_device_bytes - receipt.device_bytes);
    const auto required = scf::cuda_direct_jk_resident_value_bytes(direct);
    if (!required || required > remaining) return GENERATIVEQC_STATUS_SUCCESS;
    const auto optional = [&](generativeqc_status result) -> generativeqc_status {
      if (result != GENERATIVEQC_STATUS_OUT_OF_MEMORY) return result;
      // Optional allocation refusal is not a numerical fallback. Retire only
      // OOM; a real stream/device failure must still fail the response owner.
      const auto pending = cudaGetLastError();
      check(pending == cudaErrorMemoryAllocation ? cudaSuccess : pending);
      check(cudaStreamSynchronize(stream));
      detail.clear();
      return GENERATIVEQC_STATUS_SUCCESS;
    };
    auto result = fitted ? scf::prepare_cuda_direct_jk_range_resident_values(
                               direct, remaining, (1U << 28) - 1, scf::FockOperator::LongRange,
                               source.correction->spec.exchange.omega, detail)
                         : scf::prepare_cuda_direct_jk_resident_values(direct, remaining, detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return optional(result);
    auto diagnostic = scf::cuda_direct_jk_plan_diagnostic(direct);
    receipt.resident_integral_bytes = diagnostic.resident_value_bytes;
    receipt.resident_source_evaluations = diagnostic.resident_values_completed;
    receipt.device_bytes = arena_bytes + diagnostic.device_bytes;
    if (fitted || !receipt.resident_integral_bytes) return GENERATIVEQC_STATUS_SUCCESS;
    remaining = options.max_device_bytes - receipt.device_bytes;
    if (required > options.max_resident_integral_bytes - receipt.resident_integral_bytes ||
        base_bytes > remaining || required > remaining - base_bytes)
      return GENERATIVEQC_STATUS_SUCCESS;
    result = scf::create_cuda_direct_jk_resident_plan_on_stream(
        source.device_id, {source.primary->system()}, 0, remaining, stream, &range_direct,
        diagnostic, detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return optional(result);
    // Canonical metadata is optional inside Direct preparation for through-d
    // bases. A recovered allocation refusal may therefore publish a valid
    // replay-only plan that cannot retain values. Keep the original owner.
    if (!scf::cuda_direct_jk_linear_available(range_direct)) {
      scf::destroy_cuda_direct_jk_plan(range_direct);
      range_direct = nullptr;
      detail.clear();
      return GENERATIVEQC_STATUS_SUCCESS;
    }
    const auto allowance =
        std::min(options.max_resident_integral_bytes - receipt.resident_integral_bytes,
                 remaining - diagnostic.device_bytes);
    result = scf::prepare_cuda_direct_jk_range_resident_values(
        range_direct, allowance, (1U << 28) - 1, scf::FockOperator::LongRange,
        source.correction->spec.exchange.omega, detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) {
      scf::destroy_cuda_direct_jk_plan(range_direct);
      range_direct = nullptr;
      return optional(result);
    }
    diagnostic = scf::cuda_direct_jk_plan_diagnostic(range_direct);
    receipt.resident_integral_bytes += diagnostic.resident_value_bytes;
    receipt.resident_source_evaluations += diagnostic.resident_values_completed;
    receipt.device_bytes += diagnostic.device_bytes;
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  cudaError_t evaluate(const double* density, const double* grid_potential, const double* hcore,
                       bool response) {
    check(cudaMemsetAsync(status, 0, 2 * sizeof(int), stream));
    const auto* beta = source.spins == 2 ? density + matrix : nullptr;
    auto* beta_exchange = source.spins == 2 ? exchange + matrix : nullptr;
    std::string detail;
    if (fitted) {
      check(source.spins == 1 ? maps::launch_rks_fitted(shape, seed, density, inverse, raw, primary,
                                                        map_workspace, map_bytes, status, stream)
                              : maps::launch_uks_fitted(shape, seed, density, inverse, raw, primary,
                                                        map_workspace, map_bytes, status, stream));
      check(gate());
    } else {
      const auto result = scf::enqueue_cuda_direct_jk_compensated_device(
          direct, source.primary->strategy().spec, density, matrix, coulomb, exchange, compensation,
          compensation_elements, direct_status, 0, census, detail, true, beta, beta_exchange,
          false);
      if (result != GENERATIVEQC_STATUS_SUCCESS) return cudaErrorNotSupported;
      check(source.spins == 1 ? maps::launch_rks_exact(shape, coulomb, exchange, primary,
                                                       map_workspace, map_bytes, status, stream)
                              : maps::launch_uks_exact(shape, coulomb, exchange, primary,
                                                       map_workspace, map_bytes, status, stream));
      check(gate());
    }
    const auto result = scf::enqueue_cuda_direct_jk_compensated_device(
        range_direct ? range_direct : direct, source.correction->spec, density, matrix, nullptr,
        long_range, compensation, compensation_elements, direct_status, 0, range_census, detail,
        true, beta, source.spins == 2 ? long_range + matrix : nullptr, false);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return cudaErrorNotSupported;
    check(source.spins == 1
              ? maps::launch_rks_assemble(shape, grid_potential, hcore, long_range, primary,
                                          candidate, map_workspace, map_bytes, status, stream)
              : maps::launch_uks_assemble(shape, grid_potential, hcore, long_range, primary,
                                          candidate, map_workspace, map_bytes, status, stream));
    audit<<<maps::blocks(elements), 128, 0, stream>>>(
        candidate, response ? nullptr : source.fock, source.nao, source.spins,
        response ? options.grid.symmetry_absolute_tolerance : options.reference_absolute_tolerance,
        response ? options.grid.symmetry_relative_tolerance : options.reference_relative_tolerance,
        status);
    check(cudaGetLastError());
    check(gate());
    std::uint64_t counts[4]{};
    check(cudaMemcpyAsync(counts, census, sizeof(counts), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    auto& quartets = response ? receipt.response_quartets : receipt.reference_quartets;
    auto& eris = response ? receipt.response_eris : receipt.reference_eris;
    require(counts[0] <= options.max_direct_eri_evaluations &&
                counts[2] <= options.max_direct_eri_evaluations - counts[0] &&
                counts[1] <= options.max_direct_eri_evaluations &&
                counts[3] <= options.max_direct_eri_evaluations - counts[1],
            "physical response Direct census exceeded its admitted work bound");
    require(quartets <= std::numeric_limits<std::uint64_t>::max() - (counts[0] + counts[2]) &&
                eris <= std::numeric_limits<std::uint64_t>::max() - (counts[1] + counts[3]),
            "physical response work receipt overflow");
    quartets += counts[0] + counts[2];
    eris += counts[1] + counts[3];
    return cudaSuccess;
  }
};

CudaKsWb97mvResponse::CudaKsWb97mvResponse() = default;
CudaKsWb97mvResponse::~CudaKsWb97mvResponse() = default;

generativeqc_status CudaKsWb97mvResponse::prepare(methods::PreparedBatch& batch, std::size_t item,
                                                  const CudaKsFinalStateToken& expected,
                                                  const CudaKsWb97mvResponseOptions& options,
                                                  std::string& detail) {
  impl_.reset();
  detail.clear();
  try {
    require(options.max_device_bytes && options.max_affine_term_visits &&
                options.max_direct_eri_evaluations &&
                std::isfinite(options.reference_absolute_tolerance) &&
                options.reference_absolute_tolerance >= 0 &&
                std::isfinite(options.reference_relative_tolerance) &&
                options.reference_relative_tolerance >= 0,
            "invalid physical response resource or reference gates");
    auto owner = std::make_unique<Impl>(batch, item, expected, options);
    auto result = owner->reference.prepare(detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    result = methods::detail::dft_cuda_resident_fock_sources(batch, item, expected, owner->source,
                                                             detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    owner->validate_model();
    result = methods::detail::dft_cuda_resident_orbital_frame(batch, item, expected, owner->frame,
                                                              detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    const auto lease = owner->reference.reference();
    const auto grid_shape = owner->reference.shape();
    owner->stream = lease.stream;
    require(owner->source.device_id == lease.device_id && owner->source.nao == grid_shape.nao &&
                owner->source.spins == grid_shape.spins &&
                owner->source.stream == reinterpret_cast<void*>(lease.stream),
            "physical response grid and Fock sources differ");
    const auto binding = scf::prepared_cuda_fock_binding(*owner->source.primary);
    require(binding && binding.device_id == lease.device_id && binding.stream == lease.stream &&
                binding.nbf == grid_shape.nao,
            "physical response requires a resident primary Fock source");
    if (owner->source.primary->strategy().spec.coulomb.approximation ==
        scf::FockApproximation::DensityFitted) {
      owner->fitted =
          scf::cuda_density_fitting_readonly_source(owner->source.primary->cuda_fitted_source());
      require(owner->fitted && owner->fitted.nao == grid_shape.nao &&
                  owner->fitted.device_id == lease.device_id &&
                  owner->fitted.stream == lease.stream,
              "physical response requires an immutable full-rank fitted raw/metric source");
    }
    owner->shape = {grid_shape.nao, owner->fitted ? owner->fitted.naux : 0, grid_shape.spins};
    owner->matrix = product(grid_shape.nao, grid_shape.nao);
    owner->elements = product(grid_shape.spins, owner->matrix);
    const auto cartesian = molecule::cartesian_ao_count(owner->source.primary->system());
    const auto quartic = product(product(cartesian, cartesian), product(cartesian, cartesian));
    require(product(owner->fitted ? 1 : 2, quartic) <= options.max_direct_eri_evaluations,
            "physical response Direct ERI work exceeds its explicit bound");
    std::size_t terms{}, bytes{}, total_terms{};
    const auto admit_map = [&](auto workspace, auto work) {
      check(workspace(owner->shape, bytes));
      require(work(owner->shape, terms) && terms <= options.max_affine_term_visits - total_terms,
              "physical response affine schedule exceeds its explicit term-visit bound");
      total_terms += terms;
      owner->map_bytes = std::max(owner->map_bytes, bytes);
    };
    if (owner->fitted) {
      if (grid_shape.spins == 1)
        admit_map(maps::workspace_rks_fitted, maps::work_rks_fitted);
      else
        admit_map(maps::workspace_uks_fitted, maps::work_uks_fitted);
    } else {
      if (grid_shape.spins == 1)
        admit_map(maps::workspace_rks_exact, maps::work_rks_exact);
      else
        admit_map(maps::workspace_uks_exact, maps::work_uks_exact);
    }
    if (grid_shape.spins == 1)
      admit_map(maps::workspace_rks_assemble, maps::work_rks_assemble);
    else
      admit_map(maps::workspace_uks_assemble, maps::work_uks_assemble);
    owner->receipt.affine_action_terms = total_terms;
    if (owner->fitted) admit_map(maps::workspace_inverse, maps::work_inverse);
    owner->receipt.affine_reference_terms = total_terms;
    check(cuda_wb97mv_grid_workspace(grid_shape, options.grid, owner->grid_bytes));
    Layout layout;
    const auto grid_offset = layout.add(owner->grid_bytes / sizeof(double));
    const auto map_offset = layout.add(owner->map_bytes / sizeof(double));
    const auto coulomb_offset = layout.add(owner->matrix);
    const auto exchange_offset = layout.add(owner->elements);
    const auto range_offset = layout.add(owner->elements);
    const auto primary_offset = layout.add(owner->elements);
    const auto potential_offset = layout.add(owner->elements);
    const auto candidate_offset = layout.add(owner->elements);
    const auto zero_offset = layout.add(owner->matrix);
    const auto raw_offset =
        layout.add(owner->fitted ? product(owner->matrix, owner->fitted.naux) : 0);
    const auto inverse_offset =
        layout.add(owner->fitted ? product(owner->fitted.naux, owner->fitted.naux) : 0);
    const auto seed_offset = layout.add(1);
    owner->compensation_elements =
        product(product(2, grid_shape.spins), product(cartesian, cartesian));
    const auto compensation_offset = layout.add(owner->compensation_elements);
    const auto status_offset = layout.add(1);
    const auto census_offset = layout.add(4);
    owner->arena_bytes = layout.bytes;
    require(owner->arena_bytes < options.max_device_bytes,
            "physical response workspace exceeds its explicit device bound");
    scf::CudaDirectJkDiagnostic diagnostic;
    result = scf::create_cuda_direct_jk_resident_plan_on_stream(
        lease.device_id, {owner->source.primary->system()}, 0,
        options.max_device_bytes - owner->arena_bytes, lease.stream, &owner->direct, diagnostic,
        detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    require(scf::cuda_direct_jk_linear_available(owner->direct),
            "physical response requires admitted canonical Direct storage");
    owner->receipt.device_bytes = owner->arena_bytes + diagnostic.device_bytes;
    require(owner->receipt.device_bytes <= options.max_device_bytes,
            "physical response total device storage exceeds its explicit bound");
    check(cudaMalloc(&owner->arena, owner->arena_bytes));
    const auto panel = [&](std::size_t offset) {
      return reinterpret_cast<double*>(static_cast<char*>(owner->arena) + offset);
    };
    owner->grid_workspace = panel(grid_offset);
    owner->map_workspace = panel(map_offset);
    owner->coulomb = panel(coulomb_offset);
    owner->exchange = panel(exchange_offset);
    owner->long_range = panel(range_offset);
    owner->primary = panel(primary_offset);
    owner->potential = panel(potential_offset);
    owner->candidate = panel(candidate_offset);
    owner->zero = panel(zero_offset);
    owner->raw = panel(raw_offset);
    owner->inverse = panel(inverse_offset);
    owner->seed = panel(seed_offset);
    owner->compensation = panel(compensation_offset);
    owner->status = reinterpret_cast<int*>(panel(status_offset));
    owner->direct_status = owner->status + 1;
    owner->census = reinterpret_cast<std::uint64_t*>(panel(census_offset));
    owner->range_census = owner->census + 2;
    check(cudaMemsetAsync(owner->arena, 0, owner->arena_bytes, lease.stream));
    const double seed = 1;
    check(cudaMemcpyAsync(owner->seed, &seed, sizeof(seed), cudaMemcpyHostToDevice, lease.stream));
    if (owner->fitted) {
      result = scf::enqueue_cuda_density_fitting_raw_values(
          owner->fitted.integrals, 0, owner->matrix, reinterpret_cast<void*>(lease.stream),
          owner->raw, detail);
      if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
      check(maps::launch_inverse(owner->shape, owner->fitted.whitening, owner->inverse,
                                 owner->map_workspace, owner->map_bytes, owner->status,
                                 lease.stream));
      check(owner->gate());
    }
    check(owner->grid.prepare(grid_shape, options.grid, lease, owner->grid_workspace,
                              owner->grid_bytes));
    if (!options.defer_resident_integrals) {
      result = owner->prepare_resident(detail);
      if (result != GENERATIVEQC_STATUS_SUCCESS) return result;
    }
    const double* reference_potential{};
    check(owner->grid.reference_potential(reference_potential));
    check(owner->evaluate(lease.density, reference_potential, owner->source.hcore, false));
    check(owner->authenticate(lease.stream));
    owner->ready = true;
    impl_ = std::move(owner);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::invalid_argument& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception& error) {
    detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}

cudaError_t CudaKsWb97mvResponse::apply(const double* direction, double* output,
                                        cudaStream_t stream) noexcept {
  if (!impl_ || !impl_->ready) return cudaErrorInvalidResourceHandle;
  try {
    auto& owner = *impl_;
    auto error = owner.authenticate(stream);
    if (error != cudaSuccess) {
      owner.ready = false;
      return error;
    }
    const auto bytes = owner.elements * sizeof(double);
    for (const auto* pointer : {direction, static_cast<const double*>(output)}) {
      cudaPointerAttributes attributes{};
      if (!pointer || reinterpret_cast<std::uintptr_t>(pointer) % alignof(double) ||
          cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess ||
          attributes.type != cudaMemoryTypeDevice || attributes.device != owner.source.device_id ||
          overlap(pointer, bytes, owner.arena, owner.arena_bytes))
        return cudaErrorInvalidValue;
    }
    if (overlap(direction, bytes, output, bytes) ||
        overlap(output, bytes, owner.source.hcore, owner.matrix * sizeof(double)) ||
        overlap(output, bytes, owner.source.fock, bytes) ||
        overlap(output, bytes, owner.frame.coefficients, bytes) ||
        overlap(output, bytes, owner.frame.orbital_energies,
                owner.source.spins * owner.source.nao * sizeof(double)) ||
        overlap(output, bytes, owner.frame.density, bytes) ||
        overlap(output, bytes, owner.frame.overlap, owner.matrix * sizeof(double)))
      return cudaErrorInvalidValue;
    const auto grid_lease = owner.reference.reference();
    const auto grid_shape = owner.reference.shape();
    const auto basis_elements =
        3 * grid_shape.atoms + 2 * grid_shape.primitives + 16 * grid_shape.nao;
    if (overlap(output, bytes, grid_lease.basis, basis_elements * sizeof(double)) ||
        overlap(output, bytes, grid_lease.points, 3 * grid_shape.points * sizeof(double)) ||
        overlap(output, bytes, grid_lease.weights, grid_shape.points * sizeof(double)) ||
        (owner.fitted && overlap(output, bytes, owner.fitted.whitening,
                                 owner.fitted.naux * owner.fitted.naux * sizeof(double))))
      return cudaErrorInvalidValue;
    if (owner.receipt.actions == std::numeric_limits<std::uint64_t>::max())
      return cudaErrorInvalidValue;
    struct DrainOnFailure {
      cudaStream_t stream;
      bool active{true};
      ~DrainOnFailure() {
        // Later launch/API failure must not return while queued work still
        // borrows the caller's direction or output storage.
        if (active) (void)cudaStreamSynchronize(stream);
      }
    } drain{stream};
    check(owner.grid.apply(direction, owner.potential, stream));
    check(owner.evaluate(direction, owner.potential, owner.zero, true));
    error = owner.authenticate(stream);
    if (error != cudaSuccess) {
      owner.ready = false;
      return error;
    }
    check(cudaMemcpyAsync(output, owner.candidate, bytes, cudaMemcpyDeviceToDevice, stream));
    check(cudaStreamSynchronize(stream));
    drain.active = false;
    ++owner.receipt.actions;
    return cudaSuccess;
  } catch (...) {
    return cudaErrorInvalidValue;
  }
}

generativeqc_status CudaKsWb97mvResponse::retain_integrals(std::size_t maximum_additional_bytes,
                                                           std::string& detail) {
  if (!impl_ || !impl_->ready || impl_->receipt.actions || impl_->receipt.resident_integral_bytes) {
    detail = "optional integral admission requires an unused prepared response owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  auto& owner = *impl_;
  try {
    check(owner.authenticate(owner.stream));
    const auto remaining = owner.options.max_device_bytes - owner.receipt.device_bytes;
    owner.options.max_device_bytes =
        owner.receipt.device_bytes + std::min(remaining, maximum_additional_bytes);
    const auto result = owner.prepare_resident(detail);
    if (result != GENERATIVEQC_STATUS_SUCCESS) {
      owner.ready = false;
      return result;
    }
    check(owner.authenticate(owner.stream));
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::exception& error) {
    owner.ready = false;
    detail = error.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}

CudaKsWb97mvResponseWork CudaKsWb97mvResponse::work() const noexcept {
  if (!impl_) return {};
  auto receipt = impl_->receipt;
  receipt.grid = impl_->grid.work();
  return receipt;
}

cudaError_t CudaKsWb97mvResponse::induced_fock(void* context, const double* direction,
                                               double* output, const response::CudaCpksShape& shape,
                                               cudaStream_t stream) noexcept {
  if (!context) return cudaErrorInvalidValue;
  auto& self = *static_cast<CudaKsWb97mvResponse*>(context);
  if (!self.impl_ || shape.nao != self.impl_->source.nao || shape.spins != self.impl_->source.spins)
    return cudaErrorInvalidValue;
  return self.apply(direction, output, stream);
}

}  // namespace generativeqc::dft
