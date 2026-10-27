#include "dft/cuda_stationary_controller.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

#include "dft/cuda_grid_api.hpp"
#include "dft/cuda_stationary_capacity.hpp"
#include "generativeqc/generativeqc.hpp"
#include "runtime/cuda_resources.cuh"
#include "scf/cuda_fock_execution.hpp"
#include "scf/cuda_one_electron_gradient.hpp"
#include "xc_cpu_generated.hpp"

namespace generativeqc::dft {
namespace {
void require_stationary_call(int status, const std::array<char, 512>& error) {
  // Private executor codes are device/host OOM (2/3) and driver failure (4),
  // not the public enum. Driver failure must not trigger a numerical seed
  // retry.
  if (status == 2 || status == 3) throw std::bad_alloc();
  if (status == 4) throw Error(GENERATIVEQC_STATUS_CUDA_ERROR, error.data());
  if (status) throw std::runtime_error(error.data());
}
void require_grid_call(int status, const std::array<char, 512>& error) {
  if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status) throw Error(static_cast<generativeqc_status>(status), error.data());
}
/** Preserve driver errors through the same typed public boundary as grid calls.
 */
void stationary_cuda_check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw Error(GENERATIVEQC_STATUS_CUDA_ERROR, cudaGetErrorString(status));
}
}  // namespace

/** Full-grid VV10 seeds stay on device; only topology is retained on host. */
struct CudaStationaryController::NonlocalOwner {
  CudaMolecularGridView quadrature;
  nlc::Vv10CudaDeviceLayout layout;
  std::size_t points_per_atom{};
  std::vector<std::size_t> xc_offsets, offsets;
  std::vector<double> xc_radii, radii;
  runtime::OwnedCudaBuffer<double> effective, seeds, workspace;
  runtime::OwnedCudaBuffer<int> errors;
};

CudaStationaryController::CudaStationaryController(const core::System& system, const AoBasis& basis,
                                                   const MolecularGrid& grid,
                                                   const CudaStationaryBinding& binding, int device,
                                                   std::size_t tile_points,
                                                   std::size_t maximum_device_bytes,
                                                   const MolecularGrid* nonlocal_grid)
    : device_(device),
      binding_(&binding),
      quadrature_(grid.cuda_view()),
      aos_(basis.nao),
      tile_points_(std::min(
          tile_points,
          std::max(grid.point_count(), nonlocal_grid ? nonlocal_grid->point_count() : 0))) {
  if (!binding_ || !quadrature_ || quadrature_.device != device || !tile_points_ || !basis.natom ||
      basis.natom != system.atoms.size() || grid.spec().partition_iterations != 3 ||
      !system.ecp_terms.empty() ||
      std::any_of(system.atoms.begin(), system.atoms.end(),
                  [](const auto& atom) { return atom.ecp_core != 0; }))
    throw std::invalid_argument(
        "native CUDA stationary owner has no qualified profile/grid domain");
  points_per_atom_ = grid.spec().version == 4 ? 0 : grid_points_per_atom(grid.spec());
  coincident_tolerance_ = grid.spec().coincident_tolerance;
  const auto capacity = cuda_stationary_capacity(*binding_, basis.natom, aos_, basis.nprimitive,
                                                 grid.point_count(), tile_points_, false,
                                                 nonlocal_grid ? nonlocal_grid->point_count() : 0);
  if (basis.packed.size() != capacity.packed_elements)
    throw std::invalid_argument("native stationary capacity/basis layout mismatch");
  const auto& storage = capacity.grid;
  maximum_device_bytes = std::min(maximum_device_bytes, runtime::device_resource_available(device));
  resources_.minimum_device_bytes = capacity.minimum_device_bytes;
  if (resources_.minimum_device_bytes > maximum_device_bytes) throw std::bad_alloc();
  const bool cached = capacity.cached_device_bytes <= maximum_device_bytes;
  resources_.cached_geometry = cached;
  const auto stationary_budget =
      cached ? capacity.stationary_cached_bytes : capacity.stationary_minimum_bytes;
  resources_.device_upper_bytes =
      cached ? capacity.cached_device_bytes : capacity.minimum_device_bytes;
  resources_.grid_numeric_bytes = storage.allocation_bytes;
  resources_.stationary_numeric_bytes = stationary_budget + capacity.nonlocal_device_bytes;
  resources_.grid_provider_allowance_bytes = storage.provider_allowance;
  mixed_grid_budget_ = maximum_device_bytes - resources_.device_upper_bytes;
  unsigned order = 0;
  while ((order + 1) * (order + 2) * (order + 3) / 6 < binding_->ao_jets) ++order;
  if (order > 3 || (order + 1) * (order + 2) * (order + 3) / 6 != binding_->ao_jets)
    throw std::invalid_argument("native stationary profile AO jets have no grid producer");
  for (const auto& atom : system.atoms) {
    centers_.insert(centers_.end(), atom.position.begin(), atom.position.end());
    charges_.push_back(atom.atomic_number);
  }
  runtime::CudaDeviceScope device_scope(device, stationary_cuda_check);
  cudaDeviceProp property{};
  stationary_cuda_check(cudaGetDeviceProperties(&property, device));
  std::array<char, 512> error{};
  try {
    if (binding.model.nonlocal_b != 0) {
      if (!nonlocal_grid || !binding.nonlocal_geometry || !binding.grid_topology)
        throw std::invalid_argument("native composite lacks nonlocal grid/provider");
      nonlocal_ = std::make_unique<NonlocalOwner>();
      auto& pair = *nonlocal_;
      pair.quadrature = nonlocal_grid->cuda_view();
      if (!pair.quadrature || pair.quadrature.device != device)
        throw std::invalid_argument("native nonlocal quadrature device mismatch");
      const auto points = pair.quadrature.point_count;
      pair.points_per_atom =
          nonlocal_grid->spec().version == 4 ? 0 : grid_points_per_atom(nonlocal_grid->spec());
      pair.layout = nlc::vv10_cuda_device_layout(points, tile_points_, true, true);
      pair.effective.allocate(device, 5 * points);
      pair.seeds.allocate(device, 6 * points + 1);
      pair.workspace.allocate(device, pair.layout.workspace_bytes / sizeof(double));
      pair.errors.allocate(device, 3);
      auto topology = [&](const MolecularGrid& source, auto& offsets, auto& radii) {
        offsets.push_back(0);
        for (const auto& atom : system.atoms) {
          offsets.push_back(offsets.back() +
                            (source.spec().version == 4
                                 ? grid_atom_point_count(source.spec(), atom.atomic_number)
                                 : grid_points_per_atom(source.spec())));
          if (source.spec().version == 4)
            radii.push_back(source.spec().element_radii[atom.atomic_number]);
        }
        if (offsets.back() != source.point_count())
          throw std::invalid_argument("native nonlocal topology extent mismatch");
      };
      topology(grid, pair.xc_offsets, pair.xc_radii);
      topology(*nonlocal_grid, pair.offsets, pair.radii);
    }
    grid_provider_allowance_.emplace(device, storage.provider_allowance);
    require_stationary_call(
        binding_->create(device, property.major, property.minor, basis.natom, aos_,
                         basis.nprimitive, tile_points_, 1, binding_->spin_blocks,
                         std::max(std::size_t{1}, basis.natom * (basis.natom - 1) / 2),
                         stationary_budget, capacity.geometry_lanes, capacity.geometry_threads,
                         &stationary_, error.data(), error.size()),
        error);
    if (grid.spec().version == 4) {
      if (!binding_->grid_topology)
        throw std::invalid_argument("stationary binding lacks variable atom grid support");
      std::vector<std::size_t> offsets{0};
      std::vector<double> radii;
      for (const auto& atom : system.atoms) {
        offsets.push_back(offsets.back() + grid_atom_point_count(grid.spec(), atom.atomic_number));
        radii.push_back(grid.spec().element_radii[atom.atomic_number]);
      }
      if (offsets.back() != grid.point_count())
        throw std::invalid_argument("stationary molecular grid topology mismatch");
      require_stationary_call(binding_->grid_topology(stationary_, offsets.data(), radii.data(),
                                                      basis.natom, error.data(), error.size()),
                              error);
    }
    std::vector<std::int64_t> ranges(2 * aos_), atoms(aos_);
    // This owner never submits AO integral tasks: native providers supply those
    // already-contracted sources. Only atom labels enter geometry AD;
    // normalized AO expansions remain owned by the grid's unchanged packed
    // basis.
    std::vector<double> norms(aos_, 1.0);
    const auto* primitives = basis.packed.data() + 3 * basis.natom;
    const auto* records = primitives + 2 * basis.nprimitive;
    for (std::size_t orbital = 0; orbital < aos_; ++orbital) {
      atoms[orbital] = static_cast<std::int64_t>(records[16 * orbital]);
      ranges[2 * orbital] = static_cast<std::int64_t>(records[16 * orbital + 1]);
      ranges[2 * orbital + 1] = static_cast<std::int64_t>(records[16 * orbital + 2]);
    }
    require_stationary_call(binding_->topology(stationary_, primitives, ranges.data(), norms.data(),
                                               atoms.data(), error.data(), error.size()),
                            error);
    const std::array<std::size_t, 3> dimensions{basis.natom, basis.nprimitive, aos_};
    require_grid_call(grid_cuda_create_v3(
                          device, property.major, property.minor, dimensions.data(),
                          basis.packed.data(), tile_points_, order, storage.allocation_bytes, aos_,
                          nullptr, 0, binding_->feature_mask, &grid_, error.data(), error.size()),
                      error);
    generativeqc_tensor::Metrics metrics;
    std::array<int, 3> versions{};
    require_grid_call(
        grid_cuda_metrics_v1(grid_, &metrics, versions.data(), error.data(), error.size()), error);
    resources_.grid_provider_retained_bytes = metrics.provider_retained_bytes;
    resources_.grid_prepare_device_delta = metrics.prepare_device_delta;
    if (metrics.provider_retained_bytes > storage.provider_allowance)
      throw std::runtime_error("native stationary grid provider exceeds its retained allowance");
  } catch (...) {
    close();
    throw;
  }
}

CudaStationaryController::~CudaStationaryController() { close(); }

void CudaStationaryController::close() noexcept {
  // The stationary consumer borrows the grid stream and tile storage. It must
  // drain/die first on every ordinary and exceptional lifecycle path.
  if (stationary_) binding_->destroy(stationary_);
  stationary_ = nullptr;
  if (grid_) grid_cuda_destroy_v1(grid_);
  grid_ = nullptr;
  nonlocal_.reset();
  grid_provider_allowance_.reset();
  mixed_grid_provider_allowance_.reset();
}

generativeqc_status CudaStationaryController::gradient(
    const CudaKsResidentDensityBinding& density,
    const CudaKsResidentStationaryWeightsBinding& weights, const scf::PreparedFockPlan& fock,
    const std::function<bool()>& token_current, std::vector<double>& output, std::string& detail,
    bool allow_spatial_default, scf::DirectForcePrecisionRequest* precision,
    const scf::ResolvedFockBuild* range, const CudaKsResidentNonlocalFeaturesBinding* nonlocal) {
  detail.clear();
  try {
    const auto derivative = scf::prepared_cuda_direct_derivative_binding(fock);
    if (!stationary_ || !grid_ || !density || !weights || !derivative || !token_current ||
        !token_current() || density.device_id != device_ || weights.device_id != device_ ||
        derivative.device_id != device_ || density.matrix_elements != aos_ * aos_ ||
        weights.matrix_elements != density.matrix_elements ||
        density.spins != binding_->spin_blocks || weights.spins != density.spins ||
        weights.owner != density.owner || weights.solve_epoch != density.solve_epoch ||
        weights.generation != density.generation)
      throw std::invalid_argument(
          "native stationary final density/weight token is stale or "
          "incompatible");
    const auto& spec = fock.strategy().spec;
    const bool composite = binding_->model.range_omega != 0;
    const auto expected_exchange =
        (density.spins == 1 ? -0.5 : -1.0) *
        (composite ? binding_->model.short_range_exchange : binding_->full_range_exchange);
    const auto full_range = [](const scf::FockTermSpec& term) {
      return !term.present || (term.op == scf::FockOperator::FullRange && term.omega == 0.0 &&
                               term.approximation == scf::FockApproximation::Exact);
    };
    if (fock.strategy().backend != scf::FockBackend::Cuda || !spec.coulomb.present ||
        spec.coulomb.coefficient != 1.0 || !full_range(spec.coulomb) ||
        !full_range(spec.exchange) || spec.exchange.present != (expected_exchange != 0.0) ||
        (spec.exchange.present && spec.exchange.coefficient != expected_exchange) ||
        (spec.spin == scf::FockSpin::Restricted ? 1U : 2U) != density.spins)
      throw std::invalid_argument("native stationary profile disagrees with prepared Fock model");
    if (composite &&
        (!range || !nonlocal_ || !nonlocal || !*nonlocal || nonlocal->device_id != device_ ||
         nonlocal->owner != density.owner || nonlocal->solve_epoch != density.solve_epoch ||
         nonlocal->generation != density.generation ||
         nonlocal->point_count != nonlocal_->quadrature.point_count ||
         range->spec.exchange.omega != binding_->model.range_omega ||
         range->spec.exchange.coefficient !=
             (density.spins == 1 ? -0.5 : -1.0) *
                 (binding_->model.long_range_exchange - binding_->model.short_range_exchange)))
      throw std::invalid_argument("native composite final-state/model lease mismatch");
    runtime::CudaDeviceScope device_scope(device_, stationary_cuda_check);
    std::array<char, 512> error{};
    require_stationary_call(
        binding_->reset_geometry(stationary_, centers_.data(), coincident_tolerance_, error.data(),
                                 error.size()),
        error);
    require_stationary_call(binding_->nuclear(stationary_, charges_.data(), charges_.size(),
                                              error.data(), error.size()),
                            error);
    require_grid_call(
        grid_cuda_density_device_v1(grid_, density.alpha, density.beta, density.matrix_elements,
                                    density.spins, density.stream, error.data(), error.size()),
        error);
    if (nonlocal_) {
      const auto& nlc = *nonlocal_;
      require_stationary_call(
          binding_->grid_topology(stationary_, nlc.xc_offsets.data(),
                                  nlc.xc_radii.empty() ? nullptr : nlc.xc_radii.data(),
                                  centers_.size() / 3, error.data(), error.size()),
          error);
    }
    const bool mixed_geometry = precision && precision->mixed_geometry;
    bool mixed_grid = precision && precision->mixed_grid;
    const auto geometry = mixed_geometry ? binding_->geometry_fp32 : binding_->geometry;
    if (!geometry) throw std::invalid_argument("no typed FP32 stationary geometry binding");
    if (precision) precision->geometry_points = 0;
    if (precision) precision->grid_points = 0;
    if (mixed_grid && !mixed_grid_bytes_) {
      try {
        mixed_grid_provider_allowance_.emplace(device_, CudaGridStorage::provider_allowance);
        require_grid_call(grid_cuda_prepare_fp32_v1(grid_, mixed_grid_budget_, &mixed_grid_bytes_,
                                                    error.data(), error.size()),
                          error);
        resources_.device_upper_bytes += mixed_grid_bytes_;
        resources_.grid_numeric_bytes += mixed_grid_bytes_ - CudaGridStorage::provider_allowance;
        resources_.grid_provider_allowance_bytes += CudaGridStorage::provider_allowance;
      } catch (const std::bad_alloc&) {
        // Optional resident float panels may not displace the admitted strict
        // owner. Zero grid_points records the fallback; no clipping or CPU work.
        mixed_grid_provider_allowance_.reset();
        mixed_grid = false;
      }
    }
    for (std::size_t offset = 0; offset < quadrature_.point_count; offset += tile_points_) {
      const auto points = std::min(tile_points_, quadrature_.point_count - offset);
      if (mixed_grid) {
        require_grid_call(
            grid_cuda_run_fp32_device_deferred_v1(grid_, quadrature_.points + 3 * offset, points,
                                                  aos_, error.data(), error.size()),
            error);
        precision->grid_points += points;
      } else
        require_grid_call(grid_cuda_run_selected_device_deferred_v1(
                              grid_, quadrature_.points + 3 * offset, points, 1, nullptr, aos_,
                              nullptr, nullptr, error.data(), error.size()),
                          error);
      GridTaskView view;
      require_grid_call(grid_cuda_view_v1(grid_, &view, error.data(), error.size()), error);
      const double* work = nullptr;
      std::uint64_t flags = 0;
      require_grid_call(grid_cuda_density_jets_v2(grid_, view.generation, binding_->density_jets,
                                                  &work, &flags, error.data(), error.size()),
                        error);
      require_stationary_call(
          geometry(stationary_, &view, work, offset, points_per_atom_, quadrature_.weights + offset,
                   quadrature_.atomic_weights + offset, view.generation, flags, error.data(),
                   error.size()),
          error);
      if (mixed_geometry) precision->geometry_points += points;
    }
    if (nonlocal_) {
      auto& pair = *nonlocal_;
      const auto& quadrature = pair.quadrature;
      const auto points = quadrature.point_count;
      GridTaskView view;
      require_grid_call(grid_cuda_view_v1(grid_, &view, error.data(), error.size()), error);
      const auto stream = view.stream;
      // Order the borrowed final feature lease against the grid producer;
      // neither dense D/W nor full-grid feature arrays cross to the host.
      runtime::OwnedCudaEvent ready(device_, cudaEventDisableTiming, stationary_cuda_check);
      ready.record(reinterpret_cast<cudaStream_t>(nonlocal->stream));
      stationary_cuda_check(cudaStreamWaitEvent(stream, ready.get(), 0));
      stationary_cuda_check(cudaMemsetAsync(pair.errors.get(), 0, 3 * sizeof(int), stream));
      auto* effective = pair.effective.get();
      auto* seeds = pair.seeds.get();
      nlc::enqueue_vv10_molecular_domain_cuda(
          stream, points, generated::kMolecularVv10DensityThreshold, quadrature.weights,
          nonlocal->density, nonlocal->gradient, effective, effective + points,
          effective + 2 * points, pair.errors.get() + 1);
      const auto& model = binding_->model;
      nlc::enqueue_vv10_cuda_device(
          pair.layout,
          {nlc::Vv10Variant::vv10, model.nonlocal_b, model.nonlocal_c, model.nonlocal_coefficient},
          device_, stream, quadrature.points, effective, effective + points, effective + 2 * points,
          pair.workspace.get(), pair.layout.workspace_bytes, seeds + 6 * points, seeds,
          seeds + points, effective + 2 * points, seeds + 5 * points, pair.errors.get() + 2);
      nlc::enqueue_vv10_pack_force_seeds_cuda(stream, points, effective, effective + 2 * points,
                                              seeds, pair.errors.get(), pair.errors.get() + 1,
                                              pair.errors.get() + 2);
      // The same bounded AO/grid owner can serve a distinct nonlocal grid.
      // Changing atom-prefix/radius topology first drains prior geometry work.
      require_stationary_call(
          binding_->grid_topology(stationary_, pair.offsets.data(),
                                  pair.radii.empty() ? nullptr : pair.radii.data(),
                                  centers_.size() / 3, error.data(), error.size()),
          error);
      for (std::size_t offset = 0; offset < points; offset += tile_points_) {
        const auto count = std::min(tile_points_, points - offset);
        require_grid_call(grid_cuda_run_selected_device_deferred_v1(
                              grid_, quadrature.points + 3 * offset, count, 1, nullptr, aos_,
                              nullptr, nullptr, error.data(), error.size()),
                          error);
        require_grid_call(grid_cuda_view_v1(grid_, &view, error.data(), error.size()), error);
        const double* work = nullptr;
        std::uint64_t flags = 0;
        require_grid_call(grid_cuda_density_jets_v2(grid_, view.generation, binding_->density_jets,
                                                    &work, &flags, error.data(), error.size()),
                          error);
        require_stationary_call(binding_->nonlocal_geometry(
                                    stationary_, &view, work, offset, pair.points_per_atom,
                                    quadrature.weights + offset, quadrature.atomic_weights + offset,
                                    seeds, points, offset, error.data(), error.size()),
                                error);
      }
    }
    const scf::CudaDerivativeSourceConsumer consume = [&](const auto& view) {
      require_stationary_call(
          binding_->consume_integrals(stationary_, &view, error.data(), error.size()), error);
    };
    auto status = scf::consume_prepared_cuda_stationary_one_electron_pair(
        fock.cuda_direct_source(), weights.density, weights.weighted_density,
        weights.matrix_elements, consume, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    status = composite ? scf::consume_prepared_cuda_direct_rsh_energy_derivatives_device(
                             fock, *range, density.alpha, density.beta, density.matrix_elements,
                             consume, detail)
                       : scf::consume_prepared_cuda_direct_shell_full_range_derivatives_device(
                             fock, density.alpha, density.beta, density.matrix_elements, consume,
                             detail, allow_spatial_default, precision);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    std::vector<double> candidate(centers_.size());
    require_stationary_call(
        binding_->finish_gradient(stationary_, quadrature_.point_count, candidate.data(),
                                  candidate.size(), error.data(), error.size()),
        error);
    require_stationary_call(binding_->metrics(stationary_, resources_.stationary_work.data(),
                                              resources_.stationary_work.size()),
                            error);
    require_grid_call(
        grid_cuda_work_metrics_v1(grid_, &resources_.grid_work, error.data(), error.size()), error);
    if (!token_current()) throw std::invalid_argument("native stationary terminal token changed");
    output.swap(candidate);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "native stationary controller exceeds its admitted resource capacity";
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument& failure) {
    detail = failure.what();
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const Error& failure) {
    detail = failure.what();
    return failure.status();
  } catch (const std::exception& failure) {
    detail = failure.what();
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  }
}
}  // namespace generativeqc::dft
