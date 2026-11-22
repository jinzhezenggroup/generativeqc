#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "api/handles.hpp"
#include "dft/application_precision.hpp"
#include "dft/cuda_ks_grid_reference.hpp"
#include "dft/cuda_ks_wb97mv_hessian.hpp"
#include "dft/cuda_stationary_controller.hpp"
#include "dft/semilocal_family.hpp"
#include "generativeqc/ks_hessian.h"
#include "methods/named_ks_catalog.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda_density_fitting_device.hpp"

namespace {
using namespace generativeqc;
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }
struct Storage {
  std::vector<double*> allocations;
  ~Storage() {
    for (auto* pointer : allocations) cudaFree(pointer);
  }
  double* allocate(std::size_t count) {
    double* pointer{};
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(double)));
    allocations.push_back(pointer);
    return pointer;
  }
};
template <typename T>
std::vector<T> download(const T* pointer, std::size_t count) {
  std::vector<T> result(count);
  check(cudaMemcpy(result.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
  return result;
}
template <typename T>
void array(const char* name, const std::vector<T>& values) {
  std::cout << ",\"" << name << "\":[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << values[index];
  }
  std::cout << ']';
}
void grid_topology(dft::CudaKsGridReference& grid, const char* offsets, const char* radii) {
  const auto geometry = grid.geometry_reference();
  if (!geometry.atom_offsets) return;
  const auto atoms = grid.geometry_shape().atoms;
  array(offsets, download(geometry.atom_offsets, atoms + 1));
  array(radii, download(geometry.atomic_radii, atoms));
}
void run(bool unrestricted, const std::string& mode, bool full) {
  const bool fitted = mode.find("df") != std::string::npos,
             fitted_range = mode.find("-lrdf") != std::string::npos,
             ecp = mode.find("ecp") != std::string::npos,
             spherical = mode.find("spherical") != std::string::npos,
             auxiliary_d = mode.find("auxd") != std::string::npos,
             auxiliary_g = mode.find("auxg") != std::string::npos,
             orbital_d = mode.find("orbd") != std::string::npos,
             orbital_f = mode.find("orbf") != std::string::npos,
             benchmark_h2 = mode.find("benchmark-h2") != std::string::npos,
             benchmark_water = mode.find("benchmark-water") != std::string::npos;
  const unsigned orbital_angular = orbital_f ? 3U : orbital_d ? 2U : 0U;
  const bool split_grid = mode.find("-split-grid") != std::string::npos;
  const bool resource_ledger = mode.find("-resource-ledger") != std::string::npos;
  const bool auto_precision = mode.find("-auto-precision") != std::string::npos;
  const bool borrow_force_centers = mode.find("-borrow-force-centers") != std::string::npos;
  const bool fitted_xc_warmup = mode.find("-fitted-xc-warmup") != std::string::npos;
  const bool split_treutler = mode.find("-split-grid-treutler") != std::string::npos;
  const bool split_budget = mode.find("-split-grid-budget") != std::string::npos;
  const bool truncated_range = mode.find("-rank-lr") != std::string::npos;
  const bool truncated_primary = mode.find("-rank-primary") != std::string::npos;
  const bool truncated_both = mode.find("-rank-both") != std::string::npos;
  const bool near_metric_cutoff = mode.find("-rank-boundary") != std::string::npos;
  const bool lda = mode.find("-lda") != std::string::npos;
  const bool pbe = mode.find("-pbe") != std::string::npos;
  const bool pbe0 = mode.find("-pbe0") != std::string::npos;
  const bool pbe_d4 = mode.find("-pbe-d4") != std::string::npos;
  const bool d3_hard_cutoff = mode.find("-d3-hard-cutoff") != std::string::npos;
  const bool b3lyp = mode.find("-b3lyp") != std::string::npos;
  const bool r2scan = mode.find("-r2scan") != std::string::npos;
  const bool revised = mode.find("-r2scan-rvv10") != std::string::npos;
  const bool composite = mode.find("-r2scan-3c") != std::string::npos;
  const bool m062x = mode.find("-m06-2x") != std::string::npos;
  const bool mn15 = mode.find("-mn15") != std::string::npos;
  const bool split_hybrid = m062x || mn15;
  const bool semilocal_grid = lda || pbe || b3lyp || split_hybrid || (r2scan && !revised);
  const auto family = lda      ? dft::SemilocalFamily::Lda
                      : pbe    ? dft::SemilocalFamily::Pbe
                      : b3lyp  ? dft::SemilocalFamily::B3lyp
                      : r2scan ? dft::SemilocalFamily::R2scan
                               : dft::SemilocalFamily::Wb97mv;
  const bool separate_auxiliary = auxiliary_d || auxiliary_g || (fitted && split_hybrid);
  const bool compact_fixture = separate_auxiliary || orbital_angular || split_hybrid;
  require(!fitted_xc_warmup || (pbe && !pbe0 && !pbe_d4 && fitted && auxiliary_d && spherical &&
                                !ecp && !unrestricted && !orbital_angular),
          "fitted XC warmup needs the all-electron PBE RKS fixture");
  require(!(truncated_range || truncated_both || near_metric_cutoff) ||
              (fitted_range && !compact_fixture && !ecp && !semilocal_grid),
          "fixed-rank tests require the minimal all-electron fitted Wb97m-V model");
  require(!truncated_primary || (pbe && fitted && !fitted_range && !compact_fixture && !ecp),
          "primary-only cutoff fixture requires minimal all-electron PBE or PBE0 DF");
  require(!auxiliary_d || !auxiliary_g, "select only one independent auxiliary fixture");
  require(!orbital_d || !orbital_f, "select only one high-angular orbital fixture");
  require(!separate_auxiliary || fitted, "the independent auxiliary fixture requires DF");
  require(!benchmark_h2 || !benchmark_water, "select only one external-comparison geometry");
  require(
      !composite || (spherical && !ecp && !compact_fixture && !benchmark_h2 && !benchmark_water),
      "r2SCAN-3c requires its method-owned spherical all-electron basis");
  require(!(benchmark_h2 || benchmark_water) || (!unrestricted && (!fitted || pbe_d4) && !ecp &&
                                                 !separate_auxiliary && !orbital_angular),
          "external-comparison fixtures require all-electron RKS (DF for PBE-D4 only)");
  require(!pbe_d4 || benchmark_water, "PBE-D4 fixture requires water to exercise ATM");
  require(!d3_hard_cutoff || (pbe && !pbe_d4 && !pbe0 && !unrestricted && !ecp &&
                              !compact_fixture && !benchmark_water && !benchmark_h2),
          "D3 cutoff fixture requires the two-atom all-electron PBE case");
  core::System system;
  system.multiplicity = unrestricted ? 2 : 1;
  if (spherical) system.basis_representation = core::kSphericalBasis;
  for (unsigned atom = 0; atom < (unrestricted ? 3U : 2U); ++atom) {
    system.atoms.push_back({1, {0.15 * atom * atom, 0.13 * atom, 1.5 * atom}});
    system.shells.push_back(
        {atom,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}});
  }
  if (ecp) {
    system.atoms[0].atomic_number = 11;
    system.atoms[0].ecp_core = 10;
    system.shells.insert(system.shells.begin() + 1, {0, 1, {{0.45, 1.0}}});
    system.ecp_terms = {{0, -1, 0, 0.9, 0.25}, {0, -1, 2, 0.4, -0.1}, {0, 0, 1, 0.8, 0.3},
                        {0, 1, 2, 0.65, -0.4}, {0, 2, 3, 0.7, 0.12},  {0, 3, 4, 0.55, -0.08}};
  }
  if (compact_fixture) {
    // Single radial primitives keep the complete high-angular molecular response
    // gate bounded; they do not replace the independent auxiliary topology.
    const std::array<std::array<double, 3>, 3> centers{
        {{{0.13, -0.21, 0.17}}, {{0.43, 0.19, 2.1}}, {{-0.3, 0.4, -1.6}}}};
    system.shells.clear();
    for (unsigned atom = 0; atom < system.atoms.size(); ++atom) {
      system.atoms[atom].position = centers[atom];
      system.shells.push_back({atom, 0, {{atom == 0 ? 0.7 : 0.9, 1.0}}});
    }
    if (ecp) system.ecp_terms = {{0, -1, 2, 0.8, -0.4}, {0, 0, 2, 0.8, 0.3}};
  }
  if (fitted_xc_warmup) {
    // Exercise the deployed mixed-XC shape gate (at least 24 orbitals),
    // without making the independent fitted metric or AO fixture large.
    const auto seeds = std::exchange(system.shells, std::vector<core::Shell>{});
    for (const auto& shell : seeds) {
      system.shells.push_back(shell);
      system.shells.push_back({shell.atom_index, 0, {{0.28, 1.0}}});
      system.shells.push_back({shell.atom_index, 1, {{0.38, 1.0}}});
      system.shells.push_back({shell.atom_index, 1, {{1.3, 1.0}}});
      system.shells.push_back({shell.atom_index, 2, {{0.72, 1.0}}});
    }
  }
  if (orbital_angular)
    system.shells.insert(system.shells.begin() + 1, {0, orbital_angular, {{0.48, 1.0}}});
  if (benchmark_h2) system.atoms[0].position = {0.03, -0.02, 0.01};
  if (benchmark_water) {
    // Use the CLI's canonical STO-3G decimals, not a rounded vendor basis alias.
    system.atoms = {{8, {0.0, 0.0, 0.0}},
                    {1, {1.43233673, 0.0, 1.10715266}},
                    {1, {-1.43233673, 0.0, 1.10715266}}};
    system.shells = {
        {0,
         0,
         {{130.7093214, 0.1543289673}, {23.80886605, 0.5353281423}, {6.443608313, 0.4446345422}}},
        {0,
         0,
         {{5.033151319, -0.09996722919}, {1.169596125, 0.3995128261}, {0.38038896, 0.7001154689}}},
        {0,
         1,
         {{5.033151319, 0.155916275}, {1.169596125, 0.6076837186}, {0.38038896, 0.3919573931}}},
        {1,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
        {2,
         0,
         {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}}};
  }
  // Use an exactly representable separation at the cutoff, not a near-boundary
  // comparison whose result depends on host/device rounding.
  if (d3_hard_cutoff) system.atoms[1].position = {0, 0, 1.5};
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  std::optional<core::System> auxiliary;
  if (separate_auxiliary) {
    auxiliary = system;
    auxiliary->ecp_terms.clear();
    for (auto& atom : auxiliary->atoms) atom.ecp_core = 0;
    auxiliary->shells.clear();
    for (unsigned atom = 0; atom < system.atoms.size(); ++atom)
      auxiliary->shells.push_back({atom, 0, {{atom == 0 ? 0.8 : 0.65, 1.0}}});
    if (auxiliary_d || auxiliary_g)
      auxiliary->shells.insert(auxiliary->shells.begin() + 1,
                               {0, auxiliary_g ? 4U : 2U, {{0.6, 1.0}}});
    require(molecule::validate_and_normalize(*auxiliary, detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
  }
  const std::array<generativeqc_ks_semilocal_component, 2> components{{{lda      ? "LDA_X"
                                                                        : pbe    ? "GGA_X_PBE"
                                                                        : r2scan ? "MGGA_X_R2SCAN"
                                                                                 : "MGGA_X_WB97M_V",
                                                                        pbe0 ? 0.75 : 1.0},
                                                                       {lda      ? "LDA_C_PW"
                                                                        : pbe    ? "GGA_C_PBE"
                                                                        : r2scan ? "MGGA_C_R2SCAN"
                                                                                 : "MGGA_C_WB97M_V",
                                                                        1.0}}};
  const double occupation = unrestricted ? 1 : 2;
  const double fraction = b3lyp ? 0.20 : 0.25;
  const std::array<generativeqc_ks_semilocal_component, 4> b3_components{
      {{"LDA_X", 0.08}, {"GGA_X_B88", 0.72}, {"LDA_C_VWN_RPA", 0.19}, {"GGA_C_LYP", 0.81}}};
  const generativeqc_ks_exchange_term full_exchange{GENERATIVEQC_KS_EXCHANGE_FULL_RANGE, fraction,
                                                    0.0, -fraction / occupation};
  const std::array<generativeqc_ks_exchange_term, 2> exchange{
      {{GENERATIVEQC_KS_EXCHANGE_SHORT_RANGE, 0.15, 0.3, -0.15 / occupation},
       {GENERATIVEQC_KS_EXCHANGE_LONG_RANGE, 1.0, 0.3, -1.0 / occupation}}};
  generativeqc_ks_options ks{};
  ks.struct_size = sizeof(ks);
  ks.abi_version = GENERATIVEQC_ABI_VERSION;
  ks.scf_domain = dft::semilocal_family_scf_domain(family);
  ks.grid_version = 1;
  ks.radial_points = pbe_d4 ? 16 : compact_fixture || benchmark_h2 || benchmark_water ? 8 : 12;
  ks.angular_polar = pbe_d4 ? 8 : 4;
  ks.angular_azimuth = pbe_d4 ? 16 : 8;
  if (split_treutler) {
    ks.grid_version = 4;
    ks.radial_points = 1;  // v4 radial field selects the pruned level.
    ks.angular_polar = ks.angular_azimuth = 0;
  }
  if (fitted_xc_warmup) {
    ks.grid_version = 4;
    ks.radial_points = 3;
    ks.angular_polar = ks.angular_azimuth = 0;
    ks.tile_points = 0;
  }
  ks.partition_iterations = 3;
  ks.coincident_tolerance = 1e-12;
  ks.tile_points = 64;
  ks.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
  ks.spin_channels = unrestricted ? 2 : 1;
  ks.semilocal_components = b3lyp ? b3_components.data() : components.data();
  ks.semilocal_component_count = b3lyp ? b3_components.size() : components.size();
  if (pbe0 || b3lyp) {
    ks.exchange_terms = &full_exchange;
    ks.exchange_term_count = 1;
  } else if (revised) {
    ks.has_nonlocal_correlation = 1;
    ks.nonlocal_variant = GENERATIVEQC_NONLOCAL_RVV10;
    ks.nonlocal_b = 11.95;
    ks.nonlocal_c = 0.0093;
    ks.nonlocal_coefficient = 1;
    ks.nonlocal_maximum_bytes = 1 << 24;
  } else if (!semilocal_grid) {
    ks.semilocal_range_omega = 0.3;
    ks.exchange_terms = exchange.data();
    ks.exchange_term_count = exchange.size();
    ks.has_nonlocal_correlation = 1;
    ks.nonlocal_variant = GENERATIVEQC_NONLOCAL_VV10;
    ks.nonlocal_b = 6;
    ks.nonlocal_c = 0.01;
    ks.nonlocal_coefficient = 1;
    ks.nonlocal_maximum_bytes = 1 << 24;
  }
  generativeqc_method_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.method =
      lda      ? (unrestricted ? GENERATIVEQC_METHOD_LDA_UKS : GENERATIVEQC_METHOD_LDA_RKS)
      : r2scan ? (unrestricted ? GENERATIVEQC_METHOD_R2SCAN_UKS : GENERATIVEQC_METHOD_R2SCAN_RKS)
               : (unrestricted ? GENERATIVEQC_METHOD_PBE_UKS : GENERATIVEQC_METHOD_PBE_RKS);
  if (pbe_d4) descriptor.method = GENERATIVEQC_METHOD_PBE_D4_RKS;
  if (split_hybrid) {
    const auto* entry =
        methods::named_ks::find_method(m062x ? (unrestricted ? "m06-2x-uks" : "m06-2x-rks")
                                             : (unrestricted ? "mn15-uks" : "mn15-rks"));
    require(entry && entry->cuda, "split hybrid is absent from the CUDA catalog");
    methods::named_ks::electronic_options(*entry, ks);
    descriptor.method = entry->carrier;
  }
  descriptor.max_iterations = 250;
  descriptor.diis_history = 8;
  descriptor.energy_tolerance = 1e-13;
  descriptor.density_tolerance = 1e-11;
  descriptor.screening_tolerance = 1e-14;
  descriptor.precision_mode =
      auto_precision ? GENERATIVEQC_PRECISION_AUTO : GENERATIVEQC_PRECISION_FP64;
  descriptor.density_fitting_mode =
      fitted ? GENERATIVEQC_DENSITY_FITTING_CUDA : GENERATIVEQC_DENSITY_FITTING_NONE;
  // H2 and H3 have different full/erf gaps. These cutoffs isolate the LR
  // branch, truncate both branches, or sit within 10% of a retained LR root.
  const double cutoff = near_metric_cutoff                    ? (unrestricted ? 0.0017 : 0.0193)
                        : truncated_range                     ? (unrestricted ? 0.005 : 0.03)
                        : truncated_both || truncated_primary ? (unrestricted ? 0.026 : 0.08)
                                                              : 1e-10;
  descriptor.density_fitting_relative_threshold = cutoff;
  descriptor.ks_options = &ks;
  generativeqc_ks_options vv10{};
  generativeqc_ks_nonlocal_grid_options split{};
  if (split_grid) {
    vv10 = ks;
    vv10.radial_points = split_treutler ? 0 : 6;
    vv10.angular_polar = split_treutler ? 0 : 3;
    vv10.angular_azimuth = split_treutler ? 0 : 6;
    split.base.base = ks;
    split.base.base.struct_size = sizeof(split);
    split.version = 1;
    split.grid = &vv10;
    descriptor.ks_options = &split.base.base;
  }
  core::ContextState context;
  context.requested_backend = context.executed_backend = GENERATIVEQC_BACKEND_CUDA;
  std::unique_ptr<generativeqc_context, decltype(&generativeqc_context_destroy)> named_context(
      nullptr, generativeqc_context_destroy);
  std::unique_ptr<generativeqc_batch, decltype(&generativeqc_batch_destroy)> named_batch(
      nullptr, generativeqc_batch_destroy);
  std::unique_ptr<methods::PreparedBatch> bare_batch;
  std::shared_ptr<runtime::DeviceResourceLedger> ledger;
  std::optional<runtime::DeviceResourceScope> preparation_scope;
  if (resource_ledger) {
    require(!composite, "ledger fixture requires the unwrapped native batch");
    ledger = std::make_shared<runtime::DeviceResourceLedger>();
    check(cudaGetDevice(&ledger->device));
    ledger->limit = 2ULL << 30;
    preparation_scope.emplace(ledger);
  }
  constexpr auto flags = GENERATIVEQC_BATCH_ENABLE_WARM_STARTS;
  if (composite) {
    // Use the same catalog, basis and dispersion admission as a public named batch.
    const generativeqc_context_descriptor context_input{
        sizeof(context_input), GENERATIVEQC_ABI_VERSION, 0, GENERATIVEQC_BACKEND_CUDA};
    generativeqc_context* raw_context{};
    require(
        generativeqc_context_create(&context_input, &raw_context) == GENERATIVEQC_STATUS_SUCCESS,
        "failed to create CUDA context for r2SCAN-3c");
    named_context.reset(raw_context);
    const generativeqc_named_ks_descriptor named{sizeof(named), GENERATIVEQC_ABI_VERSION,
                                                 unrestricted ? "r2scan-3c-uks" : "r2scan-3c-rks",
                                                 nullptr, 128U << 20};
    std::vector<generativeqc_atom> atoms;
    for (const auto& atom : system.atoms)
      atoms.push_back({atom.atomic_number, atom.position[0], atom.position[1], atom.position[2]});
    generativeqc_system_descriptor input{};
    input.struct_size = sizeof(input);
    input.abi_version = GENERATIVEQC_ABI_VERSION;
    input.atoms = atoms.data();
    input.atom_count = static_cast<std::uint32_t>(atoms.size());
    input.multiplicity = system.multiplicity;
    generativeqc_system* bound_system{};
    auto status =
        generativeqc_system_create_named_ks_v1(raw_context, &input, &named, &bound_system);
    require(status == GENERATIVEQC_STATUS_SUCCESS, raw_context->last_detail);
    const std::unique_ptr<generativeqc_system, decltype(&generativeqc_system_destroy)> owned_system(
        bound_system, generativeqc_system_destroy);
    system = bound_system->data;
    const generativeqc_system* inputs[] = {bound_system};
    descriptor.method = unrestricted ? GENERATIVEQC_METHOD_PBE_UKS : GENERATIVEQC_METHOD_PBE_RKS;
    generativeqc_batch* prepared_batch{};
    status = generativeqc_batch_prepare_named_ks_v1(raw_context, inputs, 1, &descriptor, &named,
                                                    flags, &prepared_batch);
    require(status == GENERATIVEQC_STATUS_SUCCESS, raw_context->last_detail);
    named_batch.reset(prepared_batch);
  } else {
    const methods::named_ks::Dispersion hard_cutoff{
        .kind = 1, .s6 = 1, .s8 = 0.7875, .a1 = 0.4289, .a2 = 4.4407, .pair_cutoff = 1.5};
    bare_batch = methods::prepare_batch(
        context, {system}, descriptor, flags, auxiliary ? &*auxiliary : nullptr,
        d3_hard_cutoff ? &hard_cutoff : nullptr, d3_hard_cutoff ? 128U << 20 : 0);
  }
  auto* batch = composite ? named_batch->plan.get() : bare_batch.get();
  const auto solve = [&](bool forces = false) {
    auto results = batch->execute(methods::Coordinates(1), forces);
    require(results.size() == 1, "expected one native molecular KS result");
    require(results[0].status == GENERATIVEQC_STATUS_SUCCESS &&
                results[0].calculation.convergence.converged,
            "native molecular KS SCF failed: " + results[0].detail);
    return std::move(results[0].calculation);
  };
  const auto calculation = solve(composite);
  const auto energy = calculation.energy;
  dft::CudaKsFinalStateToken token;
  require(methods::detail::dft_final_state_token(*batch, 0, token, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  require(token.strict_physical_fock, "no strict physical SCF Fock in accepted generation");
  if (split_hybrid && ecp)
    require((batch->supported_properties(0).value_or(0) & GENERATIVEQC_PROPERTY_FORCES) == 0,
            "split-hybrid ECP must not advertise a native standalone force");
  if (auto_precision && (pbe0 || b3lyp || !semilocal_grid || fitted_xc_warmup)) {
    const auto& precision = calculation.precision;
    require(precision.mixed_stage_fock_builds > 0 && precision.strict_refinement_applied,
            "AUTO did not execute a mixed proposal and strict refinement");
  }
  std::vector<double> gradient;
  if (composite) {
    require(calculation.dispersion && calculation.dispersion->composite &&
                calculation.forces.size() == 3 * system.atoms.size(),
            "missing complete force and correction diagnostics");
    gradient.reserve(calculation.forces.size());
    for (double force : calculation.forces) gradient.push_back(-force);
  } else if (r2scan && !revised) {
    dft::CudaStationaryResources resources;
    const auto status = methods::detail::dft_cuda_native_stationary_gradient(
        *batch, 0, token, 128U << 20, gradient, resources, detail);
    require(status == GENERATIVEQC_STATUS_SUCCESS, detail);
  }
  if (borrow_force_centers) {
    dft::CudaKsGridReference before_force(*batch, 0, token);
    require(before_force.prepare(detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
    require(!before_force.geometry_reference().center_pairs,
            "cold Hessian must not require a force allocation");
    const auto prepare_force = [&] {
      dft::CudaStationaryResources resources;
      require(methods::detail::dft_cuda_native_stationary_gradient(*batch, 0, token, 128U << 20,
                                                                   gradient, resources, detail) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              detail);
      require(resources.cached_geometry, "force fixture did not retain its center table");
    };
    prepare_force();
    dft::CudaKsGridReference first_force(*batch, 0, token);
    require(first_force.prepare(detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
    const auto first_geometry = first_force.geometry_reference();
    require(first_geometry.center_pairs, "Hessian did not borrow the live force table");
    prepare_force();
    require(first_geometry.authenticate(first_geometry.owner) == cudaErrorInvalidResourceHandle,
            "borrow survived a force-table reset at the same KS token");
  }
  dft::CudaKsGridReference reference(*batch, 0, token);
  require(reference.prepare(detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  if (borrow_force_centers)
    require(reference.geometry_reference().center_pairs,
            "Hessian did not reborrow the new force-table publication");
  const auto shape = reference.shape();
  const auto lease = reference.reference();
  dft::CudaKsGridReference vv10_reference(*batch, 0, token, dft::CudaKsGridRole::separate_nonlocal);
  if (split_grid) {
    require(vv10_reference.prepare(detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
    require(vv10_reference.shape().points != shape.points &&
                vv10_reference.reference().density == lease.density,
            "VV10 reference did not borrow the independent grid and accepted density");
  }
  const auto coordinates = 3 * shape.atoms, panels = shape.spins * shape.nao * shape.nao;
  dft::CudaKsResidentFockBinding fock;
  dft::CudaKsResidentOrbitalFrameBinding frame;
  require(methods::detail::dft_cuda_resident_fock_sources(*batch, 0, token, fock, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  require(methods::detail::dft_cuda_resident_orbital_frame(*batch, 0, token, frame, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  if (resource_ledger) {
    preparation_scope.reset();
    generativeqc_context public_context;
    generativeqc_batch public_batch;
    public_batch.context = &public_context;
    public_batch.resource_ledger = ledger;
    public_batch.plan = std::move(bare_batch);
    public_batch.atom_counts = {static_cast<std::uint32_t>(shape.atoms)};
    std::vector<double> motion(coordinates), output(full ? coordinates * coordinates : coordinates);
    for (std::size_t index = 0; index < coordinates; ++index) motion[index] = std::cos(index + 0.7);
    const auto retained = ledger->live;
    const auto limit = ledger->limit;
    require(retained > 0 && retained < limit, "SCF state did not enter the same device ledger");
    generativeqc_ks_hessian_options_v2 controls{};
    require(generativeqc_ks_hessian_default_options_v2(&controls) == GENERATIVEQC_STATUS_SUCCESS,
            "could not initialize CUDA Hessian controls");
    controls.absolute_tolerance = 2e-12;
    controls.relative_tolerance = 2e-11;
    generativeqc_ks_hessian_receipt_v2 receipt{};
    receipt.struct_size = sizeof(receipt);
    receipt.abi_version = GENERATIVEQC_ABI_VERSION;
    const auto execute = [&] {
      return generativeqc_ks_hessian_cuda_v2(
          &public_batch, 0, &controls, full ? nullptr : motion.data(), full ? 0 : motion.size(),
          output.data(), output.size(), &receipt);
    };
    std::fill(output.begin(), output.end(), 37.0);
    receipt.completed_directions = 41;
    const auto unpublished = [&] {
      return receipt.completed_directions == 41 && ledger->live == retained &&
             std::all_of(output.begin(), output.end(), [](double value) { return value == 37; });
    };
    const auto output_capacity = controls.output_bytes;
    const auto preparation_peak = ledger->peak;
    controls.output_bytes = 1;
    require(execute() == GENERATIVEQC_STATUS_INVALID_ARGUMENT && unpublished() &&
                ledger->peak == preparation_peak,
            "short output capacity must fail before allocating device response storage");
    controls.output_bytes = output_capacity;
    ledger->limit = retained;
    require(execute() == GENERATIVEQC_STATUS_OUT_OF_MEMORY && unpublished(),
            "resource rejection failed to preserve the output, receipt or SCF allocations");
    ledger->limit = limit;
    if (benchmark_water && (lda || pbe)) {
      auto stale_epoch = token, stale_grid = token, stale_provider = token;
      ++stale_epoch.identity.determinant.solve_epoch;
      ++stale_grid.identity.model.grid.partition_iterations;
      stale_provider.identity.determinant.model.screening_tolerance *= 0.5;
      using Failure = std::pair<const dft::CudaKsFinalStateToken*, const char*>;
      for (const auto& [stale, source] :
           {Failure{&stale_epoch, "solve epoch"}, Failure{&stale_grid, "grid prescription"},
            Failure{&stale_provider, "integral provider"}}) {
        dft::CudaKsWb97mvHessian rejected;
        dft::CudaKsWb97mvHessianOptions policy;
        const auto mode = full ? dft::CudaKsWb97mvHessianMode::dense_and_action
                               : dft::CudaKsWb97mvHessianMode::action_only;
        require(rejected.prepare(*public_batch.plan, 0, *stale, mode, policy, detail) ==
                        GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                    rejected.work().device_bytes == 0 && unpublished(),
                std::string("Hessian accepted or leaked storage on stale ") + source);
      }
      // Use a real converged CUDA determinant with the explicit host XC
      // schedule. Its owner never allocated resident XC quadrature; a stale
      // token or a synthetic null pointer would test a different boundary.
      {
        auto host_xc = ks;
        host_xc.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_HOST_UNFUSED;
        auto host_xc_descriptor = descriptor;
        host_xc_descriptor.ks_options = &host_xc;
        generativeqc_context missing_context;
        generativeqc_batch missing_batch;
        missing_batch.context = &missing_context;
        missing_batch.resource_ledger = ledger;
        missing_batch.atom_counts = public_batch.atom_counts;
        {
          runtime::DeviceResourceScope scope(ledger);
          missing_batch.plan =
              methods::prepare_batch(context, {system}, host_xc_descriptor, flags, nullptr);
          const auto solved = missing_batch.plan->execute(methods::Coordinates(1), false);
          require(solved.size() == 1 && solved[0].status == GENERATIVEQC_STATUS_SUCCESS &&
                      solved[0].calculation.convergence.converged,
                  "host-unfused CUDA SCF did not converge");
        }
        dft::CudaKsFinalStateToken missing_token;
        require(methods::detail::dft_final_state_token(*missing_batch.plan, 0, missing_token,
                                                       detail) == GENERATIVEQC_STATUS_SUCCESS &&
                    missing_token.strict_physical_fock,
                "missing-grid fixture needs an accepted strict CUDA token: " + detail);
        const auto missing_live = ledger->live;
        const auto missing_status = generativeqc_ks_hessian_cuda_v2(
            &missing_batch, 0, &controls, full ? nullptr : motion.data(), full ? 0 : motion.size(),
            output.data(), output.size(), &receipt);
        require(
            missing_status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                missing_context.last_detail.find("device-fused CUDA XC owner") !=
                    std::string::npos &&
                ledger->live == missing_live,
            "missing resident XC primitive did not fail closed: " + missing_context.last_detail);
      }
      require(unpublished(), "missing resident XC primitive published output or leaked storage");
      const auto convergent = controls;
      controls.max_iterations = controls.restart = 1;
      controls.absolute_tolerance = controls.relative_tolerance = 1e-30;
      require(execute() == GENERATIVEQC_STATUS_NOT_CONVERGED && unpublished(),
              "truncated CPKS solve published output or leaked device storage");
      controls = convergent;
    }
    require(execute() == GENERATIVEQC_STATUS_SUCCESS, public_context.last_detail);
    require(receipt.completed_directions == (full ? coordinates : 1) && receipt.device_bytes &&
                ledger->peak >= retained + receipt.device_bytes && ledger->live == retained &&
                ledger->peak <= limit &&
                std::all_of(output.begin(), output.end(),
                            [](double value) { return std::isfinite(value); }),
            "response receipt not covered by the shared ledger or temporary allocations leaked: " +
                std::to_string(ledger->peak) + " vs " +
                std::to_string(retained + receipt.device_bytes));
    public_batch.plan.reset();
    require(ledger->live == 0, "batch destruction did not release its retained allocation charges");
    std::cout << "{\"response_bytes\":" << receipt.device_bytes
              << ",\"scf_retained_bytes\":" << retained
              << ",\"aggregate_peak_bytes\":" << ledger->peak << "}\n";
    return;
  }
  Storage storage;
  auto* direction = storage.allocate(coordinates);
  auto* output = storage.allocate(coordinates);
  auto* matrix = storage.allocate(coordinates * coordinates);
  std::vector<double> motion(coordinates), sentinel(coordinates, 27.0);
  for (std::size_t index = 0; index < coordinates; ++index)
    motion[index] = 0.12 * std::cos(index + 0.7);
  if (separate_auxiliary && unrestricted)
    motion = {0.02, -0.03, 0.1, -0.04, 0.01, -0.02, 0.02, 0.02, -0.08};
  check(cudaMemcpyAsync(direction, motion.data(), coordinates * sizeof(double),
                        cudaMemcpyHostToDevice, lease.stream));
  check(cudaMemcpyAsync(output, sentinel.data(), coordinates * sizeof(double),
                        cudaMemcpyHostToDevice, lease.stream));
  dft::CudaKsWb97mvHessian endpoint;
  using Mode = dft::CudaKsWb97mvHessianMode;
  const auto preparation = (full || d3_hard_cutoff) ? Mode::dense_and_action : Mode::action_only;
  dft::CudaKsWb97mvHessianOptions options;
  if (mode.find("-fp32-cpks") != std::string::npos)
    options.cpks.resident_precision = response::CudaCpksResidentPrecision::fp32_proposal;
  if (semilocal_grid)
    options.grid.max_pair_evaluations = options.induced.grid.max_pair_evaluations = 0;
  if (mode.find("directional-exact") != std::string::npos) options.max_frozen_exact_bytes = 0;
  if (mode.find("physical-cpks") != std::string::npos) options.max_resident_cpks_bytes = 0;
  if (mode.find("serial-one-electron") != std::string::npos) {
    options.one_electron.parallel_streams = 0;
    options.max_frozen_core_bytes = 0;
  }
  if (mode.find("serial-grid") != std::string::npos) options.grid.replay_nuclear_maps = false;
  if (mode.find("batch-chunked") != std::string::npos)
    options.exact.max_batch_projection_bytes = fitted ? 32ULL << 10 : 512;
  if (mode.find("unbatched") != std::string::npos) {
    options.exact.max_batch_workspace_bytes = 0;
    options.induced.max_resident_integral_bytes = 0;
    options.grid.pair_tile_points = options.induced.grid.pair_tile_points = 32;
    options.geometry.point_tile = 64;
  }
  if (mode.find("streamed") != std::string::npos) options.exact.max_resident_integral_bytes = 0;
  options.cpks.gmres.absolute_tolerance = 2e-12;
  options.cpks.gmres.relative_tolerance = 2e-11;
  auto insufficient = options;
  insufficient.max_device_bytes = 1;
  const auto capacity_status =
      endpoint.prepare(*batch, 0, token, preparation, insufficient, detail);
  dft::CudaKsWb97mvHessianResult insufficient_result;
  require(capacity_status == GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
              endpoint.apply(direction, output, lease.stream, insufficient_result) ==
                  cudaErrorInvalidResourceHandle &&
              download(output, coordinates) == sentinel,
          "stationary endpoint accepted an insufficient budget or wrote output: " + detail);
  if (auto_precision) {
    auto unproved = token;
    unproved.strict_physical_fock = false;
    dft::CudaKsWb97mvHessianResult rejected;
    require(endpoint.prepare(*batch, 0, unproved, preparation, options, detail) ==
                    GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                endpoint.apply(direction, output, lease.stream, rejected) ==
                    cudaErrorInvalidResourceHandle &&
                download(output, coordinates) == sentinel,
            "unproved SCF precision was accepted or wrote output");
    if (fitted_xc_warmup) {
      auto relaxed = token;
      relaxed.relaxed_stationarity = true;
      detail.clear();
      require(endpoint.prepare(*batch, 0, relaxed, preparation, options, detail) ==
                      GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                  detail.find("relaxed application stationarity") != std::string::npos &&
                  endpoint.apply(direction, output, lease.stream, rejected) ==
                      cudaErrorInvalidResourceHandle &&
                  download(output, coordinates) == sentinel,
              "relaxed SCF state was accepted or wrote output");
    }
  }
  if (revised) {
    auto short_pairs = options;
    short_pairs.grid.max_pair_evaluations = short_pairs.induced.grid.max_pair_evaluations = 1;
    const auto status = endpoint.prepare(*batch, 0, token, preparation, short_pairs, detail);
    dft::CudaKsWb97mvHessianResult rejected;
    require(status != GENERATIVEQC_STATUS_SUCCESS &&
                endpoint.apply(direction, output, lease.stream, rejected) ==
                    cudaErrorInvalidResourceHandle &&
                download(output, coordinates) == sentinel,
            "rVV10 pair-cap rejection changed caller storage");
  }
  if (split_budget) {
    // Both grids fit on their own, but cannot borrow each other's live AO
    // tiles. One byte below the combined minimum must leave no usable owner.
    std::size_t primary_bytes{}, vv10_bytes{}, primary_min{}, vv10_min{};
    check(dft::cuda_grid_geometry_workspace(reference.geometry_shape(), options.geometry,
                                            primary_bytes, &primary_min));
    check(dft::cuda_grid_geometry_workspace(vv10_reference.geometry_shape(), options.geometry,
                                            vv10_bytes, &vv10_min));
    auto short_grid = options;
    short_grid.geometry.max_workspace_bytes = primary_min + vv10_min - 1;
    require(endpoint.prepare(*batch, 0, token, preparation, short_grid, detail) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                detail.find("moving-grid workspace admission") != std::string::npos,
            "dual-grid owner ignored combined geometry budget: " + detail);
    dft::CudaKsWb97mvHessianResult rejected;
    require(endpoint.apply(direction, output, lease.stream, rejected) ==
                    cudaErrorInvalidResourceHandle &&
                download(output, coordinates) == sentinel,
            "failed dual-grid preparation changed caller storage");
    options.geometry.max_workspace_bytes = primary_min + vv10_min;
  }
  const auto prepared = endpoint.prepare(*batch, 0, token, preparation, options, detail);
  if (near_metric_cutoff) {
    dft::CudaKsWb97mvHessianResult rejected;
    const std::vector<double> untouched(coordinates * coordinates, 31.0);
    check(cudaMemcpyAsync(matrix, untouched.data(), untouched.size() * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    require(prepared == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                detail.find("metric cutoff separation") != std::string::npos,
            "near-cutoff fitted response was not correctly rejected: " + detail);
    require(
        endpoint.apply(direction, output, lease.stream, rejected) ==
                cudaErrorInvalidResourceHandle &&
            download(output, coordinates) == sentinel &&
            endpoint.hessian(matrix, lease.stream, rejected) == cudaErrorInvalidResourceHandle &&
            download(matrix, untouched.size()) == untouched && !endpoint.work().published_hvps &&
            !endpoint.work().published_hessians,
        "rejected metric branch published a derivative");
    std::cout << "{\"metric_cutoff_preserves_output\":true}\n";
    return;
  }
  require(prepared == GENERATIVEQC_STATUS_SUCCESS, detail);
  require(bool(fock.fitted_range) == fitted_range, "unexpected accepted LR provider");
  dft::CudaKsWb97mvHessianResult result;
  if (preparation == Mode::action_only) {
    const std::vector<double> untouched(coordinates * coordinates, 31.0);
    check(cudaMemcpyAsync(matrix, untouched.data(), untouched.size() * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    require(endpoint.hessian(matrix, lease.stream, result) == cudaErrorInvalidResourceHandle &&
                !result.converged() && !result.completed_directions &&
                !endpoint.work().attempted_directions &&
                download(matrix, untouched.size()) == untouched,
            "action-only owner performed dense work or changed the caller output");
    const auto work = endpoint.work();
    require(!work.cpks_operator_bytes && !work.frozen_exact_bytes && !work.frozen_core_bytes,
            "action-only owner reserved repeated-RHS storage");
  }
  require(endpoint.apply(direction, direction, lease.stream, result) != cudaSuccess,
          "stationary endpoint admitted input/output alias");
  require(endpoint.apply(direction, const_cast<double*>(lease.density), lease.stream, result) !=
              cudaSuccess,
          "stationary endpoint admitted immutable density output");
  if (split_grid) {
    const auto vv10_lease = vv10_reference.reference();
    const auto before = download(vv10_lease.points, coordinates);
    require(endpoint.apply(direction, const_cast<double*>(vv10_lease.points), lease.stream,
                           result) == cudaErrorInvalidValue &&
                download(vv10_lease.points, coordinates) == before,
            "stationary endpoint wrote the independent quadrature");
  }
  if (split_treutler) {
    for (auto* grid : {&reference, &vv10_reference}) {
      const auto quad = grid->geometry_reference();
      require(quad.atomic_radii && quad.atom_offsets, "missing native Treutler topology");
      const auto radii = download(quad.atomic_radii, shape.atoms);
      const auto offsets = download(quad.atom_offsets, shape.atoms + 1);
      auto* radii_output = const_cast<double*>(quad.atomic_radii);
      auto* offsets_output =
          reinterpret_cast<double*>(const_cast<std::uint64_t*>(quad.atom_offsets));
      require(
          endpoint.apply(direction, radii_output, lease.stream, result) == cudaErrorInvalidValue &&
              endpoint.apply(direction, offsets_output, lease.stream, result) ==
                  cudaErrorInvalidValue &&
              download(quad.atomic_radii, shape.atoms) == radii &&
              download(quad.atom_offsets, shape.atoms + 1) == offsets,
          "stationary endpoint wrote the retained Treutler topology");
    }
  }
  const double nonfinite = std::numeric_limits<double>::infinity();
  check(
      cudaMemcpyAsync(direction, &nonfinite, sizeof(double), cudaMemcpyHostToDevice, lease.stream));
  require(endpoint.apply(direction, output, lease.stream, result) != cudaSuccess &&
              download(output, coordinates) == sentinel,
          "nonfinite direction changed output");
  check(cudaMemcpyAsync(direction, motion.data(), coordinates * sizeof(double),
                        cudaMemcpyHostToDevice, lease.stream));
  if (d3_hard_cutoff) {
    require(calculation.dispersion && calculation.dispersion->two_body_energy != 0,
            "primal D3 plan did not evaluate the cutoff pair");
    const auto apply_status = endpoint.apply(direction, output, lease.stream, result);
    require((apply_status != cudaSuccess || !result.converged()) && result.response.converged() &&
                download(output, coordinates) == sentinel,
            "undefined cutoff HVP must fail after CPKS without changing the caller output");
    const std::vector<double> matrix_sentinel(coordinates * coordinates, 31.0);
    auto* matrix = storage.allocate(matrix_sentinel.size());
    check(cudaMemcpyAsync(matrix, matrix_sentinel.data(), matrix_sentinel.size() * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    const auto matrix_status = endpoint.hessian(matrix, lease.stream, result);
    require((matrix_status != cudaSuccess || !result.converged()) && result.response.converged() &&
                result.completed_directions < coordinates &&
                download(matrix, matrix_sentinel.size()) == matrix_sentinel,
            "undefined cutoff Hessian changed the caller output");
    const auto work = endpoint.work();
    require(!work.published_hvps && !work.published_hessians &&
                work.completed_directions == result.completed_directions,
            "undefined cutoff reported a published result");
    std::cout << "{\"hard_cutoff_preserves_output\":true}\n";
    return;
  }
  check(endpoint.apply(direction, output, lease.stream, result));
  require(result.converged() && result.response.converged(), "stationary CPKS did not converge");
  const auto hvp = download(output, coordinates);
  std::cout << std::setprecision(17) << "{\"atoms\":" << shape.atoms
            << ",\"primitives\":" << shape.primitives << ",\"nao\":" << shape.nao
            << ",\"spins\":" << shape.spins << ",\"point_count\":" << shape.points
            << ",\"orbital_angular\":" << orbital_angular
            << ",\"r2scan3c\":" << (composite ? "true" : "false")
            << ",\"benchmark_water\":" << (benchmark_water ? "true" : "false")
            << ",\"auxiliary_d\":" << (auxiliary_d ? "true" : "false")
            << ",\"auxiliary_g\":" << (auxiliary_g ? "true" : "false") << ",\"energy\":" << energy
            << ",\"fitted_xc_warmup\":" << (fitted_xc_warmup ? "true" : "false")
            << ",\"scf_mixed_fock_builds\":" << calculation.precision.mixed_stage_fock_builds
            << ",\"scf_refinement_iterations\":" << calculation.precision.refinement_iterations
            << ",\"response_residual\":" << result.response.gmres.residual_norm;
  array("basis", download(lease.basis, 3 * shape.atoms + 2 * shape.primitives + 16 * shape.nao));
  array("points", download(lease.points, 3 * shape.points));
  array("weights", download(lease.weights, shape.points));
  array("atomic", download(reference.geometry_reference().atomic_weights, shape.points));
  grid_topology(reference, "atom_offsets", "atomic_radii");
  if (split_grid) {
    const auto points = vv10_reference.shape().points;
    array("vv10_points", download(vv10_reference.reference().points, 3 * points));
    array("vv10_weights", download(vv10_reference.reference().weights, points));
    array("vv10_atomic", download(vv10_reference.geometry_reference().atomic_weights, points));
    grid_topology(vv10_reference, "vv10_atom_offsets", "vv10_atomic_radii");
  }
  array("density", download(lease.density, panels));
  array("hcore", download(fock.hcore, shape.nao * shape.nao));
  array("fock", download(fock.fock, panels));
  array("overlap", download(frame.overlap, shape.nao * shape.nao));
  if (fitted) {
    const auto source =
        scf::cuda_density_fitting_readonly_source(fock.primary->cuda_fitted_source());
    require(static_cast<bool>(source), "accepted DF source is not readable");
    require(source.naux == molecule::ao_count(auxiliary ? *auxiliary : system),
            "explicit native auxiliary basis was ignored");
    auto* raw = storage.allocate(shape.nao * shape.nao * source.naux);
    require(scf::enqueue_cuda_density_fitting_raw_values(source.integrals, 0, shape.nao * shape.nao,
                                                         lease.stream, raw,
                                                         detail) == GENERATIVEQC_STATUS_SUCCESS,
            detail);
    check(cudaStreamSynchronize(lease.stream));
    std::cout << ",\"naux\":" << source.naux;
    array("raw", download(raw, shape.nao * shape.nao * source.naux));
    array("whitening", download(source.whitening, source.naux * source.naux));
    std::cout << ",\"metric_relative_threshold\":" << cutoff;
    const auto report_rank = [&](const auto& provider, const char* name) {
      const auto metric = scf::cuda_density_fitting_readonly_metric(provider.cuda_fitted_source());
      require(metric && metric.relative_threshold == cutoff, "missing accepted DF spectrum");
      const auto values = download(metric.eigenvalues, metric.naux);
      std::size_t rank = 0;
      for (const auto value : values) rank += value > cutoff * values.back();
      require(metric.full_rank == (rank == metric.naux), "inconsistent DF metric branch");
      std::cout << ",\"" << name << "\":" << rank;
    };
    if (truncated_range || truncated_both || truncated_primary) {
      report_rank(*fock.primary, "metric_rank");
      if (fock.fitted_range) report_rank(*fock.fitted_range, "range_metric_rank");
    }
  }
  array("motion", motion);
  array("hvp", hvp);
  if (!gradient.empty()) array("gradient", gradient);
  if (composite || pbe_d4) {
    require(calculation.dispersion.has_value(), "missing primal D4 result");
    const auto& dispersion = *calculation.dispersion;
    std::cout << ",\"d4_energy\":" << dispersion.two_body_energy + dispersion.atm_energy;
    if (composite) std::cout << ",\"gcp_energy\":" << dispersion.composite->gcp_energy;
  }
  std::cout << ",\"resident_integral_bytes\":" << endpoint.work().exact.resident_integral_bytes;
  std::cout << ",\"batched_workspace_bytes\":" << endpoint.work().exact.batched_workspace_bytes;
  if (full) {
    const auto started = std::chrono::steady_clock::now();
    check(endpoint.hessian(matrix, lease.stream, result));
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started);
    require(result.converged() && result.completed_directions == coordinates,
            "complete Hessian was not accepted");
    array("hessian", download(matrix, coordinates * coordinates));
    std::cout << ",\"hessian_seconds\":" << seconds.count();
  }
  if (unrestricted) {
    // CUDA success is deliberately distinct from numerical convergence. A
    // bounded failed response must not publish even the first Hessian column.
    dft::CudaKsWb97mvHessian limited;
    auto short_solve = options;
    short_solve.cpks.gmres.max_iterations = short_solve.cpks.gmres.restart = 1;
    short_solve.cpks.gmres.absolute_tolerance = 1e-30;
    short_solve.cpks.gmres.relative_tolerance = 1e-30;
    require(limited.prepare(*batch, 0, token, Mode::dense_and_action, short_solve, detail) ==
                GENERATIVEQC_STATUS_SUCCESS,
            detail);
    const std::vector<double> untouched(coordinates * coordinates, 33.0);
    check(cudaMemcpyAsync(matrix, untouched.data(), untouched.size() * sizeof(double),
                          cudaMemcpyHostToDevice, lease.stream));
    check(limited.hessian(matrix, lease.stream, result));
    require(!result.converged() &&
                result.status == dft::CudaKsWb97mvHessianStatus::response_not_converged &&
                download(matrix, untouched.size()) == untouched,
            "nonconverged response published a partial Hessian");
  }
  const auto work = endpoint.work();
  if (split_grid) {
    require(work.grid.nuclear_response_pairs == 0 && work.induced.grid.reference_pairs == 0 &&
                work.separate_vv10.nuclear_response_pairs > 0 &&
                work.induced.separate_vv10.reference_pairs > 0,
            "only the independent VV10 grid may traverse ordered pairs");
    std::cout << ",\"vv10_grid_directions\":" << work.separate_vv10.nuclear_directions;
  }
  if (split_budget) {
    require(work.geometry.replayed_ao_primitive_term_bound &&
                work.vv10_geometry.replayed_ao_primitive_term_bound,
            "shared minimum did not exercise both bounded AO replay paths");
    std::cout << ",\"shared_geometry_budget\":true";
  }
  require(work.correction_directions == ((composite || pbe_d4) ? (full ? 1 + coordinates : 1) : 0),
          "unexpected correction work inventory");
  if (semilocal_grid)
    require(work.grid.reference_pairs == 0 && work.grid.nuclear_response_pairs == 0 &&
                work.induced.grid.reference_pairs == 0 && work.induced.grid.response_pairs == 0,
            "semilocal-grid KS response performed nonlocal pair work");
  std::cout << ",\"device_bytes\":" << work.device_bytes
            << ",\"cpks_operator_bytes\":" << work.cpks_operator_bytes
            << ",\"cpks_operator_columns\":" << work.cpks_operator_columns
            << ",\"cpks_cached_actions\":" << work.cpks_cached_actions
            << ",\"cpks_physical_checks\":" << work.cpks_physical_checks
            << ",\"cpks_proposal_seeds\":" << work.cpks_proposal_seeds
            << ",\"cpks_proposal_iterations\":" << work.cpks_proposal_iterations
            << ",\"cpks_strict_iterations\":" << work.cpks_strict_iterations
            << ",\"frozen_exact_bytes\":" << work.frozen_exact_bytes
            << ",\"frozen_exact_columns\":" << work.frozen_exact_columns
            << ",\"frozen_exact_contractions\":" << work.frozen_exact_contractions
            << ",\"frozen_core_bytes\":" << work.frozen_core_bytes
            << ",\"one_electron_components\":" << work.one_electron.integrals.component_evaluations
            << ",\"parallel_one_electron_bytes\":"
            << work.one_electron.integrals.parallel_workspace_bytes
            << ",\"parallel_one_electron_actions\":" << work.one_electron.integrals.parallel_actions
            << ",\"directions\":" << work.completed_directions
            << ",\"correction_directions\":" << work.correction_directions
            << ",\"induced_actions\":" << work.induced.actions
            << ",\"raw_df_value_passes\":" << work.raw_df.value_passes
            << ",\"raw_df_contraction_passes\":" << work.raw_df.contraction_passes
            << ",\"fitted_range\":" << (fitted_range ? "true" : "false")
            << ",\"raw_df_range_value_passes\":" << work.raw_df_range.value_passes
            << ",\"raw_df_range_contraction_passes\":" << work.raw_df_range.contraction_passes
            << ",\"raw_df_range_workspace_bytes\":" << work.raw_df_range.workspace_bytes
            << ",\"range_affine_terms\":" << work.fitted_range.affine_terms
            << ",\"exact_nuclear_components\":" << work.exact.component_evaluations
            << ",\"induced_direct_quartets\":"
            << work.induced.reference_quartets + work.induced.response_quartets
            << ",\"grid_right_directions\":" << work.grid.nuclear_directions
            << ",\"grid_mixed_contractions\":" << work.grid.nuclear_contractions;
  cudaStream_t other{};
  check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
  require(endpoint.apply(direction, output, other, result) != cudaSuccess,
          "foreign stream admitted");
  check(cudaStreamDestroy(other));
  check(cudaStreamBeginCapture(lease.stream, cudaStreamCaptureModeThreadLocal));
  require(
      endpoint.apply(direction, output, lease.stream, result) == cudaErrorStreamCaptureUnsupported,
      "stationary endpoint admitted capture");
  cudaGraph_t graph{};
  check(cudaStreamEndCapture(lease.stream, &graph));
  check(cudaGraphDestroy(graph));
  solve();
  require(
      endpoint.apply(direction, output, lease.stream, result) == cudaErrorInvalidResourceHandle &&
          download(output, coordinates) == hvp,
      "revoked endpoint changed published HVP");
  if (mode.find("-gradient-check") != std::string::npos) {
    require(fitted_range, "displaced gradient fixture requires fitted LR");
    constexpr double step = 0.002;
    for (const double sign : {-1.0, 1.0}) {
      std::vector<double> displaced(coordinates);
      for (std::size_t index = 0; index < coordinates; ++index)
        displaced[index] =
            system.atoms[index / 3].position[index % 3] + sign * step * motion[index];
      // A geometry update retires the producing stream; it cannot run on
      // the batch still borrowed by the central Hessian owner.
      auto displaced_batch = methods::prepare_batch(context, {system}, descriptor, flags,
                                                    auxiliary ? &*auxiliary : nullptr);
      const auto results =
          displaced_batch->execute(methods::Coordinates{std::move(displaced)}, false);
      require(results.size() == 1, "expected one displaced fitted LR KS result");
      require(results[0].status == GENERATIVEQC_STATUS_SUCCESS &&
                  results[0].calculation.convergence.converged,
              "displaced fitted LR KS solve failed: " + results[0].detail);
      dft::CudaKsFinalStateToken displaced_token;
      require(methods::detail::dft_final_state_token(*displaced_batch, 0, displaced_token,
                                                     detail) == GENERATIVEQC_STATUS_SUCCESS,
              detail);
      dft::CudaStationaryResources resources;
      std::vector<double> displaced_gradient;
      const auto status = methods::detail::dft_cuda_native_stationary_gradient(
          *displaced_batch, 0, displaced_token, 128U << 20, displaced_gradient, resources, detail);
      require(status == GENERATIVEQC_STATUS_SUCCESS, detail);
      array(sign < 0 ? "displaced_gradient_minus" : "displaced_gradient_plus", displaced_gradient);
    }
    std::cout << ",\"gradient_step\":" << step;
  }
  std::cout << ",\"lifecycle\":true}\n";
}
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 3 || argc == 4, "usage: probe rks|uks direct|df[-spherical][-ecp] [full]");
    require(std::string(argv[1]) == "rks" || std::string(argv[1]) == "uks", "invalid spin mode");
    run(std::string(argv[1]) == "uks", argv[2], argc == 4 && std::string(argv[3]) == "full");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
