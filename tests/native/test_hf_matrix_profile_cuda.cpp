#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "molecule/basis.hpp"
#include "runtime/lowering_binding.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda/rhf_bucket_internal.hpp"

namespace {
using namespace generativeqc;
using namespace scf;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct FixtureFile {
  std::array<char, 40> path{};
  FixtureFile() {
    std::snprintf(path.data(), path.size(), "/tmp/gqc-hf-matrix-XXXXXX");
    const int descriptor = mkstemp(path.data());
    require(descriptor >= 0, "could not create profile fixture");
    require(close(descriptor) == 0, "could not close profile fixture");
  }
  ~FixtureFile() {
    unsetenv("GENERATIVEQC_HF_MATRIX_PROFILE");
    std::remove(path.data());
  }
};

/** Synthetic ordering fixtures, never production calibration or timing claims.
 * All compatibility facts come from the real numerical owner on this device.
 */
void write_profile(const char* path, const cuda_policy::HfMatrixProfileContext& context,
                   bool library, bool stale = false, bool force_direct = false) {
  const auto& target = context.target;
  std::ofstream stream(path);
  stream << "generativeqc-hf-matrix-profile-v2\nsource_identity "
         << (stale ? std::string(64, '0') : std::string(context.source_identity.data()))
         << "\nprovider nvidia\nversions " << context.driver_version << ' '
         << context.runtime_version << ' ' << context.toolkit_version;
  for (const auto version : context.cublas_version) stream << ' ' << version;
  stream << "\nbuild_mode " << context.release_build << ' ' << context.fast_compile
         << "\nnative_build_id " << context.native_build_id.data() << "\nruntime_build_id "
         << context.runtime_build_id.data() << "\ndriver_build_id "
         << context.driver_build_id.data() << "\ncublas_build_id " << context.cublas_build_id.data()
         << "\ntarget " << target.compute_capability_major << ' ' << target.compute_capability_minor
         << ' ' << target.warp_size << ' ' << target.maximum_threads_per_block << ' '
         << target.maximum_threads_per_sm << ' ' << target.maximum_blocks_per_sm << ' '
         << target.registers_per_sm << ' ' << target.shared_memory_per_block << ' '
         << target.shared_memory_per_block_optin << ' ' << target.shared_memory_per_sm << ' '
         << target.multiprocessor_count << ' ' << target.total_global_memory << ' '
         << target.fp32_fp64_ratio << ' ' << target.clock_rate_khz << ' '
         << target.memory_clock_rate_khz << ' ' << target.memory_bus_width_bits
         << "\nbounds 1 32 1 100\nevidence_sha256 " << std::string(64, 'f')
         << (library ? "\nnative 1000000 1 1\ncublas 1 1000000 1000000"
                     : "\nnative 1 1000000 1000000\ncublas 1000000000 1 1")
         << "\nprovider_setup_nanoseconds 0\ndirect_jk "
         << (library        ? "0 1073741824 3 3"
             : force_direct ? "0 1073741824 7 6"
                            : "32 1073741824 7 6")
         << "\nend\n";
  require(static_cast<bool>(stream), "could not write profile fixture");
}

core::System system_for(bool unrestricted, bool polarized, bool include_f = false) {
  core::System system;
  system.atoms = {{2, {0., 0., -0.60}}, {1, {0., 0., 0.85}}};
  system.shells = {
      {0, 0, {{6.36242139, 0.15432897}, {1.15892300, 0.53532814}, {0.31364979, 0.44463454}}},
      {1, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}}};
  if (polarized) {
    system.shells.push_back({0, 1, {{0.8, 1.0}}});
    system.shells.push_back({0, 2, {{0.6, 1.0}}});
    system.shells.push_back({1, 1, {{0.4, 1.0}}});
    system.shells.push_back({1, 2, {{0.3, 1.0}}});
  }
  if (include_f) {
    system.shells.push_back({0, 3, {{0.5, 1.0}}});
    system.shells.push_back({1, 3, {{0.25, 1.0}}});
  }
  system.charge = unrestricted ? 0 : 1;
  system.multiplicity = unrestricted ? 2 : 1;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "profile endpoint normalization failed");
  return system;
}

/** Independent PySCF 2.14.0 RHF/UHF, spherical AOs, Bohr coordinates.
 * Polarized fixtures add the explicit uncontracted p/d shells in system_for;
 * both geometries use 1e-13 energy and 1e-10 orbital-gradient convergence.
 */
void check_reference(const ScfResult& result, bool unrestricted, bool moved, bool forces,
                     bool polarized) {
  double energy = unrestricted ? (moved ? -3.0238944398316963 : -3.0089750919335954)
                               : (moved ? -2.843497533600211 : -2.840418791231528);
  if (polarized)
    energy = unrestricted ? (moved ? -3.0943946173546464 : -3.084528237938705)
                          : (moved ? -2.8708266063637846 : -2.8695562149996334);
  require(result.converged && std::abs(result.energy - energy) < 1e-8,
          "profile endpoint failed independent energy gate");
  if (!forces) return;
  double force = unrestricted ? (moved ? 0.4815583993190885 : 0.5134714730999471)
                              : (moved ? 0.09425952363345647 : 0.1112894685854584);
  if (polarized)
    force = unrestricted ? (moved ? 0.31974655862200985 : 0.3382306192399964)
                         : (moved ? 0.0364028166953978 : 0.04854207363098417);
  const std::array<double, 6> expected{0., 0., -force, 0., 0., force};
  require(result.forces.size() == expected.size(), "profile endpoint force shape mismatch");
  for (std::size_t element = 0; element < expected.size(); ++element)
    require(std::abs(result.forces[element] - expected[element]) < 1e-7,
            "profile endpoint failed independent force gate");
}

struct Plan {
  CudaRhfBucketPlan* value{};
  ~Plan() { destroy_rhf_cuda_bucket_plan(value); }
};

void verify_endpoint(const cuda_policy::HfMatrixProfileContext& context, const char* path,
                     bool unrestricted, bool forces, std::size_t batch_size, bool polarized) {
  auto system = system_for(unrestricted, polarized);
  ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.screening_tolerance = 1e-12;
  options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  options.compute_forces = forces;
  Plan owner;
  const auto evaluate = [&](const char* phase, const core::System& geometry,
                            const std::vector<double>* density, bool library, bool calibrated,
                            bool reuse, bool moved) {
    const auto start = std::chrono::steady_clock::now();
    const std::vector<core::System> systems(batch_size, geometry);
    const std::vector<const std::vector<double>*> densities(batch_size, density);
    const auto rows =
        unrestricted
            ? run_uhf_cuda_bucket_cached(&owner.value, systems, options, densities, 0, false)
            : run_rhf_cuda_bucket_cached(&owner.value, systems, options, densities, 0, false);
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    require(rows.size() == batch_size && owner.value, "profile endpoint lost its batch");
    require(owner.value->topology.matrix_profile.has_value() == calibrated,
            "production packer did not retain the admitted profile");
    const bool resident = calibrated ? !library : !polarized;
    require(owner.value->topology.persistent_eri == resident &&
                owner.value->persistent_eri == (resident && !forces),
            "production execution disagrees with prepared resident-ERI admission");
    require(owner.value->topology.persistent_force_angular_order_count ==
                    (calibrated && library ? 3U : 7U) &&
                owner.value->topology.persistent_fock_angular_order_count ==
                    (calibrated && library ? 3U : 6U),
            "production preparation lost its runtime queue domains");
    require(owner.value->resources.matrix_binding_.has_value(), "missing prepared matrix binding");
    require(owner.value->resources.matrix_binding_->requires_provider_context() == library,
            "production execution used a different matrix recipe");
    for (const auto& row : rows) {
      require(row.status == GENERATIVEQC_STATUS_SUCCESS, "profile SCF execution failed");
      require(row.execution_plan_reused == reuse, "profile cache invalidation mismatch");
      check_reference(row.scf, unrestricted, moved, forces, polarized);
      require(row.scf.executed_fock_builds.has_value(), "missing physical Fock work census");
      std::cout << "profile-fixture=" << phase << " spin=" << (unrestricted ? "UHF" : "RHF")
                << " forces=" << forces << " batch=" << batch_size << " library=" << library
                << " calibrated=" << calibrated << " reused=" << reuse
                << " nbf=" << owner.value->topology.nbf
                << " persistent_eri=" << owner.value->persistent_eri
                << " endpoint_ms=" << elapsed_ms << " iterations=" << row.scf.iterations
                << " fock_builds=" << *row.scf.executed_fock_builds
                << " owned_device_bytes=" << hf_cuda_owned_device_bytes(owner.value)
                << " energy=" << std::setprecision(17) << row.scf.energy << '\n';
    }
    return rows.front().scf.density;
  };
  unsetenv("GENERATIVEQC_HF_MATRIX_PROFILE");
  auto default_density = evaluate("no-profile", system, nullptr, polarized, false, false, false);
  default_density =
      evaluate("no-profile-warm", system, &default_density, polarized, false, true, false);
  auto moved_system = system;
  moved_system.atoms[1].position[2] += 0.03;
  evaluate("no-profile-moved-warm", moved_system, &default_density, polarized, false, true, true);
  write_profile(path, context, false);
  require(setenv("GENERATIVEQC_HF_MATRIX_PROFILE", path, 1) == 0, "profile environment failed");
  auto density = evaluate("native-cold", system, nullptr, false, true, false, false);
  density = evaluate("native-warm", system, &density, false, true, true, false);
  evaluate("native-moved-warm", moved_system, &density, false, true, true, true);
  write_profile(path, context, true);
  density = evaluate("library-cold", system, nullptr, true, true, false, false);
  density = evaluate("library-warm", system, &density, true, true, true, false);
  system.atoms[1].position[2] += 0.03;
  density = evaluate("library-moved-warm", system, &density, true, true, true, true);
  write_profile(path, context, true, true);
  evaluate("stale-fallback", system, &density, polarized, false, false, true);
}
/** Refuse optional tensor storage through the actual allocation ledger.
 * A compatible replay must retain the bounded Cartesian direct owner and
 * admitted profile rather than repeatedly attempting the refused allocation.
 */
void verify_resident_capacity_fallback(const cuda_policy::HfMatrixProfileContext& context,
                                       const char* path) {
  const auto system = system_for(false, true, true);
  const auto check_capacity_reference = [](const ScfResult& result) {
    // Independent PySCF 2.14.0 spherical s/p/d/f RHF, same strict gates as above.
    require(result.converged && std::abs(result.energy - (-2.872528313696481)) < 1e-8,
            "capacity fallback failed its independent 32-AO energy gate");
  };
  ScfOptions options;
  options.max_iterations = 100;
  options.energy_tolerance = 1e-12;
  options.density_tolerance = 1e-10;
  options.screening_tolerance = 1e-12;
  options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  options.compute_forces = false;
  const auto probe_peak = [&](bool direct) {
    write_profile(path, context, false, false, direct);
    require(cuda_policy::load_hf_matrix_profile(path, context).profile.has_value(),
            "capacity probe profile does not match the loaded execution stack");
    auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
    ledger->device = 0;
    ledger->limit = std::numeric_limits<std::size_t>::max();
    runtime::DeviceResourceScope scope(ledger);
    {
      Plan owner;
      const auto rows = run_rhf_cuda_bucket_cached(&owner.value, {system}, options, {nullptr}, 0);
      require(rows.size() == 1 && rows.front().status == GENERATIVEQC_STATUS_SUCCESS,
              "capacity probe failed");
      require(owner.value->topology.matrix_profile.has_value() &&
                  owner.value->persistent_eri == !direct,
              "capacity probe did not execute its selected topology");
      check_capacity_reference(rows.front().scf);
    }
    require(ledger->live == 0, "capacity probe leaked tracked storage");
    return ledger->peak;
  };
  require(setenv("GENERATIVEQC_HF_MATRIX_PROFILE", path, 1) == 0, "profile environment failed");
  const auto direct_peak = probe_peak(true);
  const auto resident_peak = probe_peak(false);
  std::cout << "capacity-probe direct_peak=" << direct_peak << " resident_peak=" << resident_peak
            << '\n';
  require(resident_peak > direct_peak, "fixture does not distinguish optional resident capacity");
  auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
  ledger->device = 0;
  ledger->limit = direct_peak + (resident_peak - direct_peak) / 2;
  runtime::DeviceResourceScope scope(ledger);
  {
    Plan owner;
    auto rows = run_rhf_cuda_bucket_cached(&owner.value, {system}, options, {nullptr}, 0);
    require(rows.size() == 1 && rows.front().status == GENERATIVEQC_STATUS_SUCCESS && owner.value,
            "resident allocation refusal lost the bounded CUDA route");
    require(!owner.value->persistent_eri_enabled && !owner.value->topology.persistent_eri &&
                !owner.value->topology.ao_to_direct_transform.empty() &&
                owner.value->topology.matrix_profile.has_value(),
            "capacity fallback lost profile identity or required Cartesian transforms");
    require(rows.front().scf.precision.execution_retries == 1 && ledger->rejected > 0 &&
                ledger->peak <= ledger->limit,
            "resident fallback was not a single capacity-bounded retry");
    check_capacity_reference(rows.front().scf);
    const auto rejected = ledger->rejected;
    const auto density = rows.front().scf.density;
    rows = run_rhf_cuda_bucket_cached(&owner.value, {system}, options, {&density}, 0);
    require(rows.front().status == GENERATIVEQC_STATUS_SUCCESS &&
                rows.front().execution_plan_reused && ledger->rejected == rejected &&
                rows.front().scf.precision.execution_retries == 0,
            "compatible replay retried refused optional storage");
    check_capacity_reference(rows.front().scf);
    std::cout << "resident-capacity-fallback limit=" << ledger->limit << " peak=" << ledger->peak
              << " rejected=" << rejected << '\n';
  }
  require(ledger->live == 0, "resident fallback leaked tracked storage");
}
}  // namespace

int main(int argc, char** argv) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0 ||
      runtime::active_cuda_provider().kind != runtime::CudaProviderKind::Nvidia)
    return 77;
  try {
    runtime::CudaTargetInfo target;
    require(query_hf_cuda_target(0, target) == GENERATIVEQC_STATUS_SUCCESS, "target query failed");
    const auto context = query_hf_cuda_matrix_profile_context(target);
    FixtureFile fixture;
    if (argc == 2 && std::string(argv[1]) == "--capacity-only") {
      verify_resident_capacity_fallback(context, fixture.path.data());
      std::cout << "Bounded resident-ERI allocation fallback passed\n";
      return 0;
    }
    require(argc == 1, "only --capacity-only is supported");
    write_profile(fixture.path.data(), context, true);
    require(cuda_policy::load_hf_matrix_profile(fixture.path.data(), context).profile.has_value(),
            "actual execution stack could not admit its compatible profile");
    std::cout << "synthetic_ordering_fixture=true source=" << context.source_identity.data()
              << " native_build_id=" << context.native_build_id.data()
              << " runtime_build_id=" << context.runtime_build_id.data()
              << " driver_build_id=" << context.driver_build_id.data()
              << " cublas_build_id=" << context.cublas_build_id.data() << '\n';
    for (const bool polarized : {false, true})
      for (const bool unrestricted : {false, true})
        for (const bool forces : {false, true})
          for (const std::size_t batch_size : {1U, 2U})
            verify_endpoint(context, fixture.path.data(), unrestricted, forces, batch_size,
                            polarized);
    verify_resident_capacity_fallback(context, fixture.path.data());
    std::cout << "Real HF matrix-profile routing and independent endpoint gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
