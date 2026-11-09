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
#include <stdexcept>
#include <string>
#include <vector>

#include "molecule/basis.hpp"
#include "runtime/lowering_binding.hpp"
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
                   bool library, bool stale = false) {
  const auto& target = context.target;
  std::ofstream stream(path);
  stream << "generativeqc-hf-matrix-profile-v1\nsource_identity "
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
         << "\nprovider_setup_nanoseconds 0\nend\n";
  require(static_cast<bool>(stream), "could not write profile fixture");
}

core::System system_for(bool unrestricted) {
  core::System system;
  system.atoms = {{2, {0., 0., -0.60}}, {1, {0., 0., 0.85}}};
  system.shells = {
      {0, 0, {{6.36242139, 0.15432897}, {1.15892300, 0.53532814}, {0.31364979, 0.44463454}}},
      {1, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}}};
  system.charge = unrestricted ? 0 : 1;
  system.multiplicity = unrestricted ? 2 : 1;
  system.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "profile endpoint normalization failed");
  return system;
}

/** Independent PySCF 2.14.0 STO-3G/Bohr fixtures retained by the incremental
 * endpoint test, including its +0.03-Bohr changed-geometry determinant.
 */
void check_reference(const ScfResult& result, bool unrestricted, bool moved, bool forces) {
  const double energy = unrestricted ? (moved ? -3.0238944398316963 : -3.0089750919335954)
                                     : (moved ? -2.843497533600211 : -2.840418791231528);
  require(result.converged && std::abs(result.energy - energy) < 1e-8,
          "profile endpoint failed independent energy gate");
  if (!forces) return;
  const double force = unrestricted ? (moved ? 0.4815583993190885 : 0.5134714730999471)
                                    : (moved ? 0.09425952363345647 : 0.1112894685854584);
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
                     bool unrestricted, bool forces, std::size_t batch_size) {
  auto system = system_for(unrestricted);
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
    require(owner.value->resources.matrix_binding_.has_value(), "missing prepared matrix binding");
    require(owner.value->resources.matrix_binding_->requires_provider_context() == library,
            "production execution used a different matrix recipe");
    for (const auto& row : rows) {
      require(row.status == GENERATIVEQC_STATUS_SUCCESS, "profile SCF execution failed");
      require(row.execution_plan_reused == reuse, "profile cache invalidation mismatch");
      check_reference(row.scf, unrestricted, moved, forces);
      require(row.scf.executed_fock_builds.has_value(), "missing physical Fock work census");
      std::cout << "profile-fixture=" << phase << " spin=" << (unrestricted ? "UHF" : "RHF")
                << " forces=" << forces << " batch=" << batch_size << " library=" << library
                << " calibrated=" << calibrated << " reused=" << reuse
                << " endpoint_ms=" << elapsed_ms << " iterations=" << row.scf.iterations
                << " fock_builds=" << *row.scf.executed_fock_builds
                << " owned_device_bytes=" << hf_cuda_owned_device_bytes(owner.value)
                << " energy=" << std::setprecision(17) << row.scf.energy << '\n';
    }
    return rows.front().scf.density;
  };
  unsetenv("GENERATIVEQC_HF_MATRIX_PROFILE");
  evaluate("no-profile", system, nullptr, false, false, false, false);
  write_profile(path, context, false);
  require(setenv("GENERATIVEQC_HF_MATRIX_PROFILE", path, 1) == 0, "profile environment failed");
  auto density = evaluate("native-cold", system, nullptr, false, true, false, false);
  density = evaluate("native-warm", system, &density, false, true, true, false);
  write_profile(path, context, true);
  density = evaluate("library-cold", system, nullptr, true, true, false, false);
  density = evaluate("library-warm", system, &density, true, true, true, false);
  system.atoms[1].position[2] += 0.03;
  density = evaluate("library-moved-warm", system, &density, true, true, true, true);
  write_profile(path, context, true, true);
  evaluate("stale-fallback", system, &density, false, false, false, true);
}
}  // namespace

int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0 ||
      runtime::active_cuda_provider().kind != runtime::CudaProviderKind::Nvidia)
    return 77;
  try {
    runtime::CudaTargetInfo target;
    require(query_hf_cuda_target(0, target) == GENERATIVEQC_STATUS_SUCCESS, "target query failed");
    const auto context = query_hf_cuda_matrix_profile_context(target);
    FixtureFile fixture;
    write_profile(fixture.path.data(), context, true);
    require(cuda_policy::load_hf_matrix_profile(fixture.path.data(), context).profile.has_value(),
            "actual execution stack could not admit its compatible profile");
    std::cout << "synthetic_ordering_fixture=true source=" << context.source_identity.data()
              << " native_build_id=" << context.native_build_id.data()
              << " runtime_build_id=" << context.runtime_build_id.data()
              << " driver_build_id=" << context.driver_build_id.data()
              << " cublas_build_id=" << context.cublas_build_id.data() << '\n';
    for (const bool unrestricted : {false, true})
      for (const bool forces : {false, true})
        for (const std::size_t batch_size : {1U, 2U})
          verify_endpoint(context, fixture.path.data(), unrestricted, forces, batch_size);
    std::cout << "Real HF matrix-profile routing and independent endpoint gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
