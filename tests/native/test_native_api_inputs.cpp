#include "core/types.hpp"

#ifdef GENERATIVEQC_ABI_VERSION
#error "native core must not include the public C ABI"
#endif

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "api/ks_hessian.hpp"
#include "api/native_method_inputs.hpp"

namespace {
using namespace generativeqc;
static_assert(std::is_same_v<core::Backend, generativeqc_backend>);
static_assert(std::is_same_v<core::BasisRepresentation, generativeqc_basis_representation>);
static_assert(api::native_backend(GENERATIVEQC_BACKEND_CUDA) == core::kCudaBackend);
static_assert(api::native_basis_representation(GENERATIVEQC_BASIS_SPHERICAL) ==
              core::kSphericalBasis);

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
using Context = std::unique_ptr<generativeqc_context, decltype(&generativeqc_context_destroy)>;
using System = std::unique_ptr<generativeqc_system, decltype(&generativeqc_system_destroy)>;
using Calculation =
    std::unique_ptr<generativeqc_calculation, decltype(&generativeqc_calculation_destroy)>;
using Batch = std::unique_ptr<generativeqc_batch, decltype(&generativeqc_batch_destroy)>;

System system(generativeqc_context* context, double exponent, double half_distance = 0.7,
              int atomic_number = 1) {
  const std::array<generativeqc_atom, 2> atoms{
      {{atomic_number, 0, 0, -half_distance}, {atomic_number, 0, 0, half_distance}}};
  const std::array<generativeqc_primitive, 2> primitives{{{exponent, 1}, {exponent, 1}}};
  const std::array<generativeqc_shell, 2> shells{{{0, 0, 0, 1}, {1, 0, 1, 1}}};
  generativeqc_system_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.atoms = atoms.data();
  descriptor.atom_count = atoms.size();
  descriptor.shells = shells.data();
  descriptor.shell_count = shells.size();
  descriptor.primitives = primitives.data();
  descriptor.primitive_count = primitives.size();
  descriptor.multiplicity = 1;
  descriptor.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  generativeqc_system* out = nullptr;
  require(generativeqc_system_create(context, &descriptor, &out) == GENERATIVEQC_STATUS_SUCCESS,
          "system conversion failed");
  return {out, generativeqc_system_destroy};
}

generativeqc_method_descriptor descriptor(generativeqc_method method,
                                          const generativeqc_system* auxiliary) {
  generativeqc_method_descriptor value{};
  value.struct_size = sizeof(value);
  value.abi_version = GENERATIVEQC_ABI_VERSION;
  value.method = method;
  value.max_iterations = 200;
  value.diis_history = 8;
  value.energy_tolerance = 1e-12;
  value.density_tolerance = 1e-10;
  value.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE;
  value.density_fitting_auxiliary_basis = auxiliary;
  value.density_fitting_relative_threshold = 1e-10;
  value.precision_mode = GENERATIVEQC_PRECISION_FP64;
  return value;
}

Calculation prepare(generativeqc_context* context, const generativeqc_system* orbital,
                    const generativeqc_method_descriptor& method) {
  generativeqc_calculation* out = nullptr;
  require(generativeqc_calculation_prepare(context, orbital, &method, &out) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "single preparation failed");
  return {out, generativeqc_calculation_destroy};
}

double energy(generativeqc_calculation* calculation) {
  generativeqc_result_descriptor result{};
  result.struct_size = sizeof(result);
  result.abi_version = GENERATIVEQC_ABI_VERSION;
  require(generativeqc_calculation_execute(calculation, &result) == GENERATIVEQC_STATUS_SUCCESS &&
              result.converged && std::isfinite(result.energy),
          "single execution failed");
  require(result.executed_backend == GENERATIVEQC_BACKEND_CPU_REFERENCE,
          "backend conversion changed");
  return result.energy;
}

double batch_energy(generativeqc_batch* batch, const double* coordinates = nullptr) {
  generativeqc_batch_item_result_descriptor result{};
  result.struct_size = sizeof(result);
  result.abi_version = GENERATIVEQC_ABI_VERSION;
  generativeqc_batch_input_descriptor input{};
  input.struct_size = sizeof(input);
  input.abi_version = GENERATIVEQC_ABI_VERSION;
  input.coordinates = coordinates;
  input.coordinate_count = coordinates ? 6 : 0;
  require(generativeqc_batch_execute(batch, &input, 1, &result, 1) == GENERATIVEQC_STATUS_SUCCESS &&
              result.status == GENERATIVEQC_STATUS_SUCCESS && result.converged &&
              std::isfinite(result.energy),
          "batch execution failed");
  return result.energy;
}

void invalid_inputs(generativeqc_context* context, generativeqc_method method) {
  auto orbital = system(context, 0.8);
  auto mismatch = system(context, 0.3, 0.9);
  auto request = descriptor(method, mismatch.get());
  generativeqc_calculation* calculation = nullptr;
  require(generativeqc_calculation_prepare(context, orbital.get(), &request, &calculation) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              !calculation,
          "mismatched auxiliary geometry was accepted");
  request = descriptor(method, nullptr);
  require(generativeqc_calculation_prepare(context, nullptr, &request, &calculation) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "null orbital was accepted");
  require(generativeqc_calculation_prepare(nullptr, orbital.get(), &request, &calculation) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "null context was accepted");
  --request.struct_size;
  require(generativeqc_calculation_prepare(context, orbital.get(), &request, &calculation) ==
                  GENERATIVEQC_STATUS_ABI_MISMATCH &&
              !calculation,
          "short method descriptor was accepted");
  if (method != GENERATIVEQC_METHOD_MP2) {
    const generativeqc_system* items[]{orbital.get()};
    generativeqc_batch* batch = nullptr;
    require(generativeqc_batch_prepare(context, items, 1, &request, 0, &batch) ==
                    GENERATIVEQC_STATUS_ABI_MISMATCH &&
                !batch,
            "short batch descriptor was accepted");
    mismatch = system(context, 0.3, 0.7, 2);
    request = descriptor(method, mismatch.get());
    require(generativeqc_batch_prepare(context, items, 1, &request, 0, &batch) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                !batch,
            "batch accepted mismatched auxiliary topology");
    items[0] = nullptr;
    require(generativeqc_batch_prepare(context, items, 1, &request, 0, &batch) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                !batch,
            "null batch member was accepted");
  }
}

void auxiliary_lifetime(generativeqc_context* context, generativeqc_method method) {
  auto orbital = system(context, 0.8);
  auto auxiliary = system(context, 0.3);
  auto request = descriptor(method, auxiliary.get());
  auto single = prepare(context, orbital.get(), request);
  const auto expected = energy(single.get());
  const auto implicit = energy(prepare(context, orbital.get(), descriptor(method, nullptr)).get());
  require(std::abs(expected - implicit) > 1e-8, "distinct auxiliary input was ignored");

  Batch batch(nullptr, generativeqc_batch_destroy);
  if (method != GENERATIVEQC_METHOD_MP2) {
    const generativeqc_system* items[]{orbital.get()};
    generativeqc_batch* out = nullptr;
    require(generativeqc_batch_prepare(context, items, 1, &request, 0, &out) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "batch preparation failed");
    batch.reset(out);
    require(std::abs(batch_energy(batch.get()) - expected) < 1e-10, "batch input mismatch");
  }

  // Mutate the actual source handle before destruction to deterministically
  // expose a borrowed System, then release both public systems and descriptor.
  auxiliary->data = {};
  orbital->data = {};
  auxiliary.reset();
  orbital.reset();
  request = {};
  for (int replay = 0; replay < 2; ++replay) {
    require(std::abs(energy(single.get()) - expected) < 1e-10, "single retained caller input");
    if (batch)
      require(std::abs(batch_energy(batch.get()) - expected) < 1e-10,
              "batch retained caller input");
  }
  if (batch) {
    auto moved_orbital = system(context, 0.8, 0.8);
    auto moved_auxiliary = system(context, 0.3, 0.8);
    const auto oracle = energy(
        prepare(context, moved_orbital.get(), descriptor(method, moved_auxiliary.get())).get());
    const double coordinates[]{0, 0, -0.8, 0, 0, 0.8};
    require(std::abs(batch_energy(batch.get(), coordinates) - oracle) < 1e-10,
            "batch failed to rebind the owned auxiliary geometry");
  }
}

void correlation_auxiliary_forwarding(generativeqc_context* cpu_context) {
  auto orbital = system(cpu_context, 0.8);
  auto auxiliary = system(cpu_context, 0.3);
  auxiliary->correlation_auxiliary = true;
  // Preparation only copies host metadata. Supply CUDA intent without creating
  // a device context or executing the prepared calculation.
  generativeqc_context metadata_context{};
  metadata_context.state.requested_backend = core::kCudaBackend;
  auto request = descriptor(GENERATIVEQC_METHOD_MP2, auxiliary.get());
  request.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CORRELATION_CUDA;
  generativeqc_calculation* out = nullptr;
  const auto status =
      generativeqc_calculation_prepare(&metadata_context, orbital.get(), &request, &out);
  if (status != GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error("correlation MP2 auxiliary forwarding: " +
                             metadata_context.last_detail);
  Calculation prepared(out, generativeqc_calculation_destroy);
  require(!prepared->plan->capabilities().supports_batch,
          "correlation-only MP2 unexpectedly admitted batches");

  out = nullptr;
  request.density_fitting_auxiliary_basis = nullptr;
  require(generativeqc_calculation_prepare(&metadata_context, orbital.get(), &request, &out) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              !out,
          "correlation-only MP2 accepted an implicit auxiliary");
  auto mismatch = system(cpu_context, 0.3, 0.9);
  request.density_fitting_auxiliary_basis = mismatch.get();
  require(generativeqc_calculation_prepare(&metadata_context, orbital.get(), &request, &out) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              !out,
          "correlation-only MP2 accepted mismatched auxiliary geometry");
  request = descriptor(GENERATIVEQC_METHOD_MP2, auxiliary.get());
  require(generativeqc_calculation_prepare(&metadata_context, orbital.get(), &request, &out) ==
                  GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
              !out,
          "legacy RI-MP2 accepted a correlation-only auxiliary handle");
}

void scf_auxiliary_roles(generativeqc_context* context) {
  auto orbital = system(context, 0.8);
  auto auxiliary = system(context, 0.3);
  auxiliary->scf_auxiliary = true;
  auto request = descriptor(GENERATIVEQC_METHOD_RHF, auxiliary.get());
  request.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CUDA;
  generativeqc_ks_options options{};
  request.ks_options = &options;
  for (const auto method : {GENERATIVEQC_METHOD_RHF, GENERATIVEQC_METHOD_UHF,
                            GENERATIVEQC_METHOD_PBE_RKS, GENERATIVEQC_METHOD_PBE_UKS}) {
    request.method = method;
    for (const auto mode : {GENERATIVEQC_DENSITY_FITTING_CUDA, GENERATIVEQC_DENSITY_FITTING_AUTO}) {
      request.density_fitting_mode = mode;
      require(api::valid_system_roles(*orbital, request, core::kCudaBackend),
              "complete SCF DF role rejected");
      require(!api::valid_system_roles(*orbital, request, core::kCpuReferenceBackend),
              "SCF CUDA auxiliary entered CPU provider");
    }
    require(!api::valid_system_roles(*auxiliary, request, core::kCudaBackend),
            "SCF auxiliary became orbital");
  }
  request.method = GENERATIVEQC_METHOD_MP2;
  require(!api::valid_system_roles(*orbital, request, core::kCudaBackend),
          "SCF auxiliary entered correlation");
  request.method = GENERATIVEQC_METHOD_RHF;
  for (const auto mode :
       {GENERATIVEQC_DENSITY_FITTING_NONE, GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE,
        GENERATIVEQC_DENSITY_FITTING_FULL_CUDA}) {
    request.density_fitting_mode = mode;
    require(!api::valid_system_roles(*orbital, request, core::kCudaBackend),
            "SCF auxiliary entered another provider");
  }
}

void ks_auxiliary_roles(generativeqc_context* context) {
  auto orbital = system(context, 0.8);
  auto auxiliary = system(context, 0.3);
  auxiliary->ks_auxiliary = true;
  auto request = descriptor(GENERATIVEQC_METHOD_RHF, auxiliary.get());
  request.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CUDA;
  require(!api::valid_system_roles(*orbital, request, core::kCudaBackend),
          "HF accepted a KS-only auxiliary");
  request.method = GENERATIVEQC_METHOD_PBE_RKS;
  require(!api::valid_system_roles(*orbital, request, core::kCudaBackend),
          "KS-only auxiliary accepted absent KS options");
  generativeqc_ks_options options{};
  request.ks_options = &options;
  for (const auto mode : {GENERATIVEQC_DENSITY_FITTING_CUDA, GENERATIVEQC_DENSITY_FITTING_AUTO}) {
    request.density_fitting_mode = mode;
    require(api::valid_system_roles(*orbital, request, core::kCudaBackend),
            "CUDA KS DF source role was rejected");
    require(!api::valid_system_roles(*orbital, request, core::kCpuReferenceBackend),
            "KS CUDA auxiliary entered CPU provider");
    // Both public preparation entry points must consult the actual context.
    generativeqc_calculation* single = nullptr;
    generativeqc_batch* batch = nullptr;
    const generativeqc_system* items[]{orbital.get()};
    require(generativeqc_calculation_prepare(context, orbital.get(), &request, &single) ==
                    GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                !single,
            "CPU single preparation accepted a CUDA-only auxiliary");
    require(generativeqc_batch_prepare(context, items, 1, &request, 0, &batch) ==
                    GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                !batch,
            "CPU batch preparation accepted a CUDA-only auxiliary");
  }
  require(!api::valid_system_roles(*auxiliary, request, core::kCudaBackend),
          "KS-only auxiliary became an orbital");
  request.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE;
  require(!api::valid_system_roles(*orbital, request, core::kCudaBackend),
          "KS-only CUDA auxiliary entered another provider");

  double published = 7;
  generativeqc_ks_hessian_options_v1 policy;
  generativeqc_ks_hessian_receipt_v1 receipt;
  receipt.completed_directions = 99;
  require(generativeqc_ks_hessian_cuda_version_v1() <= 1, "unknown native Hessian bridge version");
  require(generativeqc_ks_hessian_cuda_v1(nullptr, 0, &policy, nullptr, 0, &published, 1,
                                          &receipt) == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              published == 7 && receipt.completed_directions == 99,
          "rejected Hessian call modified caller publication");

  request = descriptor(GENERATIVEQC_METHOD_PBE_RKS, nullptr);
  const generativeqc_system* items[]{orbital.get()};
  generativeqc_batch* output = nullptr;
  require(generativeqc_batch_prepare(context, items, 1, &request, 0, &output) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "CPU KS preparation for CUDA Hessian refusal failed");
  Batch batch(output, generativeqc_batch_destroy);
  (void)batch_energy(batch.get());
  std::array<double, 36> matrix;
  matrix.fill(7);
  require(generativeqc_ks_hessian_cuda_v1(batch.get(), 0, &policy, nullptr, 0, matrix.data(),
                                          matrix.size(),
                                          &receipt) == GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
              receipt.completed_directions == 99,
          "CPU KS state entered CUDA Hessian execution");
  for (double component : matrix)
    require(component == 7, "CPU KS refusal modified caller publication");
}
}  // namespace

int main() {
  try {
    generativeqc_context_descriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
    descriptor.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
    generativeqc_context* out = nullptr;
    require(generativeqc_context_create(&descriptor, &out) == GENERATIVEQC_STATUS_SUCCESS,
            "CPU context conversion failed");
    Context context(out, generativeqc_context_destroy);
    for (const auto method :
         {GENERATIVEQC_METHOD_RHF, GENERATIVEQC_METHOD_PBE_RKS, GENERATIVEQC_METHOD_MP2}) {
      invalid_inputs(context.get(), method);
      auxiliary_lifetime(context.get(), method);
    }
    correlation_auxiliary_forwarding(context.get());
    ks_auxiliary_roles(context.get());
    scf_auxiliary_roles(context.get());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
