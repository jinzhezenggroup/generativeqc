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
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
