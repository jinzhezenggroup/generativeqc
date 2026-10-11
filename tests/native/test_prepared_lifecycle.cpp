#include <cmath>
#include <iostream>
#include <stdexcept>

#include "methods/method.hpp"
#include "molecule/basis.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Operation>
void rejected(Operation operation) {
  bool failed = false;
  try {
    operation();
  } catch (const std::logic_error&) {
    failed = true;
  }
  require(failed, "illegal lifecycle transition was accepted");
}

void transitions() {
  using namespace generativeqc::runtime;
  CompiledExecutionRegion region;
  require(region.state() == CompiledExecutionState::Unbound, "initial state");
  rejected([&] { region.begin(); });
  CompiledExecutionBinding binding{"qualified", 0, nullptr, nullptr, nullptr, 0};
  region.bind(binding);
  region.begin();
  require(region.state() == CompiledExecutionState::Running, "running state");
  rejected([&] { region.begin(); });
  rejected([&] { region.bind(binding); });
  rejected([&] { region.recover(); });
  rejected([&] { region.close(); });
  require(!region.invalidate(), "running invalidation must be refused");
  region.mark_success();
  region.begin();
  region.abort();
  require(region.failed() && !region.warmed(), "failed output remained warmed");
  require(!region.bind(binding), "compatible bind must not erase sticky failure");
  rejected([&] { region.begin(); });
  rejected([&] { region.mark_success(); });
  region.recover();
  region.begin();
  region.mark_success();
  binding.backend = 1;
  require(region.bind(binding), "changing target must rebind native state");
  require(!region.warmed(), "target rebind retained old replay eligibility");
  region.close();
  region.close();
  rejected([&] { region.begin(); });
  rejected([&] { region.bind(binding); });
  rejected([&] { region.recover(); });
}

void cpu_consumers() {
  using namespace generativeqc;
  core::System system;
  system.atoms = {{1, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
  const std::vector<core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  system.shells = {{0, 0, primitives}, {1, 0, primitives}};
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "H2 setup");
  core::ContextState context;
  for (const auto method : {GENERATIVEQC_METHOD_RHF, GENERATIVEQC_METHOD_PBE_RKS}) {
    generativeqc_method_descriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
    descriptor.method = method;
    descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
    descriptor.max_iterations = 200;
    descriptor.energy_tolerance = 1e-12;
    descriptor.density_tolerance = 1e-10;
    auto owner = methods::prepare_calculation(context, system, descriptor);
    require(owner->execution_lifecycle().bound(), "real owner was not bound");
    const auto cold = owner->execute(false);
    const auto replay = owner->execute(false);
    require(cold.convergence.converged && replay.convergence.converged &&
                std::abs(cold.energy - replay.energy) < 1e-10,
            "real CPU replay changed the result");
    owner->reject_execution();
    require(owner->execution_lifecycle().failed(), "publication failure not retained");
    const auto recovery = owner->execute(false);
    require(recovery.convergence.converged && std::abs(cold.energy - recovery.energy) < 1e-10,
            "real CPU recovery changed the result");
    owner->reset_execution();
    require(!owner->execution_lifecycle().warmed(), "reset retained warm eligibility");
    owner->close_execution();
    rejected([&] { owner->execute(false); });
    rejected([&] { owner->reset_execution(); });
  }
}
}  // namespace

int main() {
  try {
    transitions();
    cpu_consumers();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
