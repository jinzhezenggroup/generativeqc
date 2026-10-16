#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "methods/batch_capabilities.hpp"
#include "methods/correlated_batch.hpp"
#include "methods/correlated_warm_reference.hpp"
#include "methods/execution_failure.hpp"
#include "molecule/basis.hpp"
#include "scf/mean_field.hpp"

namespace {
using namespace generativeqc;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Operation>
void rejected(generativeqc_status status, Operation operation) {
  try {
    operation();
  } catch (...) {
    require(methods::item_exception_status() == status, "failure status changed");
    return;
  }
  throw std::runtime_error("expected rejection was not raised");
}

scf::ScfResult accepted(scf::FockSpin spin) {
  scf::ScfResult result;
  result.converged = true;
  if (spin == scf::FockSpin::Restricted)
    result.reference = std::make_shared<hf::PhysicalReference>();
  else
    result.unrestricted_reference = std::make_shared<hf::UnrestrictedPhysicalReference>();
  return result;
}

void reference_policy() {
  const std::vector<double> seed{1};
  const auto typed_rejection = std::make_exception_ptr(
      methods::MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE, "rejected seed"));
  for (const auto spin : {scf::FockSpin::Restricted, scf::FockSpin::Unrestricted}) {
    bool fallback = true;
    auto result = methods::warm_reference::solve_reference(
        nullptr, spin, &fallback,
        [&](const std::vector<double>* density) {
          require(!density, "cold reference received a seed");
          return accepted(spin);
        },
        "reference failed");
    require(result.converged && !fallback, "cold reference claimed fallback");

    for (const bool throws : {false, true}) {
      std::weak_ptr<const hf::PhysicalReference> discarded;
      bool cold_ran = false;
      result = methods::warm_reference::solve_reference(
          &seed, spin, &fallback,
          [&](const std::vector<double>* density) {
            if (density) {
              require(density == &seed, "reference seed identity changed");
              if (throws) std::rethrow_exception(typed_rejection);
              auto proposal = accepted(scf::FockSpin::Restricted);
              discarded = proposal.reference;
              proposal.converged = false;
              return proposal;
            }
            require(discarded.expired(), "unsuccessful reference survived into cold solve");
            require(!cold_ran, "cold retry was unbounded");
            cold_ran = true;
            return accepted(spin);
          },
          "reference failed");
      require(result.converged && fallback && cold_ran, "seed rejection did not recover");
    }

    rejected(GENERATIVEQC_STATUS_NOT_CONVERGED, [&] {
      methods::warm_reference::solve_reference(
          nullptr, spin, &fallback,
          [&](const std::vector<double>*) {
            auto proposal = accepted(spin);
            proposal.reference.reset();
            proposal.unrestricted_reference.reset();
            return proposal;
          },
          "missing physical reference");
    });
    rejected(GENERATIVEQC_STATUS_CUDA_ERROR, [&] {
      methods::warm_reference::solve_reference(
          &seed, spin, &fallback,
          [&](const std::vector<double>* density) -> scf::ScfResult {
            if (density) std::rethrow_exception(typed_rejection);
            throw methods::MethodError(GENERATIVEQC_STATUS_CUDA_ERROR, "cold device failure");
          },
          "reference failed");
    });
  }

  const std::array terminal_failures{
      std::make_exception_ptr(std::bad_alloc()),
      std::make_exception_ptr(std::length_error("capacity")),
      std::make_exception_ptr(std::runtime_error("pending device operation")),
      std::make_exception_ptr(methods::MethodError(GENERATIVEQC_STATUS_CUDA_ERROR, "device")),
      std::make_exception_ptr(Error(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "capability"))};
  for (const auto& failure : terminal_failures) {
    bool fallback = true;
    const auto status = methods::item_exception_status(failure);
    rejected(status, [&] {
      methods::warm_reference::solve_reference(
          &seed, scf::FockSpin::Restricted, &fallback,
          [&](const std::vector<double>* density) -> scf::ScfResult {
            require(density, "terminal failure submitted a cold reference");
            std::rethrow_exception(failure);
          },
          "reference failed");
    });
    require(!fallback, "terminal failure claimed a cold retry");
  }
  require(methods::item_exception_status({}) == GENERATIVEQC_STATUS_INTERNAL_ERROR,
          "empty exception has no internal status");
}

core::System h2() {
  core::System system;
  system.atoms = {{1, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
  const std::vector<core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  system.shells = {{0, 0, primitives}, {1, 0, primitives}};
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "H2 setup failed");
  return system;
}

struct Control {
  bool fail_binding{}, fail_execution{}, nonconverged{}, saw_forces{};
  double last_seed_energy = std::numeric_limits<double>::quiet_NaN();
  int executable_token{};
};

std::array<Control, 2> controls;
scf::HfWarmState candidate_template;
int next_executable_token{};
double next_energy{};

/** These providers perform no device/scientific work. CUDA is only a placement
 * label here; real-device qualification belongs to the production endpoint tests. */
class ProbeCalculation final : public methods::PreparedCalculation,
                               public methods::ReferenceSeededCalculation {
 public:
  ProbeCalculation(const methods::Capabilities& capabilities, core::ContextState& context,
                   const core::System& system)
      : PreparedCalculation(context.requested_backend),
        capabilities_(capabilities),
        backend_(context.requested_backend),
        system_(system),
        control_(controls.at(static_cast<std::size_t>(system.atoms[0].position[0]))),
        executable_(std::make_unique<int>(++next_executable_token)) {}

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const methods::Capabilities& capabilities() const noexcept override { return capabilities_; }

  methods::Result execute_seeded_reference(bool forces, const scf::HfWarmState* seed,
                                           bool* fallback,
                                           std::optional<scf::HfWarmState>* candidate) override {
    control_.saw_forces = forces;
    control_.last_seed_energy = seed ? seed->energy : std::numeric_limits<double>::quiet_NaN();
    control_.executable_token = *executable_;
    if (fallback) *fallback = false;
    methods::Result result;
    result.energy = ++next_energy;
    result.executed_backend = backend_;
    result.convergence.converged = !control_.nonconverged;
    if (forces) result.forces.assign(3 * atom_count(), 1);
    if (candidate) {
      *candidate = candidate_template;
      (*candidate)->coordinates = methods::geometry::coordinates(system_);
      (*candidate)->energy = result.energy;
    }
    if (control_.fail_execution) throw std::bad_alloc();
    return result;
  }

  void transfer_reference_execution_to(methods::ReferenceSeededCalculation& target) override {
    dynamic_cast<ProbeCalculation&>(target).executable_ = std::move(executable_);
  }

  runtime::ExecutionResourceSnapshot execution_resources() const noexcept override {
    runtime::ExecutionResourceSnapshot snapshot;
    snapshot.host_numeric_peak_bytes = 123;
    return snapshot;
  }

  std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic() const override {
    return generativeqc_correlation_diagnostic{};
  }

  std::optional<methods::CcPerformanceDiagnostic> cc_performance_diagnostic() const override {
    methods::CcPerformanceDiagnostic diagnostic;
    diagnostic.reference_seconds = 17;
    return diagnostic;
  }

 private:
  methods::Result execute_impl(bool forces) override {
    return execute_seeded_reference(forces, nullptr, nullptr, nullptr);
  }
  methods::Capabilities capabilities_;
  generativeqc_backend backend_;
  core::System system_;
  Control& control_;
  std::unique_ptr<int> executable_;
};

std::unique_ptr<methods::PreparedCalculation> prepare_probe(
    const methods::Capabilities& capabilities, core::ContextState& context,
    const core::System& system, const generativeqc_method_descriptor&, const core::System*) {
  if (controls.at(static_cast<std::size_t>(system.atoms[0].position[0])).fail_binding)
    throw methods::MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "failed rebind");
  return std::make_unique<ProbeCalculation>(capabilities, context, system);
}

void batch_policy(generativeqc_backend backend, bool unrestricted, bool cc_diagnostics,
                  const core::System& source, const scf::HfWarmState& initial) {
  controls = {};
  candidate_template = initial;
  if (unrestricted) {
    for (auto& value : candidate_template.density) value *= 0.5;
    const auto beta = candidate_template.density;
    candidate_template.density.insert(candidate_template.density.end(), beta.begin(), beta.end());
  }
  core::ContextState context;
  context.requested_backend = backend;
  auto neighbor = source;
  for (auto& atom : neighbor.atoms) atom.position[0] = 1;
  generativeqc_method_descriptor descriptor{};
  const methods::Capabilities capabilities{};
  auto batch = methods::prepare_correlated_batch(
      capabilities, context, {source, neighbor}, descriptor, true, prepare_probe, "probe",
      unrestricted ? GENERATIVEQC_METHOD_UHF : GENERATIVEQC_METHOD_RHF, !cc_diagnostics,
      cc_diagnostics);
  auto& warm = dynamic_cast<methods::BatchWarmStarts&>(*batch);
  const auto* cc = dynamic_cast<methods::BatchCcPerformanceDiagnostics*>(batch.get());
  require(static_cast<bool>(cc) == cc_diagnostics, "method diagnostic facets changed");
  require(!cc || cc->cc_performance_diagnostic(0)->reference_seconds == 17,
          "CC diagnostic was not forwarded");
  require(batch->execution_resources(0).host_numeric_peak_bytes == 123,
          "resource observation was not forwarded");
  require(warm.warm_density_size(0) == candidate_template.density.size(), "spin density size");

  auto results = batch->execute({}, false);
  const auto first_seed = warm.warm_state(0)->energy;
  const auto first_token = controls[0].executable_token;
  for (std::size_t index = 0; index < results.size(); ++index) {
    require(results[index].status == GENERATIVEQC_STATUS_SUCCESS &&
                !results[index].warm_start_used && results[index].calculation.forces.empty() &&
                results[index].calculation.executed_backend == backend &&
                results[index].bucket_id == (cc_diagnostics ? 0 : index),
            "cold batch placement, ordering or output changed");
  }
  warm.set_warm_start_updates(false);
  results = batch->execute({}, true);
  require(results[0].warm_start_used && controls[0].saw_forces &&
              controls[0].last_seed_energy == first_seed &&
              warm.warm_state(0)->energy == first_seed,
          "frozen seed was not reused without replacement");
  warm.set_warm_start_updates(true);

  auto moved = methods::geometry::coordinates(source);
  moved[2] -= 0.01;
  moved[5] -= 0.01;
  controls[0].fail_binding = true;
  results = batch->execute({moved, std::nullopt}, false);
  require(results[0].status == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              std::isnan(results[0].calculation.energy) &&
              results[1].status == GENERATIVEQC_STATUS_SUCCESS &&
              warm.warm_state(0)->energy == first_seed,
          "failed rebind corrupted last-good state or a neighboring item");
  controls[0].fail_binding = false;
  results = batch->execute({moved, std::nullopt}, true);
  require(results[0].status == GENERATIVEQC_STATUS_SUCCESS &&
              controls[0].executable_token == first_token &&
              warm.warm_state(0)->coordinates == moved &&
              controls[0].last_seed_energy == first_seed,
          "geometry rebind lost executable ownership or last-good seed");

  const auto last_good = warm.warm_state(0)->energy;
  for (const bool allocation_failure : {false, true}) {
    controls[0].nonconverged = !allocation_failure;
    controls[0].fail_execution = allocation_failure;
    results = batch->execute({moved, std::nullopt}, false);
    require(results[0].status == (allocation_failure ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
                                                     : GENERATIVEQC_STATUS_NOT_CONVERGED) &&
                results[1].status == GENERATIVEQC_STATUS_SUCCESS &&
                warm.warm_state(0)->energy == last_good,
            "failed endpoint published a candidate or aborted its neighbor");
  }
  controls[0].nonconverged = controls[0].fail_execution = false;
  auto restored = warm.warm_state(0);
  restored->energy += 10;
  auto invalid = warm.warm_state(1);
  invalid->density.clear();
  rejected(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
           [&] { warm.restore_warm_states({restored, invalid}); });
  require(warm.warm_state(0)->energy == last_good, "checkpoint restore was not transactional");
  warm.restore_warm_states({restored, std::nullopt});
  require(warm.warm_state(0)->energy == restored->energy && warm.warm_state(1).has_value(),
          "partial checkpoint restore replaced an unselected seed");
  warm.clear_warm_starts();
  results = batch->execute({}, false);
  require(!results[0].warm_start_used && results[0].status == GENERATIVEQC_STATUS_SUCCESS,
          "cleared batch did not recover with a cold seed");
}
}  // namespace

int main() {
  try {
    reference_policy();
    const auto source = h2();
    scf::ScfOptions options;
    options.export_physical_reference = true;
    options.compute_forces = false;
    options.screening_tolerance = 0;
    options.reference_memory_budget_bytes = 256ULL << 20;
    const auto reference = scf::run_rhf(source, options);
    require(reference.converged && reference.reference, "physical warm-density setup");
    const auto initial =
        methods::warm_reference::capture(source, *reference.reference, reference.energy_change,
                                         reference.density_rms, reference.iterations);
    for (const auto backend : {GENERATIVEQC_BACKEND_CPU_REFERENCE, GENERATIVEQC_BACKEND_CUDA})
      for (const bool unrestricted : {false, true})
        for (const bool cc_diagnostics : {false, true})
          batch_policy(backend, unrestricted, cc_diagnostics, source, initial);
    std::cout << "shared method reference and batch execution contracts: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
