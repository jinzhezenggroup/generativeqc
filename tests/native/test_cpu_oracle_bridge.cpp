// Cross-library qualification: build this executable once, then select the
// clean main, frozen prototype or integration library using the loader path.
// No symbol introduced by the integration is referenced here. Numeric output
// is bit-exact JSONL; route observations are separate from comparable payloads.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include "molecule/basis.hpp"
#include "posthf/raw_source.hpp"
#include "runtime/host_component_trace.hpp"
#include "scf/mean_field.hpp"
#include "scf/proposal_bridge.hpp"
#include "scf/reference/observation.hpp"

// Private C ABI declarations, copied from src/posthf/bridge.cpp. These are not
// additions to, or claims about, the supported public calculator ABI.
extern "C" {
int generativeqc_posthf_rhf_density_v1(void*, int, int, unsigned, double, int, double, double*,
                                       std::size_t, double*, char*, std::size_t);
int generativeqc_posthf_uhf_density_v1(void*, int, int, unsigned, double, int, double, double*,
                                       std::size_t, double*, char*, std::size_t);
int generativeqc_accuracy_hf_probe_v1(void*, int, int, int, unsigned, unsigned, double, double,
                                      double, int, double, double*, std::size_t, double*,
                                      std::size_t, double*, std::size_t, char*, std::size_t);
int generativeqc_scf_solve_v1(void*, int, int, int, unsigned, unsigned, double, double, double,
                              const double*, ScfProposeV1, ScfObserveV1, double*, std::size_t,
                              double*, std::size_t, double*, std::size_t, char*, std::size_t);
}

namespace {
using namespace generativeqc;
namespace obs = scf::reference::observation;
constexpr unsigned kMaxIterations = 200;
constexpr double kTolerance = 1e-10;
constexpr double kMetricThreshold = 1e-10;
// A finite, nonzero sentinel lets us distinguish untouched outputs from the
// bridges' intentionally written NaNs, zeros, failed iterates and diagnostics.
constexpr std::uint64_t kSentinelBits = UINT64_C(0xc123456789abcdef);
constexpr double kSentinel = std::bit_cast<double>(kSentinelBits);
constexpr std::array<const char*, 8> kReasons{"unspecified", "overlap",          "core_guess",
                                              "final_fock",  "reference_export", "fallback",
                                              "iteration",   "seed_validation"};
using Counts = std::array<std::size_t, kReasons.size()>;
thread_local Counts calls{};
thread_local bool invalid_observation{};
thread_local runtime::host_trace::detail::State host_work;
bool expect_force_closure = true;

void require(bool condition, const std::string& detail) {
  if (!condition) throw std::runtime_error(detail);
}
std::size_t begin_observation(const char* name, std::size_t n) noexcept {
  if (std::strcmp(name, "reference_eigensolve") == 0) {
    const auto reason = static_cast<std::size_t>(obs::active_reason);
    if (reason >= calls.size() || n != 7)
      invalid_observation = true;
    else
      ++calls[reason];
  }
  return runtime::host_trace::detail::begin(name, n);
}
void end_observation(std::size_t token, int exceptions) noexcept {
  runtime::host_trace::detail::end(token, exceptions);
}
const obs::Observer observer{begin_observation, end_observation};
class Observation {
 public:
  Observation() : previous_(obs::active), previous_host_(runtime::host_trace::detail::active) {
    host_work = {};
    runtime::host_trace::detail::active = &host_work;
    calls = {};
    invalid_observation = false;
    obs::active = &observer;
  }
  ~Observation() {
    obs::active = previous_;
    runtime::host_trace::detail::active = previous_host_;
  }
  Observation(const Observation&) = delete;
  Observation& operator=(const Observation&) = delete;

 private:
  const obs::Observer* previous_;
  runtime::host_trace::detail::State* previous_host_;
};

struct Hooks {
  unsigned proposed{}, observed{};
  std::uint64_t generation{};
  bool malformed{};
  std::size_t spins{};
};
thread_local Hooks hooks{};
int propose_none(const ScfSnapshotViewV1* s, double*, std::uint64_t* generation,
                 unsigned* iteration) {
  ++hooks.proposed;
  hooks.malformed |= !s || !generation || !iteration;
  if (!s || !generation || !iteration) return 0;
  hooks.malformed |= s->nbf != 7 || s->spins != hooks.spins || !s->density || !s->fock ||
                     !s->residual || !s->overlap || !s->baseline || !s->electrons ||
                     s->iteration == 0;
  if (hooks.generation == 0) hooks.generation = s->generation;
  hooks.malformed |= hooks.generation != s->generation;
  *generation = s->generation;
  *iteration = s->iteration;
  return 0;  // No proposal; preserve the primary baseline update.
}
void observe_decision(const ScfSnapshotViewV1* s, const ScfDecisionViewV1* d) {
  ++hooks.observed;
  hooks.malformed |= !s || !d;
  if (s && d)
    hooks.malformed |=
        s->nbf != 7 || s->spins != hooks.spins || !d->reason || s->generation != hooks.generation;
}

std::string json_quote(std::string_view text) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : text) {
    if (c == '"' || c == '\\')
      out << '\\' << static_cast<char>(c);
    else if (c < 0x20)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else
      out << static_cast<char>(c);
  }
  out << '"';
  return out.str();
}
void print_bits(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << '"' << std::hex << std::setw(16) << std::setfill('0')
              << std::bit_cast<std::uint64_t>(values[i]) << '"' << std::dec;
  }
  std::cout << ']';
}
struct Record {
  std::string entry, fixture, approximation, scenario, detail;
  int status{};
  std::vector<double> density, forces, scalars;
  Counts counts{};
  std::size_t selector_focks{}, corrections{}, correction_solves{}, fixed_point_checks{},
      promotions{};
  unsigned proposed{}, observed{};
  bool physical_reference{};
};
void capture_counts(Record& r) {
  r.counts = calls;
  require(host_work.valid && host_work.current == -1, "incomplete host-work observation");
  for (const auto& row : host_work.regions) {
    require(row.finished, "unfinished host-work region");
    const std::string_view name(row.name);
    r.selector_focks += name == "final_state_fock_build";
    r.corrections += name == "strict_final_correction";
    r.correction_solves += name == "final_state_correction_solve";
    r.fixed_point_checks += name == "final_state_fixed_point";
    r.promotions += name == "final_state_fixed_point_promotion";
  }
}
void print(const Record& r) {
  std::cout << "{\"kind\":\"case\",\"entry\":" << json_quote(r.entry)
            << ",\"fixture\":" << json_quote(r.fixture)
            << ",\"approximation\":" << json_quote(r.approximation)
            << ",\"scenario\":" << json_quote(r.scenario) << ",\"status\":" << r.status
            << ",\"detail\":" << json_quote(r.detail) << ",\"density_bits\":";
  print_bits(r.density);
  std::cout << ",\"force_bits\":";
  print_bits(r.forces);
  std::cout << ",\"scalar_bits\":";
  print_bits(r.scalars);
  std::cout << ",\"reference_leaf_counts\":{";
  for (std::size_t i = 0; i < r.counts.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << json_quote(kReasons[i]) << ':' << r.counts[i];
  }
  std::cout << "},\"physical_reference\":" << (r.physical_reference ? "true" : "false")
            << ",\"proposed\":" << r.proposed << ",\"observed\":" << r.observed << "}\n";
}
std::size_t count(const Record& r, obs::EigenReason reason) {
  return r.counts[static_cast<std::size_t>(reason)];
}
void check_route(const Record& r, bool reference, unsigned iterations, std::size_t spins,
                 bool converged, bool forces = true) {
  const std::string prefix =
      r.entry + "/" + r.fixture + "/" + r.approximation + "/" + r.scenario + ": ";
  require(!invalid_observation, prefix + "invalid reference observation");
  // Setup positivity is essential: otherwise a disconnected observer could
  // falsely certify the absence of reference target work in a scalar solve.
  require(count(r, obs::EigenReason::overlap) >= 1, prefix + "overlap leaf not observed");
  require(count(r, obs::EigenReason::core_guess) == 1, prefix + "core guess leaf changed");
  require(count(r, obs::EigenReason::iteration) == (reference ? iterations * spins : 0),
          prefix + "iteration provider/count mismatch");
  std::size_t final_spin_solves = converged ? 1 : 0;
  if (expect_force_closure && converged && forces && spins == 2) {
    require(r.selector_focks == r.corrections + 1 && r.corrections <= 32 &&
                r.fixed_point_checks >= 1 && r.correction_solves + r.promotions == r.corrections &&
                r.fixed_point_checks == r.promotions + 1,
            prefix + "bounded final-state work did not reconcile");
    final_spin_solves += r.correction_solves + r.fixed_point_checks;
    if (r.scalars.size() == 6)
      require(r.scalars[5] == iterations + 1 + r.selector_focks,
              prefix + "total Fock builds do not include actual final-state work");
  } else {
    require(r.selector_focks == 0 && r.corrections == 0 && r.fixed_point_checks == 0,
            prefix + "non-force/RHF/failed solve acquired final-state correction work");
    if (converged && r.scalars.size() == 6)
      require(r.scalars[5] == iterations + 2, prefix + "ordinary Fock schedule changed");
  }
  require(count(r, obs::EigenReason::final_fock) == (reference ? spins * final_spin_solves : 0),
          prefix + "finalization provider/count mismatch");
  require(count(r, obs::EigenReason::reference_export) == 0 && !r.physical_reference,
          prefix + "physical-reference export was used as a selector");
  require(count(r, obs::EigenReason::fallback) == 0, prefix + "unexpected reference fallback");
}
void check_rejected_without_work(const Record& r) {
  require(r.status != 0 && !r.detail.empty(), r.entry + ": malformed request accepted");
  require(std::all_of(r.counts.begin(), r.counts.end(), [](auto n) { return n == 0; }),
          r.entry + ": malformed request performed eigensolve work");
}
bool untouched(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::bit_cast<std::uint64_t>(value) == kSentinelBits;
  });
}
bool finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}
bool all_nan(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(), [](double value) { return std::isnan(value); });
}
bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::bit_cast<std::uint64_t>(a[i]) != std::bit_cast<std::uint64_t>(b[i])) return false;
  return true;
}

core::System water(bool uhf) {
  core::System s;
  s.charge = uhf ? 1 : 0;
  s.multiplicity = uhf ? 2 : 1;
  s.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  s.atoms = {{8, {0, 0, 0}}, {1, {0, -1.43233673, 1.10715266}}, {1, {0, 1.43233673, 1.10715266}}};
  s.shells = {
      {0, 0, {{130.70932, .15432897}, {23.808861, .53532814}, {6.4436083, .44463454}}},
      {0, 0, {{5.0331513, -.09996723}, {1.1695961, .39951283}, {.3803890, .70011547}}},
      {0, 1, {{5.0331513, .15591627}, {1.1695961, .60768372}, {.3803890, .39195739}}},
      {1, 0, {{3.425250914, .1543289673}, {.6239137298, .5353281423}, {.168855404, .4446345422}}},
      {2, 0, {{3.425250914, .1543289673}, {.6239137298, .5353281423}, {.168855404, .4446345422}}}};
  std::string detail;
  require(molecule::validate_and_normalize(s, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "fixture normalization failed: " + detail);
  require(molecule::ao_count(s) == 7 && s.electron_count == (uhf ? 9 : 10),
          "fixture dimension/electron count changed");
  return s;
}
Record initial_record(const char* entry, bool uhf, bool df, const std::string& scenario,
                      std::size_t scalars, bool forces = false) {
  Record r;
  r.entry = entry;
  r.fixture = uhf ? "water-cation7" : "water7";
  r.approximation = df ? "df" : "exact";
  r.scenario = scenario;
  r.density.assign((uhf ? 2 : 1) * 49, kSentinel);
  r.scalars.assign(scalars, kSentinel);
  if (forces) r.forces.assign(9, kSentinel);
  return r;
}
Record oracle(posthf::RawSource& raw, bool uhf, bool df, const std::string& scenario,
              int backend = 0, std::size_t short_by = 0, unsigned iterations = kMaxIterations,
              double metric = kMetricThreshold) {
  auto r = initial_record("oracle", uhf, df, scenario, 4);
  std::array<char, 512> error{};
  Observation observation;
  const auto operation =
      uhf ? generativeqc_posthf_uhf_density_v1 : generativeqc_posthf_rhf_density_v1;
  r.status = operation(&raw, backend, 0, iterations, kTolerance, df, metric, r.density.data(),
                       r.density.size() - short_by, r.scalars.data(), error.data(), error.size());
  r.detail = error.data();
  capture_counts(r);
  print(r);
  return r;
}
void oracle_cases(posthf::RawSource& raw, bool uhf, bool df, bool reference) {
  const std::size_t spins = uhf ? 2 : 1;
  const auto good = oracle(raw, uhf, df, "converged");
  require(good.status == 0 && good.detail.empty() && finite(good.density) && finite(good.scalars),
          "oracle did not return finite converged state: " + good.detail);
  require(good.scalars[3] > 1 && good.scalars[3] <= kMaxIterations,
          "unexpected oracle iteration diagnostic");
  check_route(good, reference, static_cast<unsigned>(good.scalars[3]), spins, true);
  for (const auto& bad :
       {oracle(raw, uhf, df, "bad_backend", 7), oracle(raw, uhf, df, "bad_shape", 0, 1)}) {
    check_rejected_without_work(bad);
    require(untouched(bad.density) && untouched(bad.scalars),
            "malformed oracle output sentinel overwritten");
  }
  for (const auto& [label, threshold] :
       {std::pair{"metric_nan", std::numeric_limits<double>::quiet_NaN()},
        std::pair{"metric_zero", 0.0}}) {
    const auto metric = oracle(raw, uhf, df, label, 0, 0, kMaxIterations, threshold);
    if (df) {
      check_rejected_without_work(metric);
      require(untouched(metric.density) && untouched(metric.scalars),
              "invalid DF metric overwrote output sentinels");
    } else {
      require(metric.status == 0 && metric.detail.empty() &&
                  same_bits(metric.density, good.density) &&
                  same_bits(metric.scalars, good.scalars),
              "exact oracle must ignore unused metric threshold bit-for-bit");
      check_route(metric, reference, static_cast<unsigned>(metric.scalars[3]), spins, true);
    }
  }
  const auto failed = oracle(raw, uhf, df, "max_iterations_1", 0, 0, 1);
  require(failed.status != 0 && !failed.detail.empty() && untouched(failed.density) &&
              untouched(failed.scalars),
          "nonconverged oracle exported density or overwrote scalar sentinels");
  check_route(failed, reference, 1, spins, false);
}

Record diagnostic(posthf::RawSource& raw, bool uhf, bool df, bool sol01,
                  const std::string& scenario, unsigned iterations = kMaxIterations,
                  bool malformed = false) {
  auto r = initial_record(sol01 ? "sol01_hooks" : "num01", uhf, df, scenario, sol01 ? 6 : 5, true);
  std::array<char, 512> error{};
  const auto method = uhf ? GENERATIVEQC_METHOD_UHF : GENERATIVEQC_METHOD_RHF;
  hooks = {};
  hooks.spins = uhf ? 2 : 1;
  Observation observation;
  if (sol01) {
    r.status = generativeqc_scf_solve_v1(
        &raw, method, raw.orbital().multiplicity, df, iterations, 8, kTolerance, kTolerance,
        kMetricThreshold, nullptr, propose_none, observe_decision, r.density.data(),
        r.density.size() - (malformed ? 1 : 0), r.forces.data(), r.forces.size(), r.scalars.data(),
        r.scalars.size(), error.data(), error.size());
  } else {
    r.status = generativeqc_accuracy_hf_probe_v1(
        &raw, method, 0, 0, iterations, 8, kTolerance, kTolerance, 0, df, kMetricThreshold,
        r.density.data(), r.density.size() - (malformed ? 1 : 0), r.forces.data(), r.forces.size(),
        r.scalars.data(), r.scalars.size(), error.data(), error.size());
  }
  r.detail = error.data();
  capture_counts(r);
  r.proposed = hooks.proposed;
  r.observed = hooks.observed;
  print(r);
  require(!hooks.malformed, "SOL01 callback snapshot contract failed");
  return r;
}
void diagnostic_cases(posthf::RawSource& raw, bool uhf, bool df, bool sol01, bool reference) {
  const std::size_t spins = uhf ? 2 : 1;
  const auto good = diagnostic(raw, uhf, df, sol01, "converged");
  require(good.status == 0 && good.detail.empty() && good.scalars[4] == 1 && finite(good.density) &&
              finite(good.forces) && finite(good.scalars),
          good.entry + ": expected finite converged result: " + good.detail);
  const auto iterations = static_cast<unsigned>(good.scalars[3]);
  require(iterations > 1 && iterations <= kMaxIterations, "invalid diagnostic iterations");
  check_route(good, reference, iterations, spins, true);
  if (sol01)
    require(good.proposed > 0 && good.observed > 0 && good.scalars[5] >= iterations + 2,
            "SOL01 hooks were not executed");
  const auto bad = diagnostic(raw, uhf, df, sol01, "bad_shape", kMaxIterations, true);
  check_rejected_without_work(bad);
  require(untouched(bad.density) && untouched(bad.forces) && untouched(bad.scalars) &&
              bad.proposed == 0 && bad.observed == 0,
          "malformed diagnostic overwrote sentinel or invoked hooks");
  const auto failed = diagnostic(raw, uhf, df, sol01, "max_iterations_1", 1);
  require(failed.status == 0 && failed.detail.empty() && failed.scalars[3] == 1 &&
              failed.scalars[4] == 0 && all_nan(failed.forces),
          "diagnostic nonconvergence contract changed");
  if (sol01)
    require(finite(failed.density) && !untouched(failed.density) && failed.scalars[5] == 1 &&
                failed.proposed > 0 && failed.observed > 0,
            "SOL01 failed iterate/hooks not retained");
  else
    require(all_nan(failed.density), "NUM01 failed density was exposed as reusable");
  check_route(failed, reference, 1, spins, false);
}

void primary_case(posthf::RawSource& raw, bool uhf, bool df, bool dispatch, bool reference,
                  bool forces = true) {
  auto r = initial_record(dispatch ? "run_fock_strategy" : "run_cpu_fock_strategy", uhf, df,
                          forces ? "converged" : "energy_only", 6, forces);
  scf::ScfOptions options;
  options.compute_forces = forces;
  options.max_iterations = kMaxIterations;
  options.energy_tolerance = options.density_tolerance = kTolerance;
  options.screening_tolerance = 0;
  options.density_fitting_relative_threshold = kMetricThreshold;
  // The guard is part of the test: requesting a physical reference changes the
  // endpoint and cannot be used to make an oracle/provider test pass.
  require(!options.export_physical_reference, "test requested a physical-reference selector");
  options.resolved_fock_build = scf::resolve_fock_build(
      scf::make_hf_fock_spec(
          uhf ? scf::FockSpin::Unrestricted : scf::FockSpin::Restricted,
          df ? scf::FockApproximation::DensityFitted : scf::FockApproximation::Exact),
      scf::FockBackend::Cpu, 0, kMetricThreshold);
  Observation observation;
  const auto* auxiliary = df ? &raw.auxiliary() : nullptr;
  const auto result = dispatch ? scf::run_fock_strategy(raw.orbital(), auxiliary, options, 0)
                               : scf::run_cpu_fock_strategy(raw.orbital(), auxiliary, options);
  r.status = result.converged ? 0 : 1;
  r.density = result.density;
  r.forces = result.forces;
  r.scalars = {result.energy,
               result.energy_change,
               result.density_rms,
               static_cast<double>(result.iterations),
               result.converged ? 1.0 : 0.0,
               static_cast<double>(result.fock_builds)};
  capture_counts(r);
  r.physical_reference = bool(result.reference);
  print(r);
  require(result.converged && result.density.size() == (uhf ? 98 : 49) &&
              result.forces.size() == (forces ? 9 : 0) && finite(r.density) && finite(r.forces) &&
              finite(r.scalars),
          r.entry + ": primary solve failed");
  check_route(r, reference, result.iterations, uhf ? 2 : 1, true, forces);
}

void loaded_libraries() {
#if defined(__unix__) || defined(__APPLE__)
  for (const auto* symbol :
       {"generativeqc_posthf_rhf_density_v1", "generativeqc_posthf_uhf_density_v1",
        "generativeqc_accuracy_hf_probe_v1", "generativeqc_scf_solve_v1"}) {
    void* address = dlsym(RTLD_DEFAULT, symbol);
    Dl_info info{};
    require(address && dladdr(address, &info) != 0 && info.dli_fname,
            std::string("could not resolve loaded library for ") + symbol);
    std::cout << "{\"kind\":\"library\",\"symbol\":" << json_quote(symbol)
              << ",\"path\":" << json_quote(info.dli_fname) << "}\n";
  }
#else
  std::cout << "{\"kind\":\"library\",\"detail\":\"dladdr unavailable on this platform\"}\n";
#endif
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string mode = "integration";
    if (argc == 3 && std::string_view(argv[1]) == "--mode")
      mode = argv[2];
    else
      require(argc == 1, "usage: test_cpu_oracle_bridge [--mode main|prototype|integration]");
    require(mode == "main" || mode == "prototype" || mode == "integration", "invalid mode");
    std::cout << "{\"kind\":\"metadata\",\"schema\":1,\"mode\":" << json_quote(mode)
              << ",\"fixture_basis\":\"normalized STO-3G water7; identical orbital/auxiliary "
                 "primitives\","
                 "\"comparison\":\"integration oracle payloads equal main; integration "
                 "primary/NUM01/SOL01 payloads equal prototype\"}\n";
    expect_force_closure = mode == "integration";
    loaded_libraries();
    for (const bool uhf : {false, true}) {
      const auto system = water(uhf);
      const auto auxiliary = system;  // Copy normalized primitives; never normalize twice.
      posthf::RawSource raw(system, &auxiliary);
      require(raw.nbf() == 7 && raw.naux() == 7, "RawSource fixture dimensions changed");
      for (const bool df : {false, true}) {
        oracle_cases(raw, uhf, df, mode != "prototype");
        primary_case(raw, uhf, df, false, mode == "main");
        primary_case(raw, uhf, df, true, mode == "main");
        primary_case(raw, uhf, df, false, mode == "main", false);
        diagnostic_cases(raw, uhf, df, false, mode == "main");
        diagnostic_cases(raw, uhf, df, true, mode == "main");
      }
    }
    std::cout << "{\"kind\":\"result\",\"status\":\"PASS\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CPU oracle bridge qualification failed: " << error.what() << '\n';
    return 1;
  }
}
