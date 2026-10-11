#include <limits>
#include <stdexcept>
#include <string>

#include "methods/scf_common_options.hpp"

namespace {
using generativeqc::methods::MethodError;
using namespace generativeqc::methods::detail;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callback>
void rejects_invalid(Callback&& call, const char* detail) {
  try {
    call();
  } catch (const MethodError& error) {
    require(error.status() == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "common method validation returned the wrong status");
    require(std::string(error.what()).find(detail) != std::string::npos,
            "common method validation lost its diagnostic");
    return;
  }
  throw std::runtime_error("invalid common method option was accepted");
}

void verify_scf_defaults() {
  generativeqc_method_descriptor descriptor{};
  descriptor.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
  descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
  auto options = common_scf_options(descriptor);
  require(!options.preliminary_guess && options.max_iterations == 100 &&
              options.diis_history == 8 && options.energy_tolerance == 1.0e-10 &&
              options.density_tolerance == 1.0e-8 && options.screening_tolerance == 1.0e-12,
          "HF/KS common SCF defaults changed");

  descriptor.max_iterations = 37;
  descriptor.diis_history = 5;
  descriptor.energy_tolerance = 3.0e-11;
  descriptor.density_tolerance = 4.0e-9;
  descriptor.screening_tolerance = 5.0e-13;
  options = common_scf_options(descriptor);
  require(options.max_iterations == 37 && options.diis_history == 5 &&
              options.energy_tolerance == 3.0e-11 &&
              options.density_tolerance == 4.0e-9 &&
              options.screening_tolerance == 5.0e-13,
          "HF/KS explicit SCF overrides changed");

  require(checked_density_fitting_mode(descriptor) == GENERATIVEQC_DENSITY_FITTING_NONE,
          "none density fitting rejected");
  descriptor.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE;
  require(checked_density_fitting_mode(descriptor) == GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE,
          "CPU density fitting rejected");
  descriptor.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CUDA;
  require(checked_density_fitting_mode(descriptor) == GENERATIVEQC_DENSITY_FITTING_CUDA,
          "CUDA density fitting rejected");
  descriptor.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_AUTO;
  require(checked_density_fitting_mode(descriptor) == GENERATIVEQC_DENSITY_FITTING_AUTO,
          "automatic density fitting rejected");
  descriptor.density_fitting_mode = static_cast<generativeqc_density_fitting_mode>(999);
  rejects_invalid([&] { (void)checked_density_fitting_mode(descriptor); },
                  "unknown density-fitting execution mode");

  require(checked_precision_mode(descriptor) == GENERATIVEQC_PRECISION_FP64,
          "strict FP64 mode rejected");
  descriptor.precision_mode = GENERATIVEQC_PRECISION_AUTO;
  require(checked_precision_mode(descriptor) == GENERATIVEQC_PRECISION_AUTO,
          "automatic precision mode rejected");
  descriptor.precision_mode = static_cast<generativeqc_precision_mode>(999);
  rejects_invalid([&] { (void)checked_precision_mode(descriptor); },
                  "unknown floating-point precision mode");
}

void verify_incremental_selectors() {
  using generativeqc::scf::IncrementalDirectJkControl;
  using generativeqc::scf::classify_incremental_direct_jk_control;
  require(classify_incremental_direct_jk_control(nullptr) == IncrementalDirectJkControl::Disabled,
          "unset incremental selector changed");
  for (const char* text : {"0", "off"})
    require(classify_incremental_direct_jk_control(text) == IncrementalDirectJkControl::Disabled,
            "disabled incremental selector changed");
  for (const char* text : {"1", "on"})
    require(classify_incremental_direct_jk_control(text) == IncrementalDirectJkControl::Enabled,
            "enabled incremental selector changed");
  for (const char* text : {"", "yes", "OFF", "2", "1 "})
    require(classify_incremental_direct_jk_control(text) == IncrementalDirectJkControl::Invalid,
            "invalid incremental selector was accepted");

  constexpr const char* name = "GENERATIVEQC_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL";
  require(!parse_incremental_direct_jk_rebuild_interval(name, nullptr).has_value(),
          "unset rebuild interval did not preserve the default");
  require(parse_incremental_direct_jk_rebuild_interval(name, "0") == 0u &&
              parse_incremental_direct_jk_rebuild_interval(name, "0008") == 8u &&
              parse_incremental_direct_jk_rebuild_interval(name, "429") == 429u,
          "valid rebuild interval changed");
  const std::string maximum = std::to_string(std::numeric_limits<unsigned>::max());
  require(parse_incremental_direct_jk_rebuild_interval(name, maximum.c_str()) ==
              std::numeric_limits<unsigned>::max(),
          "maximum unsigned rebuild interval was rejected");
  for (const char* value : {"", "-1", "+1", " 1", "1 ", "1x",
                            "999999999999999999999999999"})
    rejects_invalid([&] { (void)parse_incremental_direct_jk_rebuild_interval(name, value); },
                    "must be an unsigned integer");
}
}  // namespace

int main() {
  try {
    verify_scf_defaults();
    verify_incremental_selectors();
    return 0;
  } catch (...) {
    return 1;
  }
}
