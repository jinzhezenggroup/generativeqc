#include <array>
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
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
  descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
  auto options = common_scf_options(descriptor);
  require(!options.preliminary_guess && options.max_iterations == 100 &&
              options.diis_history == 8 && options.energy_tolerance == 1.0e-10 &&
              options.density_tolerance == 1.0e-8 && options.screening_tolerance == 1.0e-12 &&
              options.application_precision_budget == std::array<double, 3>{},
          "HF/KS common SCF defaults changed");

  descriptor.max_iterations = 37;
  descriptor.diis_history = 5;
  descriptor.energy_tolerance = 3.0e-11;
  descriptor.density_tolerance = 4.0e-9;
  descriptor.screening_tolerance = 5.0e-13;
  options = common_scf_options(descriptor);
  require(options.max_iterations == 37 && options.diis_history == 5 &&
              options.energy_tolerance == 3.0e-11 && options.density_tolerance == 4.0e-9 &&
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

void verify_descriptor_prefix() {
  generativeqc_method_descriptor descriptor{};
  const auto rejects_abi = [&] {
    try {
      (void)common_scf_options(descriptor);
    } catch (const MethodError& error) {
      require(error.status() == GENERATIVEQC_STATUS_ABI_MISMATCH,
              "invalid common method prefix returned the wrong status");
      return;
    }
    throw std::runtime_error("invalid common method prefix was accepted");
  };
  rejects_abi();
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  for (std::size_t size = 0; size < sizeof(descriptor); ++size) {
    descriptor.struct_size = static_cast<std::uint32_t>(size);
    rejects_abi();
  }
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION + 1;
  rejects_abi();
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  for (std::size_t size = sizeof(descriptor) + 1; size < sizeof(ApplicationPrecisionMethodOptions);
       ++size) {
    descriptor.struct_size = static_cast<std::uint32_t>(size);
    rejects_abi();
  }
}

void verify_independent_correlated_execution() {
  using generativeqc::core::DfPreconvergence;
  CorrelationExecutionMethodOptions options{};
  auto& descriptor = options.base.base;
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  // A legacy/zero extension is still strict; adding an execution selector must
  // not grant a numerical allowance or change the public descriptor prefix.
  for (auto size : {sizeof(descriptor), sizeof(options.base), sizeof(options)}) {
    descriptor.struct_size = size;
    require(correlation_preconvergence(descriptor) == DfPreconvergence::Legacy,
            "legacy correlated execution changed");
  }
  for (auto mode : {DfPreconvergence::Fp64, DfPreconvergence::Mixed, DfPreconvergence::Automatic}) {
    options.df_preconvergence = static_cast<std::uint32_t>(mode);
    require(correlation_preconvergence(descriptor) == mode,
            "correlated execution selector was lost");
    require(application_precision_budget(descriptor) == std::array<double, 3>{},
            "execution selection relaxed the scientific allowance");
  }
  options.base.energy = 1e-5;
  options.base.maximum_force = 1e-4;
  options.df_preconvergence = static_cast<std::uint32_t>(DfPreconvergence::Fp64);
  require(application_precision_budget(descriptor)[0] == 1e-5 &&
              correlation_preconvergence(descriptor) == DfPreconvergence::Fp64,
          "relaxed convergence cannot select strict arithmetic");
  options.df_preconvergence = 4;
  rejects_invalid([&] { (void)correlation_preconvergence(descriptor); }, "selector");
  options.df_preconvergence = 0;
  options.reserved = 1;
  rejects_invalid([&] { (void)correlation_preconvergence(descriptor); }, "selector");
  options.reserved = 0;
  for (std::size_t size = sizeof(options.base) + 1; size < sizeof(options); ++size) {
    descriptor.struct_size = size;
    try {
      (void)correlation_preconvergence(descriptor);
    } catch (const MethodError& error) {
      require(error.status() == GENERATIVEQC_STATUS_ABI_MISMATCH,
              "truncated execution extension returned the wrong status");
      continue;
    }
    throw std::runtime_error("truncated execution extension accepted");
  }
}

void verify_incremental_selectors() {
  using generativeqc::scf::classify_incremental_direct_jk_control;
  using generativeqc::scf::IncrementalDirectJkControl;
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
  for (const char* value : {"", "-1", "+1", " 1", "1 ", "1x", "999999999999999999999999999"})
    rejects_invalid([&] { (void)parse_incremental_direct_jk_rebuild_interval(name, value); },
                    "must be an unsigned integer");
}
}  // namespace

int main() {
  try {
    verify_scf_defaults();
    verify_descriptor_prefix();
    verify_independent_correlated_execution();
    verify_incremental_selectors();
    return 0;
  } catch (...) {
    return 1;
  }
}
