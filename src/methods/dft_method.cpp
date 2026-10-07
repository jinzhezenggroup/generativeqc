#include "methods/dft_method.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "api/handles.hpp"
#include "dft/ao_grid.hpp"
#include "dft/dispersion/d4_runtime.hpp"
#include "dft/grid.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "dft/semilocal_family.hpp"
#include "generated_method_parameters.hpp"
#include "generativeqc/generativeqc.hpp"
#include "libxc_semilocal_cpu/generated_libxc_semilocal_registry.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/mean_field.hpp"
#include "scf/preliminary_guess.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/types.hpp"

#if GENERATIVEQC_HAS_CUDA
#include "dft/cuda_ks.hpp"
#include "generated_split_hybrid_registry.cuh"
#include "scf/cuda_direct_jk.hpp"
#include "scf/cuda_fock_execution.hpp"
#include "scf/cuda_one_electron_gradient.hpp"
#endif

namespace generativeqc::methods::detail {
namespace {

std::uint64_t next_cpu_ks_owner() {
  static std::atomic<std::uint64_t> next{1};
  auto value = next.load(std::memory_order_relaxed);
  do {
    if (value == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CPU KS owner identity exhausted");
  } while (!next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed));
  return value;
}

struct NativeKsExecutionPlan {
  std::uint32_t spin_channels{1};
  dft::SemilocalFamily semilocal_family{dft::SemilocalFamily::Lda};
  bool compiler_resolved{};
  bool d4_correction{};
  bool nonlocal_correlation{};
  dft::nlc::Vv10Parameters nonlocal_parameters{};
  std::uint64_t nonlocal_maximum_bytes{};
  bool range_exchange{};
  double short_range_exchange{};
  double long_range_exchange{};
  double range_omega{};
  std::uint32_t functional{};
  bool generated_split_hybrid{};
  const dft::SemilocalPointProgram* automatic_program{};
};

std::optional<NativeKsExecutionPlan> legacy_ks_execution_plan(generativeqc_method method) noexcept {
  switch (method) {
    case GENERATIVEQC_METHOD_LDA_RKS:
      return NativeKsExecutionPlan{1, dft::SemilocalFamily::Lda, false};
    case GENERATIVEQC_METHOD_LDA_UKS:
      return NativeKsExecutionPlan{2, dft::SemilocalFamily::Lda, false};
    case GENERATIVEQC_METHOD_PBE_D4_RKS:
      return NativeKsExecutionPlan{1, dft::SemilocalFamily::Pbe, false, true};
    case GENERATIVEQC_METHOD_PBE_RKS:
      return NativeKsExecutionPlan{1, dft::SemilocalFamily::Pbe, false};
    case GENERATIVEQC_METHOD_PBE_UKS:
      return NativeKsExecutionPlan{2, dft::SemilocalFamily::Pbe, false};
    case GENERATIVEQC_METHOD_R2SCAN_RKS:
      return NativeKsExecutionPlan{1, dft::SemilocalFamily::R2scan, false};
    case GENERATIVEQC_METHOD_R2SCAN_UKS:
      return NativeKsExecutionPlan{2, dft::SemilocalFamily::R2scan, false};
    default:
      return std::nullopt;
  }
}

bool unrestricted(const NativeKsExecutionPlan& plan) noexcept { return plan.spin_channels == 2; }

std::uint32_t scf_domain_version(const NativeKsExecutionPlan& plan) noexcept {
  if (plan.automatic_program) return plan.automatic_program->domain_version;
  return plan.generated_split_hybrid ? 4U
                                     : dft::semilocal_family_domain_version(plan.semilocal_family);
}

std::uint32_t xc_functional_code(const NativeKsExecutionPlan& plan) noexcept {
  return plan.functional ? plan.functional : dft::semilocal_family_code(plan.semilocal_family);
}

const char* semilocal_family_name(const NativeKsExecutionPlan& plan) noexcept {
  if (plan.automatic_program) return plan.automatic_program->identifier;
  return plan.generated_split_hybrid ? "generated split global hybrid"
                                     : dft::semilocal_family_name(plan.semilocal_family);
}

std::optional<double> semilocal_component(const generativeqc_ks_options& input,
                                          std::string_view component_id) {
  std::optional<double> value;
  for (std::uint32_t i = 0; i < input.semilocal_component_count; ++i) {
    const auto& term = input.semilocal_components[i];
    if (!term.component_id || !*term.component_id || !std::isfinite(term.coefficient) ||
        term.coefficient < 0.0)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "invalid KS semilocal component");
    if (std::string_view(term.component_id) == component_id) {
      if (value)
        throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "duplicate KS semilocal component");
      value = term.coefficient;
    }
  }
  return value;
}

struct SemilocalAdmission {
  dft::SemilocalFamily family{dft::SemilocalFamily::Lda};
  double exchange_scale{1.0};
  double correlation_scale{1.0};
  std::uint32_t functional{};
  bool generated_split_hybrid{};
  const dft::SemilocalPointProgram* automatic_program{};
};

std::optional<SemilocalAdmission> admit_curated_semilocal(const generativeqc_ks_options& input) {
  for (const auto& metadata : dft::kSemilocalFamilyMetadata) {
    if (input.semilocal_component_count != metadata.component_count ||
        input.semilocal_range_omega != metadata.range_omega)
      continue;
    double exchange_scale = 1.0;
    double correlation_scale = 1.0;
    bool matches = true;
    for (std::uint32_t i = 0; i < metadata.component_count; ++i) {
      const auto coefficient = semilocal_component(input, metadata.component_ids[i]);
      if (!coefficient) {
        matches = false;
        break;
      }
      if (metadata.component_coefficients_are_native_scales) {
        if (i == 0)
          exchange_scale = *coefficient;
        else if (i == 1)
          correlation_scale = *coefficient;
      } else if (*coefficient != metadata.component_coefficients[i]) {
        matches = false;
        break;
      }
    }
    if (matches) return SemilocalAdmission{metadata.family, exchange_scale, correlation_scale};
  }
  return std::nullopt;
}

SemilocalAdmission admit_semilocal(const generativeqc_ks_options& input) {
  if (!input.semilocal_components || !input.semilocal_component_count)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "KS execution plan requires semilocal components");
  if (!std::isfinite(input.semilocal_range_omega) || input.semilocal_range_omega < 0.0)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "invalid semilocal range parameter");

  if (auto curated = admit_curated_semilocal(input)) return *curated;

  if (input.semilocal_component_count == 1 && input.semilocal_range_omega == 0.0) {
    const auto& component = input.semilocal_components[0];
    if (!component.component_id || !*component.component_id ||
        !std::isfinite(component.coefficient) || component.coefficient < 0.0)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "invalid KS semilocal component");
    if (component.coefficient == 1.0) {
      const auto automatic =
          dft::generated::automatic_libxc_entry(std::string_view(component.component_id));
      if (automatic)
        return {dft::SemilocalFamily::Lda, 1.0,   1.0,
                automatic.functional_code, false, automatic.program};
    }
  }

#if GENERATIVEQC_HAS_CUDA
  if (input.semilocal_component_count == 2 && input.semilocal_range_omega == 0.0) {
    const auto& first = input.semilocal_components[0];
    const auto& second = input.semilocal_components[1];
    if (first.component_id && second.component_id && *first.component_id && *second.component_id &&
        first.coefficient == 1.0 && second.coefficient == 1.0) {
      const auto functional = dft::generated::split_hybrid_functional_code(
          std::string_view(first.component_id), std::string_view(second.component_id));
      if (functional) return {dft::SemilocalFamily::R2scan, 1.0, 1.0, functional, true};
    }
  }
#endif

  throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                    "KS semilocal primitive graph has no qualified native lowerer");
}

std::string_view expected_scf_domain(const NativeKsExecutionPlan& plan) noexcept {
  if (plan.automatic_program) return dft::generated::kAutomaticLibxcScfDomain;
  if (plan.generated_split_hybrid) return "libxc-7.0/split-global-hybrid-v1";
  return dft::semilocal_family_scf_domain(plan.semilocal_family);
}

/** Benchmark-only KS incremental Direct-J/K controls. The generic spelling is
 * capability-based; the legacy PBE0 spelling remains accepted for reproducibility. */
bool incremental_direct_jk_benchmark_requested(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0)
    return false;
  if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) return true;
  throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                    std::string(name) + " must be 0/off or 1/on");
}

std::optional<unsigned> incremental_direct_jk_benchmark_rebuild_interval(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) return std::nullopt;
  if (*value < '0' || *value > '9')
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      std::string(name) + " must be an unsigned integer");
  errno = 0;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno == ERANGE || end == value || *end != '\0' || parsed > UINT_MAX)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      std::string(name) + " must be an unsigned integer");
  return static_cast<unsigned>(parsed);
}

std::optional<double> incremental_direct_jk_benchmark_density_rms_threshold(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) return std::nullopt;
  errno = 0;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (errno == ERANGE || end == value || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      std::string(name) + " must be finite and nonnegative");
  return parsed;
}

scf::ScfOptions dft_options(const generativeqc_method_descriptor& descriptor,
                            generativeqc_backend backend, NativeKsExecutionPlan& execution_plan) {
  if (!std::isfinite(descriptor.energy_tolerance) || !std::isfinite(descriptor.density_tolerance) ||
      !std::isfinite(descriptor.screening_tolerance))
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "DFT tolerances must be finite");
  scf::ScfOptions options;
  options.preliminary_guess = scf::initial_guess::preliminary_options(descriptor.initial_guess);
  options.max_iterations = descriptor.max_iterations == 0 ? 100 : descriptor.max_iterations;
  options.diis_history = descriptor.diis_history == 0 ? 8 : descriptor.diis_history;
  options.energy_tolerance =
      descriptor.energy_tolerance > 0.0 ? descriptor.energy_tolerance : 1.0e-10;
  options.density_tolerance =
      descriptor.density_tolerance > 0.0 ? descriptor.density_tolerance : 1.0e-8;
  options.screening_tolerance =
      descriptor.screening_tolerance > 0.0 ? descriptor.screening_tolerance : 1.0e-12;

  const auto legacy_plan = legacy_ks_execution_plan(descriptor.method);
  const generativeqc_ks_options* ks_input = nullptr;
  SemilocalAdmission semilocal;
  if (descriptor.ks_options) {
    ks_input = descriptor.ks_options;
    if (ks_input->struct_size < sizeof(generativeqc_ks_options) ||
        ks_input->abi_version != GENERATIVEQC_ABI_VERSION)
      throw MethodError(GENERATIVEQC_STATUS_ABI_MISMATCH, "KS execution-plan ABI mismatch");
    if ((ks_input->spin_channels != 1 && ks_input->spin_channels != 2) ||
        (ks_input->exchange_terms == nullptr) != (ks_input->exchange_term_count == 0))
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "invalid compiler KS execution plan");
    semilocal = admit_semilocal(*ks_input);
    execution_plan = {ks_input->spin_channels, semilocal.family, true,
                      descriptor.method == GENERATIVEQC_METHOD_PBE_D4_RKS};
    execution_plan.functional = semilocal.functional;
    execution_plan.generated_split_hybrid = semilocal.generated_split_hybrid;
    execution_plan.automatic_program = semilocal.automatic_program;
    if (execution_plan.automatic_program &&
        (backend != GENERATIVEQC_BACKEND_CPU_REFERENCE || ks_input->exchange_term_count != 0 ||
         ks_input->has_nonlocal_correlation != 0))
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "automatic Libxc semilocal KS currently requires pure CPU execution");
    if (execution_plan.generated_split_hybrid && backend != GENERATIVEQC_BACKEND_CUDA)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "generated split-global-hybrid KS currently requires CUDA");
  } else {
    if (!legacy_plan)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "DFT execution requires a compiler-resolved KS plan");
    execution_plan = *legacy_plan;
    semilocal = {execution_plan.semilocal_family, 1.0, 1.0};
  }

  const auto mode = descriptor.density_fitting_mode;
  if (mode != GENERATIVEQC_DENSITY_FITTING_NONE &&
      mode != GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE &&
      mode != GENERATIVEQC_DENSITY_FITTING_CUDA && mode != GENERATIVEQC_DENSITY_FITTING_AUTO)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "unknown density-fitting execution mode");
  options.density_fitting_mode = mode;
  if ((mode == GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE &&
       backend != GENERATIVEQC_BACKEND_CPU_REFERENCE) ||
      (mode == GENERATIVEQC_DENSITY_FITTING_CUDA && backend != GENERATIVEQC_BACKEND_CUDA))
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DFT density-fitting backend must match the calculation backend");
  if (descriptor.density_fitting_auxiliary_basis != nullptr &&
      options.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DFT auxiliary basis requires density fitting");
  if (descriptor.density_fitting_relative_threshold != 0.0)
    options.density_fitting_relative_threshold = descriptor.density_fitting_relative_threshold;
  options.density_fitting_memory_budget_bytes = descriptor.density_fitting_memory_budget_bytes;
  if (descriptor.precision_mode != GENERATIVEQC_PRECISION_FP64 &&
      descriptor.precision_mode != GENERATIVEQC_PRECISION_AUTO)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "unknown floating-point precision mode");
  if (descriptor.precision_mode == GENERATIVEQC_PRECISION_AUTO &&
      backend != GENERATIVEQC_BACKEND_CUDA)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "DFT automatic precision currently requires CUDA");
  if (descriptor.precision_mode == GENERATIVEQC_PRECISION_AUTO && execution_plan.d4_correction)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "PBE-D4 currently requires strict FP64");
  options.precision_mode = descriptor.precision_mode;

  scf::FockBuildSpec fock;
  fock.spin =
      unrestricted(execution_plan) ? scf::FockSpin::Unrestricted : scf::FockSpin::Restricted;
  if (options.preliminary_guess && fock.spin != scf::FockSpin::Restricted)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "preliminary SCF requires a restricted target");
  fock.derivative_order = 0;
  fock.exchange.present = false;
  options.semilocal_exchange_scale = semilocal.exchange_scale;
  options.semilocal_correlation_scale = semilocal.correlation_scale;

  const generativeqc_ks_exchange_term* full_range = nullptr;
  const generativeqc_ks_exchange_term* short_range = nullptr;
  const generativeqc_ks_exchange_term* long_range = nullptr;
  if (ks_input) {
    const double divisor = unrestricted(execution_plan) ? 1.0 : 2.0;
    for (std::uint32_t i = 0; i < ks_input->exchange_term_count; ++i) {
      const auto& term = ks_input->exchange_terms[i];
      if (!std::isfinite(term.coefficient) || term.coefficient < 0.0 ||
          !std::isfinite(term.omega) || term.omega < 0.0 || !std::isfinite(term.fock_coefficient) ||
          term.fock_coefficient != -term.coefficient / divisor)
        throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                          "invalid KS exact-exchange contribution");
      switch (term.operator_kind) {
        case GENERATIVEQC_KS_EXCHANGE_FULL_RANGE:
          if (full_range || term.omega != 0.0)
            throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                              "invalid full-range exchange plan");
          full_range = &term;
          break;
        case GENERATIVEQC_KS_EXCHANGE_SHORT_RANGE:
          if (short_range || term.omega <= 0.0)
            throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                              "invalid short-range exchange plan");
          short_range = &term;
          break;
        case GENERATIVEQC_KS_EXCHANGE_LONG_RANGE:
          if (long_range || term.omega <= 0.0)
            throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                              "invalid long-range exchange plan");
          long_range = &term;
          break;
        default:
          throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "unknown KS exchange operator");
      }
    }
    if (full_range && (short_range || long_range))
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "KS execution cannot mix full- and range-separated exchange");
    if ((short_range == nullptr) != (long_range == nullptr))
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "range-separated exchange requires short- and long-range terms");
    if (short_range &&
        (ks_input->exchange_term_count != 2 || short_range->omega != long_range->omega))
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "range-separated exchange requires one shared omega");
    if (full_range && ks_input->exchange_term_count != 1)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "full-range exchange requires one contribution");
    if (full_range) {
      fock.exchange.present = full_range->coefficient != 0.0;
      fock.exchange.coefficient = full_range->fock_coefficient;
    } else if (short_range) {
      execution_plan.range_exchange = true;
      execution_plan.short_range_exchange = short_range->coefficient;
      execution_plan.long_range_exchange = long_range->coefficient;
      execution_plan.range_omega = short_range->omega;
      fock.exchange.present = short_range->coefficient != 0.0;
      fock.exchange.coefficient = short_range->fock_coefficient;
    }

    if (ks_input->has_nonlocal_correlation > 1)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "invalid nonlocal-correlation presence flag");
    if (ks_input->has_nonlocal_correlation) {
      dft::nlc::Vv10Variant variant;
      if (ks_input->nonlocal_variant == GENERATIVEQC_NONLOCAL_VV10)
        variant = dft::nlc::Vv10Variant::vv10;
      else if (ks_input->nonlocal_variant == GENERATIVEQC_NONLOCAL_RVV10)
        variant = dft::nlc::Vv10Variant::rvv10;
      else
        throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                          "invalid KS nonlocal-correlation variant");
      if (!std::isfinite(ks_input->nonlocal_b) || ks_input->nonlocal_b <= 0.0 ||
          !std::isfinite(ks_input->nonlocal_c) || ks_input->nonlocal_c <= 0.0 ||
          !std::isfinite(ks_input->nonlocal_coefficient) || ks_input->nonlocal_coefficient <= 0.0 ||
          !ks_input->nonlocal_maximum_bytes)
        throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                          "invalid KS nonlocal-correlation parameters or budget");
      execution_plan.nonlocal_correlation = true;
      execution_plan.nonlocal_parameters = {variant, ks_input->nonlocal_b, ks_input->nonlocal_c,
                                            ks_input->nonlocal_coefficient};
      execution_plan.nonlocal_maximum_bytes = ks_input->nonlocal_maximum_bytes;
    }
  }

  const auto& semilocal_metadata = dft::semilocal_family_metadata(execution_plan.semilocal_family);
  const bool complete_cuda_nonlocal = semilocal_metadata.cuda_nonlocal_correlation &&
                                      execution_plan.range_exchange &&
                                      execution_plan.nonlocal_correlation;
  const bool cuda_nonlocal = backend == GENERATIVEQC_BACKEND_CUDA && complete_cuda_nonlocal;
  const bool scaled_or_hybrid = options.semilocal_exchange_scale != 1.0 ||
                                options.semilocal_correlation_scale != 1.0 || fock.exchange.present;
  // AUTO is admitted per component by CudaKsPlan: Direct Coulomb J may use
  // mixed arithmetic while exact exchange K remains strict FP64. Density-fitted
  // global hybrids retain their separate strict-FP64 admission.
  const bool cuda_global_hybrid =
      backend == GENERATIVEQC_BACKEND_CUDA && !execution_plan.range_exchange &&
      !execution_plan.nonlocal_correlation &&
      (options.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE ||
       options.precision_mode != GENERATIVEQC_PRECISION_AUTO) &&
      fock.exchange.present;
  const double qualified_exact_exchange = semilocal_metadata.cuda_global_hybrid_exact_exchange;
  const double expected_semilocal_exchange =
      semilocal_metadata.component_coefficients_are_native_scales && qualified_exact_exchange > 0.0
          ? 1.0 - qualified_exact_exchange
          : 1.0;
  const double spin_divisor = fock.spin == scf::FockSpin::Restricted ? 2.0 : 1.0;
  const bool cuda_curated_global_hybrid =
      cuda_global_hybrid && qualified_exact_exchange > 0.0 &&
      options.semilocal_exchange_scale == expected_semilocal_exchange &&
      options.semilocal_correlation_scale == 1.0 &&
      fock.exchange.coefficient == -qualified_exact_exchange / spin_divisor;
  bool cuda_split_hybrid = false;
#if GENERATIVEQC_HAS_CUDA
  if (cuda_global_hybrid && execution_plan.generated_split_hybrid &&
      options.semilocal_exchange_scale == 1.0 && options.semilocal_correlation_scale == 1.0) {
    const auto composition =
        dft::generated::split_hybrid_composition(xc_functional_code(execution_plan));
    if (composition.matched && composition.exact_exchange_denominator) {
      const double exact_exchange = static_cast<double>(composition.exact_exchange_numerator) /
                                    static_cast<double>(composition.exact_exchange_denominator);
      const double divisor = unrestricted(execution_plan) ? 1.0 : 2.0;
      cuda_split_hybrid = fock.exchange.coefficient == -exact_exchange / divisor;
    }
  }
#endif
  if (scaled_or_hybrid && backend == GENERATIVEQC_BACKEND_CUDA && !execution_plan.range_exchange &&
      !cuda_curated_global_hybrid && !cuda_split_hybrid && !cuda_nonlocal)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "CUDA scaled/global-hybrid KS composition is not qualified");
  if (execution_plan.nonlocal_correlation && !semilocal_metadata.native_nonlocal_correlation)
    throw MethodError(
        GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
        "self-consistent nonlocal correlation has no lowerer for this semilocal graph");
  if (execution_plan.nonlocal_correlation && backend != GENERATIVEQC_BACKEND_CPU_REFERENCE &&
      !cuda_nonlocal)
    throw MethodError(
        GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
        "CUDA self-consistent nonlocal correlation lacks a qualified family capability");
  if (execution_plan.range_exchange && !semilocal_metadata.native_range_exchange)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "native KS range exchange has no lowerer for this semilocal graph");
  if (options.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE) {
    if (options.precision_mode == GENERATIVEQC_PRECISION_AUTO)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "DFT density fitting requires FP64 execution");
    fock.coulomb.approximation = scf::FockApproximation::DensityFitted;
    if (fock.exchange.present) fock.exchange.approximation = scf::FockApproximation::DensityFitted;
  }
  options.resolved_fock_build = scf::resolve_fock_build(
      fock, backend == GENERATIVEQC_BACKEND_CUDA ? scf::FockBackend::Cuda : scf::FockBackend::Cpu,
      options.screening_tolerance, options.density_fitting_relative_threshold);
  const bool generic_incremental_direct_jk =
      incremental_direct_jk_benchmark_requested("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK");
  const bool legacy_pbe0_incremental_direct_jk =
      incremental_direct_jk_benchmark_requested("GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK");
  if (generic_incremental_direct_jk || legacy_pbe0_incremental_direct_jk) {
    const bool strict_exact_cuda_ks =
        backend == GENERATIVEQC_BACKEND_CUDA &&
        options.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE &&
        options.precision_mode == GENERATIVEQC_PRECISION_FP64 &&
        scf::direct_jk_incremental_exact_eligible(*options.resolved_fock_build);
    if (!strict_exact_cuda_ks)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "KS incremental Direct-J/K benchmark mode requires strict-FP64 "
                        "exact-direct CUDA KS");
    if (legacy_pbe0_incremental_direct_jk) {
      const bool strict_exact_pbe0_rks =
          cuda_curated_global_hybrid && fock.spin == scf::FockSpin::Restricted &&
          execution_plan.semilocal_family == dft::SemilocalFamily::Pbe;
      if (!strict_exact_pbe0_rks)
        throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                          "legacy PBE0 incremental Direct-J/K selector requires CUDA RKS PBE0");
    }
    options.incremental_direct_jk = true;
    auto interval = incremental_direct_jk_benchmark_rebuild_interval(
        "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL");
    if (!interval)
      interval = incremental_direct_jk_benchmark_rebuild_interval(
          "GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK_REBUILD_INTERVAL");
    if (interval) options.incremental_direct_jk_rebuild_interval = *interval;
    auto threshold = incremental_direct_jk_benchmark_density_rms_threshold(
        "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK_DENSITY_RMS_THRESHOLD");
    if (!threshold)
      threshold = incremental_direct_jk_benchmark_density_rms_threshold(
          "GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK_DENSITY_RMS_THRESHOLD");
    if (threshold) options.incremental_direct_jk_density_rms_threshold = *threshold;
  }
  if (semilocal_metadata.molecular_nonlocal_domain) {
    if (!execution_plan.range_exchange || !execution_plan.nonlocal_correlation)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "WB97M-V requires complete B97M + SR/LR + VV10 primitives");
    const auto correction_backend =
        backend == GENERATIVEQC_BACKEND_CUDA ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
    const auto correction =
        scf::resolve_fock_build(scf::make_rsh_correction_fock_spec(
                                    fock.spin, execution_plan.short_range_exchange,
                                    execution_plan.long_range_exchange, execution_plan.range_omega),
                                correction_backend, options.screening_tolerance);
    scf::require_wb97mv_composition(*options.resolved_fock_build, correction,
                                    execution_plan.nonlocal_parameters);
  }
  options.compute_forces = false;
  return options;
}

/** Copy every pointee before constructing scientific owners. The public entry
 * has already admitted a complete current descriptor. A null KS-options pointer
 * retains the existing default GridSpec and tile values; production callers pass
 * the compiler-resolved grid, not a second native production profile. */
dft::GridSpec ks_grid_options(const generativeqc_method_descriptor& descriptor,
                              scf::ScfOptions& options,
                              const NativeKsExecutionPlan& execution_plan) {
  dft::GridSpec grid;
  if (!descriptor.ks_options) return grid;
  const auto& input = *descriptor.ks_options;
  if (input.struct_size < sizeof(generativeqc_ks_options) ||
      input.abi_version != GENERATIVEQC_ABI_VERSION)
    throw MethodError(GENERATIVEQC_STATUS_ABI_MISMATCH, "KS execution-plan ABI mismatch");
  if (!input.scf_domain ||
      std::string_view(input.scf_domain) != expected_scf_domain(execution_plan))
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "unsupported KS tail/spin domain policy");
  if (!input.tile_points || input.tile_points > static_cast<std::uint64_t>(INT_MAX))
    throw std::invalid_argument("invalid KS XC tile points");
  options.xc_tile_points = input.tile_points;
  switch (input.xc_execution_schedule) {
    case GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED:
      options.xc_execution_schedule = scf::ScfOptions::XcExecutionSchedule::DeviceFused;
      break;
    case GENERATIVEQC_XC_EXECUTION_HOST_UNFUSED:
      options.xc_execution_schedule = scf::ScfOptions::XcExecutionSchedule::HostUnfused;
      break;
    default:
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "unknown KS XC execution schedule");
  }
  grid.version = input.grid_version;
  grid.radial_points = input.radial_points;
  grid.angular_polar = input.angular_polar;
  grid.angular_azimuth = input.angular_azimuth;
  grid.partition_iterations = input.partition_iterations;
  grid.coincident_tolerance = input.coincident_tolerance;
  if ((input.element_radii == nullptr) != (input.element_radius_count == 0) ||
      (input.element_radii && input.element_radius_count != grid.element_radii.size()))
    throw std::invalid_argument("KS element radii require 119 entries or NULL/zero");
  if (input.element_radii) {
    for (std::size_t z = 1; z < grid.element_radii.size(); ++z) {
      const double radius = input.element_radii[z];
      if (!std::isfinite(radius) || (grid.version == 1 ? radius <= 0.0 : radius < 0.0))
        throw std::invalid_argument("invalid KS element radius");
      grid.element_radii[z] = grid.version == 1 && radius == 1.0 ? 0.0 : radius;
    }
  }
  dft::validate_grid_spec(grid);
  return grid;
}

Result adapt_result(scf::ScfResult native, generativeqc_backend backend) {
  Result result;
  result.preliminary_guess = native.preliminary_guess;
  result.energy = native.energy;
  result.convergence.iterations = native.iterations;
  result.convergence.energy_change = native.energy_change;
  result.convergence.residual_rms = native.density_rms;
  result.physical_residual_rms = native.physical_residual_rms;
  result.convergence.converged = native.converged;
  result.executed_backend = backend;
  result.fock_builds = native.fock_builds;
  result.precision = native.precision;
  result.incremental_direct_jk = native.incremental_direct_jk;
  result.precision_work = std::move(native.precision_work);
  native.dft_diagnostic.fock_builds = native.fock_builds;
  native.dft_diagnostic.initial_density_used = native.initial_density_used;
  // Move the snapshot instead of retaining another max-iteration history.
  result.ks_diagnostic = std::move(native.dft_diagnostic);
  return result;
}

/** The method's global ledger supplies the budget. Size the common direct
 * source explicitly so its standalone default cap is not a second KS limit. */
unsigned ks_direct_derivative_order(const scf::ResolvedFockBuild& strategy,
                                    generativeqc_backend backend) noexcept {
#if GENERATIVEQC_HAS_CUDA
  if (backend == GENERATIVEQC_BACKEND_CUDA && strategy.backend == scf::FockBackend::Cuda) {
    const auto exact = [](const scf::FockTermSpec& term) {
      return term.present && term.approximation == scf::FockApproximation::Exact;
    };
    if (exact(strategy.spec.coulomb) || exact(strategy.spec.exchange)) return 1;
  }
#else
  (void)strategy;
  (void)backend;
#endif
  return 0;
}

/** Retain the existing generated DF response only for the semilocal CUDA J
 * domain. Hybrid/RSH/nonlocal promotion remains a separate provider contract. */
unsigned ks_fitted_derivative_order(const scf::ResolvedFockBuild& strategy,
                                    generativeqc_backend backend) noexcept {
#if GENERATIVEQC_HAS_CUDA
  if (backend == GENERATIVEQC_BACKEND_CUDA && strategy.backend == scf::FockBackend::Cuda &&
      strategy.spec.coulomb.present &&
      strategy.spec.coulomb.approximation == scf::FockApproximation::DensityFitted &&
      !strategy.spec.exchange.present)
    return 1;
#else
  (void)strategy;
  (void)backend;
#endif
  return 0;
}

std::size_t ks_provider_bytes(const core::System& system, generativeqc_backend backend,
                              unsigned direct_derivative_order = 0) {
#if GENERATIVEQC_HAS_CUDA
  if (backend == GENERATIVEQC_BACKEND_CUDA) {
    std::size_t primitives = 0;
    for (const auto& shell : system.shells)
      primitives = runtime::add_capacity(primitives, shell.primitives.size());
    return scf::cuda_direct_coulomb_device_bytes(1, molecule::ao_count(system), system.atoms.size(),
                                                 system.shells.size(), primitives,
                                                 direct_derivative_order);
  }
#else
  (void)direct_derivative_order;
#endif
  return 0;
}

#if GENERATIVEQC_HAS_CUDA
KsTransportDiagnostic adapt_transfers(const dft::CudaKsTransfers& value) {
  return {value.setup_h2d_bytes,
          value.density_h2d_bytes,
          value.scalar_d2h_bytes,
          value.matrix_d2h_bytes,
          value.final_state_d2h_bytes,
          value.final_state_reads,
          value.synchronizations,
          value.iterations,
          value.occupation_stabilized_proposals};
}

void add_transfers(dft::CudaKsTransfers& target, const dft::CudaKsTransfers& value) {
  const auto add = [](std::uint64_t& destination, std::uint64_t increment) {
    if (increment > std::numeric_limits<std::uint64_t>::max() - destination)
      throw std::overflow_error("CUDA KS transport counter overflow");
    destination += increment;
  };
  add(target.setup_h2d_bytes, value.setup_h2d_bytes);
  add(target.density_h2d_bytes, value.density_h2d_bytes);
  add(target.scalar_d2h_bytes, value.scalar_d2h_bytes);
  add(target.matrix_d2h_bytes, value.matrix_d2h_bytes);
  add(target.final_state_d2h_bytes, value.final_state_d2h_bytes);
  add(target.final_state_reads, value.final_state_reads);
  add(target.synchronizations, value.synchronizations);
  add(target.iterations, value.iterations);
  add(target.occupation_stabilized_proposals, value.occupation_stabilized_proposals);
}
#endif

/** Own auxiliary shells and rebind centers when a batch item moves. The source
 * owner copies the result, so descriptor/temporary system lifetimes never leak
 * into a prepared calculation. An omitted auxiliary basis means the orbital basis. */
std::optional<core::System> ks_auxiliary_template(
    const generativeqc_method_descriptor& descriptor) {
  if (!descriptor.density_fitting_auxiliary_basis) return std::nullopt;
  return descriptor.density_fitting_auxiliary_basis->data;
}

void validate_ks_auxiliary_geometry(const core::System& system,
                                    const std::optional<core::System>& auxiliary) {
  if (!auxiliary) return;
  if (auxiliary->atoms.size() != system.atoms.size())
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DFT auxiliary basis atom count differs");
  for (std::size_t i = 0; i < system.atoms.size(); ++i)
    if (auxiliary->atoms[i].atomic_number != system.atoms[i].atomic_number ||
        auxiliary->atoms[i].position != system.atoms[i].position)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "DFT auxiliary basis must share the initial system geometry");
}

std::optional<core::System> ks_auxiliary_for_system(const core::System& system,
                                                    std::optional<core::System> auxiliary) {
  if (!auxiliary) return auxiliary;
  if (auxiliary->atoms.size() != system.atoms.size())
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DFT auxiliary basis atom count differs");
  for (std::size_t i = 0; i < system.atoms.size(); ++i) {
    if (auxiliary->atoms[i].atomic_number != system.atoms[i].atomic_number)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "DFT auxiliary basis atom ordering differs");
    auxiliary->atoms[i].position = system.atoms[i].position;
  }
  return auxiliary;
}

/** Backend selection must precede materialization: constructing the reference
 * grid and then uploading it hides cubic host work in CUDA preparation. */
dft::MolecularGrid ks_molecular_grid(const core::System& system, dft::GridSpec spec,
                                     generativeqc_backend backend, int device, bool retain_device) {
  if (backend == GENERATIVEQC_BACKEND_CUDA) {
#if GENERATIVEQC_HAS_CUDA
    return dft::MolecularGrid::from_cuda(system, spec, device, retain_device);
#else
    throw std::runtime_error("CUDA quadrature is unavailable in this build");
#endif
  }
  return dft::MolecularGrid(system, spec);
}

scf::FockOccupiedProjectionReservation ks_fitted_projection_reservation(
    const core::System& system, const NativeKsExecutionPlan& execution_plan,
    const scf::ResolvedFockBuild& strategy) {
  // This method owns an integer restricted determinant. Fixed-density Fock
  // callers do not acquire this promise from the same dimensions or system.
  const auto& spec = strategy.spec;
  if (execution_plan.spin_channels != 1 || execution_plan.range_exchange ||
      strategy.backend != scf::FockBackend::Cuda || spec.spin != scf::FockSpin::Restricted ||
      spec.derivative_order != 0 || !spec.coulomb.present || !spec.exchange.present ||
      spec.coulomb.approximation != scf::FockApproximation::DensityFitted ||
      spec.exchange.approximation != scf::FockApproximation::DensityFitted ||
      spec.exchange.op != scf::FockOperator::FullRange)
    return {};
  const auto [alpha, beta] = scf::initial_guess::spin_occupations(system);
  if (system.electron_count <= 0 || !alpha || alpha != beta || system.multiplicity != 1 ||
      alpha > molecule::ao_count(system))
    throw std::invalid_argument("KS projection reservation requires valid restricted occupations");
  return {alpha};
}

#if GENERATIVEQC_HAS_CUDA
bool cuda_rsh_provider_compatible(const scf::PreparedFockPlan& provider,
                                  const scf::ResolvedFockBuild& correction,
                                  const scf::PreparedFockPlan* range_provider) noexcept {
  const auto& primary = provider.strategy();
  const auto& primary_spec = primary.spec;
  const auto& correction_spec = correction.spec;
  const auto primary_binding = scf::prepared_cuda_fock_binding(provider);
  const bool correction_compatible =
      correction.backend == scf::FockBackend::Cuda && correction_spec.spin == primary_spec.spin &&
      correction_spec.derivative_order == 0 && !correction_spec.coulomb.present &&
      correction_spec.exchange.present &&
      correction_spec.exchange.approximation == scf::FockApproximation::Exact &&
      correction_spec.exchange.op == scf::FockOperator::LongRange &&
      std::isfinite(correction_spec.exchange.omega) && correction_spec.exchange.omega > 0.0 &&
      std::isfinite(correction_spec.exchange.coefficient) &&
      correction.screening_tolerance == primary.screening_tolerance;
  if (!primary_binding || primary.backend != scf::FockBackend::Cuda ||
      primary_spec.derivative_order != 0 || !primary_spec.coulomb.present ||
      primary_spec.coulomb.coefficient != 1.0 ||
      primary_spec.coulomb.op != scf::FockOperator::FullRange ||
      primary_spec.coulomb.omega != 0.0 || !correction_compatible)
    return false;

  const bool fitted_primary =
      primary_spec.coulomb.approximation == scf::FockApproximation::DensityFitted &&
      (!primary_spec.exchange.present ||
       (primary_spec.exchange.approximation == scf::FockApproximation::DensityFitted &&
        primary_spec.exchange.op == scf::FockOperator::FullRange &&
        primary_spec.exchange.omega == 0.0));
  if (fitted_primary) {
    if (!range_provider || range_provider->strategy() != correction ||
        !range_provider->matches_system(provider.system()))
      return false;
    const auto range_binding = scf::prepared_cuda_fock_binding(*range_provider);
    return range_binding && range_binding.device_id == primary_binding.device_id &&
           range_binding.nbf == primary_binding.nbf;
  }

  const bool exact_primary =
      primary_spec.coulomb.approximation == scf::FockApproximation::Exact &&
      (!primary_spec.exchange.present ||
       (primary_spec.exchange.approximation == scf::FockApproximation::Exact &&
        primary_spec.exchange.op == scf::FockOperator::FullRange &&
        primary_spec.exchange.omega == 0.0));
  return exact_primary && range_provider == nullptr;
}
#endif

class KsPreparedCalculation final : public PreparedCalculation {
 public:
  KsPreparedCalculation(Capabilities capabilities, core::System system,
                        NativeKsExecutionPlan execution_plan, scf::ScfOptions options,
                        dft::GridSpec grid, generativeqc_backend backend, int device,
                        const std::optional<core::System>& auxiliary)
      : capabilities_(capabilities),
        system_(std::move(system)),
        execution_plan_(execution_plan),
        options_(std::move(options)),
        backend_(backend),
        fock_(system_, auxiliary ? &*auxiliary : nullptr, *options_.resolved_fock_build, device,
              options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE
                  ? ks_provider_bytes(
                        system_, backend,
                        ks_direct_derivative_order(*options_.resolved_fock_build, backend))
                  : options_.density_fitting_memory_budget_bytes,
              options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE
                  ? ks_direct_derivative_order(*options_.resolved_fock_build, backend)
                  : 0U,
              options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE ? 0U : 1U,
              ks_fitted_projection_reservation(system_, execution_plan_,
                                               *options_.resolved_fock_build)),
        basis_(system_),
        grid_(ks_molecular_grid(
            system_, grid, backend_, device,
            options_.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::DeviceFused)) {
    options_.retain_ks_state = backend_ != GENERATIVEQC_BACKEND_CUDA;
    if (execution_plan_.range_exchange) prepare_range_exchange(device);
#if GENERATIVEQC_HAS_CUDA
    if (backend_ == GENERATIVEQC_BACKEND_CUDA && execution_plan_.range_exchange &&
        (!range_strategy_ ||
         !cuda_rsh_provider_compatible(fock_, *range_strategy_, range_correction_.get())))
      throw MethodError(
          GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
          "CUDA range-separated KS requires compatible primary and Direct LR providers");
#endif
    if (execution_plan_.nonlocal_correlation) prepare_nonlocal(device);
#if GENERATIVEQC_HAS_CUDA
    if (backend_ == GENERATIVEQC_BACKEND_CUDA) {
      const auto& metadata = dft::semilocal_family_metadata(execution_plan_.semilocal_family);
      if (execution_plan_.nonlocal_correlation && metadata.cuda_nonlocal_correlation &&
          options_.xc_execution_schedule != scf::ScfOptions::XcExecutionSchedule::DeviceFused)
        throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                          "public CUDA nonlocal KS requires device-fused XC/nonlocal execution");
      const auto* range = range_strategy_ ? &*range_strategy_ : nullptr;
      const auto domain = metadata.molecular_nonlocal_domain
                              ? dft::nlc::Vv10DensityDomain::MolecularV1
                              : dft::nlc::Vv10DensityDomain::StrictPositive;
      cuda_ = std::make_unique<dft::CudaKsPlan>(
          fock_, basis_, grid_, options_, xc_functional_code(execution_plan_),
          options_.xc_tile_points, range, nonlocal_.get(), domain, dft::CudaXcPreparationBudget{},
          range_correction_.get());
    }
#endif
    if (execution_plan_.d4_correction) prepare_d4(device);
    runtime::sample_cpu_capacity(host_numeric_capacity());
  }

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }
  const core::System& system() const noexcept { return system_; }
  const std::vector<scf::CudaDensityFittingMetricDiagnostic>& fitted_diagnostics() const noexcept {
    return fock_.diagnostic().fitted;
  }

  /** Explicit retained vectors; object metadata and transient setup are not
   * inferred from this lower-bound observation. Grid/basis buffers are owned. */
  std::size_t host_numeric_capacity() const noexcept {
    auto bytes =
        runtime::add_capacity(fock_.cpu_observation_capacity(),
                              runtime::vector_capacities(basis_.packed, grid_.points(),
                                                         grid_.weights(), grid_.owners(), warm_));
    if (range_correction_)
      bytes = runtime::add_capacity(bytes, range_correction_->cpu_observation_capacity());
    if (cpu_physical_)
      for (const auto* matrices : {&cpu_physical_->density, &cpu_physical_->fock})
        for (const auto& matrix : *matrices)
          bytes = runtime::add_capacity(bytes, runtime::vector_bytes(matrix));
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) bytes = runtime::add_capacity(bytes, cuda_->resources().retained_host_numeric_bytes);
#endif
    if (d4_) {
      const auto& resources = d4_->resources();
      bytes = runtime::add_capacity(bytes, static_cast<std::size_t>(resources.plan_host_bytes));
      bytes =
          runtime::add_capacity(bytes, static_cast<std::size_t>(resources.execution_host_bytes));
    }
    if (nonlocal_) {
      const auto& resources = nonlocal_->resources();
      bytes =
          runtime::add_capacity(bytes, static_cast<std::size_t>(resources.host_workspace_bytes));
    }
    return bytes;
  }

  /** Explicit output/rebuild export. Ordinary CUDA replays keep this on device. */
  std::vector<double> warm_density() {
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) return cuda_->warm_density();
#endif
    return warm_;
  }

  void clear_warm_start() noexcept {
    warm_.clear();
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) cuda_->clear_warm_start();
#endif
  }

  std::optional<std::vector<double>> prepare_cold_initial_guess(
      scf::initial_guess::PreliminaryDiagnostic& diagnostic) const {
    if (!options_.preliminary_guess) return std::nullopt;
    scf::initial_guess::validate_preliminary_target(system_, fock_.strategy(), options_);
    diagnostic.requested_kind = static_cast<std::uint32_t>(options_.preliminary_guess->kind);
    diagnostic.target_attempts = 1;
    const auto started = std::chrono::steady_clock::now();
    try {
      runtime::CpuRetainedCapacity retained(host_numeric_capacity());
      scf::initial_guess::EigenOperation eigen;
#if GENERATIVEQC_HAS_CUDA
      // The prepared CUDA owner already charges its idle solver workspace.
      // Borrow it synchronously rather than running CPU Jacobi for a CUDA seed.
      if (cuda_) eigen = cuda_->seed_eigen_operation();
#endif
      auto density = scf::initial_guess::prepare_preliminary_density(
          fock_, *options_.preliminary_guess, diagnostic, eigen);
      diagnostic.preparation_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      return density;
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception&) {
      diagnostic.outcome = scf::initial_guess::PreliminaryOutcome::PreparationFailed;
      diagnostic.work_counters_complete = false;
      diagnostic.preparation_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      return std::nullopt;
    }
  }

  void invalidate_result() override { invalidate_final_state(); }

  void invalidate_final_state() noexcept {
    cpu_physical_.reset();
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) cuda_->invalidate_final_state();
#endif
  }

#if GENERATIVEQC_HAS_CUDA
  dft::CudaKsPlan* cuda_plan() noexcept { return cuda_.get(); }
#endif

  std::optional<KsTransportDiagnostic> ks_transport_diagnostic() const override {
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) return adapt_transfers(cuda_->transfers());
#endif
    return std::nullopt;
  }

  generativeqc_status final_state_token(dft::CudaKsFinalStateToken& token,
                                        std::string& detail) const {
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) return cuda_->final_state_token(token, detail);
#endif
    token = {};
    if (!cpu_physical_) {
      detail = "CPU KS owner has no successful current final state";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    token = {1, cpu_physical_->identity};
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  generativeqc_status read_final_state(const dft::CudaKsFinalStateToken& expected,
                                       bool compute_weighted_density,
                                       dft::VerifiedKsFinalState& state, std::string& detail) {
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) return cuda_->read_final_state(expected, compute_weighted_density, state, detail);
#endif
    state = {};
    dft::CudaKsFinalStateToken current;
    const auto status = final_state_token(current, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (expected != current) {
      detail = "CPU KS final-state token is stale";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    // The SCF frame predates the last F[D] rebuild. Diagonalize that actual
    // retained physical F here; do not relabel the lagged orbital energies.
    // Explicit export costs one eigen solve and validation, zero Fock builds.
    const auto& ints = fock_.one_electron();
    const auto x = scf::reference::symmetric_orthogonalizer(ints.overlap, ints.nbf);
    std::vector<scf::reference::EigenResult> spins;
    spins.reserve(current.identity.model.spins);
    for (const auto& fock : cpu_physical_->fock)
      spins.push_back(scf::reference::generalized_eigen(fock, x, ints.nbf));
    dft::KsFinalStateCandidate candidate{current.identity,
                                         current.identity.determinant.factor.density_generation,
                                         true, std::move(spins)};
    scf::solver::FinalStateLimits limits{options_.density_tolerance, options_.energy_tolerance, 0,
                                         true};
    if (!dft::validate_ks_final_state(current.identity, ints.overlap, ints.hcore, *cpu_physical_,
                                      candidate, limits, compute_weighted_density, state, detail)) {
      invalidate_final_state();
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  generativeqc_status resident_density(const dft::CudaKsFinalStateToken& expected, int& device,
                                       const double*& alpha, const double*& beta,
                                       std::size_t& matrix_elements, unsigned& spins,
                                       void*& source_stream, std::string& detail) {
    device = -1;
    alpha = nullptr;
    beta = nullptr;
    matrix_elements = 0;
    spins = 0;
    source_stream = nullptr;
#if GENERATIVEQC_HAS_CUDA
    if (!cuda_) {
      detail = "resident final density requires a CUDA KS owner";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    dft::CudaKsResidentDensityBinding binding;
    const auto status = cuda_->resident_final_density(expected, binding, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (!binding) {
      detail = "CUDA KS resident final-density binding is invalid";
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    device = binding.device_id;
    alpha = binding.alpha;
    beta = binding.beta;
    matrix_elements = binding.matrix_elements;
    spins = binding.spins;
    source_stream = binding.stream;
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    detail = "resident final density requires a CUDA build";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  generativeqc_status fixed_density_profile(const dft::CudaKsFinalStateToken& expected,
                                            KsFixedDensityProfile& profile, std::string& detail) {
    profile = {};
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) {
      dft::CudaKsFixedDensityProfile native;
      const auto status = cuda_->profile_fixed_density_components(expected, native, detail);
      if (status == GENERATIVEQC_STATUS_SUCCESS) {
        profile.milliseconds = native.milliseconds;
        profile.present_mask = native.present_mask;
      }
      return status;
    }
#endif
    detail = "fixed-density component profiling requires a CUDA KS owner";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }

  generativeqc_status resident_grid(const dft::CudaKsFinalStateToken& expected, int& device,
                                    const double*& points, const double*& weights,
                                    const double*& atomic_weights, std::size_t& point_count,
                                    std::string& detail) {
    device = -1;
    points = nullptr;
    weights = nullptr;
    atomic_weights = nullptr;
    point_count = 0;
#if GENERATIVEQC_HAS_CUDA
    if (!cuda_) {
      detail = "resident molecular grid requires a CUDA KS owner";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    // Reuse the exact final-density lease as the zero-transfer token gate. The
    // grid is immutable for this prepared KS owner and survives for its lifetime.
    dft::CudaKsResidentDensityBinding density_binding;
    const auto status = cuda_->resident_final_density(expected, density_binding, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    const auto grid = grid_.cuda_view();
    if (density_binding && !grid &&
        options_.xc_execution_schedule == scf::ScfOptions::XcExecutionSchedule::HostUnfused) {
      detail = "host-unfused KS owner has no resident molecular grid";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    if (!density_binding || !grid || grid.device != density_binding.device_id) {
      detail = "CUDA KS resident molecular-grid binding is invalid";
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    device = grid.device;
    points = grid.points;
    weights = grid.weights;
    atomic_weights = grid.atomic_weights;
    point_count = grid.point_count;
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    detail = "resident molecular grid requires a CUDA build";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  generativeqc_status resident_nonlocal_features(const dft::CudaKsFinalStateToken& expected,
                                                 int& device, const double*& density,
                                                 const double*& gradient, std::size_t& point_count,
                                                 void*& source_stream, std::string& detail) {
    device = -1;
    density = nullptr;
    gradient = nullptr;
    point_count = 0;
    source_stream = nullptr;
#if GENERATIVEQC_HAS_CUDA
    if (!cuda_) {
      detail = "resident nonlocal features require a CUDA KS owner";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    dft::CudaKsResidentNonlocalFeaturesBinding binding;
    const auto status = cuda_->resident_final_nonlocal_features(expected, binding, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (!binding) {
      detail = "CUDA KS resident nonlocal feature binding is invalid";
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    device = binding.device_id;
    density = binding.density;
    gradient = binding.gradient;
    point_count = binding.point_count;
    source_stream = binding.stream;
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    detail = "resident nonlocal features require a CUDA build";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  generativeqc_status read_derivative_state(const dft::CudaKsFinalStateToken& expected,
                                            KsDerivativeSnapshot& output, std::string& detail) {
    output = {};
    dft::VerifiedKsFinalState state;
#if GENERATIVEQC_HAS_CUDA
    const auto before = cuda_ ? cuda_->transfers() : dft::CudaKsTransfers{};
#endif
    const auto status = read_final_state(expected, true, state, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    // These are the provider's actual metric and the collocation/grid sources
    // used by this immutable KS owner, not caller-supplied identity labels.
    output = {std::move(state), system_,        fock_.one_electron().overlap,
              basis_.packed,    grid_.points(), grid_.weights(),
              grid_.owners()};
    // Both backends collocate this owner's exact host-built quadrature. Export
    // its raw measures directly; dividing partitioned weights loses tail data.
    output.atomic_weights = grid_.atomic_weights();
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) {
      const auto after = cuda_->transfers();
      output.export_d2h_bytes = after.final_state_d2h_bytes - before.final_state_d2h_bytes;
      output.export_reads = after.final_state_reads - before.final_state_reads;
      output.export_synchronizations = after.synchronizations - before.synchronizations;
    }
#endif
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  generativeqc_status density_fitted_integral_gradient(
      const dft::CudaKsFinalStateToken& expected,
      const std::vector<scf::reference::Matrix>& density,
      const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail) {
    output.clear();
    work = {};
    if (options_.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE ||
        !system_.ecp_terms.empty() || execution_plan_.range_exchange ||
        execution_plan_.nonlocal_correlation) {
      detail = "density-fitted stationary derivatives require all-electron full-range KS";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    const auto& strategy = fock_.strategy();
    const auto fitted_term = [](const scf::FockTermSpec& term) {
      return !term.present || (term.approximation == scf::FockApproximation::DensityFitted &&
                               term.op == scf::FockOperator::FullRange);
    };
    if (!fitted_term(strategy.spec.coulomb) || !fitted_term(strategy.spec.exchange)) {
      detail = "density-fitted stationary derivative provider identity mismatch";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    dft::CudaKsFinalStateToken current;
    auto status = final_state_token(current, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (current != expected) {
      detail = "density-fitted stationary derivative token is stale";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    const auto spins = expected.identity.model.spins;
    const auto& one = fock_.one_electron();
    const auto matrix_elements = one.nbf * one.nbf;
    const auto coordinates = 3 * system_.atoms.size();
    if (spins < 1 || spins > 2 || density.size() != spins || weighted_density.size() != spins ||
        one.ncoord != coordinates || one.hcore_derivative.size() != coordinates * matrix_elements ||
        one.overlap_derivative.size() != coordinates * matrix_elements) {
      detail = "density-fitted stationary D/W or one-electron derivative shape mismatch";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    for (unsigned spin = 0; spin < spins; ++spin)
      if (density[spin].size() != matrix_elements ||
          weighted_density[spin].size() != matrix_elements) {
        detail = "density-fitted stationary D/W AO shape mismatch";
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      }
    // H'/Pulay/J'/K' vectors coexist with the four-source publication buffer.
    // This allowance covers only compact publication, not DF-provider scratch.
    if (!maximum_bytes || coordinates > maximum_bytes / (8U * sizeof(double))) {
      detail = "density-fitted stationary derivative publication exceeds its budget";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }

    std::vector<double> hcore, pulay;
    std::size_t one_electron_device_bytes = 0;
    std::size_t one_electron_h2d_bytes = 0;
    std::size_t one_electron_d2h_bytes = 0;
    bool resident_one_electron = false;
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) {
      scf::OneElectronGradientResources one_electron;
      dft::CudaKsResidentStationaryWeightsBinding resident_weights;
      const auto resident_status =
          cuda_->resident_final_stationary_weights(expected, resident_weights, detail);
      if (resident_status == GENERATIVEQC_STATUS_SUCCESS) {
        if (!resident_weights || resident_weights.device_id != expected.identity.model.device ||
            resident_weights.matrix_elements != matrix_elements ||
            resident_weights.spins != spins) {
          detail = "CUDA DF stationary D/W binding is incompatible with the final KS state";
          return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
        }
        const auto one_status = scf::execute_cuda_stationary_one_electron_pair(
            resident_weights.device_id, system_, {}, {}, 0, maximum_bytes, hcore, pulay, detail,
            &one_electron, resident_weights.density, resident_weights.weighted_density);
        if (one_status == GENERATIVEQC_STATUS_SUCCESS) {
          resident_one_electron = true;
          one_electron_device_bytes = one_electron.device_bytes;
          one_electron_h2d_bytes = one_electron.host_to_device_bytes;
          one_electron_d2h_bytes = one_electron.device_to_host_bytes;
        } else if (one_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                   one_status != GENERATIVEQC_STATUS_OUT_OF_MEMORY) {
          return one_status;
        }
      } else if (resident_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
        return resident_status;
      }
    }
#endif
    if (!resident_one_electron) {
      // CPU and bounded CUDA fallback preserve the established exact host
      // contraction. CUDA only reaches this branch when the optional resident
      // one-electron consumer cannot be admitted; J/K response ownership is
      // unaffected.
      hcore.assign(coordinates, 0.0);
      pulay.assign(coordinates, 0.0);
      for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
        const auto* dh = one.hcore_derivative.data() + coordinate * matrix_elements;
        const auto* ds = one.overlap_derivative.data() + coordinate * matrix_elements;
        for (unsigned spin = 0; spin < spins; ++spin)
          for (std::size_t item = 0; item < matrix_elements; ++item) {
            hcore[coordinate] += density[spin][item] * dh[item];
            pulay[coordinate] -= weighted_density[spin][item] * ds[item];
          }
      }
      detail.clear();
    }

    const std::vector<double> empty;
    scf::FockEnergyDerivativeComponents two;
    std::optional<scf::CudaDfBorrowedResponseDensity> response_density;
    std::optional<scf::CudaDfBorrowedFittedProjection> fitted_projection;
#if GENERATIVEQC_HAS_CUDA
    if (cuda_ && spins == 1) {
      dft::CudaKsResidentDensityBinding lease;
      std::string lease_detail;
      const auto lease_status = cuda_->resident_final_density(expected, lease, lease_detail);
      if (lease_status == GENERATIVEQC_STATUS_SUCCESS) {
        if (!lease || lease.spins != 1 || lease.matrix_elements != matrix_elements ||
            lease.device_id != expected.identity.model.device) {
          detail = "CUDA KS returned an incompatible final density lease for DF response";
          return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
        }
        response_density.emplace(scf::CudaDfBorrowedResponseDensity{
            lease.device_id, lease.alpha, lease.matrix_elements, lease.stream});
      } else if (lease_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED) {
        detail = lease_detail;
        return lease_status;
      }
    }
    if (cuda_ && spins == 1 && strategy.spec.exchange.present) {
      dft::CudaKsResidentFittedProjectionBinding lease;
      std::string lease_detail;
      const auto lease_status =
          cuda_->resident_final_fitted_projection(expected, lease, lease_detail);
      if (lease_status == GENERATIVEQC_STATUS_SUCCESS) {
        if (!lease) {
          detail = "CUDA KS returned an invalid final fitted projection lease";
          return GENERATIVEQC_STATUS_INTERNAL_ERROR;
        }
        fitted_projection.emplace(scf::CudaDfBorrowedFittedProjection{
            lease.device_id, lease.occupied_coefficients, lease.projection, lease.nbf, lease.naux,
            lease.rank, lease.stream});
      } else if (lease_status != GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
                 lease_status != GENERATIVEQC_STATUS_INVALID_ARGUMENT) {
        detail = lease_detail;
        return lease_status;
      }
    }
#endif
    try {
      two = response_density ? fock_.energy_derivative_components_with_cuda_df_state(
                                   density[0], empty, &*response_density,
                                   fitted_projection ? &*fitted_projection : nullptr)
            : fitted_projection
                ? fock_.energy_derivative_components_with_fitted_projection(density[0], empty,
                                                                            *fitted_projection)
                : fock_.energy_derivative_components(density[0], spins == 2 ? density[1] : empty);
    } catch (const std::bad_alloc&) {
      detail = "density-fitted stationary response exceeded the prepared resource budget";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument& error) {
      detail = error.what();
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
      detail = error.what();
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
    if (two.coulomb.size() != coordinates || two.exchange.size() != coordinates) {
      detail = "density-fitted stationary J/K derivative source shape mismatch";
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    std::vector<double> candidate;
    candidate.reserve(4 * coordinates);
    std::size_t publication_remaining_bytes = maximum_bytes;
    for (const auto* values : {&hcore, &pulay, &two.coulomb, &two.exchange, &candidate}) {
      if (values->capacity() > publication_remaining_bytes / sizeof(double)) {
        detail = "density-fitted stationary derivative publication exceeds its budget";
        return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
      }
      publication_remaining_bytes -= values->capacity() * sizeof(double);
    }
    const auto publication_peak_bytes = maximum_bytes - publication_remaining_bytes;
    candidate.insert(candidate.end(), hcore.begin(), hcore.end());
    candidate.insert(candidate.end(), pulay.begin(), pulay.end());
    candidate.insert(candidate.end(), two.coulomb.begin(), two.coulomb.end());
    candidate.insert(candidate.end(), two.exchange.begin(), two.exchange.end());
    if (!std::all_of(candidate.begin(), candidate.end(),
                     [](double value) { return std::isfinite(value); })) {
      detail = "density-fitted stationary derivative source is nonfinite";
      return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    }
    // H'/S' derivative tensors remain retained host provider data. CUDA may
    // instead contract the exact final resident D/W with the paired device
    // consumer; slots 2/4/5 report that consumer's device peak and actual
    // metadata/output movement. DF J/K response scratch/transfers remain
    // separate and are not inferred from spin dimensions.
    work[0] = fock_.diagnostic().device_bytes;
    work[2] = one_electron_device_bytes;
    work[3] = publication_peak_bytes;
    work[4] = one_electron_h2d_bytes;
    work[5] = one_electron_d2h_bytes;
    output = std::move(candidate);
    detail.clear();
    return GENERATIVEQC_STATUS_SUCCESS;
  }

  generativeqc_status cuda_full_range_integral_derivatives(
      const dft::CudaKsFinalStateToken& expected, std::vector<double>& output,
      std::string& detail) {
#if GENERATIVEQC_HAS_CUDA
    output.clear();
    if (!cuda_ || !system_.ecp_terms.empty() || range_strategy_ ||
        options_.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE) {
      detail = "CUDA full-range stationary shell derivatives require an all-electron Direct owner";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    const auto derivative_source = scf::prepared_cuda_direct_derivative_binding(fock_);
    if (!derivative_source) {
      detail = "prepared CUDA Fock owner has no retained Direct shell derivative lease";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    dft::CudaKsResidentDensityBinding resident_density;
    auto status = cuda_->resident_final_density(expected, resident_density, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (!resident_density || resident_density.device_id != derivative_source.device_id ||
        resident_density.matrix_elements != derivative_source.nbf * derivative_source.nbf) {
      detail = "CUDA full-range stationary density is incompatible with the Direct shell owner";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    status = scf::execute_prepared_cuda_direct_shell_full_range_derivatives_device(
        fock_, resident_density.alpha, resident_density.beta, resident_density.matrix_elements,
        output, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) {
      output.clear();
      return status;
    }
    const auto coordinates = 3 * system_.atoms.size();
    if (output.size() != 2 * coordinates) {
      output.clear();
      detail = "prepared CUDA full-range shell derivative source shape mismatch";
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    (void)expected;
    (void)output;
    detail = "CUDA full-range stationary shell derivatives are unavailable in this build";
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  generativeqc_status cuda_integral_gradient(
      const dft::CudaKsFinalStateToken& expected, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail,
      const std::vector<scf::reference::Matrix>* cached_density = nullptr,
      const std::vector<scf::reference::Matrix>* cached_weighted_density = nullptr,
      bool combined_two_electron = false) {
#if GENERATIVEQC_HAS_CUDA
    if (!cuda_ || !system_.ecp_terms.empty()) return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const bool fitted = options_.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE;
    if (fitted && ks_fitted_derivative_order(fock_.strategy(), backend_) == 0) {
      detail =
          "CUDA density-fitted stationary derivatives require the qualified semilocal DF-J domain";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    const bool range_exchange = execution_plan_.range_exchange;
    if (combined_two_electron && (fitted || range_exchange)) {
      detail = "combined stationary two-electron derivative requires full-range Direct sources";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    if (range_exchange != range_strategy_.has_value()) {
      detail = "CUDA stationary integral gradient has inconsistent range-exchange ownership";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    if ((cached_density == nullptr) != (cached_weighted_density == nullptr)) {
      detail = "cached CUDA stationary D/W must be supplied together";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    const auto transfers_before = cuda_->transfers();
    dft::VerifiedKsFinalState exported_state;
    generativeqc_status status = GENERATIVEQC_STATUS_SUCCESS;
    if (!cached_density) {
      status = read_final_state(expected, true, exported_state, detail);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      cached_density = &exported_state.density;
      cached_weighted_density = &exported_state.weighted_density;
    }
    const auto& model = expected.identity.model;
    const auto device = model.device;
    const auto bytes = maximum_bytes;
    const auto derivative_source = scf::prepared_cuda_direct_derivative_binding(fock_);
    if (!fitted && !derivative_source) {
      detail = "CUDA integral gradient requires a retained prepared Direct derivative source";
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    }
    const auto spins = model.spins;
    const auto nbf = fock_.diagnostic().nbf;
    const auto matrix_elements = nbf * nbf;
    if (!nbf) {
      detail = "CUDA stationary derivative source has an empty AO basis";
      return GENERATIVEQC_STATUS_INTERNAL_ERROR;
    }
    const auto valid_cached = [&](const auto& blocks) {
      return blocks.size() == spins &&
             std::all_of(blocks.begin(), blocks.end(),
                         [&](const auto& matrix) { return matrix.size() == matrix_elements; });
    };
    if (!valid_cached(*cached_density) || !valid_cached(*cached_weighted_density)) {
      detail = "cached CUDA stationary D/W has an incompatible spin or AO shape";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    if (!fitted && (derivative_source.device_id != device || derivative_source.nbf != nbf)) {
      detail = "CUDA stationary Direct derivative source disagrees with the live KS owner";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    dft::CudaKsResidentDensityBinding resident_density;
    status = cuda_->resident_final_density(expected, resident_density, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (!resident_density || resident_density.device_id != device ||
        resident_density.matrix_elements != matrix_elements || resident_density.spins != spins) {
      detail = "CUDA stationary derivative density is incompatible with the prepared Fock owner";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    dft::CudaKsResidentStationaryWeightsBinding resident_weights;
    status = cuda_->resident_final_stationary_weights(expected, resident_weights, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (!resident_weights || resident_weights.device_id != device ||
        resident_weights.matrix_elements != matrix_elements || resident_weights.spins != spins) {
      detail = "CUDA stationary one-electron D/W is incompatible with the prepared Fock owner";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    const auto transfers_after = cuda_->transfers();
    // Both Direct and DF sources belong to the already-budgeted SCF owner.
    // Report retained provider bytes without charging them again to the
    // stationary consumer's additional-device allowance.
    work = {fitted ? fock_.diagnostic().device_bytes : derivative_source.retained_device_bytes,
            0,
            0,
            0,
            0,
            0,
            transfers_after.final_state_d2h_bytes - transfers_before.final_state_d2h_bytes,
            transfers_after.final_state_reads - transfers_before.final_state_reads,
            transfers_after.synchronizations - transfers_before.synchronizations};
    scf::OneElectronGradientResources one;
    const auto nc = 3 * system_.atoms.size();
    std::vector<double> candidate;
    candidate.reserve((combined_two_electron ? 3 : range_exchange ? 5 : 4) * nc);
    std::vector<double> hcore, pulay, value;
    status = GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    if (!fitted)
      status = scf::execute_prepared_cuda_stationary_one_electron_pair(
          fock_.cuda_direct_source(), resident_weights.density, resident_weights.weighted_density,
          resident_weights.matrix_elements, bytes, hcore, pulay, detail, &one);
    if (status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED)
      status = scf::execute_cuda_stationary_one_electron_pair(
          device, system_, {}, {}, 0, bytes, hcore, pulay, detail, &one, resident_weights.density,
          resident_weights.weighted_density);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    work[2] = std::max<std::uint64_t>(work[2], one.device_bytes);
    work[3] = std::max<std::uint64_t>(work[3], one.host_numeric_bytes);
    work[4] += one.host_to_device_bytes;
    work[5] += one.device_to_host_bytes;
    candidate.insert(candidate.end(), hcore.begin(), hcore.end());
    candidate.insert(candidate.end(), pulay.begin(), pulay.end());
    if (fitted) {
      try {
        value = spins == 1 ? fock_.retained_energy_derivative(cached_density->at(0))
                           : fock_.retained_energy_derivative(cached_density->at(0),
                                                              cached_density->at(1));
      } catch (const std::bad_alloc&) {
        detail = "CUDA density-fitted stationary response exceeded its retained budget";
        return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
      } catch (const std::invalid_argument& error) {
        detail = error.what();
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      } catch (const std::exception& error) {
        detail = error.what();
        return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
      }
      if (value.size() != nc) {
        detail = "CUDA density-fitted stationary response returned the wrong coordinate shape";
        return GENERATIVEQC_STATUS_INTERNAL_ERROR;
      }
      candidate.insert(candidate.end(), value.begin(), value.end());
      // Semilocal DF is deliberately J-only. Keep the stationary runtime's
      // canonical four-source layout without manufacturing exchange work.
      candidate.insert(candidate.end(), nc, 0.0);
    } else {
      status = range_exchange
                   ? scf::execute_prepared_cuda_direct_rsh_energy_derivatives_device(
                         fock_, *range_strategy_, resident_density.alpha, resident_density.beta,
                         resident_density.matrix_elements, value, detail)
                   : scf::execute_prepared_cuda_direct_shell_full_range_derivatives_device(
                         fock_, resident_density.alpha, resident_density.beta,
                         resident_density.matrix_elements, value, detail, !combined_two_electron);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
      candidate.insert(candidate.end(), value.begin(), value.end());
    }
    output = std::move(candidate);
    return GENERATIVEQC_STATUS_SUCCESS;
#else
    return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
  }

  Result execute(bool compute_forces) override {
    invalidate_final_state();
    const char* method_name = semilocal_family_name(execution_plan_);
    if (compute_forces) {
      const char* issue =
          execution_plan_.automatic_program
              ? "#1122"
              : (dft::semilocal_family_requires_tau(execution_plan_.semilocal_family) ? "#164"
                                                                                      : "#163");
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        std::string(method_name) +
                            " KS nuclear gradients are tracked separately in issue " + issue);
    }
    auto result = adapt_result(run(nullptr, true, true), backend_);
    apply_d4(result);
    return result;
  }

  void apply_d4(Result& result) {
    if (!d4_) return;
    std::vector<double> coordinates;
    coordinates.reserve(3 * system_.atoms.size());
    for (const auto& atom : system_.atoms)
      coordinates.insert(coordinates.end(), atom.position.begin(), atom.position.end());
    const std::uint8_t active = 1;
    const std::uint8_t want_gradient = 0;
    std::vector<dft::dispersion::D4Status> statuses;
    std::vector<double> components, gradients, charges;
    std::string detail;
    const auto status =
        d4_->execute(coordinates, std::span(&active, 1), std::span(&want_gradient, 1), statuses,
                     components, gradients, charges, detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS || statuses.size() != 1 ||
        statuses[0] != dft::dispersion::D4Status::success || components.size() != 2)
      throw MethodError(
          status == GENERATIVEQC_STATUS_SUCCESS ? GENERATIVEQC_STATUS_NUMERICAL_FAILURE : status,
          detail.empty() ? "PBE-D4 correction failed" : detail);
    result.energy += components[0] + components[1];
  }

  /** Single-system and native batch paths share the same scientific owner. */
  scf::ScfResult run(const std::vector<double>* initial_density, bool reuse_warm, bool update_warm,
                     bool allow_preliminary = true) {
    invalidate_final_state();
#if GENERATIVEQC_HAS_CUDA
    if (cuda_) {
      // Native iterations read only scalar diagnostics. The public energy
      // result does not require a final AO matrix download; warm D stays resident.
      cuda_->set_warm_start_updates(update_warm);
      scf::initial_guess::PreliminaryDiagnostic diagnostic;
      std::optional<std::vector<double>> prepared;
      const bool policy = allow_preliminary && options_.preliminary_guess.has_value();
      const bool existing = initial_density != nullptr || (reuse_warm && cuda_->has_warm_start());
      if (policy) {
        diagnostic.requested_kind = static_cast<std::uint32_t>(options_.preliminary_guess->kind);
        diagnostic.target_attempts = 1;
        if (existing) {
          diagnostic.outcome = scf::initial_guess::PreliminaryOutcome::ExplicitDensity;
        } else {
          prepared = prepare_cold_initial_guess(diagnostic);
        }
      }
      const auto* seed = initial_density ? initial_density : (prepared ? &*prepared : nullptr);
      scf::ScfResult native;
      try {
        native = cuda_->run(seed, reuse_warm && !prepared, false);
      } catch (const generativeqc::Error& error) {
        // CUDA helpers may also throw untyped runtime errors for device faults.
        // Only an explicitly numerical failure may consume the seed retry.
        if (!prepared || error.status() != GENERATIVEQC_STATUS_NUMERICAL_FAILURE) throw;
        diagnostic.work_counters_complete = false;
      }
      // Returned physical failures and nonconvergence retain a complete work
      // census; a typed numerical exception above has no terminal result.
      if (prepared && (!native.converged || cuda_->failed())) {
        diagnostic.discarded_target_iterations = native.iterations;
        diagnostic.discarded_target_fock_builds = native.fock_builds;
        diagnostic.outcome = scf::initial_guess::PreliminaryOutcome::TargetRetried;
        diagnostic.target_attempts = 2;
        prepared.reset();
        // Release the discarded result's history before the new solve. begin()
        // revokes its final state and resets DIIS while preserving last-good warm D.
        native = {};
        native = cuda_->run(nullptr, false, false);
      }
      if (policy) native.preliminary_guess = diagnostic;
      if (cuda_->failed())
        throw MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                          "CUDA KS physical evaluation failed");
      return native;
    }
#endif
    if (cpu_epoch_ == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("CPU KS solve epoch exhausted");
    ++cpu_epoch_;
    const auto* seed =
        initial_density ? initial_density : (reuse_warm && !warm_.empty() ? &warm_ : nullptr);
    // The CPU driver already samples its provider/grid. Add only the retained
    // last-good density, which coexists with its current/proposed densities.
    runtime::CpuRetainedCapacity retained_warm(runtime::vector_bytes(warm_));
    auto execution_options = options_;
    if (!allow_preliminary) execution_options.preliminary_guess.reset();
    scf::ScfResult native;
    if (execution_plan_.automatic_program) {
      native = unrestricted(execution_plan_)
                   ? scf::run_semilocal_uks(fock_, basis_, grid_, execution_options,
                                            *execution_plan_.automatic_program, seed)
                   : scf::run_semilocal_rks(fock_, basis_, grid_, execution_options,
                                            *execution_plan_.automatic_program, seed);
    } else if (execution_plan_.generated_split_hybrid)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "generated split-global-hybrid CPU KS is unavailable");
    else if (dft::semilocal_family_uses_molecular_nonlocal_domain(
                 execution_plan_.semilocal_family)) {
      if (!range_correction_ || !nonlocal_)
        throw std::runtime_error("WB97M-V requires both range-exchange and nonlocal owners");
      native = unrestricted(execution_plan_)
                   ? scf::run_wb97mv_uks(fock_, *range_correction_, basis_, grid_,
                                         execution_options, *nonlocal_, seed)
                   : scf::run_wb97mv_rks(fock_, *range_correction_, basis_, grid_,
                                         execution_options, *nonlocal_, seed);
    } else if (execution_plan_.range_exchange) {
      if (!range_correction_)
        throw std::runtime_error("KS range-exchange correction owner is missing");
      native = unrestricted(execution_plan_)
                   ? scf::run_pbe_rsh_uks(fock_, *range_correction_, basis_, grid_,
                                          execution_options, seed, nonlocal_.get())
                   : scf::run_pbe_rsh_rks(fock_, *range_correction_, basis_, grid_,
                                          execution_options, seed, nonlocal_.get());
    } else if (execution_plan_.semilocal_family == dft::SemilocalFamily::B3lyp)
      native = unrestricted(execution_plan_)
                   ? scf::run_b3lyp_uks(fock_, basis_, grid_, execution_options, seed)
                   : scf::run_b3lyp_rks(fock_, basis_, grid_, execution_options, seed);
    else if (execution_plan_.semilocal_family == dft::SemilocalFamily::Pbe && nonlocal_)
      native =
          unrestricted(execution_plan_)
              ? scf::run_pbe_uks_nonlocal(fock_, basis_, grid_, execution_options, seed, *nonlocal_)
              : scf::run_pbe_rks_nonlocal(fock_, basis_, grid_, execution_options, seed,
                                          *nonlocal_);
    else
      native = scf::run_curated_semilocal_ks(fock_, basis_, grid_, execution_options,
                                             execution_plan_.semilocal_family,
                                             execution_plan_.spin_channels, seed);
    // This owner has immutable model/geometry/spin identity. Only successful
    // executions may replace its compatible last-good density; DIIS is fresh.
    if (native.converged && options_.retain_ks_state) {
      const auto spins = execution_plan_.spin_channels;
      const auto matrix = fock_.one_electron().nbf * fock_.one_electron().nbf;
      if (native.density.size() != spins * matrix ||
          native.ks_physical_fock.size() != spins * matrix ||
          (spins == 2 && native.dft_diagnostic.occupations.size() != spins))
        throw std::runtime_error("CPU KS retained spin-state shape mismatch");
      std::vector<scf::reference::Matrix> densities, focks;
      densities.reserve(spins);
      focks.reserve(spins);
      for (unsigned spin = 0; spin < spins; ++spin) {
        const auto begin = spin * matrix;
        densities.emplace_back(native.density.begin() + begin,
                               native.density.begin() + begin + matrix);
        focks.emplace_back(native.ks_physical_fock.begin() + begin,
                           native.ks_physical_fock.begin() + begin + matrix);
      }
      std::vector<std::size_t> occupied;
      if (spins == 1)
        occupied = {static_cast<std::size_t>(system_.electron_count / 2)};
      else
        occupied.assign(native.dft_diagnostic.occupations.begin(),
                        native.dft_diagnostic.occupations.end());
      dft::KsFinalStateIdentity identity;
      identity.determinant = {
          {cpu_owner_, 1, 1, 1}, cpu_epoch_, fock_.strategy(), std::move(occupied)};
      identity.model = {1,
                        scf_domain_version(execution_plan_),
                        grid_.spec(),
                        options_.xc_tile_points,
                        xc_functional_code(execution_plan_),
                        spins,
                        -1,
                        cpu_owner_,
                        options_.semilocal_exchange_scale,
                        options_.semilocal_correlation_scale};
      if (range_correction_) identity.model.range_correction = range_correction_->strategy();
      if (nonlocal_) identity.model.nonlocal_correlation = nonlocal_->parameters();
      if (dft::semilocal_family_uses_molecular_nonlocal_domain(execution_plan_.semilocal_family))
        identity.model.nonlocal_density_domain = dft::nlc::Vv10DensityDomain::MolecularV1;
      dft::KsPhysicalState physical{identity,
                                    true,
                                    std::move(densities),
                                    std::move(focks),
                                    native.dft_diagnostic.components,
                                    native.energy,
                                    native.dft_diagnostic.physical_residual};
      cpu_physical_ = std::move(physical);
      if (seed && spins == 2) {
        // A warm UKS proposal may pass the SCF step gate while its latest
        // physical F[D] frame still fails the stricter derivative export.
        // Certify that frame before publishing success or replacing the last
        // good seed. Rejection uses the existing single cold retry in the
        // batch owner, never a hidden solve during snapshot/force export.
        dft::VerifiedKsFinalState verified;
        std::string detail;
        const dft::CudaKsFinalStateToken token{1, cpu_physical_->identity};
        if (read_final_state(token, false, verified, detail) != GENERATIVEQC_STATUS_SUCCESS) {
          native.converged = false;
          native.ks_physical_fock.clear();
        }
      }
    }
    if (native.converged && update_warm) warm_ = std::move(native.density);
    runtime::sample_cpu_capacity(host_numeric_capacity());
    return native;
  }

 private:
  void prepare_range_exchange(int device) {
    const auto spin =
        unrestricted(execution_plan_) ? scf::FockSpin::Unrestricted : scf::FockSpin::Restricted;
    const auto fock_backend =
        backend_ == GENERATIVEQC_BACKEND_CUDA ? scf::FockBackend::Cuda : scf::FockBackend::Cpu;
    range_strategy_ = scf::resolve_fock_build(
        scf::make_rsh_correction_fock_spec(spin, execution_plan_.short_range_exchange,
                                           execution_plan_.long_range_exchange,
                                           execution_plan_.range_omega),
        fock_backend, options_.screening_tolerance);
    const bool fitted_primary =
        fock_.strategy().spec.coulomb.approximation == scf::FockApproximation::DensityFitted;
    if (fock_backend == scf::FockBackend::Cpu || fitted_primary) {
      const auto budget =
          fock_backend == scf::FockBackend::Cuda ? ks_provider_bytes(system_, backend_, 0U) : 0U;
      range_correction_ = std::make_unique<scf::PreparedFockPlan>(system_, nullptr,
                                                                  *range_strategy_, device, budget);
    }
  }

  void prepare_nonlocal(int device) {
    if (grid_.point_count() > std::numeric_limits<std::uint32_t>::max())
      throw MethodError(GENERATIVEQC_STATUS_OUT_OF_MEMORY,
                        "KS grid exceeds the VV10 public point-count domain");
    generativeqc_status status = GENERATIVEQC_STATUS_INTERNAL_ERROR;
    std::string detail;
    nonlocal_ = dft::nlc::Vv10Plan::prepare(
        backend_, device, static_cast<std::uint32_t>(grid_.point_count()),
        static_cast<std::uint32_t>(options_.xc_tile_points), execution_plan_.nonlocal_parameters,
        execution_plan_.nonlocal_maximum_bytes, detail, status);
    if (!nonlocal_)
      throw MethodError(
          status, detail.empty() ? "self-consistent nonlocal plan preparation failed" : detail);
  }

  void prepare_d4(int device) {
    const auto source = ::generativeqc::generated::method_parameters::pbeD4();
    dft::dispersion::D4Parameters parameters{dft::dispersion::D4ReferenceModel::eeq,
                                             source.s6,
                                             source.s8,
                                             source.s9,
                                             source.a1,
                                             source.a2,
                                             source.cn_cutoff,
                                             source.pair_cutoff,
                                             source.atm_cutoff,
                                             source.ga,
                                             source.gc};
    std::vector<std::uint32_t> offsets{0, static_cast<std::uint32_t>(system_.atoms.size())};
    std::vector<std::int32_t> atomic_numbers;
    std::vector<double> coordinates;
    atomic_numbers.reserve(system_.atoms.size());
    coordinates.reserve(3 * system_.atoms.size());
    for (const auto& atom : system_.atoms) {
      atomic_numbers.push_back(atom.atomic_number);
      coordinates.insert(coordinates.end(), atom.position.begin(), atom.position.end());
    }
    generativeqc_status status = GENERATIVEQC_STATUS_INTERNAL_ERROR;
    std::string detail;
    d4_ = dft::dispersion::D4Plan::prepare(
        backend_, device, std::move(offsets), std::move(atomic_numbers),
        std::vector<double>{static_cast<double>(system_.charge)}, std::move(coordinates),
        parameters, dft::dispersion::D4EEQProfile::standard, 256ull * 1024ull * 1024ull, detail,
        status);
    if (!d4_)
      throw MethodError(status,
                        detail.empty() ? "PBE-D4 production plan preparation failed" : detail);
  }

  Capabilities capabilities_;
  core::System system_;
  NativeKsExecutionPlan execution_plan_;
  scf::ScfOptions options_;
  generativeqc_backend backend_;
  scf::PreparedFockPlan fock_;
  std::optional<scf::ResolvedFockBuild> range_strategy_;
  std::unique_ptr<scf::PreparedFockPlan> range_correction_;
  dft::AoBasis basis_;
  dft::MolecularGrid grid_;
  std::vector<double> warm_;
  const std::uint64_t cpu_owner_{next_cpu_ks_owner()};
  std::uint64_t cpu_epoch_{};
  std::optional<dft::KsPhysicalState> cpu_physical_;
  std::unique_ptr<dft::dispersion::D4Plan> d4_;
  std::unique_ptr<dft::nlc::Vv10Plan> nonlocal_;
#if GENERATIVEQC_HAS_CUDA
  std::unique_ptr<dft::CudaKsPlan> cuda_;
#endif
};

std::vector<double> positions(const core::System& system) {
  std::vector<double> out;
  out.reserve(3 * system.atoms.size());
  for (const auto& atom : system.atoms)
    out.insert(out.end(), atom.position.begin(), atom.position.end());
  return out;
}

bool valid_positions(const std::vector<double>& coordinates, const core::System& system) {
  return coordinates.size() == 3 * system.atoms.size() &&
         std::all_of(coordinates.begin(), coordinates.end(),
                     [](double value) { return std::isfinite(value); });
}

void set_positions(core::System& system, const std::vector<double>& coordinates) {
  for (std::size_t i = 0; i < system.atoms.size(); ++i)
    std::copy_n(coordinates.begin() + 3 * i, 3, system.atoms[i].position.begin());
}

generativeqc_status item_exception_status() {
  try {
    throw;
  } catch (const MethodError& error) {
    return error.status();
  } catch (const generativeqc::Error& error) {
    return error.status();
  } catch (const std::bad_alloc&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return GENERATIVEQC_STATUS_INTERNAL_ERROR;
  }
}

/** Independent native KS owners, with ordinary-stream round-robin CUDA work.
 * Geometry is rebuilt per item, while model/basis/charge/spin remain immutable.
 * No HF graph or Python calculation loop participates in this schedule. */
class KsPreparedBatch final : public PreparedBatch {
 public:
  KsPreparedBatch(Capabilities capabilities, std::vector<core::System> systems,
                  NativeKsExecutionPlan execution_plan, scf::ScfOptions options, dft::GridSpec grid,
                  generativeqc_backend backend, int device, bool warm_enabled,
                  std::optional<core::System> auxiliary)
      : capabilities_(capabilities),
        systems_(std::move(systems)),
        execution_plan_(execution_plan),
        options_(std::move(options)),
        grid_spec_(std::move(grid)),
        backend_(backend),
        device_(device),
        warm_enabled_(warm_enabled),
        auxiliary_(std::move(auxiliary)),
        items_(systems_.size()) {
    for (std::size_t i = 0; i < size(); ++i) {
      runtime::CpuRetainedCapacity neighbors(host_numeric_capacity());
      items_[i].plan = make_plan(systems_[i]);
    }
  }

  std::size_t size() const noexcept override { return systems_.size(); }

  void invalidate_result() override {
    for (auto& item : items_)
      if (item.plan) item.plan->invalidate_final_state();
  }

  std::vector<BatchItemResult> execute(const Coordinates& coordinates,
                                       bool compute_forces) override {
    invalidate_result();
    if (compute_forces)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "KS nuclear gradients are tracked separately in issue #163");
    if (!coordinates.empty() && coordinates.size() != size())
      throw std::invalid_argument("KS batch coordinates do not match system count");
    std::vector<BatchItemResult> results(size());
    std::vector<bool> ready(size(), false);
    std::vector<std::optional<std::vector<double>>> preliminary_seeds(size());
    std::vector<scf::initial_guess::PreliminaryDiagnostic> preliminary_diagnostics(size());
    std::vector<bool> preliminary_requested(size(), false);
    // Allocate source-geometry metadata before launching any item. The success
    // path can then publish its last-good identity without a coordinate copy.
    std::vector<scf::HfWarmState> candidates(size());
    for (std::size_t i = 0; i < size(); ++i) {
      auto& result = results[i];
      result.bucket_id = i;  // One ordinary stream/owner per stable input slot.
      result.calculation.executed_backend = backend_;
      result.calculation.energy = std::numeric_limits<double>::quiet_NaN();
      try {
        auto target = systems_[i];
        if (!coordinates.empty() && coordinates[i]) {
          if (!valid_positions(*coordinates[i], target))
            throw std::invalid_argument("invalid KS batch item coordinates");
          set_positions(target, *coordinates[i]);
        }
        auto& item = items_[i];
        candidates[i].coordinates = positions(target);
        if (!item.plan || positions(item.plan->system()) != candidates[i].coordinates) {
          // Preserve the last GOOD seed before freeing its device owner. This
          // explicit rebuild download is never part of routine SCF iterations.
          materialize_warm(i);
#if GENERATIVEQC_HAS_CUDA
          if (auto* cuda = item.plan ? item.plan->cuda_plan() : nullptr)
            add_transfers(item.retired_transfers, cuda->transfers());
#endif
          item.plan.reset();
          item.resident_warm = false;
          item.plan = make_plan(target);
        }
        result.warm_start_used = warm_enabled_ && item.warm.has_value();
#if GENERATIVEQC_HAS_CUDA
        if (item.plan->cuda_plan() && options_.preliminary_guess) {
          auto& diagnostic = preliminary_diagnostics[i];
          preliminary_requested[i] = true;
          diagnostic.requested_kind = static_cast<std::uint32_t>(options_.preliminary_guess->kind);
          diagnostic.target_attempts = 1;
          if (result.warm_start_used) {
            diagnostic.outcome = scf::initial_guess::PreliminaryOutcome::ExplicitDensity;
          } else {
            preliminary_seeds[i] = item.plan->prepare_cold_initial_guess(diagnostic);
          }
        }
#endif
        ready[i] = true;
      } catch (...) {
        result.status = item_exception_status();
      }
    }

    const auto finish = [&](std::size_t i, scf::ScfResult native) {
      auto& result = results[i];
      if (preliminary_requested[i] && !native.preliminary_guess.requested_kind)
        native.preliminary_guess = preliminary_diagnostics[i];
      result.calculation = adapt_result(std::move(native), backend_);
      items_[i].plan->apply_d4(result.calculation);
      const auto& calculation = result.calculation;
      result.status = calculation.convergence.converged ? GENERATIVEQC_STATUS_SUCCESS
                                                        : GENERATIVEQC_STATUS_NOT_CONVERGED;
      if (calculation.convergence.converged && warm_enabled_ && warm_updates_) {
        auto& state = candidates[i];
        state.energy = calculation.energy;
        state.energy_change = calculation.convergence.energy_change;
        state.density_rms = calculation.convergence.residual_rms;
        state.iterations = calculation.convergence.iterations;
        items_[i].warm = std::move(state);
        items_[i].resident_warm = true;
      }
    };

    // A rejected/nonconverged warm solve gets one cold retry. CUDA retries
    // retain the same per-item scheduler; a failed neighbor never halts it.
    for (unsigned attempt = 0; attempt < 2; ++attempt) {
      std::vector<bool> running(size(), false);
      for (std::size_t i = 0; i < size(); ++i) {
        auto& result = results[i];
        // Retry only seed-related failures. Resource/driver failures preserve
        // their first status and leave the last-good seed for explicit replay.
        const bool seed_failure = result.status == GENERATIVEQC_STATUS_NOT_CONVERGED ||
                                  result.status == GENERATIVEQC_STATUS_NUMERICAL_FAILURE ||
                                  result.status == GENERATIVEQC_STATUS_INVALID_ARGUMENT;
        const bool preliminary_seeded = preliminary_seeds[i].has_value();
        const bool first_seeded = result.warm_start_used || preliminary_seeded;
        if (!ready[i] || (attempt && (!first_seeded || !seed_failure))) continue;
        if (attempt) {
          result.warm_start_fallback = true;
          // Release the failed attempt's exported history before starting another
          // solve, preserving the two-history resource bound.
          result.calculation.ks_diagnostic.reset();
          if (preliminary_requested[i]) {
            auto& diagnostic = preliminary_diagnostics[i];
            diagnostic.target_attempts = 2;
            diagnostic.discarded_target_iterations = result.calculation.convergence.iterations;
            diagnostic.discarded_target_fock_builds = result.calculation.fock_builds;
            diagnostic.work_counters_complete = result.status == GENERATIVEQC_STATUS_NOT_CONVERGED;
            if (preliminary_seeded)
              diagnostic.outcome = scf::initial_guess::PreliminaryOutcome::TargetRetried;
          }
          preliminary_seeds[i].reset();
        }
        auto& item = items_[i];
        const bool reuse = !attempt && result.warm_start_used;
        const auto* seed =
            reuse && !item.resident_warm
                ? &item.warm->density
                : (!attempt && preliminary_seeds[i] ? &*preliminary_seeds[i] : nullptr);
        try {
#if GENERATIVEQC_HAS_CUDA
          if (auto* cuda = item.plan->cuda_plan()) {
            cuda->set_warm_start_updates(warm_enabled_ && warm_updates_);
            cuda->begin(seed, reuse && item.resident_warm && !preliminary_seeds[i]);
            running[i] = true;
            continue;
          }
#endif
          runtime::CpuRetainedCapacity neighbors(host_numeric_capacity(i));
          const auto discarded_iterations = result.calculation.convergence.iterations;
          const auto discarded_fock_builds = result.calculation.fock_builds;
          const bool discarded_complete = result.status == GENERATIVEQC_STATUS_NOT_CONVERGED;
          auto native = item.plan->run(seed, reuse && item.resident_warm,
                                       warm_enabled_ && warm_updates_, !attempt);
          if (attempt && options_.preliminary_guess) {
            auto& diagnostic = native.preliminary_guess;
            diagnostic.requested_kind =
                static_cast<std::uint32_t>(options_.preliminary_guess->kind);
            diagnostic.outcome = scf::initial_guess::PreliminaryOutcome::ExplicitDensity;
            diagnostic.target_attempts = 2;
            diagnostic.discarded_target_iterations = discarded_iterations;
            diagnostic.discarded_target_fock_builds = discarded_fock_builds;
            diagnostic.work_counters_complete = discarded_complete;
          }
          finish(i, std::move(native));
        } catch (...) {
          result.status = item_exception_status();
        }
      }
#if GENERATIVEQC_HAS_CUDA
      while (std::any_of(running.begin(), running.end(), [](bool value) { return value; })) {
        // Submit ALL active streams before synchronizing any scalar record.
        for (std::size_t i = 0; i < size(); ++i) {
          if (!running[i]) continue;
          try {
            items_[i].plan->cuda_plan()->enqueue_iteration();
          } catch (...) {
            results[i].status = item_exception_status();
            running[i] = false;
          }
        }
        for (std::size_t i = 0; i < size(); ++i) {
          if (!running[i]) continue;
          try {
            auto* cuda = items_[i].plan->cuda_plan();
            if (cuda->finish_iteration()) continue;
            running[i] = false;
            if (cuda->failed())
              throw MethodError(GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
                                "CUDA KS physical evaluation failed");
            finish(i, cuda->result(false));
          } catch (...) {
            results[i].status = item_exception_status();
            running[i] = false;
          }
        }
      }
#endif
    }
    runtime::sample_cpu_capacity(host_numeric_capacity());
    return results;
  }

  void clear_warm_starts() override {
    for (auto& item : items_) {
      item.warm.reset();
      item.resident_warm = false;
      if (item.plan) item.plan->clear_warm_start();
    }
  }

  std::size_t warm_density_size(std::size_t index) const override {
    const auto n = molecule::ao_count(systems_.at(index));
    const std::size_t spins = execution_plan_.spin_channels;
    if (!n || n > std::numeric_limits<std::size_t>::max() / n / spins / sizeof(double))
      throw std::invalid_argument("KS warm density dimensions overflow");
    return spins * n * n;
  }

  const std::optional<scf::HfWarmState>& warm_state(std::size_t index) const override {
    materialize_warm(index);
    return items_.at(index).warm;
  }

  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>> states) override {
    if (!warm_enabled_ || states.size() != size())
      throw std::invalid_argument("KS seed restore requires a matching warm-enabled batch");
    for (std::size_t i = 0; i < size(); ++i) {
      if (!states[i]) continue;
      const auto& state = *states[i];
      if (state.density.size() != warm_density_size(i) ||
          !valid_positions(state.coordinates, systems_[i]) || state.iterations < 0 ||
          !std::isfinite(state.energy) || !std::isfinite(state.energy_change) ||
          !std::isfinite(state.density_rms) || state.density_rms < 0)
        throw std::invalid_argument("invalid KS seed dimensions or diagnostics");
      auto source = systems_[i];
      set_positions(source, state.coordinates);
      scf::initial_guess::EigenOperation eigen;
#if GENERATIVEQC_HAS_CUDA
      if (auto* cuda = items_[i].plan ? items_[i].plan->cuda_plan() : nullptr)
        eigen = cuda->seed_eigen_operation();
#endif
      // This common validation reads only source S and checks the shared
      // spin-density convention. An idle CUDA plan supplies the ordinary
      // eigensolver without overwriting its final/warm state; source coordinates
      // still determine S, including changed-geometry checkpoints. The shared
      // validator preserves its legacy near-symmetric input fallback.
      scf::validate_hf_warm_density(
          source, unrestricted(execution_plan_) ? GENERATIVEQC_METHOD_UHF : GENERATIVEQC_METHOD_RHF,
          state.density, eigen);
    }
    // All source-metric validation precedes the no-throw commit. Missing
    // entries preserve neighbors, including their resident density ownership.
    for (std::size_t i = 0; i < size(); ++i) {
      if (!states[i]) continue;
      auto& item = items_[i];
      item.warm.swap(states[i]);
      item.resident_warm = false;
      if (item.plan) item.plan->clear_warm_start();
    }
  }

  void set_warm_start_updates(bool enabled) override { warm_updates_ = enabled; }

  std::optional<KsTransportDiagnostic> ks_transport_diagnostic(std::size_t index) const override {
    const auto& item = items_.at(index);
#if GENERATIVEQC_HAS_CUDA
    if (auto* cuda = item.plan ? item.plan->cuda_plan() : nullptr) {
      auto cumulative = item.retired_transfers;
      add_transfers(cumulative, cuda->transfers());
      return adapt_transfers(cumulative);
    }
#endif
    return std::nullopt;
  }

  generativeqc_status final_state_token(std::size_t index, dft::CudaKsFinalStateToken& token,
                                        std::string& detail) const {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->final_state_token(token, detail);
    token = {};
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status read_final_state(std::size_t index,
                                       const dft::CudaKsFinalStateToken& expected,
                                       bool compute_weighted_density,
                                       dft::VerifiedKsFinalState& state, std::string& detail) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->read_final_state(expected, compute_weighted_density, state,
                                                  detail);
    state = {};
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status resident_density(std::size_t index,
                                       const dft::CudaKsFinalStateToken& expected, int& device,
                                       const double*& alpha, const double*& beta,
                                       std::size_t& matrix_elements, unsigned& spins,
                                       void*& source_stream, std::string& detail) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->resident_density(expected, device, alpha, beta, matrix_elements,
                                                  spins, source_stream, detail);
    device = -1;
    alpha = nullptr;
    beta = nullptr;
    matrix_elements = 0;
    spins = 0;
    source_stream = nullptr;
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status fixed_density_profile(std::size_t index,
                                            const dft::CudaKsFinalStateToken& expected,
                                            KsFixedDensityProfile& profile, std::string& detail) {
    profile = {};
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->fixed_density_profile(expected, profile, detail);
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status resident_grid(std::size_t index, const dft::CudaKsFinalStateToken& expected,
                                    int& device, const double*& points, const double*& weights,
                                    const double*& atomic_weights, std::size_t& point_count,
                                    std::string& detail) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->resident_grid(expected, device, points, weights, atomic_weights,
                                               point_count, detail);
    device = -1;
    points = nullptr;
    weights = nullptr;
    atomic_weights = nullptr;
    point_count = 0;
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status resident_nonlocal_features(std::size_t index,
                                                 const dft::CudaKsFinalStateToken& expected,
                                                 int& device, const double*& density,
                                                 const double*& gradient, std::size_t& point_count,
                                                 void*& source_stream, std::string& detail) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->resident_nonlocal_features(expected, device, density, gradient,
                                                            point_count, source_stream, detail);
    device = -1;
    density = nullptr;
    gradient = nullptr;
    point_count = 0;
    source_stream = nullptr;
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status read_derivative_state(std::size_t index,
                                            const dft::CudaKsFinalStateToken& expected,
                                            KsDerivativeSnapshot& output, std::string& detail) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->read_derivative_state(expected, output, detail);
    output = {};
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status density_fitted_integral_gradient(
      std::size_t index, const dft::CudaKsFinalStateToken& expected,
      const std::vector<scf::reference::Matrix>& density,
      const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->density_fitted_integral_gradient(
          expected, density, weighted_density, output, maximum_bytes, work, detail);
    output.clear();
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status cuda_full_range_integral_derivatives(
      std::size_t index, const dft::CudaKsFinalStateToken& expected, std::vector<double>& output,
      std::string& detail) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->cuda_full_range_integral_derivatives(expected, output, detail);
    output.clear();
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  generativeqc_status cuda_integral_gradient(
      std::size_t index, const dft::CudaKsFinalStateToken& expected, std::vector<double>& output,
      std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail,
      const std::vector<scf::reference::Matrix>* cached_density = nullptr,
      const std::vector<scf::reference::Matrix>* cached_weighted_density = nullptr,
      bool combined_two_electron = false) {
    if (index < items_.size() && items_[index].plan)
      return items_[index].plan->cuda_integral_gradient(
          expected, output, maximum_bytes, work, detail, cached_density, cached_weighted_density,
          combined_two_electron);
    detail = "KS batch item has no prepared final-state owner";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }

  // These profiles describe HF graph/provider layouts, not this method's
  // ordinary-stream schedule. Absence is explicit at the common interface.
  std::optional<std::vector<DirectShellClassProfileEntry>> last_direct_shell_class_profile()
      const override {
    return std::nullopt;
  }
  std::optional<DirectPppsQueueProfile> last_direct_ppps_queue_profile() const override {
    return std::nullopt;
  }
  std::vector<EigensolverDiagnostic> last_eigensolver_diagnostics() const override { return {}; }
  std::vector<scf::CudaDensityFittingMetricDiagnostic> last_density_fitting_metric_diagnostics()
      const override {
    std::vector<scf::CudaDensityFittingMetricDiagnostic> result;
    for (std::size_t index = 0; index < items_.size(); ++index)
      if (items_[index].plan)
        for (auto diagnostic : items_[index].plan->fitted_diagnostics()) {
          // KS assigns one provider/bucket per input slot, including equal shapes.
          diagnostic.bucket_id = index;
          diagnostic.system_index = index;
          result.push_back(diagnostic);
        }
    return result;
  }
  std::vector<InactiveEigensolverProfileEntry> last_inactive_eigensolver_profile() const override {
    return {};
  }

 private:
  /** All other owners remain alive while one CPU item executes. The selected
   * item's externally materialized seed is also distinct from its plan. */
  std::size_t host_numeric_capacity(std::size_t exclude_plan = SIZE_MAX) const noexcept {
    std::size_t bytes = 0;
    for (std::size_t i = 0; i < items_.size(); ++i) {
      const auto& item = items_[i];
      if (item.plan && i != exclude_plan)
        bytes = runtime::add_capacity(bytes, item.plan->host_numeric_capacity());
      if (item.warm)
        bytes = runtime::add_capacity(
            bytes, runtime::vector_capacities(item.warm->density, item.warm->coordinates));
    }
    return bytes;
  }

  struct Item {
    std::unique_ptr<KsPreparedCalculation> plan;
    // Density is materialized only for explicit output, import, or rebuilding
    // an owner. Empty density with resident_warm=true is a valid lazy snapshot.
    mutable std::optional<scf::HfWarmState> warm;
    bool resident_warm{};
#if GENERATIVEQC_HAS_CUDA
    dft::CudaKsTransfers retired_transfers;
#endif
  };
  std::unique_ptr<KsPreparedCalculation> make_plan(const core::System& system) const {
    return std::make_unique<KsPreparedCalculation>(capabilities_, system, execution_plan_, options_,
                                                   grid_spec_, backend_, device_,
                                                   ks_auxiliary_for_system(system, auxiliary_));
  }
  void materialize_warm(std::size_t i) const {
    const auto& item = items_.at(i);
    if (item.warm && item.warm->density.empty() && item.resident_warm)
      item.warm->density = item.plan->warm_density();
  }

  Capabilities capabilities_;
  std::vector<core::System> systems_;
  NativeKsExecutionPlan execution_plan_;
  scf::ScfOptions options_;
  dft::GridSpec grid_spec_;
  generativeqc_backend backend_;
  int device_;
  bool warm_enabled_, warm_updates_{true};
  std::optional<core::System> auxiliary_;
  std::vector<Item> items_;
};

}  // namespace

Result adapt_dft_result(scf::ScfResult native, generativeqc_backend backend) {
  return adapt_result(std::move(native), backend);
}

generativeqc_status dft_final_state_token(const PreparedCalculation& calculation,
                                          dft::CudaKsFinalStateToken& token, std::string& detail) {
  const auto* ks = dynamic_cast<const KsPreparedCalculation*>(&calculation);
  if (ks) return ks->final_state_token(token, detail);
  token = {};
  detail = "prepared calculation is not a KS final-state owner";
  return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
}

generativeqc_status read_dft_final_state(PreparedCalculation& calculation,
                                         const dft::CudaKsFinalStateToken& expected,
                                         bool compute_weighted_density,
                                         dft::VerifiedKsFinalState& state, std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedCalculation*>(&calculation);
  if (ks) return ks->read_final_state(expected, compute_weighted_density, state, detail);
  state = {};
  detail = "prepared calculation is not a KS final-state owner";
  return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
}

generativeqc_status dft_final_state_token(const PreparedBatch& batch, std::size_t index,
                                          dft::CudaKsFinalStateToken& token, std::string& detail) {
  const auto* ks = dynamic_cast<const KsPreparedBatch*>(&batch);
  if (ks) return ks->final_state_token(index, token, detail);
  token = {};
  detail = "prepared batch is not a KS final-state owner";
  return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
}

generativeqc_status read_dft_final_state(PreparedBatch& batch, std::size_t index,
                                         const dft::CudaKsFinalStateToken& expected,
                                         bool compute_weighted_density,
                                         dft::VerifiedKsFinalState& state, std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks) return ks->read_final_state(index, expected, compute_weighted_density, state, detail);
  state = {};
  detail = "prepared batch is not a KS final-state owner";
  return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
}

generativeqc_status dft_cuda_full_range_integral_derivatives(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    std::vector<double>& output, std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks) return ks->cuda_full_range_integral_derivatives(index, expected, output, detail);
  output.clear();
  detail = "CUDA full-range stationary shell derivatives require a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_integral_gradient(PreparedBatch& batch, std::size_t index,
                                               const dft::CudaKsFinalStateToken& expected,
                                               std::vector<double>& output,
                                               std::size_t maximum_bytes,
                                               std::array<std::uint64_t, 9>& work,
                                               std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks) return ks->cuda_integral_gradient(index, expected, output, maximum_bytes, work, detail);
  detail = "CUDA integral gradient requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_density_fitted_integral_gradient_cached(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    const std::vector<scf::reference::Matrix>& density,
    const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
    std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks)
    return ks->density_fitted_integral_gradient(index, expected, density, weighted_density, output,
                                                maximum_bytes, work, detail);
  output.clear();
  detail = "density-fitted integral gradient requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_integral_gradient_cached(
    PreparedBatch& batch, std::size_t index, const dft::CudaKsFinalStateToken& expected,
    const std::vector<scf::reference::Matrix>& density,
    const std::vector<scf::reference::Matrix>& weighted_density, std::vector<double>& output,
    std::size_t maximum_bytes, std::array<std::uint64_t, 9>& work, std::string& detail,
    bool combined_two_electron) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks)
    return ks->cuda_integral_gradient(index, expected, output, maximum_bytes, work, detail,
                                      &density, &weighted_density, combined_two_electron);
  detail = "CUDA integral gradient requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_resident_density(PreparedBatch& batch, std::size_t index,
                                              const dft::CudaKsFinalStateToken& expected,
                                              int& device, const double*& alpha,
                                              const double*& beta, std::size_t& matrix_elements,
                                              unsigned& spins, void*& source_stream,
                                              std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks)
    return ks->resident_density(index, expected, device, alpha, beta, matrix_elements, spins,
                                source_stream, detail);
  device = -1;
  alpha = nullptr;
  beta = nullptr;
  matrix_elements = 0;
  spins = 0;
  source_stream = nullptr;
  detail = "resident final density requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_fixed_density_profile(PreparedBatch& batch, std::size_t index,
                                                   const dft::CudaKsFinalStateToken& expected,
                                                   KsFixedDensityProfile& profile,
                                                   std::string& detail) {
  profile = {};
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks) return ks->fixed_density_profile(index, expected, profile, detail);
  detail = "fixed-density component profiling requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_resident_grid(PreparedBatch& batch, std::size_t index,
                                           const dft::CudaKsFinalStateToken& expected, int& device,
                                           const double*& points, const double*& weights,
                                           const double*& atomic_weights, std::size_t& point_count,
                                           std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks)
    return ks->resident_grid(index, expected, device, points, weights, atomic_weights, point_count,
                             detail);
  device = -1;
  points = nullptr;
  weights = nullptr;
  atomic_weights = nullptr;
  point_count = 0;
  detail = "resident molecular grid requires a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status dft_cuda_resident_nonlocal_features(PreparedBatch& batch, std::size_t index,
                                                        const dft::CudaKsFinalStateToken& expected,
                                                        int& device, const double*& density,
                                                        const double*& gradient,
                                                        std::size_t& point_count,
                                                        void*& source_stream, std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks)
    return ks->resident_nonlocal_features(index, expected, device, density, gradient, point_count,
                                          source_stream, detail);
  device = -1;
  density = nullptr;
  gradient = nullptr;
  point_count = 0;
  source_stream = nullptr;
  detail = "resident nonlocal features require a native KS batch";
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
}

generativeqc_status read_dft_derivative_state(PreparedBatch& batch, std::size_t index,
                                              const dft::CudaKsFinalStateToken& expected,
                                              KsDerivativeSnapshot& output, std::string& detail) {
  auto* ks = dynamic_cast<KsPreparedBatch*>(&batch);
  if (ks) return ks->read_derivative_state(index, expected, output, detail);
  output = {};
  detail = "prepared batch is not a KS final-state owner";
  return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
}

void validate_ks_spin_state(const NativeKsExecutionPlan& execution_plan,
                            const core::System& system) {
  if (dft::semilocal_family_uses_molecular_nonlocal_domain(execution_plan.semilocal_family) &&
      !system.ecp_terms.empty())
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "molecular-nonlocal KS ECP execution is not qualified");
  if (!unrestricted(execution_plan)) {
    if (system.electron_count <= 0 || system.electron_count % 2 || system.multiplicity != 1)
      throw std::invalid_argument(
          "RKS requires a positive even electron count and spin multiplicity 1");
    return;
  }
  const int spin_excess = static_cast<int>(system.multiplicity) - 1;
  if (system.electron_count <= 0 || spin_excess < 0 || spin_excess > system.electron_count ||
      (system.electron_count - spin_excess) % 2 != 0)
    throw std::invalid_argument(
        "UKS requires electron count and multiplicity to define integer "
        "nonnegative spin occupations");
}

generativeqc_status validate_dft_system(generativeqc_method, const core::System& system,
                                        std::string& detail) {
  if (system.shells.empty()) {
    detail = "DFT requires an explicit Gaussian orbital basis";
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  }
  const int spin_excess = static_cast<int>(system.multiplicity) - 1;
  if (system.electron_count > 0 && spin_excess >= 0 && spin_excess <= system.electron_count &&
      (system.electron_count - spin_excess) % 2 == 0)
    return GENERATIVEQC_STATUS_SUCCESS;
  detail =
      "DFT requires electron count and multiplicity to define integer nonnegative spin occupations";
  return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
}

std::unique_ptr<PreparedCalculation> prepare_dft_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor) {
#if !GENERATIVEQC_HAS_CUDA
  if (context.requested_backend == GENERATIVEQC_BACKEND_CUDA)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "DFT CUDA backend is not built");
#endif
  NativeKsExecutionPlan execution_plan;
  auto options = dft_options(descriptor, context.requested_backend, execution_plan);
  validate_ks_spin_state(execution_plan, system);
  if (options.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE &&
      (!system.ecp_terms.empty() ||
       std::any_of(system.atoms.begin(), system.atoms.end(),
                   [](const auto& atom) { return atom.ecp_core != 0; })))
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "DFT ECP density fitting is not qualified");
  validate_ks_auxiliary_geometry(system, ks_auxiliary_template(descriptor));
  auto grid = ks_grid_options(descriptor, options, execution_plan);
  return std::make_unique<KsPreparedCalculation>(
      capabilities, system, execution_plan, std::move(options), std::move(grid),
      context.requested_backend, context.device_id,
      ks_auxiliary_for_system(system, ks_auxiliary_template(descriptor)));
}

std::unique_ptr<PreparedBatch> prepare_dft_batch(const Capabilities& capabilities,
                                                 core::ContextState& context,
                                                 std::vector<core::System> systems,
                                                 const generativeqc_method_descriptor& descriptor,
                                                 generativeqc_batch_flags flags) {
  if ((flags & ~GENERATIVEQC_BATCH_ENABLE_WARM_STARTS) != 0)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "KS batches support warm starts but not HF-specific profiling flags");
#if !GENERATIVEQC_HAS_CUDA
  if (context.requested_backend == GENERATIVEQC_BACKEND_CUDA)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "DFT CUDA backend is not built");
#endif
  NativeKsExecutionPlan execution_plan;
  auto options = dft_options(descriptor, context.requested_backend, execution_plan);
  for (const auto& system : systems) {
    validate_ks_spin_state(execution_plan, system);
    if (options.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_NONE &&
        (!system.ecp_terms.empty() ||
         std::any_of(system.atoms.begin(), system.atoms.end(),
                     [](const auto& atom) { return atom.ecp_core != 0; })))
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "DFT ECP density fitting is not qualified");
  }
  if (!systems.empty())
    validate_ks_auxiliary_geometry(systems.front(), ks_auxiliary_template(descriptor));
  auto grid = ks_grid_options(descriptor, options, execution_plan);
  return std::make_unique<KsPreparedBatch>(
      capabilities, std::move(systems), execution_plan, std::move(options), std::move(grid),
      context.requested_backend, context.device_id,
      (flags & GENERATIVEQC_BATCH_ENABLE_WARM_STARTS) != 0, ks_auxiliary_template(descriptor));
}

}  // namespace generativeqc::methods::detail
