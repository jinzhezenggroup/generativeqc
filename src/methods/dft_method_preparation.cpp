#include "methods/dft_method_preparation.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "core/types.hpp"
#include "dft/xc.hpp"
#include "generated_method_parameters.hpp"
#include "libxc_semilocal_cpu/generated_libxc_semilocal_registry.hpp"
#include "methods/method.hpp"
#include "methods/scf_common_options.hpp"
#include "scf/mean_field.hpp"

#if GENERATIVEQC_HAS_CUDA
#include "generated_split_hybrid_registry.cuh"
#endif

namespace generativeqc::methods::detail {
namespace {

// Descriptor interpretation is deliberately kept out of mutable execution
// owners. Every owner receives an already-validated, immutable plan.
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
  const dft::RestrictedSemilocalPointProgram* automatic_restricted_program{};
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
        return {dft::SemilocalFamily::Lda,
                1.0,
                1.0,
                automatic.functional_code,
                false,
                automatic.program,
                dft::generated::automatic_restricted_libxc_program(
                    std::string_view(component.component_id))};
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
  if (plan.automatic_program)
    return ks_scf_domain_version(plan) == 5U ? dft::generated::kAutomaticDensityLibxcScfDomain
                                          : dft::generated::kAutomaticLibxcScfDomain;
  if (plan.generated_split_hybrid) return "libxc-7.0/split-global-hybrid-v1";
  return dft::semilocal_family_scf_domain(plan.semilocal_family);
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
}  // namespace

bool ks_is_unrestricted(const NativeKsExecutionPlan& plan) noexcept { return plan.spin_channels == 2; }

std::uint32_t ks_scf_domain_version(const NativeKsExecutionPlan& plan) noexcept {
  if (!ks_is_unrestricted(plan) && plan.automatic_restricted_program)
    return plan.automatic_restricted_program->domain_version;
  if (plan.automatic_program) return plan.automatic_program->domain_version;
  return plan.generated_split_hybrid ? 4U
                                     : dft::semilocal_family_domain_version(plan.semilocal_family);
}

std::uint32_t ks_xc_functional_code(const NativeKsExecutionPlan& plan) noexcept {
  return plan.functional ? plan.functional : dft::semilocal_family_code(plan.semilocal_family);
}

const char* ks_semilocal_family_name(const NativeKsExecutionPlan& plan) noexcept {
  if (plan.automatic_program) return plan.automatic_program->identifier;
  return plan.generated_split_hybrid ? "generated split global hybrid"
                                     : dft::ks_semilocal_family_name(plan.semilocal_family);
}

scf::ScfOptions dft_options(const generativeqc_method_descriptor& descriptor,
                            generativeqc_backend backend, NativeKsExecutionPlan& execution_plan) {
  if (!std::isfinite(descriptor.energy_tolerance) || !std::isfinite(descriptor.density_tolerance) ||
      !std::isfinite(descriptor.screening_tolerance))
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT, "DFT tolerances must be finite");
  scf::ScfOptions options = common_scf_options(descriptor);

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
    execution_plan.automatic_restricted_program = semilocal.automatic_restricted_program;
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

  const auto mode = checked_density_fitting_mode(descriptor);
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
  const auto precision_mode = checked_precision_mode(descriptor);
  if (precision_mode == GENERATIVEQC_PRECISION_AUTO && backend != GENERATIVEQC_BACKEND_CUDA)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "DFT automatic precision currently requires CUDA");
  if (precision_mode == GENERATIVEQC_PRECISION_AUTO && execution_plan.d4_correction)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "PBE-D4 currently requires strict FP64");
  options.precision_mode = precision_mode;

  scf::FockBuildSpec fock;
  fock.spin =
      ks_is_unrestricted(execution_plan) ? scf::FockSpin::Unrestricted : scf::FockSpin::Restricted;
  if (options.preliminary_guess && !options.preliminary_guess->automatic_cpu &&
      fock.spin != scf::FockSpin::Restricted)
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
    const double divisor = ks_is_unrestricted(execution_plan) ? 1.0 : 2.0;
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
        dft::generated::split_hybrid_composition(ks_xc_functional_code(execution_plan));
    if (composition.matched && composition.exact_exchange_denominator) {
      const double exact_exchange = static_cast<double>(composition.exact_exchange_numerator) /
                                    static_cast<double>(composition.exact_exchange_denominator);
      const double divisor = ks_is_unrestricted(execution_plan) ? 1.0 : 2.0;
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

void validate_ks_spin_state(const NativeKsExecutionPlan& execution_plan,
                            const core::System& system) {
  if (dft::semilocal_family_uses_molecular_nonlocal_domain(execution_plan.semilocal_family) &&
      !system.ecp_terms.empty())
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "molecular-nonlocal KS ECP execution is not qualified");
  if (!ks_is_unrestricted(execution_plan)) {
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

}  // namespace generativeqc::methods::detail
