#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "generativeqc/generativeqc.h"

namespace generativeqc::core {
struct System;
}

namespace generativeqc::methods::named_ks {

struct NonlocalCorrelation {
  generativeqc_nonlocal_variant variant{};  // Zero denotes an absent primitive.
  double b{}, c{}, coefficient{};
};

/** Compiler-owned, geometry-only correction. Kind: none, D3(BJ), D3(0), D4-EEQ.
 * Cutoffs and switching widths are in Bohr; zero denotes no cutoff for D3. */
struct Dispersion {
  std::uint32_t kind{};
  double s6{}, s8{}, s9{}, a1{}, a2{}, rs6{}, rs8{}, alp{};
  double cn_cutoff{}, pair_cutoff{}, atm_cutoff{}, pair_switch_width{}, atm_switch_width{};
  double ga{}, gc{}, charge_cn_cutoff{};
  // 0: ordinary D4. 1: the complete, basis-bound r2SCAN-3c D4 + gCP profile.
  std::uint32_t composite_profile{};
};

/** Build-time compiler output, owned once by the native library. These flags
 * describe catalog representability, never per-system scientific admission. */
struct Method {
  std::string_view name, source, identity, domain;
  generativeqc_method carrier;
  std::uint32_t spin, component_offset, component_count;
  std::uint32_t exchange_offset, exchange_count;
  bool cpu, cuda;
  std::string_view reason;
  double range_omega{};
  NonlocalCorrelation nonlocal{};
  Dispersion dispersion{};
  const generativeqc_basis_template_descriptor_v1* bound_basis{};
  std::string_view basis_name, basis_identity;
};

GENERATIVEQC_API std::span<const Method> methods() noexcept;
GENERATIVEQC_API const Method* find_method(std::string_view name) noexcept;
GENERATIVEQC_API std::span<const generativeqc_ks_semilocal_component> components(
    const Method& method) noexcept;
GENERATIVEQC_API std::span<const generativeqc_ks_exchange_term> exchanges(
    const Method& method) noexcept;

/** Copy the complete electronic program, preserving all caller execution controls.
 * Correction-bearing callers must also retain dispersion in their native owner. */
void electronic_options(const Method& method, generativeqc_ks_options& options) noexcept;

/** Compare actual normalized orbital data against the method's immutable
 * template. A display name or caller-provided checksum cannot authorize it. */
void validate_bound_basis(const Method& method, const core::System& system);

}  // namespace generativeqc::methods::named_ks
