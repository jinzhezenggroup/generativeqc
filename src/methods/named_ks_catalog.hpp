#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "generativeqc/generativeqc.h"

namespace generativeqc::methods::named_ks {

/** Build-time compiler output, owned once by the native library. These flags
 * describe catalog representability, never per-system scientific admission. */
struct Method {
  std::string_view name, source, identity, domain;
  generativeqc_method carrier;
  std::uint32_t spin, component_offset, component_count;
  std::uint32_t exchange_offset, exchange_count;
  bool cpu, cuda;
  std::string_view reason;
};

GENERATIVEQC_API std::span<const Method> methods() noexcept;
GENERATIVEQC_API const Method* find_method(std::string_view name) noexcept;
GENERATIVEQC_API std::span<const generativeqc_ks_semilocal_component> components(
    const Method& method) noexcept;
GENERATIVEQC_API std::span<const generativeqc_ks_exchange_term> exchanges(
    const Method& method) noexcept;

}  // namespace generativeqc::methods::named_ks
