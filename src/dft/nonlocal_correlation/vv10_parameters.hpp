#pragma once

#include <cstdint>

namespace generativeqc::dft::nlc {

enum class Vv10Variant : std::int32_t { vv10 = 1, rvv10 = 2 };

/** Scientific value identity shared by the SCF provider and grid derivatives.
 * It does not own quadrature, storage, or execution policy. */
struct Vv10Parameters {
  Vv10Variant variant{Vv10Variant::vv10};
  double b{};
  double c{};
  double coefficient{1.0};
  bool operator==(const Vv10Parameters&) const = default;
};

}  // namespace generativeqc::dft::nlc
