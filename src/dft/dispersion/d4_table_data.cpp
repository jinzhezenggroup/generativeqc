#include "dft/dispersion/d4_eeq.hpp"

#include <array>
#include <cstddef>

#include "dft/dispersion/d4_data.hpp"
#include "dft/dispersion/d4_eeq_data.hpp"
#include "dft/dispersion/d4_eeq_r2scan3c_c6.hpp"

namespace generativeqc::dft::dispersion {
namespace {

constexpr auto make_charge_elements() {
  std::array<D4EEQChargeElementData, kD4TableElementCount> result{};
  static_assert(eeq_data::kElementCount == kD4TableElementCount);
  for (std::size_t index = 0; index < result.size(); ++index) {
    const auto& source = eeq_data::kChargeElements[index];
    result[index] = {source.chi, source.eta, source.kcnchi, source.radius};
  }
  return result;
}

inline constexpr auto kEEQChargeElements = make_charge_elements();

static_assert(data::kElementCount == kD4TableElementCount);
static_assert(data::kReferenceCount == kD4TableReferenceCount);
static_assert(data::kReferenceC6.size() == kD4PackedReferenceC6Count);
static_assert(eeq_data::kReferenceCount == kD4TableReferenceCount);
static_assert(eeq_data::kReferenceC6Standard.size() == kD4PackedReferenceC6Count);
static_assert(eeq_data::kReferenceC6R2SCAN3C.size() == kD4PackedReferenceC6Count);

}  // namespace

D4Tables gfn2_d4_host_tables() {
  return {D4ReferenceModel::gfn2,
          data::kElements.data(),
          data::kReferences.data(),
          data::kReferenceC6.data(),
          kD4TableElementCount,
          kD4TableReferenceCount,
          kD4PackedReferenceC6Count,
          3.0,
          2.0};
}

EEQTables eeq2019_host_tables() {
  return {eeq_data::kElements.data(), kEEQChargeElements.data(), kD4TableElementCount};
}

D4Tables eeq_d4_host_tables(D4EEQProfile profile) {
  const bool r2scan = profile == D4EEQProfile::r2scan3c;
  const double* c6 =
      r2scan ? eeq_data::kReferenceC6R2SCAN3C.data() : eeq_data::kReferenceC6Standard.data();
  const auto r2scan_parameters = ::generativeqc::generated::method_parameters::r2scan3cD4();
  return {D4ReferenceModel::eeq,
          eeq_data::kElements.data(),
          eeq_data::kReferences.data(),
          c6,
          kD4TableElementCount,
          kD4TableReferenceCount,
          kD4PackedReferenceC6Count,
          r2scan ? r2scan_parameters.ga : 3.0,
          r2scan ? r2scan_parameters.gc : 2.0};
}

}  // namespace generativeqc::dft::dispersion
