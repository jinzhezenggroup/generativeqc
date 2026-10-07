#include "dft/dispersion/d4_eeq.hpp"

#include <array>

#include "dft/dispersion/d4_data.hpp"
#include "dft/dispersion/d4_eeq_data.hpp"
#include "dft/dispersion/d4_eeq_r2scan3c_c6.hpp"

namespace generativeqc::dft::dispersion {
namespace {
static_assert(data::kElementCount == kD4ElementCount);
static_assert(data::kReferenceCount == kD4ReferenceCount);
static_assert(data::kReferenceC6.size() == kD4ReferenceC6Count);
static_assert(eeq_data::kElementCount == kD4ElementCount);
static_assert(eeq_data::kReferenceCount == kD4ReferenceCount);
static_assert(eeq_data::kReferenceC6Standard.size() == kD4ReferenceC6Count);
static_assert(eeq_data::kReferenceC6R2SCAN3C.size() == kD4ReferenceC6Count);

const std::array<EEQChargeElementData, kD4ElementCount>& eeq_charge_elements() {
  static const auto values = [] {
    std::array<EEQChargeElementData, kD4ElementCount> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
      const auto& source = eeq_data::kChargeElements[i];
      result[i] = {source.chi, source.eta, source.kcnchi, source.radius};
    }
    return result;
  }();
  return values;
}
}  // namespace

D4Tables gfn2_d4_host_tables() {
  return {D4ReferenceModel::gfn2, data::kElements.data(), data::kReferences.data(),
          data::kReferenceC6.data(), kD4ElementCount, kD4ReferenceCount,
          kD4ReferenceC6Count, 3.0, 2.0};
}
EEQTables eeq2019_host_tables() {
  return {eeq_data::kElements.data(), eeq_charge_elements().data(), kD4ElementCount};
}
D4Tables eeq_d4_host_tables(D4EEQProfile profile) {
  const bool r2scan = profile == D4EEQProfile::r2scan3c;
  const double* c6 = r2scan ? eeq_data::kReferenceC6R2SCAN3C.data()
                            : eeq_data::kReferenceC6Standard.data();
  const auto p = ::generativeqc::generated::method_parameters::r2scan3cD4();
  return {D4ReferenceModel::eeq, eeq_data::kElements.data(), eeq_data::kReferences.data(),
          c6, kD4ElementCount, kD4ReferenceCount, kD4ReferenceC6Count,
          r2scan ? p.ga : 3.0, r2scan ? p.gc : 2.0};
}
}  // namespace generativeqc::dft::dispersion
