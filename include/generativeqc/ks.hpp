#ifndef GENERATIVEQC_KS_HPP
#define GENERATIVEQC_KS_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "generativeqc/generativeqc.hpp"

namespace generativeqc {

/** Caller-selected molecular quadrature; no implicit production grid is promised. */
struct KsGrid {
  std::uint32_t version{1};
  std::uint32_t radial_points{64};
  std::uint32_t angular_polar{12};
  std::uint32_t angular_azimuth{24};
  std::uint32_t partition_iterations{3};
  double coincident_tolerance{1.0e-12};
  std::uint64_t tile_points{256};
};

/** Build a compiler-equivalent explicit KS composition in plain C++.
 *
 * This is a transport builder, not a second scientific functional registry:
 * callers supply semilocal component IDs, coefficients and the qualified SCF
 * domain. Native preparation copies every pointer-backed field synchronously.
 * Its returned Calculation therefore does not borrow this builder's lifetime.
 * A force request still obeys the native method capabilities; no Python fallback.
 */
class KsComposition {
 public:
  KsComposition(generativeqc_method carrier, std::string scf_domain, std::uint32_t spin_channels)
      : carrier_(carrier), scf_domain_(std::move(scf_domain)), spin_channels_(spin_channels) {
    if (scf_domain_.empty() || (spin_channels_ != 1 && spin_channels_ != 2))
      throw std::invalid_argument("KS composition requires a domain and one/two spin channels");
  }

  KsComposition& set_grid(KsGrid grid) {
    grid_ = grid;
    return *this;
  }

  KsComposition& set_element_radii(std::array<double, 119> radii) {
    radii_ = std::move(radii);
    return *this;
  }

  KsComposition& set_xc_schedule(generativeqc_xc_execution_schedule schedule) {
    schedule_ = schedule;
    return *this;
  }

  KsComposition& add_semilocal(std::string component_id, double coefficient) {
    if (component_id.empty()) throw std::invalid_argument("empty KS component identifier");
    components_.emplace_back(std::move(component_id), coefficient);
    return *this;
  }

  KsComposition& add_exact_exchange(generativeqc_ks_exchange_operator operation, double coefficient,
                                    double omega = 0.0) {
    const double divisor = spin_channels_ == 1 ? 2.0 : 1.0;
    exchange_.push_back({operation, coefficient, omega, -coefficient / divisor});
    return *this;
  }

  [[nodiscard]] Calculation prepare(Context& context, const System& system,
                                    generativeqc_method_descriptor descriptor) const {
    if (descriptor.method != carrier_)
      throw std::invalid_argument("KS carrier and method descriptor must match");
    if (method_capabilities(carrier_).family != GENERATIVEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL)
      throw std::invalid_argument("KS composition requires a DFT method carrier");

    std::vector<generativeqc_ks_semilocal_component> native_components;
    native_components.reserve(components_.size());
    for (const auto& component : components_)
      native_components.push_back({component.first.c_str(), component.second});

    generativeqc_ks_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = GENERATIVEQC_ABI_VERSION;
    options.scf_domain = scf_domain_.c_str();
    options.grid_version = grid_.version;
    options.radial_points = grid_.radial_points;
    options.angular_polar = grid_.angular_polar;
    options.angular_azimuth = grid_.angular_azimuth;
    options.partition_iterations = grid_.partition_iterations;
    options.coincident_tolerance = grid_.coincident_tolerance;
    options.tile_points = grid_.tile_points;
    if (radii_) {
      options.element_radii = radii_->data();
      options.element_radius_count = static_cast<std::uint32_t>(radii_->size());
    }
    options.xc_execution_schedule = schedule_;
    options.spin_channels = spin_channels_;
    options.semilocal_components = native_components.empty() ? nullptr : native_components.data();
    options.semilocal_component_count = static_cast<std::uint32_t>(native_components.size());
    options.exchange_terms = exchange_.empty() ? nullptr : exchange_.data();
    options.exchange_term_count = static_cast<std::uint32_t>(exchange_.size());

    descriptor.ks_options = &options;
    return Calculation(context, system, descriptor);
  }

 private:
  generativeqc_method carrier_;
  std::string scf_domain_;
  std::uint32_t spin_channels_;
  KsGrid grid_{};
  generativeqc_xc_execution_schedule schedule_{GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED};
  std::vector<std::pair<std::string, double>> components_;
  std::vector<generativeqc_ks_exchange_term> exchange_;
  std::optional<std::array<double, 119>> radii_;
};

}  // namespace generativeqc

#endif  // GENERATIVEQC_KS_HPP
