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
/** @native-contract generativeqc::KsGrid
 * @behavior Value descriptor for caller-selected molecular quadrature; defaults form a small
 * reference grid and do not promise production convergence.
 * @outputs version selects reference (1) or resolved production (2) grid semantics;
 * radial_points, angular_polar, and angular_azimuth set quadrature counts;
 * partition_iterations sets Becke partition smoothing; coincident_tolerance sets the
 * coincident-center threshold; tile_points bounds the grid tile size.
 * @lifetime Owns only scalar values; copying preserves the complete descriptor.
 * @errors Construction does not validate the grid; native preparation validates it.
 * @units coincident_tolerance is in Bohr; point/iteration counts and version are dimensionless.
 */
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
/** @native-contract generativeqc::KsComposition
 * @behavior Own a caller-supplied KS primitive composition for native preparation; this builder
 * is not a functional-name registry and adds no method or property capabilities.
 * @lifetime Owns copied/moved strings, exchange terms, grid, and optional radii. Copies are
 * independent; a successfully prepared Calculation does not borrow builder storage.
 * @errors Explicit operations can throw std::invalid_argument, Error, or standard allocation
 * exceptions as documented below; native preparation validates physical admissibility.
 * @execution Serialize mutation with reads/preparation using the same builder. Prepared
 * calculations obey their own Context lifetime and execution rules.
 */
class KsComposition {
 public:
  /** @native-contract generativeqc::KsComposition::KsComposition
   * @behavior Initialize an empty explicit KS composition for a selected native DFT carrier.
   * @inputs carrier identifies the eventual DFT method; scf_domain names its compiler-owned
   * numerical domain; spin_channels is 1 for RKS or 2 for UKS.
   * @lifetime Owns the supplied domain string; retains no caller storage.
   * @errors Throws std::invalid_argument for an empty domain or a spin count other than 1/2.
   * Carrier/domain compatibility is deferred to prepare; allocation may throw.
   * @execution Synchronous value construction without native execution.
   */
  KsComposition(generativeqc_method carrier, std::string scf_domain, std::uint32_t spin_channels)
      : carrier_(carrier), scf_domain_(std::move(scf_domain)), spin_channels_(spin_channels) {
    if (scf_domain_.empty() || (spin_channels_ != 1 && spin_channels_ != 2))
      throw std::invalid_argument("KS composition requires a domain and one/two spin channels");
  }

  /** @native-contract generativeqc::KsComposition::set_grid
   * @behavior Replace the quadrature descriptor used by subsequent preparation.
   * @inputs grid follows the KsGrid field and unit conventions.
   * @outputs Returns this builder by reference for chaining.
   * @lifetime Copies the complete descriptor; previously prepared calculations are unaffected.
   * @errors No validation here; invalid grid values are rejected during native preparation.
   * @execution Synchronous mutation; serialize with other accesses to this builder.
   */
  KsComposition& set_grid(KsGrid grid) {
    grid_ = grid;
    return *this;
  }

  /** @native-contract generativeqc::KsComposition::set_element_radii
   * @behavior Replace the optional element-radius table used by subsequent preparation.
   * @inputs radii has slots 0..118 indexed by atomic number; slot zero is unused.
   * @outputs Returns this builder by reference for chaining.
   * @lifetime Owns the supplied array independently of caller storage.
   * @errors Validation is deferred to native preparation, including production-grid radii.
   * @execution Synchronous mutation; serialize with other accesses to this builder.
   * @units Radii are in Bohr.
   */
  KsComposition& set_element_radii(std::array<double, 119> radii) {
    radii_ = std::move(radii);
    return *this;
  }

  /** @native-contract generativeqc::KsComposition::set_xc_schedule
   * @behavior Select the XC execution schedule for subsequent preparation.
   * @inputs schedule is a native XC schedule identifier; the initial value is DEVICE_FUSED.
   * @outputs Returns this builder by reference for chaining.
   * @errors Native preparation validates schedule/backend support; this setter does not.
   * @execution Synchronous scalar mutation; serialize with other accesses to this builder.
   */
  KsComposition& set_xc_schedule(generativeqc_xc_execution_schedule schedule) {
    schedule_ = schedule;
    return *this;
  }

  /** @native-contract generativeqc::KsComposition::add_semilocal
   * @behavior Append one explicit semilocal component and its physical coefficient.
   * @inputs component_id names a native-supported semilocal component; coefficient is its
   * dimensionless multiplier. Composition/domain compatibility is checked during preparation.
   * @outputs Returns this builder by reference for chaining; does not deduplicate components.
   * @lifetime Owns the component string and coefficient independently of caller storage.
   * @errors Throws std::invalid_argument for an empty identifier; allocation may throw.
   * @execution Synchronous mutation; serialize with other accesses to this builder.
   */
  KsComposition& add_semilocal(std::string component_id, double coefficient) {
    if (component_id.empty()) throw std::invalid_argument("empty KS component identifier");
    components_.emplace_back(std::move(component_id), coefficient);
    return *this;
  }

  /** @native-contract generativeqc::KsComposition::add_exact_exchange
   * @behavior Append an exact-exchange term with the spin-resolved native Fock coefficient:
   * minus one half of coefficient for RKS, or minus coefficient for UKS.
   * @inputs operation selects full-, short-, or long-range exchange; coefficient is its
   * physical fraction; omega is the range parameter and defaults to zero.
   * @outputs Returns this builder by reference for chaining; does not deduplicate terms.
   * @lifetime Owns a value copy of the term.
   * @errors Allocation may throw; operator/range/composition validation is deferred to prepare.
   * @execution Synchronous mutation; serialize with other accesses to this builder.
   * @units coefficient is dimensionless; omega is in inverse Bohr and zero for full range.
   */
  KsComposition& add_exact_exchange(generativeqc_ks_exchange_operator operation, double coefficient,
                                    double omega = 0.0) {
    const double divisor = spin_channels_ == 1 ? 2.0 : 1.0;
    exchange_.push_back({operation, coefficient, omega, -coefficient / divisor});
    return *this;
  }

  /** @native-contract generativeqc::KsComposition::prepare
   * @behavior Prepare a native calculation from a snapshot of this explicit KS composition.
   * @inputs context and system must be valid; descriptor must have its ABI header initialized
   * and its method equal to this builder's DFT carrier. Replaces descriptor.ks_options in the
   * local descriptor copy; other options retain their ordinary native contract.
   * @outputs Returns an owned Calculation; no energy or forces are computed here.
   * @lifetime Native preparation copies all pointer-backed composition data before returning.
   * The builder and input system need not survive the calculation; its Context must survive it.
   * @errors Throws std::invalid_argument for carrier mismatch or a non-DFT carrier; unknown
   * carrier/query or native preparation failures throw Error. Allocation exceptions may propagate.
   * @execution Synchronous preparation; serialize with builder mutation and Context operations.
   * Native capabilities still govern execution, including rejection of unsupported DFT forces.
   * @units Grid radii/tolerances use Bohr, exchange range uses inverse Bohr, and energies use
   * Hartree when the returned calculation is subsequently executed.
   */
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
