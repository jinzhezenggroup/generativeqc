#ifndef GENERATIVEQC_CLI_OPTIONS_HPP
#define GENERATIVEQC_CLI_OPTIONS_HPP

#include <optional>
#include <stdexcept>
#include <string>

#include "cli_method_catalog.hpp"
#include "generativeqc/ks.hpp"

namespace generativeqc::cli {

class UsageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Both execution and dry-run planning resolve the same scientific inputs.
struct RunOptions {
  std::string input;
  std::string method_name{"gfn2-xtb"};
  generativeqc_method method{GENERATIVEQC_METHOD_GFN2_XTB};
  std::string basis_name{"sto-3g"};
  std::string auxiliary_basis_name;
  generativeqc_basis_representation representation{GENERATIVEQC_BASIS_CARTESIAN};
  generativeqc_density_fitting_mode density_fitting{GENERATIVEQC_DENSITY_FITTING_NONE};
  generativeqc_backend backend{GENERATIVEQC_BACKEND_CPU_REFERENCE};
  int device_id{0};
  int charge{0};
  std::uint32_t multiplicity{1};
  std::optional<std::uint32_t> max_iterations;
  std::optional<double> energy_tolerance;
  std::optional<double> density_tolerance;
  std::optional<double> screening_tolerance;
  bool input_angstrom{true};
  bool basis_explicit{false};
  bool auxiliary_basis_explicit{false};
  bool representation_explicit{false};
  bool density_fitting_explicit{false};
  const method_generated::Method* composition{nullptr};
  bool grid_explicit{false};
  generativeqc::KsGrid grid{};
  bool forces{false};
  bool json{false};
  std::uint32_t diis_history{8};
  std::uint32_t batch{1};
  std::optional<std::uint64_t> host_bytes, device_bytes, pinned_host_bytes;
  std::uint64_t host_reserve_bytes{0}, device_reserve_bytes{0};
  double headroom_fraction{0};
  generativeqc_precision_mode precision{GENERATIVEQC_PRECISION_FP64};
  std::uint64_t correlation_memory_bytes{0}, density_fitting_memory_bytes{0};
  double density_fitting_threshold{1e-10}, mp2_denominator_threshold{1e-10};
  std::uint32_t ccsd_max_iterations{100}, ccsd_diis_history{6};
  double ccsd_energy_tolerance{1e-11}, ccsd_residual_tolerance{1e-9};
  double ccsd_denominator_threshold{1e-10}, ccsd_damping{0}, ccsd_level_shift{0};
  // Native Cartesian geometry optimization controls; ignored by singlepoints.
  // Empty stages chooses the automatic GFN2 / PBE / target planner.
  std::string optimization_stages;
  std::string preopt_basis_name;
  std::string optimization_output;
  std::uint32_t optimization_max_steps{80};
  double optimization_fmax{3e-4};      // Hartree/Bohr
  double optimization_max_step{0.35};  // Bohr per atom
  bool optimization_plan_only{false};
};

}  // namespace generativeqc::cli
#endif
