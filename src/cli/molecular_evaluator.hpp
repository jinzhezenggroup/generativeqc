#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <utility>

#include "api/host_cross_overlap.hpp"
#include "api/ks_batch_capabilities.hpp"
#include "api/ks_host_hessian.hpp"
#include "api/ks_host_orbital_capacity.hpp"
#include "cli/molecular_system.hpp"
#include "cli/saddle.hpp"
#include "generativeqc/ks.hpp"
#include "generativeqc/warm_seed.h"
#include "methods/warm_seed_capacity.hpp"
#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "solver/cpu/symmetric_eigen.hpp"

namespace generativeqc::cli {
namespace molecular_detail {
inline Json rows(std::span<const double> values, std::size_t columns) {
  Json result = Json::array();
  if (!columns) return result;
  for (std::size_t begin = 0; begin < values.size(); begin += columns)
    result.push_back(std::vector<double>(values.begin() + begin, values.begin() + begin + columns));
  return result;
}
}  // namespace molecular_detail

/** Detached diagnostic state from the actual native frame, never a new solve. */
struct MolecularOrbitalSnapshot {
  generativeqc_ks_host_orbitals_receipt_v1 shape;
  std::vector<double> coefficients;
};

/** Internal fixed-Hamiltonian derivative owner, not molecular-task admission.
 * The CLI adapter applies its public qualification policy separately. Native
 * component qualification uses this same implementation without changing that
 * policy or creating a second numerical provider. Physical options and the
 * linked program are immutable borrowed state and must outlive this owner;
 * invocation resource controls may change only between synchronous calls.
 * Basis records are copied/moved into it. Context outlives all prepared frames.
 * Only explicit accepted-geometry observations update occupied-root references.
 */
class NativeMolecularEvaluator {
  using Clock = std::chrono::steady_clock;
  using Batch = std::unique_ptr<generativeqc_batch, decltype(&generativeqc_batch_destroy)>;
  using Seed = std::unique_ptr<generativeqc_warm_seed_snapshot,
                               decltype(&generativeqc_warm_seed_snapshot_destroy)>;

 public:
  NativeMolecularEvaluator(const RunOptions& options, const std::vector<generativeqc_atom>& atoms,
                           std::span<const double> masses, const SaddleInvocation& invocation,
                           const method_generated::Method& program, NativeBasisData basis)
      : options_(options),
        invocation_(invocation),
        program_(program),
        context_(generativeqc_context_descriptor{sizeof(generativeqc_context_descriptor),
                                                 GENERATIVEQC_ABI_VERSION, -1,
                                                 GENERATIVEQC_BACKEND_CPU_REFERENCE}),
        basis_(std::move(basis)),
        atoms_(atoms),
        masses_(masses.begin(), masses.end()) {
    if (atoms_.empty() || masses_.size() != atoms_.size() ||
        !std::all_of(masses_.begin(), masses_.end(),
                     [](double mass) { return std::isfinite(mass) && mass > 0; }))
      throw std::invalid_argument("occupied-root tracking requires one positive finite atom mass");
    work = {{"scf_iterations", 0},
            {"scf_fock_builds", 0},
            {"nuclear_response_solves", 0},
            {"response_iterations", 0},
            {"induced_fock_actions", 0},
            {"first_eri_primitives", 0},
            {"curvature_eri_primitives", 0},
            {"first_one_electron_primitives", 0},
            {"curvature_one_electron_primitives", 0},
            {"source_xc_point_visits", 0},
            {"curvature_xc_point_visits", 0},
            {"response_xc_point_visits", 0},
            {"ecp_kernel_blocks", 0},
            {"ecp_radial_layers", 0},
            {"ecp_ao_primitive_points", 0},
            {"ecp_contraction_terms", 0},
            {"d3_pair_visits", 0},
            {"d3_reference_terms", 0}};
  }

  double elapsed() const { return std::chrono::duration<double>(Clock::now() - started_).count(); }
  generativeqc_basis_representation representation() const { return basis_.representation; }

  Json records = Json::array(), work;

  /** Reserve externally owned panels outside the prepared electronic lease.
   * Keep the two-frame envelope fixed so the first admission also covers a
   * moved trial and no accepted frame has to be evicted to make room for it.
   */
  void configure_host_budget(std::uint64_t limit, std::size_t caller_bytes,
                             std::size_t corrector_bytes, std::size_t harmonic_bytes) {
    using runtime::size_add;
    using runtime::size_mul;
    const auto count = 3 * atoms_.size();
    const auto basis = orbital_basis_dimension();
    const auto orbitals = api::host_orbital_capacity(basis, program_.spin);
    const auto seed_bytes =
        invocation_.accepted_frame_warm_start
            ? methods::warm_seed_item_capacity(orbitals.output_elements, atoms_.size()).total_bytes
            : 0;
    auto basis_bytes = size_mul(basis_.shells.capacity(), sizeof(generativeqc_shell));
    basis_bytes = size_add(basis_bytes,
                           size_mul(basis_.primitives.capacity(), sizeof(generativeqc_primitive)));
    basis_bytes =
        size_add(basis_bytes, size_mul(basis_.core_electrons.capacity(), sizeof(std::int32_t)));
    basis_bytes =
        size_add(basis_bytes, size_mul(basis_.ecp_terms.capacity(), sizeof(generativeqc_ecp_term)));
    // Public frame handles own normalized shells independently of the native
    // copy in the batch inventory. Normalization keeps primitive counts fixed.
    auto system_bytes = size_mul(atoms_.size(), sizeof(core::Atom));
    system_bytes = size_add(system_bytes, size_mul(basis_.shells.size(), sizeof(core::Shell)));
    system_bytes = size_add(system_bytes, size_mul(basis_.ecp_terms.size(), sizeof(core::EcpTerm)));
    for (const auto& shell : basis_.shells)
      system_bytes =
          size_add(system_bytes, size_mul(shell.primitive_count, sizeof(core::Primitive)));
    const auto atoms_bytes = size_mul(atoms_.size(), sizeof(generativeqc_atom));
    const auto position_bytes = size_mul(count, sizeof(double));
    const auto frame_bytes =
        size_add(size_add(system_bytes, atoms_bytes),
                 size_add(position_bytes, size_mul(orbitals.output_elements, sizeof(double))));
    auto common = size_add(caller_bytes, basis_bytes);
    common = size_add(common, size_add(atoms_bytes, size_mul(masses_.size(), sizeof(double))));
    // Capture publishes a distinct snapshot while the batch still owns its
    // converged density. Keep room for it and the previous accepted seed.
    common = size_add(common, size_mul(2, size_add(frame_bytes, seed_bytes)));
    const auto harmonic_peak = size_add(common, harmonic_bytes);
    // Use the existing conservative root algebra bound. The current C output
    // and retained C are in the frames; physical export scratch is in the lease.
    const auto root_algebra = size_mul(root_algebra_elements(basis), sizeof(double));
    auto outside = size_add(common, size_add(corrector_bytes, root_algebra));
    outside = size_add(outside, size_add(position_bytes, 2 * sizeof(double) * program_.spin));
    if (harmonic_peak > limit || outside >= limit)
      throw std::length_error(
          "saddle --host-bytes cannot cover corrector, frame and harmonic panels");
    host_.emplace(HostBudget{limit, outside, harmonic_peak, seed_bytes});
  }

  Json host_budget_receipt() const {
    if (!host_) return nullptr;
    return {{"effective_host_limit_bytes", host_->limit},
            {"minimum_host_peak_bytes", host_->minimum ? Json(host_->minimum) : Json(nullptr)},
            {"selected_host_peak_bytes", host_->peak ? Json(host_->peak) : Json(nullptr)},
            {"harmonic_host_peak_bytes", host_->harmonic_peak},
            {"warm_seed_snapshot_bytes_per_frame", host_->seed_bytes},
            {"outside_electronic_lease_bytes", host_->outside_electronic_peak},
            {"electronic_host_peak_bytes", host_->electronic_peak},
            {"derivative_numeric_cap_bytes", host_->workspace}};
  }

  bool uses_ecp() const {
    return !basis_.ecp_terms.empty() ||
           std::any_of(basis_.core_electrons.begin(), basis_.core_electrons.end(),
                       [](std::int32_t core) { return core != 0; });
  }

  /** Match the qualified def2-ECP operator, not a user-supplied name or core count. */
  bool uses_only_spherical_def2_ecp(int atomic_number) const {
    // Pinned BSE def2-ECP (tests/data/global_hybrid_ecp/manifest.json).
    // All rows have radial power 2; -1 denotes the local channel.
    struct Term {
      int channel;
      double exponent, coefficient;
    };
    constexpr std::array cs46{Term{-1, 3.123269, -28.884309}, Term{0, 4.079750, 84.547730},
                              Term{0, 2.417406, 16.654173},   Term{0, 3.123269, 28.884309},
                              Term{1, 5.514080, 157.049059},  Term{1, 2.160316, 26.423307},
                              Term{1, 3.123269, 28.884309},   Term{2, 1.807410, 13.172753},
                              Term{2, 0.858182, 3.342833},    Term{2, 3.123269, 28.884309}};
    constexpr std::array i28{Term{-1, 19.458609, -21.842040}, Term{-1, 19.349260, -28.468191},
                             Term{-1, 4.823767, -0.243713},   Term{-1, 4.884315, -0.320804},
                             Term{0, 40.015835, 49.994293},   Term{0, 17.429747, 281.025317},
                             Term{0, 9.005484, 61.573326},    Term{0, 19.458609, 21.842040},
                             Term{0, 19.349260, 28.468191},   Term{0, 4.823767, 0.243713},
                             Term{0, 4.884315, 0.320804},     Term{1, 15.355466, 67.442841},
                             Term{1, 14.971833, 134.881137},  Term{1, 8.960164, 14.675051},
                             Term{1, 8.259096, 29.375666},    Term{1, 19.458609, 21.842040},
                             Term{1, 19.349260, 28.468191},   Term{1, 4.823767, 0.243713},
                             Term{1, 4.884315, 0.320804},     Term{2, 15.068908, 35.439529},
                             Term{2, 14.555322, 53.176057},   Term{2, 6.718647, 9.067195},
                             Term{2, 6.456393, 13.206937},    Term{2, 1.191779, 0.089335},
                             Term{2, 1.291157, 0.052380},     Term{2, 19.458609, 21.842040},
                             Term{2, 19.349260, 28.468191},   Term{2, 4.823767, 0.243713},
                             Term{2, 4.884315, 0.320804}};
    if (basis_.representation != GENERATIVEQC_BASIS_SPHERICAL || basis_.ecp_terms.empty() ||
        basis_.core_electrons.size() != atoms_.size() ||
        (atomic_number != 55 && atomic_number != 53))
      return false;
    const auto expected =
        atomic_number == 55 ? std::span<const Term>{cs46} : std::span<const Term>{i28};
    const auto expected_core = atomic_number == 55 ? 46 : 28;
    // Basis expansion emits terms in atom order, but the radial row order is immaterial.
    auto term = basis_.ecp_terms.begin();
    for (std::size_t atom = 0; atom < atoms_.size(); ++atom) {
      const auto core = basis_.core_electrons[atom];
      if (!core) continue;
      if (atoms_[atom].atomic_number != atomic_number || core != expected_core) return false;
      std::bitset<i28.size()> matched;
      for (; term != basis_.ecp_terms.end() && term->atom_index == atom; ++term) {
        const auto row = std::find_if(expected.begin(), expected.end(), [&](const Term& value) {
          return term->channel == value.channel && term->power == 2 &&
                 term->exponent == value.exponent && term->coefficient == value.coefficient;
        });
        if (row == expected.end()) return false;
        const auto index = static_cast<std::size_t>(row - expected.begin());
        if (matched.test(index)) return false;
        matched.set(index);
      }
      if (matched.count() != expected.size()) return false;
    }
    return term == basis_.ecp_terms.end();
  }

  /** Rejected trials cannot replace the occupied spaces of an accepted frame. */
  void accept_geometry(std::span<const double> positions) {
    if (current_ && std::equal(positions.begin(), positions.end(), current_->positions.begin(),
                               current_->positions.end()))
      accepted_ = current_;
  }

  /** Release a failed/current trial, retaining the last accepted root only. */
  void discard_current() noexcept { current_.reset(); }

  /** Export the exact matching current/accepted orbitals for state qualification.
   * The caller's separate bound covers this detached copy, not the retained
   * tracking owner. Root export is diagnostic only and never supplies a force
   * or Hessian. A rejected trial cannot masquerade as the accepted geometry.
   */
  MolecularOrbitalSnapshot orbital_snapshot(std::span<const double> positions,
                                            std::size_t maximum_bytes) const {
    const auto matches = [&](const auto& frame) {
      return frame && std::equal(positions.begin(), positions.end(), frame->positions.begin(),
                                 frame->positions.end());
    };
    const auto frame = matches(current_) ? current_ : matches(accepted_) ? accepted_ : nullptr;
    if (!frame) throw std::invalid_argument("molecular orbital snapshot has no matching geometry");
    if (frame->coefficients.size() > maximum_bytes / sizeof(double))
      throw std::length_error("molecular orbital snapshot exceeds its detached output budget");
    return {frame->orbitals, frame->coefficients};
  }

  /** Serialize only occupied columns from the accepted frame, without a new SCF
   * or a detached copy of the virtual coefficients. The budget counts FP64
   * payload bytes, not JSON representation overhead.
   */
  Json accepted_occupied_state(std::span<const double> positions,
                               std::uint64_t maximum_bytes) const {
    if (!accepted_ || !std::equal(positions.begin(), positions.end(), accepted_->positions.begin(),
                                  accepted_->positions.end()))
      throw std::logic_error("occupied-state export has no matching accepted frame");
    const auto& shape = accepted_->orbitals;
    const auto beta = shape.spins == 2 ? shape.beta_occupied : 0;
    const auto bytes = sizeof(double) * shape.basis * (shape.alpha_occupied + beta);
    if (bytes > maximum_bytes)
      throw std::length_error("occupied-state export exceeds --occupied-state-budget-bytes");
    Json channels = Json::array();
    for (std::size_t spin = 0; spin < shape.spins; ++spin) {
      const auto occupied = spin == 0 ? shape.alpha_occupied : beta;
      Json coefficients = Json::array();
      for (std::size_t ao = 0; ao < shape.basis; ++ao) {
        const auto begin =
            accepted_->coefficients.begin() + (spin * shape.basis + ao) * shape.basis;
        coefficients.push_back(std::vector<double>(begin, begin + occupied));
      }
      channels.push_back(
          {{"occupied_orbitals", occupied}, {"coefficients", std::move(coefficients)}});
    }
    return {{"basis_dimension", shape.basis},
            {"numeric_bytes", bytes},
            {"channels", std::move(channels)}};
  }

 private:
  struct HostBudget {
    std::uint64_t limit;
    std::size_t outside, harmonic_peak, seed_bytes;
    std::size_t minimum{}, peak{}, outside_electronic_peak{}, electronic_peak{}, workspace{};
  };

  struct Frame {
    std::unique_ptr<System> system;
    Batch batch{nullptr, generativeqc_batch_destroy};
    std::vector<generativeqc_atom> atoms;
    std::vector<double> positions, coefficients;
    generativeqc_ks_host_orbitals_receipt_v1 orbitals;
    Seed seed{nullptr, generativeqc_warm_seed_snapshot_destroy};
    std::size_t seed_bytes{};
  };

  void check(generativeqc_status status) const {
    if (status != GENERATIVEQC_STATUS_SUCCESS) {
      const char* detail = generativeqc_context_get_last_detail(context_.get());
      throw Error(status, detail && *detail ? std::string(detail)
                                            : "native CPU molecular derivative failed (status " +
                                                  std::to_string(status) + ")");
    }
  }

 protected:
  void record(std::string_view stage, std::string_view status, std::span<const double> positions,
              Json details = Json::object()) {
    details["stage"] = stage;
    details["status"] = status;
    details["elapsed_seconds"] = elapsed();
    details["coordinates_bohr"] = molecular_detail::rows(positions, 3);
    records.push_back(details);
    if (invocation_.progress_json) std::cerr << details.dump() << '\n';
  }

 private:
  /** Publish unfinished native phases even when a later call never returns.
   * Work counters remain completed-work receipts, not estimates of in-flight
   * SCF/force work. Failed phases do not publish an accepted molecular state.
   */
  template <class Operation>
  void run_phase(std::string_view stage, std::span<const double> positions, Operation&& operation) {
    const auto started = Clock::now();
    record(stage, "start", positions);
    try {
      operation();
    } catch (const std::exception& failure) {
      record(stage, "failed", positions,
             {{"detail", failure.what()},
              {"phase_seconds", std::chrono::duration<double>(Clock::now() - started).count()}});
      throw;
    }
    record(stage, "complete", positions,
           {{"phase_seconds", std::chrono::duration<double>(Clock::now() - started).count()}});
  }

  void add_work(const char* key, std::uint64_t amount) {
    work[key] = work.at(key).get<std::uint64_t>() + amount;
  }

  generativeqc_ks_host_hessian_options_v1 derivative_controls(const Frame& frame) const {
    const auto& input = invocation_.derivatives;
    auto controls = api::default_host_hessian_options();
    controls.expected_atoms = frame.atoms.data();
    controls.expected_atom_count = frame.atoms.size();
    controls.expected_charge = options_.charge;
    controls.expected_multiplicity = options_.multiplicity;
    controls.numeric_bytes = input.numeric_bytes;
    controls.integral_bytes = input.integral_bytes;
    controls.grid_bytes = input.grid_bytes;
    controls.response_bytes = input.response_bytes;
    controls.gmres_bytes = input.response_bytes;
    controls.ecp_bytes = input.ecp_bytes;
    controls.dispersion_bytes = input.dispersion_bytes;
    controls.output_bytes = input.output_bytes;
    controls.max_eri_primitives = input.max_eri_primitives;
    controls.max_map_terms = input.max_map_terms;
    controls.max_d3_pairs = input.max_d3_pairs;
    controls.relative_tolerance = input.response_rtol;
    controls.absolute_tolerance = input.response_atol_explicit ? input.response_atol : 1e-12;
    controls.restart = input.response_restart;
    controls.max_iterations = input.response_iterations;
    controls.symmetry_absolute_tolerance = invocation_.controls.raw_symmetry_tolerance;
    controls.symmetry_relative_tolerance = 0;
    return controls;
  }

  /** Admit before native provider setup or SCF, using the same normalized
   * system and exact MethodIR which the force/Hessian owner will consume.
   */
  void prepare_host_batch(Frame& frame, std::size_t basis, std::size_t overlap_workspace,
                          const generativeqc_method_descriptor& method,
                          const generativeqc_named_ks_descriptor& named,
                          const generativeqc_rks_cold_fock_options_v1* cold) {
    auto controls = derivative_controls(frame);
    std::size_t derivative_floor{};
    check(generativeqc_ks_hessian_cpu_minimum_bytes_v1(
        context_.get(), frame.system->get(), named.identity, 1, &controls, &derivative_floor));
    if (derivative_floor > controls.numeric_bytes)
      throw std::length_error("saddle Hessian exceeds --numeric-budget-bytes");
    const auto floor =
        std::max(derivative_floor, api::host_orbital_workspace_bytes(basis, program_.spin));
    const auto outside = runtime::size_add(host_->outside, overlap_workspace);
    if (outside >= host_->limit || floor > host_->limit - outside)
      throw std::length_error("saddle --host-bytes cannot cover callback and derivative panels");
    const auto available = host_->limit - outside;
    HostResources pool(available);
    generativeqc_batch* prepared{};
    std::size_t workspace{};
    const auto ceiling = std::max(floor, static_cast<std::size_t>(controls.numeric_bytes));
    const auto flags = invocation_.accepted_frame_warm_start
                           ? generativeqc_batch_flags{GENERATIVEQC_BATCH_ENABLE_WARM_STARTS}
                           : generativeqc_batch_flags{0};
    const auto status = generativeqc_saddle_prepare_cpu_host_batch_v1(
        context_.get(), frame.system->get(), &method, &named, cold, flags, pool.get(), available,
        floor, ceiling, &prepared, &workspace);
    if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) {
      const char* detail = generativeqc_context_get_last_detail(context_.get());
      throw Error(status, std::string("saddle --host-bytes preparation refused: ") + detail);
    }
    check(status);
    frame.batch.reset(prepared);
    if (flags) {
      generativeqc_warm_seed_capacity_v1 seed{};
      seed.struct_size = sizeof(seed);
      seed.abi_version = GENERATIVEQC_ABI_VERSION;
      check(generativeqc_batch_query_warm_seed_capacity_v1(frame.batch.get(), &seed));
      if (seed.total_bytes != host_->seed_bytes)
        throw std::logic_error("saddle host seed reservation disagrees with prepared dimensions");
    }
    const auto snapshot = pool.snapshot();
    const auto electronic = runtime::size_add(snapshot.retained_bytes, snapshot.workspace_bytes);
    const auto combined = runtime::size_add(outside, electronic);
    const auto peak = std::max(combined, host_->harmonic_peak);
    host_->minimum =
        std::max({host_->minimum, host_->harmonic_peak, combined - (workspace - floor)});
    if (peak >= host_->peak) {
      host_->peak = peak;
      host_->outside_electronic_peak = outside;
      host_->electronic_peak = electronic;
      host_->workspace = workspace;
    }
  }

  std::vector<double> occupied_overlaps(const Frame& current, std::array<double, 3>& translation,
                                        std::size_t cross_overlap_workspace) {
    const auto& shape = current.orbitals;
    std::vector<double> result(shape.spins, 1);
    if (!accepted_) return result;
    const auto& previous = *accepted_;
    if (shape.basis != previous.orbitals.basis || shape.spins != previous.orbitals.spins ||
        shape.alpha_occupied != previous.orbitals.alpha_occupied ||
        shape.beta_occupied != previous.orbitals.beta_occupied)
      throw std::runtime_error("occupied spin populations changed");
    const std::size_t basis = shape.basis;
    const double maximum_mass = *std::max_element(masses_.begin(), masses_.end());
    double total_weight = 0;
    for (std::size_t atom = 0; atom < masses_.size(); ++atom) {
      const double weight = masses_[atom] / maximum_mass;
      total_weight += weight;
      for (std::size_t axis = 0; axis < 3; ++axis)
        translation[axis] +=
            weight * (current.positions[3 * atom + axis] - previous.positions[3 * atom + axis]);
    }
    for (double& component : translation) component /= total_weight;
    // Registration belongs only to the overlap diagnostic. SCF and all
    // derivatives continue to use the original trial system and angular grid.
    std::vector<double> metric(basis * basis);
    check(generativeqc_system_cross_overlap_host_v1(
        context_.get(), current.system->get(), previous.system->get(), cross_overlap_workspace,
        metric.data(), metric.size(), nullptr, translation.data()));
    for (std::size_t spin = 0; spin < shape.spins; ++spin) {
      const std::size_t occupied = spin == 0 ? shape.alpha_occupied : shape.beta_occupied;
      if (!occupied) continue;
      std::vector<double> product(basis * occupied), overlaps(occupied * occupied),
          gram(occupied * occupied);
      const auto offset = spin * basis * basis;
      for (std::size_t row = 0; row < basis; ++row)
        for (std::size_t inner = 0; inner < basis; ++inner)
          for (std::size_t orbital = 0; orbital < occupied; ++orbital)
            product[row * occupied + orbital] +=
                metric[row * basis + inner] *
                previous.coefficients[offset + inner * basis + orbital];
      for (std::size_t row = 0; row < occupied; ++row)
        for (std::size_t column = 0; column < occupied; ++column)
          for (std::size_t inner = 0; inner < basis; ++inner)
            overlaps[row * occupied + column] +=
                current.coefficients[offset + inner * basis + row] *
                product[inner * occupied + column];
      for (std::size_t row = 0; row < occupied; ++row)
        for (std::size_t column = 0; column < occupied; ++column)
          for (std::size_t inner = 0; inner < occupied; ++inner)
            gram[row * occupied + column] +=
                overlaps[inner * occupied + row] * overlaps[inner * occupied + column];
      if (!std::all_of(gram.begin(), gram.end(), [](double value) { return std::isfinite(value); }))
        throw std::runtime_error("nonfinite occupied-root metric");
      const auto prepared = solver::cpu::prepare_owned_symmetric_eigen(
          occupied, solver::cpu::SymmetricEigenFamily::scalar_owned_row_major);
      const auto spectrum = solver::cpu::execute_symmetric_eigen(prepared, std::move(gram));
      const double smallest = spectrum.values.front();
      if (smallest < -1e-10) throw std::runtime_error("indefinite occupied-root Gram matrix");
      result[spin] = std::sqrt(std::max(0.0, smallest));
      if (!std::isfinite(result[spin]) || result[spin] < invocation_.minimum_occupied_overlap) {
        std::ostringstream message;
        message.precision(17);
        message << "occupied root continuity lost in spin channel " << spin << ": minimum overlap "
                << result[spin] << ", required " << invocation_.minimum_occupied_overlap;
        throw std::runtime_error(message.str());
      }
    }
    return result;
  }

 public:
  /** Prepare a cold native frame and publish energy/gradient only after root checks.
   * The matching prepared batch stays owned until Hessian consumption/discard.
   */
  void energy_gradient(std::span<const double> positions, double& energy,
                       std::span<double> gradient) {
    current_.reset();
    if (positions.empty() || positions.size() != 3 * atoms_.size() ||
        gradient.size() != positions.size() ||
        !std::all_of(positions.begin(), positions.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("invalid molecular energy/gradient coordinate binding");
    const auto basis_dimension = orbital_basis_dimension();
    const auto orbital_capacity = api::host_orbital_capacity(basis_dimension, program_.spin);
    // The immutable basis layout is shared across frames; mass registration
    // moves only the centres. Use the accepted system to size both sides.
    std::size_t cross_overlap_workspace{};
    if (accepted_)
      check(generativeqc_system_cross_overlap_host_v1(
          context_.get(), accepted_->system->get(), accepted_->system->get(),
          std::numeric_limits<std::size_t>::max(), nullptr, 0, &cross_overlap_workspace, nullptr));
    const auto tracking_bytes = runtime::size_add(
        root_tracking_bytes(basis_dimension, orbital_capacity), cross_overlap_workspace);
    if (tracking_bytes > invocation_.root_bytes)
      throw std::length_error("occupied-root tracking exceeds --root-budget-bytes");
    auto frame = std::make_shared<Frame>();
    frame->positions.assign(positions.begin(), positions.end());
    frame->atoms = atoms_;
    for (std::size_t atom = 0; atom < atoms_.size(); ++atom) {
      frame->atoms[atom].x = positions[3 * atom];
      frame->atoms[atom].y = positions[3 * atom + 1];
      frame->atoms[atom].z = positions[3 * atom + 2];
    }
    const generativeqc_system_descriptor geometry{
        sizeof(geometry),         GENERATIVEQC_ABI_VERSION,
        frame->atoms.data(),      static_cast<std::uint32_t>(frame->atoms.size()),
        basis_.shells.data(),     static_cast<std::uint32_t>(basis_.shells.size()),
        basis_.primitives.data(), static_cast<std::uint32_t>(basis_.primitives.size()),
        options_.charge,          options_.multiplicity,
        basis_.representation};
    run_phase("system_prepare", positions, [&] {
      frame->system = std::make_unique<System>(make_molecular_system(context_, geometry, basis_));
    });
    generativeqc_ks_options grid{};
    grid.struct_size = sizeof(grid);
    grid.abi_version = GENERATIVEQC_ABI_VERSION;
    grid.grid_version = options_.grid.version;
    grid.radial_points = options_.grid.radial_points;
    grid.angular_polar = options_.grid.angular_polar;
    grid.angular_azimuth = options_.grid.angular_azimuth;
    grid.partition_iterations = options_.grid.partition_iterations;
    grid.coincident_tolerance = options_.grid.coincident_tolerance;
    grid.tile_points = options_.grid.tile_points;
    methods::named_ks::electronic_options(program_, grid);
    generativeqc_method_descriptor method{};
    method.struct_size = sizeof(method);
    method.abi_version = GENERATIVEQC_ABI_VERSION;
    method.method = program_.carrier;
    method.max_iterations = options_.max_iterations.value_or(240);
    method.diis_history = options_.diis_history;
    method.energy_tolerance = options_.energy_tolerance.value_or(1e-13);
    method.density_tolerance = options_.density_tolerance.value_or(1e-11);
    method.screening_tolerance = options_.screening_tolerance.value_or(1e-14);
    method.ks_options = &grid;
    generativeqc_named_ks_descriptor named{};
    named.struct_size = sizeof(named);
    named.abi_version = GENERATIVEQC_ABI_VERSION;
    named.name = program_.name.data();
    named.identity = program_.identity.data();
    named.dispersion_maximum_bytes = invocation_.derivatives.dispersion_bytes;
    named.cpu_packed_eri_maximum_bytes = invocation_.derivatives.cpu_packed_eri_bytes;
    const generativeqc_system* systems[]{frame->system->get()};
    const auto* cold =
        invocation_.cold_fock_damping && !accepted_ ? &*invocation_.cold_fock_damping : nullptr;
    run_phase("batch_prepare", positions, [&] {
      if (host_) {
        auto aggregate_overlap = cross_overlap_workspace;
        if (!accepted_)
          check(generativeqc_system_cross_overlap_host_v1(
              context_.get(), frame->system->get(), frame->system->get(),
              std::numeric_limits<std::size_t>::max(), nullptr, 0, &aggregate_overlap, nullptr));
        prepare_host_batch(*frame, basis_dimension, aggregate_overlap, method, named, cold);
      } else {
        generativeqc_batch* prepared{};
        const auto flags =
            invocation_.accepted_frame_warm_start ? GENERATIVEQC_BATCH_ENABLE_WARM_STARTS : 0;
        if (cold)
          check(generativeqc_batch_prepare_named_ks_cold_rks_v1(context_.get(), systems, 1, &method,
                                                                &named, cold, flags, &prepared));
        else
          check(generativeqc_batch_prepare_named_ks_v1(context_.get(), systems, 1, &method, &named,
                                                       flags, &prepared));
        frame->batch.reset(prepared);
      }
      if (invocation_.accepted_frame_warm_start && accepted_) {
        // Model, basis, charge and spin are immutable within this evaluator.
        if (!accepted_->seed) throw std::logic_error("accepted derivative frame has no seed");
        check(generativeqc_batch_restore_warm_seeds_v1(frame->batch.get(), accepted_->seed.get()));
      }
      generativeqc_property_flags supported{};
      check(generativeqc_ks_batch_supported_properties_v1(frame->batch.get(), 0, &supported));
      if (!(supported & GENERATIVEQC_PROPERTY_FORCES))
        throw std::runtime_error("prepared native CPU force owner unavailable; no Python fallback");
    });
    std::vector<double> forces(positions.size());
    generativeqc_batch_item_result_descriptor primal{};
    primal.struct_size = sizeof(primal);
    primal.abi_version = GENERATIVEQC_ABI_VERSION;
    primal.forces = forces.data();
    primal.force_count = forces.size();
    std::uint64_t fock_builds{};
    run_phase("scf_forces", positions, [&] {
      check(generativeqc_batch_execute(frame->batch.get(), nullptr, 0, &primal, 1));
      // A fallback may leave no complete work receipt. Report the violated
      // seed policy before querying counters that cannot describe both solves.
      if (invocation_.accepted_frame_warm_start) {
        if (primal.warm_start_fallback)
          throw std::runtime_error("accepted-frame SCF failed; cold fallback result refused");
        if ((primal.warm_start_used != 0) != bool(accepted_))
          throw std::runtime_error("accepted-frame SCF seed policy was not satisfied");
      }
      add_work("scf_iterations", primal.iterations);
      check(generativeqc_batch_get_last_fock_builds(frame->batch.get(), 0, &fock_builds));
      add_work("scf_fock_builds", fock_builds);
      if (primal.status == GENERATIVEQC_STATUS_NOT_CONVERGED && !primal.converged)
        throw Error(primal.status, "native CPU SCF did not converge");
      check(primal.status);
      if (!primal.converged || primal.executed_backend != GENERATIVEQC_BACKEND_CPU_REFERENCE ||
          !std::isfinite(primal.energy) ||
          !std::all_of(forces.begin(), forces.end(),
                       [](double value) { return std::isfinite(value); }))
        throw std::runtime_error("native CPU SCF/forces did not converge");
    });
    auto controls = derivative_controls(*frame);
    controls.numeric_bytes = invocation_.root_bytes;
    check(generativeqc_ks_orbitals_cpu_v1(frame->batch.get(), 0, &controls, nullptr, 0,
                                          &frame->orbitals));
    const auto& shape = frame->orbitals;
    if (shape.basis != basis_dimension || shape.spins != program_.spin ||
        shape.output_elements != orbital_capacity.output_elements ||
        shape.numeric_reservation_bytes != orbital_capacity.numeric_bytes)
      throw std::runtime_error("occupied-root capacity disagrees with the prepared frame");
    frame->coefficients.resize(shape.output_elements);
    check(generativeqc_ks_orbitals_cpu_v1(frame->batch.get(), 0, &controls,
                                          frame->coefficients.data(), frame->coefficients.size(),
                                          &frame->orbitals));
    std::array<double, 3> registration{};
    const auto overlaps = occupied_overlaps(*frame, registration, cross_overlap_workspace);
    std::vector<std::uint64_t> occupied_counts{shape.alpha_occupied};
    if (shape.spins == 2) occupied_counts.push_back(shape.beta_occupied);
    record("energy_gradient", "complete", positions,
           {{"energy_hartree", primal.energy},
            {"scf_iterations", primal.iterations},
            {"scf_fock_builds", fock_builds},
            {"accepted_frame_warm_start", primal.warm_start_used != 0},
            {"cold_fock_prefix_iterations", cold ? cold->prefix_iterations : 0},
            {"minimum_occupied_overlaps_by_spin", overlaps},
            {"occupied_overlap_coordinate_frame", "mass-centroid-translation-registered"},
            {"root_registration_translation_bohr", registration},
            {"occupied_counts", occupied_counts},
            {"orbital_spin_channels", shape.spins},
            {"root_cross_overlap_workspace_bytes", cross_overlap_workspace},
            {"root_numeric_reservation_bytes", tracking_bytes}});
    energy = primal.energy;
    for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate)
      gradient[coordinate] = -forces[coordinate];
    current_ = std::move(frame);
  }

  /** Consume the exact retained frame; no SCF is repeated for its Hessian. */
  void hessian(std::span<const double> positions, std::span<double> output) {
    if (positions.empty() || positions.size() != 3 * atoms_.size() ||
        output.size() / positions.size() != positions.size() || output.size() % positions.size())
      throw std::invalid_argument("invalid molecular Hessian coordinate/output binding");
    if (!current_ || !current_->batch ||
        !std::equal(positions.begin(), positions.end(), current_->positions.begin(),
                    current_->positions.end()))
      throw std::runtime_error("Hessian geometry lacks its prepared energy/gradient batch");
    const auto controls = derivative_controls(*current_);
    auto receipt = api::empty_host_hessian_receipt();
    check(generativeqc_ks_hessian_cpu_v1(current_->batch.get(), 0, &controls, nullptr, 0,
                                         output.data(), output.size(), &receipt));
    add_work("nuclear_response_solves", receipt.completed_directions);
    add_work("response_iterations", receipt.response_iterations);
    add_work("induced_fock_actions", receipt.induced_fock_actions);
    add_work("first_eri_primitives", receipt.first_eri_primitives);
    add_work("curvature_eri_primitives", receipt.curvature_eri_primitives);
    add_work("first_one_electron_primitives", receipt.first_one_electron_primitives);
    add_work("curvature_one_electron_primitives", receipt.curvature_one_electron_primitives);
    add_work("source_xc_point_visits", receipt.source_xc_point_visits);
    add_work("curvature_xc_point_visits", receipt.curvature_xc_point_visits);
    add_work("response_xc_point_visits", receipt.response_xc_point_visits);
    // These are Hessian-source counts; the separate primal force is not included.
    add_work("ecp_kernel_blocks", receipt.ecp_kernel_blocks);
    add_work("ecp_radial_layers", receipt.ecp_radial_layers);
    add_work("ecp_ao_primitive_points", receipt.ecp_ao_primitive_points);
    add_work("ecp_contraction_terms", receipt.ecp_contraction_terms);
    add_work("d3_pair_visits", receipt.d3_pair_visits);
    add_work("d3_reference_terms", receipt.d3_reference_terms);
    if (invocation_.accepted_frame_warm_start) capture_seed(*current_, positions);
    current_->batch.reset();
    record("hessian", "complete", positions,
           {{"completed_directions", receipt.completed_directions},
            {"root_seed_snapshot_bytes", current_->seed_bytes},
            {"response_residual_norm", receipt.response_residual_norm},
            {"raw_symmetry_error", receipt.raw_symmetry_error},
            {"numeric_reservation_bytes", receipt.numeric_reservation_bytes},
            {"native_execution_seconds", receipt.execution_seconds},
            {"d3_energy", receipt.d3_energy}});
  }

 private:
  /** Geometry changes do not change the basis layout or linked spin count. Use
   * the same shell sizing as the native owner before preparing an SCF batch. */
  std::size_t orbital_basis_dimension() const {
    std::size_t dimension{};
    for (const auto& shell : basis_.shells)
      dimension = runtime::size_add(
          dimension, molecule::shell_ao_count(shell.angular_momentum, basis_.representation));
    return dimension;
  }

  /** Preserve the existing conservative overlap-panel allowance even on the
   * first frame. The accepted frame's coefficients and seed remain live while
   * the trial is being verified. Native seed capture applies its own bound. */
  std::size_t root_tracking_bytes(std::size_t basis, api::HostOrbitalCapacity capacity) const {
    const auto retained_and_overlap =
        runtime::size_add(retained_root_elements(), root_algebra_elements(basis));
    return runtime::size_add(capacity.numeric_bytes,
                             runtime::size_mul(retained_and_overlap, sizeof(double)));
  }

  static std::size_t root_algebra_elements(std::size_t basis) {
    return runtime::size_mul(8, runtime::size_mul(basis, basis));
  }

  std::size_t retained_root_elements() const {
    return accepted_ ? runtime::size_add(accepted_->coefficients.size(),
                                         accepted_->seed_bytes / sizeof(double))
                     : 0;
  }

  /** Capture only a completed derivative frame. The accepted-frame pointer
   * commits the snapshot; a rejected trial's snapshot cannot seed another SCF. */
  void capture_seed(Frame& frame, std::span<const double> positions) const {
    const auto limit = invocation_.root_bytes / sizeof(double);
    const auto retained = retained_root_elements();
    if (retained > limit || frame.coefficients.size() > limit - retained)
      throw std::length_error("occupied tracking exceeds --root-budget-bytes");
    auto available = (limit - retained - frame.coefficients.size()) * sizeof(double);
    if (host_) available = std::min(available, host_->seed_bytes);
    generativeqc_warm_seed_snapshot_info_v1 info{};
    info.struct_size = sizeof(info);
    info.abi_version = GENERATIVEQC_ABI_VERSION;
    generativeqc_warm_seed_snapshot* captured{};
    check(generativeqc_batch_capture_warm_seeds_v1(frame.batch.get(), available, &info, &captured));
    Seed candidate(captured, generativeqc_warm_seed_snapshot_destroy);
    generativeqc_warm_seed_view_v1 view{};
    view.struct_size = sizeof(view);
    view.abi_version = GENERATIVEQC_ABI_VERSION;
    check(generativeqc_warm_seed_snapshot_read_v1(candidate.get(), 0, &view));
    if (info.system_count != 1 || !view.present || !view.coordinates ||
        view.coordinate_count != positions.size() ||
        !std::equal(positions.begin(), positions.end(), view.coordinates))
      throw std::runtime_error("native SCF seed does not bind the completed derivative frame");
    frame.seed = std::move(candidate);
    frame.seed_bytes = info.numeric_bytes;
  }

  const RunOptions& options_;
  const SaddleInvocation& invocation_;
  const method_generated::Method& program_;
  Context context_;
  NativeBasisData basis_;
  std::vector<generativeqc_atom> atoms_;
  std::vector<double> masses_;
  std::optional<HostBudget> host_;
  std::shared_ptr<Frame> current_, accepted_;
  Clock::time_point started_{Clock::now()};
};

}  // namespace generativeqc::cli
