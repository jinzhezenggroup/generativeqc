#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "dft/nonlocal_correlation/vv10_integration.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "methods/dft_method.hpp"
#include "molecule/basis.hpp"

namespace {
using namespace generativeqc;

void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

core::System input_system(unsigned kind) {
  core::System system;
  if (kind == 2) {
    system.atoms = {{2, {0.0, 0.0, 0.0}}, {1, {0.1, 0.2, 1.7}}};
    system.shells = {{0, 0, {{1.8, 1.0}}}, {1, 0, {{0.6, 1.0}}}};
    system.charge = 1;
  } else {
    const unsigned count = kind == 1 ? 3 : 2;
    system.multiplicity = kind == 1 ? 2 : 1;
    for (unsigned i = 0; i < count; ++i) {
      system.atoms.push_back({1, {0.15 * i * i, 0.13 * i, 1.5 * i}});
      system.shells.push_back({i,
                               0,
                               {{3.425250914, 0.1543289673},
                                {0.6239137298, 0.5353281423},
                                {0.168855404, 0.4446345422}}});
    }
  }
  return system;
}

generativeqc_ks_options options(bool unrestricted) {
  static const std::array<generativeqc_ks_semilocal_component, 2> components{
      {{"MGGA_X_WB97M_V", 1.0}, {"MGGA_C_WB97M_V", 1.0}}};
  static const std::array<generativeqc_ks_exchange_term, 2> restricted_exchange{{
      {GENERATIVEQC_KS_EXCHANGE_SHORT_RANGE, 0.15, 0.3, -0.075},
      {GENERATIVEQC_KS_EXCHANGE_LONG_RANGE, 1.0, 0.3, -0.5},
  }};
  static const std::array<generativeqc_ks_exchange_term, 2> unrestricted_exchange{{
      {GENERATIVEQC_KS_EXCHANGE_SHORT_RANGE, 0.15, 0.3, -0.15},
      {GENERATIVEQC_KS_EXCHANGE_LONG_RANGE, 1.0, 0.3, -1.0},
  }};
  const auto& exchange = unrestricted ? unrestricted_exchange : restricted_exchange;
  generativeqc_ks_options ks{};
  ks.struct_size = sizeof(ks);
  ks.abi_version = GENERATIVEQC_ABI_VERSION;
  ks.scf_domain = "libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16";
  ks.grid_version = 1;
  ks.radial_points = 12;
  ks.angular_polar = 4;
  ks.angular_azimuth = 8;
  ks.partition_iterations = 3;
  ks.coincident_tolerance = 1e-12;
  ks.tile_points = 64;
  ks.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
  ks.spin_channels = unrestricted ? 2 : 1;
  ks.semilocal_components = components.data();
  ks.semilocal_component_count = components.size();
  ks.semilocal_range_omega = 0.3;
  ks.exchange_terms = exchange.data();
  ks.exchange_term_count = exchange.size();
  ks.has_nonlocal_correlation = 1;
  ks.nonlocal_variant = GENERATIVEQC_NONLOCAL_VV10;
  ks.nonlocal_b = 6.0;
  ks.nonlocal_c = 0.01;
  ks.nonlocal_coefficient = 1.0;
  ks.nonlocal_maximum_bytes = 1 << 24;
  return ks;
}

generativeqc_method_descriptor descriptor(const generativeqc_ks_options& ks) {
  generativeqc_method_descriptor method{};
  method.struct_size = sizeof(method);
  method.abi_version = GENERATIVEQC_ABI_VERSION;
  method.method = ks.spin_channels == 2 ? GENERATIVEQC_METHOD_PBE_UKS : GENERATIVEQC_METHOD_PBE_RKS;
  method.max_iterations = 180;
  method.diis_history = 8;
  method.energy_tolerance = 1e-12;
  method.density_tolerance = 1e-10;
  method.screening_tolerance = 1e-14;
  method.ks_options = &ks;
  return method;
}

const methods::Capabilities capabilities{GENERATIVEQC_METHOD_PBE_RKS,
                                         GENERATIVEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL,
                                         GENERATIVEQC_PROPERTY_ENERGY, false, true};

template <class Values>
void array(std::ostream& out, const Values& values) {
  out << '[';
  bool comma = false;
  for (const auto& value : values) {
    if (comma) out << ',';
    out << value;
    comma = true;
  }
  out << ']';
}

void dump(std::ostream& out, const std::string& name, const core::System& input,
          const dft::MolecularGrid& grid, bool unrestricted, const methods::Result& result,
          const dft::VerifiedKsFinalState& state) {
  out << std::setprecision(17) << "{\"name\":\"" << name << "\",\"atoms\":[";
  for (std::size_t i = 0; i < input.atoms.size(); ++i) {
    if (i) out << ',';
    const auto& a = input.atoms[i];
    out << '[' << a.atomic_number << ',';
    array(out, a.position);
    out << ']';
  }
  out << "],\"shells\":[";
  for (std::size_t i = 0; i < input.shells.size(); ++i) {
    if (i) out << ',';
    const auto& shell = input.shells[i];
    out << '[' << shell.atom_index << ',' << shell.angular_momentum << ",[";
    for (std::size_t j = 0; j < shell.primitives.size(); ++j) {
      if (j) out << ',';
      out << '[' << shell.primitives[j].exponent << ',' << shell.primitives[j].coefficient << ']';
    }
    out << "]]";
  }
  out << "],\"charge\":" << input.charge << ",\"spin\":" << input.multiplicity - 1
      << ",\"unrestricted\":" << (unrestricted ? "true" : "false")
      << ",\"energy\":" << result.energy << ",\"residual\":" << *result.physical_residual_rms
      << ",\"density\":[";
  for (std::size_t s = 0; s < state.density.size(); ++s) {
    if (s) out << ',';
    array(out, state.density[s]);
  }
  out << "],\"fock\":[";
  for (std::size_t s = 0; s < state.fock.size(); ++s) {
    if (s) out << ',';
    array(out, state.fock[s]);
  }
  out << "],\"points\":";
  array(out, grid.points());
  out << ",\"weights\":";
  array(out, grid.weights());
  out << "}\n";
}

void nonlocal_weighted_potential_execution() {
  using namespace dft::nlc;
  constexpr std::size_t points = 5;
  constexpr std::uint64_t dense_pairs = points * points;
  constexpr std::uint64_t workspace_bytes = 12 * points * sizeof(double);
  const std::vector<double> coordinates{0.0, 0.0, 0.0,  0.7, 0.2, -0.1, -0.3, 0.8,
                                        0.5, 1.1, -0.4, 0.9, 0.4, -0.2, 0.3};
  const std::vector<double> weights{0.3, -0.2, 0.0, -0.0, 0.7};
  const std::vector<double> density{0.42, 0.31, 0.18, 0.27, 0.36};
  const std::vector<double> gradient{0.05,  -0.02, 0.01,  -0.03, 0.04, 0.02, 0.01, 0.02,
                                     -0.04, -0.02, -0.01, 0.03,  0.02, 0.03, -0.01};
  struct Evaluation {
    generativeqc_status status{};
    double energy{};
    std::vector<double> vrho, vsigma, point, weight;
    std::uint64_t pairs{};
  };
  std::string detail;
  auto prepare = [&](Vv10Variant variant, std::uint32_t tile) {
    generativeqc_status status;
    auto plan = Vv10Plan::prepare(GENERATIVEQC_BACKEND_CPU_REFERENCE, -1, points, tile,
                                  {variant, 6.0, 0.01, 0.7}, workspace_bytes, detail, status);
    require(plan && status == GENERATIVEQC_STATUS_SUCCESS, detail);
    const auto& resources = plan->resources();
    require(resources.workspace_bytes == workspace_bytes &&
                resources.host_workspace_bytes == workspace_bytes &&
                resources.device_workspace_bytes == 0 &&
                resources.maximum_bytes == workspace_bytes &&
                resources.pair_evaluations == dense_pairs,
            "weighted VV10 changed the prepared workspace or dense work bound");
    require(plan->last_execution_pair_evaluations() == 0,
            "unexecuted VV10 plan reported pair work");
    return plan;
  };
  auto evaluate = [&](Vv10Plan& plan, const std::vector<double>& xyz, const std::vector<double>& w,
                      const std::vector<double>& rho, const std::vector<double>& grad,
                      bool weighted, bool geometry = false) {
    Evaluation result;
    result.vrho.resize(points);
    result.vsigma.resize(points);
    if (geometry) {
      result.point.resize(3 * points);
      result.weight.resize(points);
    }
    if (weighted)
      result.status = plan.execute(xyz, w, rho, grad, result.energy, result.vrho, result.vsigma,
                                   result.point, result.weight, detail, true);
    else
      result.status = plan.execute(xyz, w, rho, grad, result.energy, result.vrho, result.vsigma,
                                   result.point, result.weight, detail);
    result.pairs = plan.last_execution_pair_evaluations();
    require(plan.resources().workspace_bytes == workspace_bytes &&
                plan.resources().pair_evaluations == dense_pairs,
            "VV10 execution mutated its prepared resource admission");
    return result;
  };
  auto require_same = [&](const Evaluation& reference, const Evaluation& actual) {
    require(reference.status == GENERATIVEQC_STATUS_SUCCESS &&
                actual.status == GENERATIVEQC_STATUS_SUCCESS,
            detail);
    require(reference.energy == actual.energy && reference.vrho == actual.vrho &&
                reference.vsigma == actual.vsigma && reference.point == actual.point &&
                reference.weight == actual.weight,
            "VV10 fallback changed raw energy or derivative outputs");
  };

  // Five points includes both partial tiles and a tile larger than the grid.
  // Signed nonzero weights remain in both pair domains; only exact zeros leave.
  for (std::uint32_t tile : {1u, 2u, 3u, 8u}) {
    auto plan = prepare(Vv10Variant::vv10, tile);
    const auto raw = evaluate(*plan, coordinates, weights, density, gradient, false);
    const auto weighted = evaluate(*plan, coordinates, weights, density, gradient, true);
    require(
        raw.status == GENERATIVEQC_STATUS_SUCCESS && weighted.status == GENERATIVEQC_STATUS_SUCCESS,
        detail);
    require(raw.pairs == dense_pairs && weighted.pairs == 9,
            "VV10 weighted execution did not report the actual active pair count");
    require(raw.energy == weighted.energy,
            "VV10 zero-weight screening changed the ordered energy sum");
    for (std::size_t i = 0; i < points; ++i) {
      require(weights[i] * raw.vrho[i] == weights[i] * weighted.vrho[i] &&
                  weights[i] * raw.vsigma[i] == weights[i] * weighted.vsigma[i],
              "VV10 zero-weight screening changed a weighted potential");
      if (weights[i] == 0.0) {
        require(raw.vrho[i] != 0.0 && raw.vsigma[i] != 0.0,
                "raw VV10 zero-weight rows lost their feature derivatives");
        require(weighted.vrho[i] == 0.0 && weighted.vsigma[i] == 0.0,
                "weighted VV10 retained inactive feature outputs");
      } else
        require(raw.vrho[i] == weighted.vrho[i] && raw.vsigma[i] == weighted.vsigma[i],
                "VV10 screening changed an active row's ordered feature sums");
    }

    const std::vector<double> zero_weights{0.0, -0.0, 0.0, -0.0, 0.0};
    const auto vacuum = evaluate(*plan, coordinates, zero_weights, density, gradient, true);
    require(vacuum.status == GENERATIVEQC_STATUS_SUCCESS && vacuum.energy == 0.0 &&
                vacuum.pairs == 0 &&
                std::all_of(vacuum.vrho.begin(), vacuum.vrho.end(),
                            [](double x) { return x == 0.0; }) &&
                std::all_of(vacuum.vsigma.begin(), vacuum.vsigma.end(),
                            [](double x) { return x == 0.0; }),
            "all-zero VV10 weighted execution retained work or nonzero outputs");
    const auto raw_vacuum = evaluate(*plan, coordinates, zero_weights, density, gradient, false);
    require(raw_vacuum.status == GENERATIVEQC_STATUS_SUCCESS && raw_vacuum.energy == 0.0 &&
                raw_vacuum.pairs == dense_pairs &&
                std::all_of(raw_vacuum.vrho.begin(), raw_vacuum.vrho.end(),
                            [](double x) { return x != 0.0; }),
            "raw all-zero VV10 weights lost their local feature derivatives");

    // A nonzero signed weight may underflow when multiplied by density. It
    // still has raw feature derivatives and must not be classified as padding.
    auto tiny_weights = weights;
    tiny_weights[2] = std::numeric_limits<double>::denorm_min();
    tiny_weights[3] = -std::numeric_limits<double>::denorm_min();
    require(tiny_weights[2] != 0.0 && tiny_weights[2] * density[2] == 0.0 &&
                tiny_weights[3] != 0.0 && tiny_weights[3] * density[3] == 0.0,
            "VV10 signed-weight fixture did not exercise product underflow");
    const auto raw_tiny = evaluate(*plan, coordinates, tiny_weights, density, gradient, false);
    const auto weighted_tiny = evaluate(*plan, coordinates, tiny_weights, density, gradient, true);
    require_same(raw_tiny, weighted_tiny);
    require(weighted_tiny.pairs == dense_pairs,
            "VV10 classified an underflowed nonzero weight as inactive");

    // Geometry output includes dE/dweight even for zero weights, so the E/V-only
    // permission cannot be used to mask the geometry consumer's zero rows.
    const auto raw_geometry = evaluate(*plan, coordinates, weights, density, gradient, false, true);
    const auto weighted_geometry =
        evaluate(*plan, coordinates, weights, density, gradient, true, true);
    require_same(raw_geometry, weighted_geometry);
    require(weighted_geometry.pairs == dense_pairs && weighted_geometry.weight[2] != 0.0 &&
                weighted_geometry.weight[3] != 0.0,
            "weighted-potential permission masked generic geometry derivatives");

    // The opt-in belongs to the weighted feature consumer. Empty feature spans
    // must retain the energy-only domain and its original dense execution.
    double raw_energy_only = 0.0, weighted_energy_only = 0.0;
    require(plan->execute(coordinates, weights, density, gradient, raw_energy_only, {}, {}, {}, {},
                          detail) == GENERATIVEQC_STATUS_SUCCESS &&
                plan->last_execution_pair_evaluations() == dense_pairs,
            "raw energy-only VV10 did not retain its dense execution");
    require(plan->execute(coordinates, weights, density, gradient, weighted_energy_only, {}, {}, {},
                          {}, detail, true) == GENERATIVEQC_STATUS_SUCCESS &&
                plan->last_execution_pair_evaluations() == dense_pairs &&
                weighted_energy_only == raw_energy_only,
            "weighted-potential permission changed energy-only VV10 execution");

    auto distant = coordinates;
    distant[3 * 2] = 1e7;
    const auto raw_distant = evaluate(*plan, distant, weights, density, gradient, false);
    const auto weighted_distant = evaluate(*plan, distant, weights, density, gradient, true);
    require_same(raw_distant, weighted_distant);
    require(weighted_distant.pairs == dense_pairs,
            "out-of-envelope VV10 did not retain its dense fallback");

    // Exercise a scale/weight envelope guard independently of coordinates.
    auto large_weights = weights;
    large_weights[0] = 1e13;
    require(std::isfinite(large_weights[0] * density[0]) && large_weights[0] * density[0] > 1e12,
            "VV10 large-weight fixture did not exceed the masking envelope");
    const auto raw_large = evaluate(*plan, coordinates, large_weights, density, gradient, false);
    const auto weighted_large =
        evaluate(*plan, coordinates, large_weights, density, gradient, true);
    require_same(raw_large, weighted_large);
    require(raw_large.pairs == dense_pairs && weighted_large.pairs == dense_pairs,
            "out-of-envelope weighted density did not retain dense VV10 execution");

    // Zero quadrature weight cannot exempt a point from local-scale validation.
    // This finite gradient overflows sigma before the pair loop is reached.
    auto bad_local_gradient = gradient;
    bad_local_gradient[3 * 2] = 1e200;
    const auto raw_local_failure =
        evaluate(*plan, coordinates, weights, density, bad_local_gradient, false);
    require(evaluate(*plan, coordinates, weights, density, gradient, true).pairs == 9,
            "VV10 did not restore successful work before local-scale failure replay");
    const auto weighted_local_failure =
        evaluate(*plan, coordinates, weights, density, bad_local_gradient, true);
    require(raw_local_failure.status == GENERATIVEQC_STATUS_NUMERICAL_FAILURE &&
                weighted_local_failure.status == raw_local_failure.status &&
                raw_local_failure.pairs == 0 && weighted_local_failure.pairs == 0 &&
                detail.find("local scales") != std::string::npos,
            "zero-weight VV10 screening concealed a local-scale numerical failure");

    // The unused row must not conceal a numerical failure in the raw pair
    // domain: its finite coordinate overflows r^2 against an active point.
    distant[3 * 2] = 1e155;
    const auto raw_overflow = evaluate(*plan, distant, weights, density, gradient, false);
    const auto weighted_overflow = evaluate(*plan, distant, weights, density, gradient, true);
    require(raw_overflow.status == GENERATIVEQC_STATUS_NUMERICAL_FAILURE &&
                weighted_overflow.status == raw_overflow.status && raw_overflow.pairs == 0 &&
                weighted_overflow.pairs == 0,
            "zero-weight VV10 screening concealed pair overflow or published failed work");

    for (unsigned invalid = 0; invalid < 3; ++invalid) {
      require(evaluate(*plan, coordinates, weights, density, gradient, true).pairs == 9,
              "VV10 did not restore successful work before invalid-input replay");
      auto bad_xyz = coordinates, bad_density = density, bad_gradient = gradient;
      const auto nan = std::numeric_limits<double>::quiet_NaN();
      if (invalid == 0) bad_xyz[3 * 2] = nan;
      if (invalid == 1) bad_density[2] = nan;
      if (invalid == 2) bad_gradient[3 * 2] = nan;
      const auto bad = evaluate(*plan, bad_xyz, weights, bad_density, bad_gradient, true);
      require(bad.status == GENERATIVEQC_STATUS_INVALID_ARGUMENT && bad.pairs == 0,
              "VV10 zero-weight screening concealed a nonfinite input");
    }
    const auto recovered = evaluate(*plan, coordinates, weights, density, gradient, true);
    require_same(weighted, recovered);
    require(recovered.pairs == 9, "VV10 did not recover its pair count after a failed execution");
  }

  auto rvv10 = prepare(Vv10Variant::rvv10, 3);
  const auto raw_rvv10 = evaluate(*rvv10, coordinates, weights, density, gradient, false);
  const auto weighted_rvv10 = evaluate(*rvv10, coordinates, weights, density, gradient, true);
  require_same(raw_rvv10, weighted_rvv10);
  require(weighted_rvv10.pairs == dense_pairs,
          "VV10 weighted-potential permission changed the rVV10 execution domain");
}

void nonlocal_zero_weight_ao_overflow() {
  using namespace dft::nlc;
  // Finite normalized primitives do not bound the magnitude of an AO jet.
  // This nearly cancelling tight contraction has a tiny positive computed
  // norm, while its diffuse primitive keeps normalization admissible. Its
  // self-overlap and squared coefficient are normal, so admission does not
  // rely on retaining a subnormal normalization value.
  // Search a bounded set of adjacent exponents instead of relying on one
  // particular libm/compiler rounding of the cancellation and normalization.
  constexpr std::array<double, 3> center{1.0, 1.0, 1.0};
  double nearby = 1e150;
  core::System system;
  std::string detail;
  bool fixture_found = false;
  for (unsigned step = 0; step < 64; ++step) {
    nearby = std::nextafter(nearby, std::numeric_limits<double>::infinity());
    core::System candidate;
    candidate.atoms = {{2, center}};
    candidate.shells = {{0, 0, {{1e150, 1.0}, {nearby, -1.0}, {1e-100, 1e-120}}}};
    if (molecule::validate_and_normalize(candidate, detail) != GENERATIVEQC_STATUS_SUCCESS)
      continue;
    const dft::AoBasis candidate_basis(candidate);
    std::array<double, 4> center_ao{};
    candidate_basis.evaluate(center.data(), 1, 1, 0, 1, center_ao.data(), center_ao.size());
    if (!std::isfinite(center_ao[0]) || std::abs(center_ao[0]) <= 1e200 ||
        !std::all_of(center_ao.begin() + 1, center_ao.end(),
                     [](double value) { return value == 0.0; }))
      continue;
    system = candidate;
    fixture_found = true;
    break;
  }
  require(fixture_found,
          "no admitted finite large-AO overflow fixture found within 64 exponent ULPs");
  const dft::AoBasis basis(system);
  dft::GridSpec spec{1, 1, 1, 3, 3, 1e-12};
  spec.element_radii[2] = 1e-200;
  const dft::MolecularGrid grid(system, spec);
  const auto points = grid.point_count();
  require(basis.nao == 1 && points == 3 &&
              std::all_of(grid.weights().begin(), grid.weights().end(),
                          [](double weight) { return weight == 0.0; }) &&
              std::all_of(grid.points().begin(), grid.points().end(),
                          [](double coordinate) { return coordinate == 1.0; }),
          "zero-weight AO overflow fixture did not reach its coincident zero-weight grid");
  std::vector<double> ao(4 * points);
  basis.evaluate(grid.points().data(), points, 1, 0, 1, ao.data(), ao.size());
  require(
      std::all_of(ao.begin(), ao.begin() + points,
                  [](double value) { return std::isfinite(value) && std::abs(value) > 1e200; }) &&
          std::all_of(ao.begin() + points, ao.end(), [](double value) { return value == 0.0; }),
      "zero-weight AO overflow fixture lacks large finite values and zero first jets");

  generativeqc_status status;
  auto plan =
      Vv10Plan::prepare(GENERATIVEQC_BACKEND_CPU_REFERENCE, -1, static_cast<std::uint32_t>(points),
                        2, {Vv10Variant::vv10, 6.0, 0.01, 1.0}, 1 << 20, detail, status);
  require(plan && status == GENERATIVEQC_STATUS_SUCCESS, detail);
  const std::vector<double> padded_density(points, 1.0), padded_gradient(3 * points);
  std::vector<double> vrho(points), vsigma(points);
  double energy = 0.0;
  require(plan->execute(grid.points(), grid.weights(), padded_density, padded_gradient, energy,
                        vrho, vsigma, {}, {}, detail) == GENERATIVEQC_STATUS_SUCCESS &&
              !std::isfinite(vrho[0] * ao[0] * ao[0]),
          "zero-weight AO overflow fixture did not isolate finite VV10 fields with AO overflow");
  bool rejected = false;
  try {
    (void)integrate_vv10_rks(basis, grid, {0.0}, *plan, 2, {}, Vv10DensityDomain::MolecularV1);
  } catch (const std::runtime_error& error) {
    rejected = std::string(error.what()) == "nonfinite self-consistent VV10 AO contribution";
  }
  require(rejected && plan->last_execution_pair_evaluations() == points * points,
          "zero-weight VV10 row masking concealed downstream AO assembly overflow");
}

void nonlocal_density_domain() {
  using namespace dft::nlc;
  auto system = input_system(2);
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  const dft::AoBasis basis(system);
  const dft::MolecularGrid grid(system, {1, 12, 4, 8, 3, 1e-12});
  const std::size_t n = basis.nao, points = grid.point_count();
  generativeqc_status status;
  const Vv10Parameters parameters{Vv10Variant::vv10, 6.0, 0.01, 1.0};
  auto plan =
      Vv10Plan::prepare(GENERATIVEQC_BACKEND_CPU_REFERENCE, -1, static_cast<std::uint32_t>(points),
                        64, parameters, 1 << 24, detail, status);
  require(plan && status == GENERATIVEQC_STATUS_SUCCESS, detail);
  const std::vector<double> density{1.2, 0.05, 0.05, 0.5};
  const auto screened =
      integrate_vv10_rks(basis, grid, density, *plan, 64, {}, Vv10DensityDomain::MolecularV1);
  // Independent active-set compaction verifies that zero-weight padding removes
  // both pair domains, not just the outer energy quadrature.
  std::vector<double> ao(4 * points * n);
  basis.evaluate(grid.points().data(), points, 1, 0, n, ao.data(), ao.size());
  std::vector<double> xyz, weights, rho, gradients;
  std::vector<std::size_t> active;
  for (std::size_t p = 0; p < points; ++p) {
    const auto* phi = ao.data() + p * n;
    double r = 0.0, g[3]{};
    for (std::size_t mu = 0; mu < n; ++mu) {
      double weighted = 0.0;
      for (std::size_t nu = 0; nu < n; ++nu) weighted += density[mu * n + nu] * phi[nu];
      r += phi[mu] * weighted;
      for (unsigned k = 0; k < 3; ++k) g[k] += 2.0 * ao[((k + 1) * points + p) * n + mu] * weighted;
    }
    if (r >= 1e-8) {
      active.push_back(p);
      xyz.insert(xyz.end(), grid.points().begin() + 3 * p, grid.points().begin() + 3 * p + 3);
      weights.push_back(grid.weights()[p]);
      rho.push_back(r);
      gradients.insert(gradients.end(), g, g + 3);
    }
  }
  require(!active.empty() && active.size() < points,
          "molecular VV10 fixture lacks an active/tail split");
  const auto active_weights = static_cast<std::uint64_t>(
      std::count_if(weights.begin(), weights.end(), [](double weight) { return weight != 0.0; }));
  require(plan->last_execution_pair_evaluations() == active_weights * active_weights,
          "molecular VV10 integration retained padded pair evaluations");
  auto compact = Vv10Plan::prepare(GENERATIVEQC_BACKEND_CPU_REFERENCE, -1,
                                   static_cast<std::uint32_t>(active.size()), 64, parameters,
                                   1 << 24, detail, status);
  require(compact && status == GENERATIVEQC_STATUS_SUCCESS, detail);
  std::vector<double> vrho(active.size()), vsigma(active.size()), potential(n * n);
  double energy = 0.0;
  require(compact->execute(xyz, weights, rho, gradients, energy, vrho, vsigma, {}, {}, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail);
  for (std::size_t a = 0; a < active.size(); ++a) {
    const auto p = active[a];
    const auto* phi = ao.data() + p * n;
    for (std::size_t mu = 0; mu < n; ++mu)
      for (std::size_t nu = 0; nu < n; ++nu) {
        double value = vrho[a] * phi[mu] * phi[nu];
        for (unsigned k = 0; k < 3; ++k) {
          const auto* jet = ao.data() + ((k + 1) * points + p) * n;
          value += 2.0 * vsigma[a] * gradients[3 * a + k] * (jet[mu] * phi[nu] + phi[mu] * jet[nu]);
        }
        potential[mu * n + nu] += weights[a] * value;
      }
  }
  require(std::abs(energy - screened.energy) < 1e-13,
          "VV10 padding differs from compact active-set energy");
  for (std::size_t i = 0; i < potential.size(); ++i)
    require(std::abs(potential[i] - screened.potential[i]) < 1e-13,
            "VV10 padding leaked inactive points into the AO potential");
  const auto vacuum = integrate_vv10_rks(basis, grid, std::vector<double>(n * n), *plan, 64, {},
                                         Vv10DensityDomain::MolecularV1);
  require(vacuum.energy == 0.0 && std::all_of(vacuum.potential.begin(), vacuum.potential.end(),
                                              [](double x) { return x == 0.0; }),
          "screened VV10 vacuum is not zero");
  require(plan->last_execution_pair_evaluations() == 0,
          "screened molecular VV10 vacuum retained pair evaluations");
  for (unsigned invalid = 0; invalid < 3; ++invalid) {
    auto bad = density;
    if (invalid == 0)
      for (auto& x : bad) x = -x;
    if (invalid == 1) bad[0] = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try {
      (void)integrate_vv10_rks(
          basis, grid, bad, *plan, 64, {},
          invalid == 2 ? static_cast<Vv10DensityDomain>(99) : Vv10DensityDomain::MolecularV1);
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected, "VV10 molecular screening concealed invalid inputs/domain");
  }
  const std::vector<double> zero_rho(points), zero_gradient(3 * points);
  require(plan->execute(grid.points(), grid.weights(), zero_rho, zero_gradient, energy, {}, {}, {},
                        {}, detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "molecular screening weakened the strict raw-pair density contract");
}

void run_case(unsigned kind, bool unrestricted, std::ostream* output) {
  const auto input = input_system(kind);
  auto system = input;
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS, detail);
  auto ks = options(unrestricted);
  auto method = descriptor(ks);
  core::ContextState context;
  context.device_id = -1;
  auto plan = methods::detail::prepare_dft_calculation(capabilities, context, system, method);
  dft::CudaKsFinalStateToken token;
  require(
      methods::detail::dft_final_state_token(*plan, token, detail) != GENERATIVEQC_STATUS_SUCCESS,
      "unexecuted WB97M-V owner published a state");
  auto result = plan->execute(false);
  require(result.convergence.converged && result.ks_diagnostic && result.physical_residual_rms &&
              *result.physical_residual_rms < 1e-9,
          "WB97M-V native endpoint did not converge");
  // Independently reconverged PySCF 2.14.0 / Libxc 7.0.0 on the exact
  // basis/grid above; reproduce with tools/verify_wb97mv_scf.py and this test's dump.
  const double oracle = kind == 0   ? -1.1419425514308112
                        : kind == 1 ? -1.5764935673604195
                                    : -2.3282082665978300;
  require(std::abs(result.energy - oracle) < 2e-10,
          "complete WB97M-V SCF energy differs from its independent matched-grid oracle");
  require(result.ks_diagnostic->scf_domain_version == 3 && result.ks_diagnostic->ao_order == 1,
          "WB97M-V lost its production point-domain or AO-jet identity");
  require(std::abs(result.ks_diagnostic->components.total() - result.energy) < 1e-12,
          "WB97M-V energy components do not describe its final density");
  require(
      methods::detail::dft_final_state_token(*plan, token, detail) == GENERATIVEQC_STATUS_SUCCESS,
      detail);
  dft::VerifiedKsFinalState state;
  auto status = methods::detail::read_dft_final_state(*plan, token, false, state, detail);
  require(status == GENERATIVEQC_STATUS_SUCCESS, detail);
  require(state.identity.model.range_correction && state.identity.model.nonlocal_correlation,
          "WB97M-V final-state identity omitted the LR-K/VV10 owners");
  require(state.identity.model.nonlocal_density_domain == dft::nlc::Vv10DensityDomain::MolecularV1,
          "WB97M-V final state lost its explicit nonlocal density domain");
  auto forged = token;
  forged.identity.model.nonlocal_correlation->b += 0.1;
  dft::VerifiedKsFinalState rejected;
  require(methods::detail::read_dft_final_state(*plan, forged, false, rejected, detail) !=
              GENERATIVEQC_STATUS_SUCCESS,
          "changed VV10 identity authorized an old state");

  forged = token;
  forged.identity.model.nonlocal_density_domain = dft::nlc::Vv10DensityDomain::StrictPositive;
  require(methods::detail::read_dft_final_state(*plan, forged, false, rejected, detail) !=
              GENERATIVEQC_STATUS_SUCCESS,
          "changed VV10 density policy authorized an old state");

  const std::string name = std::string(kind == 0   ? "h2"
                                       : kind == 1 ? "h3"
                                                   : "heh") +
                           (unrestricted ? "-uks" : "-rks");
  if (output) {
    dft::MolecularGrid grid(system, {1, ks.radial_points, ks.angular_polar, ks.angular_azimuth,
                                     ks.partition_iterations, ks.coincident_tolerance});
    dump(*output, name, input, grid, unrestricted, result, state);
  }
  // Pointees have been snapshotted; replay must not observe caller mutation.
  ks.exchange_terms = nullptr;
  ks.exchange_term_count = 0;
  ks.nonlocal_b = 9.0;
  const auto warm = plan->execute(false);
  require(warm.convergence.converged && warm.ks_diagnostic->initial_density_used &&
              std::abs(warm.energy - result.energy) < 1e-9,
          "WB97M-V warm replay changed the immutable model/energy");
  require(methods::detail::read_dft_final_state(*plan, token, false, rejected, detail) !=
              GENERATIVEQC_STATUS_SUCCESS,
          "WB97M-V replay did not revoke the previous token");
  bool force_rejected = false;
  try {
    (void)plan->execute(true);
  } catch (const methods::MethodError& error) {
    force_rejected = error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  require(force_rejected, "energy-only WB97M-V implementation accepted analytic forces");
  std::cout << name << " E=" << std::setprecision(16) << result.energy
            << " residual=" << *result.physical_residual_rms << '\n';

  // Invalid or incomplete compositions must never execute a partial method.
  for (unsigned failure = 0; failure < 9; ++failure) {
    ks = options(unrestricted);
    std::array<generativeqc_ks_exchange_term, 2> exchange{ks.exchange_terms[0],
                                                          ks.exchange_terms[1]};
    std::array<generativeqc_ks_semilocal_component, 2> components{ks.semilocal_components[0],
                                                                  ks.semilocal_components[1]};
    ks.exchange_terms = exchange.data();
    ks.semilocal_components = components.data();
    switch (failure) {
      case 0:
        ks.exchange_terms = nullptr;
        ks.exchange_term_count = 0;
        break;
      case 1:
        ks.has_nonlocal_correlation = 0;
        break;
      case 2:
        exchange[0].omega = 0.4;
        break;
      case 3:
        ks.nonlocal_b = 5.9;
        break;
      case 4:
        ks.nonlocal_variant = GENERATIVEQC_NONLOCAL_RVV10;
        break;
      case 5:
        exchange[1].coefficient = 0.8;
        break;
      case 6:
        components[0].coefficient = 0.9;
        break;
      case 7:
        ks.scf_domain = "semilocal-scaled-v1/pbe-spin-c2-1e-18";
        break;
      case 8:
        ks.nonlocal_maximum_bytes = 1;
        break;
    }
    bool refused = false;
    try {
      (void)methods::detail::prepare_dft_calculation(capabilities, context, system, method);
    } catch (const methods::MethodError&) {
      refused = true;
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    require(refused, "incomplete/mutated WB97M-V composition was accepted");
  }
  if (kind == 2 && !unrestricted) {
    ks = options(false);
    method.max_iterations = 1;
    auto failed = methods::detail::prepare_dft_calculation(capabilities, context, system, method);
    require(!failed->execute(false).convergence.converged,
            "single-iteration WB97M-V falsely converged");
    require(methods::detail::dft_final_state_token(*failed, token, detail) !=
                GENERATIVEQC_STATUS_SUCCESS,
            "unconverged WB97M-V solve published a successful state");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::ofstream output;
    if (argc == 2) {
      output.open(argv[1]);
      require(static_cast<bool>(output), "unable to write WB97M-V independent-oracle input");
    } else
      require(argc == 1, "usage: generativeqc_wb97mv_scf_tests [oracle-input.jsonl]");
    int32_t available = 1;
    require(generativeqc_method_available(GENERATIVEQC_METHOD_PBE_RKS, &available) ==
                    GENERATIVEQC_STATUS_SUCCESS &&
                available,
            "generic RKS DFT carrier is missing its energy admission");
    nonlocal_weighted_potential_execution();
    nonlocal_zero_weight_ao_overflow();
    nonlocal_density_domain();
    run_case(0, false, output.is_open() ? &output : nullptr);
    run_case(0, true, output.is_open() ? &output : nullptr);
    run_case(1, true, output.is_open() ? &output : nullptr);
    run_case(2, false, output.is_open() ? &output : nullptr);
    run_case(2, true, output.is_open() ? &output : nullptr);
    std::cout << "WB97M-V self-consistent composition and owner gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
