// Exercise capacity authority through actual prepared owners and lazy ports.
// The H2 checks use an independent PySCF reference; larger fixtures exercise
// ownership/lifetime rather than claiming broad scientific qualification.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <generativeqc/generativeqc.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "api/handles.hpp"
#include "methods/dft_method.hpp"
#include "methods/named_ks_catalog.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda_one_electron_gradient.hpp"
#include "scf/ks_device_resources.hpp"

namespace {
namespace qc = generativeqc;
namespace detail = qc::methods::detail;
using Resources = std::unique_ptr<generativeqc_device_resources,
                                  decltype(&generativeqc_device_resources_destroy)>;
using Calculation =
    std::unique_ptr<generativeqc_calculation, decltype(&generativeqc_calculation_destroy)>;
using Batch = std::unique_ptr<generativeqc_batch, decltype(&generativeqc_batch_destroy)>;
constexpr std::size_t cap = 2ULL << 30;
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void check(generativeqc_status status, const std::string& detail) {
  require(status == GENERATIVEQC_STATUS_SUCCESS, detail + ": status=" + std::to_string(status));
}
Resources resources(std::uint64_t bytes) {
  const generativeqc_device_resource_buffer_v1 row{bytes, 4, 0, 0, 0, 0};
  generativeqc_device_resource_descriptor desc{};
  desc.struct_size = sizeof(desc);
  desc.abi_version = GENERATIVEQC_ABI_VERSION;
  desc.buffers = &row;
  desc.buffer_count = 1;
  desc.admission = GENERATIVEQC_DEVICE_RESOURCE_NUMERIC_CAP;
  generativeqc_device_resources* result{};
  qc::check(generativeqc_device_resources_create(&desc, &result));
  return Resources(result, generativeqc_device_resources_destroy);
}
auto snapshot(const Resources& resources, const std::string& phase) {
  generativeqc_device_resource_diagnostic d{};
  d.struct_size = sizeof(d);
  d.abi_version = GENERATIVEQC_ABI_VERSION;
  qc::check(generativeqc_device_resources_read(resources.get(), &d));
  require(d.numeric_live_bytes + d.provider_reserved_bytes + d.capacity_reserved_bytes <=
              d.numeric_limit,
          phase + " exceeds declared cap");
  std::cout << "{\"phase\":\"" << phase << "\",\"live\":" << d.numeric_live_bytes
            << ",\"provider\":" << d.provider_reserved_bytes
            << ",\"promise\":" << d.capacity_reserved_bytes << ",\"peak\":" << d.combined_peak_bytes
            << ",\"rejected\":" << d.rejected_allocations << "}\n"
            << std::flush;
  return d;
}
void empty(const Resources& r) {
  const auto d = snapshot(r, "closed");
  require(!d.numeric_live_bytes && !d.provider_reserved_bytes && !d.capacity_reserved_bytes,
          "teardown retained charges");
}

/** D4's complete promise includes its retained correction owner. D3 still uses
 * ordinary capped allocation until its complete lifetime is qualified. */
void test_named_correction_admission() {
  generativeqc_context_descriptor descriptor{sizeof(descriptor), GENERATIVEQC_ABI_VERSION, 0,
                                             GENERATIVEQC_BACKEND_CUDA};
  qc::Context context(descriptor);
  const std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, -.7}, {1, 0, 0, .7}}};
  qc::System system(context, atoms, "sto-3g");
  auto method = qc::default_method_descriptor("pbe-rks");
  method.energy_tolerance = 1e-12;
  method.density_tolerance = 1e-10;
  method.max_iterations = 250;
  for (const auto* selector : {"pbe-d3(bj)-rks", "pbe-d3(0)-rks", "pbe-d4(bj-eeq-atm)-rks",
                               "pbe-d3(bj)-uks", "pbe-d3(0)-uks", "pbe-d4(bj-eeq-atm)-uks"}) {
    const auto* entry = qc::methods::named_ks::find_method(selector);
    require(entry != nullptr, "named correction fixture is not registered");
    const bool promised = std::string(selector).find("-d4(") != std::string::npos;
    method.method = entry->carrier;
    const std::string identity(entry->identity);
    const generativeqc_named_ks_descriptor named{sizeof(named), GENERATIVEQC_ABI_VERSION, selector,
                                                 identity.c_str(), 256ULL << 20};
    auto r = resources(cap);
    {
      generativeqc_calculation* raw{};
      check(generativeqc_calculation_prepare_named_ks_with_device_resources_v1(
                context.get(), system.get(), &method, &named, r.get(), &raw),
            context.get()->last_detail);
      Calculation calculation(raw, generativeqc_calculation_destroy);
      require((snapshot(r, std::string(selector) + "-prepared").capacity_reserved_bytes > 0) ==
                  promised,
              "named correction capacity ownership differs from its complete bound");
      const auto result = calculation->plan->execute(true);
      require(result.convergence.converged && std::isfinite(result.energy),
              "bounded named correction did not converge");
      require((snapshot(r, std::string(selector) + "-executed").capacity_reserved_bytes > 0) ==
                  promised,
              "named correction lost its retained future-owner promise");
      generativeqc_calculation* plain{};
      check(generativeqc_calculation_prepare_named_ks_v1(context.get(), system.get(), &method,
                                                         &named, &plain),
            context.get()->last_detail);
      Calculation reference_owner(plain, generativeqc_calculation_destroy);
      const auto reference = reference_owner->plan->execute(true);
      require(std::abs(result.energy - reference.energy) < 1e-9 && result.forces.size() == 6 &&
                  result.forces.size() == reference.forces.size(),
              "capped correction changed energy or force publication");
      for (std::size_t i = 0; i < result.forces.size(); ++i)
        require(std::abs(result.forces[i] - reference.forces[i]) < 1e-8,
                "capped correction changed forces");
    }
    empty(r);
    {
      const std::array<const generativeqc_system*, 2> systems{system.get(), system.get()};
      generativeqc_batch* raw{};
      check(generativeqc_batch_prepare_named_ks_with_device_resources_v1(
                context.get(), systems.data(), systems.size(), &method, &named,
                GENERATIVEQC_BATCH_ENABLE_WARM_STARTS, r.get(), &raw),
            context.get()->last_detail);
      Batch batch(raw, generativeqc_batch_destroy);
      require(
          (snapshot(r, std::string(selector) + "-fleet").capacity_reserved_bytes > 0) == promised,
          "named fleet capacity ownership differs from its complete bound");
      for (const qc::methods::Coordinates& moved :
           {qc::methods::Coordinates{}, qc::methods::Coordinates{{{0, 0, -.7, .03, 0, .78}}, {}}}) {
        const auto values = batch->plan->execute(moved, true);
        require(values.size() == systems.size(), "named fleet omitted an item");
        for (const auto& item : values)
          require(item.status == GENERATIVEQC_STATUS_SUCCESS &&
                      item.calculation.convergence.converged &&
                      std::isfinite(item.calculation.energy),
                  "named correction fleet failed bounded replay");
      }
    }
    empty(r);
    auto insufficient = resources(1);
    generativeqc_calculation* rejected{};
    require(generativeqc_calculation_prepare_named_ks_with_device_resources_v1(
                context.get(), system.get(), &method, &named, insufficient.get(), &rejected) ==
                    GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
                !rejected,
            "named correction bypassed a below-capacity refusal");
    empty(insufficient);
  }
}
struct Model {
  std::array<generativeqc_ks_semilocal_component, 2> terms{{{"GGA_C_PBE", 1}, {"GGA_X_PBE", .75}}};
  generativeqc_ks_exchange_term exchange{GENERATIVEQC_KS_EXCHANGE_FULL_RANGE, .25, 0, -.125};
  generativeqc_ks_options options{};
  generativeqc_method_descriptor method{};
  Model() {
    options.struct_size = sizeof(options);
    options.abi_version = GENERATIVEQC_ABI_VERSION;
    options.scf_domain = "semilocal-scaled-v1/pbe-spin-c2-1e-18";
    options.grid_version = 1;
    options.radial_points = 64;
    options.angular_polar = 12;
    options.angular_azimuth = 24;
    options.partition_iterations = 3;
    options.coincident_tolerance = 1e-12;
    options.tile_points = 256;
    options.xc_execution_schedule = GENERATIVEQC_XC_EXECUTION_DEVICE_FUSED;
    options.spin_channels = 1;
    options.semilocal_components = terms.data();
    options.semilocal_component_count = terms.size();
    options.exchange_terms = &exchange;
    options.exchange_term_count = 1;
    method.struct_size = sizeof(method);
    method.abi_version = GENERATIVEQC_ABI_VERSION;
    method.method = GENERATIVEQC_METHOD_PBE_RKS;
    method.max_iterations = 250;
    method.diis_history = 8;
    method.energy_tolerance = 1e-12;
    method.density_tolerance = 1e-10;
    method.screening_tolerance = 1e-14;
    method.precision_mode = GENERATIVEQC_PRECISION_FP64;
    method.ks_options = &options;
  }
};
Calculation prepare(qc::Context& context, const qc::System& system, const Model& model,
                    const Resources& r) {
  generativeqc_calculation* output{};
  const auto status = generativeqc_calculation_prepare_with_device_resources(
      context.get(), system.get(), &model.method, r.get(), &output);
  check(status, context.get()->last_detail);
  return Calculation(output, generativeqc_calculation_destroy);
}
// Deliberately call independent owner ports with no outer ledger scope. The
// prepared owner must bind its own promise after the public prepare returns.
void single_ports(Calculation& c) {
  std::string message;
  qc::dft::CudaKsFinalStateToken token;
  check(detail::dft_final_state_token(*c->plan, token, message), message);
  qc::dft::VerifiedKsFinalState state;
  check(detail::read_dft_final_state(*c->plan, token, true, state, message), message);
  std::vector<double> sources;
  std::array<std::uint64_t, 9> work{};
  check(detail::dft_prepared_integral_gradient_cached(
            *c->plan, token, state.density, state.weighted_density, sources, cap, work, message),
        message);
  require(
      sources.size() == 12 * c->plan->atom_count() &&
          std::all_of(sources.begin(), sources.end(), [](double v) { return std::isfinite(v); }),
      "single late integral sources missing");
  const auto warm = c->plan->execute(false);
  require(std::isfinite(warm.energy), "single warm replay failed");
  require(detail::read_dft_final_state(*c->plan, token, true, state, message) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "single stale token accepted");
}
void batch_ports(Batch& batch, std::size_t index) {
  std::string message;
  qc::dft::CudaKsFinalStateToken token;
  check(detail::dft_final_state_token(*batch->plan, index, token, message), message);
  qc::dft::CudaKsResidentOrbitalFrameBinding frame;
  check(detail::dft_cuda_resident_orbital_frame(*batch->plan, index, token, frame, message, true),
        message);
  qc::dft::CudaKsResidentGridBinding grid;
  check(detail::dft_cuda_resident_grid_reference(*batch->plan, index, token, grid, message),
        message);
  qc::dft::CudaKsResidentFockBinding fock;
  check(detail::dft_cuda_resident_fock_sources(*batch->plan, index, token, fock, message), message);
  qc::methods::detail::KsDerivativeSnapshot state;
  check(detail::read_dft_derivative_state(*batch->plan, index, token, state, message), message);
  check(detail::read_dft_derivative_sources(*batch->plan, index, token, state, message), message);
  std::vector<double> gradient;
  std::array<std::uint64_t, 9> work{};
  check(
      detail::dft_cuda_integral_gradient(*batch->plan, index, token, gradient, cap, work, message),
      message);
  require(!gradient.empty() && std::all_of(gradient.begin(), gradient.end(),
                                           [](double v) { return std::isfinite(v); }),
          "batch late gradient missing");
}
void test_shared_lifetimes() {
  generativeqc_context_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.backend = GENERATIVEQC_BACKEND_CUDA;
  qc::Context context(descriptor);
  Model model;
  const std::array<generativeqc_atom, 2> h2{{{1, 0, 0, -.7}, {1, 0, 0, .7}}};
  std::vector<generativeqc_atom> waters;
  for (unsigned i = 0; i < 3; ++i) {
    const double x = 8.0 * i;
    waters.push_back({8, x, 0, 0});
    waters.push_back({1, x, 1.43, 1.1});
    waters.push_back({1, x, -1.43, 1.1});
  }
  qc::System small(context, h2, "sto-3g"), large(context, waters, "sto-3g");
  auto r = resources(cap);
  std::uint64_t small_bound{};
  {
    auto small_owner = prepare(context, small, model, r);
    const auto first = snapshot(r, "small-prepared");
    require(first.capacity_reserved_bytes, "small owner missing promise");
    small_bound =
        first.numeric_live_bytes + first.provider_reserved_bytes + first.capacity_reserved_bytes;
    auto large_owner = prepare(context, large, model, r);
    const auto both = snapshot(r, "two-owners-prepared");
    require(both.provider_reserved_bytes > 0,
            "21-AO system did not convert matrix-provider allowance");
    require(std::isfinite(large_owner->plan->execute(false).energy), "large energy failed");
    single_ports(large_owner);
    snapshot(r, "large-late-ports");
    require(std::isfinite(small_owner->plan->execute(false).energy), "small energy failed");
    single_ports(small_owner);
    snapshot(r, "small-late-ports");
    large_owner.reset();
    const auto remaining = snapshot(r, "large-closed");
    require(remaining.numeric_live_bytes + remaining.provider_reserved_bytes +
                    remaining.capacity_reserved_bytes ==
                small_bound,
            "closing large owner altered small owner's reserved bound");
  }
  empty(r);
  // Two complete promises fit exactly. A third owner must fail without
  // consuming either survivor's future credit; closing one enables retry.
  auto tight = resources(2 * small_bound);
  {
    auto a = prepare(context, small, model, tight);
    auto b = prepare(context, small, model, tight);
    const auto before = snapshot(tight, "exact-shared-cap");
    generativeqc_calculation* rejected{};
    const auto status = generativeqc_calculation_prepare_with_device_resources(
        context.get(), small.get(), &model.method, tight.get(), &rejected);
    require(status == GENERATIVEQC_STATUS_OUT_OF_MEMORY && !rejected,
            "third owner bypassed shared cap");
    const auto after = snapshot(tight, "third-owner-refused");
    require(before.numeric_live_bytes == after.numeric_live_bytes &&
                before.provider_reserved_bytes == after.provider_reserved_bytes &&
                before.capacity_reserved_bytes == after.capacity_reserved_bytes,
            "failed prepare leaked charges");
    a.reset();
    auto retry = prepare(context, small, model, tight);
    require(std::isfinite(retry->plan->execute(true).energy), "retry after shortage failed");
    single_ports(retry);
  }
  empty(tight);
  {
    const std::array<const generativeqc_system*, 2> systems{small.get(), large.get()};
    generativeqc_batch* raw{};
    check(generativeqc_batch_prepare_with_device_resources(
              context.get(), systems.data(), systems.size(), &model.method,
              GENERATIVEQC_BATCH_ENABLE_WARM_STARTS, r.get(), &raw),
          context.get()->last_detail);
    Batch batch(raw, generativeqc_batch_destroy);
    snapshot(r, "ragged-prepared");
    auto run = [&](const qc::methods::Coordinates& xyz) {
      auto values = batch->plan->execute(xyz, true);
      require(values.size() == 2, "ragged publication missing");
      for (const auto& item : values)
        require(
            item.status == GENERATIVEQC_STATUS_SUCCESS && std::isfinite(item.calculation.energy),
            "ragged execution failed: " + std::to_string(item.status));
    };
    run({});
    batch_ports(batch, 0);
    batch_ports(batch, 1);
    snapshot(r, "ragged-cold-ports");
    std::string message;
    qc::dft::CudaKsFinalStateToken stale;
    check(detail::dft_final_state_token(*batch->plan, 1, stale, message), message);
    qc::methods::Coordinates xyz(2);
    xyz[0] = {0, 0, -.7, .03, 0, .78};
    xyz[1].emplace();
    for (const auto& atom : waters) {
      xyz[1]->push_back(atom.x);
      xyz[1]->push_back(atom.y);
      xyz[1]->push_back(atom.z);
    }
    xyz[1]->at(5) += .04;
    run(xyz);
    batch_ports(batch, 0);
    batch_ports(batch, 1);
    snapshot(r, "ragged-moved-ports");
    qc::dft::VerifiedKsFinalState output;
    require(detail::read_dft_final_state(*batch->plan, 1, stale, true, output, message) ==
                GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "geometry rebuild accepted old token");
    r.reset();
    run({});
    batch_ports(batch, 1);
  }
}

/** Restore diagnostic overrides even if construction or execution throws. */
class EnvironmentSetting {
 public:
  EnvironmentSetting(const char* name, const char* value) : name_(name) {
    if (const char* previous = std::getenv(name)) previous_ = previous;
    if (setenv(name, value, 1)) throw std::runtime_error("cannot set test environment");
  }
  ~EnvironmentSetting() {
    if (previous_)
      setenv(name_.c_str(), previous_->c_str(), 1);
    else
      unsetenv(name_.c_str());
  }
  EnvironmentSetting(const EnvironmentSetting&) = delete;
  EnvironmentSetting& operator=(const EnvironmentSetting&) = delete;

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

void test_admission_boundaries() {
  generativeqc_context_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.backend = GENERATIVEQC_BACKEND_CUDA;
  qc::Context context(descriptor);
  const std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, -.7}, {1, 0, 0, .7}}};
  qc::System system(context, atoms, "sto-3g");
  Model model;
  std::uint64_t bound{};
  {
    // Measure the predeclared reservation, not an observed numeric peak.
    // Disable optional owners while obtaining this native topology bound.
    EnvironmentSetting ao("GENERATIVEQC_CUDA_KS_ACTIVE_AO", "0");
    EnvironmentSetting batches("GENERATIVEQC_CUDA_XC_BATCH_TILES", "1");
    auto r = resources(cap);
    auto c = prepare(context, system, model, r);
    const auto d = snapshot(r, "mandatory-bound");
    require(d.capacity_reserved_bytes, "missing native mandatory promise");
    bound = d.numeric_live_bytes + d.provider_reserved_bytes + d.capacity_reserved_bytes;
  }
  std::vector<qc::methods::Result> results;
  for (unsigned mode = 0; mode < 4; ++mode) {
    const auto bytes = mode == 1 ? bound : mode == 2 ? bound - 1 : cap;
    auto r = resources(bytes);
    if (mode == 3) {
      r.reset();
      const generativeqc_device_resource_buffer_v1 row{bytes, 4, 0, 0, 0, 0};
      generativeqc_device_resource_descriptor desc{};
      desc.struct_size = sizeof(desc);
      desc.abi_version = GENERATIVEQC_ABI_VERSION;
      desc.buffers = &row;
      desc.buffer_count = 1;
      desc.admission = GENERATIVEQC_DEVICE_RESOURCE_SELECTED_INVENTORY;
      generativeqc_device_resources* raw{};
      qc::check(generativeqc_device_resources_create(&desc, &raw));
      r.reset(raw);
    }
    {
      auto c = prepare(context, system, model, r);
      const auto prepared = snapshot(r, "boundary-prepared-" + std::to_string(mode));
      require((prepared.capacity_reserved_bytes > 0) == (mode < 2),
              "unexpected mandatory promise eligibility");
      auto result = c->plan->execute(true);
      require(
          result.ks_diagnostic && result.ks_diagnostic->cuda_ao_selection.selected == (mode == 0),
          "optional AO selection ignored authority or mandatory fallback credit");
      std::cout << "{\"mode\":" << mode
                << ",\"ao_selected\":" << result.ks_diagnostic->cuda_ao_selection.selected
                << ",\"energy\":" << result.energy << "}\n";
      results.push_back(std::move(result));
      single_ports(c);
      require(!snapshot(r, "boundary-executed-" + std::to_string(mode)).rejected_allocations,
              "bounded optional admission attempted to spend mandatory capacity");
    }
    empty(r);
  }
  // PySCF 2.14.0, STO-3G H2 at 1.4 Bohr, PBE0 and the explicit v1
  // 64x12x24 grid. Independent analytical forces also pass reconverged FD.
  constexpr double reference_energy = -1.1543107969376867;
  constexpr std::array<double, 6> reference_force{-3.1245442208152474e-19, 5.805979911434448e-20,
                                                  0.012126775188883943,    3.1245442208153707e-19,
                                                  -5.805979911434486e-20,  -0.012126775188884165};
  for (const auto& result : results) {
    require(std::abs(result.energy - reference_energy) < 1e-8, "independent PBE0 energy mismatch");
    require(result.forces.size() == reference_force.size(), "independent force shape mismatch");
    for (std::size_t i = 0; i < reference_force.size(); ++i)
      require(std::abs(result.forces[i] - reference_force[i]) < 1e-6,
              "independent PBE0 force mismatch");
    require(std::abs(result.energy - results[0].energy) < 1e-8, "dense fallback energy mismatch");
    require(result.forces.size() == results[0].forces.size(), "dense fallback force size mismatch");
    for (std::size_t i = 0; i < result.forces.size(); ++i)
      require(std::abs(result.forces[i] - results[0].forces[i]) < 1e-6,
              "dense fallback force mismatch");
  }
}

void test_sparse_pair() {
  // Direct allocator evidence for many atoms sharing one orbital. SCF is
  // irrelevant here: the standalone bridge still retains both atom vectors.
  qc::core::System sparse;
  for (unsigned i = 0; i < 1000; ++i) sparse.atoms.push_back({1, {double(i), 0, 0}, 0});
  sparse.shells.push_back({0, 0, {{1.0, std::pow(2.0 / std::acos(-1.0), .75)}}});
  const auto paired = qc::scf::cuda_ks_one_electron_capacity(1, 1000, 1, 1, true);
  auto ledger = std::make_shared<qc::runtime::DeviceResourceLedger>();
  ledger->limit = paired;
  ledger->device = 0;
  {
    qc::runtime::DeviceCapacityReservation reservation(ledger, paired);
    qc::runtime::DeviceCapacityScope scope(reservation);
    std::vector<double> h, p;
    const std::array<double, 1> d{.3}, w{.2};
    std::string message;
    qc::scf::OneElectronGradientResources measured;
    check(qc::scf::execute_cuda_stationary_one_electron_pair(0, sparse, d, w, 1, cap, h, p, message,
                                                             &measured),
          message);
    require(h.size() == 3000 && p.size() == 3000 && !ledger->live &&
                ledger->capacity_reserved == paired,
            "paired gradient leaked capacity");
    require(measured.device_bytes <= paired && ledger->peak <= paired && !ledger->rejected,
            "sparse paired-gradient exceeded its declared native bound");
    require(std::all_of(h.begin(), h.end(), [](double value) { return std::isfinite(value); }) &&
                std::all_of(p.begin(), p.end(), [](double value) { return std::isfinite(value); }),
            "sparse paired-gradient published a nonfinite output");
    std::cout << "{\"paired_device_bytes\":" << measured.device_bytes
              << ",\"paired_bound\":" << paired << "}\n";
  }
  require(!ledger->capacity_reserved && !ledger->live, "paired promise teardown retained charges");
}

void test_df_response_ledger_headroom() {
  // A DF response allowance can exceed the public owner's remaining capacity.
  // RHF must remain admitted at its observed full-endpoint peak. UHF has a
  // smaller auxiliary-panel fallback and must also run one byte below its
  // unconstrained peak, preserving full energy/forces without first recording
  // an unnecessary rejected allocation in the ledger.
  generativeqc_context_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.backend = GENERATIVEQC_BACKEND_CUDA;
  qc::Context context(descriptor);
  EnvironmentSetting storage("GENERATIVEQC_DF_VALUE_STORAGE", "auto");
  EnvironmentSetting exchange("GENERATIVEQC_DF_EXCHANGE", "auto");
  constexpr std::array<std::array<double, 2>, 2> energy{
      {{-1.1933554788642575, -1.1872520538646867}, {-0.5385113483224678, -0.5519790864298522}}};
  constexpr std::array<std::array<double, 6>, 2> rhf_forces{
      {{0, 0, 0.05868322397695891, 0, 0, -0.0586832239769588},
       {0.0018615223820731986, 0, 0.09183510418227792, -0.0018615223820731934, 0,
        -0.09183510418227803}}};
  constexpr std::array<std::array<double, 6>, 2> forces{
      {{0, 0, -0.19038408686848274, 0, 0, 0.19038408686848285},
       {-0.0029698593196991994, 0, -0.146513059771827, 0.0029698593196991994, 0,
        0.146513059771827}}};
  for (unsigned spin = 0; spin < 2; ++spin) {
    for (unsigned geometry = 0; geometry < 2; ++geometry) {
      const std::array<generativeqc_atom, 2> atoms{
          {{1, 0, 0, -.7}, {1, geometry ? .03 : 0, 0, geometry ? .78 : .7}}};
      qc::System system(context, atoms, "sto-3g", spin ? 1 : 0, spin ? 2 : 1);
      qc::System auxiliary(context, atoms, "sto-3g");
      Model model;
      model.method.method = spin ? GENERATIVEQC_METHOD_UHF : GENERATIVEQC_METHOD_RHF;
      model.method.ks_options = nullptr;
      model.method.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CUDA;
      model.method.density_fitting_auxiliary_basis = auxiliary.get();
      model.method.density_fitting_relative_threshold = 1e-12;
      std::uint64_t constrained = cap;
      for (unsigned pass = 0; pass < 2; ++pass) {
        auto r = resources(constrained);
        {
          qc::Calculation calculation(context, system, model.method, r.get());
          const auto result =
              calculation.execute(GENERATIVEQC_PROPERTY_ENERGY | GENERATIVEQC_PROPERTY_FORCES);
          // Independent PySCF 2.14.0 DF-HF/STO-3G, including auxiliary-basis
          // response. These are physical forces, not a native wrapper oracle.
          require(std::abs(result.energy - energy[spin][geometry]) < 1e-8,
                  "independent DF-HF energy mismatch");
          require(result.forces && result.forces->size() == 6, "DF-HF forces missing");
          for (std::size_t i = 0; i < 6; ++i)
            require(
                std::abs(result.forces->at(i) - (spin ? forces : rhf_forces)[geometry][i]) < 1e-6,
                "independent DF-HF force mismatch");
          const auto d = snapshot(r, "df-response-headroom");
          require(!d.rejected_allocations, "DF response sized a panel beyond its bound ledger");
          if (!pass) {
            require(d.combined_peak_bytes > 1, "DF response peak missing");
            constrained = d.combined_peak_bytes - (spin ? 1 : 0);
          }
        }
        empty(r);
      }
    }
  }
}

}  // namespace

int main() {
  try {
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    test_named_correction_admission();
    test_shared_lifetimes();
    test_admission_boundaries();
    test_sparse_pair();
    test_df_response_ledger_headroom();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
