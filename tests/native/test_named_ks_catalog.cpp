#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "api/ks_dispersion.hpp"
#include "dft/application_precision.hpp"
#include "generativeqc/ks.hpp"
#include "methods/application_precision.hpp"
#include "methods/named_ks_catalog.hpp"
#include "molecule/basis_template.hpp"

extern "C" generativeqc_status generativeqc_ks_catalog_plan_v1(const char*, std::size_t,
                                                               const char*, std::size_t,
                                                               generativeqc_backend,
                                                               generativeqc_ks_options*,
                                                               std::uint32_t*);
extern "C" int generativeqc_resource_ks_cpu_df_fleet_v1(const std::uint64_t*, std::size_t,
                                                        std::uint64_t, std::uint64_t*, std::size_t);
extern "C" int generativeqc_resource_ks_cpu_df_force_fleet_v1(const std::uint64_t*, std::size_t,
                                                              std::uint64_t, std::uint64_t*,
                                                              std::size_t);

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

// A client built before the packed ERI field passes only this allocation.
struct OriginalNamedKsDescriptor {
  std::uint32_t struct_size;
  std::uint32_t abi_version;
  const char* name;
  const char* identity;
  std::uint64_t dispersion_maximum_bytes;
};

static_assert(sizeof(OriginalNamedKsDescriptor) ==
              offsetof(generativeqc_named_ks_descriptor, cpu_packed_eri_maximum_bytes));

// A client built with packed ERIs passes no CPU DF host admission suffix.
struct PackedNamedKsDescriptor {
  OriginalNamedKsDescriptor prefix;
  std::uint64_t cpu_packed_eri_maximum_bytes;
};

static_assert(sizeof(PackedNamedKsDescriptor) ==
              offsetof(generativeqc_named_ks_descriptor, cpu_df_host_maximum_bytes));

const generativeqc_named_ks_descriptor* original_wire(const OriginalNamedKsDescriptor& original) {
  // Pass the historical wire bytes; neither caller nor callee should read a
  // current-layout object from that address.
  return reinterpret_cast<const generativeqc_named_ks_descriptor*>(&original);
}

generativeqc_ks_options controls() {
  generativeqc_ks_options result{};
  result.struct_size = sizeof(result);
  result.abi_version = GENERATIVEQC_ABI_VERSION;
  result.grid_version = 2;
  result.radial_points = 23;
  result.angular_polar = 7;
  result.angular_azimuth = 11;
  result.partition_iterations = 3;
  result.tile_points = 101;
  result.force_tile_points = 1024;
  result.coincident_tolerance = 1e-12;
  result.nonlocal_maximum_bytes = 1234567;
  return result;
}

void test_cpu_df_inventory_identity() {
  using generativeqc::methods::named_ks::find_method;
  for (const auto selector : {"lda_xc_pw-rks", "lda_xc_pw-uks", "pbe-rks", "pbe-uks"}) {
    const auto* method = find_method(selector);
    require(method && method->cpu, "expected a linked CPU semilocal program");
    const char* selected = nullptr;
    require(generativeqc_ks_cpu_df_energy_program_v1(method->identity.data(), &selected) == 0 &&
                selected && method->name == selected,
            "exact semilocal graph lost its CPU DF inventory");
  }
  if (const auto* alias = find_method("libxc:pbe")) {
    const auto* pbe = find_method("pbe-rks");
    if (alias->identity == pbe->identity) {
      const char* selected = nullptr;
      require(generativeqc_ks_cpu_df_energy_program_v1(alias->identity.data(), &selected) == 0 &&
                  selected && pbe->name == selected,
              "alias order changed CPU DF inventory selection");
    }
  }
  const auto* hybrid = find_method("pbe0-rks");
  require(hybrid, "expected a neighboring hybrid program");
  const char* selected = "unchanged";
  require(generativeqc_ks_cpu_df_energy_program_v1(hybrid->identity.data(), &selected) == 0 &&
              !selected,
          "hybrid unexpectedly selected the semilocal CPU DF inventory");
  selected = "unchanged";
  require(generativeqc_ks_cpu_df_energy_program_v1("pbe-rks", &selected) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              selected && std::string_view(selected) == "unchanged",
          "selector spelling bypassed MethodIR validation");
}

void test_complete_catalog() {
  using namespace generativeqc::methods::named_ks;
  require(methods().size() > 10, "catalog was lost in a library-only build");
  for (const auto& entry : methods()) {
    for (const auto backend : {GENERATIVEQC_BACKEND_CPU_REFERENCE, GENERATIVEQC_BACKEND_CUDA}) {
      auto output = controls();
      const auto before = output;
      std::uint32_t selected = 9;
      require(generativeqc_ks_catalog_plan_v1(entry.name.data(), entry.name.size(), nullptr, 0,
                                              backend, &output, &selected) == 0,
              "metadata query loaded a backend or rejected a valid row");
      const bool complete = backend == GENERATIVEQC_BACKEND_CUDA ? entry.cuda : entry.cpu;
      generativeqc_ks_catalog_binding_v1 binding{sizeof(binding), GENERATIVEQC_ABI_VERSION, nullptr,
                                                 nullptr, 0};
      require(generativeqc_ks_catalog_lookup_v1(entry.name.data(), entry.name.size(), backend,
                                                &binding) == GENERATIVEQC_STATUS_SUCCESS,
              "native selector lookup rejected a catalog spelling");
      if (complete) {
        require(binding.identifier && binding.identity && binding.identifier == entry.source &&
                    binding.identity == entry.identity && binding.spin == entry.spin,
                "selector metadata differs from the prepared program");
      } else {
        require(!binding.identifier && !binding.identity && !binding.spin,
                "unavailable selector published metadata for the wrong backend");
      }
      const bool admitted = complete && !entry.dispersion.kind;
      require(selected == static_cast<unsigned>(admitted), "catalog admission flags diverged");
      if (complete) {
        const char* name = nullptr;
        require(generativeqc_ks_catalog_name_v1(entry.identity.data(), backend, &name) == 0 && name,
                "complete named program was unavailable by identity");
      }
      if (!admitted) {
        require(std::memcmp(&output, &before, sizeof(output)) == 0,
                "explicit compiler route changed caller controls");
        continue;
      }
      require(
          output.semilocal_components == components(entry).data() &&
              output.exchange_terms == (entry.exchange_count ? exchanges(entry).data() : nullptr) &&
              output.scf_domain == entry.domain.data() && output.spin_channels == entry.spin,
          "CLI and API are not borrowing the same catalog owner");
      require(output.grid_version == before.grid_version &&
                  output.radial_points == before.radial_points &&
                  output.angular_polar == before.angular_polar &&
                  output.angular_azimuth == before.angular_azimuth &&
                  output.tile_points == before.tile_points &&
                  output.force_tile_points == before.force_tile_points &&
                  output.nonlocal_maximum_bytes == before.nonlocal_maximum_bytes &&
                  output.coincident_tolerance == before.coincident_tolerance,
              "catalog selection changed caller execution policy");
      auto by_identity = controls();
      selected = 9;
      require(
          generativeqc_ks_catalog_plan_v1(nullptr, 0, entry.identity.data(), entry.identity.size(),
                                          backend, &by_identity, &selected) == 0 &&
              selected == 1,
          "complete identity lookup failed");
      require(by_identity.semilocal_component_count == output.semilocal_component_count &&
                  by_identity.exchange_term_count == output.exchange_term_count,
              "identity lookup changed the scientific inventory");
      for (const auto& term : exchanges(entry))
        require(std::isfinite(term.fock_coefficient) &&
                    term.fock_coefficient == -term.coefficient / (entry.spin == 1 ? 2.0 : 1.0) &&
                    term.operator_kind >= GENERATIVEQC_KS_EXCHANGE_FULL_RANGE &&
                    term.operator_kind <= GENERATIVEQC_KS_EXCHANGE_LONG_RANGE,
                "compiler exchange output was lost");
      require(output.semilocal_range_omega == entry.range_omega &&
                  output.has_nonlocal_correlation == (entry.nonlocal.variant != 0) &&
                  output.nonlocal_b == entry.nonlocal.b && output.nonlocal_c == entry.nonlocal.c &&
                  output.nonlocal_coefficient == entry.nonlocal.coefficient,
              "complete nonlocal program was truncated");
    }
  }
}

void test_identity_and_transaction_boundaries() {
  const auto* entry = generativeqc::methods::named_ks::find_method("pbe0-rks");
  require(entry && entry->cpu, "PBE0 fixture unavailable");
  auto output = controls();
  const auto before = output;
  std::uint32_t selected = 99;
  const std::string different_identity(64, 'f');
  require(generativeqc_ks_catalog_plan_v1("pbe0-rks", 8, different_identity.data(), 64,
                                          GENERATIVEQC_BACKEND_CPU_REFERENCE, &output,
                                          &selected) == 0 &&
              selected == 0 && std::memcmp(&output, &before, sizeof(output)) == 0,
          "same-name custom graph was replaced with a catalog model");
  output.scf_domain = "explicit-other-domain";
  const auto domain_before = output;
  require(
      generativeqc_ks_catalog_plan_v1("pbe0-rks", 8, nullptr, 0, GENERATIVEQC_BACKEND_CPU_REFERENCE,
                                      &output, &selected) == 0 &&
          selected == 0 && std::memcmp(&output, &domain_before, sizeof(output)) == 0,
      "catalog silently replaced the explicit numerical domain");
  output = controls();
  selected = 99;
  const auto rejected = [&](const char* name, std::size_t size, const char* identity,
                            std::size_t identity_size, int backend) {
    require(generativeqc_ks_catalog_plan_v1(name, size, identity, identity_size, backend, &output,
                                            &selected) != 0 &&
                selected == 99 && std::memcmp(&output, &before, sizeof(output)) == 0,
            "invalid input changed output or selection publication");
  };
  rejected(nullptr, 8, nullptr, 0, GENERATIVEQC_BACKEND_CPU_REFERENCE);
  rejected("pbe0-rks", 257, nullptr, 0, GENERATIVEQC_BACKEND_CPU_REFERENCE);
  rejected("pbe0-rks\0tail", 13, nullptr, 0, GENERATIVEQC_BACKEND_CPU_REFERENCE);
  rejected(nullptr, 0, nullptr, 0, GENERATIVEQC_BACKEND_CPU_REFERENCE);
  rejected("pbe0-rks", 8, "short", 5, GENERATIVEQC_BACKEND_CPU_REFERENCE);
  rejected("pbe0-rks", 8, different_identity.data(), 64, 99);
  require(
      generativeqc_ks_catalog_plan_v1("pbe0-rks", 8, nullptr, 0, GENERATIVEQC_BACKEND_CPU_REFERENCE,
                                      &output, &output.spin_channels) != 0 &&
          std::memcmp(&output, &before, sizeof(output)) == 0,
      "aliased output/selection changed publication");
  require(
      generativeqc_ks_catalog_plan_v1("pbe0-rks", 8, nullptr, 0, GENERATIVEQC_BACKEND_CPU_REFERENCE,
                                      &output, &selected) == 0 &&
          selected == 1,
      "invalid query contaminated a fresh selection");
}
void test_borrowed_controls_and_extension_aliases() {
  struct Extended {
    generativeqc_ks_options prefix;
    double application_energy;
    double application_force;
  };
  Extended extended{};
  extended.prefix = controls();
  extended.prefix.struct_size = sizeof(extended);
  extended.application_energy = 1e-7;
  extended.application_force = 2e-7;
  const auto extended_before = extended;
  require(generativeqc_ks_catalog_plan_v1(
              "pbe0-rks", 8, nullptr, 0, GENERATIVEQC_BACKEND_CPU_REFERENCE, &extended.prefix,
              reinterpret_cast<std::uint32_t*>(&extended.application_energy)) != 0 &&
              std::memcmp(&extended, &extended_before, sizeof(extended)) == 0,
          "selection aliased the application extension");

  auto output = controls();
  std::array<double, 2> radii{1.0, 2.0};
  const auto radii_before = radii;
  output.element_radii = radii.data();
  output.element_radius_count = radii.size();
  const auto output_before = output;
  require(generativeqc_ks_catalog_plan_v1("pbe0-rks", 8, nullptr, 0,
                                          GENERATIVEQC_BACKEND_CPU_REFERENCE, &output,
                                          reinterpret_cast<std::uint32_t*>(radii.data())) != 0 &&
              std::memcmp(&output, &output_before, sizeof(output)) == 0 && radii == radii_before,
          "selection overwrote borrowed caller radii");

  alignas(std::uint32_t) char domain[] = "explicit-other-domain";
  const std::string domain_before(domain);
  output.scf_domain = domain;
  const auto domain_output_before = output;
  require(
      generativeqc_ks_catalog_plan_v1("pbe0-rks", 8, nullptr, 0, GENERATIVEQC_BACKEND_CPU_REFERENCE,
                                      &output, reinterpret_cast<std::uint32_t*>(domain)) != 0 &&
          std::memcmp(&output, &domain_output_before, sizeof(output)) == 0 &&
          domain_before == domain,
      "unselected query overwrote a borrowed domain");

  std::uint32_t selected = 9;
  require(
      generativeqc_ks_catalog_plan_v1("pbe0-rks", 8, nullptr, 0, GENERATIVEQC_BACKEND_CPU_REFERENCE,
                                      &extended.prefix, &selected) == 0 &&
          selected == 1 && extended.application_energy == extended_before.application_energy &&
          extended.application_force == extended_before.application_force,
      "valid selection changed the application extension after alias refusal");
}

void test_named_preparation_failures_and_lifetime() {
  generativeqc::Context context({sizeof(generativeqc_context_descriptor), GENERATIVEQC_ABI_VERSION,
                                 0, GENERATIVEQC_BACKEND_CPU_REFERENCE});
  const std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}}};
  std::unique_ptr<generativeqc::Calculation> calculation;
  std::unique_ptr<generativeqc::Calculation> nonlocal;
  std::unique_ptr<generativeqc_batch, decltype(&generativeqc_batch_destroy)> batch(
      nullptr, generativeqc_batch_destroy);
  {
    generativeqc::System system(context, atoms, "sto-3g");
    const generativeqc_system* systems[]{system.get()};
    auto method = generativeqc::default_method_descriptor("pbe-rks");
    auto grid = controls();
    grid.grid_version = 1;
    grid.radial_points = 8;
    grid.angular_polar = 4;
    grid.angular_azimuth = 8;
    method.ks_options = &grid;
    std::string selector = "pbe-d3(bj)-rks";
    std::string identity(generativeqc::methods::named_ks::find_method(selector)->identity);
    generativeqc_named_ks_descriptor named{sizeof(named),    GENERATIVEQC_ABI_VERSION,
                                           selector.c_str(), identity.c_str(),
                                           256ull << 20,     0};
    // Both public entry points must clear an old output value even when
    // rejection happens before any numerical owner is constructed.
    const auto rejected = [&](const generativeqc_method_descriptor& request,
                              const generativeqc_named_ks_descriptor* name,
                              generativeqc_status expected) {
      auto* single = reinterpret_cast<generativeqc_calculation*>(1);
      auto* fleet = reinterpret_cast<generativeqc_batch*>(1);
      require(generativeqc_calculation_prepare_named_ks_v1(context.get(), system.get(), &request,
                                                           name, &single) == expected &&
                  !single,
              "named single refusal published an owner or wrong status");
      require(generativeqc_batch_prepare_named_ks_v1(context.get(), systems, 1, &request, name, 0,
                                                     &fleet) == expected &&
                  !fleet,
              "named batch refusal published an owner or wrong status");
    };
    rejected(method, nullptr, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    auto invalid = named;
    invalid.abi_version += 1;
    rejected(method, &invalid, GENERATIVEQC_STATUS_ABI_MISMATCH);
    for (const auto bytes : {sizeof(OriginalNamedKsDescriptor) - 1,
                             sizeof(OriginalNamedKsDescriptor) + 1, sizeof(named) + 1}) {
      invalid = named;
      invalid.struct_size = static_cast<std::uint32_t>(bytes);
      rejected(method, &invalid, GENERATIVEQC_STATUS_ABI_MISMATCH);
    }
    invalid = named;
    invalid.identity = "incorrect-identity";
    rejected(method, &invalid, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    invalid = named;
    invalid.name = "unrepresented-functional-rks";
    rejected(method, &invalid, GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
    invalid = named;
    invalid.dispersion_maximum_bytes = 0;
    rejected(method, &invalid, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    auto wrong_spin = method;
    wrong_spin.method = GENERATIVEQC_METHOD_PBE_UKS;
    rejected(wrong_spin, &named, GENERATIVEQC_STATUS_INVALID_ARGUMENT);

    // Invalid extension controls must reach their existing validators, not
    // disappear when the named adapter replaces scientific fields.
    generativeqc::methods::detail::ApplicationPrecisionMethodOptions extended_method{};
    extended_method.base = method;
    extended_method.base.struct_size = sizeof(extended_method);
    extended_method.energy = -1;
    rejected(extended_method.base, &named, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    extended_method.energy = 0;
    generativeqc_ks_nonlocal_grid_options extended_grid{};
    extended_grid.version = 1;
    extended_grid.base.base = grid;
    extended_grid.base.base.struct_size = sizeof(extended_grid);
    extended_grid.base.energy = -1;
    extended_method.base.ks_options = &extended_grid.base.base;
    rejected(extended_method.base, &named, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    extended_grid.base.energy = 0;
    rejected(extended_method.base, &named, GENERATIVEQC_STATUS_ABI_MISMATCH);
    extended_grid.grid = &grid;
    rejected(extended_method.base, &named, GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    extended_grid.base.base.struct_size = sizeof(extended_grid.base);

    calculation =
        std::make_unique<generativeqc::Calculation>(context, system, extended_method.base, named);
    const OriginalNamedKsDescriptor original{sizeof(original), named.abi_version, named.name,
                                             named.identity, named.dispersion_maximum_bytes};
    generativeqc_calculation* original_single = nullptr;
    generativeqc::check(generativeqc_calculation_prepare_named_ks_v1(
        context.get(), system.get(), &extended_method.base, original_wire(original),
        &original_single));
    generativeqc_calculation_destroy(original_single);
    generativeqc_batch* raw = nullptr;
    generativeqc::check(generativeqc_batch_prepare_named_ks_v1(
        context.get(), systems, 1, &extended_method.base, original_wire(original), 0, &raw));
    batch.reset(raw);
    extended_grid.base.base.struct_size = sizeof(extended_grid);
    named.name = "wb97m-v-rks";
    named.identity = nullptr;
    require(generativeqc_ks_nonlocal_grid_version_v1() == 1,
            "independent nonlocal-grid transport is unavailable");
    extended_grid.version = 2;
    rejected(extended_method.base, &named, GENERATIVEQC_STATUS_ABI_MISMATCH);
    extended_grid.version = 1;
    extended_grid.reserved = 1;
    rejected(extended_method.base, &named, GENERATIVEQC_STATUS_ABI_MISMATCH);
    extended_grid.reserved = 0;
    grid.abi_version += 1;
    rejected(extended_method.base, &named, GENERATIVEQC_STATUS_ABI_MISMATCH);
    grid.abi_version = GENERATIVEQC_ABI_VERSION;
    nonlocal =
        std::make_unique<generativeqc::Calculation>(context, system, extended_method.base, named);
    selector.assign(selector.size(), 'x');
    identity.assign(identity.size(), '0');
    extended_method = {};
    extended_grid = {};
  }
  // Descriptors, strings and source system are now gone; both prepared
  // endpoints must still compute the same complete energy from owned inputs.
  const auto single = calculation->execute(GENERATIVEQC_PROPERTY_ENERGY);
  generativeqc_batch_item_result_descriptor item{};
  item.struct_size = sizeof(item);
  item.abi_version = GENERATIVEQC_ABI_VERSION;
  generativeqc::check(generativeqc_batch_execute(batch.get(), nullptr, 0, &item, 1));
  require(item.status == GENERATIVEQC_STATUS_SUCCESS && std::isfinite(item.energy) &&
              std::abs(item.energy - single.energy) < 1e-12,
          "named owners lost copied controls/system or diverged between endpoints");
  // Same independent PySCF/Libxc grid oracle as the CLI endpoint fixture.
  require(
      std::abs(nonlocal->execute(GENERATIVEQC_PROPERTY_ENERGY).energy + 1.150312466420175) < 2e-8,
      "named preparation lost its borrowed nonlocal-grid controls");
}

void test_bound_basis_and_system_factory() {
  generativeqc::Context context({sizeof(generativeqc_context_descriptor), GENERATIVEQC_ABI_VERSION,
                                 0, GENERATIVEQC_BACKEND_CPU_REFERENCE});
  const std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}}};
  const auto* entry = generativeqc::methods::named_ks::find_method("r2scan-3c-rks");
  require(entry && entry->bound_basis && entry->cpu, "canonical bound basis is unavailable");
  generativeqc_named_ks_descriptor named{
      sizeof(named), GENERATIVEQC_ABI_VERSION, "r2scan-3c-rks", nullptr, 256ull << 20, 0};
  generativeqc_system_descriptor descriptor{sizeof(descriptor),
                                            GENERATIVEQC_ABI_VERSION,
                                            atoms.data(),
                                            2,
                                            nullptr,
                                            0,
                                            nullptr,
                                            0,
                                            0,
                                            1,
                                            GENERATIVEQC_BASIS_SPHERICAL};
  const auto refused_factory = [&](const auto& input, const auto& program) {
    auto* result = reinterpret_cast<generativeqc_system*>(1);
    require(generativeqc_system_create_named_ks_v1(context.get(), &input, &program, &result) != 0 &&
                !result,
            "invalid named basis factory published a system");
  };
  auto wrong = named;
  wrong.identity = "wrong-identity";
  refused_factory(descriptor, wrong);
  wrong = named;
  ++wrong.struct_size;
  generativeqc_system* bad_size = reinterpret_cast<generativeqc_system*>(1);
  require(generativeqc_system_create_named_ks_v1(context.get(), &descriptor, &wrong, &bad_size) ==
                  GENERATIVEQC_STATUS_ABI_MISMATCH &&
              !bad_size,
          "bound basis factory accepted an unknown named descriptor size");
  const OriginalNamedKsDescriptor original{sizeof(original), named.abi_version, named.name,
                                           named.identity, named.dispersion_maximum_bytes};
  generativeqc_system* original_system = nullptr;
  generativeqc::check(generativeqc_system_create_named_ks_v1(
      context.get(), &descriptor, original_wire(original), &original_system));
  generativeqc_system_destroy(original_system);
  wrong = named;
  wrong.name = "pbe-rks";
  refused_factory(descriptor, wrong);
  auto invalid = descriptor;
  invalid.abi_version += 1;
  refused_factory(invalid, named);
  const generativeqc_atom potassium{19, 0, 0, 0};
  invalid = descriptor;
  invalid.atoms = &potassium;
  invalid.atom_count = 1;
  refused_factory(invalid, named);

  auto expanded = generativeqc::molecule::expand_basis_template(*entry->bound_basis, atoms,
                                                                GENERATIVEQC_BASIS_SPHERICAL);
  descriptor.shells = expanded.shells.data();
  descriptor.shell_count = expanded.shells.size();
  descriptor.primitives = expanded.primitives.data();
  descriptor.primitive_count = expanded.primitives.size();
  refused_factory(descriptor, named);  // Callers cannot override the bound basis.

  auto method = generativeqc::default_method_descriptor("pbe-rks");
  auto grid = controls();
  grid.grid_version = 1;
  grid.radial_points = 8;
  grid.angular_polar = 4;
  grid.angular_azimuth = 8;
  method.ks_options = &grid;
  const auto rejected_basis = [&](const generativeqc::System& system) {
    auto* result = reinterpret_cast<generativeqc_calculation*>(1);
    auto* batch = reinterpret_cast<generativeqc_batch*>(1);
    const generativeqc_system* systems[]{system.get()};
    require(generativeqc_calculation_prepare_named_ks_v1(context.get(), system.get(), &method,
                                                         &named, &result) != 0 &&
                !result,
            "named single accepted a changed basis");
    require(generativeqc_batch_prepare_named_ks_v1(context.get(), systems, 1, &method, &named, 0,
                                                   &batch) != 0 &&
                !batch,
            "named batch accepted a changed basis");
  };
  std::unique_ptr<generativeqc::Calculation> factory_calculation;
  std::unique_ptr<generativeqc::Calculation> explicit_calculation;
  {
    generativeqc::System factory(context, atoms, named);
    generativeqc::System explicit_basis(context, descriptor);
    factory_calculation =
        std::make_unique<generativeqc::Calculation>(context, factory, method, named);
    explicit_calculation =
        std::make_unique<generativeqc::Calculation>(context, explicit_basis, method, named);
    generativeqc::System unrelated(context, atoms, "sto-3g");
    rejected_basis(unrelated);
    descriptor.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
    generativeqc::System cartesian(context, descriptor);
    rejected_basis(cartesian);
    descriptor.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    expanded.primitives.front().exponent *= 1.001;
    generativeqc::System changed(context, descriptor);
    rejected_basis(changed);
  }
  expanded = {};
  named = {};
  grid = {};
  method = {};
  const auto actual = factory_calculation->execute(GENERATIVEQC_PROPERTY_ENERGY);
  const auto expected = explicit_calculation->execute(GENERATIVEQC_PROPERTY_ENERGY);
  require(std::isfinite(actual.energy) && std::abs(actual.energy - expected.energy) < 1e-12,
          "method-owned and explicit bound basis differ after input destruction");
}

void test_cpu_df_native_host_budget() {
  const auto* program = generativeqc::methods::named_ks::find_method("pbe-rks");
  require(program && program->cpu, "CPU PBE catalog entry is required");
  generativeqc::Context context({sizeof(generativeqc_context_descriptor), GENERATIVEQC_ABI_VERSION,
                                 0, GENERATIVEQC_BACKEND_CPU_REFERENCE});
  const std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}}};
  generativeqc::System system(context, atoms, "sto-3g");
  const std::array<const generativeqc_system*, 2> fleet_systems{system.get(), system.get()};
  const std::array<const generativeqc::System*, 2> sdk_fleet{&system, &system};
  auto method = generativeqc::default_method_descriptor("pbe-rks");
  method.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE;
  method.diis_history = 8;
  method.max_iterations = 100;
  auto grid = controls();
  grid.grid_version = 1;
  grid.radial_points = 8;
  grid.angular_polar = 4;
  grid.angular_azimuth = 8;
  grid.tile_points = 101;
  grid.force_tile_points = 400;
  method.ks_options = &grid;
  constexpr auto energy = GENERATIVEQC_PROPERTY_ENERGY;
  constexpr auto both = energy | GENERATIVEQC_PROPERTY_FORCES;
  const generativeqc_named_ks_descriptor ordinary{sizeof(ordinary),
                                                  GENERATIVEQC_ABI_VERSION,
                                                  "pbe-rks",
                                                  program->identity.data(),
                                                  0,
                                                  0,
                                                  0,
                                                  0,
                                                  0};
  const PackedNamedKsDescriptor packed{
      {sizeof(packed), ordinary.abi_version, ordinary.name, ordinary.identity, 0}, 0};
  const auto* packed_wire = original_wire(packed.prefix);
  generativeqc_calculation* legacy_calculation = nullptr;
  generativeqc::check(generativeqc_calculation_prepare_named_ks_v1(
      context.get(), system.get(), &method, packed_wire, &legacy_calculation));
  generativeqc_calculation_destroy(legacy_calculation);
  generativeqc_batch* legacy_batch = nullptr;
  generativeqc::check(generativeqc_batch_prepare_named_ks_v1(context.get(), fleet_systems.data(),
                                                             fleet_systems.size(), &method,
                                                             packed_wire, 0, &legacy_batch));
  generativeqc_batch_destroy(legacy_batch);

  const auto quote = [&](bool forces, std::size_t copies) {
    const std::array<std::uint64_t, 22> shape{
        2, 2, 2, 2, 6, 1, 0, 0, 512, 8, 4, forces ? 400u : 101u, 8, 100, 1, 0, 0, 2, 2, 2, 6, 0};
    std::array<std::uint64_t, 44> shapes{};
    std::array<std::uint64_t, 47> values{};
    std::copy(shape.begin(), shape.end(), shapes.begin());
    std::copy(shape.begin(), shape.end(), shapes.begin() + shape.size());
    const auto query = forces ? generativeqc_resource_ks_cpu_df_force_fleet_v1
                              : generativeqc_resource_ks_cpu_df_fleet_v1;
    require(query(shapes.data(), copies * shape.size(), 0, values.data(), 15 + 16 * copies) == 0,
            "CPU DF fixture capacity could not be quoted");
    return values[14];
  };
  for (bool forces : {false, true}) {
    const auto properties = forces ? both : energy;
    auto named = ordinary;
    named.cpu_df_host_properties = properties;
    named.cpu_df_host_maximum_bytes = quote(forces, 1) - 1;
    auto* rejected_single = reinterpret_cast<generativeqc_calculation*>(1);
    require(generativeqc_calculation_prepare_named_ks_v1(context.get(), system.get(), &method,
                                                         &named, &rejected_single) ==
                    GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
                !rejected_single,
            "one-byte-short native calculation published an owner");
    ++named.cpu_df_host_maximum_bytes;
    generativeqc::Calculation bounded(context, system, method, named);
    require(bounded.supported_properties() == properties,
            "calculation did not retain its budgeted properties");
    generativeqc::Calculation baseline(context, system, method, ordinary);
    const auto expected = baseline.execute(properties);
    const auto actual = bounded.execute(properties);
    require(actual.iterations == expected.iterations && actual.energy == expected.energy &&
                actual.forces == expected.forces,
            "bounded native calculation changed the science");

    named.cpu_df_host_maximum_bytes = quote(forces, 2) - 1;
    auto* rejected_batch = reinterpret_cast<generativeqc_batch*>(1);
    require(generativeqc_batch_prepare_named_ks_v1(
                context.get(), fleet_systems.data(), fleet_systems.size(), &method, &named, 0,
                &rejected_batch) == GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
                !rejected_batch,
            "one-byte-short native fleet published an owner");
    ++named.cpu_df_host_maximum_bytes;
    auto batch = generativeqc::Batch::prepare_named_ks(context, sdk_fleet, method, named);
    named = {};
    for (const auto& item : batch.execute({}, forces)) {
      require(item.converged && item.iterations == expected.iterations &&
                  item.energy == expected.energy &&
                  item.forces == expected.forces.value_or(std::vector<double>{}),
              "bounded native fleet changed the science");
    }
    if (!forces) {
      try {
        (void)batch.execute({}, true);
        require(false, "an energy-only native fleet replayed forces");
      } catch (const generativeqc::Error& error) {
        require(error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                "unbudgeted fleet force replay returned the wrong status");
      }
    }
  }
}

void dump_bound_basis() {
  const auto* entry = generativeqc::methods::named_ks::find_method("r2scan-3c-rks");
  std::cout << std::setprecision(17);
  for (int z = 1; z <= 18; ++z) {
    const std::array<generativeqc_atom, 1> atom{{{z, 0, 0, 0}}};
    const auto expanded = generativeqc::molecule::expand_basis_template(
        *entry->bound_basis, atom, GENERATIVEQC_BASIS_SPHERICAL);
    for (const auto& shell : expanded.shells) {
      std::cout << z << " " << shell.angular_momentum;
      for (std::size_t k = shell.primitive_offset;
           k < shell.primitive_offset + shell.primitive_count; ++k) {
        const auto& p = expanded.primitives[k];
        std::cout << " " << p.exponent << " " << p.coefficient;
      }
      std::cout << "\n";
    }
  }
}

}  // namespace

int main(int argc, char**) {
  try {
    if (argc > 1) {
      dump_bound_basis();
      return 0;
    }
    test_bound_basis_and_system_factory();
    test_complete_catalog();
    test_cpu_df_inventory_identity();
    test_cpu_df_native_host_budget();
    test_identity_and_transaction_boundaries();
    test_borrowed_controls_and_extension_aliases();
    test_named_preparation_failures_and_lifetime();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
