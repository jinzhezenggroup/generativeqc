#include <array>
#include <cmath>
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

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
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
  result.coincident_tolerance = 1e-12;
  result.nonlocal_maximum_bytes = 1234567;
  return result;
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
    generativeqc_named_ks_descriptor named{sizeof(named), GENERATIVEQC_ABI_VERSION,
                                           selector.c_str(), identity.c_str(), 256ull << 20};
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
    generativeqc::dft::NonlocalGridKsOptions extended_grid{};
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
    generativeqc_batch* raw = nullptr;
    generativeqc::check(generativeqc_batch_prepare_named_ks_v1(
        context.get(), systems, 1, &extended_method.base, &named, 0, &raw));
    batch.reset(raw);
    extended_grid.base.base.struct_size = sizeof(extended_grid);
    named.name = "wb97m-v-rks";
    named.identity = nullptr;
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
  generativeqc_named_ks_descriptor named{sizeof(named), GENERATIVEQC_ABI_VERSION, "r2scan-3c-rks",
                                         nullptr, 256ull << 20};
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
    test_identity_and_transaction_boundaries();
    test_borrowed_controls_and_extension_aliases();
    test_named_preparation_failures_and_lifetime();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
