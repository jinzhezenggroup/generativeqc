#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "methods/named_ks_catalog.hpp"

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
      const bool admitted = backend == GENERATIVEQC_BACKEND_CUDA ? entry.cuda : entry.cpu;
      require(selected == static_cast<unsigned>(admitted), "catalog admission flags diverged");
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
                    term.operator_kind == GENERATIVEQC_KS_EXCHANGE_FULL_RANGE,
                "compiler exchange output was lost");
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
}  // namespace

int main() {
  try {
    test_complete_catalog();
    test_identity_and_transaction_boundaries();
    test_borrowed_controls_and_extension_aliases();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
