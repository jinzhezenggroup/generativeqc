#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "api/handles.hpp"
#include "generativeqc/generativeqc.hpp"
#include "molecule/basis_catalog.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void dump_catalog() {
  std::cout << std::setprecision(17);
  for (const auto name : generativeqc::molecule::bundled_basis_names()) {
    for (int atomic_number = 1; atomic_number <= 118; ++atomic_number) {
      if (!generativeqc::molecule::bundled_basis_supports(name, atomic_number)) continue;
      const std::array<generativeqc_atom, 1> atoms{{{atomic_number, 0, 0, 0}}};
      const auto basis =
          generativeqc::molecule::expand_bundled_basis(name, atoms, GENERATIVEQC_BASIS_CARTESIAN);
      for (const auto& shell : basis.shells) {
        std::cout << name << ' ' << atomic_number << ' ' << shell.angular_momentum;
        for (std::size_t index = shell.primitive_offset;
             index < shell.primitive_offset + shell.primitive_count; ++index) {
          const auto& primitive = basis.primitives[index];
          std::cout << ' ' << primitive.exponent << ' ' << primitive.coefficient;
        }
        std::cout << '\n';
      }
    }
  }
}

void test_capacity_catalog() {
  const auto names = generativeqc::molecule::bundled_basis_names();
  require(generativeqc::molecule::bundled_basis_count() == names.size(),
          "catalog discovery count differs from compiled names");
  require(generativeqc::molecule::bundled_basis_name(names.size()).empty(),
          "out-of-range discovery must not borrow a name");
  require(generativeqc::molecule::canonical_basis_name("STO_3G") == "sto-3g",
          "canonical discovery must retain case/underscore aliases");
  require(generativeqc::molecule::bundled_basis_pack_checksum() ==
              generativeqc::molecule::bundled_basis_catalog_sha256(),
          "versioned and legacy provenance must identify the same compiled pack");
  for (std::size_t index = 0; index < names.size(); ++index) {
    require(generativeqc::molecule::bundled_basis_name(index) == names[index],
            "borrowed discovery name differs from the existing catalog");
    for (int atomic_number = 1; atomic_number <= 118; ++atomic_number) {
      if (!generativeqc::molecule::bundled_basis_supports(names[index], atomic_number)) continue;
      const std::array<generativeqc_atom, 1> atoms{{{atomic_number, 0, 0, 0}}};
      for (const auto representation :
           {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
        const auto capacity =
            generativeqc::molecule::bundled_basis_capacity(names[index], atoms, representation);
        const auto expanded =
            generativeqc::molecule::expand_bundled_basis(names[index], atoms, representation);
        std::size_t expected_aos = 0;
        for (const auto& shell : expanded.shells)
          expected_aos += representation == GENERATIVEQC_BASIS_CARTESIAN
                              ? (shell.angular_momentum + 1) * (shell.angular_momentum + 2) / 2
                              : 2 * shell.angular_momentum + 1;
        require(capacity.shell_count == expanded.shells.size() &&
                    capacity.primitive_count == expanded.primitives.size() &&
                    capacity.ao_count == expected_aos && capacity.ecp_term_count == 0 &&
                    capacity.buffer_bytes ==
                        expanded.shells.size() * sizeof(generativeqc_shell) +
                            expanded.primitives.size() * sizeof(generativeqc_primitive),
                "named capacity must agree with the shared general-template expansion");
      }
    }
  }
}

void test_constructor() {
  generativeqc_context_descriptor context_descriptor{};
  context_descriptor.struct_size = sizeof(context_descriptor);
  context_descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  context_descriptor.backend = GENERATIVEQC_BACKEND_CPU_REFERENCE;
  generativeqc::Context context(context_descriptor);
  std::array<generativeqc_atom, 2> atoms{{{1, 0, 0, -0.7}, {1, 0, 0, 0.7}}};
  generativeqc_system_descriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
  descriptor.atoms = atoms.data();
  descriptor.atom_count = atoms.size();
  descriptor.multiplicity = 1;
  descriptor.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  const char* digest = generativeqc_bundled_basis_catalog_sha256();
  require(std::string_view(digest).size() == 64, "catalog checksum must be SHA-256");

  generativeqc_system* raw = nullptr;
  require(generativeqc_system_create_bundled(context.get(), &descriptor, "STO_3G", digest, &raw) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "bundled C constructor failed");
  std::unique_ptr<generativeqc_system, decltype(&generativeqc_system_destroy)> system(
      raw, generativeqc_system_destroy);
  const std::array<generativeqc_primitive, 6> primitives{{{3.425250914, 0.1543289673},
                                                          {0.6239137298, 0.5353281423},
                                                          {0.1688554040, 0.4446345422},
                                                          {3.425250914, 0.1543289673},
                                                          {0.6239137298, 0.5353281423},
                                                          {0.1688554040, 0.4446345422}}};
  const std::array<generativeqc_shell, 2> shells{{{0, 0, 0, 3}, {1, 0, 3, 3}}};
  auto explicit_descriptor = descriptor;
  explicit_descriptor.primitives = primitives.data();
  explicit_descriptor.primitive_count = primitives.size();
  explicit_descriptor.shells = shells.data();
  explicit_descriptor.shell_count = shells.size();
  generativeqc::System explicit_system(context, explicit_descriptor);
  require(system->data.shells.size() == explicit_system.get()->data.shells.size(),
          "shell counts differ from independent H STO-3G table");
  for (std::size_t shell_index = 0; shell_index < system->data.shells.size(); ++shell_index) {
    const auto& actual = system->data.shells[shell_index];
    const auto& expected = explicit_system.get()->data.shells[shell_index];
    require(actual.atom_index == expected.atom_index &&
                actual.angular_momentum == expected.angular_momentum &&
                actual.primitives.size() == expected.primitives.size(),
            "normalized shell metadata differs");
    for (std::size_t index = 0; index < actual.primitives.size(); ++index)
      require(actual.primitives[index].exponent == expected.primitives[index].exponent &&
                  actual.primitives[index].coefficient == expected.primitives[index].coefficient,
              "normalized primitives differ from independent table");
  }
  generativeqc::System cpp_system(context, atoms, "sto-3g");
  atoms[0].z = -10;
  require(cpp_system.get()->data.atoms[0].position[2] == -0.7,
          "C++ bundled system must copy input coordinates");
  require(system->data.atoms[0].position[2] == -0.7, "C bundled system must copy atoms");

  for (const char* name : {"unknown", ""}) {
    raw = system.get();
    require(generativeqc_system_create_bundled(context.get(), &descriptor, name, nullptr, &raw) ==
                    GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                raw == nullptr,
            "unknown basis must reject and clear output");
  }
  raw = system.get();
  require(generativeqc_system_create_bundled(context.get(), &descriptor, "sto-3g", "bad", &raw) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              raw == nullptr,
          "mismatched provenance must reject and clear output");
  require(std::string_view(generativeqc_context_get_last_detail(context.get())).find("checksum") !=
              std::string_view::npos,
          "catalog mismatch must explain failed provenance");
  require(generativeqc_system_create_bundled(context.get(), &explicit_descriptor, "sto-3g", nullptr,
                                             &raw) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "constructor must not discard caller basis buffers");
  descriptor.struct_size -= 1;
  require(generativeqc_system_create_bundled(context.get(), &descriptor, "sto-3g", nullptr, &raw) ==
              GENERATIVEQC_STATUS_ABI_MISMATCH,
          "new constructor requires a complete descriptor");
  descriptor.struct_size = sizeof(descriptor);
  descriptor.basis_representation = 99;
  require(generativeqc_system_create_bundled(context.get(), &descriptor, "sto-3g", nullptr, &raw) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "invalid representation must reject");
  descriptor.basis_representation = GENERATIVEQC_BASIS_CARTESIAN;
  atoms[0].atomic_number = 26;
  require(generativeqc_system_create_bundled(context.get(), &descriptor, "sto-3g", nullptr, &raw) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "missing element must reject without partial system");
  require(raw == nullptr, "failed system must not publish a handle");
  require(generativeqc_system_create_bundled(nullptr, &descriptor, "sto-3g", nullptr, &raw) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "null context must reject");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "dump") {
      dump_catalog();
    } else {
      test_capacity_catalog();
      test_constructor();
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
