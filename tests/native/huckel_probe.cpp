#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/initial_guess/huckel.hpp"
#include "scf/preliminary_guess.hpp"
#include "scf/solver/cpu_target_eigen.hpp"
#include "scf/solver/proposal_control.hpp"

namespace allocation {
// Observe simultaneous new/delete payloads, including the shared library.
// Headers and the retained target integrals are outside the preliminary cap.
// A block keeps its creation epoch so destruction after observation is safe.
struct alignas(std::max_align_t) Header {
  void* original;
  std::size_t bytes;
  bool counted;
};
bool enabled = false;
std::size_t live = 0, peak = 0;
void* allocate(std::size_t bytes, std::size_t alignment) {
  alignment = std::max(alignment, alignof(Header));
  const auto overhead = sizeof(Header) + alignment - 1;
  if (bytes > std::numeric_limits<std::size_t>::max() - overhead) throw std::bad_alloc();
  auto* original = std::malloc(bytes + overhead);
  if (!original) throw std::bad_alloc();
  auto address = reinterpret_cast<std::uintptr_t>(original) + sizeof(Header);
  address = (address + alignment - 1) & ~(alignment - 1);
  auto* header = reinterpret_cast<Header*>(address) - 1;
  *header = {original, bytes, enabled};
  if (enabled) {
    live += bytes;
    peak = std::max(peak, live);
  }
  return reinterpret_cast<void*>(address);
}
void release(void* pointer) noexcept {
  if (!pointer) return;
  auto* header = static_cast<Header*>(pointer) - 1;
  if (header->counted) live -= header->bytes;
  std::free(header->original);
}
}  // namespace allocation
void* operator new(std::size_t bytes) {
  return allocation::allocate(bytes, alignof(std::max_align_t));
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void* operator new(std::size_t bytes, std::align_val_t alignment) {
  return allocation::allocate(bytes, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t bytes, std::align_val_t alignment) {
  return ::operator new(bytes, alignment);
}
void operator delete(void* pointer) noexcept { allocation::release(pointer); }
void operator delete[](void* pointer) noexcept { allocation::release(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { allocation::release(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { allocation::release(pointer); }
void operator delete(void* pointer, std::align_val_t) noexcept { allocation::release(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { allocation::release(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
  allocation::release(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  allocation::release(pointer);
}

/** CPU-only source-linked oracle probe. Input is raw contracted shell data;
 * the production basis validator supplies exactly the target normalization.
 * Identity overlap is a separate allocation test, never a molecular oracle. */
int main(int argc, char** argv) {
  using namespace generativeqc;
  using namespace scf::initial_guess;
  try {
    bool identity_overlap = false, native_eigen = false, summary_only = false;
    for (int index = 1; index < argc; ++index) {
      const std::string_view option = argv[index];
      if (option == "--identity-overlap")
        identity_overlap = true;
      else if (option == "--native-eigen")
        native_eigen = true;
      else if (option == "--summary-only")
        summary_only = true;
      else
        throw std::invalid_argument("unknown Huckel probe option");
    }
    core::System system;
    std::size_t atoms{}, shells{};
    int spherical{};
    if (!(std::cin >> atoms >> shells >> system.charge >> spherical)) return 3;
    system.basis_representation =
        spherical ? GENERATIVEQC_BASIS_SPHERICAL : GENERATIVEQC_BASIS_CARTESIAN;
    for (std::size_t atom = 0; atom < atoms; ++atom) {
      core::Atom item;
      std::cin >> item.atomic_number >> item.position[0] >> item.position[1] >> item.position[2];
      system.atoms.push_back(item);
    }
    for (std::size_t shell = 0; shell < shells; ++shell) {
      core::Shell item;
      std::size_t primitives{};
      std::cin >> item.atom_index >> item.angular_momentum >> primitives;
      item.primitives.resize(primitives);
      for (auto& primitive : item.primitives)
        std::cin >> primitive.exponent >> primitive.coefficient;
      system.shells.push_back(std::move(item));
    }
    std::string detail;
    if (!std::cin ||
        molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
      throw std::invalid_argument(detail.empty() ? "invalid input" : detail);
    integrals::IntegralData integrals;
    if (identity_overlap) {
      integrals.nbf = molecule::ao_count(system);
      integrals.overlap = scf::reference::identity(integrals.nbf);
    } else {
      integrals = integrals::build_integrals(system, false, false);
    }
    PreliminaryOptions policy;
    policy.kind = PreliminaryKind::Huckel;
    PreliminaryDiagnostic diagnostic;
    const auto capacity = preliminary_numeric_capacity(system, policy);
    allocation::enabled = true;
    const auto result = huckel_density(system, integrals, policy, diagnostic);
    scf::solver::validate_seed(
        integrals.overlap, result.density, integrals.nbf,
        {static_cast<unsigned>(system.electron_count)}, 2.0,
        native_eigen ? EigenOperation(scf::solver::cpu_target_eigen) : EigenOperation{});
    allocation::enabled = false;
    if (!allocation::peak || allocation::peak > capacity)
      throw std::runtime_error("Huckel preparation exceeds its preflight numeric allowance");
    std::cout << integrals.nbf << ' ' << result.atomic_solves << ' '
              << result.occupied_atomic_orbitals << ' ' << diagnostic.preliminary_iterations << ' '
              << diagnostic.preliminary_fock_builds << ' ' << capacity << ' ' << allocation::peak
              << '\n'
              << std::setprecision(17);
    if (summary_only) return 0;
    for (double value : result.density) std::cout << value << ' ';
    std::cout << '\n';
  } catch (const std::exception& error) {
    allocation::enabled = false;
    std::cerr << error.what() << '\n';
    return 2;
  }
}
