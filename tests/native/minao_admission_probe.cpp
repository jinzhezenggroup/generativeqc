// Source-linked mathematical/allocator probe, not a target SCF or GPU test.
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>

#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "scf/initial_guess/minao.hpp"
#include "scf/preliminary_guess.hpp"
#include "scf/solver/proposal_control.hpp"

namespace {
bool track = false;
std::size_t active = 0, peak = 0;
struct alignas(std::max_align_t) Header {
  std::size_t bytes;
  bool counted;
};
}  // namespace
void* operator new(std::size_t size) {
  auto* h = static_cast<Header*>(std::malloc(size + sizeof(Header)));
  if (!h) throw std::bad_alloc();
  h->bytes = size;
  h->counted = track;
  if (track) {
    active += size;
    peak = std::max(peak, active);
  }
  return h + 1;
}
void operator delete(void* pointer) noexcept {
  if (!pointer) return;
  auto* h = static_cast<Header*>(pointer) - 1;
  if (h->counted) active -= h->bytes;
  std::free(h);
}
void operator delete(void* pointer, std::size_t) noexcept { operator delete(pointer); }

int main(int argc, char**) {
  using namespace generativeqc;
  using namespace scf::initial_guess;
  std::size_t n = 0, atoms = 0;
  core::System system;
  if (!(std::cin >> n >> system.electron_count >> atoms)) return 3;
  for (std::size_t i = 0; i < atoms; ++i) {
    int z;
    std::cin >> z;
    system.atoms.push_back({z, {0, 0, 0}, 0});
  }
  // n synthetic s shells reproduce the resource extent without an integral owner.
  system.shells.resize(n);
  for (auto& shell : system.shells) shell.primitives.push_back({1.0, 1.0});
  PreliminaryOptions options;
  options.kind = PreliminaryKind::Minao;
  const auto capacity = preliminary_numeric_capacity(system, options);
  if (argc > 1) {
    std::cout << capacity << '\n';
    return 0;
  }
  integrals::IntegralData ints;
  ints.nbf = n;
  ints.overlap.resize(n * n);
  for (double& value : ints.overlap) std::cin >> value;
  try {
    track = true;
    scf::reference::Matrix raw(n * n);
    for (double& value : raw) std::cin >> value;
    const auto x = scf::reference::symmetric_orthogonalizer(ints.overlap, n);
    const auto admitted = admissible_minao_density(system, ints, x, raw);
    scf::solver::validate_seed(ints.overlap, admitted, n,
                               {static_cast<unsigned>(system.electron_count)}, 2.0);
    track = false;
    std::cout << peak << ' ' << capacity << '\n' << std::setprecision(17);
    for (double value : admitted) std::cout << value << ' ';
    std::cout << '\n';
  } catch (const std::exception& error) {
    track = false;
    std::cerr << error.what() << '\n';
    return 2;
  }
}
