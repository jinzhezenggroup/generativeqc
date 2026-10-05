#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "hf/rhf_frame_recycle.hpp"

namespace {
using namespace generativeqc;
void require(bool ok, const char* detail) {
  if (!ok) throw std::runtime_error(detail);
}
void test() {
  core::System system;
  system.electron_count = 2;
  hf::PhysicalReference ref;
  ref.nocc = 1;
  ref.nbf = 4;
  ref.orbital_energies = {-1., .1, .2, .3};
  hf::RHFFrameResponseRecycle recycle;
  const auto bytes = hf::RHFFrameIdentity::required_storage_bytes(system, ref) + 9 * sizeof(double);
  require(!recycle.prepare(system, ref, 0, true, "hash", bytes - 1),
          "short recycle budget admitted");
  require(recycle.prepare(system, ref, 0, true, "hash", bytes), "exact recycle budget refused");
  require(recycle.storage_bytes() == bytes, "recycle projection scratch not counted");
  const std::array<double, 3> x{1., 2., 3.}, ax{2., 6., 12.};
  require(recycle.initial_guess(ax).empty(), "unpopulated recycle proposed data");
  require(recycle.capture(x, ax), "finite solved direction refused");
  const std::array<double, 3> rhs{4., 12., 24.};
  const auto guess = recycle.initial_guess(rhs);
  require(guess.size() == 3, "compatible subspace not reused");
  for (std::size_t k = 0; k < 3; ++k)
    require(std::abs(guess[k] - 2 * x[k]) < 2e-15, "projection oracle mismatch");
  response::GmresOptions options;
  options.absolute_tolerance = 1e-12;
  options.relative_tolerance = 0.;
  auto action = [](auto a, auto b) {
    for (std::size_t k = 0; k < 3; ++k) b[k] = (k + 2) * a[k];
  };
  const auto solved =
      response::solve_gmres(response::prepare_gmres(3, options), action, rhs, guess);
  require(solved.converged() && solved.operator_actions == 1 && solved.iterations == 0,
          "recycled solution bypassed true residual or did not reduce actions");
  const std::array<double, 3> unrelated{2., -1., 5.};
  const auto other = response::solve_gmres(response::prepare_gmres(3, options), action, unrelated,
                                           recycle.initial_guess(unrelated));
  require(other.converged(), "recycled different RHS did not converge");
  for (std::size_t k = 0; k < 3; ++k)
    require(std::abs((k + 2) * other.solution[k] - unrelated[k]) < 1e-12,
            "different RHS exact residual gate");
  for (int variant = 0; variant < 6; ++variant) {
    auto changed = ref;
    auto molecule = system;
    int device = 0;
    bool matrix = true;
    const char* hash = "hash";
    if (variant == 0) changed.orbital_energies[1] = std::nextafter(.1, 1.);
    if (variant == 1) changed.energy = std::nextafter(0., 1.);
    if (variant == 2) molecule.charge = 1;
    if (variant == 3) device = 1;
    if (variant == 4) matrix = false;
    if (variant == 5) hash = "other";
    require(!recycle.matches(molecule, changed, device, matrix, hash), "stale operator admitted");
  }
  auto changed = ref;
  changed.energy = 1.;
  require(recycle.prepare(system, changed, 0, true, "hash", bytes) && !recycle.populated(),
          "stale subspace survived identity replacement");
  require(recycle.initial_guess(rhs).empty(), "stale proposal survived");
  require(!recycle.prepare(system, changed, 0, true, "hash", 0) && recycle.storage_bytes() == 0,
          "budget refusal retained optional payload");
  require(recycle.prepare(system, ref, 0, true, "hash", bytes), "recovery after refusal failed");
  auto bad = ax;
  bad[0] = std::numeric_limits<double>::quiet_NaN();
  require(!recycle.capture(x, bad) && !recycle.populated(), "nonfinite exact image published");
  bad.fill(std::numeric_limits<double>::max());
  require(!recycle.capture(x, bad) && !recycle.populated(), "overflowing image norm published");
}
}  // namespace
int main() {
  try {
    test();
    std::cout << "Strict response recycling and independent residual gates passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
