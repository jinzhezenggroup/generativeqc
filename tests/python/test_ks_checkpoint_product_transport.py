"""Execute checkpoint admission routing and atomicity, with scientific doubles."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cpp_source_support import cpp_function_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("cuda", [0, 1])
def test_checkpoint_borrows_products_before_commit(
    tmp_path: Path, required_native_cxx: NativeCxx, cuda: int
) -> None:
    """Both spins route optional products and reject before publishing any seed."""
    validator = cpp_function_definition(
        (ROOT / "src/scf/rhf.cpp").read_text(), "validate_hf_warm_density"
    )
    restore = cpp_function_definition(
        (ROOT / "src/methods/dft_method.cpp").read_text(), "restore_warm_states"
    ).replace(" override", "")
    source = tmp_path / "checkpoint.cpp"
    source.write_text(
        PREFIX
        + "\nnamespace scf {\n"
        + validator
        + "\n}\n"
        + OWNER
        + restore
        + "\n};\n"
        + MAIN
    )
    binary = required_native_cxx.build_executable(
        [source],
        tmp_path / "checkpoint",
        compile_args=["-std=c++20", f"-DGENERATIVEQC_HAS_CUDA={cuda}"],
    )
    subprocess.run([binary], check=True, timeout=10)


PREFIX = r"""
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
using Matrix = std::vector<double>;
enum generativeqc_method { GENERATIVEQC_METHOD_RHF, GENERATIVEQC_METHOD_UHF,
  GENERATIVEQC_METHOD_LDA_UKS, GENERATIVEQC_METHOD_PBE_UKS };
namespace core { struct System { int electron_count{2}, multiplicity{1}; double position{}; }; }
namespace posthf {
struct RawSource {
  explicit RawSource(const core::System&) {}
  size_t nbf() const { return 2; }
  enum class Operator { overlap };
  void read(Operator, std::initializer_list<int>, std::initializer_list<size_t>, double* out, size_t n) {
    assert(n == 4); out[0] = out[3] = 1; out[1] = out[2] = 0;
  }
};
}
int products_used{}, eigens_used{}, validators{};
bool use_products{}, refuse_products{};
namespace scf {
namespace initial_guess {
using EigenOperation = std::function<void()>;
using MatrixProductOperation = std::function<void()>;
}
struct HfWarmState {
  Matrix density{2, 0, 0, 0};
  std::vector<double> coordinates{0};
  int iterations{};
  double energy{}, energy_change{}, density_rms{};
};
std::pair<int, int> spin_occupations(const core::System&) { return {1, 1}; }
void validate_seed(const Matrix&, const Matrix& density, size_t n,
    const std::vector<unsigned>& electrons, double weight,
    const initial_guess::EigenOperation& eigen,
    const initial_guess::MatrixProductOperation& products = {}) {
  ++validators;
  assert(n == 2 && ((electrons.size() == 1 && weight == 2) ||
                   (electrons.size() == 2 && weight == 1)));
  if (eigen) eigen();
  if (products) products();
  // A scientific refusal after product evaluation must also be atomic.
  if (density[0] < 0) throw std::invalid_argument("invalid density");
}
}
namespace runtime { struct DeviceCapacityScope { explicit DeviceCapacityScope(void*) {} }; }
namespace geometry {
bool valid_coordinates(const std::vector<double>& p, const core::System&) { return p.size() == 1; }
void set_coordinates(core::System& system, const std::vector<double>& p) { system.position = p[0]; }
}
bool ks_is_unrestricted(bool flag) { return flag; }
struct Cuda {
  scf::initial_guess::EigenOperation seed_eigen_operation() { return [] { ++eigens_used; }; }
  scf::initial_guess::MatrixProductOperation seed_product_operation() {
    if (!use_products) return {};
    return [] { ++products_used; if (refuse_products) throw std::runtime_error("product failure"); };
  }
};
struct Plan {
  Cuda cuda;
  bool cleared{};
  Cuda* cuda_plan() { return &cuda; }
  void clear_warm_start() { cleared = true; }
};
"""

OWNER = r"""
struct Batch {
  struct Item {
    std::unique_ptr<Plan> plan{std::make_unique<Plan>()};
    std::optional<scf::HfWarmState> warm{scf::HfWarmState{}};
    bool resident_warm{true};
  };
  std::unique_ptr<int> capacity_;
  bool warm_enabled_{true}, execution_plan_{};
  std::vector<core::System> systems_{2};
  std::vector<Item> items_{2};
  size_t size() const { return 2; }
  size_t warm_density_size(size_t) const { return 4; }
"""

MAIN = r"""
int main() {
  for (bool unrestricted : {false, true})
    for (bool available : {false, true}) {
      Batch batch; batch.execution_plan_ = unrestricted;
      use_products = available; refuse_products = false;
      products_used = eigens_used = validators = 0;
      scf::HfWarmState valid; valid.energy = 42;
      batch.restore_warm_states({valid, std::nullopt});
      assert(batch.items_[0].warm->energy == 42 && !batch.items_[0].resident_warm);
      assert(batch.items_[0].plan->cleared);
      assert(batch.items_[1].warm->energy == 0 && batch.items_[1].resident_warm);
      assert(!batch.items_[1].plan->cleared);
      assert(validators > 0);
      assert((products_used > 0) == (available && GENERATIVEQC_HAS_CUDA));
      assert((eigens_used > 0) == bool(GENERATIVEQC_HAS_CUDA));
      Batch atomic; atomic.execution_plan_ = unrestricted;
      auto invalid = valid; invalid.density[0] = -1;
      bool refused = false;
      try { atomic.restore_warm_states({valid, invalid}); }
      catch (const std::invalid_argument&) { refused = true; }
      assert(refused);
      for (auto& item : atomic.items_) {
        assert(item.warm->energy == 0 && item.resident_warm && !item.plan->cleared);
      }
      if (available && GENERATIVEQC_HAS_CUDA) {
        refuse_products = true; refused = false;
        try { atomic.restore_warm_states({valid, std::nullopt}); }
        catch (const std::runtime_error&) { refused = true; }
        assert(refused);
        for (auto& item : atomic.items_) assert(item.warm->energy == 0 && !item.plan->cleared);
      }
    }
}
"""
