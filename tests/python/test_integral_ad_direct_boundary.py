"""Run the actual direct-provider C boundary with a counted SCF constructor."""

import subprocess
from pathlib import Path

import pytest
from _cpp_source_support import cpp_function_definition
from test_coulomb_optional_allocation import compile_cached_probe

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def direct_boundary_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = (ROOT / "src/api/c_api_integral_ad.cpp").read_text()
    definitions = "template <class Function>\n" + "\n".join(
        cpp_function_definition(source, name)
        for name in (
            "integral_ad_status",
            "generativeqc_integral_ad_direct_prepare_v1",
        )
    )
    directory = tmp_path_factory.mktemp("integral-direct-boundary")
    cpp, binary = directory / "probe.cpp", directory / "probe"
    cpp.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#define GENERATIVEQC_API
struct generativeqc_system {};
int constructions{};
bool fail_construction{};
struct IntegralHFState {
  using SourceFock = int (*)(const double*, const double*, const double*, const double*, double*);
  template<class... Args> explicit IntegralHFState(Args&&...) {
    ++constructions;
    if (fail_construction) throw std::runtime_error("injected SCF failure");
  }
  void publish(double* state, double* receipt) const {
    std::fill_n(state, 16, 3.0);
    std::fill_n(receipt, 8, 4.0);
  }
};
int valid_fock(const double*, const double*, const double*, const double*, double*) { return 0; }
"""
        + definitions
        + r"""
int main(int argc, char** argv) {
  assert(argc == 2);
  const int scenario = std::atoi(argv[1]);
  generativeqc_system system;
  // Direct n=2 storage contains S,h,Enuc only, not a dense 16-element ERI.
  const std::array<double, 9> values{};
  const std::array<double, 8> parameters{};
  std::array<double, 16> state;
  std::array<double, 8> receipt;
  state.fill(71.0);
  receipt.fill(83.0);
  void* const sentinel = reinterpret_cast<void*>(0x1234);
  void* owner = sentinel;
  char detail[128]{};
  fail_construction = scenario == 2;
  const int status = generativeqc_integral_ad_direct_prepare_v1(
      &system, values.data(), 2, 0, 0, 0, 10, 2, 1e-8, 1e-8, 1000000,
      scenario == 0 ? nullptr : valid_fock, parameters.data(), parameters.size(), 10,
      &owner, state.data(), receipt.data(), detail, sizeof(detail));
  if (scenario == 1) {
    assert(status == 0 && constructions == 1);
    assert(owner != sentinel && owner != nullptr);
    assert(std::all_of(state.begin(), state.end(), [](double v) { return v == 3.0; }));
    assert(std::all_of(receipt.begin(), receipt.end(), [](double v) { return v == 4.0; }));
    delete static_cast<IntegralHFState*>(owner);
  } else {
    assert(status != 0);
    assert(constructions == (scenario == 0 ? 0 : 1));
    assert(owner == sentinel);
    assert(std::all_of(state.begin(), state.end(), [](double v) { return v == 71.0; }));
    assert(std::all_of(receipt.begin(), receipt.end(), [](double v) { return v == 83.0; }));
    assert(std::string(detail).find(scenario == 0 ? "direct integral AD provider" : "injected SCF")
           != std::string::npos);
  }
}
"""
    )
    compile_cached_probe(cpp, binary)
    return binary


@pytest.mark.parametrize("scenario", [0, 1, 2], ids=["null-provider", "valid-provider", "scf-failure"])
def test_direct_prepare_provider_boundary(direct_boundary_probe: Path, scenario: int) -> None:
    subprocess.run([str(direct_boundary_probe), str(scenario)], check=True, timeout=10)
