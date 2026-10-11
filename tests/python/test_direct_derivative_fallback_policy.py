"""Host execution of the real derivative-owner policy and preparation join."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for position in range(opening, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[start : position + 1]
    raise AssertionError(f"unterminated function: {signature}")


def test_derivative_ownership_does_not_admit_uncovered_scf_values(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Exercise actual joins; stubs model capability/budget, never integral science."""
    direct = (ROOT / "src/scf/cuda/direct_jk.cpp").read_text()
    header = (ROOT / "src/scf/cuda/direct_jk_plan.hpp").read_text()
    policy = _function(direct, "bool direct_shell_derivative_fallback_requested(")
    start = direct.index(
        "    if (!resident_source_only && !through_f && budget > plan->device_bytes)"
    )
    stop = direct.index("    std::size_t span_host_preparation_bytes", start)
    preparation = direct[start:stop]
    bounded_value = _function(header, "inline bool direct_jk_bounded_value_enabled(")
    full_value = _function(
        header, "inline bool direct_jk_generated_full_range_value_available("
    )
    source = tmp_path / "policy.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
struct Shared { bool value_capability{}; };
struct Exchange {
  std::unique_ptr<Shared> shared;
  bool force_capability{}, bounded_value_capability{};
  std::size_t device_bytes{8};
};
struct Coulomb { std::size_t device_bytes{4}; };
struct CudaDirectJkPlan {
  std::unique_ptr<Exchange> generated_exchange;
  std::unique_ptr<Coulomb> generated_coulomb;
  std::size_t device_bytes{8};
  bool bounded_value_opt_in{false};
  int batch{}, stream{};
};
bool value_coverage{}, insufficient_optional_budget{};
unsigned exchange_attempts{}, coulomb_attempts{};
std::unique_ptr<Exchange> prepare_generated_exchange(
    int, int, int, int, double, std::size_t allowance,
    bool derivative, bool allow_uncovered) {
  ++exchange_attempts;
  if ((!value_coverage && !allow_uncovered) || allowance < 8 || insufficient_optional_budget)
    return {};
  auto owner = std::make_unique<Exchange>();
  owner->shared = std::make_unique<Shared>();
  owner->shared->value_capability = value_coverage;
  owner->force_capability = derivative;
  owner->bounded_value_capability = allow_uncovered && !value_coverage;
  return owner;
}
std::unique_ptr<Coulomb> prepare_generated_coulomb(int,int,int,int,double,std::size_t allowance) {
  ++coulomb_attempts;
  return value_coverage && allowance >= 4 ? std::make_unique<Coulomb>() : nullptr;
}
"""
        + policy
        + bounded_value
        + full_value
        + r"""
void prepare(CudaDirectJkPlan* plan, bool resident_source_only, bool through_f,
             unsigned derivative_order, std::size_t& budget) {
  const int host = 0, device_id = 0;
  const double screening_tolerance = 1e-12;
"""
        + preparation
        + r"""
}
int main() {
  unsetenv("GENERATIVEQC_DIRECT_SHELL_DERIVATIVE_FALLBACK");
  assert(!direct_shell_derivative_fallback_requested(1));
  for (const char* selection : {"0", "1"}) {
    setenv("GENERATIVEQC_DIRECT_SHELL_DERIVATIVE_FALLBACK", selection, 1);
    for (bool coverage : {false, true})
      for (unsigned derivative_order : {0U, 1U})
        for (bool resident_source_only : {false, true})
          for (bool through_f : {false, true})
            for (bool limited : {false, true}) {
              value_coverage = coverage;
              insufficient_optional_budget = limited;
              CudaDirectJkPlan plan;
              std::size_t budget = 64;
              exchange_attempts = coulomb_attempts = 0;
              prepare(&plan, resident_source_only, through_f, derivative_order, budget);
              const bool attempt = !resident_source_only && !through_f;
              const bool requested = derivative_order && selection[0] == '1';
              const bool retained = attempt && !limited && (coverage || requested);
              assert(exchange_attempts == unsigned(attempt));
              assert(bool(plan.generated_exchange) == retained);
              assert(direct_jk_generated_full_range_value_available(plan) == (retained && coverage));
              assert(!direct_jk_bounded_value_enabled(plan));
              if (retained) {
                assert(plan.generated_exchange->force_capability == bool(derivative_order));
                assert(plan.generated_exchange->bounded_value_capability == (requested && !coverage));
                assert(budget == 56);
              } else {
                assert(coulomb_attempts == unsigned(attempt));
                assert(budget == (attempt && coverage ? 60 : 64));
              }
            }
  }
  setenv("GENERATIVEQC_DIRECT_SHELL_DERIVATIVE_FALLBACK", "1", 1);
  value_coverage = false;
  insufficient_optional_budget = false;
  CudaDirectJkPlan plan;
  std::size_t no_optional_room = plan.device_bytes;
  exchange_attempts = 0;
  prepare(&plan, false, false, 1, no_optional_room);
  assert(exchange_attempts == 0 && !plan.generated_exchange);
  std::size_t budget = 64;
  prepare(&plan, false, false, 1, budget);
  assert(plan.generated_exchange && plan.generated_exchange->force_capability);
  assert(!direct_jk_generated_full_range_value_available(plan));
  plan.bounded_value_opt_in = true;
  assert(direct_jk_generated_full_range_value_available(plan));
  setenv("GENERATIVEQC_DIRECT_SHELL_DERIVATIVE_FALLBACK", "yes", 1);
  assert(!direct_shell_derivative_fallback_requested(0));
  try { direct_shell_derivative_fallback_requested(1); assert(false); }
  catch (const std::invalid_argument&) {}
}
""",
        encoding="utf-8",
    )
    binary = native_cxx.build_executable(
        [source], tmp_path / "policy", compile_args=("-std=c++17",)
    )
    subprocess.run([str(binary)], check=True, timeout=30)
