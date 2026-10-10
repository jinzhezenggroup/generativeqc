"""Compile selector and actual endpoint-dispatch fragments with launch recorders."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _block(source: str, signature: str) -> str:
    start = source.index(signature)
    opened = source.index("{", start)
    depth = 0
    for index in range(opened, len(source)):
        depth += (source[index] == "{") - (source[index] == "}")
        if depth == 0:
            return source[start : index + 1]
    raise AssertionError("unclosed source block")


def _branch(source: str, condition: str) -> str:
    first = _block(source, condition)
    remaining = source[source.index(first) + len(first) :]
    assert remaining.lstrip().startswith("else {")
    return first + " " + _block(remaining, "else {")


def test_selector_matrix_and_reachable_endpoint_dispatch(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    diis = (ROOT / "src/scf/cuda/scf_diis_kernels.cu").read_text()
    hf = (ROOT / "src/scf/cuda_rhf.cpp").read_text()
    ks = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    selectors = "\n".join(
        _block(diis, "bool " + name + "()")
        for name in (
            "ordered_incremental_diis_gram_requested",
            "incremental_diis_gram_requested",
        )
    )
    hf_dispatch = _branch(hf, "if (plan.ordered_diis_gram) {")
    ks_dispatch = _branch(ks, "if (ordered_diis_gram) {")
    ks_second = ks[ks.index("if (ordered_diis_gram) {") + len(ks_dispatch) :]
    ks_iteration_dispatch = _branch(ks_second, "if (ordered_diis_gram) {")
    source = tmp_path / "endpoint_dispatch.cpp"
    prefix = r"""
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
unsigned ordered_calls{}, pending_calls{}, legacy_calls{};
constexpr int cudaSuccess = 0;
int cuda_status(int status) { return status; }
void check(int status) { if (status) throw std::runtime_error("launch failure"); }
template <class... Args> void launch_update_diis_cached_gram(Args...) { ++ordered_calls; }
template <class... Args> int launch_diis_pending_gram(Args...) { ++pending_calls; return 0; }
template <class... Args> void launch_update_diis_kernel(Args...) { ++legacy_calls; }
"""
    bindings = r"""
  struct { bool ordered_diis_gram, incremental_diis_gram; } plan{ordered, incremental};
  const bool ordered_diis_gram = ordered, incremental_diis_gram = incremental;
  struct { void* stream_ = nullptr; } resources;
  void* stream = nullptr;
  const int batch_size=2, matrix_reduction_threads=32, nbf=3, n=3, spins=1;
  const bool unrestricted=false;
  const unsigned diis_history=4, history=4;
  double value{};
  double *fock=&value, *residual=&value, *fock_history=&value, *residual_history=&value,
         *diis_linear_system=&value, *diis_coefficients=&value, *eigensystem=&value,
         *gram=&value, *weights=&value, *effective=&value;
  double *diis_raw_gram=incremental?&value:nullptr, *raw_gram=diis_raw_gram;
  unsigned count{}, head{};
  unsigned *diis_count=&count, *diis_head=&head, *history_count=&count, *history_head=&head;
  unsigned char active_value=1;
  unsigned char *active=&active_value, *enabled=&active_value;
"""
    bodies = ""
    for name, body in (
        ("hf_endpoint", hf_dispatch),
        ("ks_chunk", ks_dispatch),
        ("ks_iteration", ks_iteration_dispatch),
    ):
        bodies += (
            f"int {name}(bool incremental, bool ordered) {{\n"
            + bindings
            + body
            + "\nreturn 0;\n}\n"
        )
    source.write_text(
        prefix
        + selectors
        + bodies
        + r"""
int main() {
  const char* enable = "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM";
  const char* reduction = "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION";
  for (const char* flag : {static_cast<const char*>(nullptr), "0", "1", "invalid"})
    for (const char* reducer : {static_cast<const char*>(nullptr), "cooperative", "ordered", "", "invalid", "1"})
      for (unsigned history : {0U, 1U, 2U, 64U}) {
        if (flag) setenv(enable, flag, 1); else unsetenv(enable);
        if (reducer) setenv(reduction, reducer, 1); else unsetenv(reduction);
        const bool admitted = history >= 2;
        const bool enabled = flag && std::strcmp(flag, "1") == 0;
        const bool valid_flag = !flag || std::strcmp(flag, "0") == 0 || enabled;
        const bool ordered = reducer && std::strcmp(reducer, "ordered") == 0;
        const bool valid_reducer = !reducer || std::strcmp(reducer, "cooperative") == 0 || ordered;
        const bool should_reject = admitted && (!valid_flag || (enabled && !valid_reducer));
        bool threw = false, actual_incremental = false, actual_ordered = false;
        try {
          actual_incremental = admitted && incremental_diis_gram_requested();
          actual_ordered = actual_incremental && ordered_incremental_diis_gram_requested();
        } catch (const std::invalid_argument&) { threw = true; }
        if (threw != should_reject) throw std::runtime_error("selector rejection changed");
        if (threw) continue;
        if (actual_incremental != (admitted && enabled) ||
            actual_ordered != (admitted && enabled && ordered))
          throw std::runtime_error("selector chose wrong route");
        ordered_calls = pending_calls = legacy_calls = 0;
        hf_endpoint(actual_incremental, actual_ordered);
        ks_chunk(actual_incremental, actual_ordered);
        ks_iteration(actual_incremental, actual_ordered);
        if (ordered_calls != (actual_ordered ? 3U : 0U) ||
            pending_calls != (actual_incremental && !actual_ordered ? 3U : 0U) ||
            legacy_calls != (actual_ordered ? 0U : 3U))
          throw std::runtime_error("endpoint dispatch changed or doubled Gram work");
      }
}
"""
    )
    executable = tmp_path / "endpoint_dispatch"
    required_native_cxx.build_executable(
        [source], executable, compile_args=("-std=c++20", "-O2")
    )
    subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )


def test_reducer_is_owned_by_plan_and_invalidates_hf_replay() -> None:
    bucket = (ROOT / "src/scf/cuda/rhf_bucket.cpp").read_text()
    driver = (ROOT / "src/scf/cuda_rhf.cpp").read_text()
    ks = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    recreate = bucket.index("    delete *plan;")
    for field in ("incremental_diis_gram", "ordered_diis_gram"):
        assert bucket.index(f"(*plan)->{field} != {field} ||") < recreate
        assert f"plan.{field} != requested_{field} ||" in driver
        assert f"plan.{field} = requested_{field};" in driver
    assert ks.count("ordered_incremental_diis_gram_requested()") == 1
    assert ks.count("if (ordered_diis_gram) {") == 2
    assert ks.index("if (final_closure) {") < ks.rindex("if (ordered_diis_gram) {")
    # The selector is a source-reduction choice, not a second arena reservation.
    arena = (ROOT / "src/scf/cuda/arena.cpp").read_text()
    assert "diis_gram_cache" not in arena
    assert "made.diis_raw_gram" in arena
