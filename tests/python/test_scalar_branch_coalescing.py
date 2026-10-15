"""Executable lazy-branch and IEEE contracts for multi-output scalar lowering."""

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_coalesced_regions_preserve_lazy_values_and_partials(
    tmp_path: Path, native_cxx: "NativeCxx"
) -> None:
    """Unknown predicate dependencies and inactive singular arms remain lazy."""
    graph = Graph()
    x, y = graph.variable("x"), graph.variable("y")
    reciprocal = 1 / x
    # The two derivatives and the energy must share a branch. The nested
    # opposite result is undefined at x=0, but unreachable for every input.
    safe = graph.select_le(x, 0, 7, graph.select_le(x, 0, 1 / y, reciprocal))
    energy = graph.select_le(x, 0, x * x, reciprocal * reciprocal)
    predicate = graph.select_le(y, 0, x, -x)
    dependent = graph.select_le(predicate, 0, safe, energy)
    roots = (
        safe,
        energy,
        graph.differentiate(energy, x),
        graph.select_le(x, 0, 11, reciprocal + 3),
        dependent,
        graph.select_le(y, 0, safe, graph.select_le(x, 0, x + y, reciprocal)),
    )
    functions = []
    for coalesced in (False, True):
        emitter = ScalarCEmitter(
            graph, {"x": "x", "y": "y"}, coalesce_branches=coalesced
        )
        emitter.emit(roots)
        functions.append(
            f"void evaluate_{int(coalesced)}(double x, double y, double* out) {{\n"
            + "\n".join(emitter.lines)
            + "\n"
            + "\n".join(
                f"  out[{index}] = {emitter.reference(root)};"
                for index, root in enumerate(roots)
            )
            + "\n}\n"
        )
    source = tmp_path / "branches.cpp"
    source.write_text(
        "#include <cmath>\n#include <cfenv>\n#include <cstring>\n"
        "#include <cassert>\n#include <limits>\n"
        + "\n".join(functions)
        + r"""
int main() {
  const double values[] = {-2, -0.0, 0.0, 0.125, 1, 3,
                           std::numeric_limits<double>::quiet_NaN()};
  for (double x : values) for (double y : values) {
    double a[6], b[6];
    std::feclearexcept(FE_ALL_EXCEPT);
    evaluate_0(x, y, a);
    const int before = std::fetestexcept(FE_DIVBYZERO | FE_INVALID);
    std::feclearexcept(FE_ALL_EXCEPT);
    evaluate_1(x, y, b);
    assert(before == std::fetestexcept(FE_DIVBYZERO | FE_INVALID));
    for (unsigned i=0; i<6; ++i)
      assert((std::isnan(a[i]) && std::isnan(b[i])) ||
             std::memcmp(a+i, b+i, sizeof(double)) == 0);
    if (x == 0 && y == 0) {
      assert(before == 0);
      assert(b[0] == 7 && b[1] == 0 && b[3] == 11 && b[4] == 7);
    }
  }
}
""",
        encoding="utf-8",
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "branches",
        compile_args=("-std=c++17", "-O0", "-frounding-math", "-ffp-contract=off"),
    )
    subprocess.run([str(executable)], check=True, timeout=30)


def test_coalesced_emitter_reuses_parent_scope_and_incremental_roots() -> None:
    """Later emission and lexical forks reuse only available parent values."""
    graph = Graph()
    x = graph.variable("x")
    root = graph.select_le(x, 0, 0, x * x)
    emitter = ScalarCEmitter(graph, {"x": "input"}, coalesce_branches=True)
    emitter.emit_assignment(root, "saved")
    child = emitter.fork()
    child.emit((root + 1,))
    assert child.reference(root) == "saved"
    assert emitter.reference(root) == "saved"
    emitter.emit((root + 2,))
    assert graph.evaluate(root + 2, {"x": 3}) == 11
