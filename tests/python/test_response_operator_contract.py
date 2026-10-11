from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_native_response_symmetry_contract_compiles_and_runs(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    source = tmp_path / "response_symmetry_contract.cpp"
    executable = tmp_path / "response_symmetry_contract"
    source.write_text(
        r"""
#include <array>
#include <span>
#include <stdexcept>

#include "response/linear_problem.hpp"

int main() {
  using generativeqc::response::LinearResponseProblem;
  using generativeqc::response::LinearResponseSymmetry;

  auto general_apply = [](std::span<const double> input, std::span<double> output) {
    output[0] = input[0] + 2.0 * input[1];
    output[1] = 3.0 * input[0] + 4.0 * input[1];
  };
  LinearResponseProblem general(2, general_apply);
  if (general.symmetry() != LinearResponseSymmetry::General || general.has_transpose()) return 1;
  try {
    (void)general.apply_transpose();
    return 2;
  } catch (const std::logic_error&) {
  }

  LinearResponseProblem symmetric(2, general_apply, LinearResponseSymmetry::Symmetric);
  if (symmetric.symmetry() != LinearResponseSymmetry::Symmetric || !symmetric.has_transpose())
    return 3;
  const std::array<double, 2> input{1.0, 2.0};
  std::array<double, 2> output{};
  symmetric.apply_transpose()(input, output);
  if (output[0] != 5.0 || output[1] != 11.0) return 4;

  auto transpose_apply = [](std::span<const double> input, std::span<double> output) {
    output[0] = input[0] + 3.0 * input[1];
    output[1] = 2.0 * input[0] + 4.0 * input[1];
  };
  LinearResponseProblem nonsymmetric(
      2, general_apply, LinearResponseSymmetry::General, transpose_apply);
  if (nonsymmetric.symmetry() != LinearResponseSymmetry::General || !nonsymmetric.has_transpose())
    return 5;
  output = {};
  nonsymmetric.apply_transpose()(input, output);
  if (output[0] != 7.0 || output[1] != 10.0) return 6;

  try {
    LinearResponseProblem contradictory(
        2, general_apply, LinearResponseSymmetry::Symmetric, transpose_apply);
    (void)contradictory;
    return 7;
  } catch (const std::invalid_argument&) {
  }

  return 0;
}
"""
    )
    native_cxx.build_executable(
        [source],
        executable,
        compile_args=(
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
        ),
    )
    subprocess.run([str(executable)], cwd=ROOT, check=True, timeout=30)


def test_native_response_plan_binding_and_solver_contracts(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Run identity, batch and physical residual gates without a native library."""
    executable = tmp_path / "native_response_contract"
    required_native_cxx.build_executable(
        [
            ROOT / "src/response/native_gmres.cpp",
            ROOT / "src/response/resident_gmres.cpp",
            ROOT / "tests/native/test_native_gmres.cpp",
        ],
        executable,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
        ),
    )
    subprocess.run([str(executable)], cwd=ROOT, check=True, timeout=30)


def test_mp2_response_adapter_declares_symmetric_operator() -> None:
    source = (ROOT / "src/posthf/mp2_force.cpp").read_text()
    start = source.index("response::LinearResponseProblem response_problem")
    end = source.index("ConventionalForceResult conventional_force_impl", start)
    adapter = source[start:end]

    assert "response::LinearResponseSymmetry::Symmetric" in adapter


def test_rccsd_lambda_cpu_cuda_use_shared_general_response_problem() -> None:
    cpu = (ROOT / "src/cc/lambda_response.cpp").read_text()
    cuda = (ROOT / "src/cc/lambda_response_cuda.cu").read_text()

    start = cpu.index("response::LinearResponseProblem make_lambda_response_problem")
    end = cpu.index("}  // namespace detail", start)
    adapter = cpu[start:end]

    assert "response::LinearResponseSymmetry::General" in adapter
    for source in (cpu, cuda):
        assert "detail::make_lambda_response_problem" in source
        assert "response::prepare_response(" in source
        assert "response::solve_response(" in source
        assert "response::solve_gmres(" not in source


def test_rhf_frame_response_uses_bound_provisional_and_physical_problems() -> None:
    """The exact retry cannot reuse a screened operator's prepared binding."""
    source = (ROOT / "src/hf/rhf_frame_response.cu").read_text()
    assert "response::solve_gmres(" not in source
    for name in ("provisional", "physical"):
        assert f"response::prepare_response({name}_problem, zplan.options)" in source
        assert f"response::solve_response({name}_plan, {name}_problem" in source
    assert "inverse->apply(x, y)" in source
    assert "owner.apply(z.solution, residual, true, 0.0)" in source


def test_rccsdt_z_uses_shared_problem_after_symmetry_and_curvature_gates() -> None:
    """CC retains ownership of the generated action and independent dense oracle."""
    source = (ROOT / "src/cc/rccsdt_force.cpp").read_text()
    assert "response::solve_gmres(" not in source
    certify = source.index("if (asymmetry > 1e-10)")
    stable = source.index(
        "if (!(minimum_curvature > kMinimumOrbitalCurvature))", certify
    )
    bind = source.index(
        "const response::LinearResponseProblem response_problem(", stable
    )
    solve = source.index("response::solve_response(z_plan, response_problem", bind)
    assert certify < stable < bind < solve
    assert "response::LinearResponseSymmetry::Symmetric" in source[bind:solve]
    assert "independent[row] -= correlation.orbital_rhs[row]" in source[solve:]
