"""Compile the actual host/resident controllers against a deterministic backend."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "response/resident_krylov.hpp"

using namespace generativeqc::response;

struct Backend final : ResidentResponseBackend {
  explicit Backend(std::size_t count, const ResponsePlan& plan = {})
      : ResidentResponseBackend(plan), data(count) {}
  std::vector<std::array<double, 2>> data;
  std::size_t uploads{}, downloads{}, actions{}, norms{}, fail_norm{};
  bool nan_operator{};
  bool diagonal_operator{}, upper_operator{}, right_preconditioned{}, nan_preconditioner{};

  std::size_t dimension() const noexcept override { return 2; }
  std::size_t vector_slots() const noexcept override { return data.size(); }
  std::size_t owned_resident_bytes() const noexcept override {
    return data.size() * sizeof(data[0]);
  }
  void upload(std::size_t slot, std::span<const double> values) override {
    ++uploads;
    std::copy(values.begin(), values.end(), data.at(slot).begin());
  }
  void download(std::size_t slot, std::span<double> values) override {
    ++downloads;
    std::copy(data.at(slot).begin(), data.at(slot).end(), values.begin());
  }
  void zero(std::size_t slot) override { data.at(slot).fill(0.0); }
  void copy(std::size_t dst, std::size_t src) override { data.at(dst) = data.at(src); }
  void scale(std::size_t slot, double alpha) override {
    for (double& value : data.at(slot)) value *= alpha;
  }
  void axpy(std::size_t dst, double alpha, std::size_t src) override {
    for (std::size_t i = 0; i < 2; ++i) data.at(dst)[i] += alpha * data.at(src)[i];
  }
  double dot(std::size_t left, std::size_t right) override {
    return std::inner_product(data.at(left).begin(), data.at(left).end(),
                              data.at(right).begin(), 0.0);
  }
  double norm(std::size_t slot) override {
    if (++norms == fail_norm) throw std::overflow_error("injected norm overflow");
    return stable_norm(data.at(slot));
  }
  void apply(std::size_t dst, std::size_t src) override {
    ++actions;
    copy(dst, src);
    if (diagonal_operator) {
      data.at(dst)[0] *= 2.0;
      data.at(dst)[1] *= 4.0;
      if (upper_operator) data.at(dst)[0] += data.at(src)[1];
    }
    if (nan_operator) data.at(dst)[0] = std::numeric_limits<double>::quiet_NaN();
  }
  bool has_right_preconditioner() const noexcept override { return right_preconditioned; }
  void precondition(std::size_t dst, std::size_t src) override {
    data.at(dst)[0] = data.at(src)[0] / 2.0;
    data.at(dst)[1] = data.at(src)[1] / 4.0;
    if (nan_preconditioner) data.at(dst)[0] = std::numeric_limits<double>::quiet_NaN();
  }
};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  try {
    const auto plan = prepare_gmres(2, GmresOptions{});
    const auto count = resident_gmres_workspace(plan).vector_slots;
    const std::array<double, 2> rhs{1.0, 2.0}, guess{0.5, 0.5};
    const double maximum = std::numeric_limits<double>::max();
    const std::array<double, 2> huge{maximum, maximum};

    Backend input(count);
    const auto refused = solve_gmres_resident(plan, input, huge, guess);
    require(refused.result.status == GmresStatus::nonfinite_input,
            "overflowing RHS norm did not return nonfinite_input");
    require(input.uploads == 0 && input.actions == 0 && input.downloads == 0,
            "invalid RHS norm reached device execution");
    require(std::equal(guess.begin(), guess.end(), refused.result.solution.begin()),
            "invalid RHS did not preserve the supplied initial guess");

    Backend initial(count);
    initial.nan_operator = true;
    const auto invalid = solve_gmres_resident(plan, initial, rhs, guess);
    require(invalid.result.status == GmresStatus::nonfinite_operator,
            "nonfinite initial operator residual escaped its failure status");
    require(initial.downloads == 1, "failure must publish one final solution");

    for (std::size_t checkpoint : {2U, 3U}) {
      Backend backend(count);
      backend.fail_norm = checkpoint;
      const auto failed = solve_gmres_resident(plan, backend, rhs);
      require(failed.result.status == GmresStatus::nonfinite_operator,
              "Arnoldi/candidate norm failure escaped its failure status");
      require(backend.downloads == 1, "checkpoint failure downloaded more than once");
    }

    Backend valid(count);
    const auto solved = solve_gmres_resident(plan, valid, rhs);
    require(solved.converged() && valid.downloads == 1,
            "valid resident identity solve regressed");

    const LinearResponseProblem problem(2, [](auto input, auto output) {
      output[0] = 2.0 * input[0];
      output[1] = 4.0 * input[1];
    });
    const auto bound = prepare_response(problem, GmresOptions{});
    Backend preconditioned(count, bound);
    preconditioned.diagonal_operator = true;
    preconditioned.right_preconditioned = true;
    const auto accelerated = solve_response_resident(bound, problem, preconditioned, rhs);
    require(accelerated.converged() && accelerated.result.iterations == 1 &&
                accelerated.result.preconditioner_actions == 1,
            "resident right preconditioning failed");
    require(std::abs(accelerated.result.solution[0] - 0.5) < 1e-12 &&
                std::abs(accelerated.result.solution[1] - 0.5) < 1e-12,
            "right preconditioning changed the physical solution");
    require(preconditioned.uploads == 1 && preconditioned.downloads == 1,
            "resident preconditioner staged an iteration vector");
    preconditioned.nan_preconditioner = true;
    const auto rejected = solve_response_resident(bound, problem, preconditioned, rhs);
    require(rejected.result.status == GmresStatus::nonfinite_preconditioner &&
                rejected.result.operator_actions == 0 &&
                rejected.result.preconditioner_actions == 1,
            "nonfinite preconditioner reached the physical operator");
    preconditioned.nan_preconditioner = false;
    require(solve_response_resident(bound, problem, preconditioned, rhs).converged(),
            "resident backend could not recover after preconditioner failure");

    const LinearResponseProblem nonsymmetric(2, [](auto input, auto output) {
      output[0] = 2.0 * input[0] + input[1];
      output[1] = 4.0 * input[1];
    });
    GmresOptions exact_options;
    const auto workspace = resident_gmres_workspace(bound);
    const auto exact_bytes = workspace.host_scalar_bytes + workspace.host_result_bytes;
    for (const auto budget : {exact_bytes, exact_bytes + 64}) {
      exact_options.max_workspace_bytes = budget;
      const auto nonsymmetric_plan = prepare_response(nonsymmetric, exact_options);
      Backend triangular(count, nonsymmetric_plan);
      triangular.diagonal_operator = triangular.upper_operator = true;
      triangular.right_preconditioned = true;
      const auto physical = solve_response_resident(
          nonsymmetric_plan, nonsymmetric, triangular, rhs);
      require(physical.converged() && physical.result.residual_norm < 1e-12 &&
                  std::abs(physical.result.solution[0] - 0.25) < 1e-12 &&
                  std::abs(physical.result.solution[1] - 0.5) < 1e-12,
              "resident right preconditioner changed a nonsymmetric problem");
      require(triangular.uploads == 1 && triangular.downloads == 1 &&
                  physical.result.measured_workspace_peak_bytes <= budget,
              "resident exact host workspace/transfer contract");
    }
    exact_options.max_workspace_bytes = exact_bytes - 1;
    const auto insufficient_plan = prepare_response(nonsymmetric, exact_options);
    Backend insufficient(count, insufficient_plan);
    const auto insufficient_result = solve_response_resident(
        insufficient_plan, nonsymmetric, insufficient, rhs);
    require(insufficient_result.result.status == GmresStatus::workspace_limit &&
                insufficient.uploads == 0 && insufficient.actions == 0,
            "resident workspace refusal performed device work");

    const LinearResponseProblem other(2, [](auto input, auto output) {
      std::copy(input.begin(), input.end(), output.begin());
    });
    const auto other_plan = prepare_response(other, GmresOptions{});
    const auto before_uploads = preconditioned.uploads;
    for (bool wrong_backend : {false, true}) {
      bool threw = false;
      try {
        if (wrong_backend)
          solve_response_resident(other_plan, other, preconditioned, rhs);
        else
          solve_response_resident(bound, other, preconditioned, rhs);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      require(threw && preconditioned.uploads == before_uploads,
              "wrong resident problem binding reached uploads");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
"""


def test_resident_gmres_preserves_numerical_failure_statuses(tmp_path: Path) -> None:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++20 compiler is unavailable")
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "resident_gmres_failures.cpp"
    binary = tmp_path / "resident_gmres_failures"
    source.write_text(SOURCE, encoding="utf-8")
    built = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O0",
            "-pthread",
            "-I",
            str(root / "src"),
            "-I",
            str(root / "include"),
            str(source),
            str(root / "src/response/native_gmres.cpp"),
            str(root / "src/response/resident_gmres.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    tested = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=30, check=False
    )
    assert tested.returncode == 0, tested.stdout + tested.stderr
