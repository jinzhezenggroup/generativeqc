"""The common native CC iteration policy retains physical CPU endpoints."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from _cc_owner_test_support import compile_owner, write_df_cpu_headers

from tools.generativeqc_cc.oracle import DeterminantOracle, dense_feeds, random_case

ROOT = Path(__file__).resolve().parents[2]


def _literal(value: np.ndarray) -> str:
    return "{" + ",".join(float(x).hex() for x in np.asarray(value).ravel()) + "}"


def _problem() -> tuple[str, DeterminantOracle]:
    o, v = 2, 2
    f, g, t1, t2 = random_case(o, v, 3122)
    f *= 0.04
    f += np.diag(np.r_[np.linspace(-1.2, -0.8, o), np.linspace(0.7, 1.1, v)])
    g *= 0.15
    feeds = dense_feeds(f, g, np.zeros_like(t1), np.zeros_like(t2))
    d1 = np.diag(f)[:o, None] - np.diag(f)[None, o:]
    feeds.update(d1=d1, d2=d1[:, None, :, None] + d1[None, :, None, :])
    lines = ["Problem p; p.nocc=2; p.nvir=2;"]
    for name, value in feeds.items():
        field = "initial_" + name if name in ("t1", "t2") else name
        lines.append(f"p.{field}={_literal(value)};")
    return "\n".join(lines), DeterminantOracle(f, g, o)


def test_native_cpu_shared_driver_matches_determinant_oracle(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    problem, oracle = _problem()
    write_df_cpu_headers(tmp_path)
    source = tmp_path / "shared_driver.cpp"
    source.write_text(PREFIX + problem + MAIN)
    executable = tmp_path / "shared_driver"
    compile_owner(compiler, tmp_path, [ROOT / "src/cc/solver.cpp", source], executable)
    run = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=30, check=True
    )
    rows = [np.asarray(line.split(), dtype=float) for line in run.stdout.splitlines()]
    assert len(rows) == 12
    for row in rows:
        history, damping, budget, status, iterations, updates, calls, replays = row[:8]
        energy, r1, r2 = row[8:11]
        t1, t2 = row[11:15].reshape(2, 2), row[15:].reshape(2, 2, 2, 2)
        expected_energy, expected_r1, expected_r2 = oracle.evaluate_full(t1, t2)
        assert abs(energy - expected_energy) <= 2e-12
        assert abs(r1 - np.max(np.abs(expected_r1))) <= 2e-12
        assert abs(r2 - np.max(np.abs(expected_r2))) <= 2e-12
        assert iterations == updates + 1
        assert calls >= iterations
        if not history:
            assert calls == iterations
        if budget == 1:
            assert status == 1 and updates == 1 and replays == 0
        else:
            assert status == 0 and replays == 1
            assert max(r1, r2) <= 1e-10
        assert damping in (0.0, 0.2)


def test_both_native_backends_delegate_scientific_iteration_policy() -> None:
    for filename, output in (
        ("solver.cpp", "IterationOutputs"),
        ("cuda_solver.cu", "DeviceIterationOutputs"),
    ):
        source = (ROOT / "src/cc" / filename).read_text()
        assert f"run_cc_iterations<generated::{output}>" in source
        assert "delta <= options.energy_tolerance" not in source
        assert "iteration == options.max_iterations" not in source
        assert "has_carried_output" not in source


PREFIX = r"""
#include "cc/solver.hpp"
#include <iomanip>
#include <iostream>
using namespace generativeqc::cc;
int main(){
"""
MAIN = r"""
for(unsigned history:{0U,2U,6U}) for(double damping:{0.,.2}) for(unsigned budget:{1U,100U}){
 SolverOptions options;options.diis_size=history;options.damping=damping;
 options.max_iterations=budget;options.residual_tolerance=1e-10;options.energy_tolerance=1e-12;
 const auto r=solve_cpu(p,options);const auto& d=r.diagnostic;
 std::cout<<std::setprecision(17)<<history<<' '<<damping<<' '<<budget<<' '
 <<int(r.status)<<' '<<d.iterations<<' '<<d.update_calls<<' '<<d.iteration_graph_calls<<' '
 <<d.replay_graph_calls<<' '<<r.correlation_energy<<' '<<d.r1_max<<' '<<d.r2_max;
 for(double x:r.t1)std::cout<<' '<<x;for(double x:r.t2)std::cout<<' '<<x;
 std::cout<<'\n';
}
}
"""


def test_common_driver_budget_replay_and_carry_contract(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "policy.cpp"
    source.write_text(POLICY)
    executable = tmp_path / "policy"
    compile_owner(compiler, tmp_path, [source], executable)
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=10)


POLICY = r"""
#include "cc/iteration_driver.hpp"
#include <cassert>
#include <limits>
using namespace generativeqc::cc;
int main(){
 for(unsigned budget:{0U,1U,4U,std::numeric_limits<unsigned>::max()})
 for(unsigned history:{0U,1U,2U}) for(bool reject_first:{false,true}) {
  SolverOptions options;options.max_iterations=budget;
  unsigned state=0,observations=0,evaluations=0,replays=0;
  bool converged=false;
  run_cc_iterations<unsigned>(options,
   [&](const std::optional<unsigned>& carried){
    if(!carried)++evaluations;else assert(*carried==state);
    return std::pair{state,IterationMetrics{.5,0.,0.}};
   },
   [&](unsigned count,IterationMetrics,double){observations=count;},
   [&](){++replays;return IterationMetrics{.5,reject_first&&replays==1?1.:0.,0.};},
   [&](unsigned output)->std::optional<unsigned>{
    assert(output==state);++state;
    if(history)++evaluations;
    if(history==1)return state;
    return std::nullopt;
   },
   [&](){converged=true;},
   [&](const std::runtime_error&){assert(false);});
  const unsigned needed=reject_first?2:1;
  const unsigned expected=std::min(budget,needed);
  assert(state==expected && observations==expected+1);
  assert(converged==(budget>=needed));
  assert(replays==expected);
  assert(evaluations==1+expected*(history==2?2:1));
 }
 // Backend failure policy can translate a numerical failure or rethrow a
 // transport/runtime failure. No update/replay is permitted after either.
 for(bool rethrow:{false,true}) {
  SolverOptions options;unsigned failures=0;
  try {
   run_cc_iterations<unsigned>(options,
    [&](const std::optional<unsigned>&)->std::pair<unsigned,IterationMetrics>{
     throw std::runtime_error("injected");
    },
    [&](unsigned,IterationMetrics,double){assert(false);},
    [&](){assert(false);return IterationMetrics{};},
    [&](unsigned)->std::optional<unsigned>{assert(false);return std::nullopt;},
    [&](){assert(false);},
    [&](const std::runtime_error&){++failures;if(rethrow)throw;});
   assert(!rethrow);
  }catch(const std::runtime_error&){assert(rethrow);}
  assert(failures==1);
 }
}
"""
