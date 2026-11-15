"""Keep RCCSD backend ownership executable without requiring a CUDA device."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.tensor import execute

from tools import generate_rccsd_native as codegen
from tools.generativeqc_cc.oracle import dense_feeds, random_case

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import NativeCxx


def test_cuda_admission_metadata_does_not_require_host_replay(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Both interfaces share admission; CPU replay matches unlowered TensorIR.

    Defining CUDA preprocessor macros is a host-only API probe, not a physical
    device test. Complete CUDA compilation is qualified separately.
    """
    (tmp_path / "generated_rccsd_cpu.hpp").write_text(codegen.cpu_header())
    (tmp_path / "generated_triples_fock_response_cpu.hpp").write_text(
        codegen.triples_fock_cpu_header()
    )
    occupied, virtuals = 1, 2
    fock, integrals, singles, doubles = random_case(occupied, virtuals, 4096)
    feeds = dense_feeds(fock, integrals, singles, doubles)
    program = codegen.build_ccsd_program(
        occupied, virtuals, form="expanded", diagnostics=False
    )
    expected_outputs = execute(program, feeds).outputs
    expected = np.concatenate(
        [
            np.asarray(expected_outputs[name]).ravel()
            for name in (
                "correlation_energy",
                "singles_residual",
                "doubles_residual",
            )
        ]
    )
    replay = ["Inputs inputs{};"]
    for name in codegen.INPUT_NAMES:
        if name not in feeds:
            continue
        values = ",".join(float(value).hex() for value in feeds[name].ravel())
        replay += [
            f"const double feed_{name}[] = {{{values}}};",
            f"inputs.{name} = feed_{name};",
        ]
    source = tmp_path / "boundary.cpp"
    source.write_text(_PROBE.replace("REPLAY_INPUTS", "\n".join(replay)))
    observations = []
    for cuda_interface in (False, True):
        arguments = ["-std=c++20", "-O0"]
        if cuda_interface:
            arguments += ["-D__CUDACC__", "-D__host__=", "-D__device__="]
        binary = native_cxx.build_executable(
            [source],
            tmp_path / ("cuda-admission" if cuda_interface else "host-replay"),
            compile_args=arguments,
        )
        result = subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True, timeout=30
        )
        observations.append(result.stdout.split())
    assert observations[0][: -expected.size] == observations[1]
    np.testing.assert_allclose(
        np.asarray(observations[0][-expected.size :], dtype=float),
        expected,
        atol=2e-11,
        rtol=2e-11,
    )


_PROBE = r"""
#include "generated_triples_fock_response_cpu.hpp"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace generativeqc::cc::generated;

template<class Input>
concept HasHostReplay = requires(const Input& input, double* arena) {
  run_replay_cpu(1,2,input,arena,1);
};
template<class Input>
concept HasHostResolvent = requires(const Input& input, double* arena) {
  run_triples_resolvent_cpu(1,2,2,input,arena,1);
};

int main() {
#ifdef __CUDACC__
  static_assert(!HasHostReplay<Inputs>);
  static_assert(!HasHostResolvent<TriplesResolventInputs>);
#else
  static_assert(HasHostReplay<Inputs>);
  static_assert(HasHostResolvent<TriplesResolventInputs>);
#endif
  for (const auto [occupied,virtuals] : {
      std::pair<std::size_t,std::size_t>{1,2},{20,1},{20,8},{0,2}}) {
    assert(replay_arena_elements(occupied,virtuals) <=
           replay_strict_arena_elements(occupied,virtuals));
    std::cout << iteration_arena_elements(occupied,virtuals) << ' '
              << iteration_reuse_arena_elements(occupied,virtuals) << ' '
              << replay_arena_elements(occupied,virtuals) << ' '
              << replay_selected_program_hash(occupied,virtuals) << ' '
              << triples_resolvent_arena_elements(occupied,virtuals,2) << ' ';
  }
  bool rejected = false;
  try { checked_product({std::numeric_limits<std::size_t>::max(),2}); }
  catch (const std::length_error&) { rejected = true; }
  assert(rejected);
#ifndef __CUDACC__
  REPLAY_INPUTS
  rejected = false;
  try { run_replay_cpu(1,2,inputs,nullptr,0); }
  catch (const std::length_error&) { rejected = true; }
  assert(rejected);
  std::vector<double> arena(replay_arena_elements(1,2));
  const auto result = run_replay_cpu(1,2,inputs,arena.data(),arena.size());
  std::cout << std::setprecision(17) << result.energy << ' ';
  for (std::size_t index = 0; index < 2; ++index) std::cout << result.r1[index] << ' ';
  for (std::size_t index = 0; index < 4; ++index) std::cout << result.r2[index] << ' ';
#endif
  std::cout << '\n';
}
"""
