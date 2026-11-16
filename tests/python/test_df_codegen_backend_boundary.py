"""Keep DF admission shared while retaining CPU replay only on its backend."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import numpy as np
from _cc_owner_test_support import ROOT, write_df_cpu_headers

from tools import generate_df_ao_native as ao_codegen

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import NativeCxx


def test_df_cuda_admission_does_not_require_host_replay(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Compare admission interfaces and check host factors against NumPy.

    CUDA preprocessor macros exercise the compiled interface without a device;
    real CUDA compilation and device-image comparisons are separate gates.
    """
    write_df_cpu_headers(tmp_path)
    (tmp_path / "generated_df_ao_source.hpp").write_text(ao_codegen.cpu_header())
    rng = np.random.default_rng(55810)
    raw = rng.normal(scale=0.2, size=(2, 2, 3))
    root = rng.normal(scale=0.3, size=(3, 3))
    coefficients = rng.normal(scale=0.4, size=(2, 2))
    expected = np.einsum(
        "mp,nq,mnP,PQ->pqQ", coefficients, coefficients, raw, root
    ).ravel()
    inputs = ["ao::Inputs inputs{};"]
    for name, values in (
        ("three_center", raw),
        ("inverse_root", root),
        ("coefficients", coefficients),
    ):
        literals = ",".join(float(value).hex() for value in values.ravel())
        inputs += [
            f"const double feed_{name}[] = {{{literals}}};",
            f"inputs.{name} = feed_{name};",
        ]
    source = tmp_path / "boundary.cpp"
    source.write_text(_PROBE.replace("REPLAY_INPUTS", "\n".join(inputs)))
    observations = []
    for cuda_interface in (False, True):
        arguments = [
            "-std=c++20",
            "-O0",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ]
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
        atol=2e-13,
        rtol=2e-13,
    )


_PROBE = r"""
#include "generated_df_ccsd_spectator_pairs_cpu.hpp"
#include "generated_df_ao_source.hpp"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
namespace generated = generativeqc::cc::generated;
namespace ao = generativeqc::posthf::df_ao_generated;

template<class Input>
concept HasVirtualReplay = requires(const Input& input, double* arena) {
  run_virtual_cpu(1,2,input,arena,1);
};
template<class Input>
concept HasCoreReplay = requires(const Input& input, double* arena) {
  run_iteration_cpu(1,2,input,arena,1);
};
template<class Input>
concept HasHoistedReplay = requires(const Input& input, double* arena) {
  run_prepare_cpu(1,2,input,arena,1);
};
template<class Input>
concept HasNormReplay = requires(const Input& input, double* arena) {
  run_ladder_norm_cpu(1,2,input,arena,1);
};
template<class Input>
concept HasFactorReplay = requires(const Input& input, double* arena) {
  evaluate_factors(2,3,input,arena,1);
};

int main() {
#ifdef __CUDACC__
  constexpr bool host_replay = false;
#else
  constexpr bool host_replay = true;
#endif
  static_assert(HasVirtualReplay<generated::df::Inputs> == host_replay);
  static_assert(HasCoreReplay<generated::dfcore::Inputs> == host_replay);
  static_assert(HasHoistedReplay<generated::dfhoist::Inputs> == host_replay);
  static_assert(HasNormReplay<generated::dfpairs::NormInputs> == host_replay);
  static_assert(HasFactorReplay<ao::Inputs> == host_replay);
  for (const auto [occupied,virtuals] : {
      std::pair<std::size_t,std::size_t>{1,2},{20,1},{20,8},{0,2}}) {
    std::cout << generated::df::virtual_cpu_arena_elements(occupied,virtuals) << ' '
              << generated::df::virtual_cuda_arena_elements(occupied,virtuals) << ' '
              << generated::df::virtual_replay_arena_elements(occupied,virtuals) << ' '
              << generated::df::virtual_batch_arena_elements(occupied,virtuals,3) << ' '
              << generated::dfcore::iteration_arena_elements(occupied,virtuals) << ' '
              << generated::dfhoist::auxiliary_packed_arena_elements(occupied,virtuals) << ' '
              << generated::dfpairs::auxiliary_arena_elements(occupied,virtuals) << ' '
              << generated::dfpairs::ladder_norm_arena_elements(occupied,virtuals) << ' '
              << ao::required_factors(occupied,virtuals) << ' '
              << ao::work_factor_block(occupied,virtuals,3) << ' ';
  }
  bool rejected = false;
  try {
    generated::df::checked_product({std::numeric_limits<std::size_t>::max(),2});
  } catch (const std::overflow_error&) { rejected = true; }
  assert(rejected);
#ifndef __CUDACC__
  REPLAY_INPUTS
  rejected = false;
  try { ao::evaluate_factors(2,3,inputs,nullptr,0); }
  catch (const std::length_error&) { rejected = true; }
  assert(rejected);
  std::vector<double> arena(ao::required_factors(2,3));
  const auto result = ao::evaluate_factors(2,3,inputs,arena.data(),arena.size());
  for (std::size_t index = 0; index < 12; ++index)
    std::cout << std::setprecision(17) << result.factors[index] << ' ';
#endif
  std::cout << '\n';
}
"""
