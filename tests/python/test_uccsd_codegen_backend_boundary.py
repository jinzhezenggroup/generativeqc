"""Execute the native host/CUDA header boundary without a GPU or shared library."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.cc.uccsd_equations import build_program
from generativeqc_compiler.tensor import execute

from tools import generate_uccsd_native as codegen

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import NativeCxx


def test_cuda_keeps_metadata_without_borrowing_host_evaluators(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """CPU replay stays numerical; CUDA retains sizes, work and error contracts.

    The CUDA preprocessor mode is a host-only interface probe, not physical
    CUDA qualification. Complete generated CUDA objects are compiled separately.
    """
    (tmp_path / "generated_uccsd_cpu.hpp").write_text(codegen.cpu_header())
    (tmp_path / "generated_uccsd_response.hpp").write_text(codegen.response_header())
    source = tmp_path / "boundary.cpp"
    source.write_text(_PROBE)
    observations = []
    for cuda_interface in (False, True):
        arguments = ["-std=c++20", "-O0"]
        if cuda_interface:
            arguments += ["-D__CUDACC__", "-D__host__=", "-D__device__="]
        binary = native_cxx.build_executable(
            [source],
            tmp_path / ("cuda-metadata" if cuda_interface else "host-replay"),
            compile_args=arguments,
        )
        result = subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True, timeout=30
        )
        observations.append(result.stdout.split())
    assert observations[0][:-1] == observations[1]
    program = build_program(1, 1, 1, 1)
    # A singleton same-spin doubles block vanishes by antisymmetry.
    feeds = {
        node.attrs["name"]: np.full(
            node.spec.shape,
            0.0 if node.attrs["name"] in ("t2aa", "t2bb") else 0.01,
        )
        for node in program.live_nodes
        if node.op == "input"
    }
    expected = execute(program, feeds).outputs["correlation_energy"]
    np.testing.assert_allclose(
        float(observations[0][-1]), expected, atol=3e-14, rtol=3e-14
    )


_PROBE = r"""
#include "generated_uccsd_response.hpp"
#include <cassert>
#include <iomanip>
#include <iostream>
using namespace generativeqc::cc::ucc_generated;

template<class Input>
concept HasHostResidual = requires(const Input& input, double* arena) {
  evaluate(1,1,1,1,input,arena,1);
};
template<class Feeds>
concept HasHostResponse = requires(const Feeds& feeds, double* arena) {
  response_evaluate(std::string{"cc"},1,1,1,1,2,2,1,feeds,arena,1);
};

int main() {
#ifdef __CUDACC__
  static_assert(!HasHostResidual<Inputs>);
  static_assert(!HasHostResponse<ResponseFeeds>);
#else
  static_assert(HasHostResidual<Inputs>);
  static_assert(HasHostResponse<ResponseFeeds>);
#endif
  const auto arena_size = required(1,1,1,1);
  const auto response_size = response_required("cc",1,1,1,1,2,2,1);
  const auto inputs = response_inputs("cc",1,1,1,1,2,2,1);
  const auto outputs = response_outputs("cc",1,1,1,1,2,2,1);
  assert(outputs.at("correlation_energy") == 1);
  bool rejected = false;
  try { checked_product({std::numeric_limits<std::size_t>::max(),2}); }
  catch (const std::length_error&) { rejected = true; }
  assert(rejected);
  rejected = false;
  try { response_required("unknown",1,1,1,1,2,2,1); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
  std::cout << arena_size << ' ' << work(1,1,1,1) << ' ' << response_size << ' '
            << response_work("cc",1,1,1,1,2,2,1) << ' ' << inputs.size() << ' '
            << outputs.size() << ' ' << kEquationHash << ' ' << kTriplesHash;
#ifndef __CUDACC__
  Inputs values{};
  std::vector<double> feeds(kFeeds.size(),0.01);
  for (std::size_t index = 0; index < kFeeds.size(); ++index) {
    const std::string name(kFeeds[index].name);
    if (name == "t2aa" || name == "t2bb") feeds[index] = 0.0;
    values.*(kFeeds[index].field) = feeds.data() + index;
  }
  std::vector<double> arena(std::max<std::size_t>(1,arena_size));
  const auto result = evaluate(1,1,1,1,values,arena.data(),arena.size());
  std::cout << ' ' << std::setprecision(17) << *result.correlation_energy;
#endif
  std::cout << '\n';
}
"""
