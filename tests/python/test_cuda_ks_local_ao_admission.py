"""Host admission coverage, not GPU numerics or generic default profitability."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.dft.ao_cuda import emit_native_xc_point_dispatch

from tools.generate_xc_split_hybrid_registry import emit_registry

ROOT = Path(__file__).resolve().parents[2]


def _definition(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def admission_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile real point capabilities, XC layout/resource logic and SCF guard."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    start = source.index("    const char* ao_selection = std::getenv(")
    end = source.index("    if (host_unfused &&", start)
    guard = source[start:end]
    # No method/provider/schedule stubs: restoring a whitelist is a compile error.
    for unrelated in (
        "SemilocalFamily",
        "precision_schedule",
        "provider",
        "exchange_coefficient",
    ):
        assert unrelated not in guard
    header = (ROOT / "src/dft/cuda_xc.hpp").read_text()
    declarations = "\n".join(
        _definition(header, signature) + ";"
        for signature in (
            "enum class CudaXcAoPrecision",
            "struct CudaXcPointCapabilities",
            "struct CudaXcLayout",
            "struct CudaXcExecutionCapabilities",
            "struct CudaXcAoTiles",
            "struct CudaXcAoSelectionResources",
        )
    )
    xc_source = (ROOT / "src/dft/cuda_xc.cpp").read_text()
    declarations += "\n" + _definition(xc_source, "struct CudaXcProgramTraits") + ";"
    definitions = "\n".join(
        _definition(xc_source, signature)
        for signature in (
            "CudaXcProgramTraits cuda_xc_program_traits(",
            "CudaXcLayout cuda_xc_layout_shape(",
            "CudaXcExecutionCapabilities cuda_xc_execution_capabilities(",
            "CudaXcLayout cuda_xc_local_ao_layout(",
            "CudaXcAoSelectionResources cuda_xc_ao_selection_resources(",
        )
    )
    directory = tmp_path_factory.mktemp("ks-local-ao-admission")
    (directory / "generated_split_hybrid_registry.cuh").write_text(emit_registry())
    unit, executable = directory / "probe.cpp", directory / "probe"
    unit.write_text(
        r"""
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "dft/cuda_ks_precision.hpp"
#include "dft/semilocal_family.hpp"
#include "runtime/bounded_workspace.hpp"
#include "generated_split_hybrid_registry.cuh"
using namespace generativeqc;
using namespace generativeqc::dft;
using runtime::size_add;
using runtime::size_mul;
namespace generated = generativeqc::dft::generated;
"""
        + declarations
        + r"""
namespace cuda_xc_detail {
using CudaXcPointLauncher = void (*)();
template <unsigned F, bool R> void launch_points() {}
template <unsigned Mask> void launch_split_hybrid_points() {}
"""
        + emit_native_xc_point_dispatch()
        + "\n}\n"
        + definitions
        + r"""
int main(int argc, char** argv) {
  if (argc != 7) return 2;
  if (std::string(argv[1]) == "unset") unsetenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO");
  else setenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", argv[1], 1);
  const std::string family = argv[2], mode = argv[4];
  const auto functional = family == "m062x" ? generated::kM062XFunctionalCode
                          : family == "mn15" ? generated::kMN15FunctionalCode
                          : static_cast<unsigned>(std::stoul(family));
  const bool unrestricted = std::string(argv[3]) == "uks";
  const bool host_unfused = mode == "host";
  const bool automatic = std::string(argv[5]) == "auto";
  const bool nonlocal = std::string(argv[6]) == "nonlocal";
  try {
    auto xc_layout = cuda_xc_layout_shape(
        3, 9, mode == "budget" ? 4096 : 12, mode == "budget" ? 1048576 : 64,
        functional, unrestricted, 16, false, CudaXcAoPrecision::Fp64,
        mode == "scaled" ? 0.37 : 1.0, mode == "scaled" ? 0.81 : 1.0, false);
    if (mode == "response") xc_layout.response = true;
    if (mode == "fp32-ao") xc_layout.ao_precision = CudaXcAoPrecision::Fp32ComputeFp64Storage;
    if (mode == "already-local") xc_layout.local_ao = true;
    if (mode == "zero-nao") xc_layout.nao = 0;
    if (mode == "zero-points") xc_layout.npoint = 0;
    if (mode == "zero-tile") xc_layout.tile_points = 0;
"""
        + guard
        + r"""
    // Emulate a successful owner's selected layout without any CUDA execution.
    // Empty maps are legal; this tests the arithmetic contract, not the cutoff.
    if (admit_ao) {
      CudaXcAoTiles maps;
      maps.offsets.resize(ao_selection_bound.tiles + 1);
      xc_layout = cuda_xc_local_ao_layout(xc_layout, maps);
    }
    const auto schedule = resolve_cuda_ks_precision_schedule(
        automatic ? GENERATIVEQC_PRECISION_AUTO : GENERATIVEQC_PRECISION_FP64,
        xc_layout.fast_paths, false, nonlocal);
    const auto iteration = resolve_cuda_ks_iteration_precision(
        schedule, false, cuda_xc_execution_capabilities(xc_layout).mixed_density_contraction);
    std::cout << (admit_ao ? "local" : "dense") << ":"
              << iteration.uses_lower_precision(cuda_ks_precision_region::kCoulombJ) << ":"
              << iteration.uses_lower_precision(cuda_ks_precision_region::kDensityContraction);
  } catch (const std::invalid_argument&) { std::cout << "rejected"; }
}
"""
    )
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
            str(unit),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    return executable


def _run(
    probe: Path,
    selection: str,
    family: str = "1",
    spin: str = "rks",
    mode: str = "physical",
    precision: str = "fp64",
    nonlocal_: str = "local",
) -> str:
    return subprocess.run(
        [str(probe), selection, family, spin, mode, precision, nonlocal_],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout


@pytest.mark.parametrize("family", ["0", "1", "2", "3", "4"])
@pytest.mark.parametrize("spin", ["rks", "uks"])
@pytest.mark.parametrize("precision", ["fp64", "auto"])
@pytest.mark.parametrize("selection", ["unset", "0", "1"])
def test_capable_family_spin_matrix(
    admission_probe: Path, family: str, spin: str, precision: str, selection: str
) -> None:
    automatic = precision == "auto"
    dense = selection == "0"
    expected = f"{'dense' if dense else 'local'}:{int(automatic)}:{int(dense and automatic and int(family) < 3)}"
    assert (
        _run(admission_probe, selection, family, spin, precision=precision) == expected
    )


@pytest.mark.parametrize("spin", ["rks", "uks"])
@pytest.mark.parametrize("precision", ["fp64", "auto"])
@pytest.mark.parametrize("nonlocal_", ["local", "nonlocal"])
@pytest.mark.parametrize("selection", ["unset", "1"])
def test_scaled_pbe_is_not_an_exact_composition_gate(
    admission_probe: Path, spin: str, precision: str, nonlocal_: str, selection: str
) -> None:
    assert _run(
        admission_probe, selection, "1", spin, "scaled", precision, nonlocal_
    ) == (f"local:{int(precision == 'auto')}:0")


@pytest.mark.parametrize(
    "mode",
    [
        "host",
        "response",
        "fp32-ao",
        "already-local",
        "zero-nao",
        "zero-points",
        "zero-tile",
    ],
)
@pytest.mark.parametrize("selection", ["unset", "0", "1"])
def test_actual_layout_incompatibilities_fail_closed(
    admission_probe: Path, mode: str, selection: str
) -> None:
    assert _run(admission_probe, selection, mode=mode) == (
        "rejected" if selection == "1" else "dense:0:0"
    )


@pytest.mark.parametrize("family", ["m062x", "mn15"])
@pytest.mark.parametrize("selection", ["unset", "0", "1"])
def test_point_program_without_capability_stays_dense(
    admission_probe: Path, family: str, selection: str
) -> None:
    assert _run(admission_probe, selection, family) == (
        "rejected" if selection == "1" else "dense:0:0"
    )


@pytest.mark.parametrize("selection", ["unset", "1"])
def test_host_budget_decline_keeps_dense_arithmetic_capability(
    admission_probe: Path, selection: str
) -> None:
    assert (
        _run(admission_probe, selection, mode="budget", precision="auto") == "dense:1:1"
    )


@pytest.mark.parametrize("selection", ["", "yes", "2", "-1", "01"])
def test_unknown_selection_fails_closed(admission_probe: Path, selection: str) -> None:
    assert _run(admission_probe, selection) == "rejected"
