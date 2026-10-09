"""The topology bootstrap does no CPU molecular evaluation or hidden force call."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_cuda_setup_does_not_evaluate_a_cpu_molecule() -> None:
    """Keep reference evaluators out of this production preparation owner."""
    source = (ROOT / "src/xtb/native/src/runtime/gfn2_cuda_execution.cu").read_text()
    assert not re.search(r"\b(?:evaluate|update)_\w+_cpu\s*\(", source)
    binding = source.split("build_energy_force_bindings(", 1)[1].split(
        "build_inference_bindings(", 1
    )[0]
    assert "execute_gfn2_energy_force_cuda(" not in binding


def test_cuda_first_transaction_and_recovery(tmp_path: Path) -> None:
    """Run the production private transaction with host/device input and failures."""
    if os.environ.get("GENERATIVEQC_TEST_GFN2_CUDA") != "1":
        pytest.skip("explicit GFN2 CUDA qualification is disabled")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("GFN2 CUDA qualification requires Slurm")
    compiler, nvcc = (shutil.which(name) for name in ("c++", "nvcc"))
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if compiler is None or nvcc is None:
        pytest.skip("C++ compiler and CUDA toolkit required")
    if launcher is None:
        pytest.fail("sccache or ccache is required for CUDA qualification builds")
    subprocess.run([launcher, "--version"], check=True, capture_output=True)
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    diagnostic = os.environ.get("GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS") == "1"
    if diagnostic:
        cache = (
            Path(os.environ.get("GENERATIVEQC_BUILD_DIRECTORY", library.parent))
            / "CMakeCache.txt"
        )
        assert cache.is_file() and (
            "GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS:BOOL=ON" in cache.read_text()
        ), "diagnostic fixture must link a matching opt-in build"
    # The installed/native Linux SONAME is stable even when the qualification
    # runner copied the library under a receipt-specific filename.
    (tmp_path / "libgenerativeqc.so.0").symlink_to(library)
    toolkit = Path(nvcc).resolve().parents[1]
    binary = tmp_path / "bootstrap"
    object_file = tmp_path / "bootstrap.o"
    subprocess.run(
        [
            launcher,
            compiler,
            "-std=c++17",
            "-O2",
            *(["-DGENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS=1"] if diagnostic else []),
            "-I",
            str(toolkit / "include"),
            "-I",
            str(ROOT / "src/xtb/native/src"),
            "-c",
            str(ROOT / "tests/native/test_gfn2_cuda_bootstrap.cpp"),
            "-o",
            str(object_file),
        ],
        check=True,
        timeout=180,
    )
    subprocess.run(
        [
            compiler,
            str(object_file),
            str(library),
            "-L",
            str(toolkit / "lib64"),
            "-lcudart",
            f"-Wl,-rpath,{tmp_path}",
            f"-Wl,-rpath,{toolkit / 'lib64'}",
            "-o",
            str(binary),
        ],
        check=True,
        timeout=180,
    )
    override = os.environ.get("GENERATIVEQC_GFN2_BOOTSTRAP_ORACLE")
    if override:
        fixture = Path(override).resolve()
        assert fixture.is_file(), "requested independent GFN2 oracle is missing"
    else:
        fixtures = json.loads(
            (ROOT / "tests/data/gfn2_native_tblite.json").read_text()
        )["cases"]
        numbers = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "Cl": 17}
        lines = [str(len(fixtures))]
        for case in fixtures:
            lines.append(f"{case['name']} {len(case['symbols'])} {case['energy']:.17g}")
            for symbol, position, force in zip(
                case["symbols"], case["positions"], case["forces"], strict=True
            ):
                lines.append(" ".join(map(str, [numbers[symbol], *position, *force])))
        fixture = tmp_path / "oracle.txt"
        fixture.write_text("\n".join(lines) + "\n")
    subprocess.run([str(binary), str(fixture)], check=True, timeout=180)
