"""The topology bootstrap does no CPU molecular evaluation or hidden force call."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _density_diagnostic_mode(library: Path) -> bool:
    requested = os.environ.get("GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS") == "1"
    cache = (
        Path(os.environ.get("GENERATIVEQC_BUILD_DIRECTORY", library.parent))
        / "CMakeCache.txt"
    )
    assert cache.is_file(), "density timing fixture requires matching build metadata"
    match = re.search(
        r"^GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS:BOOL=(.*)$",
        cache.read_text(),
        re.MULTILINE,
    )
    assert match is not None, "density diagnostic build mode is missing"
    value = match.group(1).strip().upper()
    assert value in {"ON", "OFF"}, "density diagnostic build mode is unrecognized"
    enabled = value == "ON"
    assert enabled == requested, (
        "density timing fixture must link a matching ON/OFF build"
    )
    return requested


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("built", [False, True])
def test_density_timing_build_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested: bool, built: bool
) -> None:
    monkeypatch.delenv("GENERATIVEQC_BUILD_DIRECTORY", raising=False)
    monkeypatch.delenv("GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS", raising=False)
    if requested:
        monkeypatch.setenv("GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS", "1")
    (tmp_path / "CMakeCache.txt").write_text(
        "GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS:BOOL="
        + ("ON" if built else "OFF")
        + "\n"
    )
    library = tmp_path / "libgenerativeqc.so"
    if requested == built:
        assert _density_diagnostic_mode(library) == requested
    else:
        with pytest.raises(AssertionError, match="matching ON/OFF build"):
            _density_diagnostic_mode(library)


@pytest.mark.parametrize(
    "contents",
    [None, "", "GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS:BOOL=unknown\n"],
)
def test_density_timing_rejects_unverified_build_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str | None
) -> None:
    monkeypatch.delenv("GENERATIVEQC_BUILD_DIRECTORY", raising=False)
    monkeypatch.delenv("GENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS", raising=False)
    if contents is not None:
        (tmp_path / "CMakeCache.txt").write_text(contents)
    with pytest.raises(AssertionError, match="metadata|missing|unrecognized"):
        _density_diagnostic_mode(tmp_path / "libgenerativeqc.so")


@pytest.mark.parametrize("diagnostic", [False, True])
def test_density_native_timing_rejects_mismatched_artifact(
    tmp_path: Path, required_native_cxx: NativeCxx, diagnostic: bool
) -> None:
    """Run the native pre-timing gate with both library capability values."""
    native = (ROOT / "tests/native/test_gfn2_cuda_bootstrap.cpp").read_text()
    prefix = native.split("void verify_density_endpoint_timing(", 1)[1].split(
        "  oracle(run(cache, molecule, device_input), molecule);", 1
    )[0]
    source = tmp_path / "timing_mode.cpp"
    source.write_text(
        "#include <stdexcept>\n#include <string>\n"
        "bool library_diagnostic = false;\n"
        "struct Molecule { std::string name; };\n"
        "struct Gfn2CudaExecutionCache {\n"
        "  Gfn2CudaExecutionCache(int, void*) {}\n"
        "  bool enable_density_diagnostics() { return library_diagnostic; }\n"
        "};\n"
        "void require(bool condition, const char* message) {\n"
        "  if (!condition) throw std::runtime_error(message);\n}\n"
        "void verify_density_endpoint_timing(" + prefix + "  (void)arm;\n}\n"
        "int main(int argc, char**) {\n"
        "  library_diagnostic = argc > 1;\n"
        '  try { verify_density_endpoint_timing({"h2o"}, false); }\n'
        "  catch (const std::runtime_error&) { return 1; }\n"
        "  return 0;\n}\n"
    )
    flags = ["-std=c++17"]
    if diagnostic:
        flags.append("-DGENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS=1")
    binary = required_native_cxx.build_executable(
        [source], tmp_path / "timing_mode", compile_args=flags
    )
    for library_diagnostic in (False, True):
        command = [str(binary), *(["on"] if library_diagnostic else [])]
        result = subprocess.run(command, check=False, timeout=10)
        assert result.returncode == (0 if diagnostic == library_diagnostic else 1)


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
    diagnostic = _density_diagnostic_mode(library)
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
