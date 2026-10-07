"""Actual shared LP64 loader/ABI ownership and fail-closed cohort checks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
PROVIDER = ROOT / "src/tensor/cpu/lp64_provider.cpp"
GFN = ROOT / "src/xtb/native/src/model/gfn2/eigensolver.cpp"


@pytest.fixture(scope="module")
def loader_builds(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
):
    folder = tmp_path_factory.mktemp("lp64-loader")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_weighted_gram_native.py"),
            "--output",
            str(folder / "generated_weighted_gram_native.hpp"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    common = [
        "-std=c++17",
        "-O2",
        "-ffunction-sections",
        "-fdata-sections",
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "src/xtb/native/src"),
        "-I" + str(folder),
    ]
    binaries = {}
    for mode in ("system", "configured", "private"):
        destination = folder / mode
        destination.mkdir()
        definitions = []
        if mode == "configured":
            definitions = [
                '-DGENERATIVEQC_XTB_CONFIGURED_CPU_LINALG_RUNTIME="/configured/reviewed-runtime.so"'
            ]
        elif mode == "private":
            definitions = [
                "-DGENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS=1",
                '-DGENERATIVEQC_XTB_CONFIGURED_WHEEL_OPENBLAS_CONFIG_PREFIX="OpenBLAS 0.test"',
            ]
        # Deliberately define the loader profile only on the moved provider TU.
        # The real method TU must link and map statuses without owning the loader.
        objects = []
        for index, source in enumerate(
            (ROOT / "tests/native/test_cpu_lp64_loader.cpp", GFN, PROVIDER)
        ):
            obj = destination / f"{index}.o"
            required_native_cxx.compile_object(
                source, obj, args=common + (definitions if source == PROVIDER else [])
            )
            objects.append(obj)
        binaries[mode] = required_native_cxx.link(
            objects,
            destination / "loader",
            args=[
                "-pthread",
                "-ldl",
                "-Wl,--gc-sections",
                "-Wl,--wrap=dlopen",
                "-Wl,--wrap=dlclose",
                "-Wl,--wrap=dlmopen",
            ],
        )
        binaries[mode] = destination / "loader"
    libraries = {}
    cases = {
        "standard": (True, []),
        "prefixed": (True, ["-DPREFIXED=1"]),
        "prefixed_threads": (True, ["-DPREFIXED=1", "-DPREFIX_THREADS=1"]),
        "mixed": (False, ["-DMIXED=1"]),
        **{f"missing_{i}": (False, [f"-DOMIT={i}"]) for i in range(1, 8)},
        **{f"failure_{i}": (False, [f"-DFAIL={i}"]) for i in range(1, 11)},
    }
    for name, (accepted, flags) in cases.items():
        library = required_native_cxx.build_shared(
            [ROOT / "tests/native/fixtures/cpu_lp64/mock_provider.cpp"],
            folder / f"{name}.so",
            compile_args=["-std=c++17", "-O2", *flags],
        )
        libraries[name] = (library, accepted)
    # A complete, globally visible rejecting prefixed cohort is the private
    # isolation adversary, not a second accepted provider.
    libraries["poison"] = (
        required_native_cxx.build_shared(
            [ROOT / "tests/native/fixtures/cpu_lp64/mock_provider.cpp"],
            folder / "poison.so",
            compile_args=["-std=c++17", "-O2", "-DPREFIXED=1", "-DFAIL=1"],
        ),
        False,
    )
    return binaries, libraries


def run_loader(
    binary: Path,
    mode: str,
    accepted: bool,
    first: Path,
    fallback: Path,
    host: str | Path = "none",
) -> str:
    result = subprocess.run(
        [str(binary), mode, str(int(accepted)), str(first), str(fallback), str(host)],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


@pytest.mark.parametrize(
    "case",
    [
        "standard",
        "prefixed",
        "prefixed_threads",
        "mixed",
        *(f"missing_{i}" for i in range(1, 8)),
        *(f"failure_{i}" for i in range(1, 11)),
    ],
)
def test_actual_runtime_cohort_admission(loader_builds, case: str) -> None:
    binaries, libraries = loader_builds
    path, accepted = libraries[case]
    run_loader(binaries["system"], "system", accepted, path, path)


def test_ordinary_configured_failure_retains_system_fallback(loader_builds) -> None:
    binaries, libraries = loader_builds
    output = run_loader(
        binaries["configured"],
        "configured",
        True,
        libraries["missing_1"][0],
        libraries["standard"][0],
    )
    assert "opens=2 closes=1" in output


def test_configured_runtime_success_retains_priority(loader_builds) -> None:
    binaries, libraries = loader_builds
    output = run_loader(
        binaries["configured"],
        "configured",
        True,
        libraries["standard"][0],
        libraries["poison"][0],
    )
    assert "opens=1 closes=0" in output


@pytest.mark.parametrize(
    "case,accepted",
    [("prefixed", True), ("poison", False), ("standard", False), ("failure_10", False)],
)
def test_private_cohort_isolation_and_no_fallback(
    loader_builds, case: str, accepted: bool
) -> None:
    binaries, libraries = loader_builds
    shim = binaries["private"].parent / "libgenerativeqc_xtb_openblas_lp64_shim.so"
    shutil.copyfile(libraries[case][0], shim)
    run_loader(
        binaries["private"],
        "private",
        accepted,
        libraries[case][0],
        libraries["prefixed"][0],
        libraries["poison"][0],
    )


def test_no_method_owned_loader_or_raw_unwrapping() -> None:
    source = GFN.read_text()
    for retired in (
        "CpuLinearAlgebraAccess",
        "load_lapacke_cblas_symbols",
        "backend_self_test",
        "dlopen(",
        "dlsym(",
        "dlmopen(",
        "set_num_threads_local_",
        "class ScopedSequentialBlas",
        "class CpuLinearAlgebraBackend",
    ):
        assert retired not in source
    assert "cpu_provider::bind_gemm(backend)" in source
    assert "cpu_provider::bind_symmetric_eigen(backend)" in source
    assert "cpu_provider::cholesky_lower(" in source
    assert "cpu_provider::reciprocal_condition_lower(" in source
    assert "cpu_provider::solve_lower_triangular(" in source
    shared = PROVIDER.read_text()
    assert "generativeqc_xtb_status_t" not in shared
    for method in (
        "compute_occupations",
        "minimum_overlap_rcond",
        "EigensolverPlan",
        "Wavefunction",
        "commit_batch_solve_results",
    ):
        assert method not in shared


def test_gpu_and_scientific_body_ownership_stays_unchanged() -> None:
    base = "dea1656"
    changes = subprocess.check_output(
        ["git", "diff", "--name-only", base, "--", "src"], cwd=ROOT, text=True
    ).splitlines()
    assert not any(Path(path).suffix in (".cu", ".cuh") for path in changes)
    old = subprocess.check_output(
        ["git", "show", f"{base}:src/tensor/cpu_linalg.cpp"], cwd=ROOT
    )
    assert old == (ROOT / "src/tensor/cpu_linalg.cpp").read_bytes()
    for directory in (
        "python/generativeqc_compiler",
        "src/scf/solver",
        "src/scf/rhf.cpp",
    ):
        assert not subprocess.check_output(
            ["git", "diff", base, "--", directory], cwd=ROOT
        )
