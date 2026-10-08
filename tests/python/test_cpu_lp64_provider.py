"""Actual shared LP64 loader/ABI ownership and fail-closed cohort checks."""

from __future__ import annotations

import hashlib
import json
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
LoaderBuilds = tuple[dict[str, Path], dict[str, tuple[Path, bool]]]


@pytest.fixture(scope="module")
def loader_builds(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> LoaderBuilds:
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
        required_native_cxx.link(
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
def test_actual_runtime_cohort_admission(
    loader_builds: LoaderBuilds, case: str
) -> None:
    binaries, libraries = loader_builds
    path, accepted = libraries[case]
    run_loader(binaries["system"], "system", accepted, path, path)


def test_ordinary_configured_failure_retains_system_fallback(
    loader_builds: LoaderBuilds,
) -> None:
    binaries, libraries = loader_builds
    output = run_loader(
        binaries["configured"],
        "configured",
        True,
        libraries["missing_1"][0],
        libraries["standard"][0],
    )
    assert "opens=2 closes=1" in output


def test_configured_runtime_success_retains_priority(
    loader_builds: LoaderBuilds,
) -> None:
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
    loader_builds: LoaderBuilds, case: str, accepted: bool
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
    assert "cpu_provider::solve_lower_triangular(" not in source
    assert "solver/cpu/generalized_eigen.hpp" in source
    assert "reduce_generalized_eigen(basis, lowering)" in source
    assert "recover_generalized_eigen(basis, lowering)" in source
    public_header = (PROVIDER.with_suffix(".hpp")).read_text()
    assert "struct CpuLinearAlgebraAccess {" not in public_header
    assert "friend struct CpuLinearAlgebraAccess;" in public_header
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


def _definition(source: str, prefix: str, occurrence: int) -> str:
    """Read the reviewed complete C++ region; no Git checkout is required."""
    start = -1
    for _ in range(occurrence + 1):
        start = source.index(prefix, start + 1)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def _contract_region(source: str, region: dict[str, object]) -> str:
    prefix, occurrence = region["prefix"], region["occurrence"]
    assert isinstance(prefix, str) and isinstance(occurrence, int)
    body = _definition(source, prefix, occurrence)
    if marker := region.get("suffix_from"):
        assert isinstance(marker, str) and body.count(marker) == 1
        body = body[body.index(marker) :]
    return body


def test_reviewed_scientific_and_admission_contracts() -> None:
    contract = json.loads(
        (ROOT / "tests/data/cpu_lp64_source_contract.json").read_text()
    )
    assert contract["schema"] == 1
    for path, expected in contract["files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    for region in contract["regions"]:
        source = (ROOT / region["path"]).read_text()
        body = _contract_region(source, region)
        assert hashlib.sha256(body.encode()).hexdigest() == region["sha256"], region


def test_cpu_bindings_are_not_gpu_consumers() -> None:
    for pattern in ("*.cu", "*.cuh"):
        for path in (ROOT / "src").rglob(pattern):
            source = path.read_text()
            for header in (
                "tensor/cpu/lp64_provider.hpp",
                "solver/cpu/symmetric_eigen.hpp",
                "tensor/weighted_gram.hpp",
            ):
                assert f'#include "{header}"' not in source, path


def test_rhf_contract_tracks_eigen_selection_without_freezing_other_admission() -> None:
    contract = json.loads(
        (ROOT / "tests/data/cpu_lp64_source_contract.json").read_text()
    )
    region = next(
        item for item in contract["regions"] if item["path"] == "src/scf/rhf.cpp"
    )
    source = (ROOT / region["path"]).read_text()
    selected = _contract_region(source, region)
    assert source.count(selected) == 1
    # Independent physical-reference budget checks may evolve before this boundary.
    outside = source.replace(
        selected, "/* independent reference admission */\n" + selected, 1
    )
    assert _contract_region(outside, region) == selected
    for before, after in (
        ("!options.export_physical_reference", "options.export_physical_reference"),
        ("solver::cpu_target_eigen", "reference::generalized_eigen"),
        ("nullptr, target_eigen", "nullptr, initial_guess::EigenOperation{}"),
    ):
        assert before in selected
        altered = source.replace(selected, selected.replace(before, after, 1), 1)
        assert _contract_region(altered, region) != selected


@pytest.fixture(scope="module")
def dispatch_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    folder = tmp_path_factory.mktemp("lp64-dispatch")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_weighted_gram_native.py"),
            "--output",
            str(folder / "generated_weighted_gram_native.hpp"),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    args = [
        "-std=c++20",
        "-O2",
        "-ffunction-sections",
        "-fdata-sections",
        "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "src/xtb/native/src"),
        "-I" + str(folder),
    ]
    sources = [
        ROOT / "tests/native/test_cpu_lp64_dispatch.cpp",
        PROVIDER,
        ROOT / "src/tensor/cpu_linalg.cpp",
        ROOT / "src/scf/solver/cpu_target_eigen.cpp",
        ROOT / "src/scf/solver/eigen_frame.cpp",
        ROOT / "src/scf/reference/linalg.cpp",
    ]
    objects = []
    for index, source in enumerate(sources):
        flags = list(args)
        if source == PROVIDER and (
            library := os.environ.get("GENERATIVEQC_TEST_LP64_LIBRARY")
        ):
            flags.append(
                f'-DGENERATIVEQC_XTB_CONFIGURED_CPU_LINALG_RUNTIME="{library}"'
            )
        obj = folder / f"{index}.o"
        required_native_cxx.compile_object(source, obj, args=flags)
        objects.append(obj)
    binary = folder / "probe"
    required_native_cxx.link(objects, binary, args=["-Wl,--gc-sections", "-ldl"])
    return binary


def test_actual_gfn_primitive_dispatch_and_thread_lifetime(
    dispatch_probe: Path,
) -> None:
    subprocess.run(
        [str(dispatch_probe), "dispatch"], check=True, capture_output=True, timeout=15
    )


def test_actual_real_lp64_and_gaussian_scalar_consumers(dispatch_probe: Path) -> None:
    if not os.environ.get("GENERATIVEQC_TEST_LP64_LIBRARY"):
        pytest.skip(
            "set GENERATIVEQC_TEST_LP64_LIBRARY for explicit real-provider qualification"
        )
    subprocess.run(
        [str(dispatch_probe), "real"], check=True, capture_output=True, timeout=30
    )


def test_borrowed_eigen_contract(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Exact N-derived counts, invalid bindings, no allocation, and raw info."""
    binary = required_native_cxx.build_executable(
        [ROOT / "tests/native/test_cpu_borrowed_eigen_contract.cpp"],
        tmp_path / "borrowed",
        compile_args=["-std=c++17", "-O2", "-I" + str(ROOT / "src")],
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=15)


def test_opaque_binding_types(tmp_path: Path, required_native_cxx: NativeCxx) -> None:
    source = tmp_path / "opaque.cpp"
    source.write_text(
        '#include "tensor/cpu/lp64_provider.hpp"\n#include <type_traits>\n'
        "namespace cpu=generativeqc::tensor::cpu;\n"
        "static_assert(sizeof(cpu::LapackInt)==4);\n"
        "static_assert(!std::is_convertible_v<cpu::GemmBinding,cpu::CblasDgemm>);\n"
        "static_assert(!std::is_convertible_v<cpu::SymmetricEigenBinding,cpu::LapackDsyevdWork>);\n"
        "int main(){}\n"
    )
    binary = required_native_cxx.build_executable(
        [source],
        tmp_path / "opaque",
        compile_args=["-std=c++17", "-pedantic-errors", "-I" + str(ROOT / "src")],
    )
    subprocess.run([str(binary)], check=True, timeout=10)


@pytest.mark.parametrize(
    "access",
    [
        "cpu::CpuLinearAlgebraAccess::dsyevd(backend)",
        "backend.dsyevd_work_",
        "cpu::bind_symmetric_eigen(backend).call_",
        "cpu::bind_gemm(backend).call_",
    ],
)
def test_raw_provider_access_is_not_public(
    tmp_path: Path, required_native_cxx: NativeCxx, access: str
) -> None:
    source = tmp_path / "private.cpp"
    source.write_text(
        '#include "tensor/cpu/lp64_provider.hpp"\n'
        "namespace cpu=generativeqc::tensor::cpu;\n"
        "int main(){cpu::CpuLinearAlgebraBackend backend;\n"
        f"(void)({access});"
        "}\n"
    )
    with pytest.raises(subprocess.CalledProcessError) as failure:
        required_native_cxx.compile_object(
            source,
            tmp_path / "private.o",
            args=["-std=c++17", "-I" + str(ROOT / "src")],
        )
    assert "private.cpp" in failure.value.stderr


@pytest.mark.parametrize("local_threads", (False, True))
def test_build_linked_owned_eigen_profile(
    tmp_path: Path, required_native_cxx: NativeCxx, local_threads: bool
) -> None:
    binary = required_native_cxx.build_executable(
        [
            ROOT / "tests/native/test_cpu_linked_eigen_profile.cpp",
            ROOT / "src/tensor/cpu_linalg.cpp",
            ROOT / "src/scf/solver/cpu_target_eigen.cpp",
            ROOT / "src/scf/solver/eigen_frame.cpp",
            ROOT / "src/scf/reference/linalg.cpp",
        ],
        tmp_path / "linked",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            "-I" + str(ROOT / "tests/native/fixtures/cpu_linked_provider"),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-DGENERATIVEQC_HAS_OPENBLAS=1",
            "-DGENERATIVEQC_OPENBLAS_HAS_LAPACKE=1",
            f"-DGENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS={int(local_threads)}",
            "-DGENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS=1",
        ],
        link_args=["-Wl,--gc-sections"],
    )
    subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=15
    )
