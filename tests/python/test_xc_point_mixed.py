"""Independent interior point gates; these do not qualify molecular Fast."""

import ctypes
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_installed_fp32_point_headers_compile(native_cxx: Any, tmp_path: Path) -> None:
    """The typed point path must work without checkout include fallbacks."""
    entries = dict(
        re.findall(
            r'^"([^"]+)"\s*=\s*"([^"]+)"$',
            (ROOT / "pyproject.toml").read_text(),
            re.MULTILINE,
        )
    )
    for relative in ("src/dft/xc_point.hpp", "src/dft/xc_point_mixed.hpp"):
        target = tmp_path / entries[relative]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    native_cxx.build_shared(
        [ROOT / "tests/native/xc_point_mixed_probe.cpp"],
        tmp_path / "point.so",
        compile_args=(
            "-std=c++17",
            "-O2",
            f"-I{tmp_path / 'generativeqc_compiler/assets/src'}",
        ),
    )
    from generativeqc_compiler.dft.ao_cuda import emit_grid_source

    assert ROOT / "src/dft/xc_point_mixed.hpp" in emit_grid_source()[2]


@pytest.fixture(scope="module", params=("host", "cuda"))
def point_probe(
    native_cxx: Any,
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Any:
    """Bind only the point equation, without loading a production library."""
    directory = tmp_path_factory.mktemp("mixed-point")
    path = directory / "point.so"
    compiler = native_cxx
    suffix = "cpp"
    flags = ("-std=c++20", "-O2", "-ffp-contract=off", f"-I{ROOT / 'src'}")
    if request.param == "cuda":
        if os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") != "1":
            pytest.skip("scheduled CUDA qualification is opt-in")
        if not os.environ.get("SLURM_JOB_ID"):
            pytest.fail("CUDA qualification requires finite srun")
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            pytest.fail("scheduled CUDA qualification requires nvcc")
        compiler = replace(native_cxx, compiler=nvcc)
        suffix = "cu"
        flags = (
            "-forward-unknown-to-host-compiler",
            "-std=c++20",
            "-O2",
            "--fmad=false",
            "--ftz=false",
            "-arch=native",
            f"-I{ROOT / 'src'}",
        )
    compiler.build_shared(
        [ROOT / f"tests/native/xc_point_mixed_probe.{suffix}"],
        path,
        compile_args=flags,
    )
    library = ctypes.CDLL(str(path))
    function = library.mixed_point
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    function.argtypes = (pointer, pointer, ctypes.c_int)
    function.restype = ctypes.c_int

    def evaluate(
        features: Any,
        mixed: bool | int = True,
        exchange: float = 0.75,
        correlation: float = 1.0,
    ) -> tuple[int, np.ndarray]:
        inputs = np.asarray([*np.asarray(features).ravel(), exchange, correlation])
        output = np.empty(9)
        status = function(inputs, output, int(mixed))
        return status, output

    return evaluate


@pytest.mark.parametrize("density", [0.0, 1e-30, 1e-13, 1e7])
@pytest.mark.parametrize("mode", (1, 2))
def test_tail_retains_exact_strict_equation(
    point_probe: Any, density: float, mode: int
) -> None:
    features = np.array([[density, 0.0, 0.0, 0.0]] * 2)
    status, mixed = point_probe(features, mixed=mode)
    strict_status, strict = point_probe(features, mixed=False)
    assert status == strict_status == 0
    np.testing.assert_array_equal(mixed, strict)


def test_unequal_spin_and_invalid_inputs_are_not_narrowed(point_probe: Any) -> None:
    features = np.array([[0.4, 0.1, -0.2, 0.3], [0.3, 0.1, -0.2, 0.3]])
    status, mixed = point_probe(features)
    strict_status, strict = point_probe(features, mixed=False)
    assert status == strict_status == 0
    np.testing.assert_array_equal(mixed, strict)
    for density in (-1.0, np.inf, np.nan):
        features[:, 0] = density
        assert point_probe(features)[0] == -1


@pytest.mark.parametrize(("mode", "spin_ratio"), ((1, 1.0), (2, 1.0), (2, 0.3)))
def test_interior_executes_fp32_with_independent_libxc_gate(
    point_probe: Any, mode: int, spin_ratio: float
) -> None:
    """Use independent Libxc E/vxc, including Cartesian gradient conversion."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    generator = np.random.default_rng(20261013)
    maximum = np.zeros(9)
    for density in np.geomspace(1e-10, 1e3, 17):
        for ratio in (0.0, 1e-3, 1.0, 1e3):
            gradient = generator.normal(size=3) * density * ratio
            features = np.array([[density, *gradient]] * 2)
            features[1] *= spin_ratio
            status, actual = point_probe(features, mixed=mode)
            assert status == 1
            energy, potential, *_ = libxc.eval_xc(
                "0.75*GGA_X_PBE+GGA_C_PBE", features[:, :, None], spin=1, deriv=1
            )
            vrho, vsigma = potential[:2]
            expected = np.empty(9)
            expected[0] = energy[0] * (1 + spin_ratio) * density
            for spin in range(2):
                expected[1 + 4 * spin] = vrho[0, spin]
                expected[2 + 4 * spin : 5 + 4 * spin] = (
                    2 * vsigma[0, 2 * spin] * features[spin, 1:]
                    + vsigma[0, 1] * features[1 - spin, 1:]
                )
            maximum = np.maximum(maximum, abs(actual - expected))
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-7)
    assert np.any(maximum > 0)
