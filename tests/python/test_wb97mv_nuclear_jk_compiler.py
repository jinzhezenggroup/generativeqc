"""Independent signed-density tests of build-time quartet AD and emitted C++."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import pytest
from generativeqc_compiler.method.wb97mv_nuclear_jk import (
    PARTS,
    quartet_operator,
    quartet_program,
)

from tools.generate_wb97mv_nuclear_jk import generate


@pytest.fixture(scope="module")
def quartet_library(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., bool]:
    """Host-compile the same generated FP64 maps linked by the CUDA owner."""
    compiler = shutil.which("c++")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if not compiler or not launcher:
        pytest.skip("C++ compiler and verified compiler cache are required")
    subprocess.run([launcher, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("quartet-ad")
    source, library = directory / "maps.cpp", directory / "maps.so"
    source.write_text(
        "#include <cmath>\nusing std::isfinite;\n"
        "#define __device__\n#define __forceinline__ inline\n"
        + generate()
        + '\nextern "C" bool evaluate(unsigned kernel, unsigned spins, '
        "const double* density, const double* delta, std::size_t nao, "
        "const std::size_t* ao, double value, double motion, double* output) {\n"
        "using namespace generativeqc_wb97mv_nuclear_jk_ad; Result result;\n"
        "if (!evaluate(static_cast<Kernel>(kernel),spins,density,delta,nao,ao,value,motion,result)) return false;\n"
        "output[0]=result.energy; output[1]=result.weight; output[2]=result.delta_weight;\n"
        "for (unsigned index=0; index<8; ++index) { output[3+index]=result.fock[index]; output[11+index]=result.motion[index]; }\n"
        "return true; }\n"
    )
    subprocess.run(
        [
            launcher,
            compiler,
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-shared",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    loaded = ctypes.CDLL(str(library))
    evaluate = loaded.evaluate
    panel = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    evaluate.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        panel,
        panel,
        ctypes.c_size_t,
        np.ctypeslib.ndpointer(dtype=np.uintp, flags="C_CONTIGUOUS"),
        ctypes.c_double,
        ctypes.c_double,
        panel,
    ]
    evaluate.restype = ctypes.c_bool
    return evaluate


def _oracle(
    spins: int,
    part: str,
    density: np.ndarray,
    delta: np.ndarray,
    ao: tuple[int, ...],
    value: float,
    motion: float,
) -> np.ndarray:
    """Published WB97M-V fractions: SR=.15, LR=1; RKS D is spin-summed."""
    coulomb = 0.5 if part in ("primary", "coulomb") else 0.0
    exchange = {
        "primary": 0.15,
        "correction": 0.85,
        "coulomb": 0.0,
        "short": 0.15,
        "long": 1.0,
    }[part] / (-4 if spins == 1 else -2)
    first, second, third, fourth = ao
    left, right = density[:, first, second], density[:, third, fourth]
    left_delta, right_delta = delta[:, first, second], delta[:, third, fourth]
    exchange_left, exchange_right = density[:, first, third], density[:, second, fourth]
    exchange_left_delta, exchange_right_delta = (
        delta[:, first, third],
        delta[:, second, fourth],
    )
    weight = coulomb * left.sum() * right.sum() + exchange * (
        exchange_left @ exchange_right
    )
    delta_weight = coulomb * (
        left_delta.sum() * right.sum() + left.sum() * right_delta.sum()
    )
    delta_weight += exchange * (
        exchange_left_delta @ exchange_right + exchange_left @ exchange_right_delta
    )
    incidence = np.zeros((4, 2))
    incidence[0, :spins], incidence[1, :spins] = (
        coulomb * right.sum(),
        coulomb * left.sum(),
    )
    incidence[2, :spins], incidence[3, :spins] = (
        exchange * exchange_right,
        exchange * exchange_left,
    )
    return np.r_[
        weight * value,
        weight,
        delta_weight,
        (incidence * value).ravel(),
        (incidence * motion).ravel(),
    ]


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("part", PARTS)
def test_compiled_quartet_ad_against_independent_spin_equations(
    quartet_library: Callable[..., bool],
    spins: int,
    part: str,
) -> None:
    random = np.random.default_rng(27182)
    for ao in ((0, 1, 2, 3), (2, 2, 2, 2), (3, 0, 0, 3), (1, 3, 2, 0)):
        density = random.normal(size=(spins, 4, 4))
        delta = random.normal(size=(spins, 4, 4))
        output = np.full(19, np.nan)
        assert quartet_library(
            PARTS.index(part),
            spins,
            density,
            delta,
            4,
            np.array(ao, dtype=np.uintp),
            -0.71,
            0.39,
            output,
        )
        np.testing.assert_allclose(
            output,
            _oracle(spins, part, density, delta, ao, -0.71, 0.39),
            rtol=3e-14,
            atol=3e-14,
        )


def test_compiled_quartet_fails_closed_on_nonfinite_and_invalid_selectors(
    quartet_library: Callable[..., bool],
) -> None:
    density, delta = np.ones((2, 2, 2)), np.zeros((2, 2, 2))
    ao, output = np.zeros(4, dtype=np.uintp), np.zeros(19)
    for kernel, spins, value in (
        (99, 1, 1.0),
        (0, 3, 1.0),
        (0, 1, np.nan),
        (0, 2, np.inf),
    ):
        assert not quartet_library(
            kernel, spins, density, delta, 2, ao, value, 1.0, output
        )
    delta.fill(np.nan)
    assert not quartet_library(0, 2, density, delta, 2, ao, 1.0, 1.0, output)


def test_quartet_scientific_selectors_and_ad_admission() -> None:
    assert quartet_operator("primary") == ("full_range", 0.0)
    assert quartet_operator("short") == ("short_range", 0.3)
    assert quartet_operator("correction") == ("long_range", 0.3)
    for spins, part in ((0, "primary"), (True, "primary"), (2, "exact")):
        with pytest.raises(ValueError, match="invalid WB97M-V"):
            quartet_program(spins, part)
    with pytest.raises(ValueError, match="invalid WB97M-V"):
        quartet_operator("exact")


def test_generated_quartet_header_is_reproducible() -> None:
    assert generate() == generate()
