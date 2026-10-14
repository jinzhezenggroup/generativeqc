"""Transactional FP32 shell runtime against independent contracted libcint forces."""

from __future__ import annotations

import ctypes
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda
from generativeqc_compiler.integral.df_rys import emit_df_rys_cuda
from generativeqc_compiler.integral.df_rys_shell import (
    RYS_SHELL_CLASSES,
    emit_df_rys_policy_cpp,
    emit_df_rys_shell_cuda,
)
from generativeqc_compiler.integral.df_screening import emit_sss_force_screening_cuda
from generativeqc_compiler.integral.df_shell_derivatives import (
    emit_df_shell_derivatives_cuda,
)

from tools.generativeqc_validation.df_derivatives import make_df_derivative_fixture

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RUN_CUDA_TESTS") != "1",
    reason="requires explicitly scheduled CUDA shell runtime qualification",
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", params=((0, 0, 0), (1, 1, 1), (2, 1, 3), (3, 3, 3)))
def native_shell_probe(
    request: pytest.FixtureRequest, tmp_path_factory: Any, native_cxx: Any
) -> tuple[tuple[int, int, int], Any]:
    """Keep compiled variants class-local, just as the production package does."""
    assert os.environ.get("SLURM_JOB_ID"), "native CUDA tests require finite srun"
    compiler = shutil.which("nvcc")
    assert compiler, "scheduled shell qualification requires nvcc"
    angular = request.param
    directory = tmp_path_factory.mktemp("df-native-fp32")
    for name, source in (
        ("generated_df_derivatives.cuh", emit_df_derivatives_cuda()),
        ("generated_df_screening.cuh", emit_sss_force_screening_cuda()),
        ("generated_df_rys.cuh", emit_df_rys_cuda()),
        ("generated_df_rys_policy.hpp", emit_df_rys_policy_cpp(classes=(angular,))),
        (
            "generated_df_shell_derivatives.cuh",
            "\n".join(
                emit_df_shell_derivatives_cuda(
                    classes=(angular,), scalar_type=scalar_type
                )
                for scalar_type in ("double", "float")
            ),
        ),
        (
            "generated_df_rys_shell.cuh",
            "\n".join(
                emit_df_rys_shell_cuda(
                    classes=(angular,),
                    shell_header="generated_df_shell_derivatives.cuh",
                    scalar_type=scalar_type,
                )
                for scalar_type in ("double", "float")
            ),
        ),
    ):
        (directory / name).write_text(source)
    basis = directory / "basis.o"
    native_cxx.compile_object(
        ROOT / "src/molecule/basis.cpp",
        basis,
        args=(
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ),
    )
    cuda_cxx = replace(native_cxx, compiler=compiler)
    binary = directory / "probe.so"
    cuda_cxx.build_shared(
        [ROOT / "tests/native/fixtures/df_shell_fp32_probe.cu"],
        binary,
        compile_args=(
            "-forward-unknown-to-host-compiler",
            "-std=c++20",
            "-O2",
            "-arch=native",
            "--ftz=false",
            "-I",
            str(directory),
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT),
            *(
                f"-DDF_TEST_ANGULAR_{axis}={value}"
                for axis, value in zip("ABC", angular, strict=True)
            ),
        ),
        link_args=(str(basis),),
        compile_timeout=240,
    )
    library = ctypes.CDLL(str(binary))
    vector = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    counter = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
    library.probe.argtypes = (
        ctypes.c_uint,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_uint,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_uint,
        ctypes.c_uint,
        vector,
        vector,
        vector,
        vector,
        vector,
        counter,
    )
    library.probe.restype = ctypes.c_int

    def evaluate(
        fixture: Any,
        weights: np.ndarray,
        *,
        variant: int = 2,
        rys: bool = False,
        mixed: bool = True,
        pairs: int = 0,
        packet: bool = False,
        spherical: bool = False,
        clipped: bool = False,
        fault: int = 0,
        replicas: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        shells = fixture.inputs["shells"]
        exponents = np.array(
            [primitive[0] for shell in shells for primitive in shell["primitives"]]
        )
        coefficients = np.array(
            [primitive[1] for shell in shells for primitive in shell["primitives"]]
        )
        output, counts = np.empty((3, 3)), np.empty(3, dtype=np.uint64)
        assert (
            library.probe(
                variant,
                rys,
                mixed,
                pairs,
                packet,
                spherical,
                clipped,
                fault,
                replicas,
                np.asarray(fixture.inputs["coordinates"]),
                exponents,
                coefficients,
                np.ascontiguousarray(weights),
                output,
                counts,
            )
            == 0
        )
        return output, counts

    return angular, evaluate


@pytest.mark.parametrize("spherical", (False, True))
@pytest.mark.parametrize("variant", (0, 1, 2))
def test_production_shell_schedules_match_libcint(
    native_shell_probe: Any, spherical: bool, variant: int
) -> None:
    """Cover real weight folding, primitive sums, partial panels and both launches."""
    angular, evaluate = native_shell_probe
    fixture = make_df_derivative_fixture(angular, primitive_lengths=(2, 2, 2))
    reference = fixture.spherical_reference if spherical else fixture.reference
    weights = np.random.default_rng(291).uniform(-0.7, 0.9, size=reference.shape[2:])
    for rys in (False, True) if angular in RYS_SHELL_CLASSES else (False,):
        for pairs in (0, 1, 2):
            for packet, clipped in ((False, False), (True, True)):
                visible = weights.copy()
                if clipped and weights.shape[-1] > 1:
                    visible[..., 0] = 0
                expected = np.einsum("acijk,ijk->ac", reference, visible)
                for mixed in (False, True):
                    actual, counts = evaluate(
                        fixture,
                        weights,
                        variant=variant,
                        rys=rys,
                        mixed=mixed,
                        pairs=pairs,
                        packet=packet,
                        spherical=spherical,
                        clipped=clipped,
                    )
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        atol=1e-5 if mixed else 8e-11,
                        rtol=3e-6 if mixed else 8e-11,
                    )
                    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=1e-12)
                    np.testing.assert_array_equal(
                        counts, [1, 8, 0] if mixed else [0, 0, 0]
                    )


@pytest.mark.parametrize("fault", (1, 2, 3))
def test_whole_task_retry_precedes_scatter(native_shell_probe: Any, fault: int) -> None:
    """Unsafe casts reject one whole subgroup, including already computed work."""
    angular, evaluate = native_shell_probe
    fixture = make_df_derivative_fixture(angular, primitive_lengths=(2, 2, 2))
    weights = np.random.default_rng(961).uniform(
        0.1, 0.9, size=fixture.reference.shape[2:]
    )
    for rys in (False, True) if angular in RYS_SHELL_CLASSES else (False,):
        strict, _ = evaluate(fixture, weights, mixed=False, rys=rys, fault=fault)
        actual, counts = evaluate(fixture, weights, rys=rys, fault=fault)
        assert np.isfinite(actual).all() and np.any(actual != 0)
        np.testing.assert_allclose(actual, strict, atol=0, rtol=2e-13)
        np.testing.assert_array_equal(counts, [1, 2 if fault == 2 else 1, 1])


def test_zero_work_is_not_counted_as_mixed(native_shell_probe: Any) -> None:
    angular, evaluate = native_shell_probe
    fixture = make_df_derivative_fixture(angular, primitive_lengths=(2, 2, 2))
    actual, counts = evaluate(fixture, np.zeros(fixture.reference.shape[2:]))
    np.testing.assert_array_equal(actual, 0)
    np.testing.assert_array_equal(counts, 0)


@pytest.mark.parametrize("fault", (0, 2))
def test_concurrent_subgroups_and_tail(native_shell_probe: Any, fault: int) -> None:
    """Multiple active subgroup owners share a warp without sharing retry state."""
    angular, evaluate = native_shell_probe
    fixture = make_df_derivative_fixture(angular, primitive_lengths=(2, 2, 2))
    weights = np.random.default_rng(562).uniform(
        0.1, 0.9, size=fixture.reference.shape[2:]
    )
    for rys in (False, True) if angular in RYS_SHELL_CLASSES else (False,):
        strict, _ = evaluate(fixture, weights, rys=rys, mixed=False, fault=fault)
        for packet in (False, True):
            actual, counts = evaluate(
                fixture, weights, rys=rys, fault=fault, replicas=3, packet=packet
            )
            np.testing.assert_allclose(actual, 3 * strict, atol=1e-5, rtol=3e-6)
            np.testing.assert_array_equal(counts, [3, 6, 3] if fault else [3, 24, 0])
