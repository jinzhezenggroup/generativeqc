"""Real CPU/CUDA Johnson consumers against an independent chronological oracle.

The oracle materializes a chronological list and uses NumPy's dense solve. It
never calls the ordered-history IR, interpreter, emitter, or generated helpers.
CUDA execution is optional and explicitly skipped when nvcc/device is absent;
compilation or numerical errors on an available toolchain are test failures.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "src/xtb/native/src"
OFFSETS = (0, 10, 31, 62)
TOTAL = OFFSETS[-1]
DAMPING = 0.3
DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
UINT64_PTR = ctypes.POINTER(ctypes.c_uint64)


@pytest.fixture(scope="module")
def consumer_includes(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("ordered-history-consumer-generated")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_ordered_history_native.py"),
            "--output-directory",
            str(directory),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
    )
    return directory


def _load(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.consumer_available.restype = ctypes.c_int
    library.consumer_create.argtypes = [ctypes.c_int, ctypes.c_int]
    library.consumer_create.restype = ctypes.c_void_p
    library.consumer_destroy.argtypes = [ctypes.c_void_p]
    library.consumer_error.argtypes = [ctypes.c_void_p]
    library.consumer_error.restype = ctypes.c_char_p
    library.consumer_step.argtypes = [ctypes.c_void_p, DOUBLE_PTR, ctypes.c_int]
    library.consumer_fault.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.consumer_copy.argtypes = [ctypes.c_void_p, DOUBLE_PTR, UINT64_PTR]
    return library


@pytest.fixture(scope="module")
def cpu_consumer(
    tmp_path_factory: pytest.TempPathFactory,
    required_native_cxx: NativeCxx,
    consumer_includes: Path,
) -> ctypes.CDLL:
    directory = tmp_path_factory.mktemp("ordered-history-real-cpu")
    library = directory / "consumer.so"
    required_native_cxx.build_shared(
        [
            ROOT / "tests/native/test_ordered_history_consumers.cpp",
            NATIVE / "model/common/scc_mixer.cpp",
        ],
        library,
        compile_args=[
            "-std=c++17",
            "-O2",
            "-I" + str(NATIVE),
            "-I" + str(ROOT / "src"),
            "-I" + str(consumer_includes),
        ],
    )
    return _load(library)


@pytest.fixture(scope="module")
def cuda_consumer(
    tmp_path_factory: pytest.TempPathFactory,
    required_native_cxx: NativeCxx,
    consumer_includes: Path,
) -> ctypes.CDLL:
    nvcc = shutil.which(os.environ.get("NVCC", "nvcc"))
    if nvcc is None:
        pytest.skip("real CUDA consumer was not compiled or run: nvcc unavailable")
    directory = tmp_path_factory.mktemp("ordered-history-real-cuda")
    objects = []
    # Keep CUDA's incumbent contraction policy. In particular, no --fmad=false
    # is added to make CPU and CUDA falsely appear bit-identical.
    for index, source in enumerate(
        (
            ROOT / "tests/native/test_ordered_history_consumers.cu",
            NATIVE / "backends/cuda/gfn2_scc_mixer.cu",
        )
    ):
        obj = directory / f"consumer-{index}.o"
        subprocess.run(
            [
                required_native_cxx.cache,
                nvcc,
                "-std=c++17",
                "-O2",
                "-Xcompiler=-fPIC",
                "-I" + str(NATIVE),
                "-I" + str(ROOT / "src"),
                "-I" + str(consumer_includes),
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=ROOT,
            env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        )
        objects.append(obj)
    library = directory / "consumer.so"
    subprocess.run(
        [nvcc, "-shared", *map(str, objects), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = _load(library)
    if result.consumer_available() != 1:
        pytest.skip(
            "real CUDA consumer compiled but was not run: no usable CUDA device"
        )
    return result


class ChronologicalJohnson:
    """Independent model with chronological history, no physical-slot reads."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.current = np.concatenate(
            [
                0.01 * (s + 1) + 0.001 * np.arange(1, end - begin + 1)
                for s, (begin, end) in enumerate(pairwise(OFFSETS))
            ]
        )
        self.previous = np.zeros(TOTAL)
        self.previous_residual = np.zeros(TOTAL)
        self.history: list[list[tuple[np.ndarray, np.ndarray, float]]] = [[], [], []]
        self.df = [
            np.zeros((capacity, end - begin)) for begin, end in pairwise(OFFSETS)
        ]
        self.u = [array.copy() for array in self.df]
        self.weights = np.zeros((3, capacity))
        self.iteration = 0
        self.rms = np.zeros(3)
        self.maximum = np.zeros(3)

    def next_raw(self) -> np.ndarray:
        t = self.iteration + 1
        residual = np.concatenate(
            [
                (0.004 + 0.027 * (1 + np.sin(0.53 * t + s)))
                * (
                    np.sin(
                        np.arange(1, end - begin + 1) * (0.19 + 0.013 * t)
                        + 0.61 * t
                        + s
                    )
                    + 0.31 * np.cos(np.arange(1, end - begin + 1) * 0.47 - t * 0.29)
                )
                for s, (begin, end) in enumerate(pairwise(OFFSETS))
            ]
        )
        return self.current + residual

    def advance(self, raw: np.ndarray, poison: bool = True) -> None:
        residual = raw - self.current
        next_input = np.empty(TOTAL)
        for s, (begin, end) in enumerate(pairwise(OFFSETS)):
            x, f = self.current[begin:end], residual[begin:end]
            norm = np.linalg.norm(f)
            self.rms[s], self.maximum[s] = (
                norm / np.sqrt(end - begin),
                np.max(np.abs(f)),
            )
            if poison:
                # Poison physical backing independently of chronological data.
                slots = list(range(self.iteration, self.capacity))
                slots.append(
                    (self.iteration - 1) % self.capacity if self.iteration else 0
                )
                self.df[s][slots] = np.nan
                self.u[s][slots] = np.nan
                self.weights[s, slots] = np.nan
            result = x + DAMPING * f
            if self.iteration:
                difference = f - self.previous_residual[begin:end]
                scale = max(np.linalg.norm(difference), np.finfo(float).eps)
                df = difference / scale
                u = DAMPING * df + (x - self.previous[begin:end]) / scale
                weight = max(1.0, 0.01 / norm) if norm > 1e-7 else 100000.0
                self.history[s].append((df.copy(), u.copy(), weight))
                self.history[s] = self.history[s][-self.capacity :]
                derivatives = np.stack([entry[0] for entry in self.history[s]])
                updates = np.stack([entry[1] for entry in self.history[s]])
                weights = np.array([entry[2] for entry in self.history[s]])
                panel = weights[:, None] * derivatives
                gram = panel @ panel.T + 1e-4 * np.eye(len(weights))
                coefficients = np.linalg.solve(gram, panel @ f)
                result -= (weights * coefficients) @ updates
                # This ring is only an expected publication image, never input
                # to the oracle algebra above.
                slot = (self.iteration - 1) % self.capacity
                self.df[s][slot], self.u[s][slot] = df, u
                self.weights[s, slot] = weight
            next_input[begin:end] = result
        self.previous = self.current.copy()
        self.previous_residual = residual
        self.current = next_input
        self.iteration += 1

    def observe_committed_vectors(self, values: np.ndarray) -> None:
        # Compare every transition before rebasing its next input state. This
        # prevents arbitrary open-loop residual forcing from amplifying tiny
        # solver roundoff; chronological history remains independently built.
        self.current = values[TOTAL : 2 * TOTAL].copy()
        self.previous = values[2 * TOTAL : 3 * TOTAL].copy()
        self.previous_residual = values[3 * TOTAL : 4 * TOTAL].copy()

    def values(self) -> np.ndarray:
        return np.concatenate(
            [
                self.current,
                self.current,
                self.previous,
                self.previous_residual,
                *[array.ravel() for array in self.df],
                *[array.ravel() for array in self.u],
                self.weights.ravel(),
                self.rms,
                self.maximum,
            ]
        )

    def metadata(self) -> np.ndarray:
        return np.array(
            [
                *([self.iteration] * 3),
                *([0] * 3),
                *([0] * 3),
                *([1] * 3),
                *((self.rms < 0.04) & (self.maximum < 0.08)).astype(int),
            ],
            dtype=np.uint64,
        )


def _trajectory(
    library: ctypes.CDLL,
    capacity: int,
    graph: bool,
    *,
    rtol: float = 3e-9,
    atol: float = 3e-11,
) -> None:
    handle = library.consumer_create(capacity, graph)
    assert handle, library.consumer_error(handle).decode()
    oracle = ChronologicalJohnson(capacity)
    output = np.empty(254 + 127 * capacity)
    metadata = np.empty(15, dtype=np.uint64)
    try:
        # Every capacity reaches partial/full history and wraps twice. The
        # capacity-one case also exercises repeated tentative-only history.
        for step in range(2 * capacity + 5):
            raw = oracle.next_raw()
            assert (
                library.consumer_step(handle, raw.ctypes.data_as(DOUBLE_PTR), 1) == 0
            ), library.consumer_error(handle).decode()
            oracle.advance(raw)
            assert (
                library.consumer_copy(
                    handle,
                    output.ctypes.data_as(DOUBLE_PTR),
                    metadata.ctypes.data_as(UINT64_PTR),
                )
                == 0
            )
            # Dense LAPACK and the retained scalar Cholesky have different
            # reduction/solve rounding. No CPU/CUDA bitwise claim is made.
            np.testing.assert_allclose(
                output,
                oracle.values(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
                err_msg=f"capacity={capacity}, graph={graph}, step={step + 1}",
            )
            np.testing.assert_array_equal(metadata, oracle.metadata())
            oracle.observe_committed_vectors(output)
        before_failure = output.copy()
        before_metadata = metadata.copy()
        faults = [0, 3, 4]
        if capacity >= 2:
            faults.extend([1, 2, 5])
        if capacity >= 4:
            faults.append(6)
        for kind in faults:
            assert library.consumer_fault(handle, kind) == 0, library.consumer_error(
                handle
            ).decode()
            assert (
                library.consumer_copy(
                    handle,
                    output.ctypes.data_as(DOUBLE_PTR),
                    metadata.ctypes.data_as(UINT64_PTR),
                )
                == 0
            )
            # The fixture restores the injected fault after a bytewise check
            # of failure atomicity, so repeated failures start identically.
            np.testing.assert_array_equal(output[TOTAL:], before_failure[TOTAL:])
            np.testing.assert_array_equal(metadata, before_metadata)
        # A successful transition after failures verifies recovery, including
        # reuse of the very same captured graph when graph=True.
        raw = oracle.next_raw()
        assert library.consumer_step(handle, raw.ctypes.data_as(DOUBLE_PTR), 1) == 0, (
            library.consumer_error(handle).decode()
        )
        oracle.advance(raw)
        assert (
            library.consumer_copy(
                handle,
                output.ctypes.data_as(DOUBLE_PTR),
                metadata.ctypes.data_as(UINT64_PTR),
            )
            == 0
        )
        np.testing.assert_allclose(
            output, oracle.values(), rtol=rtol, atol=atol, equal_nan=True
        )
        np.testing.assert_array_equal(metadata, oracle.metadata())
    finally:
        library.consumer_destroy(handle)


@pytest.mark.parametrize("capacity", (1, 2, 4, 64))
def test_real_cpu_consumer_multistep_state_and_failure_retention(
    cpu_consumer: ctypes.CDLL, capacity: int
) -> None:
    _trajectory(cpu_consumer, capacity, False)


@pytest.mark.parametrize("capacity", (1, 2, 4, 64))
@pytest.mark.parametrize("graph", (False, True), ids=("stream", "graph-replay"))
def test_real_cuda_consumer_multistep_state_and_failure_retention(
    cuda_consumer: ctypes.CDLL, capacity: int, graph: bool
) -> None:
    _trajectory(cuda_consumer, capacity, graph, rtol=2e-8, atol=2e-10)


def _zero_difference_trajectory(library: ctypes.CDLL, graph: bool) -> None:
    """Exercise epsilon normalization, max omega and converged-but-active calls."""
    capacity = 4
    handle = library.consumer_create(capacity, graph)
    assert handle, library.consumer_error(handle).decode()
    oracle = ChronologicalJohnson(capacity)
    output = np.empty(254 + 127 * capacity)
    metadata = np.empty(15, dtype=np.uint64)
    try:
        for scale in (0.0, 0.0, 1e-18, 1e-18, 0.0, 1e-8, 1e-8, 0.0, 0.0):
            raw = oracle.current.copy()
            for begin in OFFSETS[:-1]:
                raw[begin] += scale
            assert (
                library.consumer_step(handle, raw.ctypes.data_as(DOUBLE_PTR), 1) == 0
            ), library.consumer_error(handle).decode()
            oracle.advance(raw)
            assert (
                library.consumer_copy(
                    handle,
                    output.ctypes.data_as(DOUBLE_PTR),
                    metadata.ctypes.data_as(UINT64_PTR),
                )
                == 0
            )
            np.testing.assert_allclose(
                output, oracle.values(), rtol=2e-8, atol=2e-10, equal_nan=True
            )
            np.testing.assert_array_equal(metadata, oracle.metadata())
            assert np.all(metadata[12:] == 1), (
                "residual convergence must not suppress requested mixing"
            )
            oracle.observe_committed_vectors(output)
        assert np.any(oracle.weights == 100000.0)
    finally:
        library.consumer_destroy(handle)


def test_real_cpu_zero_history_and_epsilon_normalization(
    cpu_consumer: ctypes.CDLL,
) -> None:
    _zero_difference_trajectory(cpu_consumer, False)


@pytest.mark.parametrize("graph", (False, True), ids=("stream", "graph-replay"))
def test_real_cuda_zero_history_and_epsilon_normalization(
    cuda_consumer: ctypes.CDLL, graph: bool
) -> None:
    _zero_difference_trajectory(cuda_consumer, graph)
