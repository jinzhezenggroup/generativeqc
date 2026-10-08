"""Real CPU/CUDA Johnson consumers against an independent chronological oracle.

The oracle materializes a chronological list and uses NumPy's dense solve. It
never calls the ordered-history IR, interpreter, emitter, or generated helpers.
CUDA execution is optional and explicitly skipped when nvcc/device is absent;
compilation or numerical errors on an available toolchain are test failures.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
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
BYTE_PTR = ctypes.POINTER(ctypes.c_uint8)


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


def _cpu_owner_bridge(directory: Path, *, frozen: bool) -> Path:
    """Adapt only operation names/results; both owners retain their own arithmetic."""
    fixture = (
        ROOT / "tests/native/fixtures/johnson_prechange_3b97c234/consumer.cpp"
        if frozen
        else ROOT / "tests/native/test_ordered_history_consumers.cpp"
    )
    names = (
        {
            "BATCH": "mix_scc_broyden_batch_cpu",
            "RESTART": "restart_scc_mixer_system_cpu",
            "INITIALIZE": "initialize_scc_mixer_state_cpu",
            "MIX": "mix_scc_broyden_system_cpu",
            "NUMERICAL_FAILURE": "GENERATIVEQC_XTB_STATUS_INTERNAL_ERROR",
            "ALLOCATION_FAILED": "GENERATIVEQC_XTB_STATUS_ALLOCATION_FAILED",
        }
        if frozen
        else {
            "BATCH": "mix_broyden_batch",
            "RESTART": "restart_broyden_system",
            "INITIALIZE": "initialize_broyden_state",
            "MIX": "mix_broyden_system",
            "NUMERICAL_FAILURE": "static_cast<int>(BroydenResult::numerical_failure)",
            "ALLOCATION_FAILED": "static_cast<int>(BroydenResult::allocation_failed)",
        }
    )
    source = directory / "owner-bridge.cpp"
    source.write_text(
        f'#include "{fixture}"\n'
        + "".join(f"#define OWNER_{key} {value}\n" for key, value in names.items())
        + f'#include "{ROOT / "tests/native/ordered_history_owner_probe.inc"}"\n'
    )
    return source


def _load_cpu_owner(path: Path) -> ctypes.CDLL:
    library = _load(path)
    library.consumer_state_size.argtypes = [ctypes.c_void_p]
    library.consumer_state_size.restype = ctypes.c_size_t
    library.consumer_copy_state.argtypes = [ctypes.c_void_p, BYTE_PTR]
    library.consumer_copy_state.restype = None
    library.consumer_copy_layout.argtypes = [ctypes.c_void_p, UINT64_PTR]
    library.consumer_copy_layout.restype = None
    library.consumer_batch.argtypes = [ctypes.c_void_p, DOUBLE_PTR]
    library.consumer_restart.argtypes = [ctypes.c_void_p, ctypes.c_int, DOUBLE_PTR]
    library.consumer_initialize.argtypes = [ctypes.c_void_p, DOUBLE_PTR]
    library.consumer_set_counter.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
    ]
    library.consumer_set_counter.restype = None
    library.consumer_reject_binding.argtypes = [ctypes.c_void_p, ctypes.c_int]
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
            _cpu_owner_bridge(directory, frozen=False),
            ROOT / "src/solver/cpu/johnson_broyden.cpp",
        ],
        library,
        compile_args=[
            "-std=c++17",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(consumer_includes),
        ],
    )
    return _load_cpu_owner(library)


@pytest.fixture(scope="module")
def frozen_cpu_consumer(
    tmp_path_factory: pytest.TempPathFactory,
    required_native_cxx: NativeCxx,
) -> ctypes.CDLL:
    """Build the exact pre-migration owner without current sources or generators."""
    frozen = ROOT / "tests/native/fixtures/johnson_prechange_3b97c234"
    manifest = json.loads((frozen / "manifest.json").read_text())
    assert manifest["commit"] == "3b97c234eb18f5e6f354842b458a4e56c1619f95"
    hashes = {
        name: record["sha256"] for name, record in manifest["sources"].items()
    } | manifest["generated"]
    for name, expected in hashes.items():
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == expected
    directory = tmp_path_factory.mktemp("ordered-history-frozen-cpu")
    library = directory / "consumer.so"
    required_native_cxx.build_shared(
        [
            _cpu_owner_bridge(directory, frozen=True),
            frozen / "model/common/scc_mixer.cpp",
        ],
        library,
        compile_args=["-std=c++17", "-O2", "-I" + str(frozen)],
    )
    return _load_cpu_owner(library)


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


@pytest.mark.parametrize("capacity", (1, 2, 4, 64))
def test_shared_cpu_matches_frozen_prechange_ragged_state_and_failures(
    cpu_consumer: ctypes.CDLL,
    frozen_cpu_consumer: ctypes.CDLL,
    capacity: int,
) -> None:
    """An exact old-owner differential gate, independent of current CPU source."""
    libraries = (frozen_cpu_consumer, cpu_consumer)
    handles = [library.consumer_create(capacity, 0) for library in libraries]
    values = [np.empty(254 + 127 * capacity) for _ in libraries]
    metadata = [np.empty(15, dtype=np.uint64) for _ in libraries]
    driver = ChronologicalJohnson(capacity)
    state_bytes: list[np.ndarray] = []

    def copy_and_compare() -> None:
        for library, handle, output, info in zip(
            libraries, handles, values, metadata, strict=True
        ):
            assert (
                library.consumer_copy(
                    handle,
                    output.ctypes.data_as(DOUBLE_PTR),
                    info.ctypes.data_as(UINT64_PTR),
                )
                == 0
            )
        # Both CPU implementations keep the identical reduction/solve order.
        # This exact comparison is separate from the tolerant NumPy/CUDA gates.
        np.testing.assert_array_equal(values[0], values[1])
        np.testing.assert_array_equal(metadata[0], metadata[1])
        for library, handle, state in zip(libraries, handles, state_bytes, strict=True):
            library.consumer_copy_state(handle, state.ctypes.data_as(BYTE_PTR))
        np.testing.assert_array_equal(state_bytes[0], state_bytes[1])

    try:
        for library, handle in zip(libraries, handles, strict=True):
            assert handle, library.consumer_error(handle).decode()
        state_bytes = [
            np.empty(library.consumer_state_size(handle), dtype=np.uint8)
            for library, handle in zip(libraries, handles, strict=True)
        ]
        copy_and_compare()
        for _ in range(2 * capacity + 5):
            raw = driver.next_raw()
            for library, handle in zip(libraries, handles, strict=True):
                assert (
                    library.consumer_step(handle, raw.ctypes.data_as(DOUBLE_PTR), 1)
                    == 0
                ), library.consumer_error(handle).decode()
            copy_and_compare()
            driver.iteration += 1
            driver.observe_committed_vectors(values[0])
        faults = [0, 3, 4]
        if capacity >= 2:
            faults.extend([1, 2, 5])
        if capacity >= 4:
            faults.append(6)
        for fault in faults:
            for library, handle in zip(libraries, handles, strict=True):
                assert library.consumer_fault(handle, fault) == 0, (
                    library.consumer_error(handle).decode()
                )
            assert libraries[0].consumer_error(handles[0]) == libraries[
                1
            ].consumer_error(handles[1])
            copy_and_compare()
        raw = driver.next_raw()
        for library, handle in zip(libraries, handles, strict=True):
            assert (
                library.consumer_step(handle, raw.ctypes.data_as(DOUBLE_PTR), 1) == 0
            ), library.consumer_error(handle).decode()
        copy_and_compare()
    finally:
        for library, handle in zip(libraries, handles, strict=True):
            if handle:
                library.consumer_destroy(handle)


@pytest.mark.parametrize("capacity", (1, 4))
def test_shared_cpu_matches_frozen_admission_restart_and_mixed_peer_failures(
    cpu_consumer: ctypes.CDLL,
    frozen_cpu_consumer: ctypes.CDLL,
    capacity: int,
) -> None:
    libraries = (frozen_cpu_consumer, cpu_consumer)
    handles = [library.consumer_create(capacity, 0) for library in libraries]
    values = [np.empty(254 + 127 * capacity) for _ in libraries]
    metadata = [np.empty(15, dtype=np.uint64) for _ in libraries]
    state_bytes: list[np.ndarray] = []
    driver = ChronologicalJohnson(capacity)

    def compare() -> None:
        for library, handle, output, info, state in zip(
            libraries, handles, values, metadata, state_bytes, strict=True
        ):
            assert (
                library.consumer_copy(
                    handle,
                    output.ctypes.data_as(DOUBLE_PTR),
                    info.ctypes.data_as(UINT64_PTR),
                )
                == 0
            )
            library.consumer_copy_state(handle, state.ctypes.data_as(BYTE_PTR))
        np.testing.assert_array_equal(values[0], values[1])
        np.testing.assert_array_equal(metadata[0], metadata[1])
        np.testing.assert_array_equal(state_bytes[0], state_bytes[1])
        assert libraries[0].consumer_error(handles[0]) == libraries[1].consumer_error(
            handles[1]
        )

    def invoke(name: str, arguments: tuple, expected: int) -> None:
        for library, handle in zip(libraries, handles, strict=True):
            assert getattr(library, name)(handle, *arguments) == expected, (
                library.consumer_error(handle).decode()
            )
        compare()

    try:
        for library, handle in zip(libraries, handles, strict=True):
            assert handle, library.consumer_error(handle).decode()
        state_bytes = [
            np.empty(library.consumer_state_size(handle), dtype=np.uint8)
            for library, handle in zip(libraries, handles, strict=True)
        ]
        layouts = [np.empty(22, dtype=np.uint64) for _ in libraries]
        for library, handle, layout in zip(libraries, handles, layouts, strict=True):
            library.consumer_copy_layout(handle, layout.ctypes.data_as(UINT64_PTR))
        np.testing.assert_array_equal(layouts[0], layouts[1])
        compare()
        for _ in range(capacity + 3):
            raw = driver.next_raw()
            invoke("consumer_batch", (raw.ctypes.data_as(DOUBLE_PTR),), 0)
            driver.iteration += 1
            driver.observe_committed_vectors(values[0])

        # A middle-peer NaN leaves its complete state intact except its status;
        # both finite peers still advance, including their physical histories.
        raw = driver.next_raw()
        raw[OFFSETS[1]] = np.nan
        before_values, before_info = values[0].copy(), metadata[0].copy()
        invoke("consumer_batch", (raw.ctypes.data_as(DOUBLE_PTR),), 6)
        np.testing.assert_array_equal(metadata[0][:3], before_info[:3] + [1, 0, 1])
        np.testing.assert_array_equal(metadata[0][6:9], [0, 6, 0])
        np.testing.assert_array_equal(
            values[0][TOTAL + OFFSETS[1] : TOTAL + OFFSETS[2]],
            before_values[TOTAL + OFFSETS[1] : TOTAL + OFFSETS[2]],
        )
        before = state_bytes[0].copy()
        invoke("consumer_initialize", (raw.ctypes.data_as(DOUBLE_PTR),), 1)
        np.testing.assert_array_equal(state_bytes[0], before)
        invoke("consumer_restart", (1, raw.ctypes.data_as(DOUBLE_PTR)), 1)
        np.testing.assert_array_equal(state_bytes[0], before)

        for kind in range(6):
            invoke("consumer_reject_binding", (kind,), 1)
            np.testing.assert_array_equal(state_bytes[0], before)

        raw[OFFSETS[1]] = before_values[TOTAL + OFFSETS[1]] + 0.02
        invoke("consumer_restart", (1, raw.ctypes.data_as(DOUBLE_PTR)), 0)
        assert metadata[0][1] == 0 and metadata[0][4] == before_info[4] + 1
        assert metadata[0][7] == 0
        for library, handle in zip(libraries, handles, strict=True):
            library.consumer_set_counter(handle, 1, 1, 2**64 - 1)
        compare()
        before = state_bytes[0].copy()
        invoke("consumer_restart", (1, raw.ctypes.data_as(DOUBLE_PTR)), 1)
        np.testing.assert_array_equal(state_bytes[0], before)
        for library, handle in zip(libraries, handles, strict=True):
            library.consumer_set_counter(handle, 1, 1, 1)
            library.consumer_set_counter(handle, 1, 0, 2**64 - 1)
        invoke("consumer_batch", (raw.ctypes.data_as(DOUBLE_PTR),), 6)
        assert metadata[0][1] == 2**64 - 1 and metadata[0][7] == 6
        invoke("consumer_restart", (1, raw.ctypes.data_as(DOUBLE_PTR)), 0)
        assert metadata[0][1] == 0 and metadata[0][4] == 2
        invoke("consumer_batch", (raw.ctypes.data_as(DOUBLE_PTR),), 0)
    finally:
        for library, handle in zip(libraries, handles, strict=True):
            if handle:
                library.consumer_destroy(handle)
