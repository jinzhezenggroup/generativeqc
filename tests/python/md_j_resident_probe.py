"""Test-only bridge to the exact device-resident Fock seam used by native KS.

The public host Fock evaluator is a retained independent compatibility route;
its preparation schedule alone is not evidence that resident MD kernels ran.
This adapter adds no production ABI and owns neither an oracle nor SCF policy.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import os
import tempfile
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from typing_extensions import Self


@cache
def _probe_library() -> Any:
    """Cache a host-compiled test adapter against the exact selected library."""
    root = Path(os.environ.get("SOURCE", Path(__file__).resolve().parents[2]))
    source = Path(
        os.environ.get(
            "GENERATIVEQC_MD_J_PROBE_CPP", root / "tests/native/md_j_resident_probe.cpp"
        )
    )
    native = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    identity = hashlib.sha256(source.read_bytes() + native.read_bytes()).hexdigest()[
        :20
    ]
    directory = Path(tempfile.gettempdir()) / f"md-j-resident-probe-{identity}"
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "probe.so"
    if not binary.exists():
        from generativeqc_compiler.common.compiler_cache import (
            resolve_compiler_cache,
            run_cached_compiler,
        )
        from generativeqc_compiler.common.compiler_process import run_compiler

        compiler = os.environ.get("CXX", "c++")
        launcher = resolve_compiler_cache(timeout=15)
        command = [
            compiler,
            "-std=c++20",
            "-fPIC",
            "-O2",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
            "-I" + str(Path(os.environ["CUDA_PATH"]) / "include"),
            "-c",
            str(source),
        ]
        # Keep the cached compile separate from the non-cacheable link. Publish
        # atomically so concurrent tests never load a partially written library.
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            obj = Path(temporary) / "probe.o"
            linked = Path(temporary) / "probe.so"
            result = run_cached_compiler(
                [*command, "-o", str(obj)],
                timeout=120,
                label="resident MD-J probe",
                launcher=launcher,
            )
            if result.returncode:
                raise RuntimeError(result.stderr or "resident probe compilation failed")
            result = run_compiler(
                [
                    compiler,
                    "-shared",
                    str(obj),
                    str(native),
                    "-L" + str(Path(os.environ["CUDA_PATH"]) / "lib64"),
                    "-lcudart",
                    "-Wl,-rpath," + str(native.parent),
                    "-o",
                    str(linked),
                ],
                timeout=120,
                label="resident MD-J probe link",
            )
            if result.returncode:
                raise RuntimeError(result.stderr or "resident probe link failed")
            linked.replace(binary)
    # Load the selected backend before the bridge's SONAME dependency. An
    # experimental snapshot may contain an older versioned sibling beside an
    # explicitly selected .so; letting the loader search that sibling first
    # would time a different backend from the one whose bytes keyed this probe.
    native_handle = ct.CDLL(str(native))
    library = ct.CDLL(str(binary))
    symbol = "generativeqc_get_source_identity"
    if (
        ct.cast(getattr(library, symbol), ct.c_void_p).value
        != ct.cast(getattr(native_handle, symbol), ct.c_void_p).value
    ):
        raise RuntimeError("resident probe resolved a different native backend")
    library._selected_native_handle = native_handle
    library.md_j_probe_create.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.c_int,
        ct.c_double,
        ct.c_size_t,
    ]
    library.md_j_probe_create.restype = ct.c_void_p
    library.md_j_probe_execute.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
        ct.POINTER(ct.c_double),
    ]
    library.md_j_probe_execute.restype = ct.c_int
    library.md_j_probe_propose.argtypes = [
        *library.md_j_probe_execute.argtypes[:-1],
        ct.c_double,
        ct.POINTER(ct.c_double),
    ]
    library.md_j_probe_propose.restype = ct.c_int
    library.md_j_probe_capture.argtypes = library.md_j_probe_execute.argtypes[:-1]
    library.md_j_probe_capture.restype = ct.c_int
    library.md_j_probe_replay.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double)]
    library.md_j_probe_replay.restype = ct.c_int
    for name in (
        "md_j_probe_graph_nodes",
        "md_j_probe_graph_kernels",
        "md_j_probe_graph_device_copies",
    ):
        function = getattr(library, name)
        function.argtypes = [ct.c_void_p]
        function.restype = ct.c_size_t
    for name in ("md_j_probe_schedule", "md_j_probe_error"):
        function = getattr(library, name)
        function.restype = ct.c_char_p
        function.argtypes = [] if name.endswith("error") else [ct.c_void_p]
    library.md_j_probe_device_bytes.argtypes = [ct.c_void_p]
    library.md_j_probe_device_bytes.restype = ct.c_size_t
    library.md_j_probe_destroy.argtypes = [ct.c_void_p]
    library.md_j_probe_destroy.restype = None
    return library


class ResidentMdJProbe:
    """Own a serialized resident plan; all device buffers are test-local."""

    def __init__(
        self,
        basis: Any,
        spec: Any,
        *,
        device: str = "cuda",
        screening_tolerance: float = 1e-12,
        device_budget_bytes: int = 0,
    ) -> None:
        import cupy as cp
        from generativeqc import Calculator, _native

        if device != "cuda" or not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError(
                "resident qualification requires finite Slurm CUDA execution"
            )
        self.basis, self.spec = basis, spec
        self._probe = _probe_library()
        self._calculator = Calculator(
            device="cuda",
            basis_representation=(
                "spherical" if basis.representation == "real_spherical" else "cartesian"
            ),
        )
        self._native = self._calculator._library
        self._context, self._system, self._handle = (
            ct.c_void_p(),
            ct.c_void_p(),
            ct.c_void_p(),
        )
        try:
            _native.check(
                self._native,
                self._native.generativeqc_context_create(
                    ct.byref(self._calculator._context_descriptor()),
                    ct.byref(self._context),
                ),
            )
            self._system = self._calculator._create_native_system(
                self._context,
                basis.atoms,
                basis.charge,
                basis.multiplicity,
                basis.shells,
            )
            self._handle = ct.c_void_p(
                self._probe.md_j_probe_create(
                    self._system,
                    spec.spin == "unrestricted",
                    spec.exchange.present,
                    screening_tolerance,
                    device_budget_bytes,
                )
            )
            if not self._handle.value:
                raise RuntimeError(self._probe.md_j_probe_error().decode())
            self.diagnostics = {
                "direct_schedule": self._probe.md_j_probe_schedule(
                    self._handle
                ).decode(),
                "device_bytes": self._probe.md_j_probe_device_bytes(self._handle),
            }
            spins = 2 if spec.spin == "unrestricted" else 1
            self._density = cp.empty((spins, basis.nao, basis.nao), dtype=cp.float64)
            self._coulomb = cp.empty((basis.nao, basis.nao), dtype=cp.float64)
            self._exchange = cp.empty_like(self._density)
            self._error = cp.empty(1, dtype=cp.int32)
            self.last_device_seconds = 0.0
            self._graph_captured = False
        except BaseException:
            self.close()
            raise

    def _require_open(self) -> None:
        if not self._handle.value:
            raise RuntimeError("resident probe is closed")

    def _buffer_arguments(self) -> tuple[Any, ...]:
        separate = self.spec.spin == "unrestricted"
        exchange = self.spec.exchange.present
        return (
            self._handle,
            self._density[0].data.ptr,
            self._density[1].data.ptr if separate else None,
            self.basis.nao**2,
            self._coulomb.data.ptr,
            self._exchange[0].data.ptr if exchange else None,
            self._exchange[1].data.ptr if separate and exchange else None,
            self._error.data.ptr,
        )

    def _upload_density(self, density: Any) -> np.ndarray:
        import cupy as cp

        self._require_open()
        values = np.ascontiguousarray(density, dtype=np.float64)
        stream = cp.cuda.get_current_stream()
        self._density.set(values.reshape(self._density.shape), stream=stream)
        # Upload is outside capture and precedes native work on the owner stream.
        # Fence the actual producer, including a caller's nondefault CuPy stream.
        stream.synchronize()
        return values

    def _publish(self, values: np.ndarray, status: int, seconds: ct.c_double) -> Any:
        import cupy as cp

        if status or int(cp.asnumpy(self._error)[0]):
            raise RuntimeError(
                self._probe.md_j_probe_error().decode() or "numerical error"
            )
        self.last_device_seconds = seconds.value
        np.testing.assert_array_equal(
            cp.asnumpy(self._density).reshape(values.shape), values
        )
        separate = self.spec.spin == "unrestricted"
        return SimpleNamespace(
            coulomb=cp.asnumpy(self._coulomb),
            exchange=(
                cp.asnumpy(self._exchange)
                if separate
                else cp.asnumpy(self._exchange[0])
            )
            if self.spec.exchange.present
            else None,
        )

    def evaluate(self, density: Any) -> Any:
        """Upload test input, run resident raw J/K, and publish synchronized arrays."""
        values = self._upload_density(density)
        seconds = ct.c_double()
        status = self._probe.md_j_probe_execute(
            *self._buffer_arguments(), ct.byref(seconds)
        )
        return self._publish(values, status, seconds)

    def evaluate_proposal(self, density: Any, screening: float) -> Any:
        """Exercise a provisional cutoff without rebuilding the prepared owner."""
        values = self._upload_density(density)
        seconds = ct.c_double()
        status = self._probe.md_j_probe_propose(
            *self._buffer_arguments(), screening, ct.byref(seconds)
        )
        return self._publish(values, status, seconds)

    def capture_graph(self) -> None:
        """Capture native resident execution against this owner's stable buffers.

        Preparation and all device allocations precede capture. The graph contains
        only kernels, memsets and verified device copies; work-count downloads
        must be disabled at construction. Capture does not execute or upload density. This probe must
        stay alive for every replay; close destroys graphs before the native plan.
        """
        self._require_open()
        if self._probe.md_j_probe_capture(*self._buffer_arguments()):
            raise RuntimeError(self._probe.md_j_probe_error().decode())
        self._graph_captured = True
        self.diagnostics["graph_nodes"] = self._probe.md_j_probe_graph_nodes(
            self._handle
        )
        self.diagnostics["graph_kernel_nodes"] = self._probe.md_j_probe_graph_kernels(
            self._handle
        )
        self.diagnostics["graph_device_copy_nodes"] = (
            self._probe.md_j_probe_graph_device_copies(self._handle)
        )

    def evaluate_graph(self, density: Any) -> Any:
        """Replay with changed density, without re-entering native host dispatch.

        Only test input upload and completed-result download touch the host; they
        are outside capture/replay timing. Native replay launches the retained graph
        and fences the owner stream once when publishing the completed result.
        """
        self._require_open()
        if not self._graph_captured:
            raise RuntimeError("resident graph has not been captured")
        values = self._upload_density(density)
        seconds = ct.c_double()
        status = self._probe.md_j_probe_replay(self._handle, ct.byref(seconds))
        return self._publish(values, status, seconds)

    def close(self) -> None:
        """Release the resident borrower before its system/context lifetimes."""
        if self._handle.value:
            self._probe.md_j_probe_destroy(self._handle)
            self._handle = ct.c_void_p()
            self._graph_captured = False
        if self._system.value:
            self._native.generativeqc_system_destroy(self._system)
            self._system = ct.c_void_p()
        if self._context.value:
            self._native.generativeqc_context_destroy(self._context)
            self._context = ct.c_void_p()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *arguments: object) -> None:
        self.close()
