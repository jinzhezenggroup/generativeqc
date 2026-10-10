"""Test-only bridge to the exact device-resident Fock seam used by native KS.

The public host Fock evaluator is a retained independent compatibility route;
its preparation schedule alone is not evidence that resident MD kernels ran.
This adapter adds no production ABI and owns neither an oracle nor SCF policy.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import os
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import numpy as np


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
        compiler = os.environ.get("CXX", "c++")
        cache = shutil.which("ccache")
        command = ([cache] if cache else []) + [
            compiler,
            "-std=c++20",
            "-shared",
            "-fPIC",
            "-O2",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
            "-I" + str(Path(os.environ["CUDA_PATH"]) / "include"),
            str(source),
            str(native),
            "-L" + str(Path(os.environ["CUDA_PATH"]) / "lib64"),
            "-lcudart",
            "-Wl,-rpath," + str(native.parent),
            "-o",
            str(binary),
        ]
        try:
            subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=120
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(error.stderr) from error
    library = ct.CDLL(str(binary))
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
        except BaseException:
            self.close()
            raise

    def evaluate(self, density: Any) -> Any:
        """Upload test input, run resident raw J/K, and publish synchronized arrays."""
        import cupy as cp

        values = np.ascontiguousarray(density, dtype=np.float64)
        self._density.set(values.reshape(self._density.shape))
        cp.cuda.Stream.null.synchronize()
        separate = self.spec.spin == "unrestricted"
        exchange = self.spec.exchange.present
        seconds = ct.c_double()
        status = self._probe.md_j_probe_execute(
            self._handle,
            self._density[0].data.ptr,
            self._density[1].data.ptr if separate else None,
            self.basis.nao**2,
            self._coulomb.data.ptr,
            self._exchange[0].data.ptr if exchange else None,
            self._exchange[1].data.ptr if separate and exchange else None,
            self._error.data.ptr,
            ct.byref(seconds),
        )
        if status or int(cp.asnumpy(self._error)[0]):
            raise RuntimeError(
                self._probe.md_j_probe_error().decode() or "numerical error"
            )
        self.last_device_seconds = seconds.value
        np.testing.assert_array_equal(
            cp.asnumpy(self._density).reshape(values.shape), values
        )
        return SimpleNamespace(
            coulomb=cp.asnumpy(self._coulomb),
            exchange=(
                cp.asnumpy(self._exchange)
                if separate
                else cp.asnumpy(self._exchange[0])
            )
            if exchange
            else None,
        )

    def close(self) -> None:
        """Release the resident borrower before its system/context lifetimes."""
        if self._handle.value:
            self._probe.md_j_probe_destroy(self._handle)
            self._handle = ct.c_void_p()
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
