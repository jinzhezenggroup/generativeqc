"""Bounded host execution of the shared CUDA Cartesian range-ERI owner."""

import ctypes as ct
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc_compiler.common.native_runtime import compile_runtime
from generativeqc_compiler.common.paths import asset_path
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.integral.range_exchange_cpu import emit_range_exchange_cpu


class PreparedCpuRangeExchange:
    """One operator-frozen artifact; no per-component plans or primitive arrays.

    Basis storage is borrowed for this synchronous force lease. The host
    runtime visits normalized Cartesian expansions in bounded scalar storage
    and publishes only a compact physical-atom gradient (or twelve primitive
    slot derivatives for qualification). No reference library is loaded.
    """

    def __init__(self, basis: Any, radial: Any, compiler: Any, cache: Any) -> None:
        from ._stationary_cpu import _publish_source

        source = emit_range_exchange_cpu(radial)
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (canonical_hash(source) + ".cpp")
        _publish_source(path, source)
        names = (
            "integrals/execution_qualifiers.hpp",
            "integrals/range_moments.hpp",
            "integrals/range_exchange_runtime.hpp",
            "core/types.hpp",
            "molecule/basis.hpp",
            "scf/generated_shell_task.hpp",
            "scf/cuda/packed_basis.hpp",
            "scf/cuda/scalar_math.cuh",
            "scf/cuda/cartesian_angular.cuh",
            "scf/cuda/integral_limits.hpp",
            "scf/cuda/gaussian_geometry.cuh",
            "scf/cuda/hermite_recurrence.cuh",
            "scf/cuda/coulomb_auxiliary.cuh",
            "scf/cuda/boys_table.cuh",
        )
        public_header = asset_path("include/generativeqc/generativeqc.h")
        headers = tuple(asset_path("src/" + name) for name in names) + (
            public_header,
            asset_path("include/generativeqc/generated_method_ids.h"),
        )
        artifact = compile_runtime(
            compiler,
            cache,
            path,
            headers=headers,
            options=(
                "-std=c++20",
                "-ffp-contract=off",
                f"-I{headers[0].parents[1]}",
                f"-I{public_header.parents[1]}",
            ),
        )
        self.library = ct.CDLL(str(artifact.library))
        self.call = self.library.generativeqc_range_exchange_cpu
        self.call.argtypes = [
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.c_size_t,
            ct.c_size_t,
            ct.POINTER(ct.c_int64),
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.c_bool,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_uint64),
        ]
        self.call.restype = ct.c_int
        self.basis = basis

    def execute(
        self, indices: Any, weights: Any, *, independent_slots: bool = False
    ) -> tuple[np.ndarray, int, int]:
        """Publish a complete finite batch; invalid inputs cannot modify output."""
        indices = np.ascontiguousarray(indices, dtype=np.int64)
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        if (
            indices.ndim != 2
            or indices.shape[1] != 4
            or weights.shape != (len(indices),)
        ):
            raise ValueError("CPU range exchange requires weighted AO quartets")
        basis = self.basis
        output = np.empty((4 if independent_slots else basis.natom, 3))
        visits = (ct.c_uint64 * 2)()
        status = self.call(
            basis.packed.ctypes.data_as(ct.POINTER(ct.c_double)),
            basis.natom,
            basis.nprimitive,
            basis.nao,
            indices.ctypes.data_as(ct.POINTER(ct.c_int64)),
            weights.ctypes.data_as(ct.POINTER(ct.c_double)),
            len(indices),
            independent_slots,
            output.ctypes.data_as(ct.POINTER(ct.c_double)),
            visits,
        )
        if status:
            raise ArithmeticError(
                f"shared CPU range exchange failed with status {status}"
            )
        return output, visits[0], visits[1]

    def close(self) -> None:
        """Release the synchronous borrowed basis lease; no device owner exists."""
        self.basis = None
