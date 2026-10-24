"""Finite CPU compilation and bounded streaming of compiler-AD DF primitives."""

import ctypes as ct
from itertools import product
from pathlib import Path

import numpy as np

from ..common.cpp_adapter import CppCompilerAdapter
from ..common.native_runtime import compile_runtime
from ..common.paths import asset_path
from ..common.provenance import canonical_hash
from ..common.source_cache import cache_source
from .df_second import emit_df_second


class DFSecondEvaluator:
    """One component tile; no molecular arrays or global Jacobians are retained."""

    def __init__(
        self, family: str, angular: tuple, indices: tuple, cache: Path
    ) -> None:
        source = emit_df_second(family, angular, indices)
        directory = Path(cache) / "generated-sources"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (canonical_hash({"source": source}) + ".cpp")
        cache_source(path, source)
        headers = tuple(
            asset_path(name)
            for name in (
                "src/integrals/first_component_runtime.hpp",
                "src/integrals/range_moments.hpp",
            )
        )
        self.artifact = compile_runtime(
            CppCompilerAdapter(Path("c++")),
            Path(cache) / "native",
            path,
            headers=headers,
            options=("-ffp-contract=off", f"-I{headers[0].parents[1]}"),
        )
        self.library = ct.CDLL(str(self.artifact.library))
        self.run = self.library.generativeqc_df_second_v1
        self.run.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_void_p,
            ct.c_size_t,
        ]
        self.run.restype = ct.c_int
        self.centers = len(angular)
        self.shape = (len(indices), 1 + 6 * self.centers)

    def contract(
        self, primitives: tuple, centers: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Stream at most 64 radial products, publishing only complete finite tiles."""
        if centers.shape != (self.centers, 3) or direction.shape != centers.shape:
            raise ValueError("DF primitive geometry/direction shape mismatch")
        stride = 7 * self.centers + 1
        records = np.empty((64, stride))
        result = np.zeros(self.shape)
        staged = np.zeros_like(result)
        cursor = 0

        def flush(count: int) -> None:
            status = self.run(
                records.ctypes.data, count, stride, staged.ctypes.data, staged.size
            )
            if status:
                raise FloatingPointError(
                    f"generated DF primitive failed with status {status}"
                )
            result[:] += staged

        for primitive in product(*primitives):
            records[cursor] = (
                *(entry[0] for entry in primitive),
                *centers.reshape(-1),
                *direction.reshape(-1),
                np.prod([entry[1] for entry in primitive]),
            )
            cursor += 1
            if cursor == len(records):
                flush(cursor)
                cursor = 0
        if cursor:
            flush(cursor)
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite contracted DF derivative")
        return result
