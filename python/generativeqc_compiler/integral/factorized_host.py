"""Bounded CPU execution of the existing full-range factorized ERI jets.

One identity-checked artifact accepts runtime angular/component selections.
The compiler's existing Hermite/Boys factor algebra owns all derivatives;
this adapter owns only finite primitive pages and explicit external weights.
"""

from __future__ import annotations

import ctypes as ct
import math
import os
import tempfile
import threading
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.native_runtime import compile_runtime
from generativeqc_compiler.common.paths import asset_path
from generativeqc_compiler.common.provenance import canonical_hash, file_hash

from .factorized_second import emit_factorized_second

_HEADERS = (
    "src/integrals/factorized_host_runtime.hpp",
    "src/integrals/eri_geometry.hpp",
    "src/integrals/range_moments.hpp",
)
_ARTIFACTS: dict[object, Any] = {}
_LOCK = threading.RLock()


@cache
def _source() -> str:
    return (
        emit_factorized_second(backend="cpu")
        + r"""
#include "integrals/factorized_host_runtime.hpp"
namespace {
struct Program {
  using Prepared = generativeqc::scf::factorized_second::Geometry;
  static bool prepare(const unsigned* angular, const double* record, Prepared& geometry) {
    unsigned maximum_order=2;
    for (unsigned slot=0; slot<4; ++slot) maximum_order+=angular[slot];
    return generativeqc::integrals::make_eri_geometry<Prepared,14>(
        record, record+4, maximum_order, generativeqc::integrals::CoulombRange::Full, 0, geometry);
  }
  static bool evaluate(const unsigned* angular, unsigned component,
                       const double* record, const double* direction,
                       const Prepared& geometry, double* output) {
    return generativeqc::scf::factorized_second::primitive(
        0, 0, angular, component, record, direction, output, &geometry);
  }
};
}
extern "C" int generativeqc_factorized_eri_host_v1(
    unsigned operation, const unsigned* angular, const unsigned* components,
    std::size_t component_count, const double* records, std::size_t count,
    std::size_t stride, const double* weights, const double* direction,
    double* output, std::size_t output_size, std::uint64_t* visits) {
  return generativeqc::integrals::factorized_eri_records<Program>(
      operation, angular, components, component_count, records, count, stride,
      weights, direction, output, output_size, visits);
}
"""
    )


def _artifact(compiler: CppCompilerAdapter, cache_directory: Path) -> Any:
    headers = tuple(asset_path(name) for name in _HEADERS)
    header_identity = tuple(file_hash(path) for path in headers)
    key = (compiler, cache_directory.resolve(), header_identity)
    with _LOCK:
        artifact = _ARTIFACTS.get(key)
        if artifact is None or not artifact.library.is_file():
            source = _source()
            directory = cache_directory / "generated-sources"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (canonical_hash({"source": source}) + ".cpp")
            if not path.exists():
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=directory, delete=False
                ) as stream:
                    stream.write(source)
                    temporary = Path(stream.name)
                try:
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            elif path.read_text() != source:
                raise ValueError("factorized ERI source cache integrity failure")
            artifact = compile_runtime(
                compiler,
                cache_directory / "native",
                path,
                headers=headers,
                options=("-ffp-contract=off", f"-I{headers[0].parents[2] / 'src'}"),
            )
            _ARTIFACTS[key] = artifact
        metadata = artifact.metadata
        if (
            canonical_hash(metadata["identity"]) != metadata["key"]
            or metadata["identity"].get("backend") != "cpu"
            or file_hash(artifact.library) != metadata["binary_sha256"]
        ):
            raise ValueError("factorized ERI native cache integrity failure")
        return artifact


class FactorizedERIHost:
    """Stream unit full-range s/p/d/f ERI first derivatives or weighted HVPs.

    Primitive lists carry radial normalization; Cartesian component factors
    belong to the caller's existing public-weight pullback. Mathematical slots
    are independent, including repeated physical centers. No molecular tensor
    or native allocation survives a call. ``numeric_bytes`` conservatively
    counts record pages, Python/native publication tiles, directions, weights
    and component metadata. Generated scalar/jet stack temporaries, compiler
    processes, library mappings and Python object overhead are excluded.
    Each primitive record owns one transient geometry/moment binding shared
    across its component page; no binding is cached across records or calls.
    """

    @staticmethod
    def required_bytes(record_capacity: int = 128) -> int:
        """Preflight the conservative numeric lease without compiling anything."""
        if type(record_capacity) is not int or not 1 <= record_capacity <= 128:
            raise ValueError("factorized record capacity must be from one to 128")
        return 8 * (record_capacity * 17 + 4 * 64 * 13 + 64 + 2 * 12 + 25) + 4 * (
            4 + 64
        )

    def __init__(
        self,
        compiler: CppCompilerAdapter,
        cache_directory: Path,
        *,
        record_capacity: int = 128,
        budget_bytes: int = 1 << 20,
    ) -> None:
        if not isinstance(compiler, CppCompilerAdapter):
            raise TypeError("factorized host execution requires a CPU compiler")
        if type(record_capacity) is not int or not 1 <= record_capacity <= 128:
            raise ValueError("factorized record capacity must be from one to 128")
        if type(budget_bytes) is not int or not 0 < budget_bytes < 2**63:
            raise ValueError("factorized host budget must be a positive int64")
        self.numeric_bytes = self.required_bytes(record_capacity)
        if self.numeric_bytes > budget_bytes:
            raise MemoryError("factorized ERI host numeric budget exceeded")
        self.artifact = _artifact(compiler, Path(cache_directory))
        self.library = ct.CDLL(str(self.artifact.library))
        self.run = self.library.generativeqc_factorized_eri_host_v1
        self.run.argtypes = (
            ct.c_uint,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_void_p,
        )
        self.run.restype = ct.c_int
        self.record_capacity = record_capacity
        self._closed = False

    def raw_first(
        self, angular: Any, components: Any, primitives: Any, centers: Any
    ) -> np.ndarray:
        """Return radially contracted raw value/gradient rows in slot order."""
        return self._contract(1, angular, components, primitives, centers)[0]

    def weighted_hvp(
        self,
        angular: Any,
        components: Any,
        primitives: Any,
        centers: Any,
        weights: Any,
        direction: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Contract fixed raw Cartesian weights with a fixed slot direction."""
        if weights is None or direction is None:
            raise ValueError(
                "factorized ERI HVP requires explicit weights and direction"
            )
        return self._contract(
            2, angular, components, primitives, centers, weights, direction
        )

    def _contract(
        self,
        operation: int,
        angular: Any,
        components: Any,
        primitives: Any,
        centers: Any,
        weights: Any = None,
        direction: Any = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("factorized ERI host owner is closed")
        angular = tuple(angular)
        components = tuple(components)
        if len(angular) != 4 or any(
            type(order) is not int or not 0 <= order <= 3 for order in angular
        ):
            raise ValueError("factorized ERI requires four s/p/d/f shells")
        size = math.prod((order + 1) * (order + 2) // 2 for order in angular)
        if (
            not 1 <= len(components) <= 64
            or len(set(components)) != len(components)
            or any(
                type(index) is not int or not 0 <= index < size for index in components
            )
        ):
            raise ValueError("factorized ERI requires one to 64 distinct components")
        primitive_lists = tuple(tuple(shell) for shell in primitives)
        if len(primitive_lists) != 4 or any(not shell for shell in primitive_lists):
            raise ValueError("factorized ERI requires four nonempty primitive lists")
        for shell in primitive_lists:
            for primitive in shell:
                pair = np.asarray(primitive)
                if (
                    pair.shape != (2,)
                    or np.iscomplexobj(pair)
                    or not np.isfinite(pair).all()
                    or pair[0] <= 0
                ):
                    raise ValueError(
                        "factorized ERI primitives must be finite and positive"
                    )
        positions = np.asarray(centers)
        if (
            positions.shape != (4, 3)
            or np.iscomplexobj(positions)
            or not np.isfinite(positions).all()
        ):
            raise ValueError("factorized ERI centers must be finite real slot xyz")
        motion = np.zeros((4, 3)) if direction is None else np.asarray(direction)
        if (
            motion.shape != (4, 3)
            or np.iscomplexobj(motion)
            or not np.isfinite(motion).all()
        ):
            raise ValueError("factorized ERI direction must be finite real slot xyz")
        motion = np.ascontiguousarray(motion, dtype=np.float64)
        cotangent = np.ones(len(components)) if weights is None else np.asarray(weights)
        if (
            cotangent.shape != (len(components),)
            or np.iscomplexobj(cotangent)
            or not np.isfinite(cotangent).all()
        ):
            raise ValueError(
                "factorized ERI weights must match the finite component tile"
            )
        cotangent = np.ascontiguousarray(cotangent, dtype=np.float64)
        orders = np.array(angular, dtype=np.uint32)
        selection = np.array(components, dtype=np.uint32)
        shape = (len(components), 13) if operation == 1 else (4, 3)
        result = np.zeros(shape)
        chunk = np.empty(shape)
        records = np.empty((self.record_capacity, 17))
        visits = np.zeros(2, dtype=np.uint64)
        batches = primitive_records = component_evaluations = 0

        def flush(count: int) -> None:
            nonlocal batches, primitive_records, component_evaluations
            status = self.run(
                operation,
                orders.ctypes.data,
                selection.ctypes.data,
                len(selection),
                records.ctypes.data,
                count,
                17,
                cotangent.ctypes.data,
                motion.ctypes.data,
                chunk.ctypes.data,
                chunk.size,
                visits.ctypes.data,
            )
            if status:
                error = ValueError if status == 1 else FloatingPointError
                raise error(f"factorized ERI host failed with native status {status}")
            with np.errstate(over="raise", invalid="raise"):
                np.add(result, chunk, out=result)
            batches += 1
            primitive_records += int(visits[0])
            component_evaluations += int(visits[1])

        count = 0
        for combination in product(*primitive_lists):
            exponents, coefficients = zip(*combination, strict=True)
            records[count, :4] = exponents
            records[count, 4:16] = positions.ravel()
            records[count, 16] = math.prod(coefficients)
            if not math.isfinite(records[count, 16]):
                raise FloatingPointError("factorized ERI radial scale overflow")
            count += 1
            if count == self.record_capacity:
                flush(count)
                count = 0
        if count:
            flush(count)
        return result, {
            "program_identity": canonical_hash(
                {
                    "schema": "generativeqc.factorized-eri-host/v1",
                    "operation": operation,
                    "angular": angular,
                    "components": components,
                }
            ),
            "native_artifact": self.artifact.metadata["key"],
            "records": primitive_records,
            "geometry_evaluations": primitive_records,
            "component_evaluations": component_evaluations,
            "chunks": batches,
            "resources": {"peak_bytes": {"host": self.numeric_bytes}},
        }

    def close(self) -> None:
        """Expire this synchronous owner; no native allocation is retained."""
        self._closed = True
