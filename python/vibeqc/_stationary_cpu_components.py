"""Bounded multicomponent public-AO consumer of generated primitive gradients."""

import ctypes as ct
import typing
from itertools import product
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.integral.first_derivative_schedule import (
    COMPONENT_LABELS,
    CPU_AOT_SHARDS,
    REQUESTS_PER_UNIT,
    cpu_aot_symbol,
    derivative_binding,
    derivative_requests,
    derivative_sources,
)


def _configure_dispatch(call: typing.Any) -> typing.Any:
    call.argtypes = [
        ct.c_uint,
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.POINTER(ct.c_double),
    ]
    call.restype = ct.c_int
    return call


def _packaged_aot_dispatchers(library: typing.Any) -> tuple[typing.Any, ...]:
    if library is None:
        return ()
    dispatchers = []
    for shard in range(CPU_AOT_SHARDS):
        try:
            call = getattr(library, cpu_aot_symbol(shard))
        except AttributeError:
            return ()
        dispatchers.append(_configure_dispatch(call))
    return tuple(dispatchers)


class ComponentPrimitiveExecutor:
    """Stream every public component weight; share only compiled scalar kernels."""

    def __init__(
        self,
        basis: typing.Any,
        cache: str | Path,
        primitive_tile: int,
        compiler: CppCompilerAdapter,
        *,
        aot_library: typing.Any = None,
    ) -> None:
        from ._stationary_cpu import _compile_primitive_library

        if primitive_tile < 1:
            raise ValueError("primitive tile must be positive")
        if any(shell.angular_momentum > 2 for shell in basis.shells):
            raise NotImplementedError("component executor supports s/p/d bases only")
        start = 3 * basis.natom
        self.centers = basis.packed[:start].reshape(-1, 3)
        self.primitives = basis.packed[start : start + 2 * basis.nprimitive].reshape(
            -1, 2
        )
        self.aos = basis.packed[start + 2 * basis.nprimitive :].reshape(-1, 16)
        if any(row[3] not in (1, 2, 3) for row in self.aos):
            raise ValueError("public AOs require one to three Cartesian components")
        self.expansions = tuple(
            tuple(
                (
                    "".join(
                        axis * int(power)
                        for axis, power in zip("xyz", row[4 + 4 * t : 7 + 4 * t])
                    ),
                    row[7 + 4 * t],
                )
                for t in range(int(row[3]))
            )
            for row in self.aos
        )
        domain = tuple(
            sorted({c for expansion in self.expansions for c, _ in expansion})
        )
        self.libraries, self.dispatchers, self.calls = [], [], {}
        requests = derivative_requests(domain)
        groups = tuple(
            requests[begin : begin + REQUESTS_PER_UNIT]
            for begin in range(0, len(requests), REQUESTS_PER_UNIT)
        )
        packaged = (
            _packaged_aot_dispatchers(aot_library) if domain == COMPONENT_LABELS else ()
        )
        if packaged:
            if len(packaged) != len(groups):
                raise RuntimeError("packaged CPU derivative shard inventory mismatch")
            self.libraries.append(aot_library)
            self.dispatchers.extend(packaged)
            for shard, selected in enumerate(groups):
                for kind, request in enumerate(selected):
                    self.calls[request] = (packaged[shard], kind)
            source_bytes = largest_source = 0
            runtime_compilations = 0
        else:
            sources = derivative_sources(domain)
            for selected, source in sources:
                library, call = _compile_primitive_library(source, cache, compiler)
                self.libraries.append(library)
                self.dispatchers.append(call)
                for kind, request in enumerate(selected):
                    self.calls[request] = (call, kind)
            source_bytes = sum(len(source.encode("utf-8")) for _, source in sources)
            largest_source = max(len(source.encode("utf-8")) for _, source in sources)
            runtime_compilations = len(sources)
        self.compilation_work = {
            "primitive_compiled_kernels": len(self.calls),
            "primitive_translation_units": len(groups),
            "primitive_generated_source_bytes": source_bytes,
            "primitive_largest_source_bytes": largest_source,
            "primitive_runtime_compilations": runtime_compilations,
            "primitive_packaged_aot": int(bool(packaged)),
        }
        self.buffer = np.zeros((primitive_tile, 17))
        self.records = 0

    def _run(self, request: tuple[str, tuple[str, ...]], count: int) -> np.ndarray:
        call, kind = self.calls[request]
        out = np.empty((4, 3))
        if call(
            kind,
            self.buffer.ctypes.data_as(ct.POINTER(ct.c_double)),
            count,
            out.ctypes.data_as(ct.POINTER(ct.c_double)),
        ):
            raise ArithmeticError("generated CPU component derivative failed")
        self.records += count
        return out

    def integral(
        self,
        operator: str,
        indices: tuple[int, ...],
        weight: float,
        nucleus: int | None = None,
    ) -> tuple[list[int], np.ndarray]:
        rows = self.aos[list(indices)]
        rank = len(indices)
        owners = [int(row[0]) for row in rows]
        if nucleus is not None:
            owners.append(nucleus)
        ranges = [range(int(row[1]), int(row[1] + row[2])) for row in rows]
        result = np.zeros((len(owners), 3))
        for terms in product(*(self.expansions[i] for i in indices)):
            binding = derivative_binding(operator, tuple(c for c, _ in terms))
            self.buffer.fill(0)
            self.buffer[:, 4 : 4 + 3 * len(owners)] = self.centers[owners][
                np.ix_(binding.centers, binding.axes)
            ].reshape(-1)
            norm = weight * np.prod([coefficient for _, coefficient in terms])
            count = 0
            # Do not fold ordered weights or assume a density symmetry. Only
            # exponent/coordinate slots change to reuse a compiled primitive.
            for ids in product(*ranges):
                primitives = self.primitives[list(ids)]
                self.buffer[count, :rank] = primitives[list(binding.centers[:rank]), 0]
                self.buffer[count, 16] = norm * np.prod(primitives[:, 1])
                count += 1
                if count == len(self.buffer):
                    result[np.ix_(binding.centers, binding.axes)] += self._run(
                        binding.request, count
                    )[: len(owners)]
                    count = 0
            if count:
                result[np.ix_(binding.centers, binding.axes)] += self._run(
                    binding.request, count
                )[: len(owners)]
        return owners, result

    def nuclear(self, a: int, b: int, charges: np.ndarray) -> np.ndarray:
        self.buffer.fill(0)
        self.buffer[0, :2] = charges[[a, b]]
        self.buffer[0, 4:10] = self.centers[[a, b]].reshape(-1)
        self.buffer[0, 16] = 1
        return self._run(("nuclear", ()), 1)[:2]
