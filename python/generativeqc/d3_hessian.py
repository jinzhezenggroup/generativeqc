"""Low-level native CPU D3(BJ) curvature; no electronic capability promotion."""

from __future__ import annotations

import os
import time
import typing
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.geometry.d3 import D3_REFERENCE_SLOTS, compile_d3_bj
from generativeqc_compiler.tensor.cpu import NativeTensorProgram
from generativeqc_compiler.tensor.types import checked_size


@dataclass(frozen=True)
class D3HVPResult:
    """Raw D3 gradient derivative in atom-major xyz order, Hartree/Bohr²."""

    direction: np.ndarray
    value: np.ndarray
    identity: str
    diagnostics: typing.Mapping[str, typing.Any]


@dataclass(frozen=True)
class D3HessianResult:
    """Unmodified D3 nuclear Hessian in atom-major xyz order, Hartree/Bohr²."""

    matrix: np.ndarray
    identity: str
    diagnostics: typing.Mapping[str, typing.Any]


class NativeD3SecondOrder:
    """Bind one immutable uncut two-body BJ geometry to a generated CPU HVP.

    ``spec`` is a canonical D3 specification, not a method-name shortcut. The
    shared GeometryIR/TensorIR energy supplies all derivatives; no finite
    differences, electronic response solve or external oracle runs here.
    ``cache`` retains native artifacts compiled with the existing finite-time
    compiler adapter and verified compiler-cache policy. Workspace and work
    admission precede compiler discovery. A separate generation budget admits
    a conservative 32-panel CN/C6 table reserve before building the AD graph;
    lowering checks actual native execution byte/work requirements separately.
    This low-level source does not
    advertise a total-method Calculator Hessian/HVP capability.
    """

    def __init__(
        self,
        spec: typing.Any,
        elements: typing.Iterable[int],
        coordinates: object,
        *,
        cache: Path,
        compiler: typing.Any = None,
        max_generation_bytes: int = 64 << 20,
        max_workspace_bytes: int = 64 << 20,
        max_work: int = 100_000_000,
    ) -> None:
        started = time.perf_counter()
        checked_size(max_generation_bytes, "D3 generation budget")
        checked_size(max_workspace_bytes, "D3 workspace budget")
        checked_size(max_work, "D3 work budget")
        numbers = np.asarray(tuple(elements))
        if numbers.ndim != 1 or numbers.dtype.kind not in "iu":
            raise ValueError("D3 elements must be a one-dimensional integer sequence")
        pairs = numbers.size * (numbers.size - 1) // 2
        reference_entries = pairs * D3_REFERENCE_SLOTS**2
        if reference_entries > max_work:
            raise ValueError("D3 table generation exceeds work budget")
        table_reserve = 32 * 8 * (reference_entries + numbers.size * D3_REFERENCE_SLOTS)
        if table_reserve >= max_generation_bytes:
            raise MemoryError(
                "D3 table-generation workspace exceeds max_generation_bytes"
            )
        compiled = compile_d3_bj(spec, numbers.tolist(), coordinates)
        second = compiled.coordinate_hvp()
        positions = immutable(np.asarray(coordinates, dtype=np.float64))
        reserve = 8 * positions.nbytes
        if reserve >= max_workspace_bytes:
            raise MemoryError("D3 coordinate/response workspace exceeds budget")
        self._native = NativeTensorProgram(
            second.program,
            compiler=compiler
            if compiler is not None
            else lambda: CppCompilerAdapter(Path(os.environ.get("CXX", "c++"))),
            cache=cache,
            max_bytes=max_workspace_bytes - reserve,
            max_work=max_work,
        )
        reverse = compiled.coordinate_vjp()
        self._seed = next(
            name for name, source in reverse.input_map.items() if source == "energy"
        )
        self._energy_seed = immutable(
            np.ones(compiled.program.outputs["energy"].spec.shape)
        )
        self._tangent = next(iter(second.input_map))
        self._output = next(iter(second.output_map))
        self._coordinate = compiled.geometry.coordinate_name
        self._positions = positions
        self._spec_identity = compiled.spec.identity
        self._pair_count = len(compiled.pair_state.topology.pairs)
        self._table_generation_reserve = table_reserve
        self._max_workspace_bytes = max_workspace_bytes
        self._reserved_bytes = reserve
        self.identity = canonical_hash(
            {
                "kind": "native-d3-bj-second-order/v1",
                "primal": compiled.identity,
                "second_equation": second.derivative_hash,
                "coordinates": positions.tolist(),
            }
        )
        self.preparation_seconds = time.perf_counter() - started

    def _diagnostics(self, evaluations: int, seconds: float) -> dict:
        return {
            "component": "d3_bj_two_body",
            "source_identity": self.identity,
            "specification_identity": self._spec_identity,
            "backend": "cpu",
            "derivative_method": "compiler-automatic-differentiation",
            "public_calculator_endpoint": False,
            "units": "hartree/bohr^2",
            "ordering": "atom-major/cartesian-xyz",
            "native_hvp_evaluations": evaluations,
            "electronic_response_solves": 0,
            "displaced_scf_evaluations": 0,
            "pairs": self._pair_count,
            "scalar_work_per_hvp": self._native.resources["scalar_work"],
            "workspace_bytes": self._native.resources["required_bytes"]
            + self._reserved_bytes,
            "table_generation_reserve_bytes": self._table_generation_reserve,
            "preparation_seconds": self.preparation_seconds,
            "assembly_seconds": seconds,
        }

    def hvp(self, direction: object) -> D3HVPResult:
        """Return the complete analytic D3 H@v without modifying the bound state."""
        started = time.perf_counter()
        vector = np.asarray(direction)
        if (
            vector.shape != self._positions.shape
            or np.iscomplexobj(vector)
            or not np.isfinite(vector).all()
        ):
            raise ValueError("D3 direction must be finite real [atom,3] data")
        vector = np.asarray(vector, dtype=np.float64)
        output = self._native.execute(
            {
                self._coordinate: self._positions,
                self._tangent: vector,
                self._seed: self._energy_seed,
            }
        )
        return D3HVPResult(
            immutable(vector),
            output[self._output],
            canonical_hash({"source": self.identity, "direction": vector.tolist()}),
            MappingProxyType(self._diagnostics(1, time.perf_counter() - started)),
        )

    def hessian(self, *, output_budget_bytes: int = 64 << 20) -> D3HessianResult:
        """Assemble every raw unit column transactionally within joint budgets."""
        started = time.perf_counter()
        checked_size(output_budget_bytes, "D3 Hessian output budget")
        dimension = self._positions.size
        output_bytes = dimension**2 * 8
        if output_bytes > output_budget_bytes:
            raise ValueError("D3 Hessian exceeds output_budget_bytes")
        required = (
            self._native.resources["required_bytes"]
            + self._reserved_bytes
            + 2 * output_bytes
        )
        if required > self._max_workspace_bytes:
            raise MemoryError("simultaneous D3 Hessian/output workspace exceeds budget")
        matrix = np.empty((dimension, dimension))
        direction = np.zeros_like(self._positions)
        for column in range(dimension):
            direction.reshape(-1)[column] = 1
            matrix[:, column] = self.hvp(direction).value.reshape(-1)
            direction.reshape(-1)[column] = 0
        diagnostics = self._diagnostics(dimension, time.perf_counter() - started)
        diagnostics["output_bytes"] = output_bytes
        diagnostics["simultaneous_bytes"] = required
        return D3HessianResult(
            immutable(matrix),
            canonical_hash({"source": self.identity, "kind": "raw-hessian"}),
            MappingProxyType(diagnostics),
        )
