"""Private device-array owner for the compiler-AD molecular Hessian endpoint.

CuPy supplies resident storage and CUDA linear algebra, not differentiation.
Scalar derivatives are emitted from the compiler's Graph; density/DF derivatives
are the existing TensorIR programs. There is deliberately no NumPy fallback.
"""

from __future__ import annotations

import ctypes as ct
import typing
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from generativeqc_compiler.common.native_runtime import compile_runtime
from generativeqc_compiler.common.paths import asset_path
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.source_cache import cache_source
from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.dft.ao_resident_cuda import emit_resident_ao
from generativeqc_compiler.integral.array_graph_cuda import emit_array_graph
from generativeqc_compiler.tensor.interpreter import Execution
from generativeqc_compiler.tensor.namespace_interpreter import execute_namespace


class _Namespace:
    """Adapt device placement and stacked inputs without implicit host copies."""

    __name__ = "generativeqc.cuda.resident"

    def __init__(self, cp: typing.Any) -> None:
        self.cp = cp

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self.cp, name)

    def asarray(
        self,
        value: typing.Any,
        dtype: typing.Any = None,
        *,
        device: typing.Any = None,
        copy: bool | None = None,
        order: str | None = None,
    ) -> typing.Any:
        if device is not None and device != self.cp.cuda.Device():
            raise ValueError("CUDA array feed belongs to a different device")
        if (
            isinstance(value, (list, tuple))
            and value
            and any(isinstance(entry, self.cp.ndarray) for entry in value)
        ):
            result = self.cp.stack([self.asarray(entry, dtype) for entry in value])
        else:
            result = self.cp.asarray(value, dtype=dtype, order=order)
        return result.copy() if copy else result

    def errstate(self, **kwargs: typing.Any) -> typing.Any:
        return nullcontext()

    def zeros(
        self, shape: tuple, dtype: typing.Any = None, *, device: typing.Any = None
    ) -> typing.Any:
        if device is not None and device != self.cp.cuda.Device():
            raise ValueError("CUDA tensor allocation requested on a different device")
        return self.cp.zeros(shape, dtype=dtype)

    def permute_dims(self, value: typing.Any, axes: tuple) -> typing.Any:
        return self.cp.transpose(value, axes)

    def astype(
        self, value: typing.Any, dtype: typing.Any, *, copy: bool = True
    ) -> typing.Any:
        return value.astype(dtype, copy=copy)


class CudaHessianExecution:
    """Own one stream, generated artifacts and aggregate device error status."""

    def __init__(self, compiler: typing.Any, *, device_id: int) -> None:
        try:
            import cupy
        except ImportError as error:
            raise RuntimeError(
                "fully resident CUDA Hessians require generativeqc[cuda-hessian]; "
                "the endpoint does not fall back to host differentiation"
            ) from error
        self.cp = cupy
        self.namespace = _Namespace(cupy)
        self.device = cupy.cuda.Device(device_id)
        with self.device:
            self.stream = cupy.cuda.Stream(non_blocking=True)
            self.pool = cupy.cuda.MemoryPool()
        self.compiler = compiler
        self.cache = None
        self.functions = {}
        self.graphs = {}
        self.status = None
        self.statistics = {
            "scalar_graph_launches": 0,
            "tensor_programs": 0,
            "ao_jet_launches": 0,
            "branch_metadata_bytes": 0,
            "numerical_host_fallbacks": 0,
            "intermediate_panel_downloads": 0,
        }

    def initialize(self, cache: typing.Any) -> None:
        self.cache = Path(cache)
        self.status = self.cp.zeros(1, dtype=self.cp.int32)

    def function(
        self, source: str, symbol: str, arguments: list, *, integral: bool = False
    ) -> typing.Any:
        key = canonical_hash({"source": source, "symbol": symbol})
        if key not in self.functions:
            directory = self.cache / "resident-hessian-sources"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (key + ".cu")
            cache_source(path, source)
            headers = (
                tuple(
                    asset_path("src/integrals/" + name)
                    for name in ("eri_geometry.hpp", "range_moments.hpp")
                )
                if integral
                else ()
            )
            options = ("--fmad=false",)
            if headers:
                options += ("-I" + str(headers[0].parents[1]),)
            artifact = compile_runtime(
                self.compiler,
                self.cache / "resident-hessian-native",
                path,
                headers=headers,
                options=options,
            )
            library = ct.CDLL(str(artifact.library))
            function = getattr(library, symbol)
            function.argtypes, function.restype = arguments, ct.c_int
            self.functions[key] = (library, function)
        return self.functions[key][1]

    def launch(self, function: typing.Any, *arguments: typing.Any) -> None:
        status = function(*arguments, self.status.data.ptr, self.stream.ptr)
        if status:
            raise RuntimeError(
                f"resident CUDA kernel launch failed: CUDA status {status}"
            )

    def check(self) -> None:
        self.stream.synchronize()
        if int(self.status[0]):
            raise FloatingPointError(
                "nonfinite or unsupported resident compiler-AD evaluation"
            )

    def freeze(self, value: typing.Any, *, shape: tuple | None = None) -> typing.Any:
        array = self.namespace.asarray(value)
        if array.dtype.kind == "c":
            raise ValueError("complex references are unsupported")
        array = self.cp.asarray(array, dtype=self.cp.float64, order="C")
        if shape is not None and array.shape != shape:
            raise ValueError(f"expected shape {shape}, got {array.shape}")
        if not bool(self.cp.isfinite(array).all()):
            raise ValueError("reference arrays must be finite")
        return array.copy()

    def tensor(
        self, program: typing.Any, feeds: dict, *, debug: bool, max_bytes: int
    ) -> Execution:
        result = execute_namespace(
            program,
            feeds,
            self.namespace,
            debug=debug,
            max_bytes=max_bytes,
            finite_check=self._finite_check,
        )
        self.statistics["tensor_programs"] += 1
        return Execution(*result)

    def _finite_check(self, value: typing.Any) -> None:
        self.status[:] |= (~self.cp.isfinite(value).all()).astype(self.cp.int32)

    def graph(self, graph: typing.Any, roots: tuple, variables: dict) -> tuple:
        roots = tuple(roots)
        names = tuple(sorted(variables))
        key = (id(graph), tuple(root.identifier for root in roots), names)
        if key not in self.graphs:
            source = emit_array_graph(graph, roots, names)
            function = self.function(
                source,
                "generativeqc_array_graph",
                [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_void_p, ct.c_void_p],
            )
            self.graphs[key] = (graph, function)
        function = self.graphs[key][1]
        shape = np.broadcast_shapes(*(np.shape(variables[name]) for name in names))
        inputs = (
            self.cp.stack(
                [
                    self.cp.broadcast_to(
                        self.cp.asarray(variables[name], dtype=self.cp.float64), shape
                    )
                    for name in names
                ]
            )
            if names
            else self.cp.empty((0, *shape))
        )
        outputs = self.cp.empty((len(roots), *shape), dtype=self.cp.float64)
        self.launch(function, inputs.data.ptr, int(np.prod(shape)), outputs.data.ptr)
        self.statistics["scalar_graph_launches"] += 1
        return tuple(outputs[index] for index in range(len(roots)))

    def branch_bytes(self, value: typing.Any) -> bytes:
        """Only discrete guard metadata may cross the boundary during a solve."""
        result = (
            value.tobytes()
            if isinstance(value, np.ndarray)
            else value.get(stream=self.stream).tobytes()
        )
        self.statistics["branch_metadata_bytes"] += len(result)
        return result

    def publish(self, array: typing.Any) -> np.ndarray:
        """The sole numerical tensor download: a completed public result."""
        self.check()
        return self.cp.asnumpy(array, stream=self.stream)


class CudaAO:
    """Borrow the native basis definition; evaluate all AO jets on the device."""

    def __init__(self, basis: typing.Any, execution: CudaHessianExecution) -> None:
        self.host, self.execution = basis, execution
        self.packed_device = execution.cp.asarray(basis.packed)
        self.function = execution.function(
            emit_resident_ao(),
            "generativeqc_resident_ao",
            [
                ct.c_void_p,
                ct.c_size_t,
                ct.c_size_t,
                ct.c_size_t,
                ct.c_void_p,
                ct.c_size_t,
                ct.c_size_t,
                ct.c_void_p,
                ct.c_void_p,
                ct.c_void_p,
            ],
        )

    def __getattr__(self, name: str) -> typing.Any:
        return getattr(self.host, name)

    def evaluate(
        self, points: typing.Any, order: int, *, budget_bytes: int = 256 << 20
    ) -> typing.Any:
        points = self.execution.cp.ascontiguousarray(
            self.execution.cp.asarray(points, dtype=self.execution.cp.float64)
        )
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or not bool(self.execution.cp.isfinite(points).all())
        ):
            raise ValueError("resident AO points require finite [point,3] coordinates")
        shape = (len(jet_indices(order)), len(points), self.nao)
        if 8 * np.prod(shape) > budget_bytes:
            raise MemoryError("resident AO jets exceed the requested byte budget")
        result = self.execution.cp.empty(shape)
        self.execution.launch(
            self.function,
            self.packed_device.data.ptr,
            self.natom,
            self.nprimitive,
            self.nao,
            points.data.ptr,
            len(points),
            shape[0],
            result.data.ptr,
        )
        self.execution.statistics["ao_jet_launches"] += 1
        return result
