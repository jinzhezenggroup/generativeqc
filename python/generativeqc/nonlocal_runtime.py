"""Bounded native fixed-grid execution for VV10/rVV10 pair mathematics.

This is a primitive runtime, not a public KS capability. It intentionally stays
separate from Calculator until the MethodIR/native KS contribution seam and
backend-specific capability gates are qualified.
"""

from __future__ import annotations

import ctypes
import typing
from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256

import numpy as np
from generativeqc_compiler.common.nonlocal_correlation import NonlocalCorrelationSpec
from generativeqc_compiler.common.provenance import canonical_hash

from . import _native

if typing.TYPE_CHECKING:
    from typing_extensions import Self


@dataclass(frozen=True)
class NonlocalRuntimeDiagnostic:
    """Owned numeric capacity and deterministic pair-work evidence."""

    backend: str
    workspace_bytes: int
    host_workspace_bytes: int
    device_workspace_bytes: int
    maximum_bytes: int
    pair_evaluations: int
    tiles: int
    point_count: int
    tile_points: int


@dataclass(frozen=True)
class NonlocalFixedGridResult:
    """One fixed-grid nonlocal energy plus requested derivative families."""

    energy: float
    vrho: np.ndarray | None
    vsigma: np.ndarray | None
    point_derivative: np.ndarray | None
    weight_derivative: np.ndarray | None
    backend: str
    identity: str
    plan_identity: str
    provider_identity: str | None
    host_workspace_bytes: int
    device_workspace_bytes: int
    pair_evaluations: int
    tiles: int
    tile_points: int


@dataclass(frozen=True)
class NonlocalBatchResult:
    """Atomic Python publication for a bounded ragged primitive batch."""

    results: tuple[NonlocalFixedGridResult, ...]
    pair_evaluations: int
    peak_owned_workspace_bytes: int


@dataclass(frozen=True)
class NonlocalResidentSeedView:
    """Borrowed device-resident [6,stride] force seeds and stream identity."""

    pointer: int
    stride: int
    stream: int
    generation: int


@dataclass(frozen=True)
class NonlocalResidentForceDiagnostic:
    """Bounded native resident-force owner state without forcing a CUDA fence."""

    device_bytes: int
    point_count: int
    collected_points: int
    generation: int
    executed: bool
    stream_bound: bool


def _positive_uint32(value: typing.Any, label: str) -> int:
    if type(value) is not int or not 0 < value < 2**32:
        raise ValueError(f"{label} must be a positive uint32 integer")
    return value


def _backend_name(value: int) -> str:
    if value == _native.BACKEND_CPU_REFERENCE:
        return "cpu"
    if value == _native.BACKEND_CUDA:
        return "cuda"
    return f"backend-{value}"


def _array(value: typing.Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise TypeError(f"{label} must be real")
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{label} requires shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


class NonlocalFixedGridPlan:
    """Persistent bounded CPU/CUDA plan for VV10/rVV10 pair execution."""

    def __init__(
        self,
        spec: NonlocalCorrelationSpec,
        point_count: int,
        *,
        coefficient: Fraction = Fraction(1),
        tile_points: int = 256,
        maximum_bytes: int = 256 << 20,
        device: str = "cpu",
        device_id: int = 0,
        library: typing.Any = None,
    ) -> None:
        if not isinstance(spec, NonlocalCorrelationSpec):
            raise TypeError("spec must be NonlocalCorrelationSpec")
        if not isinstance(coefficient, Fraction) or coefficient <= 0:
            raise ValueError("coefficient must be a positive Fraction")
        point_count = _positive_uint32(point_count, "point_count")
        if point_count > (2**32 - 1) // 3:
            raise ValueError(
                "point_count exceeds the flattened-coordinate uint32 domain"
            )
        tile_points = _positive_uint32(tile_points, "tile_points")
        if type(maximum_bytes) is not int or not 0 < maximum_bytes < 2**64:
            raise ValueError("maximum_bytes must be a positive uint64 integer")
        if device not in ("cpu", "cuda"):
            raise ValueError("native VV10 device must be cpu or cuda")
        if type(device_id) is not int or device_id < 0:
            raise ValueError("device_id must be a nonnegative integer")

        self.spec = spec
        self.coefficient = coefficient
        self.point_count = point_count
        self.device = device
        self.device_id = device_id
        self.maximum_bytes = maximum_bytes
        self._library = (
            _native.load_library(device=device, device_id=device_id)
            if library is None
            else library
        )
        if not hasattr(self._library, "generativeqc_nonlocal_plan_prepare"):
            raise RuntimeError(
                "loaded GENERATIVEQC library does not expose native nonlocal execution"
            )
        self._context = ctypes.c_void_p()
        self._plan = ctypes.c_void_p()
        context_descriptor = _native.ContextDescriptor(
            ctypes.sizeof(_native.ContextDescriptor),
            _native.ABI_VERSION,
            device_id,
            _native.BACKEND_CUDA if device == "cuda" else _native.BACKEND_CPU_REFERENCE,
        )
        variant = {
            "vv10": _native.NONLOCAL_VV10,
            "rvv10": _native.NONLOCAL_RVV10,
        }.get(spec.variant)
        if variant is None:
            raise NotImplementedError(f"unsupported nonlocal variant {spec.variant!r}")
        model = _native.NonlocalDescriptor(
            ctypes.sizeof(_native.NonlocalDescriptor),
            _native.ABI_VERSION,
            variant,
            float(spec.b),
            float(spec.c),
            float(coefficient),
            point_count,
            tile_points,
            maximum_bytes,
        )
        try:
            _native.check(
                self._library,
                self._library.generativeqc_context_create(
                    ctypes.byref(context_descriptor), ctypes.byref(self._context)
                ),
            )
            _native.check(
                self._library,
                self._library.generativeqc_nonlocal_plan_prepare(
                    self._context, ctypes.byref(model), ctypes.byref(self._plan)
                ),
                context=self._context,
            )
        except Exception:
            self.close()
            raise
        source = self._library.generativeqc_get_source_identity().decode()
        diagnostic = self.diagnostic()
        self._identity = canonical_hash(
            {
                "schema": "generativeqc.nonlocal-fixed-grid-plan/v2",
                "source": source,
                "spec": spec.identity,
                "variant": spec.variant,
                "b": str(spec.b),
                "c": str(spec.c),
                "coefficient": str(coefficient),
                "backend": diagnostic.backend,
                "device_id": device_id if diagnostic.backend == "cuda" else None,
                "point_count": point_count,
                "tile_points": diagnostic.tile_points,
                "maximum_bytes": maximum_bytes,
                "workspace_bytes": diagnostic.workspace_bytes,
                "host_workspace_bytes": diagnostic.host_workspace_bytes,
                "device_workspace_bytes": diagnostic.device_workspace_bytes,
                "derivatives": ("energy", "vrho", "vsigma", "geometry"),
                "response": False,
            }
        )

    @property
    def identity(self) -> str:
        return self._identity

    def close(self) -> None:
        if self._plan.value:
            self._library.generativeqc_nonlocal_plan_destroy(self._plan)
            self._plan.value = None
        if self._context.value:
            self._library.generativeqc_context_destroy(self._context)
            self._context.value = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if not self._plan.value or not self._context.value:
            raise RuntimeError("native nonlocal plan is closed")

    def diagnostic(self) -> NonlocalRuntimeDiagnostic:
        self._require_open()
        native = _native.NonlocalRuntimeDiagnostic(
            ctypes.sizeof(_native.NonlocalRuntimeDiagnostic),
            _native.ABI_VERSION,
        )
        _native.check(
            self._library,
            self._library.generativeqc_nonlocal_plan_get_diagnostic(
                self._plan, ctypes.byref(native)
            ),
            context=self._context,
        )
        return NonlocalRuntimeDiagnostic(
            backend=_backend_name(native.backend),
            workspace_bytes=int(native.workspace_bytes),
            host_workspace_bytes=int(native.host_workspace_bytes),
            device_workspace_bytes=int(native.device_workspace_bytes),
            maximum_bytes=int(native.maximum_bytes),
            pair_evaluations=int(native.pair_evaluations),
            tiles=int(native.tiles),
            point_count=int(native.point_count),
            tile_points=int(native.tile_points),
        )

    def execute(
        self,
        coordinates: typing.Any,
        weights: typing.Any,
        density: typing.Any,
        density_gradient: typing.Any,
        *,
        features: bool = True,
        geometry: bool = False,
    ) -> NonlocalFixedGridResult:
        self._require_open()
        if type(features) is not bool or type(geometry) is not bool:
            raise TypeError("features and geometry must be boolean")
        n = self.point_count
        coordinates = _array(coordinates, (n, 3), "coordinates")
        weights = _array(weights, (n,), "weights")
        density = _array(density, (n,), "density")
        density_gradient = _array(density_gradient, (n, 3), "density_gradient")
        if np.any(density <= 0):
            raise ValueError("native VV10 execution requires strictly positive density")

        vrho = np.empty(n, dtype=np.float64) if features else None
        vsigma = np.empty(n, dtype=np.float64) if features else None
        point = np.empty((n, 3), dtype=np.float64) if geometry else None
        weight = np.empty(n, dtype=np.float64) if geometry else None

        def pointer(value: np.ndarray | None) -> typing.Any:
            return (
                None
                if value is None
                else value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )

        input_descriptor = _native.NonlocalInputDescriptor(
            ctypes.sizeof(_native.NonlocalInputDescriptor),
            _native.ABI_VERSION,
            pointer(coordinates),
            coordinates.size,
            pointer(weights),
            weights.size,
            pointer(density),
            density.size,
            pointer(density_gradient),
            density_gradient.size,
        )
        output = _native.NonlocalResultDescriptor(
            ctypes.sizeof(_native.NonlocalResultDescriptor),
            _native.ABI_VERSION,
            0.0,
            pointer(vrho),
            0 if vrho is None else vrho.size,
            pointer(vsigma),
            0 if vsigma is None else vsigma.size,
            pointer(point),
            0 if point is None else point.size,
            pointer(weight),
            0 if weight is None else weight.size,
            0,
        )
        _native.check(
            self._library,
            self._library.generativeqc_nonlocal_plan_execute(
                self._plan, ctypes.byref(input_descriptor), ctypes.byref(output)
            ),
            context=self._context,
        )
        diagnostic = self.diagnostic()
        if _backend_name(output.executed_backend) != diagnostic.backend:
            raise RuntimeError(
                "native nonlocal execution backend changed after preparation"
            )
        result_identity = canonical_hash(
            {
                "schema": "generativeqc.nonlocal-fixed-grid-evaluation/v2",
                "plan": self.identity,
                "coordinates_sha256": sha256(coordinates.tobytes()).hexdigest(),
                "weights_sha256": sha256(weights.tobytes()).hexdigest(),
                "density_sha256": sha256(density.tobytes()).hexdigest(),
                "density_gradient_sha256": sha256(
                    density_gradient.tobytes()
                ).hexdigest(),
                "features": features,
                "geometry": geometry,
            }
        )
        return NonlocalFixedGridResult(
            energy=float(output.energy),
            vrho=None if vrho is None else vrho.copy(),
            vsigma=None if vsigma is None else vsigma.copy(),
            point_derivative=None if point is None else point.copy(),
            weight_derivative=None if weight is None else weight.copy(),
            backend=_backend_name(output.executed_backend),
            identity=result_identity,
            plan_identity=self.identity,
            provider_identity=None,
            host_workspace_bytes=diagnostic.host_workspace_bytes,
            device_workspace_bytes=diagnostic.device_workspace_bytes,
            pair_evaluations=diagnostic.pair_evaluations,
            tiles=diagnostic.tiles,
            tile_points=diagnostic.tile_points,
        )


class _ResidentNonlocalForceOwner:
    """Internal full-grid CUDA VV10/rVV10 force composition owner.

    Grid features are collected D2D from borrowed task leases, pair/geometry
    seeds stay resident, and publication returns only a borrowed pointer/stride
    identity. The caller must keep this owner alive through the stationary
    geometry drain that consumes the returned seed view.
    """

    _CREATE = "generativeqc_internal_nonlocal_cuda_force_create_v1"
    _DESTROY = "generativeqc_internal_nonlocal_cuda_force_destroy_v1"
    _COLLECT = "generativeqc_internal_nonlocal_cuda_force_collect_v1"
    _EXECUTE = "generativeqc_internal_nonlocal_cuda_force_execute_v1"
    _SEED_VIEW = "generativeqc_internal_nonlocal_cuda_force_seed_view_v1"
    _RESET = "generativeqc_internal_nonlocal_cuda_force_reset_v1"
    _METRICS = "generativeqc_internal_nonlocal_cuda_force_metrics_v1"

    def __init__(
        self,
        spec: NonlocalCorrelationSpec,
        coordinates: typing.Any,
        weights: typing.Any,
        *,
        coefficient: Fraction,
        tile_points: int,
        maximum_bytes: int,
        density_threshold: float,
        device_id: int,
        context: typing.Any,
        library: typing.Any,
    ) -> None:
        if not isinstance(spec, NonlocalCorrelationSpec):
            raise TypeError("spec must be NonlocalCorrelationSpec")
        if not isinstance(coefficient, Fraction) or coefficient <= 0:
            raise ValueError("coefficient must be a positive Fraction")
        if type(tile_points) is not int or tile_points <= 0:
            raise ValueError("tile_points must be a positive integer")
        if type(maximum_bytes) is not int or not 0 < maximum_bytes < 2**64:
            raise ValueError("maximum_bytes must be a positive uint64 integer")
        if type(device_id) is not int or device_id < 0:
            raise ValueError("device_id must be a nonnegative integer")
        if (
            not isinstance(density_threshold, (int, float))
            or not np.isfinite(density_threshold)
            or density_threshold <= 0
        ):
            raise ValueError("density_threshold must be finite and positive")
        points = np.ascontiguousarray(coordinates, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
            raise ValueError("resident nonlocal coordinates require shape [N,3]")
        quadrature = _array(weights, (len(points),), "weights")
        if not np.isfinite(points).all():
            raise ValueError("resident nonlocal coordinates must be finite")
        point_count = _positive_uint32(len(points), "point_count")
        if point_count > (2**32 - 1) // 3:
            raise ValueError(
                "point_count exceeds the flattened-coordinate uint32 domain"
            )
        context_pointer = (
            context
            if isinstance(context, ctypes.c_void_p)
            else ctypes.cast(context, ctypes.c_void_p)
        )
        if not context_pointer.value:
            raise ValueError("resident nonlocal force requires a live native context")
        required = (
            self._CREATE,
            self._DESTROY,
            self._COLLECT,
            self._EXECUTE,
            self._SEED_VIEW,
            self._RESET,
            self._METRICS,
        )
        missing = tuple(name for name in required if not hasattr(library, name))
        if missing:
            raise RuntimeError(
                "loaded GENERATIVEQC library lacks resident nonlocal force ABI: "
                + ", ".join(missing)
            )
        variant = {
            "vv10": _native.NONLOCAL_VV10,
            "rvv10": _native.NONLOCAL_RVV10,
        }.get(spec.variant)
        if variant is None:
            raise NotImplementedError(f"unsupported nonlocal variant {spec.variant!r}")
        model = _native.NonlocalDescriptor(
            ctypes.sizeof(_native.NonlocalDescriptor),
            _native.ABI_VERSION,
            variant,
            float(spec.b),
            float(spec.c),
            float(coefficient),
            point_count,
            tile_points,
            maximum_bytes,
        )
        double = ctypes.POINTER(ctypes.c_double)
        getattr(library, self._CREATE).argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_native.NonlocalDescriptor),
            double,
            ctypes.c_size_t,
            double,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        getattr(library, self._DESTROY).argtypes = [ctypes.c_void_p]
        getattr(library, self._DESTROY).restype = None
        getattr(library, self._COLLECT).argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        getattr(library, self._EXECUTE).argtypes = [ctypes.c_void_p]
        getattr(library, self._SEED_VIEW).argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        getattr(library, self._RESET).argtypes = [ctypes.c_void_p]
        getattr(library, self._METRICS).argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        for name in (
            self._CREATE,
            self._COLLECT,
            self._EXECUTE,
            self._SEED_VIEW,
            self._RESET,
            self._METRICS,
        ):
            getattr(library, name).restype = ctypes.c_int

        self.spec = spec
        self.coefficient = coefficient
        self.point_count = point_count
        self.device_id = device_id
        self.maximum_bytes = maximum_bytes
        self._library = library
        self._context = context_pointer
        self._owner = ctypes.c_void_p()
        self._executions = 0

        def pointer(array: np.ndarray) -> typing.Any:
            return array.ctypes.data_as(double)

        _native.check(
            library,
            getattr(library, self._CREATE)(
                context_pointer,
                ctypes.byref(model),
                pointer(points),
                points.size,
                pointer(quadrature),
                quadrature.size,
                float(density_threshold),
                ctypes.byref(self._owner),
            ),
            context=context_pointer,
        )

    def _require_open(self) -> None:
        if not self._owner.value:
            raise RuntimeError("resident nonlocal force owner is closed")

    def close(self) -> None:
        if self._owner.value:
            getattr(self._library, self._DESTROY)(self._owner)
            self._owner.value = None

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def collect(self, task: typing.Any, offset: int) -> None:
        """Enqueue one ordered resident feature tile without a host fence."""
        self._require_open()
        if type(offset) is not int or offset < 0:
            raise ValueError("resident nonlocal tile offset must be nonnegative")
        if task._owner.device_id != self.device_id:
            raise ValueError("resident nonlocal/grid current owner device mismatch")
        _native.check(
            self._library,
            getattr(self._library, self._COLLECT)(
                self._owner, ctypes.byref(task.view), offset
            ),
            context=self._context,
        )

    def execute(self) -> NonlocalResidentSeedView:
        """Enqueue MolecularV1 + VV10/rVV10 force work and borrow its seed view."""
        self._require_open()
        _native.check(
            self._library,
            getattr(self._library, self._EXECUTE)(self._owner),
            context=self._context,
        )
        pointer = ctypes.c_void_p()
        stride = ctypes.c_size_t()
        stream = ctypes.c_void_p()
        generation = ctypes.c_uint64()
        _native.check(
            self._library,
            getattr(self._library, self._SEED_VIEW)(
                self._owner,
                ctypes.byref(pointer),
                ctypes.byref(stride),
                ctypes.byref(stream),
                ctypes.byref(generation),
            ),
            context=self._context,
        )
        if (
            not pointer.value
            or not stream.value
            or stride.value != self.point_count
            or generation.value == 0
        ):
            raise RuntimeError(
                "resident nonlocal force owner returned an invalid seed view"
            )
        self._executions += 1
        return NonlocalResidentSeedView(
            int(pointer.value),
            int(stride.value),
            int(stream.value),
            int(generation.value),
        )

    def reset(self) -> None:
        """Reuse the same bounded owner for the next full-grid feature sequence."""
        self._require_open()
        _native.check(
            self._library,
            getattr(self._library, self._RESET)(self._owner),
            context=self._context,
        )

    def diagnostic(self) -> NonlocalResidentForceDiagnostic:
        self._require_open()
        values = (ctypes.c_uint64 * 6)()
        _native.check(
            self._library,
            getattr(self._library, self._METRICS)(self._owner, values, len(values)),
            context=self._context,
        )
        return NonlocalResidentForceDiagnostic(
            device_bytes=int(values[0]),
            point_count=int(values[1]),
            collected_points=int(values[2]),
            generation=int(values[3]),
            executed=bool(values[4]),
            stream_bound=bool(values[5]),
        )


class NativeNonlocalPairProvider:
    """Method-name-independent bounded provider injected into compiler integration."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        device_id: int = 0,
        memory_budget_bytes: int = 256 << 20,
        library: typing.Any = None,
    ) -> None:
        if device not in ("cpu", "cuda"):
            raise ValueError("native nonlocal provider device must be cpu or cuda")
        if type(device_id) is not int or device_id < 0:
            raise ValueError("device_id must be a nonnegative integer")
        if type(memory_budget_bytes) is not int or not 0 < memory_budget_bytes < 2**64:
            raise ValueError("memory_budget_bytes must be a positive uint64 integer")
        self.device = device
        self.device_id = device_id
        self.memory_budget_bytes = memory_budget_bytes
        self._library = (
            _native.load_library(device=device, device_id=device_id)
            if library is None
            else library
        )
        self._source_identity = (
            self._library.generativeqc_get_source_identity().decode()
        )
        self._identity = canonical_hash(
            {
                "schema": "generativeqc.native-nonlocal-pair-provider/v1",
                "source": self._source_identity,
                "backend": device,
                "device_id": device_id if device == "cuda" else None,
                "memory_budget_bytes": memory_budget_bytes,
                "precision": "float64",
                "capabilities": {
                    "energy": True,
                    "potential": True,
                    "first_nuclear_geometry": True,
                    "response_hv": False,
                },
            }
        )

    @property
    def identity(self) -> str:
        return self._identity

    @classmethod
    def from_fock(
        cls, fock: typing.Any, *, memory_budget_bytes: int = 256 << 20
    ) -> typing.Any:
        diagnostics = fock.diagnostics
        device = diagnostics["backend"]
        device_id = diagnostics["device_id"] if device == "cuda" else 0
        return cls(
            device=device,
            device_id=0 if device_id is None else int(device_id),
            memory_budget_bytes=memory_budget_bytes,
            library=fock._library,
        )

    def evaluate(
        self,
        coordinates: typing.Any,
        weights: typing.Any,
        density: typing.Any,
        density_gradient: typing.Any,
        spec: NonlocalCorrelationSpec,
        coefficient: Fraction,
        *,
        tile_points: int = 256,
        geometry: bool = False,
    ) -> NonlocalFixedGridResult:
        point_count = len(np.asarray(weights))
        with NonlocalFixedGridPlan(
            spec,
            point_count,
            coefficient=coefficient,
            tile_points=tile_points,
            maximum_bytes=self.memory_budget_bytes,
            device=self.device,
            device_id=self.device_id,
            library=self._library,
        ) as plan:
            result = plan.execute(
                coordinates,
                weights,
                density,
                density_gradient,
                features=True,
                geometry=geometry,
            )
        return replace(result, provider_identity=self.identity)

    def evaluate_batch(
        self,
        requests: typing.Iterable[tuple[typing.Any, ...]],
        *,
        tile_points: int = 256,
        geometry: bool = False,
    ) -> NonlocalBatchResult:
        staged: list[NonlocalFixedGridResult] = []
        pair_evaluations = 0
        peak = 0
        for index, request in enumerate(requests):
            try:
                result = self.evaluate(
                    *request, tile_points=tile_points, geometry=geometry
                )
            except Exception as error:
                raise RuntimeError(f"ragged member {index} failed: {error}") from error
            staged.append(result)
            pair_evaluations += result.pair_evaluations
            peak = max(
                peak,
                result.host_workspace_bytes + result.device_workspace_bytes,
            )
        return NonlocalBatchResult(tuple(staged), pair_evaluations, peak)
