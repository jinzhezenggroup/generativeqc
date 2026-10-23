"""Source-only executable witnesses for prepared stationary backend selection.

The real owner runs against inert compiler/device boundaries. No CUDA library,
scientific compilation or calculation participates in the capacity report.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


def check_prepared_selection(module: ModuleType) -> None:
    """Check request identity, artifact routing, precision and early budgets."""
    ns = SimpleNamespace
    target = ns(architecture="sm_120", to_payload=lambda: {"architecture": "sm_120"})
    state = ns(
        identity=ns(
            method="pbe",
            functional_identity="functional",
            regularization_identity="regularization",
        ),
        _source=ns(backend="cuda", ecp_cores=(), ecp_terms=()),
    )
    arguments: dict[str, Any] = {
        "state": state,
        "basis": ns(
            identity="basis", topology="topology", natom=2, nao=4, nprimitive=4
        ),
        "contract": ns(family="dft", spin="rks"),
        "plan": ns(identity="plan", spin_blocks=1),
        "tensor_plans": {},
        "target": target,
        "aot_directory": Path("/packaged"),
        "native_grid_library": Path("/native-grid.so"),
        "ecp": False,
        "device": 0,
        "spec": ns(partition_iterations=3),
        "grid_plan": ns(allocation_bytes=64, peak_bytes=128, active_ao_capacity=4),
        "source_bytes": 256,
        "tile_points": 16,
        "primitive_tile": 8,
        "integral_terms": 4,
        "page_work_budget": 128,
        "resident_ao_cutoff": 1e-16,
        "resident_ao_cache_bytes": 4096,
        "resident_ao_producer": "sampled-jets",
        "resident_ao_max_active_fraction": 0.75,
        "integral_derivatives": True,
        "fp32_semilocal_geometry": True,
    }
    module._basis_topology_identity = lambda basis: basis.topology
    module._resolve_restricted_point_policy = lambda: False
    owner_type = module.PreparedStationaryCudaExecution
    owner = owner_type()
    baseline = owner._request(**arguments)
    # These are scientific/schedule/resource identity changes, not source-shape
    # checks. Equal requests would allow an incompatible prepared replay.
    variants = {
        "resident_ao_cutoff": None,
        "resident_ao_cache_bytes": 0,
        "resident_ao_producer": "exact-jets-native-bitmask",
        "resident_ao_max_active_fraction": 0.5,
        "fp32_semilocal_geometry": False,
        "integral_derivatives": False,
        "aot_directory": None,
        "native_grid_library": None,
        "device": 1,
        "tile_points": 32,
        "primitive_tile": 16,
        "integral_terms": 8,
        "page_work_budget": 256,
        "source_bytes": 512,
        "ecp": True,
        "basis": ns(topology="changed"),
        "plan": ns(identity="changed"),
        "contract": ns(family="dft", spin="uks"),
        "target": ns(to_payload=lambda: {"architecture": "sm_90"}),
        "grid_plan": ns(allocation_bytes=128, peak_bytes=256),
        "tensor_plans": {"x": ns(identity="tensor", peak_bytes=64, host_bytes=32)},
        "spec": ns(partition_iterations=4),
    }
    for key, value in variants.items():
        if owner._request(**(arguments | {key: value})) == baseline:
            raise RuntimeError(
                "prepared stationary AO request contract changed: " + key
            )
    module._resolve_restricted_point_policy = lambda: True
    if owner._request(**arguments) == baseline:
        raise RuntimeError(
            "prepared stationary AO request contract changed: restricted point"
        )

    class ReachedSourceBoundary(Exception):
        """Stop after observing all cold backend choices, before native work."""

    seen: dict[str, Any] = {}
    artifact = ns(library=Path("/inert-artifact.so"), metadata={})

    def stationary(route: str, *args: Any, **kwargs: Any) -> Any:
        seen["stationary"] = route
        return artifact

    def grid(route: str, *args: Any, **kwargs: Any) -> Any:
        seen["grid"] = route
        return artifact

    def sources(*args: Any, **kwargs: Any) -> Any:
        seen["fp32"] = kwargs["fp32_geometry"]
        raise ReachedSourceBoundary

    module.compile_stationary_cuda = lambda *a, **k: stationary("jit", *a, **k)
    module.load_stationary_aot_artifact = lambda *a, **k: stationary("aot", *a, **k)
    module.compile_grid = lambda *a, **k: grid("jit", *a, **k)
    module._native_grid_artifact = lambda *a, **k: grid("native", *a, **k)
    module._prepare_force_memory = lambda *a: {}
    module._layout = lambda *a: (None, None, (), None)
    module._component_mode = lambda *a: False
    module.stationary_runtime_sources = lambda *a: ()
    module.CudaGrid = lambda *a, **k: nullcontext(ns(artifact=artifact))
    module._CudaSources = sources
    call = arguments | {
        "compiler": ns(target=target),
        "cache": Path("/inert-cache"),
        "requests": (("nuclear", ()),),
        "functional": 1,
        "max_device_bytes": 8192,
        "max_host_bytes": 8192,
        "host_bound": 512,
    }
    for aot in (None, Path("/packaged")):
        for ecp in (False, True):
            for native in (None, Path("/native-grid.so")):
                for fp32 in (False, True):
                    seen.clear()
                    owner = owner_type()
                    original_request = owner._request

                    def request(
                        _original: Any = original_request, **kwargs: Any
                    ) -> Any:
                        seen["request_fp32"] = kwargs["fp32_semilocal_geometry"]
                        return _original(**kwargs)

                    owner._request = request
                    try:
                        owner.ensure(
                            **(
                                call
                                | {
                                    "aot_directory": aot,
                                    "ecp": ecp,
                                    "native_grid_library": native,
                                    "fp32_semilocal_geometry": fp32,
                                }
                            )
                        )
                    except ReachedSourceBoundary:
                        pass
                    else:
                        raise RuntimeError(
                            "prepared stationary AOT selection contract changed: source boundary"
                        )
                    expected = {
                        "stationary": "jit" if aot is None or ecp else "aot",
                        "grid": "jit" if native is None else "native",
                        "fp32": fp32,
                        "request_fp32": fp32,
                    }
                    if seen != expected:
                        raise RuntimeError(
                            "prepared stationary AOT selection contract changed: routing/precision"
                        )
    for limit in ("max_device_bytes", "max_host_bytes"):
        seen.clear()
        try:
            owner_type().ensure(**(call | {limit: 1}))
        except ValueError:
            if seen:
                raise RuntimeError(
                    "prepared stationary AOT selection contract changed: late budget"
                ) from None
        else:
            raise RuntimeError(
                "prepared stationary AOT selection contract changed: budget accepted"
            )
