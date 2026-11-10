"""Source-only executable witnesses for prepared stationary backend selection.

The real owner runs against inert compiler/device boundaries. No CUDA library,
scientific compilation or calculation participates in the capacity report.
"""

from __future__ import annotations

import ast
import builtins
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


def check_snapshot_reserve(owner: ast.FunctionDef) -> None:
    """Validate live-token and lazy-owner reserve behavior, not a source digest."""
    from generativeqc import _ks_snapshot as runtime
    from generativeqc_compiler.method.stationary_resources import (
        stationary_fitted_integral_reserve,
    )

    namespace = dict(vars(runtime))
    exec(  # noqa: S102 - inspected dry helper, inert token/capacity boundaries
        compile(
            ast.Module(body=[owner], type_ignores=[]), "<snapshot reserve>", "exec"
        ),
        namespace,
    )
    reserve = namespace[owner.name]
    for fitted, ranged, stale in (
        (True, False, False),
        (True, True, False),
        (False, False, False),
        (True, True, True),
    ):
        checked = []

        def current(stale: bool = stale, checked: list[bool] = checked) -> None:
            checked.append(True)
            if stale:
                raise LookupError("revoked native token")

        snapshot = SimpleNamespace(
            density_fitted=fitted,
            check_current=current,
            cuda_integral_gradient_additional_device_bytes=lambda: 5000,
        )
        try:
            actual = reserve(
                snapshot, atoms=3, aos=24, primitives=22, range_exchange=ranged
            )
        except LookupError:
            assert stale
        except ValueError:
            assert not fitted and not stale
        else:
            assert fitted and not stale
            assert actual == stationary_fitted_integral_reserve(
                atoms=3, aos=24, primitives=22
            ) + (5000 if ranged else 0)
        assert checked


def check_fitted_geometry(module: ModuleType) -> None:
    """Exercise real planning and provider admission without CUDA allocation.

    Independent scalar bounds cover known fitted, unknown fitted and Direct
    reserves. The complete endpoint then runs against inert grid/source owners
    until the first reset, after real native staging validation has completed.
    """
    import numpy as np
    from generativeqc._ks_snapshot import _integral_source_resources
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.method import resolve_method
    from generativeqc_compiler.method.stationary_resources import (
        stationary_cuda_allocation_bytes,
        stationary_native_pair_reserve,
    )

    ns = SimpleNamespace
    target = cuda_target_info("sm_120")
    basis = ns(
        natom=3,
        nao=24,
        nprimitive=22,
        numeric_bytes=2608 * 3,
        packed=ns(size=3 * 3 + 44 + 128 * 3),
    )
    expected = 12 << 20
    for fitted, known, allowance in (
        (False, True, 512 << 20),
        (True, False, 512 << 20),
        (True, True, 512 << 20),
        (True, True, 104 << 20),
    ):

        def reserve(*, atoms: int, aos: int, primitives: int) -> int:
            assert (atoms, aos, primitives) == (3, 24, 22)
            return expected

        source = ns(density_fitted=fitted, cuda_integral_derivatives=lambda: None)
        source.stationary_integral_device_reserve = reserve if known else None
        layout = module._plan_stationary_cuda_tile(
            ns(_source=source),
            basis,
            plan=ns(spin_blocks=1, range_exchange_sources=()),
            target=target,
            needs_first=True,
            tile_points=16,
            primitive_tile=32,
            integral_terms=4,
            source_names=tuple(range(8)),
            ecp=False,
            max_device_bytes=allowance,
            max_host_bytes=256 << 20,
            max_ecp_pair_samples=100_000_000,
        )
        minimum = stationary_cuda_allocation_bytes(
            atoms=3,
            aos=24,
            primitives=22,
            points=16,
            tasks=32,
            spins=1,
            sources=8,
            geometry_lanes=16,
        )
        available = max(0, allowance - layout.grid_plan.peak_bytes - minimum)
        required = (
            min(available, expected)
            if fitted and known
            else available
            if fitted
            else min(
                available,
                stationary_native_pair_reserve(atoms=3, aos=24, primitives=22),
            )
        )
        assert layout.native_geometry_reserve == required
        assert (
            layout.source_resources.allocation_bytes
            + layout.grid_plan.peak_bytes
            + required
            <= allowance
        )

    class Admitted(Exception):
        pass

    class Owner:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def configure_grid(self, *args: Any) -> None:
            pass

        def reset_geometry(self, *args: Any) -> None:
            raise Admitted

    class Compiler:
        def __init__(self) -> None:
            self.target = target

    module.CudaCompilerAdapter = Compiler
    module.os = ns(environ={"GENERATIVEQC_DIRECT_FORCE_REDUCTION": "separate"})

    class Contract(module.StationaryDerivativeContract):
        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            return ns(family="gga", spin="unpolarized", validate=lambda state: state)

    module.StationaryDerivativeContract = Contract
    module.native_ao_geometry_identity = lambda basis: "geometry"
    module._layout = lambda basis: (
        np.array([[1.0, 1.0]]),
        np.array([[0, 0, 1]]),
        ((("", 1.0),),),
        module.qualified_sp_requests(),
    )
    layout = ns(
        grid_plan=ns(
            peak_bytes=16, tile_points=1, active_ao_capacity=None, features_only=True
        ),
        tensor_plans={},
        host_bound=16,
        native_integral_host_reserve=32,
        native_geometry_reserve=32,
        ecp_workspace=0,
        ecp_pair_samples=0,
        source_resources=ns(allocation_bytes=16),
    )
    module._plan_stationary_cuda_tile = lambda *args, **kwargs: layout
    module._prepare_force_memory = lambda *args: {}
    module.compile_stationary_cuda = lambda *args, **kwargs: object()
    module._native_grid_artifact = lambda *args, **kwargs: object()
    module.CudaGrid = module._CudaSources = lambda *args, **kwargs: Owner()

    for atoms, fitted, known, large_device, large_host, actual in (
        (2, True, True, False, False, 32),
        (2, True, True, False, False, 33),
        (2, False, True, False, False, 33),
        (96, False, True, False, False, 32),
        (96, True, True, False, False, 32),
        (96, True, True, True, True, 32),
        (96, True, True, True, False, 32),
        (96, True, True, False, True, 32),
        (96, True, False, False, False, 32),
        (2, True, True, True, True, 32),
    ):
        source = ns(
            backend="cuda",
            density_fitted=fitted,
            method_ir=resolve_method("PBE"),
            hamiltonian="all-electron",
            ecp_cores=(0,) * atoms,
            ecp_terms=(),
            ecp_derivatives=lambda: None,
            metadata=(3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            grid_spec=ns(partition_iterations=3, coincident_tolerance=1e-12),
            functional=ns(ingredients=("rho", "sigma")),
        )
        if known:
            source.stationary_integral_device_reserve = lambda **kwargs: 32

        def provider(
            *args: Any, atoms: int = atoms, actual: int = actual, **kwargs: Any
        ) -> Any:
            return np.zeros((4, atoms, 3)), _integral_source_resources(
                (0, 0, actual, 0, 0, 0, 0, 0, 0)
            )

        source.cuda_integral_derivatives = (
            source.density_fitted_integral_derivatives
        ) = provider
        basis = ns(
            identity="basis",
            natom=atoms,
            nao=1,
            nprimitive=1,
            charge=0,
            atoms=tuple(ns(atomic_number=1) for _ in range(atoms)),
        )
        state = ns(
            _source=source,
            identity=ns(basis_identity="basis", geometry_identity="geometry"),
            occupations=np.array([float(atoms)]),
            density=np.zeros((1, 1, 1)),
            weighted_density=np.zeros((1, 1, 1)),
            grid=ns(points=np.zeros((1, 3))),
        )
        preferences = []

        def schedule(preferences: list[int] = preferences, **kwargs: Any) -> Any:
            preferences.append(kwargs["preferred_tile_points"])
            admitted, work = kwargs["admit"](1, False)
            return admitted, work, None

        module._plan_stationary_optional_ao_schedule = schedule
        rejected = False
        try:
            module._complete_rks_cuda_gradient_diagnostic(
                state,
                basis,
                compiler=Compiler(),
                target=target,
                cache=Path("/inert-cache"),
                native_grid_library=Path("/inert-grid.so"),
                tile_points=1,
                max_device_bytes=(1 << 30) if large_device else 64,
                max_host_bytes=(1 << 30) if large_host else 256 << 20,
            )
        except Admitted:
            pass
        except RuntimeError as error:
            assert "device staging exceeds" in str(error)
            rejected = True
        else:
            raise AssertionError("provider probe executed beyond admission")
        assert rejected == (fitted and actual > 32)
        expected_preference = (
            1024
            if atoms == 96 and large_device and large_host
            else 256
            if atoms == 96 and fitted and known
            else 512
        )
        assert preferences == [expected_preference]


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
        "plan": ns(identity="plan", spin_blocks=1, range_exchange_sources=()),
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
    module.stationary_runtime_sources = lambda *args, **kwargs: ()
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


def check_resident_ao_binding(owner: ast.FunctionDef) -> None:
    """Execute the current map binding with inert storage and revocable state."""
    ns = SimpleNamespace

    def require(condition: bool, message: str = "binding invariant") -> None:
        if not condition:
            raise RuntimeError(message)

    previous = None
    fail_allocation = False

    class Cache:
        def __init__(self, grid: Any, domain: Any, **policy: Any) -> None:
            require(previous is None or previous() is None, "old map still retained")
            if fail_allocation:
                raise MemoryError("map staging")
            self.grid, self.domain, self.policy = grid, domain, policy
            self.work: dict[str, Any] = {}

        def reset_work(self) -> None:
            self.work.clear()

    adapters = ns(
        ResidentAoMapDomain=lambda **fields: ns(**fields),
        ResidentAoMapCache=Cache,
        ResidentDeviceAoMapOwner=Cache,
    )

    def imports(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "_resident_ao_maps":
            return adapters
        return builtins.__import__(name, *args, **kwargs)

    scope: dict[str, Any] = {
        "__builtins__": {**vars(builtins), "__import__": imports},
    }
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            owner,
        ],
        type_ignores=[],
    )
    # Execute only the repository-owned helper with inert map adapters.
    exec(  # noqa: S102
        compile(ast.fix_missing_locations(code), "<resident-ao-binding>", "exec"), scope
    )
    cache = scope[owner.name]
    binding = ns(_resident_ao_maps=None, _resident_ao_map_key=None)
    grid = ns(
        basis_identity="basis",
        geometry_generation=1,
        basis_generation=1,
        plan=ns(tile_points=16, order=2),
    )
    resident = ns(device=1, points=4096, point_count=32)

    def current() -> None:
        pass

    state = ns(
        _source=ns(check_current=current),
        identity=ns(geometry_identity="geometry", grid_identity="grid"),
    )

    def run(**policy: Any) -> Any:
        return cache(
            binding,
            grid,
            state,
            resident,
            **({"cutoff": 1e-16, "budget_bytes": 64} | policy),
        )

    try:
        first = run()
        require(
            vars(first.domain)
            == {
                "basis_identity": "basis",
                "geometry_identity": "geometry",
                "grid_identity": "grid",
                "device": 1,
                "point_pointer": 4096,
                "point_count": 32,
                "tile_points": 16,
                "derivative_order": 2,
            }
        )
        first.work["previous_endpoint"] = 7
        state.identity.density_generation = 2
        require(run() is first and not first.work)
        del first
        for obj, field, value in (
            (state.identity, "geometry_identity", "moved"),
            (state.identity, "grid_identity", "new-grid"),
            (grid, "basis_identity", "new-basis"),
            (grid, "geometry_generation", 2),
            (grid, "basis_generation", 2),
            (grid.plan, "order", 1),
            (grid.plan, "tile_points", 8),
            (resident, "device", 2),
            (resident, "points", 8192),
            (resident, "point_count", 16),
        ):
            previous = weakref.ref(binding._resident_ao_maps)
            setattr(obj, field, value)
            run()
            require(previous() is None)
            previous = None
        previous = weakref.ref(binding._resident_ao_maps)
        grid = ns(**vars(grid))
        run()
        require(previous() is None)
        previous = None

        def revoked() -> None:
            raise RuntimeError("revoked snapshot")

        state._source.check_current = revoked
        try:
            run()
        except RuntimeError as error:
            require(str(error) == "revoked snapshot")
        else:
            raise AssertionError("revoked snapshot reused")
        state._source.check_current = current
        previous = weakref.ref(binding._resident_ao_maps)
        fail_allocation = True
        try:
            run(cutoff=1e-18)
        except MemoryError:
            pass
        else:
            raise AssertionError("replacement failure swallowed")
        require(previous() is None)
        require(binding._resident_ao_maps is binding._resident_ao_map_key is None)
        previous = None
        fail_allocation = False
        for disabled in (True, False):
            run()
            require(
                cache(
                    binding,
                    grid,
                    state,
                    resident if disabled else None,
                    cutoff=None if disabled else 1e-16,
                    budget_bytes=64,
                )
                is None
            )
            require(binding._resident_ao_maps is binding._resident_ao_map_key is None)
    except Exception as error:
        raise RuntimeError(
            f"stationary CUDA resident AO policy lifecycle contract changed: {error}"
        ) from error
