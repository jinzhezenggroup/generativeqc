"""Complete numeric-capacity planning for native semilocal prepared KS owners.

The dry run resolves shell metadata and integer shapes, never quadrature,
integrals, density matrices, eigensolvers or a CUDA execution context. CPU
workspace bounds cover the existing dense reference route. CUDA device state
uses allocator-owned layout queries and the common #202 direct-J boundary.
All estimates compose through the existing #203 ResourcePlan and one budget.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import typing
from dataclasses import asdict, replace

from generativeqc_compiler.common.resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    checked_bytes,
)

from ._cpu_force_resources import CPU_FORCE_HOST_CAP
from ._cuda_force_resources import ordinary_cuda_force_limits
from ._snapshot_grid_cache import SNAPSHOT_GRID_CACHE_BYTES
from .basis import BasisSet
from .basis_capabilities import require_basis
from .calculator import Atom, _snapshot_basis
from .elements import electron_state
from .ks import (
    KsOptions,
    _native_semilocal_code,
    composite_stationary_second_order_eligible,
    cuda_pbe_range_stationary_second_order_eligible,
    is_canonical_r2scan_rvv10,
    resolve_ks_method,
    resolve_ks_options,
)
from .resources_hf import (
    _basis_record,
    _cuda_library_identity,
    _ecp_inventory_owner,
    _ecp_workspace,
)
from .resources_planning import plan_resources

_CPU_AO_GRID_CACHE_CAP = 64 << 20
_CUDA_FITTED_METHODS = frozenset(
    {"pbe-rks", "pbe-uks", "pbe0-rks", "pbe0-uks", "b3lyp-rks", "b3lyp-uks"}
)
_CUDA_FITTED_ENERGY_METHODS = _CUDA_FITTED_METHODS | {
    "lda-rks",
    "lda-uks",
    "r2scan-rks",
    "r2scan-uks",
}
_CUDA_STATIONARY_INVENTORY_KEYS = (
    "grid_numeric",
    "provider_allowance",
    "stationary_minimum",
    "stationary_cached",
    "minimum",
    "cached",
    "tile_points",
    "geometry_lanes",
    "geometry_threads",
)
_CPU_HOST_INVENTORY_KEYS = (
    "metadata",
    "grid",
    "basis",
    "warm_and_matrices",
    "history",
    "provider",
    "solver_host",
    "solver_binding_host",
    "matrix_binding_host",
    "structured_binding_host",
    "xc_schedule_staging",
    "nonlocal_provider",
    "ao_grid_cache",
    "retained",
    "setup_workspace",
    "scf_workspace",
)


_CPU_HOST_PERSISTENT_KEYS = _CPU_HOST_INVENTORY_KEYS[:12]
_CPU_PACKED_FLEET_SYMBOL = "generativeqc_resource_ks_cpu_packed_fleet_v1"
_CPU_PACKED_FORCE_FLEET_SYMBOL = "generativeqc_resource_ks_cpu_packed_force_fleet_v1"
_CPU_FORCE_FLEET_SYMBOL = "generativeqc_resource_ks_cpu_force_fleet_v1"
_CPU_DF_FLEET_SYMBOL = "generativeqc_resource_ks_cpu_df_fleet_v1"
_CPU_DF_FORCE_FLEET_SYMBOL = "generativeqc_resource_ks_cpu_df_force_fleet_v1"


def native_cpu_df_program(request: ResourceRequest) -> bytes | None:
    """Recover the prepared selector from a freshly resolved CPU DF request."""
    if request.name != "ks" or request.identity.backend != "cpu":
        return None
    controls = json.loads(request.identity.schedule)
    if controls.get("fleet_inventory") not in (
        "native-cpu-df-v1",
        "native-cpu-df-force-v1",
    ):
        return None
    return controls["cpu_df_named_program"].encode("ascii")


def native_dense_force_request(request: ResourceRequest) -> bool:
    """The smaller native bound cannot fund the Python compatibility consumer."""
    return (
        request.name == "ks"
        and request.identity.backend == "cpu"
        and "forces" in request.identity.observables
        and json.loads(request.identity.schedule).get("fleet_inventory")
        == "native-cpu-force-v1"
    )


class _PackedEriCapacityExceeded(ValueError):
    """The exact retained fleet does not fit its separate provider cap."""


class _MissingCpuKsInventory(NotImplementedError):
    """A selected CPU runtime cannot authoritatively size this KS fleet."""


def _cpu_host_shape(
    item: typing.Any,
    *,
    library: typing.Any,
    diis_history: int,
    max_iterations: int,
    requires_first_ao_derivatives: bool,
    model: typing.Any,
    df_forces: bool = False,
) -> tuple[int, ...]:
    """Marshal the same compact topology for the per-item and fleet ABIs.

    Automatic CPU and numeric-budget owners retain the 256-point incumbent;
    the unpromised CUDA preparation policy cannot enlarge a quoted owner.
    """
    orbital = item["orbital"]
    dimensions = (
        item["atoms"],
        orbital["nbf"],
        orbital["cartesian_nbf"],
        orbital["shells"],
        orbital["primitives"],
        item["spins"],
        orbital["maximum_angular"],
        orbital.get("ecp_terms", 0),
        item["grid_points"],
        model.grid.radial_points,
        model.grid.angular_polar,
        max(
            model.tile_points or 256, (model.force_tile_points or 0) if df_forces else 0
        ),
        diis_history,
        max_iterations,
        int(requires_first_ao_derivatives),
        int(model.has_nonlocal_correlation),
        _ecp_workspace(item, library=library),
    )
    return tuple(checked_bytes(value, "KS host shape") for value in dimensions)


def _cpu_host_result(
    values: typing.Sequence[int],
    model: typing.Any,
    *,
    check_nonlocal_budget: bool = True,
) -> dict[str, int]:
    """Decode capacities without waiving the explicit nonlocal sub-budget."""
    inventory = {
        key: checked_bytes(value, f"KS {key}")
        for key, value in zip(_CPU_HOST_INVENTORY_KEYS, values, strict=True)
    }
    if (
        check_nonlocal_budget
        and model.has_nonlocal_correlation
        and inventory["nonlocal_provider"] > model.nonlocal_memory_budget_bytes
    ):
        raise ValueError(
            "KS nonlocal provider workspace exceeds nonlocal_memory_budget_bytes"
        )
    return inventory


def _native_cpu_fleet_inventory(
    library: typing.Any,
    items: typing.Sequence[typing.Any],
    *,
    workspace_floor: int,
    **options: typing.Any,
) -> tuple[list[dict[str, int]], dict[str, int], int] | None:
    """Preserve the CPU fleet ABI and its transient-only consumer allowance."""
    return _native_host_fleet_inventory(
        library, items, backend="cpu", workspace_floor=workspace_floor, **options
    )


def _native_host_fleet_inventory(
    library: typing.Any,
    items: typing.Sequence[typing.Any],
    *,
    backend: str,
    workspace_floor: int = 0,
    retained_allowance: int = 0,
    packed_cap: int = 0,
    packed_forces: bool = False,
    dense_forces: bool = False,
    cpu_fitted: bool = False,
    cpu_fitted_forces: bool = False,
    **options: typing.Any,
) -> tuple[list[dict[str, int]], dict[str, int], int] | None:
    """Query all retained owners and serialized workspace before publication.

    Consumer allowances describe lifetimes, not provider admission. CUDA cache
    and force owners remain retained across replay; CPU force scratch remains
    a transient floor. A selected CPU runtime must provide fleet sizing and
    any active ECP workspace sizing. CUDA retains its separate compatibility.
    Transport consistency checks validate publication without retrying sizing.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("KS host inventory requires cpu or cuda backend")
    cuda = backend == "cuda"
    if cpu_fitted:
        symbol = (
            _CPU_DF_FORCE_FLEET_SYMBOL if cpu_fitted_forces else _CPU_DF_FLEET_SYMBOL
        )
    elif dense_forces:
        symbol = _CPU_FORCE_FLEET_SYMBOL
    elif packed_forces:
        symbol = _CPU_PACKED_FORCE_FLEET_SYMBOL
    elif packed_cap:
        symbol = _CPU_PACKED_FLEET_SYMBOL
    elif cuda:
        symbol = "generativeqc_resource_ks_cuda_host_fleet_v1"
    else:
        symbol = "generativeqc_resource_ks_cpu_fleet_v1"
    query = getattr(library, symbol, None)
    if query is None:
        if cpu_fitted:
            raise NotImplementedError("native library lacks CPU DF KS host inventory")
        if packed_cap:
            raise NotImplementedError(
                "packed ERI plans require a native host inventory"
            )
        if not cuda and library is not None:
            raise _MissingCpuKsInventory(
                f"selected library lacks the CPU KS fleet inventory {symbol}"
            )
        return None
    if (
        not cuda
        and any(item["orbital"].get("ecp_terms") for item in items)
        and getattr(library, "generativeqc_resource_ecp_workspace_v1", None) is None
    ):
        raise _MissingCpuKsInventory(
            "selected library lacks the CPU KS ECP workspace inventory "
            "generativeqc_resource_ecp_workspace_v1"
        )
    if not cuda and retained_allowance:
        raise ValueError("the CPU fleet ABI has no retained consumer allowance")
    dimensions = tuple(
        value
        for item in items
        for value in (
            *_cpu_host_shape(
                item, library=library, df_forces=cpu_fitted_forces, **options
            ),
            *(
                (
                    item["auxiliary"][key]
                    for key in (
                        "nbf",
                        "cartesian_nbf",
                        "shells",
                        "primitives",
                        "maximum_angular",
                    )
                )
                if cpu_fitted
                else ()
            ),
            *((int(options["model"].xc_schedule == "host_unfused"),) if cuda else ()),
        )
    )
    shapes = (ctypes.c_uint64 * len(dimensions))(*dimensions)
    values = (ctypes.c_uint64 * (15 + 16 * len(items)))()
    floor = checked_bytes(workspace_floor, "KS workspace allowance")
    allowance = checked_bytes(retained_allowance, "KS retained consumer allowance")
    controls = (
        () if dense_forces else (allowance, floor) if cuda else (packed_cap or floor,)
    )
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        *([ctypes.c_uint64] * len(controls)),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    status = query(
        shapes,
        len(shapes),
        *controls,
        values,
        len(values),
    )
    if packed_cap and status == 1:
        raise _PackedEriCapacityExceeded(
            "packed spherical ERI capacity exceeded for retained KS batch"
        )
    if packed_forces and status == 2:
        raise NotImplementedError("this build omits native CPU stationary forces")
    if dense_forces and status == 2:
        return None
    if cpu_fitted and status == 2:
        raise NotImplementedError(
            "CPU DF inventory excludes materialized DF derivatives"
        )
    if status:
        raise ValueError(
            f"native {backend.upper()} KS fleet inventory rejected shape or capacity"
        )
    aggregate = [
        checked_bytes(value, f"{backend.upper()} KS fleet capacity")
        for value in values[:15]
    ]
    if aggregate[12] != sum(aggregate[:12]) + allowance or aggregate[14] != sum(
        aggregate[12:14]
    ):
        raise ValueError(
            f"native {backend.upper()} KS fleet inventory returned inconsistent peak"
        )
    host = [
        _cpu_host_result(
            values[15 + 16 * index : 31 + 16 * index],
            options["model"],
            check_nonlocal_budget=not cuda,
        )
        for index in range(len(items))
    ]
    if dense_forces and any(
        not row["retained"] or not row["scf_workspace"] for row in host
    ):
        raise ValueError("native CPU KS fleet inventory omitted a required force item")
    totals = dict(zip(_CPU_HOST_PERSISTENT_KEYS, aggregate[:12], strict=True))
    if any(
        row["retained"] != sum(row[key] for key in _CPU_HOST_PERSISTENT_KEYS)
        for row in host
    ):
        raise ValueError(
            "native KS fleet inventory returned inconsistent retained item"
        )
    if any(totals[key] != sum(row[key] for row in host) for key in totals):
        raise ValueError(
            "native KS fleet inventory returned inconsistent component totals"
        )
    if aggregate[13] != max(
        floor, *(max(row["setup_workspace"], row["scf_workspace"]) for row in host)
    ):
        raise ValueError(
            "native KS fleet inventory returned inconsistent serialized workspace"
        )
    return host, totals, aggregate[13]


def _incremental_direct_jk_controls() -> dict[str, bool]:
    """Match both native benchmark selectors, including legacy OR semantics."""
    controls = {}
    for name in (
        "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK",
        "GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK",
    ):
        value = os.environ.get(name)
        if value not in (None, "0", "off", "1", "on"):
            raise ValueError(f"{name} must be 0/off or 1/on")
        controls[name] = value in ("1", "on")
    return controls


def _item_host_inventory(
    item: typing.Any,
    *,
    diis_history: typing.Any,
    max_iterations: typing.Any,
    requires_first_ao_derivatives: typing.Any,
    backend: typing.Any,
    model: typing.Any,
    library: typing.Any = None,
) -> typing.Any:
    """Bound numeric and LP64 value metadata by actual execution lifetimes.

    Every plan retains its host grid, basis, warm/source metadata and provider.
    Preparation and CPU SCF are serialized across items; CUDA SCF state and
    histories coexist for all active streams. A geometry rebuild destroys its
    old owner before allocating the new one, retaining only its last-good seed.
    """
    orbital = item["orbital"]
    a, n, c = item["atoms"], orbital["nbf"], orbital["cartesian_nbf"]
    s, p, spins = orbital["shells"], orbital["primitives"], item["spins"]
    n2, c2 = byte_product(n, n), byte_product(c, c)
    points = item["grid_points"]
    packed = 3 * a + 2 * p + 16 * n
    # One owned grid has xyz/weights/owner. Source-system copies, AO records,
    # descriptors and plan objects have a conservative LP64 value allowance.
    # Include fixed element-radius tables copied into method/grid/XC snapshots.
    metadata = 8192 + 512 * (a + s + p + n + c + orbital.get("ecp_terms", 0))
    # Optional terminal DIIS readback: two uint64 counters and count/head.
    # The matching device record is charged by the native state layout query.
    if backend == "cuda" and os.environ.get("GENERATIVEQC_DF_PROGRESS_TRACE"):
        metadata += 24
    # Shared native XC potential binding, including bounded preparation scratch.
    if backend == "cuda" and model.xc_schedule != "host_unfused":
        metadata += 16 << 10
    grid = byte_product(36, points)
    basis = byte_product(8, packed)
    warm_and_matrices = byte_product(8, n2, 4 + 4 * spins) + 8 * 3 * a * 4
    # 128 bytes covers ScfIteration (checked by native inventory v1); the
    # factor two includes CPU vector growth or the CUDA owner plus its exported
    # history. The adapter/ABI move that export through to its cached handle.
    history = byte_product(256, max_iterations)
    provider = byte_product(8, 2 * n2 + (n2 * n2 if backend == "cpu" else 0))
    # The provider retains S/H; do not count those again as a second owner.
    retained = metadata + grid + basis + warm_and_matrices + history + provider
    # Ordinary KS reuses one serialized Xsyevd workspace across spins. Match
    # the shared native admission bound; actual provider queries are checked
    # before either allocation. Small native solves need no provider workspace.
    solver_host = (
        1024 * 1024 + byte_product(16, 8, n2) if backend == "cuda" and n > 16 else 0
    )
    # Match the native two-candidate solver metadata reservation independently
    # of queried numeric workspace; tiny native matrices also retain a binding.
    solver_binding_host = 16 * 1024 if backend == "cuda" else 0
    # The immutable matrix-product binding has its own preparation envelope,
    # including generated small-shape and host-unfused XC execution. It is not
    # part of the independent eigensolver or XC binding reservations.
    matrix_binding_host = 16 * 1024 if backend == "cuda" else 0
    # Structured Gram/basis bindings borrow the same provider, but retain a
    # separate preparation envelope even while library qualification is closed.
    structured_binding_host = 32 * 1024 if backend == "cuda" else 0
    retained += (
        solver_host
        + solver_binding_host
        + matrix_binding_host
        + structured_binding_host
    )
    quadrature = (
        16 * (2 * model.grid.radial_points + 3 * model.grid.angular_polar) + 8 * a
    )
    if getattr(model.grid, "version", 1) == 4:
        # Native immutable element rules coexist with their contiguous upload
        # source and the final grid. Reserve their atom-expanded upper bound,
        # pair corrections, prefix metadata and bounded vector bookkeeping.
        quadrature += byte_product(80, points) + byte_product(8, a, a) + 64 * a + 8
    if backend == "cuda":
        # Shared rule/center/radius upload coexists with the small rule vectors.
        quadrature += 8 * (4 * a + 2 * (512 + 256))
    matrix_work = byte_product(8, spins, n2, 128 + 2 * (diis_history + 1))
    matrix_work += byte_product(16, diis_history + 1, diis_history + 1)
    host_unfused = backend == "cuda" and model.xc_schedule == "host_unfused"
    xc_tile = (
        byte_product(
            8,
            min(points, model.tile_points or 256),
            n,
            4 if requires_first_ao_derivatives else 1,
        )
        + byte_product(8, spins, n2)
        if backend == "cpu" or host_unfused
        else 0
    )
    # run_rks owns this optional cache only for the current serialized solve.
    # Reserve its full eligible footprint even when an experimental XC route
    # declines it; never charge all batch caches as persistent/coexisting state.
    ao_grid_cache = 0
    if (
        backend == "cpu"
        and requires_first_ao_derivatives
        and spins == 1
        and n > 0
        and points <= _CPU_AO_GRID_CACHE_CAP // 32 // n
    ):
        ao_grid_cache = byte_product(32, points, n)
    # Match CudaKsPlan's retained host staging exactly. Host-unfused owns one
    # density and one Vxc matrix per spin; UKS additionally owns split alpha/
    # beta matrices for the audited CPU integrator.
    xc_schedule_staging = (
        byte_product(8, n2, 2 * spins + (2 if spins == 2 else 0)) if host_unfused else 0
    )
    retained += xc_schedule_staging
    nonlocal_provider = 0
    nonlocal_work = 0
    if model.has_nonlocal_correlation:
        # Vv10Plan retains omega/kappa/weighted-density and three local
        # derivatives: six FP64 arrays. The AO bridge separately owns rho,
        # grad-rho, vrho/vsigma, one first-derivative AO tile and V_nlc.
        nonlocal_provider = byte_product(8, points, 6)
        if backend == "cpu" and nonlocal_provider > model.nonlocal_memory_budget_bytes:
            raise ValueError(
                "KS nonlocal provider workspace exceeds nonlocal_memory_budget_bytes"
            )
        matrix_factor = 3 if spins == 2 else 1
        nonlocal_work = (
            byte_product(8, points, 6)
            + byte_product(8, min(points, model.tile_points or 256), n, 4)
            + byte_product(8, n2, matrix_factor)
        )
        retained += nonlocal_provider
    if backend == "cpu":
        # Value-only Jet objects retain no derivative arrays. Raw Cartesian
        # Jet integrals coexist with unpacked and spherical transform buffers.
        angular = orbital["maximum_angular"]
        hermite = 6 * (angular + 3) ** 2 * (2 * angular + 4)
        recurrence = 64 * ((4 * angular + 1) ** 4 + hermite + 256 + a)
        integrals = byte_product(64, 2 * c2 + byte_product(c2, c2))
        integrals += max(recurrence, 32 * (c2 * c2 + n2 * n2 + c * n + n2))
        setup = integrals + quadrature + basis
    else:
        pairs = s * (s + 1) // 2
        # Common HostBatch packing retains primitive/shell-pair metadata and
        # may form PSSS task descriptors. Bound vector growth, source copies,
        # AO transforms and Cartesian one-electron downloads without packing.
        setup = 2048 * (1 + a + s + p + n + c + pairs + pairs * pairs)
        setup += 64 * (c2 + n2 + n * c) + quadrature + basis
    setup += _ecp_workspace(item, library=library)
    return {
        key: checked_bytes(value, f"KS {key}")
        for key, value in {
            "metadata": metadata,
            "grid": grid,
            "basis": basis,
            "warm_and_matrices": warm_and_matrices,
            "history": history,
            "provider": provider,
            "solver_host": solver_host,
            "solver_binding_host": solver_binding_host,
            "matrix_binding_host": matrix_binding_host,
            "structured_binding_host": structured_binding_host,
            "xc_schedule_staging": xc_schedule_staging,
            "nonlocal_provider": nonlocal_provider,
            "ao_grid_cache": ao_grid_cache,
            "retained": retained,
            "setup_workspace": setup,
            "scf_workspace": matrix_work + xc_tile + nonlocal_work + ao_grid_cache,
        }.items()
    }


def _cuda_leaf_inventory(
    library: typing.Any,
    item: typing.Any,
    *,
    diis_history: typing.Any,
    requires_first_ao_derivatives: typing.Any,
    functional_code: int,
    requires_tau: bool,
    tile: typing.Any,
    rebuild_exchange: bool | None = None,
    rebuild_range: bool = False,
    nonlocal_limit: int | None = None,
    wb97mv: bool = False,
) -> typing.Any:
    """Query the existing allocator-owned leaves without composing lifetimes."""
    orbital = item["orbital"]
    nonlocal_host = extra_quadrature = 0
    if nonlocal_limit is not None:
        from . import _native

        query = getattr(library, "generativeqc_resource_ks_cuda_nonlocal_v1", None)
        if query is None:
            raise NotImplementedError("native library lacks CUDA nonlocal KS inventory")
        args = (
            orbital["nbf"],
            item["atoms"],
            orbital["primitives"],
            item["grid_points"],
            diis_history,
            item["spins"],
            tile,
            item["nonlocal_grid_points"],
            int("nonlocal_grid_host_bytes" in item),
            int(wb97mv),
        )
        output = (ctypes.c_uint64 * 4)()
        query.argtypes = [ctypes.c_size_t] * 10 + [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        if any(value > 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1 for value in args):
            raise ValueError("KS resource shape exceeds the host size_t ABI")
        status = query(*args, nonlocal_limit, output, len(output))
        if status == _native.STATUS_OUT_OF_MEMORY:
            raise MemoryError(
                "KS nonlocal workspace exceeds nonlocal_memory_budget_bytes"
            )
        if status:
            raise NotImplementedError(
                "native CUDA nonlocal inventory rejected this shape/build"
            )
        extra_quadrature, nonlocal_host = output[2:]
    elif rebuild_exchange is not None:
        # This query counts only mutable KS state and XC/grid. Direct J/K can
        # use it too, provided the fleet supplies its own Fock source capacity.
        query = getattr(library, "generativeqc_resource_ks_cuda_rebuild_v1", None)
        if query is None:
            raise NotImplementedError(
                "native library lacks CUDA KS rebuild state/XC inventory"
            )
        args = (
            orbital["nbf"],
            item["atoms"],
            orbital["primitives"],
            item["grid_points"],
            diis_history,
            item["spins"],
            functional_code,
            tile,
            int(rebuild_exchange),
            int(rebuild_range),
        )
    else:
        query = getattr(library, "generativeqc_resource_ks_cuda_v2", None)
        selector = functional_code
        if query is None:
            if functional_code >= 0x30000 or requires_tau:
                raise NotImplementedError(
                    "tau-dependent CUDA KS planning requires point-program allocation inventory v2"
                )
            query = getattr(library, "generativeqc_resource_ks_cuda_v1", None)
            selector = int(requires_first_ao_derivatives)
        if query is None:
            raise NotImplementedError(
                "native library has no KS CUDA allocation inventory v1/v2"
            )
        args = (
            orbital["nbf"],
            item["atoms"],
            orbital["shells"],
            orbital["primitives"],
            item["grid_points"],
            diis_history,
            item["spins"],
            selector,
            tile,
        )
    if nonlocal_limit is None:
        if any(value > 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1 for value in args):
            raise ValueError("KS resource shape exceeds the host size_t ABI")
        query.argtypes = [ctypes.c_size_t] * len(args) + [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        output = (ctypes.c_uint64 * (3 if rebuild_exchange is None else 2))()
        if query(*args, output, len(output)):
            if functional_code >= 0x30000:
                raise NotImplementedError(
                    "automatic KS CUDA inventory requires a packaged static point provider; "
                    "build with GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=ON and use "
                    "valid launch/DIIS dimensions"
                )
            raise NotImplementedError(
                "KS CUDA inventory requires a CUDA build and valid launch/DIIS dimensions"
            )
    quadrature_query = getattr(
        library, "generativeqc_resource_quadrature_cuda_v1", None
    )
    if quadrature_query is None:
        raise NotImplementedError(
            "native library has no CUDA quadrature allocation inventory"
        )
    quadrature_query.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    quadrature_query.restype = ctypes.c_int
    quadrature = ctypes.c_uint64()
    if quadrature_query(item["atoms"], item["grid_points"], ctypes.byref(quadrature)):
        raise NotImplementedError("invalid CUDA quadrature resource shape/build")
    # Quadrature is built after the Coulomb provider. The native v1 query
    # includes bounded generation scratch plus the resident full-grid buffer.
    # The XC slot is the borrowed private arena plus the full shared grid owner,
    # including atomic weights retained for force consumers. Charge that combined
    # lifetime once per item; only setup excess is serialized across the fleet.
    provider_query = getattr(
        library, "generativeqc_resource_ks_matrix_provider_cuda_v1", None
    )
    if provider_query is None:
        raise NotImplementedError("native library has no KS matrix-provider inventory")
    provider_query.argtypes = [ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
    provider_query.restype = ctypes.c_int
    matrix_provider = ctypes.c_uint64()
    if provider_query(orbital["nbf"], ctypes.byref(matrix_provider)):
        raise NotImplementedError(
            "invalid CUDA KS matrix-provider resource shape/build"
        )
    return {
        "state": checked_bytes(int(output[0])),
        "xc": checked_bytes(int(output[1])),
        "coulomb": checked_bytes(int(output[2]))
        if rebuild_exchange is None and nonlocal_limit is None
        else 0,
        "quadrature_setup": checked_bytes(quadrature.value + extra_quadrature),
        "nonlocal_host": checked_bytes(
            nonlocal_host + item.get("nonlocal_grid_host_bytes", 0)
        ),
        "matrix_provider": checked_bytes(matrix_provider.value),
        "ecp_workspace": checked_bytes(
            _ecp_workspace(item, cuda=True, library=library)
        ),
    }


def _cuda_item_inventory(
    library: typing.Any, item: typing.Any, **options: typing.Any
) -> typing.Any:
    """Retain explicit per-item compatibility for libraries without composition."""
    leaf = _cuda_leaf_inventory(library, item, **options)
    orbital = item["orbital"]
    cartesian_aos, shells = orbital["cartesian_nbf"], orbital["shells"]
    one_electron = byte_product(
        64,
        1
        + item["atoms"]
        + shells
        + orbital["primitives"]
        + cartesian_aos
        + shells * (shells + 1) // 2
        + cartesian_aos * cartesian_aos,
    )
    return {
        "state": leaf["state"],
        "xc": leaf["xc"],
        "coulomb": leaf["coulomb"],
        "setup": max(
            one_electron,
            leaf["ecp_workspace"],
            checked_bytes(leaf["coulomb"] + leaf["quadrature_setup"]),
        ),
        "quadrature_setup": leaf["quadrature_setup"],
        "matrix_provider": leaf["matrix_provider"],
    }


class _CudaFockCapacity(typing.NamedTuple):
    retained: int
    initial_setup: int
    late_excess: int
    host: int


def _native_fitted_fock_inventory(
    library: typing.Any, item: typing.Any, *, memory_budget: int, exchange: bool
) -> _CudaFockCapacity:
    """Size the full-range DF owner and its restricted determinant projection."""
    query = getattr(library, "generativeqc_resource_prepared_df_fock_cuda_v1", None)
    if query is None:
        raise NotImplementedError("native library lacks CUDA fitted Fock inventory")
    dimensions = [item["atoms"]]
    for basis in (item["orbital"], item["auxiliary"]):
        dimensions.extend(
            basis[key] for key in ("nbf", "cartesian_nbf", "shells", "primitives")
        )
    restricted = item["spins"] == 1
    rank = item["electrons"]["nalpha"] if exchange and restricted else 0
    # Only restricted J-only DF promotes pair storage without an occupied rank.
    dimensions.extend((memory_budget, 0, rank, 1, int(restricted and not exchange)))
    shape = (ctypes.c_uint64 * len(dimensions))(*dimensions)
    values = (ctypes.c_uint64 * 3)()
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    if query(shape, len(shape), values, len(values)):
        raise NotImplementedError(
            "native CUDA fitted Fock inventory rejected the source"
        )
    retained, setup, host = (checked_bytes(value) for value in values)
    if (
        not 0 < retained <= setup
        or not host
        or (memory_budget and retained != memory_budget)
    ):
        raise ValueError("native CUDA fitted Fock inventory is inconsistent")
    return _CudaFockCapacity(retained, setup, 0, host)


def _native_radial_fock_inventory(
    library: typing.Any, item: typing.Any, *, memory_budget: int, radial_model: int
) -> tuple[_CudaFockCapacity, str]:
    """Decode the complete native initial/retry envelope and its LR selection."""
    from . import _native

    query = getattr(library, "generativeqc_resource_ks_cuda_radial_fock_v1", None)
    if query is None:
        raise NotImplementedError("native library lacks CUDA radial Fock inventory")
    orbital = item["orbital"]
    auxiliary = item.get("auxiliary")
    dimensions = [item["atoms"]]
    for basis in (orbital, auxiliary):
        dimensions.extend(
            basis[key] if basis is not None else 0
            for key in ("nbf", "cartesian_nbf", "shells", "primitives")
        )
    fitted = auxiliary is not None
    rank = item["electrons"]["nalpha"] if fitted and item["spins"] == 1 else 0
    dimensions.extend(
        (
            memory_budget,
            rank,
            int(fitted),
            radial_model,
            orbital.get("ecp_terms", 0),
        )
    )
    shape = (ctypes.c_uint64 * len(dimensions))(*dimensions)
    values = (ctypes.c_uint64 * 5)()
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    status = query(shape, len(shape), values, len(values))
    if status == _native.STATUS_INVALID_ARGUMENT:
        raise ValueError("invalid radial Fock shape or GENERATIVEQC_DF_RANGE_EXCHANGE")
    if status == _native.STATUS_OUT_OF_MEMORY:
        raise MemoryError("KS radial owners exceed the available DF sub-budget")
    if status:
        raise NotImplementedError(
            "native CUDA radial Fock inventory rejected the source"
        )
    modes = ("direct", "df", "auto")
    retained, setup, late, host, policy = (checked_bytes(value) for value in values)
    if (
        not retained
        or not setup
        or not host
        or policy >= len(modes)
        or (not fitted and policy)
    ):
        raise ValueError("native CUDA radial Fock inventory is inconsistent")
    return _CudaFockCapacity(retained, setup, late, host), modes[policy]


def _native_cuda_fleet_inventory(
    library: typing.Any,
    items: typing.Sequence[typing.Any],
    stationary: typing.Sequence[typing.Any],
    *,
    device_fused_xc: bool,
    fock: typing.Sequence[_CudaFockCapacity] | None = None,
    **options: typing.Any,
) -> tuple[list[dict[str, int]], dict[str, int]] | None:
    """Compose native leaves without Python setup/peak/candidate arithmetic.

    Only absence permits the legacy per-item route. Rejection or an incoherent
    wire publication is authoritative and never retries portable sizing. The
    query itself is shape-only, including in CPU-only builds; leaf availability
    remains the selected CUDA library's separate contract.
    """
    symbol = (
        "generativeqc_resource_ks_cuda_fleet_v2"
        if fock is not None
        else "generativeqc_resource_ks_cuda_fleet_v1"
    )
    query = getattr(library, symbol, None)
    if query is None:
        if fock is not None:
            raise NotImplementedError(
                "native Fock capacity requires CUDA fleet inventory v2"
            )
        return None
    if len(items) != len(stationary):
        raise ValueError("CUDA KS fleet and stationary inventory lengths differ")
    native_stationary = all(consumer is not None for consumer in stationary)
    leaves = [_cuda_leaf_inventory(library, item, **options) for item in items]
    if fock is not None:
        for leaf, owner in zip(leaves, fock, strict=True):
            leaf["coulomb"] = owner.retained
            leaf["fock_setup"] = owner.initial_setup
            leaf["deferred_setup"] = owner.late_excess
    dimensions = []
    for item, leaf, consumer in zip(items, leaves, stationary, strict=True):
        orbital = item["orbital"]
        dimensions.extend(
            checked_bytes(value, "CUDA KS fleet shape")
            for value in (
                orbital["cartesian_nbf"],
                item["atoms"],
                orbital["shells"],
                orbital["primitives"],
                leaf["state"],
                leaf["xc"],
                leaf["coulomb"],
                leaf["quadrature_setup"],
                leaf["matrix_provider"],
                leaf["ecp_workspace"],
                int(device_fused_xc),
                consumer["minimum"] if native_stationary else 0,
                consumer["cached"] if native_stationary else 0,
                *(
                    (leaf["fock_setup"], leaf["deferred_setup"])
                    if fock is not None
                    else ()
                ),
            )
        )
    shapes = (ctypes.c_uint64 * len(dimensions))(*dimensions)
    values = (ctypes.c_uint64 * (9 + 6 * len(items)))()
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    if query(shapes, len(shapes), values, len(values)):
        raise ValueError("native CUDA KS fleet inventory rejected shape or capacity")
    capacities = [checked_bytes(value, "CUDA KS fleet capacity") for value in values]
    aggregate = capacities[:9]
    if (
        aggregate[7] != sum(aggregate[:5]) + aggregate[5]
        or aggregate[8] != sum(aggregate[:5]) + aggregate[6]
    ):
        raise ValueError("native CUDA KS fleet inventory returned inconsistent peaks")
    keys = ("state", "xc", "coulomb", "setup", "quadrature_setup", "matrix_provider")
    device = [
        dict(zip(keys, capacities[9 + 6 * index : 15 + 6 * index], strict=True))
        for index in range(len(items))
    ]
    for inventory, leaf in zip(device, leaves, strict=True):
        if leaf.get("nonlocal_host"):
            inventory["nonlocal_host"] = leaf["nonlocal_host"]
        if leaf.get("deferred_setup"):
            inventory["deferred_setup"] = leaf["deferred_setup"]
        for key in ("state", "xc", "coulomb", "quadrature_setup", "matrix_provider"):
            expected = 0 if key == "xc" and not device_fused_xc else leaf[key]
            if inventory[key] != expected:
                raise ValueError("native CUDA KS fleet changed allocator-owned leaves")
        if inventory["setup"] < max(
            leaf["ecp_workspace"],
            leaf["coulomb"] + leaf["quadrature_setup"],
            leaf.get("fock_setup", 0),
        ):
            raise ValueError("native CUDA KS fleet omitted a declared setup owner")
    for index, key in enumerate(("state", "xc", "coulomb", "matrix_provider")):
        if aggregate[index] != sum(inventory[key] for inventory in device):
            raise ValueError(
                "native CUDA KS fleet returned inconsistent retained totals"
            )
    expected_setup_extra = max(
        (
            max(
                0,
                inventory["setup"]
                - sum(inventory[key] for key in ("state", "xc", "coulomb")),
                inventory.get("deferred_setup", 0),
            )
            for inventory in device
        ),
        default=0,
    )
    if aggregate[4] != expected_setup_extra:
        raise ValueError("native CUDA KS fleet returned inconsistent setup excess")
    for index, mode in ((5, "minimum"), (6, "cached")):
        expected = (
            sum(consumer[mode] for consumer in stationary) if native_stationary else 0
        )
        if aggregate[index] != expected:
            raise ValueError("native CUDA KS fleet changed stationary consumer totals")
    return device, dict(
        zip(
            (
                "state",
                "xc",
                "coulomb",
                "matrix_provider",
                "setup_extra",
                "minimum",
                "cached",
                "minimum_peak",
                "cached_peak",
            ),
            aggregate,
            strict=True,
        )
    )


def _cuda_stationary_descriptor(
    model: typing.Any,
    *,
    precision: str,
    max_iterations: int,
    diis_history: int,
    energy_tolerance: float,
    density_tolerance: float,
    screening_tolerance: float,
    application_precision_budget: typing.Any = None,
    library: typing.Any = None,
) -> typing.Any:
    """Serialize the resolved model; native admission owns scientific predicates."""
    from . import _generated_methods, _native
    from .ks import native_dft_carrier_for_ir, native_ks_options

    descriptor_type = (
        _native.ApplicationPrecisionMethodDescriptor
        if application_precision_budget is not None
        else _native.MethodDescriptor
    )
    descriptor = descriptor_type(
        struct_size=ctypes.sizeof(descriptor_type),
        abi_version=_native.ABI_VERSION,
        method=_generated_methods.METHOD_NAME_TO_ID[
            native_dft_carrier_for_ir(model.method_ir)
        ],
        max_iterations=max_iterations,
        diis_history=diis_history,
        energy_tolerance=energy_tolerance,
        density_tolerance=density_tolerance,
        screening_tolerance=screening_tolerance,
        density_fitting_mode=_native.DENSITY_FITTING_NONE,
        precision_mode=_native.PRECISION_FP64
        if precision == "fp64"
        else _native.PRECISION_AUTO,
    )
    descriptor.ks_options = ctypes.pointer(
        native_ks_options(model, library=library, backend=_native.BACKEND_CUDA)
    )
    if application_precision_budget is not None:
        descriptor.application_energy_budget = application_precision_budget.energy_abs
        descriptor.application_max_force_budget = (
            application_precision_budget.force_max_abs
        )
        descriptor.application_rms_force_budget = (
            application_precision_budget.force_rms or 0.0
        )
    return descriptor


def native_cuda_stationary_model_available(
    library: typing.Any, model: typing.Any, **controls: typing.Any
) -> bool:
    """Discover a linked model through native policy before a molecule exists.

    This upper bound is useful for Calculator output discovery. A real shape
    still needs the capacity query and prepared-item capability; this function
    neither probes CUDA nor substitutes a Python functional whitelist.
    """
    from . import _native

    query = getattr(library, "generativeqc_ks_stationary_cuda_model_available_v1", None)
    if query is None:
        return False
    descriptor = _cuda_stationary_descriptor(model, library=library, **controls)
    available = ctypes.c_int()
    query.argtypes = [
        ctypes.POINTER(_native.MethodDescriptor),
        ctypes.POINTER(ctypes.c_int),
    ]
    query.restype = ctypes.c_int
    status = query(ctypes.byref(descriptor), ctypes.byref(available))
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return False
    _native.check(library, status)
    return bool(available.value)


def _native_cuda_stationary_inventory(
    library: typing.Any, item: typing.Any, descriptor: typing.Any
) -> dict[str, int] | None:
    """Discover a complete per-item consumer, without CUDA or final-state work.

    Missing/unsupported bindings retain the incumbent fleet route. An invalid
    descriptor or capacity is not a fallback: surface the native failure.
    """
    from . import _native

    query = getattr(library, "generativeqc_resource_ks_stationary_cuda_v1", None)
    if query is None:
        return None
    orbital = item["orbital"]
    dimensions = (
        item["atoms"],
        orbital["nbf"],
        orbital["primitives"],
        item["grid_points"],
        orbital["maximum_angular"],
        orbital.get("ecp_terms", 0),
        # Basis metadata retains one core count per atom; the native shape ABI
        # takes their scalar total to distinguish ECP from all-electron owners.
        sum(
            checked_bytes(core, "ECP core electrons")
            for core in orbital.get("ecp_core_electrons", ())
        ),
    )
    shape = (ctypes.c_uint64 * len(dimensions))(
        *(checked_bytes(value, "CUDA KS stationary shape") for value in dimensions)
    )
    output = (ctypes.c_uint64 * len(_CUDA_STATIONARY_INVENTORY_KEYS))()
    query.argtypes = [
        ctypes.POINTER(_native.MethodDescriptor),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    status = query(ctypes.byref(descriptor), shape, len(shape), output, len(output))
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status)
    inventory = dict(zip(_CUDA_STATIONARY_INVENTORY_KEYS, output, strict=True))
    if (
        inventory["minimum"]
        != inventory["grid_numeric"]
        + inventory["provider_allowance"]
        + inventory["stationary_minimum"]
        or inventory["cached"]
        != inventory["grid_numeric"]
        + inventory["provider_allowance"]
        + inventory["stationary_cached"]
        or not 0 < inventory["minimum"] <= inventory["cached"]
    ):
        raise ValueError("native CUDA KS stationary inventory is inconsistent")
    return {
        key: checked_bytes(value, f"CUDA KS stationary {key}")
        for key, value in inventory.items()
    }


def configure_native_cuda_stationary_batch(
    library: typing.Any, batch: typing.Any, plan: typing.Any
) -> None:
    """Authenticate every planned receipt before applying per-item cache bounds.

    Uniform cached/minimum fleet candidates reserve all items together. A greedy
    first controller must not consume another item's mandatory minimum, and the
    native prepared owner retains the selected bound across geometry rebuilds.
    """
    from . import _native

    request = next((item for item in plan.requests if item.name == "ks"), None)
    if request is None or request.identity.backend != "cuda":
        return
    selected = dict(plan.selections)[request.name]
    candidate = next(item for item in request.candidates if item.name == selected)
    decisions = dict(candidate.decisions)
    mode = decisions.get("cuda_stationary_cache")
    if mode is None:
        return
    query = getattr(library, "generativeqc_ks_batch_stationary_cuda_capacity_v1", None)
    limit = getattr(library, "generativeqc_ks_batch_stationary_cuda_limit_v1", None)
    if query is None or limit is None:
        raise NotImplementedError(
            "native CUDA KS stationary prepared admission is unavailable"
        )
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    limit.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
    limit.restype = ctypes.c_int
    inventories = json.loads(decisions["item_device_inventory"])
    for index, inventory in enumerate(inventories):
        output = (ctypes.c_uint64 * len(_CUDA_STATIONARY_INVENTORY_KEYS))()
        _native.check(library, query(batch, index, output, len(output)))
        expected = tuple(
            inventory["stationary"][key] for key in _CUDA_STATIONARY_INVENTORY_KEYS
        )
        if tuple(output) != expected:
            raise ValueError(
                "prepared CUDA KS stationary capacity differs from the plan"
            )
    for index, inventory in enumerate(inventories):
        _native.check(library, limit(batch, index, inventory["stationary"][mode]))


def ks_resource_request(
    systems: typing.Any,
    *,
    method: typing.Any = "pbe-rks",
    basis: typing.Any = "sto-3g",
    backend: typing.Any = "cpu",
    precision: typing.Any = "fp64",
    basis_representation: typing.Any = None,
    charges: typing.Any = None,
    multiplicities: typing.Any = None,
    diis_history: typing.Any = 8,
    max_iterations: typing.Any = 100,
    energy_tolerance: typing.Any = 1e-10,
    density_tolerance: typing.Any = 1e-8,
    screening_tolerance: typing.Any = 1e-12,
    ks_options: typing.Any = None,
    device_id: typing.Any = 0,
    library: typing.Any = None,
    include_forces: typing.Any = False,
    cpu_packed_eri_maximum_bytes: int = 0,
    name: typing.Any = "ks",
    first_phase: typing.Any = 0,
    last_phase: typing.Any = 0,
    _fitted: tuple[typing.Any, int, float] | None = None,
    _application_precision_budget: typing.Any = None,
    _cuda_energy_only: bool = False,
    _allow_native_dense_forces: bool = True,
) -> typing.Any:
    """Resolve one complete KS request for the shared global planner.

    A positive ``cpu_packed_eri_maximum_bytes`` gates the exact retained value
    arrays for the whole fleet, independently of the composed host budget.
    This opt-in requires native named full-range CPU execution. Forces need
    the separate native force inventory and prepared derivative capability.
    """
    if type(include_forces) is not bool:
        raise TypeError("include_forces must be boolean")
    if backend not in ("cpu", "cuda"):
        raise NotImplementedError("KS planning requires a native CPU or CUDA backend")
    packed_cap = cpu_packed_eri_maximum_bytes
    if type(packed_cap) is not int or not 0 <= packed_cap < 2**64:
        raise ValueError("CPU packed ERI capacity must be a nonnegative uint64 integer")
    if packed_cap and (
        backend != "cpu" or str(precision).lower() != "fp64" or _fitted is not None
    ):
        raise NotImplementedError("packed ERI plans require direct FP64 CPU KS")
    packed_forces = bool(packed_cap and include_forces)
    if _application_precision_budget is not None:
        from .force_aware_numerics import TargetErrorBudget

        if not isinstance(_application_precision_budget, TargetErrorBudget):
            raise TypeError(
                "application precision requires an immutable TargetErrorBudget"
            )
        if backend != "cuda":
            raise NotImplementedError("application precision requires CUDA")
        if getattr(ks_options, "_precision_budget", None) is not None:
            raise ValueError(
                "application precision must have one method/KS transport owner"
            )
    try:
        # Calculator has already validated this frozen snapshot against the
        # exact selector. Reuse it for native metadata; changed options, aliases
        # and separately selected profiles take ordinary resolution. All shape,
        # precision, native-provider and capacity admission below still runs.
        model = (
            ks_options
            if (backend == "cuda" or _fitted is not None)
            and type(ks_options) is KsOptions
            and ks_options._resolved_selector is not None
            and ks_options._resolved_selector == method
            else resolve_ks_options(method, ks_options)
        )
    except (TypeError, ValueError, NotImplementedError) as error:
        raise NotImplementedError(
            "KS planning requires a compiler-resolved native KS execution plan"
        ) from error
    if model.execution_plan.post_scf:
        raise NotImplementedError(
            "KS planning supports electronic contributions only; "
            "post-SCF corrections require a separate resource owner"
        )
    from .ks import _automatic_semilocal_name

    cpu_fitted = backend == "cpu" and _fitted is not None
    cpu_fitted_forces = cpu_fitted and include_forces
    automatic_name = None if cpu_fitted else _automatic_semilocal_name(model.method_ir)
    automatic = automatic_name is not None
    automatic_cuda = automatic and backend == "cuda"
    rvv10 = (
        backend == "cuda"
        and model.has_nonlocal_correlation
        and is_canonical_r2scan_rvv10(model.method_ir)
    )
    wb97mv = (
        backend == "cuda"
        and model.has_range_exchange
        and composite_stationary_second_order_eligible(model.method_ir)
    )
    range_pbe = backend == "cuda" and cuda_pbe_range_stationary_second_order_eligible(
        model.method_ir
    )
    cuda_nonlocal = rvv10 or wb97mv
    from ._native_hessian import (
        canonical_cuda_split_hybrid_domain,
        has_cuda_r2scan_stationary_domain,
    )

    split_hybrid_energy = (
        backend == "cuda"
        and not include_forces
        and library is not None
        and precision == "fp64"
        and model.xc_schedule == "device_fused"
        and model.grid.partition_iterations == 3
        and _application_precision_budget is None
        and (_fitted is None or _fitted[0] is not None)
        and canonical_cuda_split_hybrid_domain(library, model.method_ir) is not None
    )
    if model.has_separate_nonlocal_grid and not cuda_nonlocal:
        raise NotImplementedError(
            "KS resource-ledger planning does not yet support independent XC/VV10 grids"
        )
    requires_tau = "tau" in model.functional.ingredients
    if packed_cap:
        from . import _native
        from ._named_ks import select_named_program

        if requires_tau or model.has_range_exchange or model.has_nonlocal_correlation:
            raise NotImplementedError(
                "packed ERI plans require local LDA/GGA and full-range J/K"
            )
        if library is None:
            library = _native.load_library(device="cpu")
        symbol = (
            _CPU_PACKED_FORCE_FLEET_SYMBOL
            if packed_forces
            else _CPU_PACKED_FLEET_SYMBOL
        )
        if not hasattr(library, symbol) or not select_named_program(
            library, model.method_ir, _native.BACKEND_CPU_REFERENCE
        ):
            raise NotImplementedError(
                "packed ERI plans require a linked native KS program and inventory"
            )
    if requires_tau and not automatic and backend == "cpu":
        raise NotImplementedError(
            "KS planning does not yet account for tau-dependent native XC workspace"
        )
    precision = str(precision).lower()
    if precision not in ("fp64", "auto"):
        raise ValueError("KS precision must be 'fp64' or 'auto'")
    if precision == "auto" and backend != "cuda":
        raise NotImplementedError("KS automatic precision currently requires CUDA")
    if (
        backend == "cuda"
        and model.has_range_exchange
        and not wb97mv
        and (
            not range_pbe
            or include_forces
            or precision != "fp64"
            or model.xc_schedule != "device_fused"
            or model.grid.partition_iterations != 3
            or _application_precision_budget is not None
        )
    ):
        raise NotImplementedError(
            "CUDA range-only KS capacity requires strict FP64 PBE with fused "
            "order-3 grids and energy-only SCF preparation"
        )
    if automatic_cuda and (precision != "fp64" or model.xc_schedule != "device_fused"):
        raise NotImplementedError(
            "automatic CUDA KS requires strict-FP64 device-fused execution"
        )
    if backend == "cuda" and model.has_nonlocal_correlation and not cuda_nonlocal:
        raise NotImplementedError(
            "CUDA KS resource planning does not yet cover self-consistent nonlocal correlation"
        )
    fitted_exchange = None
    cpu_df_name = None
    if cpu_fitted:
        from ._named_ks import select_cpu_df_energy_program

        if precision == "fp64":
            cpu_df_name = select_cpu_df_energy_program(library, model.method_ir)
        if cpu_df_name is None:
            raise NotImplementedError(
                "CPU DF KS host resource plans require linked FP64 LDA/PBE energy or forces"
            )
    elif _fitted is not None:
        # Named corrections lower to an ingredient carrier (e.g. pbe-rks
        # for B3LYP-D4). Match the resolved electronic graph for energy.
        canonical_fitted = (
            method in _CUDA_FITTED_METHODS
            and model.method_ir.identity == resolve_ks_method(method)[0].identity
            if include_forces
            else any(
                model.method_ir.identity == resolve_ks_method(name)[0].identity
                for name in _CUDA_FITTED_ENERGY_METHODS
            )
        )
        if (
            backend != "cuda"
            or precision != "fp64"
            or model.xc_schedule != "device_fused"
            or not (
                cuda_nonlocal or canonical_fitted or split_hybrid_energy or range_pbe
            )
            or _application_precision_budget is not None
            or (
                cuda_nonlocal
                and model.method_ir.identity != resolve_ks_method(method)[0].identity
            )
        ):
            raise NotImplementedError(
                "CUDA fitted KS resource planning requires canonical strict-FP64 "
                "LDA, PBE, PBE0, B3LYP, r2SCAN, r2SCAN+rVV10, WB97M-V, "
                "or linked M06-2X/MN15 with a qualified property inventory"
            )
        fitted_exchange = bool(model.coefficients[2])
    if cuda_nonlocal and (
        precision != "fp64"
        or model.xc_schedule != "device_fused"
        or _application_precision_budget is not None
        or model.grid.partition_iterations != 3
        or (
            model.nonlocal_grid is not None
            and model.nonlocal_grid.partition_iterations != 3
        )
    ):
        raise NotImplementedError(
            "CUDA nonlocal KS inventory requires FP64 and fused order-3 grids"
        )
    if cuda_nonlocal and include_forces:
        raise NotImplementedError(
            "CUDA nonlocal KS resource planning does not cover forces"
        )
    incremental_controls = _incremental_direct_jk_controls()
    incremental_requested = any(incremental_controls.values())
    systems = tuple(tuple(Atom.from_value(a) for a in atoms) for atoms in systems)
    if not systems or any(not atoms for atoms in systems):
        raise ValueError("KS resource planning requires nonempty systems")
    charges = (0,) * len(systems) if charges is None else tuple(charges)
    multiplicities = (
        (1,) * len(systems) if multiplicities is None else tuple(multiplicities)
    )
    if len(charges) != len(systems) or len(multiplicities) != len(systems):
        raise ValueError("charges and multiplicities must match the KS batch")
    for label, value in (
        ("DIIS history", diis_history),
        ("maximum iterations", max_iterations),
        ("device", device_id),
    ):
        checked_bytes(value, label)
    # Match descriptor defaults, including explicitly supplied zero controls.
    diis_history, max_iterations = diis_history or 8, max_iterations or 100
    for value in (energy_tolerance, density_tolerance, screening_tolerance):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("KS numerical tolerances must be positive finite")
    automatic_cuda_force_model = False
    if automatic_cuda:
        if library is None:
            from . import _native

            library = _native.load_library(device="cpu")
        automatic_cuda_force_model = native_cuda_stationary_model_available(
            library,
            model,
            precision=precision,
            max_iterations=max_iterations,
            diis_history=diis_history,
            energy_tolerance=energy_tolerance,
            density_tolerance=density_tolerance,
            screening_tolerance=screening_tolerance,
        )
    selected = _snapshot_basis(basis, basis_representation)
    if _fitted is not None:
        auxiliary_basis, fitted_budget, metric_threshold = _fitted
        checked_bytes(fitted_budget, "DF native memory budget")
        if auxiliary_basis is not None and len(systems) != 1:
            raise NotImplementedError(
                "a fitted KS plan with an explicit auxiliary basis requires one system"
            )
    # Imported CPU tau uses the same four AO jets and dense potential matrices
    # as GGA; tau and its point differential are fixed-size stack values.
    requires_first_ao_derivatives = bool(model.ao_order)
    unrestricted = model.method_ir.spin == "polarized"
    items = []
    for atoms, charge, multiplicity in zip(
        systems, charges, multiplicities, strict=True
    ):
        electrons = electron_state(
            atoms,
            charge=charge,
            multiplicity=multiplicity,
            element_metadata=selected.by_element
            if isinstance(selected, BasisSet)
            else {},
        )
        if not unrestricted and (
            multiplicity != 1 or electrons.nalpha != electrons.nbeta
        ):
            raise ValueError("RKS requires a closed-shell singlet")
        orbital = _basis_record(
            atoms,
            selected,
            basis_representation,
            backend,
            charge,
            multiplicity,
            "orbital",
            derivative_order=0,
            library=library,
        )
        if (
            not electrons.nalpha
            or max(electrons.nalpha, electrons.nbeta) > orbital["nbf"]
        ):
            raise ValueError("KS spin populations do not fit the AO space")
        if packed_cap and (
            orbital["representation"] != "spherical" or orbital["maximum_angular"] > 3
        ):
            raise ValueError(
                "packed Coulomb ERIs require a real spherical s/p/d/f basis"
            )
        require_basis(
            selected,
            atoms,
            backend=backend,
            operator="ao",
            derivative_order=int(requires_first_ao_derivatives),
            representation=orbital["representation"],
            role="orbital",
        )
        items.append(
            {
                "atoms": len(atoms),
                "orbital": orbital,
                "electrons": asdict(electrons),
                "spins": 2 if unrestricted else 1,
                "grid_points": model.grid.point_count(a.atomic_number for a in atoms),
            }
        )
        if _fitted is not None:
            if range_pbe and orbital.get("ecp_terms") and auxiliary_basis is None:
                raise NotImplementedError(
                    "CUDA range-PBE ECP density fitting requires an explicit "
                    "ECP-free auxiliary basis"
                )
            auxiliary = (
                orbital
                if auxiliary_basis is None
                else _basis_record(
                    atoms,
                    auxiliary_basis,
                    basis_representation,
                    backend,
                    charge,
                    multiplicity,
                    "auxiliary",
                    derivative_order=0,
                    library=library,
                    operator="df_three_center",
                    consumer=None
                    if cpu_fitted
                    else "wb97mv_df_values"
                    if wb97mv
                    else "scf_df",
                )
            )
            # SCF's orbital ECP value export retires before the fitted source
            # is built, and is already covered by the shared ECP setup leaf.
            # A separate auxiliary basis is only a Coulomb fitting space.
            if auxiliary_basis is not None and auxiliary.get("ecp_terms"):
                raise NotImplementedError(
                    "a fitted KS auxiliary basis must not carry ECPs"
                )
            if cpu_fitted and orbital.get("ecp_terms"):
                raise NotImplementedError("CPU DF KS host resource plans exclude ECPs")
            if include_forces and orbital.get("ecp_terms"):
                raise NotImplementedError(
                    "fitted KS ECP resource planning does not cover forces"
                )
            # Hessians prepare only fitted values; the stationary package
            # checks second-order closure independently of this SCF envelope.
            orbital_limit, auxiliary_limit = (
                (3, 3) if cpu_fitted else (1, 1) if include_forces else (3, 4)
            )
            if (
                orbital["maximum_angular"] > orbital_limit
                or auxiliary["maximum_angular"] > auxiliary_limit
            ):
                raise NotImplementedError(
                    "fitted KS resource planning exceeds the orbital/auxiliary angular domain"
                )
            items[-1]["auxiliary"] = auxiliary
        if (
            rvv10
            and (orbital.get("ecp_terms") or any(orbital.get("ecp_core_electrons", ())))
            and (
                library is None
                or not has_cuda_r2scan_stationary_domain(library)
                or orbital["maximum_angular"] > 3
            )
        ):
            raise NotImplementedError(
                "r2SCAN+rVV10 ECP planning requires linked variational CUDA "
                "and orbital shells through f"
            )
        if split_hybrid_energy and (
            orbital.get("ecp_terms") or any(orbital.get("ecp_core_electrons", ()))
        ):
            spaces = [(orbital, 3)]
            if _fitted is not None:
                spaces.append((items[-1]["auxiliary"], 4))
            if any(
                space["maximum_angular"]
                > (limit if space["representation"] == "spherical" else 1)
                for space, limit in spaces
            ):
                raise NotImplementedError(
                    "split-hybrid ECP planning requires orbital shells through f and "
                    "auxiliary shells through g (spherical for d and higher)"
                )
        if cuda_nonlocal:
            row = items[-1]
            nonlocal_grid = (
                model.nonlocal_grid if model.has_separate_nonlocal_grid else model.grid
            )
            points = nonlocal_grid.point_count(a.atomic_number for a in atoms)
            row["nonlocal_grid_points"] = points
            if model.has_separate_nonlocal_grid:
                # The second grid deep-copies the system; its construction
                # rules, packing copies and point buffers may overlap that copy.
                atom_count = row["atoms"]
                grid_rules = (
                    2 * nonlocal_grid.radial_points + 3 * nonlocal_grid.angular_polar
                )
                row["nonlocal_grid_host_bytes"] = checked_bytes(
                    116 * points
                    + 8 * atom_count**2
                    + 104 * atom_count
                    + 16 * grid_rules
                    + 8 * (2 * (512 + 256) + 1)
                    + 512 * (atom_count + orbital["shells"] + orbital["primitives"])
                    + 64 * orbital.get("ecp_terms", 0)
                )
    # Preserve conservative s/p/d ECP force capacity, including all-electron
    # fragments of an ECP catalog. Only active shells determine this capacity:
    # an unused higher-angular element must not erase the force reservation.
    # This is not the narrower compatibility consumer's admission predicate.
    cpu_forces = (
        backend == "cpu"
        and not cpu_fitted
        and (
            include_forces
            or (
                not packed_cap
                and isinstance(selected, BasisSet)
                and any(element.ecp_core_electrons for element in selected.elements)
                and all(item["orbital"]["maximum_angular"] <= 2 for item in items)
            )
        )
    )
    cuda_force_inventory = (
        backend == "cuda"
        and not cuda_nonlocal
        and not range_pbe
        and (
            include_forces
            or (not automatic_cuda and _fitted is None and not _cuda_energy_only)
        )
    )
    controls = {
        "diis_history": diis_history,
        "max_iterations": max_iterations,
        "energy_tolerance": energy_tolerance,
        "density_tolerance": density_tolerance,
        "screening_tolerance": screening_tolerance,
        "ks_options": model.to_payload(),
        "outputs": "energy+forces" if cuda_force_inventory or cpu_forces else "energy",
        "inventory_version": 1,
        "schedule": "ordinary-stream-round-robin"
        if backend == "cuda"
        else "serialized-native",
    }
    has_ecp = any(item["orbital"].get("ecp_terms") for item in items)
    # Serialized generated-force ownership can retain the largest member's
    # arena across the next SCF replay. Charge exactly the execution policy's
    # maximum caps, including a mixed fleet's all-electron and ECP members.
    # A bounded automatic package may omit its force artifacts. Charge their
    # retained owners only when native model discovery finds a complete graph.
    force_caps = tuple(
        (0, 0)
        if not cuda_force_inventory
        or (automatic_cuda and not automatic_cuda_force_model)
        else ordinary_cuda_force_limits(
            item["atoms"],
            density_fitted=_fitted is not None,
            hamiltonian="scalar-semilocal-ecp"
            if item["orbital"].get("ecp_terms")
            else "all-electron",
        )
        for item in items
    )
    force_device_cap = max(cap[0] for cap in force_caps)
    force_host_cap = max(cap[1] for cap in force_caps)
    if has_ecp:
        controls["ecp_host_inventory"] = _ecp_inventory_owner(library)
    if _application_precision_budget is not None:
        controls["application_precision_budget"] = asdict(_application_precision_budget)
    if backend == "cpu":
        controls["ao_grid_cache_cap"] = _CPU_AO_GRID_CACHE_CAP
        cpu_inventory = (
            "portable-metadata-v1"
            if library is None
            else "native-cpu-v1"
            if getattr(library, "generativeqc_resource_ks_cpu_fleet_v1", None)
            is not None
            else "unavailable"
        )
        controls.update(host_inventory=cpu_inventory, fleet_inventory=cpu_inventory)
        if cpu_df_name is not None:
            inventory_name = (
                "native-cpu-df-force-v1" if cpu_fitted_forces else "native-cpu-df-v1"
            )
            controls.update(
                host_inventory=inventory_name,
                fleet_inventory=inventory_name,
                cpu_df_named_program=cpu_df_name.decode("ascii"),
            )
        if packed_cap:
            inventory_name = (
                "native-cpu-packed-force-v1"
                if packed_forces
                else "native-cpu-packed-v1"
            )
            controls.update(
                cpu_packed_eri_maximum_bytes=packed_cap,
                host_inventory=inventory_name,
                fleet_inventory=inventory_name,
            )
    if backend == "cuda":
        controls.update(
            device_id=device_id,
            convergence_policy=os.environ.get("GENERATIVEQC_CUDA_KS_CONVERGENCE"),
            generated_force_device_cap=force_device_cap,
            generated_force_host_cap=force_host_cap,
            one_electron_mapping=os.environ.get(
                "GENERATIVEQC_ONE_ELECTRON_VALUE_MAPPING"
            ),
            dddd_force_class_major=os.environ.get(
                "GENERATIVEQC_EXPERIMENT_DDDD_FORCE_CLASS_MAJOR"
            ),
        )
    if incremental_requested:
        controls["incremental_direct_jk"] = incremental_controls
    if _fitted is not None:
        controls["fitted_fock"] = {
            "budget_bytes": fitted_budget,
            "metric_threshold": metric_threshold,
            "value_storage": os.environ.get("GENERATIVEQC_DF_VALUE_STORAGE", "auto"),
        }
    # AUTO runs two separately bounded nonlinear stages. Reserve the native
    # owner and exported history for both stages plus strict closure corrections.
    history_iterations = max_iterations
    if precision == "auto":
        history_iterations = checked_bytes(2 * max_iterations + 4, "KS mixed history")
        controls["history_iterations"] = history_iterations
    elif automatic_cuda:
        # Automatic RKS/UKS share a bounded physical F[D] closure after DIIS.
        history_iterations = checked_bytes(max_iterations + 4, "KS automatic history")
        controls["history_iterations"] = history_iterations
    identity = ResourceIdentity(
        method,
        "native-ks-direct-v1" if _fitted is None else "native-ks-fitted-v1",
        backend,
        precision,
        json.dumps({"items": items}),
        ("energy", "forces")
        if cuda_force_inventory or cpu_forces or cpu_fitted_forces
        else ("energy",),
        json.dumps(controls, sort_keys=True),
    )
    exclusions = (
        "caller-owned inputs and previously retained explicit output/snapshot objects",
        "Python objects, allocator bookkeeping, arenas and page rounding",
    )
    if backend == "cuda":
        exclusions += (
            "CUDA driver/context/modules, compiler-managed stacks and pool/page retention",
        )
    if incremental_requested:
        # Inventory v1 has no incremental-mode input. Its ordinary device
        # layout and strict host-history bound omit the anchor/delta buffers
        # and final-closure records. Fail before querying/loading a library;
        # never admit that experimental owner under an ordinary resource plan.
        return ResourceRequest(
            name,
            identity,
            (),
            exclusions,
            unsupported_reason=(
                "KS resource planning does not support experimental incremental "
                "Direct-J/K storage and closure history; unset "
                "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK and "
                "GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK or set both to 0/off"
            ),
        )
    if backend == "cuda" and library is None:
        from . import _native

        try:
            library = _native.load_library(device="cpu")
        except (NotImplementedError, RuntimeError, OSError) as error:
            return ResourceRequest(
                name, identity, (), exclusions, unsupported_reason=str(error)
            )
        if has_ecp:
            controls["ecp_host_inventory"] = _ecp_inventory_owner(library)
    try:
        dense_forces = False
        native_inventory = None
        if (
            backend == "cpu"
            and cpu_forces
            and not packed_cap
            and _allow_native_dense_forces
            and hasattr(library, _CPU_FORCE_FLEET_SYMBOL)
            and not automatic
            and not requires_tau
            and not model.has_range_exchange
            and not model.has_nonlocal_correlation
            and all(item["orbital"]["maximum_angular"] <= 3 for item in items)
        ):
            from . import _native
            from ._named_ks import select_named_program

            if select_named_program(
                library, model.method_ir, _native.BACKEND_CPU_REFERENCE
            ):
                native_inventory = _native_host_fleet_inventory(
                    library,
                    items,
                    backend=backend,
                    dense_forces=True,
                    diis_history=diis_history,
                    max_iterations=history_iterations,
                    requires_first_ao_derivatives=requires_first_ao_derivatives,
                    model=model,
                )
                dense_forces = native_inventory is not None
        if dense_forces:
            controls.update(
                host_inventory="native-cpu-force-v1",
                fleet_inventory="native-cpu-force-v1",
            )
            identity = replace(identity, schedule=json.dumps(controls, sort_keys=True))
        else:
            native_inventory = _native_host_fleet_inventory(
                library,
                items,
                backend=backend,
                workspace_floor=CPU_FORCE_HOST_CAP
                if cpu_forces and not packed_forces
                else 0,
                retained_allowance=(SNAPSHOT_GRID_CACHE_BYTES + force_host_cap)
                if backend == "cuda"
                else 0,
                packed_cap=packed_cap,
                packed_forces=packed_forces,
                cpu_fitted=cpu_fitted,
                cpu_fitted_forces=cpu_fitted_forces,
                diis_history=diis_history,
                max_iterations=history_iterations,
                requires_first_ao_derivatives=requires_first_ao_derivatives,
                model=model,
            )
    except _PackedEriCapacityExceeded as error:
        return ResourceRequest(
            name, identity, (), exclusions, infeasible_reason=str(error)
        )
    except _MissingCpuKsInventory as error:
        return ResourceRequest(
            name, identity, (), exclusions, unsupported_reason=str(error)
        )
    if native_inventory is not None:
        host, totals, host_phase_peak = native_inventory
        if backend == "cuda":
            controls["cuda_host_fleet_composition"] = "native-declared-capacity"
    else:
        host = [
            _item_host_inventory(
                item,
                diis_history=diis_history,
                max_iterations=history_iterations,
                requires_first_ao_derivatives=requires_first_ao_derivatives,
                backend=backend,
                model=model,
                library=library,
            )
            for item in items
        ]
        totals = {
            key: checked_bytes(sum(inventory[key] for inventory in host))
            for key in _CPU_HOST_PERSISTENT_KEYS
        }
        host_phase_peak = max(
            max(inventory["setup_workspace"], inventory["scf_workspace"])
            for inventory in host
        )
        if cpu_forces:
            host_phase_peak = max(host_phase_peak, CPU_FORCE_HOST_CAP)
    estimates = []
    for key in _CPU_HOST_PERSISTENT_KEYS:
        estimates.append(
            ResourceEstimate(
                f"all KS host {key}",
                totals[key],
                "pageable",
                first_phase,
                last_phase,
                kind="persistent",
            )
        )
    # CPU force workspaces are transient. The prepared CUDA force owner is
    # retained across calls and overlaps the next setup/SCF replay; reserve it
    # separately rather than treating serialized execution as retired storage.
    # Keep the established diagnostic labels; kind records allocation lifetime.
    if packed_forces or dense_forces:
        exclusions += (
            "native integral recurrence temporaries and BLAS/runtime internals",
        )
    elif cpu_forces:
        exclusions += (
            "force JIT/compiler processes, loaded code, BLAS/runtime internal storage",
        )
    if backend == "cuda":
        # One immutable exact grid survives snapshot closure for public forces.
        # The cache replaces rather than accumulates entries across ragged items.
        estimates.append(
            ResourceEstimate(
                "serialized KS snapshot grid cache cap",
                SNAPSHOT_GRID_CACHE_BYTES,
                "pageable",
                first_phase,
                last_phase,
                kind="persistent",
            )
        )
        estimates.append(
            ResourceEstimate(
                "serialized generated KS force host staging cap",
                force_host_cap,
                "pageable",
                first_phase,
                last_phase,
                kind="persistent",
            )
        )
    estimates.append(
        ResourceEstimate(
            "serialized KS transient host phase peak",
            host_phase_peak,
            "pageable",
            first_phase,
            last_phase,
        )
    )
    device = []
    device_totals = None
    native_stationary = False
    try:
        if library is not None:
            options_version = getattr(library, "generativeqc_ks_options_version", None)
            if callable(options_version):
                options_version.argtypes, options_version.restype = [], ctypes.c_uint32
            if not callable(options_version) or options_version() != 1:
                raise NotImplementedError(
                    "native library does not support the current semantic KS execution-plan ABI"
                )
            version = getattr(
                library, "generativeqc_ks_resource_inventory_version_v1", None
            )
            if version is not None:
                version.argtypes, version.restype = [], ctypes.c_int
            if version is None or version() != 1:
                raise NotImplementedError(
                    "native library does not implement KS allocation inventory v1"
                )
        if backend == "cuda":
            if _application_precision_budget is not None:
                transport = getattr(
                    library, "generativeqc_application_precision_version_v2", None
                )
                if transport is not None:
                    transport.argtypes, transport.restype = [], ctypes.c_uint32
                if transport is None or transport() != 2:
                    raise NotImplementedError(
                        "native library lacks method precision transport"
                    )
            from .ks import cuda_semilocal_stationary_second_order_eligible

            direct_hybrid_energy = (
                _cuda_energy_only
                and _fitted is None
                and not include_forces
                and precision == "fp64"
                and model.xc_schedule == "device_fused"
                and _application_precision_budget is None
                and bool(model.coefficients[2])
                and (
                    split_hybrid_energy
                    or cuda_semilocal_stationary_second_order_eligible(model.method_ir)
                )
            )
            fock = None
            radial_fock = (
                wb97mv
                or range_pbe
                or (rvv10 and _fitted is None)
                or direct_hybrid_energy
            )
            if radial_fock:
                entries = [
                    _native_radial_fock_inventory(
                        library,
                        item,
                        memory_budget=fitted_budget if _fitted is not None else 0,
                        radial_model=1 if wb97mv else 2 if range_pbe else 0,
                    )
                    for item in items
                ]
                fock, modes = zip(*entries, strict=True)
                if len(set(modes)) != 1:
                    raise ValueError(
                        "inconsistent LR exchange policy across one resource request"
                    )
                if wb97mv:
                    controls["wb97mv_range_exchange"] = modes[0]
                if range_pbe:
                    controls["pbe_range_exchange"] = modes[0]
            elif _fitted is not None:
                fock = [
                    _native_fitted_fock_inventory(
                        library,
                        item,
                        memory_budget=fitted_budget,
                        exchange=fitted_exchange,
                    )
                    for item in items
                ]
            if fock is not None:
                estimates.append(
                    ResourceEstimate(
                        "all KS radial Fock host envelopes"
                        if radial_fock
                        else "all KS fitted Fock host envelopes",
                        checked_bytes(sum(owner.host for owner in fock)),
                        "pageable",
                        first_phase,
                        last_phase,
                        kind="persistent",
                    )
                )
            if (
                _fitted is not None
                or range_pbe
                or (cuda_nonlocal and not include_forces)
                or (automatic_cuda and not automatic_cuda_force_model)
            ):
                stationary = [None for _ in items]
            else:
                descriptor = _cuda_stationary_descriptor(
                    model,
                    library=library,
                    precision=precision,
                    max_iterations=max_iterations,
                    diis_history=diis_history,
                    energy_tolerance=energy_tolerance,
                    density_tolerance=density_tolerance,
                    screening_tolerance=screening_tolerance,
                    application_precision_budget=_application_precision_budget,
                )
                stationary = [
                    _native_cuda_stationary_inventory(library, item, descriptor)
                    for item in items
                ]
            native_stationary = all(item is not None for item in stationary)
            if automatic_cuda and include_forces and not native_stationary:
                raise NotImplementedError(
                    "automatic CUDA KS forces require a packaged native stationary "
                    "profile and an admitted all-electron s/p FP64 context"
                )
            if (
                model.has_nondefault_composition
                and not native_stationary
                and not automatic_cuda
                and fock is None
            ):
                raise NotImplementedError(
                    "CUDA KS planning does not claim this scaled/global-hybrid execution"
                )
            if has_ecp:
                controls["ecp_device_inventory"] = _ecp_inventory_owner(library)
            identity = replace(
                identity,
                schedule=json.dumps(
                    {**controls, **_cuda_library_identity(library)}, sort_keys=True
                ),
            )
            device_options = {
                "diis_history": diis_history,
                "requires_first_ao_derivatives": requires_first_ao_derivatives,
                "functional_code": _native_semilocal_code(model.method_ir),
                "requires_tau": requires_tau,
                "tile": model.tile_points or 256,
                "rebuild_exchange": (
                    bool(model.coefficients[2])
                    if range_pbe
                    else True
                    if direct_hybrid_energy
                    else fitted_exchange
                ),
                "rebuild_range": range_pbe,
                "nonlocal_limit": model.nonlocal_memory_budget_bytes
                if cuda_nonlocal
                else None,
                "wb97mv": wb97mv,
            }
            native_device = _native_cuda_fleet_inventory(
                library,
                items,
                stationary,
                device_fused_xc=model.xc_schedule != "host_unfused",
                fock=fock,
                **device_options,
            )
            if native_device is None:
                device = [
                    _cuda_item_inventory(library, item, **device_options)
                    for item in items
                ]
                if model.xc_schedule == "host_unfused":
                    for item in device:
                        item["xc"] = 0
            else:
                device, device_totals = native_device
            if cuda_nonlocal:
                estimates.append(
                    ResourceEstimate(
                        "all KS nonlocal host envelopes",
                        checked_bytes(sum(item["nonlocal_host"] for item in device)),
                        "pageable",
                        first_phase,
                        last_phase,
                        kind="persistent",
                    )
                )
            for inventory, consumer in zip(device, stationary, strict=True):
                inventory["stationary"] = consumer
            for key in ("state", "xc", "coulomb"):
                estimates.append(
                    ResourceEstimate(
                        f"all KS device {key}",
                        device_totals[key]
                        if device_totals is not None
                        else checked_bytes(sum(x[key] for x in device)),
                        f"device:{device_id}",
                        first_phase,
                        last_phase,
                        kind="persistent",
                    )
                )
            # Every eligible item retains its own opaque cuBLAS handle. Admit
            # this separately from numeric arenas so the explicit allocation
            # ledger cannot spend the provider reservation on other buffers.
            estimates.append(
                ResourceEstimate(
                    "all KS matrix-provider retention allowance",
                    device_totals["matrix_provider"]
                    if device_totals is not None
                    else checked_bytes(sum(x["matrix_provider"] for x in device)),
                    f"device:{device_id}",
                    first_phase,
                    last_phase,
                    kind="library",
                    accounting="runtime_allowance",
                )
            )
            # At most one owner is rebuilt at a time. Its retired allocation
            # need not coexist with its setup temporary. Quadrature setup
            # already includes the new owner's live Coulomb allocation.
            extra = (
                device_totals["setup_extra"]
                if device_totals is not None
                else max(
                    max(0, x["setup"] - sum(x[k] for k in ("state", "xc", "coulomb")))
                    for x in device
                )
            )
            # The prepared force arena remains live during a later SCF replay.
            # The native transient peak also includes ECP derivative scratch
            # alongside live SCF owners. Add only its excess here; the force
            # arena itself has a separate persistent reservation below.
            estimates.append(
                ResourceEstimate(
                    "serialized KS transient device phase peak",
                    extra,
                    f"device:{device_id}",
                    first_phase,
                    last_phase,
                )
            )
            if not native_stationary and cuda_force_inventory and not automatic_cuda:
                estimates.append(
                    ResourceEstimate(
                        "serialized generated KS force device staging cap",
                        force_device_cap,
                        f"device:{device_id}",
                        first_phase,
                        last_phase,
                        kind="persistent",
                    )
                )
    except (NotImplementedError, RuntimeError, OSError) as error:
        return ResourceRequest(
            name, identity, (), exclusions, unsupported_reason=str(error)
        )
    cuda_composition_decisions = ()
    if backend == "cuda":
        cuda_composition_decisions = (
            (
                "cuda_fleet_composition",
                "native-declared-capacity"
                if device_totals is not None
                else "legacy-leaf-compatibility",
            ),
        )
    candidate = ResourceCandidate(
        f"{backend}-ks-resident",
        "resident",
        tuple(estimates),
        decisions=(
            ("batch_execution", controls["schedule"]),
            ("cpu_worker_limit", "1"),
            ("item_host_inventory", json.dumps(host, sort_keys=True)),
            ("item_device_inventory", json.dumps(device, sort_keys=True)),
            *cuda_composition_decisions,
            (
                "xc_ao_selection",
                "dense under the public device ledger; no optional map allowance"
                if backend == "cuda"
                else "not applicable",
            ),
            (
                "xc_point_batching",
                "one-tile under the public device ledger; no optional panel allowance"
                if backend == "cuda"
                else "not applicable",
            ),
            (
                "preparation",
                "included in common observation and persistent device ledger",
            ),
            (
                "transient_phase_accounting",
                "transient phase maxima plus retained CUDA force owners across replays",
            ),
        ),
    )
    if not native_stationary:
        return ResourceRequest(name, identity, (candidate,), exclusions)
    candidates = []
    for mode, cost in (("cached", 0), ("minimum", 1)):
        consumer = ResourceEstimate(
            "all native KS stationary CUDA consumers",
            device_totals[mode]
            if device_totals is not None
            else checked_bytes(sum(item[mode] for item in stationary)),
            f"device:{device_id}",
            first_phase,
            last_phase,
            kind="persistent",
        )
        candidates.append(
            replace(
                candidate,
                name=f"cuda-ks-resident-stationary-{mode}",
                estimates=(*candidate.estimates, consumer),
                relative_cost=cost,
                decisions=(*candidate.decisions, ("cuda_stationary_cache", mode)),
            )
        )
    return ResourceRequest(name, identity, tuple(candidates), exclusions)


def estimate_ks_resources(
    systems: typing.Any, *, budget: typing.Any = None, **options: typing.Any
) -> typing.Any:
    """Plan native KS energy capacity without creating scientific arrays."""
    return plan_resources(
        (ks_resource_request(systems, **options),),
        ResourceBudget() if budget is None else budget,
        library=options.get("library"),
    )
