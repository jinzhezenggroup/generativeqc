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

from ._cpu_force_resources import CPU_FORCE_HOST_CAP, qualified_basis
from ._cuda_force_resources import ordinary_cuda_force_limits
from ._snapshot_grid_cache import SNAPSHOT_GRID_CACHE_BYTES
from .basis import BasisSet
from .basis_capabilities import require_basis
from .calculator import Atom, _snapshot_basis
from .elements import electron_state
from .ks import KsOptions, _native_semilocal_code, resolve_ks_options
from .resources_hf import (
    _basis_record,
    _cuda_library_identity,
    _ecp_inventory_owner,
    _ecp_workspace,
)
from .resources_planning import plan_resources

_CPU_AO_GRID_CACHE_CAP = 64 << 20
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


def _cpu_host_shape(
    item: typing.Any,
    *,
    library: typing.Any,
    diis_history: int,
    max_iterations: int,
    requires_first_ao_derivatives: bool,
    model: typing.Any,
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
        model.tile_points or 256,
        diis_history,
        max_iterations,
        int(requires_first_ao_derivatives),
        int(model.has_nonlocal_correlation),
        _ecp_workspace(item, library=library),
    )
    return tuple(checked_bytes(value, "KS host shape") for value in dimensions)


def _cpu_host_result(values: typing.Sequence[int], model: typing.Any) -> dict[str, int]:
    """Decode capacities without waiving the explicit nonlocal sub-budget."""
    inventory = {
        key: checked_bytes(value, f"KS {key}")
        for key, value in zip(_CPU_HOST_INVENTORY_KEYS, values, strict=True)
    }
    if (
        model.has_nonlocal_correlation
        and inventory["nonlocal_provider"] > model.nonlocal_memory_budget_bytes
    ):
        raise ValueError(
            "KS nonlocal provider workspace exceeds nonlocal_memory_budget_bytes"
        )
    return inventory


def _native_cpu_host_inventory(
    library: typing.Any,
    item: typing.Any,
    **options: typing.Any,
) -> dict[str, int] | None:
    """Use discovered authority, retaining absent/old-library metadata sizing.

    A rejected discovered query is never retried. This adapter neither loads a
    provider nor performs numerical work during metadata-only planning.
    """
    query = getattr(library, "generativeqc_resource_ks_cpu_host_v1", None)
    if query is None:
        return None
    dimensions = _cpu_host_shape(item, library=library, **options)
    shape = (ctypes.c_uint64 * len(dimensions))(*dimensions)
    values = (ctypes.c_uint64 * len(_CPU_HOST_INVENTORY_KEYS))()
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    if query(shape, len(shape), values, len(values)):
        raise ValueError("native CPU KS host inventory rejected shape or capacity")
    return _cpu_host_result(values, options["model"])


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
    **options: typing.Any,
) -> tuple[list[dict[str, int]], dict[str, int], int] | None:
    """Query all retained owners and serialized workspace before publication.

    Consumer allowances describe lifetimes, not provider admission. CUDA cache
    and force owners remain retained across replay; CPU force scratch remains
    a transient floor. Only missing symbols permit the older per-item path.
    Transport consistency checks validate publication without retrying sizing.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("KS host inventory requires cpu or cuda backend")
    cuda = backend == "cuda"
    symbol = (
        "generativeqc_resource_ks_cuda_host_fleet_v1"
        if cuda
        else "generativeqc_resource_ks_cpu_fleet_v1"
    )
    query = getattr(library, symbol, None)
    if query is None:
        return None
    if not cuda and retained_allowance:
        raise ValueError("the CPU fleet ABI has no retained consumer allowance")
    dimensions = tuple(
        value
        for item in items
        for value in (
            *_cpu_host_shape(item, library=library, **options),
            *((int(options["model"].xc_schedule == "host_unfused"),) if cuda else ()),
        )
    )
    shapes = (ctypes.c_uint64 * len(dimensions))(*dimensions)
    values = (ctypes.c_uint64 * (15 + 16 * len(items)))()
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        *([ctypes.c_uint64] * (2 if cuda else 1)),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    floor = checked_bytes(workspace_floor, "KS workspace allowance")
    allowance = checked_bytes(retained_allowance, "KS retained consumer allowance")
    if query(
        shapes,
        len(shapes),
        *((allowance, floor) if cuda else (floor,)),
        values,
        len(values),
    ):
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
        _cpu_host_result(values[15 + 16 * index : 31 + 16 * index], options["model"])
        for index in range(len(items))
    ]
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
    if backend == "cpu":
        inventory = _native_cpu_host_inventory(
            library,
            item,
            diis_history=diis_history,
            max_iterations=max_iterations,
            requires_first_ao_derivatives=requires_first_ao_derivatives,
            model=model,
        )
        if inventory is not None:
            return inventory
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
        if nonlocal_provider > model.nonlocal_memory_budget_bytes:
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
) -> typing.Any:
    """Query the existing allocator-owned leaves without composing lifetimes."""
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
    orbital = item["orbital"]
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
    if any(value > 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1 for value in args):
        raise ValueError("KS resource shape exceeds the host size_t ABI")
    query.argtypes = [ctypes.c_size_t] * 9 + [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    output = (ctypes.c_uint64 * 3)()
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
        "coulomb": checked_bytes(int(output[2])),
        "quadrature_setup": checked_bytes(quadrature.value),
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


def _native_cuda_fleet_inventory(
    library: typing.Any,
    items: typing.Sequence[typing.Any],
    stationary: typing.Sequence[typing.Any],
    *,
    device_fused_xc: bool,
    **options: typing.Any,
) -> tuple[list[dict[str, int]], dict[str, int]] | None:
    """Compose native leaves without Python setup/peak/candidate arithmetic.

    Only absence permits the legacy per-item route. Rejection or an incoherent
    wire publication is authoritative and never retries portable sizing. The
    query itself is shape-only, including in CPU-only builds; leaf availability
    remains the selected CUDA library's separate contract.
    """
    query = getattr(library, "generativeqc_resource_ks_cuda_fleet_v1", None)
    if query is None:
        return None
    if len(items) != len(stationary):
        raise ValueError("CUDA KS fleet and stationary inventory lengths differ")
    native_stationary = all(consumer is not None for consumer in stationary)
    leaves = [_cuda_leaf_inventory(library, item, **options) for item in items]
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
        for key in ("state", "xc", "coulomb", "quadrature_setup", "matrix_provider"):
            expected = 0 if key == "xc" and not device_fused_xc else leaf[key]
            if inventory[key] != expected:
                raise ValueError("native CUDA KS fleet changed allocator-owned leaves")
        if inventory["setup"] < max(
            leaf["ecp_workspace"], leaf["coulomb"] + leaf["quadrature_setup"]
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
    name: typing.Any = "ks",
    first_phase: typing.Any = 0,
    last_phase: typing.Any = 0,
    _application_precision_budget: typing.Any = None,
) -> typing.Any:
    """Resolve one complete KS request for the shared global planner."""
    if type(include_forces) is not bool:
        raise TypeError("include_forces must be boolean")
    if backend not in ("cpu", "cuda"):
        raise NotImplementedError("KS planning requires a native CPU or CUDA backend")
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
        # exact selector. Reuse it for CUDA metadata; changed options, aliases
        # and separately selected profiles take ordinary resolution. All shape,
        # precision, native-provider and capacity admission below still runs.
        model = (
            ks_options
            if backend == "cuda"
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

    automatic_name = _automatic_semilocal_name(model.method_ir)
    automatic = automatic_name is not None
    automatic_cuda = automatic and backend == "cuda"
    if model.has_separate_nonlocal_grid:
        raise NotImplementedError(
            "KS resource-ledger planning does not yet support independent XC/VV10 grids"
        )
    requires_tau = "tau" in model.functional.ingredients
    if requires_tau and not automatic and backend == "cpu":
        raise NotImplementedError(
            "KS planning does not yet account for tau-dependent native XC workspace"
        )
    precision = str(precision).lower()
    if precision not in ("fp64", "auto"):
        raise ValueError("KS precision must be 'fp64' or 'auto'")
    if precision == "auto" and backend != "cuda":
        raise NotImplementedError("KS automatic precision currently requires CUDA")
    if automatic_cuda and (precision != "fp64" or model.xc_schedule != "device_fused"):
        raise NotImplementedError(
            "automatic CUDA KS requires strict-FP64 device-fused execution"
        )
    if backend == "cuda" and model.has_nonlocal_correlation:
        raise NotImplementedError(
            "self-consistent nonlocal correlation currently requires CPU"
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
    cpu_forces = backend == "cpu" and (qualified_basis(selected) or include_forces)
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
    controls = {
        "diis_history": diis_history,
        "max_iterations": max_iterations,
        "energy_tolerance": energy_tolerance,
        "density_tolerance": density_tolerance,
        "screening_tolerance": screening_tolerance,
        "ks_options": model.to_payload(),
        "outputs": "energy+forces"
        if (backend == "cuda" and (not automatic_cuda or include_forces)) or cpu_forces
        else "energy",
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
        if automatic_cuda and not automatic_cuda_force_model
        else ordinary_cuda_force_limits(
            item["atoms"],
            density_fitted=False,
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
        native_fleet = getattr(library, "generativeqc_resource_ks_cpu_fleet_v1", None)
        controls["host_inventory"] = (
            "native-cpu-v1"
            if native_fleet is not None
            or getattr(library, "generativeqc_resource_ks_cpu_host_v1", None)
            is not None
            else "portable-metadata-v1"
        )
        controls["fleet_inventory"] = (
            "native-cpu-v1" if native_fleet is not None else "portable-metadata-v1"
        )
    if backend == "cuda":
        controls.update(
            device_id=device_id,
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
        "native-ks-direct-v1",
        backend,
        precision,
        json.dumps({"items": items}),
        ("energy", "forces")
        if (backend == "cuda" and (not automatic_cuda or include_forces)) or cpu_forces
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
    native_inventory = _native_host_fleet_inventory(
        library,
        items,
        backend=backend,
        workspace_floor=CPU_FORCE_HOST_CAP if cpu_forces else 0,
        retained_allowance=(SNAPSHOT_GRID_CACHE_BYTES + force_host_cap)
        if backend == "cuda"
        else 0,
        diis_history=diis_history,
        max_iterations=history_iterations,
        requires_first_ao_derivatives=requires_first_ao_derivatives,
        model=model,
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
    if cpu_forces:
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
            if automatic_cuda and not automatic_cuda_force_model:
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
            }
            native_device = _native_cuda_fleet_inventory(
                library,
                items,
                stationary,
                device_fused_xc=model.xc_schedule != "host_unfused",
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
            if not native_stationary and not automatic_cuda:
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
