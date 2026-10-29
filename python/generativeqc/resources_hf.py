"""Compact HF topology resolution and conservative CPU allocation inventory.

Only shell/primitive metadata is built here. In particular, a dry run never
constructs overlap, ERI, DF, density or derivative arrays. CUDA HF needs its
native arena/provider query; absence of that query is reported explicitly.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import threading
import typing
from collections import OrderedDict
from dataclasses import asdict, replace
from pathlib import Path

from generativeqc_compiler.common.resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    checked_bytes,
)

from .basis import BasisSet
from .basis_capabilities import require_basis, resolved_basis_metadata
from .calculator import Atom, _snapshot_basis
from .elements import electron_state
from .resources_planning import plan_resources

# Newly tracked controls may be absent from earlier schema-1 checkpoints.
# Preserve that distinction in restart provenance; absence is not a claim
# that an older build implemented today's automatic execution policy.
_CUDA_SCHEDULE_EXTENSION_VARIABLES = (
    # Optional in historical checkpoints, but part of every new resource and
    # restart identity: changing the force schedule requires explicit admission.
    "GENERATIVEQC_BOUNDED_ANGULAR_FORCE",
    "GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS",
    "GENERATIVEQC_DIRECT_WEIGHTED_FORCE_RESPONSES",
    "GENERATIVEQC_DIRECT_FP32_BLOCK_SCREENING",
    "GENERATIVEQC_CUDA_KS_DEVICE_FINAL_VALIDATION",
    "GENERATIVEQC_DIRECT_COULOMB_REACHABLE",
    "GENERATIVEQC_DIRECT_HERMITE_CONVOLUTION",
    "GENERATIVEQC_CANONICAL_RSH_VALUES",
    "GENERATIVEQC_DF_VALUE_STORAGE",
    "GENERATIVEQC_DF_SHARED_SOURCE",
    "GENERATIVEQC_DF_REFERENCE_FINAL_VALIDATION",
    "GENERATIVEQC_DF_VALUE_RAW_MAPPING",
    "GENERATIVEQC_DF_VALUE_MATH",
    "GENERATIVEQC_DF_SEED_EXCHANGE",
    "GENERATIVEQC_DF_SEED_VERIFY",
    "GENERATIVEQC_DF_FINAL_EXCHANGE",
    "GENERATIVEQC_DF_SHELL_POLICY",
    "GENERATIVEQC_DF_FORCE_SCREEN_ABS",
    "GENERATIVEQC_DF_SHELL_SCREEN_ABS",
    "GENERATIVEQC_DF_RESIDENT_EXCHANGE",
    "GENERATIVEQC_DF_RAW_REUSE",
    "GENERATIVEQC_DF_RESPONSE_BATCHING",
    "GENERATIVEQC_DF_FINAL_PROJECTION",
    "GENERATIVEQC_DF_DIIS_DOTS",
    "GENERATIVEQC_DF_RESPONSE_STORAGE",
    "GENERATIVEQC_DF_RESPONSE_SPACE",
    "GENERATIVEQC_DF_OCCUPIED_METRIC",
    "GENERATIVEQC_DF_COULOMB_RESPONSE",
    "GENERATIVEQC_DF_HOST_MAX_BYTES",
)

# These runtime switches are read by CUDA HF/DF policy code. Preserve even
# switches dormant for a small shape: a prepared resource identity must not
# silently accept a different arithmetic, profiling or scheduling policy.
_CUDA_SCHEDULE_VARIABLES = (
    "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM",
    "GENERATIVEQC_SCF_INCREMENTAL_DIIS_GRAM_REDUCTION",
    "GENERATIVEQC_FINAL_FOCK_REBUILD",
    "GENERATIVEQC_MIXED_PRECISION_FOCK_THRESHOLD",
    "GENERATIVEQC_GRAPH_EIGENSOLVER_OVERRIDE",
    "GENERATIVEQC_XSYEV_PROBE_SKIP_DIAGNOSTIC",
    "GENERATIVEQC_BOUNDED_DIRECT_STREAMING",
    "GENERATIVEQC_BOUNDED_DIRECT_COUNT_DIAGNOSTIC",
    "GENERATIVEQC_BOUNDED_DIRECT_AOT_ONLY_DIAGNOSTIC",
    "GENERATIVEQC_BOUNDED_DIRECT_FOCK_ONLY_DIAGNOSTIC",
    "GENERATIVEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE",
    "GENERATIVEQC_DIRECT_TILE_VALIDATION",
    "GENERATIVEQC_FORCE_DENSITY_PRODUCT_SCREENING",
    "GENERATIVEQC_PPPS_RESIDENT_BRA",
    "GENERATIVEQC_PPPS_SIGNATURE_BUCKETING",
    "GENERATIVEQC_PSPS_SIGNATURE_BUCKETING",
    "GENERATIVEQC_PPSS_SIGNATURE_BUCKETING",
    "GENERATIVEQC_PPPS_BLOCK_THREADS",
    "GENERATIVEQC_ONE_ELECTRON_VALUE_MAPPING",
    "GENERATIVEQC_HF_COLD_SCREENING",
    "GENERATIVEQC_EXPERIMENT_DDDD_FORCE_CLASS_MAJOR",
    "GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_MAPPING",
    "GENERATIVEQC_PSSS_RESIDENT_BRA",
    "GENERATIVEQC_DF_VALUE_MAPPING",
    "GENERATIVEQC_DF_MD_CHARGE_SPLITS",
    "GENERATIVEQC_DF_DERIVATIVE_MAPPING",
    "GENERATIVEQC_DF_WEIGHTED_EXECUTION",
    "GENERATIVEQC_DF_SHELL_SCHEDULE",
    "GENERATIVEQC_DF_DERIVATIVE_PAIRS",
    "GENERATIVEQC_DF_METRIC_DERIVATIVE_PAIRS",
    "GENERATIVEQC_DF_PRIMITIVE_BUCKETS",
    "GENERATIVEQC_DF_PACKED_AO_BLOCK_ROWS",
    "GENERATIVEQC_DF_RESPONSE_ALGEBRA",
    "GENERATIVEQC_DF_RAW_STAGING",
    "GENERATIVEQC_DF_EXCHANGE",
    *_CUDA_SCHEDULE_EXTENSION_VARIABLES,
)


# A retained CDLL keeps its loader handle alive, so another wrapper for that
# handle denotes the same loaded image even if its original path was replaced.
# Bound ownership to 32 loaded libraries; ordinary wrappers retain their own
# digest after eviction. Do not use stat-keyed file caching: same-tick in-place
# edits can preserve every observed timestamp on supported filesystems.
_RESOURCE_LIBRARY_IDENTITIES: OrderedDict[int, tuple[ctypes.CDLL, str]] = OrderedDict()
_RESOURCE_LIBRARY_IDENTITY_LOCK = threading.Lock()


def _cuda_library_identity(library: typing.Any) -> typing.Any:
    """Reuse provenance of a retained loaded image without probing a GPU.

    New loader handles are hashed independently. Metadata-only adapters have no
    loader lifetime guarantee and retain the uncached file-hash behavior. Profile
    identity remains local to each wrapper, independent of shared binary bytes.
    """
    from .profiles import file_hash

    digest = getattr(library, "_generativeqc_resource_binary_sha256", None)
    path = getattr(library, "_name", None)
    if digest is None and path is not None:
        handle = library._handle if isinstance(library, ctypes.CDLL) else None
        if handle is None:
            digest = file_hash(Path(path))
        else:
            # Hashing releases the GIL. Serialize first publication and LRU
            # updates so concurrent calculator construction cannot lose a lease.
            with _RESOURCE_LIBRARY_IDENTITY_LOCK:
                retained = _RESOURCE_LIBRARY_IDENTITIES.get(handle)
                if retained is not None:
                    digest = retained[1]
                    _RESOURCE_LIBRARY_IDENTITIES.move_to_end(handle)
                else:
                    digest = file_hash(Path(path))
                    # Keep the loaded owner, not just its reusable integer address.
                    # No scientific handles, device probes or CUDA buffers are made.
                    _RESOURCE_LIBRARY_IDENTITIES[handle] = (library, digest)
                    if len(_RESOURCE_LIBRARY_IDENTITIES) > 32:
                        _RESOURCE_LIBRARY_IDENTITIES.popitem(last=False)
        library._generativeqc_resource_binary_sha256 = digest
    profile = getattr(library, "_generativeqc_profile_diagnostics", {})
    return {"native_binary_sha256": digest, "profile_identity": profile.get("identity")}


def _basis_record(
    atoms: typing.Any,
    basis: typing.Any,
    representation: typing.Any,
    backend: typing.Any,
    charge: typing.Any,
    multiplicity: typing.Any,
    role: typing.Any,
    *,
    derivative_order: typing.Any = 1,
    library: typing.Any = None,
) -> typing.Any:
    selected = _snapshot_basis(basis, representation)
    mode = (
        selected.representation
        if isinstance(selected, BasisSet)
        else representation or "cartesian"
    )
    require_basis(
        selected,
        atoms,
        backend=backend,
        operator="eri",
        derivative_order=derivative_order,
        representation=mode,
        role=role,
    )
    shells = selected.shells_for(atoms) if isinstance(selected, BasisSet) else selected
    metadata = resolved_basis_metadata(
        selected,
        shells,
        atoms,
        representation=mode,
        charge=charge,
        multiplicity=multiplicity,
        role=role,
    )
    from .resources_basis import _basis_shape

    angular = tuple(int(shell.angular_momentum) for shell in shells)
    primitives = tuple(len(shell.primitives) for shell in shells)
    record = {
        # AO counts depend on the representation and expanded contractions,
        # not the execution device. Reuse the already selected native owner
        # for CUDA too; this metadata query does not create a device context.
        **_basis_shape(angular, primitives, mode, library=library),
        "shell_angular": angular,
        "shell_primitive_counts": primitives,
        "representation": mode,
        "basis_hash": metadata["mathematical_identity"],
    }
    from .ecp import resolve_ecp

    cores, terms = resolve_ecp(selected, atoms)
    if terms:
        record["ecp_terms"] = len(terms)
        record["ecp_core_electrons"] = cores
    return record


def _ecp_inventory_owner(library: typing.Any) -> str:
    """Identify ECP sizing separately from the enclosing HF/KS inventory."""
    return (
        "native-ecp-v1"
        if getattr(library, "generativeqc_resource_ecp_workspace_v1", None) is not None
        else "portable-metadata-v1"
    )


def _ecp_workspace(
    item: typing.Any, *, cuda: typing.Any = False, library: typing.Any = None
) -> int:
    """Use the native two-grid owner when already loaded, without probing a GPU.

    Old/no-library dry runs retain the portable bound. A discovered query's
    rejection is authoritative, not permission to retry Python sizing. Public
    AOs select the CUDA radial tile; Cartesian AOs determine storage capacity.
    """
    orbital = item["orbital"]
    if not orbital.get("ecp_terms"):
        return 0
    query = getattr(library, "generativeqc_resource_ecp_workspace_v1", None)
    if query is None:
        return _portable_ecp_workspace(item, cuda=cuda)
    dimensions = (
        orbital["nbf"],
        orbital["cartesian_nbf"],
        item["atoms"],
        orbital["primitives"],
        orbital["ecp_terms"],
    )
    dimensions = tuple(checked_bytes(value, "ECP shape") for value in dimensions)
    if any(
        value > 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1 for value in dimensions
    ):
        raise ValueError("ECP resource shape exceeds the host size_t ABI")
    query.argtypes = [ctypes.c_size_t] * 5 + [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    query.restype = ctypes.c_int
    output = ctypes.c_uint64()
    if query(*dimensions, int(bool(cuda)), ctypes.byref(output)):
        raise ValueError("native ECP workspace inventory rejected shape or capacity")
    return checked_bytes(output.value, "ECP two-grid workspace")


def _portable_ecp_workspace(item: typing.Any, *, cuda: typing.Any = False) -> int:
    """Compatibility-only two-grid bound for old or absent native libraries."""
    from generativeqc_compiler.integral.ecp_policy import (
        REFINED_POLAR_POINTS,
        REFINED_RADIAL_POINTS,
    )
    from generativeqc_compiler.integral.ecp_schedule import cuda_radial_tile

    orbital = item["orbital"]
    if not orbital.get("ecp_terms"):
        return 0
    n, d = orbital["cartesian_nbf"], 3 * item["atoms"]
    points = 2 * REFINED_POLAR_POINTS * REFINED_POLAR_POINTS
    tile = cuda_radial_tile(orbital["nbf"], REFINED_POLAR_POINTS) if cuda else 1
    return checked_bytes(
        32 * n * n * (1 + d)
        + 32 * tile * n * (points + 16)
        + 256
        * (
            points
            + REFINED_RADIAL_POINTS
            + n
            + orbital["primitives"]
            + orbital["ecp_terms"]
        )
        + 4096,
        "ECP two-grid workspace",
    )


def _native_cpu_fleet_inventory(
    library: typing.Any, items: typing.Sequence[typing.Any], diis_history: int
) -> tuple[tuple[dict[str, int], ...], int, int] | None:
    """Marshal compact metadata to the native retained/workspace fleet owner.

    One query validates and sizes the whole fleet, including per-item phase
    diagnostics. A rejected discovered query is never retried with older phase
    queries or portable arithmetic. No provider is loaded by this adapter.
    """
    query = getattr(library, "generativeqc_resource_hf_cpu_fleet_v1", None)
    if query is None:
        return None
    dimensions = []
    for item in items:
        orbital, auxiliary = item["orbital"], item.get("auxiliary")
        dimensions.extend(
            (
                item["atoms"],
                orbital["nbf"],
                orbital["cartesian_nbf"],
                item["spins"],
                orbital["maximum_angular"],
                diis_history,
                auxiliary["nbf"] if auxiliary else 0,
                auxiliary["cartesian_nbf"] if auxiliary else 0,
                auxiliary["maximum_angular"] if auxiliary else 0,
                _ecp_workspace(item, library=library),
                orbital["shells"],
                orbital["primitives"],
                auxiliary["shells"] if auxiliary else 0,
                auxiliary["primitives"] if auxiliary else 0,
            )
        )
    shapes = (ctypes.c_uint64 * len(dimensions))(
        *(checked_bytes(value, "CPU HF fleet shape") for value in dimensions)
    )
    values = (ctypes.c_uint64 * (3 + 4 * len(items)))()
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    if query(shapes, len(shapes), values, len(values)):
        raise ValueError("native CPU HF fleet inventory rejected shape or capacity")
    retained, workspace, peak = (
        checked_bytes(values[index], "CPU HF fleet capacity") for index in range(3)
    )
    if peak != retained + workspace:
        raise ValueError("native CPU HF fleet inventory returned inconsistent peak")
    keys = ("preparation", "iteration", "finalization", "integral_state")
    inventories = tuple(
        {
            key: checked_bytes(values[3 + 4 * index + slot], f"CPU {key} capacity")
            for slot, key in enumerate(keys)
        }
        for index in range(len(items))
    )
    return inventories, retained, workspace


def _cpu_item_inventory(
    item: typing.Any, diis_history: typing.Any, *, library: typing.Any = None
) -> dict[str, int]:
    """Bound the existing resident CPU route by its preparation/iteration phases.

    The CPU DF implementation currently retains the ordinary four-center
    integral record while preparing RI tensors. That real allocation belongs
    here even though DF contractions do not use its ERI values. Forward-mode
    Jet arrays coexist with unpacked tensors during preparation; spherical
    conversion can retain Cartesian and public tensors simultaneously.

    Bounds include requested numeric storage and conservative LP64 C++ value
    metadata. Malloc bookkeeping, allocator arenas and language/runtime objects
    remain explicit exclusions, covered by user reserve/headroom when needed.

    A discovered native query owns phase sizing, shared with the native CLI.
    Rejections propagate; only no-library dry runs and older binaries use the
    portable bound below. Metadata planning never loads a library implicitly.
    """
    query = getattr(library, "generativeqc_resource_hf_cpu_host_v1", None)
    if query is not None:
        orbital, auxiliary = item["orbital"], item.get("auxiliary")
        dimensions = (
            item["atoms"],
            orbital["nbf"],
            orbital["cartesian_nbf"],
            item["spins"],
            orbital["maximum_angular"],
            diis_history,
            auxiliary["nbf"] if auxiliary else 0,
            auxiliary["cartesian_nbf"] if auxiliary else 0,
            auxiliary["maximum_angular"] if auxiliary else 0,
            _ecp_workspace(item, library=library),
        )
        shape = (ctypes.c_uint64 * len(dimensions))(
            *(checked_bytes(value, "CPU HF host shape") for value in dimensions)
        )
        keys = ("preparation", "iteration", "finalization", "integral_state")
        values = (ctypes.c_uint64 * len(keys))()
        query.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        query.restype = ctypes.c_int
        if query(shape, len(shape), values, len(values)):
            raise ValueError("native CPU HF host inventory rejected shape or capacity")
        return {
            key: checked_bytes(value, f"CPU {key} capacity")
            for key, value in zip(keys, values, strict=True)
        }
    orbital, auxiliary = item["orbital"], item.get("auxiliary")
    n, c, d = orbital["nbf"], orbital["cartesian_nbf"], 3 * item["atoms"]
    spin = item["spins"]
    n2, c2 = byte_product(n, n), byte_product(c, c)
    conventional = byte_product(8, d + 1, 2 * n2 + byte_product(n2, n2)) + 8 * d
    cartesian = byte_product(8, d + 1, 2 * c2 + byte_product(c2, c2)) + 8 * d
    # Jet: scalar, derivative vector and vector value metadata. The 64-byte
    # per-value allowance exceeds the current LP64 Jet/vector object sizes.
    jet_bytes = 8 * (d + 1) + 64
    jet_arrays = byte_product(jet_bytes, 2 * c2 + byte_product(c2, c2))
    l = orbital["maximum_angular"]
    a_l = auxiliary["maximum_angular"] if auxiliary else 0
    coulomb_order = max(4 * l, 2 * l + a_l, 2 * a_l)
    hermite = 6 * (max(l, a_l) + 3) ** 2 * (2 * max(l, a_l) + 4)
    # fill_coulomb allocates its full four-dimensional recurrence box. Only
    # one primitive quartet executes at a time; contraction length does not
    # multiply this workspace by the primitive-quartet count.
    recurrence = byte_product(jet_bytes, (coulomb_order + 1) ** 4 + hermite + 256 + d)
    preparation = jet_arrays + max(
        recurrence, cartesian + conventional + 8 * (2 * n2 + n2 * n2)
    )
    preparation += _ecp_workspace(item, library=library)
    resident = conventional
    matrix_work = byte_product(8, spin, n2, 64 + 2 * (diis_history + 1))
    matrix_work += byte_product(16, diis_history + 1, diis_history + 1)
    iteration = resident + matrix_work
    finalization = iteration
    if auxiliary:
        a, ac = auxiliary["nbf"], auxiliary["cartesian_nbf"]
        raw_count = byte_product(a, a) + byte_product(n2, a)
        cart_raw_count = byte_product(ac, ac) + byte_product(c2, ac)
        raw = byte_product(8, d + 1, raw_count)
        raw_cart = byte_product(8, d + 1, cart_raw_count)
        df_jets = byte_product(jet_bytes, cart_raw_count)
        transformed = byte_product(8, n2, a)
        preparation = max(
            preparation,
            conventional + df_jets + max(recurrence, raw_cart + raw + 8 * raw_count),
            conventional + raw + transformed + 8 * 16 * a * a,
        )
        resident += raw + transformed
        iteration = resident + matrix_work + 8 * (a + 4 * n2)
        # Metric pseudoinverse differentiation plus two full response tensors
        # coexist with SCF state during analytic DF force assembly.
        finalization = resident + matrix_work + 8 * (32 * a * a + 2 * n2 * a + 8 * d)
    return {
        name: checked_bytes(value, f"CPU {name} capacity")
        for name, value in {
            "preparation": preparation,
            "iteration": iteration,
            "finalization": finalization,
            "integral_state": resident,
        }.items()
    }


def _small_cuda_item_inventory(
    library: typing.Any,
    item: typing.Any,
    diis_history: typing.Any,
    precision: typing.Any,
    energy_tolerance: typing.Any,
    screening_tolerance: typing.Any,
) -> typing.Any:
    """Query the topology-aware small CUDA HF energy/force envelope.

    v2 receives shell angular momenta and contraction lengths so the native
    query can reproduce the quartet-direct force layout without constructing a
    CUDA plan or selecting a device profile. The aggregate v1 query is not a
    safe fallback because it cannot represent the force-route topology. Return
    raw allocator/ECP leaves; the fleet owner supplies the retained host bound.
    """
    query = getattr(library, "generativeqc_resource_small_hf_cuda_v2", None)
    if query is None:
        raise NotImplementedError(
            "native library has no topology-aware CUDA HF allocation inventory v2"
        )
    orbital = item["orbital"]
    scalar_args = (
        orbital["nbf"],
        orbital["cartesian_nbf"],
        item["atoms"],
        orbital["shells"],
        diis_history,
        item["spins"],
    )
    if any(
        value > 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1 for value in scalar_args
    ):
        raise ValueError("CUDA resource shape exceeds this host's size_t ABI")
    angular_values = tuple(orbital["shell_angular"])
    primitive_values = tuple(orbital["shell_primitive_counts"])
    if (
        len(angular_values) != orbital["shells"]
        or len(primitive_values) != orbital["shells"]
    ):
        raise ValueError("CUDA resource shell topology is incomplete")
    angular = (ctypes.c_uint8 * len(angular_values))(*angular_values)
    primitive_counts = (ctypes.c_size_t * len(primitive_values))(*primitive_values)
    query.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    output = (ctypes.c_uint64 * 2)()
    status = query(
        orbital["nbf"],
        orbital["cartesian_nbf"],
        item["atoms"],
        orbital["shells"],
        angular,
        primitive_counts,
        diis_history,
        item["spins"],
        0 if precision == "fp64" else 1,
        energy_tolerance,
        screening_tolerance,
        output,
        2,
    )
    if status:
        raise NotImplementedError(
            "CUDA HF inventory v2 requires an ordinary <=16-AO s/p/d/f topology "
            "with the small native eigensolver"
        )
    return {
        "arena": checked_bytes(int(output[0])),
        "host_staging": checked_bytes(int(output[1])),
        "ecp_workspace": _ecp_workspace(item, cuda=True, library=library),
    }


def _portable_small_cuda_item_inventory(
    item: typing.Any, capacities: typing.Any
) -> dict[str, int]:
    """Compatibility-only host bound for an absent native fleet symbol."""
    orbital = item["orbital"]
    pairs = orbital["shells"] * (orbital["shells"] + 1) // 2
    # Four copies allow vector growth, HostBatch staging, the cached topology
    # and the temporary cold retry retained alongside a warm bucket. At most
    # 16 shells reach this route; PSSS metadata is bounded by pair^2 records.
    metadata = 2048 * (
        1
        + item["atoms"]
        + orbital["shells"]
        + orbital["primitives"]
        + orbital["nbf"]
        + orbital["cartesian_nbf"]
        + pairs
        + pairs * pairs
    )
    host = (
        4 * capacities["host_staging"]
        + metadata
        + 8 * item["spins"] * orbital["nbf"] ** 2 * 64
        + 8 * 3 * item["atoms"] * 16
    )
    ecp_workspace = capacities["ecp_workspace"]
    return {
        "arena": capacities["arena"],
        "host": checked_bytes(host + ecp_workspace),
        "ecp_workspace": ecp_workspace,
    }


def _native_cuda_fleet_inventory(
    library: typing.Any,
    items: typing.Sequence[typing.Any],
    capacities: typing.Sequence[typing.Any],
) -> tuple[tuple[dict[str, int], ...], int, int, int] | None:
    """Marshal allocator/ECP leaves to the shared native direct-HF fleet owner.

    Missing-symbol compatibility is distinct from native refusal. The owner
    retains the independent arena/ECP retry maxima without creating a CUDA
    context; Python only validates publication and presents named capacities.
    """
    query = getattr(library, "generativeqc_resource_hf_cuda_fleet_v1", None)
    if query is None:
        return None
    dimensions = []
    for item, capacity in zip(items, capacities, strict=True):
        orbital = item["orbital"]
        dimensions.extend(
            (
                item["atoms"],
                orbital["nbf"],
                orbital["cartesian_nbf"],
                orbital["shells"],
                orbital["primitives"],
                item["spins"],
                capacity["arena"],
                capacity["host_staging"],
                capacity["ecp_workspace"],
            )
        )
    shapes = (ctypes.c_uint64 * len(dimensions))(
        *(checked_bytes(value, "CUDA HF fleet shape") for value in dimensions)
    )
    values = (ctypes.c_uint64 * (4 + 3 * len(items)))()
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    if query(shapes, len(shapes), values, len(values)):
        raise ValueError("native CUDA HF fleet inventory rejected shape or capacity")
    resident, retry, peak, host = (
        checked_bytes(values[index], "CUDA HF fleet capacity") for index in range(4)
    )
    if peak != resident + retry:
        raise ValueError("native CUDA HF fleet inventory returned inconsistent peak")
    keys = ("arena", "host", "ecp_workspace")
    inventories = tuple(
        {
            key: checked_bytes(values[4 + 3 * index + slot], f"CUDA HF {key} capacity")
            for slot, key in enumerate(keys)
        }
        for index in range(len(items))
    )
    for inventory, capacity in zip(inventories, capacities, strict=True):
        if (
            inventory["arena"] != capacity["arena"]
            or inventory["ecp_workspace"] != capacity["ecp_workspace"]
            or inventory["host"] <= inventory["ecp_workspace"]
        ):
            raise ValueError(
                "native CUDA HF fleet inventory returned inconsistent item"
            )
    return inventories, resident, retry, host


def hf_resource_request(
    systems: typing.Any,
    *,
    method: typing.Any = "rhf",
    basis: typing.Any = "sto-3g",
    backend: typing.Any = "cpu",
    basis_representation: typing.Any = None,
    charges: typing.Any = None,
    multiplicities: typing.Any = None,
    density_fitting: typing.Any = "none",
    auxiliary_basis: typing.Any = None,
    diis_history: typing.Any = 8,
    max_iterations: typing.Any = 100,
    energy_tolerance: typing.Any = 1e-10,
    density_tolerance: typing.Any = 1e-8,
    screening_tolerance: typing.Any = 1e-12,
    precision: typing.Any = "fp64",
    density_fitting_relative_threshold: typing.Any = 1e-10,
    density_fitting_memory_budget_bytes: typing.Any = 0,
    name: typing.Any = "hf",
    first_phase: typing.Any = 0,
    last_phase: typing.Any = 0,
    device_id: typing.Any = 0,
    library: typing.Any = None,
) -> typing.Any:
    """Resolve real basis/spin inputs into one serialized-fleet provider request.

    Workspace scales with the largest serialized item. Retained item state and
    output storage scale with the entire batch. A geometry-only change reuses
    this topology identity; geometry-derived caches and warm densities still
    obey their normal scientific invalidation rules at execution.
    """
    systems = tuple(tuple(Atom.from_value(a) for a in atoms) for atoms in systems)
    if not systems or any(not atoms for atoms in systems):
        raise ValueError("HF resource planning requires nonempty systems")
    charges = (0,) * len(systems) if charges is None else tuple(charges)
    multiplicities = (
        (1,) * len(systems) if multiplicities is None else tuple(multiplicities)
    )
    if len(charges) != len(systems) or len(multiplicities) != len(systems):
        raise ValueError("charges and multiplicities must match the fleet size")
    if method not in ("rhf", "uhf") or backend not in ("cpu", "cuda"):
        raise NotImplementedError(
            "HF resource planning requires an explicit RHF/UHF CPU/CUDA provider"
        )
    if density_fitting not in ("none", "cpu", "cpu_reference", "cuda", "auto"):
        raise ValueError("unknown density-fitting provider")
    if auxiliary_basis is not None and density_fitting == "none":
        raise ValueError("auxiliary basis requires density fitting")
    checked_bytes(diis_history, "DIIS history")
    checked_bytes(density_fitting_memory_budget_bytes, "DF memory sub-budget")
    checked_bytes(device_id, "device ordinal")
    checked_bytes(max_iterations, "maximum iterations")
    if not max_iterations:
        raise ValueError("maximum iterations must be positive")
    for value in (
        energy_tolerance,
        density_tolerance,
        screening_tolerance,
        density_fitting_relative_threshold,
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("HF numerical controls must be positive finite")
    if density_fitting_relative_threshold >= 1:
        raise ValueError("DF metric threshold must be below one")
    if precision not in ("fp64", "auto"):
        raise ValueError("precision must be 'fp64' or 'auto'")
    fitted = density_fitting != "none"
    executed_backend = "cpu" if density_fitting in ("cpu", "cpu_reference") else backend
    if density_fitting == "cuda" and backend != "cuda":
        raise ValueError("CUDA DF requires a CUDA backend")
    items = []
    for atoms, charge, multiplicity in zip(
        systems, charges, multiplicities, strict=True
    ):
        selected_basis = _snapshot_basis(basis, basis_representation)
        electrons = electron_state(
            atoms,
            charge=charge,
            multiplicity=multiplicity,
            element_metadata=selected_basis.by_element
            if isinstance(selected_basis, BasisSet)
            else {},
        )
        if method == "rhf" and multiplicity != 1:
            raise ValueError("RHF requires a closed-shell singlet")
        orbital = _basis_record(
            atoms,
            basis,
            basis_representation,
            executed_backend,
            charge,
            multiplicity,
            "orbital",
            library=library,
        )
        if max(electrons.nalpha, electrons.nbeta) > orbital["nbf"]:
            raise ValueError(
                "basis has fewer orbitals than the requested spin population"
            )
        auxiliary = None
        if fitted:
            if orbital.get("ecp_terms"):
                raise NotImplementedError(
                    "ECP density-fitting execution is not yet validated"
                )
            auxiliary = (
                orbital
                if auxiliary_basis is None
                else _basis_record(
                    atoms,
                    auxiliary_basis,
                    None
                    if isinstance(auxiliary_basis, (BasisSet, os.PathLike))
                    or (
                        isinstance(auxiliary_basis, str)
                        and auxiliary_basis.endswith(".json")
                    )
                    else basis_representation,
                    executed_backend,
                    charge,
                    multiplicity,
                    "auxiliary",
                    library=library,
                )
            )
        items.append(
            {
                "atoms": len(atoms),
                "orbital": orbital,
                "auxiliary": auxiliary,
                "electrons": asdict(electrons),
                "spins": 1 if method == "rhf" else 2,
            }
        )
    controls = {
        "diis_history": diis_history,
        "max_iterations": max_iterations,
        "energy_tolerance": energy_tolerance,
        "density_tolerance": density_tolerance,
        "screening_tolerance": screening_tolerance,
        "precision": precision,
        "density_fitting_relative_threshold": density_fitting_relative_threshold,
        "density_fitting": density_fitting,
        "density_fitting_memory_budget_bytes": density_fitting_memory_budget_bytes,
    }
    has_ecp = any(item["orbital"].get("ecp_terms") for item in items)
    if has_ecp:
        controls["ecp_inventory"] = _ecp_inventory_owner(library)
    if executed_backend == "cpu":
        native_fleet = getattr(library, "generativeqc_resource_hf_cpu_fleet_v1", None)
        controls["host_inventory"] = (
            "native-cpu-v1"
            if native_fleet is not None
            or getattr(library, "generativeqc_resource_hf_cpu_host_v1", None)
            is not None
            else "portable-metadata-v1"
        )
        controls["fleet_inventory"] = (
            "native-cpu-v1" if native_fleet is not None else "portable-metadata-v1"
        )
    if executed_backend == "cuda":
        controls.update(
            device_id=device_id,
            runtime_policy={
                key: os.environ.get(key) for key in _CUDA_SCHEDULE_VARIABLES
            },
        )
    identity = ResourceIdentity(
        method,
        "native-hf-df-v1" if fitted else "native-hf-direct-v1",
        executed_backend,
        precision,
        json.dumps({"items": items}),
        ("energy", "forces"),
        json.dumps(controls, sort_keys=True),
    )
    exclusions = (
        "caller-owned inputs and previously retained output objects",
        "Python/runtime objects, allocator bookkeeping, arenas and page rounding",
    )
    if (
        auxiliary_basis is not None
        and len({item["auxiliary"]["basis_hash"] for item in items}) > 1
    ):
        return ResourceRequest(
            name,
            identity,
            (),
            exclusions,
            unsupported_reason="native prepared fleets require one compatible explicit auxiliary template",
        )
    if executed_backend != "cpu":
        cuda_exclusions = exclusions + (
            "CUDA context, modules, graphs, stacks and allocator pool/page retention",
        )
        try:
            if os.environ.get("GENERATIVEQC_BOUNDED_DIRECT_FOCK_CLASS_PROFILE") in (
                "1",
                "profile",
            ):
                raise NotImplementedError(
                    "CUDA resource plans do not include optional class profiling"
                )
            if (
                os.environ.get("GENERATIVEQC_GRAPH_EIGENSOLVER_OVERRIDE")
                == "graph_native"
            ):
                raise NotImplementedError(
                    "CUDA inventory v1 requires the default small native eigensolver"
                )
            if library is None:
                from . import _native

                # CPU loading deliberately avoids select_library's GPU probe
                # and profile selection. The explicit library path can still
                # identify a CUDA-linked library for scalar layout queries.
                library = _native.load_library(device="cpu")
            if has_ecp:
                controls["ecp_inventory"] = _ecp_inventory_owner(library)
            if (
                not fitted
                and getattr(library, "generativeqc_resource_hf_cuda_fleet_v1", None)
                is not None
            ):
                controls["cuda_hf_fleet_composition"] = "native-declared-capacity"
            identity = replace(
                identity,
                schedule=json.dumps(
                    {**controls, **_cuda_library_identity(library)}, sort_keys=True
                ),
            )
            inventories = tuple(
                _small_cuda_item_inventory(
                    library,
                    item,
                    diis_history,
                    precision,
                    energy_tolerance,
                    screening_tolerance,
                )
                for item in items
            )
            if fitted:
                from .resources_hf_df import cuda_df_candidates

                candidates = cuda_df_candidates(
                    library,
                    items,
                    diis_history=diis_history,
                    device=device_id,
                    first_phase=first_phase,
                    last_phase=last_phase,
                    requested_budget=density_fitting_memory_budget_bytes,
                )
                return ResourceRequest(
                    name,
                    identity,
                    candidates,
                    cuda_exclusions
                    + (
                        "opaque CUDA library retention above the reported allowance is untracked",
                    ),
                )
            native_inventory = _native_cuda_fleet_inventory(library, items, inventories)
        except ValueError as error:
            if "budget" not in str(error):
                raise
            return ResourceRequest(
                name, identity, (), cuda_exclusions, infeasible_reason=str(error)
            )
        except (NotImplementedError, RuntimeError, OSError) as error:
            return ResourceRequest(
                name, identity, (), cuda_exclusions, unsupported_reason=str(error)
            )
        if native_inventory is None:
            inventories = tuple(
                _portable_small_cuda_item_inventory(item, capacity)
                for item, capacity in zip(items, inventories, strict=True)
            )
            resident = checked_bytes(sum(row["arena"] for row in inventories))
            retry = checked_bytes(
                max(row["arena"] for row in inventories)
                + max(row["ecp_workspace"] for row in inventories)
            )
            host = checked_bytes(sum(row["host"] for row in inventories))
        else:
            inventories, resident, retry, host = native_inventory
        candidate = ResourceCandidate(
            "cuda-small-resident",
            "resident",
            (
                ResourceEstimate(
                    "all retained direct-HF bucket arenas",
                    resident,
                    f"device:{device_id}",
                    first_phase,
                    last_phase,
                    kind="persistent",
                ),
                ResourceEstimate(
                    "one cold numerical retry alongside warm bucket",
                    retry,
                    f"device:{device_id}",
                    first_phase,
                    last_phase,
                ),
                ResourceEstimate(
                    "fleet host state, packed topology and staging capacity",
                    host,
                    "pageable",
                    first_phase,
                    last_phase,
                    kind="persistent",
                ),
            ),
            decisions=(
                ("batch_execution", "serialized buckets; every retained arena summed"),
                ("eigensolver", "small native; no external library workspace"),
                ("item_inventory", json.dumps(inventories, sort_keys=True)),
            ),
        )
        return ResourceRequest(name, identity, (candidate,), cuda_exclusions)
    native_inventory = _native_cpu_fleet_inventory(library, items, diis_history)
    if native_inventory is not None:
        inventories, retained, peak_work = native_inventory
    else:
        inventories = tuple(
            _cpu_item_inventory(item, diis_history, library=library) for item in items
        )
        retained = sum(
            8
            * (4 * item["spins"] * item["orbital"]["nbf"] ** 2 + 8 * 3 * item["atoms"])
            + 128
            * (
                item["orbital"]["primitives"]
                + item["orbital"]["shells"]
                + item["atoms"]
            )
            + (
                128 * (item["auxiliary"]["primitives"] + item["auxiliary"]["shells"])
                if item["auxiliary"]
                else 0
            )
            for item in items
        )
        peak_work = max(
            max(row["preparation"], row["iteration"], row["finalization"])
            for row in inventories
        )
    candidate = ResourceCandidate(
        "cpu-resident",
        "resident",
        (
            ResourceEstimate(
                "fleet topology, warm state and outputs",
                checked_bytes(retained),
                "pageable",
                first_phase,
                last_phase,
                kind="persistent",
            ),
            ResourceEstimate(
                "largest serialized item workspace",
                peak_work,
                "pageable",
                first_phase,
                last_phase,
            ),
        ),
        decisions=(
            ("batch_execution", "serialized items; retained state summed across fleet"),
            ("cpu_worker_limit", "1"),
            ("item_phase_inventory", json.dumps(inventories, sort_keys=True)),
        ),
    )
    return ResourceRequest(name, identity, (candidate,), exclusions)


def estimate_hf_resources(
    systems: typing.Any, *, budget: typing.Any = None, **options: typing.Any
) -> typing.Any:
    """Return an HF dry-run plan using compact topology and no native execution."""
    budget = ResourceBudget() if budget is None else budget
    return plan_resources(
        (hf_resource_request(systems, **options),),
        budget,
        library=options.get("library"),
    )
