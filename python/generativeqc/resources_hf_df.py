"""Presentation of the shared native CUDA DF numeric bucket/fleet inventory."""

from __future__ import annotations

import ctypes
import json
import os
from typing import Any

from generativeqc_compiler.common.native_call import checked_native_call
from generativeqc_compiler.common.resources import (
    ResourceCandidate,
    ResourceEstimate,
    checked_bytes,
)

from .resources_df import density_fitting_value_layout

_BUCKET_KEYS = (
    "batch",
    "nbf",
    "naux",
    "cartesian_nbf",
    "cartesian_naux",
    "coordinates",
    "source_bytes",
    "diis_device_bytes",
    "occupied",
    "rhf_occupied",
    "host_metadata",
    "preparation_minimum",
    "preparation_retained",
    "preparation_temporary",
    "response_minimum",
    "resident_response_capacity",
    "response_capacity",
    "response_host_capacity",
    "resident_host_bytes",
    "overlap_cache_host_bytes",
    "ordinary_eigen_device_bytes",
    "final_snapshot_device_bytes",
    "final_selection_host_bytes",
    "ordinary_eigen_host_workspace_bytes",
    "resident_device_bytes",
    "rhf_authorized",
)
_TILE_KEYS = (
    "batch_tile",
    "ao_pair_tile",
    "auxiliary_tile",
    "occupied_tile",
    "peak_workspace_bytes",
    "stores_full_three_center",
    "budget_bytes",
    "fixed_device_bytes",
    "generated_source",
    "rank_capacity",
    "stored_factor_bytes",
    "raw_factor_bytes",
    "contraction_scratch_bytes",
    "projection_capacity_elements",
    "panel_capacity_elements",
    "automatic_rhf_rank",
    "value_layout_elements_per_system",
    "dense_equivalent_elements_per_system",
    "bounded_materialization_bytes",
    "bounded_conversion_traffic_bytes",
)
_BUCKET_WIDTH = len(_BUCKET_KEYS) + 2 * len(_TILE_KEYS)


def _tile(values: list[int], row: dict[str, Any], pair_storage: str) -> dict[str, Any]:
    """Decode numeric capacities; the compiler supplies only the layout label."""
    result = dict(zip(_TILE_KEYS, values, strict=True))
    for key in ("stores_full_three_center", "generated_source"):
        if result[key] not in (0, 1):
            raise ValueError("native CUDA DF fleet returned invalid tile flags")
        result[key] = bool(result[key])
    if (
        not 0 < result["batch_tile"] <= row["batch"]
        or not 0 < result["ao_pair_tile"] <= row["nbf"] ** 2
        or not 0 < result["auxiliary_tile"] <= row["naux"]
        or not 0 < result["occupied_tile"] <= row["nbf"]
        or result["peak_workspace_bytes"] < result["fixed_device_bytes"]
        or (
            result["budget_bytes"]
            and result["peak_workspace_bytes"] > result["budget_bytes"]
        )
    ):
        raise ValueError("native CUDA DF fleet returned incoherent tile capacity")
    result["pair_storage"] = pair_storage
    layout = density_fitting_value_layout(row["nbf"], row["naux"], pair_storage)
    result["value_layout_identity"] = layout.identity
    dense_elements = (
        layout.dense_elements if pair_storage == "packed" else layout.storage_elements
    )
    if (
        result["value_layout_elements_per_system"] != layout.storage_elements
        or result["dense_equivalent_elements_per_system"] != dense_elements
    ):
        raise ValueError(
            "native CUDA DF capacity disagrees with compiler layout contract"
        )
    if pair_storage == "packed":
        factor = 8 * row["batch"] * layout.storage_elements
        if (
            result["stored_factor_bytes"] != factor
            or result["raw_factor_bytes"] != factor
        ):
            raise ValueError(
                "native packed DF factor disagrees with compiler layout contract"
            )
    return result


def cuda_df_candidates(
    library: Any,
    items: Any,
    *,
    diis_history: Any,
    device: Any,
    first_phase: Any,
    last_phase: Any,
    requested_budget: Any = 0,
) -> tuple[ResourceCandidate, ...]:
    """Render authoritative native choices without Python capacity composition.

    All retained buckets coexist with serialized work and an independent cold
    singleton retry. The existing CUDA SCF/CPU numerical-recovery contract is
    unchanged. Only a missing symbol permits historical Python composition;
    native rejection or an incoherent wire result is final, never a fallback.
    """
    query = getattr(library, "generativeqc_resource_hf_df_cuda_fleet_v1", None)
    if query is None:
        from ._resources_hf_df_compat import cuda_df_candidates as legacy_candidates

        return legacy_candidates(
            library,
            items,
            diis_history=diis_history,
            device=device,
            first_phase=first_phase,
            last_phase=last_phase,
            requested_budget=requested_budget,
        )
    storage = os.environ.get("GENERATIVEQC_DF_VALUE_STORAGE", "auto")
    selectors = ("auto", "dense", "packed", "packed-single")
    if storage not in selectors:
        raise ValueError(
            "GENERATIVEQC_DF_VALUE_STORAGE must be auto, dense, packed or packed-single"
        )
    pair_storage = "packed" if storage in ("packed", "packed-single") else "dense"
    expected_kinds = (
        (2,) if pair_storage == "packed" else (1,) if requested_budget else (0, 1)
    )
    checked_bytes(diis_history, "DF DIIS history")
    checked_bytes(requested_budget, "DF sub-budget")
    if diis_history > 2 ** (8 * ctypes.sizeof(ctypes.c_uint)) - 1:
        raise ValueError("DF DIIS history exceeds the native unsigned domain")
    items = tuple(items)
    dimensions = []
    for item in items:
        orbital, auxiliary, state = (
            item["orbital"],
            item["auxiliary"],
            item["electrons"],
        )
        dimensions.extend(
            checked_bytes(value, "CUDA DF fleet shape")
            for value in (
                item["atoms"],
                orbital["nbf"],
                orbital["cartesian_nbf"],
                orbital["shells"],
                orbital["primitives"],
                auxiliary["nbf"],
                auxiliary["cartesian_nbf"],
                auxiliary["shells"],
                auxiliary["primitives"],
                state["nalpha"],
                state["nbeta"],
                item["spins"],
                1,
            )
        )
    shapes = (ctypes.c_uint64 * len(dimensions))(*dimensions)
    query.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_uint64,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    query.restype = ctypes.c_int
    arguments = (
        shapes,
        len(shapes),
        diis_history,
        requested_budget,
        selectors.index(storage),
        int(os.environ.get("GENERATIVEQC_DF_EXCHANGE") == "occupied"),
    )
    header = (ctypes.c_uint64 * 3)()
    checked_native_call(
        query,
        *arguments,
        header,
        len(header),
        error_type=ValueError,
        status_error_types={2: NotImplementedError},
    )
    buckets, choices, length = header
    if (
        not 0 < buckets <= len(items)
        or choices != len(expected_kinds)
        or length != 3 + choices * (8 + buckets * _BUCKET_WIDTH)
    ):
        raise ValueError("native CUDA DF fleet returned invalid schema")
    output = (ctypes.c_uint64 * length)()
    checked_native_call(
        query,
        *arguments,
        output,
        len(output),
        error_type=ValueError,
        status_error_types={2: NotImplementedError},
    )
    values = [checked_bytes(value, "native CUDA DF capacity") for value in output]
    if values[:3] != list(header):
        raise ValueError("native CUDA DF fleet changed its publication schema")
    candidates = []
    offset = 3
    for expected_kind in expected_kinds:
        (
            kind,
            sub_budget,
            cost,
            host,
            host_work,
            device_bytes,
            device_work,
            allowance,
        ) = values[offset : offset + 8]
        offset += 8
        if (
            kind != expected_kind
            or (kind == 0) != (sub_budget == 0)
            or (requested_budget and sub_budget != requested_budget)
            or not host_work
            or not device_work
        ):
            raise ValueError("native CUDA DF fleet returned invalid candidate")
        source = sub_budget > 0
        inventories = []
        for _ in range(buckets):
            row = dict(
                zip(
                    _BUCKET_KEYS,
                    values[offset : offset + len(_BUCKET_KEYS)],
                    strict=True,
                )
            )
            offset += len(_BUCKET_KEYS)
            restricted = row.pop("rhf_authorized")
            if (
                restricted not in (0, 1)
                or not row["batch"]
                or not 0 < row["nbf"] <= 16
                or not 0 < row["naux"] <= 128
            ):
                raise ValueError("native CUDA DF fleet returned invalid bucket")
            if not restricted:
                row["rhf_occupied"] = None
            for route in ("energy_tiles", "force_tiles"):
                row[route] = _tile(
                    values[offset : offset + len(_TILE_KEYS)], row, pair_storage
                )
                expected_budget = (
                    sub_budget if route == "energy_tiles" else sub_budget // 2
                )
                if (
                    row[route]["budget_bytes"] != expected_budget
                    or row[route]["generated_source"] != source
                ):
                    raise ValueError(
                        "native CUDA DF fleet changed its budget/source contract"
                    )
                offset += len(_TILE_KEYS)
            row["tiles"] = row["force_tiles"]
            inventories.append(row)
        if sum(row["batch"] for row in inventories) != len(items):
            raise ValueError("native CUDA DF fleet omitted or duplicated items")
        if host != sum(
            row["resident_host_bytes"] for row in inventories
        ) or device_bytes != sum(row["resident_device_bytes"] for row in inventories):
            raise ValueError(
                "native CUDA DF fleet returned inconsistent retained totals"
            )
        estimates = (
            ResourceEstimate(
                "all DF bucket caches and native solver workspaces",
                device_bytes,
                f"device:{device}",
                first_phase,
                last_phase,
                kind="persistent",
                recomputable=source,
            ),
            ResourceEstimate(
                "largest serialized DF setup/force/retry workspace",
                device_work,
                f"device:{device}",
                first_phase,
                last_phase,
            ),
            ResourceEstimate(
                "all retained DF host state",
                host,
                "pageable",
                first_phase,
                last_phase,
                kind="persistent",
                cacheable=not source,
            ),
            ResourceEstimate(
                "largest serialized DF host workspace",
                host_work,
                "pageable",
                first_phase,
                last_phase,
            ),
            ResourceEstimate(
                "opaque CUDA library retention allowance",
                allowance,
                f"device:{device}",
                first_phase,
                last_phase,
                kind="library",
                accounting="runtime_allowance",
            ),
        )
        identities = sorted(
            {
                row[route]["value_layout_identity"]
                for row in inventories
                for route in ("energy_tiles", "force_tiles")
            }
        )
        candidates.append(
            ResourceCandidate(
                ("cuda-df-resident", "cuda-df-source", "cuda-df-packed")[kind],
                "recomputed" if kind == 1 else "resident",
                estimates,
                relative_cost=cost,
                decisions=(
                    ("df_pair_storage", pair_storage),
                    ("df_value_layout_identities", json.dumps(identities)),
                    ("density_fitting_memory_budget_bytes", str(sub_budget)),
                    (
                        "batch_execution",
                        "serialized buckets; all retained caches summed",
                    ),
                    (
                        "scf_driver",
                        "CUDA SCF with existing CPU numerical recovery; budget-selected CUDA forward storage, generated response and J/K"
                        if source
                        else "CUDA SCF with existing CPU numerical recovery; CUDA J/K",
                    ),
                    ("bucket_inventory", json.dumps(inventories, sort_keys=True)),
                ),
            )
        )
    return tuple(candidates)
