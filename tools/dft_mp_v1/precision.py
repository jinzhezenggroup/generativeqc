"""Fail-closed consumption of public execution-owned precision evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def require_public_precision(value: object) -> dict[str, Any]:
    """Reject missing/partial public evidence without repairing any field."""

    if type(value) is not dict or type(value.get("detail_version")) is not int:
        raise ValueError("versioned public precision work missing")
    if value["detail_version"] != 1:
        raise ValueError("unsupported public precision-work version")
    if value.get("complete") is not True:
        raise ValueError("public precision work is incomplete")
    if value.get("operator_inventory_complete") is not True:
        raise ValueError("partial operator census")
    if type(value.get("returned_state")) is not str or not value["returned_state"]:
        raise ValueError("public precision returned-state identity missing")
    native = value.get("native_provenance")
    if (
        type(native) is not dict
        or native.get("operator_work_counters_valid") is not True
    ):
        raise ValueError("native operator work counters invalid")
    for key, native_value in native.items():
        if key in value and value[key] != native_value:
            raise ValueError(f"public aggregate/native {key} mismatch")
    if (
        type(value.get("scf_fock_timeline")) is not list
        or not value["scf_fock_timeline"]
    ):
        raise ValueError("SCF/Fock event timeline missing")
    if type(value.get("operators")) is not list or not value["operators"]:
        raise ValueError("executed operator census missing")
    return value


def precision_from_result(result: object) -> dict[str, Any]:
    """Copy Result.precision exactly for an acceptance adapter.

    No fallback accepts effective bits, an AUTO request, aggregate counts, or
    iteration counts in place of execution-owned events/operators.
    """

    return deepcopy(require_public_precision(getattr(result, "precision", None)))
