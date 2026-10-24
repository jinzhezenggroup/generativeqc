"""Capture returned observables without touching optional result metadata."""

from __future__ import annotations

from typing import Any


def capture(
    result: Any, call: dict, atoms: list, elapsed: float
) -> tuple[dict, tuple | list | None]:
    availability = {}
    errors = []

    def get(owner: Any, field: str, label: str) -> Any:
        try:
            value = getattr(owner, field)
        except Exception as error:  # noqa: BLE001 -- preserve other already returned observables before reporting this failure
            message = f"{type(error).__name__}: {error}"
            errors.append({"field": label, "error": message})
            availability[label] = "UNAVAILABLE"
            return {"availability": "UNAVAILABLE", "error": message}
        availability[label] = "RETURNED_NONE" if value is None else "RETURNED"
        return value

    core = {
        "schema": "generativeqc.h100-small-cbs-observables.v1",
        "call": call,
        "atoms_bohr": atoms,
        "complete_seconds": elapsed,
        "complete_seconds_scope": "workflow construction when cold plus singlepoint through return; excludes retention/enrichment/gates",
        "energy": get(result, "energy", "energy"),
        "forces": get(result, "forces", "forces"),
    }
    refs = get(result, "components", "components")
    if isinstance(refs, (list, tuple)):
        constituents = []
        for index, component in enumerate(refs):
            native = get(component, "result", f"components.{index}.result")
            constituents.append(
                {
                    "energy": get(native, "energy", f"components.{index}.energy"),
                    "forces": get(native, "forces", f"components.{index}.forces"),
                }
            )
        core["components"] = constituents
    else:
        core["components"] = refs
        refs = None
    core.update(observable_availability=availability, capture_errors=errors)
    return core, refs
