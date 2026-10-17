"""Compiled CUDA DF composition contracts; these tests do not execute a GPU."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import (
    ResourceBudget,
    _native,
    _resources_hf_df_compat,
    resources_hf_df,
)
from generativeqc.resources_planning import plan_resources
from generativeqc_compiler.common.resources import ResourceIdentity, ResourceRequest


@pytest.fixture(scope="module")
def library() -> Any:
    """Require the real compiled owner, including in a CPU-only library."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_hf_df_cuda_fleet_v1")
    return loaded


def topology(nbf: int = 2, *, spins: int = 1) -> dict[str, Any]:
    """Supply expanded scalar topology without constructing scientific arrays."""
    alpha = max(1, nbf // 2)
    return {
        "atoms": 2 if nbf == 2 else 3,
        "orbital": {
            "nbf": nbf,
            "cartesian_nbf": nbf if nbf == 2 else nbf + 1,
            "shells": 2 if nbf == 2 else 3,
            "primitives": 6 if nbf == 2 else 9,
        },
        "auxiliary": {
            "nbf": 2 if nbf == 2 else 9,
            "cartesian_nbf": 2 if nbf == 2 else 10,
            "shells": 2 if nbf == 2 else 4,
            "primitives": 6 if nbf == 2 else 12,
        },
        "electrons": {"nalpha": alpha, "nbeta": alpha if spins == 1 else alpha - 1},
        "spins": spins,
    }


def candidates(library: Any, items: list[dict[str, Any]], **options: Any) -> Any:
    """Use stable public phase/device labels for both native and legacy routes."""
    return resources_hf_df.cuda_df_candidates(
        library,
        items,
        diis_history=options.pop("history", 8),
        device=3,
        first_phase=4,
        last_phase=12,
        **options,
    )


class OldLibrary:
    """Model an older real SDK, not a second numerical implementation."""

    def __init__(self, library: Any) -> None:
        self.library = library

    def __getattr__(self, name: str) -> Any:
        if name == "generativeqc_resource_hf_df_cuda_fleet_v1":
            raise AttributeError(name)
        return getattr(self.library, name)


@pytest.mark.parametrize("storage", ("auto", "dense", "packed", "packed-single"))
@pytest.mark.parametrize("exchange", ("auto", "occupied"))
@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("history", (0, 1, 8))
@pytest.mark.parametrize("ragged", (False, True))
@pytest.mark.parametrize("budget", (0, 64 << 20))
def test_complete_publication_matches_old_library_contract(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
    storage: str,
    exchange: str,
    spins: int,
    history: int,
    ragged: bool,
    budget: int,
) -> None:
    """Preserve every capacity, tile, layout, lifetime and diagnostic decision."""
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", storage)
    monkeypatch.setenv("GENERATIVEQC_DF_EXCHANGE", exchange)
    first = topology(spins=spins)
    items = [first, copy.deepcopy(first)]
    items[1]["auxiliary"]["primitives"] += 3
    if ragged:
        items.insert(0, topology(5, spins=spins))
    reference = candidates(
        OldLibrary(library), items, history=history, requested_budget=budget
    )
    actual = candidates(library, items, history=history, requested_budget=budget)
    assert actual == reference
    assert (
        candidates(library, items[::-1], history=history, requested_budget=budget)
        == actual
    )


@pytest.mark.parametrize("storage", ("dense", "packed", "packed-single"))
def test_sub_budget_exact_and_one_byte_below(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
    storage: str,
) -> None:
    """Find the actual native admission boundary, then check both SDK routes."""
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", storage)
    items = [topology(5)]
    lower, upper = 1, 64 << 20
    while lower < upper:
        middle = (lower + upper) // 2
        try:
            candidates(library, items, requested_budget=middle)
        except ValueError:
            lower = middle + 1
        else:
            upper = middle
    assert candidates(library, items, requested_budget=lower) == candidates(
        OldLibrary(library), items, requested_budget=lower
    )
    for selected in (library, OldLibrary(library)):
        with pytest.raises(ValueError):
            candidates(selected, items, requested_budget=lower - 1)


@pytest.mark.parametrize("kind", (0, 1))
@pytest.mark.parametrize("space", ("host", "device"))
def test_complete_candidate_exact_and_below_global_cap(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
    kind: int,
    space: str,
) -> None:
    """Exercise native global composition with the actual DF inventory."""
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "dense")
    selected = candidates(library, [topology(5), topology(), topology()])[kind]
    identity = ResourceIdentity(
        "rhf",
        "cuda-df",
        "cuda",
        "fp64",
        json.dumps({"test": "DF scalar inventory"}),
        ("energy", "forces"),
        "serialized DF buckets",
    )
    request = ResourceRequest("hf", identity, (selected,))
    plan = plan_resources(
        (request,), ResourceBudget(), library=library
    ).require_feasible()
    exact = ResourceBudget(
        host_bytes=plan.peak_bytes["host"],
        device_bytes=plan.peak_bytes["device"],
    )
    plan_resources((request,), exact, library=library).require_feasible()
    short = replace(exact, **{f"{space}_bytes": plan.peak_bytes[space] - 1})
    with pytest.raises(MemoryError):
        plan_resources((request,), short, library=library).require_feasible()


@pytest.mark.parametrize(
    "corruption",
    (
        "refusal",
        "header",
        "length",
        "changed_header",
        "candidate",
        "overflow",
        "retained_total",
        "tile_flag",
        "tile_shape",
        "rhf_flag",
    ),
)
def test_native_refusal_and_malformed_output_never_retry_python(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    """Native publication is authoritative even when old-library math could fit."""
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "dense")
    query = library.generativeqc_resource_hf_df_cuda_fleet_v1

    def forbidden(*arguments: Any, **options: Any) -> Any:
        raise AssertionError("native DF refusal retried compatibility composition")

    monkeypatch.setattr(_resources_hf_df_compat, "cuda_df_candidates", forbidden)

    def corrupt(*arguments: Any) -> int:
        if corruption == "refusal":
            return 1
        status = query(*arguments)
        assert status == 0
        output, length = arguments[6:8]
        if length == 3:
            if corruption == "header":
                output[0] = 0
            elif corruption == "length":
                output[2] += 1
        else:
            slot = {
                "changed_header": 0,
                "candidate": 3,
                "overflow": 6,
                "retained_total": 6,
                "tile_flag": 42,
                "tile_shape": 37,
                "rhf_flag": 36,
            }.get(corruption)
            if slot is not None:
                output[slot] = (
                    (2**64 - 1) if corruption == "overflow" else output[slot] + 100
                )
        return status

    selected = SimpleNamespace(generativeqc_resource_hf_df_cuda_fleet_v1=corrupt)
    with pytest.raises(ValueError):
        candidates(selected, [topology()])


@pytest.mark.parametrize(
    "field,value",
    (
        ("atoms", 0),
        ("spins", 3),
        ("primitives", 2**62),
        ("nbf", 17),
        ("cartesian_nbf", 1),
        ("naux", 129),
        ("nalpha", 3),
    ),
)
def test_native_topology_refusal_is_transactional(
    library: Any,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
) -> None:
    """Invalid later items cannot leave usable partial candidate publications."""
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "dense")
    first, invalid = topology(), topology()
    target = (
        invalid
        if field in ("atoms", "spins")
        else invalid["electrons"]
        if field == "nalpha"
        else invalid["auxiliary"]
        if field == "naux"
        else invalid["orbital"]
    )
    target["nbf" if field == "naux" else field] = value
    expected_error = NotImplementedError if field in ("nbf", "naux") else ValueError
    with pytest.raises(expected_error):
        candidates(library, [first, invalid])


def test_existing_bucket_compatibility_guard(library: Any) -> None:
    """Equal orbital/electron keys do not authorize incompatible auxiliaries."""
    first, second = topology(), topology()
    second["auxiliary"]["nbf"] = 3
    second["auxiliary"]["cartesian_nbf"] = 3
    with pytest.raises(NotImplementedError, match="incompatible"):
        candidates(library, [first, second])
