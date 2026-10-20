"""Synthetic protocol controls; no native execution or scientific acceptance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
import copy

import pytest

from benchmarks.cc_source_census import triples_expected, validate_cc_census
from benchmarks.cc_trace_reader import CC_PLAN, PHASE, T_PLAN, InvalidTrace


def fixture() -> tuple[dict[str, Any], dict[str, int], tuple[int, int, int]]:
    o, v, q = 2, 3, 5
    phases = {}
    plans = {}
    order = []
    stack = []
    next_id = 1
    plan = 0

    def begin(site: int, index: int = 0, detail: int = 0, dependency: int = 0) -> int:
        nonlocal next_id
        identity = next_id
        next_id += 1
        row = (
            PHASE,
            1,
            identity,
            stack[-1][0] if stack else 0,
            site,
            index,
            detail,
            plan,
            dependency,
            0,
            77,
            0,
        )
        phases[identity] = {
            "words": row,
            "start": identity * 100,
            "end": identity * 100 + 1,
            "status": 0,
        }
        order.append(identity)
        stack.append((identity, plan))
        return identity

    def end() -> None:
        nonlocal plan
        _, previous = stack.pop()
        plan = previous

    def admit(words: tuple[int, ...]) -> None:
        nonlocal next_id, plan
        identity = next_id
        next_id += 1
        plans[identity] = (words[0], 1, identity, stack[-1][0], *words[1:])
        plan = identity

    begin(1)
    begin(2, o + v, q)
    for i in range(o + v):
        begin(3, i, q)
        end()
        begin(22, i, o + v)
        end()
    begin(22, o + v, o + v)
    end()
    begin(22, o + v + 1, q)
    end()
    begin(23, o, v)
    end()
    copies = [
        q * o * o,
        q * o * v,
        q * v * v,
        (o * v) ** 2,
        (o * v) ** 2,
        (o * v) ** 2,
        o**3 * v,
        o**4,
    ]
    for index, count in enumerate(copies):
        if index >= 3:
            begin(27, index - 3, q)
            end()
        for site, detail in ((26, count), (24, 8 * count), (25, count)):
            begin(site, 0, detail)
            end()
    end()
    begin(16)
    admit((CC_PLAN, 1, q, 2, 1, 23, 8000000, 4000000, 3000000, 77, 0))
    tile_count = 0
    for site, index, batch in ((14, 0, 2), (14, 1, 2), (15, 0, 1)):
        begin(site, index, q)
        for offset in range(0, q, batch):
            width = min(batch, q - offset)
            begin(4, offset, width)
            begin(5, offset, width)
            end()
            producer = begin(6, offset, width)
            end()
            begin(7, offset, width, producer)
            end()
            end()
            tile_count += 1
        end()
    end()
    begin(8, o, v)
    admit((T_PLAN, o, v, q, 3, 4096, 0, 64, 64, 64, 8000000, 4000000, 77, 0))
    model = triples_expected(o, 3, True, True)
    panel_ids = {}
    for canonical, packed in model["tiles"]:
        begin(28, canonical, packed)
        for tile, value, slot in model["panels"]:
            if tile == canonical:
                panel_ids[slot] = begin(10, value, slot)
                end()
        for tile, a, b, slot in model["pairs"]:
            if tile == canonical:
                begin(29, a * 6 + b, 2, panel_ids[slot])
                end()
        for tile, seed, slot in model["singles"]:
            if tile == canonical:
                begin(11, seed, 1, panel_ids[slot])
                end()
        begin(12, canonical, 1)
        end()
        begin(30, canonical, 1)
        end()
        end()
    begin(30, len(model["tiles"]), 1)
    end()
    end()
    end()
    assert not stack
    semantic = {
        "df_source_scans": 1,
        "df_source_reads": o + v,
        "df_transform_stages": o + v + 7,
        "df_block_gemms": 5,
        "df_cuda_transform_calls": 1,
        "df_transform_fmas": 2 * q * (o + v) ** 3
        + q**2 * (o + v) ** 2
        + q * (3 * (o * v) ** 2 + o**3 * v + o**4),
        "ccsd_evaluations": 2,
        "ccsd_replay_evaluations": 1,
        "ccsd_q_tiles": tile_count,
        "ccsd_q_slices": 3 * q,
        "triples_panel_calls": len(model["panels"]),
        "triples_epilogue_kernels": len(model["tiles"]),
        "triples_panel_capacity": 3,
        "triples_reduction_kernels": len(model["tiles"]) + 1,
        "triples_moment_products": 2 * o**3,
        "triples_fp64_gemms": model["vendor_gemm_calls_if_all_pairs_accepted"],
    }
    return {"phases": phases, "plans": plans, "begin_order": order}, semantic, (o, v, q)


def controls() -> dict[str, Any]:
    inventory, semantic, geometry = fixture()
    positive = validate_cc_census(inventory, semantic, geometry)
    cases = []

    def reject(
        name: str, change: Callable[[dict[str, Any], dict[str, int]], None]
    ) -> None:
        actual, work = copy.deepcopy((inventory, semantic))
        change(actual, work)
        try:
            validate_cc_census(actual, work, geometry)
        except (InvalidTrace, KeyError) as error:
            cases.append({"case": name, "rejected": True, "reason": str(error)})
        else:
            raise AssertionError(name + " was accepted")

    def omit(site: int) -> Callable[[dict[str, Any], dict[str, int]], None]:
        def modify(actual: dict[str, Any], _: dict[str, int]) -> None:
            identity = next(
                i for i, p in actual["phases"].items() if p["words"][4] == site
            )
            del actual["phases"][identity]
            actual["begin_order"].remove(identity)

        return modify

    reject("joint journal/correlation host-range omission", omit(5))
    reject("joint projection omission", omit(22))
    reject("joint triples epilogue omission", omit(12))
    reject("joint panel-production omission", omit(10))

    def bad_dependency(actual: dict[str, Any], _: dict[str, int]) -> None:
        p = next(p for p in actual["phases"].values() if p["words"][4] == 11)
        words = list(p["words"])
        words[8] = 0
        p["words"] = tuple(words)

    reject("late-enable cache-hit producer remains unknown", bad_dependency)
    reject(
        "logical products incorrectly used as vendor calls",
        lambda _, s: s.__setitem__("triples_fp64_gemms", s["triples_moment_products"]),
    )
    reject(
        "existing semantic Q count mismatch",
        lambda _, s: s.__setitem__("ccsd_q_tiles", s["ccsd_q_tiles"] - 1),
    )

    def extra_subtree(actual: dict[str, Any], _: dict[str, int]) -> None:
        first = max([*actual["phases"], *actual["plans"]]) + 1
        root = next(
            identity for identity, p in actual["phases"].items() if p["words"][4] == 1
        )
        plan = next(
            identity for identity, p in actual["plans"].items() if p[0] == CC_PLAN
        )
        for offset, site in enumerate((4, 5, 6, 7)):
            identity = first + offset
            parent = root if site == 4 else first
            dependency = first + 2 if site == 7 else 0
            row = (PHASE, 1, identity, parent, site, 0, 1, plan, dependency, 0, 77, 0)
            actual["phases"][identity] = {
                "words": row,
                "start": identity * 100,
                "end": identity * 100 + 1,
                "status": 0,
            }
            actual["begin_order"].append(identity)

    reject(
        "joint extra complete Q subtree with unchanged independent work", extra_subtree
    )
    return {
        "scope": "synthetic independent-census controls only; not actual source/device/science coverage",
        "positive": positive,
        "negative_controls": cases,
    }


def test_independent_census_rejects_joint_omissions_and_extra_work() -> None:
    result = controls()
    assert len(result["negative_controls"]) == 8
    assert all(row["rejected"] for row in result["negative_controls"])


def test_logical_vendor_and_kernel_layers_remain_distinct() -> None:
    model = triples_expected(9, 3, True, True)
    assert len(model["tiles"]) == 165
    assert model["logical_w_products"] == 1458
    assert model["vendor_gemm_calls_if_all_pairs_accepted"] == 908
    assert model["panel_probes"] == 405


@pytest.mark.parametrize("site", [3, 22, 23, 24, 25, 26, 27])
@pytest.mark.parametrize("corruption", ["index", "shape", "parent"])
def test_df_domains_reject_count_preserving_corruption(
    site: int, corruption: str
) -> None:
    inventory, semantic, geometry = fixture()
    phase = next(p for p in inventory["phases"].values() if p["words"][4] == site)
    words = list(phase["words"])
    if corruption == "index":
        # Replacing one row/projection/block by a duplicate keeps all counts.
        words[5] = 1
    elif corruption == "shape":
        words[6] += 1
    else:
        words[3] = next(i for i, p in inventory["phases"].items() if p["words"][4] == 1)
    phase["words"] = tuple(words)
    with pytest.raises(InvalidTrace):
        validate_cc_census(inventory, semantic, geometry)


def test_df_domain_checks_bound_huge_ao_before_enumeration() -> None:
    inventory, semantic, (_, v, q) = fixture()
    o = 2**52
    for identity, original in inventory["plans"].items():
        if original[0] == T_PLAN:
            words = list(original)
            words[4] = o
            inventory["plans"][identity] = tuple(words)
    with pytest.raises(InvalidTrace):
        validate_cc_census(inventory, semantic, (o, v, q))


def test_df_raw_values_are_not_contraction_work() -> None:
    inventory, semantic, geometry = fixture()
    o, v, q = geometry
    raw_values = (o + v) ** 2 * q
    result = validate_cc_census(inventory, semantic, geometry)
    assert result["df_independent_raw_values"] == raw_values
    semantic["df_transform_fmas"] += raw_values
    with pytest.raises(InvalidTrace, match="independent DF geometry/source work"):
        validate_cc_census(inventory, semantic, geometry)


def test_huge_q_domain_is_rejected_before_materializing_tiles() -> None:
    inventory, semantic, (o, v, old_q) = fixture()
    q = 2**52
    n = o + v
    semantic["df_transform_fmas"] = (
        2 * q * n**3 + q * q * n * n + q * (3 * (o * v) ** 2 + o**3 * v + o**4)
    )
    for identity, original in inventory["plans"].items():
        words = list(original)
        words[5 if words[0] == CC_PLAN else 6] = q
        inventory["plans"][identity] = tuple(words)
    for phase in inventory["phases"].values():
        words = list(phase["words"])
        if words[4] in (2, 3, 27) or (words[4] == 22 and words[5] == o + v + 1):
            words[6] = q
            phase["words"] = tuple(words)
    for site in (24, 25, 26):
        copies = [p for p in inventory["phases"].values() if p["words"][4] == site]
        for phase in copies[:3]:
            words = list(phase["words"])
            words[6] = words[6] // old_q * q
            phase["words"] = tuple(words)
    with pytest.raises(InvalidTrace, match="Q tile cardinality"):
        validate_cc_census(inventory, semantic, (o, v, q))
