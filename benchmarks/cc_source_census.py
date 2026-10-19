"""Finite independent input/plan/work census. No observed count invents completeness."""

from __future__ import annotations

from itertools import permutations
from typing import Any

try:
    from benchmarks.cc_trace_reader import CC_PLAN, T_PLAN, require
except ModuleNotFoundError:
    from cc_trace_reader import CC_PLAN, T_PLAN, require

PERMUTATIONS = tuple(permutations(range(3)))


def triples_expected(
    occupied: int, panel_slots: int, paired: bool, serpentine: bool
) -> dict[str, Any]:
    require(
        type(occupied) is int and 1 <= occupied <= 64 and 1 <= panel_slots <= 3,
        "unsupported triples input domain",
    )
    identities = [None] * panel_slots
    ages = [0] * panel_slots
    epoch = 0
    tiles = []
    panels = []
    pairs = []
    singles = []
    probes = 0
    for largest in range(occupied):
        if serpentine:
            visits = [
                (largest, middle, smallest)
                for smallest in (
                    range(largest, -1, -1) if largest % 2 else range(largest + 1)
                )
                for middle in (
                    range(largest, smallest - 1, -1)
                    if smallest % 2
                    else range(smallest, largest + 1)
                )
            ]
        else:
            visits = [
                (largest, middle, smallest)
                for middle in range(largest + 1)
                for smallest in range(middle + 1)
            ]
        for physical in visits:
            i, j, k = physical
            canonical = i * (i + 1) * (i + 2) // 6 + j * (j + 1) // 2 + k
            tiles.append((canonical, (i * occupied + j) * occupied + k))
            unique_seeds = {}
            seed_indices = []
            for index, order in enumerate(PERMUTATIONS):
                ordered = tuple(physical[x] for x in order)
                if ordered not in unique_seeds:
                    unique_seeds[ordered] = index
                    seed_indices.append(index)
            for value in dict.fromkeys(physical):
                probes += 1
                if value not in identities:
                    slot = min(range(panel_slots), key=lambda x: ages[x])
                    identities[slot] = value
                    panels.append((canonical, value, slot))
                else:
                    slot = identities.index(value)
                epoch += 1
                ages[slot] = epoch
                seeds = [
                    index
                    for index in seed_indices
                    if physical[PERMUTATIONS[index][0]] == value
                ]
                if paired and len(seeds) == 2:
                    a, b = seeds
                    oa, ob = PERMUTATIONS[a], PERMUTATIONS[b]
                    swapped = (
                        physical[ob[0]] == physical[oa[0]]
                        and physical[ob[1]] == physical[oa[2]]
                        and physical[ob[2]] == physical[oa[1]]
                    )
                    if swapped and b == a + 1:
                        pairs.append((canonical, a, b, slot))
                        continue
                singles.extend((canonical, index, slot) for index in seeds)
    # A separate mathematical domain check catches omissions in the visit model.
    expected = {
        (
            i * (i + 1) * (i + 2) // 6 + j * (j + 1) // 2 + k,
            (i * occupied + j) * occupied + k,
        )
        for i in range(occupied)
        for j in range(i + 1)
        for k in range(j + 1)
    }
    require(
        set(tiles) == expected and len(tiles) == len(expected),
        "triangular enumeration incomplete",
    )
    return {
        "tiles": tiles,
        "panels": panels,
        "pairs": pairs,
        "singles": singles,
        "panel_probes": probes,
        "logical_w_products": 2 * (len(singles) + 2 * len(pairs)),
        "vendor_gemm_calls_if_all_pairs_accepted": len(panels)
        + 2 * (len(singles) + len(pairs)),
    }


def validate_cc_census(
    inventory: dict[str, Any],
    semantic: dict[str, int],
    geometry: tuple[int, int, int],
    allowed_lowerings: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    phases = inventory["phases"]
    plans = inventory["plans"]
    o, v, q = geometry
    cc = [(identity, row) for identity, row in plans.items() if row[0] == CC_PLAN]
    tp = [(identity, row) for identity, row in plans.items() if row[0] == T_PLAN]
    require(len(cc) == len(tp) == 1, "missing/extra prepared plan facts")
    cc_id, cc_plan = cc[0]
    t_id, t_plan = tp[0]
    require(
        cc_plan[4] == 1 and phases[cc_plan[3]]["words"][4] == 16,
        "CC plan owner/native parent mismatch",
    )
    require(
        cc_plan[5] == q
        and cc_plan[6] > 0
        and cc_plan[7] > 0
        and cc_plan[8] & 3 == 3
        and cc_plan[8] & ~63 == 0
        and cc_plan[8] & (8 | 32) == 0,
        "unsupported CC plan/unknown route/mixed domain",
    )
    require(
        tuple(t_plan[4:7]) == (o, v, q) and tuple(t_plan[10:13]) == (64, 64, 64),
        "triples geometry/precision mismatch",
    )
    require(
        phases[t_plan[3]]["words"][4] == 8 and t_plan[9] in allowed_lowerings,
        "unadmitted triples parent/lowering",
    )
    require(
        t_plan[14] <= t_plan[13] and t_plan[7] == semantic["triples_panel_capacity"],
        "triples reservation/panel admission mismatch",
    )
    groups = {
        site: [phase for phase in phases.values() if phase["words"][4] == site]
        for site in range(1, 33)
    }
    require(len(groups[2]) == semantic["df_source_scans"] == 1, "DF source scan census")
    require(len(groups[3]) == semantic["df_source_reads"] == o + v, "DF raw row census")
    require(
        len(groups[22]) + len(groups[27]) == semantic["df_transform_stages"],
        "DF projection/block census",
    )
    require(len(groups[27]) == semantic["df_block_gemms"], "DF block census")
    require(
        len(groups[23]) == semantic["df_cuda_transform_calls"] == 1, "DF pack census"
    )
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
    require(
        [phase["words"][6] for phase in groups[24]] == [8 * count for count in copies],
        "independent DF publication bytes",
    )
    require(
        [phase["words"][6] for phase in groups[25]] == copies,
        "DF finite-validation inventory",
    )
    require(
        [phase["words"][6] for phase in groups[26]] == copies,
        "DF host allocation-preparation inventory",
    )
    require(
        len(groups[14]) == semantic["ccsd_evaluations"]
        and len(groups[15]) == semantic["ccsd_replay_evaluations"],
        "primal/replay invocation census",
    )
    expected_tiles = []
    for site, batch in ((14, cc_plan[6]), (15, cc_plan[7])):
        for parent in groups[site]:
            require(
                parent["words"][7] == cc_id and parent["words"][3] == cc_plan[3],
                "Q invocation is outside admitted native solver",
            )
            children = [p for p in groups[4] if p["words"][3] == parent["words"][2]]
            require(
                (q + batch - 1) // batch == len(children),
                "Q tile cardinality differs from bounded source inventory",
            )
            expected = [
                (offset, min(batch, q - offset)) for offset in range(0, q, batch)
            ]
            require(
                [(p["words"][5], p["words"][6]) for p in children] == expected,
                "independent Q domain/order census",
            )
            for tile in children:
                tid = tile["words"][2]
                leaves = {
                    kind: [p for p in groups[kind] if p["words"][3] == tid]
                    for kind in (5, 6, 7)
                }
                require(
                    all(len(items) == 1 for items in leaves.values()),
                    "Q host/production/accumulation inventory",
                )
                producer = leaves[6][0]
                consumer = leaves[7][0]
                require(
                    all(
                        p["words"][5:8] == tile["words"][5:8]
                        for items in leaves.values()
                        for p in items
                    ),
                    "Q payload/plan mismatch",
                )
                require(
                    producer["words"][7] == cc_id
                    and consumer["words"][8] == producer["words"][2],
                    "actual Q producer/consumer source edge",
                )
                expected_tiles.append(tile)
    consumed = {tile["words"][2] for tile in expected_tiles}
    require(
        consumed == {tile["words"][2] for tile in groups[4]},
        "extra/orphan Q subtree outside independent invocation domain",
    )
    require(
        all(
            leaf["words"][3] in consumed for site in (5, 6, 7) for leaf in groups[site]
        ),
        "extra/orphan Q leaf outside actual tile domain",
    )
    require(
        len(expected_tiles) == semantic["ccsd_q_tiles"]
        and sum(t["words"][6] for t in expected_tiles) == semantic["ccsd_q_slices"],
        "independent existing Q work totals",
    )
    model = triples_expected(o, t_plan[7], t_plan[8] > 0, t_plan[7] == 3)
    require(
        [(p["words"][5], p["words"][6]) for p in groups[28]] == model["tiles"],
        "independent occupied-tile census",
    )
    require(
        len(groups[10]) == len(model["panels"]) == semantic["triples_panel_calls"],
        "independent panel LRU census",
    )

    def tile_of(phase: dict[str, Any]) -> int:
        parent = phases[phase["words"][3]]
        require(
            parent["words"][4] == 28, "triples operation outside physical occupied tile"
        )
        return parent["words"][5]

    require(
        [(tile_of(p), p["words"][5], p["words"][6]) for p in groups[10]]
        == model["panels"],
        "panel logical identity/slot/order mismatch",
    )
    require(
        len(groups[12]) == semantic["triples_epilogue_kernels"] == len(model["tiles"]),
        "triples epilogue census",
    )
    require(
        len(groups[30])
        == semantic["triples_reduction_kernels"]
        == len(model["tiles"]) + 1,
        "triples reduction census",
    )
    require(
        semantic["triples_moment_products"] == 2 * o**3 == model["logical_w_products"],
        "logical W product census",
    )
    # Declined pairs are explicit, and cannot be silently counted as production.
    require(
        all(p["status"] == 0 for p in groups[29]),
        "paired decline outside currently admitted frozen route",
    )
    require(
        len(groups[29]) == len(model["pairs"])
        and len(groups[11]) == len(model["singles"]),
        "paired/single seed census",
    )
    expected_pairs = [(tile, a * 6 + b) for tile, a, b, _ in model["pairs"]]
    expected_singles = [(tile, index) for tile, index, _ in model["singles"]]
    require(
        [(tile_of(p), p["words"][5]) for p in groups[29]] == expected_pairs,
        "paired seed/order mismatch",
    )
    require(
        [(tile_of(p), p["words"][5]) for p in groups[11]] == expected_singles,
        "single seed/order mismatch",
    )
    panel_slots = {}
    operations = {}
    for item in model["pairs"]:
        operations[(29, item[0], item[1] * 6 + item[2])] = item[3]
    for item in model["singles"]:
        operations[(11, item[0], item[1])] = item[2]
    for identity in inventory["begin_order"]:
        phase = phases[identity]
        row = phase["words"]
        site = row[4]
        if site == 10:
            panel_slots[row[6]] = identity
        elif site in (11, 29):
            slot = operations[(site, tile_of(phase), row[5])]
            require(
                row[7] == t_id and row[8] > 0 and row[8] == panel_slots.get(slot),
                "missing/late/stale panel producer identity",
            )
    require(
        semantic["triples_fp64_gemms"]
        == model["vendor_gemm_calls_if_all_pairs_accepted"],
        "logical products/vendor calls are separate checked layers",
    )
    return {
        "source_coverage": "PASS_FINITE_STRICT_FP64_DOMAIN",
        "q_tiles": len(expected_tiles),
        "triples_tiles": len(model["tiles"]),
        "logical_w_products": model["logical_w_products"],
        "vendor_gemm_calls": model["vendor_gemm_calls_if_all_pairs_accepted"],
        "device_kernel_count": "separate measured CUPTI family, never inferred from logical/vendor totals",
    }
