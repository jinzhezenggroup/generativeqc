"""Build an exact finite shell/component closure before native execution.

This tool emits an inventory, not a runtime compiler or capability assertion.
Consumers must still qualify their physical source, closure and contractions.
Existing entries for unselected families and scalar/AO programs are preserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from math import prod
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.hessian_aot import (
    FAMILIES,
    RADIALS,
    IntegralRequest,
)
from generativeqc_compiler.integral.shell_spec import cartesian_components

from tools.generate_df_kernels import write_if_changed
from tools.generate_hessian_aot import SCHEMA, Inventory, parse_inventory


def complete_inventory(
    inventory: Inventory,
    families: tuple[str, ...],
    max_angular: int,
    *,
    components_per_tile: int = 16,
    radials: tuple[str, ...] = RADIALS,
) -> Inventory:
    """Replace selected families with non-overlapping complete ordered tiles.

    All shell-slot permutations are explicit: no center recovery or symmetry
    convention is silently inferred by the package builder. The finite through-f
    bound matches IntegralRequest; an unsupported bound fails before generation.
    """
    if (
        type(max_angular) is not int
        or not 0 <= max_angular <= 3
        or type(components_per_tile) is not int
        or not 1 <= components_per_tile <= 16
        or not families
        or len(set(families)) != len(families)
        or any(family not in FAMILIES for family in families)
        or not radials
        or len(set(radials)) != len(radials)
        or any(radial not in RADIALS for radial in radials)
    ):
        raise ValueError("invalid finite Hessian AOT closure")
    requests = [item for item in inventory.integrals if item.family not in families]
    for family in families:
        slots = 4 if family == "eri" else 3 if family == "three_center_eri" else 2
        for angular in product(range(max_angular + 1), repeat=slots):
            size = prod(len(cartesian_components(order)) for order in angular)
            for radial in radials if family == "eri" else ("full_range",):
                for begin in range(0, size, components_per_tile):
                    requests.append(
                        IntegralRequest(
                            family,
                            angular,
                            tuple(range(begin, min(begin + components_per_tile, size))),
                            radial,
                        )
                    )
    return Inventory(
        inventory.ao,
        inventory.arrays,
        tuple(sorted(requests, key=lambda item: item.name)),
    )


def inventory_payload(inventory: Inventory) -> dict:
    """Serialize only the public selection schema, without derived selectors."""
    return {
        "schema": SCHEMA,
        "ao": inventory.ao,
        "arrays": list(inventory.arrays),
        "integrals": [
            {
                "family": item.family,
                "angular": list(item.angular),
                "components": list(item.components),
                **({"radial": item.radial} if item.family == "eri" else {}),
            }
            for item in inventory.integrals
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", choices=FAMILIES, nargs="+", required=True)
    parser.add_argument("--max-angular", type=int, required=True)
    parser.add_argument("--components-per-tile", type=int, default=16)
    parser.add_argument("--radials", choices=RADIALS, nargs="+", default=list(RADIALS))
    args = parser.parse_args()
    inventory = complete_inventory(
        parse_inventory(json.loads(args.inventory.read_text())),
        tuple(args.families),
        args.max_angular,
        components_per_tile=args.components_per_tile,
        radials=tuple(args.radials),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(
        args.output, json.dumps(inventory_payload(inventory), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
