"""Independent bounded canonical layouts and Libcint inputs for an LR value sink.

This is fixture preparation, never a native plan or production allocation owner.
The compiler supplies only existing radial/component normalization and headers;
Libcint supplies all reference integrals. No public runtime or GPU is imported.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REJECTED = 0xFFFF
SIZE_MAX = (1 << 64) - 1
STYLES = ((False, 0.0), (True, 0.0), (False, 0.125), (True, 0.125), (True, 2.0))


def admit_request(
    angular: tuple[int, ...],
    *,
    precision: str = "fp64",
    operator: str = "long_range",
    derivative_order: int = 0,
    omega: float = 0.3,
    capacity: int = 192,
) -> int:
    """Refuse outside the fixture envelope before constructing publication metadata."""
    if (
        len(angular) != 4
        or any(type(x) is not int or not 0 <= x <= 2 for x in angular)
        or not 3 <= sum(angular) <= 5
        or precision != "fp64"
        or operator != "long_range"
        or derivative_order != 0
        or not math.isfinite(omega)
        or omega < 0
    ):
        raise ValueError("strict FP64, derivative-free SPD LR orders 3..5 required")
    slots = {3: 1, 4: 3, 5: 6}[sum(angular)]
    if capacity < 32 * slots:
        raise ValueError("whole-shell component-map capacity refused")
    return 32 * slots


def checked_extent(first: int, second: int, same: bool, limit: int = SIZE_MAX) -> int:
    """Check products in the same finite address space as the standalone reader."""
    if min(first, second, limit) < 0 or max(first, second) > limit:
        raise ValueError("invalid canonical extent")
    count = first * (first + 1) // 2 if same else first * second
    if (same and first != second) or count > limit:
        raise ValueError("canonical extent overflow/capacity refused")
    return count


def pair(first: int, second: int) -> tuple[int, int]:
    return max(first, second), min(first, second)


def quartet_key(indices: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """An eightfold-orbit identity independent of either source's component ordinal."""
    return tuple(sorted((pair(*indices[:2]), pair(*indices[2:]))))


@dataclass(frozen=True)
class CanonicalLayout:
    first: tuple[tuple[int, int], ...]
    second: tuple[tuple[int, int], ...]
    prefix: tuple[int, ...] | None
    same: bool
    bounds: dict[tuple[int, int], float]
    threshold: float

    @property
    def count(self) -> int:
        if self.prefix is not None:
            return self.prefix[-1] if self.prefix else 0
        return checked_extent(len(self.first), len(self.second), self.same)

    def decode(self, offset: int) -> tuple[int, int, int, int]:
        """Incumbent row decoding: first inclusive prefix strictly greater than slot."""
        if not 0 <= offset < self.count:
            raise ValueError("canonical slot outside layout")
        if self.prefix is not None:
            first = bisect.bisect_right(self.prefix, offset)
            second = offset - (self.prefix[first - 1] if first else 0)
        elif self.same:
            first = (math.isqrt(8 * offset + 1) - 1) // 2
            second = offset - first * (first + 1) // 2
        else:
            first, second = divmod(offset, len(self.second))
        return (*self.first[first], *self.second[second])

    def slot(self, indices: tuple[int, ...]) -> int | None:
        """Candidate inverse ranks; triangle ownership is sorted rank, never AO rank."""
        left, right = pair(*indices[:2]), pair(*indices[2:])
        first_rank = {p: rank for rank, p in enumerate(self.first)}
        second_rank = {p: rank for rank, p in enumerate(self.second)}
        if self.same:
            a, b = first_rank[left], first_rank[right]
            a, b = max(a, b), min(a, b)
        elif left in first_rank and right in second_rank:
            a, b = first_rank[left], second_rank[right]
        else:
            a, b = first_rank[right], second_rank[left]
        # Keep the direct product comparison, including exact equality and zeros.
        if self.bounds[left] * self.bounds[right] < self.threshold:
            return None
        if self.prefix is not None:
            previous = self.prefix[a - 1] if a else 0
            if b >= self.prefix[a] - previous:
                raise ValueError("rank outside the independently screened row")
            return previous + b
        return a * (a + 1) // 2 + b if self.same else a * len(self.second) + b


def canonical_layout(
    first: list[tuple[int, int]],
    second: list[tuple[int, int]],
    bounds: dict[tuple[int, int], float],
    threshold: float,
    *,
    same: bool,
    compact: bool,
    capacity: int = 192,
) -> CanonicalLayout:
    """Sort tied bounds deterministically, then count each row by product predicate."""
    if (
        not math.isfinite(threshold)
        or threshold < 0
        or any(not math.isfinite(v) or v < 0 for v in bounds.values())
    ):
        raise ValueError("finite nonnegative Schwarz inputs required")
    first = sorted(set(first), key=lambda p: (-bounds[p], p))
    second = first if same else sorted(set(second), key=lambda p: (-bounds[p], p))
    extent = checked_extent(len(first), len(second), same)
    if extent > capacity:
        raise ValueError("bounded fixture destination capacity refused")
    prefix = []
    total = 0
    for rank, left in enumerate(first):
        row = second[: rank + 1] if same else second
        total += sum(bounds[left] * bounds[right] >= threshold for right in row)
        prefix.append(total)
    return CanonicalLayout(
        tuple(first),
        tuple(second),
        tuple(prefix) if compact else None,
        same,
        bounds,
        threshold,
    )


def shell_pairs(offsets: list[int], first: int, second: int) -> list[tuple[int, int]]:
    """Independent Cartesian shell component enumeration, including diagonal AOs."""
    return [
        pair(i, j)
        for i in range(offsets[first], offsets[first + 1])
        for j in range(offsets[second], offsets[second + 1])
        if first != second or i >= j
    ]


def task_components(
    offsets: list[int],
    first: tuple[int, int],
    second: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    left, right = shell_pairs(offsets, *first), shell_pairs(offsets, *second)
    return [
        (*a, *b)
        for ai, a in enumerate(left)
        for bi, b in enumerate(right)
        if first != second or bi <= ai
    ]


def publication_maps(
    angular: tuple[int, ...],
    offsets: list[int],
    owners: list[tuple[int, int]],
    layout: CanonicalLayout,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Prove that the component and incumbent slot inventories are inverses."""
    capacity = admit_request(angular)
    tasks = (
        [(a, b) for ai, a in enumerate(owners) for b in owners[: ai + 1]]
        if layout.same
        else [(owners[0], owners[1])]
    )
    visits = [0] * layout.count
    result = []
    for first, second in tasks:
        components = task_components(offsets, first, second)
        if len(components) > capacity:
            raise ValueError("whole-shell map overflow")
        mapping = [REJECTED] * capacity
        for ordinal, component in enumerate(components):
            target = layout.slot(component)
            if target is not None:
                if quartet_key(component) != quartet_key(layout.decode(target)):
                    raise ValueError("canonical source and component inverse disagree")
                mapping[ordinal] = target
                visits[target] += 1
        result.append(
            {
                "shell_pairs": [list(first), list(second)],
                "components": [list(c) for c in components],
                "map": mapping,
            }
        )
    expected = [
        int(
            layout.bounds[pair(*layout.decode(i)[:2])]
            * layout.bounds[pair(*layout.decode(i)[2:])]
            >= layout.threshold
        )
        for i in range(layout.count)
    ]
    if visits != expected:
        raise ValueError("canonical inventory has duplicates or holes")
    return result, visits


def geometry_inputs(angular: tuple[int, ...], variant: str) -> dict[str, Any]:
    coordinates = [
        [0.13, -0.31, 0.24],
        [-0.43, 0.27, 0.51],
        [0.68, -0.14, -0.22],
        [-0.21, 0.48, -0.63],
    ]
    if variant == "coincident":
        coordinates = [coordinates[0]] * 4
    elif variant == "reversed_centers":
        coordinates = coordinates[::-1]
    elif variant != "separated":
        raise ValueError("unknown geometry variant")
    return {
        "atomic_numbers": [1] * 4,
        "coordinates": coordinates,
        "shells": [
            {
                "atom_index": i,
                "angular_momentum": l,
                "primitives": [
                    [0.57 + 0.13 * i + 0.41 * p, 0.83 if p == 0 else -0.17 / p]
                    for p in range((1, 2, 2, 3)[i])
                ],
            }
            for i, l in enumerate(angular)
        ],
        "basis_representation": "cartesian",
        "charge": 0,
        "multiplicity": 1,
    }


def native_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    from generativeqc_compiler.integral.shell_spec import cartesian_components
    from generativeqc_compiler.integral.weight_pullback import (
        normalized_radial_primitives,
    )

    angular = [s["angular_momentum"] for s in inputs["shells"]]
    offsets, primitive_offsets = [0], [0]
    powers, coefficients, exponents, weights = [], [], [], []
    for l, shell in zip(angular, inputs["shells"], strict=True):
        labels = cartesian_components(l)
        offsets.append(offsets[-1] + len(labels))
        for label in labels:
            component = [label.count(axis) for axis in "xyz"]
            powers.extend(component)
            coefficients.append(
                1
                / math.sqrt(math.prod(math.prod(range(1, 2 * x, 2)) for x in component))
            )
        for exponent, coefficient in normalized_radial_primitives(
            l, shell["primitives"]
        ):
            exponents.append(exponent)
            weights.append(coefficient)
        primitive_offsets.append(len(exponents))
    return {
        "angular": angular,
        "offsets": offsets,
        "primitive_offsets": primitive_offsets,
        "powers": powers,
        "coefficients": coefficients,
        "exponents": exponents,
        "weights": weights,
        "positions": list(itertools.chain.from_iterable(inputs["coordinates"])),
    }


def raw_exchange(values: list[float], layout: CanonicalLayout, density: Any) -> Any:
    """Independent unique eightfold orbit; raw positive K, with signed density."""
    import numpy as np

    from tools.generativeqc_validation.f_shell_numerics import eri_orbit

    output = np.zeros_like(density)
    for slot, value in enumerate(values):
        indices = layout.decode(slot)
        if layout.slot(indices) is None:
            continue
        for a, b, c, d in eri_orbit(indices):
            output[a, c] += value * density[b, d]
    return output


def reference_identity() -> dict[str, str]:
    """Bind the pinned wheel's actual Libcint library on Linux or macOS."""
    import numpy as np
    import pyscf
    from pyscf.gto import moleintor

    if pyscf.__version__ != "2.14.0":
        raise ValueError("reference requires pinned PySCF 2.14.0")
    libraries = {
        p.resolve()
        for pattern in ("libcint*.so*", "libcint*.dylib")
        for p in (Path(pyscf.__file__).parent / "lib/deps/lib").glob(pattern)
    }
    if len(libraries) != 1:
        raise ValueError("cannot uniquely authenticate the pinned Libcint library")
    libcint = libraries.pop()
    cgto = Path(moleintor.libcgto._name).resolve()
    return {
        "pyscf": pyscf.__version__,
        "numpy": np.__version__,
        "libcint": str(libcint),
        "libcint_sha256": hashlib.sha256(libcint.read_bytes()).hexdigest(),
        "cgto": str(cgto),
        "cgto_sha256": hashlib.sha256(cgto.read_bytes()).hexdigest(),
    }


def prepare(output: Path) -> dict[str, Any]:
    """Write data plus generated headers; no compile, device probing or execution."""
    import numpy as np
    import pyscf
    from generativeqc_compiler.integral.direct_cartesian_contraction_cuda import (
        emit_direct_cartesian_contraction_headers,
    )
    from generativeqc_compiler.integral.direct_order2_shell_cuda import (
        emit_direct_order2_shell_header,
    )
    from generativeqc_compiler.integral.direct_pair_cache_cuda import (
        emit_direct_pair_cache_header,
    )
    from generativeqc_compiler.integral.direct_pair_support_cuda import (
        emit_direct_pair_support_headers,
    )
    from generativeqc_compiler.integral.direct_recurrence_cuda import (
        emit_direct_recurrence_headers,
    )
    from generativeqc_compiler.integral.direct_source_contraction_cuda import (
        emit_direct_source_contraction_header,
    )
    from generativeqc_compiler.integral.lowering.fock_accumulation import (
        emit_direct_fock_accumulation_header,
    )

    from tools.generate_validation_references import pyscf_molecule

    identity = reference_identity()  # Authenticate before any oracle work/output.
    pyscf.lib.num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for angular in itertools.product(range(3), repeat=4):
        if not 3 <= sum(angular) <= 5:
            continue
        for variant in ("separated", "coincident", "reversed_centers"):
            inputs = geometry_inputs(angular, variant)
            native = native_inputs(inputs)
            mol, scale, _ = pyscf_molecule(inputs)
            offsets = native["offsets"]
            # Equal angular-pair buckets include both shell pairs and all three
            # shell tasks. This tests triangle ownership in sorted-rank space.
            owner_sets = [("two_pairs", [(1, 0), (3, 2)])]
            if angular == (1, 1, 1, 1):
                owner_sets.append(("same_shell_pair", [(0, 0)]))
            for owner_name, owners in owner_sets:
                same = len(owners) == 1 or sum(angular[:2]) == sum(angular[2:])
                pair_sets = [shell_pairs(offsets, *owner) for owner in owners]
                if same:
                    pair_sets = [
                        list(set(itertools.chain.from_iterable(pair_sets)))
                    ] * 2
                bounds = {
                    p: (0.0, 0.125, 0.25, 0.5, 1.0)[(p[0] + 2 * p[1]) % 5]
                    for p in set(pair_sets[0] + pair_sets[1])
                }
                n = offsets[-1]
                density = np.array(
                    [
                        [math.sin(0.31 * i + 0.17 * j) / n for j in range(n)]
                        for i in range(n)
                    ]
                )
                for omega in (0.0, 0.03, 0.3, 3.0):
                    reference_cache = {}
                    for compact, threshold in STYLES:
                        layout = canonical_layout(
                            *pair_sets, bounds, threshold, same=same, compact=compact
                        )
                        tasks, visits = publication_maps(
                            angular, offsets, owners, layout
                        )
                        expected = []
                        with mol.with_range_coulomb(omega):
                            for slot in range(layout.count):
                                indices = layout.decode(slot)
                                shells = tuple(
                                    bisect.bisect_right(offsets, i) - 1 for i in indices
                                )
                                if shells not in reference_cache:
                                    # PySCF omega=0 means Full. LR(0) is analytically
                                    # zero; do not silently ask Libcint for full ERIs.
                                    reference_cache[shells] = (
                                        np.zeros(
                                            tuple(
                                                offsets[s + 1] - offsets[s]
                                                for s in shells
                                            )
                                        )
                                        if omega == 0
                                        else mol.intor_by_shell("int2e", shells)
                                    )
                                local = tuple(
                                    i - offsets[s]
                                    for i, s in zip(indices, shells, strict=True)
                                )
                                expected.append(
                                    float(
                                        reference_cache[shells][local]
                                        * math.prod(scale[list(indices)])
                                    )
                                )
                        if not np.isfinite(expected).all():
                            raise ValueError("nonfinite independent LR reference")
                        cases.append(
                            {
                                "name": f"{''.join(map(str, angular))}/{owner_name}/{variant}/{omega}/{compact}/{threshold}",
                                **native,
                                "omega": omega,
                                "threshold": threshold,
                                "tasks": tasks,
                                "bounds": [[*p, b] for p, b in sorted(bounds.items())],
                                "canonical": [
                                    list(layout.decode(i)) for i in range(layout.count)
                                ],
                                "expected": expected,
                                "visits": visits,
                                "density": density.ravel().tolist(),
                                "raw_k": raw_exchange(expected, layout, density)
                                .ravel()
                                .tolist(),
                            }
                        )
    headers = {
        **emit_direct_cartesian_contraction_headers(),
        **emit_direct_pair_support_headers(),
        **emit_direct_recurrence_headers(),
        "generated_direct_pair_cache.cuh": emit_direct_pair_cache_header(),
        "generated_direct_order2_shell.cuh": emit_direct_order2_shell_header(),
        "generated_direct_source_contraction.cuh": emit_direct_source_contraction_header(),
        "generated_direct_fock_accumulation.cuh": emit_direct_fock_accumulation_header(),
    }
    for name, content in headers.items():
        (output / name).write_text(content)
    # JSON is retained for review; the standalone reader consumes this bounded
    # whitespace format so its build has no JSON/runtime-library dependency.
    (output / "fixtures.json").write_text(
        json.dumps(cases, separators=(",", ":"), allow_nan=False)
    )
    with (output / "fixtures.txt").open("w") as stream:
        stream.write(f"GQCLR1 {len(cases)}\n")

        def row(values: Any) -> None:
            stream.write(" ".join(map(str, values)) + "\n")

        for case in cases:
            row(
                [
                    case["name"],
                    *case["angular"],
                    case["omega"],
                    case["threshold"],
                    len(case["expected"]),
                    len(case["tasks"]),
                ]
            )
            for key in (
                "offsets",
                "primitive_offsets",
                "positions",
                "powers",
                "coefficients",
                "exponents",
                "weights",
                "density",
                "expected",
                "visits",
                "raw_k",
            ):
                row([len(case[key]), *case[key]])
            row([len(case["bounds"]), *itertools.chain.from_iterable(case["bounds"])])
            row(
                [
                    len(case["canonical"]),
                    *itertools.chain.from_iterable(case["canonical"]),
                ]
            )
            for task in case["tasks"]:
                row(
                    [
                        *itertools.chain.from_iterable(task["shell_pairs"]),
                        len(task["components"]),
                        len(task["map"]),
                        *task["map"],
                    ]
                )
    hashes = {
        p.name: {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in sorted(output.iterdir())
    }
    receipt = {
        "schema": "generativeqc.materialized-lr-fixture.v1",
        "cases": len(cases),
        "compositions": 51,
        "orders": [3, 4, 5],
        **identity,
        "emitted_gate": {"atol": 2e-12, "rtol": 2e-10},
        "component_gate": {"atol": 3e-11, "rtol": 3e-11},
        "device_qualification": "PENDING; no build or GPU execution",
        "files": hashes,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()
