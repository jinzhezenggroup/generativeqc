"""Nested-water RHF/UHF/moved cases for the #1598 incremental Direct-J/K gate.

All cases reuse the audited comparator's nested def2-SVP spherical geometry
prefixes. The UHF population ionizes one water molecule without moving any
atom, so its AO count matches the RHF population exactly. Moved cases apply a
deterministic per-atom displacement to the same prefix; the displacement
changes every intramolecular distance without altering the element or basis
identity.
"""

from dataclasses import replace

from benchmarks import compare_gpu4pyscf_batch as comparator
from benchmarks._cases import benchmark_cases

_MOVED_DISPLACEMENT_BOHR = 0.06


def _displaced(atoms: tuple) -> tuple:
    """Displace atom i along axis i%3 by +-0.06 Bohr depending on index parity."""

    moved = []
    for index, (element, position) in enumerate(atoms):
        axis = index % 3
        sign = 1.0 if index % 2 == 0 else -1.0
        shifted = list(position)
        shifted[axis] += sign * _MOVED_DISPLACEMENT_BOHR
        moved.append((element, tuple(shifted)))
    return tuple(moved)


def qualification_cases() -> dict:
    """Return RHF, UHF-cation and moved nested-water cases for 12..96 atoms."""

    base = benchmark_cases()["water-32mer-4s4-def2-svp-spherical"]
    cases: dict = {}
    for atoms in (12, 24, 48, 96):
        prefix = base.atoms[:atoms]
        cases[f"water-{atoms}"] = replace(
            base,
            description=f"nested {atoms}-atom water cluster, def2-SVP spherical",
            atoms=prefix,
            expected_ao_count=atoms * 8,
        )
        cases[f"water-{atoms}-cation-uhf"] = replace(
            base,
            description=(
                f"nested {atoms}-atom ionized water cluster, def2-SVP spherical, "
                "UHF doublet"
            ),
            atoms=prefix,
            charge=1,
            multiplicity=2,
            method="uhf",
            expected_ao_count=atoms * 8,
        )
        cases[f"water-{atoms}-moved"] = replace(
            base,
            description=(
                f"nested {atoms}-atom water cluster, def2-SVP spherical, "
                f"per-atom {_MOVED_DISPLACEMENT_BOHR:g}-Bohr deterministic "
                "displacement"
            ),
            atoms=_displaced(prefix),
            expected_ao_count=atoms * 8,
        )
    return cases


if __name__ == "__main__":
    comparator.benchmark_cases = qualification_cases
    comparator.main()
