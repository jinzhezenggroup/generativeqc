"""Build the CRBH20 input plan from immutable benchmark data snapshots."""

import argparse
import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

MOLPRO = "https://raw.githubusercontent.com/molpro/pymolpro/1d45741076c19a0b1f87464a53a256254a662035"
ACCDB = "https://raw.githubusercontent.com/peverati/ACCDB/bd1f68947daab2370edccd4339cc0390dc562c29"
CRBH_GEOMETRIES = f"{MOLPRO}/pymolpro/share/database/Karton_CRBH20.json"
CRBH_DOI = "https://doi.org/10.1016/j.chemphys.2015.07.005"
CRBH_TABLE_2 = (
    198.0,
    213.2,
    202.5,
    208.8,
    199.4,
    183.0,
    141.1,
    211.6,
    189.0,
    184.4,
    195.6,
    218.5,
    207.3,
    219.3,
    204.1,
    173.2,
    144.1,
    191.0,
    192.1,
    184.1,
)
HARTREE_KCAL = 627.5094740631
ANGSTROM_PER_BOHR = 0.529177210903
NUCLEAR_CHARGE = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16}
ANCHORS = {"CRBH20": [1, 11]}


def acquire(url: str) -> tuple[str, str]:
    with urlopen(url, timeout=30) as response:  # noqa: S310
        contents = response.read()
    return contents.decode(), hashlib.sha256(contents).hexdigest()


def geometry(lines: list[str]) -> list:
    result = []
    for line in lines:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid geometry row {line!r}")
        symbol, xyz = fields[0], fields[1:]
        xyz = [float(value) for value in xyz]
        if not all(map(math.isfinite, xyz)):
            raise ValueError(f"nonfinite position {xyz}")
        xyz = [value / ANGSTROM_PER_BOHR for value in xyz]
        result.append([symbol.capitalize(), xyz])
    return result


def crbh20() -> tuple[dict, list]:
    text, digest = acquire(CRBH_GEOMETRIES)
    data = json.loads(text)
    if data["references"]["doi"] != CRBH_DOI or len(data["reactions"]) != 20:
        raise ValueError("unexpected CRBH20 source identity")
    points, rows = {}, []

    def point(task: tuple[str, str, str]) -> tuple[str, dict]:
        key, source_key, xyz_key = task
        molpro = geometry(data["molecules"][source_key]["geometry"].splitlines())
        xyz_url = f"{ACCDB}/Geometries/{xyz_key}_CRBH20.xyz"
        text, xyz_digest = acquire(xyz_url)
        lines = text.strip().splitlines()
        accdb = geometry(lines[2:])
        same_geometry = len(accdb) == len(molpro) and all(
            a == b and all(abs(x - y) < 1e-8 for x, y in zip(left, right, strict=True))
            for (a, left), (b, right) in zip(accdb, molpro, strict=True)
        )
        if (
            int(lines[0]) != len(molpro)
            or lines[1].split() != ["0", "1"]
            or not same_geometry
        ):
            raise ValueError(
                f"CRBH20 geometry or state disagrees between mirrors: {key}"
            )
        return key, {
            "bohr": molpro,
            "charge": 0,
            "multiplicity": 1,
            "source": CRBH_GEOMETRIES,
            "source_key": source_key,
            "source_sha256": digest,
            "crosscheck": xyz_url,
            "crosscheck_sha256": xyz_digest,
        }

    tasks = []
    for index, kj in enumerate(CRBH_TABLE_2, 1):
        stoichiometry = {}
        source_terms = data["reactions"][str(index)]["stoichiometry"]
        if source_terms != {str(2 * index - 1): -1, str(2 * index): 1}:
            raise ValueError(f"unexpected CRBH20 reaction {index}: {source_terms}")
        for source_key, coefficient in source_terms.items():
            role = "react" if coefficient == -1 else "ts"
            key = f"CRBH20/{index:02d}{role}"
            tasks.append((key, source_key, f"{index}{role}"))
            stoichiometry[key] = coefficient
        kcal = kj / 4.184
        if abs(data["reaction_energies"][str(index)] * HARTREE_KCAL - kcal) > 0.02:
            raise ValueError(f"CRBH20 Table 2 and mirror disagree for reaction {index}")
        rows.append(
            {
                "set": "CRBH20",
                "index": index,
                "stoichiometry": stoichiometry,
                "reference_kcal_mol": kcal,
                "reference": {
                    "benchmark": CRBH_DOI,
                    "table": 2,
                    "row": index,
                    "reference_kj_mol": kj,
                    "method": "W1-F12" if index <= 10 else "W1w",
                },
            }
        )
    with ThreadPoolExecutor(max_workers=12) as pool:
        points.update(pool.map(point, tasks))
    return points, rows


def validate(systems: dict, barriers: list, anchors: dict) -> None:
    sizes = Counter(row["set"] for row in barriers)
    if sizes != {"CRBH20": 20}:
        raise ValueError(f"unexpected benchmark sizes: {sizes}")
    expected_systems = {
        f"CRBH20/{index:02d}{role}"
        for index in range(1, 21)
        for role in ("react", "ts")
    }
    if set(systems) != expected_systems:
        raise ValueError("unexpected CRBH20 systems")
    composition = {}
    for key, system in systems.items():
        atoms = Counter(symbol for symbol, _ in system["bohr"])
        electrons = (
            sum(NUCLEAR_CHARGE[s] * n for s, n in atoms.items()) - system["charge"]
        )
        if not atoms or electrons <= 0 or electrons % 2 or system["multiplicity"] != 1:
            raise ValueError(f"unexpected closed-shell state for {key}")
        composition[key] = dict(atoms, charge=system["charge"])
    seen = set()
    for row in barriers:
        identity = row["set"], row["index"]
        expected_terms = {
            f"CRBH20/{row['index']:02d}react": -1,
            f"CRBH20/{row['index']:02d}ts": 1,
        }
        if identity in seen or row["stoichiometry"] != expected_terms:
            raise ValueError(f"unexpected or duplicate entry {identity}")
        seen.add(identity)
        balance: Counter = Counter()
        for key, coefficient in row["stoichiometry"].items():
            for label, count in composition[key].items():
                balance[label] += coefficient * count
        if any(balance.values()):
            raise ValueError(f"unbalanced reaction {identity}: {balance}")
    if seen != {("CRBH20", index) for index in range(1, 21)} or anchors != ANCHORS:
        raise ValueError("unexpected CRBH20 directions or canonical anchors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    systems, barriers = crbh20()
    validate(systems, barriers, ANCHORS)
    with options.output.open("x") as stream:
        json.dump(
            {
                "systems": systems,
                "barriers": barriers,
                "canonical_anchors": ANCHORS,
            },
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    print(f"CRBH20: {len(systems)} endpoints")


if __name__ == "__main__":
    main()
