"""Build BH76 inputs from the original Minnesota /08 pages and archives."""

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from html import unescape
from pathlib import Path
from typing import Any

BASE = "https://comp.chem.umn.edu/db/dbs/"
# One entry per forward/reverse pair, in the order of the author's REF1 table.
# A dot in the HTML description can denote a bound complex: those reactants
# use the matching complex file, whereas separated fragments are distinct terms.
REACTIONS = {
    "HTBH38": [
        ("TS1", "H HCl", "H2 Cl"),
        ("TS2", "OH H2", "H2O H"),
        ("TS3", "CH3 H2", "CH4 H"),
        ("TS4", "OH CH4", "H2O CH3"),
        ("TS5", "H H2", "H2 H"),
        ("TS6", "OH NH3", "H2O NH2"),
        ("TS7", "HCl CH3", "CH4 Cl"),
        ("TS8", "OH C2H6", "H2O C2H5"),
        ("TS9", "F H2", "HF H"),
        ("TS10", "O CH4", "OH CH3"),
        ("TS11", "H PH3", "H2 PH2"),
        ("TS12", "H OH", "H2 O"),
        ("TS13", "H H2S", "H2 HS"),
        ("TS14", "O HCl", "OH Cl"),
        ("TS15", "CH3 NH2", "CH4 NH"),
        ("TS16", "C2H5 NH2", "C2H6 NH"),
        ("TS17", "NH2 C2H6", "NH3 C2H5"),
        ("TS18", "NH2 CH4", "NH3 CH3"),
        ("TS19", "C5H8", "C5H8"),
    ],
    "NHTBH38": [
        ("TS1", "H N2O", "OH N2"),
        ("TS2", "H HF", "HF H"),
        ("TS3", "H HCl", "HCl H"),
        ("TS4", "H CH3F", "HF CH3"),
        ("TS5", "H F2", "HF F"),
        ("TS6", "CH3 FCl", "CH3F Cl"),
        ("TS7", "F- CH3F", "CH3F F-"),
        ("TS7", "FCH3Fcomp", "FCH3Fcomp"),
        ("TS9", "Cl- CH3Cl", "CH3Cl Cl-"),
        ("TS9", "ClCH3Clcomp", "ClCH3Clcomp"),
        ("TS11", "F- CH3Cl", "CH3F Cl-"),
        ("TS11", "FCH3Clcomp1", "FCH3Clcomp2"),
        ("TS13", "OH- CH3F", "CH3OH F-"),
        ("TS13", "HOCH3Fcomp2", "HOCH3Fcomp1"),
        ("TS15", "H N2", "HN2"),
        ("TS16", "H CO", "HCO"),
        ("TS17", "H C2H4", "CH3CH2"),
        ("TS18", "CH3 C2H4", "CH3CH2CH2"),
        ("TS19", "HCN", "HNC"),
    ],
}


def table_rows(html: str) -> list[tuple[int, str, str, float, float]]:
    """Extract only the numbered rows, preserving the authors' reference column."""
    found = []
    for row in re.findall(
        r"<tr[^>]*>(.*?)</tr>", html, flags=re.DOTALL | re.IGNORECASE
    ):
        cells = [
            unescape(re.sub(r"<[^>]*>", "", cell)).strip()
            for cell in re.findall(
                r"<td[^>]*>(.*?)</td>", row, flags=re.DOTALL | re.IGNORECASE
            )
        ]
        if len(cells) == 5 and cells[0].isdigit():
            index, label, direction, scalar, best = cells
            found.append((int(index), label, direction, float(scalar), float(best)))
    assert [row[0] for row in found] == list(range(1, 39))
    assert [row[2] for row in found] == ["Vf", "Vr"] * 19
    return found


def read_molecule(
    g09: tarfile.TarFile, xyz: tarfile.TarFile, stem: str
) -> dict[str, Any]:
    gaussian = g09.extractfile(stem + ".g09").read().decode()
    xyz_lines = xyz.extractfile(stem + ".xyz").read().decode().splitlines()
    state = re.search(r"^\s*(-?\d+)\s+(\d+)[ \t]*\n", gaussian, re.MULTILINE)
    assert state, stem
    charge, multiplicity = map(int, state.groups())
    # The group also distributes some Gaussian inputs as internal coordinates.
    # Use their Cartesian XYZ release consistently for the geometry.
    atoms = []
    for line in xyz_lines[2:]:
        if line.strip():
            symbol, x, y, z = line.split()
            atoms.append([symbol, [float(x), float(y), float(z)]])
    assert len(atoms) == int(xyz_lines[0]), stem
    return {"charge": charge, "multiplicity": multiplicity, "angstrom": atoms}


def build(directory: Path) -> dict[str, Any]:
    inputs = {"sources": {}, "systems": {}, "barriers": []}
    systems = inputs["systems"]
    for dataset, equations in REACTIONS.items():
        name = dataset.lower()
        html, gz, xyz_gz = (
            directory / f
            for f in (name + ".html", name + ".tar.gz", name + "_xyz.tar.gz")
        )
        for prefix, path in zip(("", "tar/", "tar/"), (html, gz, xyz_gz), strict=True):
            inputs["sources"][BASE + prefix + path.name] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        rows = table_rows(html.read_text())
        with tarfile.open(gz) as gaussian, tarfile.open(xyz_gz) as xyz:
            archive_dir = next(iter(gaussian)).name
            for reaction, (ts, left, right) in enumerate(equations):
                label = rows[2 * reaction][1]
                for offset, fragments in enumerate((left.split(), right.split())):
                    index, _, direction, ref1, ref2 = rows[2 * reaction + offset]
                    # All members of the anionic reactions use the augmented series.
                    family = (
                        "aug" if dataset == "NHTBH38" and 13 <= index <= 28 else "plain"
                    )
                    suffix = "@aug" if family == "aug" else ""
                    keys = [f"{dataset}/{bare}{suffix}" for bare in (ts, *fragments)]
                    for key in keys:
                        if key not in systems:
                            stem = (
                                archive_dir
                                + "/"
                                + key.split("/")[1].removesuffix("@aug")
                            )
                            systems[key] = read_molecule(gaussian, xyz, stem)
                    # Balance both nuclei and charge before accepting the manual map.
                    transition, *endpoints = (systems[key] for key in keys)
                    assert Counter(a[0] for a in transition["angstrom"]) == Counter(
                        a[0] for m in endpoints for a in m["angstrom"]
                    ), keys
                    assert transition["charge"] == sum(m["charge"] for m in endpoints)
                    assert (family == "aug") == (transition["charge"] < 0), keys
                    for key in keys:
                        assert systems[key].setdefault("family", family) == family
                    inputs["barriers"].append(
                        {
                            "set": dataset,
                            "index": index,
                            "reaction": label,
                            "direction": direction,
                            "ref1_kcal_mol": ref1,
                            "ref2_kcal_mol": ref2,
                            "transition": keys[0],
                            "side": keys[1:],
                        }
                    )
    return inputs


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("download_directory", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    data = build(args.download_directory)
    args.output.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    print(len(data["systems"]), "systems;", len(data["barriers"]), "barriers")
