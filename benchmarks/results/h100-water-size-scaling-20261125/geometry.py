"""Extract five nested neutral water clusters from the ErgoSCF SPC snapshot."""

import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path

SOURCE_URL = (
    "https://www.ergoscf.org/files/molecules/water_clusters_b_005_to_100.tar.gz"
)
ARCHIVE_SHA256 = "f1bfeb2a7a5b83978d7db1709668524a4dda942185c1d0b8b0885e4ff7dc81e6"
MEMBER = "water_clusters_b_005_to_100/035.xyz"
SIZES = (96, 144, 192, 240, 300)
BOHR_ANGSTROM = 0.529177210903


def generate(archive):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("unexpected ErgoSCF archive")
    with tarfile.open(archive) as source:
        raw = source.extractfile(MEMBER).read()
    count, _, *lines = raw.decode("ascii").splitlines()
    if len(lines) != int(count) or len(lines) % 3:
        raise ValueError("invalid source XYZ atom count")
    fields = [line.split() for line in lines]
    atoms = [(row[0], tuple(map(float, row[1:]))) for row in fields]
    waters = [atoms[i : i + 3] for i in range(0, len(atoms), 3)]
    if any([symbol for symbol, _ in water] != ["O", "H", "H"] for water in waters):
        raise ValueError("expected O/H/H ordering")
    oxygens = [water[0][1] for water in waters]
    centroid = tuple(sum(axis) / len(oxygens) for axis in zip(*oxygens))
    center_index = min(
        range(len(oxygens)), key=lambda i: math.dist(centroid, oxygens[i])
    )
    center = oxygens[center_index]
    nearest = sorted(
        range(len(waters)), key=lambda i: (math.dist(center, oxygens[i]), i)
    )

    cases = {}
    for size in SIZES:
        chosen = nearest[: size // 3]
        local_oxygens = [oxygens[i] for i in chosen]
        min_oo = min(
            math.dist(a, b)
            for i, a in enumerate(local_oxygens)
            for b in local_oxygens[:i]
        )
        if min_oo < 2.3:
            raise ValueError("overlapping water molecules")
        for i in chosen:
            if any(not 0.9 < math.dist(oxygens[i], h) < 1.1 for _, h in waters[i][1:]):
                raise ValueError("unexpected O–H geometry or periodic wrapping")
        cases[str(size)] = {
            "source_molecule_indices_zero_based": chosen,
            "oxygen_radius_angstrom": math.dist(center, local_oxygens[-1]),
            "minimum_oxygen_distance_angstrom": min_oo,
            "geometry_bohr": [
                [
                    symbol,
                    [(x - origin) / BOHR_ANGSTROM for x, origin in zip(xyz, center)],
                ]
                for i in chosen
                for symbol, xyz in waters[i]
            ],
        }

    return {
        "archive_url": SOURCE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "archive_member": MEMBER,
        "member_sha256": hashlib.sha256(raw).hexdigest(),
        "center_molecule_index_zero_based": center_index,
        "center_oxygen_angstrom": center,
        "bohr_angstrom": BOHR_ANGSTROM,
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(args.archive), indent=2) + "\n")
