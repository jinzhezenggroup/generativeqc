"""Partition complete DT_D/TQ_D endpoint pairs across production H100 jobs."""

import argparse
import json
from pathlib import Path

from protocol import ROOT, basis, load, sha


def estimated_work(system: dict) -> int:
    # Approximate triples effort in the DZ correction and high-basis MP2 effort.
    # This is a scheduling heuristic and is excluded from measurement statistics.
    from pyscf import gto

    dz, qz = (
        gto.M(
            atom=system["bohr"],
            charge=system["charge"],
            spin=0,
            basis=basis(cardinal),
            unit="Bohr",
            verbose=0,
        )
        for cardinal in (2, 4)
    )
    occupied = dz.nelectron // 2
    return occupied**3 * (dz.nao_nr() - occupied) ** 4 + occupied**2 * qz.nao_nr() ** 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--crbh20", type=int, default=2)
    args = parser.parse_args()
    path = ROOT / "inputs.json"
    systems = load(path)["systems"]
    args.output.mkdir()
    dataset, count = "CRBH20", args.crbh20
    work = {
        key: estimated_work(system)
        for key, system in systems.items()
        if key.startswith(dataset + "/")
    }
    if not 1 <= count <= len(work):
        raise ValueError(f"invalid shard count for {dataset}: {count}")
    shards = [[] for _ in range(count)]
    estimates = [0] * count
    for key in sorted(work, key=lambda key: (-work[key], key)):
        rank = min(range(count), key=estimates.__getitem__)
        shards[rank].append(key)
        estimates[rank] += work[key]
    for rank, (keys, estimate) in enumerate(zip(shards, estimates, strict=True)):
        path_out = args.output / f"{dataset.lower()}-{rank + 1}.json"
        path_out.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "systems": keys,
                    "relative_work_estimate": estimate,
                    "inputs_sha256": sha(path),
                },
                indent=2,
            )
            + "\n"
        )
        print(path_out.name, len(keys), estimate)


if __name__ == "__main__":
    main()
