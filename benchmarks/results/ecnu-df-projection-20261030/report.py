"""Display retained complete endpoints without suppressing accuracy failures."""

import json
import lzma
from pathlib import Path
from statistics import median


def main() -> None:
    """Report warm medians; independent qualification is a separate column."""
    path = Path(__file__).with_name("observations.json.xz")
    data = json.loads(lzma.decompress(path.read_bytes()))
    for device in ("a100", "n2", "n5"):
        cohorts = data["cohorts"]
        candidate = cohorts[device + "/candidate"]["records"]
        baseline = cohorts[device + "/baseline"]["records"]
        reference = cohorts.get(device + "/gpu4pyscf", {}).get("records", {})
        gates = {
            f"{q['method']}-{q['mode']}-{q['atoms']}": q
            for q in data["qualification"][device]
        }
        assert len(candidate) == len(gates) == 64
        print(device, "case baseline candidate GPU4PySCF gate (warm median seconds)")
        for name, row in candidate.items():
            assert len(row["samples"]) == 4
            if not name.endswith("-96"):
                continue
            values = []
            for population in (baseline, candidate, reference):
                values.append(
                    f"{median(s[0] for s in population[name]['samples'][1:]):.3f}"
                    if name in population
                    else "unmeasured"
                )
            if name == "hf-direct-96":
                values[0] = "invalid-force"
            print(name, *values, gates[name]["status"])
        failed = [n for n, q in gates.items() if q["status"] != "qualified"]
        print("Failed gates:", ", ".join(failed))


if __name__ == "__main__":
    main()
