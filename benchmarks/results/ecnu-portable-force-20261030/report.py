"""Report complete portable endpoints and retain independent failed gates."""

import hashlib
import json
import lzma
import subprocess
from pathlib import Path
from statistics import median


def main() -> None:
    """Read the verified Git baseline; never relabel an older measured image."""
    data = json.loads(lzma.decompress(Path(__file__).with_name("observations.json.xz").read_bytes()))
    snapshot = data["baseline_snapshot"]
    entry = next(row for row in snapshot["files"] if row["path"].endswith("observations.json.xz"))
    raw = subprocess.check_output(["git", "show", f"{snapshot['source_revision']}:{entry['path']}"])
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("historical baseline checksum mismatch")
    previous = json.loads(lzma.decompress(raw))
    for device in ("a100", "n2", "n5"):
        populations = [
            previous["cohorts"][device + "/candidate"]["records"],
            data["cohorts"][device + "/seed21"]["records"],
            data["cohorts"].get(device + "/work-guard", {}).get("records", {}),
            previous["cohorts"].get(device + "/gpu4pyscf", {}).get("records", {}),
        ]
        gates = data["qualification"][device + "/seed21"]
        assert len(populations[1]) == len(gates) == 64
        print(device, "case original seed21 work-guard GPU4PySCF (warm seconds)")
        for name in sorted(populations[1]):
            if not name.endswith("-96") and name not in ("hf-direct-3", "hf-direct-12"):
                continue
            values = [
                f"{median(sample[0] for sample in arm[name]['samples'][1:]):.6f}"
                if name in arm else "unmeasured"
                for arm in populations
            ]
            print(name, *values)
        print("Failed seed gates:", [(q["method"], q["mode"], q["atoms"]) for q in gates if q["status"] != "qualified"])


if __name__ == "__main__":
    main()
