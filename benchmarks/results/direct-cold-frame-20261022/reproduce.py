"""Check retained errors/timings; optionally requalify the original force arrays."""

import hashlib
import json
import lzma
import statistics
import sys
from pathlib import Path


def main() -> None:
    """Accept an optional full artifact path, whose checksum is retained in Git."""
    root = Path(__file__).resolve().parent
    evidence = json.loads(lzma.decompress((root / "evidence.json.xz").read_bytes()))
    if len(sys.argv) > 1:
        from benchmarks.readme_defaults_qualification import qualify

        raw = Path(sys.argv[1]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == evidence["raw_artifact"]["sha256"]
        full = json.loads(lzma.decompress(raw))
        oracles = json.loads(
            lzma.decompress((root / evidence["reference_oracles"]).read_bytes())
        )
        reuse = json.loads((root / evidence["reference_reuse"]).read_text())
        records = (
            full["records"] | full["ablations"] | {"memcheck": full["memcheck_record"]}
        )
        for name, record in records.items():
            result = qualify([record], oracles, reference_reuse=reuse)[0]
            for key, value in evidence["records"][name]["qualification"].items():
                assert result[key] == value
        print("Original force arrays independently requalified")
    for record in evidence["records"].values():
        q = record["qualification"]
        assert q["status"] == "qualified"
        for error in q["cold"]["errors"] + q["warm"]["errors"]:
            assert all(0 <= error[k] <= v for k, v in record["gate"].items())
        for error in [q["cpu_gpu_spot"], *q["grid_stability"]]:
            assert all(0 <= error[k] <= v / 4 for k, v in record["gate"].items())
    for atoms in (3, 12, 48, 96):
        for variant in ("baseline", "candidate", "gpu4pyscf"):
            samples = [
                r["samples"]
                for name, r in evidence["records"].items()
                if name.startswith(variant + "-") and r["atoms"] == atoms
            ]
            cold = statistics.median(s[0]["seconds"] for s in samples)
            warm = statistics.median(
                statistics.median(x["seconds"] for x in s[1:]) for s in samples
            )
            print(atoms, variant, cold, warm)


if __name__ == "__main__":
    main()
