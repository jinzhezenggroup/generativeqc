"""Compare complete installed CLI resource commands, including process startup."""

import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

base = Path("/tmp/qc-2227-cli-d4-cpu-install/bin/generativeqc")  # noqa: S108 -- recorded immutable installation
candidate = Path("/tmp/qc-2227-cli-d3-cpu-install/bin/generativeqc")  # noqa: S108 -- recorded immutable installation
geometry = Path("tests/data/h2_bohr.xyz").resolve()
environment = {**os.environ, "PATH": "/nonexistent", "GENERATIVEQC_PROFILE": "off"}
for key in (
    "GENERATIVEQC_LIBRARY",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK",
    "GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK",
):
    environment.pop(key, None)
samples = []
reference = {}
for group in range(8):
    for role in "ABBA" if group % 2 == 0 else "BAAB":
        for method in ("rhf", "pbe-rks", "pbe-uks", "pbe-d4(bj-eeq-atm)-rks", "pbe-d4(bj-eeq-atm)-uks"):
            for batch in (1, 4):
                key = f"{method}/{batch}"
                binary = base if role == "A" else candidate
                command = [
                    str(binary),
                    "resources",
                    str(geometry),
                    "--units",
                    "bohr",
                    "--method",
                    method,
                    "--batch",
                    str(batch),
                ]
                timings = []
                for _ in range(21):
                    started = time.perf_counter()
                    result = subprocess.run(
                        command,
                        env=environment,
                        text=True,
                        capture_output=True,
                        check=True,
                        timeout=30,
                    )
                    timings.append(time.perf_counter() - started)
                    record = json.loads(result.stdout)
                    record.pop("identity")
                    record.pop("source_identity")
                    if key in reference:
                        assert record == reference[key], (role, key)
                    else:
                        reference[key] = record
                samples.append(
                    {"group": group, "role": role, "case": key, "seconds": timings}
                )
ratios = {}
for key in reference:
    times = {
        role: [
            t
            for row in samples
            if row["role"] == role and row["case"] == key
            for t in row["seconds"]
        ]
        for role in ("A", "B")
    }
    ratios[key] = statistics.median(times["B"]) / statistics.median(times["A"])
report = {
    "scope": "complete installed metadata-only CLI commands; no scientific/GPU execution",
    "products": {
        role: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for role, path in [("A", base), ("B", candidate)]
    },
    "controls": {
        "groups": 8,
        "orders": ["ABBA", "BAAB"],
        "samples_per_block": 21,
        "runtime_path": "/nonexistent",
    },
    "calls": sum(len(row["seconds"]) for row in samples),
    "ratios": ratios,
    "reference": reference,
    "samples": samples,
}
Path(".artifacts/cli-d3/cli-samples.json").write_text(
    json.dumps(report, separators=(",", ":")) + "\n"
)
print(
    json.dumps(
        {
            "calls": report["calls"],
            "ratios": ratios,
            "maximum_ratio": max(ratios.values()),
        },
        indent=2,
    )
)
assert max(ratios.values()) <= 1.05
