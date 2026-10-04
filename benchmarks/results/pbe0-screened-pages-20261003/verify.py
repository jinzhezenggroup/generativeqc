"""Recheck the corrected campaign without relabeling historical timings."""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import sys
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from benchmarks.readme_pbe0 import SCHEMA
from tools.render_omol25_benchmarks import validate


def require(condition: bool, detail: str) -> None:
    """Reject invalid evidence even under ``python -O`` or PYTHONOPTIMIZE.

    These are publication acceptance gates, so disabling Python assertions
    must never remove them or allow an invalid archive to print PASS.
    """
    if not condition:
        raise ValueError(detail)


def main() -> None:
    """Validate all 144 corrected endpoints and the original unchanged gates."""
    directory = Path(__file__).resolve().parent
    storage = json.loads((directory / "storage.json").read_text())
    compressed = (directory / "corrected-campaign.json.xz").read_bytes()
    require(
        hashlib.sha256(compressed).hexdigest() == storage["sha256"],
        "compressed archive SHA-256 mismatch",
    )
    raw = lzma.decompress(compressed)
    require(
        hashlib.sha256(raw).hexdigest() == storage["uncompressed_sha256"],
        "uncompressed archive SHA-256 mismatch",
    )
    members = json.loads(raw)
    require(
        set(members) == set(storage["members"]), "archive member inventory mismatch"
    )
    for name, value in members.items():
        require(
            hashlib.sha256(value.encode()).hexdigest() == storage["members"][name],
            "archive member SHA-256 mismatch",
        )
    summary = json.loads(members["summary.json"])
    qualification = json.loads(members["native-qualification.json"])
    require(qualification["status"] == "passed", "native qualification did not pass")
    require(
        qualification["memcheck_errors"] == qualification["initcheck_errors"] == 0,
        "native sanitizer qualification failed",
    )
    for key in ("source_identity", "library_sha256"):
        require(
            storage[key] == summary[key] == qualification[key],
            "source or library identity mismatch",
        )
    cases = {case["atoms"]: case for case in summary["cases"]}
    require(set(cases) == {3, 6, 12, 24, 48, 96}, "incomplete case inventory")
    checked = 0
    for atoms, case in sorted(cases.items()):
        reference_raw = gzip.decompress(
            (
                directory
                / storage["reference_root"]
                / f"water{atoms}-reference.json.gz"
            ).read_bytes()
        )
        reference = json.loads(reference_raw)
        validate(reference, reference, schema=SCHEMA)
        grids = []
        for variant in ("baseline", "candidate"):
            prefix = f"{variant}/{atoms}/"
            raw_record = members[prefix + "native.json"]
            record = json.loads(raw_record)
            campaign = json.loads(members[prefix + "campaign.json"])
            retained = case["variants"][variant]
            require(
                json.loads(members[prefix + "native.outcome"])["exit_code"] == 0,
                "native endpoint process failed",
            )
            require(
                hashlib.sha256(raw_record.encode()).hexdigest() == retained["sha256"],
                "native record SHA-256 mismatch",
            )
            require(
                record["reference_sha256"]
                == retained["reference_sha256"]
                == hashlib.sha256(reference_raw).hexdigest(),
                "reference record SHA-256 mismatch",
            )
            require(campaign == retained["campaign"], "campaign metadata mismatch")
            require(
                campaign["claim_consumption_barrier"] is True,
                "missing claim-consumption barrier",
            )
            require(
                campaign["bounded_schwarz_schedule"]
                == ("0" if variant == "baseline" else "1"),
                "unexpected bounded scheduling policy",
            )
            for key in ("source_identity", "library_sha256"):
                require(
                    campaign[key] == summary[key],
                    "campaign source or library identity mismatch",
                )
            require(
                record["native_build"]["library_sha256"] == summary["library_sha256"],
                "loaded native library identity mismatch",
            )
            require(
                record["native_build"]["probe"]["source_identity"]
                == summary["source_identity"],
                "loaded native source identity mismatch",
            )
            require(
                record["status"] == "measured" and record["stage"] == "complete",
                "native endpoint did not finish",
            )
            validate(record, reference, schema=SCHEMA)
            rows = record["records"]
            require(len(rows) == 12, "incomplete endpoint records")
            checked += len(rows)
            errors = {"energy": 0.0, "force": 0.0}
            for row in rows:
                require(
                    row["converged"] and row["status"] == 0 and row["gate"],
                    "native endpoint did not pass",
                )
                for oracle in reference["records"]:
                    if row["geometry"] != oracle["geometry"]:
                        continue
                    errors["energy"] = max(
                        errors["energy"], abs(row["energy"] - oracle["energy"])
                    )
                    errors["force"] = max(
                        errors["force"],
                        float(
                            np.max(np.abs(np.asarray(row["forces"]) - oracle["forces"]))
                        ),
                    )
            require(
                errors["energy"] <= 1e-08 and errors["force"] <= 1e-07,
                "independent energy/force acceptance failed",
            )
            require(
                errors == retained["maximum_errors_all_same_geometry_reference_pairs"],
                "retained maximum errors mismatch",
            )
            for phase in ("cold", "warm", "moved", "moved-warm"):
                selected = [row for row in rows if row["phase"] == phase]
                expected = retained["phases"][phase]
                seconds = [row["complete_seconds"] for row in selected]
                require(
                    seconds == expected["seconds"], "complete phase samples mismatch"
                )
                require(
                    median(seconds) == expected["median"],
                    "complete phase median mismatch",
                )
                require(
                    [row["iterations"] for row in selected] == expected["iterations"],
                    "phase iteration counts mismatch",
                )
            grid = rows[0]["native_force_components"]["grid_work_plan"]
            require(
                all(
                    (
                        row["native_force_components"]["grid_work_plan"] == grid
                        for row in rows
                    )
                ),
                "grid work varies within the case",
            )
            require(
                grid["grid_points"] == atoms * 48 * 16 * 32, "grid point count mismatch"
            )
            require(
                grid["grid_pair_visits"]
                == (1 + 2 * grid["grid_points"]) * atoms * (atoms - 1) // 2,
                "grid pair count mismatch",
            )
            grids.append(grid)
        before, after = (
            case["variants"][variant] for variant in ("baseline", "candidate")
        )
        for key in (
            "host",
            "job",
            "cuda_visible_devices",
            "source_identity",
            "library_sha256",
        ):
            require(
                before["campaign"][key] == after["campaign"][key],
                "baseline/candidate allocation or source mismatch",
            )
        require(grids[0] == grids[1], "baseline/candidate grid work mismatch")
        reduction = (
            1 - after["phases"]["warm"]["median"] / before["phases"]["warm"]["median"]
        )
        require(reduction == case["warm_reduction"], "retained warm reduction mismatch")
        print(f"water{atoms}: corrected warm reduction {100 * reduction:.3f}%")
    require(
        checked == summary["native_endpoints_checked"] == 144,
        "incomplete endpoint inventory",
    )
    require(
        storage["historical"]["commit"] == "0b99c6ce298f2726373f1909ad10a37b5acffc43",
        "historical source identity mismatch",
    )
    print("PASS: 144 corrected endpoints; historical pre-fix bytes remain separate")


if __name__ == "__main__":
    main()
