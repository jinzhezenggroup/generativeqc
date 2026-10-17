"""Reduce newly executed native arrays against unchanged retained reference data."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import runpy
from collections import Counter
from pathlib import Path

from tools.render_readme_dft import collect_point

SOURCE = "1593d3b2fea9ed3e9232efc49eedede1beb1787b"
ROOT = Path(__file__).resolve().parents[3]
PUBLICATION = ROOT / "benchmarks/results/readme-refresh-20261015"


def receipt(path: Path) -> dict:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> None:
    """Keep historical engines only in their existing publications, never rerun."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--df96-raw", type=Path)
    args = parser.parse_args()
    raw = args.raw.resolve()
    previous = runpy.run_path(
        str(PUBLICATION.parent / "readme-wb97mv-20261014/refresh.py")
    )
    refresh = runpy.run_path(str(PUBLICATION / "refresh.py"))
    campaign = {
        "schema": "generativeqc.readme-native-refresh.v1",
        "source_commit": SOURCE,
        "prerequisites": {
            "2531": "ba0dadbdcdef7af18715b72a2366595719df4bde",
            "2532": "27d3b8558b7db2ae4b7d4051ae9122522379a809",
        },
        "populations": {},
    }
    libraries = set()
    identities = set()
    for mode, name in (("none", "summary.json.xz"), ("cuda", "df-summary.json.xz")):
        campaign["populations"][mode] = {}
        baseline = previous["combined_summary"](name)
        if mode == "cuda":
            baseline["methods"]["wb97m-v"].append(
                {"atoms": 96, "protocol": refresh["df96_protocol"]()}
            )
        for method, points in baseline["methods"].items():
            updates = []
            for historical in points:
                atoms = historical["atoms"]
                directory = raw / "measurements" / mode / method
                if (
                    mode == "cuda"
                    and method == "wb97m-v"
                    and atoms == 96
                    and args.df96_raw
                ):
                    directory = args.df96_raw / "measurements/cuda/wb97m-v"
                if not all(
                    (directory / str(atoms) / filename).exists()
                    for filename in ("native.json", "native.outcome")
                ):
                    raise ValueError(
                        f"unfinished native population: {mode}/{method}/{atoms}"
                    )
                reduced = collect_point(directory, atoms)
                if reduced["protocol"] != historical["protocol"]:
                    raise ValueError(f"protocol drift: {mode}/{method}/{atoms}")
                native = reduced["engines"]["native"]
                for row in native.get("records", []):
                    diagnostic = row.get("native_ks_diagnostic")
                    if isinstance(diagnostic, dict):
                        diagnostic.pop("history", None)
                if native.get("native_build"):
                    libraries.add(native["native_build"]["library_sha256"])
                    identities.add(native["native_build"]["probe"]["source_identity"])
                update = {
                    "atoms": atoms,
                    "protocol": reduced["protocol"],
                    "native": native,
                }
                if atoms in (3, 12):
                    primer_dir = raw / "priming" / mode / method / str(atoms)
                    primer = json.loads((primer_dir / "native.json").read_bytes())
                    update["cache_priming"] = {
                        "raw_sha256": receipt(primer_dir / "native.json")["sha256"],
                        "outcome": json.loads(
                            (primer_dir / "native.outcome").read_bytes()
                        ),
                        "phase_counts": dict(
                            Counter(row["phase"] for row in primer["records"])
                        ),
                        "complete_seconds": [
                            row["complete_seconds"] for row in primer["records"]
                        ],
                        "checked_calls": sum(
                            row.get("gate") is True for row in primer["records"]
                        ),
                        "scheduler": primer["scheduler"],
                        "native_build": primer.get("native_build"),
                        "fresh_measurement_process": True,
                    }
                updates.append(update)
            if updates:
                campaign["populations"][mode][method] = updates
    if len(libraries) != 1 or len(identities) != 1:
        raise ValueError("campaign must have one native source and binary")
    settings = json.loads((PUBLICATION / "build-settings.json").read_bytes())
    expected = settings["native_build"]
    if not (
        settings["source_commit"] == SOURCE
        and libraries == {expected["library_sha256"]}
        and identities == {expected["probe"]["source_identity"]}
    ):
        raise ValueError("raw observations do not belong to the frozen build")
    payload = (
        json.dumps(campaign, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    (PUBLICATION / "native-summary.json.xz").write_bytes(
        lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)
    )
    manifest = {
        "schema": "generativeqc.readme-native-refresh-manifest.v1",
        "source_commit": SOURCE,
        "prerequisites": {
            "2531": "ba0dadbdcdef7af18715b72a2366595719df4bde",
            "2532": "27d3b8558b7db2ae4b7d4051ae9122522379a809",
        },
        "native_library_sha256": libraries.pop(),
        "native_source_identity": identities.pop(),
        "reference_policy": "historical timing/work/geometry observations unchanged; no reference execution",
        "files": {},
        "baseline_files": {},
    }
    for relative in [
        "readme-dft-20261013/summary.json.xz",
        "readme-dft-20261013/df-summary.json.xz",
        "readme-wb97mv-20261014/summary.json.xz",
        "readme-wb97mv-20261014/df-summary.json.xz",
    ]:
        manifest["baseline_files"][relative] = receipt(PUBLICATION.parent / relative)
    for path in PUBLICATION.iterdir():
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.name] = receipt(path)
    (PUBLICATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
