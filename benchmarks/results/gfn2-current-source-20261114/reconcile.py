"""Recompute current-source acceptance from immutable, independently gated rows."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HISTORICAL = ROOT.parent / "gfn2-sharp-cutoff-20261113"
GATE = 5e-7
CASES = ("water-8", "water-32", "water-128")


def load(path: Path) -> dict:
    return json.loads(lzma.decompress(path.read_bytes()))


def reconcile() -> dict:
    provenance = json.loads((ROOT / "provenance.json").read_text())
    for name, identity in provenance["archive_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != identity:
            raise ValueError("archive identity mismatch: " + name)
    oracle = load(HISTORICAL / "matched-oracle.json.xz")
    oracle_identity = hashlib.sha256(
        lzma.decompress((HISTORICAL / "matched-oracle.json.xz").read_bytes())
    ).hexdigest()
    inputs = lzma.decompress((HISTORICAL / "inputs.json.xz").read_bytes())
    input_identity = hashlib.sha256(inputs).hexdigest()
    if (
        oracle["xtb_config"] != {"smooth_cutoff": 0.0}
        or oracle["input_sha256"] != input_identity
    ):
        raise ValueError("independent model/input mismatch")
    references = {(row["case"], row["geometry"]): row for row in oracle["rows"]}
    summary = {
        "source_commit": provenance["source_commit"],
        "gate": GATE,
        "reports": {},
    }
    all_errors = []
    for name, device, properties in (
        ("cpu-force", "cpu", "force"),
        ("cuda-force", "cuda", "force"),
        ("cuda-energy", "cuda", "energy"),
        ("xtbloom-force", "cuda", "force"),
    ):
        report = load(ROOT / (name + ".json.xz"))
        native = name != "xtbloom-force"
        if (
            report["status"] != ("PASS" if native else "measured")
            or report["input_sha256"] != input_identity
        ):
            raise ValueError("incomplete or input-mismatched receipt: " + name)
        bound = provenance["reports"][name]
        if report["library_sha256"] != bound["library_sha256"]:
            raise ValueError("library identity mismatch: " + name)
        if native:
            for key in ("source_manifest_sha256", "driver_sha256"):
                if report[key] != bound[key]:
                    raise ValueError("execution/source identity mismatch: " + name)
            if report["settings"]["fresh_scc_every_call"] is not True:
                raise ValueError("not a FRESH receipt")
            configuration = report.get("xtb_config", {})
            if configuration != oracle["xtb_config"] or type(
                configuration.get("smooth_cutoff")
            ) not in (int, float):
                raise ValueError("unverified oracle model metadata: " + name)
            controls = {
                "max_iterations": 300,
                "history": 64 if device == "cpu" else 8,
                "energy_tolerance": 1e-10,
                "charge_tolerance": 1e-8,
                "temperature_kelvin": 300,
            }
            if any(
                report["settings"].get(key) != value for key, value in controls.items()
            ):
                raise ValueError("scientific control mismatch: " + name)
            manifest_identity = hashlib.sha256(
                lzma.decompress((ROOT / bound["source_manifest_archive"]).read_bytes())
            ).hexdigest()
            if (
                manifest_identity != report["source_manifest_sha256"]
                or report["oracle_sha256"] != oracle_identity
                or report["driver_sha256"]
                != hashlib.sha256((ROOT / "replay.py").read_bytes()).hexdigest()
            ):
                raise ValueError("unbound source manifest, driver or oracle: " + name)
        cases = CASES if native else ("water-128",)
        expected_keys = {
            (case, geometry, repeat)
            for case in cases
            for geometry in (0, 1)
            for repeat in range(6)
        }
        seen = set()
        grouped = {}
        errors = []
        for row in report["rows"]:
            # The historical peer numbers its five warm repeats 0..4.
            repeat = (
                row["repeat"]
                if native or row["phase"] in ("cold", "moved")
                else row["repeat"] + 1
            )
            key = row["case"], row["geometry"], repeat
            if key not in expected_keys or key in seen:
                raise ValueError("unexpected or duplicate protocol row: " + name)
            seen.add(key)
            reference = references[key[:2]]
            expected_phase = (
                ("cold" if key[1] == 0 else "moved")
                if repeat == 0
                else ("warm" if key[1] == 0 else "moved-warm")
            )
            if (
                row["geometry_sha256"] != reference["geometry_sha256"]
                or row["phase"] != expected_phase
                or row["converged"] is not True
                or row["executed_backend"]
                != ("cpu_reference" if device == "cpu" else "cuda")
                or not 0 < row["iterations"] <= 300
            ):
                raise ValueError("geometry/execution/SCC provenance mismatch: " + name)
            energy_error = abs(row["energy"] - reference["energy"])
            force_error = None
            if properties == "force":
                if len(row["forces"]) != len(reference["forces"]) or any(
                    len(atom) != 3 for atom in row["forces"]
                ):
                    raise ValueError("incorrect complete force shape: " + name)
                actual = [value for atom in row["forces"] for value in atom]
                expected = [value for atom in reference["forces"] for value in atom]
                differences = [
                    abs(value - target)
                    for value, target in zip(actual, expected, strict=True)
                ]
                if not all(math.isfinite(value) for value in differences):
                    raise ValueError("nonfinite force: " + name)
                force_error = max(differences)
            elif row["forces"] is not None:
                raise ValueError("unexpected force publication: " + name)
            if (
                not math.isfinite(energy_error)
                or energy_error > GATE
                or (force_error or 0) > GATE
            ):
                raise ValueError("failed unchanged scientific gate: " + name)
            if (
                not math.isfinite(row["complete_seconds"])
                or not row["complete_seconds"] >= row["seconds"] > 0
            ):
                raise ValueError("invalid complete endpoint timing: " + name)
            errors.append((energy_error, force_error or 0))
            grouped.setdefault((key[0], row["phase"]), []).append(row)
        if seen != expected_keys:
            raise ValueError("incomplete original/moved/five-repeat protocol: " + name)
        summary["reports"][name] = {
            "observations": len(seen),
            "maximum_energy_error": max(error[0] for error in errors),
            "maximum_force_error": max(error[1] for error in errors)
            if properties == "force"
            else None,
            "total_scc_iterations": sum(row["iterations"] for row in report["rows"]),
            "timings": [
                {
                    "case": case,
                    "phase": phase,
                    "complete_seconds_median": statistics.median(
                        row["complete_seconds"] for row in rows
                    ),
                    "iterations": sorted({row["iterations"] for row in rows}),
                }
                for (case, phase), rows in grouped.items()
            ],
        }
        all_errors.extend(errors)
    summary.update(
        status="PASS",
        observations=sum(
            report["observations"] for report in summary["reports"].values()
        ),
        maximum_energy_error=max(error[0] for error in all_errors),
        maximum_force_error=max(error[1] for error in all_errors),
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(reconcile(), indent=2, allow_nan=False))
