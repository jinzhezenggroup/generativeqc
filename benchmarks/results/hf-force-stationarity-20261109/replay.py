"""Replay default-control force acceptance without accepting the failed control."""

from __future__ import annotations

import hashlib
import json
import math
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    """Keep scientific and integrity gates active under optimized Python."""
    if not condition:
        raise ValueError(message)


def load(directory: Path = ROOT) -> tuple[dict, dict[str, str]]:
    """Authenticate every complete original text before decoding measurements."""
    cohort = json.loads((directory / "cohort.json").read_text())
    require(cohort["schema"] == "generativeqc.hf-force-stationarity.v2", "force schema")
    payload = (directory / "sha256.txt").read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == cohort["stream_manifest_sha256"],
        "changed force capsule",
    )
    codec = runpy.run_path(str(ROOT.parent.parent / "_exact_fp64_evidence.py"))
    _, files = codec["load"](directory)
    require(len(files) == cohort["member_count"], "incomplete force evidence")
    historical = json.loads(files["before-format/cohort.json"])
    for name, digest in historical["members"].items():
        require(
            hashlib.sha256(files["before-format/" + name].encode()).hexdigest()
            == digest,
            "changed historical force evidence member: " + name,
        )
    return cohort, files


def observable_error(actual: dict, reference: dict) -> dict:
    """Score full finite force arrays against the unchanged independent gates."""
    require(actual["converged"] and reference["converged"], "unconverged endpoint")
    require(len(actual["forces"]) == len(reference["forces"]) == 48, "force shape")
    differences = []
    for actual_row, reference_row in zip(
        actual["forces"], reference["forces"], strict=True
    ):
        require(len(actual_row) == len(reference_row) == 3, "force row shape")
        for value, target in zip(actual_row, reference_row, strict=True):
            require(math.isfinite(value) and math.isfinite(target), "nonfinite force")
            differences.append(abs(value - target))
    require(
        math.isfinite(actual["energy"]) and math.isfinite(reference["energy"]),
        "nonfinite energy",
    )
    energy = abs(actual["energy"] - reference["energy"])
    maximum_force = max(differences)
    return {
        "energy": energy,
        "maximum_force": maximum_force,
        "passed": energy < 1e-8 and maximum_force < 1e-7,
    }


def replay(directory: Path = ROOT) -> dict:
    """Retain every cold/warm observation and disclose the stricter work cost."""
    cohort, files = load(directory)
    require(
        cohort["gates"] == {"absolute_energy": 1e-8, "maximum_force": 1e-7},
        "changed independent gates",
    )
    reader = runpy.run_path(
        str(ROOT.parent / "a100-shared-rys-xy-20261108" / "replay.py")
    )
    reference_cohort, references = reader["load"]()
    binding = cohort["reference"]
    require(
        reference_cohort["reference_files"]["48"] == binding["reference_member"],
        "changed reference selection",
    )
    reference_text = references[binding["reference_member"]]
    require(
        hashlib.sha256(reference_text.encode()).hexdigest() == binding["sha256"],
        "changed independent reference",
    )
    reference = json.loads(reference_text)
    require(reference["direct_scf"] is False, "reference is not full Fock")
    require(reference["actual_orbital_gradient_norm"] < 1e-10, "reference stationarity")
    summary = {"gates": cohort["gates"], "scope": cohort["scope"], "arms": {}}
    for arm in ("baseline", "candidate-before-format", "candidate"):
        record = json.loads(files[arm + ".json"])
        identity = cohort["arms"][arm]
        for name in ("source_identity", "library_sha256"):
            require(record["native_build"][name] == identity[name], "native " + name)
        require(record["source_tree"] == identity["source_tree"], "source tree")
        require(
            record["scheduler"]["SLURM_JOB_ID"] == identity["slurm_job_id"]
            and record["scheduler"]["CUDA_VISIBLE_DEVICES"] == "0",
            "scheduler identity",
        )
        require(record["geometry_bohr"] == reference["geometry_bohr"], "geometry")
        require(
            record["explicit_controls"] == {}
            and record["environment_controls"] == {}
            and record["xy_control"] is None
            and record["effective_controls"]
            == {
                "energy_tolerance": 1e-9,
                "density_tolerance": 1e-5,
                "screening_tolerance": 1e-12,
            },
            "changed public/default controls",
        )
        samples = record["samples"]
        require(
            [sample["phase"] for sample in samples] == ["cold", "warm", "warm", "warm"],
            "incomplete cold/warm population",
        )
        results = []
        for sample in samples:
            require(
                math.isfinite(sample["seconds"]) and sample["seconds"] > 0,
                "invalid complete endpoint timing",
            )
            results.append(
                {
                    **{
                        name: sample[name]
                        for name in ("phase", "seconds", "iterations", "fock_builds")
                    },
                    **observable_error(sample, reference),
                }
            )
        summary["arms"][arm] = {
            "samples": results,
            "all_passed": all(result["passed"] for result in results),
        }
    require(not summary["arms"]["baseline"]["all_passed"], "lost failed baseline")
    for arm in ("candidate-before-format", "candidate"):
        require(summary["arms"][arm]["all_passed"], arm + " force gate")
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
