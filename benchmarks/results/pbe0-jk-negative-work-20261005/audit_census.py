"""Audit intrusive work observations against frozen independent energy results.

The counter probe is a separate energy-only solve. Its observations cannot fill
missing work fields in the earlier clean energy/force timing campaign.
"""

import gzip
import hashlib
import json
import math
from pathlib import Path

from tools.generativeqc_validation.publication import validate_publication

ROOT = Path(__file__).resolve().parent
REF = ROOT.parent / "pbe0-composed-baseline-20261005"
SOURCE = "cb81f4c481e5cadd85999a7ff791f93a95a61f9fcf7bb5917ba3c3bfffa849b8"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_input(path: Path, geometry: list, basis: dict) -> int:
    """Check every atom and unnormalized primitive consumed by the C++ probe."""
    tokens = iter(path.read_text().split())
    atoms, shells = int(next(tokens)), int(next(tokens))
    require(atoms == len(geometry), "atom inventory changed")
    expected = []
    for atom, (symbol, position) in enumerate(geometry):
        z = {"H": 1, "O": 8}[symbol]
        require(int(next(tokens)) == z, "element changed")
        require([float(next(tokens)) for _ in range(3)] == position, "geometry changed")
        for shell in basis[z]["shells"]:
            for coefficients in shell["coefficients"]:
                expected.append(
                    (atom, shell["angular_momentum"], shell["exponents"], coefficients)
                )
    require(shells == len(expected), "shell inventory changed")
    aos = 0
    for atom, angular, exponents, coefficients in expected:
        require(
            [int(next(tokens)) for _ in range(3)] == [atom, angular, len(exponents)],
            "shell changed",
        )
        for exponent, coefficient in zip(exponents, coefficients, strict=True):
            require(float(next(tokens)) == float(exponent), "exponent changed")
            require(float(next(tokens)) == float(coefficient), "coefficient changed")
        aos += 2 * angular + 1
    require(next(tokens, None) is None, "extra input data")
    return aos


def audit() -> dict:
    """Authenticate the permanent oracle before checking independent observations."""
    manifest = json.loads((REF / "publication.json").read_text())
    validate_publication(
        manifest,
        {
            entry["path"]: (REF / entry["path"]).read_bytes()
            for entry in manifest["files"]
        },
    )
    artifacts = ROOT / "census"
    provenance = json.loads((ROOT / "provenance.json").read_text())["census"]
    require(provenance["job.exit"].strip() == "0", "job failed")
    require(
        provenance["source-check.txt"].split()
        == ["SOURCE_MATCH", "7e5343ff567deae29e4feb6232cdb8620e35ec4f", SOURCE],
        "source identity changed",
    )
    for line in (artifacts / "probe.sha256").read_text().splitlines():
        digest, name = line.split()
        filename = Path(name).name
        # The executable is not a portable source artifact. Its original hash
        # and successful remote check remain receipts, not a new binary audit.
        if filename == "ks-work-probe":
            require(
                digest
                == "37b2d5d683f7bd186087e98f2d1d03da065327e21dd3aabdf9502b693599dcf9",
                "binary receipt changed",
            )
            continue
        if filename.endswith(".cpp"):
            filename += ".txt"
        require(
            hashlib.sha256((artifacts / filename).read_bytes()).hexdigest() == digest,
            "probe/input receipt changed",
        )
    for filename in ("probe-check.txt", "binary-check.txt"):
        require(
            all(line.endswith(": OK") for line in provenance[filename].splitlines()),
            "remote receipt failed",
        )
    basis_record = json.loads(
        (REF.parent / "pbe0-def2-svp-20261003/def2-svp-ho.json").read_text()
    )
    basis = {entry["atomic_number"]: entry for entry in basis_record["elements"]}
    summary = {
        "kind": "separate-intrusive-energy-only-work-census",
        "source_identity": SOURCE,
        "job": 5779,
        "timing_claim": None,
        "force_claim": None,
        "cases": {},
    }
    for atoms in (48, 96):
        reference = json.loads(
            gzip.decompress((REF / f"reference-{atoms}.json.gz").read_bytes())
        )
        require(
            reference["protocol"]["basis_identity"] == basis_record["checksum"],
            "basis identity mismatch",
        )
        for geometry, positions in enumerate(reference["protocol"]["geometries_bohr"]):
            require(
                check_input(
                    artifacts / f"input-{atoms}-{geometry}.txt", positions, basis
                )
                == atoms * 8,
                "AO inventory changed",
            )
        rows = [
            json.loads(line)
            for line in gzip.decompress(
                (artifacts / f"work-{atoms}.jsonl.gz").read_bytes()
            )
            .decode()
            .splitlines()
        ]
        remaining = iter(rows)
        errors, residuals, geometries = [], [], []
        for geometry in (0, 1):
            identity = next(remaining)
            require(
                identity["kind"] == "identity" and identity["geometry"] == geometry,
                "identity order changed",
            )
            require(
                identity["source_identity"] == SOURCE and identity["job"] == "5779",
                "observation identity mismatch",
            )
            require(
                identity["atoms"] == atoms and identity["aos"] == atoms * 8,
                "system size changed",
            )
            require(
                identity["covered_mask"] == identity["present_mask"] == (1 << 21) - 1,
                "incomplete streaming coverage",
            )
            require(
                identity["observer_device_bytes"] == 880, "counter allocation changed"
            )
            builds, admitted = [], []
            for replay in range(6):
                steps = []
                while (row := next(remaining))["kind"] == "iteration":
                    require(
                        row["geometry"] == geometry and row["replay"] == replay,
                        "iteration order changed",
                    )
                    require(
                        row["iteration"] == len(steps) + 1 and row["failed"] is False,
                        "failed/nonconsecutive observation",
                    )
                    require(
                        row["build"] == "strict-full-density", "build policy changed"
                    )
                    for channel in ("j_admitted", "k_admitted"):
                        counts = row[channel]
                        require(
                            len(counts) == 55
                            and all(type(n) is int and n >= 0 for n in counts),
                            "invalid count array",
                        )
                        require(
                            all(n > 0 for n in counts[:21]) and not any(counts[21:]),
                            "count outside present class mask",
                        )
                    steps.append(row)
                require(
                    row["kind"] == "result"
                    and row["geometry"] == geometry
                    and row["replay"] == replay,
                    "result order changed",
                )
                require(
                    row["converged"] is True and row["fock_builds"] == len(steps) > 0,
                    "owner/observed builds mismatch",
                )
                require(
                    row["xc_evaluations"] == len(steps)
                    and row["final_residual_audits"] == 1,
                    "final audit/XC count mismatch",
                )
                require(
                    0 < row["ao_point_square_sum"] < row["ao_dense_point_square_sum"],
                    "sparse AO work absent",
                )
                require(
                    all(
                        math.isfinite(row[key])
                        for key in ("energy", "energy_change", "physical_residual_rms")
                    ),
                    "nonfinite result",
                )
                require(
                    0 <= row["physical_residual_rms"] <= 1e-10,
                    "strict physical residual failed",
                )
                residuals.append(row["physical_residual_rms"])
                for oracle in reference["records"]:
                    if oracle["geometry"] == geometry:
                        require(
                            oracle["converged"]
                            and oracle["status"] == 0
                            and oracle["gate"],
                            "reference failed",
                        )
                        error = abs(row["energy"] - oracle["energy"])
                        require(
                            error <= reference["protocol"]["energy_gate"],
                            "independent energy gate failed",
                        )
                        errors.append(error)
                builds.append(row["fock_builds"])
                admitted.append(
                    {
                        key: [sum(step[key]) for step in steps]
                        for key in ("j_admitted", "k_admitted")
                    }
                )
            geometries.append(
                {
                    "geometry": geometry,
                    "fock_builds": builds,
                    "admitted_per_build": admitted,
                }
            )
        require(
            next(remaining, None) is None and len(errors) == 72,
            "sample inventory mismatch",
        )
        summary["cases"][str(atoms)] = {
            "independent_energy_pairings": len(errors),
            "max_energy_error": max(errors),
            "max_physical_residual_rms": max(residuals),
            "total_fock_builds": sum(sum(g["fock_builds"]) for g in geometries),
            "geometries": geometries,
        }
    return summary


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
