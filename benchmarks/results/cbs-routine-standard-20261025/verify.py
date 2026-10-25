"""Recompute independent CBS energy, Cartesian-force and relative-energy gates."""

from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path
from statistics import median

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def error(actual: object, expected: object, tolerance: float) -> dict:
    delta = np.asarray(actual) - np.asarray(expected)
    require(bool(np.all(np.isfinite(delta))), "nonfinite observable")
    maximum = float(np.max(np.abs(delta)))
    return {
        "passed": maximum <= tolerance,
        "atol": tolerance,
        "rtol": 0.0,
        "max_absolute_error": maximum,
        "max_scaled_error": maximum / tolerance,
        "rms_error": float(np.sqrt(np.mean(delta * delta))),
        "shape": list(np.shape(delta)) or [1],
    }


def reference_forces(record: dict) -> dict[float, dict[str, np.ndarray]]:
    """Rebuild two full Cartesian gradients from retained independent energies."""
    n = len(record["protocol"]["atoms_bohr"])
    fields = tuple(record["central"]["energies"])
    derivatives = record["derivatives"]
    expected = {
        (h, atom, axis) for h in (1e-4, 3e-5) for atom in range(n) for axis in range(3)
    }
    keys = [(r["step"], r["atom"], r["axis"]) for r in derivatives]
    require(
        set(keys) == expected and len(keys) == len(expected),
        "incomplete Cartesian oracle",
    )
    out = {h: {key: np.empty((n, 3)) for key in fields} for h in (1e-4, 3e-5)}
    for row in derivatives:
        require(len(row["points"]) == 2, "incomplete +/- stencil")
        minus, plus = (p["energies"] for p in row["points"])
        require(set(minus) == set(plus) == set(fields), "oracle component drift")
        for key in fields:
            out[row["step"]][key][row["atom"], row["axis"]] = -(
                plus[key] - minus[key]
            ) / (2 * row["step"])
    return out


def verify(payload: dict) -> dict:
    report = {
        "errors": {},
        "standard_timings": [],
        "work": [],
        "failures": [],
        "relative_energies": {},
    }
    refs, native_cases = {}, {}
    for record in payload["reference"].values():
        for name, case in record["cases"].items():
            key = (case["protocol"]["model"], name)
            require(key not in refs, "duplicate reference population")
            refs[key] = case
    for native in payload["native"].values():
        gates = native["gates"]
        for name, case in native["cases"].items():
            model = case["protocol"]["model"]
            key = (model, name)
            tag = model + "/" + name
            require(key not in native_cases, "duplicate native population")
            native_cases[key] = case
            require(key in refs, "missing independent reference")
            ref = refs[key]
            require(
                case["protocol"] == ref["protocol"],
                "native/reference scientific input mismatch",
            )
            if case["status"] != "completed" or ref["status"] != "completed":
                report["failures"].append(
                    {
                        "case": tag,
                        "native": case["status"],
                        "reference": ref["status"],
                        "error": case.get("error", ref.get("error")),
                    }
                )
                continue
            force = reference_forces(ref)
            # Equal +/- branch switches can cancel in a derivative. Check every
            # component's displaced energy stays in the central neighborhood,
            # in addition to independent two-step derivative stability.
            for index, stencil in enumerate(ref["derivatives"]):
                for sign, point in zip(
                    ("minus", "plus"), stencil["points"], strict=True
                ):
                    report["errors"][f"{tag}/oracle-locality/{index}/{sign}"] = error(
                        [
                            point["energies"][field]
                            for field in ref["central"]["energies"]
                        ],
                        list(ref["central"]["energies"].values()),
                        gates["oracle_locality_energy"],
                    )
            for field in ref["central"]["energies"]:
                report["errors"][tag + "/fd-stability/" + field] = error(
                    force[1e-4][field], force[3e-5][field], gates["fd_stability"]
                )
            samples = case["samples"]
            sample_keys = [(r["policy"], r["repeat"]) for r in samples]
            expected = {
                (p, i)
                for p in native["policies"]
                for i in range(native["samples_per_policy"])
            }
            require(
                len(sample_keys) == len(expected) and set(sample_keys) == expected,
                "incomplete native population",
            )
            for row in samples + [case["energy_only"]]:
                require(
                    row["converged"] and row["executed_backend"] == "cuda",
                    "invalid native endpoint",
                )
                require(
                    len(row["components"]) == 5 and len(row["execution_groups"]) == 2,
                    "incomplete CBS endpoint",
                )
                energy_only = row["forces"] is None
                policy = "standard" if energy_only else row["policy"]
                prefix = (
                    tag
                    + "/"
                    + ("energy-only" if energy_only else f"{policy}/{row['repeat']}")
                )
                require(
                    row["convergence"]["requested_policy"] == policy,
                    "requested profile drift",
                )
                limits = row["convergence"]["limits"]
                require(
                    limits["density_tolerance"] <= 1e-11,
                    "physical reference density cap drift",
                )
                require(
                    limits["ccsd_residual_tolerance"] <= 1e-7,
                    "CC physical residual cap drift",
                )
                require(
                    limits["ccsd_energy_tolerance"]
                    == (1e-12 if policy == "tight" else 1e-8),
                    "CC energy limit drift",
                )
                if policy == "tight":
                    require(
                        limits["ccsd_residual_tolerance"] == 1e-10
                        and row["convergence"]["budget"] is None,
                        "strict escape changed",
                    )
                else:
                    require(
                        row["convergence"]["budget"]
                        == {
                            "energy_abs": 1e-5,
                            "force_max_abs": 1e-4,
                            "force_rms": 3e-5,
                        },
                        "routine allowance drift",
                    )
                report["errors"][prefix + "/energy"] = error(
                    row["energy"],
                    ref["central"]["energies"]["composite"],
                    gates["energy"],
                )
                if not energy_only:
                    report["errors"][prefix + "/force"] = error(
                        row["forces"], force[3e-5]["composite"], gates["force_max"]
                    )
                    rms = np.sqrt(
                        np.mean(
                            (np.asarray(row["forces"]) - force[3e-5]["composite"]) ** 2
                        )
                    )
                    report["errors"][prefix + "/force-rms"] = error(
                        rms, 0, gates["force_rms"]
                    )
                coefficient_energy, coefficient_force = (
                    0.0,
                    np.zeros_like(force[3e-5]["composite"]),
                )
                for component in row["components"]:
                    label = (
                        "high"
                        if any(r.endswith("_high") for r in component["roles"])
                        else "low"
                    )
                    field = label + "/" + component["method"]
                    result = component["result"]
                    report["errors"][prefix + "/component-energy/" + field] = error(
                        result["energy"],
                        ref["central"]["energies"][field],
                        gates["energy"],
                    )
                    coefficient_energy += component["coefficient"] * result["energy"]
                    if not energy_only:
                        report["errors"][prefix + "/component-force/" + field] = error(
                            result["forces"], force[3e-5][field], gates["force_max"]
                        )
                        coefficient_force += component["coefficient"] * np.asarray(
                            result["forces"]
                        )
                    if component["method"] == "df-ccsd(t)":
                        cc, perf = result["correlation"], result["cc_performance"]
                        residual = max(
                            cc["ccsd_replay_singles_residual_max"],
                            cc["ccsd_replay_doubles_residual_max"],
                        )
                        require(
                            residual <= limits["ccsd_residual_tolerance"],
                            "independent CC replay failed",
                        )
                        report["work"].append(
                            {
                                "case": tag,
                                "endpoint": prefix,
                                "policy": policy,
                                "ccsd_iterations": cc["ccsd_iterations"],
                                "residual": residual,
                                "cc_performance": perf,
                            }
                        )
                require(
                    abs(row["energy"] - coefficient_energy) < 1e-11,
                    "energy composition failed",
                )
                if not energy_only:
                    require(
                        bool(
                            np.allclose(
                                row["forces"], coefficient_force, rtol=0, atol=1e-12
                            )
                        ),
                        "force composition failed",
                    )
            routine = [r for r in samples if r["policy"] == "standard"]
            strict = next(r for r in samples if r["policy"] == "tight")
            for row in routine:
                prefix = f"{tag}/strict-control/{row['repeat']}"
                report["errors"][prefix + "/energy"] = error(
                    row["energy"], strict["energy"], gates["energy"]
                )
                report["errors"][prefix + "/force"] = error(
                    row["forces"], strict["forces"], gates["force_max"]
                )
            seconds = [r["complete_seconds"] for r in routine]
            report["standard_timings"].append(
                {"case": tag, "seconds": seconds, "median": median(seconds)}
            )
            report["errors"][tag + "/energy-only-parity"] = error(
                case["energy_only"]["energy"], routine[0]["energy"], 1e-8
            )
    relative = {
        "water-stretch": {"water-stretched": 1, "water": -1},
        "nitrogen-stretch": {"nitrogen-stretched": 1, "nitrogen": -1},
        "water-dimer-binding": {"water-dimer": 1, "water": -2},
        "hydroxide-diffuse-shift": {"hydroxide-diffuse": 1, "hydroxide": -1},
    }
    for name, terms in relative.items():
        keys = [("full-df", term) for term in terms]
        if not all(
            key in native_cases
            and native_cases[key]["status"] == refs[key]["status"] == "completed"
            for key in keys
        ):
            report["failures"].append(
                {"relative": name, "error": "incomplete relative-energy population"}
            )
            continue
        expected = sum(
            terms[key[1]] * refs[key]["central"]["energies"]["composite"]
            for key in keys
        )
        actual = sum(
            terms[key[1]]
            * next(
                r["energy"]
                for r in native_cases[key]["samples"]
                if r["policy"] == "standard"
            )
            for key in keys
        )
        report["relative_energies"][name] = {
            "standard": actual,
            "reference": expected,
            "terms": terms,
        }
        report["errors"]["relative/" + name] = error(
            actual, expected, sum(abs(v) for v in terms.values()) * 1e-7
        )
    report["passed"] = not report["failures"] and all(
        e["passed"] for e in report["errors"].values()
    )
    report["case_count"] = len(native_cases)
    require(report["case_count"] == 13, "qualification population changed")
    verify_timings(payload, report)
    report["passed"] = report["passed"] and all(
        e["passed"] for e in report["errors"].values()
    )
    return report


def verify_timings(payload: dict, report: dict) -> None:
    """Qualify Standard timings without presenting internal controls as a tier.

    The historical all-FP64 endpoints are cross-revision numerical controls,
    never same-build speed comparisons. C2 also retains independent PySCF
    energies and two-step directional forces from the earlier campaign.
    """
    controls, oracle = payload["qm7_controls"], payload["qm7_reference"]
    measured, allocations = {"samples": [], "protocols": {}}, {}
    native = payload["native"]["native-final5"]
    # Supplementary sizes keep their original job/card/startup receipts.
    # A curve may span cards of the same model; it is not one paired trial.
    for run in payload["qm7_runs"].values():
        require(run["status"] == "completed", "incomplete QM7 endpoint")
        require(
            run["library_sha256"] == native["library_sha256"]
            and run["source_artifact_sha256"] == native["source_patch_sha256"],
            "QM7 source/library drift",
        )
        require(
            not measured["protocols"].keys() & run["protocols"].keys(),
            "duplicate QM7 population",
        )
        measured["samples"].extend(run["samples"])
        measured["protocols"].update(run["protocols"])
        allocations.update(
            {
                carbon: {"slurm_job_id": run["slurm_job_id"], "gpu": run["gpu"]}
                for carbon in run["protocols"]
            }
        )
    keys = [(r["carbon"], r["repeat"]) for r in measured["samples"]]
    require(
        len(keys) == 14 and set(keys) == {(c, i) for c in range(1, 8) for i in (0, 1)},
        "incomplete QM7 population",
    )
    report["qm7_standard_timings"], report["qm7_work"] = [], []
    for carbon in range(1, 8):
        control = controls[str(carbon)]
        protocol = measured["protocols"][str(carbon)]
        require(protocol == control["protocol"], "QM7 control input mismatch")
        rows = [r for r in measured["samples"] if r["carbon"] == carbon]
        seconds = [r["complete_seconds"] for r in rows]
        require(all(np.isfinite(t) and t > 0 for t in seconds), "invalid endpoint time")
        report["qm7_standard_timings"].append(
            {
                "carbon": carbon,
                "seconds": seconds,
                "median": median(seconds),
                **allocations[str(carbon)],
            }
        )
        for row in rows:
            prefix = f"qm7/C{carbon}/{row['repeat']}"
            require(
                row["policy"] == "standard"
                and row["preconvergence"] == "auto"
                and row["converged"]
                and row["executed_backend"] == "cuda",
                "invalid Standard endpoint",
            )
            require(
                row["convergence"]["limits"]["ccsd_residual_tolerance"] == 1e-7
                and row["convergence"]["limits"]["density_tolerance"] == 1e-11,
                "final default controls drift",
            )
            require(
                len(row["components"]) == 5 and len(row["execution_groups"]) == 2,
                "incomplete QM7 endpoint",
            )
            pairs = [("composite", row, control["result"])]
            for component, expected in zip(
                row["components"], control["result"]["components"], strict=True
            ):
                require(
                    component["roles"] == expected["roles"]
                    and component["basis_identity"] == expected["basis_identity"]
                    and component["method"] == expected["method"],
                    "component identity drift",
                )
                label = (
                    "high"
                    if any(r.endswith("_high") for r in component["roles"])
                    else "low"
                )
                pairs.append(
                    (
                        label + "/" + component["method"],
                        component["result"],
                        expected["result"],
                    )
                )
            for field, actual, expected in pairs:
                report["errors"][prefix + "/control-energy/" + field] = error(
                    actual["energy"], expected["energy"], 1e-7
                )
                report["errors"][prefix + "/control-force/" + field] = error(
                    actual["forces"], expected["forces"], 3e-6
                )
                if carbon == 2:
                    require(protocol == oracle["protocol"], "QM7 oracle input mismatch")
                    report["errors"][prefix + "/oracle-energy/" + field] = error(
                        actual["energy"], oracle["energies"][field], 1e-7
                    )
                    for stencil in oracle["derivatives"]:
                        report["errors"][
                            prefix + f"/oracle-force/{stencil['step_bohr']}/" + field
                        ] = error(
                            -np.vdot(actual["forces"], oracle["direction"]),
                            stencil["values"][field],
                            3e-6,
                        )
            result = next(
                c["result"] for c in row["components"] if c["method"] == "df-ccsd(t)"
            )
            cc, perf = result["correlation"], result["cc_performance"]
            residual = max(
                cc["ccsd_replay_singles_residual_max"],
                cc["ccsd_replay_doubles_residual_max"],
            )
            require(residual <= 1e-7, "QM7 final physical replay failed")
            report["qm7_work"].append(
                {
                    "carbon": carbon,
                    "repeat": row["repeat"],
                    "ccsd_iterations": cc["ccsd_iterations"],
                    "residual": residual,
                    "cc_performance": perf,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(
        lzma.decompress((args.directory / "records.json.xz").read_bytes())
    )
    report = verify(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        "PASS" if report["passed"] else "FAIL",
        report["case_count"],
        "cases;",
        len(report["errors"]),
        "observable gates",
    )
    for name, check in report["errors"].items():
        if not check["passed"]:
            print(name, check)
    for failure in report["failures"]:
        print(failure)
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
