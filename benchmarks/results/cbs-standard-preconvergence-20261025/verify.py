"""Recompute retained convergence, observable gates and complete endpoint tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(name: str) -> dict:
    return json.loads(lzma.decompress((HERE / name).read_bytes()))


def error(actual: object, expected: object, tolerance: float) -> dict:
    delta = np.asarray(actual) - np.asarray(expected)
    maximum = float(np.max(np.abs(delta)))
    require(
        np.all(np.isfinite(delta)) and maximum <= tolerance,
        f"observable gate failed: {maximum} > {tolerance}",
    )
    return {
        "passed": True,
        "atol": tolerance,
        "rtol": 0.0,
        "max_absolute_error": maximum,
        "max_scaled_error": maximum / tolerance,
        "rms_error": float(np.sqrt(np.mean(delta * delta))),
        "shape": list(np.shape(delta)) or [1],
    }


def verify() -> dict:
    payload = read("samples.json.xz")
    provenance = read("provenance.json.xz")
    oracle = read("reference.json.xz")
    require(oracle["status"] == "completed", "incomplete independent oracle")
    patch = lzma.decompress((HERE / "candidate2.patch.xz").read_bytes())
    patch_hash = hashlib.sha256(patch).hexdigest()
    library_hash = provenance["environment"]["libraries"]["candidate2"]["sha256"]
    report = {"errors": {}, "timings": [], "work": [], "resource_failures": []}
    samples = []
    for name, carbons, repeats in (
        ("four-cell", (1, 2, 4), 2),
        ("large-four-cell", (6,), 1),
        ("energy-only", (2, 6), 2),
    ):
        group = payload[name]
        require(group["status"] == "completed", f"incomplete {name}")
        require(group["source_artifact_sha256"] == patch_hash, "wrong source patch")
        require(group["library_sha256"] == library_hash, "wrong native binary")
        expected_arms = {("standard", "fp64"), ("standard", "auto")}
        if name != "energy-only":
            expected_arms |= {("fast", "fp64"), ("fast", "auto")}
        keys = [
            (r["carbon"], r["policy"], r["preconvergence"], r["repeat"])
            for r in group["samples"]
        ]
        expected = {
            (c, p, a, i)
            for c in carbons
            for p, a in expected_arms
            for i in range(repeats)
        }
        require(
            len(keys) == len(set(keys)) and set(keys) == expected,
            "incomplete population",
        )
        for row in group["samples"]:
            require(
                row["converged"] and row["executed_backend"] == "cuda",
                "invalid backend/result",
            )
            require(
                math.isfinite(row["complete_seconds"]) and row["complete_seconds"] > 0,
                "invalid full endpoint time",
            )
            require(
                (row["forces"] is None) == (name == "energy-only"), "wrong endpoint"
            )
            protocol = group["protocols"][str(row["carbon"])]
            require(
                np.allclose(
                    row["hf_weights"], protocol["hf_weights"], atol=1e-15, rtol=0
                )
                and np.allclose(
                    row["mp2_weights"], protocol["mp2_weights"], atol=1e-15, rtol=0
                ),
                "extrapolation changed",
            )
            require(
                len(row["components"]) == 5 and len(row["execution_groups"]) == 2,
                "incomplete focal point composition",
            )
            require(
                abs(
                    row["energy"]
                    - math.fsum(
                        c["coefficient"] * c["result"]["energy"]
                        for c in row["components"]
                    )
                )
                < 5e-12,
                "composite energy is inconsistent",
            )
            cc = next(
                c["result"] for c in row["components"] if c["method"] == "df-ccsd(t)"
            )
            limits = row["convergence"]["limits"]
            residual = max(
                cc["correlation"][k]
                for k in (
                    "ccsd_replay_singles_residual_max",
                    "ccsd_replay_doubles_residual_max",
                )
            )
            require(
                residual <= limits["ccsd_residual_tolerance"],
                "independent replay failed",
            )
            if row["policy"] == "standard":
                require(
                    row["convergence"]["scale"] == 0
                    and row["convergence"]["budget"] is None,
                    "strict arm received a convergence allowance",
                )
                require(
                    limits["ccsd_energy_tolerance"] == 1e-12
                    and limits["ccsd_residual_tolerance"] == 1e-10,
                    "strict controls changed",
                )
            performance = cc["cc_performance"]
            mixed = row["preconvergence"] == "auto" and (
                row["policy"] == "fast"
                or (row["carbon"] >= (6 if name == "energy-only" else 2))
            )
            require(
                performance["ccsd_fp32_requested"] is mixed,
                "unexpected automatic admission",
            )
            if mixed:
                require(
                    performance["ccsd_fp32_executed"]
                    and performance["ccsd_fp32_gemm_calls"] > 0,
                    "mixed label lacks executed work",
                )
            if row["forces"] is not None:
                require(
                    performance["lambda_fp32_requested"] is mixed,
                    "unexpected Lambda admission",
                )
                require(
                    performance["lambda_fp64_actions"] > 0,
                    "no final strict Lambda actions",
                )
            report["work"].append(
                {
                    "group": name,
                    "carbon": row["carbon"],
                    "policy": row["policy"],
                    "preconvergence": row["preconvergence"],
                    "repeat": row["repeat"],
                    "ccsd_iterations": cc["correlation"]["ccsd_iterations"],
                    "residual": residual,
                    **{
                        k: performance[k]
                        for k in (
                            "ccsd_fp32_evaluations",
                            "lambda_seconds",
                            "lambda_fp32_actions",
                            "lambda_fp64_actions",
                            "lambda_auxiliary_batch_size",
                        )
                    },
                }
            )
            samples.append((name, row, protocol))
        for carbon in carbons:
            arms = {
                arm: [
                    r
                    for r in group["samples"]
                    if r["carbon"] == carbon
                    and (r["policy"], r["preconvergence"]) == arm
                ]
                for arm in expected_arms
            }
            control = arms["standard", "fp64"][0]
            for arm, rows in arms.items():
                values = [r["complete_seconds"] for r in rows]
                report["timings"].append(
                    {
                        "group": name,
                        "carbon": carbon,
                        "policy": arm[0],
                        "preconvergence": arm[1],
                        "seconds": values,
                        "median": median(values),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                )
                for i, row in enumerate(rows):
                    prefix = f"{name}/C{carbon}/{arm[0]}/{arm[1]}/{i}"
                    budget = row["convergence"]["budget"]
                    if row["policy"] == "standard":
                        require(
                            row["convergence"]["limits"]
                            == control["convergence"]["limits"],
                            "execution selection changed strict stopping limits",
                        )
                    report["errors"][prefix + "/control-energy"] = error(
                        row["energy"],
                        control["energy"],
                        budget["energy_abs"] if budget else 1e-9,
                    )
                    if row["forces"] is not None:
                        report["errors"][prefix + "/control-forces"] = error(
                            row["forces"],
                            control["forces"],
                            budget["force_max_abs"] if budget else 1e-8,
                        )
                        if budget:
                            rms = np.sqrt(
                                np.mean(
                                    (np.asarray(row["forces"]) - control["forces"]) ** 2
                                )
                            )
                            report["errors"][prefix + "/control-force-rms"] = error(
                                rms, 0, budget["force_rms"]
                            )
                    controls = {
                        (c["method"], c["basis_identity"]): c["result"]
                        for c in control["components"]
                    }
                    component_budget = row["convergence"]["component_budget"]
                    for component in row["components"]:
                        key = component["method"], component["basis_identity"]
                        require(key in controls, "component model changed")
                        tag = prefix + "/component/" + "/".join(component["roles"])
                        actual, expected = component["result"], controls[key]
                        report["errors"][tag + "/energy"] = error(
                            actual["energy"],
                            expected["energy"],
                            component_budget["energy_abs"]
                            if component_budget
                            else 1e-9,
                        )
                        if actual["forces"] is not None:
                            report["errors"][tag + "/forces"] = error(
                                actual["forces"],
                                expected["forces"],
                                component_budget["force_max_abs"]
                                if component_budget
                                else 1e-8,
                            )
                            if component_budget:
                                rms = np.sqrt(
                                    np.mean(
                                        (
                                            np.asarray(actual["forces"])
                                            - expected["forces"]
                                        )
                                        ** 2
                                    )
                                )
                                report["errors"][tag + "/force-rms"] = error(
                                    rms, 0, component_budget["force_rms"]
                                )
    for name, row, protocol in samples:
        if row["carbon"] != 2:
            continue
        for key in (
            "atoms_bohr",
            "orbital_identities",
            "auxiliary_identities",
            "hf_weights",
            "mp2_weights",
        ):
            require(
                protocol[key] == oracle["protocol"][key],
                "independent reference model differs",
            )
        prefix = f"{name}/C2/{row['policy']}/{row['preconvergence']}/{row['repeat']}"
        values = {"composite": (row["energy"], row["forces"])}
        for c in row["components"]:
            basis = (
                "low"
                if c["basis_identity"] == protocol["orbital_identities"][0]
                else "high"
            )
            values[basis + "/" + c["method"]] = (
                c["result"]["energy"],
                c["result"]["forces"],
            )
        for key, (energy, forces) in values.items():
            report["errors"][prefix + "/oracle-energy/" + key] = error(
                energy, oracle["energies"][key], 1e-8
            )
            if forces is not None:
                for d in oracle["derivatives"]:
                    report["errors"][
                        prefix + f"/oracle-force/{d['step_bohr']}/" + key
                    ] = error(
                        -np.vdot(forces, oracle["direction"]), d["values"][key], 3e-6
                    )
    for name in ("resource-probe", "resource-probe-high"):
        for row in payload[name]["samples"]:
            if row["status"] == "failed":
                report["resource_failures"].append(
                    {k: row[k] for k in ("budget_mib", "preconvergence", "error")}
                )
            else:
                control = next(
                    r
                    for r in payload["four-cell"]["samples"]
                    if r["carbon"] == 2
                    and r["policy"] == "standard"
                    and r["preconvergence"] == "fp64"
                )
                prefix = f"resource/{row['budget_mib']}/{row['preconvergence']}"
                report["errors"][prefix + "/energy"] = error(
                    row["result"]["energy"], control["energy"], 1e-9
                )
                report["errors"][prefix + "/forces"] = error(
                    row["result"]["forces"], control["forces"], 1e-8
                )
    baseline = payload["master-compatibility"]["samples"][0]
    require(
        baseline["convergence"]["preconvergence_fallback"]
        == "native_execution_transport_unavailable",
        "older native compatibility fallback is missing",
    )
    control = next(
        r
        for r in payload["four-cell"]["samples"]
        if r["carbon"] == 2
        and r["policy"] == "standard"
        and r["preconvergence"] == "fp64"
    )
    report["errors"]["master/energy"] = error(
        baseline["energy"], control["energy"], 1e-12
    )
    report["errors"]["master/forces"] = error(
        baseline["forces"], control["forces"], 1e-12
    )
    report["status"] = "passed"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"PASS: {len(report['errors'])} observable gates; {len(report['timings'])} timing populations"
    )
