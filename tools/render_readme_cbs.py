"""Render full DF-CBS endpoints without converting timeouts into timings.

Independent energy and two-step directional-derivative gates cover both the
composite and all constituents. They are not a full-vector force oracle or a
CBS basis-set accuracy certificate. Missing references remain explicitly null.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
from pathlib import Path

import numpy as np

from benchmarks.readme_cbs import SCHEMA, STEPS, digest

ENERGY_GATE = 5e-8
DIRECTION_GATE = 5e-6
TRANSLATION_GATE = 1e-6


def reference_label(summary: dict) -> str:
    """Do not merge or silently relabel different fitted Hamiltonians in a series."""
    models = {
        point["native"]["protocol"].get("hf_reference", "exact")
        for point in summary["points"]
        if point.get("native") is not None
    }
    if len(models) > 1 or not models.issubset({"exact", "df"}):
        raise ValueError("a CBS series must use one explicit HF reference model")
    return (
        "Full DF: RI-JK reference/response; shared JKFIT auxiliaries for correlation."
        if models == {"df"}
        else "Correlation-only DF; final RHF uses exact J/K."
    )


def constituents(sample: dict, protocol: dict) -> dict:
    """Identify shared native companions by orbital identity, not elapsed time."""
    names = dict(zip(protocol["orbital_identities"], ("low", "high"), strict=True))
    result = {"composite": sample}
    for component in sample["components"]:
        key = f"{names[component['basis_identity']]}/{component['method']}"
        if key in result:
            raise ValueError("duplicate CBS constituent")
        result[key] = component["result"]
    expected = {
        "composite",
        "low/rhf",
        "high/rhf",
        "low/mp2",
        "high/mp2",
        "low/df-ccsd(t)",
    }
    if result.keys() != expected:
        raise ValueError("incomplete DF-CBS constituents")
    return result


def qualification(sample: dict, protocol: dict, reference: dict | None) -> dict:
    """Fail closed on partial/nonfinite native E+F, independently of the oracle."""
    rows = constituents(sample, protocol)
    shape = (len(protocol["atoms_bohr"]), 3)
    execution = (
        math.isfinite(sample["complete_seconds"])
        and sample["complete_seconds"] > 0
        and all(
            row["converged"]
            and row["executed_backend"] == "cuda"
            and math.isfinite(row["energy"])
            and np.shape(row["forces"]) == shape
            and np.isfinite(row["forces"]).all()
            and np.max(np.abs(np.sum(row["forces"], axis=0))) <= TRANSLATION_GATE
            for row in rows.values()
        )
    )
    result = {
        "execution_gate": bool(execution),
        "accuracy_gate": None,
        "maximum_energy_error_hartree": None,
        "maximum_directional_error_hartree_per_bohr": None,
        "component_errors": None,
    }
    if reference is None:
        return result
    if reference["protocol"] != protocol:
        raise ValueError("independent reference has a different scientific input")
    derivatives = reference.get("derivatives", [])
    if reference.get("status") != "completed" or len(derivatives) != len(STEPS):
        return result
    if tuple(row["step_bohr"] for row in derivatives) != STEPS:
        raise ValueError("independent reference lacks the prescribed two-step check")
    direction = np.asarray(reference["direction"])
    if (
        direction.shape != shape
        or not np.isfinite(direction).all()
        or not np.isclose(np.linalg.norm(direction), 1)
    ):
        raise ValueError("invalid independent force direction")
    errors = {}
    for key, row in rows.items():
        values = [
            reference["energies"][key],
            *(step["values"][key] for step in derivatives),
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite independent oracle")
        errors[key] = {
            "energy": abs(row["energy"] - values[0]),
            "directional_derivatives": [
                abs(-float(np.vdot(row["forces"], direction)) - value)
                for value in values[1:]
            ]
            if execution
            else None,
        }
    maximum_energy = max(error["energy"] for error in errors.values())
    maximum_force = (
        max(
            value
            for error in errors.values()
            for value in error["directional_derivatives"]
        )
        if execution
        else None
    )
    # Fast's native stopping policy has an observable allowance. A normalized
    # force projection is bounded by either its max norm or its RMS norm; this
    # remains a directional check, never a claim of independent vector coverage.
    convergence = sample.get("convergence", {})
    limits = {}
    for key in rows:
        budget = convergence.get("budget" if key == "composite" else "component_budget")
        limits[key] = {
            "energy": budget["energy_abs"] if budget else ENERGY_GATE,
            "directional": min(
                budget["force_max_abs"] * float(np.abs(direction).sum()),
                budget["force_rms"] * math.sqrt(direction.size)
                if budget.get("force_rms") is not None
                else math.inf,
            )
            if budget
            else DIRECTION_GATE,
        }
    result.update(
        accuracy_gate=bool(
            execution
            and all(
                error["energy"] <= limits[key]["energy"]
                and max(error["directional_derivatives"]) <= limits[key]["directional"]
                for key, error in errors.items()
            )
        ),
        maximum_energy_error_hartree=maximum_energy,
        maximum_directional_error_hartree_per_bohr=maximum_force,
        component_errors=errors,
        component_limits=limits,
    )
    return result


def collect(directory: Path, carbon: int) -> dict:
    """Preserve whole-point outcomes and earlier completed calls separately."""
    point = {
        "carbon": carbon,
        "files": {},
        "native": None,
        "reference": None,
        "checks": [],
    }
    for engine in ("native", "reference"):
        for suffix in ("json", "outcome"):
            path = directory / str(carbon) / f"{engine}.{suffix}"
            if path.exists():
                record = json.loads(path.read_text())
                if suffix == "json" and (
                    record.get("schema") != SCHEMA or record.get("engine") != engine
                ):
                    raise ValueError(f"unexpected CBS receipt: {path}")
                point[engine if suffix == "json" else f"{engine}_outcome"] = record
                point["files"][f"{engine}.{suffix}"] = digest(path)
    native = point["native"]
    if native is not None:
        if native["protocol"]["carbon"] != carbon:
            raise ValueError("receipt has the wrong carbon count")
        point["checks"] = [
            qualification(sample, native["protocol"], point["reference"])
            for sample in native["samples"]
        ]
    return point


def arithmetic_provenance(sample: dict, protocol: dict) -> dict | None:
    """Check actual FP32 work; older convergence-only receipts remain explicit.

    A resource-refused request is valid FP64 execution, not mixed execution.
    Missing telemetry cannot substantiate a new FP32 preconvergence claim.
    """
    convergence = sample.get("convergence", {})
    if "arithmetic_request" not in convergence:
        return None
    request = convergence["arithmetic_request"]
    if request == "fp64":
        if convergence.get("arithmetic") != "fp64":
            raise ValueError("FP64 request has inconsistent arithmetic provenance")
        return {"arithmetic": "fp64"}
    if request != "df_ccsd_fp32_preconvergence":
        raise ValueError("unknown arithmetic request")
    cc = constituents(sample, protocol)["low/df-ccsd(t)"].get("cc_performance") or {}
    flags = ("requested", "executed", "refused")
    counts = ("evaluations", "gemm_calls", "cast_elements")
    if (
        any(type(cc.get(f"ccsd_fp32_{key}")) is not bool for key in flags)
        or cc.get("ccsd_fp32_requested") is not True
        or any(
            type(cc.get(f"ccsd_fp32_{key}")) is not int or cc[f"ccsd_fp32_{key}"] < 0
            for key in counts
        )
    ):
        raise ValueError("FP32 request lacks native arithmetic telemetry")
    mixed = cc["ccsd_fp32_executed"]
    if mixed != (cc["ccsd_fp32_gemm_calls"] > 0) or (
        mixed and (cc["ccsd_fp32_refused"] or cc["ccsd_fp32_evaluations"] == 0)
    ):
        raise ValueError("FP32 counters disagree with reported arithmetic")
    result = {key: cc[f"ccsd_fp32_{key}"] for key in (*flags, *counts)}
    # Historical receipts predate the separate Lambda request. Preserve their
    # exact diagnostic shape and never infer unrecorded work from today's policy.
    if "lambda_arithmetic_request" in convergence:
        request = convergence["lambda_arithmetic_request"]
        if request == "df_lambda_fp32_preconvergence":
            work = (
                "fp32_iterations",
                "fp64_iterations",
                "fp32_actions",
                "fp64_actions",
                "fp32_gemm_calls",
                "fp32_cast_elements",
                "auxiliary_batch_size",
                "numeric_capacity_bytes",
                "owned_device_bytes",
            )
            if (
                any(type(cc.get(f"lambda_fp32_{key}")) is not bool for key in flags)
                or cc.get("lambda_fp32_requested") is not True
                or any(
                    type(cc.get(f"lambda_{key}")) is not int or cc[f"lambda_{key}"] < 0
                    for key in work
                )
            ):
                raise ValueError(
                    "Lambda FP32 request lacks native arithmetic telemetry"
                )
            executed = cc["lambda_fp32_executed"]
            if executed != (cc["lambda_fp32_gemm_calls"] > 0) or (
                executed
                and (
                    cc["lambda_fp32_refused"]
                    or not cc["lambda_fp32_actions"]
                    or not cc["lambda_fp32_cast_elements"]
                )
            ):
                raise ValueError(
                    "Lambda FP32 counters disagree with reported arithmetic"
                )
            arithmetic = "mixed" if executed else "fp64"
            result["lambda"] = {
                "arithmetic": arithmetic,
                **{key: cc[f"lambda_fp32_{key}"] for key in flags},
                **{key: cc[f"lambda_{key}"] for key in work},
            }
            mixed |= executed
        elif request in ("fp64", "not_requested"):
            arithmetic = request
            result["lambda"] = {"arithmetic": arithmetic}
        else:
            raise ValueError("unknown Lambda arithmetic request")
        if convergence.get("lambda_arithmetic") != arithmetic:
            raise ValueError("Lambda counters disagree with reported arithmetic")
    arithmetic = "mixed" if mixed else "fp64"
    if convergence.get("arithmetic") != arithmetic:
        raise ValueError("FP32 counters disagree with reported arithmetic")
    return {"arithmetic": arithmetic, **result}


def attach_fast(summary: dict, fast: dict) -> None:
    """Attach measured Fast points only after matched-source full-vector checks.

    Standard is a convergence control, not an independent scientific oracle.
    Both its existing independent checks and the Fast checks remain in the
    receipt. A failed or missing control cannot authorize a plotted Fast time.
    """
    controls = {point["carbon"]: point for point in summary["points"]}
    for point in fast["points"]:
        control = controls.get(point["carbon"])
        point["standard_comparison"] = []
        candidate = point.get("native")
        standard = control.get("native") if control else None
        if candidate is None or standard is None:
            continue
        for field in (
            "protocol",
            "source_revision",
            "driver_sha256",
            "library_sha256",
            "gpu",
        ):
            if not candidate.get(field) or not standard.get(field):
                raise ValueError(f"Fast/Standard {field} provenance is missing")
            if candidate.get(field) != standard.get(field):
                raise ValueError(
                    f"Fast/Standard {field} differs for C{point['carbon']}"
                )
        if candidate.get("policy") != "fast" or standard.get("policy") != "standard":
            raise ValueError("comparison requires explicit Fast and Standard receipts")
        if not completed_times(control):
            continue
        baseline = constituents(standard["samples"][0], standard["protocol"])
        for index, sample in enumerate(candidate["samples"]):
            if not point["checks"][index]["execution_gate"]:
                point["standard_comparison"].append({"passed": False, "components": {}})
                continue
            convergence = sample.get("convergence", {})
            if convergence.get("requested_policy") != "fast":
                raise ValueError("Fast receipt lacks executed convergence provenance")
            errors = {}
            accepted = True
            for key, row in constituents(sample, candidate["protocol"]).items():
                budget = convergence.get(
                    "budget" if key == "composite" else "component_budget"
                )
                if not budget:
                    raise ValueError("Fast receipt lacks observable allowances")
                delta = np.asarray(row["forces"]) - np.asarray(baseline[key]["forces"])
                values = {
                    "energy_abs": abs(row["energy"] - baseline[key]["energy"]),
                    "force_max_abs": float(np.max(np.abs(delta))),
                    "force_rms": float(np.sqrt(np.mean(delta**2))),
                }
                passed = all(
                    math.isfinite(value)
                    and (budget[name] is None or value <= budget[name])
                    for name, value in values.items()
                )
                errors[key] = {**values, "budget": budget, "passed": passed}
                accepted = accepted and passed
            point["standard_comparison"].append(
                {
                    "passed": accepted,
                    "components": errors,
                    "arithmetic": arithmetic_provenance(sample, candidate["protocol"]),
                }
            )
    summary["fast"] = fast


def completed_times(point: dict, phase: str = "cold") -> list[float]:
    """Never average an incomplete replay population or plot failed accuracy."""
    native = point["native"]
    if native is None or not native["samples"]:
        return []
    if phase == "cold":
        indices = range(1)
    elif phase == "replay":
        if len(native["samples"]) != native["requested_samples"]:
            return []
        indices = range(1, len(native["samples"]))
    else:
        raise ValueError("unknown endpoint phase")
    if "standard_comparison" in point and (
        len(point["standard_comparison"]) < len(native["samples"])
        or any(not point["standard_comparison"][index]["passed"] for index in indices)
    ):
        return []
    if any(
        not point["checks"][index]["execution_gate"]
        or point["checks"][index]["accuracy_gate"] is False
        for index in indices
    ):
        return []
    return [native["samples"][index]["complete_seconds"] for index in indices]


def outcome(point: dict) -> str:
    """A deadline is a whole-process outcome, never a completed E+F time."""
    native = point["native"]
    if completed_times(point):
        return (
            "checked (directional)"
            if point["checks"][0]["accuracy_gate"]
            else "unchecked ‡"
        )
    if (
        "standard_comparison" in point
        and point["native"]
        and point["native"]["samples"]
    ):
        return "Fast control missing/failed"
    if point["checks"]:
        return "numerical gate failed"
    exit_code = point.get("native_outcome", {}).get("exit_code")
    if exit_code == 124:
        return "point time limit"
    if native is None:
        return "not completed"
    return native["status"] + (": " + native["error"] if native.get("error") else "")


def table(summary: dict) -> str:
    """Display complete API cold calls, with work diagnostics kept alongside."""
    if "fast" in summary:
        return comparison_table(summary)
    lines = [
        "# DF-CCSD(T)/CBS complete energy + analytic forces",
        "",
        reference_label(summary),
        "",
        "One fresh-calculator cold observation per molecule unless noted. Seconds include",
        "both native basis groups and returned host forces; they are not RHF-only times.",
        "‡ = independent accuracy unchecked. Checked means constituent + composite energy",
        "and one normalized force projection at two finite-difference steps, not every force component.",
        "",
        "| Alkane | QM7 index | AO DZ/TZ | Auxiliary DZ/TZ | Complete E+F (s) | Outcome |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for point in summary["points"]:
        record = point["native"] or point["reference"]
        protocol = record["protocol"] if record else {}
        times = completed_times(point)
        seconds = f"{times[0]:.3f}" if times else "—"
        counts = {
            field: "/".join(str(value) for value in protocol.get(field, [])) or "—"
            for field in ("nao", "naux")
        }
        lines.append(
            f"| C{point['carbon']}H{2 * point['carbon'] + 2} | {protocol.get('qm7_index_zero_based', '—')} | "
            f"{counts['nao']} | {counts['naux']} | {seconds} | {outcome(point)} |"
        )
    lines.extend(
        [
            "",
            "## Observed work",
            "",
            "DZ SCF iteration counts are unavailable: the frozen companion ABI returns",
            "the primary CC convergence counter, not the HF counter. They are not inferred.",
            "TZ has an MP2 primary whose convergence counter records its HF reference.",
            "Response iterations are separate for DZ CCSD(T), DZ MP2 and TZ MP2.",
            "Group times are non-overlapping; individual companion timings must not be added.",
            "",
            "| Alkane | SCF iterations DZ/TZ | CCSD iterations | Response iterations CC/DZ-MP2/TZ-MP2 | Group seconds DZ/TZ | Max ΔE (Eh) | Max directional error (Eh/Bohr) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for point in summary["points"]:
        native = point["native"]
        if not native or not native["samples"]:
            continue
        sample = native["samples"][0]
        rows = constituents(sample, native["protocol"])
        scf = f"—/{rows['high/rhf']['iterations']}"
        correlation = [
            rows[key]["correlation"]
            for key in ("low/df-ccsd(t)", "low/mp2", "high/mp2")
        ]
        response = "/".join(str(value["response_iterations"]) for value in correlation)
        groups = {
            group["basis_identity"]: group["elapsed_seconds"]
            for group in sample["execution_groups"]
        }
        seconds = "/".join(
            f"{groups[identity]:.3f}"
            for identity in native["protocol"]["orbital_identities"]
        )
        check = point["checks"][0]
        energy = check["maximum_energy_error_hartree"]
        force = check["maximum_directional_error_hartree_per_bohr"]
        error_text = "/".join(
            "—" if value is None else f"{value:.3e}" for value in (energy, force)
        ).split("/")
        lines.append(
            f"| C{point['carbon']} | {scf} | {correlation[0]['ccsd_iterations']} | {response} | "
            f"{seconds} | {error_text[0]} | {error_text[1]} |"
        )
    return "\n".join(lines) + "\n"


def comparison_table(summary: dict) -> str:
    """Retain complete observations, vector differences and CC/response work."""
    lines = [
        "# Standard and Fast DF-CCSD(T)/CBS",
        "",
        reference_label(summary),
        "",
        "One API-cold complete E+F observation per policy and molecule; same binary and GPU within each pair.",
        "Fast differences are against Standard, not independent oracle errors. Constituent budgets also apply.",
        "",
        "| Alkane | Standard (s) | Fast (s) | Max composite ΔE (Eh) | Max force difference (Eh/Bohr) | RMS force difference (Eh/Bohr) | Fast control |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    controls = {point["carbon"]: point for point in summary["points"]}
    for point in summary["fast"]["points"]:
        st, ft = completed_times(controls[point["carbon"]]), completed_times(point)
        checks = point.get("standard_comparison", [])
        values = checks[0]["components"].get("composite", {}) if checks else {}
        cells = [f"{times[0]:.3f}" if times else "—" for times in (st, ft)]
        cells += [
            f"{values[key]:.3e}" if key in values else "—"
            for key in ("energy_abs", "force_max_abs", "force_rms")
        ]
        status = "passed" if ft else outcome(point)
        lines.append(f"| C{point['carbon']} | " + " | ".join(cells) + f" | {status} |")
    lines += [
        "",
        "## Fast arithmetic",
        "",
        "FP32 counts cover auxiliary CCSD preconvergence; final checks and response use FP64.",
        "Unknown means the older receipt did not record precision telemetry.",
        "",
        "| Alkane | Executed arithmetic | FP32 refused | FP32 evaluations | FP32 GEMMs | Cast elements |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for point in summary["fast"]["points"]:
        checks = point.get("standard_comparison", [])
        precision = (checks[0].get("arithmetic") if checks else None) or {}
        cells = [
            str(precision.get(key, "unknown"))
            for key in (
                "arithmetic",
                "refused",
                "evaluations",
                "gemm_calls",
                "cast_elements",
            )
        ]
        lines.append(f"| C{point['carbon']} | " + " | ".join(cells) + " |")
    if any(
        "lambda_arithmetic_request" in sample.get("convergence", {})
        for point in summary["fast"]["points"]
        for sample in (point.get("native") or {}).get("samples", [])
    ):
        lines += [
            "",
            "## Lambda work",
            "",
            "FP32 auxiliary warmup is followed by a fresh FP64 solve and independent audit.",
            "Actions count the shared transpose operator, excluding the separate audit; phase time includes both.",
            "Q is the actual auxiliary batch. Numeric capacity is not measured peak VRAM.",
            "",
            "| Alkane | Lambda Standard/Fast (s) | Standard FP64 actions | Fast FP32/FP64 actions | Fast Q | Fast FP32 GEMMs | Fast numeric capacity (GiB) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for point in summary["fast"]["points"]:
            work = []
            for item in (controls[point["carbon"]], point):
                native = item.get("native") or {}
                samples = native.get("samples", [])
                cc = (
                    constituents(samples[0], native["protocol"])["low/df-ccsd(t)"].get(
                        "cc_performance"
                    )
                    if samples
                    else None
                )
                work.append(cc or {})
            standard, fast = work
            times = "/".join(
                f"{cc['lambda_seconds']:.3f}"
                if cc.get("lambda_seconds") is not None
                else "—"
                for cc in work
            )
            actions = "/".join(
                str(fast.get(f"lambda_fp{width}_actions", "—")) for width in (32, 64)
            )
            capacity = fast.get("lambda_numeric_capacity_bytes")
            cells = [
                times,
                str(standard.get("lambda_fp64_actions", "—")),
                actions,
                str(fast.get("lambda_auxiliary_batch_size", "—")),
                str(fast.get("lambda_fp32_gemm_calls", "—")),
                f"{capacity / (1 << 30):.3f}" if capacity is not None else "—",
            ]
            lines.append(f"| C{point['carbon']} | " + " | ".join(cells) + " |")
    for name, series in (("Standard", summary), ("Fast", summary["fast"])):
        # The existing single-series table contains actual semantic counters,
        # independent checks and per-group timings without double-counting companions.
        plain = {key: value for key, value in series.items() if key != "fast"}
        lines += [
            "",
            f"## {name} diagnostics",
            "",
            table(plain)
            .replace(
                "# DF-CCSD(T)/CBS complete energy + analytic forces",
                "### Complete endpoints",
            )
            .replace("## Observed work", "### Observed work"),
        ]
    return "\n".join(lines).rstrip() + "\n"


def render(summary: dict, destination: Path, *, readme_style: bool = False) -> None:
    """Plot only measured complete calls; gaps retain an explicit outcome label."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update(
        {"svg.fonttype": "none", "svg.hashsalt": SCHEMA, "font.family": "DejaVu Sans"}
    )
    figure, axes = plt.subplots(figsize=(10, 4.8))
    series = [("Standard", summary, "#087e8b", "o")]
    if "fast" in summary:
        series.append(("Fast", summary["fast"], "#e07824", "s"))
    measured = {
        name: {point["carbon"]: completed_times(point) for point in data["points"]}
        for name, data, _, _ in series
    }
    any_values = False
    for name, data, color, marker in series:
        values = [measured[name][point["carbon"]] for point in data["points"]]
        any_values = any_values or any(values)
        axes.plot(
            [point["carbon"] for point in data["points"]],
            [times[0] if times else math.nan for times in values],
            color=color,
            linewidth=1.6,
            label=name,
        )
        for point, times in zip(data["points"], values, strict=True):
            carbon = point["carbon"]
            if times:
                checked = point["checks"][0]["accuracy_gate"]
                label = f"{times[0]:.1f} s"
                offset = 10 if name == "Standard" else -17
                # Place labels outside the pair even when Fast is slower.
                # Fixed per-policy offsets overlap on crossing curves.
                peer = measured.get("Fast" if name == "Standard" else "Standard", {})
                if peer.get(carbon) and times[0] != peer[carbon][0]:
                    offset = 10 if times[0] > peer[carbon][0] else -17
                if not readme_style and not checked:
                    label += " ‡"
                axes.plot(
                    carbon,
                    times[0],
                    marker=marker,
                    color=color,
                    markerfacecolor=color if (readme_style or checked) else "white",
                )
                axes.annotate(
                    label,
                    (carbon, times[0]),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=color if len(series) > 1 else "black",
                )
            else:
                axes.text(
                    carbon,
                    0.04 if name == "Standard" else 0.42,
                    (name + ": " if len(series) > 1 else "")
                    + outcome(point).split(":")[0],
                    transform=axes.get_xaxis_transform(),
                    rotation=90,
                    va="bottom",
                    ha="center",
                    fontsize=9,
                    color="#93422e",
                )
    if any_values:
        axes.set_yscale("log")
        axes.margins(y=0.22)
    axes.set(
        xlim=(0.5, 7.5),
        xticks=range(1, 8),
        xlabel="QM7 alkane representative (carbon count)",
        ylabel="Complete E+F wall time / s",
        title="DF-CCSD(T)/CBS · cc-pV[D,T]Z + DZ correction · RTX 5090",
    )
    axes.grid(axis="y", alpha=0.2)
    if "fast" in summary:
        axes.legend(fontsize=9, loc="upper left")
    elif not readme_style:
        axes.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="#087e8b",
                    linestyle="none",
                    label="Independent energy + directional-force checks",
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    color="#087e8b",
                    markerfacecolor="white",
                    linestyle="none",
                    label="Complete, accuracy unchecked (‡)",
                ),
            ],
            fontsize=8,
            loc="upper left",
        )
    footer = reference_label(summary) + (
        "\nOne cold call per policy; same-build pairs. Fast timings require full-vector checks against Standard."
        if "fast" in summary
        else " One cold call; compiler caches retained."
    )
    if not readme_style:
        footer += (
            "\nIncomplete/failed E+F has no plotted time. "
            "No full-vector or CBS basis-limit accuracy claim."
        )
    figure.text(
        0.5,
        0.02,
        footer,
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(destination, metadata={"Date": None})
    plt.close(figure)
    destination.write_text(
        "\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--fast-runs", type=Path)
    parser.add_argument("--fast-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Render only a performance-focused README SVG; retain all scientific gates.",
    )
    args = parser.parse_args()
    if bool(args.runs) == bool(args.summary):
        parser.error("choose --runs or --summary")
    if args.runs:
        summary = {
            "schema": SCHEMA,
            "points": [collect(args.runs, carbon) for carbon in range(1, 8)],
            "gates": {
                "energy_hartree": ENERGY_GATE,
                "directional_hartree_per_bohr": DIRECTION_GATE,
                "translation_hartree_per_bohr": TRANSLATION_GATE,
            },
        }
    else:
        summary = json.loads(lzma.decompress(args.summary.read_bytes()))
    if args.fast_runs and args.fast_summary:
        parser.error("choose --fast-runs or --fast-summary")
    if args.fast_runs or args.fast_summary:
        fast = (
            {
                "schema": SCHEMA,
                "points": [collect(args.fast_runs, carbon) for carbon in range(1, 8)],
            }
            if args.fast_runs
            else json.loads(lzma.decompress(args.fast_summary.read_bytes()))
        )
        attach_fast(summary, fast)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.readme_only:
        render(summary, args.output / "readme-cbs-performance.svg", readme_style=True)
        return
    (args.output / "summary.json.xz").write_bytes(
        lzma.compress(json.dumps(summary, allow_nan=False).encode())
    )
    (args.output / "timings.md").write_text(table(summary))
    render(summary, args.output / "df-cbs.svg")


if __name__ == "__main__":
    main()
