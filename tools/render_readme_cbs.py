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
    result.update(
        accuracy_gate=bool(
            execution
            and maximum_energy <= ENERGY_GATE
            and maximum_force <= DIRECTION_GATE
        ),
        maximum_energy_error_hartree=maximum_energy,
        maximum_directional_error_hartree_per_bohr=maximum_force,
        component_errors=errors,
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
    values = [completed_times(point) for point in summary["points"]]
    axes.plot(
        [point["carbon"] for point in summary["points"]],
        [times[0] if times else math.nan for times in values],
        color="#087e8b",
        linewidth=1.6,
    )
    for point, times in zip(summary["points"], values, strict=True):
        carbon = point["carbon"]
        if times:
            checked = point["checks"][0]["accuracy_gate"]
            label = f"{times[0]:.1f} s"
            if not readme_style and not checked:
                label += " ‡"
            axes.plot(
                carbon,
                times[0],
                marker="o",
                color="#087e8b",
                markerfacecolor="#087e8b" if (readme_style or checked) else "white",
            )
            axes.annotate(
                label,
                (carbon, times[0]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        else:
            axes.text(
                carbon,
                0.04,
                outcome(point).split(":")[0],
                transform=axes.get_xaxis_transform(),
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=9,
                color="#93422e",
            )
    if any(values):
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
    if not readme_style:
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
    footer = reference_label(summary) + " One cold call; compiler caches retained."
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
