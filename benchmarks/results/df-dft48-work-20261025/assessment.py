"""Recompute the scoped improvement and independent numerical comparisons."""

from __future__ import annotations

import json
import lzma
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from benchmarks.readme_defaults_accuracy import errors

if TYPE_CHECKING:
    from pathlib import Path


def read(path: Path) -> dict | list:
    """Decode retained values without rounding or source-identity rewriting."""
    return json.loads(lzma.decompress(path.read_bytes()))


def key(record: dict) -> tuple:
    """Identify an entire cold/three-warm population."""
    return tuple(record[k] for k in ("arm", "method", "mode", "atoms"))


def timing(record: dict) -> dict:
    """Keep all warm observations; the minimum is never the headline."""
    cold = [s["seconds"] for s in record["samples"] if s["phase"] == "cold"]
    warm = [s["seconds"] for s in record["samples"] if s["phase"] == "warm"]
    assert len(cold) == 1 and len(warm) == 3
    return {
        "cold": cold[0],
        "warm": median(warm),
        "warm_min": min(warm),
        "warm_max": max(warm),
    }


def assess(root: Path) -> dict:
    """Compare the clean control, updated figure and unchanged numerical gates."""
    rows = [
        r
        for arm in ("standard", "fast", "gpu4pyscf")
        for r in read(root / f"{arm}.json.xz")
    ]
    index = {key(r): r for r in rows}
    assert len(index) == 192
    previous = root.parent / "readme-final-20261024"
    old = {
        key(r): r
        for arm in ("standard", "fast")
        for r in read(previous / f"{arm}.json.xz")
    }
    improvements = []
    for k, r in sorted(index.items()):
        if r["arm"] == "gpu4pyscf" or not (
            r["method"] == "hf" or (r["mode"] == "df" and r["atoms"] == 48)
        ):
            continue
        b = old[k]
        g = index[("gpu4pyscf", *k[1:])]
        for field in ("geometry_bohr", "runner_sha256", "auxiliary_input_sha256"):
            assert b.get(field) == r.get(field) == g.get(field), field
        improvements.append(
            {
                **{x: r[x] for x in ("arm", "method", "mode", "atoms")},
                "previous_readme": timing(b),
                "current": timing(r),
                "gpu4": timing(g),
            }
        )
    controls = []
    for b in read(root / "control.json.xz"):
        r = index[key(b)]
        changes = []
        for before, after in zip(b["samples"], r["samples"], strict=True):
            assert (before["phase"], before["repeat"]) == (
                after["phase"],
                after["repeat"],
            )
            d = np.asarray(after["forces"]) - np.asarray(before["forces"])
            delta = {
                "phase": after["phase"],
                "repeat": after["repeat"],
                "energy_abs": abs(after["energy"] - before["energy"]),
                "force_max_abs": float(abs(d).max()),
                "force_rms": float(np.sqrt(np.mean(d * d))),
                "focks_before": before["fock_builds"],
                "focks_after": after["fock_builds"],
            }
            delta["passed"] = delta["energy_abs"] <= 1e-8 and delta[
                "force_max_abs"
            ] <= (3e-7 if r["arm"] == "fast" else 1e-9)
            changes.append(delta)
        controls.append(
            {
                **{x: r[x] for x in ("arm", "method", "mode", "atoms")},
                "control": timing(b),
                "candidate": timing(r),
                "changes": changes,
            }
        )
    wins = []
    for mode in ("direct", "df"):
        for phase in ("cold", "warm"):
            entry = {"mode": mode, "phase": phase}
            for arm in ("standard", "fast"):
                selected = [r for r in rows if r["mode"] == mode and r["arm"] == arm]
                entry[arm] = {
                    "wins": sum(
                        timing(r)[phase]
                        < timing(index[("gpu4pyscf", r["method"], mode, r["atoms"])])[
                            phase
                        ]
                        for r in selected
                    ),
                    "pairs": len(selected),
                }
            wins.append(entry)
    moved = []
    for r in read(root / "regressions.json.xz")["moved"]:
        check = (
            errors(r["energy"], r["forces"], r["reference"], 48)
            if "forces" in r["properties"]
            else {
                "energy_abs_per_atom_hartree": abs(
                    r["energy"] - r["reference"]["energy"]
                )
                / 48,
                "passed": abs(r["energy"] - r["reference"]["energy"]) / 48 <= 1e-5,
            }
        )
        assert check == r["errors"]
        moved.append({**{k: r[k] for k in ("method", "arm", "properties")}, **check})
    return {
        "improvements": improvements,
        "controls": controls,
        "wins": wins,
        "moved": moved,
    }


def table(value: dict) -> str:
    """Keep the requested HF/DF48 updates visible beside their prior README data."""
    lines = [
        "# HF and DF 48-atom updates",
        "",
        "One cold observation; warm is the median of all three frozen-density calls. The previous README is a separate source cohort, so these changes include the prior HF/DF preparation work as well as this force-policy repair. Same-binary force-policy controls are retained in comparisons.json.xz.",
        "",
        "| Arm | Method | Mode | Atoms | Previous cold | Current cold | GPU4 cold | Previous warm | Current warm [range] | GPU4 warm |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in value["improvements"]:
        b, a, g = (r[k] for k in ("previous_readme", "current", "gpu4"))
        lines.append(
            f"| {r['arm']} | {r['method']} | {r['mode']} | {r['atoms']} | {b['cold']:.4f} | {a['cold']:.4f} | {g['cold']:.4f} | {b['warm']:.4f} | {a['warm']:.4f} [{a['warm_min']:.4f}, {a['warm_max']:.4f}] | {g['warm']:.4f} |"
        )
    return "\n".join(lines) + "\n"
