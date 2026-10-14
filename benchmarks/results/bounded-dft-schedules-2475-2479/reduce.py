"""Recheck this campaign's source bindings and every returned E/force array.

Run from the repository root with PYTHONPATH=.:python. The ignored campaign
directory retains frozen *-source trees, installed images, and raw endpoints.
Incomplete populations and changed SCF work remain visible in the publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import subprocess
from pathlib import Path
from statistics import median

from benchmarks.readme_omol25 import source_hashes
from tools.generate_build_identity import _inventory, _source_identity
from tools.render_readme_dft import collect_point, phase_rows


def digest(path: Path) -> str:
    """Hash retained bytes without depending on filesystem timestamps."""
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def source_arm(label: str) -> str:
    """Map same-binary runtime ablations to their frozen source/image owner."""
    if label.startswith("ao-"):
        return "ao"
    if label.startswith("force-"):
        return "experiment"
    return label


def planned_points() -> set[tuple[str, str, str, int]]:
    """Retain absent/failed points instead of selecting only existing JSON files."""
    points = {("baseline", "none", "r2scan-3c", atoms) for atoms in (48, 96)}
    for method in ("b3lyp", "m06-2x", "pbe0", "r2scan"):
        for atoms in (3, 12, 48, 96):
            for route in ("host", "device", "dense"):
                if route != "dense" or atoms <= 12:
                    points.add((f"ao-{route}", "none", method, atoms))
    for method, sizes in (
        ("r2scan-3c", (3, 12, 48, 96)),
        ("pbe0", (48, 96)),
        ("hf", (3, 12)),
    ):
        for atoms in sizes:
            for route in ("default", "dddd", "page16", "page8"):
                points.add((f"force-{route}", "none", method, atoms))
    for tile in (8, 16, 32):
        for method in ("m06-2x", "r2scan", "pbe0"):
            for mode in ("none", "cuda"):
                for atoms in (48, 96):
                    points.add((f"matrix{tile}", mode, method, atoms))
    return points


def freeze_source(campaign: Path, arm: str) -> dict:
    """Bind dirty pilots explicitly; a Git revision alone cannot identify them."""
    source = (campaign / f"{arm}-source").resolve()

    def git(*args: str) -> str:
        # A copied snapshot can retain the original core.worktree. Read the
        # frozen files here without mutating its Git config or following it to
        # an unrelated checkout on another measurement node.
        return subprocess.check_output(
            ["git", "-C", str(source), f"--work-tree={source}", *args],
            text=True,
        )

    inventory = _inventory(source, source / "cmake/GenerativeQCSourceIdentity.json")
    tracked = set(git("ls-files").splitlines())
    inputs = {str(path): digest(source / path) for path in inventory}
    return {
        "revision": git("rev-parse", "HEAD").strip(),
        "tracked_patch": git("diff", "HEAD", "--"),
        "untracked_native_inputs": {
            str(path): (source / path).read_text()
            for path in inventory
            if str(path) not in tracked
        },
        "native_source_identity": _source_identity(source, inventory),
        "native_inventory_sha256": hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest(),
        "native_input_count": len(inputs),
        "consumer_source_file_sha256": {
            name: digest(source / name) for name in source_hashes()
        },
        "installed_library_sha256": digest(campaign / arm / "lib/libgenerativeqc.so"),
    }


def source_binding(native: dict, source: dict) -> dict:
    """Keep incomplete or mismatched receipts without certifying their source."""
    reasons = []
    build = native.get("native_build")
    if not isinstance(build, dict):
        reasons.append("missing or malformed native_build")
    else:
        probe = build.get("probe")
        if (
            not isinstance(probe, dict)
            or probe.get("source_identity") != source["native_source_identity"]
        ):
            reasons.append("missing or mismatched native source identity")
        if build.get("library_sha256") != source["installed_library_sha256"]:
            reasons.append("missing or mismatched installed library hash")
    observed = native.get("source_file_sha256")
    expected = source["consumer_source_file_sha256"]
    if not isinstance(observed, dict) or set(observed) != set(expected):
        reasons.append("missing, malformed or incomplete consumer source inventory")
    elif observed != expected:
        reasons.append("consumer source hash mismatch")
    return {"status": "unqualified" if reasons else "verified", "reasons": reasons}


def compact_point(point: dict) -> None:
    """Select publication telemetry only after checking the untouched raw arrays.

    Preserve every endpoint time, error, final residual and iteration/build
    count. Full SCF energy trajectories and clean-run component clocks remain
    in the hash-bound raw files; separate profiles own detailed attribution.
    Independent cold/moved oracle vectors remain in independent_references.
    """
    point["engines"]["reference"].pop("records", None)
    native = point["engines"]["native"]
    native.pop("native_force_components", None)  # Duplicates the last record.
    for row in native.get("records", []):
        diagnostic = row.get("native_ks_diagnostic")
        if diagnostic:
            history = diagnostic.pop("history", [])
            if history:
                diagnostic["history_iterations"] = [r["iteration"] for r in history]
            row["native_ks_diagnostic"] = {
                key: value
                for key, value in diagnostic.items()
                if key
                in {
                    "ao_order", "coulomb_exchange_work", "fock_builds", "grid_points",
                    "history_iterations", "initial_density_used", "physical_residual_max",
                    "density_change_max", "scf_domain", "tile_points",
                }
            }
        force = row.get("native_force_components")
        if force:
            row["native_force_components"] = {
                key: value
                for key, value in force.items()
                if key
                in {
                    "schema", "source_route", "stationary_integral_derivative_route",
                    "native_integrals_required", "traffic", "resource_bounds",
                    "work_counts", "scf_force_memory",
                }
            }


def write_table(points: list[dict], path: Path) -> None:
    """Keep all phase samples and observed work beside complete median timings."""
    lines = [
        "# Complete endpoint observations",
        "",
        (
            "Seconds, including preparation. Warm cells show median [min, max] over "
            "all five calls. Parentheses retain observed iteration/Fock-build pairs. "
            "A timed-out point can retain a complete earlier phase; missing/failed "
            "phases and source-unqualified points never receive accepted medians."
        ),
        "",
        "| Method | Provider | Atoms | Arm | Point status | Cold | Warm | Moved | Moved-warm |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for point in points:
        native = point["engines"]["native"]
        cells = []
        for phase in ("cold", "warm", "moved", "moved-warm"):
            if point["source_binding"]["status"] != "verified":
                cells.append("source_unqualified")
                continue
            rows = phase_rows(native, phase)
            if native.get("phase_status", {}).get(phase) != "measured":
                cells.append(
                    native.get("phase_status", {}).get(phase, native["status"])
                )
                continue
            values = [row["complete_seconds"] for row in rows]
            cell = f"{median(values):.3f}"
            if len(values) > 1:
                cell += f" [{min(values):.3f}, {max(values):.3f}]"
            work = sorted({(row["iterations"], row["fock_builds"]) for row in rows})
            cells.append(
                cell
                + " ("
                + ", ".join(f"{i}/{f if f is not None else '—'}" for i, f in work)
                + ")"
            )
        lines.append(
            f"| {point['method']} | {point['mode']} | {point['atoms']} | "
            f"{point['arm']} | {native['status']} | " + " | ".join(cells) + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Reject stale identities and recompute acceptance using the shared reducer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    sources, points = {}, []
    planned = planned_points()
    observed = {
        (*path.parts[-5:-2], int(path.parts[-2]))
        for path in (args.campaign / "endpoints").glob("*/*/*/*/native.json")
    }
    for label, mode, method, atoms in sorted(planned | observed):
        path = (
            args.campaign / "endpoints" / label / mode / method / str(atoms) / "native.json"
        )
        arm = source_arm(label)
        if arm not in sources:
            sources[arm] = freeze_source(args.campaign, arm)
        source = sources[arm]
        point = collect_point(path.parent.parent, atoms)
        point.update(arm=label, source_arm=arm, method=method, mode=mode)
        native = point["engines"]["native"]
        point["source_binding"] = source_binding(native, source)
        point["reference_role"] = "retained independent geometry oracle"
        point["accepted_phase_seconds"] = {
            phase: [row["complete_seconds"] for row in phase_rows(native, phase)]
            if point["source_binding"]["status"] == "verified"
            and native.get("phase_status", {}).get(phase) == "measured"
            else []
            for phase in ("cold", "warm", "moved", "moved-warm")
        }
        compact_point(point)
        points.append(point)
    points.sort(key=lambda p: (p["method"], p["mode"], p["atoms"], p["arm"]))
    payload = {
        "schema": "generativeqc.bounded-dft-schedules.v1",
        "planned_populations": len(planned),
        "retention": (
            "Every endpoint sample/work/error and independent geometry oracle is retained. "
            "Detailed SCF trajectories and clean component clocks remain in raw_sha256-bound "
            "ignored inputs; intrusive attribution is published separately. Native input "
            "inventories can be reconstructed from the retained revision and source patch."
        ),
        "sources": sources,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        lzma.compress(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    )
    if args.table:
        write_table(points, args.table)
    print(f"Rechecked {len(points)} populations from {len(sources)} frozen images")


if __name__ == "__main__":
    main()
