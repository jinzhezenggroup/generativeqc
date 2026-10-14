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


def freeze_source(campaign: Path, arm: str) -> dict:
    """Bind dirty pilots explicitly; a Git revision alone cannot identify them."""
    source = (campaign / f"{arm}-source").resolve()

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(source), *args], text=True)

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
        "native_input_sha256": inputs,
        "installed_library_sha256": digest(campaign / arm / "lib/libgenerativeqc.so"),
    }


def write_table(points: list[dict], path: Path) -> None:
    """Keep all phase samples and observed work beside complete median timings."""
    lines = [
        "# Complete endpoint observations",
        "",
        (
            "Seconds, including preparation. Warm cells show median [min, max] over "
            "all five calls. Parentheses retain observed iteration/Fock-build pairs. "
            "Failed or missing populations never receive a complete-population median."
        ),
        "",
        "| Method | Provider | Atoms | Arm | Cold | Warm | Moved | Moved-warm |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for point in points:
        native = point["engines"]["native"]
        cells = []
        for phase in ("cold", "warm", "moved", "moved-warm"):
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
            f"{point['arm']} | " + " | ".join(cells) + " |"
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
    for path in sorted((args.campaign / "endpoints").glob("*/*/*/*/native.json")):
        label, mode, method, atoms = path.parts[-5:-1]
        arm = source_arm(label)
        if arm not in sources:
            sources[arm] = freeze_source(args.campaign, arm)
        source = sources[arm]
        point = collect_point(path.parent.parent, int(atoms))
        point.update(arm=label, source_arm=arm, method=method, mode=mode)
        native = point["engines"]["native"]
        build = native.get("native_build")
        if build:
            assert (
                build["probe"]["source_identity"] == source["native_source_identity"]
            ), path
            assert build["library_sha256"] == source["installed_library_sha256"], path
        for name, checksum in native.get("source_file_sha256", {}).items():
            assert digest(args.campaign / f"{arm}-source" / name) == checksum, (
                path,
                name,
            )
        point["reference_role"] = "retained independent geometry oracle"
        point["accepted_phase_seconds"] = {
            phase: [row["complete_seconds"] for row in phase_rows(native, phase)]
            if native["phase_status"][phase] == "measured"
            else []
            for phase in ("cold", "warm", "moved", "moved-warm")
        }
        points.append(point)
    points.sort(key=lambda p: (p["method"], p["mode"], p["atoms"], p["arm"]))
    payload = {
        "schema": "generativeqc.bounded-dft-schedules.v1",
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
