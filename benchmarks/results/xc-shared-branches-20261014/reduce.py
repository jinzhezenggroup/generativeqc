"""Recheck raw endpoint arrays and publish compact, source-bound observations.

Run with PYTHONPATH=.:python from the repository root. The campaign directory
contains endpoints/{baseline,candidate}/{none,cuda}/{method}/{atoms}, plus clean
baseline-source and candidate-source checkouts. Raw arrays remain there.
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


def write_table(points: list[dict], path: Path) -> None:
    """Show complete phases and all SCF work without timing partial populations."""
    lines = [
        "# Paired complete energy/force observations",
        "",
        (
            "Seconds; warm cells show median [minimum, maximum] over all five calls. "
            "Parentheses give every observed iteration/Fock-build pair. "
            "Incomplete populations retain their returned-call count without a median."
        ),
        "",
        "| Method | Provider | Atoms | Arm | Cold | Warm | Moved | Moved-warm |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(
        points, key=lambda p: (p["method"], p["mode"], p["atoms"], p["arm"])
    )
    for point in ordered:
        native = point["engines"]["native"]
        cells = []
        for phase in ("cold", "warm", "moved", "moved-warm"):
            rows = (
                phase_rows(native, phase)
                if native["phase_status"][phase] == "measured"
                else []
            )
            if not rows:
                count = sum(row["phase"] == phase for row in native["records"])
                expected = 5 if "warm" in phase else 1
                cells.append(f"{native['phase_status'][phase]} ({count}/{expected})")
                continue
            values = [row["complete_seconds"] for row in rows]
            timing = f"{median(values):.3f}"
            if len(rows) > 1:
                timing += f" [{min(values):.3f}, {max(values):.3f}]"
            work = sorted({(row["iterations"], row["fock_builds"]) for row in rows})
            cells.append(timing + " (" + ", ".join(f"{i}/{f}" for i, f in work) + ")")
        provider = "DF" if point["mode"] == "cuda" else "direct"
        lines.append(
            f"| {point['method']} | {provider} | {point['atoms']} | {point['arm']} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "| Method | Provider | Atoms | Arm | Accepted / requested calls | Max energy error (Eh) | Max force error (Eh/Bohr) | Slurm job |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for point in ordered:
        native = point["engines"]["native"]
        accepted = [row for row in native["records"] if row.get("gate") is True]
        provider = "DF" if point["mode"] == "cuda" else "direct"
        errors = {
            name: f"{max(row[name] for row in accepted):.3e}" if accepted else "—"
            for name in ("energy_error", "force_error")
        }
        lines.append(
            f"| {point['method']} | {provider} | {point['atoms']} | {point['arm']} | "
            f"{len(accepted)}/12 | {errors['energy_error']} | {errors['force_error']} | "
            f"{native['scheduler']['SLURM_JOB_ID']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Reject stale native/consumer identities before reducing accepted phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args()
    sources = {}
    points = []
    raw_native = {}
    for arm in ("baseline", "candidate"):
        source = (args.campaign / f"{arm}-source").resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        assert not subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ), f"dirty measured source: {arm}"
        identity = _source_identity(
            source, _inventory(source, source / "cmake/GenerativeQCSourceIdentity.json")
        )
        sources[arm] = {"revision": revision, "native_source_identity": identity}
        for path in sorted(
            (args.campaign / "endpoints" / arm).glob("*/*/*/native.json")
        ):
            mode, method, atoms = path.parts[-4:-1]
            raw_native[arm, mode, method, atoms] = json.loads(path.read_text())
            point = collect_point(path.parent.parent, int(atoms))
            point.update(arm=arm, method=method, mode=mode)
            native = point["engines"]["native"]
            assert native["native_build"]["probe"]["source_identity"] == identity
            for name, digest in native["source_file_sha256"].items():
                assert (
                    hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
                )
            # Historical oracle timings remain identified as historical. Only
            # native calls collected in this campaign enter the paired table.
            point["reference_role"] = "retained independent geometry oracle"
            point["accepted_phase_seconds"] = {
                phase: [row["complete_seconds"] for row in phase_rows(native, phase)]
                if native["phase_status"][phase] == "measured"
                else []
                for phase in ("cold", "warm", "moved", "moved-warm")
            }
            points.append(point)
    paired = []
    for (arm, mode, method, atoms), candidate in raw_native.items():
        if arm != "candidate":
            continue
        baseline = raw_native.get(("baseline", mode, method, atoms))
        if baseline is None:
            continue
        # Compare every common observation, including returned calls from a
        # timed-out population. Phase completeness remains separately gated.
        key = lambda row: (row["phase"], row["geometry"], row["repeat"])
        before = {key(row): row for row in baseline["records"]}
        pairs = [
            (before[key(row)], row)
            for row in candidate["records"]
            if key(row) in before
        ]
        if not pairs:
            continue
        paired.append(
            {
                "mode": mode,
                "method": method,
                "atoms": int(atoms),
                "compared_calls": len(pairs),
                "max_energy_difference": max(
                    abs(a["energy"] - b["energy"]) for a, b in pairs
                ),
                "max_force_difference": max(
                    abs(x - y)
                    for a, b in pairs
                    for fa, fb in zip(a["forces"], b["forces"], strict=True)
                    for x, y in zip(fa, fb, strict=True)
                ),
                "identical_scf_work": all(
                    (a["iterations"], a["fock_builds"])
                    == (b["iterations"], b["fock_builds"])
                    for a, b in pairs
                ),
            }
        )
    payload = {
        "schema": "generativeqc.xc-shared-branches-endpoints.v1",
        "sources": sources,
        "points": points,
        "paired_observations": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        lzma.compress(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    )
    if args.table is not None:
        write_table(points, args.table)
    print(f"Rechecked {len(points)} endpoint populations; wrote {args.output}")


if __name__ == "__main__":
    main()
