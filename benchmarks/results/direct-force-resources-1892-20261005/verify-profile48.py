"""Authenticate completed single-warm traces without treating them as clean timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def digest(path: Path) -> str:
    """Hash large profiler databases without loading them into host memory."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def summarize(root: Path) -> dict:
    """Require the measured binary, numerical gates and actual force-range launches."""
    if (root / "job.exit").read_text().strip() != "0":
        raise ValueError("profile job did not complete successfully")
    hashes = {}
    for inventory in ("evidence.sha256", "drivers.sha256"):
        for line in (root / inventory).read_text().splitlines():
            expected, remote_path = line.split("  ", 1)
            name = Path(remote_path).name
            if digest(root / name) != expected:
                raise ValueError(f"profile receipt changed: {name}")
            hashes[name] = expected
    arms = {}
    for arm in ("baseline", "default", "128"):
        payload = json.loads((root / f"{arm}.json").read_text())
        rows = payload["records"]
        if [row["phase"] for row in rows] != ["cold", "prime", "profiled-warm"]:
            raise ValueError(f"partial profile journal: {arm}")
        if any(
            not row["gate"]
            or row["energy_error"] > 1e-8
            or row["force_error"] > 1e-7
            for row in rows
        ):
            raise ValueError(f"independent numerical profile gate failed: {arm}")
        if rows[-1]["iterations"] != 1:
            raise ValueError(f"unmatched profiled SCF branch: {arm}")
        expected_binary = (
            "daf5d5b0f3bdca15bdf0e56243f0163455767333c2fa8ff6b9285f4a561ce2af"
            if arm == "baseline"
            else "cca2d24221b6c040a32907b70cb77590526a6dd891da1ba14352e9b8b4472098"
        )
        if payload["native_build"]["library_sha256"] != expected_binary:
            raise ValueError(f"unexpected profiled binary: {arm}")
        with sqlite3.connect(f"file:{root / f'{arm}.sqlite'}?mode=ro", uri=True) as db:
            ranges = db.execute(
                "SELECT e.start,e.end,coalesce(e.text,s.value) FROM NVTX_EVENTS e "
                "LEFT JOIN StringIds s ON e.textId=s.id "
                "WHERE coalesce(e.text,s.value) LIKE 'p0b/%'"
            ).fetchall()
            if sorted(row[2] for row in ranges) != [
                "p0b/profiled-warm",
                "p0b/public-stationary-force",
            ]:
                raise ValueError(f"unexpected executed NVTX ranges: {arm}")
            force_start, force_end, _ = next(
                row for row in ranges if row[2] == "p0b/public-stationary-force"
            )
            kernels = db.execute(
                "SELECT s.value,k.blockX,k.gridX,k.registersPerThread,"
                "k.staticSharedMemory,k.localMemoryPerThread,k.start,k.end "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON k.demangledName=s.id "
                "WHERE s.value LIKE '%bounded_direct_shell_quartet_kernel%' "
                "AND k.start>=? AND k.end<=?",
                (force_start, force_end),
            ).fetchall()
        if len(kernels) != 1:
            raise ValueError(f"unexpected generic force launch inventory: {arm}")
        kernel = kernels[0]
        if "DirectScreeningPurpose)1, (bool)1" not in kernel[0]:
            raise ValueError(f"selected launch is not the generic force consumer: {arm}")
        if kernel[1] != (128 if arm == "128" else 256):
            raise ValueError(f"CTA experiment was not actually executed: {arm}")
        arms[arm] = {
            "library_sha256": expected_binary,
            "executed_nvtx_ranges": [row[2] for row in ranges],
            "generic_force": {
                "symbol": kernel[0],
                "launches": len(kernels),
                "block_x": kernel[1],
                "grid_x": kernel[2],
                "registers_per_thread": kernel[3],
                "static_shared_bytes": kernel[4],
                "local_bytes_per_thread": kernel[5],
                "gpu_interval_seconds": (kernel[7] - kernel[6]) / 1e9,
            },
            "maximum_energy_error": max(row["energy_error"] for row in rows),
            "maximum_force_error": max(row["force_error"] for row in rows),
        }
    return {
        "scope": "single-warm Nsight Systems attribution, not clean endpoint timing",
        "job": 6111,
        "exit": 0,
        "atoms": 48,
        "method": "pbe0",
        "arms": arms,
        "evidence_sha256": hashes,
        "hardware_counters": "not measured; no occupancy/stall/local-traffic claim",
        "caveat": "CUDA Event tracing is intrusive; use separate job 6112 for speed claims",
    }


def main() -> None:
    """Write a derived receipt separately from immutable profiler inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.write_text(
        json.dumps(summarize(arguments.profile), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
