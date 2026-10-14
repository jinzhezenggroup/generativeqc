"""Opt-in integrity/phase audit and same-environment figure byte replay.

This validates retained publication evidence. It does not recompute omitted
native force vectors or qualify a later source revision/GPU build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import tempfile
from pathlib import Path

from tools.render_readme_dft import phase_rows, render


def verify(*, rerender: bool = False) -> None:
    """Check exact retained bytes, complete phase counts and scalar acceptance."""
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    for name, receipt in manifest["files"].items():
        payload = (root / name).read_bytes()
        assert len(payload) == receipt["bytes"], name
        assert hashlib.sha256(payload).hexdigest() == receipt["sha256"], name
    # Source inventories are shared losslessly outside the readable manifest;
    # storage compaction must preserve every original provenance field.
    followup = manifest["benchmark_issue_followup"]
    audit = lzma.decompress((root / followup["source_audit"]).read_bytes())
    assert len(audit) == followup["source_audit_uncompressed"]["bytes"]
    assert (
        hashlib.sha256(audit).hexdigest()
        == followup["source_audit_uncompressed"]["sha256"]
    )
    assert isinstance(json.loads(audit), dict)
    for name, prefix in (("summary.json.xz", ""), ("df-summary.json.xz", "df_")):
        summary = json.loads(lzma.decompress((root / name).read_bytes()))
        complete = observations = 0
        for points in summary["methods"].values():
            for point in points:
                native = point["engines"]["native"]
                if native["status"] == "measured":
                    assert len(native["records"]) == 12
                    complete += 1
                    observations += len(native["records"])
                for phase, expected in (
                    ("cold", 1),
                    ("warm", 5),
                    ("moved", 1),
                    ("moved-warm", 5),
                ):
                    rows = phase_rows(native, phase)
                    if not rows:
                        continue
                    assert len(rows) == expected, (name, point["atoms"], phase)
                    for row in rows:
                        if row["gate"] is None:
                            # Original independent WB observations are explicitly
                            # unchecked and keep null errors; no retroactive pass.
                            assert row["energy_error"] is row["force_error"] is None
                            continue
                        assert row["gate"] is True
                        assert 0 <= row["energy_error"] <= 1e-8
                        assert 0 <= row["force_error"] <= 1e-7
        assert complete == manifest[prefix + "acceptance"]["native_points"]
        assert observations == manifest[prefix + "acceptance"]["native_observations"]
        if rerender:
            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory)
                render(summary, destination)
                for output in destination.iterdir():
                    assert output.read_bytes() == (root / output.name).read_bytes(), (
                        output.name
                    )
    print(
        "Publication hashes, complete phases and scalar gates verified"
        + ("; figures replay byte-for-byte." if rerender else ".")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerender", action="store_true")
    verify(rerender=parser.parse_args().rerender)
