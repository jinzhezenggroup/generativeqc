"""Recheck every retained E/F call and source lifecycle without loading CUDA."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record
from tools.generativeqc_validation.retention import safe_relative


def verify(directory: Path) -> dict:
    """Authenticate the bundle, then rerun scientific gates on exact report bytes."""
    manifest = json.loads((directory / "publication.json").read_text())
    files = {
        safe_relative(e["path"]): (directory / safe_relative(e["path"])).read_bytes()
        for e in manifest["files"]
    }
    validate_publication(manifest, files)
    samples = load_publication_record(directory, role="samples")
    expected = load_publication_record(directory, role="summary", name="summary.json")
    observed = {}
    with tempfile.TemporaryDirectory(prefix="tzvpd-evidence-") as workspace:
        for atoms, point in samples["points"].items():
            target = Path(workspace) / atoms
            target.mkdir()
            for variant, report in point["reports"].items():
                (target / f"{variant}.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
                (target / f"{variant}.outcome").write_text(
                    json.dumps(point["outcomes"][variant])
                )
            (target / "source-identity.json").write_text(json.dumps(point["identity"]))
            command = [
                sys.executable,
                *(["-O"] if sys.flags.optimize else []),
                str(directory / "validate-point.py"),
                str(target),
            ]
            checked = subprocess.run(
                command, check=False, text=True, capture_output=True
            )
            if checked.returncode:
                raise ValueError(
                    f"{atoms}-atom endpoint verification failed:\n{checked.stderr}"
                )
            observed[atoms] = json.loads(checked.stdout)
    if observed != expected:
        raise ValueError(
            "retained summary differs from independently rechecked reports"
        )
    return {
        "accepted_atoms": list(map(int, observed)),
        "accepted_endpoint_calls": sum(
            len(report["records"])
            for point in samples["points"].values()
            for report in point["reports"].values()
        ),
        "scope": "numerical acceptance and observed timings; no default promotion",
    }


if __name__ == "__main__":
    directory = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    )
    print(json.dumps(verify(directory), indent=2))
