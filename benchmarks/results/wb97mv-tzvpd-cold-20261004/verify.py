"""Recheck every retained E/F call and source lifecycle without loading CUDA."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Imports follow this trusted checkout, independently of the selected data
# directory, caller working directory, or inherited PYTHONPATH.
_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_ROOT), str(_ROOT / "python")]

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record
from tools.generativeqc_validation.retention import safe_relative


def verify(directory: Path) -> dict:
    """Check bundle consistency, then apply the trusted checkout's scientific gates."""
    manifest = json.loads((directory / "publication.json").read_text())
    files = {
        safe_relative(e["path"]): (directory / safe_relative(e["path"])).read_bytes()
        for e in manifest["files"]
    }
    validate_publication(manifest, files)
    samples = load_publication_record(directory, role="samples")
    expected = load_publication_record(directory, role="summary", name="summary.json")
    observed = {}
    variants = ("reference", "none", "lda16")
    with tempfile.TemporaryDirectory(prefix="tzvpd-evidence-") as workspace:
        for index, (atoms, point) in enumerate(samples["points"].items()):
            if (
                not isinstance(atoms, str)
                or not atoms.isascii()
                or not atoms.isdecimal()
                or int(atoms) <= 0
                or str(int(atoms)) != atoms
            ):
                raise ValueError("invalid canonical atom-count label")
            if set(point["reports"]) != set(variants) or set(point["outcomes"]) != set(
                variants
            ):
                raise ValueError(
                    "point requires exactly reference, none and lda16 reports/outcomes"
                )
            # Publication labels are data, never workspace filenames. Only
            # local indices and the fixed protocol variants determine paths.
            target = Path(workspace) / str(index)
            target.mkdir()
            for variant in variants:
                report = point["reports"][variant]
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
                str(Path(__file__).resolve().with_name("validate-point.py")),
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
