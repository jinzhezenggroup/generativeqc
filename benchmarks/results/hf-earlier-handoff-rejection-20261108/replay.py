"""Apply the identical HF96 independent-oracle and paired-warm replay gates."""

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replay(directory: Path = ROOT) -> dict:
    """Reuse validation logic, never the older experiment's identity or samples."""
    validator = runpy.run_path(
        str(ROOT.parent / "hf-fp64-predictor-rejection-20261108" / "replay.py")
    )["replay"]
    return validator(directory)


if __name__ == "__main__":
    support = runpy.run_path(
        str(
            ROOT.parent
            / "hf-screened-handoff-rejections-20261108"
            / "replay_support.py"
        )
    )
    print(json.dumps(support["compact_summary"](replay()), indent=2, allow_nan=False))
