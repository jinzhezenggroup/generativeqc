"""Opt-in byte-preservation reproduction for the source-matched AOT audit only.

This historical receipt is not a required CI fingerprint. The baseline restores
only the two edited scientific emitter owners; all their other inputs were
unchanged in this change. Refuse other revisions rather than silently comparing
a historical emitter against newer mathematics.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.common.paths import source_hashes
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.integral import first_derivative_schedule as primitive
from generativeqc_compiler.method import stationary_cuda as stationary

BASELINE = "a16443baced95f5434c31b74cd993c970c39927b"


def main() -> None:
    """Emit scientific source hashes in one isolated baseline/candidate process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "candidate"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    expected = json.loads(
        gzip.decompress(Path(__file__).with_name("census.json.gz").read_bytes())
    )
    actual = canonical_hash(
        source_hashes(
            "common",
            "integral",
            "tensor",
            "xc",
            "dft",
            "method",
            assets=stationary.STATIONARY_AOT_ASSETS,
        )
    )
    if actual != expected["compiler_inputs_sha256"]:
        raise RuntimeError("one-off comparison requires the audited compiler inputs")
    if args.mode == "baseline":
        for module in (primitive, stationary):
            path = Path(module.__file__).relative_to(ROOT)
            original = subprocess.check_output(
                ["git", "show", f"{BASELINE}:{path}"], cwd=ROOT, text=True
            )
            # Restore definitions under their canonical module identity, so
            # lazy compiler imports see the same baseline scientific owner.
            exec(compile(original, str(path), "exec"), module.__dict__)  # noqa: S102 -- pinned local Git source
    from generativeqc_compiler.method.stationary_native_cuda import (
        NATIVE_STATIONARY_CUDA_PROFILES,
    )

    records = {}

    def record(name: str, source: str) -> None:
        records[name] = {
            "bytes": len(source.encode()),
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
        }

    record(
        "shared-sp",
        primitive.emit_first_derivative_cuda(stationary.qualified_sp_requests()),
    )
    for index, (_, source) in enumerate(
        primitive.derivative_cuda_sources(stationary.QUALIFIED_SPD_COMPONENTS)
    ):
        record(f"shared-spd-{index}", source)
    for profile in stationary.QUALIFIED_STATIONARY_AOT_PROFILE_NAMES:
        record(
            f"legacy-sp-{profile}",
            stationary.emit_stationary_profile_aot_wrapper_cuda(profile),
        )
        record(
            f"legacy-spd-{profile}",
            stationary.emit_stationary_profile_component_aot_wrapper_cuda(profile),
        )
        if profile in NATIVE_STATIONARY_CUDA_PROFILES:
            record(
                f"native-scientific-wrapper-{profile}",
                stationary.emit_stationary_profile_aot_wrapper_cuda(
                    profile, include_reduction=False
                ),
            )
    args.output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(args.mode, len(records), sum(row["bytes"] for row in records.values()))


if __name__ == "__main__":
    main()
