"""Generate the independent stationary primitive reference once per build.

This is deliberately uncached: generated sources may use semantic source reuse,
but every build retains one independent scientific lowering verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.integral.first_derivative_native import (
    emit_first_derivative_cuda,
)
from generativeqc_compiler.integral.first_derivative_schedule import (
    CUDA_REQUESTS_PER_UNIT,
    derivative_cuda_sources,
)
from generativeqc_compiler.method.stationary_cuda import (
    QUALIFIED_SPD_AOT_SHARD_WIDTH,
    QUALIFIED_SPD_AOT_SHARDS,
    QUALIFIED_SPD_COMPONENTS,
    qualified_sp_requests,
)

REFERENCE_SCHEMA = "generativeqc.stationary-aot-expected-primitives.v1"


def build_expected_primitives() -> dict[str, object]:
    """Hash independently lowered source bytes; do not read the source cache."""
    shards = derivative_cuda_sources(QUALIFIED_SPD_COMPONENTS)
    if (
        CUDA_REQUESTS_PER_UNIT != QUALIFIED_SPD_AOT_SHARD_WIDTH
        or len(shards) != QUALIFIED_SPD_AOT_SHARDS
    ):
        raise RuntimeError("stationary s/p/d AOT shard contract drift")
    return {
        "schema": REFERENCE_SCHEMA,
        "sp": [canonical_hash(emit_first_derivative_cuda(qualified_sp_requests()))],
        "spd": [canonical_hash(source) for _, source in shards],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_expected_primitives()
    rendered = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != rendered:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
