"""Emit the packaged s/p/d stationary CPU first-derivative inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.first_derivative_schedule import (
    CPU_AOT_SHARDS,
    derivative_cpu_aot_sources,
)

from tools import stationary_cpu_source_receipt as source_receipt


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    if path.exists() and path.read_bytes() == encoded:
        return
    path.write_bytes(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    identity = source_receipt.input_identity()
    sources = derivative_cpu_aot_sources()
    if len(sources) != CPU_AOT_SHARDS:
        raise RuntimeError("stationary CPU derivative AOT shard contract drift")
    for shard, (_, source) in enumerate(sources):
        _write_if_changed(
            args.output_directory
            / f"generativeqc_stationary_cpu_derivative_{shard}.cpp",
            source,
        )
    source_receipt.publish(
        args.output_directory, identity, tuple(source for _, source in sources)
    )


if __name__ == "__main__":
    main()
