#!/usr/bin/env python3
"""Emit the packaged s/p/d stationary CPU first-derivative inventory."""

from __future__ import annotations

import argparse
from pathlib import Path

from vibeqc_compiler.integral.first_derivative_schedule import (
    CPU_AOT_SHARDS,
    derivative_cpu_aot_sources,
)


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    sources = derivative_cpu_aot_sources()
    if len(sources) != CPU_AOT_SHARDS:
        raise RuntimeError("stationary CPU derivative AOT shard contract drift")
    for shard, (_, source) in enumerate(sources):
        _write_if_changed(
            args.output_directory / f"vibeqc_stationary_cpu_derivative_{shard}.cpp",
            source,
        )


if __name__ == "__main__":
    main()
