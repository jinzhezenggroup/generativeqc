"""Emit the packaged WB97M-V s/p SR/LR CPU derivative inventories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.integral.first_derivative_schedule import (
    CPU_RSH_AOT_FAMILIES,
    CPU_RSH_AOT_SHARDS,
    derivative_cpu_rsh_aot_sources,
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

    for family in CPU_RSH_AOT_FAMILIES:
        sources = derivative_cpu_rsh_aot_sources(family)
        if len(sources) != CPU_RSH_AOT_SHARDS:
            raise RuntimeError("stationary CPU RSH derivative AOT shard contract drift")
        for shard, (_, source) in enumerate(sources):
            _write_if_changed(
                args.output_directory
                / f"vibeqc_wb97mv_{family}_cpu_derivative_{shard}.cpp",
                source,
            )


if __name__ == "__main__":
    main()
