"""Generate bounded s/p/d CPU value ERIs from the shared scientific DAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from generativeqc_compiler.integral.eri_cpu import emit_eri_cpu, eri_cpu_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    outputs = [(args.output, emit_eri_cpu())]
    if args.inventory:
        outputs.append(
            (
                args.inventory,
                json.dumps(eri_cpu_inventory(), sort_keys=True, indent=2) + "\n",
            )
        )
    for path, source in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != source:
            path.write_text(source)


if __name__ == "__main__":
    main()
