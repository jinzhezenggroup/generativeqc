"""Generate target-gated CUDA exact-shell derivative AOT package dispatch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.integral.derivative_aot_registry import (
    radial_inventory_from_payload,
)
from generativeqc_compiler.integral.derivative_cuda_shell_aot import (
    emit_cuda_derivative_shell_aot_header,
)
from generativeqc_compiler.integral.production_profile import resolve_production_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--radial-manifest",
        type=Path,
        default=ROOT / "manifests/derivative_aot_radials.json",
    )
    parser.add_argument(
        "--production-manifest",
        type=Path,
        default=(
            ROOT / "python/generativeqc_compiler/integral/production_shell_classes.json"
        ),
    )
    parser.add_argument("--target-architecture", default="sm_120")
    args = parser.parse_args()

    radial_payload = json.loads(args.radial_manifest.read_text(encoding="utf-8"))
    radials = radial_inventory_from_payload(radial_payload, backend="cuda")
    if not radials:
        raise RuntimeError("CUDA derivative AOT radial inventory is empty")
    profile = resolve_production_profile(
        args.production_manifest, args.target_architecture
    )
    source = emit_cuda_derivative_shell_aot_header(profile, radials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != source:
        args.output.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
