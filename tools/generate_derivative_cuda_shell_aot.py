"""Generate target-gated CUDA exact-shell derivative AOT package dispatch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.common.cuda_target import normalize_cuda_architecture
from generativeqc_compiler.integral.derivative_aot_registry import (
    radial_inventory_from_payload,
)
from generativeqc_compiler.integral.derivative_cuda_shell_aot import (
    emit_cuda_derivative_shell_aot_bundle,
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
    parser.add_argument("--target-architecture", action="append", required=True)
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--profile-map", action="append", default=[])
    args = parser.parse_args()

    radial_payload = json.loads(args.radial_manifest.read_text(encoding="utf-8"))
    radials = radial_inventory_from_payload(radial_payload, backend="cuda")
    if not radials:
        raise RuntimeError("CUDA derivative AOT radial inventory is empty")
    architectures = sorted(
        {normalize_cuda_architecture(value) for value in args.target_architecture}
    )
    overrides = {}
    for entry in args.profile_map:
        architecture, separator, profile = entry.partition("=")
        if not separator or not profile:
            parser.error("--profile-map requires sm_XX=profile")
        architecture = normalize_cuda_architecture(architecture)
        if architecture not in architectures or architecture in overrides:
            parser.error("--profile-map requires a unique configured target")
        overrides[architecture] = profile
    profiles = tuple(
        resolve_production_profile(
            args.production_manifest,
            architecture,
            overrides.get(architecture, args.profile),
        )
        for architecture in architectures
    )
    source = emit_cuda_derivative_shell_aot_bundle(profiles, radials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != source:
        args.output.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
