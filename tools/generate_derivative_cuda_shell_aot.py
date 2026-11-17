"""Generate target-gated CUDA exact-shell derivative AOT package dispatch."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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
from generativeqc_compiler.integral.production_policy import (
    cuda_image_inventory,
    profile_overrides,
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
    parser.add_argument("--cuda-image", action="append", default=[])
    parser.add_argument(
        "--disable-shell-packages",
        action="store_true",
        help="Emit empty derivative registries regardless of manifest packaging policy",
    )
    args = parser.parse_args()

    radial_payload = json.loads(args.radial_manifest.read_text(encoding="utf-8"))
    radials = radial_inventory_from_payload(radial_payload, backend="cuda")
    if not radials:
        raise RuntimeError("CUDA derivative AOT radial inventory is empty")
    architectures = sorted(
        {normalize_cuda_architecture(value) for value in args.target_architecture}
    )
    try:
        overrides = profile_overrides(architectures, args.profile_map)
        cuda_image_inventory(architectures, args.cuda_image)
    except ValueError as error:
        parser.error(str(error))
    profiles = tuple(
        resolve_production_profile(
            args.production_manifest,
            architecture,
            overrides.get(architecture, args.profile),
        )
        for architecture in architectures
    )
    if args.disable_shell_packages:
        # The global build switch is authoritative even for a custom portable
        # profile that explicitly opts into derivative compilation.
        profiles = tuple(
            replace(
                profile, packaging=replace(profile.packaging, derivative_shells=False)
            )
            for profile in profiles
        )
    source = emit_cuda_derivative_shell_aot_bundle(profiles, radials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != source:
        args.output.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
