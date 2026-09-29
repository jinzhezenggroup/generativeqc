"""Write the installed identity record for one stationary CUDA AOT artifact."""

from __future__ import annotations

import argparse
import json
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.integral.first_derivative_schedule import (
    CUDA_REQUESTS_PER_UNIT,
    derivative_cuda_sources,
)
from generativeqc_compiler.method.stationary_cuda import (
    QUALIFIED_SPD_AOT_SHARD_WIDTH,
    QUALIFIED_SPD_AOT_SHARDS,
    QUALIFIED_SPD_COMPONENTS,
    QUALIFIED_STATIONARY_AOT_PROFILE_NAMES,
    _qualified_aot_profile,
    emit_stationary_component_aot_wrapper_cuda,
    emit_stationary_profile_component_aot_wrapper_cuda,
    stationary_aot_contract_identity,
    stationary_aot_plan_identity,
    stationary_aot_profile_contract_identity,
    stationary_aot_profile_plan_identity,
)


def _code_objects(values: list[str]) -> list[dict[str, str]]:
    result = set()
    for value in values:
        number = value.removesuffix("-real").removesuffix("-virtual")
        if not number.isdigit():
            raise ValueError(f"invalid stationary CUDA architecture: {value}")
        kinds = (
            ("cubin",)
            if value.endswith("-real")
            else ("ptx",)
            if value.endswith("-virtual")
            else ("cubin", "ptx")
        )
        result.update((f"sm_{number}", kind) for kind in kinds)
    return [
        {"architecture": architecture, "kind": kind}
        for architecture, kind in sorted(result)
    ]


def _expected_component_sources(
    functional: int | None,
    *,
    spin: str | None,
    profile: str | None,
    iterations: int,
) -> tuple[str, tuple[str, ...]]:
    units = derivative_cuda_sources(QUALIFIED_SPD_COMPONENTS)
    if (
        CUDA_REQUESTS_PER_UNIT != QUALIFIED_SPD_AOT_SHARD_WIDTH
        or len(units) != QUALIFIED_SPD_AOT_SHARDS
    ):
        raise RuntimeError("stationary s/p/d AOT shard contract drift")
    wrapper = (
        emit_stationary_profile_component_aot_wrapper_cuda(
            profile, iterations=iterations
        )
        if profile is not None
        else emit_stationary_component_aot_wrapper_cuda(
            typing.cast("int", functional),
            spin=typing.cast("str", spin),
            iterations=iterations,
        )
    )
    return (wrapper, tuple(source for _, source in units))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--primitive-source", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--functional", type=int, choices=(0, 1, 2))
    parser.add_argument("--spin", choices=("unpolarized", "polarized"))
    parser.add_argument("--profile", choices=QUALIFIED_STATIONARY_AOT_PROFILE_NAMES)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--component-domain", choices=("sp", "spd"), default="sp")
    parser.add_argument("--architecture", action="append", default=[])
    parser.add_argument("--compile-architecture", action="append", default=[])
    args = parser.parse_args()
    if args.profile is None:
        if args.functional is None or args.spin is None:
            parser.error("--profile or --functional/--spin is required")
        functional = args.functional
        spin = args.spin
        profile = None
    else:
        if args.functional is not None or args.spin is not None:
            parser.error("--profile cannot be combined with --functional/--spin")
        selected = _qualified_aot_profile(args.profile)
        functional = selected.functional
        spin = selected.spin
        profile = args.profile
    code_objects = _code_objects(args.compile_architecture)
    architectures = sorted({item["architecture"] for item in code_objects})
    if not architectures or architectures != sorted(set(args.architecture)):
        raise ValueError("stationary AOT target/code-object architecture mismatch")

    component_domain = None
    source_identity = canonical_hash(args.source.read_text())
    schema = "generativeqc.stationary-cuda-aot.v2"
    component_payload: dict[str, object] = {}
    if args.component_domain == "spd":
        if len(args.primitive_source) != QUALIFIED_SPD_AOT_SHARDS:
            raise ValueError(
                "stationary s/p/d AOT manifest requires the complete primitive shard set"
            )
        component_domain = QUALIFIED_SPD_COMPONENTS
        actual_identity = canonical_hash(
            {
                "wrapper": canonical_hash(args.source.read_text()),
                "primitives": tuple(
                    canonical_hash(path.read_text()) for path in args.primitive_source
                ),
            }
        )
        expected_wrapper, expected_primitives = _expected_component_sources(
            functional,
            spin=spin,
            profile=profile,
            iterations=args.iterations,
        )
        source_identity = canonical_hash(
            {
                "wrapper": canonical_hash(expected_wrapper),
                "primitives": tuple(
                    canonical_hash(source) for source in expected_primitives
                ),
            }
        )
        if actual_identity != source_identity:
            raise ValueError("stationary s/p/d generated source identity mismatch")
        schema = "generativeqc.stationary-cuda-aot.v3"
        component_payload = {
            "component_domain": list(component_domain),
            "primitive_shard_width": QUALIFIED_SPD_AOT_SHARD_WIDTH,
            "primitive_shards": QUALIFIED_SPD_AOT_SHARDS,
            "primitive_source_sha256": [
                file_hash(path) for path in args.primitive_source
            ],
        }
    elif args.primitive_source:
        raise ValueError("--primitive-source requires --component-domain spd")

    payload = {
        "schema": schema,
        "functional": functional,
        "spin": spin,
        **({"profile": profile} if profile is not None else {}),
        "plan_identity": (
            stationary_aot_profile_plan_identity(profile)
            if profile is not None
            else stationary_aot_plan_identity(functional, spin=spin)
        ),
        "partition_iterations": args.iterations,
        **component_payload,
        "architectures": architectures,
        "compile_architectures": sorted(set(args.compile_architecture)),
        "code_objects": code_objects,
        "source_identity": source_identity,
        "source_sha256": file_hash(args.source),
        "contract_identity": (
            stationary_aot_profile_contract_identity(
                profile,
                iterations=args.iterations,
                component_domain=component_domain,
            )
            if profile is not None
            else stationary_aot_contract_identity(
                functional,
                spin=spin,
                iterations=args.iterations,
                component_domain=component_domain,
            )
        ),
        "binary_sha256": file_hash(args.library),
        "binary_bytes": args.library.stat().st_size,
        "compile_contract": {
            "fp64": True,
            "fmad": False,
            "relaxed_constexpr": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if not args.output.exists() or args.output.read_text() != text:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
