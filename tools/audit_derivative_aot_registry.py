"""Audit packaged derivative AOT identities, source inventory, and symbols."""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.common.provenance import file_hash
from generativeqc_compiler.integral.derivative_aot_registry import (
    AOT_ANGULAR_DOMAIN,
    AOT_COMPONENT_CONTRACTION_CONTRACT,
    AOT_COMPONENT_OUTPUT_CONTRACT,
    AOT_SPIN_CONTRACT,
    AOT_WEIGHTED_CONTRACTION_CONTRACT,
    AOT_WEIGHTED_OUTPUT_CONTRACT,
    DerivativeAotBundleKey,
    DerivativeAotPackageKey,
    component_groups,
    make_key,
    radial_inventory_from_payload,
)
from generativeqc_compiler.integral.first_derivative_schedule import (
    CPU_AOT_COMPONENTS,
    CPU_AOT_SHARDS,
    cpu_aot_symbol,
    derivative_cpu_aot_sources,
)
from generativeqc_compiler.integral.range_separation import CoulombKernel
from generativeqc_compiler.integral.rsh_cpu_aot import program_source


def _source_statistics(sources: tuple[str, ...]) -> dict[str, int]:
    sizes = tuple(len(source.encode("utf-8")) for source in sources)
    return {
        "translation_units": len(sizes),
        "generated_source_bytes": sum(sizes),
        "largest_translation_unit_bytes": max(sizes, default=0),
    }


def _range_programs(
    radials: tuple[CoulombKernel, ...],
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    records: list[dict[str, object]] = []
    sources: list[str] = []
    for radial in radials:
        for angular_value in product(AOT_ANGULAR_DOMAIN, repeat=4):
            angular = tuple(angular_value)
            for group_index in range(len(component_groups(angular))):
                source, components, prefix = program_source(
                    radial, angular, group_index
                )
                key = make_key(radial, angular, group_index, backend="cpu")
                sources.append(source)
                package_key = DerivativeAotPackageKey(
                    backend="cpu",
                    target="native-host",
                    scientific_identity=key.identity,
                    output_contract=AOT_WEIGHTED_OUTPUT_CONTRACT,
                    spin_contract=AOT_SPIN_CONTRACT,
                    contraction_contract=AOT_WEIGHTED_CONTRACTION_CONTRACT,
                )
                records.append(
                    {
                        "identity": key.identity,
                        "identity_payload": key.to_payload(),
                        "package_identity": package_key.identity,
                        "package_identity_payload": package_key.to_payload(),
                        "radial": radial.to_payload(),
                        "angular": list(angular),
                        "group_index": group_index,
                        "component_indices": list(components),
                        "entry_prefix": prefix,
                        "source_bytes": len(source.encode("utf-8")),
                    }
                )
    return records, tuple(sources)


def _symbol_status(
    library_path: Path | None,
    *,
    range_records: list[dict[str, object]],
) -> dict[str, object] | None:
    if library_path is None:
        return None
    library_path = library_path.resolve()
    library = ct.CDLL(str(library_path))
    missing = []
    for shard in range(CPU_AOT_SHARDS):
        symbol = cpu_aot_symbol(shard)
        if not hasattr(library, symbol):
            missing.append(symbol)
    for record in range_records:
        prefix = str(record["entry_prefix"])
        for suffix in ("identity_v2", "create_v2", "run_v2"):
            symbol = f"{prefix}_{suffix}"
            if not hasattr(library, symbol):
                missing.append(symbol)
    source_identity = None
    if hasattr(library, "generativeqc_get_source_identity"):
        library.generativeqc_get_source_identity.argtypes = []
        library.generativeqc_get_source_identity.restype = ct.c_char_p
        value = library.generativeqc_get_source_identity()
        source_identity = None if value is None else value.decode()
    native_abi = None
    if hasattr(library, "generativeqc_get_abi_version"):
        library.generativeqc_get_abi_version.argtypes = []
        library.generativeqc_get_abi_version.restype = ct.c_uint32
        native_abi = int(library.generativeqc_get_abi_version())
    return {
        "path": str(library_path),
        "binary_bytes": library_path.stat().st_size,
        "binary_sha256": file_hash(library_path),
        "native_source_identity": source_identity,
        "native_abi": native_abi,
        "missing_symbols": sorted(missing),
        "complete": not missing,
    }


def audit(
    *,
    radial_manifest: Path,
    library: Path | None = None,
) -> dict[str, object]:
    payload = json.loads(radial_manifest.read_text(encoding="utf-8"))
    radials = radial_inventory_from_payload(payload, backend="cpu")
    if not radials:
        raise RuntimeError("CPU derivative AOT radial inventory is empty")

    full_units = derivative_cpu_aot_sources()
    full_sources = tuple(source for _, source in full_units)
    range_records, range_sources = _range_programs(radials)
    full_key = DerivativeAotBundleKey(
        backend="cpu",
        radial=CoulombKernel("full_range", 0.0),
        component_domain=CPU_AOT_COMPONENTS,
    )
    full_package_key = DerivativeAotPackageKey(
        backend="cpu",
        target="native-host",
        scientific_identity=full_key.identity,
        output_contract=AOT_COMPONENT_OUTPUT_CONTRACT,
        spin_contract=AOT_SPIN_CONTRACT,
        contraction_contract=AOT_COMPONENT_CONTRACTION_CONTRACT,
    )
    return {
        "schema": "generativeqc.derivative-aot.audit.v1",
        "provenance": {
            "radial_manifest": str(radial_manifest),
            "radial_manifest_sha256": file_hash(radial_manifest),
            "registry_sha256": file_hash(
                ROOT
                / "python/generativeqc_compiler/integral/derivative_aot_registry.py"
            ),
        },
        "full_range": {
            "backend": "cpu",
            "target": "native-host",
            "identity": full_key.identity,
            "identity_payload": full_key.to_payload(),
            "package_identity": full_package_key.identity,
            "package_identity_payload": full_package_key.to_payload(),
            "radial": {"version": 1, "family": "full_range", "omega": 0.0},
            "component_domain": list(CPU_AOT_COMPONENTS),
            "shards": CPU_AOT_SHARDS,
            "symbols": [cpu_aot_symbol(shard) for shard in range(CPU_AOT_SHARDS)],
            **_source_statistics(full_sources),
        },
        "range": {
            "radials": [radial.to_payload() for radial in radials],
            "programs": range_records,
            **_source_statistics(range_sources),
        },
        "package_library": _symbol_status(
            library,
            range_records=range_records,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--radial-manifest",
        type=Path,
        default=ROOT / "manifests/derivative_aot_radials.json",
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit(radial_manifest=args.radial_manifest, library=args.library)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
