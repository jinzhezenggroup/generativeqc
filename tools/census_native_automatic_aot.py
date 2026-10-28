"""Census the active native automatic AOT build and its installed CUDA kernels.

Only objects in the current CMake compile database are counted; stale outputs
from a larger previous selection cannot inflate a shrinking package. This
read-only offline tool does not initialize CUDA or infer scientific admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.common.cuda_resources import parse_resources
from generativeqc_compiler.common.provenance import atomic_json, file_hash
from generativeqc_compiler.xc.libxc_bulk_capabilities import (
    STAGE_EVIDENCE_SCHEMA,
    functional_capability,
)

from tools.validate_direct_jk_aot_resources import cuobjdump_resources


def artifact(path: Path) -> dict[str, Any]:
    """Bind the inspected bytes, without treating a source digest as a binary key."""
    return {
        "path": str(path.resolve()),
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
    }


def device_image(path: Path) -> dict[str, Any]:
    """Hash the exact embedded device bytes across legitimate host RPATH edits."""
    sections = subprocess.run(
        ["readelf", "-SW", str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    matches = re.findall(
        r"\]\s+\.nv_fatbin\s+PROGBITS\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)",
        sections,
    )
    if len(matches) != 1:
        raise ValueError("expected one native CUDA fatbin section")
    offset, size = (int(value, 16) for value in matches[0])
    if not size or offset + size > path.stat().st_size:
        raise ValueError("invalid CUDA image extent")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            data = stream.read(min(remaining, 1024 * 1024))
            if not data:
                raise ValueError("truncated CUDA image")
            digest.update(data)
            remaining -= len(data)
    return {"sha256": digest.hexdigest(), "bytes": size}


def active_objects(build: Path) -> dict[str, list[dict[str, Any]]]:
    """Join active compilation commands to existing source/object pairs."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in json.loads((build / "compile_commands.json").read_text()):
        command = row.get("arguments") or shlex.split(row["command"])
        output = row.get("output") or command[command.index("-o") + 1]
        if "generativeqc_native_stationary.dir/" in output:
            group = (
                "force_bindings" if row["file"].endswith(".cpp") else "force_consumers"
            )
            if "native_stationary_shared_kernels" in output:
                group = "shared_reduction"
        elif "generativeqc_libxc_semilocal_cuda.dir/" in output:
            group = "point"
        elif "generativeqc_stationary_sp_primitives.dir/" in output:
            group = "integral_primitives"
        elif "/generated/libxc_semilocal_cpu/" in row["file"]:
            group = "cpu_point"
        else:
            continue
        obj = Path(row["directory"]) / output
        source = Path(row["directory"]) / row["file"]
        if not obj.is_file() or not source.is_file():
            raise ValueError(f"active AOT input/output is absent: {source}, {obj}")
        groups.setdefault(group, []).append(
            {"source": artifact(source), "object": artifact(obj), "command": command}
        )
    if not groups.get("point") or not groups.get("force_consumers"):
        raise ValueError("census requires a compiled point and force package")
    return groups


def device_scope(name: str) -> str | None:
    """Classify owned symbols, leaving unrelated core CUDA kernels out of scope."""
    if "generativeqc_native_shared_" in name or "generativeqc_native_libxc_" in name:
        return "automatic_force"
    if "executable_" in name or "automatic_" in name:
        return "automatic_point"
    return None


def spill_records(log: str) -> dict[str, dict[str, int]]:
    """Read actual ptxas spill counts; LOCAL=0 alone never proves zero spills."""
    result: dict[str, dict[str, int]] = {}
    for record in parse_resources(log):
        name = record.function
        if device_scope(name):
            counters = {
                "stack_bytes": record.stack_bytes,
                "spill_store_bytes": record.spill_store_bytes,
                "spill_load_bytes": record.spill_load_bytes,
            }
            if name in result and result[name] != counters:
                raise ValueError(f"ambiguous ptxas resource record: {name}")
            result[name] = counters
    return result


def compiled_capability_evidence(
    name: str, *, census: dict[str, Any], reference: str
) -> dict[str, Any]:
    """Expose artifact economics to Lane F without granting SCF or force science."""
    envelope = {
        "schema": STAGE_EVIDENCE_SCHEMA,
        "subject_identity": functional_capability(name).identity,
        "stage": "compiled-cuda",
        "status": "pass",
        "evidence": reference,
        "reason": None,
        "qualification": {
            "scope": "static native automatic E/vxc and requested stationary AOT package",
            "library": census["library"],
            "installed_library": census["installed_library"],
            "artifact_census": reference,
            "compilation": census.get("compilation", []),
            "endpoint_performance": census.get("endpoint_performance", []),
            "numerical_promotion": False,
        },
    }
    # Exercise the existing promotion owner, including subject identity and
    # prerequisite validation, instead of inventing a parallel capability gate.
    capability = functional_capability(name, evidence={"compiled-cuda": envelope})
    if (
        "compiled-cuda" not in capability.qualified_stages
        or "forces" in capability.qualified_stages
    ):
        raise ValueError("compilation evidence must not promote molecular endpoints")
    return {"compiled-cuda": envelope}


def main() -> None:
    """Inspect a frozen linked/installed pair and retain reproducible joins."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-tree", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--installed-library", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--ptxas-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capability-output", type=Path)
    parser.add_argument("--compile-measurement", type=Path, action="append", default=[])
    parser.add_argument(
        "--performance-evidence", type=Path, action="append", default=[]
    )
    args = parser.parse_args()
    groups = active_objects(args.build_tree)
    library, installed = artifact(args.library), artifact(args.installed_library)
    linked_image, installed_image = (
        device_image(args.library),
        device_image(args.installed_library),
    )
    if linked_image != installed_image:
        raise ValueError("installed CUDA bytes differ from the linked package")
    dynamic_symbols = subprocess.run(
        ["nm", "-D", "--defined-only", str(args.library)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    exported_workers = [
        line.split()[-1]
        for line in dynamic_symbols.splitlines()
        if re.search(r"cuda_xc_detail.*executable_[0-9a-f]{64}", line)
    ]
    if exported_workers:
        raise ValueError(
            "shared point workers must remain private to their authenticated module"
        )

    def dump(option: str, path: Path = args.library) -> str:
        return subprocess.run(
            [str(args.cuobjdump), option, str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        ).stdout

    resources = cuobjdump_resources(
        "\n".join(
            match[0]
            for match in re.finditer(
                r"[ \t]*Function (\w+):\n[^\n]+", dump("--dump-resource-usage")
            )
            if device_scope(match[1])
        )
    )
    entries = [
        line.split()[-1]
        for line in dump("--dump-elf-symbols").splitlines()
        if "STO_ENTRY" in line
    ]
    kernels = {name for name in entries if device_scope(name)}
    spills = spill_records(args.ptxas_log.read_text())
    if not kernels or kernels - resources.keys() or kernels - spills.keys():
        missing = sorted(kernels - spills.keys())
        raise ValueError(f"incomplete kernel/resource/spill join: {missing[:8]}")
    names: set[str] = set()
    for row in groups["point"]:
        names.update(
            re.findall(
                r'const CudaXcPointProgram program\{\s*"([A-Z0-9_]+)"',
                Path(row["source"]["path"]).read_text(),
            )
        )
    if not names:
        raise ValueError("no compiled automatic registrations")
    census = {
        "schema": "generativeqc.native-automatic-aot-census.v1",
        "library": library,
        "installed_library": installed,
        "linked_device_image": linked_image,
        "installed_device_image": installed_image,
        "exported_point_workers": exported_workers,
        "archive": artifact(args.build_tree / "libgenerativeqc_native_stationary.a"),
        "device_link_object": artifact(
            args.build_tree
            / "CMakeFiles/generativeqc_native_stationary.dir/cmake_device_link.o"
        ),
        "compile_database": artifact(args.build_tree / "compile_commands.json"),
        "compiler_resource_log": artifact(args.ptxas_log),
        "compilation": [artifact(path) for path in args.compile_measurement],
        "endpoint_performance": [artifact(path) for path in args.performance_evidence],
        "registrations": sorted(names),
        "groups": {
            name: {
                "count": len(rows),
                "source_bytes": sum(row["source"]["bytes"] for row in rows),
                "object_bytes": sum(row["object"]["bytes"] for row in rows),
                "rows": rows,
            }
            for name, rows in groups.items()
        },
        "kernels": {
            name: {"scope": device_scope(name), **resources[name], **spills[name]}
            for name in sorted(kernels)
        },
        "kernel_counts": {
            scope: sum(device_scope(name) == scope for name in kernels)
            for scope in ("automatic_force", "automatic_point")
        },
        "complete_library_entry_counts": {
            "instances": len(entries),
            "unique_symbols": len(set(entries)),
            "scope": "all linked cubins, including unrelated core and TU-local helper kernels",
        },
    }
    if (
        artifact(args.library) != library
        or artifact(args.installed_library) != installed
    ):
        raise ValueError("library changed during census")
    atomic_json(args.output, census)
    if args.capability_output:
        atomic_json(
            args.capability_output,
            {
                name: compiled_capability_evidence(
                    name, census=census, reference=str(args.output)
                )
                for name in sorted(names)
            },
        )


if __name__ == "__main__":
    main()
