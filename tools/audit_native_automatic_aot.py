"""Join executed automatic-force receipts to native source/object/cubin evidence.

This offline audit never initializes CUDA or compiles. Source/object records
supplement the actual loaded-library digest in each receipt; they cannot replace
that execution binding or independently promote a scientific endpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from generativeqc_compiler.common.provenance import atomic_json, file_hash

from tools.validate_direct_jk_aot_resources import cuobjdump_resources


def artifact(path: Path) -> dict[str, Any]:
    """Identify the exact frozen file inspected by this offline qualification."""
    return {
        "path": str(path.resolve()),
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
    }


def audit_receipt(
    receipt: dict[str, Any], library: dict[str, Any], resources: dict[str, Any]
) -> dict[str, Any]:
    """Require real execution, exact binary identity and complete cubin counters."""
    if not receipt["force_completed"] or receipt["fallback_reason"] not in (
        "none",
        "minimum-workspace",
    ):
        raise ValueError("receipt does not certify a completed native force")
    artifact_kinds = ("point_artifact", "force_artifact")
    if receipt.get("schema") == "generativeqc.native-automatic-artifact.v2":
        artifact_kinds += ("point_executable_artifact", "force_executable_artifact")
        for field in (
            "polarized_executable_identity",
            "restricted_executable_identity",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", receipt.get(field, "")):
                raise ValueError("missing or invalid point executable identity")
    for kind in artifact_kinds:
        if (
            not isinstance(receipt.get(kind), dict)
            or receipt[kind].get("sha256") != library["sha256"]
            or receipt[kind].get("bytes") != library["bytes"]
        ):
            raise ValueError("executed artifact differs from the inspected binary")
    profile = receipt["force_profile"]
    if not re.fullmatch(r"libxc_[0-9]+_(rks|uks)", profile):
        raise ValueError("receipt is not an automatic native force profile")
    point = "automatic_" + receipt["identifier"].lower() + "_point_provider"
    executable_keys = tuple(
        filter(
            None,
            (
                receipt.get("polarized_executable_identity"),
                receipt.get("restricted_executable_identity"),
            ),
        )
    )
    if any(not re.fullmatch(r"[0-9a-f]{64}", key) for key in executable_keys):
        raise ValueError("invalid point executable identity")
    consumer = receipt.get("force_consumer_profile") or profile
    if not re.fullmatch(r"shared_[0-9a-f]{24}|libxc_[0-9]+_(rks|uks)", consumer):
        raise ValueError("invalid force consumer identity")
    if consumer.startswith("shared_"):
        if not re.fullmatch(
            r"[0-9a-f]{64}", receipt.get("force_consumer_identity", "")
        ):
            raise ValueError("missing or invalid force consumer source identity")
        point_work = receipt.get("staged_point_work", {})
        if any(
            point_work.get(field, 0) <= 0
            for field in ("tile_bytes", "launches", "points")
        ):
            raise ValueError("staged point storage/work is not accounted")
    scopes = {
        "point": {
            name: row
            for name, row in resources.items()
            if any("executable_" + key in name for key in executable_keys)
            or (not executable_keys and point in name)
        },
        "force": {
            name: row
            for name, row in resources.items()
            if "generativeqc_native_" + consumer + "_" in name
        },
    }
    for scope, rows in scopes.items():
        if not rows:
            raise ValueError(f"executed {scope} has no compiled cubin resources")
        for row in rows.values():
            if (
                set(row) != {"REG", "STACK", "LOCAL", "SHARED"}
                or not 0 <= row["REG"] <= 255
                or any(row[field] < 0 for field in row)
            ):
                raise ValueError("incomplete or invalid cubin resource counters")
        # Device linking can retain a zero-register auxiliary function record
        # (for example stationary_evaluate_point). It is not a launchable
        # kernel. Keep that record, but never accept an all-empty scope as
        # evidence for the actual point/force kernels.
        if not any(row["REG"] > 0 for row in rows.values()):
            raise ValueError(f"executed {scope} has no kernel resource counters")
    return {"profile": profile, "scopes": scopes, "runtime": receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--build-tree", type=Path, required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    # Resource policy belongs to the consumer/target qualification, not parser
    # defaults. Stack includes call frames, and does not claim zero spills.
    parser.add_argument("--max-stack", type=int, required=True)
    parser.add_argument("--max-local", type=int, required=True)
    parser.add_argument("--max-shared", type=int, required=True)
    args = parser.parse_args()
    if min(args.max_stack, args.max_local, args.max_shared) < 0:
        parser.error("resource limits must be nonnegative")
    library = artifact(args.library)
    tool = subprocess.run(
        [str(args.cuobjdump), "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    dump = subprocess.run(
        [str(args.cuobjdump), "--dump-resource-usage", str(args.library)],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    # A linked DSO embeds unrelated fatbins with repeated internal tensor
    # helpers. Audit the uniquely owned automatic point/force symbols here;
    # duplicates within those scopes still fail in the common parser.
    automatic_blocks = [
        match[0]
        for match in re.finditer(r"[ \t]*Function (\w+):\n[^\n]+", dump.stdout)
        if any(
            marker in match[1]
            for marker in (
                "automatic_",
                "executable_",
                "generativeqc_native_libxc_",
                "generativeqc_native_shared_",
            )
        )
    ]
    resources = cuobjdump_resources("\n".join(automatic_blocks))
    profiles = {}
    executions = []
    for path in args.receipt:
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            receipt = row["artifact"]
            joined = audit_receipt(receipt, library, resources)
            # Becke geometry uses dynamic shared storage, which cuobjdump's
            # static SHARED counter cannot describe. Bind its execution value
            # to the same explicit per-block ceiling as the static audit.
            if not 0 <= receipt["stationary_work_v1"][22] <= args.max_shared:
                raise ValueError("executed Becke dynamic shared memory exceeds limit")
            profile = joined["profile"]
            if profile not in profiles:
                consumer = receipt.get("force_consumer_profile") or profile
                staged = consumer.startswith("shared_")
                source = (
                    args.build_tree
                    / "generated/stationary_force"
                    / (
                        f"native_{consumer}_bindings.cpp"
                        if staged
                        else f"native_{profile}.cu"
                    )
                )
                objects = list(args.build_tree.rglob(source.name + ".o"))
                if len(objects) != 1:
                    raise ValueError(f"expected one compiled object for {profile}")
                identity = receipt["force_source_identity"].encode()
                if (
                    len(identity) != 64
                    or identity not in source.read_bytes()
                    or identity not in objects[0].read_bytes()
                ):
                    raise ValueError(
                        "executed source identity is missing from source/object"
                    )
                point_identity = receipt["point_source_identity"].encode()
                point_sources = [
                    path
                    for path in (
                        args.build_tree / "generated/libxc_semilocal_cuda"
                    ).glob("*.cu")
                    if point_identity in path.read_bytes()
                ]
                if len(point_identity) != 64 or len(point_sources) != 1:
                    raise ValueError("executed point has no unique generated source")
                point_objects = list(
                    args.build_tree.rglob(point_sources[0].name + ".o")
                )
                if (
                    len(point_objects) != 1
                    or point_identity not in point_objects[0].read_bytes()
                ):
                    raise ValueError(
                        "executed point identity is missing from its compiled object"
                    )
                for scope in joined["scopes"].values():
                    for name, limits in scope.items():
                        for field, maximum in (
                            ("STACK", args.max_stack),
                            ("LOCAL", args.max_local),
                            ("SHARED", args.max_shared),
                        ):
                            if limits[field] > maximum:
                                raise ValueError(
                                    f"{profile}: {name} exceeds {field}={maximum}"
                                )
                profiles[profile] = {
                    "source": artifact(source),
                    "object": artifact(objects[0]),
                    "source_identity": receipt["force_source_identity"],
                    "point_source": artifact(point_sources[0]),
                    "point_object": artifact(point_objects[0]),
                    "point_source_identity": receipt["point_source_identity"],
                    "scopes": joined["scopes"],
                }
                if staged:
                    shared_source = source.with_name(f"native_{consumer}.cu")
                    shared_objects = list(
                        args.build_tree.rglob(shared_source.name + ".o")
                    )
                    shared_identity = receipt["force_consumer_identity"].encode()
                    if (
                        len(shared_identity) != 64
                        or len(shared_objects) != 1
                        or shared_identity not in shared_source.read_bytes()
                        or shared_identity not in shared_objects[0].read_bytes()
                    ):
                        raise ValueError(
                            "shared consumer is not bound to its compiled object"
                        )
                    profiles[profile]["shared_consumer"] = {
                        "source": artifact(shared_source),
                        "object": artifact(shared_objects[0]),
                        "identity": receipt["force_consumer_identity"],
                    }
                executable_objects = {}
                for key in set(
                    filter(
                        None,
                        (
                            receipt.get("polarized_executable_identity"),
                            receipt.get("restricted_executable_identity"),
                        ),
                    )
                ):
                    marker = (
                        "namespace generativeqc::dft::cuda_xc_detail::executable_"
                        + key
                        + " {\nusing generativeqc_tensor::I;"
                    ).encode()
                    worker_sources = [
                        path
                        for path in (
                            args.build_tree / "generated/libxc_semilocal_cuda"
                        ).glob("*.cu")
                        if marker in path.read_bytes()
                    ]
                    worker_objects = (
                        list(args.build_tree.rglob(worker_sources[0].name + ".o"))
                        if len(worker_sources) == 1
                        else []
                    )
                    if (
                        len(worker_objects) != 1
                        or key.encode() not in worker_objects[0].read_bytes()
                    ):
                        raise ValueError(
                            "point executable has no unique compiled source/object"
                        )
                    executable_objects[key] = {
                        "source": artifact(worker_sources[0]),
                        "object": artifact(worker_objects[0]),
                    }
                profiles[profile]["point_executables"] = executable_objects
            executions.append(
                {
                    "receipt": str(path),
                    "phase": row["phase"],
                    "profile": profile,
                    "workspace": receipt["capacity"],
                    "fallback_reason": receipt["fallback_reason"],
                    "dynamic_becke_shared_bytes": receipt["stationary_work_v1"][22],
                }
            )
    if artifact(args.library) != library:
        raise ValueError("inspected library changed during audit")
    atomic_json(
        args.output,
        {
            "schema": "generativeqc.native-automatic-aot-audit.v1",
            "library": library,
            "tool": tool.stdout.strip(),
            "limits": {
                "registers": 255,
                "stack": args.max_stack,
                "local": args.max_local,
                "static_shared": args.max_shared,
            },
            "profiles": profiles,
            "executions": executions,
        },
    )
    print(
        f"Validated {len(profiles)} executed profiles and {len(executions)} endpoint receipts"
    )


if __name__ == "__main__":
    main()
