"""Lossless offline replay of one spent PBE0 pilot; stdlib, no archived imports.

Published Git inputs are read only from an explicitly named local repository.
No command fetches history, launches science, imports an SDK or repairs caches.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

LIMIT = 32 << 20
MEMBER_LIMIT = 1 << 20
AUDIT_SHA = "fcab7709e199a6817a780e6afd6f4a3827477109dfc0b157e2619fc846e10b08"
PLAN_SHA = "019f0f5945c34805fa39007a0e1e496dea7ec57daac29123e76d91d1bf0456f3"
RELEASE_SHA = "ad95a9f8a5e3206b4957f58c6187c45e3c0b468f4d3acfd94b28fb18b3b6babc"
SOURCE = "cbd4d8074de35105717c8aac7cd1f86babbe90ec"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def pairs(items: list) -> dict:
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_json(raw: bytes) -> dict:
    def constant(value: str) -> None:
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def relative(value: str) -> str:
    require(
        isinstance(value, str) and value and "\\" not in value and ":" not in value,
        "unsafe relative name",
    )
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and all(p not in ("", ".", "..") for p in value.split("/")),
        "unsafe relative path",
    )
    require(
        path.name not in (".env", "auth.json", "token", "token.json")
        and not path.name.startswith(".env."),
        "credential path refused",
    )
    return value


def checked(raw: bytes, entry: dict) -> bytes:
    require(
        type(entry["bytes"]) is int
        and len(raw) == entry["bytes"]
        and sha(raw) == entry["sha256"],
        "raw byte identity differs",
    )
    return raw


def decode(raw: bytes, limit: int) -> bytes:
    require(
        len(raw) <= MEMBER_LIMIT and type(limit) is int and 0 <= limit <= LIMIT,
        "decode cap",
    )
    decoder = lzma.LZMADecompressor(memlimit=96 << 20)
    try:
        result = decoder.decompress(raw, max_length=limit + 1)
    except lzma.LZMAError as error:
        raise ValueError("invalid compressed stream") from error
    require(
        len(result) <= limit and decoder.eof and not decoder.unused_data,
        "truncated/trailing/oversized XZ",
    )
    return result


def git_blob(repo: Path | None, entry: dict) -> bytes:
    require(repo is not None, "published Git input needs --repo; no implicit fetch")
    require(
        re.fullmatch(r"[0-9a-f]{40}", entry["revision"]) is not None,
        "fixed Git revision required",
    )
    path = relative(entry["path"])
    listing = subprocess.check_output(
        ["git", "ls-tree", entry["revision"], "--", path], cwd=repo
    )
    fields = listing.decode().split("\t")[0].split()
    require(
        len(fields) == 3 and fields[0] == entry["mode"] and fields[2] == entry["blob"],
        "published Git mode/blob differs",
    )
    return subprocess.check_output(
        ["git", "show", entry["revision"] + ":" + path], cwd=repo
    )


def recover(
    directory: Path,
    capsule: dict,
    repo: Path | None = None,
    resolver: Callable[[dict], bytes] | None = None,
) -> dict[str, bytes]:
    require(
        capsule["schema"] == "generativeqc.h100-pbe0-failed-pilot-originals.v1",
        "wrong capsule schema",
    )
    pool_entry = capsule["pool"]
    pool_path = directory / relative(pool_entry["file"])
    require(not pool_path.is_symlink(), "pool symlink refused")
    pool = decode(checked(pool_path.read_bytes(), pool_entry), pool_entry["raw_bytes"])
    require(sha(pool) == pool_entry["raw_sha256"], "pool raw hash")
    objects, spans, used = {}, [], set()
    require(
        len(capsule["objects"]) <= 512 and len(capsule["files"]) <= 512,
        "member count cap",
    )
    require(
        sum(e["bytes"] for e in capsule["objects"].values()) <= LIMIT,
        "aggregate raw cap",
    )
    for digest, entry in capsule["objects"].items():
        require(
            digest == entry["sha256"] and re.fullmatch(r"[0-9a-f]{64}", digest),
            "object key identity",
        )
        encoding = entry["encoding"]
        if encoding == "byte-pool":
            offset, size = entry["offset"], entry["bytes"]
            require(
                type(offset) is int
                and type(size) is int
                and offset >= 0
                and size >= 0
                and offset + size <= len(pool),
                "pool extent",
            )
            raw = pool[offset : offset + size]
            if size:
                spans.append((offset, offset + size))
        elif encoding == "xz-parts":
            require(0 < len(entry["parts"]) <= 16, "part count")
            pieces = []
            for part in entry["parts"]:
                path = directory / relative(part["file"])
                require(not path.is_symlink(), "part symlink refused")
                data = decode(checked(path.read_bytes(), part), part["raw_bytes"])
                require(sha(data) == part["raw_sha256"], "part raw hash")
                pieces.append(data)
            raw = b"".join(pieces)
        elif encoding == "published-git":
            raw = resolver(entry) if resolver is not None else git_blob(repo, entry)
        else:
            raise ValueError("unrecognized object encoding")
        objects[digest] = checked(raw, entry)
    end = 0
    for start, stop in sorted(spans):
        require(start == end, "pool overlap/gap")
        end = stop
    require(end == len(pool), "unindexed pool bytes")
    result = {}
    for row in capsule["files"]:
        name = relative(row["path"])
        require(name not in result, "duplicate original path")
        require(
            row["role"] in ("ORIGINAL", "SUPPLEMENTAL_ORIGINAL"),
            "unknown original role",
        )
        mode = row["linux_mode"]
        require(
            mode is None or (type(mode) is int and 0 <= mode <= 0o777),
            "noncanonical mode",
        )
        require(
            (row["role"] == "ORIGINAL") == (mode is not None),
            "original mode witness missing or invented",
        )
        require(row["sha256"] in objects, "missing raw object")
        result[name] = checked(objects[row["sha256"]], row)
        used.add(row["sha256"])
    require(used == set(objects), "unreferenced raw object")
    require(
        not any(
            str(parent) in result
            for name in result
            for parent in PurePosixPath(name).parents
        ),
        "file/directory path collision",
    )
    return result


def replay(files: dict[str, bytes], capsule: dict) -> dict:
    def j(name: str) -> dict:
        return read_json(files[name])

    audit_raw = files["audit/root-terminal-failure-audit.json"]
    require(sha(audit_raw) == AUDIT_SHA, "not the authoritative spent-phase audit")
    audit = read_json(audit_raw)
    index = j("audit/terminal-originals-index.json")
    originals = {r["path"]: r for r in index["retained_files"]}
    rows = {r["path"]: r for r in capsule["files"]}
    require(
        len(originals) == 77
        and sum(r["bytes"] for r in originals.values()) == 10312946,
        "original retention scope",
    )
    require(
        {name for name in files if not name.startswith("audit/")} == set(originals),
        "missing or extra terminal original",
    )
    inventory = {r["path"]: r for r in index["inventory"]}
    for name, row in originals.items():
        checked(files[name], row)
        require(
            rows[name]["linux_mode"] == row["mode"]
            and rows[name]["role"] == "ORIGINAL",
            "restoration mode/role differs from original",
        )
        require(
            inventory[name]["sha256"] == row["sha256"]
            and inventory[name]["mode"] == row["mode"],
            "original inventory/mode differs",
        )
    plan = j("scripts/prepared-plan.json")
    require(
        sha(files["scripts/prepared-plan.json"]) == PLAN_SHA
        and plan["source_driver"]["revision"] == SOURCE,
        "deployed source/plan",
    )
    for name, digest in {**plan["task_code"], **plan["task_data"]}.items():
        require(
            sha(files["scripts/" + name]) == digest
            and oct(originals["scripts/" + name]["mode"]) == plan["task_modes"][name],
            "code/data mode/byte mismatch",
        )
    require(
        len(plan["task_code"]) + len(plan["task_data"]) == 29, "deployed input count"
    )
    require(
        sha(files["root-release.json"]) == RELEASE_SHA
        and files["root-release.json"] == files["output/root-release-snapshot.json"],
        "release snapshot",
    )
    for name in ("batch-consumed.json", "slot-0-consumed.json"):
        require(
            j(name)["release_sha256"] == RELEASE_SHA
            and j(name)["root_nonce"] == plan["root_nonce"],
            "consumed release binding",
        )
    requested = [p for p in inventory if p.endswith("/scientific-request.json")]
    complete = [
        p for p in inventory if "/return-" in p and p.endswith(".complete.json")
    ]
    partial = [p for p in inventory if "/partial-" in p]
    require(
        requested == ["output/slot-0/scientific-request.json"]
        and not complete
        and not partial,
        "not one failed cold request with absent returns",
    )
    req = j(requested[0])
    require(
        req["arm"] == "standard"
        and req["phase"] == "cold"
        and type(req["request_index"]) is int
        and req["request_index"] == 0,
        "request phase/ordinal",
    )
    failure = j("output/slot-0/failure.json")
    require(
        failure["complete_endpoints"] == 0
        and "missing packaged stationary CUDA artifact for pbe0_rks_spd"
        in failure["message"],
        "failure cause",
    )
    worker = j("output/slot-0/outcome.json")
    require(
        worker["retained_complete_returns"] == 0 and worker["comparisons"] == [],
        "invented numerical return",
    )
    offline = j("output/offline-report.json")
    require(
        offline["requests_consumed"] == 1
        and offline["complete_returns"] == 0
        and [r["status"] for r in offline["rows"]] == ["FAILED", "NOT_RUN", "NOT_RUN"],
        "request/arm accounting",
    )
    require(
        all(r["seconds"] is None for r in offline["rows"])
        and offline["speedups"] is None,
        "zero-filled timing or speedup",
    )
    before, after = j("output/static-before.json"), j("output/static-after.json")
    pre, device = j("output/preflight.json"), j("output/device-after.json")
    require(before == after and pre["device"] == device, "stored identity changed")
    guard, born, ack = (
        j("process-0/control/" + name)
        for name in (
            "guard-outcome.json",
            "command-identity.json",
            "start-command.json",
        )
    )
    identity = j("process-0/control/identity.json")
    require(sha(canonical(guard["command"])) == ack["command_sha256"], "argv ACK")
    require(
        guard["nonce"] == born["nonce"] == identity["nonce"] == ack["nonce"],
        "guard nonce",
    )
    for kind, row in (("guard", identity), ("command", born)):
        require(
            all(
                guard[kind + "_identity"][k] == row[k]
                for k in ("pid", "ppid", "pgid", "sid", "start_ticks")
            ),
            "terminal birth",
        )
    require(
        born["ppid"] == identity["pid"]
        and guard["cleanup_success"] is True
        and guard["remaining_owned"] == [],
        "owned child cleanup",
    )
    batch, final = j("output/batch-outcome.json"), j("output/finalizer-complete.json")
    require(
        batch["status"] == final["status"] == "FAILED"
        and final["batch_sha256"] == sha(files["output/batch-outcome.json"])
        and final["offline_report_sha256"] == sha(files["output/offline-report.json"]),
        "terminal marker/report hash",
    )
    api_statuses = []
    for name in sorted(files):
        if name.startswith("root-api/observations/"):
            observation = j(name)
            for kind in ("status", "instances"):
                raw = base64.b64decode(observation[kind + "_raw_base64"], validate=True)
                require(
                    sha(raw) == observation[kind + "_raw_sha256"]
                    and read_json(raw) == observation[kind + "_response"],
                    "API exact raw/parsed binding",
                )
            api_statuses.append(observation["status_response"]["data"]["status"])
    require(api_statuses == ["RUNNING"] * 6, "preterminal poll scope changed")
    final_status = read_json(
        j("audit/heartbeat-20261025-2235-status-followup.json")["stdout"].encode()
    )["data"]
    require(
        final_status["status"] == "FAILED"
        and final_status["created_at"] == "1792947534000"
        and final_status["finished_at"] == "1792967890000",
        "final platform outcome",
    )
    logs = read_json(j("audit/terminal-all-logs.json")["stdout"].encode())["data"]
    events = read_json(j("audit/terminal-events.json")["stdout"].encode())["data"][
        "items"
    ]
    require(
        len(logs["logs"]) == 4 and logs["truncated"] is False and len(events) == 32,
        "incomplete logs/events",
    )
    stat = j("audit/exact-stationary-pair-stat-result.json")
    required = {
        "libgenerativeqc_stationary_pbe0_rks_spd.so",
        "generativeqc_stationary_pbe0_rks_spd.json",
    }
    require(
        {r["name"] for r in stat["files"]} == required
        and all(r["lstat_exists"] is False for r in stat["files"]),
        "exact companion stat",
    )
    require(stat["parent_inode_mtime_unchanged"] is True, "stat changed parent")
    binding = j("audit/owner-static-diagnosis-binding-readback.json")
    require(
        sha(files["audit/owner-independent-terminal-review.json"])
        == binding["owner_report_sha256"],
        "independent source diagnosis binding",
    )
    require(
        binding["all9_Git_LF_bytes_match_expected_2e_source_rows"] is True
        and binding[
            "historical_HF_actual_outcome_text_bytes_SHA_and_core_only_build_command_independently_match"
        ]
        is True,
        "static provenance binding",
    )
    return {
        "schema": "generativeqc.h100-pbe0-failed-pilot-replay.v1",
        "job": final_status["name"],
        "status": "FAILED_SPENT",
        "source_revision": SOURCE,
        "core_sha256": before["library_sha256"],
        "environment_identity": before["environment_sha256"],
        "request_count": 1,
        "complete_EF_returns": 0,
        "arms": {"standard": "FAILED_COLD", "fast": "NOT_RUN", "gpu4pyscf": "NOT_RUN"},
        "energy": "UNAVAILABLE",
        "forces": "UNAVAILABLE",
        "endpoint_seconds": "UNAVAILABLE",
        "ABI_scalar": "UNAVAILABLE_NOT_EXPLICITLY_RECORDED",
        "work_precision_fallback_peak_billing": "UNAVAILABLE",
        "source_env_device_receipts_equal": True,
        "cleanup": "PASS",
        "raw_originals_restorable": 77,
        "raw_original_bytes": 10312946,
        "supplemental_originals": 8,
        "cache_binaries": "INVENTORY_ONLY_REMOTE_RETAINED_NOT_PUBLISHED_AS_RAW",
        "exact_companion_paths": "BOTH_ABSENT_IN_POST_FAILURE_ROOT_STAT",
        "artifact_key": "UNAVAILABLE_MANIFEST_NOT_LOADED",
        "time_scopes_do_not_add": {
            k: audit[k]
            for k in (
                "Root_elapsed_seconds",
                "worker_command_wall_seconds",
                "supervisor_wall_before_export_seconds",
                "engineering_charge_before_export_seconds",
            )
        },
        "scientific_acceptance": "INCOMPLETE",
        "numerical_precision_OOM_verdict": "NOT_ESTABLISHED",
        "full2731_DF96_formal_parent": "INCOMPLETE",
        "speedups": None,
        "new_science_or_retry_authority": "NONE_NO_UNUSED_BALANCE",
    }


def restore(directory: Path, files: dict[str, bytes], rows: list[dict]) -> None:
    # Validate before creating any output, including file/directory collisions.
    for name in files:
        relative(name)
    require(
        not any(
            str(parent) in files
            for name in files
            for parent in PurePosixPath(name).parents
        ),
        "file/directory path collision",
    )
    require(
        not directory.exists() and not directory.is_symlink(),
        "restore destination must be new",
    )
    directory.mkdir(parents=True, exist_ok=False)
    modes = {r["path"]: r["linux_mode"] for r in rows}
    for name, raw in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        if os.name == "posix" and modes[name] is not None:
            path.chmod(modes[name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--repo", type=Path, help="local full Git history; never fetched implicitly"
    )
    parser.add_argument(
        "--restore",
        type=Path,
        help="new directory, no overwrite, never executes restored code",
    )
    args = parser.parse_args()
    artifacts = read_json((args.directory / "artifacts.json").read_bytes())
    for entry in artifacts["files"]:
        checked((args.directory / relative(entry["path"])).read_bytes(), entry)
    raw = (args.directory / "originals.json").read_bytes()
    require(len(raw) <= MEMBER_LIMIT, "original index size cap")
    capsule = read_json(raw)
    files = recover(args.directory, capsule, args.repo)
    result = replay(files, capsule)
    # This is an explicit history audit, not a normal-CI source fingerprint test.
    # These inputs are fixed published Git objects and are never imported.
    for entry in capsule.get("published_diagnosis_inputs", []):
        checked(git_blob(args.repo, entry), entry)
    expected = read_json((args.directory / "summary.json").read_bytes())
    require(result == expected, "stored summary differs from raw replay")
    if args.restore:
        restore(args.restore, files, capsule["files"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
