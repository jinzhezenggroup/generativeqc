"""Verify or restore one historical initialization case without executing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import stat
from pathlib import Path, PurePosixPath
from typing import NoReturn

STORED_LIMIT = 1 << 20
EXPANDED_LIMIT = 8 << 20
MEMORY_LIMIT = 96 << 20
MEMBER_LIMIT = 128


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unique(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def decode(data: bytes | str) -> dict | list:
    def nonfinite(value: str) -> NoReturn:
        raise ValueError("nonfinite JSON number: " + value)

    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def safe(name: str) -> str:
    require(isinstance(name, str) and 0 < len(name) < 512, "invalid member name")
    require("\\" not in name and ":" not in name, "nonportable member name")
    require(not PurePosixPath(name).is_absolute(), "absolute member name")
    require(
        all(part not in {"", ".", ".."} for part in name.split("/")),
        "unsafe member name",
    )
    return name


def authenticate(raw: bytes, record: dict, label: str) -> None:
    require(len(raw) == record["bytes"], label + ": byte count")
    require(digest(raw) == record["sha256"], label + ": SHA256")


def read_limited(path: Path, limit: int) -> bytes:
    require(path.is_file() and not path.is_symlink(), "nonordinary storage file")
    require(path.stat().st_size <= limit, "stored file exceeds limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "stored file grew beyond limit")
    return raw


def bounded_xz(raw: bytes) -> bytes:
    require(len(raw) <= STORED_LIMIT, "stored record exceeds limit")
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=MEMORY_LIMIT)
    data = decoder.decompress(raw, max_length=EXPANDED_LIMIT + 1)
    require(len(data) <= EXPANDED_LIMIT, "expanded record exceeds limit")
    require(decoder.eof and not decoder.unused_data, "truncated or trailing XZ stream")
    return data


def rows(events: list[dict], event: str) -> list[dict]:
    return [row for row in events if row.get("event") == event]


def verify(
    directory: Path,
) -> tuple[dict, dict[str, bytes], dict[str, tuple[bytes, int]]]:
    manifest_raw = read_limited(directory / "manifest.json", STORED_LIMIT)
    manifest = decode(manifest_raw)
    require(
        manifest["schema"] == "generativeqc.provider-init-engineering-evidence.v1",
        "manifest schema",
    )
    descriptors = {safe(row["name"]): row for row in manifest["members"]}
    require(
        len(descriptors) == len(manifest["members"]) <= MEMBER_LIMIT,
        "member count or duplicates",
    )
    originals = {}
    for storage in manifest["files"]:
        path = directory / safe(storage["path"])
        require(not path.is_symlink(), "symlink storage")
        raw = read_limited(path, STORED_LIMIT)
        authenticate(raw, storage, path.name)
        expanded = bounded_xz(raw)
        require(len(expanded) == storage["expanded_bytes"], "expanded bytes")
        require(digest(expanded) == storage["expanded_sha256"], "expanded SHA256")
        record = decode(expanded)
        require(
            record["schema"] == "generativeqc.finite-init-original-records.v1",
            "record schema",
        )
        require(len(record["members"]) == storage["members"], "group member count")
        for name, text in record["members"].items():
            safe(name)
            require(
                name not in originals and name in descriptors,
                "unknown or repeated member",
            )
            require(descriptors[name]["group"] == storage["group"], "member group")
            require(isinstance(text, str), "non-text original")
            data = text.encode("utf-8")
            authenticate(data, descriptors[name], name)
            if descriptors[name]["byte_identical_to_original"]:
                require(
                    digest(data) == descriptors[name]["original_sha256"],
                    "original digest",
                )
                require(
                    len(data) == descriptors[name]["original_bytes"], "original size"
                )
            else:
                require(
                    bool(descriptors[name]["redactions"]),
                    "undocumented public redaction",
                )
            originals[name] = data
    require(originals.keys() == descriptors.keys(), "incomplete corpus")

    def record(name: str) -> dict | list:
        return decode(originals[name])

    params = record("Root-bound-parameters.json")
    source = record("frozen-executed-inputs/source-files.json")
    reviewed = record("frozen-executed-inputs/e87-source-packet.json")
    require(
        digest(originals["frozen-executed-inputs/source-files.json"])
        == params["source_manifest_sha256"],
        "bound source manifest",
    )
    require(
        digest(originals["frozen-executed-inputs/e87-source-packet.json"])
        == source["reviewed_source_packet_sha256"],
        "bound review packet",
    )
    require(
        params["root_decision_sha256"] == manifest["root_decision_sha256"],
        "opaque decision binding",
    )
    require(
        reviewed["head_commit"] == manifest["measured_head"], "measured source head"
    )
    for name, text in reviewed["files"].items():
        authenticate(text.encode(), reviewed["manifest"][name], name)
    for row in source["reviewed21"]:
        authenticate(reviewed["files"][row["name"]].encode(), row, row["name"])
        require(row["mode"] == reviewed["manifest"][row["name"]]["mode"], "review mode")

    inventory = record("Root-terminal-inventory-result.json")
    actual = {safe(row["name"]): row for row in inventory["files"]}
    require(len(actual) == len(inventory["files"]) == 56, "actual inventory count")
    require(
        sum(row["bytes"] for row in actual.values())
        == inventory["total_bytes"]
        == 838711,
        "actual inventory bytes",
    )
    literal = record("frozen-executed-inputs/literal-payload.json")
    literal["root-bound-parameters.json"] = originals[
        "Root-bound-parameters.json"
    ].decode()
    executed = {}
    for name, text in literal.items():
        safe(name)
        require(name in actual, "source missing from actual inventory")
        data = text.encode()
        authenticate(data, actual[name], name)
        mode = int(actual[name]["mode"], 8)
        require(mode in {0o600, 0o640, 0o644, 0o755}, "unexpected source mode")
        executed[name] = (data, mode)
    require(len(executed) == 53, "executed source count")
    require(len(source["staged_files"]) == 51, "staged manifest count")
    for row in source["staged_files"]:
        data, mode = executed[row["name"]]
        authenticate(data, row, row["name"])
        require(mode == int(row["mode"], 8) & 0o777, "staged source mode")
    compiled = [row for name, row in actual.items() if name not in executed]
    require(
        compiled == manifest["compiled_inventory_only"], "compiled inventory differs"
    )
    require(len(compiled) == 3, "compiled inventory count")

    commands = [record(f"dispatch-v1/command-{index:02}.json") for index in range(38)]
    preexisting = {
        row["index"]: row for row in manifest["preexisting_cli_stdout_redactions"]
    }
    require(
        len(preexisting) == len(manifest["preexisting_cli_stdout_redactions"]),
        "duplicate redaction record",
    )
    for index, command in enumerate(commands):
        require(
            command["index"] == index
            and command["returncode"] == 0
            and not command["timed_out"],
            "CLI failure or missing call",
        )
        stdout = command["stdout"].encode()
        if index in preexisting:
            redaction = preexisting[index]
            require(
                command["stdout_raw_bytes"] == redaction["raw_bytes"]
                and command["stdout_raw_sha256"] == redaction["raw_sha256"],
                "pre-scrub identity changed",
            )
            require(
                len(stdout) == redaction["captured_bytes"]
                and digest(stdout) == redaction["captured_sha256"],
                "captured stdout identity",
            )
            require(
                command["stdout"].count("[REDACTED_URL]") == redaction["redacted_URLs"],
                "preexisting stdout redaction",
            )
        else:
            require(
                len(stdout) == command["stdout_raw_bytes"]
                and digest(stdout) == command["stdout_raw_sha256"],
                "CLI stdout identity",
            )
    stage_commands = record("frozen-executed-inputs/staging-commands-win-v1.json")
    require(len(stage_commands) == 36, "staging command count")
    require(
        all(
            digest(text.encode()) == commands[index]["command_sha256"]
            for index, text in enumerate(stage_commands)
        ),
        "staging command binding",
    )
    runner = record("actual-runner-events.json")
    bridge = record("actual-bridge-events.json")
    captured = [
        decode(line)
        for line in commands[-1]["stdout"].splitlines()
        if line.startswith('{"event":')
    ]
    require(captured == runner, "runner events differ from original CLI stdout")
    native = [
        row
        for row in rows(runner, "child_complete")
        if row["stage"] == "one_real_vendor_owner_process"
    ]
    require(len(native) == 1 and native[0]["returncode"] == 0, "native child outcome")
    vendor_stdout = native[0]["stdout_utf8"].encode("utf-8")
    require(
        len(vendor_stdout) == native[0]["stdout_bytes"]
        and digest(vendor_stdout) == native[0]["stdout_sha256"],
        "actual native child stdout identity",
    )
    authenticate(vendor_stdout, manifest["actual_vendor_child_stdout"], "vendor stdout")
    require(
        [decode(line) for line in native[0]["stdout_utf8"].splitlines()] == bridge,
        "bridge events differ from native stdout",
    )
    identities = rows(bridge, "native_identity")
    keys = [
        "status",
        "loaded_status",
        "error_domain",
        "error_status",
        "device",
        "context_id",
        "cuda_uuid_hex",
        "driver_api_version",
        "runtime_version",
        "direct_build_ids",
        "pci_query_status",
        "cuda_pci_bus_id",
    ]
    require(
        len(identities) == 2
        and all(identities[0][key] == identities[1][key] for key in keys),
        "native identity changed",
    )
    require(
        identities[0]["status"] == identities[0]["loaded_status"] == 0,
        "native identity unavailable",
    )
    for event, keys in [
        (
            "physical_identity",
            ["nvidia_index", "name", "pci_bus_id", "uuid_sha256", "driver"],
        ),
        (
            "container_identity",
            [
                "proc1_start_ticks",
                "pid_namespace",
                "boot_id_sha256",
                "container_fingerprint_sha256",
            ],
        ),
        ("direct_library_snapshot", ["libraries"]),
    ]:
        before, after = rows(runner, event)
        require(all(before[key] == after[key] for key in keys), event + " changed")
    guards = rows(bridge, "guard_check")
    require(len(guards) == 2, "guard count")
    for guard in guards:
        require(
            guard["observed_delta_bytes"]
            == max(0, guard["free_before_bytes"] - guard["free_after_bytes"]),
            "delta arithmetic",
        )
        require(guard["query_status"] == 0 and guard["contract_valid"], "guard query")
    legacy, explicit = guards
    require(
        legacy["kind"] == "legacy_default"
        and legacy["allowance_bytes"] == 201326592
        and legacy["refused"],
        "legacy refusal",
    )
    require(
        explicit["kind"] == "explicit_selected"
        and explicit["allowance_bytes"] == params["allowance_bytes"] == 536870912
        and not explicit["refused"],
        "selected allowance",
    )
    require(
        legacy["observed_delta_bytes"] == explicit["observed_delta_bytes"] == 375390208,
        "observed allocation",
    )
    (cleanup,) = rows(bridge, "cleanup")
    require(
        all(
            cleanup[key] == 0
            for key in [
                "drain_rc",
                "parameters_rc",
                "solver_rc",
                "blas_rc",
                "stream_rc",
            ]
        ),
        "cleanup failure",
    )
    (terminal,) = rows(bridge, "process_terminal")
    require(
        terminal["engineering_contract"] == "PASS"
        and terminal["actual_vendor_owner_setups"] == 1
        and terminal["actual_guard_calls"] == 2,
        "finite native outcome",
    )
    require(
        all(
            terminal[key] == 0
            for key in [
                "numeric_solver_invocations",
                "matrix_operations",
                "scientific_endpoints",
            ]
        ),
        "unexpected numerical work",
    )
    report = {
        "status": "PASS_OFFLINE_CAPTURED_RECORD_AND_SOURCE_VERIFICATION",
        "measured_head": manifest["measured_head"],
        "captured_records": len(originals),
        "publicly_redacted_records": sum(
            not row["byte_identical_to_original"] for row in descriptors.values()
        ),
        "restorable_executed_sources": len(executed),
        "actual_inventory_files": len(actual),
        "compiled_inventory_only": len(compiled),
        "CLI_calls": len(commands),
        "CLI_pre_scrub_stdout_verified": len(commands) - len(preexisting),
        "CLI_pre_scrub_stdout_UNAVAILABLE": len(preexisting),
        "actual_vendor_child_stdout_bytes_verified": len(vendor_stdout),
        "observed_delta_bytes": explicit["observed_delta_bytes"],
        "qualification": manifest["qualification"],
        "new_GPU_or_scientific_work": False,
    }
    return report, originals, executed


def restore(destination: Path, originals: dict, executed: dict) -> dict:
    require(
        not destination.exists() and not destination.is_symlink(),
        "restore requires a fresh directory",
    )
    destination.mkdir(parents=True)
    checked = 0
    for prefix, records in [
        ("originals", {name: (raw, 0o644) for name, raw in originals.items()}),
        ("executed", executed),
    ]:
        for name, (raw, mode) in records.items():
            path = destination / prefix / safe(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
            path.chmod(mode)
            require(path.read_bytes() == raw, "restored bytes differ")
            if os.name == "posix":
                require(
                    stat.S_IMODE(path.stat().st_mode) == mode,
                    "restored POSIX mode differs",
                )
                checked += 1
    return {
        "restored_files": len(originals) + len(executed),
        "restored_POSIX_modes_checked": checked,
        "mode_execution_platform": os.name,
        "Windows_permissions_are_POSIX_validation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore",
        type=Path,
        help="fresh output directory; never executes restored code",
    )
    args = parser.parse_args()
    report, originals, executed = verify(Path(__file__).resolve().parent)
    if args.restore:
        report["restoration"] = restore(args.restore, originals, executed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
