"""Bounded, standard-library replay of immutable Direct WB97M-V evidence.

Nothing in this module imports an archived worker, solver, SDK or GPU library.
The public byte pool restores selected originals, not the three omitted native
cache binaries or the original compressed transport envelopes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
import math
import os
from pathlib import Path, PurePosixPath
from statistics import median
import struct

MAX_ENVELOPE = 64 << 20
XZ_MEMORY = 96 << 20
ARMS = ("standard", "fast", "gpu4pyscf")
ROTATION = ARMS + ("fast", "gpu4pyscf", "standard", "gpu4pyscf", "standard", "fast")
INPUTS = {
    48: (29, "17d44df7afa9996b0abf7df72301bf4758eadbde9f4523ce321001d14d2596ea"),
    96: (39, "03c18d243ced417acc7ae0dc324f54145f1665febded4f50c62f8d73b1d4d0e3"),
}
JOBS = {
    "2732-pilot-v1": (
        48,
        False,
        "4838629a44a14a0e9543e83093d0d0ed8e06fa317a359b86b4d243ecd6130c72",
        "ONE_FINITE_BATCH",
    ),
    "2732-pilot-v2": (
        48,
        False,
        "7c89819da88873a1b20a67e6f93e3dfde7e801fe2d94533107013939f7190341",
        "ONE_FINITE_BATCH",
    ),
    "2732-formal48-run": (
        48,
        True,
        "73666ff3874fbde91678f6d0eb673a755b1ebc0e62b089aee1ba7ca913e50473",
        "ONE_FINITE_FORMAL48_BATCH",
    ),
    "2732-pilot96-run": (
        96,
        False,
        "1e31c579042593a38af5dc0e223db1fd88f784af43cb39ae9e0a83a530cdb09c",
        "ONE_FINITE_PILOT96_BATCH",
    ),
    "2732-formal96-run": (
        96,
        True,
        "5c5f60fee9e5a52eeb56b74a13a1f2a2008cdf2abbf2774fc5f942f84aa35648",
        "ONE_FINITE_FORMAL96_BATCH",
    ),
}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def unique_pairs(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def json_bytes(data: bytes) -> dict | list:
    return json.loads(data, object_pairs_hook=unique_pairs)


def decode_xz(data: bytes, *, limit: int = MAX_ENVELOPE) -> dict | list:
    require(len(data) <= 1 << 20, "compressed artifact exceeds single-file cap")
    decoder = lzma.LZMADecompressor(memlimit=XZ_MEMORY)
    raw = decoder.decompress(data, max_length=limit + 1)
    require(
        len(raw) <= limit and decoder.eof and not decoder.unused_data,
        "expanded limit, truncated stream or trailing data",
    )
    return json_bytes(raw)


def safe_path(name: str) -> str:
    require(
        type(name) is str
        and name
        and "\\" not in name
        and ":" not in name
        and all(ord(char) >= 32 for char in name),
        "unsafe member path",
    )
    path = PurePosixPath(name)
    require(
        path.parts
        and not path.is_absolute()
        and ".." not in path.parts
        and str(path) == name,
        "unsafe or noncanonical member path",
    )
    devices = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }
    require(
        all(
            not part.endswith((".", " ")) and part.split(".")[0].upper() not in devices
            for part in path.parts
        ),
        "nonportable member path",
    )
    return name


class Evidence:
    def __init__(self, document: dict):
        require(
            document.get("schema") == "generativeqc.h100-direct-content-objects.v1",
            "unknown evidence schema",
        )
        self.document = document
        self.objects = {}
        total = 0
        for digest, encoded in document["objects"].items():
            require(type(digest) is str and len(digest) == 64, "invalid object key")
            raw = base64.b64decode(encoded, validate=True)
            require(sha(raw) == digest, "object SHA mismatch")
            total += len(raw)
            require(total <= MAX_ENVELOPE, "decoded object limit")
            self.objects[digest] = raw
        self.rows = {}
        referenced = set()
        excluded = {}
        for job, envelope in document["original_job_member_inventories"].items():
            safe_path(job)
            require("/" not in job, "unsafe job name")
            rows = {}
            for row in envelope["members"]:
                name = safe_path(row["path"])
                require(name not in rows, "duplicate member path")
                require(
                    type(row["bytes"]) is int and row["bytes"] >= 0,
                    "noncanonical member size",
                )
                require(
                    row["mode"] in ("0o600", "0o644", "0o755"), "invalid mode witness"
                )
                if row["availability"] == "PUBLIC_BYTE_OBJECT":
                    raw = self.objects[row["sha256"]]
                    require(len(raw) == row["bytes"], "object size mismatch")
                    referenced.add(row["sha256"])
                else:
                    require(
                        row["availability"] == "INVENTORY_ONLY_GENERATED_BINARY"
                        and job == "2732-pilot-v1"
                        and name.startswith(".cache/stationary-cuda/")
                        and name.endswith(("/object.o", "/runtime.so")),
                        "unadmitted missing byte object",
                    )
                    excluded[job, name] = row
                rows[name] = row
            self.rows[job] = rows
            require(
                all(
                    str(parent) not in rows
                    for name in rows
                    for parent in PurePosixPath(name).parents
                    if str(parent) != "."
                ),
                "file/directory member collision",
            )
        require(referenced == set(self.objects), "unreferenced object")
        listed = document["excluded_generated_binary_inventory"]
        require(len(listed) == len(excluded) == 3, "excluded binary coverage differs")
        seen = set()
        for row in listed:
            key = row["job"], row["path"]
            require(key not in seen and key in excluded, "duplicate/foreign exclusion")
            seen.add(key)
            require(
                all(
                    row[k] == excluded[key][k]
                    for k in ("bytes", "sha256", "mode", "availability")
                ),
                "exclusion mismatch",
            )

    def has(self, job: str, name: str) -> bool:
        return name in self.rows[job]

    def get(self, job: str, name: str) -> bytes:
        row = self.rows[job][safe_path(name)]
        require(
            row["availability"] == "PUBLIC_BYTE_OBJECT",
            "generated binary is inventory-only; public restoration unavailable",
        )
        return self.objects[row["sha256"]]

    def read(self, job: str, name: str) -> dict | list:
        return json_bytes(self.get(job, name))

    def restore(self, job: str, destination: Path) -> dict:
        # A newly and exclusively created directory avoids overwrites and existing
        # symlink children. Paths were checked before any output is created.
        destination.mkdir(exist_ok=False)
        unavailable = []
        count = 0
        for name, row in self.rows[job].items():
            if row["availability"] != "PUBLIC_BYTE_OBJECT":
                unavailable.append(row)
                continue
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(self.get(job, name))
            if os.name == "posix":
                path.chmod(int(row["mode"], 8))
            count += 1
        return dict(
            restored_public_members=count,
            unavailable=unavailable,
            modes="POSIX witnesses applied"
            if os.name == "posix"
            else "POSIX witnesses retained; Windows mode equivalence not claimed",
        )


def load(root: Path) -> Evidence:
    artifacts = json_bytes((root / "artifacts.json").read_bytes())
    require(
        artifacts["schema"] == "generativeqc.h100-direct-public-artifacts.v1",
        "artifact schema",
    )
    seen = set()
    for row in artifacts["files"]:
        safe_path(row["path"])
        require(
            row["path"] not in seen and type(row["bytes"]) is int, "artifact inventory"
        )
        seen.add(row["path"])
        raw = (root / row["path"]).read_bytes()
        require(
            len(raw) == row["bytes"] and sha(raw) == row["sha256"], "artifact binding"
        )
    return Evidence(decode_xz((root / "terminal-evidence.json.xz").read_bytes()))


def host_history(root: Path) -> dict:
    history = decode_xz((root / "host-history.json.xz").read_bytes())
    require(
        history["schema"] == "generativeqc.h100-direct-host-history.v1",
        "HOST history schema",
    )
    objects = {
        k: base64.b64decode(v, validate=True) for k, v in history["objects"].items()
    }
    require(all(sha(v) == k for k, v in objects.items()), "HOST object SHA")
    seen, referenced = set(), set()
    for row in history["members"] + history["standalone_original_compressed_files"]:
        name = safe_path(row["path"])
        require(name not in seen and type(row["bytes"]) is int, "HOST member inventory")
        seen.add(name)
        if name.endswith(".xz"):
            raw = (root / name).read_bytes()
        else:
            raw = objects[row["sha256"]]
            referenced.add(row["sha256"])
        require(
            len(raw) == row["bytes"] and sha(raw) == row["sha256"],
            "HOST original binding",
        )
    require(referenced == set(objects), "unreferenced HOST object")
    failed = decode_xz((root / "root-host-v2-failed-originals.json.xz").read_bytes())
    require(
        failed["probe_returncode"] == 1
        and failed["verified_control_originals"] == 27
        and failed["full_suite_qualification"].startswith("UNAVAILABLE"),
        "HOST failure history",
    )
    return dict(
        original_uncompressed_members=len(history["members"]),
        standalone_original_compressed_files=len(
            history["standalone_original_compressed_files"]
        ),
        V2_probe_returncode=failed["probe_returncode"],
        V2_verified_control_originals=failed["verified_control_originals"],
        V2_full_suite=failed["full_suite_qualification"],
        source_transport_sha256=failed["source_transport_sha256"],
        mode_scope="initial local policy; not a remote POSIX witness",
    )


def errors(energy: float, forces: list, reference: dict, atoms: int) -> dict:
    require(
        len(forces) == len(reference["forces"]) == atoms
        and all(len(row) == 3 for row in forces),
        "full force vector required",
    )
    difference = [
        x - y
        for row, ref in zip(forces, reference["forces"], strict=True)
        for x, y in zip(row, ref, strict=True)
    ]
    return dict(
        energy_per_atom=abs(energy - reference["energy"]) / atoms,
        force_max=max(map(abs, difference)),
        force_rms=math.sqrt(sum(x * x for x in difference) / len(difference)),
    )


def gate(values: dict, fraction: float = 1) -> bool:
    return all(
        math.isfinite(values[k]) and values[k] <= bound * fraction
        for k, bound in (
            ("energy_per_atom", 1e-5),
            ("force_max", 1e-4),
            ("force_rms", 3e-5),
        )
    )


def frame(
    raw: bytes, index: dict, complete: dict, index_raw: bytes, atoms: int, ordinal: int
) -> dict:
    require(
        type(complete["index"]) is int
        and complete["index"] == ordinal
        and complete["index_sha256"] == sha(index_raw),
        "completion binding",
    )
    require(
        index["schema"] == "generativeqc.h100-direct-return.v1"
        and type(index["data_bytes"]) is int
        and index["data_sha256"] == sha(raw)
        and index["data_bytes"] == len(raw),
        "IEEE data binding",
    )
    seconds_offset = 8 + atoms * 3 * 8
    expected = [
        (["energy"], 0, 8, "<f8", []),
        (["forces"], 8, atoms * 3 * 8, "<f8", [atoms, 3]),
        (["seconds"], seconds_offset, 8, "<f8", []),
        (["repeat"], seconds_offset + 8, 8, "<i8", []),
    ]
    require(
        all(
            type(x["offset"]) is int
            and type(x["bytes"]) is int
            and all(type(n) is int for n in x["shape"])
            for x in index["entries"]
        ),
        "frame integer fields",
    )
    require(
        [
            (x["field"], x["offset"], x["bytes"], x["dtype"], x["shape"])
            for x in index["entries"]
        ]
        == expected
        and len(raw) == seconds_offset + 16,
        "noncanonical complete E/F frame",
    )
    values = struct.unpack(f"<{atoms * 3 + 2}dq", raw)
    require(
        all(math.isfinite(x) for x in values[:-1])
        and values[-2] > 0
        and values[-1] == ordinal,
        "nonfinite return/timer or raw ordinal",
    )
    return dict(
        energy=values[0],
        forces=[list(values[1 + i * 3 : 4 + i * 3]) for i in range(atoms)],
        seconds=values[-2],
        repeat=values[-1],
    )


def birth(e: Evidence, job: str, slot: int, plan: dict, release: dict) -> dict:
    def r(name: str) -> dict | list:
        return e.read(job, f"process-{slot}/control/{name}.json")

    guard, child, ack, terminal = (
        r(n) for n in ("identity", "command-identity", "start-command", "guard-outcome")
    )
    fields = ("pid", "ppid", "pgid", "sid", "start_ticks")
    for value in (guard, child):
        require(
            all(type(value[k]) is int and value[k] >= 0 for k in fields), "birth types"
        )
    nonce = guard["nonce"]
    require(
        nonce and nonce == child["nonce"] == ack["nonce"] == terminal["nonce"],
        "birth nonce",
    )
    require(
        child["ppid"] == guard["pid"]
        and guard["pgid"] == guard["sid"] == guard["pid"]
        and child["start_ticks"] >= guard["start_ticks"],
        "child/guard birth",
    )
    require(
        all(
            ack["guard_identity"][k] == terminal["guard_identity"][k] == guard[k]
            and terminal["command_identity"][k] == child[k]
            for k in fields
        ),
        "terminal birth",
    )
    process = e.read(job, f"process-{slot}/process-outcome.json")
    argv = terminal["command"]
    require(
        ack["command_sha256"] == sha(canonical(argv)) and argv == process["command"],
        "ACK argv",
    )
    task = e.document["original_job_member_inventories"][job][
        "original_envelope_fields"
    ]["task_root"]
    expected = {
        "--plan": task + "/scripts/prepared-plan.json",
        "--plan-sha": release["plan_sha256"],
        "--release": task + "/output/root-release-snapshot.json",
        "--release-sha": sha(canonical(release)),
        "--static": task + "/output/static-before.json",
        "--slot": str(slot),
        "--output": task + f"/output/slot-{slot}",
    }
    for flag, value in expected.items():
        require(
            argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value,
            "actual command binding",
        )
    require(
        terminal["cleanup_success"]
        and process["cleanup_success"]
        and terminal["remaining_owned"] == process["remaining_owned"] == [],
        "guard cleanup",
    )
    if job != "2732-pilot-v1":
        require(
            terminal["returncode"] == process["returncode"] == 0
            and process["status"] == "RETURNED",
            "process completion",
        )
    else:
        require(
            process["status"] == "FAILED"
            and process["returncode"] == 1
            and process["guard_returncode"] == 125,
            "historical failure was changed",
        )
    return dict(
        command={k: child[k] for k in fields},
        guard={k: guard[k] for k in fields},
        nonce=nonce,
        wall_seconds=process["elapsed_seconds"],
    )


def statistics(rows: list[dict]) -> dict:
    require(
        len(rows) == 3
        and all(type(r["process_repeat"]) is int for r in rows)
        and all(r["role"] == "formal" and r.get("cohort_key") for r in rows)
        and len({r["cohort_key"] for r in rows}) == 1
        and {r["process_repeat"] for r in rows} == {0, 1, 2}
        and len(
            {
                tuple(r["birth"]["command"][k] for k in ("pid", "start_ticks"))
                for r in rows
            }
        )
        == 3
        and len({r["birth"]["nonce"] for r in rows}) == 3
        and all(
            len(r["samples"]) == 4
            and all(
                s["gate"] == "PASS"
                and type(s["seconds"]) in (int, float)
                and math.isfinite(s["seconds"])
                and s["seconds"] > 0
                for s in r["samples"]
            )
            for r in rows
        ),
        "three independent complete formal processes required",
    )
    cold = [r["samples"][0]["seconds"] for r in rows]
    warm = [[s["seconds"] for s in r["samples"][1:]] for r in rows]
    medians = [median(values) for values in warm]
    return dict(
        independent_processes=3,
        cold_median=median(cold),
        cold_range=[min(cold), max(cold)],
        process_warm_medians=medians,
        median_of_process_warm_medians=median(medians),
        range_of_process_warm_medians=[min(medians), max(medians)],
        each_warm_array=warm,
    )


def replay(e: Evidence, job: str) -> dict:
    atoms, formal, plan_hash, authority = JOBS[job]

    def r(name: str) -> dict | list:
        return e.read(job, name)

    plan = r("scripts/prepared-plan.json")
    require(
        sha(e.get(job, "scripts/prepared-plan.json"))
        == sha(canonical(plan))
        == plan_hash,
        "frozen plan identity",
    )
    for section in ("task_code", "task_data"):
        for name, digest in plan[section].items():
            require(
                sha(e.get(job, "scripts/" + safe_path(name))) == digest,
                "frozen runtime member",
            )
            require(
                e.rows[job]["scripts/" + name]["mode"] == plan["task_modes"][name],
                "frozen packet POSIX mode witness",
            )
    release = r("root-release.json")
    expected_schema = (
        "generativeqc.h100-direct-wb97mv-formal96-release.v1"
        if formal and atoms == 96
        else "generativeqc.h100-direct-wb97mv-formal-release.v1"
        if formal
        else "generativeqc.h100-direct-wb97mv-pilot96-release.v1"
        if atoms == 96
        else "generativeqc.h100-direct-wb97mv-release.v1"
    )
    require(
        release["authority"] == authority
        and release["schema"] == expected_schema
        and release.get("execution_mode") == plan.get("execution_mode")
        and release["root_coordinator_chat"] == "01a10e8f-7b0a-7b52-a330-b2de34309210"
        and release["plan_sha256"] == plan_hash
        and release["root_nonce"] == plan["root_nonce"]
        and type(release["created_at"]) is str
        and release["created_at"].isascii()
        and release["created_at"].isdecimal(),
        "Root release identity",
    )
    require(
        plan["source_driver"]["revision"] == "cbd4d8074de35105717c8aac7cd1f86babbe90ec"
        and sha(e.get(job, "scripts/frozen-driver.py.txt"))
        == "74f61cc0eb9d1afb54e210eefb2620bbf01e91c2e02d55e5fbe65fbfc88c9055",
        "measured science source identity",
    )
    require(
        plan["creation_window_ms"][0]
        <= int(release["created_at"])
        <= plan["creation_window_ms"][1],
        "creation window",
    )
    order = ROTATION if formal else ARMS
    require(
        plan["first_proposed_stage"]["order"] == list(order)
        and type(release["authorized_processes"]) is int
        and release["authorized_processes"] == len(order)
        and type(release["authorized_ef_calls"]) is int
        and release["authorized_ef_calls"] == 4 * len(order),
        "finite released schedule",
    )
    require(
        e.get(job, "root-release.json")
        == e.get(job, "output/root-release-snapshot.json")
        and r("batch-consumed.json")["release_sha256"] == sha(canonical(release)),
        "one-shot release snapshot",
    )
    allocation = r("root-control/allocation-proof.json")
    require(
        sha(e.get(job, "root-control/allocation-proof.json"))
        == release["allocation_proof"]["sha256"],
        "allocation witness binding",
    )
    require(
        all(
            allocation[k] == release[k]
            for k in ("job_name", "created_at", "rank", "node")
        )
        and allocation["status"] == "RUNNING"
        and allocation["gpu_count"] == 1,
        "allocation locator",
    )
    for name, key in (
        ("status-original", "platform_status_raw_sha256"),
        ("instances-original", "instances_raw_sha256"),
    ):
        require(
            sha(e.get(job, f"root-control/{name}.json")) == allocation[key],
            "platform original SHA",
        )
    for kind in ("static", "device"):
        require(
            e.get(job, f"output/{kind}-before.json")
            == e.get(job, f"output/{kind}-after.json"),
            "source/environment/device changed",
        )
    static, device = r("output/static-before.json"), r("output/device-before.json")
    identity_keys = (
        "source_sha256",
        "environment_sha256",
        "library_sha256",
        "code_sha256",
    )
    require(
        all(static[k] == release[k] for k in identity_keys)
        and static["code_sha256"] == sha(canonical(plan["task_code"]))
        and static["source_sha256"]
        == plan["runtime_bindings"]["source_inventory"]["sha256"]
        and static["environment_sha256"]
        == plan["runtime_bindings"]["qualified_input_identity_sha256"]
        and static["library_sha256"] == plan["runtime_bindings"]["library"]["sha256"],
        "source/environment/code/DSO",
    )
    require(
        all(
            device[k] == release[k]
            for k in ("gpu_uuid", "pci_bus_id", "cgroup_sha256", "hostname")
        )
        and device["gpu_count"] == 1
        and device["cuda_ordinal"] == 0
        and device["sm"] == 90,
        "single H100 cohort",
    )
    oracle_raw = e.get(job, "scripts/independent-oracles.json.xz")
    require(
        sha(oracle_raw)
        == "d61a3e8ba0a8f8a8c08ee4c4b30c8e3e397e67863b8d75e7ba974e90cf0f85a4",
        "oracle identity",
    )
    references = decode_xz(oracle_raw)
    index, input_hash = INPUTS[atoms]
    audit = r("scripts/host-audit.json")
    cell = next(
        t for t in audit["targets"] if t["cell"] == f"water{atoms}-wb97m-v-direct"
    )
    require(
        sha(canonical(cell["input"])) == input_hash
        and references[index]["geometry_bohr"] == cell["input"]["geometry_bohr"],
        "input geometry identity",
    )
    dense = {s["level"]: s for s in references[index]["samples"]}
    require(
        set(dense) == {5, 6} and all(s["converged"] for s in dense.values()),
        "dense references",
    )
    grid = errors(dense[5]["energy"], dense[5]["forces"], dense[6], atoms)
    cpu = next(s for s in references[33]["samples"] if s["level"] == 6)
    gpu = next(s for s in references[27]["samples"] if s["level"] == 6)
    require(
        references[33]["geometry_bohr"] == references[27]["geometry_bohr"]
        and cpu["converged"]
        and gpu["converged"],
        "CPU/GPU spot identity",
    )
    spot = errors(cpu["energy"], cpu["forces"], gpu, 3)
    require(gate(grid, 0.25) and gate(spot, 0.25), "independent quarter gates")
    expected_settings = {
        h["arm"]: h["samples"][0]["settings"] for h in cell["historical"]
    }
    rows, births, guards, nonces = [], set(), set(), set()
    for slot, arm in enumerate(order):
        samples = []
        row = dict(
            slot=slot,
            arm=arm,
            process_repeat=slot // 3 if formal else 0,
            role="formal" if formal else "diagnostic_pilot",
            status="NOT_RUN",
            samples=samples,
        )
        if e.has(job, f"process-{slot}/process-outcome.json"):
            b = birth(e, job, slot, plan, release)
            for values, key in (
                (births, tuple(b["command"][k] for k in ("pid", "start_ticks"))),
                (guards, tuple(b["guard"][k] for k in ("pid", "start_ticks"))),
                (nonces, b["nonce"]),
            ):
                require(key not in values, "duplicate actual birth/nonce")
                values.add(key)
            row["birth"] = b
            worker = r(f"output/slot-{slot}/outcome.json")
            require(
                worker["derivation"] == plan["runtime_adapter_derivation"]
                and worker["execution_decision_sha256"]
                == plan["execution_decision_sha256"],
                "scientific observer/decision provenance",
            )
            require(
                all(
                    worker["actual_identity"][k] == release[k]
                    for k in identity_keys
                    + (
                        "gpu_uuid",
                        "pci_bus_id",
                        "cgroup_sha256",
                        "created_at",
                        "job_name",
                        "node",
                        "rank",
                    )
                ),
                "worker actual cohort",
            )
            if formal or atoms == 96:
                require(
                    type(worker["slot"]) is int
                    and worker["slot"] == slot
                    and worker["arm"] == arm
                    and type(worker["process_repeat"]) is int
                    and worker["process_repeat"] == row["process_repeat"]
                    and worker["role"] == ("formal" if formal else "diagnostic_pilot"),
                    "worker schedule role",
                )
                proof = {k: b[k] for k in ("command", "guard", "nonce")}
                require(
                    all(worker["worker_birth_evidence"][k] == proof[k] for k in proof),
                    "worker birth bridge",
                )
                cache_key = (
                    "formal_cache_cohort_sha256"
                    if atoms == 48
                    else "formal96_cache_cohort_sha256"
                    if formal
                    else "pilot96_cache_cohort_sha256"
                )
                require(
                    worker[cache_key] == plan[cache_key]
                    and worker["input_sha256"] == input_hash,
                    "worker input/cache cohort",
                )
                row["cohort_key"] = sha(
                    canonical(
                        {
                            k: release[k]
                            for k in identity_keys
                            + (
                                "gpu_uuid",
                                "pci_bus_id",
                                "cgroup_sha256",
                                "hostname",
                                "job_name",
                                "created_at",
                            )
                        }
                        | {
                            "cache_cohort_sha256": plan[cache_key],
                            "input_sha256": input_hash,
                        }
                    )
                )
            row["cache"] = {
                phase: dict(
                    hit_miss_status=worker[phase]["hit_miss_status"],
                    roots={
                        name: dict(
                            status=value["status"],
                            files=len(value["files"]),
                            observed_bytes=value["observed_bytes"],
                            missing_during_read=value["missing_during_read"],
                        )
                        for name, value in worker[phase]["roots"].items()
                    },
                )
                for phase in ("cache_before", "cache_after")
            }
            for i in range(4):
                stem = f"output/slot-{slot}/return-{i:03d}"
                if not e.has(job, stem + ".complete.json"):
                    continue
                original = frame(
                    e.get(job, stem + ".bin"),
                    r(stem + ".index.json"),
                    r(stem + ".complete.json"),
                    e.get(job, stem + ".index.json"),
                    atoms,
                    i,
                )
                metadata = r(f"output/slot-{slot}/metadata-{i:03d}.json")
                require(
                    metadata["energy"] == original["energy"]
                    and metadata["forces"] == original["forces"]
                    and metadata["seconds"] == original["seconds"]
                    and metadata["converged"]
                    and metadata["phase"] == ("cold" if i == 0 else "warm")
                    and type(metadata["repeat"]) is int
                    and metadata["repeat"] == max(0, i - 1)
                    and metadata["settings"] == expected_settings[arm],
                    "original phase/default metadata",
                )
                err = errors(original["energy"], original["forces"], dense[6], atoms)
                require(gate(err), "original observable gate")
                precision = metadata.get("precision") or {}
                validation = metadata.get("precision_validation") or {}
                diagnostic = metadata.get("ks_diagnostic") or {}
                samples.append(
                    dict(
                        index=i,
                        seconds=original["seconds"],
                        errors=err,
                        gate="PASS",
                        raw_sha256=sha(e.get(job, stem + ".bin")),
                        iterations=metadata["iterations"],
                        fock_builds=metadata.get("fock_builds"),
                        effective_potential_builds=metadata.get(
                            "effective_potential_builds"
                        ),
                        initial_guess=metadata.get("initial_guess"),
                        precision={
                            key: precision.get(key)
                            for key in (
                                "requested_mode",
                                "effective_bits",
                                "strict_stage_fock_builds",
                                "post_scf_fock_builds",
                                "final_residual_audits",
                                "fallback_count",
                                "execution_retries",
                                "operator_inventory_complete",
                            )
                        },
                        validation={
                            key: validation.get(key)
                            for key in (
                                "status",
                                "scope",
                                "solve_count",
                                "fallback_reasons",
                            )
                        },
                        nonlocal_work=diagnostic.get("nonlocal_work"),
                    )
                )
            if job != "2732-pilot-v1":
                require(
                    len(samples) == 4
                    and worker["status"] == "NUMERIC_PASS_PENDING_JOB_CLOSURE",
                    "incomplete successful process",
                )
                row["status"] = "NUMERIC_PASS"
            else:
                require(
                    len(samples) == 2 and "phase/index mismatch" in worker["error"],
                    "failed pilot history",
                )
                row["status"] = "FAILED_CONTROL_WITH_TWO_NUMERIC_RETURNS"
        rows.append(row)
    batch = r("output/batch-outcome.json")
    require(
        batch["status"]
        == ("FAILED" if job == "2732-pilot-v1" else "NUMERIC_RETURNS_PENDING_CLOSURE"),
        "batch history status",
    )
    return dict(
        atoms=atoms,
        formal=formal,
        source_device_release="PASS archived witness chain",
        plan_sha256=plan_hash,
        complete_EF_returns=sum(len(row["samples"]) for row in rows),
        not_run=4 * len(order) - sum(len(row["samples"]) for row in rows),
        processes=rows,
        statistics_by_arm={
            arm: statistics([row for row in rows if row["arm"] == arm]) for arm in ARMS
        }
        if formal
        else None,
        quarter_grid_errors=grid,
        quarter_spot_errors=spot,
        supervisor_wall_seconds=batch["process_wall_seconds"],
        qualified_speedup=None,
    )


def report(root: Path) -> dict:
    evidence = load(root)
    require(set(evidence.rows) == set(JOBS), "unexpected/missing original job")
    jobs = {job: replay(evidence, job) for job in JOBS}
    require(
        sum(x["complete_EF_returns"] for x in jobs.values()) == 98
        and sum(x["not_run"] for x in jobs.values()) == 10,
        "history coverage",
    )
    return dict(
        schema="generativeqc.h100-direct-public-replay.v1",
        jobs=jobs,
        host_history=host_history(root),
        formal_EF_returns=sum(
            x["complete_EF_returns"] for x in jobs.values() if x["formal"]
        ),
        actual_complete_EF_returns=98,
        not_run=10,
        formal_independent_processes=18,
        scope=(
            "source-matched portable-sm90 formal72 E/F gates; "
            "sizes and physical/cache cohorts separate"
        ),
        unknowns=[
            "complete force-operator precision",
            "peak memory",
            "GPU4PySCF operator precision/fallback/VV10 pairs",
            "actual billing/GPU busy",
            "original measured5852 source manifest",
        ],
        hardware_only=False,
        qualified_speedup=None,
        parent_issue_completion="INCOMPLETE",
        excluded_generated_binary_inventory=evidence.document[
            "excluded_generated_binary_inventory"
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", choices=JOBS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent
    if args.restore:
        parser.error(
            "--output is required for restoration"
        ) if args.output is None else None
        result = load(root).restore(args.restore, args.output)
    else:
        parser.error("--output is only used with --restore") if args.output else None
        result = report(root)
    print(json.dumps(result, indent=2, allow_nan=False))
