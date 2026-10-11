"""Qualify one stable AOT shard's static cubin resource identity (issue #2101).

Issue #2101 established that NVCC register/stack allocation for the ordinary
RHF J/K streaming kernels (dsss/ddss/ddps, stable AOT shard 6) is sensitive to
compilation context even when the generated mathematical source is unchanged.
A default-off candidate perturbed the shared stream header and flipped
unselected kernels from 114/114/158 registers (STACK 0) to 255/154/204-206
(STACK 544) without touching their emitted math.

This tool is the fail-closed qualification gate for that failure mode. It
generates the production shards with the exact CMake generator command,
compiles the selected shard with the exact production release flags through
the verified compiler-cache launcher, dumps the per-kernel cubin resource
table with cuobjdump, and requires the measured identity and every kernel's
REG/STACK/SHARED/LOCAL to match the committed ledger under
``manifests/maintenance/``. Any identity, toolchain, or resource drift fails
with a per-kernel, attributable report. No GPU is required.

``--record-output`` measures and writes a candidate ledger for deliberate
re-qualification. It never modifies the committed ledger and is not a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.evidence import (
    canonical_hash,
    file_hash,
)

MANIFEST = (
    ROOT
    / "python"
    / "generativeqc_compiler"
    / "integral"
    / "production_shell_classes.json"
)
GENERATOR = ROOT / "tools" / "generate_shell_kernels.py"
LEDGER = ROOT / "manifests" / "maintenance" / "direct_jk_aot_resource_ledger.json"

SCHEMA = "generativeqc.direct-jk-aot-resource-ledger"
REPORT_SCHEMA = "generativeqc.direct-jk-aot-resource-report"
SCHEMA_VERSION = 1
COVERAGE = (
    "stable AOT shard 6 (sm_120): ordinary RHF J/K streaming consumers "
    "dsss/ddss/ddps and every co-located kernel"
)
ISSUE = "#2101"

QUALIFIED_SHARDS = 8
QUALIFIED_SHARD_INDEX = 6
QUALIFIED_ARCHITECTURE = "sm_120"
QUALIFIED_UNIT_MODE = "stable-shards"
QUALIFIED_STANDARD = "c++20"
QUALIFIED_SPLIT_COMPILE_THREADS = 8
QUALIFIED_DEFINITIONS = ("GENERATIVEQC_HAS_CUDA=1",)
# Source-tree headers every production shard prelude includes; the AOT header
# contract whose perturbation triggered issue #2101.
PRELUDE_HEADERS = (
    "scf/generated_shell_task.hpp",
    "runtime/compensated_atomic.cuh",
)
RESOURCE_FIELDS = ("REG", "STACK", "SHARED", "LOCAL")


class LedgerError(ValueError):
    """Raised when the committed ledger is missing or structurally invalid."""


def _tool_version(tool: Path) -> str:
    result = subprocess.run(
        [str(tool), "--version"], check=True, capture_output=True, text=True, timeout=30
    )
    return (result.stdout + result.stderr).strip()


def toolchain_versions(nvcc: Path) -> dict[str, str]:
    nvcc = nvcc.resolve()
    return {
        name: _tool_version(nvcc.with_name(name))
        for name in ("nvcc", "ptxas", "cuobjdump")
    }


def generator_abi() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return int(payload["architectures"][QUALIFIED_ARCHITECTURE]["generator_abi"])


def generate_production_shards(output_directory: Path) -> tuple[Path, ...]:
    """Run the exact CMake generator command and return the shard sources."""
    arguments = [
        sys.executable,
        str(GENERATOR),
        "--production-manifest",
        str(MANIFEST),
        "--output-directory",
        str(output_directory),
        "--shards",
        str(QUALIFIED_SHARDS),
        "--unit-mode",
        QUALIFIED_UNIT_MODE,
        "--target-architecture",
        QUALIFIED_ARCHITECTURE,
        "--profile",
        QUALIFIED_ARCHITECTURE,
    ]
    result = subprocess.run(
        arguments, check=False, capture_output=True, text=True, timeout=600
    )
    if result.returncode:
        raise RuntimeError(
            "production AOT generation failed: "
            + (result.stderr or result.stdout).strip()
        )
    profile = output_directory / QUALIFIED_ARCHITECTURE
    shards = tuple(
        profile
        / f"generativeqc_generated_shell_sm{QUALIFIED_ARCHITECTURE[3:]}_shard_{index}.cu"
        for index in range(QUALIFIED_SHARDS)
    )
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise RuntimeError(
            "production AOT generation missed shard sources: " + ", ".join(missing)
        )
    return shards


def _preprocess_command(nvcc: Path, shard: Path, generated: Path) -> list[str]:
    return [
        str(nvcc),
        f"-std={QUALIFIED_STANDARD}",
        f"-arch={QUALIFIED_ARCHITECTURE}",
        *(f"-D{definition}" for definition in QUALIFIED_DEFINITIONS),
        f"-I{ROOT / 'src'}",
        f"-I{generated}",
        "-E",
        str(shard),
    ]


def preprocessed_identity_sha256(nvcc: Path, shard: Path, generated: Path) -> str:
    """SHA256 of the preprocessed translation unit, line markers included.

    Line markers embed the compilation context whose sensitivity issue #2101
    established; the identity deliberately pins them so the gate always runs
    against the same qualified context.
    """
    result = subprocess.run(
        _preprocess_command(nvcc, shard, generated),
        check=False,
        capture_output=True,
        timeout=300,
    )
    if result.returncode:
        raise RuntimeError(
            "preprocessing failed: " + result.stderr.decode(errors="replace")
        )
    return hashlib.sha256(result.stdout).hexdigest()


def cuobjdump_resources(output: str) -> dict[str, dict[str, int]]:
    """Decode per-kernel cubin resources; incomplete or duplicate rows fail.

    Mirrors ``tools/generativeqc_validation/f_shell.py`` so both gates parse
    the same CUDA 12.9 ``--dump-resource-usage`` layout.
    """
    rows: dict[str, dict[str, int]] = {}
    for match in re.finditer(r"[ \t]*Function (\w+):\n([^\n]+)", output):
        fields = {
            key: int(value)
            for key, value in re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)", match[2])
        }
        if set(fields) != set(RESOURCE_FIELDS) or match[1] in rows:
            raise ValueError("incomplete or duplicate CUOBJDump kernel resources")
        rows[match[1]] = fields
    if not rows:
        raise ValueError("CUOBJDump reported no kernel resources")
    return rows


def qualified_flags() -> dict[str, Any]:
    return {
        "architecture": QUALIFIED_ARCHITECTURE,
        "standard": QUALIFIED_STANDARD,
        "optimization": "-O3",
        "split_compile_threads": QUALIFIED_SPLIT_COMPILE_THREADS,
        "definitions": list(QUALIFIED_DEFINITIONS),
        "ptxas_verbose": True,
    }


def compile_identity(
    *,
    shard_sha256: str,
    preprocessed_sha256: str,
    header_sha256: dict[str, str],
    manifest_sha256: str,
    abi: int,
    toolchain: dict[str, str],
) -> dict[str, Any]:
    return {
        "architecture": QUALIFIED_ARCHITECTURE,
        "unit_mode": QUALIFIED_UNIT_MODE,
        "shard_source_sha256": shard_sha256,
        "preprocessed_sha256": preprocessed_sha256,
        "header_sha256": dict(sorted(header_sha256.items())),
        "manifest_sha256": manifest_sha256,
        "generator_abi": abi,
        "toolchain": dict(sorted(toolchain.items())),
        "flags": qualified_flags(),
    }


def compile_shard(
    *,
    nvcc: Path,
    shard: Path,
    generated: Path,
    cache: Path,
    identity: dict[str, Any],
    timeout: float,
) -> tuple[Path, Path, dict[str, Any]]:
    """Compile one shard through the verified cache launcher with a bounded
    identity-keyed object cache; successful hits require matching metadata."""
    key = canonical_hash(identity)
    directory = cache / key
    directory.mkdir(parents=True, exist_ok=True)
    object_path = directory / "shard.o"
    metadata_path = directory / "compile.json"
    log_path = directory / "nvcc.log"
    metadata = None
    if metadata_path.exists():
        candidate = json.loads(metadata_path.read_text())
        if (
            candidate.get("status") == "pass"
            and candidate.get("identity") == identity
            and object_path.is_file()
            and log_path.is_file()
            and candidate.get("object_hash") == file_hash(object_path)
            and candidate.get("log_hash") == file_hash(log_path)
        ):
            metadata = {**candidate, "cache_hit": True}
    if metadata is None:
        compiler = CudaCompilerAdapter(
            nvcc, cuda_target_info(QUALIFIED_ARCHITECTURE), compile_timeout=timeout
        )
        compiled = compiler.compile(
            shard,
            object_path,
            includes=(ROOT / "src", generated),
            options=(
                f"--split-compile={QUALIFIED_SPLIT_COMPILE_THREADS}",
                *(f"-D{definition}" for definition in QUALIFIED_DEFINITIONS),
            ),
            standard=QUALIFIED_STANDARD,
        )
        log_path.write_text(compiled.stdout + compiled.stderr)
        metadata = {
            "status": "pass" if compiled.returncode == 0 else "fail",
            "reason": None
            if compiled.returncode == 0
            else "NVCC timeout"
            if compiled.timed_out
            else "NVCC compilation failed",
            "identity": identity,
            "cache_key": key,
            "cache_hit": False,
            "seconds": compiled.duration_seconds,
            "returncode": compiled.returncode,
            "object_hash": file_hash(object_path) if compiled.returncode == 0 else None,
            "log_hash": file_hash(log_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if metadata["status"] != "pass":
        raise RuntimeError(f"shard compilation failed: {metadata['reason']}")
    return object_path, directory, metadata


def measure_resources(
    nvcc: Path, object_path: Path, directory: Path
) -> dict[str, dict[str, int]]:
    cuobjdump = nvcc.resolve().with_name("cuobjdump")
    dumped = subprocess.run(
        [str(cuobjdump), "--dump-resource-usage", str(object_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    dump_path = directory / "cuobjdump.txt"
    previous_dump = dump_path.read_text() if dump_path.exists() else None
    dump_path.write_text(dumped.stdout)
    with TemporaryDirectory(dir=directory) as extracted:
        subprocess.run(
            [str(cuobjdump), "--extract-elf", "all", str(object_path.resolve())],
            cwd=extracted,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        cubins = [path for path in Path(extracted).glob("*.cubin")]
    if len(cubins) != 1:
        raise RuntimeError(f"expected exactly one release cubin, found {len(cubins)}")
    resources = cuobjdump_resources(dumped.stdout)
    return resources


def validate_ledger(payload: Any) -> None:
    """Strictly validate the committed ledger shape; unknown content fails."""
    if not isinstance(payload, dict):
        raise LedgerError("ledger root must be an object")
    if payload.get("schema") != SCHEMA:
        raise LedgerError("unknown ledger schema")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise LedgerError("unsupported ledger schema_version")
    if payload.get("coverage") != COVERAGE:
        raise LedgerError("ledger coverage does not match the qualified scope")
    if payload.get("issue") != ISSUE:
        raise LedgerError("ledger issue reference does not match the qualified issue")
    for field in (
        "architecture",
        "unit_mode",
        "shard_source_sha256",
        "preprocessed_sha256",
        "manifest_sha256",
    ):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise LedgerError(f"ledger field {field!r} must be a non-empty string")
    if payload.get("architecture") != QUALIFIED_ARCHITECTURE:
        raise LedgerError(
            "ledger architecture does not match the qualified architecture"
        )
    if payload.get("unit_mode") != QUALIFIED_UNIT_MODE:
        raise LedgerError("ledger unit_mode does not match the qualified unit mode")
    if payload.get("generator_abi") != generator_abi():
        raise LedgerError(
            "ledger generator_abi does not match the qualified manifest ABI"
        )
    headers = payload.get("header_sha256")
    if not isinstance(headers, dict) or set(headers) != set(PRELUDE_HEADERS):
        raise LedgerError("ledger header_sha256 must cover exactly the prelude headers")
    if not all(
        isinstance(value, str) and len(value) == 64 for value in headers.values()
    ):
        raise LedgerError("ledger header_sha256 values must be 64-character hashes")
    toolchain = payload.get("toolchain")
    if not isinstance(toolchain, dict) or set(toolchain) != {
        "nvcc",
        "ptxas",
        "cuobjdump",
    }:
        raise LedgerError("ledger toolchain must record nvcc, ptxas, and cuobjdump")
    flags = payload.get("flags")
    if (
        not isinstance(flags, dict)
        or flags.get("architecture") != QUALIFIED_ARCHITECTURE
    ):
        raise LedgerError(
            "ledger flags must be an object for the qualified architecture"
        )
    kernels = payload.get("kernels")
    if not isinstance(kernels, dict) or not kernels:
        raise LedgerError("ledger kernels must be a non-empty object")
    for name, resources in kernels.items():
        if not isinstance(name, str) or not name.strip():
            raise LedgerError("ledger kernel names must be non-empty strings")
        if not isinstance(resources, dict) or set(resources) != set(RESOURCE_FIELDS):
            raise LedgerError(
                f"ledger kernel {name!r} must record exactly REG/STACK/SHARED/LOCAL"
            )
        for field in RESOURCE_FIELDS:
            value = resources.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LedgerError(
                    f"ledger kernel {name!r} {field} must be a non-negative integer"
                )


def compare(ledger: dict[str, Any], measurement: dict[str, Any]) -> list[str]:
    """Return attributable failures; an empty list means exact qualification."""
    failures: list[str] = []
    toolchain = measurement["toolchain"]
    for component in ("nvcc", "ptxas", "cuobjdump"):
        if ledger["toolchain"][component] != toolchain[component]:
            failures.append(
                f"toolchain drift ({component}): measured {toolchain[component]!r} "
                f"!= qualified {ledger['toolchain'][component]!r}; re-qualify with --record-output"
            )
    identity = measurement["identity"]
    for field in (
        "architecture",
        "unit_mode",
        "shard_source_sha256",
        "preprocessed_sha256",
        "manifest_sha256",
        "generator_abi",
    ):
        if ledger.get(field) != identity.get(field):
            failures.append(
                f"identity drift ({field}): measured {identity.get(field)!r} "
                f"!= qualified {ledger.get(field)!r}; regenerate the ledger deliberately"
            )
    if ledger["header_sha256"] != identity["header_sha256"]:
        for header in PRELUDE_HEADERS:
            if ledger["header_sha256"].get(header) != identity["header_sha256"].get(
                header
            ):
                failures.append(
                    f"identity drift (header {header}): measured "
                    f"{identity['header_sha256'].get(header)!r} != qualified "
                    f"{ledger['header_sha256'].get(header)!r}; regenerate the ledger deliberately"
                )
    for field, value in ledger["flags"].items():
        if identity["flags"].get(field) != value:
            failures.append(
                f"identity drift (flags.{field}): measured {identity['flags'].get(field)!r} "
                f"!= qualified {value!r}; regenerate the ledger deliberately"
            )
    qualified_kernels = ledger["kernels"]
    measured_kernels = measurement["kernels"]
    added = sorted(set(measured_kernels) - set(qualified_kernels))
    removed = sorted(set(qualified_kernels) - set(measured_kernels))
    if added or removed:
        failures.append(f"kernel inventory drift: added={added!r} removed={removed!r}")
    for name in sorted(set(qualified_kernels) & set(measured_kernels)):
        for field in RESOURCE_FIELDS:
            qualified = qualified_kernels[name][field]
            measured = measured_kernels[name][field]
            if qualified != measured:
                failures.append(
                    f"kernel resource drift: {name} {field} {qualified} -> {measured}"
                )
    return failures


def measured_ledger(measurement: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "coverage": COVERAGE,
        "issue": ISSUE,
        **measurement["identity"],
        "kernels": dict(sorted(measurement["kernels"].items())),
    }


def run_measurement(
    *,
    nvcc: Path,
    cache: Path,
    output: Path,
    timeout: float,
) -> dict[str, Any]:
    """Generate, preprocess, compile, and resource-dump the qualified shard."""
    generated = output / "generated"
    shards = generate_production_shards(generated)
    shard = shards[QUALIFIED_SHARD_INDEX]
    toolchain = toolchain_versions(nvcc)
    identity = compile_identity(
        shard_sha256=file_hash(shard),
        preprocessed_sha256=preprocessed_identity_sha256(nvcc, shard, generated),
        header_sha256={
            header: file_hash(ROOT / "src" / header) for header in PRELUDE_HEADERS
        },
        manifest_sha256=file_hash(MANIFEST),
        abi=generator_abi(),
        toolchain=toolchain,
    )
    object_path, cache_directory, compile_metadata = compile_shard(
        nvcc=nvcc,
        shard=shard,
        generated=generated,
        cache=cache,
        identity=identity,
        timeout=timeout,
    )
    kernels = dict(
        sorted(measure_resources(nvcc, object_path, cache_directory).items())
    )
    return {
        "identity": identity,
        "toolchain": toolchain,
        "kernels": kernels,
        "object_path": str(object_path),
        "compile": {
            key: compile_metadata[key]
            for key in ("cache_hit", "cache_key", "seconds", "returncode")
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compile-timeout", type=float, default=900)
    parser.add_argument(
        "--record-output",
        type=Path,
        help="write the measured candidate ledger here instead of gating",
    )
    args = parser.parse_args(argv)

    nvcc = args.nvcc.resolve()
    if not nvcc.is_file():
        print(f"nvcc not found: {nvcc}", file=sys.stderr)
        return 2
    cache = args.cache.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    measurement = run_measurement(
        nvcc=nvcc, cache=cache, output=output, timeout=args.compile_timeout
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": None,
        "failures": [],
        "measurement": {
            "identity": measurement["identity"],
            "toolchain": measurement["toolchain"],
            "kernels": measurement["kernels"],
            "object_path": measurement["object_path"],
            "compile": measurement["compile"],
        },
        "ledger": str(LEDGER),
    }

    if args.record_output is not None:
        record = args.record_output.resolve()
        if record == LEDGER.resolve():
            print(
                "--record-output must not overwrite the committed ledger",
                file=sys.stderr,
            )
            return 2
        record.parent.mkdir(parents=True, exist_ok=True)
        payload = measured_ledger(measurement)
        record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        report["status"] = "recorded"
        report["recorded_kernels"] = len(payload["kernels"])
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"recorded {len(payload['kernels'])} kernels to {record}; "
            "commit the ledger deliberately to enable the gate"
        )
        return 0

    if not LEDGER.is_file():
        report["status"] = "fail"
        report["failures"] = [
            f"ledger is missing: {LEDGER}; qualify with --record-output"
        ]
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(report["failures"][0], file=sys.stderr)
        return 1

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    try:
        validate_ledger(ledger)
    except LedgerError as error:
        report["status"] = "fail"
        report["failures"] = [f"ledger validation failed: {error}"]
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(report["failures"][0], file=sys.stderr)
        return 1

    failures = compare(ledger, measurement)
    report["failures"] = failures
    report["status"] = "fail" if failures else "pass"
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(
            f"direct J/K AOT resource qualification failed: {len(failures)} drift(s)",
            file=sys.stderr,
        )
        return 1
    print(
        f"direct J/K AOT resource qualification passed: "
        f"{len(measurement['kernels'])} kernels match the committed ledger"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
