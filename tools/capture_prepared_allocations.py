"""Capture a pinned prepared RHF/UHF lifecycle, not a process-wide heap claim.

Freeze source independently, pin the contract before execution, then collect
both admitted owners continuously. GPU pin/capture must run inside Slurm. The
optional Memray environment and compiled marker are qualification tools only.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_replay_allocations import (
    SCHEMA_V2,
    InvalidReceipt,
    _identity,
    _integer,
    _require,
    _unique_object,
    verify_receipt,
)
from tools.replay_allocation_capture import (
    JournalWindows,
    read_memray_heap,
)

HOST = {
    "owner": "memray-malloc-family",
    "space": "host",
    "counter": "memray-1.20.0-native-heap",
}
DEVICE = {"owner": "hf", "space": "device:0", "counter": "native-device-journal.v1"}
ENDPOINTS = (
    ("energy-first", "endpoint", ("energy",)),
    ("energy-warm", "replay", ("energy",)),
    ("force-first", "endpoint", ("energy", "forces")),
    ("force-warm", "replay", ("energy", "forces")),
    ("moved", "geometry_rebuild", ("energy", "forces")),
)


def digest(path: Path) -> str:
    """Hash exact installed artifact bytes, never a nearby source substitute."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    """Reject duplicate JSON keys before interpreting pinned evidence."""
    return json.loads(path.read_text(), object_pairs_hook=_unique_object)


def write_new(path: Path, value: Any) -> None:
    """Do not overwrite an independently pinned contract or retained evidence."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_tree(root: Path, entries: list[dict[str, str]]) -> str:
    """Reconstruct the Git tree from verified bytes without a remote Git index.

    The export includes executable modes and symlink targets. Git directory
    ordering differs from ordinary lexicographic path ordering; directories
    compare as if suffixed with '/'. Submodules are deliberately unsupported.
    """
    tree: dict[bytes, Any] = {}
    for entry in entries:
        _require(
            set(entry) == {"path", "mode", "blob_sha1", "sha256"},
            "invalid source manifest entry",
        )
        relative = Path(entry["path"])
        _require(
            not relative.is_absolute()
            and relative.as_posix() == entry["path"]
            and all(part not in ("..", ".git") for part in relative.parts)
            and bool(relative.parts),
            "unsafe source manifest path",
        )
        path = root / relative
        mode = entry["mode"]
        _require(mode in ("100644", "100755", "120000"), "unsupported source mode")
        _require(
            not any(parent.is_symlink() for parent in path.parents if parent != root),
            "source path traverses a symlink",
        )
        _require(path.is_symlink() == (mode == "120000"), "source mode mismatch")
        content = (
            os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes()
        )
        if mode != "120000":
            _require(
                bool(path.stat().st_mode & 0o111) == (mode == "100755"),
                "source executable mode mismatch",
            )
        blob = hashlib.sha1(
            f"blob {len(content)}\0".encode() + content, usedforsecurity=False
        ).hexdigest()
        _require(
            blob == entry["blob_sha1"]
            and hashlib.sha256(content).hexdigest() == entry["sha256"],
            f"source bytes mismatch: {relative}",
        )
        branch = tree
        for part in relative.parts[:-1]:
            branch = branch.setdefault(os.fsencode(part), {})
            _require(type(branch) is dict, "source path collision")
        name = os.fsencode(relative.name)
        _require(name not in branch, "duplicate source manifest path")
        branch[name] = (mode, blob)

    def hash_tree(branch: dict[bytes, Any]) -> str:
        ordered = sorted(
            branch,
            key=lambda name: name + (b"/" if type(branch[name]) is dict else b""),
        )
        payload = bytearray()
        for name in ordered:
            value = branch[name]
            mode, object_hash = (
                ("40000", hash_tree(value)) if type(value) is dict else value
            )
            payload.extend(
                mode.encode() + b" " + name + b"\0" + bytes.fromhex(object_hash)
            )
        return hashlib.sha1(
            f"tree {len(payload)}\0".encode() + payload, usedforsecurity=False
        ).hexdigest()

    return hash_tree(tree)


def freeze_source(root: Path, output: Path) -> None:
    """Export a working-tree identity without touching the user's real index."""
    with tempfile.TemporaryDirectory() as folder:
        environment = {**os.environ, "GIT_INDEX_FILE": str(Path(folder) / "index")}

        def git(*arguments: str) -> bytes:
            return subprocess.check_output(
                ["git", *arguments], cwd=root, env=environment
            )

        git("read-tree", "HEAD")
        git("add", "-A")
        entries = []
        for row in git("ls-files", "--stage", "-z").split(b"\0"):
            if not row:
                continue
            metadata, name = row.split(b"\t", 1)
            mode, blob, stage = metadata.decode().split()
            _require(stage == "0", "unmerged source entry")
            path = root / os.fsdecode(name)
            content = (
                os.fsencode(os.readlink(path))
                if path.is_symlink()
                else path.read_bytes()
            )
            entries.append(
                {
                    "path": os.fsdecode(name),
                    "mode": mode,
                    "blob_sha1": blob,
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        manifest = {
            "source_commit": git("rev-parse", "HEAD").decode().strip(),
            "source_tree": git("write-tree").decode().strip(),
            "entries": entries,
        }
        _require(
            source_tree(root, entries) == manifest["source_tree"],
            "source tree mismatch",
        )
        write_new(output, manifest)


def verify_source(root: Path, manifest: dict[str, Any]) -> None:
    """Reject mismatched exports and stale extra production-source files.

    A remote copy may lack Git metadata. Checking only listed hashes would miss
    an added module that can change imports or native generation. Bytecode and
    ignored build artifacts are not source inputs to this fixed direct endpoint.
    """
    _require(
        set(manifest) == {"source_commit", "source_tree", "entries"}, "invalid manifest"
    )
    _require(
        source_tree(root, manifest["entries"]) == manifest["source_tree"],
        "source mismatch",
    )
    paths = {entry["path"] for entry in manifest["entries"]}
    suffixes = {".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp"}
    for folder in ("python", "src", "include", "tools"):
        for path in (root / folder).rglob("*"):
            if path.is_file() and path.suffix in suffixes:
                _require(
                    path.relative_to(root).as_posix() in paths,
                    f"unmanifested source input: {path.relative_to(root)}",
                )


def workload(method: str) -> dict[str, Any]:
    """Pin ragged inputs and an internal displacement, not a rigid translation.

    Moving only the final atom changes molecular distances, so unchanged stale
    integrals/results cannot pass merely through translation invariance. The
    exact moved coordinates belong to the pinned workload, not observer policy.
    """
    hydrogen = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
    water = [(8, (0.0, 0.0, 0.0)), (1, (1.43, 0.0, 1.11)), (1, (-1.43, 0.0, 1.11))]
    systems = [hydrogen, water, hydrogen]
    moved_dz = 0.01
    moved_coordinates = [
        [
            (
                coordinate_x,
                coordinate_y,
                coordinate_z + (moved_dz if atom_index == len(atoms) - 1 else 0.0),
            )
            for atom_index, (
                _,
                (coordinate_x, coordinate_y, coordinate_z),
            ) in enumerate(atoms)
        ]
        for atoms in systems
    ]
    return {
        "method": method,
        "basis": "sto-3g",
        "precision": "fp64",
        "density_fitting": "none",
        "device_id": 0,
        "systems": systems,
        "moved_dz": moved_dz,
        "moved_coordinates": moved_coordinates,
        "endpoints": ENDPOINTS,
        "energy_gate": 1e-10,
        "force_gate": 1e-9,
    }


def schedule() -> list[dict[str, Any]]:
    """Publication is explicit: no observer events disappear between replays."""
    rows = [("prepare", "setup")]
    for name, phase, _ in ENDPOINTS:
        rows.extend(((name, phase), (name + "-publication", "publication")))
    rows.extend((("close", "publication"), ("tail", "publication")))
    return [
        {
            "id": name,
            "phase": phase,
            "zero_new_allocation_domains": [DEVICE] if phase == "replay" else [],
        }
        for name, phase in rows
    ]


def identity(arguments: argparse.Namespace) -> dict[str, Any]:
    """Verify exported source, installed artifacts and the Slurm-assigned GPU."""
    _require(bool(os.environ.get("SLURM_JOB_ID")), "GPU capture/pinning requires Slurm")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _require(bool(visible) and "," not in visible, "request exactly one Slurm GPU")
    _require(not visible.startswith("MIG-"), "MIG capture is not supported")
    # NVML ordinals need not match CUDA's ordering or scheduler remapping.
    # Resolve visible ordinal zero through the same library used for capture.
    native = ctypes.CDLL(str(arguments.library.resolve()))
    count = ctypes.c_int()
    native.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    native.cudaGetDeviceCount.restype = ctypes.c_int
    _require(
        native.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1,
        "expected exactly one visible CUDA device",
    )
    pci_bus = native.cudaDeviceGetPCIBusId
    pci_bus.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    pci_bus.restype = ctypes.c_int
    bus = ctypes.create_string_buffer(32)
    _require(
        pci_bus(bus, len(bus), 0) == 0 and bool(bus.value),
        "cannot identify visible CUDA device 0",
    )
    bus_id = bus.value.decode("ascii")
    device = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={bus_id}",
            "--query-gpu=uuid,mig.mode.current",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    fields = [value.strip() for value in device.split(",")]
    _require(len(fields) == 2 and "\n" not in device, "ambiguous assigned GPU UUID")
    uuid, mig_mode = fields
    _require(
        uuid.startswith("GPU-") and "\n" not in uuid, "ambiguous assigned GPU UUID"
    )
    # A numeric visibility token can also select a MIG instance. Its PCI ID
    # names the parent GPU, not the instance; do not publish that parent UUID.
    _require(
        mig_mode in ("Disabled", "[N/A]"),
        "MIG-enabled or unknown GPU mode is unsupported",
    )
    manifest = load(arguments.source_manifest)
    verify_source(ROOT, manifest)
    paths = {entry["path"] for entry in manifest["entries"]}
    _require(
        {
            "tools/capture_prepared_allocations.py",
            "tools/replay_allocation_capture.py",
            "tools/allocation_audit_marker.cpp",
            "tools/audit_replay_allocations.py",
            "python/generativeqc/resources_native.py",
            "src/runtime/resource_ledger.hpp",
            "src/api/c_api_resources.cpp",
        }
        <= paths,
        "source manifest omits capture/runtime owners",
    )
    case = workload(arguments.method)
    return {
        "source_commit": manifest["source_commit"],
        "source_tree": manifest["source_tree"],
        "library_sha256": digest(arguments.library),
        "artifact_sha256": digest(arguments.marker),
        "workload_sha256": hashlib.sha256(
            json.dumps(case, sort_keys=True).encode()
        ).hexdigest(),
        "toolchain": arguments.toolchain,
        "device": uuid + "/visible:0",
        "endpoint": f"Calculator.prepare_batch({arguments.method},direct,sto-3g).execute(energy,forces)",
    }


def assemble_windows(
    heap: dict[str, Any],
    boundaries: list[tuple[str, bool, dict[str, Any]]],
    tail: dict[str, Any],
) -> list[dict[str, Any]]:
    """Join marker boundaries and device fences without gaps or observer aliasing.

    Endpoint boundaries are before the marker, assigning marker/snapshot payloads
    to publication. Setup/publication boundaries include their ending marker.
    Finalized host EOF is consumed by tail; it cannot be silently discarded.
    """
    specifications = schedule()
    _require(
        [name for name, _, _ in boundaries]
        == [row["id"] for row in specifications[:-1]],
        "missing/reordered capture boundary",
    )
    host_windows, device_windows = JournalWindows(), JournalWindows()
    windows = []
    for specification, (name, after, device) in zip(
        specifications, boundaries, strict=False
    ):
        _require(
            type(after) is bool
            and after == (specification["phase"] in ("setup", "publication")),
            "wrong marker side for execution/publication",
        )
        position = heap["marker_positions"][name] + int(after)
        windows.append(
            {
                "id": name,
                "phase": specification["phase"],
                "observations": [
                    host_windows.take(heap, position),
                    device_windows.take(device),
                ],
            }
        )
    windows.append(
        {
            "id": "tail",
            "phase": "publication",
            "observations": [
                host_windows.take(heap),
                device_windows.take(tail),
            ],
        }
    )
    _require(
        windows[-1]["observations"][1]["metrics"]["live_bytes"] == 0,
        "device owner leaked",
    )
    return windows


def numerical_evidence(
    results: list[Any], targets: list[Any], *, match_iterations: bool = True
) -> list[dict[str, Any]]:
    """Compare energies/forces and, by default, matched ordinary iteration work.

    An independently executing reference backend may have different convergence
    histories. Its caller can retain those counts without requiring equality;
    the energy/force gates and successful convergence remain mandatory.
    """
    import numpy as np

    _require(type(match_iterations) is bool, "invalid iteration comparison policy")
    evidence = []
    for (name, _, properties), actual, target in zip(
        ENDPOINTS, results, targets, strict=True
    ):
        pairs = list(zip(actual.items, target.items, strict=True))
        _require(
            bool(pairs)
            and all(
                item.succeeded and item.converged for pair in pairs for item in pair
            ),
            "failed/unconverged numerical comparison",
        )
        energies = [abs(item.energy - expected.energy) for item, expected in pairs]
        _require(
            all(bool(np.isfinite(error)) for error in energies), "nonfinite energy"
        )
        energy = max(energies)
        force = None
        if "forces" in properties:
            _require(
                all(
                    item.forces.shape == expected.forces.shape
                    for item, expected in pairs
                ),
                "force shape mismatch",
            )
            forces = [
                float(np.max(np.abs(item.forces - expected.forces)))
                for item, expected in pairs
            ]
            _require(
                all(bool(np.isfinite(error)) for error in forces), "nonfinite force"
            )
            force = max(forces)
        _require(
            bool(np.isfinite(energy)) and energy <= 1e-10, "energy acceptance failed"
        )
        _require(
            force is None or (bool(np.isfinite(force)) and force <= 1e-9),
            "force acceptance failed",
        )
        if match_iterations:
            _require(
                all(item.iterations == expected.iterations for item, expected in pairs),
                "matched ordinary iteration counts differ",
            )
        for item, expected in pairs:
            _integer(item.iterations, "actual iterations")
            _integer(expected.iterations, "reference iterations")
        evidence.append(
            {
                "id": name,
                "energy_error": energy,
                "force_error": force,
                "iterations": [item.iterations for item, _ in pairs],
                "reference_iterations": [item.iterations for _, item in pairs],
            }
        )
    return evidence


def capture(arguments: argparse.Namespace, actual_identity: dict[str, Any]) -> None:
    """Execute only public prepared endpoints, with both observation owners alive."""
    import memray

    _require(memray.__version__ == "1.20.0", "requires optional Memray 1.20.0")
    import generativeqc
    from generativeqc import Calculator, ResourceBudget
    from generativeqc.resources_native import NativeDeviceJournal, NativeDeviceLedger

    _require(
        Path(generativeqc.__file__).resolve()
        == ROOT / "python/generativeqc/__init__.py",
        "imported package does not match pinned checkout",
    )

    expected = load(arguments.expected)
    _require(
        expected
        == {
            "schema": SCHEMA_V2,
            "identity": actual_identity,
            "domains": [HOST, DEVICE],
            "windows": schedule(),
        },
        "pinned contract mismatch",
    )
    _require(not arguments.output.exists(), "capture output already exists")
    arguments.output.mkdir(parents=True)
    _require(
        Path(os.environ.get("GENERATIVEQC_LIBRARY", "")).resolve()
        == arguments.library.resolve(),
        "GENERATIVEQC_LIBRARY must select the pinned library",
    )
    native = ctypes.CDLL(str(arguments.library.resolve()))
    native.cudaSetDevice.argtypes = [ctypes.c_int]
    native.cudaSetDevice.restype = ctypes.c_int
    native.cudaDeviceSynchronize.argtypes = []
    native.cudaDeviceSynchronize.restype = ctypes.c_int
    _require(native.cudaSetDevice(0) == 0, "cannot select visible GPU 0")

    def synchronize() -> None:
        _require(native.cudaDeviceSynchronize() == 0, "CUDA fence failed")

    marker = ctypes.CDLL(str(arguments.marker.resolve()))
    allocate = marker.generativeqc_audit_marker_allocate
    release = marker.generativeqc_audit_marker_release
    allocate.argtypes, allocate.restype = [ctypes.c_size_t], ctypes.c_void_p
    release.argtypes, release.restype = [ctypes.c_void_p], None
    case = workload(arguments.method)
    systems = case["systems"]
    coordinates = case["moved_coordinates"]
    config = {
        key: case[key]
        for key in ("method", "basis", "precision", "density_fitting", "device_id")
    }
    reference = Calculator(device="cuda", **config)
    plan = reference.estimate_resources(systems).require_feasible()
    budget = ResourceBudget(
        device_bytes=plan.peak_bytes["device"], host_bytes=plan.peak_bytes["host"]
    )
    reference = Calculator(device="cuda", resource_budget=budget, **config)
    targets = []
    with reference.prepare_batch(systems) as ordinary:
        for name, _, properties in ENDPOINTS:
            targets.append(
                ordinary.execute(
                    strict=True,
                    properties=properties,
                    **({"coordinates": coordinates} if name == "moved" else {}),
                )
            )
    synchronize()
    captures: list[Any] = []

    class CapturedLedger(NativeDeviceLedger):
        """Attach to the production owner before its preparation binding."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            _require(
                Path(self.library._name).resolve() == arguments.library.resolve(),
                "prepared owner loaded a different library",
            )
            captures.append(NativeDeviceJournal(self, capacity=4096))

    markers: dict[str, tuple[int, int]] = {}
    boundaries: list[tuple[str, bool, dict[str, Any]]] = []
    results: list[Any] = []
    prepared = None

    def boundary(name: str, *, after: bool) -> None:
        synchronize()
        size = 3001 + 2 * len(markers)
        snapshot = captures[0].snapshot() if after else None
        pointer = allocate(size)
        _require(bool(pointer), "marker allocation failed")
        markers[name] = (pointer, size)
        if not after:
            snapshot = captures[0].snapshot()
        boundaries.append((name, after, snapshot))

    raw = arguments.output / "host.bin"
    with memray.Tracker(
        str(raw),
        file_format=memray.FileFormat.ALL_ALLOCATIONS,
        trace_python_allocators=False,
    ):
        try:
            calculator = Calculator(device="cuda", resource_budget=budget, **config)
            with patch(
                "generativeqc.resources_native.NativeDeviceLedger", CapturedLedger
            ):
                prepared = calculator.prepare_batch(systems)
            _require(len(captures) == 1, "expected exactly one native owner")
            boundary("prepare", after=True)
            for name, _, properties in ENDPOINTS:
                results.append(
                    prepared.execute(
                        strict=True,
                        properties=properties,
                        **({"coordinates": coordinates} if name == "moved" else {}),
                    )
                )
                boundary(name, after=False)
                boundary(name + "-publication", after=True)
            prepared.close()
            boundary("close", after=True)
            synchronize()
            tail = captures[0].snapshot()
        finally:
            if prepared is not None:
                prepared.close()
            synchronize()
            for pointer, _ in markers.values():
                release(pointer)
            for journal in captures:
                journal.close()
    _require(
        identity(arguments) == actual_identity,
        "source/artifacts changed during capture",
    )
    heap = read_memray_heap(raw, markers=markers, max_events=arguments.max_host_events)
    evidence = numerical_evidence(results, targets)
    receipt = {
        "schema": SCHEMA_V2,
        "identity": actual_identity,
        "execution": {"kind": "runtime", "completed": True, "source_matched": True},
        "windows": assemble_windows(heap, boundaries, tail),
    }
    verification = verify_receipt(receipt, expected)
    write_new(arguments.output / "receipt.json", receipt)
    write_new(arguments.output / "verification.json", verification)
    write_new(
        arguments.output / "diagnostics.json",
        {
            "numerical": evidence,
            "host_exclusions": heap["excluded"],
            "host_raw_events": heap["raw_count"],
            "markers": markers,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "host_scope": "intercepted malloc-family requests begun after tracker activation; excludes reference warmup, Python suballocators, mapped layers and untracked frees",
            "no_endpoint_timing_claim": True,
        },
    )
    _require(verification["status"] == "PASS", str(verification["errors"]))
    print(json.dumps({"status": "PASS", "output": str(arguments.output)}))


def main(argv: list[str] | None = None) -> int:
    """Freeze on the checkout host; pin/capture inside a finite Slurm GPU job."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-source", type=Path)
    mode.add_argument("--pin", action="store_true")
    mode.add_argument("--capture", action="store_true")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--method", choices=("rhf", "uhf"), default="rhf")
    parser.add_argument("--toolchain")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-host-events", type=int, default=1 << 20)
    arguments = parser.parse_args(argv)
    try:
        if arguments.freeze_source:
            freeze_source(ROOT, arguments.freeze_source)
            return 0
        _require(
            all(
                (
                    arguments.source_manifest,
                    arguments.expected,
                    arguments.library,
                    arguments.marker,
                    arguments.toolchain,
                )
            ),
            "missing pinned inputs",
        )
        actual = identity(arguments)
        _identity(actual)
        _require(1 <= arguments.max_host_events <= 1 << 20, "invalid host event limit")
        if arguments.pin:
            write_new(
                arguments.expected,
                {
                    "schema": SCHEMA_V2,
                    "identity": actual,
                    "domains": [HOST, DEVICE],
                    "windows": schedule(),
                },
            )
        else:
            _require(arguments.output is not None, "missing output directory")
            capture(arguments, actual)
        return 0
    except (
        InvalidReceipt,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
