"""Collect source-pinned prepared CUDA residency activity, never a fake PASS.

This command emits actual CUPTI transfer/sync diagnostics plus matched numerical
evidence. Scientific payload/dependency annotation is deliberately unresolved;
a successful collection exits 2 (INCOMPLETE), matching the residency auditor's
semantics. Use one fresh process per case inside a finite Slurm GPU allocation.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_replay_allocations import InvalidReceipt, _require
from tools.capture_prepared_allocations import (
    ENDPOINTS,
    digest,
    freeze_source,
    load,
    numerical_evidence,
    verify_source,
    workload,
    write_new,
)
from tools.cupti_graph_inventory import summarize_graphs
from tools.cupti_residency_capture import CuptiCapture, summarize
from tools.cupti_source_capture import SourceInventory, summarize_sources
from tools.cupti_source_ratchet import check_work_ratchet, load_work_ratchet


def regions() -> dict[int, dict[str, str]]:
    """Observer completion fences remain visible, outside production counters."""
    return {
        1: {"name": "prepare", "role": "prepare"},
        **{
            index: {"name": name, "role": phase}
            for index, (name, phase, _) in enumerate(ENDPOINTS, 2)
        },
        7: {"name": "close", "role": "publication"},
        8: {"name": "collector-fences", "role": "observer"},
    }


def contract(arguments: argparse.Namespace) -> dict[str, Any]:
    """Check actual source/artifacts and the GPU assigned by Slurm before work."""
    _require(bool(os.environ.get("SLURM_JOB_ID")), "pin/capture requires Slurm")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _require(bool(visible) and "," not in visible, "request exactly one Slurm GPU")
    uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={visible}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    _require(
        uuid.startswith("GPU-") and "\n" not in uuid, "invalid assigned GPU identity"
    )
    manifest = load(arguments.source_manifest)
    verify_source(ROOT, manifest)
    _require(
        {
            "tools/cupti_residency_capture.cpp",
            "tools/cupti_residency_capture.py",
            "tools/capture_prepared_residency.py",
            "tools/cupti_graph_inventory.cpp",
            "tools/cupti_graph_inventory.py",
            "tools/cupti_source_capture.cpp",
            "tools/cupti_source_capture.py",
            "tools/cupti_source_ratchet.py",
            "src/runtime/residency_observer.hpp",
            "src/runtime/residency_boundaries.hpp",
            "src/runtime/residency_cuda.cuh",
            "src/runtime/resource_cuda.cuh",
            "src/scf/cuda_rhf.cpp",
            "src/scf/cuda/rhf_graph.cpp",
            "src/scf/cuda/rhf_graph.hpp",
            "src/scf/cuda/resources.cpp",
            "src/scf/cuda/eigensolver.cpp",
            "src/api/c_api_resources.cpp",
        }
        <= {row["path"] for row in manifest["entries"]},
        "manifest omits collector source",
    )
    case = json.loads(json.dumps(workload(arguments.method)))
    ratchet = (
        load_work_ratchet(arguments.work_ratchet, case)
        if arguments.work_ratchet is not None
        else None
    )
    return {
        "schema": "generativeqc.cupti-prepared.v1",
        "identity": {
            "source_commit": manifest["source_commit"],
            "source_tree": manifest["source_tree"],
            "library_sha256": digest(arguments.library),
            "collector_sha256": digest(arguments.collector),
            "cupti_sha256": digest(arguments.cupti_library),
            "cupti_api_version": 28,
            "source_observer_layout": 2,
            "device": uuid + "/visible:0",
            "toolchain": arguments.toolchain,
        },
        "workload": case,
        "observed_work_ratchet": ratchet,
        "numerical_references": {
            "matched_history": "ordinary CUDA prepared history",
            "independent_backend": "native CPU FP64 direct prepared history",
            "ordinary_iterations_must_match": True,
        },
        "capacity": arguments.capacity,
        "graph_node_limit": arguments.graph_node_limit,
        "graph_depth_limit": arguments.graph_depth_limit,
        "regions": {str(key): value for key, value in regions().items()},
        "required_remaining_evidence": "source-owned role/payload/dependency links and graph/implicit-wait coverage",
    }


def capture(arguments: argparse.Namespace, expected: dict[str, Any]) -> int:
    """Measure actual public endpoints; keep observer synchronization separate."""
    import generativeqc
    from generativeqc import Calculator, ResourceBudget

    _require(
        Path(generativeqc.__file__).resolve()
        == ROOT / "python/generativeqc/__init__.py",
        "imported checkout mismatch",
    )
    _require(
        Path(os.environ.get("GENERATIVEQC_LIBRARY", "")).resolve()
        == arguments.library.resolve(),
        "GENERATIVEQC_LIBRARY must select pinned native library",
    )
    _require(load(arguments.expected) == expected, "independent contract mismatch")
    _require(not arguments.output.exists(), "output directory already exists")
    arguments.output.mkdir(parents=True)
    case = workload(arguments.method)
    systems = case["systems"]
    coordinates = [
        [
            (coordinate_x, coordinate_y, coordinate_z + case["moved_dz"])
            for _, (coordinate_x, coordinate_y, coordinate_z) in atoms
        ]
        for atoms in systems
    ]
    config = {
        key: case[key]
        for key in ("method", "basis", "precision", "density_fitting", "device_id")
    }
    native = ctypes.CDLL(str(arguments.library.resolve()))
    native.cudaSetDevice.argtypes, native.cudaSetDevice.restype = (
        [ctypes.c_int],
        ctypes.c_int,
    )
    native.cudaDeviceSynchronize.argtypes, native.cudaDeviceSynchronize.restype = (
        [],
        ctypes.c_int,
    )
    _require(native.cudaSetDevice(0) == 0, "cannot select assigned visible device")

    def fence() -> None:
        _require(native.cudaDeviceSynchronize() == 0, "CUDA completion fence failed")

    ordinary = Calculator(device="cuda", **config)
    plan = ordinary.estimate_resources(systems).require_feasible()
    budget = ResourceBudget(
        device_bytes=plan.peak_bytes["device"], host_bytes=plan.peak_bytes["host"]
    )
    ordinary = Calculator(device="cuda", resource_budget=budget, **config)
    targets = []
    with ordinary.prepare_batch(systems) as prepared:
        for name, _, properties in ENDPOINTS:
            targets.append(
                prepared.execute(
                    strict=True,
                    properties=properties,
                    **({"coordinates": coordinates} if name == "moved" else {}),
                )
            )
    cpu_targets = []
    cpu_reference = Calculator(device="cpu", **config)
    with cpu_reference.prepare_batch(systems) as prepared:
        for name, _, properties in ENDPOINTS:
            cpu_targets.append(
                prepared.execute(
                    strict=True,
                    properties=properties,
                    **({"coordinates": coordinates} if name == "moved" else {}),
                )
            )
    fence()
    profiler = CuptiCapture(arguments.collector, capacity=arguments.capacity)
    results = []
    prepared = None
    source_profiler = None
    try:
        mapped = {
            Path(row.split()[-1]).resolve()
            for row in Path("/proc/self/maps").read_text().splitlines()
            if "libcupti.so" in row and row.split()[-1].startswith("/")
        }
        _require(
            mapped == {arguments.cupti_library.resolve()},
            "loaded CUPTI library mismatch",
        )
        source_profiler = SourceInventory(
            arguments.collector,
            native,
            capacity=arguments.capacity,
            node_limit=arguments.graph_node_limit,
            depth_limit=arguments.graph_depth_limit,
        )
        with profiler.region(1):
            calculator = Calculator(device="cuda", resource_budget=budget, **config)
            _require(
                Path(calculator._library._name).resolve()
                == arguments.library.resolve(),
                "calculator loaded different library",
            )
            prepared = calculator.prepare_batch(systems)
        with profiler.region(8):
            fence()
        for index, (name, _, properties) in enumerate(ENDPOINTS, 2):
            with profiler.region(index):
                results.append(
                    prepared.execute(
                        strict=True,
                        properties=properties,
                        **({"coordinates": coordinates} if name == "moved" else {}),
                    )
                )
            with profiler.region(8):
                fence()
    finally:
        try:
            if prepared is not None:
                with profiler.region(7):
                    prepared.close()
            with profiler.region(8):
                fence()
        finally:
            try:
                if source_profiler is not None:
                    graph_raw, source_raw = source_profiler.stop()
                    write_new(arguments.output / "graphs.json", graph_raw)
                    write_new(arguments.output / "source-boundaries.json", source_raw)
            finally:
                raw = profiler.stop()
                write_new(arguments.output / "activity.json", raw)
    _require(
        contract(arguments) == expected, "source/artifacts changed during collection"
    )
    result = summarize(raw, regions())
    graph_result = summarize_graphs(graph_raw, raw, regions())
    source_result = summarize_sources(source_raw, raw, regions())
    ratchet_result = (
        check_work_ratchet(
            expected["observed_work_ratchet"],
            source_raw,
            raw,
            regions(),
            expected["workload"],
        )
        if expected["observed_work_ratchet"] is not None
        else None
    )
    write_new(arguments.output / "graph-diagnostics.json", graph_result)
    write_new(arguments.output / "source-diagnostics.json", source_result)
    if ratchet_result is not None:
        write_new(arguments.output / "source-work-ratchet.json", ratchet_result)
    write_new(arguments.output / "diagnostics.json", result)
    write_new(
        arguments.output / "numerical.json",
        {
            "source_matched": True,
            "completed": True,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "numerical": numerical_evidence(results, targets),
            "reference_kind": expected["numerical_references"]["matched_history"],
            "independent_reference_kind": expected["numerical_references"][
                "independent_backend"
            ],
            "independent_numerical": numerical_evidence(
                results, cpu_targets, match_iterations=False
            ),
            "no_endpoint_timing_claim": True,
        },
    )
    _require(result["activity_stream_intact"], str(result["issues"]))
    _require(graph_result["source_inventory_intact"], str(graph_result["issues"]))
    _require(source_result["source_annotations_intact"], str(source_result["issues"]))
    if ratchet_result is not None:
        _require(
            ratchet_result["status"] == "WITHIN_OBSERVED_RATCHET",
            "observed work ratchet requires review: "
            + json.dumps(
                {key: ratchet_result[key] for key in ("status", "issues", "violations")}
            ),
        )
    print(
        json.dumps(
            {
                "status": "INCOMPLETE",
                "collection": "intact",
                "output": str(arguments.output),
                "remaining": expected["required_remaining_evidence"],
                "observed_work_ratchet": ratchet_result["status"]
                if ratchet_result
                else "not-configured",
            }
        )
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    """Freeze independently, pin before capture, retain actual raw activity."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-source", type=Path)
    mode.add_argument("--pin", action="store_true")
    mode.add_argument("--capture", action="store_true")
    for name in (
        "source-manifest",
        "library",
        "collector",
        "cupti-library",
        "expected",
        "output",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--toolchain")
    parser.add_argument(
        "--work-ratchet",
        type=Path,
        help="independently reviewed explicit-work limits; never a full residency PASS",
    )
    parser.add_argument("--method", choices=("rhf", "uhf"), default="rhf")
    parser.add_argument("--capacity", type=int, default=1 << 18)
    parser.add_argument("--graph-node-limit", type=int, default=4096)
    parser.add_argument("--graph-depth-limit", type=int, default=16)
    arguments = parser.parse_args(argv)
    try:
        if arguments.freeze_source:
            freeze_source(ROOT, arguments.freeze_source)
            return 0
        _require(
            all(
                (
                    arguments.source_manifest,
                    arguments.library,
                    arguments.collector,
                    arguments.cupti_library,
                    arguments.expected,
                    arguments.toolchain,
                )
            ),
            "missing pinned inputs",
        )
        _require(1 <= arguments.capacity <= 1 << 20, "invalid activity capacity")
        _require(1 <= arguments.graph_node_limit <= 4096, "invalid graph node limit")
        _require(1 <= arguments.graph_depth_limit <= 16, "invalid graph depth limit")
        expected = contract(arguments)
        if arguments.pin:
            write_new(arguments.expected, expected)
            return 0
        _require(arguments.output is not None, "missing output directory")
        return capture(arguments, expected)
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
