"""Source integration gates with CPU doubles, never CUDA qualification."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_prepared_allocation_capture import device_runtime

from tools import capture_prepared_residency as capture
from tools.audit_replay_allocations import InvalidReceipt
from tools.capture_prepared_allocations import workload


@pytest.fixture
def contract_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-contract-double")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    owners = (
        "tools/capture_prepared_allocations.py",
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
    )
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "source_commit": "4" * 40,
                "source_tree": "5" * 40,
                "entries": [{"path": path} for path in owners],
            }
        )
    )
    binary = tmp_path / "not-a-library"
    binary.write_bytes(b"metadata-double-not-executable")
    monkeypatch.setattr(capture, "verify_source", lambda *_: None)
    return SimpleNamespace(
        source_manifest=manifest,
        library=binary,
        collector=binary,
        cupti_library=binary,
        work_ratchet=None,
        method="rhf",
        toolchain="CPU metadata test, not GPU proof",
        capacity=64,
        graph_node_limit=64,
        graph_depth_limit=4,
    )


@pytest.mark.parametrize("visible", ["0", "3", "GPU-assigned"])
def test_residency_contract_uses_runtime_visible_device(
    monkeypatch: pytest.MonkeyPatch, contract_inputs: Any, visible: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    loaded = []
    calls = []
    monkeypatch.setattr(
        capture.ctypes, "CDLL", lambda path: (loaded.append(path), device_runtime())[1]
    )

    def query(arguments: list[str], **_: Any) -> str:
        calls.append(arguments)
        return "GPU-assigned, Disabled\n"

    monkeypatch.setattr(capture.subprocess, "check_output", query)
    pinned = capture.contract(contract_inputs)
    assert pinned["identity"]["device"] == "GPU-assigned/visible:0"
    assert loaded == [str(contract_inputs.library.resolve())]
    assert calls == [
        [
            "nvidia-smi",
            "--id=0000:03:00.0",
            "--query-gpu=uuid,mig.mode.current",
            "--format=csv,noheader",
        ]
    ]
    assert pinned["workload"]["moved_coordinates"] == json.loads(
        json.dumps(workload("rhf")["moved_coordinates"])
    )
    assert pinned["observed_work_ratchet"] is None


@pytest.mark.parametrize(
    "options",
    [
        {"count_status": 1},
        {"count": 0},
        {"count": 2},
        {"pci_status": 1},
        {"bus": b""},
    ],
)
def test_residency_contract_rejects_failed_runtime_probe(
    monkeypatch: pytest.MonkeyPatch, contract_inputs: Any, options: dict[str, Any]
) -> None:
    monkeypatch.setattr(capture.ctypes, "CDLL", lambda _: device_runtime(**options))
    monkeypatch.setattr(
        capture.subprocess,
        "check_output",
        lambda *_args, **_kwargs: pytest.fail("no NVML fallback"),
    )
    with pytest.raises(InvalidReceipt):
        capture.contract(contract_inputs)


@pytest.mark.parametrize("mode", ["Enabled", "Unknown", ""])
def test_residency_contract_rejects_numeric_mig_parent(
    monkeypatch: pytest.MonkeyPatch, contract_inputs: Any, mode: str
) -> None:
    monkeypatch.setattr(capture.ctypes, "CDLL", lambda _: device_runtime())
    monkeypatch.setattr(
        capture.subprocess,
        "check_output",
        lambda *_args, **_kwargs: f"GPU-assigned, {mode}\n",
    )
    with pytest.raises(InvalidReceipt, match="MIG"):
        capture.contract(contract_inputs)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_residency_histories_execute_exact_pinned_internal_displacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str
) -> None:
    import generativeqc

    calls = []
    case = json.loads(json.dumps(workload(method)))
    arguments = SimpleNamespace(
        method=method,
        library=tmp_path / "native-double",
        expected=tmp_path / "expected",
        output=tmp_path / "output",
        collector=tmp_path / "collector",
        capacity=64,
        cupti_library=tmp_path / "libcupti.so",
        graph_node_limit=64,
        graph_depth_limit=4,
    )
    expected = {"workload": case}
    arguments.expected.write_text(json.dumps(expected))
    monkeypatch.setenv("GENERATIVEQC_LIBRARY", str(arguments.library))

    class Prepared:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def execute(self, **options: Any) -> Any:
            calls.append(options)
            return SimpleNamespace()

        def close(self) -> None:
            pass

    class Calculator:
        def __init__(self, **_: Any) -> None:
            self._library = SimpleNamespace(_name=str(arguments.library))

        def estimate_resources(self, _: Any) -> Any:
            return SimpleNamespace(
                require_feasible=lambda: SimpleNamespace(
                    peak_bytes={"device": 1, "host": 1}
                )
            )

        def prepare_batch(self, _: Any) -> Any:
            return Prepared()

    def success(*_: Any) -> int:
        return 0

    monkeypatch.setattr(generativeqc, "Calculator", Calculator)
    monkeypatch.setattr(
        capture.ctypes,
        "CDLL",
        lambda _: SimpleNamespace(cudaSetDevice=success, cudaDeviceSynchronize=success),
    )

    original_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            f"synthetic-mapping {arguments.cupti_library}\n"
            if str(path) == "/proc/self/maps"
            else original_read(path, *args, **kwargs)
        ),
    )
    monkeypatch.setattr(
        capture,
        "CuptiCapture",
        lambda *_args, **_kwargs: SimpleNamespace(
            region=lambda _: nullcontext(), stop=dict
        ),
    )
    monkeypatch.setattr(
        capture,
        "SourceInventory",
        lambda *_args, **_kwargs: SimpleNamespace(stop=lambda: ({}, {})),
    )
    # Stop after all actual endpoint calls, before any numerical/receipt verdict.
    # No synthetic observation is persisted as runtime evidence.
    monkeypatch.setattr(capture, "write_new", lambda *_: None)

    class AfterObservedEndpoints(Exception):
        pass

    def stop_after_endpoints(*_args: Any, **_kwargs: Any) -> Any:
        raise AfterObservedEndpoints

    monkeypatch.setattr(capture, "contract", stop_after_endpoints)
    with pytest.raises(AfterObservedEndpoints):
        capture.capture(arguments, expected)
    assert len(calls) == 15
    moved = [options["coordinates"] for options in calls if "coordinates" in options]
    assert moved == [case["moved_coordinates"]] * 3
    for atoms, coordinates in zip(case["systems"], moved[0], strict=True):
        assert coordinates[0] == atoms[0][1]
        assert coordinates[-1][2] == atoms[-1][1][2] + case["moved_dz"]
