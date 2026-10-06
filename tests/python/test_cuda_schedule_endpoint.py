"""Schedule evidence must observe production policy, not replace it."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from benchmarks import qualify_cuda_schedule
from benchmarks.readme_pbe0 import PBE0

if TYPE_CHECKING:
    from pathlib import Path


def test_schedule_source_receipt_binds_all_trial_axes_and_ao_policy() -> None:
    from benchmarks.readme_omol25 import source_hashes

    hashes = source_hashes()
    assert {
        "python/generativeqc/_force_active_ao.py",
        "python/generativeqc/batch.py",
        "python/generativeqc_compiler/dft/ao_cuda.py",
        "python/generativeqc_compiler/dft/xc_contraction_cuda.py",
        "python/generativeqc_compiler/method/stationary_cuda.py",
        "python/generativeqc_compiler/method/stationary_resources.py",
    } <= hashes.keys()


@pytest.mark.parametrize("intrusive", [False, True])
def test_schedule_endpoint_preserves_policy_and_protocol(
    monkeypatch: pytest.MonkeyPatch, intrusive: bool
) -> None:
    calls = []
    payload = {"resident_ao_cutoff": 1e-16, "tile_points": 1024}
    result = object()

    def diagnostic(*values: Any, **kwargs: Any) -> object:
        calls.append((values, kwargs))
        return result

    def endpoint(specification: Any) -> None:
        assert specification is PBE0
        assert sys.argv == ["schedule", "native", "--atoms", "48"]
        actual = qualify_cuda_schedule._stationary_cuda.complete_rks_cuda_gradient_diagnostic(
            "state", **payload
        )
        assert actual is result

    monkeypatch.setattr(
        qualify_cuda_schedule._stationary_cuda,
        "complete_rks_cuda_gradient_diagnostic",
        diagnostic,
    )
    monkeypatch.setattr(qualify_cuda_schedule, "main", endpoint)
    arguments = ["schedule", "native", "--atoms", "48"]
    if intrusive:
        arguments.insert(1, "--intrusive")
    monkeypatch.setattr(sys, "argv", arguments)
    qualify_cuda_schedule.run()
    assert calls == [(("state",), {**payload, "profile_device": intrusive})]
    assert (
        qualify_cuda_schedule._stationary_cuda.complete_rks_cuda_gradient_diagnostic
        is diagnostic
    )
    assert sys.argv is arguments


@pytest.mark.parametrize("fail", [False, True])
def test_schedule_journals_actual_force_work_even_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail: bool
) -> None:
    output = tmp_path / "endpoint.json"
    forces = np.zeros((1, 3))
    work = {"selected_tile": np.int64(1024), "array": np.asarray([1, 2])}
    calls = []

    def force(self: Any, *values: Any, **kwargs: Any) -> Any:
        calls.append((self, values, kwargs))
        return forces, work

    def endpoint(specification: Any) -> None:
        actual_forces, actual_work = (
            qualify_cuda_schedule.PreparedBatch._public_dft_cuda_force(
                "owner", "state", preserve_policy=True
            )
        )
        assert actual_forces is forces
        assert actual_work is work
        output.write_text(json.dumps({"status": "failed" if fail else "measured"}))
        if fail:
            raise RuntimeError("endpoint failed")

    monkeypatch.setattr(
        qualify_cuda_schedule.PreparedBatch, "_public_dft_cuda_force", force
    )
    monkeypatch.setattr(qualify_cuda_schedule, "main", endpoint)
    monkeypatch.setattr(sys, "argv", ["schedule", "native", "--output", str(output)])
    if fail:
        with pytest.raises(RuntimeError, match="endpoint failed"):
            qualify_cuda_schedule.run()
    else:
        qualify_cuda_schedule.run()
    record = json.loads(output.read_text())
    assert record["cuda_schedule_force_work"] == [
        {"selected_tile": 1024, "array": [1, 2]}
    ]
    assert record["cuda_schedule_intrusive"] is False
    assert calls == [("owner", ("state",), {"preserve_policy": True})]
    assert qualify_cuda_schedule.PreparedBatch._public_dft_cuda_force is force
