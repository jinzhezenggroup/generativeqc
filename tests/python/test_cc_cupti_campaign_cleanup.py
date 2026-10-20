"""Primary failures survive bounded cleanup and evidence publication faults."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest

from benchmarks import cc_cupti_campaign as campaign


@pytest.mark.parametrize(
    "stage", ("kill", "reap", "join", "log_close", "inputs", "publication")
)
def test_primary_failure_survives_every_cleanup_and_reap_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    c = campaign
    calls = []
    primary = ValueError("controlled primary attestation failure")
    secondary = PermissionError("controlled " + stage)
    spec = {"sha256": {}, "environment": {}, "driver_libraries": {}}
    for name in c.INPUTS:
        p = tmp_path / name
        p.write_text("{}" if name == "oracle" else name)
        spec[name] = str(p)
        spec["sha256"][name] = c.digest(p)
    allocation = tmp_path / "allocation.json"
    allocation.write_text("{}")
    spec.update(
        allocation_receipt=str(allocation),
        allocation_receipt_sha256=c.canonical_hash({}),
    )
    spec["mapped_libraries"] = {
        "libgenerativeqc.so": {
            "path": spec["library"],
            "sha256": spec["sha256"]["library"],
        }
    }

    class Process:
        pid = 12345

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            calls.append("kill")
            if stage == "kill":
                raise secondary

        def wait(self, timeout: float | None = None) -> int:
            assert timeout == 10
            calls.append("reap")
            if stage == "reap":
                raise secondary
            return -9

    class Supervisor:
        failure = None

        def __init__(self, *a: object) -> None:
            self.cleanup_errors = []

        def start(self) -> None:
            pass

        def join(self, timeout: float) -> bool:
            assert timeout == 10
            calls.append("join")
            if stage == "join":
                raise secondary
            return True

    class Clock:
        def __init__(self, *a: object) -> None:
            pass

        def sample(self) -> dict:
            return {"clock": "controlled", "cupti_ns": 1}

    monkeypatch.setattr(
        c.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout="libgenerativeqc.so => " + spec["library"] + " (0x1)"
        ),
    )
    monkeypatch.setattr(c.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(c, "NativeSupervisor", Supervisor)
    monkeypatch.setattr(c, "CallerClock", Clock)
    monkeypatch.setattr(c, "process_identity", lambda p: {"pid": p, "starttime": "1"})

    primary_observed = False

    def observe(*args: object) -> None:
        nonlocal primary_observed
        primary_observed = True
        raise primary

    monkeypatch.setattr(c, "observe_native", observe)
    real_frozen = c.frozen_inputs

    def frozen(s: dict) -> dict:
        if stage == "inputs" and primary_observed:
            raise secondary
        return real_frozen(s)

    monkeypatch.setattr(c, "frozen_inputs", frozen)
    real_atomic = c.atomic_json
    writes = []

    def atomic(path: Path, value: dict) -> None:
        if path.name == "receipt.json" and value.get("failure_type") == "ValueError":
            writes.append("terminal failure evidence")
            if stage == "publication":
                raise secondary
        return real_atomic(path, value)

    monkeypatch.setattr(c, "atomic_json", atomic)
    real_open = Path.open

    class Log:
        def __init__(self, stream: object) -> None:
            self.stream = stream

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *a: object) -> None:
            self.close()

        def close(self) -> None:
            calls.append("log_close")
            self.stream.close()
            if stage == "log_close":
                raise secondary

    def open_path(path: Path, *a: object, **k: object) -> object:
        s = real_open(path, *a, **k)
        return Log(s) if path.name == "native.log" else s

    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(ValueError) as raised:
        c.run(spec, tmp_path / "run", diagnostic=False)
    assert raised.value is primary
    assert all(action in calls for action in ("kill", "reap", "join", "log_close"))
    assert writes, "terminal failure evidence was never attempted"
    receipt = json.loads((tmp_path / "run/receipt.json").read_text())
    if stage == "publication":
        assert receipt["science"] == "INCOMPLETE" and receipt["capture"] == "INCOMPLETE"
        assert "publication failed" in raised.value.__notes__[0]
    else:
        assert receipt["failure_type"] == "ValueError"
        assert any(
            row["type"] == "PermissionError" for row in receipt["cleanup_errors"]
        )
        assert (
            receipt["science"].startswith("INCOMPLETE")
            and receipt["capture"] == "INCOMPLETE"
        )
