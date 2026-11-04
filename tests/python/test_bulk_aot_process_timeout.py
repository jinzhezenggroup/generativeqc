"""Offline AOT probes must terminate their compiler children on timeout."""

from __future__ import annotations

import os
import shlex
import shutil
import time
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common import compiler_process
from generativeqc_compiler.xc import bulk_aot

if TYPE_CHECKING:
    from pathlib import Path
    from subprocess import Popen
    from typing import Any


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX compiler process groups")
@pytest.mark.parametrize("phase", ("identification", "compilation"))
@pytest.mark.parametrize("startup_delay", (0.0, 0.5))
def test_aot_timeout_stops_real_compiler_children(
    tmp_path: Path, phase: str, startup_delay: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("requires a POSIX shell")
    started, escaped = tmp_path / "started", tmp_path / "escaped"
    release = tmp_path / "release"
    compiler = tmp_path / "compiler"
    compiler.write_text(
        f"#!{shell}\n"
        f'if [ "$1" = "--version" ] && [ "{phase}" = "compilation" ]; then\n'
        '  echo "fake cc"; exit 0\n'
        "fi\n"
        f"(sleep {startup_delay}; : > {shlex.quote(str(started))}; "
        f"while [ ! -f {shlex.quote(str(release))} ]; do sleep 0.01; done; "
        "sleep 0.8; "
        f": > {shlex.quote(str(escaped))}) &\n"
        "exec sleep 30\n"
    )
    compiler.chmod(0o700)
    # This is a process-tree cancellation test, not a wall-clock budget test.
    # Synchronize fault injection before real communicate() starts its timeout;
    # a busy CI host may otherwise exhaust 0.3 s before the child is scheduled.
    # Forward the production timeout unchanged and exercise real group cleanup.
    start_process = compiler_process.start_owned_process

    def start_ready_process(command: list[str], **options: Any) -> Popen[str]:
        process = start_process(command, **options)
        if phase == "compilation" and command[-1] == "--version":
            return process
        communicate = process.communicate
        first = True

        def communicate_ready(*args: Any, **kwargs: Any) -> tuple[str, str]:
            nonlocal first
            if first:
                first = False
                deadline = time.monotonic() + 10.0
                while not started.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not started.exists():
                    compiler_process.kill_compiler_group(process)
                    communicate()
                    pytest.fail("fault fixture child did not become ready within 10 s")
                release.touch()
            return communicate(*args, **kwargs)

        monkeypatch.setattr(process, "communicate", communicate_ready)
        return process

    monkeypatch.setattr(compiler_process, "start_owned_process", start_ready_process)
    item = bulk_aot.SourceVariant(
        name="TIMEOUT_PROBE",
        family="lda",
        spin="unpolarized",
        import_identity="test-only-source",
        domain="process-test/v1",
        features=("rho",),
        derivative_order=0,
        backend="cpu",
        source="void bulk_xc_point(const double *x, double *y) { y[0] = x[0]; }\n",
        energy_nodes=1,
        ssa={"root_count": 1},
    )
    result = bulk_aot.compile_probe(item, compiler=str(compiler), timeout=0.3)
    # The child keeps inherited pipes and would write this marker if only its
    # parent were killed. It must have started, then be stopped as a group.
    time.sleep(1.0)
    assert started.is_file(), "fault fixture child never started"
    assert not escaped.exists(), "compiler child outlived the requested timeout"
    assert result["status"] == "timed-out"
