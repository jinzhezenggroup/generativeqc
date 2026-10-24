"""Real Linux-only descendant lifecycle tests; never import scientific modules."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from process_guard import process_identity
from transport import bounded, read


@unittest.skipUnless(
    sys.platform.startswith("linux"),
    "Linux proc/pidfd/subreaper cannot be established on Windows",
)
class OwnedDescendantLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.jit = self.root / "jit"
        self.jit.mkdir()
        self.pidfile = self.root / "owned-child.json"

    def command(
        self, *, sleep_parent: bool = False, ignore_child_term: bool = False
    ) -> list[str]:
        child = "import time;time.sleep(20)"
        if ignore_child_term:
            child = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)"
        code = (
            "import json,pathlib,subprocess,sys,time;"
            f'child=subprocess.Popen([sys.executable,"-c",{child!r}]);'
            'fields=pathlib.Path("/proc/"+str(child.pid)+"/stat").read_text().rpartition(") ")[2].split();'
            f'pathlib.Path({str(self.pidfile)!r}).write_text(json.dumps({{"pid":child.pid,"start_ticks":int(fields[19])}}));'
            + ("time.sleep(20)" if sleep_parent else "sys.exit(0)")
        )
        return [sys.executable, "-c", code]

    def assert_original_child_gone(self) -> None:
        child = read(self.pidfile)
        current = process_identity(child["pid"])
        self.assertTrue(
            current is None or current["start_ticks"] != child["start_ticks"]
        )

    def test_normal_leader_exit_with_live_child_is_control_error_and_reaped(
        self,
    ) -> None:
        result = bounded(
            self.command(),
            self.root / "process",
            self.root,
            self.jit,
            10,
            1 << 20,
            1 << 20,
            grace_seconds=1,
            kill_reap_seconds=1,
        )
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["status"], "CONTROL_ERROR")
        self.assertEqual(result["trigger"], "DESCENDANT_LEFT_AFTER_LEADER")
        self.assertTrue(result["cleanup_success"])
        self.assertEqual(result["remaining_owned"], [])
        self.assert_original_child_gone()
        self.assertLess(result["elapsed_seconds"], 5.0)

    def test_timeout_child_is_reaped_and_external_sentinel_is_untouched(self) -> None:
        sentinel = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(20)"], start_new_session=True
        )
        try:
            result = bounded(
                self.command(sleep_parent=True, ignore_child_term=True),
                self.root / "process",
                self.root,
                self.jit,
                1,
                1 << 20,
                1 << 20,
                grace_seconds=1,
                kill_reap_seconds=1,
            )
            self.assertEqual(result["status"], "TIMEOUT")
            self.assertEqual(result["trigger"], "TIMEOUT")
            self.assertTrue(result["cleanup_success"])
            self.assert_original_child_gone()
            self.assertIsNone(sentinel.poll())
            record = read(self.root / "process/control/guard-outcome.json")
            self.assertNotIn(sentinel.pid, [r["pid"] for r in record["signal_records"]])
        finally:
            if sentinel.poll() is None:
                sentinel.terminate()
                try:
                    sentinel.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    sentinel.kill()
                    sentinel.wait(timeout=2)

    def test_guard_failure_before_ack_never_executes_command(self) -> None:
        isolated = self.root / "isolated"
        isolated.mkdir()
        shutil.copyfile(
            Path(__file__).with_name("transport.py"), isolated / "transport.py"
        )
        (isolated / "process_guard.py").write_text("raise SystemExit(17)\n")
        spec = importlib.util.spec_from_file_location(
            "isolated_guard_failure_transport", isolated / "transport.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        marker = self.root / "COMMAND_MUST_NOT_START"
        command = [
            sys.executable,
            "-c",
            f'from pathlib import Path;Path({str(marker)!r}).write_text("started")',
        ]
        with self.assertRaisesRegex(RuntimeError, "guard.*acknowledgement"):
            module.bounded(
                command,
                self.root / "process",
                self.root,
                self.jit,
                2,
                1 << 20,
                1 << 20,
                grace_seconds=1,
                kill_reap_seconds=1,
            )
        self.assertFalse(marker.exists())
        outcome = read(self.root / "process/process-outcome.json")
        self.assertEqual(outcome["status"], "CONTROL_ERROR")
        self.assertEqual(outcome["guard_returncode"], 17)


if __name__ == "__main__":
    unittest.main()
