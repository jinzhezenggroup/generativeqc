"""Stdlib-only replay fault injection. No scientific module or native binary loads.

Adapted only for formatting and annotations from the reviewed 34-case suite
(SHA256 e525a50eece29ff8e6f94c3f8d1ae15f2ea591afec5054f033a2045d8dad1e06).
The normal test wrapper decodes the retained reviewed replay and invokes this
helper in a temporary directory under isolated, site-free Python.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from typing import Any
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT = Path(os.environ.get("REPLAY_INPUT", HERE / "public-n-pentane.json"))
SCRIPT = Path(os.environ.get("REPLAY_SCRIPT", HERE / "fresh_endpoint.py"))
if "MOCK_CPU" in os.environ:
    assert os.sched_getaffinity(0) == {int(os.environ["MOCK_CPU"])}, (
        "Mock CPU allocation mismatch"
    )
assert not any(x in sys.modules for x in ("numpy", "pyscf", "generativeqc"))


class ReplayCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(dir=HERE, prefix="mock-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.lib = self.source / "build/libmock.so"
        self.lib.parent.mkdir()
        self.lib.write_bytes(b"mock library; never load")
        (self.source / "python/generativeqc").mkdir(parents=True)
        (self.source / "python/generativeqc/__init__.py").write_text("# mock package")
        self.output = self.root / "out"
        self.spec = importlib.util.spec_from_file_location("subject", SCRIPT)
        self.mod = importlib.util.module_from_spec(self.spec)
        self.spec.loader.exec_module(self.mod)
        self.case = json.loads(INPUT.read_text())
        self.events = []
        self.calls = []
        self.existing = []
        self.live = weakref.WeakSet()
        self.created = 0
        self.item_mutation = {}
        self.fail_at = None
        self.library_source = "a" * 64
        self.elapsed = 0
        self.high_rss = 1024
        self.supervisor_events = []
        self.waits = None
        self.poll_rss = 1024
        self.bad_calc_library = False
        self.emit_real = self.mod.emit

    def run_main(
        self, child: bool = True, dangling: bool = False
    ) -> BaseException | None:
        m = self.mod
        cmake = (
            "\n".join(
                (
                    f"CMAKE_HOME_DIRECTORY:INTERNAL={self.source}",
                    "CMAKE_BUILD_TYPE:STRING=Release",
                    "GENERATIVEQC_ENABLE_CUDA:BOOL=OFF",
                    "GENERATIVEQC_CPU_LINALG_PROVIDER:STRING=openblas",
                    "GENERATIVEQC_ENABLE_CXX_PCH:BOOL=OFF",
                    "CMAKE_CXX_COMPILER_LAUNCHER:UNINITIALIZED=ccache",
                    "CMAKE_CUDA_COMPILER_LAUNCHER:UNINITIALIZED=ccache",
                )
            )
            + "\n"
        )
        (self.lib.parent / "CMakeCache.txt").write_text(cmake)
        (self.lib.parent / "generated").mkdir(exist_ok=True)
        (self.lib.parent / "generated/build_identity.hpp").write_text(
            'kGenerativeQCSourceIdentity = "' + "a" * 64 + '";'
        )
        (self.lib.parent / "generated/build_identity_inputs.txt").write_text(
            "mock-inputs"
        )
        (self.lib.parent / "compile_commands.json").write_text("[]")
        args = [
            str(SCRIPT),
            "--reviewed",
            "--source",
            str(self.source),
            "--source-sha",
            "mock-sha",
            "--library",
            str(self.lib),
            "--library-sha",
            hashlib.sha256(self.lib.read_bytes()).hexdigest(),
            "--input",
            str(INPUT),
            "--output",
            str(self.output),
            "--arm",
            "scalar",
            "--cpu",
            "6",
            "--monitor-cpu",
            "7",
        ]
        if child:
            self.output.mkdir()
            args += ["--child"]
        if dangling:
            self.output.symlink_to(self.root / "new-target", target_is_directory=True)
        case = self.case
        owner = self

        class Atom:
            from_value = staticmethod(lambda a: a)

        class Batch:
            def __init__(self, calc: Any) -> None:
                self._calculator = calc

            def execute(self, **kw: Any) -> Any:
                owner.calls.append(("execute", kw))
                if owner.fail_at == "execute":
                    raise RuntimeError("injected execute fault")
                d = dict(  # noqa: C408 - Keep reviewed mock construction unchanged.
                    status="success",
                    status_message="",
                    succeeded=True,
                    converged=True,
                    energy=m.ORACLE[0 if owner.created < 4 else 1],
                    energy_change=1e-13,
                    density_rms=1e-12,
                    iterations=12,
                    fock_builds=14,
                    precision={"requested_mode": "fp64", "effective_bits": 64},
                    restart_origin="none",
                    warm_start_used=False,
                    warm_start_fallback=False,
                    physical_residual_rms=None,
                    initial_guess=None,
                    incremental_direct_jk={"requested": False, "active": False},
                    forces=None,
                    executed_backend="cpu_reference",
                )
                d.update(owner.item_mutation)
                return types.SimpleNamespace(items=[types.SimpleNamespace(**d)])

            def close(self) -> None:
                owner.calls.append(("close", {}))
                if owner.fail_at == "close":
                    raise RuntimeError("injected close fault")

        class Calculator:
            def __init__(self, **kw: Any) -> None:
                owner.existing.append(len(owner.live))
                owner.live.add(self)
                owner.created += 1
                self._library = (
                    lib
                    if not owner.bad_calc_library
                    else types.SimpleNamespace(
                        _name="wrong", generativeqc_get_source_identity=lambda: b"wrong"
                    )
                )
                owner.calls.append(("construct", kw))
                if owner.fail_at == "construct":
                    raise RuntimeError("injected construct fault")

            def prepare_batch(self, xyz: Any, **kw: Any) -> Any:
                owner.calls.append(("prepare", kw))
                if owner.fail_at == "prepare":
                    raise RuntimeError("injected prepare fault")
                return Batch(self)

            def _shells_for_atoms(self, atoms: Any) -> Any:
                if owner.fail_at == "shells":
                    raise RuntimeError("injected post-return shell fault")
                return [
                    types.SimpleNamespace(
                        atom_index=i,
                        angular_momentum=s[0],
                        primitives=[
                            types.SimpleNamespace(exponent=p[0], coefficient=p[1])
                            for p in s[1:]
                        ],
                    )
                    for i, label in enumerate(case["pyscf_atom_labels"])
                    for s in case["exact_primitive_input"][label]
                ]

        lib = types.SimpleNamespace(
            _name=str(self.lib),
            generativeqc_get_source_identity=lambda: self.library_source.encode(),
        )
        fake = types.ModuleType("generativeqc")
        fake.__file__ = str(self.source / "python/generativeqc/__init__.py")
        fake.Atom = Atom
        fake.Calculator = Calculator
        fake._native = types.SimpleNamespace(load_library=lambda **kw: lib)
        real_read = Path.read_text

        def read_text(p: Path, *a: Any, **kw: Any) -> str:
            if str(p) == "/proc/meminfo":
                return f"MemAvailable: {10 * (1 << 30) // 1024} kB\n"
            if str(p) == "/proc/self/cgroup":
                return "1:mock:/\n"
            if str(p).startswith("/proc/424242/"):
                return f"VmRSS: {self.poll_rss} kB\n"
            return real_read(p, *a, **kw)

        def git(argv: list[str], **kw: Any) -> str:
            self.assertEqual(argv[:2], ["git", "-C"])
            if argv[-1] == "HEAD":
                return "mock-sha\n"
            if argv[-1] == "HEAD^{tree}":
                return "mock-tree\n"
            if argv[-1] == "--porcelain":
                return ""
            raise AssertionError(argv)

        def emit(p: Path, obj: dict) -> None:
            self.events.append(copy.deepcopy(obj))
            self.emit_real(p, obj)

        usage = types.SimpleNamespace(
            ru_maxrss=self.high_rss,
            ru_utime=0.1,
            ru_stime=0.1,
            ru_minflt=1,
            ru_majflt=0,
            ru_nvcsw=0,
            ru_nivcsw=0,
        )
        proc = types.SimpleNamespace(pid=424242, returncode=None)
        ticks = iter((0.0, self.elapsed, self.elapsed, self.elapsed, self.elapsed))

        def popen(*a: Any, **kw: Any) -> Any:
            for event in self.supervisor_events:
                self.emit_real(self.output / "events.jsonl", event)
            return proc

        waits = iter(self.waits or [(proc.pid, 0, usage)])

        def wait4(*a: Any) -> Any:
            value = next(waits)
            if isinstance(value, BaseException):
                raise value
            return (value[0], value[1], value[2] or usage)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", args))
            stack.enter_context(
                patch.object(sys, "pycache_prefix", str(self.root / "absent-cache"))
            )
            stack.enter_context(patch.object(sys, "dont_write_bytecode", True))
            stack.enter_context(patch.dict(os.environ, {k: "1" for k in m.THREADS}))
            stack.enter_context(patch.dict(sys.modules, {"generativeqc": fake}))
            stack.enter_context(patch.object(m.subprocess, "check_output", git))
            stack.enter_context(patch.object(m.subprocess, "Popen", side_effect=popen))
            stack.enter_context(patch.object(m.os, "sched_setaffinity"))
            stack.enter_context(patch.object(m.resource, "setrlimit"))
            stack.enter_context(
                patch.object(m.resource, "getrusage", return_value=usage)
            )
            stack.enter_context(patch.object(m.os, "wait4", side_effect=wait4))
            stack.enter_context(patch.object(m.os, "killpg"))
            stack.enter_context(patch.object(m.signal, "signal"))
            stack.enter_context(
                patch.object(
                    m.time, "monotonic", side_effect=lambda: next(ticks, self.elapsed)
                )
            )
            stack.enter_context(patch.object(m, "emit", emit))
            stack.enter_context(patch.object(Path, "read_text", read_text))
            if hasattr(m, "native_source_identity"):
                stack.enter_context(
                    patch.object(m, "native_source_identity", return_value="a" * 64)
                )
            err = None
            try:
                m.main()
            except BaseException as exc:  # noqa: BLE001 - Includes injected KeyboardInterrupt.
                err = exc
        return err

    def test_valid_child(self) -> None:
        self.assertIsNone(self.run_main())
        self.assertEqual(self.created, 4)
        self.assertEqual(
            [e["phase"] for e in self.events if e["state"] == "returned"],
            ["cold", "warm", "warm", "changed_geometry"],
        )
        self.assertEqual(sum((k == "close" for k, _ in self.calls)), 4)
        self.assertEqual(self.events[-1]["state"], "completed")

    def make_supervisor_receipt(self) -> None:
        self.assertIsNone(self.run_main())
        self.supervisor_events = copy.deepcopy(self.events)
        self.output.rename(self.root / "prior-child")
        self.events = []

    def test_valid_supervisor(self) -> None:
        self.make_supervisor_receipt()
        self.assertIsNone(self.run_main(child=False))

    def test_missing_child_receipt_rejected(self) -> None:
        self.assertIsNotNone(self.run_main(child=False))

    def test_incomplete_child_receipt_rejected(self) -> None:
        self.make_supervisor_receipt()
        self.supervisor_events = [
            e for e in self.supervisor_events if e.get("index") != 3
        ]
        self.assertIsNotNone(self.run_main(child=False))

    def test_bad_repeated_child_receipt_rejected(self) -> None:
        self.make_supervisor_receipt()
        for e in self.supervisor_events:
            if e["state"] == "returned" and e["index"] == 1:
                e["energy_error"] = 1.0
        self.assertIsNotNone(self.run_main(child=False))

    def test_live_polled_rss_guard(self) -> None:
        self.poll_rss = 7 * (1 << 30) // 1024
        self.waits = [(0, 0, None), (424242, 9, None)]
        self.assertIsNotNone(self.run_main(child=False))
        self.assertEqual(
            next(e for e in self.events if e["state"] == "supervisor-complete")[
                "kill_reason"
            ],
            "rss-limit",
        )

    def test_live_timeout_guard(self) -> None:
        self.elapsed = 901
        self.waits = [(0, 0, None), (424242, 9, None)]
        self.assertIsNotNone(self.run_main(child=False))
        self.assertEqual(
            next(e for e in self.events if e["state"] == "supervisor-complete")[
                "kill_reason"
            ],
            "timeout",
        )

    def test_interrupt_retained(self) -> None:
        self.waits = [KeyboardInterrupt("injected"), (424242, 9, None)]
        self.assertIsNotNone(self.run_main(child=False))
        self.assertTrue(
            any(e["state"] == "supervisor-interrupted" for e in self.events)
        )

    def test_calculator_library_owner(self) -> None:
        self.bad_calc_library = True
        self.assertIsNotNone(self.run_main())

    def test_original_input_identity(self) -> None:
        self.assertEqual(
            hashlib.sha256(INPUT.read_bytes()).hexdigest(), self.mod.INPUT_SHA
        )
        self.assertEqual(self.case["dimensions"]["electron_count"], 42)
        self.assertEqual(self.case["dimensions"]["spherical_ao_count"], 130)

    def test_gate_iteration_upper(self) -> None:
        self.item_mutation = {"iterations": 151}
        self.assertIsNotNone(self.run_main())

    def test_gate_density_nonnegative(self) -> None:
        self.item_mutation = {"density_rms": -1.0}
        self.assertIsNotNone(self.run_main())

    def test_gate_density_boolean(self) -> None:
        self.item_mutation = {"density_rms": False}
        self.assertIsNotNone(self.run_main())

    def test_gate_requested_precision(self) -> None:
        self.item_mutation = {
            "precision": {"requested_mode": "auto", "effective_bits": 64}
        }
        self.assertIsNotNone(self.run_main())

    def test_gate_executed_backend(self) -> None:
        self.item_mutation = {"executed_backend": "cuda"}
        self.assertIsNotNone(self.run_main())

    def test_gate_incremental_route(self) -> None:
        self.item_mutation = {
            "incremental_direct_jk": {"requested": True, "active": True}
        }
        self.assertIsNotNone(self.run_main())

    def test_gate_final_true_rss(self) -> None:
        self.high_rss = int(7 * (1 << 30) / 1024)
        self.assertIsNotNone(self.run_main(child=False))
        self.assertEqual(
            next(e for e in self.events if e["state"] == "supervisor-complete")[
                "kill_reason"
            ],
            "final-rss-limit",
        )

    def test_gate_final_wall_time(self) -> None:
        self.elapsed = 901
        self.assertIsNotNone(self.run_main(child=False))
        self.assertEqual(
            next(e for e in self.events if e["state"] == "supervisor-complete")[
                "kill_reason"
            ],
            "final-timeout",
        )

    def test_reject_dangling_output(self) -> None:
        self.assertIsNotNone(self.run_main(child=False, dangling=True))
        self.assertFalse((self.root / "new-target").exists())

    def test_dispose_old_python_owner(self) -> None:
        self.assertIsNone(self.run_main())
        self.assertEqual(self.existing, [0, 0, 0, 0])

    def test_record_construction_failure(self) -> None:
        self.fail_at = "construct"
        self.assertIsNotNone(self.run_main())
        self.assertTrue(
            any(e["state"] == "failed" and e.get("index") == 0 for e in self.events)
        )

    def test_record_prepare_failure(self) -> None:
        self.fail_at = "prepare"
        self.assertIsNotNone(self.run_main())
        self.assertTrue(
            any(e["state"] == "failed" and e.get("index") == 0 for e in self.events)
        )

    def test_record_execute_failure(self) -> None:
        self.fail_at = "execute"
        self.assertIsNotNone(self.run_main())
        self.assertEqual(sum((k == "close" for k, _ in self.calls)), 1)
        self.assertTrue(
            any(e["state"] == "failed" and e.get("index") == 0 for e in self.events)
        )

    def test_record_close_failure(self) -> None:
        self.fail_at = "close"
        self.assertIsNotNone(self.run_main())
        self.assertTrue(
            any(e["state"] == "failed" and e.get("index") == 0 for e in self.events)
        )

    def test_record_post_return_failure(self) -> None:
        self.fail_at = "shells"
        self.assertIsNotNone(self.run_main())
        self.assertTrue(
            any(e["state"] == "failed" and e.get("index") == 0 for e in self.events)
        )

    def test_gate_loaded_source_identity(self) -> None:
        self.library_source = "unrelated-source"
        self.assertIsNotNone(self.run_main())

    def test_explicit_controls(self) -> None:
        self.assertIsNone(self.run_main())
        kw = next((kw for k, kw in self.calls if k == "construct"))
        self.assertEqual(kw.get("precision"), "fp64")
        self.assertEqual(kw.get("density_fitting"), "none")

    def test_nonfinite_energy_retained(self) -> None:
        self.item_mutation = {"energy": float("nan")}
        self.assertIsNotNone(self.run_main())
        self.assertTrue(
            any(
                e["state"] == "returned" and isinstance(e["energy"], dict)
                for e in self.events
            )
        )

    def test_failed_energy_none_retained(self) -> None:
        self.item_mutation = {
            "energy": None,
            "succeeded": False,
            "converged": False,
            "status_message": "native failure",
        }
        self.assertIsNotNone(self.run_main())
        self.assertTrue(
            any(
                e["state"] == "returned" and e.get("status_message") == "native failure"
                for e in self.events
            )
        )

    def test_bad_energy_gate(self) -> None:
        self.item_mutation = {"energy": 0.0}
        self.assertIsNotNone(self.run_main())

    def test_large_finite_energy_change_not_extra_gate(self) -> None:
        self.item_mutation = {"energy_change": 100.0}
        self.assertIsNone(self.run_main())

    def test_zero_fock_rejected(self) -> None:
        self.item_mutation = {"fock_builds": 0}
        self.assertIsNotNone(self.run_main())

    def test_nonconvergence_rejected(self) -> None:
        self.item_mutation = {"converged": False}
        self.assertIsNotNone(self.run_main())

    def test_nan_density_rejected(self) -> None:
        self.item_mutation = {"density_rms": float("nan")}
        self.assertIsNotNone(self.run_main())


if __name__ == "__main__":
    unittest.main(verbosity=2)
