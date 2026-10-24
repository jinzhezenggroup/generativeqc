"""Bounded host checks for geometry integrity and observable derivative behavior."""

import copy
import importlib.util
import json
import lzma
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "force_replay", Path(__file__).with_name("replay.py")
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class OptimizedModeTests(unittest.TestCase):
    def test_optimized_entry_import_and_inherited_environment_refuse_before_io(
        self,
    ) -> None:
        family = Path(__file__).resolve().parent
        repo = family.parents[2]
        harness = """
import importlib.util, os, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('guard_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if sys.argv[2] != 'import':
    os.environ['PYTHONOPTIMIZE'] = sys.argv[2]
    def forbidden(*args, **kwargs):
        raise RuntimeError('Evidence I/O reached before optimization refusal')
    Path.read_bytes = forbidden
    import subprocess
    subprocess.run = subprocess.check_output = forbidden
    sys.argv = [sys.argv[1], *sys.argv[3:]]
    module.main()
"""
        for name in ("replay.py", "verify.py", "publish.py"):
            for mode, flag, optimize in (
                ("direct", "-O", None),
                ("direct", "-OO", None),
                ("direct", None, "1"),
                ("direct", None, "2"),
                ("import", "-O", None),
                ("import", "-OO", None),
                ("import", None, "1"),
                ("import", None, "2"),
                ("late-env", None, "1"),
                ("late-env", None, "2"),
            ):
                with (
                    self.subTest(name=name, mode=mode, flag=flag, optimize=optimize),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    script = root / name
                    script.write_bytes((family / name).read_bytes())
                    for retained in (
                        "publication.json",
                        "comparison.json.xz",
                        "evidence.json.xz",
                        "inputs.json.xz",
                    ):
                        (root / retained).write_bytes(b"retained original sentinel\n")
                    before = {p.name: p.read_bytes() for p in root.iterdir()}
                    args = (
                        ["--working-tree"]
                        if name == "verify.py"
                        else ["--raw", str(root / "raw")]
                    )
                    if name == "replay.py":
                        args += ["--output", str(root / "comparison.json.xz")]
                    env = os.environ.copy()
                    env.pop("PYTHONOPTIMIZE", None)
                    env["PYTHONDONTWRITEBYTECODE"] = "1"
                    env["PYTHONPATH"] = os.pathsep.join(
                        (str(repo), str(repo / "python"))
                    )
                    if optimize and mode != "late-env":
                        env["PYTHONOPTIMIZE"] = optimize
                    command = [sys.executable, "-B"] + ([flag] if flag else [])
                    if mode == "direct":
                        command += [str(script), *args]
                    else:
                        command += [
                            "-c",
                            harness,
                            str(script),
                            optimize if mode == "late-env" else "import",
                            *args,
                        ]
                    process = subprocess.run(
                        command,
                        check=False,
                        cwd=root,
                        env=env,
                        capture_output=True,
                        timeout=30,
                    )
                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn(b"requires unoptimized Python", process.stderr)
                    self.assertNotIn(b"PASS", process.stdout)
                    self.assertEqual(
                        {p.name: p.read_bytes() for p in root.iterdir()}, before
                    )


class ReplayTests(unittest.TestCase):
    def test_terminal_metadata_rejects_source_or_execution_drift(self) -> None:
        # Synthetic execution metadata only; this fixture contains no oracle energies.
        initial = replay.execution_records()
        freeze = json.loads(initial["source-freeze-final.json"])
        environment = json.loads(initial["environment.json"])
        inputs = json.loads(
            lzma.decompress(Path(__file__).with_name("inputs.json.xz").read_bytes())
        )
        report = {
            "status": "PASS",
            "source_revision": freeze["revision"],
            "host": environment["host"],
            "dataset_sha256": inputs["dataset_sha256"],
            "science_workers": [],
            "mode_errors": [],
            "cgroup": {k: environment["cgroup"][k] for k in ("cpu.max", "memory.max")},
            "task_outcomes": {k: {"exit_code": 0} for k in inputs["tasks"]},
            "controller_outcomes": {
                k: {"exit_code": 0}
                for k in ("suite.outcome.json", "launch/outcome.json")
            },
        }
        for field, original in (
            ("source", freeze["files"]),
            ("libraries", environment["libraries"]),
        ):
            report[field] = {
                k: {
                    "expected_sha256": v["sha256"],
                    "observed_sha256": v["sha256"],
                    "bytes": v["bytes"],
                }
                for k, v in original.items()
            }

        def encoded(value: dict) -> dict:
            return {
                "terminal-postflight.json": json.dumps(value).encode(),
                **{
                    k: json.dumps(v).encode()
                    for k, v in value["controller_outcomes"].items()
                },
            }

        self.assertEqual(
            replay.check_terminal(initial, encoded(report), inputs)["status"], "PASS"
        )
        for mutation in ("library", "size", "task", "worker", "mode", "exit"):
            bad = copy.deepcopy(report)
            if mutation == "library":
                bad["libraries"][next(iter(bad["libraries"]))]["observed_sha256"] = (
                    "0" * 64
                )
            elif mutation == "size":
                bad["source"]["oracle.py"]["bytes"] += 1
            elif mutation == "task":
                del bad["task_outcomes"]["C7-h1-plus"]
            elif mutation == "worker":
                bad["science_workers"] = [{"pid": 1}]
            elif mutation == "mode":
                bad["mode_errors"] = [{"path": "fixture"}]
            else:
                bad["controller_outcomes"]["launch/outcome.json"]["exit_code"] = 124
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                replay.check_terminal(initial, encoded(bad), inputs)

    def test_assembled_force_terms_match_constituents(self) -> None:
        tools = replay.central_tools()
        bundle, _ = tools.native_bundle()
        native = copy.deepcopy(
            next(p["native"] for p in bundle["points"] if p["carbon"] == 4)
        )
        native["samples"][0]["hf_forces"][0][0] += 1e-4
        with self.assertRaises(AssertionError):
            replay.vectors(native, native["protocol"], tools)

    def test_timeout_retained_and_strict_replay_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "C7-h1-plus"
            task.mkdir()
            outcome = task / "outcome.json"
            raw = b'{"exit_code":124,"time_limit_seconds":21600}\n'
            outcome.write_bytes(raw)
            command = [
                sys.executable,
                str(Path(__file__).with_name("replay.py")),
                "--raw",
                str(root),
                "--output",
                str(root / "comparison.json"),
                "--require-pass",
            ]
            process = subprocess.run(command, capture_output=True, check=False)
            self.assertEqual(process.returncode, 1, process.stderr)
            result = json.loads((root / "comparison.json").read_bytes())
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(
                [r["status"] for r in result["molecules"]], ["INCOMPLETE", "FAIL"]
            )
            self.assertEqual(
                result["molecules"][1]["tasks"]["C7-h1-plus"]["outcome"]["exit_code"],
                124,
            )
            self.assertEqual(outcome.read_bytes(), raw)

    def test_exact_inputs(self) -> None:
        inputs = json.loads(
            lzma.decompress(Path(__file__).with_name("inputs.json.xz").read_bytes())
        )
        replay.validate_inputs(inputs)
        for field in ("direction", "atoms_bohr"):
            altered = copy.deepcopy(inputs)
            task = altered["tasks"]["C4-h0-minus"]
            if field == "direction":
                task[field][0][0] += 1e-10
            else:
                task[field][0][1][0] += 1e-10
            with self.assertRaises(AssertionError):
                replay.validate_inputs(altered)

    def test_missing_task_rejected(self) -> None:
        inputs = json.loads(
            lzma.decompress(Path(__file__).with_name("inputs.json.xz").read_bytes())
        )
        del inputs["tasks"]["C7-h1-plus"]
        with self.assertRaises((AssertionError, KeyError)):
            replay.validate_inputs(inputs)

    def test_force_sign_from_independent_quadratic(self) -> None:
        center = np.array([[0.5, -0.2, 0.8], [-0.3, 0.4, 0.1]])
        direction = np.random.default_rng(709).normal(size=center.shape)
        direction /= np.linalg.norm(direction)
        # E=sum(x**2), hence analytic F=-2x. Both prescribed steps must agree.
        for step in (1e-4, 3e-5):
            minus = {"composite": float(np.sum((center - step * direction) ** 2))}
            plus = {"composite": float(np.sum((center + step * direction) ** 2))}
            error = replay.projection_errors(
                {"composite": -2 * center}, direction, minus, plus, step
            )
            self.assertLess(error["composite"]["error_eh_per_bohr"], 1e-10)
            wrong = replay.projection_errors(
                {"composite": 2 * center}, direction, minus, plus, step
            )
            self.assertGreater(
                wrong["composite"]["error_eh_per_bohr"], replay.FORCE_GATE
            )

    def test_nonfinite_refused(self) -> None:
        with self.assertRaises(AssertionError):
            replay.projection_errors(
                {"a": np.zeros((1, 3))},
                [[1, 0, 0]],
                {"a": 0},
                {"a": float("nan")},
                1e-4,
            )


if __name__ == "__main__":
    unittest.main()
