"""Host-only controls: no scheduler, native, GPU or network workloads."""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/md-j-sparse-support-20261016"
REPRODUCE = BUNDLE / "reproduce"


def module_from_source(path: Path) -> types.ModuleType:
    module = types.ModuleType(path.stem.replace("-", "_"))
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)  # noqa: S102 - trusted checked-in host helper
    return module


class ReproductionSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.safety = module_from_source(REPRODUCE / "output_safety.py")
        self.temporary = tempfile.TemporaryDirectory(prefix="md-output-safety-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        # Only this test's newly created temporary directory is removed.
        self.temporary.cleanup()

    def main_namespace(self) -> dict[str, Any]:
        tree = ast.parse((BUNDLE / "summarize.py").read_text())
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "REPRODUCERS" for t in node.targets
            )
        )
        names = ast.literal_eval(assignment.value)
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        namespace = {
            "Path": Path,
            "Any": object,
            "argparse": argparse,
            "ExitStack": ExitStack,
            "ROOT": BUNDLE,
            "REPRODUCERS": names,
            "SAFETY": self.safety.__dict__,
            "summarize": lambda: {"endpoint_cases": {}},
        }
        exec(  # noqa: S102 - isolate the checked-in CLI from data imports
            compile(ast.Module(body=[main], type_ignores=[]), "<main>", "exec"),
            namespace,
        )
        return namespace

    def test_existing_objects_and_aliases_are_untouched(self) -> None:
        file = self.root / "file"
        file.write_bytes(b"retained")
        directory = self.root / "directory"
        directory.mkdir()
        link = self.root / "link"
        link.symlink_to(file)
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "absent")
        directory_link = self.root / "directory-link"
        directory_link.symlink_to(directory, target_is_directory=True)
        for path in (file, directory, link, dangling, directory_link / "new"):
            with (
                self.subTest(path=path),
                self.assertRaises((ValueError, FileExistsError)),
            ):
                self.safety.claim_file(path)
        self.assertEqual(file.read_bytes(), b"retained")
        self.assertEqual(list(directory.iterdir()), [])

    def test_traversal_retained_tree_and_overlaps_rejected(self) -> None:
        for path in (self.root / "x/../new", self.root / "benchmarks/results/new"):
            with self.assertRaises(ValueError):
                self.safety.check_destination(path)
        for paths in (
            (self.root / "a", self.root / "a"),
            (self.root / "a", self.root / "a/b"),
        ):
            with self.assertRaises(ValueError):
                self.safety.check_outputs(*paths)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_exclusive_file_and_directory_reservations(self) -> None:
        file = self.root / "file"
        with self.safety.claim_file(file) as stream:
            stream.write("original")
        with self.assertRaises(FileExistsError):
            self.safety.claim_file(file)
        directory = self.safety.claim_directory(self.root / "directory")
        with self.assertRaises(FileExistsError):
            self.safety.claim_directory(directory)
        self.assertEqual(file.read_text(), "original")

    def test_failure_retained_and_rerun_rejected(self) -> None:
        output = self.root / "oracle.json"
        original = RuntimeError("workload failed")
        with self.assertRaises(RuntimeError) as caught, self.safety.json_output(output):
            raise original
        self.assertIs(caught.exception, original)
        saved = output.read_bytes()
        self.assertEqual(json.loads(saved)["detail"], "workload failed")
        with self.assertRaises(FileExistsError), self.safety.json_output(output):
            self.fail("existing output admitted")
        self.assertEqual(output.read_bytes(), saved)

    def test_secondary_write_and_close_cannot_mask_original_error(self) -> None:
        class Broken(io.StringIO):
            def close(self) -> None:
                raise OSError("close failure")

        stream = Broken()
        original = RuntimeError("original")
        with (
            patch.object(self.safety, "claim_file", return_value=stream),
            patch.object(
                self.safety, "write_json", side_effect=[None, OSError("full")]
            ),
            self.assertRaises(RuntimeError) as caught,
            self.safety.json_output(self.root / "new"),
        ):
            raise original
        self.assertIs(caught.exception, original)
        with (
            patch.object(self.safety, "claim_file", side_effect=OSError("full")),
            self.assertRaises(RuntimeError) as caught,
            self.safety.run_receipt(self.root, "diagnostic"),
        ):
            raise original
        self.assertIs(caught.exception, original)

    def test_exporter_checks_both_outputs_before_writing(self) -> None:
        namespace = self.main_namespace()
        existing = self.root / "existing"
        existing.write_text("retained")
        fresh = self.root / "fresh"
        for output, export in ((fresh, existing), (existing, fresh)):
            argv = [
                "summarize",
                "--output",
                str(output),
                "--export-reproducers",
                str(export),
            ]
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                namespace["main"]()
            self.assertFalse(fresh.exists())
            self.assertEqual(existing.read_text(), "retained")

    def test_exporter_includes_helper_and_reserves_before_population(self) -> None:
        namespace = self.main_namespace()
        names = namespace["REPRODUCERS"]
        self.assertEqual(
            set(names),
            {p.name for p in REPRODUCE.iterdir() if p.suffix in {".py", ".sh"}},
        )
        self.assertIn("output_safety.py", names)
        events = []

        @contextmanager
        def output(path: Path) -> Iterator[Any]:
            events.append("reserve-output")
            yield lambda value: events.append("publish")

        @contextmanager
        def receipt(*args: object) -> Iterator[None]:
            yield

        @contextmanager
        def file(path: Path) -> Iterator[io.StringIO]:
            events.append(path.name)
            yield io.StringIO()

        def directory(path: Path) -> Path:
            events.append("reserve-export")
            return path

        argv = [
            "summarize",
            "--output",
            str(self.root / "out"),
            "--export-reproducers",
            str(self.root / "export"),
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(self.safety, "claim_directory", side_effect=directory),
            patch.object(self.safety, "json_output", side_effect=output),
            patch.object(self.safety, "run_receipt", side_effect=receipt),
            patch.object(self.safety, "claim_file", side_effect=file),
        ):
            namespace["main"]()
        self.assertEqual(events[:2], ["reserve-export", "reserve-output"])
        self.assertEqual(set(events[2:-1]), set(names))
        self.assertEqual(events[-1], "publish")
        source = (BUNDLE / "summarize.py").read_text()
        self.assertIn(
            'runpy.run_path(str(ROOT / "reproduce/output_safety.py"))', source
        )
        for name in (
            "resident-diagnostic.py",
            "selected-j-oracle.py",
            "summarize-calls.py",
        ):
            self.assertIn("from output_safety import", (REPRODUCE / name).read_text())

    def test_python_workloads_reject_collision_before_scientific_imports(self) -> None:
        existing = self.root / "existing"
        existing.mkdir()
        with (
            patch.dict(sys.modules, {"output_safety": self.safety}),
            patch.dict(
                os.environ,
                {
                    "SLURM_JOB_ID": "host-test",
                    "GQC_MD_REPRO_ROOT": str(self.root),
                    "GQC_MD_ORACLE_OUTPUT": str(existing),
                },
            ),
        ):
            oracle = module_from_source(REPRODUCE / "selected-j-oracle.py")
            with self.assertRaises(FileExistsError):
                oracle.main()
            diagnostic = module_from_source(REPRODUCE / "resident-diagnostic.py")
            with (
                patch.dict(os.environ, DIAGNOSTIC_ARM="arm", DIAGNOSTIC_LABEL="trial"),
                patch.object(
                    diagnostic,
                    "claim_directory",
                    side_effect=FileExistsError("occupied"),
                ),
                self.assertRaises(FileExistsError),
            ):
                diagnostic.main()

    @unittest.skipUnless(hasattr(os, "memfd_create"), "Linux memfd host control")
    def test_shell_failure_receipt_and_repeat_reservation(self) -> None:
        # Execute only the reservation/trap prefix; all external work is stubbed.
        source = (REPRODUCE / "run-endpoint.sh").read_text()
        source = source[: source.index('\ncd "$root/$arm/source"')]
        source = (
            "\n".join(
                "script_dir=/host-stub" if line.startswith("script_dir=") else line
                for line in source.splitlines()
            )
            + "\n"
        )
        fd = os.memfd_create("md-output-safety", 0)
        try:
            os.write(fd, (REPRODUCE / "output_safety.py").read_bytes())
            stub = (
                "python3() { shift; "
                + sys.executable
                + " /proc/self/fd/"
                + str(fd)
                + ' "$@"; }\nccache() { return 9; }\n'
            )
            script = (
                stub
                + source
                + '\nprintf partial > "$directory/native.json"\nexit 124\n'
            )
            environment = dict(
                os.environ,
                GQC_MD_REPRO_ROOT=str(self.root),
                SLURM_JOB_ID="stub",
                CUDA_VISIBLE_DEVICES="stub",
            )
            command = ["bash", "-s", "--", "arm", "method", "48", "trial"]
            first = subprocess.run(
                command,
                input=script,
                text=True,
                env=environment,
                pass_fds=(fd,),
                capture_output=True,
                check=False,
            )
            self.assertEqual(first.returncode, 124, first.stderr)
            destination = self.root / "results/arm/endpoint/method/48/trial"
            self.assertEqual(
                json.loads((destination / "exit-status.json").read_text()),
                {"exit_code": 124},
            )
            saved = {p.name: p.read_bytes() for p in destination.iterdir()}
            second = subprocess.run(
                command,
                input=script,
                text=True,
                env=environment,
                pass_fds=(fd,),
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(
                {p.name: p.read_bytes() for p in destination.iterdir()}, saved
            )
        finally:
            os.close(fd)

    def test_shell_relative_roots_fail_before_reservation(self) -> None:
        environment = dict(
            os.environ,
            GQC_MD_REPRO_ROOT="relative-root",
            SLURM_JOB_ID="stub",
            CUDA_VISIBLE_DEVICES="stub",
        )
        for name in (
            "run-endpoint.sh",
            "run-probe.sh",
            "run-oracle.sh",
            "run-guarded.sh",
        ):
            source = (REPRODUCE / name).read_text()
            prefix = source.split('python3 "$safety" directory', 1)[0]
            prefix = "\n".join(
                "script_dir=/host-stub" if line.startswith("script_dir=") else line
                for line in prefix.splitlines()
            )
            # The old prefix reaches this sentinel; the corrected one exits first.
            checked = subprocess.run(
                ["bash", "-s", "--", "arm", "method", "48", "trial"],
                input=prefix + "\nexit 77\n",
                text=True,
                env=environment,
                cwd=self.root,
                capture_output=True,
                check=False,
            )
            self.assertEqual(checked.returncode, 2, name + checked.stderr)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_shell_contracts(self) -> None:
        for name in (
            "run-endpoint.sh",
            "run-probe.sh",
            "run-oracle.sh",
            "run-guarded.sh",
        ):
            source = (REPRODUCE / name).read_text()
            self.assertIn('python3 "$safety" directory', source)
            self.assertIn("local original_status=$?", source)
            self.assertIn('exit "$original_status"', source)
            self.assertNotIn("--force-overwrite=true", source)
            syntax = subprocess.run(
                ["bash", "-n"],
                input=source,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
        probe = (REPRODUCE / "run-probe.sh").read_text()
        self.assertIn('export DIAGNOSTIC_ARM="$arm"', probe)
        self.assertIn('export DIAGNOSTIC_LABEL="$operation/$trial/raw"', probe)


if __name__ == "__main__":
    unittest.main()
