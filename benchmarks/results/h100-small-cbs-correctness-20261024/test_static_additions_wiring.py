"""HOST-only caller wiring; scientific/native/cache executables are mocked."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import supervisor
from transport import read, sha, write_new


class StaticAdditionWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        old = os.environ.get("PYTHONDONTWRITEBYTECODE")
        flag = sys.dont_write_bytecode

        def restore() -> None:
            if old is None:
                os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
            else:
                os.environ["PYTHONDONTWRITEBYTECODE"] = old
            sys.dont_write_bytecode = flag

        self.addCleanup(restore)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        for name, value in (
            (
                "inputs.json",
                {"external_dependencies": {"files": {}}, "conventional": {}},
            ),
            ("manifest.json", {"SYNTHETIC": "base"}),
            ("proof.json", {"SYNTHETIC": "proof"}),
        ):
            write_new(self.root / name, value)
        self.additions = {
            "derivation_proof_file": "proof.json",
            "derivation_proof_sha256": sha(self.root / "proof.json"),
        }
        write_new(self.root / "additions.json", self.additions)
        library = self.root / "synthetic-library"
        library.write_bytes(b"SYNTHETIC; never loaded")
        self.plan = {
            "task_files": {},
            "science_source": {
                "root": str(self.root),
                "inventory": "SYNTHETIC",
                "inventory_sha256": "0" * 64,
                "library": str(library),
                "library_sha256": sha(library),
            },
            "input_bundle_file": "inputs.json",
            "input_bundle_sha256": sha(self.root / "inputs.json"),
            "environment_additions": {
                "file": "additions.json",
                "sha256": sha(self.root / "additions.json"),
            },
            "environment_inventory": {
                "path": str(self.root / "manifest.json"),
                "sha256": sha(self.root / "manifest.json"),
            },
            "public_readonly_runtime": str(self.root),
            "expected_python_executable": "SYNTHETIC-PYTHON",
            "expected_python_version": "SYNTHETIC-VERSION",
            "packages": {},
        }
        self.adapter = SimpleNamespace(
            checked_source_tree=lambda *a: {"synthetic": True},
            environment_observation=lambda: {
                "executable": "SYNTHETIC-PYTHON",
                "python": "SYNTHETIC-VERSION",
                "packages": {},
            },
        )

    def test_caller_passes_checked_base_addition_and_proof(self) -> None:
        with (
            patch.object(supervisor, "helper", return_value=self.adapter),
            patch.object(
                supervisor, "verify", return_value={"status": "PASS", "synthetic": True}
            ) as verify,
            patch.object(supervisor.shutil, "which", return_value="SYNTHETIC-sccache"),
            patch.object(
                supervisor.subprocess,
                "run",
                return_value=SimpleNamespace(
                    returncode=0, stdout="sccache 0.16.0", stderr=""
                ),
            ),
        ):
            supervisor.static_unbounded(
                self.root / "plan.json", self.plan, self.root / "static.json"
            )
        self.assertEqual(verify.call_args.kwargs["additions"], self.additions)
        self.assertEqual(
            verify.call_args.kwargs["base_manifest_sha256"],
            sha(self.root / "manifest.json"),
        )
        self.assertEqual(
            verify.call_args.kwargs["derivation_proof_sha256"],
            sha(self.root / "proof.json"),
        )
        self.assertEqual(
            read(self.root / "static.json")["whole_original_inputs"]["status"], "PASS"
        )

    def test_missing_outer_control_refuses_before_immutable_reads(self) -> None:
        os.environ["PYTHONDONTWRITEBYTECODE"] = "0"
        with (
            patch.object(supervisor, "verify") as verify,
            self.assertRaises(ValueError),
        ):
            supervisor.static_unbounded(
                self.root / "plan.json", self.plan, self.root / "must-not-exist.json"
            )
        verify.assert_not_called()
        self.assertFalse((self.root / "must-not-exist.json").exists())


if __name__ == "__main__":
    unittest.main()
