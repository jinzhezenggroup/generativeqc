"""Synthetic host evidence only: these fixtures never establish an allocation."""

from __future__ import annotations

import copy
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from transport import (
    ROOT_CHAT,
    bounded,
    can_continue_scientific_failure,
    read,
    release,
    retain_return,
    sha,
    write_new,
)


class TransportAdmission(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proof = self.root / "SYNTHETIC-NOT-AN-ALLOCATION.json"
        write_new(
            self.proof,
            {
                "name": "SYNTHETIC-EXPECTED-JOB",
                "created_at": 1,
                "provider": "qz",
                "status": "RUNNING",
                "gpu_count": 1,
                "platform_status": {"synthetic": True, "created_at": 1},
                "instances": [{"synthetic": True}],
                "resources": {"synthetic": True},
            },
        )
        self.plan = self.root / "plan.json"
        write_new(
            self.plan,
            {
                "expected_job_name": "SYNTHETIC-EXPECTED-JOB",
                "execution_authority": "SCI_NOT_AUTHORIZED",
                "caps": {"job_seconds": 14400},
                "task_files": {},
                "calls": [{"index": i} for i in range(32)],
                "processes": [{}] * 4,
            },
        )
        self.approval = {
            "schema": "generativeqc.h100-small-cbs-root-release.v2",
            "root_coordinator_chat": ROOT_CHAT,
            "prepared_plan_sha256": sha(self.plan),
            "expected_job_name": "SYNTHETIC-EXPECTED-JOB",
            "authorized_calls": 32,
            "authorized_science_processes": 4,
            "authority": "ONE_FINITE_BATCH",
            "allocation_proof": str(self.proof),
            "allocation_proof_sha256": sha(self.proof),
            "allocation_created_at": "1",
            "allocation_created_at_raw": 1,
            "expected_created_at_canonical": "1",
            "submission_window_ms": {"not_before": "1", "not_after": "2"},
            "released_unix_seconds": time.time(),
        }

    def approve(self, value: dict | None = None) -> Path:
        path = self.root / "release.json"
        write_new(path, value or self.approval)
        return path

    def test_synthetic_release_hash_validation_only(self) -> None:
        plan, _ = release(self.plan, self.approve())
        self.assertEqual(len(plan["calls"]), 32)

    def test_wrong_job_refuses(self) -> None:
        bad = copy.deepcopy(self.approval)
        bad["expected_job_name"] = "RENAMED"
        with self.assertRaises(ValueError):
            release(self.plan, self.approve(bad))

    def test_consistently_renamed_observations_refuse_fixed_plan(self) -> None:
        proof = read(self.proof)
        proof["name"] = "RENAMED"
        other = self.root / "renamed-observations.json"
        write_new(other, proof)
        bad = copy.deepcopy(self.approval)
        bad.update(allocation_proof=str(other), allocation_proof_sha256=sha(other))
        with self.assertRaises(ValueError):
            release(self.plan, self.approve(bad))

    def test_new_job_creation_identity_required(self) -> None:
        bad = copy.deepcopy(self.approval)
        bad["allocation_created_at"] = 2
        with self.assertRaises(ValueError):
            release(self.plan, self.approve(bad))

    def test_changed_plan_refuses(self) -> None:
        approved = self.approve()
        self.plan.write_bytes(self.plan.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            release(self.plan, approved)

    def test_call_33_refuses(self) -> None:
        bad = copy.deepcopy(self.approval)
        bad["authorized_calls"] = 33
        with self.assertRaises(ValueError):
            release(self.plan, self.approve(bad))

    def test_cleanup_or_timeout_cannot_become_continuable_science(self) -> None:
        worker = {
            "status": "FAILED",
            "stage": "SCI_NATIVE",
            "failure_kind": "SCIENTIFIC_REQUEST",
            "calls_attempted": 1,
        }
        for metadata in (
            {"trigger": "TIMEOUT"},
            {"trigger": "STORAGE_LIMIT"},
            {"cleanup_failed": True},
        ):
            self.assertFalse(
                can_continue_scientific_failure(
                    {"status": "FAILED", "returncode": 1, **metadata}, worker
                )
            )

    def test_post_return_and_control_failures_stop_other_models(self) -> None:
        for stage, kind in (
            ("TRANSPORT_RETURN", "TRANSPORT"),
            ("POST_DEVICE", "INTEGRITY"),
            ("SCI_NATIVE", "CONTROL_INTERRUPTION"),
            ("SCI_NUMERIC", "INTEGRITY"),
        ):
            self.assertFalse(
                can_continue_scientific_failure(
                    {"status": "FAILED", "returncode": 1},
                    {
                        "status": "FAILED",
                        "stage": stage,
                        "failure_kind": kind,
                        "calls_attempted": 1,
                    },
                )
            )

    def test_expired_release_refuses(self) -> None:
        bad = copy.deepcopy(self.approval)
        bad["released_unix_seconds"] = time.time() - 14401
        with self.assertRaises(ValueError):
            release(self.plan, self.approve(bad))

    def test_nan_timestamp_refuses(self) -> None:
        path = self.root / "release.json"
        value = json.dumps(self.approval).replace(
            str(self.approval["released_unix_seconds"]), "NaN"
        )
        path.write_text(value)
        with self.assertRaises(ValueError):
            release(self.plan, path)

    def test_raw_ieee_saved_before_rejection(self) -> None:
        out = self.root / "return"
        hashes = retain_return(
            out, {"energy": -0.0, "forces": [[float("nan"), float("inf"), 1.0]]}
        )
        data = (out / "values.bin").read_bytes()
        self.assertEqual(data[:8], struct.pack("<d", -0.0))
        self.assertTrue(math.isnan(struct.unpack("<d", data[8:16])[0]))
        self.assertEqual(
            set(hashes), {"return.json", "values.bin", "values.index.json"}
        )
        self.assertEqual(
            read(out / "return.json")["forces"][0][0], {"raw_nonfinite": "nan"}
        )

    def test_raw_no_clobber(self) -> None:
        out = self.root / "return"
        retain_return(out, {"energy": 1.0})
        before = (out / "values.bin").read_bytes()
        with self.assertRaises(FileExistsError):
            retain_return(out, {"energy": 2.0})
        self.assertEqual((out / "values.bin").read_bytes(), before)

    def test_missing_release_has_no_output(self) -> None:
        with self.assertRaises(FileNotFoundError):
            release(self.plan, self.root / "missing-release.json")
        self.assertFalse((self.root / "output").exists())

    def test_optimization_refuses_import(self) -> None:
        result = subprocess.run(
            [sys.executable, "-O", "-c", "import transport"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires non-optimized Python", result.stderr)

    @unittest.skipUnless(
        os.name == "posix",
        "real POSIX child-group lifecycle cannot be established on Windows",
    )
    def test_timeout_reaps_owned_group(self) -> None:
        jit = self.root / "jit"
        jit.mkdir()
        result = bounded(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            self.root / "process",
            self.root,
            jit,
            1,
            1 << 20,
            1 << 20,
        )
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertNotEqual(result["returncode"], 0)

    @unittest.skipUnless(
        os.name == "posix",
        "real POSIX child-group lifecycle cannot be established on Windows",
    )
    def test_failure_preserves_both_streams(self) -> None:
        jit = self.root / "jit"
        jit.mkdir()
        result = bounded(
            [
                sys.executable,
                "-c",
                'import sys;print("rawout");print("rawerr",file=sys.stderr);sys.exit(7)',
            ],
            self.root / "process",
            self.root,
            jit,
            10,
            1 << 20,
            1 << 20,
        )
        self.assertEqual(result["returncode"], 7)
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("rawout", (self.root / "process/stdout.txt").read_text())
        self.assertIn("rawerr", (self.root / "process/stderr.txt").read_text())


if __name__ == "__main__":
    unittest.main()
