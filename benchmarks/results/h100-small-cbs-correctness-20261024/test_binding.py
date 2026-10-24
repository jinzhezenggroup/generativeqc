"""Synthetic host tests of device admission and root-release publication."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import finalize
import native_worker
from transport import read, sha, write_new


class AllocationBinding(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.boot = {
            "nvidia_smi": "SYNTHETIC GPU-UUID-1 PCI-1",
            "cuda_visible_devices": "SYNTHETIC-assigned",
            "hostname": "SYNTHETIC-container",
            "memory_bytes": 80 << 30,
            "compute_capability": [9, 0],
            "visible_device_count": 1,
        }
        self.preflight = self.root / "preflight.json"
        self.plan = self.root / "plan.json"
        write_new(self.plan, {"expected_job_name": "SYNTHETIC-new-job"})
        self.identity = self.root / "static.json"
        write_new(
            self.identity,
            {"whole_original_inputs": {"status": "PASS", "synthetic": True}},
        )
        write_new(
            self.preflight,
            {
                "device": self.boot,
                "job_name": "SYNTHETIC-new-job",
                "prepared_plan_sha256": sha(self.plan),
                "observed_utc": datetime.now(timezone.utc).isoformat(),
                "endpoints_started": 0,
                "static_identity": {
                    "path": str(self.identity),
                    "sha256": sha(self.identity),
                },
            },
        )
        self.proof = {
            "runtime_preflight": {
                "path": str(self.preflight),
                "sha256": sha(self.preflight),
            }
        }

    def test_changed_cuda_zero_refuses_after_sidecar(self) -> None:
        observed = {**self.boot, "nvidia_smi": "SYNTHETIC GPU-UUID-2 PCI-2"}
        mock = SimpleNamespace(
            device_observation=lambda: observed, bind_allocated_instance=lambda *a: None
        )
        sidecar = self.root / "actual-device.json"
        with (
            patch.object(native_worker, "helper", return_value=mock),
            self.assertRaises(ValueError),
        ):
            native_worker.device(
                {"expected_job_name": "SYNTHETIC-new-job"}, self.proof, sidecar
            )
        self.assertEqual(read(sidecar), observed)

    def test_changed_assigned_visibility_refuses(self) -> None:
        observed = {**self.boot, "cuda_visible_devices": "SYNTHETIC-other"}
        mock = SimpleNamespace(
            device_observation=lambda: observed, bind_allocated_instance=lambda *a: None
        )
        with (
            patch.object(native_worker, "helper", return_value=mock),
            self.assertRaises(ValueError),
        ):
            native_worker.device(
                {"expected_job_name": "SYNTHETIC-new-job"},
                self.proof,
                self.root / "actual-device.json",
            )

    def arguments(self) -> SimpleNamespace:
        status = self.root / "status.json"
        instances = self.root / "instances.json"
        write_new(
            status,
            {"success": True, "data": {"name": "SYNTHETIC-new-job", "created_at": 1}},
        )
        write_new(
            instances,
            {
                "success": True,
                "data": {
                    "name": "SYNTHETIC-new-job",
                    "items": [{"rank": 0, "node": "SYNTHETIC-node"}],
                },
            },
        )
        return SimpleNamespace(
            expected_created_at="1",
            submission_not_before_ms="1",
            submission_not_after_ms="2",
            plan=self.plan,
            plan_sha256=sha(self.plan),
            preflight=self.preflight,
            preflight_sha256=sha(self.preflight),
            status=status,
            instances=instances,
            proof=self.root / "allocation-proof.json",
            release=self.root / "release.json",
        )

    def test_terminal_gate_cannot_release(self) -> None:
        args = self.arguments()
        write_new(self.root / "gate-outcome.json", {"synthetic": "TIMEOUT"})
        with self.assertRaises(ValueError):
            finalize.finalize(args)
        self.assertFalse(args.release.exists())

    def test_wrong_plan_hash_cannot_release(self) -> None:
        args = self.arguments()
        args.plan_sha256 = "0" * 64
        with self.assertRaises(ValueError):
            finalize.finalize(args)
        self.assertFalse(args.release.exists())

    def test_expired_preflight_cannot_release(self) -> None:
        args = self.arguments()
        old = self.root / "expired.json"
        value = read(self.preflight)
        value["observed_utc"] = (
            datetime.now(timezone.utc) - timedelta(seconds=301)
        ).isoformat()
        write_new(old, value)
        args.preflight = old
        args.preflight_sha256 = sha(old)
        with self.assertRaises(ValueError):
            finalize.finalize(args)
        self.assertFalse(args.release.exists())

    def test_allocation_binding_failure_has_no_release(self) -> None:
        args = self.arguments()

        def reject(*unused: object) -> None:
            raise ValueError("SYNTHETIC wrong allocation")

        mock = SimpleNamespace(bind_allocated_instance=reject)
        with (
            patch.object(finalize, "helper", return_value=mock),
            self.assertRaises(ValueError),
        ):
            finalize.finalize(args)
        self.assertFalse(args.proof.exists())
        self.assertFalse(args.release.exists())

    def test_publication_is_no_clobber(self) -> None:
        args = self.arguments()
        mock = SimpleNamespace(bind_allocated_instance=lambda *a: None)
        with patch.object(finalize, "helper", return_value=mock):
            finalize.finalize(args)
        before = args.release.read_bytes()
        with self.assertRaises(ValueError):
            finalize.finalize(args)
        self.assertEqual(args.release.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
