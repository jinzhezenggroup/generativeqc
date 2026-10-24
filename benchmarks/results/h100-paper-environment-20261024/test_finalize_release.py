"""CPU-only tests with explicitly synthetic CLI/preflight fixtures, never GPU science."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import finalize_release as finalizer
import gate_wait as gate
import qz_smoke as adapter


class FinalizerTests(unittest.TestCase):
    def test_consistently_renamed_observations_cannot_change_expected_job(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args = self.fixture(Path(name))
            for path in (args.status, args.instances):
                value = gate.read_json(path)
                value["data"]["name"] = "different-consistent-job"
                adapter.persist(path, value)
            preflight = gate.read_json(args.preflight)
            preflight["job_name"] = "different-consistent-job"
            adapter.persist(args.preflight, preflight)
            args.preflight_sha256 = gate.sha(args.preflight)
            context = gate.read_json(args.query_context)
            context.update(
                job_name="different-consistent-job",
                status_sha256=gate.sha(args.status),
                instances_sha256=gate.sha(args.instances),
            )
            adapter.persist(args.query_context, context)
            with self.assertRaises(ValueError):
                finalizer.finalize(args, adapter)
            self.assertFalse(args.release_marker.exists())

    def test_deadline_rechecked_immediately_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args = self.fixture(Path(name))
            with (
                patch.object(
                    finalizer,
                    "gate_open",
                    side_effect=[None, ValueError("expired during validation")],
                ),
                self.assertRaises(ValueError),
            ):
                finalizer.finalize(args, adapter)
            self.assertFalse(args.release_marker.exists())

    def test_allocation_and_preflight_conditions_fail_closed(self) -> None:
        cases = [
            "status_failure",
            "instances_failure",
            "not_running",
            "project",
            "group",
            "workspace",
            "gpu",
            "cpu",
            "memory_gib",
            "shared_memory_gib",
            "worker_status",
            "worker_type",
            "duplicate",
            "endpoint_started",
            "gpu_name",
            "sm",
            "visible",
            "small_memory",
        ]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as name:
                args = self.fixture(Path(name))
                status, instances, preflight = (
                    gate.read_json(args.status),
                    gate.read_json(args.instances),
                    gate.read_json(args.preflight),
                )
                context = gate.read_json(args.query_context)
                if case == "status_failure":
                    status["success"] = False
                elif case == "instances_failure":
                    instances["success"] = False
                elif case == "not_running":
                    status["data"]["status"] = "QUEUING"
                elif case == "project":
                    status["data"]["project"] = "another-project"
                elif case == "group":
                    status["data"]["compute_group"] = "another-group"
                elif case == "workspace":
                    context["workspace"] = "another-workspace"
                elif case in {"gpu", "cpu", "memory_gib", "shared_memory_gib"}:
                    status["data"]["resource"][case] = 2
                elif case == "worker_status":
                    instances["data"]["items"][0]["status"] = "instance_pending"
                elif case == "worker_type":
                    instances["data"]["items"][0]["type"] = "another-type"
                elif case == "duplicate":
                    instances["data"]["items"].append(
                        dict(instances["data"]["items"][0])
                    )
                elif case == "endpoint_started":
                    preflight["endpoints_started"] = 1
                elif case == "gpu_name":
                    preflight["device"]["name"] = "H200"
                elif case == "sm":
                    preflight["device"]["compute_capability"] = [12, 0]
                elif case == "visible":
                    preflight["device"]["visible_device_count"] = 2
                elif case == "small_memory":
                    preflight["device"]["memory_bytes"] = 40 * (1 << 30)
                adapter.persist(args.status, status)
                adapter.persist(args.instances, instances)
                adapter.persist(args.preflight, preflight)
                args.preflight_sha256 = gate.sha(args.preflight)
                context.update(
                    status_sha256=gate.sha(args.status),
                    instances_sha256=gate.sha(args.instances),
                )
                adapter.persist(args.query_context, context)
                with self.assertRaises(ValueError):
                    finalizer.finalize(args, adapter)
                self.assertFalse(args.release_marker.exists())

    def fixture(self, root: Path) -> SimpleNamespace:
        prepared = {
            "expected_job_name": "synthetic-test-job",
            "schema": "generativeqc.h100-environment-smoke-plan.v1",
            "sources": {"fixture": True},
            "expected_packages": {"fixture": True},
            "hf_inputs": "fixture",
            "qm7": "fixture",
            "qm7_sha256": "fixture",
            "protocol_manifest": "fixture",
            "protocol_manifest_sha256": "fixture",
            "adapter": "fixture",
            "adapter_sha256": "a" * 64,
            "setup_calls": list(gate.ARMS),
            "caps": {
                "calls": 3,
                "fresh_processes": 3,
                "automatic_retries": 0,
                "per_process_seconds": 3600,
                "job_seconds": 14400,
                "termination_grace_seconds": 30,
            },
        }
        prepared_path = root / "prepared.json"
        gate.write_new(prepared_path, prepared)
        job = root / "job-output"
        job.mkdir()
        preflight = job / "preflight.json"
        gate.write_new(
            preflight,
            {
                "schema": "generativeqc.h100-environment-gate-preflight.v1",
                "fixture": "SYNTHETIC CPU ONLY",
                "job_name": "synthetic-test-job",
                "observed_utc": datetime.now(timezone.utc).isoformat(),
                "wait_cap_seconds": 300,
                "endpoints_started": 0,
                "adapter_sha256": prepared["adapter_sha256"],
                "prepared_plan_sha256": gate.sha(prepared_path),
                "device": {
                    "hostname": "synthetic-worker-0",
                    "visible_device_count": 1,
                    "compute_capability": [9, 0],
                    "name": "SYNTHETIC H100 80GB CPU TEST ONLY",
                    "memory_bytes": 85019590656,
                },
            },
        )
        status = root / "status.json"
        gate.write_new(
            status,
            {
                "success": True,
                "data": {
                    "name": "synthetic-test-job",
                    "status": "RUNNING",
                    "project": adapter.EXPECTED_PROJECT,
                    "compute_group": adapter.EXPECTED_GROUP,
                    "nodes": ["synthetic-other-node"],
                    "resource": {
                        "gpu": 1,
                        "nodes": 1,
                        "cpu": 20,
                        "memory_gib": 200,
                        "shared_memory_gib": 32,
                        "gpu_type": "NVIDIA H100 (80GB)",
                    },
                },
            },
        )
        instances = root / "instances.json"
        gate.write_new(
            instances,
            {
                "success": True,
                "data": {
                    "name": "synthetic-test-job",
                    "items": [
                        {
                            "status": "instance_running",
                            "type": "instance_type_worker",
                            "node": "synthetic-other-node",
                            "rank": 0,
                        }
                    ],
                },
            },
        )
        context = root / "query-context.json"
        gate.write_new(
            context,
            {
                "account_alias": "qz",
                "workspace": adapter.EXPECTED_WORKSPACE,
                "job_name": "synthetic-test-job",
                "scope_source": "root_cli_query_arguments",
                "status_sha256": gate.sha(status),
                "instances_sha256": gate.sha(instances),
            },
        )
        return SimpleNamespace(
            prepared_plan=prepared_path,
            prepared_plan_sha256=gate.sha(prepared_path),
            preflight=preflight,
            preflight_sha256=gate.sha(preflight),
            status=status,
            query_context=context,
            instances=instances,
            output=root / "control",
            release_marker=job / "release.json",
            observed_utc=datetime.now(timezone.utc).isoformat(),
            excluded_node="qb-prod-gpu095",
        )

    def test_actual_missing_id_shape_gets_explicit_unavailable_and_bound_marker(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as name:
            args = self.fixture(Path(name))
            result = finalizer.finalize(args, adapter)
            proof = gate.read_json(Path(result["proof_path"]))
            self.assertIsNone(proof["instances"][0]["platform_instance_id"])
            self.assertEqual(
                proof["instances"][0]["platform_instance_id_status"], "UNAVAILABLE"
            )
            self.assertEqual(proof["platform_instances"]["items"][0]["rank"], 0)
            self.assertNotIn("id", proof["platform_instances"]["items"][0])
            self.assertEqual(
                gate.read_json(args.release_marker)["preflight_sha256"],
                args.preflight_sha256,
            )
            with self.assertRaises(ValueError):
                finalizer.finalize(args, adapter)

    def test_expired_or_terminal_gate_gets_no_release(self) -> None:
        for terminal in (False, True):
            with tempfile.TemporaryDirectory() as name:
                args = self.fixture(Path(name))
                if terminal:
                    gate.write_new(
                        args.release_marker.parent / "job-outcome.json",
                        {"status": "FAILED", "fixture": True},
                    )
                else:
                    preflight = gate.read_json(args.preflight)
                    preflight["observed_utc"] = "2020-01-01T00:00:00+00:00"
                    adapter.persist(args.preflight, preflight)
                    args.preflight_sha256 = gate.sha(args.preflight)
                with self.assertRaises(ValueError):
                    finalizer.finalize(args, adapter)
                self.assertFalse(args.release_marker.exists())

    def test_fabricated_id_or_wrong_rank_or_excluded_node_refused(self) -> None:
        for change in ("id", "rank", "node"):
            with tempfile.TemporaryDirectory() as name:
                args = self.fixture(Path(name))
                response = gate.read_json(args.instances)
                row = response["data"]["items"][0]
                if change == "id":
                    row["id"] = "--pick=1"
                elif change == "rank":
                    row["rank"] = 1
                else:
                    row["node"] = args.excluded_node
                adapter.persist(args.instances, response)
                with self.assertRaises(ValueError):
                    finalizer.finalize(args, adapter)
                self.assertFalse(args.release_marker.exists())


if __name__ == "__main__":
    unittest.main()
