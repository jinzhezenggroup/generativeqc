"""Synthetic CPU gate tests; fixtures are not allocation/science evidence."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import gate_wait as gate


class GateTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[SimpleNamespace, SimpleNamespace]:
        adapter_path = root / "synthetic_cpu_adapter.py"
        adapter_path.write_text(
            "import sys\nprint('SYNTHETIC CPU FIXTURE ONLY', sys.argv)\n"
        )
        prepared = {
            "schema": "generativeqc.h100-environment-smoke-plan.v1",
            "expected_job_name": "synthetic-unit-test",
            "adapter": str(adapter_path),
            "adapter_sha256": gate.sha(adapter_path),
            "sources": {"synthetic": True},
            "expected_packages": {"synthetic": True},
            "hf_inputs": "synthetic",
            "qm7": "synthetic",
            "qm7_sha256": "synthetic",
            "protocol_manifest": "synthetic",
            "protocol_manifest_sha256": "synthetic",
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
        path = root / "prepared.json"
        gate.write_new(path, prepared)
        args = SimpleNamespace(
            job_name="synthetic-unit-test",
            adapter=adapter_path,
            adapter_sha256=gate.sha(adapter_path),
            prepared_plan=path,
            prepared_plan_sha256=gate.sha(path),
            output=root / "attempt",
            release_marker=root / "release.json",
            wait_seconds=0.05,
            process_cap=2,
            job_cap=60,
            excluded_node="qb-prod-gpu095",
        )
        adapter = SimpleNamespace(
            device_observation=lambda: {
                "fixture": "CPU synthetic only",
                "hostname": "fixture",
            },
            environment_observation=lambda: {"fixture": True},
        )
        return args, adapter

    def release(
        self, args: SimpleNamespace, *, node: str = "other-synthetic-node"
    ) -> None:
        prepared = gate.read_json(args.prepared_plan)
        proof = args.output.parent / "synthetic-proof.json"
        gate.write_new(
            proof,
            {
                "name": "synthetic-unit-test",
                "status": "RUNNING",
                "instances": [{"node": node}],
                "synthetic": True,
                "runtime_preflight": {
                    "path": str(args.output / "preflight.json"),
                    "sha256": gate.sha(args.output / "preflight.json"),
                },
            },
        )
        final = copy.deepcopy(prepared)
        final.update(
            job_name="synthetic-unit-test",
            allocation_proof=str(proof),
            allocation_proof_sha256=gate.sha(proof),
        )
        path = args.output.parent / "final.json"
        gate.write_new(path, final)
        marker = {
            "schema": "generativeqc.h100-environment-release.v1",
            "coordinator_chat": gate.ROOT_CHAT,
            "preflight_sha256": gate.sha(args.output / "preflight.json"),
            "plan_path": str(path),
            "plan_sha256": gate.sha(path),
        }
        temporary = args.release_marker.with_suffix(".tmp")
        gate.write_new(temporary, marker)
        temporary.rename(args.release_marker)

    def test_no_release_expires_without_starting_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args, adapter = self.fixture(Path(name))
            with self.assertRaises(TimeoutError):
                gate.gate(args, adapter)
            result = gate.read_json(args.output / "job-outcome.json")
            self.assertEqual(result["endpoints_started"], 0)
            self.assertEqual(result["attempts"], [])
            self.assertTrue((args.output / "preflight.json").exists())

    def test_existing_output_or_stale_marker_never_starts(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args, adapter = self.fixture(Path(name))
            args.output.mkdir()
            sentinel = args.output / "keep"
            sentinel.write_text("preserve")
            with self.assertRaises(FileExistsError):
                gate.gate(args, adapter)
            self.assertEqual(sentinel.read_text(), "preserve")
        with tempfile.TemporaryDirectory() as name:
            args, adapter = self.fixture(Path(name))
            args.release_marker.write_text("{}")
            with self.assertRaises(ValueError):
                gate.gate(args, adapter)
            self.assertEqual(
                gate.read_json(args.output / "job-outcome.json")["endpoints_started"], 0
            )

    def test_source_change_or_excluded_node_refused(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args, adapter = self.fixture(Path(name))
            with self.assertRaises(TimeoutError):
                gate.gate(args, adapter)
            self.release(args, node=args.excluded_node)
            marker = gate.read_json(args.release_marker)
            with self.assertRaises(ValueError):
                gate.final_plan(
                    marker,
                    gate.read_json(args.prepared_plan),
                    gate.sha(args.output / "preflight.json"),
                    args.excluded_node,
                )
            final_path = Path(marker["plan_path"])
            final = gate.read_json(final_path)
            final["sources"] = {"corruption": True}
            final_path.write_text(json.dumps(final))
            marker["plan_sha256"] = gate.sha(final_path)
            with self.assertRaises(ValueError):
                gate.final_plan(
                    marker,
                    gate.read_json(args.prepared_plan),
                    gate.sha(args.output / "preflight.json"),
                    "another-node",
                )

    @unittest.skipUnless(os.name == "posix", "actual owned POSIX child processes")
    def test_release_runs_exactly_three_fresh_cpu_fixture_processes(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args, adapter = self.fixture(Path(name))
            args.wait_seconds = 2

            def release() -> None:
                while not (args.output / "preflight.json").exists():
                    time.sleep(0.01)
                self.release(args)

            publisher = threading.Thread(target=release)
            publisher.start()
            result = gate.gate(args, adapter)
            publisher.join(timeout=3)
            self.assertEqual(result["status"], "PROCESSES_RETURNED")
            self.assertEqual([x["arm"] for x in result["attempts"]], list(gate.ARMS))
            self.assertEqual(len({x["pid"] for x in result["attempts"]}), 3)
            self.assertTrue(all(x["returncode"] == 0 for x in result["attempts"]))

    @unittest.skipUnless(os.name == "posix", "actual owned POSIX child processes")
    def test_child_timeout_is_reaped_and_logs_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = gate.run_child(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import time;print('token=synthetic-secret https://example.invalid/secret');time.sleep(10)",
                ],
                root,
                0.1,
            )
            self.assertTrue(result["timed_out"])
            self.assertIsNotNone(result["returncode"])
            text = (root / "stdout.log").read_text()
            self.assertNotIn("synthetic-secret", text)
            self.assertNotIn("https://", text)

    def test_header_and_url_redaction(self) -> None:
        text = gate.redact(
            'Authorization: Bearer synthetic-secret\n{"token":"synthetic-secret"}\nwss://example.invalid/secret'
        )
        self.assertNotIn("synthetic-secret", text)
        self.assertNotIn("wss://", text)


if __name__ == "__main__":
    unittest.main()
