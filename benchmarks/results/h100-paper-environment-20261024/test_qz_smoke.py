"""CPU transport tests; synthetic endpoints are not device/scientific evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import stat
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import replay as cpu_replay

SPEC = importlib.util.spec_from_file_location(
    "qz_smoke", Path(__file__).with_name("qz_smoke.py")
)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class TransportTests(unittest.TestCase):
    def test_rank_binding_rejects_fabricated_or_ambiguous_proof(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            plan, _ = self.fixture(Path(name))
            original = smoke.load_json(Path(plan["allocation_proof"]))
            changes = [
                lambda p: p["instances"][0].update(platform_instance_id="rank=0"),
                lambda p: p["instances"][0].update(id="--pick=1"),
                lambda p: p["platform_instances"].update(name="another-job"),
                lambda p: p["platform_instances"]["items"][0].update(rank=1),
                lambda p: p["platform_status"].update(nodes=["another-node"]),
                lambda p: p["platform_instances"]["items"].append(
                    copy.deepcopy(p["platform_instances"]["items"][0])
                ),
                lambda p: p["runtime_preflight"].update(sha256="0" * 64),
                lambda p: p["instances"][0].update(hostname="another-host"),
            ]
            for change in changes:
                proof = copy.deepcopy(original)
                change(proof)
                with (
                    self.subTest(change=change),
                    self.assertRaises((ValueError, RuntimeError)),
                ):
                    smoke.bind_allocated_instance(
                        proof, plan["job_name"], {"hostname": "synthetic"}
                    )

    def test_observed_rank_node_and_preflight_bind_without_invented_id(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan, module = self.fixture(root)
            with (
                patch.object(
                    smoke,
                    "device_observation",
                    return_value={"synthetic": True, "hostname": "synthetic"},
                ),
                patch.object(smoke.importlib, "import_module", return_value=module),
                patch.dict(smoke.os.environ),
                patch.object(smoke.sys, "path", smoke.sys.path.copy()),
            ):
                result = smoke.execute(plan, "cbs-c2", root / "attempt")
            self.assertEqual(result["status"], "FINITE_ENDPOINT_RETURNED")
            self.assertIsNone(result["allocated_instance"]["platform_instance_id"])
            self.assertEqual(
                result["allocated_instance"]["platform_instance_id_status"],
                "UNAVAILABLE",
            )

    def test_backend_observation_uses_loaded_selector_without_import_or_override(
        self,
    ) -> None:
        key = "gpu4pyscf.lib.cutensor"
        for module, expected in [
            (SimpleNamespace(contract_engine="cupy"), "cupy"),
            (SimpleNamespace(contract_engine=None, cutensor=object()), "cutensor"),
        ]:
            with (
                patch.dict(smoke.sys.modules, {key: module}),
                patch.object(smoke.importlib, "import_module") as imported,
            ):
                result = smoke.reference_backend_observation()
                self.assertEqual(result["engine"], expected)
                self.assertFalse(result["backend_override_applied_by_adapter"])
                imported.assert_not_called()
        with patch.dict(smoke.sys.modules, {key: SimpleNamespace()}):
            self.assertEqual(
                smoke.reference_backend_observation()["status"], "UNAVAILABLE"
            )

    def test_hf_uses_original_function_for_one_complete_call(self) -> None:
        driver = SimpleNamespace(run_native=Mock(), run_reference=Mock())
        atoms = [[8, [0.0, 0.0, 0.0]]]
        record, retain = {}, Mock()
        for arm, target in [
            ("native-hf", driver.run_native),
            ("gpu4pyscf-hf", driver.run_reference),
        ]:
            smoke.call_hf(driver, arm, atoms, Path("inputs"), record, retain)
            args, actual_atoms, actual_record, actual_retain = target.call_args.args
            self.assertEqual(
                (args.method, args.mode, args.warm_repeats), ("hf", "direct", 0)
            )
            self.assertIs(actual_atoms, atoms)
            self.assertIs(actual_record, record)
            self.assertIs(actual_retain, retain)

    def test_nonfinite_missing_and_wrong_shape_rejected(self) -> None:
        for energy, forces in [
            (float("nan"), [[0, 0, 0]]),
            (0, []),
            (0, [[0, 0]]),
            (0, [[0, True, 0]]),
            (0, [[0, float("inf"), 0]]),
        ]:
            with (
                self.subTest(energy=energy, forces=forces),
                self.assertRaises(ValueError),
            ):
                smoke.finite_endpoint(energy, forces, 1)
        smoke.finite_endpoint(-1.0, [[0.0, 0.0, 0.0]], 1)

    def test_nonfinite_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "bad.json"
            p.write_text('{"x": NaN}')
            with self.assertRaises(ValueError):
                smoke.load_json(p)

    def test_ieee_originals_preserve_nan_payload_inf_and_signed_zero(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            payload_nan = struct.unpack("<d", bytes.fromhex("420000000000f87f"))[0]
            sample = {"energy": -0.0, "forces": [[payload_nan, float("inf"), -0.0]]}
            manifest = smoke.retain_numeric_bytes(output, sample, 0)
            raw = (output / manifest["values_file"]).read_bytes()
            self.assertEqual(raw[:8], struct.pack("<d", -0.0))
            self.assertEqual(raw[8:16], bytes.fromhex("420000000000f87f"))
            self.assertTrue(math.isinf(struct.unpack("<d", raw[16:24])[0]))
            self.assertEqual(raw[24:32], struct.pack("<d", -0.0))
            with self.assertRaises(FileExistsError):
                smoke.retain_numeric_bytes(output, sample, 0)

    def test_cpu_replay_recovers_array_bytes_and_rejects_tampering(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            sample = {
                "energy": -1.0,
                "forces": np.array([[0.0, -0.0, 0.25]], dtype=">f8"),
            }
            originals = smoke.retain_numeric_bytes(root, sample, 0)
            smoke.persist(root / "display.json", sample)
            display = smoke.load_json(root / "display.json")
            result = cpu_replay.recover_sample(root, display, originals)
            self.assertEqual(
                struct.pack("<d", result["forces"][0][1]), struct.pack("<d", -0.0)
            )
            display["forces"][0][2] = 0.5
            with self.assertRaises(ValueError):
                cpu_replay.recover_sample(root, display, originals)
            display = smoke.load_json(root / "display.json")
            display["forces"][0][0] = False
            with self.assertRaises(ValueError):
                cpu_replay.recover_sample(root, display, originals)
            values = root / originals["values_file"]
            values.write_bytes(values.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                cpu_replay.recover_sample(
                    root, smoke.load_json(root / "display.json"), originals
                )

    def fixture(self, root: Path) -> tuple[dict, SimpleNamespace]:
        source = root / "source"
        (source / "benchmarks").mkdir(parents=True)
        driver = source / "benchmarks/readme_cbs.py"
        driver.write_text("# synthetic CPU test fixture\n")
        inventory = root / "source-inventory.json"
        inventory.write_text(
            json.dumps(
                {
                    "source_tree": {
                        "benchmarks/readme_cbs.py": {
                            "sha256": smoke.digest(driver),
                            "bytes": driver.stat().st_size,
                            "mode": oct(stat.S_IMODE(driver.stat().st_mode)),
                        }
                    }
                }
            )
        )
        files = {}
        for name in ("library", "manifest", "qm7", "allocation"):
            p = root / name
            p.write_text(
                json.dumps(
                    {
                        "provider": "qz",
                        "name": "synthetic-test-only",
                        "gpu_count": 1,
                        "status": "RUNNING",
                        "observed_utc": "synthetic-CPU-test",
                        "platform_status": {"synthetic": True},
                        "instances": [
                            {
                                "hostname": "synthetic",
                                "node": "synthetic",
                                "id": "synthetic",
                            }
                        ],
                        "resources": {"synthetic": True},
                    }
                )
                if name == "allocation"
                else "synthetic CPU fixture\n"
            )
            files[name] = p
        preflight = root / "runtime-preflight.json"
        smoke.persist(
            preflight,
            {
                "schema": "generativeqc.h100-environment-gate-preflight.v1",
                "fixture": "CPU ONLY; not an allocation",
                "job_name": "synthetic-test-only",
                "device": {"hostname": "synthetic"},
            },
        )
        allocation = {
            "schema": "generativeqc.h100-environment-allocation.v2",
            "provider": "qz",
            "name": "synthetic-test-only",
            "gpu_count": 1,
            "status": "RUNNING",
            "observed_utc": "synthetic-CPU-unit-test",
            "resources": {"cpu": 20, "memory_gib": 200, "gpu": 1},
            "platform_status": {
                "name": "synthetic-test-only",
                "status": "RUNNING",
                "nodes": ["synthetic-node"],
                "resource": {"gpu": 1, "nodes": 1, "cpu": 20, "memory_gib": 200},
            },
            "platform_instances": {
                "name": "synthetic-test-only",
                "items": [
                    {
                        "status": "instance_running",
                        "type": "instance_type_worker",
                        "node": "synthetic-node",
                        "rank": 0,
                    }
                ],
            },
            "instances": [
                {
                    "rank": 0,
                    "node": "synthetic-node",
                    "platform_instance_id": None,
                    "platform_instance_id_status": "UNAVAILABLE",
                    "hostname": "synthetic",
                    "hostname_source": "runtime_preflight",
                }
            ],
            "runtime_preflight": {
                "path": str(preflight),
                "sha256": smoke.digest(preflight),
            },
        }
        allocation["platform_status"].update(
            project=smoke.EXPECTED_PROJECT, compute_group=smoke.EXPECTED_GROUP
        )
        allocation["platform_status"]["resource"].update(
            shared_memory_gib=32, gpu_type="NVIDIA H100 (80GB)"
        )
        allocation["platform_job_uid"] = None
        allocation["platform_job_uid_status"] = "UNAVAILABLE"
        allocation["raw_status_response"] = {
            "success": True,
            "data": copy.deepcopy(allocation["platform_status"]),
        }
        allocation["raw_instances_response"] = {
            "success": True,
            "data": copy.deepcopy(allocation["platform_instances"]),
        }
        status_path, instances_path = (
            root / "status-original.json",
            root / "instances-original.json",
        )
        smoke.persist(status_path, allocation["raw_status_response"])
        smoke.persist(instances_path, allocation["raw_instances_response"])
        allocation["original_status_sha256"] = smoke.digest(status_path)
        allocation["original_instances_sha256"] = smoke.digest(instances_path)
        allocation["query_context"] = {
            "account_alias": "qz",
            "workspace": smoke.EXPECTED_WORKSPACE,
            "job_name": "synthetic-test-only",
            "scope_source": "root_cli_query_arguments",
            "status_sha256": smoke.digest(status_path),
            "instances_sha256": smoke.digest(instances_path),
        }
        boot = smoke.load_json(preflight)
        boot.update(endpoints_started=0, wait_cap_seconds=300)
        boot["device"].update(
            visible_device_count=1,
            compute_capability=[9, 0],
            name="SYNTHETIC H100 80GB CPU FIXTURE",
            memory_bytes=85019590656,
        )
        smoke.persist(preflight, boot)
        allocation["runtime_preflight"]["sha256"] = smoke.digest(preflight)
        smoke.persist(files["allocation"], allocation)
        plan = {
            "schema": "generativeqc.h100-environment-smoke-plan.v1",
            "expected_job_name": "synthetic-test-only",
            "expected_packages": smoke.environment_observation()["packages"],
            "sources": {
                "cbs": {
                    "root": str(source),
                    "driver": "benchmarks/readme_cbs.py",
                    "driver_sha256": smoke.digest(driver),
                    "inventory": str(inventory),
                    "inventory_sha256": smoke.digest(inventory),
                    "library": str(files["library"]),
                    "library_sha256": smoke.digest(files["library"]),
                }
            },
            "protocol_manifest": str(files["manifest"]),
            "protocol_manifest_sha256": smoke.digest(files["manifest"]),
            "allocation_proof": str(files["allocation"]),
            "allocation_proof_sha256": smoke.digest(files["allocation"]),
            "job_name": "synthetic-test-only",
            "qm7": str(files["qm7"]),
            "qm7_sha256": smoke.digest(files["qm7"]),
        }
        atoms = [[6, [0.0, 0.0, 0.0]]]
        endpoint = {
            "energy": -1.0,
            "forces": [[0.0, 0.0, 0.0]],
            "converged": True,
            "executed_backend": "cuda",
            "complete_seconds": 0.125,
        }
        module = SimpleNamespace(
            __file__=str(driver),
            geometry=Mock(return_value=atoms),
            setup=Mock(
                return_value=(
                    {"synthetic": True},
                    ("dz", "tz"),
                    ("dz-jkfit", "tz-jkfit"),
                )
            ),
            native_call=Mock(return_value=endpoint),
        )
        return plan, module

    def test_cbs_parameter_boundary_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan, module = self.fixture(root)
            output = root / "attempt"
            with (
                patch.object(
                    smoke,
                    "device_observation",
                    return_value={"synthetic": True, "hostname": "synthetic"},
                ),
                patch.object(smoke.importlib, "import_module", return_value=module),
                patch.dict(smoke.os.environ),
                patch.object(smoke.sys, "path", smoke.sys.path.copy()),
            ):
                result = smoke.execute(plan, "cbs-c2", output)
                module.setup.assert_called_once_with(
                    module.geometry.return_value, 2, 34359738368, "df"
                )
                module.native_call.assert_called_once_with(
                    module.geometry.return_value,
                    ("dz", "tz"),
                    ("dz-jkfit", "tz-jkfit"),
                    34359738368,
                    "df",
                    "standard",
                )
                self.assertEqual(result["scientific_acceptance"], "INCOMPLETE")
                self.assertEqual(result["samples"][0]["complete_seconds"], 0.125)
                before = (output / "raw.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    smoke.execute(plan, "cbs-c2", output)
                self.assertEqual(before, (output / "raw.json").read_bytes())

    def test_hash_failure_precedes_any_device_or_endpoint_call(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan, module = self.fixture(root)
            Path(plan["sources"]["cbs"]["library"]).write_text("corrupt")
            with (
                patch.object(smoke, "device_observation") as device,
                self.assertRaises(ValueError),
            ):
                smoke.execute(plan, "cbs-c2", root / "attempt")
            device.assert_not_called()
            module.native_call.assert_not_called()

    def test_changed_python_source_is_refused_before_device(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan, module = self.fixture(root)
            (Path(plan["sources"]["cbs"]["root"]) / "unexpected.py").write_text(
                "# source drift\n"
            )
            with (
                patch.object(smoke, "device_observation") as device,
                self.assertRaises(ValueError),
            ):
                smoke.execute(plan, "cbs-c2", root / "attempt")
            device.assert_not_called()
            module.native_call.assert_not_called()

    def test_package_drift_is_refused_before_device(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan, module = self.fixture(root)
            plan["expected_packages"]["numpy"] = "synthetic-wrong-version"
            with (
                patch.object(smoke, "device_observation") as device,
                self.assertRaises(ValueError),
            ):
                smoke.execute(plan, "cbs-c2", root / "attempt")
            device.assert_not_called()
            module.native_call.assert_not_called()

    def test_three_field_allocation_claim_is_not_allocation_proof(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan, module = self.fixture(root)
            proof = Path(plan["allocation_proof"])
            proof.write_text(
                json.dumps(
                    {"provider": "qz", "name": "synthetic-test-only", "gpu_count": 1}
                )
            )
            plan["allocation_proof_sha256"] = smoke.digest(proof)
            with (
                patch.object(smoke, "device_observation") as device,
                self.assertRaises(ValueError),
            ):
                smoke.execute(plan, "cbs-c2", root / "attempt")
            device.assert_not_called()
            module.native_call.assert_not_called()

    def test_runtime_hostname_must_bind_to_real_platform_instance(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan, module = self.fixture(root)
            with (
                patch.object(
                    smoke,
                    "device_observation",
                    return_value={"hostname": "other-synthetic-instance"},
                ),
                self.assertRaises(RuntimeError),
            ):
                smoke.execute(plan, "cbs-c2", root / "attempt")
            module.native_call.assert_not_called()
            raw = smoke.load_json(root / "attempt/raw.json")
            self.assertEqual(raw["status"], "FAILED")
            self.assertEqual(raw["samples"], [])

    def test_returned_raw_saved_before_numerical_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan, module = self.fixture(root)
            module.native_call.return_value["forces"] = [[float("nan"), 0.0, 0.0]]
            with (
                patch.object(
                    smoke,
                    "device_observation",
                    return_value={"synthetic": True, "hostname": "synthetic"},
                ),
                patch.object(smoke.importlib, "import_module", return_value=module),
                patch.dict(smoke.os.environ),
                patch.object(smoke.sys, "path", smoke.sys.path.copy()),
                self.assertRaises(ValueError),
            ):
                smoke.execute(plan, "cbs-c2", root / "attempt")
            raw = smoke.load_json(root / "attempt/raw.json")
            self.assertEqual(raw["status"], "FAILED")
            self.assertEqual(raw["samples"][0]["energy"], -1.0)
            self.assertEqual(
                raw["samples"][0]["forces"][0][0], {"raw_nonfinite": "nan"}
            )


if __name__ == "__main__":
    unittest.main()
