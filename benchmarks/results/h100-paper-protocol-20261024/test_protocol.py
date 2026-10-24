"""Host-only behavioral admission tests. No scientific calls or source-body checks."""

import copy
import json
import math
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import protocol as p


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(p.HERE.joinpath("manifest.json").read_text())

    def attempt(self, index: int = 0) -> dict:
        slot = self.manifest["schedule"][index]
        return p.attempt_template(self.manifest, slot, f"attempt-{index}")

    def completed(self, index: int = 0) -> dict:
        attempt = self.attempt(index)
        cell = next(c for c in self.manifest["cells"] if c["id"] == attempt["cell"])
        source = self.manifest["sources"][cell["source"]]
        atoms = len(
            cell["input"].get("atoms_bohr", cell["input"].get("geometry_bohr", []))
        )
        attempt.update(
            outcome="PASS",
            exit_code=0,
            preflight="PASS",
            science="PASS",
            timing="PASS",
            runtime={
                "library_sha256": "a" * 64,
                "driver_sha256": "b" * 64,
                "environment_sha256": "c" * 64,
                "raw_sha256": "d" * 64,
                "source_driver_sha256": source["driver"]["sha256"],
                "cuda_architecture": "sm90",
                "host": "host",
                "threads": 1,
                "gpu": {"uuid": "synthetic-fixture"},
                "versions": {"python": "fixture"},
                "cache_state": "primed",
                "job_id": "fixture",
                "process_id": str(index),
                "process_wall_seconds": 30.0,
            },
            actual_policy={"precision": "fp64", "fallback": False, "refinement": None},
            work={
                k: {"status": "UNAVAILABLE", "value": None}
                for k in ("iterations", "fock_builds", "actions")
            },
            raw_artifacts=[{"path": "fixture.json", "sha256": "d" * 64}],
            failure_reason=None,
        )
        attempt["samples"] = [
            {
                "phase": phase,
                "seconds": float(i + 1),
                "energy": -1.0,
                "forces": [[0.0, 0.0, 0.0] for _ in range(atoms)],
                "gate_replay": "PASS",
                "gate_receipt_sha256": "e" * 64,
                "cold_density_sha256": "f" * 64,
            }
            for i, phase in enumerate(cell["calls_per_process"])
        ]
        return attempt

    def test_core_cardinality_and_repetition_hierarchy(self) -> None:
        self.assertEqual(len(self.manifest["cells"]), 11)
        self.assertEqual(len(self.manifest["schedule"]), 90)
        self.assertEqual(
            sum(
                len(
                    next(c for c in self.manifest["cells"] if c["id"] == s["cell"])[
                        "calls_per_process"
                    ]
                )
                for s in self.manifest["schedule"]
            ),
            306,
        )
        for cell in self.manifest["cells"]:
            slots = [s for s in self.manifest["schedule"] if s["cell"] == cell["id"]]
            for repeat in range(3):
                arms = cell["arms"]
                expected = arms[repeat % len(arms) :] + arms[: repeat % len(arms)]
                self.assertEqual(
                    [s["arm"] for s in slots if s["repeat"] == repeat], expected
                )

    def test_scientific_contract_mutations_refused(self) -> None:
        for field, replacement in (
            ("source", "archive-is-not-measured"),
            ("input_sha256", "0" * 64),
            ("gates", {"energy": 1.0}),
            ("arms", ["fast"]),
            ("input", {"model": "conventional"}),
        ):
            with self.subTest(field=field):
                modified = copy.deepcopy(self.manifest)
                modified["cells"][0][field] = replacement
                with (
                    patch.object(p, "build_manifest", return_value=self.manifest),
                    self.assertRaises(ValueError),
                ):
                    p.validate_manifest(modified)

    def test_manifest_source_and_hash_mutations_refused(self) -> None:
        for field in ("revision", "patch_sha256", "old_sm120_library_sha256"):
            modified = copy.deepcopy(self.manifest)
            modified["sources"]["cbs"][field] = "0" * 64
            with (
                patch.object(p, "build_manifest", return_value=self.manifest),
                self.assertRaises(ValueError),
            ):
                p.validate_manifest(modified)

    def test_incomplete_is_retained(self) -> None:
        report = p.validate_attempts(self.manifest, [self.attempt()])
        self.assertEqual(report["status"], "INCOMPLETE")
        with self.assertRaises(ValueError):
            p.validate_attempts(self.manifest, [], require_complete=True)

    def test_failed_attempts_not_discarded_or_zero_filled(self) -> None:
        for outcome in ("FAIL", "TIMEOUT", "OOM", "UNAVAILABLE"):
            attempt = self.attempt()
            attempt["outcome"] = outcome
            p.validate_attempt(self.manifest, attempt)
            self.assertIsNone(attempt["runtime"])
            attempt["failure_reason"] = None
            with self.assertRaises(ValueError):
                p.validate_attempt(self.manifest, attempt)

    def test_duplicate_attempt_and_process_refused(self) -> None:
        a = self.completed()
        p.validate_attempt(self.manifest, a)
        with self.assertRaises(ValueError):
            p.validate_attempts(self.manifest, [a, a])
        b = copy.deepcopy(a)
        b["attempt_id"] = "new-id"
        with self.assertRaises(ValueError):
            p.validate_attempts(self.manifest, [a, b])

    def test_finite_retry_cap_and_preserved_first_failure(self) -> None:
        a, b, c = (self.attempt() for _ in range(3))
        a.update(outcome="TIMEOUT", attempt_id="timeout")
        b["attempt_id"], c["attempt_id"] = "second", "third"
        p.validate_attempts(self.manifest, [a, b])
        with self.assertRaises(ValueError):
            p.validate_attempts(self.manifest, [a, b, c])

    def test_source_input_model_arm_and_target_refusals(self) -> None:
        mutations = [
            ("input_sha256", "0" * 64),
            ("source", {}),
            ("arm", "gpu4"),
            ("cohort", "current-master"),
            ("repeat", True),
            ("exit_code", 1),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                a = self.completed()
                a[key] = value
                with self.assertRaises(ValueError):
                    p.validate_attempt(self.manifest, a)
        for key, value in (
            ("cuda_architecture", "sm120"),
            (
                "library_sha256",
                self.manifest["sources"]["cbs"]["old_sm120_library_sha256"],
            ),
            ("source_driver_sha256", "0" * 64),
        ):
            a = self.completed()
            a["runtime"][key] = value
            with self.assertRaises(ValueError):
                p.validate_attempt(self.manifest, a)

    def test_missing_samples_nan_infinity_and_force_shape(self) -> None:
        for mutation in (
            "missing",
            "duplicate",
            "nan",
            "inf-force",
            "shape",
            "negative-time",
        ):
            a = self.completed(18)  # first HF/DFT slot
            if mutation == "missing":
                a["samples"].pop()
            elif mutation == "duplicate":
                a["samples"].append(a["samples"][0])
            elif mutation == "nan":
                a["samples"][0]["energy"] = math.nan
            elif mutation == "inf-force":
                a["samples"][0]["forces"][0][0] = math.inf
            elif mutation == "shape":
                a["samples"][0]["forces"].pop()
            else:
                a["samples"][0]["seconds"] = -1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                p.validate_attempt(self.manifest, a)

    def test_immutable_own_cold_density(self) -> None:
        a = self.completed(18)
        p.validate_attempt(self.manifest, a)
        a["samples"][2]["cold_density_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            p.validate_attempt(self.manifest, a)

    def test_unknown_work_must_be_explicit(self) -> None:
        a = self.completed()
        a["work"]["actions"] = 0
        with self.assertRaises(ValueError):
            p.validate_attempt(self.manifest, a)

    def test_receipt_contract_does_not_certify_science(self) -> None:
        attempts = [self.completed(i) for i in range(90)]
        report = p.validate_attempts(self.manifest, attempts, require_complete=True)
        self.assertEqual(report["status"], "COMPLETE_RECEIPT_CONTRACT")
        self.assertIn(
            "REQUIRES_INDEPENDENT_RAW_REPLAY", report["scientific_acceptance"]
        )
        rows = p.timing_statistics(self.manifest, attempts)
        self.assertEqual(len(rows), 30)
        water = rows[-1]
        self.assertEqual(water["warm_process_medians"], [3.0, 3.0, 3.0])
        self.assertEqual(water["warm_within_process_samples"], [[2.0, 3.0, 4.0]] * 3)

    def pilot(self) -> dict:
        return {
            c["id"]: {
                arm: {
                    "status": "PASS",
                    "receipt_sha256": "a" * 64,
                    "process_wall_seconds": 10.0,
                    "setup_seconds": 2.0,
                    "price_per_gpu_hour": 3.0,
                }
                for arm in c["arms"]
            }
            for c in self.manifest["cells"]
        }

    def test_pilot_cost_counts_retained_attempt_budget(self) -> None:
        report = p.cost_worksheet(self.manifest, self.pilot())
        self.assertEqual(len(report["rows"]), 30)
        self.assertAlmostEqual(
            report["rows"][0]["pilot_plus_max_attempt_gpu_hour_ceiling"],
            (10 + 6 * (52 + 30) + 2 * 2) / 3600,
        )
        self.assertIn("REQUIRES_ROOT_RESOURCE_APPROVAL", report["status"])

    def test_no_invented_pilot_or_unbounded_cap(self) -> None:
        for value in (None, math.nan, math.inf, -1.0, 4000.0):
            pilot = self.pilot()
            pilot["cbs-c2"]["standard"]["process_wall_seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                p.cost_worksheet(self.manifest, pilot)
        with self.assertRaises(ValueError):
            p.cost_worksheet(self.manifest, {})

    def test_stepwise_pilot_does_not_require_full_matrix_launch(self) -> None:
        pilot = self.pilot()
        report = p.cost_worksheet(self.manifest, {"cbs-c2": pilot["cbs-c2"]})
        self.assertEqual(len(report["rows"]), 2)
        self.assertIsNone(report["full_matrix_cost_ceiling"])
        self.assertEqual(len(report["not_estimated"]), 28)

    def test_optimized_python_retains_refusal(self) -> None:
        code = "import json,protocol as p;m=json.loads(p.HERE.joinpath('manifest.json').read_text());a=p.attempt_template(m,m['schedule'][0],'test');a['input_sha256']='bad';p.validate_attempt(m,a)"
        for flag in ([], ["-O"], ["-OO"]):
            env = os.environ.copy()
            env["PYTHONOPTIMIZE"] = "2"
            result = subprocess.run(
                [sys.executable, *flag, "-c", code],
                cwd=p.HERE,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("input hash", result.stderr)


if __name__ == "__main__":
    unittest.main()
