"""Host-only raw return integrity tests; never perform a scientific call."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from replay import recover, replay
from transport import retain_return, sha, write_new


class OfflineReplay(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_energy_only_force_missing_remains_missing(self) -> None:
        folder = self.root / "return"
        retain_return(folder, {"energy": -1.0, "forces": None, "complete_seconds": 1.2})
        self.assertIsNone(recover(folder)["forces"])

    def test_corrupt_json_numeric_refuses(self) -> None:
        folder = self.root / "return"
        retain_return(folder, {"energy": -1.0, "forces": None})
        (folder / "return.json").write_text(
            json.dumps({"energy": -2.0, "forces": None})
        )
        with self.assertRaises(ValueError):
            recover(folder)

    def test_missing_ieee_field_refuses(self) -> None:
        folder = self.root / "return"
        retain_return(folder, {"energy": -1.0, "forces": None})
        (folder / "values.bin").write_bytes(b"")
        (folder / "values.index.json").write_text("[]")
        with self.assertRaises(ValueError):
            recover(folder)

    def test_negative_zero_cannot_become_positive(self) -> None:
        folder = self.root / "return"
        retain_return(folder, {"energy": -0.0})
        (folder / "return.json").write_text('{"energy": 0.0}')
        with self.assertRaises(ValueError):
            recover(folder)

    def test_missing_cases_never_promote_to_pass(self) -> None:
        family = Path(__file__).parent
        plan = json.loads((family / "prepared-plan-v1.json").read_bytes())
        (self.root / "inputs-v1.json").write_bytes(
            (family / "inputs-v1.json").read_bytes()
        )
        plan["input_bundle_file"] = "inputs-v1.json"
        path = self.root / "plan.json"
        write_new(path, plan)
        output = self.root / "output"
        output.mkdir()
        write_new(
            output / "exclusive-batch-lease.json", {"prepared_plan_sha256": sha(path)}
        )
        write_new(output / "batch-outcome.json", {"status": "SYNTHETIC_NOT_RUN"})
        result = replay(path, output)
        self.assertEqual(result["status"], "FAIL_OR_INCOMPLETE")
        self.assertEqual(result["attempted_calls"], 0)
        self.assertTrue(all(r["status"] == "NOT_RUN" for r in result["calls"]))
        self.assertEqual(result["full_scientific_acceptance"], "INCOMPLETE")

    def test_metadata_failure_preserves_recoverable_return_without_numerical_promotion(
        self,
    ) -> None:
        family = Path(__file__).parent
        plan = json.loads((family / "prepared-plan-v1.json").read_bytes())
        (self.root / "inputs-v1.json").write_bytes(
            (family / "inputs-v1.json").read_bytes()
        )
        plan["input_bundle_file"] = "inputs-v1.json"
        path = self.root / "plan.json"
        write_new(path, plan)
        output = self.root / "output"
        output.mkdir()
        write_new(
            output / "exclusive-batch-lease.json", {"prepared_plan_sha256": sha(path)}
        )
        write_new(
            output / "batch-outcome.json", {"status": "SYNTHETIC_TRANSPORT_FAILURE"}
        )
        call = plan["calls"][0]
        process = output / call["process"]
        process.mkdir()
        write_new(process / "attempt-000.json", {"call": call})
        receipt = process / "call-000"
        receipt.mkdir()
        hashes = retain_return(
            receipt / "observables",
            {
                "schema": "generativeqc.h100-small-cbs-observables.v1",
                "call": call,
                "energy": -1.0,
                "forces": None,
                "components": None,
                "complete_seconds": 1.0,
            },
        )
        write_new(receipt / "observables-binding.json", hashes)
        result = replay(path, output)
        self.assertEqual(result["calls"][0]["status"], "TRANSPORT_INCOMPLETE")
        self.assertEqual(result["retained_native_returns"], 1)
        self.assertEqual(result["returned_EF_requests_with_metadata"], 0)
        self.assertIsNone(result["calls"][0]["retained_observables"]["forces"])


if __name__ == "__main__":
    unittest.main()
