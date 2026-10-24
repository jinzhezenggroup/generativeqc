"""Reader checks using retained archives and synthetic perturbations, no science execution."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import oracle_replay as reader

RESULTS = Path(__file__).resolve().parents[1]


class OracleReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hf = reader.reference(
            RESULTS / "readme-repaired-20261021/oracles.json.xz",
            reader.HF_REFERENCE_SHA,
        )
        cls.cbs = reader.reference(
            RESULTS / "cbs-fast-memory-20261021/summary.json.xz",
            reader.CBS_REFERENCE_SHA,
        )
        cls.c2 = next(
            p["native"]
            for p in cls.cbs["points"]
            if p["native"]["protocol"]["carbon"] == 2
        )

    def test_retained_c2_comparison_does_not_fill_missing_originals(self) -> None:
        result = reader.compare_cbs(self.c2, self.c2["samples"][0], self.cbs)
        for row in result["constituents_and_composite"].values():
            self.assertEqual(row["independent_energy_gate"], "PASS")
            self.assertEqual(row["retained_direction_gate"], "PASS")
            self.assertEqual(row["translation_gate"], "PASS")
        self.assertIn("UNAVAILABLE", result["displaced_energy_originals"])
        self.assertIn("INCOMPLETE", result["scientific_acceptance"])

    def test_wrong_cbs_model_budget_or_missing_component_refused(self) -> None:
        for field, value in [
            ("hf_reference", "exact"),
            ("correlation_memory_budget_bytes", 1 << 30),
        ]:
            raw = copy.deepcopy(self.c2)
            raw["protocol"][field] = value
            with self.assertRaises(ValueError):
                reader.compare_cbs(raw, raw["samples"][0], self.cbs)
        sample = copy.deepcopy(self.c2["samples"][0])
        sample["components"].pop()
        with self.assertRaises(ValueError):
            reader.compare_cbs(self.c2, sample, self.cbs)

    def test_bad_energy_is_reported_as_fail(self) -> None:
        sample = copy.deepcopy(self.c2["samples"][0])
        sample["energy"] += 1.0
        result = reader.compare_cbs(self.c2, sample, self.cbs)
        self.assertEqual(
            result["constituents_and_composite"]["composite"][
                "independent_energy_gate"
            ],
            "FAIL",
        )

    def test_synthetic_hf_perturbation_and_quarter_spot_checks(self) -> None:
        ref = next(
            r
            for r in self.hf
            if r["method"] == "hf" and r["atoms"] == 3 and r["backend"] == "cuda"
        )
        raw = {"arm": "native-hf", "geometry_bohr": ref["geometry_bohr"]}
        sample = copy.deepcopy(ref["samples"][0])
        sample["energy"] += 3e-6
        sample["forces"][0][0] += 3e-6
        result = reader.compare_hf(raw, sample, self.hf)
        self.assertAlmostEqual(result["errors"]["energy_eh_per_atom"], 1e-6)
        self.assertEqual(result["observable_gate"], "PASS")
        self.assertEqual(result["quarter_spot_gate"], "PASS")
        sample["forces"][0][0] += 1.0
        self.assertEqual(
            reader.compare_hf(raw, sample, self.hf)["observable_gate"], "FAIL"
        )


if __name__ == "__main__":
    unittest.main()
