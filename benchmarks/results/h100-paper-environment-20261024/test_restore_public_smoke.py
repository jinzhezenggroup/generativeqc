"""Opt-in recovery tests using existing device returns; no scientific execution."""

from __future__ import annotations

import copy
import json
import lzma
import tempfile
import unittest
from pathlib import Path

import restore_public_smoke as retained


class RetainedOutputTests(unittest.TestCase):
    def test_exact_originals_restore_and_independent_cpu_replay(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            destination = Path(name) / "new"
            result = retained.restore(destination)
            self.assertEqual(result["raw_files"], 27)
            for relative, (data, _) in retained.members().items():
                self.assertEqual((destination / relative).read_bytes(), data)
            results = retained.replay_all(destination)
            self.assertEqual(set(results), {"native-hf", "gpu4pyscf-hf", "cbs-c2"})
            for arm, result in results.items():
                self.assertEqual(result["numeric_replay"]["sample_count"], 1)
                self.assertTrue(
                    result["numeric_replay"]["samples"][0][
                        "finite_complete_energy_forces"
                    ]
                )
                comparison = result["reference_comparison"]
                if arm == "cbs-c2":
                    for term in comparison["constituents_and_composite"].values():
                        for gate in (
                            "independent_energy_gate",
                            "translation_gate",
                            "retained_direction_gate",
                        ):
                            self.assertEqual(term[gate], "PASS")
                    self.assertIn(
                        "UNAVAILABLE", comparison["independent_full_force_vector"]
                    )
                    self.assertIn("INCOMPLETE", comparison["scientific_acceptance"])
                else:
                    self.assertEqual(comparison["observable_gate"], "PASS")
                    self.assertEqual(comparison["quarter_spot_gate"], "PASS")

    def test_corrupt_capsule_is_refused_before_destination_creation(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            data = bytearray(retained.CAPSULE.read_bytes())
            data[len(data) // 2] ^= 1
            corrupt = root / "corrupt.xz"
            corrupt.write_bytes(data)
            destination = root / "never-created"
            with self.assertRaises(ValueError):
                retained.restore(destination, corrupt)
            self.assertFalse(destination.exists())

    def test_existing_destination_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            destination = Path(name)
            marker = destination / "existing.txt"
            marker.write_bytes(b"retain")
            with self.assertRaises(FileExistsError):
                retained.restore(destination)
            self.assertEqual(marker.read_bytes(), b"retain")

    def test_path_and_size_corruption_fail_closed(self) -> None:
        payload = json.loads(lzma.decompress(retained.CAPSULE.read_bytes()))
        source = str(Path(payload["output_root"]).parent.parent).replace("\\", "/")
        for change in ("traversal", "duplicate", "size"):
            value = copy.deepcopy(payload)
            if change == "traversal":
                value["files"][0]["path"] = source + "/../escape.json"
            elif change == "duplicate":
                value["files"].append(copy.deepcopy(value["files"][0]))
            else:
                value["files"][0]["bytes"] = retained.MAX_MEMBER_BYTES + 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                retained.validated_members(value)


if __name__ == "__main__":
    unittest.main()
