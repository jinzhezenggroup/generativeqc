"""Offline integrity checks for the MD working view and recovery instructions."""

from __future__ import annotations

import hashlib
import json
import lzma
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/md-j-sparse-support-20261016"


class EvidenceRetention(unittest.TestCase):
    def test_current_manifest_covers_each_retained_member(self) -> None:
        entries = {}
        for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
            digest, path = line.split("  ", 1)
            self.assertNotIn(path, entries)
            entries[path] = digest
            self.assertEqual(
                hashlib.sha256((BUNDLE / path).read_bytes()).hexdigest(), digest, path
            )
        actual = {
            str(path.relative_to(BUNDLE))
            for path in BUNDLE.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.name != "SHA256SUMS"
        }
        self.assertEqual(set(entries), actual)

    def test_full_archive_and_provenance_aliases_are_explicit(self) -> None:
        recovery = json.loads((BUNDLE / "recovery.json").read_text())
        self.assertEqual(recovery["commit"], "ef02ef322e809a905cca09e7d2b3323b9d1972ae")
        self.assertIn(recovery["commit"], recovery["raw_url_template"])
        entries = {entry["path"]: entry for entry in recovery["files"]}
        self.assertEqual(len(entries), 44)
        for path, entry in entries.items():
            self.assertFalse(Path(path).is_absolute())
            self.assertNotIn("..", Path(path).parts)
            self.assertEqual(len(entry["git_blob"]), 40)
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertGreater(entry["bytes"], 0)
        self.assertIn("SHA256SUMS", entries)
        self.assertTrue(any(path.startswith("trials/") for path in entries))
        self.assertIn("validation/logs.json.xz", entries)
        self.assertIn("raw/reference-r2scan-3c-96.json.xz", entries)
        with lzma.open(BUNDLE / "evidence.json.xz", "rt") as stream:
            evidence = json.load(stream)
        for alias, binding in recovery["provenance_aliases"].items():
            archived = entries[binding["archive_path"]]
            pooled = evidence["raw_record_provenance"][alias]
            self.assertEqual(archived["sha256"], binding["decoded_sha256"])
            self.assertEqual(archived["bytes"], binding["decoded_bytes"])
            self.assertEqual(pooled["sha256"], binding["pooled_storage_sha256"])
            self.assertEqual(pooled["bytes"], binding["pooled_storage_bytes"])
            self.assertNotIn(alias, entries)
        for path in evidence["raw_record_provenance"]:
            self.assertTrue(
                path in entries or path in recovery["provenance_aliases"], path
            )

    def test_pooled_endpoint_gates_recompute_offline(self) -> None:
        namespace = runpy.run_path(str(BUNDLE / "summarize.py"))
        summary = namespace["summarize"]()
        populations = [
            case[arm]
            for case in summary["endpoint_cases"].values()
            for arm in ("baseline", "candidate")
        ]
        self.assertEqual(sum(len(row["calls"]) for row in populations), 96)
        for population in populations:
            self.assertLessEqual(population["maximum_energy_error"], 1e-8)
            self.assertLessEqual(population["maximum_force_error"], 1e-7)
            self.assertEqual(len(population["calls"]), 12)

    def test_historical_capability_table_decodes_exactly(self) -> None:
        directory = ROOT / "benchmarks/results/cg01"
        storage = json.loads((directory / "storage.json").read_text())["files"][0]
        stored = (directory / storage["path"]).read_bytes()
        decoded = lzma.decompress(stored)
        self.assertEqual(len(stored), storage["stored_bytes"])
        self.assertEqual(hashlib.sha256(stored).hexdigest(), storage["stored_sha256"])
        self.assertEqual(len(decoded), storage["decoded_bytes"])
        self.assertEqual(hashlib.sha256(decoded).hexdigest(), storage["decoded_sha256"])
        self.assertIsInstance(json.loads(decoded), dict)
        self.assertFalse((directory / storage["original_path"]).exists())


if __name__ == "__main__":
    unittest.main()
