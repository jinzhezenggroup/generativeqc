"""Synthetic HOST admission tests; no bytecode is imported or executed."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from identity import verify
from transport import sha


class ExactDerivedAddition(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = (
            self.root / "env/lib/python3.11/site-packages/_distutils_hack/__init__.py"
        )
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"SYNTHETIC HOST SOURCE\n")
        os.utime(self.source, (1, 1))
        self.cache = self.source.parent / "__pycache__/__init__.cpython-311.pyc"
        self.cache.parent.mkdir()
        self.cache.write_bytes(b"SYNTHETIC DERIVED INPUT; NOT EXECUTABLE")
        source_row = self.row(self.source)
        self.manifest = {
            "status": "PASS",
            "mismatches": [],
            "regular_files": 1,
            "regular_bytes": source_row["bytes"],
            "verified_file_rows": [source_row],
            "symlinks": [],
            "symlink_count": 0,
        }
        self.base_sha = hashlib.sha256(json.dumps(self.manifest).encode()).hexdigest()
        self.proof = {
            "scope": "SYNTHETIC metadata shape only; not a real bytecode qualification",
            "no_site": 1,
            "isolated": 1,
            "distutils_hack_imported": False,
            "path": self.row(self.cache)["path"],
            "bytes": self.cache.stat().st_size,
            "sha256": sha(self.cache),
            "source_path": source_row["path"],
            "source_bytes": source_row["bytes"],
            "source_sha256": source_row["sha256"],
            "source_matches_original_manifest": True,
            "header_flags": 0,
            "header_source_mtime": 1,
            "actual_source_mtime": 1,
            "header_source_size": source_row["bytes"],
            "actual_source_size": source_row["bytes"],
            "co_filename": str(self.source),
            "expected_filename": str(self.source),
            "code_objects_equal": True,
            "marshal_payload_exact": True,
            "cache_bytes_unchanged_after": True,
            "source_bytes_unchanged_after": True,
        }
        self.proof_sha = hashlib.sha256(json.dumps(self.proof).encode()).hexdigest()
        self.additions = {
            "schema": "generativeqc.environment-derived-additions.v1",
            "base_manifest_sha256": self.base_sha,
            "derivation_proof_sha256": self.proof_sha,
            "added_files": [self.row(self.cache)],
        }

    def row(self, path: Path) -> dict:
        return {
            "path": path.relative_to(self.root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha(path),
            "mode": oct(stat.S_IMODE(path.stat().st_mode)),
        }

    def check(self, additions: dict | None = None, proof: dict | None = None) -> dict:
        return verify(
            self.root,
            self.manifest,
            {"files": {}},
            additions=additions or self.additions,
            base_manifest_sha256=self.base_sha,
            derivation_proof=proof or self.proof,
            derivation_proof_sha256=self.proof_sha,
        )

    def test_exact_addition_admitted_without_mutating_base(self) -> None:
        before = copy.deepcopy(self.manifest)
        result = self.check()
        self.assertEqual(result["base_regular_files"], 1)
        self.assertEqual(result["qualified_added_files"], 1)
        self.assertEqual(result["regular_files"], 2)
        self.assertEqual(self.manifest, before)

    def test_without_qualification_original_nameset_still_refuses(self) -> None:
        with self.assertRaises(ValueError):
            verify(self.root, self.manifest, {"files": {}})

    def test_other_extra_file_still_refuses(self) -> None:
        self.cache.with_name("other.cpython-311.pyc").write_bytes(b"not qualified")
        with self.assertRaises(ValueError):
            self.check()

    def test_changed_added_bytes_or_mode_refuses(self) -> None:
        bad = copy.deepcopy(self.additions)
        bad["added_files"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.check(bad)
        bad = copy.deepcopy(self.additions)
        bad["added_files"][0]["mode"] = "0o000"
        with self.assertRaises(ValueError):
            self.check(bad)

    def test_wrong_base_or_proof_identity_refuses(self) -> None:
        for key in ("base_manifest_sha256", "derivation_proof_sha256"):
            bad = copy.deepcopy(self.additions)
            bad[key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(bad)

    def test_failed_equivalence_or_source_binding_refuses(self) -> None:
        for key in (
            "marshal_payload_exact",
            "code_objects_equal",
            "source_matches_original_manifest",
        ):
            bad = copy.deepcopy(self.proof)
            bad[key] = False
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(proof=bad)
        bad = copy.deepcopy(self.proof)
        bad["source_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.check(proof=bad)

    def test_replacement_or_second_added_row_refuses(self) -> None:
        bad = copy.deepcopy(self.additions)
        bad["added_files"].append(self.row(self.source))
        with self.assertRaises(ValueError):
            self.check(bad)
        bad = copy.deepcopy(self.additions)
        bad["added_files"] = [self.row(self.source)]
        with self.assertRaises(ValueError):
            self.check(bad)

    def test_source_timestamp_pairing_change_refuses(self) -> None:
        os.utime(self.source, (2, 2))
        with self.assertRaises(ValueError):
            self.check()


if __name__ == "__main__":
    unittest.main()
