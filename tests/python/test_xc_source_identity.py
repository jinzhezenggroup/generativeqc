"""Reject changed compact XC identities without re-running measured GPU work."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[2]


class SourceIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.replay = runpy.run_path(str(
                ROOT / "benchmarks/results/xc-bounded-cross-20261024/reproduce.py"
            ))

    def test_complete_replay(self):
        self.assertEqual(self.replay["report"]["qualified_calls"], 352)
        self.assertEqual(self.replay["report"]["records"], 108)

    def test_rehashed_wrong_identity(self):
        for arm, field, value in (
            ("baseline", "revision", "0" * 40),
            ("baseline", "tree", "0" * 40),
            ("baseline", "source_patch", "dirty"),
            ("guarded", "revision", "0" * 40),
            ("guarded", "tree", "0" * 40),
            ("guarded", "baseline_revision", "0" * 40),
            ("guarded", "changed_files", {}),
            ("guarded", "files_verified", 0),
            ("guarded", "manifest_bytes", 0),
            ("guarded", "manifest_sha256", "0" * 64),
        ):
            with self.subTest(arm=arm, field=field):
                p = copy.deepcopy(self.replay["provenance"])
                p["source_identities"][arm][field] = value
                p["source_identities_sha256"] = hashlib.sha256(json.dumps(
                    p["source_identities"], sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest()
                with self.assertRaises(AssertionError):
                    self.replay["verify_source_identities"](p)

    def test_changed_difference_hash(self):
        p = copy.deepcopy(self.replay["provenance"])
        changes = p["source_identities"]["guarded"]["changed_files"]
        changes[next(iter(changes))]["guarded_sha256"] = "0" * 64
        with self.assertRaises(AssertionError):
            self.replay["verify_source_identities"](p)
