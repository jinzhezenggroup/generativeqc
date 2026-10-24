"""Host-only corruption and scope tests; no original environment is modified."""

from __future__ import annotations

import copy
import stat
import tempfile
import unittest
from pathlib import Path

from identity import safe_relative, verify
from transport import sha


class InputIdentity(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / "source/cbs-qualified/input.dat"
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(b"SYNTHETIC HOST INPUT\n")
        self.manifest = {
            "status": "PASS",
            "mismatches": [],
            "regular_files": 1,
            "regular_bytes": self.file.stat().st_size,
            "symlink_count": 0,
            "symlinks": [],
            "verified_file_rows": [
                {
                    "path": "source/cbs-qualified/input.dat",
                    "sha256": sha(self.file),
                    "bytes": self.file.stat().st_size,
                    "mode": oct(stat.S_IMODE(self.file.stat().st_mode)),
                }
            ],
        }

    def test_exact_synthetic_input(self) -> None:
        self.assertEqual(
            verify(self.root, self.manifest, {"files": {}})["regular_files"], 1
        )

    def test_corrupt_bytes_fail_closed(self) -> None:
        self.file.write_bytes(b"X" * self.file.stat().st_size)
        with self.assertRaises(ValueError):
            verify(self.root, self.manifest, {"files": {}})

    def test_additional_file_refuses(self) -> None:
        (self.file.parent / "unexpected").write_bytes(b"not an original input")
        with self.assertRaises(ValueError):
            verify(self.root, self.manifest, {"files": {}})

    def test_mode_change_in_manifest_refuses(self) -> None:
        bad = copy.deepcopy(self.manifest)
        bad["verified_file_rows"][0]["mode"] = "0o000"
        with self.assertRaises(ValueError):
            verify(self.root, bad, {"files": {}})

    def test_duplicate_identity_refuses(self) -> None:
        bad = copy.deepcopy(self.manifest)
        bad["verified_file_rows"] *= 2
        bad["regular_files"] = 2
        bad["regular_bytes"] *= 2
        with self.assertRaises(ValueError):
            verify(self.root, bad, {"files": {}})

    def test_credential_and_escape_paths_refuse(self) -> None:
        for name in (
            ".env",
            "env/.env.local",
            "../other",
            "/absolute",
            "dir\\other",
            "auth.json",
            "token.json",
        ):
            with self.assertRaises(ValueError):
                safe_relative(name)


if __name__ == "__main__":
    unittest.main()
