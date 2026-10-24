"""Small synthetic integrity cases; ordinary tests do not replay the campaign."""

from __future__ import annotations

import base64
import copy
import json
import lzma
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import restore_actual


class ActualRecoverySafety(unittest.TestCase):
    def value(self) -> dict:
        data = b"original -0.0\n"
        return {
            "raw_members": 1,
            "raw_bytes": len(data),
            "members": [
                {
                    "path": "output/original.txt",
                    "bytes": len(data),
                    "sha256": restore_actual.digest(data),
                    "mode": "0o644",
                    "base64": base64.b64encode(data).decode(),
                }
            ],
        }

    def test_original_bytes_preserved(self) -> None:
        self.assertEqual(
            restore_actual.validate_members(self.value())["output/original.txt"][0],
            b"original -0.0\n",
        )

    def test_tamper_or_duplicate_refuses(self) -> None:
        value = self.value()
        value["members"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            restore_actual.validate_members(value)
        value = self.value()
        value["members"] *= 2
        with self.assertRaises(ValueError):
            restore_actual.validate_members(value)

    def test_unsafe_paths_refuse(self) -> None:
        for name in (
            "../outside",
            "/absolute",
            "folder\\outside",
            "C:/outside",
            ".env",
        ):
            value = copy.deepcopy(self.value())
            value["members"][0]["path"] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                restore_actual.validate_members(value)

    def test_trailing_or_truncated_xz_refuses(self) -> None:
        packed = lzma.compress(json.dumps(self.value()).encode())
        self.assertEqual(restore_actual.decode(packed), self.value())
        for bad in (packed + b"trailing", packed[:-1], packed + packed):
            with self.assertRaises(ValueError):
                restore_actual.decode(bad)

    def test_restore_is_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "new"
            with patch.object(
                restore_actual,
                "members",
                return_value=restore_actual.validate_members(self.value()),
            ):
                restore_actual.restore(target)
                before = (target / "output/original.txt").read_bytes()
                with self.assertRaises(FileExistsError):
                    restore_actual.restore(target)
                self.assertEqual((target / "output/original.txt").read_bytes(), before)

    def test_optimization_cannot_bypass_integrity(self) -> None:
        result = subprocess.run(
            [sys.executable, "-B", "-O", "-c", "import restore_actual"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires non-optimized Python", result.stderr)


if __name__ == "__main__":
    unittest.main()
