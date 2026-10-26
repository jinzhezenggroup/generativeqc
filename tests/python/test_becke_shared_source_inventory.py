"""The original source manifest stays byte-pinned before any GPU setup."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIRED = "becke-paired-warm-20261006"
PHYSICAL = "becke-physical-endpoints-20261006"


@unittest.skipUnless(
    shutil.which("bash") and shutil.which("sha256sum"),
    "POSIX reproduction tools required",
)
class SharedSourceInventoryTests(unittest.TestCase):
    def test_identity_checked_before_cuda_setup(self) -> None:
        results = ROOT / "benchmarks/results"
        original = (results / PHYSICAL / "source-files.sha256.gz").read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, shared = root / PAIRED, root / PHYSICAL
            bundle.mkdir()
            shared.mkdir()
            script = bundle / "reproduce.sh"
            script.write_bytes((results / PAIRED / "reproduce.sh").read_bytes())
            inventory = shared / "source-files.sha256.gz"
            env = dict(os.environ)
            env.pop("CUDA_ROOT", None)
            for raw in (original, original[:-1] + bytes([original[-1] ^ 1]), None):
                with self.subTest(valid=raw == original, missing=raw is None):
                    if raw is None:
                        inventory.unlink()
                    else:
                        inventory.write_bytes(raw)
                    result = subprocess.run(
                        ["bash", str(script)],
                        cwd=root,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    if raw == original:
                        self.assertIn("set CUDA_ROOT", result.stderr)
                    else:
                        self.assertNotIn("set CUDA_ROOT", result.stderr)
                    self.assertFalse((root / ".artifacts").exists())


if __name__ == "__main__":
    unittest.main()
