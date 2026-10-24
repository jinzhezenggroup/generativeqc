"""Host-only reader regressions; never execute the retained scientific runners."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
OPTIMIZED = (("-O", None), ("-OO", None), (None, "1"), (None, "2"))


def child(code=None, *, flag=None, optimize=None, script=None):
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    if optimize is not None:
        env["PYTHONOPTIMIZE"] = optimize
    command = [sys.executable, "-B"]
    if flag:
        command.append(flag)
    command += ["-c", code] if code is not None else [str(HERE / script), "--help"]
    return subprocess.run(command, cwd=HERE, env=env, capture_output=True,
                          text=True, timeout=30)


class ReaderIntegrity(unittest.TestCase):
    def test_corrupt_raw_checksum_is_rejected_in_every_mode(self):
        # An in-memory capsule isolates the integrity gate without editing raw.
        code = """
import json, lzma
import replay
class Capsule:
    def read_bytes(self):
        return lzma.compress(json.dumps({
            'files': {'probe': 'retained bytes'},
            'file_sha256': {'probe': '0' * 64}
        }).encode())
replay.load_raw(Capsule())
print('ACCEPTED_CORRUPT_CAPSULE')
"""
        for flag, optimize in ((None, None), *OPTIMIZED):
            with self.subTest(flag=flag, optimize=optimize):
                result = child(code, flag=flag, optimize=optimize)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertNotIn("ACCEPTED_CORRUPT_CAPSULE", result.stdout)
                self.assertRegex(result.stderr, "AssertionError|RuntimeError")

    def test_optimized_direct_entrypoints_fail_before_output(self):
        for script in ("replay.py", "unpack.py"):
            for flag, optimize in OPTIMIZED:
                with self.subTest(script=script, flag=flag, optimize=optimize):
                    result = child(script=script, flag=flag, optimize=optimize)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("requires non-optimized Python", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_optimization_environment_reaches_child_verifier(self):
        code = """
import subprocess, sys
result = subprocess.run([sys.executable, '-B', '-c', 'import replay'],
                        capture_output=True, text=True)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
sys.exit(result.returncode)
"""
        for optimize in ("1", "2"):
            with self.subTest(optimize=optimize):
                result = child(code, optimize=optimize)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires non-optimized Python", result.stderr)

    def test_normal_reader_accepts_valid_capsule(self):
        result = child("""
import hashlib, json, lzma
import replay
data = 'retained bytes'
class Capsule:
    def read_bytes(self):
        return lzma.compress(json.dumps({
            'files': {'probe': data},
            'file_sha256': {'probe': hashlib.sha256(data.encode()).hexdigest()}
        }).encode())
if replay.load_raw(Capsule())['files']['probe'] != data:
    raise RuntimeError('changed payload')
print('VALID_CAPSULE')
""")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "VALID_CAPSULE")

    def test_normal_entrypoints_remain_available(self):
        for script in ("replay.py", "unpack.py"):
            with self.subTest(script=script):
                result = child(script=script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
