"""Exercise cache administration using isolated temporary directories only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(sys.argv.pop(1)).resolve()


class ProfileCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="gqc-cli-profile-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.env = {
            **os.environ,
            "HOME": str(self.root / "home"),
            "XDG_CACHE_HOME": str(self.root / "xdg"),
            "GENERATIVEQC_PROFILE_CACHE": str(self.cache),
        }

    def cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(CLI), *args], env=self.env, text=True, capture_output=True, check=False
        )

    def test_missing_valid_and_malformed_indexes(self) -> None:
        result = self.cli("profile", "show")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["active"], {})
        self.assertFalse((self.cache / "active.json").exists())
        valid = {"compatibility": 'bundle-"\\\n\u2603'}
        (self.cache / "active.json").write_text(json.dumps(valid))
        result = self.cli("autotune", "--show-profile")
        self.assertEqual(json.loads(result.stdout)["active"], valid)
        for invalid in (
            "{garbage}",
            '{"a":"b",}',
            '{"a":"b"} {"injected":"yes"}',
            '{"a":"\\q"}',
            '{"a":"\\u123"}',
            '{"a":"unterminated}',
            '{"a":"literal\nnewline"}',
            '{"a":42}',
            "[]",
        ):
            with self.subTest(invalid=invalid):
                (self.cache / "active.json").write_text(invalid)
                result = self.cli("profile", "show")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("profile index", result.stderr)

    def test_utf8_is_validated_before_printing(self) -> None:
        valid = {"snowman": "☃", "emoji": "🐈", "accent": "é"}
        (self.cache / "active.json").write_text(json.dumps(valid, ensure_ascii=False))
        self.assertEqual(
            json.loads(self.cli("profile", "show").stdout)["active"], valid
        )
        for invalid in (
            b"\xff",
            b"\x80",
            b"\xc0\x80",
            b"\xe0\x80\x80",
            b"\xed\xa0\x80",
            b"\xf0\x80\x80\x80",
            b"\xf4\x90\x80\x80",
            b"\xf5\x80\x80\x80",
            b"\xc2",
        ):
            with self.subTest(invalid=invalid):
                (self.cache / "active.json").write_bytes(b'{"x":"' + invalid + b'"}')
                result = subprocess.run(
                    [str(CLI), "profile", "show"],
                    env=self.env,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")
                self.assertIn(b"profile index", result.stderr)

    def test_cache_precedence(self) -> None:
        result = self.cli("profile", "show")
        self.assertEqual(json.loads(result.stdout)["cache"], str(self.cache))
        self.env.pop("GENERATIVEQC_PROFILE_CACHE")
        result = self.cli("profile", "show")
        self.assertEqual(
            json.loads(result.stdout)["cache"],
            str(self.root / "xdg/generativeqc/profiles"),
        )
        self.env.pop("XDG_CACHE_HOME")
        result = self.cli("profile", "show")
        self.assertEqual(
            json.loads(result.stdout)["cache"],
            str(self.root / "home/.cache/generativeqc/profiles"),
        )
        self.env["GENERATIVEQC_PROFILE_CACHE"] = "~/custom"
        result = self.cli("profile", "show")
        self.assertEqual(
            json.loads(result.stdout)["cache"], str(self.root / "home/custom")
        )

    @unittest.skipUnless(os.name == "posix", "POSIX atomic cache administration")
    def test_clear_retains_bundles_and_does_not_follow_temporary_symlinks(self) -> None:
        bundle = self.cache / "bundles/immutable"
        bundle.mkdir(parents=True)
        sentinel = bundle / "libgenerativeqc.so"
        sentinel.write_text("keep this immutable bundle")
        unrelated = self.root / "unrelated"
        unrelated.write_text("keep this unrelated file")
        (self.cache / "active.json").write_text('{"key":"immutable"}')
        # exec preserves the PID, reproducing the old predictable temporary path.
        helper = (
            "import os, pathlib, sys; root=pathlib.Path(os.environ['GENERATIVEQC_PROFILE_CACHE']);"
            "(root / ('active.json.tmp.' + str(os.getpid()))).symlink_to(sys.argv[2]);"
            "os.execv(sys.argv[1], [sys.argv[1], 'profile', 'clear'])"
        )
        result = subprocess.run(
            [sys.executable, "-c", helper, str(CLI), str(unrelated)],
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(unrelated.read_text(), "keep this unrelated file")
        self.assertEqual(sentinel.read_text(), "keep this immutable bundle")
        self.assertFalse((self.cache / "active.json").is_symlink())
        self.assertEqual(json.loads((self.cache / "active.json").read_text()), {})
        self.assertEqual(self.cli("autotune", "--clear-profile").returncode, 0)

    @unittest.skipUnless(os.name == "posix", "POSIX atomic cache administration")
    def test_failed_replace_leaves_existing_target_and_no_temporary_files(self) -> None:
        (self.cache / "active.json").mkdir()
        result = self.cli("profile", "clear")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.cache / "active.json").is_dir())
        self.assertEqual(list(self.cache.glob("active.json.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
