"""Portable PR build-tier regressions for the monolithic CUDA CLI."""

from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tools/native_cli_ci_scope.py"
spec = importlib.util.spec_from_file_location("native_cli_ci_scope", MODULE)
assert spec is not None and spec.loader is not None
scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scope)

BASE = "a" * 40
HEAD = "b" * 40


class NativeCliScopeTests(unittest.TestCase):
    def test_release_build_covers_link_and_distribution_changes(self) -> None:
        for path in (
            "CMakeLists.txt",
            "CMakePresets.json",
            "cmake/GenerativeQCGfn2Runtime.cmake",
            "src/cli/main.cpp",
            "tools/package_native_cli.py",
            "tests/python/test_native_cli_link.py",
            ".github/workflows/ci.yml",
            ".github/actions/install-native-cli-deps/action.yml",
        ):
            with self.subTest(path=path):
                self.assertEqual(scope.classify_paths([path]), "release")

    def test_compiled_sources_still_receive_real_cli_link_and_packaging(self) -> None:
        for path in (
            "src/xtb/native/src/runtime/gfn2_cuda_execution.cu",
            "src/scf/cuda/direct_bounded_fallback.cu",
            "include/generativeqc.h",
            "python/generativeqc_compiler/xc/spec.py",
            "tools/generate_gfn2_density_cuda.py",
            "upstream/manifest.json",
        ):
            with self.subTest(path=path):
                self.assertEqual(scope.classify_paths([path]), "fast")
        self.assertEqual(
            scope.classify_paths(["src/scf/cuda/a.cu", "cmake/CUDA.cmake"]),
            "release",
        )

    def test_docs_only_are_skipped(self) -> None:
        self.assertEqual(
            scope.classify_paths(["docs/user/native_cli.md", "README.md"]), "skip"
        )
        self.assertEqual(scope.classify_paths([]), "skip")

    def test_unknown_base_and_bad_diff_fail_closed(self) -> None:
        self.assertEqual(scope.select_tier("pull_request", "", HEAD), "release")
        self.assertEqual(scope.select_tier("merge_group", BASE, "?"), "release")
        with patch.object(
            scope.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            self.assertEqual(scope.select_tier("pull_request", BASE, HEAD), "release")

    def test_tag_and_master_publish_builds_do_not_skip(self) -> None:
        for event in ("push", "workflow_dispatch"):
            self.assertEqual(scope.select_tier(event, "", HEAD), "release")

    def test_git_diff_paths_keep_spaces_and_detect_deletions(self) -> None:
        paths = b"docs/user/read me.md\0src/xtb/native/deleted.cpp\0"
        with patch.object(
            scope.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                [], 0, stdout=paths, stderr=b""
            ),
        ) as command:
            self.assertEqual(scope.select_tier("merge_group", BASE, HEAD), "fast")
            self.assertIn("-z", command.call_args.args[0])
            self.assertNotIn("--diff-filter=ACMRT", command.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
