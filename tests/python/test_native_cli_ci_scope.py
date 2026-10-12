"""Portable PR build-tier regressions for the monolithic CUDA CLI."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
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
            return_value=subprocess.CompletedProcess([], 0, stdout=paths, stderr=b""),
        ) as command:
            self.assertEqual(scope.select_tier("merge_group", BASE, HEAD), "fast")
            self.assertIn("-z", command.call_args.args[0])
            self.assertNotIn("--diff-filter=ACMRT", command.call_args.args[0])

    def test_real_git_rename_and_deletion_keep_original_native_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args: str) -> str:
                return subprocess.check_output(
                    ["git", "-C", str(root), *args], text=True, timeout=10
                ).strip()

            git("init", "-q")
            source = root / "src/cli/old source.cpp"
            source.parent.mkdir(parents=True)
            source.write_text("int native_only() { return 42; }\n")
            git("add", ".")
            git(
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.com",
                "commit",
                "-qm",
                "native source",
            )
            base = git("rev-parse", "HEAD")
            (root / "docs").mkdir()
            git("mv", "src/cli/old source.cpp", "docs/renamed.md")
            git(
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.com",
                "commit",
                "-qm",
                "move native source into docs",
            )
            renamed = git("rev-parse", "HEAD")
            git("rm", "docs/renamed.md")
            git(
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.com",
                "commit",
                "-qm",
                "remove moved source",
            )
            for head in (renamed, git("rev-parse", "HEAD")):
                for event in ("pull_request", "merge_group"):
                    with self.subTest(head=head, event=event):
                        result = subprocess.check_output(
                            [
                                sys.executable,
                                str(MODULE),
                                "--event",
                                event,
                                "--base-sha",
                                base,
                                "--head-sha",
                                head,
                            ],
                            cwd=root,
                            text=True,
                            timeout=10,
                        )
                        self.assertEqual(result.strip(), "release")

    def test_synthetic_merge_compares_against_its_current_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args: str) -> str:
                return subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(root),
                        "-c",
                        "user.name=fixture",
                        "-c",
                        "user.email=fixture@example.com",
                        *args,
                    ],
                    text=True,
                    timeout=10,
                ).strip()

            git("init", "-q", "-b", "base")
            (root / "src").mkdir()
            (root / "src/core.cpp").write_text("int value = 1;\n")
            git("add", ".")
            git("commit", "-qm", "initial source")
            git("checkout", "-qb", "docs")
            (root / "README.md").write_text("documentation only\n")
            git("add", ".")
            git("commit", "-qm", "documentation change")
            git("checkout", "-q", "base")
            (root / "src/core.cpp").write_text("int value = 2;\n")
            git("commit", "-qam", "concurrent native base change")
            base = git("rev-parse", "HEAD")
            git("merge", "--no-ff", "-qm", "synthetic merge", "docs")
            head = git("rev-parse", "HEAD")
            for event in ("pull_request", "merge_group"):
                result = subprocess.check_output(
                    [
                        sys.executable,
                        str(MODULE),
                        "--event",
                        event,
                        "--base-sha",
                        base,
                        "--head-sha",
                        head,
                    ],
                    cwd=root,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.strip(), "skip")


if __name__ == "__main__":
    unittest.main()
