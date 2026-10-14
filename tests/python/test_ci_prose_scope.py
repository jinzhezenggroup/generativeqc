"""Synthetic-Git negative controls for the proposed README-only CI scope."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import ci_prose_scope as scope


class ProseScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.git("init", "-q", "-b", "target")
        self.write("README.md", b"# Project\nOriginal prose\n")
        self.write("src/core.cpp", b"int original;\n")
        self.base = self.commit("initial")
        self.git("remote", "add", "origin", str(self.root))
        self.git("checkout", "-qb", "queue")

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.com",
                *args,
            ],
            text=True,
            timeout=10,
        ).strip()

    def write(self, path: str, content: bytes) -> None:
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    def commit(self, title: str = "change") -> str:
        self.git("add", "-A")
        self.git("commit", "-qm", title)
        return self.git("rev-parse", "HEAD")

    def tier(
        self,
        event: str = "merge_group",
        base: str | None = None,
        head: str | None = None,
        target: str = "refs/heads/target",
        pr_head: str = "",
    ) -> str:
        return subprocess.check_output(
            [
                sys.executable,
                str(ROOT / "tools/ci_prose_scope.py"),
                "--event",
                event,
                "--base-sha",
                self.base if base is None else base,
                "--head-sha",
                self.git("rev-parse", "HEAD") if head is None else head,
                "--target-ref=" + target,
                "--pr-head-sha=" + pr_head,
            ],
            cwd=self.root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()

    def readme(self) -> str:
        self.write("README.md", b"# Project\nNew ordinary prose\n")
        return self.commit()

    def test_readme_only_merge_queue(self) -> None:
        self.readme()
        self.assertEqual(self.tier(), "prose")

    def test_two_readme_queue_nodes(self) -> None:
        prefix = self.readme()
        self.write("README.md", b"# Project\nAnother paragraph\n")
        self.commit()
        self.assertEqual(self.tier(base=prefix), "prose")

    def test_all_other_paths_retain_full_coverage(self) -> None:
        paths = [
            "docs/user/guide.md",
            "docs/conf.py",
            "docs/requirements.txt",
            "docs/cuda_ownership/inventory.json",
            "docs/assets/figure.svg",
            "docs/fixture.md",
            "assets/figure.svg",
            "src/numerical.cpp",
            "python/generativeqc/example.py",
            "tests/data/fixture.json",
            "tests/native/fixture.md",
            "tests/python/test_ci_prose_scope.py",
            "tools/ci_prose_scope.py",
            "tools/native_cli_ci_scope.py",
            ".github/workflows/ci.yml",
            ".github/actions/test/action.yml",
            ".github/SECURITY.md",
            "SECURITY.md",
            "AGENTS.md",
            ".agents/notes/proposed/claim.md",
            ".pre-commit-config.yaml",
            "pyproject.toml",
            "uv.lock",
            "CMakeLists.txt",
            "cmake/config.cmake",
            "manifests/acceptance.json",
            "benchmarks/results/run/README.md",
            "benchmarks/results/run/summary.json.xz",
            "LICENSE",
            "unknown",
            "docs/space name.md",
            "docs/new\nline.md",
            "Readme.md",
            "README.md.py",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.git("reset", "--hard", "-q", self.base)
                self.write(path, b"changed\n")
                self.commit()
                self.assertEqual(self.tier(), "full")

    def test_source_deletion_hidden_by_readme_edit_stays_full(self) -> None:
        self.git("rm", "-q", "src/core.cpp")
        self.readme()
        self.assertEqual(self.tier(), "full")

    def test_source_rename_to_allowlisted_readme_stays_full(self) -> None:
        self.git("rm", "-q", "README.md")
        self.git("mv", "src/core.cpp", "README.md")
        self.commit()
        self.assertEqual(self.tier(), "full")

    def test_readme_rename_stays_full(self) -> None:
        self.git("mv", "README.md", "docs.md")
        self.commit()
        self.assertEqual(self.tier(), "full")

    def test_readme_deletion_stays_full(self) -> None:
        self.git("rm", "-q", "README.md")
        self.commit()
        self.assertEqual(self.tier(), "full")

    def test_readme_addition_stays_full(self) -> None:
        self.git("rm", "-q", "README.md")
        base = self.commit()
        self.git("update-ref", "refs/heads/target", base)
        self.readme()
        self.assertEqual(self.tier(base=base), "full")

    def test_symlink_mode_binary_invalid_utf8_and_empty_stay_full(self) -> None:
        for variant in (
            "symlink",
            "executable",
            "nul",
            "invalid",
            "blank",
            "empty",
            "control",
            "oversize",
        ):
            with self.subTest(variant=variant):
                self.git("reset", "--hard", "-q", self.base)
                path = self.root / "README.md"
                if variant == "symlink":
                    path.unlink()
                    path.symlink_to("src/core.cpp")
                elif variant == "executable":
                    path.chmod(0o755)
                else:
                    path.write_bytes(
                        {
                            "nul": b"text\0binary",
                            "invalid": b"\xff",
                            "blank": b" \n",
                            "empty": b"",
                            "control": b"text\x1b",
                            "oversize": b"x" * (1024 * 1024 + 1),
                        }[variant]
                    )
                self.commit()
                self.assertEqual(self.tier(), "full")

    def test_binary_base_cannot_become_prose_only(self) -> None:
        self.write("README.md", b"binary\0old")
        base = self.commit()
        self.git("update-ref", "refs/heads/target", base)
        self.readme()
        self.assertEqual(self.tier(base=base), "full")

    def test_earlier_source_survives_readme_only_queue_tail(self) -> None:
        self.write("src/core.cpp", b"int changed;\n")
        prefix = self.commit()
        self.readme()
        self.assertEqual(self.tier(base=prefix), "full")

    def test_source_queue_needs_no_remote_fetch(self) -> None:
        self.write("src/core.cpp", b"int changed;\n")
        self.commit()
        trace = self.root / "git-trace"
        with patch.dict(os.environ, {"GIT_TRACE": str(trace)}):
            self.assertEqual(self.tier(), "full")
        self.assertNotIn(" fetch ", trace.read_text())

    def test_earlier_workflow_survives_readme_only_queue_tail(self) -> None:
        self.write(".github/workflows/ci.yml", b"workflow\n")
        prefix = self.commit()
        self.readme()
        self.assertEqual(self.tier(base=prefix), "full")

    def test_landed_source_prefix_can_be_excluded(self) -> None:
        self.write("src/core.cpp", b"int changed;\n")
        prefix = self.commit()
        self.readme()
        self.git("update-ref", "refs/heads/target", prefix)
        self.assertEqual(self.tier(base=prefix), "prose")

    def test_full_stacked_merge_nodes_include_first_source_pr(self) -> None:
        self.git("checkout", "-qb", "source-pr", self.base)
        self.write("src/core.cpp", b"int changed;\n")
        self.commit()
        self.git("checkout", "-q", "queue")
        self.git("merge", "--no-ff", "-qm", "first queue node", "source-pr")
        prefix = self.git("rev-parse", "HEAD")
        self.git("checkout", "-qb", "readme-pr", self.base)
        self.readme()
        self.git("checkout", "-q", "queue")
        self.git("merge", "--no-ff", "-qm", "last queue node", "readme-pr")
        self.assertEqual(self.tier(base=prefix), "full")

    def test_current_tested_pr_base_excludes_already_landed_work(self) -> None:
        source_head = self.readme()
        self.git("checkout", "-q", "target")
        self.write("src/core.cpp", b"int newer_master;\n")
        self.commit()
        self.git("merge", "--no-ff", "-qm", "PR test merge", "queue")
        self.assertEqual(self.tier(event="pull_request", pr_head=source_head), "prose")

    def test_source_pr_never_uses_prose_tier(self) -> None:
        self.readme()
        self.write("src/core.cpp", b"int changed;\n")
        source_head = self.commit()
        self.git("checkout", "-q", "target")
        self.git("merge", "--no-ff", "-qm", "PR test merge", "queue")
        self.assertEqual(self.tier(event="pull_request", pr_head=source_head), "full")

    def test_unverified_pr_merge_stays_full(self) -> None:
        head = self.readme()
        for source_head in ("", "invalid", "f" * 40, head):
            with self.subTest(source_head=source_head):
                self.assertEqual(
                    self.tier(event="pull_request", pr_head=source_head), "full"
                )

    def test_wrong_pr_second_parent_stays_full(self) -> None:
        self.readme()
        self.git("checkout", "-q", "target")
        self.git("merge", "--no-ff", "-qm", "PR test merge", "queue")
        self.assertEqual(self.tier(event="pull_request", pr_head=self.base), "full")

    def test_push_schedule_manual_and_unknown_events_stay_full(self) -> None:
        self.readme()
        for event in (
            "push",
            "schedule",
            "workflow_dispatch",
            "pull_request_target",
            "unknown",
            "",
        ):
            with self.subTest(event=event):
                self.assertEqual(self.tier(event=event), "full")

    def test_missing_invalid_nonancestor_and_mismatched_refs_stay_full(self) -> None:
        head = self.readme()
        for base in ("", "bad", "0" * 40, "f" * 40):
            self.assertEqual(self.tier(base=base), "full")
        for candidate in ("", "bad", "f" * 40, self.base):
            self.assertEqual(self.tier(head=candidate), "full")
        tree = self.git("rev-parse", "HEAD^{tree}")
        unrelated = self.git("commit-tree", tree, "-m", "unrelated")
        self.assertEqual(self.tier(base=unrelated, head=head), "full")

    def test_empty_range_stays_full(self) -> None:
        self.assertEqual(self.tier(), "full")

    def test_invalid_missing_and_injection_shaped_target_stay_full(self) -> None:
        self.readme()
        for target in (
            "",
            "master",
            "--upload-pack=bad",
            "refs/tags/target",
            "refs/heads/../target",
            "refs/heads/target:other",
            "refs/heads/target\nother",
            "refs/heads/*",
            "refs/heads/missing",
            "refs/heads/$(touch injected)",
        ):
            with self.subTest(target=target):
                self.assertEqual(self.tier(target=target), "full")
                self.assertFalse((self.root / "injected").exists())

    def test_failed_fetch_cannot_reuse_stale_target(self) -> None:
        self.readme()
        self.assertEqual(self.tier(), "prose")
        self.git("remote", "set-url", "origin", str(self.root / "missing-origin"))
        self.assertEqual(self.tier(), "full")

    def test_divergent_target_cannot_hide_reintroduced_source(self) -> None:
        self.write("src/core.cpp", b"int changed;\n")
        prefix = self.commit()
        self.readme()
        self.git("checkout", "-q", "target")
        self.git("merge", "--no-ff", "-s", "ours", "-qm", "reject source", prefix)
        self.git("checkout", "-q", "queue")
        self.assertEqual(self.tier(base=prefix), "full")

    def test_target_at_head_or_unrelated_tip_stays_full(self) -> None:
        head = self.readme()
        self.git("update-ref", "refs/heads/target", head)
        self.assertEqual(self.tier(), "full")
        tree = self.git("rev-parse", "HEAD^{tree}")
        unrelated = self.git("commit-tree", tree, "-m", "unrelated")
        self.git("update-ref", "refs/heads/target", unrelated)
        self.assertEqual(self.tier(), "full")

    def test_multiple_merge_bases_stay_full(self) -> None:
        first = self.readme()
        self.git("checkout", "-q", "target")
        self.write("docs/side.md", b"side\n")
        second = self.commit()
        tree = self.git("rev-parse", "HEAD^{tree}")
        target = self.git(
            "commit-tree", tree, "-p", first, "-p", second, "-m", "target"
        )
        head = self.git("commit-tree", tree, "-p", second, "-p", first, "-m", "queue")
        self.git("update-ref", "refs/heads/target", target)
        self.git("checkout", "--detach", "-q", head)
        self.assertEqual(self.tier(), "full")

    def test_git_error_timeout_and_corrupt_output_stay_full(self) -> None:
        for error in (
            OSError("missing git"),
            subprocess.CalledProcessError(1, "git"),
            subprocess.TimeoutExpired("git", 60),
            UnicodeDecodeError("utf8", b"\xff", 0, 1, "bad"),
        ):
            with patch.object(scope, "_git", side_effect=error):
                self.assertEqual(
                    scope.select_scope(
                        "merge_group", "a" * 40, "b" * 40, "refs/heads/target"
                    ),
                    "full",
                )
        with patch.object(scope, "_git", return_value=b"malformed"):
            self.assertEqual(scope._readme_change("a" * 40, "b" * 40), False)


if __name__ == "__main__":
    unittest.main()
