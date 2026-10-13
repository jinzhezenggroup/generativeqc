"""Portable PR build-tier regressions for the monolithic CUDA CLI."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
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

    def test_master_only_cuda_cache_seed_does_not_duplicate_pr_or_queue_builds(
        self,
    ) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        cli_job = workflow.split("\n  native-cli-artifact:\n", 1)[1]
        dependencies = cli_job.split("\n    steps:", 1)[0]
        self.assertIn("needs: cuda-compile", dependencies)
        self.assertIn("needs.cuda-compile.result == 'success'", dependencies)

        def step_condition(name: str) -> str:
            step = cli_job.split(f"\n      - name: {name}\n", 1)[1].split(
                "\n      - name:", 1
            )[0]
            return next(
                line.strip()
                for line in step.splitlines()
                if line.strip().startswith("if:")
            )

        self.assertEqual(
            step_condition("Seed fast CUDA CI objects in the shared cache"),
            "if: github.event_name == 'push' && github.ref == 'refs/heads/master' "
            "&& steps.cli_scope.outputs.tier != 'skip'",
        )
        self.assertIn(
            "steps.fast_cuda_seed.outputs.cache_stats_reset == 'true'",
            step_condition("Report fast CUDA seed compiler cache"),
        )
        self.assertEqual(
            step_condition("Build monolithic C++ CUDA CLI (release or fast PR)"),
            "if: steps.cli_scope.outputs.tier != 'skip'",
        )
        self.assertIn(
            "github.event_name == 'push' && github.ref == 'refs/heads/master'",
            step_condition("Save complete shared CUDA ccache on master"),
        )

    def test_cuda_cache_age_prune_and_capacity_before_master_publication(
        self,
    ) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        cuda_compile = workflow.split("\n  cuda-compile:\n", 1)[1].split(
            "\n  master-cache-cleanup:\n", 1
        )[0]
        cli_job = workflow.split("\n  native-cli-artifact:\n", 1)[1]
        # Both consumer and publisher must have matching capacity; the
        # repository-wide Actions cache quota is a separate bound.
        for job in (cuda_compile, cli_job):
            self.assertIn("CCACHE_MAXSIZE: 5G", job)
            self.assertNotIn("CCACHE_MAXSIZE: 3G", job)

        name = "Retire unused CUDA objects before publishing the cache"
        maintenance = cli_job.split(f"\n      - name: {name}\n", 1)[1].split(
            "\n      - name:", 1
        )[0]
        self.assertIn(
            "always() && github.event_name == 'push' && "
            "github.ref == 'refs/heads/master'",
            maintenance,
        )
        self.assertIn("steps.fast_cuda_seed.outcome != 'skipped'", maintenance)
        self.assertIn("ccache --evict-older-than 7d", maintenance)
        self.assertIn("ccache --cleanup", maintenance)
        self.assertNotIn("ccache --clear", maintenance)
        # A failing CLI can still publish successful partial compiler work.
        # Both immutable cache saves and the trusted artifact mirror receive
        # the same pruned snapshot, never a pre-pruning directory.
        sequence = (
            name,
            "Save complete shared CUDA ccache on master",
            "Preserve partial master CUDA compiler work",
            "Publish trusted master CUDA ccache snapshot",
        )
        positions = [cli_job.index(f"\n      - name: {step}\n") for step in sequence]
        self.assertEqual(positions, sorted(positions))

    def test_git_diff_paths_keep_spaces_and_detect_deletions(self) -> None:
        paths = b"docs/user/read me.md\0src/xtb/native/deleted.cpp\0"
        with patch.object(
            scope.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, stdout=paths, stderr=b""),
        ) as command:
            self.assertEqual(scope.select_tier("pull_request", BASE, HEAD), "fast")
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
            git("branch", "target", base)
            git("remote", "add", "origin", str(root))
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
                git("checkout", "--detach", "-q", head)
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
                                "--target-ref=refs/heads/target",
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
            git("branch", "target", base)
            git("remote", "add", "origin", str(root))
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
                        "--target-ref",
                        "refs/heads/target",
                    ],
                    cwd=root,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.strip(), "skip")


class MergeQueueScopeTests(unittest.TestCase):
    """Exercise real stacked Git history and the exact workflow entry point."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.git("init", "-q", "-b", "target")
        self.base = self.commit("CMakeLists.txt", "initial configuration\n")
        self.git("remote", "add", "origin", str(self.root))
        self.git("checkout", "-qb", "queue")

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=fixture",
                "-c",
                "user.email=fixture@example.com",
                *args,
            ],
            text=True,
            timeout=10,
        ).strip()

    def commit(self, path: str, contents: str) -> str:
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents)
        self.git("add", ".")
        self.git("commit", "-qm", path)
        return self.git("rev-parse", "HEAD")

    def tier(
        self,
        base: str | None = None,
        head: str | None = None,
        target: str = "refs/heads/target",
        event: str = "merge_group",
    ) -> str:
        return subprocess.check_output(
            [
                sys.executable,
                str(MODULE),
                "--event",
                event,
                "--base-sha",
                base or self.base,
                "--head-sha",
                head or self.git("rev-parse", "HEAD"),
                f"--target-ref={target}",
            ],
            cwd=self.root,
            text=True,
            timeout=15,
        ).strip()

    def test_earlier_release_survives_latest_fast_or_docs_queue_node(self) -> None:
        self.git("checkout", "-qb", "release-pr", self.base)
        self.commit("CMakeLists.txt", "changed configuration\n")
        self.git("checkout", "-q", "queue")
        self.git("merge", "--no-ff", "-qm", "first queue node", "release-pr")
        prefix = self.git("rev-parse", "HEAD")
        self.git("checkout", "-qb", "fast-pr", self.base)
        self.commit("src/latest.cpp", "int value = 1;\n")
        self.git("checkout", "-q", "queue")
        self.git("merge", "--no-ff", "-qm", "second queue node", "fast-pr")
        # The former immediate-predecessor comparison loses the release tier.
        self.assertEqual(self.tier(base=prefix, event="pull_request"), "fast")
        self.assertEqual(self.tier(base=prefix), "release")
        prefix = self.git("rev-parse", "HEAD")
        self.git("checkout", "-qb", "docs-pr", self.base)
        self.commit("README.md", "latest docs\n")
        self.git("checkout", "-q", "queue")
        self.git("merge", "--no-ff", "-qm", "third queue node", "docs-pr")
        self.assertEqual(self.tier(base=prefix, event="pull_request"), "skip")
        self.assertEqual(self.tier(base=prefix), "release")

    def test_all_fast_queue_and_latest_docs_keep_fast_tier(self) -> None:
        prefix = self.commit("src/first.cpp", "int first;\n")
        self.commit("include/second.h", "int second;\n")
        self.assertEqual(self.tier(base=prefix), "fast")
        prefix = self.git("rev-parse", "HEAD")
        self.commit("README.md", "latest docs\n")
        self.assertEqual(self.tier(base=prefix), "fast")

    def test_all_docs_queue_can_skip(self) -> None:
        prefix = self.commit("README.md", "first docs\n")
        self.commit("docs/next.md", "latest docs\n")
        self.assertEqual(self.tier(base=prefix), "skip")

    def test_target_advanced_through_queue_prefix_excludes_landed_changes(self) -> None:
        prefix = self.commit("CMakeLists.txt", "landed configuration\n")
        self.commit("src/latest.cpp", "int latest;\n")
        self.assertEqual(self.tier(base=prefix), "release")
        self.git("update-ref", "refs/heads/target", prefix)
        self.assertEqual(self.tier(base=prefix), "fast")

    def test_target_advanced_to_head_or_beyond_never_skips(self) -> None:
        head = self.commit("README.md", "docs\n")
        self.git("update-ref", "refs/heads/target", head)
        self.assertEqual(self.tier(), "release")
        self.git("checkout", "-q", "target")
        self.commit("docs/later.md", "later target\n")
        self.git("checkout", "-q", "queue")
        self.assertEqual(self.tier(), "release")

    def test_divergent_target_cannot_hide_reintroduced_release_change(self) -> None:
        prefix = self.commit("CMakeLists.txt", "queued configuration\n")
        head = self.commit("README.md", "latest docs\n")
        self.git("checkout", "-q", "target")
        self.git("merge", "--no-ff", "-s", "ours", "-qm", "reject config", prefix)
        target = self.git("rev-parse", "HEAD")
        self.assertEqual(self.git("merge-base", target, head), prefix)
        self.git("checkout", "-q", "queue")
        self.assertEqual(self.tier(base=prefix), "release")

    def test_unrelated_target_advance_fails_closed(self) -> None:
        self.commit("README.md", "queue docs\n")
        self.git("checkout", "-q", "target")
        self.commit("docs/target.md", "target docs\n")
        self.git("checkout", "-q", "queue")
        self.assertEqual(self.tier(), "release")

    def test_missing_or_invalid_target_ref_requires_release(self) -> None:
        self.commit("README.md", "queue docs\n")
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
        ):
            with self.subTest(target=target):
                self.assertEqual(self.tier(target=target), "release")

    def test_failed_fetch_never_uses_stale_target_ref(self) -> None:
        self.commit("README.md", "queue docs\n")
        self.assertEqual(self.tier(), "skip")
        self.git("remote", "set-url", "origin", str(self.root / "missing-origin"))
        self.assertEqual(self.tier(), "release")

    def test_target_with_no_common_ancestor_requires_release(self) -> None:
        self.commit("README.md", "queue docs\n")
        tree = self.git("rev-parse", "HEAD^{tree}")
        unrelated = self.git("commit-tree", tree, "-m", "unrelated target")
        self.git("update-ref", "refs/heads/target", unrelated)
        self.assertEqual(self.tier(), "release")

    def test_criss_cross_multiple_merge_bases_require_release(self) -> None:
        first = self.commit("README.md", "first side\n")
        self.git("checkout", "-q", "target")
        second = self.commit("docs/side.md", "second side\n")
        tree = self.git("rev-parse", "HEAD^{tree}")
        target = self.git(
            "commit-tree", tree, "-p", first, "-p", second, "-m", "target"
        )
        head = self.git("commit-tree", tree, "-p", second, "-p", first, "-m", "queue")
        self.git("update-ref", "refs/heads/target", target)
        self.git("checkout", "--detach", "-q", head)
        self.assertEqual(
            len(self.git("merge-base", "--all", target, head).splitlines()), 2
        )
        self.assertEqual(self.tier(), "release")

    def test_invalid_or_mismatched_head_and_nonancestor_event_base_require_release(
        self,
    ) -> None:
        head = self.commit("README.md", "queue docs\n")
        for candidate in ("invalid", "f" * 40, self.base):
            with self.subTest(head=candidate):
                self.assertEqual(self.tier(head=candidate), "release")
        tree = self.git("rev-parse", "HEAD^{tree}")
        unrelated = self.git("commit-tree", tree, "-m", "unrelated event base")
        self.assertEqual(self.tier(base=unrelated, head=head), "release")

    def test_git_timeout_requires_release(self) -> None:
        with patch.object(
            scope.subprocess, "run", side_effect=subprocess.TimeoutExpired("git", 60)
        ):
            self.assertEqual(
                scope.select_tier("merge_group", BASE, HEAD, "refs/heads/target"),
                "release",
            )

    def test_workflow_passes_ref_as_data_and_preserves_classification_output(
        self,
    ) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        step = workflow.split(
            "      - name: Classify native CLI packaging impact\n", 1
        )[1].split("\n      - name:", 1)[0]
        script = textwrap.dedent(step.split("        run: |\n", 1)[1])
        self.assertIn(
            "CLI_TARGET_REF: ${{ github.event.merge_group.base_ref || '' }}", step
        )
        self.assertNotIn("${{", script)
        prefix = self.commit("CMakeLists.txt", "queued config\n")
        head = self.commit("README.md", "latest docs\n")
        (self.root / "tools").mkdir()
        (self.root / "tools/native_cli_ci_scope.py").write_bytes(MODULE.read_bytes())
        for target in (
            "refs/heads/target",
            "refs/heads/$(touch injected)",
            "--upload-pack=bad",
        ):
            with self.subTest(target=target):
                output = self.root / "workflow-output"
                output.write_text("")
                result = subprocess.run(
                    ["bash", "-c", script],
                    cwd=self.root,
                    env={
                        **os.environ,
                        "GITHUB_EVENT_NAME": "merge_group",
                        "CLI_BASE_SHA": prefix,
                        "GITHUB_SHA": head,
                        "CLI_TARGET_REF": target,
                        "GITHUB_OUTPUT": str(output),
                    },
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(output.read_text(), "tier=release\n")
                self.assertFalse((self.root / "injected").exists())


if __name__ == "__main__":
    unittest.main()
