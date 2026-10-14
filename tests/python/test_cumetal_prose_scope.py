"""Execute CuMetal README selection, full-tier gates and report provenance."""

from __future__ import annotations

import ast
import itertools
import json
import os
import re
import runpy
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/cumetal-cuda.yml"
SELECTOR = "Select README-only CuMetal scope"


def job(name: str) -> str:
    return re.split(
        r"\n  [\w-]+:\n", WORKFLOW.read_text().split(f"\n  {name}:\n", 1)[1]
    )[0]


def steps() -> dict[str, str]:
    result = {}
    for part in re.split(r"(?=^      - )", job("cuda-tests"), flags=re.MULTILINE):
        match = re.match(r"      - (?:name|uses): (.+)", part)
        if match:
            result[match[1]] = part
    return result


def condition(body: str, indent: int = 8) -> str:
    matches = list(re.finditer(r"^" + " " * indent + r"if: (.+)$", body, re.MULTILINE))
    assert len(matches) <= 1, "duplicate condition"
    if not matches:
        return "success()"
    match = matches[0]
    if match[1] != ">-":
        return match[1]
    lines = body[match.end() :].strip("\n").splitlines()
    continuation = list(
        itertools.takewhile(lambda line: line.startswith(" " * (indent + 2)), lines)
    )
    return " ".join(line.strip() for line in continuation)


def expression(value: str, context: dict[str, str | bool | int]) -> str | bool | int:
    value = value.removeprefix("${{ ").removesuffix(" }}")
    value = re.sub(
        r"\b(?:steps|needs|github)\.[\w.-]+",
        lambda match: repr(context.get(match[0], "")),
        value,
    )
    for name in ("always", "success", "failure", "cancelled"):
        value = value.replace(
            f"{name}()", repr(context.get(f"{name}()", name in {"always", "success"}))
        )
    value = value.replace("&&", " and ").replace("||", " or ")
    value = re.sub(r"!(?!=)", " not ", value).strip()

    def evaluate(node: ast.AST) -> str | bool | int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bool, int)):
            return node.value
        if isinstance(node, ast.BoolOp):
            values = [evaluate(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return next((v for v in values if not v), values[-1])
            return next((v for v in values if v), values[-1])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not evaluate(node.operand)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            equal = evaluate(node.left) == evaluate(node.comparators[0])
            if isinstance(node.ops[0], ast.Eq):
                return equal
            if isinstance(node.ops[0], ast.NotEq):
                return not equal
        raise AssertionError(ast.dump(node))

    return evaluate(ast.parse(value, mode="eval").body)


def selected(body: str, context: dict[str, str | bool | int], indent: int = 8) -> bool:
    value = condition(body, indent)
    if not re.search(
        r"\b(?:always|success|failure|cancelled)\(\)", value
    ) and not context.get("success()", True):
        return False
    return bool(expression(value, context))


def selector_source() -> str:
    shell = textwrap.dedent(steps()[SELECTOR].split("        run: |\n", 1)[1])
    return shell.split("python3 - <<'PYTHON'\n", 1)[1].rsplit("PYTHON", 1)[0]


class CuMetalWorkflowTests(unittest.TestCase):
    def test_every_existing_step_fails_full_and_prose_invokes_no_absent_tools(
        self,
    ) -> None:
        for name, body in steps().items():
            if name == SELECTOR or name.startswith("actions/checkout@"):
                continue
            for event, status, cache_hit in itertools.product(
                (
                    "pull_request",
                    "merge_group",
                    "schedule",
                    "workflow_dispatch",
                    "push",
                    "unknown",
                ),
                ("success", "failure", "cancelled"),
                ("true", "false", ""),
            ):
                context = {
                    "github.event_name": event,
                    "success()": status == "success",
                    "failure()": status == "failure",
                    "cancelled()": status == "cancelled",
                }
                for key in re.findall(r"\bsteps\.[\w.-]+", body):
                    context[key] = "success" if key.endswith(".outcome") else cache_hit
                context.update(
                    {
                        "steps.prose_scope.outcome": "success",
                        "steps.prose_scope.outputs.scope": "full",
                    }
                )
                expected = selected(body, context)
                for outcome, scope in (
                    ("success", ""),
                    ("success", "unknown"),
                    ("failure", "prose"),
                    ("cancelled", "prose"),
                    ("skipped", "prose"),
                    ("", "prose"),
                ):
                    with self.subTest(
                        step=name,
                        event=event,
                        status=status,
                        outcome=outcome,
                        scope=scope,
                    ):
                        context.update(
                            {
                                "steps.prose_scope.outcome": outcome,
                                "steps.prose_scope.outputs.scope": scope,
                            }
                        )
                        self.assertEqual(selected(body, context), expected)
                context.update(
                    {
                        "steps.prose_scope.outcome": "success",
                        "steps.prose_scope.outputs.scope": "prose",
                    }
                )
                self.assertFalse(selected(body, context), name)

    def test_full_scope_keeps_build_ctest_and_both_scientific_tiers(self) -> None:
        work = steps()
        for event in (
            "pull_request",
            "merge_group",
            "schedule",
            "workflow_dispatch",
            "push",
            "unknown",
        ):
            context = {
                "github.event_name": event,
                "steps.prose_scope.outputs.scope": "full",
            }
            for name in (
                "Configure GenerativeQC with CUDA tests enabled",
                "Build CUDA runtime test target",
                "Run CUDA runtime CTest on the Apple GPU",
            ):
                self.assertTrue(selected(work[name], context), (name, event))
            self.assertEqual(
                selected(work["Run bounded CuMetal QC endpoint gate"], context),
                event not in {"schedule", "workflow_dispatch"},
            )
            self.assertEqual(
                selected(
                    work["Run bounded CuMetal QC qualification on the Apple GPU"],
                    context,
                ),
                event in {"schedule", "workflow_dispatch"},
            )
            context.update({"success()": False, "failure()": True})
            self.assertTrue(selected(work["Preserve CUDA test results"], context))
            self.assertTrue(
                selected(work["Preserve CuMetal QC endpoint diagnostics"], context)
            )

    def test_single_producer_upload_needs_positive_successful_prose(self) -> None:
        producer = job("cuda-tests").split("    steps:\n", 1)[0]
        output = re.search(r"^      test_scope: (.+)$", producer, re.MULTILINE)[1]
        uploader = job("upload-test-results")
        for outcome, scope, result, cancelled, unrelated in itertools.product(
            ("success", "failure", "cancelled", "skipped", ""),
            ("prose", "full", "", "unknown"),
            ("success", "failure", "cancelled", "skipped", ""),
            (False, True),
            ("prose", "full", ""),
        ):
            marker = expression(
                output,
                {
                    "steps.prose_scope.outcome": outcome,
                    "steps.prose_scope.outputs.scope": scope,
                },
            )
            proven = outcome == "success" and scope == "prose"
            self.assertEqual(marker, "prose" if proven else "full")
            admitted = selected(
                uploader.split("    steps:\n", 1)[0],
                {
                    "needs.cuda-tests.outputs.test_scope": marker,
                    "needs.cuda-tests.result": result,
                    "needs.cpu.outputs.test_scope": unrelated,
                    "needs.python.outputs.test_scope": unrelated,
                    "cancelled()": cancelled,
                },
                indent=4,
            )
            self.assertEqual(
                admitted,
                not cancelled
                and result in {"success", "failure"}
                and not (result == "success" and proven),
            )
        # Absent producer output is also full, irrespective of other workflows.
        self.assertTrue(
            selected(
                uploader.split("    steps:\n", 1)[0],
                {"needs.cuda-tests.result": "success"},
                indent=4,
            )
        )
        self.assertNotIn("matrix:", producer)
        self.assertIn("use_oidc: true", uploader)
        self.assertNotIn("touch ", WORKFLOW.read_text())

    def test_one_shot_diagnostics_preserve_every_original_authorization_predicate(
        self,
    ) -> None:
        cases = (
            (
                "Compare exact retained CuMetal PTX",
                "48e7afd1cb8e52f0abb85af08727e7684b969631",
                False,
                False,
            ),
            (
                "Diagnose one CuMetal RHF timeout",
                "de9eebb88be4d7ded840a35541daff2ce43a7a44",
                True,
                True,
            ),
            (
                "Capture one CuMetal PTX compiler diagnostic",
                "0f67b413da60fa3d146eead32360901adeda4698",
                True,
                False,
            ),
        )
        for title, before, failed, timeout in cases:
            context = {
                "github.event_name": "pull_request",
                "github.event.pull_request.number": 1997,
                "github.event.action": "synchronize",
                "github.event.before": before,
                "github.run_attempt": 1,
                "steps.qc_gate.outcome": "failure",
                "steps.qc_gate.outputs.rhf_timed_out": "true",
                "steps.prose_scope.outcome": "success",
                "steps.prose_scope.outputs.scope": "full",
                "success()": not failed,
                "failure()": failed,
            }
            body = steps()[title]
            self.assertTrue(selected(body, context), title)
            negatives = {
                "github.event_name": "merge_group",
                "github.event.pull_request.number": 2465,
                "github.event.action": "opened",
                "github.event.before": "0" * 40,
                "github.run_attempt": 2,
                "steps.prose_scope.outputs.scope": "prose",
            }
            if failed:
                negatives.update(
                    {"failure()": False, "steps.qc_gate.outcome": "success"}
                )
            if timeout:
                negatives["steps.qc_gate.outputs.rhf_timed_out"] = "false"
            for key, value in negatives.items():
                self.assertFalse(selected(body, {**context, key: value}), (title, key))

    def test_cache_saves_and_diagnostic_artifacts_keep_independent_admission(
        self,
    ) -> None:
        work = steps()
        for event, hit, gate, qualification, failed in itertools.product(
            ("pull_request", "merge_group", "schedule", "workflow_dispatch"),
            ("true", "false", ""),
            ("success", "failure", "skipped", ""),
            ("success", "failure", "skipped", ""),
            (False, True),
        ):
            context = {
                "github.event_name": event,
                "success()": not failed,
                "failure()": failed,
                "steps.prose_scope.outputs.scope": "full",
                "steps.cache-cumetal.outputs.cache-hit": hit,
                "steps.cumetal_ccache.outputs.cache-hit": hit,
                "steps.generativeqc_ccache.outputs.cache-hit": hit,
                "steps.cumetal_qc_jit_cache.outputs.cache-hit": hit,
                "steps.qc_gate.outcome": gate,
                "steps.qc_qualification.outcome": qualification,
            }
            for title in (
                "Save CuMetal compiler cache",
                "Save CuMetal toolchain",
                "Save GenerativeQC ccache",
            ):
                self.assertEqual(
                    selected(work[title], context),
                    not failed and event != "merge_group" and hit != "true",
                )
            self.assertEqual(
                selected(work["Save CuMetal QC JIT cache"], context),
                event != "merge_group"
                and hit != "true"
                and "success" in {gate, qualification},
            )
        for title, step in (
            ("Preserve exact-input CuMetal compiler comparison", "retained_ptx"),
            ("Preserve one-shot CuMetal timeout diagnostic", "qc_timeout_diagnostic"),
            ("Preserve one-shot CuMetal compiler diagnostic", "qc_ptx_diagnostic"),
        ):
            for outcome, status in itertools.product(
                ("success", "failure", "cancelled", "skipped", ""),
                ("success", "failure", "cancelled"),
            ):
                context = {
                    f"steps.{step}.outcome": outcome,
                    "steps.prose_scope.outputs.scope": "full",
                    "success()": status == "success",
                    "failure()": status == "failure",
                    "cancelled()": status == "cancelled",
                }
                self.assertEqual(
                    selected(work[title], context), outcome not in {"skipped", ""}
                )

    def test_duplicate_conditions_and_guard_removal_are_rejected(self) -> None:
        build = steps()["Build CUDA runtime test target"]
        with self.assertRaises(AssertionError):
            selected(build + "        if: false\n", {})
        broken = re.sub(r"^        if: .+\n", "", build, flags=re.MULTILINE)
        context = {
            "steps.prose_scope.outcome": "success",
            "steps.prose_scope.outputs.scope": "prose",
        }
        self.assertFalse(selected(build, context))
        self.assertTrue(selected(broken, context))


class CuMetalSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.env = {
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_NAME": "Fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        }
        self.git(self.source, "init", "-q", "-b", "target")
        (self.source / "README.md").write_text("# Project\nOriginal prose\n")
        (self.source / "src").mkdir()
        (self.source / "src/core.cpp").write_text("int original;\n")
        (self.source / "tools").mkdir()
        for name in ("ci_prose_scope.py", "native_cli_ci_scope.py"):
            (self.source / "tools" / name).write_bytes(
                (ROOT / "tools" / name).read_bytes()
            )
        self.base = self.commit("base")

    def git(self, cwd: Path, *args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            env=self.env,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()

    def commit(self, title: str) -> str:
        self.git(self.source, "add", "-A")
        self.git(self.source, "commit", "-qm", title)
        return self.git(self.source, "rev-parse", "HEAD")

    def pr(self, path: str = "README.md") -> tuple[str, str]:
        self.git(self.source, "checkout", "-qb", "topic")
        destination = self.source / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("Changed contents\n")
        branch = self.commit("topic")
        self.git(self.source, "checkout", "target")
        self.git(self.source, "merge", "--no-ff", "-qm", "test merge", "topic")
        return self.git(self.source, "rev-parse", "HEAD"), branch

    def checkout(self, head: str) -> Path:
        root = self.root / f"checkout-{len(list(self.root.glob('checkout-*')))}"
        root.mkdir()
        self.git(root, "init", "-q")
        self.git(root, "remote", "add", "origin", self.source.as_uri())
        self.git(
            root,
            "fetch",
            "--depth=1",
            "--no-tags",
            "origin",
            "+" + head + ":refs/ci/event",
        )
        self.git(root, "checkout", "-q", "--detach", head)
        return root

    def run_scope(
        self, checkout: Path, head: str, pr_head: str = "", **overrides: str
    ) -> str:
        output = self.root / "output"
        output.write_text("")
        env = {
            **self.env,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_SHA": head,
            "PROSE_BASE_SHA": self.base,
            "PROSE_PR_HEAD_SHA": pr_head,
            "PROSE_TARGET_REF": "refs/heads/target",
            "PROSE_PR_CHANGED_FILES": "1",
            **overrides,
        }
        subprocess.run(
            [sys.executable, "-c", selector_source()],
            cwd=checkout,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return output.read_text().strip()

    def test_real_shallow_readme_pr_reuses_classifier_without_changing_tree(
        self,
    ) -> None:
        head, pr_head = self.pr()
        root = self.checkout(head)
        before = self.git(root, "ls-tree", "-r", "HEAD")
        self.assertEqual(self.run_scope(root, head, pr_head), "scope=prose")
        self.assertEqual(self.git(root, "rev-parse", "HEAD"), head)
        self.assertEqual(self.git(root, "ls-tree", "-r", "HEAD"), before)
        self.assertEqual(self.git(root, "rev-list", "--count", "HEAD"), "3")

    def test_count_is_only_a_no_fetch_rejection(self) -> None:
        head, pr_head = self.pr()
        root = self.checkout(head)
        self.git(root, "remote", "set-url", "origin", str(self.root / "missing"))
        for count in ("0", "2", "", "unknown", "01", "-1", "$(touch sentinel)"):
            self.assertEqual(
                self.run_scope(root, head, pr_head, PROSE_PR_CHANGED_FILES=count),
                "scope=full",
            )
        self.assertEqual(self.git(root, "rev-list", "--count", "HEAD"), "1")
        with self.assertRaises(subprocess.CalledProcessError):
            self.git(root, "show-ref", "--verify", "refs/ci/prose-event")

    def test_prefilter_rejections_launch_no_subprocess_at_all(self) -> None:
        script = self.root / "selector.py"
        script.write_text(selector_source())
        output = self.root / "output"
        for event, count in itertools.product(
            ("pull_request", "schedule", "workflow_dispatch", "push", "unknown"),
            ("0", "2", "", "01", "$(touch sentinel)"),
        ):
            env = {
                "GITHUB_OUTPUT": str(output),
                "GITHUB_EVENT_NAME": event,
                "GITHUB_SHA": "a" * 40,
                "PROSE_PR_CHANGED_FILES": count,
            }
            with (
                patch.dict(os.environ, env, clear=True),
                patch("subprocess.Popen") as popen,
            ):
                runpy.run_path(str(script))
                popen.assert_not_called()
        self.assertFalse((self.root / "sentinel").exists())

    def test_non_pr_and_unknown_events_do_not_fetch(self) -> None:
        head, pr_head = self.pr()
        root = self.checkout(head)
        for event in ("push", "schedule", "workflow_dispatch", "unknown", ""):
            self.assertEqual(
                self.run_scope(root, head, pr_head, GITHUB_EVENT_NAME=event),
                "scope=full",
            )
        self.assertEqual(self.git(root, "rev-list", "--count", "HEAD"), "1")

    def test_source_cannot_borrow_count_one(self) -> None:
        head, pr_head = self.pr("src/core.cpp")
        self.assertEqual(
            self.run_scope(self.checkout(head), head, pr_head), "scope=full"
        )

    def test_failed_fetch_or_missing_refs_stay_full(self) -> None:
        head, pr_head = self.pr()
        root = self.checkout(head)
        for values in (
            {"PROSE_BASE_SHA": ""},
            {"PROSE_PR_HEAD_SHA": ""},
            {"GITHUB_SHA": "bad"},
            {"GITHUB_SHA": "0" * 40},
        ):
            self.assertEqual(
                self.run_scope(root, head, pr_head, **values), "scope=full"
            )
        self.git(root, "remote", "set-url", "origin", str(self.root / "missing"))
        self.assertEqual(self.run_scope(root, head, pr_head), "scope=full")

    def test_stacked_queue_never_approves_only_the_readme_tail(self) -> None:
        self.git(self.source, "checkout", "-qb", "queue")
        (self.source / "src/core.cpp").write_text("int earlier_queued_code;\n")
        predecessor = self.commit("first queued source")
        (self.source / "README.md").write_text("# Project\nLast queued prose\n")
        head = self.commit("last queued README")
        root = self.checkout(head)
        self.assertEqual(
            self.run_scope(
                root, head, GITHUB_EVENT_NAME="merge_group", PROSE_BASE_SHA=predecessor
            ),
            "scope=full",
        )
        # Depth two is insufficient proof of the complete three-commit queue.
        self.assertEqual(
            self.git(root, "diff", "--name-only", predecessor, head), "README.md"
        )

    def test_single_queue_readme_and_diverged_target(self) -> None:
        self.git(self.source, "checkout", "-qb", "queue")
        (self.source / "README.md").write_text("# Project\nQueued prose\n")
        head = self.commit("README")
        root = self.checkout(head)
        self.assertEqual(
            self.run_scope(root, head, GITHUB_EVENT_NAME="merge_group"), "scope=prose"
        )
        self.git(self.source, "checkout", "target")
        (self.source / "src/core.cpp").write_text("int advanced_target;\n")
        self.commit("target advanced independently")
        self.assertEqual(
            self.run_scope(root, head, GITHUB_EVENT_NAME="merge_group"), "scope=full"
        )

    def test_moving_remote_refs_cannot_change_the_recorded_event_tree(self) -> None:
        head, pr_head = self.pr()
        root = self.checkout(head)
        (self.source / "src/core.cpp").write_text("int later_remote_source;\n")
        self.commit("remote target moved")
        self.assertEqual(self.run_scope(root, head, pr_head), "scope=prose")
        self.assertEqual(self.git(root, "rev-parse", "HEAD"), head)

    def test_stacked_workflow_tail_and_landed_prefix_use_trusted_target(self) -> None:
        self.git(self.source, "checkout", "-qb", "queue")
        (self.source / ".github/workflows").mkdir(parents=True)
        (self.source / ".github/workflows/ci.yml").write_text("changed workflow\n")
        predecessor = self.commit("earlier queued workflow")
        (self.source / "README.md").write_text("# Project\nLast queued README\n")
        head = self.commit("README tail")
        root = self.checkout(head)
        self.assertEqual(
            self.run_scope(
                root, head, GITHUB_EVENT_NAME="merge_group", PROSE_BASE_SHA=predecessor
            ),
            "scope=full",
        )
        # Once that prefix is in trusted master, only the README remains unmerged.
        self.git(self.source, "update-ref", "refs/heads/target", predecessor)
        self.assertEqual(
            self.run_scope(
                root, head, GITHUB_EVENT_NAME="merge_group", PROSE_BASE_SHA=predecessor
            ),
            "scope=prose",
        )
        # An old successful target ref may never substitute for a failed new fetch.
        self.assertEqual(
            self.run_scope(
                root,
                head,
                GITHUB_EVENT_NAME="merge_group",
                PROSE_BASE_SHA=predecessor,
                PROSE_TARGET_REF="refs/heads/missing",
            ),
            "scope=full",
        )

    def test_two_parent_synthetic_queue_requires_its_complete_unmerged_range(
        self,
    ) -> None:
        self.git(self.source, "checkout", "-qb", "source-feature")
        (self.source / "src/core.cpp").write_text("int queued_source;\n")
        self.commit("source feature")
        self.git(self.source, "checkout", "-qb", "queue", self.base)
        self.git(
            self.source,
            "merge",
            "--no-ff",
            "-qm",
            "first queue merge",
            "source-feature",
        )
        predecessor = self.git(self.source, "rev-parse", "HEAD")
        self.git(self.source, "checkout", "-qb", "readme-feature", self.base)
        (self.source / "README.md").write_text("# Project\nQueued README\n")
        self.commit("README feature")
        self.git(self.source, "checkout", "queue")
        self.git(
            self.source,
            "merge",
            "--no-ff",
            "-qm",
            "second queue merge",
            "readme-feature",
        )
        head = self.git(self.source, "rev-parse", "HEAD")
        root = self.checkout(head)
        self.assertEqual(
            self.run_scope(
                root, head, GITHUB_EVENT_NAME="merge_group", PROSE_BASE_SHA=predecessor
            ),
            "scope=full",
        )
        self.assertEqual(
            self.git(root, "diff", "--name-only", predecessor, head), "README.md"
        )
        self.assertEqual(
            len(self.git(root, "rev-list", "--parents", "-n", "1", head).split()), 3
        )
        self.git(self.source, "update-ref", "refs/heads/target", predecessor)
        self.assertEqual(
            self.run_scope(
                root, head, GITHUB_EVENT_NAME="merge_group", PROSE_BASE_SHA=predecessor
            ),
            "scope=prose",
        )

    def test_readme_stack_with_insufficient_ancestry_stays_full(self) -> None:
        self.git(self.source, "checkout", "-qb", "queue")
        (self.source / "README.md").write_text("# First paragraph\n")
        predecessor = self.commit("first README")
        (self.source / "README.md").write_text("# Second paragraph\n")
        head = self.commit("second README")
        self.assertEqual(
            self.run_scope(
                self.checkout(head),
                head,
                GITHUB_EVENT_NAME="merge_group",
                PROSE_BASE_SHA=predecessor,
            ),
            "scope=full",
        )

    def test_classifier_outputs_require_success_and_exact_marker(self) -> None:
        script = self.root / "selector.py"
        script.write_text(selector_source())
        output = self.root / "output"
        env = {
            "GITHUB_OUTPUT": str(output),
            "GITHUB_EVENT_NAME": "merge_group",
            "GITHUB_SHA": "a" * 40,
        }
        for text, code, expected in (
            ("prose\n", 0, "prose"),
            ("full\n", 0, "full"),
            ("", 0, "full"),
            ("unknown", 0, "full"),
            ("prose\nextra", 0, "full"),
            (" prose\n", 0, "full"),
            ("prose\n\n", 0, "full"),
            ("prose", 0, "full"),
            ("prose", 2, "full"),
        ):
            output.write_text("")
            with (
                patch.dict(os.environ, env, clear=True),
                patch("subprocess.Popen") as popen,
            ):
                fetch, classify = MagicMock(), MagicMock()
                fetch.returncode = 0
                fetch.communicate.return_value = ("", "")
                classify.returncode = code
                classify.communicate.return_value = (text, "")
                popen.side_effect = [fetch, classify]
                runpy.run_path(str(script))
                self.assertLessEqual(
                    classify.communicate.call_args.kwargs["timeout"],
                    fetch.communicate.call_args.kwargs["timeout"],
                )
            self.assertEqual(output.read_text(), f"scope={expected}\n")
        output.write_text("")
        with (
            patch.dict(os.environ, env, clear=True),
            patch("subprocess.Popen") as popen,
        ):
            popen.return_value.communicate.side_effect = UnicodeDecodeError(
                "utf-8", b"\xff", 0, 1, "invalid"
            )
            runpy.run_path(str(script))
        self.assertEqual(output.read_text(), "scope=full\n")

    def test_real_timeout_reaps_fetch_and_its_inherited_pipe_child(self) -> None:
        output = self.root / "output"
        pid_file = self.root / "owned-pids.json"
        binary = self.root / "bin"
        binary.mkdir()
        fake_git = binary / "git"
        fake_git.write_text(
            f"#!{sys.executable}\nimport json,os,subprocess,sys,time\n"
            "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "open(os.environ['OWNED_PIDS'], 'w').write(json.dumps([os.getpid(),child.pid]))\n"
            "time.sleep(30)\n"
        )
        fake_git.chmod(0o755)
        source = selector_source()
        self.assertIn("deadline = time.monotonic() + 75", source)
        source = source.replace(
            "deadline = time.monotonic() + 75", "deadline = time.monotonic() + 1"
        )
        env = {
            **self.env,
            "PATH": str(binary) + os.pathsep + os.environ.get("PATH", ""),
            "OWNED_PIDS": str(pid_file),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_EVENT_NAME": "merge_group",
            "GITHUB_SHA": "a" * 40,
        }
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            started = time.monotonic()
            subprocess.run(
                [sys.executable, "-c", source],
                env=env,
                cwd=self.root,
                capture_output=True,
                check=True,
                timeout=8,
            )
            self.assertLess(time.monotonic() - started, 6)
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)
        self.assertEqual(output.read_text(), "scope=full\n")
        for pid in json.loads(pid_file.read_text()):
            # A terminated orphan may briefly be a zombie until the OS reaps it.
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "stat="],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            self.assertTrue(
                result.returncode != 0 or result.stdout.strip().startswith("Z"),
                result.stdout,
            )

    def test_subprocess_timeout_kills_only_the_owned_group_and_falls_full(self) -> None:
        output = self.root / "output"
        env = {
            "GITHUB_OUTPUT": str(output),
            "GITHUB_EVENT_NAME": "merge_group",
            "GITHUB_SHA": "a" * 40,
        }
        script = self.root / "selector.py"
        script.write_text(selector_source())
        with (
            patch.dict(os.environ, env, clear=True),
            patch("subprocess.Popen") as popen,
            patch("os.killpg") as kill,
        ):
            process = popen.return_value
            process.pid = 34567
            process.communicate.side_effect = [
                subprocess.TimeoutExpired("git", 75),
                ("", ""),
            ]
            runpy.run_path(str(script))
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            kill.assert_called_once_with(34567, signal.SIGKILL)
            self.assertEqual(process.communicate.call_count, 2)
        self.assertEqual(output.read_text(), "scope=full\n")


if __name__ == "__main__":
    unittest.main()
