"""Execute README-tier admission and cleanup/coverage workflow contracts."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
JOBS = ("cpu", "cuda-compile", "cuda-resources", "cuda-direct-aot", "python")
CHEAP_CPU = {
    "Audit generated CUDA ABI call coverage",
    "Test native CLI archive/metadata contracts",
    "Generate current CUDA ownership report",
    "Preserve current CUDA ownership report",
}
SELECTOR = "Select README-only CI scope"


def job(name: str, workflow: str | None = None) -> str:
    source = WORKFLOW.read_text() if workflow is None else workflow
    body = source.split(f"\n  {name}:\n", 1)[1]
    return re.split(r"\n  [a-zA-Z0-9_-]+:\n", body, maxsplit=1)[0]


def steps(name: str, workflow: str | None = None) -> dict[str, str]:
    body = job(name, workflow).split("    steps:\n", 1)[1]
    return {
        (re.search(r"^      - (?:name|uses): (.+)$", part, re.MULTILINE))[1]: part
        for part in re.split(r"(?=^      - )", body, flags=re.MULTILINE)
        if part.strip()
    }


def condition(body: str, indent: int = 8) -> str:
    match = re.search(r"^" + " " * indent + r"if: (.+)$", body, re.MULTILINE)
    return match[1] if match else "success()"


def evaluate(node: ast.AST) -> str | bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bool)):
        return node.value
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        result = evaluate(node.values[0])
        for value in node.values[1:]:
            if (isinstance(node.op, ast.And) and result) or (
                isinstance(node.op, ast.Or) and not result
            ):
                result = evaluate(value)
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not evaluate(node.operand)
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
    ):
        equal = evaluate(node.left) == evaluate(node.comparators[0])
        return equal if isinstance(node.ops[0], ast.Eq) else not equal
    raise AssertionError(f"Unsupported workflow expression: {ast.dump(node)}")


def expression(value: str, context: dict[str, str | bool]) -> str | bool:
    value = value.removeprefix("${{ ").removesuffix(" }}")

    def replace(match: re.Match[str]) -> str:
        key = match[0]
        default = "success" if key.endswith(".outcome") else ""
        return repr(context.get(key, default))

    value = re.sub(
        r"\b(?:steps|needs|github|matrix|env)\.[a-zA-Z0-9_.-]+", replace, value
    )
    for name in ("always", "success", "failure", "cancelled"):
        value = value.replace(
            f"{name}()", repr(context.get(f"{name}()", name in {"always", "success"}))
        )
    value = value.replace("&&", " and ").replace("||", " or ")
    value = re.sub(r"!(?!=)", " not ", value).strip()
    return evaluate(ast.parse(value, mode="eval").body)


def selected(body: str, context: dict[str, str | bool]) -> bool:
    value = condition(body)
    # GitHub implicitly adds success() only without a status-check function.
    if not re.search(
        r"\b(?:always|success|failure|cancelled)\(\)", value
    ) and not context.get("success()", True):
        return False
    return bool(expression(value, context))


def instances() -> list[tuple[str, dict[str, str | bool]]]:
    result: list[tuple[str, dict[str, str | bool]]] = [
        ("cpu", {"matrix.compiler": compiler}) for compiler in ("gcc", "clang")
    ]
    result.extend((name, {}) for name in JOBS[1:4])
    shards = re.search(r"shard: \[([^\]]+)\]", job("python"))
    assert shards is not None
    result.extend(
        ("python", {"matrix.shard": value.strip()}) for value in shards[1].split(",")
    )
    return result


class ProseWorkflowTests(unittest.TestCase):
    def test_all_thirteen_instances_keep_cheap_checks_and_skip_only_heavy_steps(
        self,
    ) -> None:
        self.assertEqual(len(instances()), 13)
        for name, matrix in instances():
            for event in ("pull_request", "merge_group"):
                for failed in (False, True):
                    base = {
                        **matrix,
                        "github.event_name": event,
                        "steps.prose_scope.outcome": "success",
                        "steps.prose_scope.outputs.scope": "full",
                        "success()": not failed,
                        "failure()": failed,
                        "cancelled()": False,
                        "always()": True,
                    }
                    prose = {**base, "steps.prose_scope.outputs.scope": "prose"}
                    for title, step in steps(name).items():
                        cheap = (
                            title == SELECTOR
                            or title.startswith("actions/checkout@")
                            or (name == "cpu" and title in CHEAP_CPU)
                        )
                        with self.subTest(
                            job=name,
                            matrix=matrix,
                            event=event,
                            failed=failed,
                            step=title,
                        ):
                            self.assertEqual(
                                selected(step, prose),
                                selected(step, base) if cheap else False,
                            )
                            for outcome, scope in (
                                ("success", ""),
                                ("success", "unknown"),
                                ("failure", "prose"),
                                ("cancelled", "prose"),
                                ("skipped", "prose"),
                                ("", "prose"),
                            ):
                                uncertain = {
                                    **base,
                                    "steps.prose_scope.outcome": outcome,
                                    "steps.prose_scope.outputs.scope": scope,
                                }
                                self.assertEqual(
                                    selected(step, uncertain), selected(step, base)
                                )

    def test_classifier_executes_before_heavy_steps_without_another_runner(
        self,
    ) -> None:
        for name in JOBS:
            with self.subTest(job=name):
                header = job(name).split("    steps:\n", 1)[0]
                self.assertIn("    needs: merge_queue_liveness\n", header)
                sequence = list(steps(name))
                self.assertTrue(sequence[0].startswith("actions/checkout@"))
                self.assertEqual(sequence[1], SELECTOR)
                selector = steps(name)[SELECTOR]
                self.assertIn("        continue-on-error: true\n", selector)
                self.assertIn("        timeout-minutes: 2\n", selector)

    def test_actual_shell_requires_successful_exact_output_and_bounded_event(
        self,
    ) -> None:
        for name in JOBS:
            step = steps(name)[SELECTOR]
            script = textwrap.dedent(step.split("        run: |\n", 1)[1])
            self.assertNotIn("${{", script)
            for event in (
                "pull_request",
                "merge_group",
                "push",
                "schedule",
                "workflow_dispatch",
                "unknown",
            ):
                for output, code in (
                    ("prose\n", 0),
                    ("prose\n", 1),
                    ("prose\n", 124),
                    ("", 0),
                    ("full\n", 0),
                    ("unknown\n", 0),
                    ("prose\nother=prose\n", 0),
                    ("prose\nwarning", 0),
                ):
                    with (
                        self.subTest(job=name, event=event, output=output, code=code),
                        tempfile.TemporaryDirectory() as directory,
                    ):
                        root = Path(directory)
                        (root / "tools").mkdir()
                        (root / "tools/ci_prose_scope.py").write_text(
                            "import json, pathlib, sys\n"
                            "pathlib.Path('args.json').write_text(json.dumps(sys.argv[1:]))\n"
                            f"sys.stdout.write({output!r})\nsys.exit({code})\n"
                        )
                        destination = root / "output"
                        environment = {
                            **os.environ,
                            "GITHUB_EVENT_NAME": event,
                            "PROSE_BASE_SHA": "a" * 40,
                            "GITHUB_SHA": "b" * 40,
                            "PROSE_TARGET_REF": "refs/heads/$(touch injected)",
                            "PROSE_PR_HEAD_SHA": "c" * 40,
                            "GITHUB_OUTPUT": str(destination),
                        }
                        result = subprocess.run(
                            ["bash", "-c", script],
                            cwd=root,
                            env=environment,
                            capture_output=True,
                            text=True,
                            timeout=5,
                            check=False,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        eligible = event in {"pull_request", "merge_group"}
                        expected = (
                            "prose"
                            if eligible and code == 0 and output == "prose\n"
                            else "full"
                        )
                        self.assertEqual(destination.read_text(), f"scope={expected}\n")
                        self.assertFalse((root / "injected").exists())
                        if eligible:
                            args = json.loads((root / "args.json").read_text())
                            self.assertEqual(
                                args,
                                [
                                    "--event",
                                    event,
                                    "--base-sha",
                                    "a" * 40,
                                    "--head-sha",
                                    "b" * 40,
                                    "--target-ref=refs/heads/$(touch injected)",
                                    "--pr-head-sha=" + "c" * 40,
                                ],
                            )
                        else:
                            self.assertFalse((root / "args.json").exists())

    def test_gcc_retains_portable_scope_and_packaging_contract_invocations(
        self,
    ) -> None:
        step = steps("cpu")["Test native CLI archive/metadata contracts"]
        script = textwrap.dedent(step.split("        run: |\n", 1)[1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "python3"
            fake.write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$CALLS"\n')
            fake.chmod(0o755)
            subprocess.run(
                ["bash", "-e", "-c", script],
                cwd=root,
                env={
                    **os.environ,
                    "PATH": f"{root}:{os.environ['PATH']}",
                    "CALLS": str(root / "calls"),
                },
                check=True,
                capture_output=True,
                timeout=5,
            )
            calls = (root / "calls").read_text().splitlines()
            for test in (
                "test_native_cli_package.py",
                "test_native_cli_ci_scope.py",
                "test_ci_prose_scope.py",
                "test_ci_prose_workflow.py",
            ):
                self.assertIn(f"-m unittest discover -s tests/python -p {test}", calls)

    def test_mixed_producer_scopes_upload_only_genuinely_downloaded_artifacts(
        self,
    ) -> None:
        upload_job = job("upload-coverage")
        admission = condition(upload_job, indent=4)
        upload_steps = steps("upload-coverage")
        artifacts = re.findall(r"- artifact: (coverage-report-[^\s]+)", upload_job)
        self.assertEqual(len(artifacts), 9)
        self.assertNotIn("test_scope", upload_job)
        # Model all-prose, GCC-only fallback, an independent Python fallback,
        # mixed full/prose producers, and ordinary complete source coverage.
        for available in (
            set(),
            {artifacts[0]},
            {artifacts[1]},
            set(artifacts[::2]),
            set(artifacts),
        ):
            for classifier_outcome, classifier_scope in (
                ("success", "prose"),
                ("success", "full"),
                ("failure", "prose"),
                ("", ""),
            ):
                context = {
                    "github.event_name": "pull_request",
                    "cancelled()": False,
                    "needs.cpu.result": "success",
                    "needs.python.result": "success",
                    "steps.prose_scope.outcome": classifier_outcome,
                    "steps.prose_scope.outputs.scope": classifier_scope,
                }
                self.assertTrue(expression(admission, context))
                for index, artifact in enumerate(artifacts):
                    slot = index % 3 + 1
                    download = upload_steps[f"Download coverage report {slot}"]
                    upload = upload_steps[f"Upload coverage report {slot} to Codecov"]
                    self.assertIn("        continue-on-error: true\n", download)
                    self.assertIn("uses: actions/download-artifact@", download)
                    self.assertIn(
                        f"name: ${{{{ matrix.reports[{slot - 1}].artifact }}}}",
                        download,
                    )
                    self.assertTrue(selected(download, context))
                    for outcome in (
                        "success" if artifact in available else "failure",
                        "cancelled",
                        "skipped",
                        "",
                    ):
                        with self.subTest(
                            artifact=artifact,
                            available=sorted(available),
                            outcome=outcome,
                            classifier=(classifier_outcome, classifier_scope),
                        ):
                            specific = {
                                **context,
                                f"steps.download_{slot}.outcome": outcome,
                            }
                            self.assertEqual(
                                selected(upload, specific), outcome == "success"
                            )
                            self.assertFalse(
                                selected(upload, {**specific, "cancelled()": True})
                            )

    def test_coverage_job_keeps_existing_event_and_result_admission(self) -> None:
        admission = condition(job("upload-coverage"), indent=4)
        for event in (
            "pull_request",
            "merge_group",
            "push",
            "schedule",
            "workflow_dispatch",
        ):
            for cpu_result in ("success", "failure", "cancelled", "skipped"):
                for python_result in ("success", "failure", "cancelled", "skipped"):
                    for cancelled in (False, True):
                        context = {
                            "github.event_name": event,
                            "cancelled()": cancelled,
                            "needs.cpu.result": cpu_result,
                            "needs.python.result": python_result,
                        }
                        expected = (
                            event not in {"merge_group", "push"}
                            and cpu_result == python_result == "success"
                            and not cancelled
                        )
                        self.assertEqual(bool(expression(admission, context)), expected)


if __name__ == "__main__":
    unittest.main()
