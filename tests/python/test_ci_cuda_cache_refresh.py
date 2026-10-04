"""Execute the workflow's cache-refresh decision without network or a compiler."""

from __future__ import annotations

import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text()


def _step(name: str) -> str:
    marker = f"      - name: {name}\n"
    start = WORKFLOW.index(marker)
    end = WORKFLOW.find("      - name: ", start + len(marker))
    return WORKFLOW[start:] if end < 0 else WORKFLOW[start:end]


SCRIPT = textwrap.dedent(
    _step("Locate trusted master CUDA ccache snapshot").split("        run: |\n", 1)[1]
)
NOW = 1_800_000_000
MOCK_GH = """#!{python}
import json
import os
import sys
endpoint = next(value for value in sys.argv if value.startswith("/repos/"))
with open(os.environ["MOCK_CALLS"], "a") as stream:
    stream.write(endpoint + "\\n")
responses = json.loads(os.environ["MOCK_RESPONSES"])
response = responses.get(endpoint)
if response is None:
    sys.exit(1)
print(json.dumps(response))
"""
MOCK_DATE = """#!{python}
import datetime
import os
import sys
arguments = sys.argv[1:]
if arguments == ["-u", "+%s"]:
    print(os.environ["MOCK_NOW"])
elif len(arguments) == 4 and arguments[:2] == ["-u", "-d"] and arguments[3] == "+%s":
    stamp = datetime.datetime.fromisoformat(arguments[2].replace("Z", "+00:00"))
    print(int(stamp.timestamp()))
else:
    sys.exit(2)
"""


@unittest.skipUnless(
    all(shutil.which(command) for command in ("bash", "git", "jq")),
    "requires the workflow's host shell tools",
)
class CudaCacheRefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.env = {
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "CI fixture",
            "GIT_AUTHOR_EMAIL": "ci@example.invalid",
            "GIT_COMMITTER_NAME": "CI fixture",
            "GIT_COMMITTER_EMAIL": "ci@example.invalid",
        }
        self._git("init", "-q")
        (self.repository / "README.md").write_text("fixture\n")
        (self.repository / "src").mkdir()
        (self.repository / "src/kernel.cu").write_text("original\n")
        self.base = self._commit()
        self.head = self.base
        binaries = self.root / "bin"
        binaries.mkdir()
        gh = binaries / "gh"
        gh.write_text(MOCK_GH.format(python=sys.executable))
        gh.chmod(0o755)
        date = binaries / "date"
        date.write_text(MOCK_DATE.format(python=sys.executable))
        date.chmod(0o755)
        self.env["PATH"] = str(binaries) + os.pathsep + self.env["PATH"]

    def _git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=self.repository, env=self.env, text=True
        ).strip()

    def _commit(self) -> str:
        self._git("add", ".")
        self._git("commit", "-qm", "fixture")
        return self._git("rev-parse", "HEAD")

    def _change(self, path: str) -> None:
        self._git("reset", "--hard", self.base)
        self._git("clean", "-fdq")
        target = self.repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("changed\n")
        self.head = self._commit()

    def _run(
        self,
        *,
        age: int = 60,
        source_sha: str | None = None,
        event: str = "push",
        ref: str = "refs/heads/master",
        runs: list[dict] | None = None,
        artifacts: dict[int, list[dict]] | None = None,
        discovery_fails: bool = False,
    ) -> tuple[dict[str, str], list[str]]:
        created = datetime.datetime.fromtimestamp(
            NOW - age, datetime.timezone.utc
        ).isoformat()
        if runs is None:
            runs = [
                {
                    "id": 71,
                    "head_sha": self.base if source_sha is None else source_sha,
                    "event": "push",
                    "head_branch": "master",
                    "conclusion": "success",
                }
            ]
        if artifacts is None:
            artifacts = {71: [{"expired": False, "created_at": created}]}
        responses: dict[str, object] = {}
        if not discovery_fails:
            responses["/repos/owner/repo/actions/workflows/ci.yml/runs"] = {
                "workflow_runs": runs
            }
        for run_id, entries in artifacts.items():
            responses[f"/repos/owner/repo/actions/runs/{run_id}/artifacts"] = {
                "artifacts": entries
            }
        output = self.root / "outputs"
        calls = self.root / "calls"
        output.write_text("")
        calls.write_text("")
        script = SCRIPT.replace("${{ github.event_name }}", event).replace(
            "${{ github.ref }}", ref
        )
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", script],
            cwd=self.repository,
            env={
                **self.env,
                "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_SHA": self.head,
                "GITHUB_OUTPUT": str(output),
                "MOCK_RESPONSES": json.dumps(responses),
                "MOCK_CALLS": str(calls),
                "MOCK_NOW": str(NOW),
            },
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return dict(line.split("=", 1) for line in output.read_text().splitlines()), (
            calls.read_text().splitlines()
        )

    def test_young_unchanged_and_docs_only_stay_throttled(self) -> None:
        for path in (None, "README.md", "docs/developer/example.md"):
            with self.subTest(path=path):
                if path is not None:
                    self._change(path)
                output, _ = self._run()
                self.assertEqual(output, {"run_id": "71", "publish": "false"})

    def test_young_changed_compiler_build_and_toolchain_inputs_refresh(self) -> None:
        paths = (
            "cmake/toolchain.cmake",
            "python/generativeqc_compiler/integral/example.py",
            "python/generativeqc/runtime.py",
            "CMakeLists.txt",
            "CMakePresets.json",
            "include/generativeqc/example.h",
            "src/kernel.cu",
            "tests/native/example.cpp",
            "tools/generate_example.py",
            "manifests/xtbloom-d3.json",
            "data/parameters/d3_production.bin",
            "upstream/example/template.in",
            "pyproject.toml",
            "uv.lock",
            ".github/workflows/ci.yml",
            ".github/actions/compiler/action.yml",
        )
        for path in paths:
            with self.subTest(path=path):
                self._change(path)
                output, _ = self._run()
                self.assertEqual(output["publish"], "true")

    def test_deleted_compiler_input_refreshes(self) -> None:
        (self.repository / "src/kernel.cu").unlink()
        self.head = self._commit()
        self.assertEqual(self._run()[0]["publish"], "true")

    def test_aged_unchanged_snapshot_refreshes(self) -> None:
        for age, expected in ((10799, "false"), (10800, "true"), (10801, "true")):
            with self.subTest(age=age):
                self.assertEqual(self._run(age=age)[0]["publish"], expected)

    def test_unknown_source_requests_non_gating_refresh(self) -> None:
        for sha in ("f" * 40, "", "--output=unexpected"):
            with self.subTest(sha=sha):
                self.assertEqual(self._run(source_sha=sha)[0]["publish"], "true")
        self.assertFalse((self.repository / "unexpected").exists())

    def test_discovery_failure_or_expired_artifact_refreshes(self) -> None:
        output, _ = self._run(discovery_fails=True)
        self.assertEqual(output, {"run_id": "", "publish": "true"})
        output, _ = self._run(
            artifacts={71: [{"expired": True, "created_at": "2026-01-01T00:00:00Z"}]}
        )
        self.assertEqual(output, {"run_id": "", "publish": "true"})

    def test_selected_artifact_owns_the_compared_source_identity(self) -> None:
        self._change("src/kernel.cu")
        runs = [
            {
                "id": 80,
                "head_sha": self.head,
                "event": "push",
                "head_branch": "master",
                "conclusion": "success",
            },
            {
                "id": 71,
                "head_sha": self.base,
                "event": "push",
                "head_branch": "master",
                "conclusion": "success",
            },
        ]
        young = datetime.datetime.fromtimestamp(
            NOW - 60, datetime.timezone.utc
        ).isoformat()
        output, calls = self._run(
            runs=runs,
            artifacts={80: [], 71: [{"expired": False, "created_at": young}]},
        )
        self.assertEqual(output, {"run_id": "71", "publish": "true"})
        self.assertEqual(
            calls[-2:],
            [
                "/repos/owner/repo/actions/runs/80/artifacts",
                "/repos/owner/repo/actions/runs/71/artifacts",
            ],
        )

    def test_only_successful_master_push_runs_can_supply_artifacts(self) -> None:
        runs = [
            {
                "id": index,
                "head_sha": self.head,
                "event": event,
                "head_branch": branch,
                "conclusion": conclusion,
            }
            for index, (event, branch, conclusion) in enumerate(
                (
                    ("pull_request", "master", "success"),
                    ("push", "feature", "success"),
                    ("push", "master", "failure"),
                    ("push", "master", "success"),
                ),
                start=68,
            )
        ]
        output, calls = self._run(runs=runs)
        self.assertEqual(output, {"run_id": "71", "publish": "false"})
        self.assertEqual(calls[-1], "/repos/owner/repo/actions/runs/71/artifacts")
        self.assertEqual(len(calls), 2)

    def test_only_master_pushes_publish(self) -> None:
        self._change("src/kernel.cu")
        for event, ref in (
            ("merge_group", "refs/heads/gh-readonly-queue/master/pr-1"),
            ("pull_request", "refs/pull/1/merge"),
            ("push", "refs/heads/feature"),
        ):
            with self.subTest(event=event, ref=ref):
                self.assertEqual(
                    self._run(age=10860, event=event, ref=ref)[0]["publish"], "false"
                )

    def test_refresh_pathspecs_cover_all_existing_cuda_cache_key_inputs(self) -> None:
        key = next(
            line
            for line in _step("Restore CUDA ccache").splitlines()
            if "key: ccache-cuda-" in line
        )
        dependencies = re.findall(r"'([^']+)'", key)
        match = re.search(r"cuda_inputs=\(\s*(.*?)\s*\)", SCRIPT, re.DOTALL)
        self.assertIsNotNone(match)
        paths = shlex.split(match.group(1))
        for dependency in dependencies:
            dependency = dependency.removesuffix("/**")
            with self.subTest(dependency=dependency):
                self.assertTrue(
                    any(
                        dependency == path or dependency.startswith(path + "/")
                        for path in paths
                    )
                )
        self.assertIn(".github", paths)
        self.assertIn('"$source_sha" "$GITHUB_SHA"', SCRIPT)
        self.assertNotIn("origin/master", SCRIPT)


if __name__ == "__main__":
    unittest.main()
