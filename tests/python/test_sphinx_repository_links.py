"""Check source-link rendering while preserving real Sphinx warning gates."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.sphinx_repository_links import repository_source_link

ROOT = Path(__file__).resolve().parents[2]


class RepositorySourceLinkTests(unittest.TestCase):
    def test_existing_external_document_keeps_revision_and_anchor(self) -> None:
        result = repository_source_link(
            str(ROOT / "README"),
            root=ROOT,
            source_dir=ROOT / "docs",
            suffixes=(".md", ".rst"),
            revision="a" * 40,
            fragment="installation",
        )
        assert result is not None
        self.assertEqual(result[0], ROOT / "README.md")
        self.assertEqual(
            result[1],
            "https://github.com/jinzhezenggroup/generativeqc/blob/"
            + "a" * 40
            + "/README.md#installation",
        )

    def test_internal_missing_and_outside_repository_links_are_not_externalized(
        self,
    ) -> None:
        for target in (
            "user/methods",
            str(ROOT / "docs/user/methods"),
            str(ROOT / "missing"),
            str(ROOT.parent / "outside"),
        ):
            with self.subTest(target=target):
                self.assertIsNone(
                    repository_source_link(
                        target,
                        root=ROOT,
                        source_dir=ROOT / "docs",
                        suffixes=(".md",),
                        revision="master",
                    )
                )

    def test_symlink_cannot_escape_the_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "repo"
            root.mkdir()
            (base / "outside.md").write_text("# Outside\n", encoding="utf-8")
            (root / "alias.md").symlink_to(base / "outside.md")
            self.assertIsNone(
                repository_source_link(
                    str(root / "alias"),
                    root=root,
                    source_dir=root / "docs",
                    suffixes=(".md",),
                    revision="master",
                )
            )

    @unittest.skipUnless(
        importlib.util.find_spec("sphinx"), "requires documentation dependencies"
    )
    def test_sphinx_preserves_internal_checks_and_renders_source_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (root / "README.md").write_text("# Repository\n", encoding="utf-8")
            (docs / "conf.py").write_text(
                "import sys\n"
                f"sys.path.insert(0, {str(ROOT)!r})\n"
                "from tools.sphinx_repository_links import setup_repository_links\n"
                "extensions = ['myst_parser']\n"
                "source_suffix = {'.md': 'markdown'}\n"
                "root_doc = 'index'\n"
                "repository_source_revision = 'test-revision'\n"
                "def setup(app): setup_repository_links(app)\n",
                encoding="utf-8",
            )
            index = docs / "index.md"
            index.write_text(
                "# Test\n\n[**Source**](../README.md#repository)\n", encoding="utf-8"
            )
            output = root / "html"
            command = [
                sys.executable,
                "-m",
                "sphinx",
                "-W",
                "--keep-going",
                "-E",
                "-b",
                "html",
                str(docs),
                str(output),
            ]
            result = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=30
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            html = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("/blob/test-revision/README.md#repository", html)
            self.assertIn("<strong>Source</strong>", html)
            for target in ("missing.md", "../missing.md"):
                with self.subTest(target=target):
                    index.write_text(
                        f"# Test\n\n[Broken]({target})\n", encoding="utf-8"
                    )
                    result = subprocess.run(
                        command, check=False, capture_output=True, text=True, timeout=30
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("myst.xref_missing", result.stdout + result.stderr)
