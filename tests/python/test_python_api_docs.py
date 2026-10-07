"""Check automatic Python API module discovery without importing runtime modules."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import render_python_api_doc as renderer

ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class PythonApiDocumentationTests(unittest.TestCase):
    def test_public_modules_are_discovered_recursively_from_all(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "generativeqc"
            _write(package / "__init__.py", '__all__ = ["Root"]\n')
            _write(package / "future.py", '__all__ = ["Thing", "run"]\n')
            _write(package / "ordinary_internal.py", "def helper(): pass\n")
            _write(package / "_private.py", '__all__ = ["Hidden"]\n')
            _write(package / "__main__.py", '__all__ = ["main"]\n')
            _write(package / "nested/__init__.py", '__all__ = ["feature"]\n')
            _write(package / "nested/feature.py", '__all__ = ["Feature"]\n')

            modules = renderer.public_api_modules(package)
            self.assertEqual(
                [module.name for module in modules],
                [
                    "generativeqc",
                    "generativeqc.future",
                    "generativeqc.nested",
                    "generativeqc.nested.feature",
                ],
            )
            rendered = renderer.render_python_api_markdown(package)
            self.assertIn("generativeqc.future", rendered)
            self.assertIn("generativeqc.nested.feature", rendered)
            self.assertNotIn("ordinary_internal", rendered)
            self.assertNotIn("_private", rendered)
            self.assertNotIn(":no-index:", rendered)

    def test_nonliteral_public_all_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "generativeqc"
            _write(package / "__init__.py", '__all__ = ["Root"]\n')
            _write(package / "dynamic.py", "__all__ = names()\n")

            with self.assertRaisesRegex(ValueError, "literal list or tuple"):
                renderer.public_api_modules(package)

    def test_current_reference_covers_declared_public_facades(self) -> None:
        names = {module.name for module in renderer.public_api_modules()}

        self.assertIn("generativeqc", names)
        self.assertIn("generativeqc.torch", names)
        self.assertIn("generativeqc.extensions", names)
        self.assertIn("generativeqc.extensions.method", names)
        self.assertIn("generativeqc.extensions.tensor", names)
        self.assertIn("generativeqc.extensions.xc", names)

    def test_sphinx_source_is_generated_and_tracks_package_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "generativeqc"
            _write(package / "__init__.py", '__all__ = ["Root"]\n')
            _write(package / "future.py", '__all__ = ["Thing"]\n')

            registered: list[str] = []
            app = SimpleNamespace(
                env=SimpleNamespace(note_dependency=registered.append)
            )
            source = ["source shell"]
            with mock.patch.object(renderer, "PACKAGE", package):
                renderer.render_python_api_source(app, "reference/api", source)

            self.assertIn("generativeqc.future", source[0])
            self.assertIn(str(package), registered)
            self.assertIn(str(package / "future.py"), registered)

            registered.clear()
            source = ["unrelated"]
            with mock.patch.object(renderer, "PACKAGE", package):
                renderer.render_python_api_source(app, "index", source)
            self.assertEqual(source, ["unrelated"])
            self.assertFalse(registered)

    @unittest.skipUnless(
        importlib.util.find_spec("sphinx"), "requires documentation dependencies"
    )
    def test_rendered_reference_preserves_module_and_member_targets(self) -> None:
        from sphinx.util.inventory import InventoryFile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            output = root / "html"
            _write(
                docs / "conf.py",
                "import sys\n"
                f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'python')!r}]\n"
                "from tools.render_python_api_doc import render_python_api_source\n"
                "extensions = ['myst_parser', 'sphinx.ext.autodoc', "
                "'sphinx.ext.autosummary']\n"
                "root_doc = 'index'\n"
                "autodoc_mock_imports = ['torch']\n"
                "autodoc_typehints_format = 'fully-qualified'\n"
                "def setup(app):\n"
                "    app.connect('source-read', render_python_api_source)\n",
            )
            _write(docs / "index.md", "# Test\n\n```{toctree}\nreference/api\n```\n")
            _write(docs / "reference/api.md", "# Generated shell\n")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sphinx",
                    "-W",
                    "--keep-going",
                    "-b",
                    "html",
                    str(docs),
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            html = (output / "reference/api.html").read_text(encoding="utf-8")
            with (output / "objects.inv").open("rb") as stream:
                inventory = InventoryFile.load(stream, "", lambda _, uri: uri)
            for module in renderer.public_api_modules():
                with self.subTest(module=module.name):
                    self.assertIn(f'id="module-{module.name}"', html)
                    self.assertIn(f'href="#module-{module.name}"', html)
                    self.assertIn(module.name, inventory["py:module"])
            for kind, name in (
                ("class", "generativeqc.Calculator"),
                ("class", "generativeqc.FunctionalSpec"),
                ("class", "generativeqc.response_problem.ResponseProblem"),
                ("function", "generativeqc.extensions.method.compose"),
                ("function", "generativeqc.extensions.tensor.compile"),
                ("function", "generativeqc.extensions.xc.named"),
                ("function", "generativeqc.torch.energy"),
            ):
                with self.subTest(member=name):
                    self.assertIn(f'id="{name}"', html)
                    self.assertIn(name, inventory[f"py:{kind}"])


if __name__ == "__main__":
    unittest.main()
