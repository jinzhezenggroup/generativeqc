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
        self.assertIn("generativeqc.experimental", names)
        self.assertIn("generativeqc.experimental.array_api", names)
        self.assertIn("generativeqc.torch", names)
        self.assertIn("generativeqc.extensions", names)
        self.assertIn("generativeqc.extensions.method", names)
        self.assertIn("generativeqc.extensions.tensor", names)
        self.assertIn("generativeqc.extensions.xc", names)

    def test_shared_classes_link_to_public_targets_without_hiding_unique_members(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "generativeqc"
            _write(package / "__init__.py", '__all__ = ["Root"]\n')
            _write(
                package / "experimental/array_api.py",
                '__all__ = ["Index", "Program", "compile", "Unique"]\n',
            )
            _write(package / "extensions/tensor.py", '__all__ = ["Index", "Program"]\n')
            rendered = renderer.render_python_api_markdown(package)
            experimental = rendered.split(
                "## `generativeqc.experimental.array_api`", 1
            )[1].split("## `generativeqc.extensions.tensor`", 1)[0]
            self.assertIn(":members: Index, Program, compile, Unique", experimental)
            self.assertIn(":exclude-members: Index, Program\n", experimental)
            for name in ("Index", "Program"):
                self.assertIn(
                    f"{{py:class}}`{name} <generativeqc.extensions.tensor.{name}>`",
                    experimental,
                )
            self.assertNotIn(":no-index:", rendered)

            _write(package / "extensions/tensor.py", '__all__ = ["Index"]\n')
            with self.assertRaisesRegex(ValueError, "Program is not public"):
                renderer.render_python_api_markdown(package)

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

    def test_public_docstring_audit_follows_reexports(self) -> None:
        from tools.check_public_api_docstrings import missing_public_docstrings

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "generativeqc"
            _write(
                package / "__init__.py",
                '"""Public facade."""\nfrom .impl import Documented, Missing\n'
                '__all__ = ["Documented", "Missing"]\n',
            )
            _write(
                package / "impl.py",
                'class Documented:\n    """Documented class."""\n'
                "class Missing:\n    pass\n",
            )
            failures = missing_public_docstrings(package)
            self.assertEqual(len(failures), 1)
            self.assertIn("generativeqc.Missing", failures[0])
            self.assertIn("impl.py", failures[0])

    def test_public_docstring_audit_rejects_new_undocumented_module(self) -> None:
        from tools.check_public_api_docstrings import missing_public_docstrings

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "generativeqc"
            _write(package / "__init__.py", '"""Public facade."""\n__all__ = []\n')
            _write(
                package / "fresh.py",
                '"""Future public module."""\n'
                "def undocumented():\n    return 1\n"
                '__all__ = ["undocumented"]\n',
            )
            failures = missing_public_docstrings(package)
            self.assertEqual(len(failures), 1)
            self.assertIn("generativeqc.fresh.undocumented", failures[0])

    def test_public_docstring_audit_accepts_documented_reexports(self) -> None:
        from tools.check_public_api_docstrings import missing_public_docstrings

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "generativeqc"
            _write(
                package / "__init__.py",
                '"""Public facade."""\nfrom .impl import Entry\n__all__ = ["Entry"]\n',
            )
            _write(
                package / "impl.py",
                'class Entry:\n    """A documented entry point."""\n',
            )
            self.assertEqual(missing_public_docstrings(package), ())

    def test_public_docstring_audit_checks_reexported_class_members(self) -> None:
        from tools.check_public_api_docstrings import missing_public_docstrings

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "generativeqc"
            _write(
                package / "__init__.py",
                '"""Public facade."""\nfrom .bridge import PublicEntry\n'
                '__all__ = ["PublicEntry"]\n',
            )
            _write(package / "bridge.py", "from ._impl import Entry as PublicEntry\n")
            _write(
                package / "_impl.py",
                "raise RuntimeError('Static discovery must never import this module')\n"
                'class Entry:\n    """A documented class."""\n'
                "    def __init__(self): pass\n"
                "    def missing_method(self): pass\n"
                "    async def missing_async(self): pass\n"
                "    def __enter__(self): return self\n"
                "    @property\n    def value(self): return 1\n"
                "    def _private(self): pass\n"
                "    class Nested:\n        def nested_method(self): pass\n"
                "class Unrelated:\n    def method(self): pass\n",
            )
            failures = missing_public_docstrings(package)
            self.assertEqual(len(failures), 7)
            for member in (
                "__init__",
                "missing_method",
                "missing_async",
                "__enter__",
                "value",
                "Nested",
                "Nested.nested_method",
            ):
                self.assertTrue(
                    any(
                        f"generativeqc.PublicEntry.{member} -> generativeqc/_impl.py:"
                        in failure
                        for failure in failures
                    ),
                    failures,
                )

    def test_public_docstring_audit_limits_member_scope(self) -> None:
        from tools.check_public_api_docstrings import missing_public_docstrings

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "generativeqc"
            _write(
                package / "__init__.py",
                '"""Public facade."""\nfrom ._impl import Entry\n__all__ = ["Entry"]\n',
            )
            _write(
                package / "_impl.py",
                "class Unrelated:\n    def inherited(self): pass\n"
                'class Entry(Unrelated):\n    """An exported class."""\n'
                '    def __init__(self):\n        """Initialize the entry."""\n'
                '    def method(self):\n        """Run the entry."""\n'
                "    def _private(self): pass\n"
                "    class _Private:\n        def method(self): pass\n",
            )
            self.assertEqual(missing_public_docstrings(package), ())

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
                    self.assertEqual(html.count(f'id="module-{module.name}"'), 1)
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
                ("class", "generativeqc.experimental.array_api.CompiledFunction"),
                ("class", "generativeqc.experimental.array_api.VibeArray"),
                ("function", "generativeqc.experimental.array_api.compile"),
                ("function", "generativeqc.experimental.array_api.asarray"),
                ("function", "generativeqc.experimental.array_api.add"),
                ("function", "generativeqc.experimental.array_api.import_dlpack"),
            ):
                with self.subTest(member=name):
                    self.assertEqual(html.count(f'id="{name}"'), 1)
                    self.assertIn(name, inventory[f"py:{kind}"])
            for name, implementation in (
                ("Index", "generativeqc_compiler.tensor.types.Index"),
                ("IndexSpace", "generativeqc_compiler.tensor.types.IndexSpace"),
                ("Program", "generativeqc_compiler.tensor.program.Program"),
                ("TensorSpec", "generativeqc_compiler.tensor.types.TensorSpec"),
                ("FunctionalSpec", "generativeqc_compiler.xc.spec.FunctionalSpec"),
            ):
                facade = (
                    "generativeqc"
                    if name == "FunctionalSpec"
                    else "generativeqc.extensions.tensor"
                )
                target = f"{facade}.{name}"
                with self.subTest(shared_class=name):
                    self.assertEqual(html.count(f'id="{target}"'), 1)
                    self.assertIn(f'href="#{target}"', html)
                    self.assertIn(target, inventory["py:class"])
                    self.assertIn(implementation, inventory["py:class"])
                    self.assertEqual(
                        inventory["py:class"][implementation].uri,
                        inventory["py:class"][target].uri,
                    )


if __name__ == "__main__":
    unittest.main()
