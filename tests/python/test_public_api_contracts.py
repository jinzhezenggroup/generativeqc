"""Negative static contract fixtures; no optional/native dependency is imported."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from tools import check_public_api_docstrings as checker


def write(path: Path, content: str) -> None:
    """Write one source/document fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class PublicContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "python/generativeqc"
        self.manifest = self.root / "manifests/python_api_contracts.json"
        write(self.package / "__init__.py", '"""Public facade."""\n__all__ = []\n')

    def exported(self, source: str, names: tuple[str, ...] = ("run",)) -> None:
        write(self.package / "__init__.py", source + f"\n__all__ = {list(names)!r}\n")

    def errors(self) -> tuple[str, ...]:
        return checker.missing_public_docstrings(self.package)

    def contracts(self) -> tuple[str, ...]:
        return checker.public_contract_errors(self.package, self.manifest, self.root)

    def policy(self, profile: str = "execution") -> dict:
        """Build an explicit fixture policy, not an automatic production baseline."""
        exports, declarations, failures = checker.public_declarations(self.package)
        self.assertFalse(failures)
        sections = []
        for facet in checker.PROFILES[profile] - {"behavior"}:
            sections.append(
                f"(run-{facet})=\n## {facet}\n\nThe exact {facet} contract for this fixture.\n"
            )
        write(self.root / "docs/contract.md", "\n".join(sections))
        policy = {
            "schema_version": 1,
            "debt": [],
            "exports": exports,
            "declarations": {
                key: {
                    "kind": declaration.kind,
                    "profile": profile,
                    "facets": {
                        facet: "docstring"
                        if facet == "behavior"
                        else f"docs/contract.md#run-{facet}"
                        for facet in checker.PROFILES[profile]
                    },
                }
                for key, declaration in declarations.items()
            },
        }
        self.save_policy(policy)
        return policy

    def save_policy(self, policy: dict) -> None:
        write(self.manifest, json.dumps(policy))

    def test_literal_method_lazy_map_rejects_missing_class_documentation(self) -> None:
        self.exported("from compiler.method import Hidden", ("Hidden",))
        write(
            self.package.parent / "compiler/method/__init__.py",
            '_EXPORTS = {"Hidden": ".implementation"}\n'
            'raise RuntimeError("must not import lazy facade")\n',
        )
        write(
            self.package.parent / "compiler/method/implementation.py",
            "class Hidden: pass\n",
        )
        errors = self.errors()
        self.assertEqual(len(errors), 1)
        self.assertIn("Hidden", errors[0])
        self.assertIn("implementation.py", errors[0])

    def test_literal_tensor_lazy_map_rejects_missing_member(self) -> None:
        self.exported("from compiler.tensor import Export", ("Export",))
        write(
            self.package.parent / "compiler/tensor/__init__.py",
            '_LAZY_EXPORTS = {"Export": ("implementation", "Owner")}\n',
        )
        write(
            self.package.parent / "compiler/tensor/implementation.py",
            'raise RuntimeError("must not import implementation")\n'
            'class Owner:\n    """Own a reusable plan."""\n'
            "    def execute(self): pass\n",
        )
        self.assertEqual(len(self.errors()), 1)
        self.assertIn("Export.execute", self.errors()[0])

    def test_annotation_only_all_cannot_hide_a_new_undocumented_facade(self) -> None:
        self.policy("accessor")
        write(
            self.package / "fresh.py",
            'def run(): pass\n__all__ = ["run"]\n__all__: list[str]\n',
        )
        self.assertIn("generativeqc.fresh.run", "\n".join(self.errors()))
        self.assertIn(
            "missing/stale authoritative export binding", "\n".join(self.contracts())
        )

    def test_public_class_assignment_aliases_fail_closed(self) -> None:
        self.exported("from ._impl import Owner", ("Owner",))
        baseline = (
            'class Owner:\n    """Public owner."""\n    def _execute(self): pass\n'
        )
        write(self.package / "_impl.py", baseline)
        self.policy("accessor")
        for assignment in (
            "execute = _execute",
            "execute: object = _execute",
            "execute = staticmethod(_execute)",
            "execute = lambda self: None",
            "execute = factory()",
            "execute: object = external_callback",
            "execute: object = helper.execute",
            "execute: object = factory()",
            "_alias = _execute\n    execute: object = _alias",
        ):
            with self.subTest(assignment=assignment):
                write(self.package / "_impl.py", baseline + "    " + assignment + "\n")
                self.assertIn(
                    "unsupported public class assignment", "\n".join(self.errors())
                )
                self.assertIn(
                    "unsupported public class assignment", "\n".join(self.contracts())
                )

    def test_class_callable_shadowed_by_a_literal_fails_closed(self) -> None:
        self.exported("from ._impl import Owner", ("Owner",))
        baseline = (
            'class Owner:\n    """Public owner."""\n'
            '    def execute(self):\n        """Original operation."""\n'
        )
        write(self.package / "_impl.py", baseline)
        self.policy("accessor")
        for assignment in ("execute = None", "execute: int = 1"):
            with self.subTest(assignment=assignment):
                write(self.package / "_impl.py", baseline + "    " + assignment + "\n")
                self.assertIn(
                    "unsupported public class assignment", "\n".join(self.errors())
                )
                self.assertIn(
                    "unsupported public class assignment", "\n".join(self.contracts())
                )
        write(self.package / "_impl.py", baseline + "    execute: object\n")
        self.assertFalse(self.errors())
        self.assertFalse(self.contracts())

    def test_conditional_public_class_definitions_fail_closed(self) -> None:
        self.exported("from ._impl import Owner", ("Owner",))
        baseline = 'class Owner:\n    """Public owner."""\n'
        write(self.package / "_impl.py", baseline)
        self.policy("accessor")
        write(
            self.package / "_impl.py",
            baseline + "    if enabled:\n        def execute(self): pass\n",
        )
        self.assertIn(
            "unsupported conditional/dynamic class member", "\n".join(self.errors())
        )
        self.assertIn(
            "unsupported conditional/dynamic class member", "\n".join(self.contracts())
        )

    def test_ordinary_typed_class_data_fields_remain_supported(self) -> None:
        self.exported("from ._impl import Owner", ("Owner",))
        write(
            self.package / "_impl.py",
            "from dataclasses import dataclass, field\nVERSION = 1\n"
            '@dataclass\nclass Owner:\n    """Public record."""\n'
            "    version: int = VERSION\n    name: str\n"
            "    values: list = field(default_factory=list)\n"
            '    policy: object = make_policy()\n    KIND = "record"\n',
        )
        self.policy("record")
        self.assertFalse(self.errors())
        self.assertFalse(self.contracts())

    def test_progressive_docs_match_source_execution_modes(self) -> None:
        def definition(filename: str, name: str) -> ast.FunctionDef:
            tree = ast.parse((checker.ROOT / filename).read_text())
            return next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )

        for filename, name, strict in (
            ("python/generativeqc/progressive.py", "projected_singlepoint", True),
            (
                "python/generativeqc/progressive_controller.py",
                "run_progressive_hf",
                False,
            ),
        ):
            calls = [
                node
                for node in ast.walk(definition(filename, name))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "target_batch"
            ]
            self.assertEqual(len(calls), 1)
            self.assertIs(
                next(
                    keyword.value.value
                    for keyword in calls[0].keywords
                    if keyword.arg == "strict"
                    and isinstance(keyword.value, ast.Constant)
                ),
                strict,
            )
        seed_installer = definition(
            "python/generativeqc/progressive.py", "initialize_from"
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute"
                for node in ast.walk(seed_installer)
            )
        )
        docs = " ".join(
            (checker.ROOT / "docs/reference/python_contracts.md").read_text().split()
        )
        self.assertIn(
            "`projected_singlepoint` executes the target with `strict=True`", docs
        )
        self.assertIn(
            "`run_progressive_hf` executes the target with `strict=False`", docs
        )
        batch_docs = " ".join(
            (checker.ROOT / "docs/reference/python_execution_contracts.md")
            .read_text()
            .split()
        )
        self.assertIn(
            "`PreparedBatch.initialize_from` only installs compatible projected seed densities",
            batch_docs,
        )
        self.assertIn(
            "caller must subsequently call `execute` to solve the target", batch_docs
        )

    def test_assignment_alias_follows_owner_and_members(self) -> None:
        self.exported("from .implementation import Owner\nAlias = Owner", ("Alias",))
        write(
            self.package / "implementation.py",
            'class Owner:\n    """An owned value."""\n    def value(self): pass\n',
        )
        self.assertIn("generativeqc.Alias.value", self.errors()[0])
        exports, _, failures = checker.public_declarations(self.package)
        self.assertFalse(failures)
        self.assertEqual(
            exports["generativeqc.Alias"], "generativeqc/implementation.py:Owner"
        )

    def test_same_named_submodule_does_not_hide_a_function(self) -> None:
        self.exported("from .bridge import run")
        write(self.package / "bridge/__init__.py", "from .run import run\n")
        write(
            self.package / "bridge/run.py",
            '"""The module is documented."""\ndef run(): pass\n',
        )
        self.assertIn("missing docstring", self.errors()[0])

    def test_later_assignment_shadows_a_documented_definition(self) -> None:
        self.exported(
            'def run():\n    """Earlier documented function."""\n'
            "def _replacement(): pass\nrun = _replacement"
        )
        self.assertIn("missing docstring", self.errors()[0])

    def test_alias_retains_the_binding_at_assignment_time(self) -> None:
        self.exported(
            "def owner(): pass\nrun = owner\n"
            'def owner():\n    """Later documentation cannot repair the alias."""\n'
        )
        self.assertIn("missing docstring", self.errors()[0])

    def test_conditional_rebinding_cannot_use_an_earlier_docstring(self) -> None:
        self.exported(
            'def run():\n    """Earlier documented function."""\n'
            "if enabled:\n    run = replacement\n"
        )
        self.assertIn("unsupported conditional/dynamic binding", self.errors()[0])

    def test_shadowed_type_checking_guard_cannot_hide_rebinding(self) -> None:
        self.exported(
            "from typing import TYPE_CHECKING\nTYPE_CHECKING = True\n"
            'def run():\n    """Earlier documented function."""\n'
            "if TYPE_CHECKING:\n    def run(): pass\n"
        )
        self.assertIn("unsupported conditional/dynamic binding", self.errors()[0])

    def test_mutated_typing_module_guard_cannot_hide_rebinding(self) -> None:
        self.exported(
            "import typing\ntyping.TYPE_CHECKING = True\n"
            'def run():\n    """Earlier documented function."""\n'
            "if typing.TYPE_CHECKING:\n    def run(): pass\n"
        )
        self.assertIn("unsupported conditional/dynamic binding", self.errors()[0])

    def test_typing_flag_mutation_before_import_is_not_proven_false(self) -> None:
        self.exported(
            "import typing\ntyping.TYPE_CHECKING = True\n"
            "from typing import TYPE_CHECKING\n"
            'def run():\n    """Earlier implementation."""\n'
            "if TYPE_CHECKING:\n    def run(): pass\n"
        )
        self.assertIn("unsupported conditional/dynamic binding", self.errors()[0])

    def test_unshadowed_type_checking_imports_remain_static_only(self) -> None:
        self.exported(
            "from typing import TYPE_CHECKING\n"
            'def run():\n    """Runtime implementation."""\n'
            "if TYPE_CHECKING:\n    from unavailable import run\n"
        )
        self.assertFalse(self.errors())

    def test_distinct_live_definitions_cannot_share_one_owner_key(self) -> None:
        self.exported(
            'def public(x):\n    """One argument."""\nold = public\n'
            'def public(x, y):\n    """Two arguments."""\n',
            ("old", "public"),
        )
        self.assertIn("conflicting live definitions", "\n".join(self.errors()))
        _, _, failures = checker.public_declarations(self.package)
        self.assertIn("conflicting live definitions", "\n".join(failures))

    def test_package_attribute_beats_same_named_child_module(self) -> None:
        self.exported("def public(): pass", ())
        write(self.package / "public.py", '"""Unrelated documented child."""\n')
        write(
            self.package / "facade.py", 'from . import public\n__all__ = ["public"]\n'
        )
        self.assertIn("facade.public", self.errors()[0])
        self.assertIn("missing docstring", self.errors()[0])

    def test_unaliased_dotted_import_binds_the_root_package(self) -> None:
        self.exported("import other.child", ("other",))
        write(self.package.parent / "other/__init__.py", "value = 1\n")
        write(self.package.parent / "other/child.py", '"""Documented child only."""\n')
        self.assertIn("other/__init__.py: missing docstring", self.errors()[0])

    def test_import_from_module_fallback_when_package_attribute_is_absent(self) -> None:
        self.exported("from other import child", ("child",))
        write(self.package.parent / "other/__init__.py", '"""Package."""\n')
        write(self.package.parent / "other/child.py", '"""Child."""\n')
        self.assertFalse(self.errors())

    def test_literal_lazy_module_requires_its_own_documentation(self) -> None:
        self.exported('_PUBLIC_MODULES = frozenset({"feature"})', ("feature",))
        write(self.package / "feature.py", "x = 1\n")
        self.assertIn("feature.py: missing docstring", self.errors()[0])

    def test_direct_relative_module_export(self) -> None:
        self.exported("from . import feature", ("feature",))
        write(self.package / "feature.py", '"""Feature construction namespace."""\n')
        exports, declarations, failures = checker.public_declarations(self.package)
        self.assertFalse(failures)
        self.assertEqual(declarations[exports["generativeqc.feature"]].kind, "module")

    def test_unknown_export_fails_instead_of_silently_skipping(self) -> None:
        self.exported('"""Facade."""')
        self.assertIn("unresolved export generativeqc.run", self.errors()[0])
        self.assertIn("supported literal lazy map", self.errors()[0])

    def test_callable_factory_assignment_is_not_a_constant(self) -> None:
        self.exported("run = make_callable()")
        self.assertIn("unresolved export", self.errors()[0])

    def test_mutated_literal_lazy_map_fails_closed(self) -> None:
        for mutation in ('_EXPORTS["run"] = dynamic()', "_EXPORTS.update(dynamic())"):
            with self.subTest(mutation=mutation):
                self.exported('_EXPORTS = {"run": ".impl"}\n' + mutation)
                write(
                    self.package / "impl.py",
                    'def run():\n    """Old documented target."""\n',
                )
                self.assertIn("unsupported use/mutation", self.errors()[0])

    def test_shadowed_frozenset_is_not_a_literal_constant(self) -> None:
        self.exported("def frozenset(x): return make_callable()\nrun = frozenset({1})")
        self.assertIn("unsupported assignment", self.errors()[0])

    def test_shadowed_numpy_alias_is_not_an_explicit_dtype(self) -> None:
        self.exported('import numpy as np\nnp = factory()\nrun = np.dtype("float32")')
        self.assertIn("unsupported assignment", self.errors()[0])

    def test_unknown_dynamic_lazy_map_fails_closed(self) -> None:
        self.exported("_EXPORTS = discover_exports()")
        self.assertIn("must be a literal mapping", self.errors()[0])

    def test_unknown_literal_lazy_target_form_fails_closed(self) -> None:
        self.exported('_LAZY_EXPORTS = {"run": ["impl", "run"]}')
        self.assertIn("unsupported target", self.errors()[0])

    def test_alias_cycle_fails_with_diagnostic(self) -> None:
        self.exported('_EXPORTS = {"run": "."}')
        self.assertIn("cyclic export", self.errors()[0])

    def test_lazy_path_cannot_escape_source_root(self) -> None:
        self.exported('_EXPORTS = {"run": "outside/../../secret"}')
        self.assertIn("invalid module path", self.errors()[0])

    def test_symlink_source_cannot_escape_source_root(self) -> None:
        outside = self.root / "secret.py"
        write(outside, 'def run():\n    """Outside source."""\n')
        (self.package / "impl.py").symlink_to(outside)
        self.exported("from .impl import run")
        self.assertIn("escapes root", self.errors()[0])

    def test_explicit_constant_type_dtype_categories(self) -> None:
        self.exported(
            "import numpy as np\nVERSION = 1\nExact = int | str\n"
            'float32 = np.dtype("float32")\n'
            'bool_ = np.dtype("bool")',
            ("VERSION", "Exact", "float32", "bool_"),
        )
        _, declarations, failures = checker.public_declarations(self.package)
        self.assertFalse(failures)
        self.assertEqual(
            {d.kind for d in declarations.values()}, {"constant", "type", "dtype"}
        )
        self.assertFalse(self.errors())
        write(
            self.manifest,
            json.dumps(
                {"schema_version": 1, "debt": [], "exports": {}, "declarations": {}}
            ),
        )
        self.assertTrue(
            any("missing contract classification" in item for item in self.contracts())
        )

    def test_new_documented_export_still_requires_contract_classification(self) -> None:
        self.exported('def run():\n    """Execute a checked program."""\n')
        policy = self.policy()
        policy["declarations"].clear()
        self.save_policy(policy)
        self.assertIn("missing contract classification", "\n".join(self.contracts()))

    def test_placeholder_docstring_is_rejected(self) -> None:
        self.exported('def run():\n    """TODO."""\n')
        self.assertIn("placeholder documentation", self.errors()[0])

    def test_short_accessor_is_valid_without_execution_facets(self) -> None:
        self.exported('def run():\n    """Return the rank."""\n')
        self.policy("accessor")
        self.assertFalse(self.errors())
        self.assertFalse(self.contracts())

    def test_execution_with_all_applicable_facets_is_valid(self) -> None:
        self.exported('def run():\n    """Execute a checked program."""\n')
        self.policy()
        self.assertFalse(self.contracts())

    def test_loss_of_an_applicable_execution_facet_is_rejected(self) -> None:
        self.exported('def run():\n    """Execute a checked program."""\n')
        for facet in ("values", "errors", "ownership", "backends"):
            with self.subTest(facet=facet):
                policy = self.policy()
                next(iter(policy["declarations"].values()))["facets"].pop(facet)
                self.save_policy(policy)
                self.assertIn(
                    f"missing applicable {facet} contract", "\n".join(self.contracts())
                )

    def test_nonempty_docstring_cannot_substitute_for_execution_facets(self) -> None:
        self.exported('def run():\n    """Execute a checked program."""\n')
        policy = self.policy()
        next(iter(policy["declarations"].values()))["facets"]["errors"] = "docstring"
        self.save_policy(policy)
        self.assertIn("non-behavior facets require", "\n".join(self.contracts()))

    def test_authoritative_document_symlink_cannot_escape_root(self) -> None:
        self.exported('def run():\n    """Execute a checked program."""\n')
        policy = self.policy()
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "external.md"
            write(outside, "(run-errors)=\n## Errors\n\nExternal contract.\n")
            (self.root / "docs/external.md").symlink_to(outside)
            next(iter(policy["declarations"].values()))["facets"]["errors"] = (
                "docs/external.md#run-errors"
            )
            self.save_policy(policy)
            self.assertIn("escapes repository root", "\n".join(self.contracts()))

    def test_lost_authoritative_section_is_rejected(self) -> None:
        self.exported('def run():\n    """Execute a checked program."""\n')
        self.policy()
        path = self.root / "docs/contract.md"
        path.write_text(
            path.read_text().replace("(run-errors)=", "(unrelated-errors)=")
        )
        self.assertIn("missing contract label", "\n".join(self.contracts()))

    def test_empty_facet_cannot_borrow_next_sections_text(self) -> None:
        self.exported('def run():\n    """Execute a checked program."""\n')
        self.policy()
        path = self.root / "docs/contract.md"
        path.write_text(
            path.read_text().replace("The exact errors contract for this fixture.", "")
        )
        self.assertIn("run-errors: missing docstring", "\n".join(self.contracts()))

    def test_placeholder_reference_section_is_rejected(self) -> None:
        self.exported('def run():\n    """Execute a checked program."""\n')
        self.policy()
        path = self.root / "docs/contract.md"
        path.write_text(
            path.read_text().replace(
                "The exact errors contract for this fixture.", "TODO: explain failures."
            )
        )
        self.assertIn("placeholder documentation", "\n".join(self.contracts()))

    def test_binding_change_requires_inventory_update(self) -> None:
        self.exported('def run():\n    """Execute a checked program."""\n')
        self.policy()
        self.exported("from .other import run")
        write(
            self.package / "other.py",
            'def run():\n    """Execute a checked program."""\n',
        )
        errors = "\n".join(self.contracts())
        self.assertIn("missing/stale authoritative export binding", errors)
        self.assertIn("stale contract record", errors)

    def test_new_member_requires_inventory_update(self) -> None:
        self.exported('class Owner:\n    """Describe a value."""\n', ("Owner",))
        self.policy("accessor")
        self.exported(
            'class Owner:\n    """Describe a value."""\n'
            '    def size(self):\n        """Return the size."""\n',
            ("Owner",),
        )
        self.assertIn(
            "Owner.size: missing contract classification", "\n".join(self.contracts())
        )

    def test_substantive_debt_is_not_a_completion_exemption(self) -> None:
        self.exported('def run():\n    """Execute a checked program."""\n')
        policy = self.policy()
        policy["debt"] = ["generativeqc/__init__.py:run"]
        self.save_policy(policy)
        self.assertIn("empty debt list", "\n".join(self.contracts()))

    def test_current_inventory_is_complete_and_debt_free(self) -> None:
        self.assertEqual(checker.missing_public_docstrings(), ())
        self.assertEqual(checker.public_contract_errors(), ())
        exports, declarations, failures = checker.public_declarations()
        self.assertFalse(failures)
        self.assertTrue(exports)
        self.assertTrue(declarations)
        policy = json.loads(checker.MANIFEST.read_text())
        for key in (
            "generativeqc/calculator.py:Calculator.singlepoint",
            "generativeqc/batch.py:PreparedBatch.execute",
            "generativeqc/torch.py:energy",
            "generativeqc/torch.py:batched_energy",
            "generativeqc/extensions/tensor.py:CompiledTensorProgram.execute",
            "generativeqc/experimental/array_api.py:CompiledFunction.__call__",
        ):
            self.assertEqual(policy["declarations"][key]["profile"], "execution")
        for name in ("T", "mT"):
            key = f"generativeqc_compiler/array_api/array.py:VibeArray.{name}"
            self.assertEqual(policy["declarations"][key]["profile"], "transformation")
        for name in ("dispersion_diagnostic", "ks_transport_diagnostics"):
            key = f"generativeqc/batch.py:PreparedBatch.{name}"
            self.assertEqual(policy["declarations"][key]["profile"], "execution")


if __name__ == "__main__":
    unittest.main()
