"""Static, compiler-free native declaration/contract regression checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.check_native_api_contracts import (
    HEADERS,
    MANIFEST,
    REFERENCE,
    ROOT,
    ContractError,
    contract_facets,
    discover,
    executable_tokens,
    render,
    reviewed_entry,
    scan_c,
    scan_cpp,
    validate,
)

C = """/** @native-contract generativeqc_example
 * @behavior Read an example count without numerical execution.
 * @outputs Returns the example count by value.
 * @execution Synchronous immutable lookup with no owner.
 */
GENERATIVEQC_API uint32_t generativeqc_example(void);
"""
CPP = """namespace generativeqc {
/** @native-contract generativeqc::Example
 * @behavior Own a copied count value.
 * @outputs count holds the copied integer count.
 */
struct Example {
  int count{};
  /** @native-contract generativeqc::Example::get
   * @behavior Read the stored count.
   * @outputs Returns count by value.
   * @errors noexcept; no native access or allocation.
   */
  int get() const noexcept { return count; }
};
}
"""


class NativeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for header, source in zip(HEADERS, (C, CPP), strict=True):
            path = self.root / header
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        entries = {
            d.name: reviewed_entry(d, list(contract_facets(d)))
            for d in discover(self.root)
        }
        path = self.root / MANIFEST
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"schema_version": 1, "declarations": entries}), encoding="utf-8"
        )

    def mutate(self, header: str, before: str, after: str) -> None:
        path = self.root / header
        source = path.read_text(encoding="utf-8")
        self.assertIn(before, source)
        path.write_text(source.replace(before, after, 1), encoding="utf-8")

    def test_repository_coverage_and_reference(self) -> None:
        declarations = validate(ROOT)
        # Exact current inventory is a reviewed audit, not a wildcard allowance.
        self.assertEqual(sum(d.kind == "c-function" for d in declarations), 80)
        self.assertEqual(sum(d.kind == "cpp-type" for d in declarations), 9)
        self.assertEqual(sum(d.kind == "cpp-operation" for d in declarations), 33)
        self.assertEqual(
            (ROOT / REFERENCE).read_text(encoding="utf-8"), render(declarations)
        )
        self.assertEqual(len({d.anchor for d in declarations}), len(declarations))
        for d in declarations:
            self.assertIn(f"({d.anchor})=", render(declarations))

    def test_minimal_reviewed_source(self) -> None:
        self.assertEqual(len(validate(self.root)), 3)

    def test_utf8_contract_round_trip_under_ascii_locale(self) -> None:
        self.mutate(HEADERS[0], "Read an example count", "Read a λ count in Å")
        self.mutate(HEADERS[1], "Own a copied count value.", "Own a copied λ value.")
        manifest_path = self.root / MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["review_note"] = "Reviewed λ and Å units"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        reference = self.root / REFERENCE
        reference.parent.mkdir(parents=True)
        environment = {
            **os.environ,
            "PYTHONUTF8": "0",
            "PYTHONCOERCECLOCALE": "0",
            "LC_ALL": "C",
        }
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import codecs, locale; print(codecs.lookup(locale.getpreferredencoding(False)).name)",
            ],
            env=environment,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )
        if probe.stdout.strip() != "ascii":
            self.skipTest("Runtime does not expose an ASCII C locale")
        command = [
            sys.executable,
            str(ROOT / "tools/check_native_api_contracts.py"),
            "--root",
            str(self.root),
        ]
        for arguments in (["--write-reference"], []):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [*command, *arguments],
                    env=environment,
                    capture_output=True,
                    encoding="utf-8",
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rendered = reference.read_text(encoding="utf-8")
        self.assertIn("Read a λ count in Å", rendered)
        self.assertIn("Own a copied λ value.", rendered)

    def test_new_undocumented_c_function_fails(self) -> None:
        self.mutate(
            HEADERS[0], C, C + "GENERATIVEQC_API void generativeqc_new(void);\n"
        )
        with self.assertRaisesRegex(ContractError, "inventory drift.*generativeqc_new"):
            validate(self.root)

    def test_new_undocumented_cpp_operation_fails(self) -> None:
        self.mutate(HEADERS[1], "int count{};", "int count{};\n  void run() {}")
        with self.assertRaisesRegex(ContractError, "inventory drift.*Example::run"):
            validate(self.root)

    def test_arbitrary_adjacent_comment_is_not_a_contract(self) -> None:
        self.mutate(
            HEADERS[0],
            C[: C.index("GENERATIVEQC_API")],
            "/** This nearby comment explains an unrelated internal detail. */\n",
        )
        with self.assertRaisesRegex(ContractError, "missing exact source-owned"):
            validate(self.root)

    def test_missing_applicable_facet_fails(self) -> None:
        self.mutate(HEADERS[0], " * @outputs Returns the example count by value.\n", "")
        with self.assertRaisesRegex(ContractError, "missing facets.*outputs"):
            validate(self.root)

    def test_placeholder_facet_fails(self) -> None:
        self.mutate(HEADERS[0], "Returns the example count by value.", "TODO.")
        with self.assertRaisesRegex(ContractError, "unusable outputs"):
            validate(self.root)

    def test_new_record_field_requires_review(self) -> None:
        self.mutate(HEADERS[1], "int count{};", "int count{};\n  double energy{};")
        with self.assertRaisesRegex(ContractError, "signature/field drift"):
            validate(self.root)

    def test_record_field_type_and_default_require_review(self) -> None:
        for change in ("double count{};", "int count{7};"):
            with self.subTest(change=change):
                path = self.root / HEADERS[1]
                path.write_text(CPP.replace("int count{};", change), encoding="utf-8")
                with self.assertRaisesRegex(ContractError, "signature/field drift"):
                    validate(self.root)

    def test_constructor_destructor_and_c_cpp_anchors_are_distinct(self) -> None:
        declarations = discover(ROOT)
        anchors = {d.name: d.anchor for d in declarations}
        self.assertNotEqual(
            anchors["generativeqc::Context::Context"],
            anchors["generativeqc::Context::~Context"],
        )
        self.assertNotEqual(
            anchors["generativeqc_batch_execute"],
            anchors["generativeqc::Batch::execute"],
        )

    def test_signature_change_requires_review(self) -> None:
        self.mutate(
            HEADERS[0], "generativeqc_example(void)", "generativeqc_example(int mode)"
        )
        with self.assertRaisesRegex(ContractError, "signature/field drift"):
            validate(self.root)

    def test_unknown_c_declaration_fails_closed(self) -> None:
        for source in (
            "GENERATIVEQC_API int generativeqc_variable;",
            "GENERATIVEQC_API void generativeqc_callback(void (*fn)(int));",
            "GENERATIVEQC_API void generativeqc_body(void) {}",
        ):
            with self.subTest(source=source), self.assertRaises(ContractError):
                scan_c(source)

    def test_unknown_cpp_declarations_fail_closed(self) -> None:
        for body in (
            "using Alias = int;",
            "template<class T> void run(T value) {}",
            "struct Record { using Value = int; };",
            "struct Record { int (*callback)(int); };",
        ):
            with self.subTest(body=body), self.assertRaises(ContractError):
                scan_cpp("namespace generativeqc { " + body + " }")

    def test_export_aliases_and_generators_fail_closed(self) -> None:
        cases = (
            "#define OTHER_EXPORT GENERATIVEQC_API\nOTHER_EXPORT void generativeqc_new(void);",
            "#define OTHER_EXPORT __declspec(dllexport)\nOTHER_EXPORT void generativeqc_new(void);",
            "#define DECLARE(name) GENERATIVEQC_API void name(void);\nDECLARE(generativeqc_new)",
            "#define DECLARE(name) void name(void);\nDECLARE(generativeqc_new)",
            "#define EMPTY_EXPORT\nEMPTY_EXPORT void generativeqc_new(void);",
            '__attribute__((visibility("default"))) void generativeqc_new(void);',
            "__declspec(dllexport) void generativeqc_new(void);",
            "extern void generativeqc_new(void);",
        )
        for declaration in cases:
            with (
                self.subTest(declaration=declaration),
                self.assertRaises(ContractError),
            ):
                scan_c(C + declaration)

    def test_hidden_friends_fail_closed(self) -> None:
        for access in ("public", "private", "protected"):
            source = (
                "namespace generativeqc { class Owner { "
                + access
                + ": friend void undocumented(Owner&) {} }; }"
            )
            with (
                self.subTest(access=access),
                self.assertRaisesRegex(ContractError, "friend"),
            ):
                scan_cpp(source)

    def test_macro_contract_accepts_existing_exports_and_ignores_decoys(self) -> None:
        for export in (
            "__declspec(dllexport)",
            "__declspec(dllimport)",
            '__attribute__((visibility("default")))',
        ):
            source = "#define GENERATIVEQC_API " + export + "\n" + C
            self.assertEqual(len(scan_c(source)), 1)
        self.assertEqual(len(scan_c(C + "\n/* #define BAD OTHER_EXPORT */")), 1)

    def test_lexer_preserves_strings_and_macros(self) -> None:
        source = (
            '#define VALUE "http://x/*literal*/"\nconst char* s = R"x(/*literal*/)x";'
        )
        self.assertEqual(
            executable_tokens(source), executable_tokens("/** comment */" + source)
        )
        self.assertIn('"http://x/*literal*/"', executable_tokens(source))
        self.assertIn('R"x(/*literal*/)x"', executable_tokens(source))
        self.assertNotEqual(
            executable_tokens(source),
            executable_tokens(source.replace("VALUE", "OTHER")),
        )

    def test_private_members_and_comment_decoys_are_exempt(self) -> None:
        self.mutate(
            HEADERS[1],
            "int count{};",
            "int count{};\nprivate:\n  void hidden() {}\npublic:",
        )
        self.mutate(
            HEADERS[0], C, C + "/* GENERATIVEQC_API void generativeqc_decoy(void); */"
        )
        self.assertEqual(len(validate(self.root)), 3)

    def test_renamed_contract_does_not_cover_another_declaration(self) -> None:
        self.mutate(
            HEADERS[0],
            "@native-contract generativeqc_example",
            "@native-contract generativeqc_other",
        )
        with self.assertRaisesRegex(ContractError, "missing exact source-owned"):
            validate(self.root)


if __name__ == "__main__":
    unittest.main()
