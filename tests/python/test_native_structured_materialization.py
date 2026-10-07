"""Native source certificates and fail-closed counterexamples; no native build."""

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.audit_structured_materialization import audit_native, audit_tree, main


def producer(body: str, setup: str = "", shape: str = "n*n*n*n") -> str:
    return f"""
std::vector<double> producer(std::size_t n, std::size_t occupied) {{
  {setup}
  std::vector<double> weights({shape}, 0.0);
  {body}
  return weights;
}}
"""


OV = """
for (std::size_t i = 0; i < occupied; ++i)
  for (std::size_t a = occupied; a < n; ++a)
    for (std::size_t j = 0; j < occupied; ++j)
      for (std::size_t b = occupied; b < n; ++b)
        weights[((i*n+a)*n+j)*n+b] += 1.0;
"""
DENSE = """
for (std::size_t i = 0; i < n; ++i)
  for (std::size_t a = 0; a < n; ++a)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t b = 0; b < n; ++b)
        weights[((i*n+a)*n+j)*n+b] = 1.0;
"""


class NativeStructuredMaterializationTests(unittest.TestCase):
    def test_occupied_virtual_block_and_independent_address_enumeration(self) -> None:
        finding = audit_native(producer(OV))[0]
        self.assertEqual(finding["classification"], "exact-structured-write-support")
        self.assertEqual(finding["dense_elements"], "n*n*n*n")
        self.assertEqual(
            finding["written_elements"], "occupied*(n-occupied)*occupied*(n-occupied)"
        )
        self.assertEqual(finding["written_growth_degree"], 4)
        self.assertFalse(finding["strict_reduction_proved"])
        for n in range(2, 8):
            for occupied in range(1, n):
                addresses = {
                    ((i * n + a) * n + j) * n + b
                    for i in range(occupied)
                    for a in range(occupied, n)
                    for j in range(occupied)
                    for b in range(occupied, n)
                }
                self.assertEqual(len(addresses), occupied**2 * (n - occupied) ** 2)
                self.assertTrue(all(0 <= index < n**4 for index in addresses))
                self.assertLess(len(addresses), n**4)

    def test_offset_partition_requires_verified_extent_relation(self) -> None:
        body = (
            OV.replace("a = occupied; a < n", "a = 0; a < virtuals")
            .replace("b = occupied; b < n", "b = 0; b < virtuals")
            .replace("((i*n+a)*n+j)*n+b", "((i*n+(occupied+a))*n+j)*n+(occupied+b)")
        )
        valid = producer(body, "const auto virtuals = n - occupied;")
        finding = audit_native(valid)[0]
        self.assertEqual(
            finding["written_elements"], "occupied*virtuals*occupied*virtuals"
        )
        self.assertEqual(finding["classification"], "exact-structured-write-support")
        invalid = valid.replace("n - occupied", "n + occupied")
        self.assertEqual(audit_native(invalid)[0]["classification"], "unknown")

    def test_genuine_dense_writes_not_mislabeled(self) -> None:
        finding = audit_native(producer(DENSE))[0]
        self.assertEqual(finding["classification"], "dense-write-domain")
        self.assertIsNone(finding["recommendation"])
        self.assertIsNone(finding["expansion_ratio"])
        for n in range(1, 5):
            addresses = {
                ((i * n + a) * n + j) * n + b
                for i in range(n)
                for a in range(n)
                for j in range(n)
                for b in range(n)
            }
            self.assertEqual(addresses, set(range(n**4)))

    def test_chained_immutable_extent_aliases_do_not_hide_dense_writes(self) -> None:
        for setup in (
            "const auto all = n;",
            "const auto first = n; const auto all = first;",
            "const auto all = n + 0;",
            "const auto all = n * 1;",
            "const auto all = n + n - n;",
            "const auto all = 2*n - n;",
        ):
            with self.subTest(setup=setup):
                finding = audit_native(producer(DENSE.replace("< n", "< all"), setup))[
                    0
                ]
                self.assertEqual(finding["classification"], "dense-write-domain")
                self.assertIn("all", finding["extent_aliases"])
                self.assertIsNone(finding["recommendation"])

    def test_zero_offset_alias_does_not_hide_a_dense_write_domain(self) -> None:
        body = (
            "for (std::size_t i=0; i<extent; ++i) "
            "for (std::size_t j=0; j<n; ++j) weights[(zero+i)*n+j] = 1;"
        )
        for setup in (
            "const auto zero = 0; const auto extent = n - zero;",
            "const auto first = 0; const auto zero = first; const auto extent = n-zero;",
            "const auto zero = n+n-2*n; const auto extent = n-zero;",
        ):
            with self.subTest(setup=setup):
                finding = audit_native(producer(body, setup, shape="n*n"))[0]
                self.assertEqual(finding["classification"], "dense-write-domain")
                self.assertIsNone(finding["recommendation"])
                self.assertIsNone(finding["expansion_ratio"])
                self.assertEqual(finding["domains"][0]["axes"][0]["offset"], "0")
        for n in (1, 2, 4, 7):
            zero = 0
            addresses = {(zero + i) * n + j for i in range(n - zero) for j in range(n)}
            self.assertEqual(addresses, set(range(n * n)))

    def test_zero_offset_alias_preserves_lower_triangle(self) -> None:
        source = producer(
            "for (std::size_t i=0; i<extent; ++i) "
            "for (std::size_t j=0; j<=i; ++j) weights[(zero+i)*n+j] = 1;",
            "const auto zero = n-n; const auto extent = n-zero;",
            shape="n*n",
        )
        finding = audit_native(source)[0]
        self.assertEqual(finding["domains"][0]["kind"], "lower-triangle")
        self.assertEqual(finding["written_elements"], "n*(n+1)/2")

    def test_fixed_extent_write_and_allocation_have_constant_growth(self) -> None:
        source = producer(
            "for (std::size_t i=0; i<small; ++i) weights[i*n+i] = 1;",
            "const auto small = 3;",
            shape="n*n",
        )
        finding = audit_native(source)[0]
        self.assertEqual(finding["written_growth_degree"], 0)

        cancelled = source.replace("small = 3", "small = n*n - n*n + 3")
        finding = audit_native(cancelled)[0]
        self.assertEqual(finding["written_growth_degree"], 0)
        self.assertEqual(finding["dense_growth_degree"], 2)
        self.assertEqual(finding["extent_aliases"], {"small": "3"})
        for n in (3, 5, 11):
            addresses = {i * n + i for i in range(3)}
            self.assertEqual(len(addresses), 3)
            self.assertTrue(all(0 <= index < n * n for index in addresses))
        fixed = source.replace("n*n, 0.0", "small*small, 0.0").replace(
            "i*n+i", "i*small+i"
        )
        finding = audit_native(fixed)[0]
        self.assertEqual(finding["dense_growth_degree"], 0)
        self.assertEqual(finding["written_growth_degree"], 0)

    def test_triangle_bound_alias_preserves_count_and_quadratic_growth(self) -> None:
        source = producer(
            "for (std::size_t i=0; i<all; ++i) for (std::size_t j=0; j<=i; ++j) weights[i*n+j] = 1;",
            "const auto all = n;",
            shape="n*n",
        )
        finding = audit_native(source)[0]
        self.assertEqual(finding["domains"][0]["kind"], "lower-triangle")
        self.assertEqual(finding["written_elements"], "n*(n+1)/2")
        self.assertEqual(finding["written_growth_degree"], 2)

    def test_lower_triangle_retains_quadratic_growth(self) -> None:
        source = producer(
            """
for (std::size_t i=0; i<n; ++i) {
  for (std::size_t j=0; j<=i; ++j) {
    weights[i*n+j] = 1;
  }
}
""",
            shape="n*n",
        )
        finding = audit_native(source)[0]
        self.assertEqual(finding["domains"][0]["kind"], "lower-triangle")
        self.assertEqual(finding["written_elements"], "n*(n+1)/2")
        self.assertEqual(finding["written_growth_degree"], 2)
        self.assertEqual(finding["dense_growth_degree"], 2)
        self.assertEqual(finding["strict_reduction_conditions"], ["n > 1"])
        for n in range(1, 8):
            self.assertEqual(
                len({i * n + j for i in range(n) for j in range(i + 1)}),
                n * (n + 1) // 2,
            )

    def test_diagonal_has_linear_address_support(self) -> None:
        finding = audit_native(
            producer("for (std::size_t i=0; i<n; ++i) weights[i*n+i] = 1;", shape="n*n")
        )[0]
        self.assertEqual(finding["written_elements"], "n")
        self.assertEqual(finding["written_growth_degree"], 1)

    def test_arithmetic_helpers_are_verified_not_named_oracles(self) -> None:
        helpers = """
std::size_t square(std::size_t x) { return x*x; }
std::size_t fourth(std::size_t x) { return square(square(x)); }
std::size_t index(std::size_t n, std::size_t i, std::size_t a, std::size_t j, std::size_t b) {
  return ((i*n+a)*n+j)*n+b;
}
"""
        body = OV.replace("((i*n+a)*n+j)*n+b", "index(n,i,a,j,b)")
        finding = audit_native(helpers + producer(body, shape="fourth(n)"))[0]
        self.assertEqual(finding["classification"], "exact-structured-write-support")
        for bad in [
            helpers.replace("return x*x;", "touch(); return x*x;"),
            helpers + "std::size_t fourth(int x) { return x*x; }",
            helpers.replace("return ((i*n+a)*n+j)*n+b;", "return n*n*n*n-1;"),
        ]:
            self.assertEqual(
                audit_native(bad + producer(body, shape="fourth(n)"))[0][
                    "classification"
                ],
                "unknown",
            )

    def test_alias_calls_mutations_control_flow_and_bad_indices_fail_closed(
        self,
    ) -> None:
        cases = [
            "auto& alias = weights; " + OV,
            "auto pointer = weights.data(); " + OV,
            "mutate(weights); " + OV,
            "mutate(n); " + OV,
            "n = occupied; " + OV,
            "weights.resize(n*n*n*n); " + OV,
            "weights.assign(n*n*n*n, 1.0); " + OV,
            "if (occupied) { " + OV + " }",
            "if (occupied) return weights; " + OV,
            OV.replace("+= 1.0", "+= compute(n)"),
            OV.replace("+= 1.0", "+= ++i"),
            OV.replace("+= 1.0", "+= (n = occupied)"),
            OV.replace("++i", "i += 2"),
            OV.replace("++i", "++n"),
            OV.replace("((i*n+a)*n+j)*n+b", "0"),
            OV.replace("((i*n+a)*n+j)*n+b", "n*n*n*n-1"),
            OV.replace("a < n", "a <= n"),
            OV.replace("a = occupied", "a = n"),
            OV.replace("+= 1.0", "= i++"),
            "while (n) { " + OV + " }",
            OV + "consume(std::move(weights));",
            "auto callback = [&]() { " + OV + " }; callback();",
            "for (std::size_t tile=0; tile<n; ++tile) { " + OV + " }",
        ]
        for body in cases:
            with self.subTest(body=body):
                findings = audit_native(producer(body))
                self.assertTrue(findings)
                self.assertTrue(all(f["classification"] == "unknown" for f in findings))
                self.assertTrue(all(f["recommendation"] is None for f in findings))

    def test_nonfresh_or_aggregate_assign_is_inventory_only(self) -> None:
        sources = [
            producer(OV).replace(
                "std::vector<double> weights", "static std::vector<double> weights"
            ),
            "void producer(int n) { result.two_electron.assign(fourth_power(n), 0.0); }",
            "void producer(int n) { std::vector<double> weights(n); weights.assign(n*n, 0.0); }",
            producer(OV).replace(
                "std::vector<double> weights",
                "thread_local std::vector<double> weights",
            ),
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(audit_native(source)[0]["classification"], "unknown")

    def test_fresh_local_default_vector_assign_is_supported(self) -> None:
        source = producer(OV).replace(
            "std::vector<double> weights(n*n*n*n, 0.0);",
            "std::vector<double> weights; weights.assign(n*n*n*n, 0.0);",
        )
        self.assertEqual(
            audit_native(source)[0]["classification"], "exact-structured-write-support"
        )

    def test_declared_overload_and_foreign_namespace_are_not_index_proofs(self) -> None:
        helper = "std::size_t square(std::size_t x) { return x*x; }"
        sources = [
            helper
            + "std::size_t square(int x);"
            + producer(OV, shape="square(square(n))"),
            helper + "double square(int x);" + producer(OV, shape="square(square(n))"),
            "namespace other {"
            + helper
            + "}"
            + producer(OV, shape="square(square(n))"),
        ]
        for source in sources:
            self.assertEqual(audit_native(source)[0]["classification"], "unknown")

    def test_helper_cannot_capture_a_global_with_same_caller_dimension_name(
        self,
    ) -> None:
        source = """
std::size_t n = 100;
std::size_t index(std::size_t i, std::size_t j) { return i*n+j; }
""" + producer("for (std::size_t i=0; i<n; ++i) weights[index(i,i)] = 1;", shape="n*n")
        self.assertEqual(audit_native(source)[0]["classification"], "unknown")

    def test_unsupported_setup_preprocessor_and_shape_are_unknown(self) -> None:
        for source in [
            producer(OV, "escape();"),
            "#define INDEX(x) x\n" + producer(OV),
            "#if ENABLED\n" + producer(OV) + "#endif\n",
            producer(OV, shape="unknown_count(n)"),
            producer(OV, shape="n*occupied*n*n"),
            producer(OV).replace("std::size_t n", "const Custom& n"),
            producer(OV, "const auto n = occupied;"),
        ]:
            with self.subTest(source=source):
                self.assertEqual(audit_native(source)[0]["classification"], "unknown")

    def test_comments_literals_and_nonzero_initialization_not_allocations(self) -> None:
        self.assertEqual(audit_native("// " + producer(OV).replace("\n", "\n// ")), [])
        self.assertEqual(
            audit_native('void f() { const char* x = "weights.assign(n*n,0.0);"; }'), []
        )
        self.assertEqual(audit_native(producer(OV).replace("0.0", "1.0")), [])

    def test_multiple_domains_do_not_claim_strict_union_or_savings(self) -> None:
        finding = audit_native(producer(OV + OV.replace("i < occupied", "i < n")))[0]
        self.assertEqual(finding["classification"], "write-domain-union")
        self.assertEqual(finding["written_count_kind"], "union-upper-bound")
        self.assertIsNone(finding["recommendation"])
        self.assertIsNone(finding["expansion_ratio"])

    def test_real_mp2_source_candidate_does_not_claim_production_defect(self) -> None:
        root = Path(__file__).resolve().parents[2]
        report = audit_tree(root, ("src/posthf/mp2_gradient.cpp",))
        finding = next(
            f
            for f in report["findings"]
            if f["function"] == "initial_orbital_weights"
            and f["buffer"] == "result.two_electron"
        )
        self.assertEqual(finding["allocation_expression"], "fourth_power(n)")
        self.assertEqual(finding["classification"], "unknown")
        self.assertIn("assign freshness", finding["unknown_reason"])
        self.assertIsNone(finding["recommendation"])
        self.assertTrue(report["provenance"]["source_hashes"])
        self.assertEqual(len(report["provenance"]["scanner_hashes"]), 3)

    def _symlink(self, link: Path, target: Path) -> None:
        try:
            link.symlink_to(target, target_is_directory=target.is_dir())
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"source symlinks unavailable: {error}")

    def test_recursive_outside_root_source_link_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            root = folder / "root"
            source_dir = root / "src"
            source_dir.mkdir(parents=True)
            outside = folder / "outside.cpp"
            outside.write_text(producer(OV), encoding="utf-8")
            self._symlink(source_dir / "leak.cpp", outside)
            with self.assertRaises(ValueError):
                audit_tree(root, ("src",))

    def test_recursive_same_root_source_aliases_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "src"
            source_dir.mkdir()
            source = source_dir / "actual.cpp"
            source.write_text(producer(OV), encoding="utf-8")
            self._symlink(source_dir / "alias.cpp", source)
            report = audit_tree(root, ("src", "src/alias.cpp"))
            self.assertEqual(report["scanned_files"], 1)
            self.assertEqual(len(report["findings"]), 1)
            self.assertEqual(report["findings"][0]["path"], "src/actual.cpp")
            self.assertEqual(
                report["provenance"]["source_hashes"],
                {"src/actual.cpp": hashlib.sha256(source.read_bytes()).hexdigest()},
            )

    def test_recursive_alias_of_excluded_native_source_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = root / "src" / "xtb" / "native"
            vendor.mkdir(parents=True)
            source = vendor / "vendor.cpp"
            source.write_text(producer(OV), encoding="utf-8")
            self._symlink(root / "src" / "alias.cpp", source)
            report = audit_tree(root, ("src", "src/alias.cpp"))
            self.assertEqual(report["scanned_files"], 0)
            self.assertEqual(report["findings"], [])
            self.assertEqual(report["provenance"]["source_hashes"], {})

    def _git(self, root: Path, *args: str) -> None:
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "core.symlinks=true",
                "-c",
                "user.name=Provenance fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_alias_scan_detects_modified_untracked_and_ignored_canonical_sources(
        self,
    ) -> None:
        for state in ("modified", "untracked", "ignored"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "actual.cpp"
                source.write_text(producer(OV), encoding="utf-8")
                self._symlink(root / "alias.cpp", source)
                self._git(root, "init", "-q")
                self._git(root, "add", "--", "alias.cpp")
                if state == "modified":
                    self._git(root, "add", "--", "actual.cpp")
                elif state == "ignored":
                    (root / ".gitignore").write_text("actual.cpp\n", encoding="utf-8")
                    self._git(root, "add", "--", ".gitignore")
                self._git(root, "commit", "-qm", "provenance fixture")
                if state == "modified":
                    source.write_text(producer(OV) + "\n// changed\n", encoding="utf-8")
                report = audit_tree(root, ("alias.cpp",))
                self.assertIs(report["provenance"]["scanned_source_dirty"], True)
                self.assertEqual(
                    report["provenance"]["source_hashes"]["actual.cpp"],
                    hashlib.sha256(source.read_bytes()).hexdigest(),
                )

    def test_clean_ordinary_source_is_false_but_alias_topology_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "actual.cpp"
            source.write_text(producer(OV), encoding="utf-8")
            self._symlink(root / "alias.cpp", source)
            self._symlink(root / "chain.cpp", root / "alias.cpp")
            self._git(root, "init", "-q")
            self._git(root, "add", "--", "actual.cpp", "alias.cpp", "chain.cpp")
            self._git(root, "commit", "-qm", "provenance fixture")
            ordinary = audit_tree(root, ("actual.cpp",))["provenance"]
            self.assertIs(ordinary["scanned_source_dirty"], False)
            for selected in ("alias.cpp", "chain.cpp"):
                with self.subTest(selected=selected):
                    alias = audit_tree(root, (selected,))["provenance"]
                    self.assertIsNone(alias["scanned_source_dirty"])
                    self.assertIs(alias["alias_topology_unverified"], True)

    def test_broken_recursive_source_link_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "src"
            source_dir.mkdir()
            self._symlink(source_dir / "broken.cpp", source_dir / "missing.cpp")
            with self.assertRaises(ValueError):
                audit_tree(root, ("src",))

    def test_directory_alias_detects_deleted_canonical_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical"
            canonical.mkdir()
            source = canonical / "actual.cpp"
            source.write_text(producer(OV), encoding="utf-8")
            self._symlink(root / "directory-alias", canonical)
            self._git(root, "init", "-q")
            self._git(root, "add", "--", "canonical", "directory-alias")
            self._git(root, "commit", "-qm", "provenance fixture")
            source.unlink()
            report = audit_tree(root, ("directory-alias",))
            self.assertEqual(report["scanned_files"], 0)
            self.assertIs(report["provenance"]["scanned_source_dirty"], True)

    def test_unscanned_ignored_files_do_not_taint_clean_source_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "src"
            source_dir.mkdir()
            (source_dir / "actual.cpp").write_text(producer(OV), encoding="utf-8")
            (source_dir / "ignored.log").write_text("unused", encoding="utf-8")
            (root / ".gitignore").write_text("/src/ignored.log\n", encoding="utf-8")
            self._git(root, "init", "-q")
            self._git(root, "add", "--", "src/actual.cpp", ".gitignore")
            self._git(root, "commit", "-qm", "provenance fixture")
            report = audit_tree(root, ("src",))
            self.assertEqual(
                set(report["provenance"]["source_hashes"]), {"src/actual.cpp"}
            )
            self.assertIs(report["provenance"]["scanned_source_dirty"], False)

    def test_missing_git_preserves_hashes_with_unknown_git_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "actual.cpp"
            source.write_text(producer(OV), encoding="utf-8")
            with patch(
                "tools.audit_structured_materialization.subprocess.run",
                side_effect=FileNotFoundError("git unavailable"),
            ):
                provenance = audit_tree(root, ("actual.cpp",))["provenance"]
            for field in (
                "commit",
                "tree",
                "working_tree_dirty",
                "scanned_source_dirty",
            ):
                self.assertIsNone(provenance[field])
            self.assertEqual(
                provenance["source_hashes"]["actual.cpp"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )

    def test_cli_identities_duplicate_inputs_and_missing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.cpp"
            source.write_text(producer(OV), encoding="utf-8")
            output = root / "scan.json"
            self.assertEqual(
                main(
                    [
                        "--root",
                        str(root),
                        "--path",
                        "fixture.cpp",
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["scanned_files"], 1)
            self.assertIsNone(report["provenance"]["scanned_source_dirty"])
            self.assertIs(report["provenance"]["alias_topology_unverified"], False)
            self.assertEqual(
                report["provenance"]["source_hashes"]["fixture.cpp"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                audit_tree(root, ("fixture.cpp", "fixture.cpp"))["scanned_files"], 1
            )
            with self.assertRaises(ValueError):
                audit_tree(root, ("missing.cpp",))
            with self.assertRaises(ValueError):
                audit_tree(root, ("../outside",))


if __name__ == "__main__":
    unittest.main()
