"""CPU-only MP2 source boundary census; never treats a dense path as production."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.audit_structured_materialization import (
    MP2_REPRESENTATION_SOURCE,
    audit_mp2_representation_boundary,
    audit_native,
    audit_tree,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / MP2_REPRESENTATION_SOURCE


class MP2RepresentationSourceTests(unittest.TestCase):
    def test_real_canonical_and_factorized_owners_are_visible_without_sparsity(self) -> (
        None
    ):
        source = SOURCE.read_text(encoding="utf-8")
        observed = audit_mp2_representation_boundary(source)
        self.assertEqual(observed["status"], "SOURCE_VISIBLE")
        self.assertEqual(observed["source_sha256"], hashlib.sha256(source.encode()).hexdigest())
        self.assertEqual(len(observed["observed_source_roles"]), 8)
        self.assertEqual(observed["missing_roles"], [])
        self.assertEqual(
            {row["role"] for row in observed["observed_source_roles"]},
            {
                "checked-square-helper",
                "rank-four-extent-helper",
                "dense-canonical-RHS-allocation",
                "canonical-RHS-caller",
                "streamed-RHS-factor-owner",
                "streamed-RHS-caller",
                "factorized-Lagrangian-owner",
                "RI-reverse-factorized-consumer",
            },
        )
        for key in (
            "exact_write_support_proven",
            "runtime_endpoint_selection_proven",
            "consumer_abi_verified",
        ):
            self.assertIs(observed[key], False)

        # The generic closed lexical certifier remains deliberately unable
        # to prove the struct-member source's exact write support.
        dense = [
            row
            for row in audit_native(source, MP2_REPRESENTATION_SOURCE)
            if row["function"] == "initial_orbital_weights"
            and row["buffer"] == "result.two_electron"
        ]
        self.assertEqual(len(dense), 1)
        self.assertEqual(dense[0]["classification"], "unknown")
        self.assertIsNone(dense[0]["recommendation"])

    def test_deleted_or_comment_only_dense_path_is_not_source_confirmed(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        needle = "result.two_electron.assign(fourth_power(n), 0.0);"
        self.assertIn(needle, source)
        changed = source.replace(needle, "result.two_electron.clear();")
        changed = "// " + needle + "\n" + changed
        evidence = audit_mp2_representation_boundary(changed)
        self.assertEqual(evidence["status"], "INCOMPLETE")
        self.assertTrue(
            any("dense-canonical-RHS-allocation" in item for item in evidence["missing_roles"])
        )
        self.assertFalse(evidence["exact_write_support_proven"])

    def test_changed_factorized_path_is_explicitly_unknown(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        needle = "result.two_electron_factors.correlation_iajb = std::move(adjoint.integrals_iajb);"
        self.assertIn(needle, source)
        changed = source.replace(needle, "result.two_electron_factors.correlation_iajb.clear();")
        evidence = audit_mp2_representation_boundary(changed)
        self.assertEqual(evidence["status"], "INCOMPLETE")
        self.assertTrue(
            any("factorized-Lagrangian-owner" in item for item in evidence["missing_roles"])
        )

    def test_tree_report_attaches_provenance_boundaries_only_for_selected_source(self) -> (
        None
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / MP2_REPRESENTATION_SOURCE
            path.parent.mkdir(parents=True)
            path.write_bytes(SOURCE.read_bytes())
            report = audit_tree(root, (MP2_REPRESENTATION_SOURCE,))
            self.assertEqual(report["scanned_files"], 1)
            self.assertEqual(len(report["production_boundaries"]), 1)
            self.assertEqual(report["production_boundaries"][0]["status"], "SOURCE_VISIBLE")
            self.assertEqual(report["provenance"]["source_hashes"][MP2_REPRESENTATION_SOURCE], hashlib.sha256(SOURCE.read_bytes()).hexdigest())
            report_other = audit_tree(root, ("src",))
            self.assertEqual(len(report_other["production_boundaries"]), 1)


if __name__ == "__main__":
    unittest.main()
