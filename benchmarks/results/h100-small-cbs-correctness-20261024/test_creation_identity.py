"""Real CLI value shape and external creation-snapshot binding, HOST only."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import finalize
import test_binding
import transport


class CreationIdentity(unittest.TestCase):
    def fixture(self) -> tuple[test_binding.AllocationBinding, SimpleNamespace]:
        case = test_binding.AllocationBinding("test_publication_is_no_clobber")
        case.setUp()
        self.addCleanup(case.doCleanups)
        args = case.arguments()
        args.expected_created_at = "1"
        args.submission_not_before_ms = "1"
        args.submission_not_after_ms = "2"
        return case, args

    def test_real_cli_decimal_string_preserves_original_type(self) -> None:
        _, args = self.fixture()
        status = transport.read(args.status)
        status["data"]["created_at"] = "1792845081000"
        args.status.write_text(json.dumps(status), encoding="utf-8")
        args.expected_created_at = "1792845081000"
        args.submission_not_before_ms = "1792845080999"
        args.submission_not_after_ms = "1792845081001"
        original = args.status.read_bytes()
        with patch.object(
            finalize,
            "helper",
            return_value=SimpleNamespace(bind_allocated_instance=lambda *a: None),
        ):
            finalize.finalize(args)
        proof = transport.read(args.proof)
        approval = transport.read(args.release)
        self.assertEqual(
            proof["raw_status_response"]["data"]["created_at"], "1792845081000"
        )
        self.assertEqual(approval["allocation_created_at_raw"], "1792845081000")
        self.assertEqual(args.status.read_bytes(), original)

    def test_canonical_positive_formats(self) -> None:
        for value in (1, "1", 1792845081000, "1792845081000"):
            self.assertEqual(transport.creation_identity(value), str(value))

    def test_noncanonical_or_ambiguous_values_refuse(self) -> None:
        for value in (
            True,
            False,
            1.0,
            -1,
            0,
            "",
            "0",
            "01",
            "+1",
            "-1",
            " 1",
            "1 ",
            "１",
            "1.0",
            str(1 << 63),
            1 << 63,
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                transport.creation_identity(value)

    def test_external_creation_snapshot_mismatch_refuses(self) -> None:
        _, args = self.fixture()
        args.expected_created_at = "2"
        with (
            patch.object(
                finalize,
                "helper",
                return_value=SimpleNamespace(bind_allocated_instance=lambda *a: None),
            ),
            self.assertRaises(ValueError),
        ):
            finalize.finalize(args)
        self.assertFalse(args.release.exists())

    def test_external_submission_window_refuses_old_creation(self) -> None:
        _, args = self.fixture()
        args.submission_not_before_ms = "2"
        args.submission_not_after_ms = "3"
        with (
            patch.object(
                finalize,
                "helper",
                return_value=SimpleNamespace(bind_allocated_instance=lambda *a: None),
            ),
            self.assertRaises(ValueError),
        ):
            finalize.finalize(args)
        self.assertFalse(args.release.exists())


if __name__ == "__main__":
    unittest.main()
