"""Adversarial receipt fixtures; these are not runtime/CUDA qualification evidence."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.audit_replay_allocations import (
    SCHEMA,
    assess_native_device_ledger,
    main,
    verify_receipt,
)


def fixture() -> tuple[dict, dict]:
    identity = {
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "library_sha256": "3" * 64,
        "artifact_sha256": "4" * 64,
        "workload_sha256": "5" * 64,
        "toolchain": "fixture compiler; not execution evidence",
        "device": "fixture CUDA UUID / visible ordinal 0",
        "endpoint": "Calculator.prepare_batch(rhf,direct).execute(energy)",
    }
    domains = [
        {"owner": "hf", "space": "host", "counter": "fixture-host-journal"},
        {"owner": "hf", "space": "device:0", "counter": "fixture-device-journal"},
    ]
    schedule = [
        {"id": "prepare", "phase": "setup", "zero_new_allocations": False},
        {"id": "cold", "phase": "endpoint", "zero_new_allocations": False},
        {"id": "warm", "phase": "replay", "zero_new_allocations": True},
        {"id": "publish", "phase": "publication", "zero_new_allocations": False},
    ]
    expected = {
        "schema": SCHEMA,
        "identity": identity,
        "domains": domains,
        "windows": schedule,
    }
    receipt = {
        "schema": SCHEMA,
        "identity": copy.deepcopy(identity),
        # Deliberately exercises the runtime branch; fixture use is test-only.
        "execution": {"kind": "runtime", "completed": True, "source_matched": True},
        "windows": [],
    }
    for index, window in enumerate(schedule):
        observations = []
        for domain in domains:
            setup = index == 0
            event = {
                "sequence": 0,
                "kind": "allocate",
                "allocation_id": "arena",
                "requested_bytes": 64,
            }
            observations.append(
                {
                    "domain": copy.deepcopy(domain),
                    "coverage": {
                        "complete": True,
                        "started_before_window": True,
                        "ended_after_window": True,
                        "synchronized": True,
                        "dropped_events": 0,
                    },
                    "initial_live": {} if setup else {"arena": 64},
                    "event_begin": 0 if setup else 1,
                    "event_end": 1,
                    "events": [event] if setup else [],
                    "metrics": {
                        "allocation_count": int(setup),
                        "requested_bytes": 64 if setup else 0,
                        "peak_live_bytes": 64,
                        "live_bytes": 64,
                    },
                }
            )
        receipt["windows"].append(
            {"id": window["id"], "phase": window["phase"], "observations": observations}
        )
    return receipt, expected


def ledger_fixture() -> dict:
    return {
        "device": 0,
        "owner": "hf",
        "scope": "owned CUDA buffers only",
        "limit_bytes": 128,
        "live_bytes": 64,
        "peak_bytes": 64,
        "allocations": 0,
        "rejected_allocations": 0,
    }


class ReplayAllocationAuditTests(unittest.TestCase):
    def assert_fails(self, receipt: dict, expected: dict, message: str) -> None:
        result = verify_receipt(receipt, expected)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(message, result["errors"][0])
        self.assertEqual(result["windows"], [])

    def test_setup_allocations_are_not_hot_and_peak_is_not_requested_bytes(
        self,
    ) -> None:
        receipt, expected = fixture()
        result = verify_receipt(receipt, expected)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["windows"][0]["observations"][0]["metrics"]["allocation_count"], 1
        )
        self.assertEqual(
            result["windows"][2]["observations"][0]["metrics"],
            {
                "allocation_count": 0,
                "requested_bytes": 0,
                "peak_live_bytes": 64,
                "live_bytes": 64,
            },
        )

    def test_observed_allocation_events_in_each_hot_role_fail(self) -> None:
        for phase in ("replay", "iteration", "tile"):
            for space_index in (0, 1):
                with self.subTest(phase=phase, space_index=space_index):
                    receipt, expected = fixture()
                    expected["windows"][2]["phase"] = phase
                    receipt["windows"][2]["phase"] = phase
                    observed = receipt["windows"][2]["observations"][space_index]
                    observed["events"] = [
                        {
                            "sequence": 1,
                            "kind": "allocate",
                            "allocation_id": "temporary",
                            "requested_bytes": 32,
                        },
                        {
                            "sequence": 2,
                            "kind": "release",
                            "allocation_id": "temporary",
                            "requested_bytes": 32,
                        },
                    ]
                    observed["event_end"] = 3
                    observed["metrics"].update(
                        allocation_count=1, requested_bytes=32, peak_live_bytes=96
                    )
                    self.assert_fails(
                        receipt, expected, "hot replay made new allocations"
                    )

    def test_transient_allocations_cannot_hide_in_unchanged_live_bytes(self) -> None:
        receipt, expected = fixture()
        observed = receipt["windows"][2]["observations"][1]
        observed["metrics"]["allocation_count"] = 1
        self.assert_fails(receipt, expected, "metrics disagree")

    def test_releases_reconcile_peak_and_ownership_across_windows(self) -> None:
        receipt, expected = fixture()
        observed = receipt["windows"][-1]["observations"][1]
        observed["events"] = [
            {
                "sequence": 1,
                "kind": "release",
                "allocation_id": "arena",
                "requested_bytes": 64,
            }
        ]
        observed["event_end"] = 2
        observed["metrics"]["live_bytes"] = 0
        self.assertEqual(verify_receipt(receipt, expected)["status"], "PASS")
        observed["events"][0]["requested_bytes"] = 63
        self.assert_fails(receipt, expected, "release ownership/size mismatch")

    def test_requested_bytes_are_sum_not_peak_or_count(self) -> None:
        receipt, expected = fixture()
        observed = receipt["windows"][-1]["observations"][0]
        observed["events"] = [
            {
                "sequence": 1,
                "kind": "allocate",
                "allocation_id": "tmp",
                "requested_bytes": 16,
            },
            {
                "sequence": 2,
                "kind": "release",
                "allocation_id": "tmp",
                "requested_bytes": 16,
            },
            {
                "sequence": 3,
                "kind": "allocate",
                "allocation_id": "tmp",
                "requested_bytes": 16,
            },
            {
                "sequence": 4,
                "kind": "release",
                "allocation_id": "tmp",
                "requested_bytes": 16,
            },
        ]
        observed["event_end"] = 5
        observed["metrics"].update(
            allocation_count=2, requested_bytes=32, peak_live_bytes=80
        )
        self.assertEqual(verify_receipt(receipt, expected)["status"], "PASS")
        for wrong in (2, 80, 64):
            observed["metrics"]["requested_bytes"] = wrong
            self.assert_fails(receipt, expected, "metrics disagree")

    def test_zero_byte_allocation_still_has_nonzero_count(self) -> None:
        receipt, expected = fixture()
        observed = receipt["windows"][2]["observations"][0]
        observed["events"] = [
            {
                "sequence": 1,
                "kind": "allocate",
                "allocation_id": "empty",
                "requested_bytes": 0,
            }
        ]
        observed["event_end"] = 2
        observed["metrics"]["allocation_count"] = 1
        self.assert_fails(receipt, expected, "hot replay made new allocations")

    def test_unknown_resize_capacity_is_unknown_not_definitely_allocation(self) -> None:
        receipt, expected = fixture()
        observed = receipt["windows"][2]["observations"][0]
        observed["events"] = [
            {
                "sequence": 1,
                "kind": "resize",
                "allocation_id": "arena",
                "requested_bytes": 64,
            }
        ]
        observed["event_end"] = 2
        self.assert_fails(receipt, expected, "capacity uncertainty")

    def test_identity_mismatch_is_not_source_matched_evidence(self) -> None:
        for key in fixture()[0]["identity"]:
            with self.subTest(key=key):
                receipt, expected = fixture()
                original = receipt["identity"][key]
                receipt["identity"][key] = (
                    "f" * len(original)
                    if key.endswith(("sha256", "commit", "tree"))
                    else "other"
                )
                self.assert_fails(receipt, expected, "identity mismatch")

    def test_unobserved_incomplete_and_dropped_events_fail(self) -> None:
        for key in (
            "complete",
            "started_before_window",
            "ended_after_window",
            "synchronized",
            "dropped_events",
        ):
            with self.subTest(key=key):
                receipt, expected = fixture()
                coverage = receipt["windows"][2]["observations"][0]["coverage"]
                coverage[key] = 1 if key == "dropped_events" else False
                self.assert_fails(
                    receipt,
                    expected,
                    "dropped events"
                    if key == "dropped_events"
                    else "incomplete window",
                )

    def test_missing_metric_and_invalid_numbers_never_default_to_zero(self) -> None:
        for key in (
            "allocation_count",
            "requested_bytes",
            "peak_live_bytes",
            "live_bytes",
        ):
            for invalid in (None, False, -1, 0.0, "0", 1 << 64):
                with self.subTest(key=key, invalid=invalid):
                    receipt, expected = fixture()
                    receipt["windows"][2]["observations"][0]["metrics"][key] = invalid
                    self.assert_fails(receipt, expected, "expected uint64")
            receipt, expected = fixture()
            del receipt["windows"][2]["observations"][0]["metrics"][key]
            self.assert_fails(receipt, expected, "missing/unknown fields")

    def test_missing_duplicate_wrong_owner_or_counter_fail(self) -> None:
        receipt, expected = fixture()
        receipt["windows"][2]["observations"].pop()
        self.assert_fails(receipt, expected, "missing ownership observation")
        for key in ("owner", "counter", "space"):
            receipt, expected = fixture()
            receipt["windows"][2]["observations"][0]["domain"][key] = (
                "device:0" if key == "space" else "other"
            )
            self.assert_fails(receipt, expected, "wrong observation domain")
        receipt, expected = fixture()
        expected["domains"].append(copy.deepcopy(expected["domains"][0]))
        self.assert_fails(receipt, expected, "duplicate ownership domain")

    def test_phase_missing_relabelled_or_reordered_fail(self) -> None:
        for action in ("missing", "relabelled", "reordered"):
            receipt, expected = fixture()
            if action == "missing":
                receipt["windows"].pop(2)
                message = "phase windows"
            elif action == "relabelled":
                receipt["windows"][2]["phase"] = "setup"
                message = "wrong phase"
            else:
                receipt["windows"].reverse()
                message = "wrong phase"
            self.assert_fails(receipt, expected, message)

    def test_lost_or_reordered_event_and_interwindow_changes_fail(self) -> None:
        for action in ("lost", "reordered", "gap", "live"):
            receipt, expected = fixture()
            if action == "lost":
                receipt["windows"][0]["observations"][0]["events"].clear()
                message = "lost event range"
            elif action == "reordered":
                receipt["windows"][0]["observations"][0]["events"][0]["sequence"] = 1
                message = "lost/reordered event"
            else:
                observed = receipt["windows"][2]["observations"][0]
                if action == "gap":
                    observed["event_begin"] = observed["event_end"] = 2
                else:
                    observed["initial_live"] = {"different": 64}
                message = "inter-window"
            self.assert_fails(receipt, expected, message)

    def test_static_synthetic_unexecuted_and_incomplete_endpoints_fail(self) -> None:
        for key, value in (
            ("kind", "static"),
            ("kind", "synthetic"),
            ("completed", False),
            ("source_matched", False),
        ):
            receipt, expected = fixture()
            receipt["execution"][key] = value
            self.assert_fails(
                receipt,
                expected,
                "runtime receipt" if key == "kind" else "endpoint incomplete",
            )

    def test_contract_requires_explicit_hot_assertion(self) -> None:
        receipt, expected = fixture()
        expected["windows"][2]["zero_new_allocations"] = False
        self.assert_fails(receipt, expected, "explicit hot")
        expected["windows"][0]["zero_new_allocations"] = True
        self.assert_fails(receipt, expected, "requires hot phase")

    def test_native_ledger_preserves_known_values_without_zero_claim(self) -> None:
        for allocations in (0, 3):
            ledger = ledger_fixture()
            ledger["allocations"] = allocations
            result = assess_native_device_ledger(ledger)
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["allocation_count"], allocations)
            self.assertEqual(result["peak_live_bytes"], 64)
            self.assertIsNone(result["requested_bytes"])
        ledger = ledger_fixture()
        del ledger["allocations"]
        self.assertEqual(assess_native_device_ledger(ledger)["status"], "FAIL")

    def test_cli_pass_fail_incomplete_and_duplicate_json_fields(self) -> None:
        receipt, expected = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            contract = Path(directory) / "expected.json"
            contract.write_text(json.dumps(expected), encoding="utf-8")
            for value, mode, exit_code, status in (
                (receipt, ["--expected", str(contract)], 0, "PASS"),
                ({}, ["--expected", str(contract)], 1, "FAIL"),
                (ledger_fixture(), ["--native-ledger-v1"], 1, "INCOMPLETE"),
            ):
                path.write_text(json.dumps(value), encoding="utf-8")
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main([str(path), *mode]), exit_code)
                self.assertEqual(json.loads(output.getvalue())["status"], status)
            path.write_text('{"allocations": 1, "allocations": 0}', encoding="utf-8")
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main([str(path), "--native-ledger-v1"]), 1)
            self.assertIn("duplicate JSON", output.getvalue())


if __name__ == "__main__":
    unittest.main()
