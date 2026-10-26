"""Exercise retained qualification populations, not packet filename spelling."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "validation_gemm_receipt",
    ROOT / "benchmarks/results/scf-validation-gemm-20261024/reproduce.py",
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class PopulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = REPLAY.read("endpoints.json.xz")
        self.decoded = {
            cohort: {name: json.loads(raw) for name, raw in records.items()}
            for cohort, records in self.packet["cohorts"].items()
        }

    def test_changed_source_identity_rejected(self) -> None:
        original = REPLAY.read
        for field, value in (
            ("revision", "0" * 40),
            ("tree", "0" * 40),
            ("source_patch", "dirty"),
            ("manifest_bytes", 0),
            ("manifest_sha256", "0" * 64),
            ("files_verified", 0),
        ):
            with self.subTest(field=field):
                provenance = original("provenance.json.xz")
                provenance["source_identity"][field] = value
                provenance["source_identity_sha256"] = hashlib.sha256(
                    json.dumps(
                        provenance["source_identity"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                try:
                    REPLAY.read = lambda name: (
                        provenance if name == "provenance.json.xz" else original(name)
                    )
                    with self.assertRaises(AssertionError):
                        REPLAY.main()
                finally:
                    REPLAY.read = original

    def test_complete_population(self) -> None:
        REPLAY.validate_populations(self.decoded)

    def test_duplicate_missing_or_extra_members(self) -> None:
        for cohort in self.decoded:
            names = list(self.decoded[cohort])
            for mutation in ("duplicate", "missing", "extra"):
                with self.subTest(cohort=cohort, mutation=mutation):
                    data = copy.deepcopy(self.decoded)
                    if mutation == "duplicate":
                        data[cohort][names[-1]] = data[cohort][names[0]]
                    elif mutation == "missing":
                        del data[cohort][names[0]]
                    else:
                        data[cohort]["extra"] = data[cohort][names[0]]
                    with self.assertRaises(AssertionError):
                        REPLAY.validate_populations(data)

    def test_wrong_mode_and_arm(self) -> None:
        for cohort in self.decoded:
            name = next(iter(self.decoded[cohort]))
            for field, value in (("mode", "invalid"), ("arm", "invalid")):
                with self.subTest(cohort=cohort, field=field):
                    data = copy.deepcopy(self.decoded)
                    data[cohort][name][field] = value
                    with self.assertRaises(AssertionError):
                        REPLAY.validate_populations(data)

    def test_unknown_cohort(self) -> None:
        self.decoded["unexpected"] = {}
        with self.assertRaises(AssertionError):
            REPLAY.validate_populations(self.decoded)

    def test_rehashed_duplicate_rejected_by_full_replay(self) -> None:
        cohort = "validation-gemm-pr-moved-df"
        names = list(self.packet["cohorts"][cohort])
        raw = self.packet["cohorts"][cohort][names[0]]
        self.packet["cohorts"][cohort][names[1]] = raw
        self.packet["raw_record_sha256"][cohort + "/" + names[1]] = hashlib.sha256(
            raw.encode()
        ).hexdigest()
        original = REPLAY.read
        try:
            REPLAY.read = lambda name: (
                self.packet if name == "endpoints.json.xz" else original(name)
            )
            with (
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(AssertionError),
            ):
                REPLAY.main()
        finally:
            REPLAY.read = original


if __name__ == "__main__":
    unittest.main()
