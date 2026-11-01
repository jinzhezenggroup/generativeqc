"""Offline evidence integrity, numeric replay and bounded restoration checks."""

import base64
import copy
import json
import lzma
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import reader

ROOT = Path(__file__).parent


class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = reader.decode_xz(
            (ROOT / "terminal-evidence.json.xz").read_bytes()
        )
        cls.evidence = reader.load(ROOT)

    def test_full_original_history_and_independent_two_size_statistics(self):
        report = reader.report(ROOT)
        self.assertEqual(report["actual_complete_EF_returns"], 98)
        self.assertEqual(report["not_run"], 10)
        self.assertEqual(report["formal_EF_returns"], 72)
        self.assertEqual(report["formal_independent_processes"], 18)
        self.assertEqual(
            report["jobs"]["2732-pilot-v1"]["processes"][0]["status"],
            "FAILED_CONTROL_WITH_TWO_NUMERIC_RETURNS",
        )
        for job, expected in (
            (
                "2732-formal48-run",
                [8.331081944983453, 7.6576017959741876, 5.876181058003567],
            ),
            (
                "2732-formal96-run",
                [26.020890172570944, 23.901594722643495, 13.648962665349245],
            ),
        ):
            stats = report["jobs"][job]["statistics_by_arm"]
            self.assertEqual(
                [stats[arm]["median_of_process_warm_medians"] for arm in reader.ARMS],
                expected,
            )
            self.assertTrue(
                all(len(stats[a]["each_warm_array"]) == 3 for a in reader.ARMS)
            )
        self.assertFalse(report["hardware_only"])
        self.assertIsNone(report["qualified_speedup"])
        self.assertIn("peak memory", report["unknowns"])

    def test_bounded_decoder_rejects_trailing_truncated_oversized_and_duplicate_keys(
        self,
    ):
        good = lzma.compress(b'{"valid":true}')
        self.assertEqual(reader.decode_xz(good), {"valid": True})
        for bad in (
            good + b"trailing",
            good[:-2],
            lzma.compress(b"x" * 65),
            lzma.compress(b'{"duplicate":1,"duplicate":2}'),
        ):
            with (
                self.subTest(bad=bad[:10]),
                self.assertRaises((ValueError, lzma.LZMAError)),
            ):
                reader.decode_xz(bad, limit=64)

    def test_text_storage_preserves_v1_bytes_and_rejects_ambiguous_encodings(self):
        legacy = copy.deepcopy(self.document)
        legacy["schema"] = "generativeqc.h100-direct-content-objects.v1"
        legacy["objects"] = {
            digest: base64.b64encode(raw).decode()
            for digest, raw in self.evidence.objects.items()
        }
        self.assertEqual(reader.Evidence(legacy).objects, self.evidence.objects)
        digest = next(iter(self.document["objects"]))
        for encoding in (
            {"utf8": 1},
            {"unknown": "x"},
            {"utf8": "x", "base64": "eA=="},
        ):
            bad = copy.deepcopy(self.document)
            bad["objects"][digest] = encoding
            with self.subTest(encoding=encoding), self.assertRaises(ValueError):
                reader.Evidence(bad)

    def test_unsafe_paths_fail_before_restoration(self):
        for name in (
            ".",
            "",
            "con",
            "aux.txt",
            "a/COM1.log",
            "trailing.",
            "trailing ",
            "null\x00byte",
            "../escape",
            "/absolute",
            "C:/escape",
            "a\\b",
            "a/../b",
            "a//b",
            "a/.",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                reader.safe_path(name)
        self.assertEqual(reader.safe_path(".cache/original.bin"), ".cache/original.bin")

    def test_corrupt_missing_and_unreferenced_objects_are_rejected(self):
        digest = next(iter(self.document["objects"]))
        bad = copy.deepcopy(self.document)
        bad["objects"][digest] = {"base64": "bad-base64!"}
        with self.assertRaises(ValueError):
            reader.Evidence(bad)
        bad["objects"][digest] = {"utf8": "wrong original"}
        with self.assertRaises(ValueError):
            reader.Evidence(bad)
        bad = copy.deepcopy(self.document)
        del bad["objects"][digest]
        with self.assertRaises(KeyError):
            reader.Evidence(bad)
        bad = copy.deepcopy(self.document)
        bad["objects"][reader.sha(b"unused")] = {"utf8": "unused"}
        with self.assertRaises(ValueError):
            reader.Evidence(bad)

    def test_duplicate_paths_and_unadmitted_exclusions_are_rejected(self):
        bad = copy.deepcopy(self.document)
        rows = bad["original_job_member_inventories"]["2732-pilot-v1"]["members"]
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaises(ValueError):
            reader.Evidence(bad)
        bad = copy.deepcopy(self.document)
        bad["original_job_member_inventories"]["2732-formal96-run"]["members"][0][
            "availability"
        ] = "INVENTORY_ONLY_GENERATED_BINARY"
        with self.assertRaises(ValueError):
            reader.Evidence(bad)

    def test_inventory_only_binary_and_existing_output_preserved(
        self,
    ):
        row = self.document["excluded_generated_binary_inventory"][0]
        with self.assertRaisesRegex(ValueError, "inventory-only"):
            self.evidence.get(row["job"], row["path"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-output"
            receipt = self.evidence.restore("2732-pilot-v1", output)
            self.assertEqual(len(receipt["unavailable"]), 3)
            original = "output/slot-0/return-000.bin"
            self.assertEqual(
                (output / original).read_bytes(),
                self.evidence.get("2732-pilot-v1", original),
            )
            self.assertFalse((output / row["path"]).exists())
            with self.assertRaises(FileExistsError):
                self.evidence.restore("2732-pilot-v1", output)
            self.assertEqual(
                (output / original).read_bytes(),
                self.evidence.get("2732-pilot-v1", original),
            )

    def test_full_1176_and_2328_byte_frames_refuse_partial_nan_and_noncanonical_ordinal(
        self,
    ):
        for job, atoms, size in (
            ("2732-formal48-run", 48, 1176),
            ("2732-formal96-run", 96, 2328),
        ):
            stem = "output/slot-0/return-000"
            raw = self.evidence.get(job, stem + ".bin")
            index_raw = self.evidence.get(job, stem + ".index.json")
            index = json.loads(index_raw)
            complete = self.evidence.read(job, stem + ".complete.json")
            self.assertEqual(len(raw), size)
            self.assertEqual(
                len(reader.frame(raw, index, complete, index_raw, atoms, 0)["forces"]),
                atoms,
            )
            with self.assertRaises(ValueError):
                reader.frame(raw[:8], index, complete, index_raw, atoms, 0)
            with self.assertRaises(ValueError):
                reader.frame(
                    raw, index, dict(complete, index=False), index_raw, atoms, 0
                )
            damaged = struct.pack("<d", float("nan")) + raw[8:]
            changed = dict(index, data_sha256=reader.sha(damaged))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                reader.frame(damaged, changed, complete, index_raw, atoms, 0)

    def test_independent_process_statistics_do_not_pool_warm_arrays(self):
        rows = [
            dict(
                role="formal",
                process_repeat=i,
                cohort_key="HOST_FIXTURE",
                birth=dict(command=dict(pid=i + 1, start_ticks=i + 10), nonce=str(i)),
                samples=[
                    dict(seconds=t, gate="PASS")
                    for t in (10 * (i + 1), i + 1, i + 1, 100 * (i + 1))
                ],
            )
            for i in range(3)
        ]
        stats = reader.statistics(rows)
        self.assertEqual(stats["cold_median"], 20)
        self.assertEqual(stats["median_of_process_warm_medians"], 2)
        self.assertEqual(stats["each_warm_array"][0], [1, 1, 100])
        for key, value in (
            ("role", "diagnostic_pilot"),
            ("process_repeat", 1.0),
            ("process_repeat", False),
            ("cohort_key", "different"),
        ):
            bad = copy.deepcopy(rows)
            bad[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reader.statistics(bad)
        for defect in ("missingfourth", "duplicatebirth", "duplicatenonce"):
            bad = copy.deepcopy(rows)
            if defect == "missingfourth":
                bad[0]["samples"].pop()
            elif defect == "duplicatebirth":
                bad[1]["birth"]["command"] = bad[0]["birth"]["command"]
            else:
                bad[1]["birth"]["nonce"] = bad[0]["birth"]["nonce"]
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                reader.statistics(bad)

    def test_original_host_failure_is_not_promoted_to_a_full_suite_pass(self):
        # Historical HOST capsules have a separately authenticated Git snapshot.
        # Let the explicit offline audit supply that restored fixture; never
        # fetch or silently skip missing/corrupt history in this integrity test.
        history_root = Path(os.environ.get("GQC_H100_HOST_HISTORY_ROOT", ROOT))
        history = reader.host_history(history_root, reference_root=ROOT)
        self.assertEqual(history["original_uncompressed_members"], 37)
        self.assertEqual(history["standalone_original_compressed_files"], 4)
        self.assertEqual(history["V2_probe_returncode"], 1)
        self.assertEqual(history["V2_verified_control_originals"], 27)
        self.assertTrue(history["V2_full_suite"].startswith("UNAVAILABLE"))

    def test_noncanonical_public_repeat_and_foreign_guard_nonce_fail_replay(self):
        original = self.evidence.read
        job = "2732-formal96-run"
        for name, field, value in (
            ("output/slot-0/metadata-000.json", "repeat", False),
            ("process-0/control/start-command.json", "nonce", "foreign"),
        ):

            def injected(selected_job, selected_name):
                result = original(selected_job, selected_name)
                if (selected_job, selected_name) == (job, name):
                    result[field] = value
                return result

            with (
                self.subTest(name=name),
                patch.object(self.evidence, "read", side_effect=injected),
            ):
                with self.assertRaises(ValueError):
                    reader.replay(self.evidence, job)


if __name__ == "__main__":
    unittest.main()
