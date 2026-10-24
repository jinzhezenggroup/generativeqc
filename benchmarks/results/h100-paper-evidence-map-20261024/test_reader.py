"""Behavioral integrity checks; no compiler, oracle or device invocation."""

import copy
import importlib.util
import lzma
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("paper_map_reader", HERE / "reader.py")
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


class ReaderTests(unittest.TestCase):
    def test_original_metadata_path_does_not_replace_capsule_key(self) -> None:
        raw = b"{}\n"
        bundle = {
            "files": {"native/endpoints.jsonl": raw.decode()},
            "file_sha256": {"native/endpoints.jsonl": reader.sha(raw)},
            "file_metadata": {
                "native/endpoints.jsonl": {
                    "path": "historical-root/native/endpoints.jsonl",
                    "bytes": len(raw),
                    "sha256": reader.sha(raw),
                    "mode": 0o644,
                }
            },
        }
        self.assertEqual(set(reader.raw_records(bundle)), {"native/endpoints.jsonl"})
        bundle["file_metadata"]["native/endpoints.jsonl"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            reader.raw_records(bundle)

    def test_xz_bounds_and_stream_completion(self) -> None:
        raw = lzma.compress(b"x" * 1024)
        self.assertEqual(reader.unpack(raw), b"x" * 1024)
        for payload in (raw[:-1], raw + b"tail", raw + raw):
            with self.subTest(payload=len(payload)), self.assertRaises(ValueError):
                reader.unpack(payload)
        with self.assertRaises(ValueError):
            reader.unpack(raw, limit=100)

    def test_strict_json(self) -> None:
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                reader.json_bytes(raw)

    def test_raw_paths_bytes_modes_duplicates(self) -> None:
        entry = {
            "path": "science/receipt.json",
            "utf8": "{}\n",
            "bytes": 3,
            "sha256": reader.sha(b"{}\n"),
            "original_mode": "100644",
        }
        self.assertEqual(
            reader.raw_records({"files": [entry]}), {entry["path"]: b"{}\n"}
        )
        changes = [{"path": p} for p in ("../x", "/x", "C:/x", "a\\b", "a//b", "a/./b")]
        changes += [{"bytes": 4}, {"sha256": "0" * 64}, {"original_mode": "120777"}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                reader.raw_records({"files": [{**entry, **change}]})
        with self.assertRaises(ValueError):
            reader.raw_records({"files": [entry, entry]})

    def test_fixed_commit_and_mode_binding(self) -> None:
        sources = reader.json_bytes((HERE / "sources.json").read_bytes())
        for change in ({"commit": "master"}, {"path": "../x"}, {"mode": "120000"}):
            mutated = copy.deepcopy(sources)
            mutated["blobs"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                reader.Evidence(mutated)
        for change in (
            {"bytes": 1},
            {"sha256": "0" * 64},
            {"blob": "0" * 40},
            {"mode": "100755"},
        ):
            mutated = copy.deepcopy(sources)
            mutated["blobs"][0].update(change)
            evidence = reader.Evidence(mutated)
            entry = mutated["blobs"][0]
            with self.subTest(change=change), self.assertRaises(ValueError):
                evidence.raw(entry["commit"], entry["path"])

    def test_numeric_failure_and_optimization(self) -> None:
        script = f"""
import importlib.util
s=importlib.util.spec_from_file_location('r', {str(HERE / "reader.py")!r})
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
for f in (lambda:r.require(False,'gate'), lambda:r.json_bytes(b'{{\"x\":NaN}}'), lambda:r.finite([float('nan')])):
 try:f()
 except ValueError:pass
 else:raise SystemExit('gate bypassed')
"""
        for flags in ([], ["-O"], ["-OO"]):
            with self.subTest(flags=flags):
                result = subprocess.run(
                    [sys.executable, *flags, "-c", script],
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_raw_oracle_algebra_and_convergence_fail_closed(self) -> None:
        family = HERE.parent / "cbs-independent-energy-20261023"
        bundle = reader.json_bytes(
            reader.unpack((family / "receipts/C4.json.xz").read_bytes())
        )
        original = reader.json_bytes(bundle["receipt_json"])
        self.assertEqual(set(reader.reference(original)), set(original["energies"]))
        changed = copy.deepcopy(original)
        changed["energies"]["composite"] += 1e-3
        with self.assertRaises(ValueError):
            reader.reference(changed)
        changed = copy.deepcopy(original)
        changed["stages"][0]["rhf"]["converged"] = False
        with self.assertRaises(ValueError):
            reader.reference(changed)

    def test_batch_rejects_bad_metadata(self) -> None:
        sources = reader.json_bytes((HERE / "sources.json").read_bytes())
        for change in ({"bytes": 1}, {"sha256": "0" * 64}, {"mode": "100755"}):
            mutated = {"blobs": [copy.deepcopy(sources["blobs"][0])]}
            mutated["blobs"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                reader.Evidence(mutated).verify()

    def test_full_offline_replay(self) -> None:
        report = reader.replay(
            reader.json_bytes((HERE / "sources.json").read_bytes()),
            reader.json_bytes((HERE / "coverage-status.json").read_bytes()),
        )
        self.assertEqual(
            report, reader.json_bytes((HERE / "compact-report.json").read_bytes())
        )
        self.assertEqual(report["energy"]["count"], 72)
        self.assertEqual(report["directional"]["count"], 48)
        self.assertEqual(report["directional"]["raw_files"], 40)
        self.assertEqual(report["h100_core_status"], "INCOMPLETE")
        self.assertEqual(report["performance"]["ratios"], [])
        self.assertEqual(len(report["coverage"]["h100_core"]), 11)
        self.assertEqual(
            [
                r["formal_latest_head_approval"]
                for r in report["coverage"]["review_delivery"]
            ],
            ["APPROVED", "ABSENT", "ABSENT"],
        )

    def test_prepared_contract_needs_no_unmerged_git_object(self) -> None:
        sources = reader.json_bytes((HERE / "sources.json").read_bytes())
        entry = next(e for e in sources["blobs"] if "snapshot" in e)
        evidence = reader.Evidence({"blobs": [entry]})
        with mock.patch.object(
            reader.subprocess,
            "check_output",
            side_effect=AssertionError("Git object should not be required"),
        ):
            raw = evidence.raw(entry["commit"], entry["path"])
        self.assertEqual(reader.sha(raw), entry["sha256"])
        self.assertEqual(
            reader.json_bytes(raw)["schema"], "generativeqc.h100-paper-protocol.v1"
        )
        changed = copy.deepcopy(entry)
        changed["snapshot"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            reader.Evidence({"blobs": [changed]}).raw(entry["commit"], entry["path"])


if __name__ == "__main__":
    unittest.main()
