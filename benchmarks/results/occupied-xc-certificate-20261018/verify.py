"""Independently check retained full force arrays and print complete-population timings."""

import argparse
import hashlib
import json
import lzma
import math
import os
import statistics
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _evidence_bytes(evidence: Path | None) -> bytes:
    """Read explicit bytes or an existing Git blob, never lazily fetch a remote."""
    if evidence is not None:
        return evidence.read_bytes()
    receipt = json.loads((ROOT / "evidence-source.json").read_text())
    result = subprocess.run(
        ["git", "cat-file", "blob", receipt["git_blob"]],
        cwd=ROOT,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": "", "GIT_TERMINAL_PROMPT": "0"},
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "The retained evidence Git object is unavailable in this shallow/archive checkout. "
            "Recover the exact packet from commit " + receipt["git_commit"]
            + " as documented in README.md, then pass --evidence PATH. No network was attempted."
        )
    return result.stdout


def verify(evidence: Path | None = None) -> None:
    """Verify retained packet bytes, numerical gates and actual convergence counts."""
    if not __debug__:
        raise RuntimeError(
            "Evidence verification requires assertions; rerun without -O or PYTHONOPTIMIZE."
        )
    raw = _evidence_bytes(evidence)
    for row in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = row.split(maxsplit=1)
        content = raw if name == "evidence.json.xz" else (ROOT / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected, name
    packet = json.loads(lzma.decompress(raw))
    print("measured native identity:", packet["candidate_source"]["native_identity"])
    print("default enabled:", packet["default_enabled"])
    completed = 0
    for key, population in packet["populations"].items():
        native = population["native"]
        oracle = packet["oracles"][population["case"]]
        arm = population.get("arm", key.split("/")[0])
        if arm != "peer":
            source = packet["original_source"] if arm == "original" else packet["candidate_source"]
            assert native["native_build"]["probe"]["source_identity"] == source["native_identity"], key
            assert native["native_build"]["library_sha256"] == source["runtime_files"]["libgenerativeqc.so"], key
            assert native["native_build"]["probe"]["policy"]["precision"] == "fp64", key
            expected_selector = "1" if arm == "candidate" else "0"
            assert population["receipts"]["selector.txt"].strip() == expected_selector, key
        if key.startswith("profile/"):
            assert native["stage"] == "cold-and-first-warm-only", key
            assert native["status"] == "diagnostic_completed", key
        else:
            assert native["stage"] == "complete" and native["status"] == "measured", key
        assert native["protocol"] == oracle["protocol"], key
        phases = ["cold", "warm"] if key.startswith("profile/") else [
            "cold", *(["warm"] * 5), "moved", *(["moved-warm"] * 5)
        ]
        assert [row["phase"] for row in native["records"]] == phases, key
        errors = []
        for row in native["records"]:
            target = oracle["geometries"][str(row["geometry"])]
            anchor = b"".join(struct.pack("<d", value) for vector in target["forces"] for value in vector)
            planes = bytes.fromhex(row["forces_xor_f64le_byteplanes"])
            assert len(anchor) == len(planes), key
            width = len(anchor) // 8
            delta = bytearray(len(anchor))
            for byte in range(8):
                delta[byte::8] = planes[byte * width:(byte + 1) * width]
            original = bytes(left ^ right for left, right in zip(anchor, delta, strict=True))
            assert hashlib.sha256(original).hexdigest() == row["forces_f64le_sha256"], key
            values = [value[0] for value in struct.iter_unpack("<d", original)]
            row["forces"] = [values[begin:begin + 3] for begin in range(0, len(values), 3)]
            assert row["converged"] and target["converged"], key
            assert math.isfinite(row["energy"]) and math.isfinite(row["complete_seconds"]), key
            assert row["complete_seconds"] > 0, key
            assert row["iterations"] > 0, key
            if "fock_builds" in row:
                assert row["fock_builds"] > 0, key
            assert len(row["forces"]) == len(target["forces"]), key
            force_error = 0.0
            for vector, expected in zip(row["forces"], target["forces"], strict=True):
                assert len(vector) == len(expected) == 3, key
                for value, reference in zip(vector, expected, strict=True):
                    assert math.isfinite(value) and math.isfinite(reference), key
                    force_error = max(force_error, abs(value - reference))
            energy_error = abs(row["energy"] - target["energy"])
            assert energy_error <= 1e-8 and force_error <= 1e-7, (key, errors)
            errors.append((energy_error, force_error))
        times = [statistics.median(row["complete_seconds"] for row in native["records"]
                                   if row["phase"] == phase)
                 for phase in dict.fromkeys(phases)]
        counts = [(row["iterations"], row.get("fock_builds", row.get("scf_jk_builds")))
                  for row in native["records"]]
        print(key, "seconds:", *(f"{value:.6f}" for value in times),
              "max errors:", max(value[0] for value in errors), max(value[1] for value in errors),
              "iterations/builds:", counts)
        completed += len(native["records"])
    print("completed calls:", completed)
    print("incomplete populations:", packet["incomplete"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, help="Explicit lossless packet for a shallow/archive checkout")
    verify(parser.parse_args().evidence)
