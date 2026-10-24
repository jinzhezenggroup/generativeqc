"""Compare independently reconverged energies to frozen full native receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import re
import subprocess
import tempfile
from pathlib import Path

BASE = "4a783a27e104e9392a7227b7efca79abcfba60ae"
NATIVE = "benchmarks/results/cbs-fast-memory-20261021/"
TARGETS = {1: 0, 3: 8, 4: 48, 5: 183, 6: 293, 7: 5604}
GATE = 5e-8


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def blob(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"])


def native_bundle() -> tuple[dict, dict]:
    """Bind native data to committed LF bytes, independent of Windows checkout."""
    manifest = json.loads(blob(NATIVE + "publication.json"))
    for entry in manifest["files"]:
        raw = blob(NATIVE + entry["path"])
        assert sha(raw) == entry["sha256"] and len(raw) == entry["bytes"], entry["path"]
    raw = blob(NATIVE + "summary.json.xz")
    summary = json.loads(lzma.decompress(raw))
    source = summary["qualification"]["source"]["candidate"]
    patch = lzma.decompress(blob(NATIVE + "qualified-source.patch.xz"))
    assert sha(patch) == source["patch_sha256"]
    # Reconstruct the historical snapshot rather than assigning today's source
    # identity to an old library. This is opt-in evidence replay, not a CI test.
    changed = re.findall(rb"^diff --git a/(\S+) b/\S+$", patch, re.MULTILINE)
    with tempfile.TemporaryDirectory(prefix="cbs2679-source-") as folder:
        root = Path(folder)
        for name in {p.decode() for p in changed} | set(source["file_sha256"]):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                subprocess.check_output(
                    ["git", "show", f"{source['base_revision']}:{name}"]
                )
            )
        subprocess.run(
            ["git", "-c", "core.autocrlf=false", "apply", "--check", "-"],
            input=patch,
            cwd=root,
            check=True,
        )
        subprocess.run(
            ["git", "-c", "core.autocrlf=false", "apply", "-"],
            input=patch,
            cwd=root,
            check=True,
        )
        for name, expected in source["file_sha256"].items():
            assert sha((root / name).read_bytes()) == expected, name
    return summary, {
        "commit": BASE,
        "path": NATIVE + "summary.json.xz",
        "sha256": sha(raw),
        "publication_sha256": sha(blob(NATIVE + "publication.json")),
        "source": source,
    }


def coefficients() -> dict[str, float]:
    h = [-1 / math.expm1(1.63), 1 + 1 / math.expm1(1.63)]
    m = [-8 / 19, 27 / 19]
    return {
        "low/df-ccsd(t)": 1.0,
        "low/mp2": m[0] - 1.0,
        "low/rhf": h[0] - m[0],
        "high/mp2": m[1],
        "high/rhf": h[1] - m[1],
    }


def native_values(native: dict, protocol: dict) -> dict[str, float]:
    assert native["status"] == "completed" and len(native["samples"]) == 1
    assert native["protocol"] == protocol
    sample = native["samples"][0]
    assert sample["converged"] and sample["executed_backend"] == "cuda"
    meta = sample["basis_metadata"]
    assert meta["hf_reference"] == "df" and meta["density_fitting"]
    assert meta["frozen_core"] == 0 and meta["representation"] == "spherical"
    assert meta["hf_exponent"] == 1.63 and meta["correlation_exponent"] == 3.0
    assert meta["auxiliary_basis_identities"] == dict(
        zip(
            protocol["orbital_identities"],
            protocol["auxiliary_identities"],
            strict=True,
        )
    )
    values = {}
    for component in sample["components"]:
        level = (
            "low"
            if component["basis_identity"] == protocol["orbital_identities"][0]
            else "high"
        )
        assert component["basis_identity"] in protocol["orbital_identities"]
        key = level + "/" + component["method"]
        assert key not in values and component["result"]["converged"]
        assert abs(component["coefficient"] - coefficients()[key]) < 1e-15
        values[key] = component["result"]["energy"]
    assert set(values) == set(coefficients())
    assert (
        abs(sum(coefficients()[k] * v for k, v in values.items()) - sample["energy"])
        < 5e-12
    )
    values["composite"] = sample["energy"]
    assert all(math.isfinite(v) for v in values.values())
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument(
        "--inputs", type=Path, default=Path(__file__).with_name("inputs.json")
    )
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    input_hashes = {sha(args.inputs.read_bytes())}
    measured_input = args.inputs.with_name("inputs-measured.json.xz")
    if measured_input.is_file():
        original = lzma.decompress(measured_input.read_bytes())
        assert json.loads(original) == inputs
        input_hashes.add(sha(original))
    summary, provenance = native_bundle()
    assert inputs["native_source"] == provenance["source"]
    result = {
        "schema": "generativeqc.independent-cbs-energy-comparison.v1",
        "gate_eh": GATE,
        "native_provenance": provenance,
        "molecules": [],
        "scope": "energy only; no new native timing or force qualification",
    }
    for carbon, index in TARGETS.items():
        protocol = inputs["protocols"][str(carbon)]
        assert protocol["qm7_index_zero_based"] == index
        assert protocol["orbital_bases"] == ["cc-pvdz", "cc-pvtz"]
        assert protocol["auxiliary_bases"] == ["cc-pvdz-jkfit", "cc-pvtz-jkfit"]
        assert protocol["hf_reference"] == "df" and protocol["frozen_core"] is False
        native = {}
        for policy, series in (("standard", summary), ("fast", summary["fast"])):
            point = next(p for p in series["points"] if p["carbon"] == carbon)
            assert point["native_outcome"]["exit_code"] == 0
            assert (
                point["native"]["library_sha256"]
                == provenance["source"]["library_sha256"]
            )
            assert (
                point["native"]["driver_sha256"]
                == provenance["source"]["file_sha256"]["benchmarks/readme_cbs.py"]
            )
            native[policy] = native_values(point["native"], protocol)
        row = {
            "carbon": carbon,
            "qm7_index": index,
            "status": "INCOMPLETE",
            "native": native,
        }
        result["molecules"].append(row)
        path = args.receipts / f"C{carbon}.json"
        bundle = None
        if not path.is_file() and path.with_suffix(".json.xz").is_file():
            bundle = json.loads(
                lzma.decompress(path.with_suffix(".json.xz").read_bytes())
            )
            raw = bundle["receipt_json"].encode()
            assert sha(raw) == bundle["receipt_sha256"]
        elif path.is_file():
            raw = path.read_bytes()
        else:
            raw = None
        if raw is None:
            row["reason"] = "Independent raw receipt unavailable"
            continue
        reference = json.loads(raw)
        row.update(
            receipt_sha256=sha(raw),
            reference_status=reference["status"],
            seconds=reference.get("seconds"),
            max_rss_kib=reference.get("max_rss_kib"),
        )
        if reference["status"] != "completed":
            row["reason"] = reference.get(
                "error", "Reference interrupted or unfinished"
            )
            continue
        assert reference["protocol"] == protocol
        assert reference["dataset_sha256"] == inputs["dataset_sha256"]
        assert reference["inputs_sha256"] in input_hashes
        assert reference["packages"]["pyscf"] == "2.14.0"
        assert reference["driver_sha256"] == sha(
            Path(__file__).with_name("oracle.py").read_bytes()
        )
        for stage in reference["stages"]:
            assert stage["status"] == "completed" and stage["rhf"]["converged"]
            assert stage["mp2"]["frozen"] == 0 and stage["mp2"]["reference_df_shared"]
            if stage["cardinal"] == 2:
                assert stage["ccsd"]["converged"] and stage["ccsd"]["frozen"] == 0
                assert stage["ccsd"]["reference_df_shared"]
            if bundle is not None:
                assert (
                    sha(bundle["logs"][str(stage["cardinal"])].encode())
                    == stage["log_sha256"]
                )
            for role, b in stage["basis_records"].items():
                records = {int(k): v for k, v in b["records"].items()}
                assert b["sha256"] == sha(json.dumps(records, sort_keys=True).encode())
                expected = inputs["bases"][stage[role]]
                assert b["sha256"] == expected["pyscf_records_sha256"]
                assert b["native_identity"] == expected["identity"]
        energies = reference["energies"]
        low, high = reference["stages"]
        assert [low["cardinal"], high["cardinal"]] == [2, 3]
        assert {k: v for k, v in energies.items() if k != "composite"} == {
            "low/rhf": low["rhf"]["energy"],
            "low/mp2": low["mp2"]["energy"],
            "low/df-ccsd(t)": low["triples"]["energy"],
            "high/rhf": high["rhf"]["energy"],
            "high/mp2": high["mp2"]["energy"],
        }
        assert (
            low["triples"]["energy"]
            == low["ccsd"]["energy"] + low["triples"]["correction"]
        )
        assert set(energies) == {*coefficients(), "composite"}
        assert all(math.isfinite(v) for v in energies.values())
        assert (
            abs(
                sum(coefficients()[k] * energies[k] for k in coefficients())
                - energies["composite"]
            )
            < 5e-12
        )
        row["reference"] = energies
        if bundle is not None:
            assert json.loads(bundle["outcome_json"])["exit_code"] == 0
        row["errors_eh"] = {
            p: {k: abs(v[k] - energies[k]) for k in energies} for p, v in native.items()
        }
        row["status"] = (
            "PASS"
            if all(
                e <= GATE
                for errors in row["errors_eh"].values()
                for e in errors.values()
            )
            else "FAIL"
        )
    result["status"] = (
        "PASS"
        if all(r["status"] == "PASS" for r in result["molecules"])
        else "FAIL"
        if any(r["status"] == "FAIL" for r in result["molecules"])
        else "INCOMPLETE"
    )
    args.output.write_bytes(
        (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "molecules": [
                    {k: r[k] for k in ("carbon", "status")} for r in result["molecules"]
                ],
            }
        )
    )
    if args.require_pass and result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
