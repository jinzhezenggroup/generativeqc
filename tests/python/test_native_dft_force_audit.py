"""Adversarial receipts exercise the complete R1 paired endpoint gate offline."""

from __future__ import annotations

import hashlib
import json
import lzma
import subprocess
import sys
from pathlib import Path

import pytest
from test_exact_fp64_evidence import write_bundle


@pytest.fixture(params=("cpu", "cuda"))
def backend(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def paired_receipts(tmp_path: Path, backend: str) -> Path:
    """Retain sample-dependent history without granting work-count tolerance."""
    cases = {
        f"{method}-{spin}-{fitted}"
        for method in (
            "lda",
            "pbe",
            "pbe0",
            "b3lyp",
            *(("pbe50",) if backend == "cpu" else ()),
            "r2scan",
            "wb97m-v",
            *(("m06-2x", "mn15") if backend == "cuda" else ()),
            "pbe-d3(bj)",
            "pbe0-d4(bj-eeq-atm)",
            "r2scan-3c",
        )
        for spin in (0, 1)
        for fitted in (0, 1)
    }
    if backend == "cuda":
        cases |= {"pbe-water36-df-d", "pbe-water48-df-d"}
    cases |= {
        f"libxc-{functional}-{spin}-0"
        for functional in (
            (
                "LDA_C_VWN_4",
                "LDA_C_CHACHIYO",
                "LDA_C_PZ_MOD",
                "GGA_X_APBE",
                "GGA_X_PBE_SOL",
                "GGA_X_RPBE",
                "GGA_X_B86",
                "GGA_C_APBE",
                "MGGA_X_LTA",
                "MGGA_X_GVT4",
                "MGGA_C_TPSS",
            )
            if backend == "cuda"
            else ()
        )
        for spin in (0, 1)
    }
    expected = {
        variant: {"binary_sha256": variant} for variant in ("baseline", "candidate")
    }
    (tmp_path / "expected-inputs.json").write_text(json.dumps(expected))
    for order in ("ABBA", "BAAB"):
        for block, letter in enumerate(order):
            variant = "baseline" if letter == "A" else "candidate"
            header = {
                **expected[variant],
                "variant": variant,
                "samples": 11,
                "warmups": 3,
                "tracing": False,
                "cases": sorted(cases),
            }
            records = [{"metadata": header}]
            for case in sorted(cases):
                atoms = (
                    36
                    if "water36" in case
                    else 48
                    if "water48" in case
                    else 4
                    if case.startswith("libxc-")
                    else 3
                )
                for phase in ("cold", "warm", "moved"):
                    for sample in range(11):
                        records.append(
                            {
                                "case": case,
                                "phase": phase,
                                "sample": sample,
                                "seconds": 1.0,
                                "native": True,
                                "executed_backend": "cuda"
                                if backend == "cuda"
                                else "cpu_reference",
                                "energy": -1.0,
                                "forces": [[0.0] * 3 for _ in range(atoms)],
                                "iterations": 13 + sample % 3,
                                "fock_builds": 15 + sample % 3,
                                "warm_start_used": phase != "cold",
                                "grid_points": 384,
                                "tile_points": 37,
                                "coulomb_exchange_work": 100,
                                "nonlocal_work": 0,
                            }
                        )
            path = tmp_path / f"paired-{order}-{block}-{variant}.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return tmp_path


@pytest.mark.parametrize(
    "mutation", (None, "work", "force", "time", "identity", "missing")
)
def test_paired_audit_preserves_science_work_and_completeness(
    paired_receipts: Path, mutation: str | None, backend: str
) -> None:
    """History parity alone is insufficient, but real same-age skew must fail."""
    path = paired_receipts / "paired-ABBA-1-candidate.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "identity":
        records[0]["metadata"]["binary_sha256"] = "stale"
    elif mutation == "missing":
        records.pop()
    elif mutation == "work":
        records[1]["iterations"] += 1
    elif mutation == "force":
        records[1]["forces"][0][0] = 1e-5
    elif mutation == "time":
        for record in records[1:]:
            record["seconds"] = 1.2
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    output = paired_receipts / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.native_dft_force_audit",
            str(paired_receipts),
            "--output",
            str(output),
            "--backend",
            backend,
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == (0 if mutation is None else 1), completed.stderr
    result = json.loads(output.read_text())
    assert result["status"] == (
        "PASS"
        if mutation is None
        else "INCOMPLETE"
        if mutation == "missing"
        else "FAIL"
    )
    if mutation is None:
        assert len(result["comparisons"]) == (408 if backend == "cuda" else 240)
        assert not result["failures"] and not result["incomplete"]
    elif mutation == "work":
        assert any(failure.get("work_mismatches", 0) for failure in result["failures"])


@pytest.mark.parametrize("tamper", (False, True))
@pytest.mark.parametrize("representation", ("inline", "collection"))
def test_retained_samples_replay_without_raw_logs(
    paired_receipts: Path, backend: str, tamper: bool, representation: str
) -> None:
    """Compact JSON values recover the exact measured block bytes and gates."""
    blocks = {}
    for path in paired_receipts.glob("paired-*.jsonl"):
        contents = path.read_bytes()
        blocks[path.name] = {
            "sha256": hashlib.sha256(contents).hexdigest(),
            "records": [json.loads(line) for line in contents.splitlines()],
        }
    if tamper:
        next(iter(blocks.values()))["records"][1]["seconds"] = 0.9
    payload = {
        "schema": "generativeqc.native-dft-force-observations.v1",
        "backend": backend,
        "expected_inputs": json.loads(
            (paired_receipts / "expected-inputs.json").read_text()
        ),
        "blocks": blocks,
    }
    retained_path = paired_receipts / "observations.json.xz"
    retained_path.write_bytes(lzma.compress(json.dumps(payload).encode()))
    if representation == "collection":
        retained_path = paired_receipts / "collection"
        retained_path.mkdir()
        member = retained_path / "samples"
        member.mkdir()
        cohort = {
            "schema": "generativeqc.native-dft-force-cohort.v1",
            "backend": backend,
            "expected_inputs": payload["expected_inputs"],
            "blocks": {name: f"{backend}/{name}.json" for name in blocks},
        }
        texts = {f"{backend}/cohort.json": json.dumps(cohort)}
        texts.update(
            {
                cohort["blocks"][name]: json.dumps(block)
                for name, block in blocks.items()
            }
        )
        rows = [
            {
                "path": name,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "text": text,
            }
            for name, text in texts.items()
        ]
        write_bundle(
            member,
            {
                "schema": "generativeqc.exact-fp64-evidence.v3",
                "float_count": 0,
                "files": rows,
            },
            [],
        )
        (retained_path / "bundles.json").write_text(
            json.dumps(
                {
                    "schema": "generativeqc.native-dft-force-evidence-set.v1",
                    "bundles": {
                        "samples": hashlib.sha256(
                            (member / "sha256.txt").read_bytes()
                        ).hexdigest()
                    },
                }
            )
        )
    output = paired_receipts / "retained-audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.native_dft_force_audit",
            str(retained_path),
            "--backend",
            backend,
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == int(tamper), completed.stderr
    result = json.loads(output.read_text())
    assert result["status"] == ("FAIL" if tamper else "PASS")
    assert len(result["comparisons"]) == (408 if backend == "cuda" else 240)
