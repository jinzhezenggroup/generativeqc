"""Offline numerical replay; compiler-active seconds are never performance data."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.readme_defaults import COMMON_GATES  # noqa: E402
from benchmarks.readme_defaults_qualification import qualify  # noqa: E402
from generativeqc.basis import load_basis  # noqa: E402
from generativeqc.elements import SYMBOLS  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = HERE / name
        assert path.stat().st_size == entry["bytes"]
        assert digest(path) == entry["sha256"]
    for entry in manifest["executed_adapters"].values():
        source = lzma.decompress((HERE / entry["path"]).read_bytes())
        assert len(source) == entry["original_bytes"]
        assert hashlib.sha256(source).hexdigest() == entry["original_sha256"]
    blob = lzma.decompress((HERE / "observations.json.xz").read_bytes())
    assert b"/global_user/" not in blob
    bundle = json.loads(blob)
    assert bundle["source_revision"] == manifest["measured_revision"]
    oracle_path = ROOT / manifest["oracle"]["path"]
    assert digest(oracle_path) == manifest["oracle"]["sha256"]
    oracles = json.loads(lzma.decompress(oracle_path.read_bytes()))
    auxiliary = json.loads((HERE / "hf-reference.json").read_text())
    basis_hash = hashlib.sha256(
        json.dumps(auxiliary["basis"], sort_keys=True).encode()
    ).hexdigest()
    assert basis_hash == auxiliary["basis_sha256"]
    assert auxiliary["selection"] == "cc-pvtz-jkfit"
    native_basis = load_basis(HERE / "hf-auxiliary.json")
    assert native_basis.representation == "spherical"
    roundtrip = {
        SYMBOLS[element.atomic_number - 1]: [
            [
                shell.angular_momentum,
                *[
                    [float(exponent), *(float(c[i]) for c in shell.coefficients)]
                    for i, exponent in enumerate(shell.exponents)
                ],
            ]
            for shell in element.shells
        ]
        for element in native_basis.elements
    }
    assert roundtrip == auxiliary["basis"]
    assert {
        SYMBOLS[e.atomic_number - 1]: sum(
            (2 * s.angular_momentum + 1) * len(s.coefficients) for s in e.shells
        )
        for e in native_basis.elements
    } == {"H": 30, "O": 79}
    cpu = bundle["cpu_fit_scan"]
    cpu_oracle = next(
        r for r in oracles if (r["method"], r["atoms"], r["backend"]) == ("hf", 3, "cpu")
    )
    assert cpu["geometry_bohr"] == cpu_oracle["geometry_bohr"]
    assert cpu["direct"]["energy"] == cpu_oracle["samples"][-1]["energy"]
    assert cpu["direct"]["forces"] == cpu_oracle["samples"][-1]["forces"]
    summary = []
    for population in bundle["populations"]:
        record = population["record"]
        plan = population["plan"]
        assert record["status"] == "complete" and len(record["samples"]) == 4
        assert record["source_revision"] == manifest["measured_revision"]
        fitted = population["fitting_contract"] == "cc-pvtz-jkfit"
        adapter = "endpoint.py" if record["arm"] == "standard" else (
            "hf-fit-reference.py" if fitted else "endpoint.py"
        )
        assert record["runner_sha256"] == (
            manifest["executed_adapters"][adapter]["original_sha256"]
        )
        assert plan["endpoint_sha256"] == record["runner_sha256"]
        assert record["geometry_bohr"] == plan["geometry_bohr"]
        assert record["accelerator"]["nvidia_smi"]["uuid"] == (
            "GPU-b62d2463-27e7-bf5f-41a5-3ea1a4f99508"
        )
        assert record["slurm_job_id"] is None
        assert record["scheduler"]["kind"] == "qz_job"
        assert all(value == "1" for value in record["thread_controls"].values())
        if fitted:
            for name in ("hf-auxiliary.json", "hf-reference.json"):
                assert plan["input_sha256"][name] == digest(HERE / name)
        result = qualify(
            [record], oracles, reference_reuse=bundle["reference_reuse"]
        )[0]
        assert result["status"] == ("qualified" if fitted else "failed")
        target = next(
            row
            for row in oracles
            if (row["method"], row["atoms"], row["backend"])
            == ("hf", record["atoms"], "cuda")
        )
        assert target["geometry_bohr"] == record["geometry_bohr"]
        expected = target["samples"][-1]
        checks = []
        for sample in record["samples"]:
            assert sample["converged"]
            differences = [
                x - y
                for a, b in zip(sample["forces"], expected["forces"], strict=True)
                for x, y in zip(a, b, strict=True)
            ]
            assert len(differences) == 3 * record["atoms"]
            assert all(math.isfinite(x) for x in differences)
            errors = {
                "energy_abs_per_atom_hartree": abs(
                    sample["energy"] - expected["energy"]
                )
                / record["atoms"],
                "force_max_abs_hartree_per_bohr": max(map(abs, differences)),
                "force_rms_hartree_per_bohr": math.sqrt(
                    math.fsum(x * x for x in differences) / len(differences)
                ),
            }
            assert all(
                errors[key] <= threshold
                for key, threshold in COMMON_GATES.items()
                if fitted or key != "energy_abs_per_atom_hartree"
            )
            if not fitted:
                assert (
                    errors["energy_abs_per_atom_hartree"]
                    > COMMON_GATES["energy_abs_per_atom_hartree"]
                )
            checks.append(errors)
        if record["arm"] == "standard":
            assert record["native_build"]["library_sha256"] == (
                bundle["source_build"]["stage"]["artifacts"]["libgenerativeqc.so"]
            )
        else:
            assert record["environment"]["packages"]["gpu4pyscf"] == "1.9.0"
            if fitted:
                assert all(
                    s["settings"]["auxiliary_basis_sha256"] == basis_hash
                    and s["settings"]["auxiliary_aos"] == 139 * record["atoms"] // 3
                    and s["settings"]["auxbasis_response"] is True
                    for s in record["samples"]
                )
        summary.append(
            {
                "case": population["id"],
                "status": result["status"],
                "max_errors": {
                    key: max(c[key] for c in checks) for key in COMMON_GATES
                },
            }
        )
    assert len(summary) == 7
    assert sum(row["status"] == "qualified" for row in summary) == 6
    print(json.dumps({"status": "PASS", "populations": summary}, indent=2))


if __name__ == "__main__":
    main()
