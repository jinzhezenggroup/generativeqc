"""Regression coverage for the scheduled/manual broad Libxc promotion matrix."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS

from tools import qualify_libxc_broad_matrix as matrix

ROOT = Path(__file__).resolve().parents[2]


def _payload(stage: str) -> dict:
    return {
        "stage_evidence": {
            "stage": stage,
            "status": "pass",
        }
    }


def test_default_representatives_cover_required_noncurated_families() -> None:
    selected = matrix.select_representatives()
    counts = Counter(item.family for item in selected)

    assert counts == Counter(matrix.DEFAULT_QUOTAS)
    assert len({item.name for item in selected}) == sum(matrix.DEFAULT_QUOTAS.values())
    assert all(item.name in AUTO_BULK_COMPONENTS for item in selected)
    assert all(item.production_domain_profile.eligible for item in selected)
    assert all(
        item.required_ingredients
        == {
            "lda": ("rho",),
            "gga": ("rho", "sigma"),
            "mgga": ("rho", "sigma", "tau"),
        }[item.family]
        for item in selected
    )


def test_broad_matrix_runs_full_chain_without_per_functional_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capability = SimpleNamespace(
        name="GGA_X_PBE_SOL",
        family="gga",
        required_ingredients=("rho", "sigma"),
        identity="a" * 64,
    )
    calls: list[str] = []

    def compiled(name: str, **kwargs: object) -> dict:
        calls.append(f"compiled:{name}")
        return _payload("compiled-cpu")

    def production(name: str, **kwargs: object) -> dict:
        calls.append(f"production:{name}")
        return _payload("production-domain")

    def molecular(name: str, **kwargs: object) -> dict:
        calls.append(f"molecular:{name}")
        return _payload("molecular-scf")

    def public(name: str, **kwargs: object) -> dict:
        calls.append(f"public:{name}")
        return _payload("public-method")

    monkeypatch.setattr(matrix, "qualify_compiled_cpu", compiled)
    monkeypatch.setattr(matrix, "qualify_functional", production)
    monkeypatch.setattr(matrix, "qualify_molecular_scf", molecular)
    monkeypatch.setattr(matrix, "qualify_public_method", public)

    summary = matrix.run_matrix(
        (capability,),
        output=tmp_path,
        evidence_prefix="test://broad-matrix",
        build_dir=tmp_path,
        pyscf_version="2.14.0",
        libxc=SimpleNamespace(__version__="7.0.0"),
    )

    assert calls == [
        "compiled:GGA_X_PBE_SOL",
        "production:GGA_X_PBE_SOL",
        "molecular:GGA_X_PBE_SOL",
        "public:GGA_X_PBE_SOL",
    ]
    assert summary["public_pass_counts"] == {"gga": 1}
    assert summary["functionals"][0]["status"] == "pass"
    assert (tmp_path / "GGA_X_PBE_SOL" / "public-method.json").is_file()
    assert (tmp_path / "summary.json").is_file()


def test_broad_matrix_stops_endpoint_promotion_after_failed_prerequisite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capability = SimpleNamespace(
        name="GGA_X_PBE_SOL",
        family="gga",
        required_ingredients=("rho", "sigma"),
        identity="a" * 64,
    )
    monkeypatch.setattr(
        matrix,
        "qualify_compiled_cpu",
        lambda *args, **kwargs: _payload("compiled-cpu"),
    )
    monkeypatch.setattr(
        matrix,
        "qualify_functional",
        lambda *args, **kwargs: {
            "stage_evidence": {"stage": "production-domain", "status": "fail"}
        },
    )
    monkeypatch.setattr(
        matrix,
        "qualify_molecular_scf",
        lambda *args, **kwargs: pytest.fail("molecular SCF must remain gated"),
    )
    monkeypatch.setattr(
        matrix,
        "qualify_public_method",
        lambda *args, **kwargs: pytest.fail("public promotion must remain gated"),
    )

    summary = matrix.run_matrix(
        (capability,),
        output=tmp_path,
        evidence_prefix="test://broad-matrix",
        build_dir=tmp_path,
        pyscf_version="2.14.0",
        libxc=SimpleNamespace(__version__="7.0.0"),
    )

    row = summary["functionals"][0]
    assert row["stages"]["molecular-scf"] == "not-run"
    assert row["stages"]["public-method"] == "not-run"
    assert row["status"] == "fail"


def test_broad_matrix_cli_imports_outside_repository(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "qualify_libxc_broad_matrix.py"),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "broad automatic Libxc CPU promotion matrix" in completed.stdout
