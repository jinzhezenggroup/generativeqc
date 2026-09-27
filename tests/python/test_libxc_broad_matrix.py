"""Regression coverage for the scheduled/manual broad Libxc promotion matrix."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS

from tools import qualify_libxc_broad_matrix as matrix

ROOT = Path(__file__).resolve().parents[2]


def _payload(stage: str, *, status: str = "pass", reason: str | None = None) -> dict:
    return {
        "stage_evidence": {
            "stage": stage,
            "status": status,
            "reason": reason,
        }
    }


def _candidate(name: str, family: str) -> SimpleNamespace:
    ingredients = {
        "lda": ("rho",),
        "gga": ("rho", "sigma"),
        "mgga": ("rho", "sigma", "tau"),
    }[family]
    return SimpleNamespace(
        name=name,
        family=family,
        required_ingredients=ingredients,
        identity=(name.encode().hex() + "0" * 64)[:64],
        production_domain_profile=SimpleNamespace(eligible=True),
    )


def test_candidate_inventory_can_supply_required_noncurated_families() -> None:
    inventory = matrix.candidate_inventory()
    counts = Counter(item.family for item in inventory)

    assert all(
        counts[family] >= quota for family, quota in matrix.DEFAULT_QUOTAS.items()
    )
    assert len({item.name for item in inventory}) == len(inventory)
    assert all(item.name in AUTO_BULK_COMPONENTS for item in inventory)
    assert all(item.production_domain_profile.eligible for item in inventory)
    assert all(
        item.required_ingredients
        == {
            "lda": ("rho",),
            "gga": ("rho", "sigma"),
            "mgga": ("rho", "sigma", "tau"),
        }[item.family]
        for item in inventory
    )


def test_broad_matrix_runs_admitted_candidate_through_full_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    capability = _candidate("candidate-gga", "gga")
    calls: list[str] = []

    def production(name: str, **kwargs: object) -> dict:
        calls.append(f"production:{name}")
        return _payload("production-domain")

    def compiled(name: str, **kwargs: object) -> dict:
        calls.append(f"compiled:{name}")
        return _payload("compiled-cpu")

    def molecular(name: str, **kwargs: object) -> dict:
        calls.append(f"molecular:{name}")
        return _payload("molecular-scf")

    def public(name: str, **kwargs: object) -> dict:
        calls.append(f"public:{name}")
        return _payload("public-method")

    monkeypatch.setattr(matrix, "_eligible", lambda capability: True)
    monkeypatch.setattr(matrix, "qualify_functional", production)
    monkeypatch.setattr(matrix, "qualify_compiled_cpu", compiled)
    monkeypatch.setattr(matrix, "qualify_molecular_scf", molecular)
    monkeypatch.setattr(matrix, "qualify_public_method", public)

    summary = matrix.run_matrix(
        (capability,),
        output=tmp_path,
        evidence_prefix="test://broad-matrix",
        build_dir=tmp_path,
        pyscf_version="2.14.0",
        libxc=SimpleNamespace(__version__="7.0.0"),
        quotas={"lda": 1, "gga": 1, "mgga": 1},
    )

    assert calls == [
        "production:candidate-gga",
        "compiled:candidate-gga",
        "molecular:candidate-gga",
        "public:candidate-gga",
    ]
    assert summary["attempted_counts"]["gga"] == 1
    assert summary["production_pass_counts"]["gga"] == 1
    assert summary["public_pass_counts"]["gga"] == 1
    assert summary["functionals"][0]["status"] == "pass"
    assert (tmp_path / "candidate-gga" / "public-method.json").is_file()
    assert (tmp_path / "summary.json").is_file()


def test_production_blocker_is_retained_and_next_candidate_fills_quota(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    blocked = _candidate("blocked-gga", "gga")
    admitted = _candidate("admitted-gga", "gga")
    calls: list[str] = []

    def production(name: str, **kwargs: object) -> dict:
        calls.append(f"production:{name}")
        if name == blocked.name:
            return _payload(
                "production-domain",
                status="fail",
                reason="zero-spin boundary is unsupported",
            )
        return _payload("production-domain")

    def compiled(name: str, **kwargs: object) -> dict:
        calls.append(f"compiled:{name}")
        return _payload("compiled-cpu")

    def molecular(name: str, **kwargs: object) -> dict:
        calls.append(f"molecular:{name}")
        return _payload("molecular-scf")

    def public(name: str, **kwargs: object) -> dict:
        calls.append(f"public:{name}")
        return _payload("public-method")

    monkeypatch.setattr(matrix, "_eligible", lambda capability: True)
    monkeypatch.setattr(matrix, "qualify_functional", production)
    monkeypatch.setattr(matrix, "qualify_compiled_cpu", compiled)
    monkeypatch.setattr(matrix, "qualify_molecular_scf", molecular)
    monkeypatch.setattr(matrix, "qualify_public_method", public)

    summary = matrix.run_matrix(
        (blocked, admitted),
        output=tmp_path,
        evidence_prefix="test://broad-matrix",
        build_dir=tmp_path,
        pyscf_version="2.14.0",
        libxc=SimpleNamespace(__version__="7.0.0"),
        quotas={"lda": 1, "gga": 1, "mgga": 1},
    )

    assert calls == [
        "production:blocked-gga",
        "production:admitted-gga",
        "compiled:admitted-gga",
        "molecular:admitted-gga",
        "public:admitted-gga",
    ]
    assert summary["attempted_counts"]["gga"] == 2
    assert summary["production_pass_counts"]["gga"] == 1
    assert summary["public_pass_counts"]["gga"] == 1
    first, second = summary["functionals"]
    assert first["status"] == "blocked"
    assert first["blocker_stage"] == "production-domain"
    assert first["blocker"] == "zero-spin boundary is unsupported"
    assert first["stages"]["compiled-cpu"] == "not-run"
    assert second["status"] == "pass"


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
