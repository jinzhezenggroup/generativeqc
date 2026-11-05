"""Pinned external Ni/H inputs and independent real-spherical AO conventions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tomllib
from generativeqc import import_bse
from generativeqc_compiler.dft import NativeAO
from test_integral_autodiff_second import require_native_library  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests/data/global_hybrid"


def test_pinned_ni_def2_svp_spherical_jets_match_independent_pyscf() -> None:
    """Compare an actual d/f spherical variational space, not an s-only alias."""
    gto = pytest.importorskip("pyscf.gto")
    numint = pytest.importorskip("pyscf.dft.numint")
    from tools.generate_external_basis_references import pyscf_basis

    manifest = json.loads((DATA / "manifest.json").read_text())
    repositories = tomllib.loads((ROOT / "tools/reference_sources.toml").read_text())[
        "repository"
    ]
    source = next(item for item in repositories if item["name"] == "basis_set_exchange")
    assert manifest["source_commit"] == source["commit"]
    payload = DATA / manifest["file"]
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == manifest["sha256"]
    license_file = DATA / manifest["license_file"]
    assert (
        hashlib.sha256(license_file.read_bytes()).hexdigest()
        == source["license_sha256"]
        == manifest["license_sha256"]
    )
    raw = json.loads(payload.read_text())
    basis = import_bse(
        payload,
        source=manifest["source_repository"],
        source_version=manifest["source_commit"],
        license=manifest["license"],
        representation="spherical",
    )
    assert basis.provenance.checksum == manifest["sha256"]
    atoms = [
        ("Ni", (0.01, -0.02, 0.03)),
        ("H", (2.9, 0.1, 0.03)),
        ("H", (-3.05, 0.05, -0.02)),
        ("H", (0.03, 2.97, 0.07)),
        ("H", (-0.04, -3.1, -0.03)),
    ]
    reference = gto.M(
        atom=atoms,
        basis=pyscf_basis(raw),
        unit="Bohr",
        charge=-2,
        spin=0,
        cart=False,
        verbose=0,
    )
    points = np.array(
        ((0.2, 0.3, 0.4), (1.1, -0.4, 0.7), (4.4, 0.8, -0.6), (-0.2, 2.7, 0.1))
    )
    with NativeAO(atoms, basis=basis, representation="spherical", charge=-2) as native:
        assert native.nao == reference.nao_nr() < reference.nao_cart()
        actual = native.evaluate(points, 2)
        independent = numint.eval_ao(reference, points, deriv=2)
        np.testing.assert_allclose(actual, independent, atol=5e-11, rtol=2e-11)
