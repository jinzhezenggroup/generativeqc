"""Offline integrity and independent-oracle consistency, not native qualification."""

import hashlib
import json
from pathlib import Path

import numpy as np

REFERENCES = Path(__file__).parents[1] / "data/global_hybrid/ni-second-order"


def test_retained_independent_arrays_are_complete_and_authenticated() -> None:
    """Protect scientific input bytes and ordinary numeric shapes without pickle."""
    manifest = json.loads((REFERENCES / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        path = REFERENCES / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == set(record["arrays"])
            for field, expected in record["arrays"].items():
                value = archive[field]
                assert list(value.shape) == expected["shape"]
                assert str(value.dtype) == expected["dtype"]
                assert value.dtype.kind != "O"
                if value.dtype.kind in "fc":
                    assert np.isfinite(value).all()


def test_independent_ni_oracles_are_mutually_consistent_before_native_comparison() -> (
    None
):
    """A reference pass cannot stand in for executing the installed public endpoint."""
    matrix = np.load(REFERENCES / "reference-matrix.npz")
    actions = np.load(REFERENCES / "reference-actions.npz")
    components = np.load(REFERENCES / "reference-components.npz")
    center = np.load(REFERENCES / "reference-center.npz")
    assert matrix["complete"].item()
    assert matrix["matrices"].shape == (3, 15, 15)
    assert float(matrix["raw_matrix_gate"]) == 1e-5
    assert float(center["orbital_residual"]) < 2e-9
    for raw in matrix["matrices"]:
        np.testing.assert_allclose(raw, raw.T, atol=2e-7, rtol=0)
        np.testing.assert_allclose(
            raw @ np.tile(np.eye(3), (5, 1)), 0, atol=2e-7, rtol=0
        )
        errors = abs(
            raw @ actions["direction"].ravel() - actions["reference_hvp"].reshape(3, 15)
        )
        assert np.max(errors) / np.linalg.norm(actions["direction"]) < 1e-5
    np.testing.assert_allclose(
        components["components"].sum(axis=1),
        actions["reference_hvp"],
        atol=3e-8,
        rtol=0,
    )
    for name, points in (("reference-grid.npz", 5760), ("default-grid.npz", 59872)):
        grid = np.load(REFERENCES / name)
        assert grid["points"].shape == (points, 3)
        assert grid["weights"].shape == grid["owners"].shape == (points,)
