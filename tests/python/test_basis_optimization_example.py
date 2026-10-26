"""End-to-end basis optimization uses public derivatives and an independent oracle."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from test_integral_autodiff_second import require_native_library  # noqa: F401

from examples.optimize_gaussian_basis import optimize_basis


def test_basis_optimization_descends_and_matches_independent_engine(
    tmp_path: Path,
) -> None:
    report = optimize_basis(steps=2, cache=str(tmp_path), reference=True)
    energies = np.array(report["accepted_energies_hartree"])
    assert np.all(np.diff(energies) < 0)
    assert energies[-1] < energies[0] - 1e-4
    assert np.all(np.asarray(report["exponents"]) > 0)
    assert max(report["independent_reference"]["absolute_errors"]) < 5e-8
    assert all(
        item["seconds"] > 0 and item["work"]["primitive_pullback_evaluations"] > 0
        for item in report["endpoint_evaluations"]
    )
