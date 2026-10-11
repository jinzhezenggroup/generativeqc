"""Reference-oracle center binding checks without the optional PySCF dependency."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from tools.generativeqc_validation.second_derivatives import libcint_primitive_gradient


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction", "eri"])
def test_primitive_gradient_binds_shell_centers_and_preserves_charge_origin(
    monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    shell_count = 4 if family == "eri" else 2
    center_count = 3 if family == "nuclear_attraction" else shell_count
    centers = np.arange(center_count * 3, dtype=float).reshape(center_count, 3) / 10
    captured = {}
    origins = []

    class Molecule:
        def intor(self, name: str) -> np.ndarray:
            assert name == "int1e_ovlp_cart"
            return np.eye(shell_count)

        def intor_by_shell(
            self, name: str, slots: tuple[int, ...], *, comp: int
        ) -> np.ndarray:
            assert name.startswith("int2e_" if family == "eri" else "int1e_")
            assert len(slots) == shell_count
            return np.arange(1, comp + 1, dtype=float).reshape(
                (comp,) + (1,) * shell_count
            )

        def with_rinv_origin(self, origin: np.ndarray | tuple[int, ...]) -> object:
            origins.append(np.asarray(origin).copy())
            return nullcontext()

    def molecule(**kwargs: object) -> Molecule:
        captured.update(kwargs)
        return Molecule()

    monkeypatch.setitem(
        sys.modules, "pyscf", SimpleNamespace(gto=SimpleNamespace(M=molecule))
    )
    result = libcint_primitive_gradient(
        family, (0,) * shell_count, (0.8,) * shell_count, centers, charge=2.3
    )

    assert [name for name, _ in captured["atom"]] == [
        f"ghost-{name}" for name in ("H", "He", "Li", "Be")[:shell_count]
    ]
    np.testing.assert_array_equal(
        [position for _, position in captured["atom"]], centers[:shell_count]
    )
    assert result.shape == (center_count, 3) + (1,) * shell_count
    assert np.isfinite(result).all()
    if family == "nuclear_attraction":
        np.testing.assert_array_equal(origins, centers[2:3])
        np.testing.assert_allclose(result[2], -result[0] - result[1])
    elif family != "eri":
        np.testing.assert_array_equal(origins, np.zeros((1, 3)))
    else:
        assert origins == []
