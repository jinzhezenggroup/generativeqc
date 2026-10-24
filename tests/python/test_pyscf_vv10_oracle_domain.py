"""The optional matched-domain oracle cannot silently change PySCF defaults."""

import pytest

from tools._pyscf_vv10_domain import vv10_density_domain


def test_explicit_oracle_domain_is_scoped_and_restored() -> None:
    pyscf = pytest.importorskip("pyscf")
    from pyscf.dft import libxc, numint

    if pyscf.__version__ != "2.14.0" or libxc.libxc_version() != "7.0.0":
        pytest.skip("requires the pinned independent qualification oracle")
    original = numint._vv10nlc
    with vv10_density_domain(1e-8):
        assert numint._vv10nlc is original
    with pytest.raises(RuntimeError, match="deliberate"), vv10_density_domain(1e-10):
        assert numint._vv10nlc is not original
        raise RuntimeError("deliberate oracle consumer failure")
    assert numint._vv10nlc is original
    with pytest.raises(ValueError, match="explicitly"), vv10_density_domain(1e-9):
        pass
    assert numint._vv10nlc is original


def test_matched_domain_retains_tail_on_both_pair_legs() -> None:
    pyscf = pytest.importorskip("pyscf")
    import numpy as np
    from pyscf.dft import libxc, numint

    if pyscf.__version__ != "2.14.0" or libxc.libxc_version() != "7.0.0":
        pytest.skip("requires the pinned independent qualification oracle")
    rho = np.array([[1e-9, 0.4], [1e-10, 0.04], [0.0, 0.0], [0.0, 0.0]])
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    weights = np.array([0.2, 0.3])
    with vv10_density_domain(1e-8):
        original, _ = numint._vv10nlc(rho, points, rho, weights, points, (6.0, 0.01))
    with vv10_density_domain(1e-10):
        matched, _ = numint._vv10nlc(rho, points, rho, weights, points, (6.0, 0.01))
        without_inner_tail, _ = numint._vv10nlc(
            rho, points, rho[:, 1:].copy(), weights[1:], points[1:], (6.0, 0.01)
        )
    assert original[0] == 0.0
    assert matched[0] != 0.0
    assert matched[1] != without_inner_tail[1]
