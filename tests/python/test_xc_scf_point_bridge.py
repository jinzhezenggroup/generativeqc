"""Private #163-A point bridge preserves the exact #162 SCF-domain ABI."""

from pathlib import Path

import numpy as np
import pytest
from vibeqc import _native
from vibeqc._ks_snapshot import _generic_scf_xc_points, _scf_xc_points


def test_scf_point_bridge_matches_independent_domain_fixture() -> None:
    data = np.loadtxt(Path(__file__).resolve().parents[1] / "data/xc/scf_domain.tsv")
    library = _native.load_library(device="cpu")
    for pbe in (False, True):
        rows = data[data[:, 0] == int(pbe)]
        rho = rows[:, 2:4].T
        gradient = rows[:, 4:10].reshape(-1, 2, 3).transpose(1, 0, 2)
        expected = rows[:, 10:19]
        actual = _scf_xc_points(library, pbe, rho, gradient)
        packed = np.column_stack(
            [
                actual["energy"],
                actual["rho"].T,
                actual["gradient"].transpose(1, 0, 2).reshape(-1, 6),
            ]
        )
        tolerance = 5.0e-10 * np.abs(expected) + 1.0e-322
        assert np.all(np.isfinite(packed))
        assert np.all(np.abs(packed - expected) <= tolerance)


def test_scf_point_bridge_rejects_layout_and_invalid_domain() -> None:
    library = _native.load_library(device="cpu")
    with pytest.raises(ValueError, match=r"rho\[2,n\]"):
        _scf_xc_points(library, False, np.ones((3, 2)), np.zeros((2, 2, 3)))
    rho = np.array([[-1.0], [1.0]])
    with pytest.raises(RuntimeError, match="numerical failure"):
        _scf_xc_points(library, True, rho, np.zeros((2, 1, 3)))



def test_generic_scf_point_bridge_preserves_native_program_layout() -> None:
    program = _native.KsSemilocalProgramDescriptor(
        b"GGA_X_TEST",
        b"expression",
        7,
        3,
        1,
    )

    class Evaluate:
        def __call__(
            self,
            descriptor: typing.Any,
            rho: typing.Any,
            gradient: typing.Any,
            tau: typing.Any,
            point_count: int,
            values: typing.Any,
            value_count: int,
        ) -> int:
            assert descriptor._obj.identifier == b"GGA_X_TEST"
            assert point_count == 2
            assert value_count == 22
            output = np.ctypeslib.as_array(values, shape=(value_count,))
            output[:] = np.arange(value_count, dtype=np.float64) + 0.5
            return _native.STATUS_SUCCESS

    library = typing.cast(
        typing.Any,
        type(
            "Library",
            (),
            {"vibeqc_libxc_semilocal_program_evaluate_v1": Evaluate()},
        )(),
    )
    actual = _generic_scf_xc_points(
        library,
        program,
        np.ones((2, 2)),
        np.zeros((2, 2, 3)),
    )
    assert actual["energy"].tolist() == [0.5, 11.5]
    assert actual["rho"].shape == (2, 2)
    assert actual["gradient"].shape == (2, 2, 3)
    assert actual["kinetic"].shape == (2, 2)


def test_generic_meta_gga_point_bridge_requires_tau() -> None:
    program = _native.KsSemilocalProgramDescriptor(
        b"MGGA_X_TEST",
        b"expression",
        15,
        3,
        1,
    )
    with pytest.raises(ValueError, match="requires tau"):
        _generic_scf_xc_points(
            object(),
            program,
            np.ones((2, 1)),
            np.zeros((2, 1, 3)),
        )
