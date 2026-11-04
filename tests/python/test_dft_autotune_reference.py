"""Oracle dispatch contracts; synthetic owners are not scientific GPU evidence."""

from __future__ import annotations

import copy
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import _dft_autotune, tune_cli


@pytest.fixture
def oracle(monkeypatch: pytest.MonkeyPatch) -> Any:
    state = SimpleNamespace(
        workload={
            "kind": "dft",
            "reference_engine": "gpu4pyscf",
            "method": "pbe0-rks",
            "atoms": [["H", [0.0, 0.0, -0.7]], ["H", [0.0, 0.0, 0.7]]],
            "basis": "sto-3g",
            "charge": 0,
            "multiplicity": 1,
            "representation": "spherical",
            "grid_level": 3,
            "device_id": 2,
            "batch": 2,
        },
        device=None,
        events=[],
        converged=True,
        host_gradient=False,
    )

    class Device:
        def __init__(self, ordinal: int) -> None:
            self.ordinal = ordinal

        def __enter__(self) -> None:
            state.device = self.ordinal
            state.events.append("enter")

        def __exit__(self, *args: object) -> None:
            state.events.append("exit")
            state.device = None

    class Gradient:
        __module__ = "gpu4pyscf.grad.rks"

        def kernel(self) -> np.ndarray:
            assert state.device == 2 and self.grid_response is True
            state.events.append("gradient")
            return np.full((2, 3), 0.125)

    class Solver:
        __module__ = "gpu4pyscf.dft.rks"

        def __init__(self, molecule: Any) -> None:
            assert state.device == 2
            self.grids = SimpleNamespace()
            self.converged = state.converged
            self.e_tot = -1.0
            state.solver = self

        def kernel(self) -> None:
            assert state.device == 2
            state.events.append("scf")

        def nuc_grad_method(self) -> Any:
            if state.host_gradient:
                return SimpleNamespace()
            return Gradient()

    class Unrestricted(Solver):
        __module__ = "gpu4pyscf.dft.uks"

    def molecule(**kwargs: Any) -> Any:
        state.molecule = kwargs
        return SimpleNamespace()

    def forbidden_cpu_solver(*args: Any) -> None:
        pytest.fail("GPU reference must not execute a CPU SCF/gradient")

    def synchronize() -> None:
        assert state.device == 2
        state.events.append("synchronize")

    monkeypatch.setitem(
        sys.modules,
        "pyscf",
        SimpleNamespace(
            __version__="synthetic",
            dft=SimpleNamespace(RKS=forbidden_cpu_solver, UKS=forbidden_cpu_solver),
            gto=SimpleNamespace(M=molecule),
            lib=SimpleNamespace(num_threads=lambda count: None),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "gpu4pyscf", SimpleNamespace(__version__="synthetic")
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu4pyscf.dft",
        SimpleNamespace(
            rks=SimpleNamespace(RKS=Solver), uks=SimpleNamespace(UKS=Unrestricted)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            asnumpy=np.asarray,
            cuda=SimpleNamespace(
                Device=Device,
                get_current_stream=lambda: SimpleNamespace(synchronize=synchronize),
            ),
        ),
    )
    return state


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks", "b3lyp-rks"))
def test_gpu_oracle_keeps_model_response_and_visible_ordinal(
    oracle: Any, method: str
) -> None:
    oracle.workload["method"] = method
    oracle.workload["representation"] = "cartesian"
    oracle.workload["charge"] = 1
    oracle.workload["multiplicity"] = 2
    original = copy.deepcopy(oracle.workload)
    reference = _dft_autotune.reference_workload(oracle.workload)
    assert oracle.workload == original
    assert oracle.events == ["enter", "scf", "gradient", "synchronize", "exit"]
    assert oracle.molecule == {
        "atom": original["atoms"],
        "basis": "sto-3g",
        "unit": "Bohr",
        "charge": 1,
        "spin": 1,
        "cart": True,
        "verbose": 0,
    }
    assert oracle.solver.xc == _dft_autotune.DFT_FUNCTIONALS[method]
    assert oracle.solver.grids.level == 3 and oracle.solver.small_rho_cutoff == 0.0
    assert (oracle.solver.conv_tol, oracle.solver.conv_tol_grad) == (1e-12, 1e-10)
    assert oracle.solver.direct_scf_tol == 1e-14 and oracle.solver.max_cycle == 150
    assert type(oracle.solver).__module__.endswith(
        "uks" if method.endswith("uks") else "rks"
    )
    assert reference["reference_engine"] == "gpu4pyscf"
    assert reference["energies"] == [-1.0, -1.0]
    assert np.asarray(reference["forces"]).shape == (2, 2, 3)
    assert np.all(np.asarray(reference["forces"]) == -0.125)


@pytest.mark.parametrize("failure", ("convergence", "host-gradient", "missing-package"))
def test_gpu_oracle_fails_closed_without_cpu_fallback(
    oracle: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    oracle.converged = failure != "convergence"
    oracle.host_gradient = failure == "host-gradient"
    if failure == "missing-package":
        monkeypatch.setitem(sys.modules, "gpu4pyscf", None)
    with pytest.raises((RuntimeError, ImportError)):
        _dft_autotune.reference_workload(oracle.workload)
    assert "gradient" not in oracle.events
    assert oracle.device is None


def test_gpu_oracle_cli_is_explicit_and_dft_only() -> None:
    assert tune_cli.parser().parse_args(["unused.xyz"]).reference_engine == "pyscf"
    with pytest.raises(SystemExit):
        tune_cli.main(
            ["unused.xyz", "--method", "rhf", "--reference-engine", "gpu4pyscf"]
        )
