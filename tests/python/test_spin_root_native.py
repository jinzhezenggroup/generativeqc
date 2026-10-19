"""Public native spin-root endpoint against pinned independent molecular oracles.

Identical input basis/grid, independent libcint/Libxc/SCF/analytic gradients and
finite rotations. PySCF is forbidden during the production call itself.
"""

import builtins
import typing

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._spin_root_native import _NativeSpinModel
from generativeqc.spin_roots import SpinRootPolicy, SpinRootUnqualified
from generativeqc_compiler.dft import NativeAO
from test_dft_complete_cpu import independent_global_hybrid_gradient

GRID = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)


def calculator(method: str) -> typing.Any:
    return Calculator(
        method,
        basis="sto-3g",
        ks_options=KsOptions(grid=GRID) if method.endswith("-uks") else None,
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-11,
        initial_guess=None,
    )


def h2(distance: typing.Any) -> typing.Any:
    return [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, distance))]


def execute(
    calc: typing.Any,
    atoms: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: typing.Any,
) -> typing.Any:
    original = builtins.__import__

    def guarded(name: typing.Any, *args: typing.Any, **kw: typing.Any) -> typing.Any:
        if name == "pyscf" or name.startswith(
            ("pyscf.", "tools.generativeqc_response")
        ):
            raise AssertionError(
                "production root endpoint imported an independent oracle"
            )
        return original(name, *args, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guarded)
        return calc.stable_singlepoint(atoms, **kwargs)


def oracle(
    basis: typing.Any, root: typing.Any, method: str, model: typing.Any
) -> typing.Any:
    pyscf = pytest.importorskip("pyscf")
    assert pyscf.__version__ == "2.14.0"
    from pyscf import gto, lib, scf
    from pyscf.data.elements import ELEMENTS

    lib.num_threads(1)
    if method.endswith("-uks"):
        _, derivative, mf = independent_global_hybrid_gradient(
            basis,
            model.state,
            method,
            xc="PBE" if method == "pbe-uks" else "LDA_X,LDA_C_PW",
            initial_density=np.array(root.reference.density),
            return_mean_field=True,
        )
    else:
        labels = [f"{ELEMENTS[a.atomic_number]}{i}" for i, a in enumerate(basis.atoms)]
        shells = {label: [] for label in labels}
        for shell in basis.shells:
            shells[labels[shell.atom_index]].append(
                [
                    shell.angular_momentum,
                    *[(p.exponent, p.coefficient) for p in shell.primitives],
                ]
            )
        mol = gto.M(
            atom=[
                (label, a.position)
                for label, a in zip(labels, basis.atoms, strict=True)
            ],
            basis=shells,
            charge=basis.charge,
            spin=basis.multiplicity - 1,
            unit="Bohr",
            cart=basis.representation == "cartesian",
            verbose=0,
        )
        mf = scf.UHF(mol)
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
        mf.kernel(dm0=np.array(root.reference.density))
        derivative = mf.nuc_grad_method().kernel()
    assert mf.converged
    np.testing.assert_allclose(
        mf.make_rdm1(), root.reference.density, rtol=0, atol=3e-8
    )
    np.testing.assert_allclose(mf.e_tot, root.reference.energy, rtol=0, atol=3e-9)
    np.testing.assert_allclose(
        mf.spin_square()[0], root.spin_squared, rtol=0, atol=3e-8
    )
    for spin in range(2):
        np.testing.assert_allclose(
            np.diag(mf.make_rdm1()[spin] @ mf.get_ovlp()),
            (root.alpha_populations, root.beta_populations)[spin],
            rtol=0,
            atol=3e-8,
        )
    if root.forces is not None:
        np.testing.assert_allclose(root.forces, -derivative, rtol=0, atol=3e-7)
    return mf


def independent_hessian(root: typing.Any, mf: typing.Any) -> None:
    """Libcint/Libxc response in independent packed occupied-major coordinates."""
    c, eps = root.reference.coefficients, root.reference.orbital_energies
    counts = root.reference.occupied
    n = c.shape[1]
    sizes = [o * (n - o) for o in counts]
    action = mf.gen_response(hermi=1)

    def apply(vector: typing.Any) -> typing.Any:
        blocks = np.split(vector, (sizes[0],))
        density = []
        for cs, o, x in zip(c, counts, blocks, strict=True):
            co, cv = cs[:, :o], cs[:, o:]
            delta = co @ x.reshape(o, n - o) @ cv.T
            density.append(delta + delta.T)
        potentials = action(np.array(density))
        return 2 * np.concatenate(
            [
                (
                    (e[o:][None, :] - e[:o][:, None]) * x.reshape(o, n - o)
                    + cs[:, :o].T @ v @ cs[:, o:]
                ).ravel()
                for cs, e, o, x, v in zip(
                    c, eps, counts, blocks, potentials, strict=True
                )
            ]
        )

    eye = np.eye(sum(sizes))
    matrix = np.column_stack([apply(x) for x in eye])
    np.testing.assert_allclose(matrix, matrix.T, rtol=0, atol=3e-9)
    values, vectors = np.linalg.eigh(matrix)
    np.testing.assert_allclose(root.eigenvalues, values, rtol=3e-8, atol=3e-8)
    if len(values):
        from scipy.linalg import expm

        x = vectors[:, 0]
        energies = []
        for step in (1e-3, 3e-4):
            trials = []
            for sign in (1, -1):
                density = []
                for cs, o, block in zip(
                    c, counts, np.split(x, (sizes[0],)), strict=True
                ):
                    k = np.zeros((n, n))
                    k[:o, o:] = block.reshape(o, n - o)
                    k[o:, :o] = -k[:o, o:].T
                    co = (cs @ expm(-sign * step * k))[:, :o]
                    density.append(co @ co.T)
                trials.append(mf.energy_tot(dm=np.array(density)))
            energies.append((sum(trials) - 2 * mf.e_tot) / step**2)
        np.testing.assert_allclose(energies, values[0], rtol=2e-5, atol=3e-6)


@pytest.fixture(
    params=[(m, d) for m in ("uhf", "lda-uks", "pbe-uks") for d in (1.4, 4.0)],
    scope="module",
)
def molecular(request: typing.Any) -> typing.Any:
    method, distance = request.param
    calc = calculator(method)
    root = calc.stable_singlepoint(h2(distance), properties=("energy", "forces"))
    return method, distance, calc, root


def test_public_stable_and_broken_singlets_independent_energy_force_and_spectrum(
    molecular: typing.Any,
) -> None:
    method, distance, calc, root = molecular
    assert root.initial_status == ("STABLE" if distance == 1.4 else "UNSTABLE")
    assert root.recovered == (distance == 4.0)
    assert root.diagnostics["status"] == "STABLE"
    assert root.diagnostics["scf_residual"] <= 1e-8
    if root.recovered:
        assert (
            root.reference.energy
            < root.diagnostics["native_scf_calls"][0]["energy"] - 1e-3
        )
        assert root.diagnostics["distinct_roots"] == 1
        assert root.diagnostics["equivalent_restarts"] > 0
    with NativeAO(h2(distance), basis="sto-3g") as basis:
        model = _NativeSpinModel(
            calc,
            basis,
            root.reference.model_identity,
            charge=0,
            multiplicity=1,
            properties=("energy",),
            policy=SpinRootPolicy(),
            seed=root.reference.density,
            seed_energy=root.reference.energy,
        )
        try:
            mf = oracle(basis, root, method, model)
            independent_hessian(root, mf)
        finally:
            model.close()


def test_reconverged_geometry_force_and_native_tracking(
    molecular: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _method, distance, calc, root = molecular
    step = 2e-4
    energies = []
    for sign in (1, -1):
        target = execute(
            calc,
            h2(distance + sign * step),
            monkeypatch,
            previous=root,
            properties=("energy",),
        )
        assert target.root_relation in ("same", "spin_equivalent")
        assert target.diagnostics["previous"] == root.identity
        assert all(
            x["execution_owner"] == "native_cpu"
            for x in target.diagnostics["projection"]
        )
        assert any(x["seed_consumed"] for x in target.diagnostics["native_scf_calls"])
        energies.append(target.energy)
    np.testing.assert_allclose(
        root.forces[1, 2], -(energies[0] - energies[1]) / (2 * step), rtol=0, atol=3e-6
    )


@pytest.mark.parametrize("method", ["uhf", "lda-uks", "pbe-uks"])
def test_ordinary_open_shell_li_independent_reference(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calc = calculator(method)
    atoms = [("Li", (0.1, -0.2, 0.3))]
    root = execute(
        calc, atoms, monkeypatch, multiplicity=2, properties=("energy", "forces")
    )
    assert root.reference.occupied == (2, 1) and not root.recovered
    with NativeAO(atoms, basis="sto-3g", multiplicity=2) as basis:
        model = _NativeSpinModel(
            calc,
            basis,
            root.reference.model_identity,
            charge=0,
            multiplicity=2,
            properties=("energy",),
            policy=SpinRootPolicy(),
            seed=root.reference.density,
            seed_energy=root.reference.energy,
        )
        try:
            mf = oracle(basis, root, method, model)
            independent_hessian(root, mf)
        finally:
            model.close()


@pytest.mark.parametrize("method", ["ump2"])
def test_unrestricted_posthf_uses_actual_qualified_reference(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyscf import cc, mp

    calc = calculator(method)
    root = execute(calc, h2(4.0), monkeypatch, properties=("energy",))
    assert root.recovered
    assert root.result.correlation.reference_energy == root.reference.energy
    with NativeAO(h2(4.0), basis="sto-3g") as basis:
        mf = oracle(basis, root, method, None)
        correlated = mp.UMP2(mf) if method == "ump2" else cc.UCCSD(mf)
        correlated.conv_tol = 1e-11
        correlated.kernel()
        energy = correlated.e_tot
        if method == "uccsd(t)":
            energy += correlated.ccsd_t()
        np.testing.assert_allclose(root.energy, energy, rtol=0, atol=3e-8)


@pytest.mark.parametrize("properties", [("energy",), ("energy", "forces")])
def test_failed_qualification_blocks_requested_publication(
    properties: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calc = calculator("uhf")
    with pytest.raises(SpinRootUnqualified, match="UNSTABLE"):
        execute(
            calc,
            h2(4.0),
            monkeypatch,
            properties=properties,
            policy=SpinRootPolicy(recover=False),
        )
    with pytest.raises(SpinRootUnqualified, match="before native SCF"):
        execute(
            calc,
            h2(4.0),
            monkeypatch,
            properties=properties,
            policy=SpinRootPolicy(maximum_actions=1),
        )


def test_root_switch_requires_explicit_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calc = calculator("uhf")
    root = execute(calc, h2(4.0), monkeypatch, properties=("energy",))
    with pytest.raises(SpinRootUnqualified, match="intended"):
        execute(
            calc,
            h2(4.02),
            monkeypatch,
            properties=("energy", "forces"),
            previous=root,
            policy=SpinRootPolicy(tracking_tolerance=1e-4),
        )
    switched = execute(
        calc,
        h2(4.02),
        monkeypatch,
        properties=("energy",),
        previous=root,
        policy=SpinRootPolicy(tracking_tolerance=1e-4, allow_root_switch=True),
    )
    assert switched.root_relation == "switched"
    with pytest.raises(SpinRootUnqualified, match="different model"):
        calculator("pbe-uks").stable_singlepoint(
            h2(4.02), previous=root, properties=("energy",)
        )


def test_empty_real_rotation_domain_still_qualifies_stationarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = execute(
        calculator("uhf"),
        [("H", (0.0, 0.0, 0.0))],
        monkeypatch,
        multiplicity=2,
        properties=("energy",),
    )
    assert root.eigenvalues.size == 0
    assert root.diagnostics["status"] == "EMPTY"
    assert root.diagnostics["lowest_curvature"] is None
    assert not root.recovered


@pytest.mark.parametrize("properties", [("energy",), ("energy", "forces")])
def test_retained_native_density_drift_cannot_publish(
    properties: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import _spin_root_native

    calc = calculator("uhf")
    good = execute(calc, h2(1.4), monkeypatch, properties=("energy",))
    read_density = _spin_root_native._read_density

    def drifted_density(batch: typing.Any, nbf: int) -> np.ndarray:
        density = np.array(read_density(batch, nbf))
        # Preserve exact electron counts/idempotency and the native convergence
        # flag, but move the actual occupied space beyond the physical gate.
        c = good.reference.coefficients[0]
        occupied = c[:, 0] * np.cos(4e-8) + c[:, 1] * np.sin(4e-8)
        density[0] = np.outer(occupied, occupied)
        return density

    monkeypatch.setattr(_spin_root_native, "_read_density", drifted_density)
    with pytest.raises(SpinRootUnqualified, match="actual model"):
        execute(calc, h2(1.4), monkeypatch, properties=properties)
