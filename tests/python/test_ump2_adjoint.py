"""Independent four-spin fixed-orbital acceptance for the #1823 MO slice."""

from __future__ import annotations

import typing
from itertools import product

import numpy as np
import pytest

from tools.generativeqc_mp2 import unrestricted_adjoint as adj

SPINS = ("alpha", "beta")
CHANNELS = ("alpha_alpha", "beta_beta", "alpha_beta")


def problem(occupied: tuple[int, int] = (2, 1), nmo: int = 5) -> tuple:
    rng = np.random.default_rng(1823)
    counts = dict(zip(SPINS, occupied))
    eps = {
        s: np.linspace(-1.8, -0.7, no).tolist()
        + np.linspace(0.2, 1.3, nmo - no).tolist()
        for s, no in counts.items()
    }
    eps = {
        s: np.array(e, dtype=np.float64) + k * 0.03
        for k, (s, e) in enumerate(eps.items())
    }
    blocks = {}
    for channel in CHANNELS:
        left, right = channel.split("_")
        nl, nr = counts[left], counts[right]
        blocks[channel] = rng.normal(scale=0.1, size=(nl, nr, nmo - nl, nmo - nr))
    return blocks, eps, counts


def four_spin_oracle(blocks: dict, eps: dict, counts: dict) -> tuple:
    """All ordered *spin-orbital* excitations with spin Kronecker deltas.

    This independently enumerates aa, ab, ba, bb plus virtual spin exchanges;
    it never uses the three-channel TensorIR or its derivatives. beta-alpha
    Coulomb reads use pair interchange of the owned alpha-beta block.
    """
    occ = [(s, p) for s in SPINS for p in range(counts[s])]
    vir = [(s, p) for s in SPINS for p in range(counts[s], len(eps[s]))]
    gbars = {c: np.zeros_like(g) for c, g in blocks.items()}
    ebars = {s: np.zeros_like(e) for s, e in eps.items()}

    def coulomb(i: tuple, j: tuple, a: tuple, b: tuple) -> tuple:
        if i[0] != a[0] or j[0] != b[0]:
            return 0.0, None
        si, sj = i[0], j[0]
        index = (i[1], j[1], a[1] - counts[si], b[1] - counts[sj])
        channel = f"{si}_{sj}"
        if channel == "beta_alpha":
            channel = "alpha_beta"
            index = (index[1], index[0], index[3], index[2])
        return blocks[channel][index], (channel, index)

    energy = 0.0
    for i, j, a, b in product(occ, occ, vir, vir):
        direct, gd = coulomb(i, j, a, b)
        exchange, gx = coulomb(i, j, b, a)
        v = direct - exchange
        d = eps[i[0]][i[1]] + eps[j[0]][j[1]] - eps[a[0]][a[1]] - eps[b[0]][b[1]]
        energy += 0.25 * v * v / d
        for source, sign in ((gd, 1), (gx, -1)):
            if source is not None:
                gbars[source[0]][source[1]] += sign * 0.5 * v / d
        for orbital, sign in ((i, 1), (j, 1), (a, -1), (b, -1)):
            ebars[orbital[0]][orbital[1]] -= sign * 0.25 * v * v / d**2
    return energy, gbars, ebars


def canonical(
    blocks: dict, eps: dict, counts: dict, **kwargs: typing.Any
) -> adj.UMP2CanonicalAdjoint:
    return adj.canonical_energy_adjoint(
        blocks,
        eps,
        counts,
        reference_identity="fixed-orbital-seed-1823",
        hamiltonian_id="unscreened-test-eri",
        **kwargs,
    )


def inner(result: adj.UMP2CanonicalAdjoint, dg: dict, de: dict) -> float:
    return sum(float(np.sum(result.integrals_iajb[c] * dg[c])) for c in CHANNELS) + sum(
        float(np.dot(result.orbital_energies[s], de[s])) for s in SPINS
    )


@pytest.mark.parametrize("counts", [(2, 1), (2, 2), (0, 1), (5, 5), (0, 0)])
def test_four_spin_oracle_and_rectangular_tile_global_assembly(counts: tuple) -> None:
    blocks, eps, occupied = problem(counts)
    _, gbar, ebar = four_spin_oracle(blocks, eps, occupied)
    tiled = canonical(blocks, eps, occupied, tile_shape=(1, 2, 2, 3))
    global_result = canonical(blocks, eps, occupied, tile_shape=(9, 9, 9, 9))
    for c in CHANNELS:
        np.testing.assert_allclose(
            tiled.integrals_iajb[c], gbar[c], atol=2e-15, rtol=2e-13
        )
        np.testing.assert_allclose(
            tiled.integrals_iajb[c], global_result.integrals_iajb[c], atol=2e-15
        )
    for s in SPINS:
        np.testing.assert_allclose(
            tiled.orbital_energies[s], ebar[s], atol=2e-15, rtol=2e-13
        )
        np.testing.assert_allclose(
            tiled.orbital_energies[s], global_result.orbital_energies[s], atol=2e-15
        )
    if counts in ((5, 5), (0, 0)):
        assert tiled.tile_count == 0
        assert tiled.equation_hashes == tiled.derivative_hashes == ()
        assert tiled.minimum_absolute_denominator == 0.0


def test_fixed_orbital_directional_perturbations_and_negative_controls() -> None:
    blocks, eps, counts = problem()
    rng = np.random.default_rng(91823)
    dg = {c: rng.normal(size=g.shape) for c, g in blocks.items()}
    de = {s: rng.normal(size=e.shape) for s, e in eps.items()}
    result = canonical(blocks, eps, counts)
    exact = inner(result, dg, de)
    for h in (1e-4, 2e-5, 1e-6):
        energies = [
            four_spin_oracle(
                {c: g + sign * h * dg[c] for c, g in blocks.items()},
                {s: e + sign * h * de[s] for s, e in eps.items()},
                counts,
            )[0]
            for sign in (1, -1)
        ]
        fd = (energies[0] - energies[1]) / (2 * h)
        assert abs(fd - exact) < 2e-7
    assert abs(fd + exact) > 1e-3  # Wrong derivative sign cannot pass.
    wrong_factor = exact + sum(
        float(np.sum(result.integrals_iajb[c] * dg[c])) for c in CHANNELS[:2]
    )
    assert abs(fd - wrong_factor) > 1e-3
    wrong_eps = exact - 2 * sum(
        float(np.dot(result.orbital_energies[s], de[s])) for s in SPINS
    )
    assert abs(fd - wrong_eps) > 1e-3


@pytest.mark.parametrize("channel", CHANNELS)
def test_rectangular_disjoint_feed_partials(channel: str) -> None:
    rng = np.random.default_rng(12823)
    shape = (2, 1, 3, 2)
    feeds = {
        "g": rng.normal(size=shape),
        "ei": np.array([-1.2, -0.8]),
        "ej": np.array([-1.1]),
        "ea": np.array([0.1, 0.4, 0.7]),
        "eb": np.array([0.3, 0.9]),
    }
    if channel != "alpha_beta":
        feeds["x"] = rng.normal(size=shape)
    result = adj.tile_energy_adjoint(feeds, channel=channel)
    bars = dict(
        zip(
            ("g", "x", *adj.ENERGY_FEEDS),
            (
                result.direct,
                result.exchange,
                result.occupied_i,
                result.occupied_j,
                result.virtual_a,
                result.virtual_b,
            ),
        )
    )

    def energy(values: dict) -> float:
        total = 0.0
        for i, j, a, b in np.ndindex(shape):
            d = values["ei"][i] + values["ej"][j] - values["ea"][a] - values["eb"][b]
            v = values["g"][i, j, a, b]
            if "x" in values:
                total += 0.25 * (v - values["x"][i, j, a, b]) ** 2 / d
            else:
                total += v**2 / d
        return total

    for name, value in feeds.items():
        direction = rng.normal(size=value.shape)
        h = 1e-6
        plus, minus = dict(feeds), dict(feeds)
        plus[name], minus[name] = value + h * direction, value - h * direction
        fd = (energy(plus) - energy(minus)) / (2 * h)
        assert abs(fd - np.sum(bars[name] * direction)) < 2e-8
    if result.exchange is not None:
        np.testing.assert_allclose(result.exchange, -result.direct, atol=1e-14)
    else:
        assert channel == "alpha_beta"
    assert result.derivative_hash and result.equation_hash


def test_alpha_beta_permutation() -> None:
    blocks, eps, counts = problem()
    result = canonical(blocks, eps, counts)
    flipped = {
        "alpha_alpha": blocks["beta_beta"],
        "beta_beta": blocks["alpha_alpha"],
        "alpha_beta": blocks["alpha_beta"].transpose(1, 0, 3, 2),
    }
    other = canonical(
        flipped,
        dict(zip(SPINS, (eps["beta"], eps["alpha"]))),
        dict(zip(SPINS, (counts["beta"], counts["alpha"]))),
    )
    np.testing.assert_allclose(
        result.integrals_iajb["alpha_alpha"], other.integrals_iajb["beta_beta"]
    )
    np.testing.assert_allclose(
        result.integrals_iajb["beta_beta"], other.integrals_iajb["alpha_alpha"]
    )
    np.testing.assert_allclose(
        result.integrals_iajb["alpha_beta"],
        other.integrals_iajb["alpha_beta"].transpose(1, 0, 3, 2),
    )
    for left, right in (("alpha", "beta"), ("beta", "alpha")):
        np.testing.assert_allclose(
            result.orbital_energies[left], other.orbital_energies[right], atol=2e-15
        )


def test_strided_fortran_buffers_and_spin_equation_identity() -> None:
    blocks, eps, counts = problem((2, 2))
    expected = canonical(blocks, eps, counts)
    strided = {}
    for c, g in blocks.items():
        backing = np.zeros((*g.shape[:-1], 2 * g.shape[-1]))
        backing[..., ::2] = g
        strided[c] = backing[..., ::2]
    strided["beta_beta"] = np.asfortranarray(blocks["beta_beta"])
    other = canonical(strided, eps, counts)
    for c in CHANNELS:
        np.testing.assert_allclose(other.integrals_iajb[c], expected.integrals_iajb[c])
    shape = (2, 2, 3, 3)
    feeds = {
        "g": blocks["alpha_alpha"],
        "x": blocks["alpha_alpha"].swapaxes(2, 3),
        "ei": eps["alpha"][:2],
        "ej": eps["alpha"][:2],
        "ea": eps["alpha"][2:],
        "eb": eps["alpha"][2:],
    }
    assert feeds["g"].shape == shape
    alpha = adj.tile_energy_adjoint(feeds, channel="alpha_alpha")
    beta = adj.tile_energy_adjoint(feeds, channel="beta_beta")
    assert alpha.equation_hash != beta.equation_hash
    assert alpha.derivative_hash != beta.derivative_hash


@pytest.mark.parametrize(
    "kind",
    [
        "unknown",
        "os-exchange",
        "missing-exchange",
        "exchange-shape",
        "eps-shape",
        "empty",
        "overflow",
    ],
)
def test_tile_failure_contracts(kind: str) -> None:
    feeds = {
        "g": np.ones((1, 1, 1, 1)),
        "x": np.zeros((1, 1, 1, 1)),
        "ei": np.array([-1.0]),
        "ej": np.array([-1.0]),
        "ea": np.array([1.0]),
        "eb": np.array([1.0]),
    }
    channel = "alpha_alpha"
    if kind == "unknown":
        channel = "beta_alpha"
    elif kind == "os-exchange":
        channel = "alpha_beta"
    elif kind == "missing-exchange":
        del feeds["x"]
    elif kind == "exchange-shape":
        feeds["x"] = np.zeros((1, 1, 1, 2))
    elif kind == "eps-shape":
        feeds["ei"] = np.ones(2)
    elif kind == "empty":
        feeds["g"] = np.empty((0, 1, 1, 1))
    else:
        feeds["g"].fill(1e308)
    with pytest.raises(ValueError):
        adj.tile_energy_adjoint(feeds, channel=channel)


def test_restricted_limit_cumulative_weights_independent_rhf_perturbations() -> None:
    blocks, eps, counts = problem((2, 2))
    g, e = blocks["alpha_alpha"], eps["alpha"]
    result = canonical(dict.fromkeys(CHANNELS, g), dict.fromkeys(SPINS, e), counts)
    gbar = sum(result.integrals_iajb.values())
    ebar = sum(result.orbital_energies.values())
    rng = np.random.default_rng(8823)
    dg, de = rng.normal(size=g.shape), rng.normal(size=e.shape)

    def rhf(values: np.ndarray, energies: np.ndarray) -> float:
        return sum(
            values[i, j, a, b]
            * (2 * values[i, j, a, b] - values[i, j, b, a])
            / (energies[i] + energies[j] - energies[a + 2] - energies[b + 2])
            for i, j, a, b in np.ndindex(values.shape)
        )

    h = 1e-6
    fd = (rhf(g + h * dg, e + h * de) - rhf(g - h * dg, e - h * de)) / (2 * h)
    assert abs(fd - np.sum(gbar * dg) - np.dot(ebar, de)) < 2e-9
    assert (
        abs(fd - np.sum(result.integrals_iajb["alpha_beta"] * dg) - np.dot(ebar, de))
        > 1e-3
    )


def test_budget_rejected_before_vjp_or_output_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocks, eps, counts = problem()
    baseline = canonical(blocks, eps, counts)
    assert (
        canonical(
            blocks, eps, counts, budget_bytes=baseline.numeric_capacity_bytes
        ).numeric_capacity_bytes
        == baseline.numeric_capacity_bytes
    )
    monkeypatch.setattr(
        adj, "vjp", lambda *a, **k: pytest.fail("VJP before budget rejection")
    )
    monkeypatch.setattr(
        adj.np,
        "zeros_like",
        lambda *a, **k: pytest.fail("output before budget rejection"),
    )
    with pytest.raises(MemoryError):
        canonical(blocks, eps, counts, budget_bytes=baseline.numeric_capacity_bytes - 1)
    g = blocks["alpha_beta"]
    feeds = {
        "g": g,
        "ei": eps["alpha"][:2],
        "ej": eps["beta"][:1],
        "ea": eps["alpha"][2:],
        "eb": eps["beta"][1:],
    }
    with pytest.raises(MemoryError):
        adj.tile_energy_adjoint(feeds, channel="alpha_beta", budget_bytes=1)


@pytest.mark.parametrize(
    "kind",
    [
        "float32",
        "complex",
        "nan",
        "shape",
        "occupation",
        "near-zero",
        "positive-denominator",
        "threshold",
        "spin",
        "identity",
        "tile",
        "budget",
    ],
)
def test_fail_closed_contracts(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    blocks, eps, counts = problem()
    options = {}
    if kind == "float32":
        blocks["alpha_beta"] = blocks["alpha_beta"].astype(np.float32)
    elif kind == "complex":
        eps["alpha"] = eps["alpha"].astype(complex)
    elif kind == "nan":
        blocks["beta_beta"].flat[0] = np.nan
    elif kind == "shape":
        blocks["alpha_beta"] = blocks["alpha_beta"].transpose(1, 0, 3, 2)
    elif kind == "occupation":
        counts["alpha"] = True
    elif kind == "near-zero":
        eps = {s: np.zeros_like(e) for s, e in eps.items()}
    elif kind == "positive-denominator":
        eps = {s: -e for s, e in eps.items()}
    elif kind == "threshold":
        options["denominator_threshold"] = 0
    elif kind == "spin":
        blocks["beta_alpha"] = blocks.pop("alpha_beta")
    elif kind == "identity":
        with pytest.raises(ValueError):
            adj.canonical_energy_adjoint(
                blocks, eps, counts, reference_identity="", hamiltonian_id="test"
            )
        return
    elif kind == "tile":
        options["tile_shape"] = (1, 0, 2, 2)
    else:
        options["budget_bytes"] = False
    monkeypatch.setattr(adj, "vjp", lambda *a, **k: pytest.fail("AD must not run"))
    with pytest.raises((ValueError, TypeError)):
        canonical(blocks, eps, counts, **options)


def test_owned_immutable_results_and_late_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocks, eps, counts = problem()
    saved = {c: g.copy() for c, g in blocks.items()}
    result = canonical(blocks, eps, counts)
    outputs = [*result.integrals_iajb.values(), *result.orbital_energies.values()]
    for out in outputs:
        assert all(
            not np.shares_memory(out, a)
            for a in (
                *blocks.values(),
                *eps.values(),
                *[b for b in outputs if b is not out],
            )
        )
        with pytest.raises(ValueError):
            out.setflags(write=True)
    with pytest.raises(TypeError):
        result.integrals_iajb["alpha_beta"] = blocks["alpha_beta"]
    original = adj.vjp
    calls = 0

    def fail_second(*args: typing.Any, **kwargs: typing.Any) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected late failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(adj, "vjp", fail_second)
    with pytest.raises(ValueError, match="late failure"):
        canonical(blocks, eps, counts)
    assert calls == 2
    for c in CHANNELS:
        np.testing.assert_array_equal(blocks[c], saved[c])


@pytest.mark.parametrize("case", ["li-open-shell", "h2-broken-symmetry"])
def test_actual_uhf_fixed_orbital_perturbations(case: str) -> None:
    """Linux reference CI includes pinned PySCF; NumPy-only hosts skip this gate."""
    pyscf = pytest.importorskip("pyscf")
    assert pyscf.__version__ == "2.14.0"
    from pyscf import gto, mp, scf

    atoms = (
        [("Li", (0, 0, 0))]
        if case == "li-open-shell"
        else [("H", (0, 0, -2)), ("H", (0, 0, 2))]
    )
    mol = gto.M(
        atom=atoms,
        basis="sto-3g",
        spin=int(case == "li-open-shell"),
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = scf.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 200
    seed = None
    if case == "h2-broken-symmetry":
        seed = np.zeros((2, 2, 2))
        seed[0, 0, 0], seed[1, 1, 1] = 1, 1
    mf.kernel(dm0=seed)
    assert mf.converged
    if seed is not None:
        assert np.linalg.norm(mf.make_rdm1()[0] - mf.make_rdm1()[1]) > 0.1
    counts = dict(zip(SPINS, map(int, mol.nelec)))
    eps = {s: np.asarray(mf.mo_energy[k]) for k, s in enumerate(SPINS)}
    coeff = dict(zip(SPINS, mf.mo_coeff))
    eri = mol.intor("int2e", aosym="s1")
    blocks = {}
    for c in CHANNELS:
        left, right = c.split("_")
        nl, nr = counts[left], counts[right]
        blocks[c] = np.einsum(
            "uvwx,ui,va,wj,xb->ijab",
            eri,
            coeff[left][:, :nl],
            coeff[left][:, nl:],
            coeff[right][:, :nr],
            coeff[right][:, nr:],
            optimize=True,
        )
    oracle_energy, gbars, ebars = four_spin_oracle(blocks, eps, counts)
    assert abs(oracle_energy - mp.UMP2(mf).run().e_corr) < 1e-10
    result = canonical(blocks, eps, counts)
    for c in CHANNELS:
        np.testing.assert_allclose(result.integrals_iajb[c], gbars[c], atol=1e-12)
    for s in SPINS:
        np.testing.assert_allclose(result.orbital_energies[s], ebars[s], atol=1e-12)
    rng = np.random.default_rng(1823988)
    dg = {c: rng.normal(size=g.shape) for c, g in blocks.items()}
    de = {s: rng.normal(size=e.shape) for s, e in eps.items()}
    h = 1e-6
    energies = [
        four_spin_oracle(
            {c: g + sign * h * dg[c] for c, g in blocks.items()},
            {s: e + sign * h * de[s] for s, e in eps.items()},
            counts,
        )[0]
        for sign in (1, -1)
    ]
    assert abs((energies[0] - energies[1]) / (2 * h) - inner(result, dg, de)) < 2e-8
