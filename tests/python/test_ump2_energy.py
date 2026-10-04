"""Canonical unrestricted MP2 energy contracts for #1820."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from generativeqc_compiler.mp2.equations import unrestricted_energy_program

from tools.generativeqc_mp2 import PreparedMP2Energy, PreparedUMP2Energy
from tools.generativeqc_posthf.reference import ReferenceSnapshot
from tools.generativeqc_response.uhf import UHFReferenceSnapshot


class DenseSource:
    backend = "dense-test-oracle"

    def __init__(self, eri: np.ndarray, reference: UHFReferenceSnapshot) -> None:
        self.ao = np.ascontiguousarray(eri)
        self.nbf = reference.nbf
        self.geometry_hash = reference.geometry_hash
        self.basis_hash = reference.basis_hash
        self.representation = reference.representation
        self.shell_sizes = (self.nbf,)
        self.numeric_bytes = self.ao.nbytes
        self.identity = "ump2-dense-source"
        self.reads = 0
        self.closed = False

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("fixture source is closed")

    def requests(self, operator, *, axis_tile, budget_bytes):
        assert operator == "four_center_eri"
        assert budget_bytes > 0
        for starts in product(range(0, self.nbf, axis_tile), repeat=4):
            yield starts, tuple(min(axis_tile, self.nbf - begin) for begin in starts)

    def tile(self, request):
        self.reads += 1
        starts, sizes = request
        return np.ascontiguousarray(
            self.ao[
                tuple(slice(begin, begin + size) for begin, size in zip(starts, sizes))
            ]
        )

    def global_offsets(self, request):
        return request[0]


def _orthogonal(seed: int, nbf: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(nbf, nbf)))
    q *= np.sign(np.diag(r))
    return q


def _reference(
    *,
    alpha_coefficients: np.ndarray | None = None,
    beta_coefficients: np.ndarray | None = None,
    alpha_energies: np.ndarray | None = None,
    beta_energies: np.ndarray | None = None,
    alpha_occupied: int = 2,
    beta_occupied: int = 1,
) -> UHFReferenceSnapshot:
    nbf = 4
    ca = _orthogonal(1820, nbf) if alpha_coefficients is None else alpha_coefficients
    cb = _orthogonal(1821, nbf) if beta_coefficients is None else beta_coefficients
    ea = (
        np.array([-1.25, -0.43, 0.31, 0.88])
        if alpha_energies is None
        else alpha_energies
    )
    eb = (
        np.array([-1.02, 0.12, 0.57, 1.04])
        if beta_energies is None
        else beta_energies
    )
    oa = np.zeros(nbf)
    ob = np.zeros(nbf)
    oa[:alpha_occupied] = 1
    ob[:beta_occupied] = 1
    return UHFReferenceSnapshot(
        overlap=np.eye(nbf),
        hcore=np.zeros((nbf, nbf)),
        fock_alpha=ca @ np.diag(ea) @ ca.T,
        fock_beta=cb @ np.diag(eb) @ cb.T,
        coefficients_alpha=ca,
        coefficients_beta=cb,
        orbital_energies_alpha=ea,
        orbital_energies_beta=eb,
        occupations_alpha=oa,
        occupations_beta=ob,
        reference_energy=-7.0,
        scf_residual=1e-12,
        geometry_hash="ump2-geometry",
        basis_hash="ump2-basis",
        generation_id="ump2-generation",
    )


def _eri(nbf: int = 4) -> np.ndarray:
    rng = np.random.default_rng(1822)
    factor = rng.normal(scale=0.15, size=(nbf, nbf, nbf + 1))
    factor = 0.5 * (factor + factor.swapaxes(0, 1))
    return np.einsum("uvP,wxP->uvwx", factor, factor, optimize=True)


def _oracle(reference: UHFReferenceSnapshot, eri: np.ndarray):
    components = {}
    for name, left, right in (
        ("alpha_alpha", "alpha", "alpha"),
        ("beta_beta", "beta", "beta"),
        ("alpha_beta", "alpha", "beta"),
    ):
        nl, nr = reference.nocc(left), reference.nocc(right)
        cl = getattr(reference, f"coefficients_{left}")
        cr = getattr(reference, f"coefficients_{right}")
        el = getattr(reference, f"orbital_energies_{left}")
        er = getattr(reference, f"orbital_energies_{right}")
        if not nl or not nr or nl == reference.nmo or nr == reference.nmo:
            components[name] = 0.0
            continue
        g = np.einsum(
            "uvwx,ui,va,wj,xb->ijab",
            eri,
            cl[:, :nl],
            cl[:, nl:],
            cr[:, :nr],
            cr[:, nr:],
            optimize=True,
        )
        denominator = (
            el[:nl, None, None, None]
            + er[None, :nr, None, None]
            - el[None, None, nl:, None]
            - er[None, None, None, nr:]
        )
        if left == right:
            antisymmetrized = g - g.swapaxes(2, 3)
            components[name] = float(
                0.25 * np.sum(antisymmetrized * antisymmetrized / denominator)
            )
        else:
            components[name] = float(np.sum(g * g / denominator))
    return components


def test_tensorir_keeps_spin_in_index_space_identity() -> None:
    same = unrestricted_energy_program((2, 1, 3, 2), channel="alpha_alpha")
    mixed = unrestricted_energy_program((2, 1, 3, 2), channel="alpha_beta")
    inputs = {
        node.attrs["name"]: tuple(index.space.spin for index in node.spec.indices)
        for node in mixed.live_nodes
        if node.op == "input"
    }
    assert inputs["g"] == ("alpha", "beta", "alpha", "beta")
    assert inputs["ei"] == ("alpha",)
    assert inputs["ej"] == ("beta",)
    assert same.logical_hash != mixed.logical_hash


def test_spin_resolved_tiles_match_independent_dense_contractions() -> None:
    reference = _reference()
    eri = _eri()
    source = DenseSource(eri, reference)
    expected = _oracle(reference, eri)
    with PreparedUMP2Energy(
        reference,
        source,
        occupied_tile=1,
        virtual_tile=2,
        axis_tile=2,
    ) as prepared:
        result = prepared.execute()
        np.testing.assert_allclose(
            [result.alpha_alpha, result.beta_beta, result.alpha_beta],
            [
                expected["alpha_alpha"],
                expected["beta_beta"],
                expected["alpha_beta"],
            ],
            atol=2e-12,
            rtol=2e-12,
        )
        assert abs(result.correlation_energy - sum(expected.values())) < 2e-12
        assert abs(
            result.energy - (reference.reference_energy + sum(expected.values()))
        ) < 2e-12
        assert result.numeric_capacity_bytes <= 256 << 20
        assert result.tile_count > 0
        assert result.equation_hashes
    assert source.reads > 0


def test_restricted_limit_matches_existing_mp2_components() -> None:
    coefficients = _orthogonal(1823, 4)
    energies = np.array([-1.1, -0.5, 0.35, 0.92])
    unrestricted = _reference(
        alpha_coefficients=coefficients,
        beta_coefficients=coefficients,
        alpha_energies=energies,
        beta_energies=energies,
        alpha_occupied=2,
        beta_occupied=2,
    )
    eri = _eri()
    source = DenseSource(eri, unrestricted)
    restricted = ReferenceSnapshot(
        overlap=np.eye(4),
        hcore=np.zeros((4, 4)),
        fock=coefficients @ np.diag(energies) @ coefficients.T,
        coefficients=coefficients,
        orbital_energies=energies,
        occupations=np.array([2.0, 2.0, 0.0, 0.0]),
        electron_count=4,
        reference_energy=unrestricted.reference_energy,
        scf_residual=1e-12,
        geometry_hash=unrestricted.geometry_hash,
        basis_hash=unrestricted.basis_hash,
        generation_id="restricted-limit",
    )
    with PreparedUMP2Energy(unrestricted, source) as u, PreparedMP2Energy(
        restricted, source
    ) as r:
        ur = u.execute()
        rr = r.execute()
    np.testing.assert_allclose(
        ur.alpha_beta, rr.opposite_spin, atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(
        ur.alpha_alpha + ur.beta_beta,
        rr.same_spin,
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        ur.correlation_energy, rr.correlation_energy, atol=2e-12, rtol=2e-12
    )


def test_spin_swap_and_failure_boundaries() -> None:
    reference = _reference()
    eri = _eri()
    original_source = DenseSource(eri, reference)
    original = PreparedUMP2Energy(reference, original_source).execute()

    swapped = UHFReferenceSnapshot(
        overlap=reference.overlap,
        hcore=reference.hcore,
        fock_alpha=reference.fock_beta,
        fock_beta=reference.fock_alpha,
        coefficients_alpha=reference.coefficients_beta,
        coefficients_beta=reference.coefficients_alpha,
        orbital_energies_alpha=reference.orbital_energies_beta,
        orbital_energies_beta=reference.orbital_energies_alpha,
        occupations_alpha=reference.occupations_beta,
        occupations_beta=reference.occupations_alpha,
        reference_energy=reference.reference_energy,
        scf_residual=reference.scf_residual,
        geometry_hash=reference.geometry_hash,
        basis_hash=reference.basis_hash,
        generation_id="ump2-spin-swapped",
    )
    swapped_result = PreparedUMP2Energy(
        swapped, DenseSource(eri, swapped)
    ).execute()
    np.testing.assert_allclose(
        original.alpha_alpha, swapped_result.beta_beta, atol=2e-12
    )
    np.testing.assert_allclose(
        original.beta_beta, swapped_result.alpha_alpha, atol=2e-12
    )
    np.testing.assert_allclose(
        original.alpha_beta, swapped_result.alpha_beta, atol=2e-12
    )

    source = DenseSource(eri, reference)
    probe = PreparedUMP2Energy(reference, source)
    required = probe.numeric_capacity_bytes
    probe.close()
    assert source.reads == 0
    with pytest.raises(MemoryError, match="UMP2 needs"):
        PreparedUMP2Energy(reference, source, budget_bytes=required - 1)
    assert source.reads == 0

    prepared = PreparedUMP2Energy(reference, source)
    source.identity = "changed-source"
    with pytest.raises(ValueError, match="source changed"):
        prepared.execute()
    assert prepared.state == "failed" and prepared.last_result is None
    prepared.close()


def test_denominator_and_reference_fail_closed_before_source_reads() -> None:
    reference = _reference()
    source = DenseSource(_eri(), reference)
    close_alpha = np.array([-1.0, -0.2, -0.19999999999, 0.8])
    close = replace(
        reference,
        orbital_energies_alpha=close_alpha,
        fock_alpha=reference.coefficients_alpha
        @ np.diag(close_alpha)
        @ reference.coefficients_alpha.T,
    )
    with pytest.raises(ValueError, match="near-zero UMP2 denominator"):
        PreparedUMP2Energy(close, source, denominator_threshold=1e-9)
    assert source.reads == 0
    with pytest.raises(NotImplementedError, match="forces/amplitudes"):
        PreparedUMP2Energy(reference, source).execute(
            properties=("energy", "forces")
        )


def test_pyscf_open_shell_ump2_total_energy() -> None:
    pyscf = pytest.importorskip("pyscf")
    assert pyscf.__version__ == "2.14.0"
    from pyscf import gto, mp, scf

    mol = gto.M(
        atom=[("Li", (0.0, 0.0, 0.0))],
        basis="sto-3g",
        spin=1,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    independent = mp.UMP2(mf).run()

    density = mf.make_rdm1()
    fock = mf.get_fock(dm=density)
    coefficients = mf.mo_coeff
    energies = mf.mo_energy
    occupations = mf.mo_occ
    reference = UHFReferenceSnapshot(
        overlap=mf.get_ovlp(),
        hcore=mf.get_hcore(),
        fock_alpha=fock[0],
        fock_beta=fock[1],
        coefficients_alpha=coefficients[0],
        coefficients_beta=coefficients[1],
        orbital_energies_alpha=energies[0],
        orbital_energies_beta=energies[1],
        occupations_alpha=occupations[0],
        occupations_beta=occupations[1],
        reference_energy=mf.e_tot,
        scf_residual=1e-12,
        geometry_hash="pyscf-li-geometry",
        basis_hash="pyscf-sto3g",
        generation_id="pyscf-ump2-open-shell",
        validation_tolerance=1e-7,
    )
    source = DenseSource(
        mol.intor("int2e", aosym="s1").reshape((mol.nao_nr(),) * 4),
        reference,
    )
    result = PreparedUMP2Energy(
        reference, source, occupied_tile=1, virtual_tile=2, axis_tile=2
    ).execute()
    assert abs(result.energy - independent.e_tot) <= 1e-9
