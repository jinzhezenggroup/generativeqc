"""Cold native DF-CCSD(T) force closure; independent libcint/PySCF energy FDs."""

from __future__ import annotations

import copy
import ctypes as ct
import json
import os
import typing
from pathlib import Path

import numpy as np
import pytest
from test_rhf_frame_response_cuda import reference

from tools.generativeqc_posthf.fixtures import load_fixture, source_arguments
from tools.generativeqc_posthf.sources import NativeSource
from tools.generativeqc_validation.df_gradient import reference_df_matrices

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_COMPLETE_FORCE_TEST") != "1",
    reason="requires finite Slurm real-GPU allocation and native complete endpoint",
)


@pytest.fixture(scope="module")
def probe() -> typing.Any:
    assert os.environ.get("SLURM_JOB_ID")
    lib = ct.CDLL(
        str(Path(os.environ["GENERATIVEQC_DF_COMPLETE_FORCE_PROBE"]).resolve())
    )
    call = lib.df_complete_force_probe
    dp = ct.POINTER(ct.c_double)
    call.argtypes = [
        ct.c_void_p,
        ct.c_bool,
        ct.c_bool,
        ct.c_size_t,
        dp,
        dp,
        ct.POINTER(ct.c_size_t),
        ct.c_void_p,
        ct.c_size_t,
    ]
    call.restype = ct.c_int
    return call


def run(
    probe: typing.Any,
    metadata: dict,
    *,
    forces: bool = True,
    triples: bool = True,
    budget: int = 1 << 30,
) -> tuple:
    dp = ct.POINTER(ct.c_double)
    output = np.full((len(metadata["inputs"]["atomic_numbers"]), 3), 12345.0)
    values = np.full(16, 12345.0)
    counts = np.zeros(12, dtype=np.uintp)
    error = ct.create_string_buffer(2048)
    with NativeSource(**source_arguments(metadata)) as source:
        code = probe(
            source._handle,
            forces,
            triples,
            budget,
            output.ctypes.data_as(dp),
            values.ctypes.data_as(dp),
            counts.ctypes.data_as(ct.POINTER(ct.c_size_t)),
            error,
            len(error),
        )
    return code, error.value.decode(), output, values, counts


def independent(metadata: dict, triples: bool) -> tuple[float, float, float, float]:
    """Dense tensors here are tiny independent validation oracles only.

    Preserve exact RHF Fock/orbitals and fit only the correlation interaction.
    PySCF's DF-SCF or density_fit CC setup would change that Hamiltonian.
    """
    from pyscf import ao2mo, cc, lib
    from pyscf.cc.ccsd import _ChemistsERIs

    arrays, mf = reference(metadata)
    orbital = copy.deepcopy(metadata["inputs"])
    orbital["basis_representation"] = (
        "spherical"
        if orbital["basis_representation"] == "real_spherical"
        else "cartesian"
    )
    auxiliary = {**orbital, "shells": metadata["auxiliary_shells"]}
    raw, metric, _, _ = reference_df_matrices(orbital, auxiliary)
    eig, u = np.linalg.eigh(metric)
    keep = eig > eig.max() * 1e-10
    scale = np.zeros_like(eig)
    scale[keep] = 1 / np.sqrt(eig[keep])
    root = (u * scale) @ u.T
    c = arrays[0]
    b = np.einsum("mp,nq,mnP,PQ->pqQ", c, c, raw, root, optimize=True)
    g = np.einsum("pqQ,rsQ->pqrs", b, b)
    coupled = cc.CCSD(mf)
    coupled.conv_tol = 1e-13
    coupled.conv_tol_normt = 1e-11
    coupled.max_cycle = 200
    eris = _ChemistsERIs()
    eris._common_init_(coupled)
    o, n = coupled.nocc, len(c)
    v = n - o
    eris.oooo = g[:o, :o, :o, :o].copy()
    eris.ovoo = g[:o, o:, :o, :o].copy()
    eris.ovov = g[:o, o:, :o, o:].copy()
    eris.ovvo = g[:o, o:, o:, :o].copy()
    eris.oovv = g[:o, :o, o:, o:].copy()
    eris.ovvv = lib.pack_tril(g[:o, o:, o:, o:].reshape(o * v, v, v)).reshape(o, v, -1)
    eris.vvvv = ao2mo.restore(4, g[o:, o:, o:, o:], v)
    coupled.kernel(eris=eris)
    assert coupled.converged
    perturbative = float(coupled.ccsd_t(eris=eris)) if triples else 0.0
    return (
        float(mf.e_tot + coupled.e_corr + perturbative),
        float(mf.e_tot),
        float(coupled.e_corr),
        perturbative,
    )


@pytest.mark.parametrize("name", ["h2", "water", "lih"])
@pytest.mark.parametrize("triples", [False, True])
def test_complete_native_force_against_independent_energy_directions(
    probe: typing.Any, name: str, triples: bool, tmp_path: Path
) -> None:
    metadata, _ = load_fixture(name)
    status, error, forces, values, counts = run(probe, metadata, triples=triples)
    assert status == 0, error
    expected = independent(metadata, triples)
    np.testing.assert_allclose(values[:4], expected, atol=3e-9, rtol=0)
    assert max(values[4:8]) < 1e-8
    # Canonical bilinear contraction uses one pass; admitted specialized shell
    # leases and bounded capacity fallback retain three-pass polarization.
    assert counts[6] == 0 and counts[7] in (1, 3)
    assert counts[1] > 0 and counts[2] > 0 and counts[0] <= 1 << 30
    np.testing.assert_allclose(forces.sum(axis=0), 0, atol=3e-8, rtol=0)
    direction = np.random.default_rng(1764).normal(size=forces.shape)
    direction /= np.linalg.norm(direction)
    analytic = -float(np.sum(forces * direction))
    errors = []
    for step in (1e-4, 3e-5):
        energies = []
        for sign in (-1, 1):
            displaced = copy.deepcopy(metadata)
            displaced["inputs"]["coordinates"] = (
                np.asarray(metadata["inputs"]["coordinates"]) + sign * step * direction
            ).tolist()
            energies.append(independent(displaced, triples)[0])
        numeric = (energies[1] - energies[0]) / (2 * step)
        errors.append(abs(numeric - analytic))
        np.testing.assert_allclose(numeric, analytic, atol=3e-7, rtol=3e-7)
    (tmp_path / (name + ("-t" if triples else "") + ".json")).write_text(
        json.dumps(
            {
                "values": values.tolist(),
                "counts": counts.tolist(),
                "forces": forces.tolist(),
                "independent": expected,
                "direction_errors": errors,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("duplicate", [False, True])
def test_complete_force_auxiliary_g_and_fixed_rank_metric(
    probe: typing.Any, spherical: bool, duplicate: bool
) -> None:
    metadata, _ = load_fixture("h2")
    metadata["inputs"]["basis_representation"] = (
        "real_spherical" if spherical else "cartesian"
    )
    metadata["auxiliary_shells"].append(
        {"atom_index": 0, "angular_momentum": 4, "primitives": [[0.7, 1.0]]}
    )
    if duplicate:
        metadata["auxiliary_shells"].append(
            copy.deepcopy(metadata["auxiliary_shells"][0])
        )
    status, error, force, values, _ = run(probe, metadata)
    assert status == 0, error
    np.testing.assert_allclose(
        values[:4], independent(metadata, True), atol=3e-9, rtol=0
    )
    step = 3e-5
    energies = []
    for sign in (-1, 1):
        displaced = copy.deepcopy(metadata)
        displaced["inputs"]["coordinates"][1][2] += sign * step
        energies.append(independent(displaced, True)[0])
    np.testing.assert_allclose(
        -force[1, 2], (energies[1] - energies[0]) / (2 * step), atol=3e-7, rtol=3e-7
    )


def test_native_energy_force_finite_difference_and_failure_publication(
    probe: typing.Any,
) -> None:
    metadata, _ = load_fixture("water")
    status, error, forces, values, _ = run(probe, metadata)
    assert status == 0, error
    energy = run(probe, metadata, forces=False)
    assert energy[0] == 0, energy[1]
    np.testing.assert_allclose(values[:4], energy[3][:4], atol=3e-10, rtol=0)
    step = 1e-4
    shifted = []
    for sign in (-1, 1):
        displaced = copy.deepcopy(metadata)
        displaced["inputs"]["coordinates"][1][0] += sign * step
        result = run(probe, displaced, forces=False)
        assert result[0] == 0, result[1]
        shifted.append(result[3][0])
    np.testing.assert_allclose(
        -forces[1, 0], (shifted[1] - shifted[0]) / (2 * step), atol=3e-7, rtol=3e-7
    )
    rejected = run(probe, metadata, budget=1)
    assert rejected[0] != 0
    assert np.all(rejected[2] == 12345.0) and np.all(rejected[3] == 12345.0)


def test_complete_water_all_nuclear_coordinates(
    probe: typing.Any, tmp_path: Path
) -> None:
    """Every component, two independent energy steps, nonzero triples included."""
    metadata, _ = load_fixture("water")
    status, error, force, values, _ = run(probe, metadata)
    assert status == 0, error
    assert abs(values[3]) > 1e-6
    errors = []
    for step in (1e-4, 3e-5):
        numeric = np.zeros_like(force)
        for atom in range(len(force)):
            for axis in range(3):
                energies = []
                for sign in (-1, 1):
                    displaced = copy.deepcopy(metadata)
                    displaced["inputs"]["coordinates"][atom][axis] += sign * step
                    energies.append(independent(displaced, True)[0])
                numeric[atom, axis] = -(energies[1] - energies[0]) / (2 * step)
        np.testing.assert_allclose(force, numeric, atol=3e-7, rtol=3e-7)
        errors.append(float(np.max(np.abs(force - numeric))))
    (tmp_path / "water-all-coordinates.json").write_text(
        json.dumps(
            {
                "forces": force.tolist(),
                "coordinate_max_errors": errors,
                "values": values.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
