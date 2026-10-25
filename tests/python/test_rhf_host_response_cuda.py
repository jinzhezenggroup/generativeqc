"""Tiered exact response against independent molecular derivatives on Slurm."""

from __future__ import annotations

import copy
import ctypes as ct
import os
import typing
from pathlib import Path

import numpy as np
import pytest
from test_rhf_frame_response_cuda import probe as frame_probe  # noqa: F401
from test_rhf_frame_response_cuda import reference, run, seeds

from tools.generativeqc_posthf.fixtures import load_fixture

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RHF_FRAME_CUDA_TEST") != "1",
    reason="requires finite Slurm allocation and native response probe",
)


@pytest.fixture
def cache_probe(frame_probe: typing.Any) -> typing.Any:  # noqa: F811
    """Retain the original probe contract and expose actual tier consumption."""
    assert os.environ.get("SLURM_JOB_ID")
    library = ct.CDLL(str(Path(os.environ["GENERATIVEQC_RHF_FRAME_PROBE"]).resolve()))
    function = library.rhf_frame_response_cache_probe
    ip, dp = ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_double)
    function.argtypes = [*frame_probe.argtypes, ip, dp]
    function.restype = ct.c_int

    def call(
        metadata: dict, arrays: list, sources: list, **options: typing.Any
    ) -> tuple:
        cache, phases = np.zeros(8, dtype=np.uintp), np.zeros(8)

        def invoke(*arguments: typing.Any) -> int:
            return function(
                *arguments, cache.ctypes.data_as(ip), phases.ctypes.data_as(dp)
            )

        result = run(invoke, metadata, arrays, sources, **options)
        return (*result, cache, phases)

    return call


@pytest.mark.parametrize(
    "name,spherical", [("water", False), ("water_d", True), ("h2_f", True)]
)
def test_tiered_response_preserves_independent_derivatives(
    cache_probe: typing.Any, monkeypatch: pytest.MonkeyPatch, name: str, spherical: bool
) -> None:
    """Changed geometry, signed Z actions and final uncached audits share no oracle ERIs."""
    metadata, _ = load_fixture("h2" if name == "h2_f" else "water")
    if name in ("water_d", "h2_f"):
        metadata["inputs"]["shells"].append(
            {
                "atom_index": len(metadata["inputs"]["atomic_numbers"]) - 1,
                "angular_momentum": 3 if name == "h2_f" else 2,
                "primitives": [[0.8, 1.0]],
            }
        )
    if spherical:
        metadata["inputs"]["basis_representation"] = "real_spherical"
    arrays, mf = reference(metadata)
    sources, weights = seeds(arrays, mf.mol.nelectron // 2, 0.03)
    monkeypatch.setenv("GENERATIVEQC_TEST_RHF_RESIDENT_JK_BYTES", str(64 << 20))
    monkeypatch.setenv("GENERATIVEQC_RHF_RESIDENT_BUCKET_MASK", "1")
    monkeypatch.setenv("GENERATIVEQC_RHF_RESPONSE_HOST_TILE_BYTES", "512")
    monkeypatch.setenv("GENERATIVEQC_RHF_RESPONSE_HOST_MAX_BYTES", "0")
    gpu = cache_probe(metadata, arrays, sources)
    assert gpu[0] == 0, gpu[1]
    monkeypatch.delenv("GENERATIVEQC_RHF_RESPONSE_HOST_MAX_BYTES")
    cached = cache_probe(metadata, arrays, sources)
    assert cached[0] == 0, cached[1]
    assert cached[5][0] == gpu[5][0] > 0  # GPU priority is preserved.
    assert cached[5][3] > 0 and cached[5][4] > cached[5][3]
    assert cached[5][5] > 0 and cached[5][6] == cached[5][3] * cached[5][5]
    assert cached[3][10] > 0 and cached[4][0] < 1e-10
    assert 0 < cached[3][14] < gpu[3][14]  # Final audit still recomputes.
    for actual, expected in zip(cached[2], gpu[2], strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=3e-10)

    gradient = cached[2][0] + mf.nuc_grad_method().grad_nuc()
    direction = np.random.default_rng(1967).normal(size=gradient.shape)
    direction /= np.linalg.norm(direction)
    for step in (1e-4, 3e-5):
        energies = []
        for sign in (-1, 1):
            moved = copy.deepcopy(metadata)
            moved["inputs"]["coordinates"] = (
                np.asarray(metadata["inputs"]["coordinates"]) + sign * step * direction
            ).tolist()
            shifted, independent = reference(moved)
            energies.append(
                float(
                    independent.e_tot
                    + np.sum(weights * shifted[4])
                    + np.diag(sources[0]) @ shifted[5]
                )
            )
        np.testing.assert_allclose(
            (energies[1] - energies[0]) / (2 * step),
            np.sum(gradient * direction),
            atol=3e-7,
            rtol=3e-7,
        )
    # A new owner must regenerate the displaced source, not reuse same-shape values.
    moved_result = cache_probe(
        moved, shifted, seeds(shifted, mf.mol.nelectron // 2, 0.0)[0]
    )
    assert moved_result[0] == 0, moved_result[1]
    expected_gradient = independent.nuc_grad_method()
    np.testing.assert_allclose(
        moved_result[2][0] + expected_gradient.grad_nuc(),
        expected_gradient.kernel(),
        atol=2e-8,
        rtol=2e-8,
    )


@pytest.mark.parametrize(
    "capacity,mask", [("0", None), ("1", None), (str(1 << 20), "0")]
)
def test_host_capacity_refusal_keeps_gpu_and_exact_response(
    cache_probe: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    capacity: str,
    mask: str | None,
) -> None:
    """Disabling or refusing optional RAM must preserve already admitted GPU values."""
    metadata, _ = load_fixture("water")
    arrays, mf = reference(metadata)
    sources, _ = seeds(arrays, mf.mol.nelectron // 2, 0.03)
    monkeypatch.setenv("GENERATIVEQC_TEST_RHF_RESIDENT_JK_BYTES", str(64 << 20))
    monkeypatch.setenv("GENERATIVEQC_RHF_RESIDENT_BUCKET_MASK", "1")
    monkeypatch.setenv("GENERATIVEQC_RHF_RESPONSE_HOST_MAX_BYTES", capacity)
    if mask is not None:
        monkeypatch.setenv("GENERATIVEQC_RHF_RESPONSE_HOST_BUCKET_MASK", mask)
    result = cache_probe(metadata, arrays, sources)
    assert result[0] == 0, result[1]
    assert result[5][0] > 0 and result[5][3] == 0 and result[5][6] == 0
    assert result[4][0] < 1e-10 and result[4][1] < 1e-8


def test_small_automatic_response_retains_uncached_schedule(
    cache_probe: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An available RAM allowance cannot override the existing small-frame policy."""
    metadata, _ = load_fixture("water")
    arrays, mf = reference(metadata)
    sources, _ = seeds(arrays, mf.mol.nelectron // 2, 0.03)
    monkeypatch.delenv("GENERATIVEQC_TEST_RHF_RESIDENT_JK_BYTES", raising=False)
    result = cache_probe(metadata, arrays, sources)
    assert result[0] == 0, result[1]
    assert result[5][0] == 0 and result[5][3] == 0 and result[3][14] > 0


@pytest.mark.parametrize("tile", ["0", "7", str((512 << 20) + 8), "bad"])
def test_invalid_staging_policy_cannot_publish_response(
    cache_probe: typing.Any, monkeypatch: pytest.MonkeyPatch, tile: str
) -> None:
    """Resource-policy errors leave the caller's scientific destinations untouched."""
    metadata, _ = load_fixture("water")
    arrays, mf = reference(metadata)
    sources, _ = seeds(arrays, mf.mol.nelectron // 2, 0.03)
    monkeypatch.setenv("GENERATIVEQC_TEST_RHF_RESIDENT_JK_BYTES", str(64 << 20))
    monkeypatch.setenv("GENERATIVEQC_RHF_RESPONSE_HOST_MAX_BYTES", str(1 << 20))
    monkeypatch.setenv("GENERATIVEQC_RHF_RESPONSE_HOST_TILE_BYTES", tile)
    result = cache_probe(metadata, arrays, sources)
    assert result[0] != 0 and "host ERI" in result[1]
    assert all(np.all(output == 12345.0) for output in result[2])


def test_automatic_large_frame_uses_remaining_ram(
    cache_probe: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real 70-AO frame admits partial GPU values and RAM without an opt-in."""
    metadata, _ = load_fixture("water")
    original = copy.deepcopy(metadata)
    inputs = metadata["inputs"]
    inputs["atomic_numbers"], inputs["coordinates"], inputs["shells"] = [], [], []
    metadata["auxiliary_shells"] = []
    for molecule in range(10):
        shift = np.array([8.0 * (molecule % 5), 8.0 * (molecule // 5), 0.0])
        inputs["atomic_numbers"].extend(original["inputs"]["atomic_numbers"])
        inputs["coordinates"].extend(
            (np.asarray(original["inputs"]["coordinates"]) + shift).tolist()
        )
        for destination, source in (
            (inputs["shells"], original["inputs"]["shells"]),
            (metadata["auxiliary_shells"], original["auxiliary_shells"]),
        ):
            for shell in source:
                item = copy.deepcopy(shell)
                item["atom_index"] += 3 * molecule
                destination.append(item)
    metadata["records"]["conventional"]["electron_count"] *= 10
    arrays, mf = reference(metadata)
    assert len(arrays[0]) == 70
    sources, _ = seeds(arrays, mf.mol.nelectron // 2, 0.01)
    monkeypatch.delenv("GENERATIVEQC_TEST_RHF_RESIDENT_JK_BYTES", raising=False)
    monkeypatch.delenv("GENERATIVEQC_RHF_RESPONSE_HOST_MAX_BYTES", raising=False)
    monkeypatch.delenv("GENERATIVEQC_RHF_RESPONSE_HOST_TILE_BYTES", raising=False)
    monkeypatch.delenv("GENERATIVEQC_RHF_RESIDENT_BUCKET_MASK", raising=False)
    monkeypatch.setenv("GENERATIVEQC_RHF_RESIDENT_MAX_BYTES", str(4 << 20))
    partial = cache_probe(metadata, arrays, sources)
    assert partial[0] == 0, partial[1]
    assert 0 < partial[5][0] <= 4 << 20
    assert partial[5][3] > 0 and partial[5][5] > 0
    assert partial[3][0] <= 1 << 30
    assert partial[4][0] < 1e-10 and partial[4][1] < 1e-8
    # Plenty of VRAM must retain complete GPU residency, without staging RAM.
    monkeypatch.delenv("GENERATIVEQC_RHF_RESIDENT_MAX_BYTES")
    full = cache_probe(metadata, arrays, sources)
    assert full[0] == 0, full[1]
    assert full[5][0] == partial[5][0] + partial[5][3]
    assert full[5][3] == 0 and full[5][6] == 0
    for actual, expected in zip(partial[2], full[2], strict=True):
        np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=3e-10)
