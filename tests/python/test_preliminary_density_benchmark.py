"""Host guards for private same-basis cold experiments and policy isolation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from generativeqc import load_basis

from benchmarks import ks_preliminary_density as seed
from benchmarks.readme_omol25 import force_execution_options


def test_force_capacity_overrides_preserve_omitted_defaults_and_dense_fallback() -> (
    None
):
    assert force_execution_options(active_ao=False) == {}
    assert force_execution_options(active_ao=False, max_device_bytes=4 << 30) == {
        "max_device_bytes": 4 << 30
    }
    assert force_execution_options(active_ao=True, max_host_bytes=4 << 30) == {
        "max_host_bytes": 4 << 30,
        "active_ao_cutoff": 1e-16,
        "active_ao_cache_bytes": 64 << 20,
    }


@pytest.mark.parametrize("field", ["max_device_bytes", "max_host_bytes"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, (1 << 40) + 1])
def test_force_capacity_override_rejects_invalid_before_gpu_work(
    field: str, invalid: object
) -> None:
    with pytest.raises(ValueError, match="force .* must be an integer"):
        force_execution_options(active_ao=False, **{field: invalid})


def target() -> SimpleNamespace:
    """Use the full diffuse/f-shell snapshot, never a renamed small basis."""
    basis = load_basis(
        Path(__file__).parents[2]
        / "benchmarks/results/omol25-wb97mv-20261001/def2-tzvpd-ho.json"
    )
    return SimpleNamespace(
        _basis_metadata=[{"orbital": basis.identity}],
        _calculator=SimpleNamespace(
            _basis=basis, _energy_tolerance=1e-12, _density_tolerance=1e-10
        ),
    )


def test_no_seed_needs_no_gpu_and_preserves_absent_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.delenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", raising=False)
    result = seed.prepare_seed(target(), [], "none")
    assert result["selected"] is False
    assert result["complete_source_seconds"] is None
    assert result["source_fock_builds"] is None
    assert not result["source_reference_density_used"]
    assert "GENERATIVEQC_CUDA_KS_ACTIVE_AO" not in seed.os.environ


@pytest.mark.parametrize("provider", ["lda16", "pbe16"])
def test_source_construction_uses_exact_target_basis_and_restores_policy_on_failure(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    owner = target()
    calls = []
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_ACTIVE_AO", "1")
    monkeypatch.setitem(
        sys.modules,
        "cupy",
        SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(deviceSynchronize=lambda: None)
            )
        ),
    )

    def failed_constructor(**kwargs: object) -> None:
        assert seed.os.environ["GENERATIVEQC_CUDA_KS_ACTIVE_AO"] == "0"
        calls.append(kwargs)
        raise RuntimeError("independent source construction failed")

    monkeypatch.setattr(seed, "Calculator", failed_constructor)
    with pytest.raises(RuntimeError, match="source construction"):
        seed.prepare_seed(owner, [], provider)
    assert calls[0]["basis"] is owner._calculator._basis
    assert calls[0]["method"] == {"lda16": "lda-rks", "pbe16": "pbe-rks"}[provider]
    assert calls[0]["energy_tolerance"] == 1e-6
    assert owner._calculator._energy_tolerance == 1e-12
    assert seed.os.environ["GENERATIVEQC_CUDA_KS_ACTIVE_AO"] == "1"


def test_unknown_source_is_rejected_before_gpu_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(ValueError, match="unknown seed"):
        seed.prepare_seed(target(), [], "reference-density")
