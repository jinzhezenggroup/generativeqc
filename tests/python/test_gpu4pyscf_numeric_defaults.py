"""The public/native GPU4PySCF-compatible *numerical* defaults, not benchmark flags."""

import inspect
from pathlib import Path
from types import SimpleNamespace

from generativeqc import Calculator, InitialGuessSpec
from generativeqc_compiler.dft.nonlocal_policy import (
    MOLECULAR_VV10_DENSITY_POLICY,
    MOLECULAR_VV10_DENSITY_THRESHOLD,
)

ROOT = Path(__file__).resolve().parents[2]


def test_public_default_controls():
    p = inspect.signature(Calculator).parameters
    assert p["max_iterations"].default == 50
    assert p["energy_tolerance"].default == 1e-9
    assert p["density_tolerance"].default == 1e-8
    assert p["diis_history"].default == 8
    assert p["initial_guess"].default == "auto"
    assert "else 1e-13" in (ROOT / "python/generativeqc/calculator.py").read_text()


def test_native_defaults_match_public_values():
    scf = (ROOT / "src/scf/types.hpp").read_text()
    common = (ROOT / "src/methods/scf_common_options.hpp").read_text()
    fock = (ROOT / "src/scf/fock_build.hpp").read_text()
    for needle in (
        "max_iterations{50}",
        "energy_tolerance{1.0e-9}",
        "screening_tolerance{1.0e-13}",
    ):
        assert needle in scf
    for needle in (
        "descriptor.max_iterations == 0 ? 50",
        "descriptor.energy_tolerance : 1.0e-9",
        "descriptor.screening_tolerance : 1.0e-13",
    ):
        assert needle in common
    assert "screening_tolerance = 1.0e-13" in fock


def test_cuda_ao_and_vv10_cutoffs():
    src = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    assert "ao_selection_cutoff = 1.0e-10" in src
    assert "select_local_ao(ao_selection_cutoff," in src
    assert "prepared_ao_work.cutoff = select_ao ? ao_selection_cutoff : 0" in src
    assert float(MOLECULAR_VV10_DENSITY_THRESHOLD) == 1e-10
    assert MOLECULAR_VV10_DENSITY_POLICY == "vv10-molecular-rho-ge-1e-10-v2"


def test_auto_guess_prefers_minao(monkeypatch):
    import generativeqc.initial_guess as guess

    monkeypatch.setattr(guess, "_automatic_minao_model_eligible", lambda *a, **kw: True)

    def fail(*a, **kw):
        raise AssertionError("must select MINAO before querying Huckel")

    monkeypatch.setattr(guess, "supports_automatic_huckel", fail)
    assert guess.automatic_initial_guess_candidate(
        SimpleNamespace(_library=object()), restricted=True, has_ecp=False
    ) == InitialGuessSpec("minao")


def test_auto_guess_fallback_remains_guarded(monkeypatch):
    import generativeqc.initial_guess as guess

    monkeypatch.setattr(
        guess, "_automatic_minao_model_eligible", lambda *a, **kw: False
    )
    monkeypatch.setattr(guess, "supports_automatic_huckel", lambda *a, **kw: True)
    monkeypatch.setattr(
        guess, "_native_automatic_minao_eligible", lambda *a, **kw: True
    )
    obj = SimpleNamespace(
        _library=object(),
        _device_name="cuda",
        _method=guess._native.METHOD_RHF,
        _density_fitting_mode=guess._native.DENSITY_FITTING_NONE,
    )
    assert guess.automatic_initial_guess_candidate(
        obj, restricted=True, has_ecp=False
    ) == InitialGuessSpec("huckel")
