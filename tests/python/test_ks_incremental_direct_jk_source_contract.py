from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _source(path: str) -> str:
    return (ROOT / path).read_text()


def test_cuda_ks_incremental_direct_jk_is_capability_based() -> None:
    method = _source("src/methods/dft_method.cpp")
    ks = _source("src/dft/cuda_ks.cpp")

    assert "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK" in method
    assert (
        "direct_jk_incremental_exact_eligible(*options.resolved_fock_build)" in method
    )
    assert (
        "CUDA KS incremental Direct-J/K requires one strict-FP64 exact Direct provider"
        in ks
    )

    # The legacy experiment remains reproducible, but the generic selector is
    # not restricted to one semilocal family or one spin mode.
    assert "GENERATIVEQC_PBE0_INCREMENTAL_DIRECT_JK" in method
    generic = method.index("const bool generic_incremental_direct_jk")
    legacy = method.index("if (legacy_pbe0_incremental_direct_jk)", generic)
    generic_block = method[generic:legacy]
    assert "SemilocalFamily::Pbe" not in generic_block
    assert "FockSpin::Restricted" not in generic_block


def test_cuda_ks_incremental_direct_jk_only_changes_linear_fock_inputs() -> None:
    ks = _source("src/dft/cuda_ks.cpp")

    assert "const double* jk_density = prepare_incremental_jk_density();" in ks
    assert "provider, jk_density, spins == 2 ? jk_density + matrix : nullptr" in ks
    assert "provider, *range_correction, jk_density" in ks
    assert "finalize_incremental_jk_components();" in ks

    # XC/nonlocal correlation remain nonlinear full-density functionals and
    # must never consume the anchor-relative delta.
    assert "xc->enqueue(density, elements, next_generation, phase);" in ks
    assert "xc->enqueue_density_features(density, elements, next_generation" in ks
    assert "xc->enqueue(incremental_delta_density" not in ks
    assert "xc->enqueue_density_features(incremental_delta_density" not in ks


def test_cuda_ks_incremental_direct_jk_preserves_full_finalization() -> None:
    ks = _source("src/dft/cuda_ks.cpp")

    assert (
        "!precision_schedule.any_lower_precision() && !incremental_direct_jk && spins == 1"
        in ks
    )
    assert (
        "const bool strict_final_closure =\n"
        "        incremental_direct_jk || spins == 2 || !provider.system().ecp_terms.empty();"
        in ks
    )
    assert "++output.incremental_direct_jk.post_scf_full_builds;" in ks
    assert "incremental_anchor_range_exchange" in ks
    assert "output.incremental_direct_jk.anchor_updates" in ks


def test_cuda_ks_incremental_storage_is_explicitly_budgeted() -> None:
    header = _source("src/dft/cuda_ks.hpp")
    ks = _source("src/dft/cuda_ks.cpp")

    assert "bool incremental_direct_jk = false" in header
    assert "bool incremental_direct_jk, void* storage" in ks
    assert "reserve(incremental_anchor_density, elements);" in ks
    assert "reserve(incremental_delta_density, elements);" in ks
    assert "reserve(incremental_anchor_j, matrix);" in ks
