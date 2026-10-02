"""Source-contract tests for the PBE0-DF final occupied-projection force reuse."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_dft_force_consumes_ks_projection_before_generic_response() -> None:
    method = (ROOT / "src/methods/dft_method.cpp").read_text()
    prepared = (ROOT / "src/scf/fock_prepared.cpp").read_text()
    lower = (ROOT / "src/scf/cuda/df_force_response.cpp").read_text()

    assert "resident_final_fitted_projection(expected, lease" in method
    assert "energy_derivative_components_with_fitted_projection" in method
    # The final U aliases provider scratch: exchange must consume it before J
    # can run an ordinary response that revokes/reuses the same allocation.
    body = prepared[
        prepared.index(
            "energy_derivative_components_with_fitted_projection"
        ) : prepared.index("PreparedFockPlan::retained_energy_derivative")
    ]
    assert body.index("provider.derivative(exchange") < body.index(
        "provider.derivative(coulomb"
    )
    assert "borrowed.projection != plan->auxiliary_tile_values" in lower
    assert 'trace_counter("response_reused_final_fitted_projection", 1)' in lower


def test_dft_projection_reuse_keeps_explicit_fallback_controls() -> None:
    lower = (ROOT / "src/scf/cuda/df_force_response.cpp").read_text()
    assert 'projection != "off"' in lower
    assert 'occupied_source != "raw"' in lower
    assert 'space != "dense"' in lower
    assert 'storage != "jk-scratch"' in lower


def test_optional_projection_admission_preserves_bounded_fallback() -> None:
    lower = (ROOT / "src/scf/cuda/df_force_response.cpp").read_text()
    selection = lower[
        lower.index("const bool select_borrowed_projection") : lower.index(
            "bool borrowed_fitted_occupied"
        )
    ]
    assert (
        "plan->value_storage.pairs == DfPairStorage::SymmetricLowerSingle" in selection
    )
    assert lower.index("const std::string_view projection =") < lower.index(
        "const bool select_borrowed_projection"
    )
    retry = lower[
        lower.index(
            "if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY && borrowed_fitted_occupied"
        ) : lower.index(
            "if (status == GENERATIVEQC_STATUS_SUCCESS && borrow && matching_source"
        )
    ]
    assert 'space == "auto" && occupied_source == "auto"' in retry
    assert "final_state, nullptr" in retry
