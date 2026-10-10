"""Automatic CUDA DF routing must remain on the bounded source-backed path."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_automatic_df_uses_bounded_source_backed_storage_selection() -> None:
    source = (ROOT / "src/scf/rhf.cpp").read_text()
    assert "automatic_dense_resident_df_owner" not in source
    assert "preferred_automatic_resident_df_value_peak" not in source

    start = source.index(
        "[[maybe_unused]] DensityFittingScfData prepare_density_fitting_data("
    )
    end = source.index("Matrix build_density_fitting_rhf_fock(", start)
    preparation = source[start:end]
    assert "if (data.resolved_budget.value_bytes != 0U ||" in preparation
    assert "assemble_density_fitting_metadata" in preparation

    start = source.index("CudaDensityFittingPlanPtr make_cuda_density_fitting_plan(")
    end = source.index("ScfResult run_cuda_independent_fock_strategy(", start)
    planner = source[start:end]
    assert "const auto planning_budget = data.resolved_budget.value_bytes;" in planner
    assert "plan_requested_density_fitting_tiles(" in source
    assert "requested_df_pair_storage_request()" in source
    assert (
        "set_cuda_density_fitting_scf_value_budget(owned_plan.get(), planning_budget)"
        in planner
    )


def test_auto_budget_has_no_provider_specific_peak_override() -> None:
    header = (ROOT / "src/scf/df_preparation_budget.hpp").read_text()
    assert "preferred_value_peak_bytes" not in header
    assert (
        "const auto workload_target = "
        "std::max(df_budget_bytes(value_demand + response_demand), min_auto);" in header
    )


def test_prepared_ks_df_uses_method_owned_capacity_crossover() -> None:
    """The common owner must route validated occupations through actual planning."""
    source = (ROOT / "src/scf/fock_prepared.cpp").read_text()
    begin = source.index("      const auto plan_values =")
    end = source.index("      cuda_df.reset(raw_plan);", begin)
    planning = source[begin:end]
    assert "plan_requested_density_fitting_tiles(" in planning
    assert "reserved_rank != 0" in planning
    assert "resolve_method_owned_df_resident_budget(" in planning
    assert "data.value_storage = tiles.value_storage.pairs;" in planning
    assert "diagnostic.variant.df_pair_storage = data.value_storage;" in planning
    assert (
        "reserved_rank != static_cast<std::size_t>(system.electron_count / 2)" in source
    )
    assert (
        "variant.df_pair_storage_request == DfPairStorageRequest::Automatic" in source
    )


def test_retained_df_forces_defer_coordinate_matrices_until_host_fallback() -> None:
    """Energy preparation must not eagerly export a future force fallback."""
    prepared = (ROOT / "src/scf/fock_prepared.cpp").read_text()
    method = (ROOT / "src/methods/dft_method.cpp").read_text()
    assert "const bool export_derivatives = derivatives &&" in " ".join(
        prepared.split()
    )
    assert "strategy.spec.derivative_order == 0" in prepared
    assert 'Scope export_trace("deferred_one_electron_derivatives")' in prepared
    begin = method.index("    if (!resident_one_electron) {")
    end = method.index("    const std::vector<double> empty;", begin)
    fallback = method[begin:end]
    assert fallback.index("fock_.ensure_one_electron_derivatives();") < fallback.index(
        "one.hcore_derivative.data()"
    )
