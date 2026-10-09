"""Keep energy-only dense ERIs scalar without changing the integral producer."""

from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2] / "src/integrals/s_integrals.cpp"
).read_text()


def test_generated_value_producer_writes_scalar_storage() -> None:
    quartet = SOURCE.split("std::size_t build_value_eri_shell_quartet(", 1)[1].split(
        "std::vector<std::size_t> cartesian_shell_offsets(", 1
    )[0]
    assert "ValueEriComponents& components" in quartet
    assert "Jet" not in quartet
    assert "generated_eri_cpu::make_geometry" in quartet
    assert "generated_eri_cpu::prepare_coulomb" in quartet
    assert "generated_eri_cpu::prepared_primitive" in quartet
    assert "return count;" in quartet
    consumer = SOURCE.split("void build_value_eri_shell_quartets(", 1)[1].split(
        "using ValueEriCartesianBlock", 1
    )[0]
    assert "std::vector<double>& eri" in consumer
    assert "store_eri_symmetry" in consumer


def test_dense_jet_storage_and_unpack_are_derivative_only() -> None:
    build = SOURCE.split("IntegralData build_integrals(", 1)[1].split(
        "std::vector<double> build_range_eri(", 1
    )[0]
    assert "include_derivatives ? sizeof(Jet) : sizeof(double)" in build
    assert (
        "if (include_derivatives)\n      eri.assign(n4, Jet(0.0, out.ncoord));" in build
    )
    assert "out.eri.assign(n4, 0.0);" in build
    assert "build_value_eri_shell_quartets(system, aos, out.eri);" in build
    assert (
        "if (include_eri && include_derivatives) "
        "unpack_jets(eri, out.eri, out.eri_derivative, out.ncoord);"
    ) in " ".join(build.split())
    # Higher-angular value execution retains its independent recurrence but
    # only the resulting scalar is retained in the rank-four output.
    assert "production_eri_cartesian(" in build
    assert "store_eri_symmetry(out.eri, n, {i, j, k, l}, value.value);" in build


def test_full_and_range_values_share_the_existing_symmetry_scatter() -> None:
    assert (
        "template <typename Value>\nvoid store_eri_symmetry(std::vector<Value>&"
        in SOURCE
    )
    range_values = SOURCE.split("std::vector<double> build_range_eri(", 1)[1].split(
        "std::array<double, 12> contract_weighted_eri_shell_derivative(", 1
    )[0]
    assert (
        "store_eri_symmetry(eri, cartesian_nbf, {i, j, k, l}, value);" in range_values
    )


def test_mixed_basis_keeps_low_l_geometry_reuse_and_disjoint_fallback() -> None:
    """Adding f/g must not reroute existing s/p/d primitive-component work."""
    shell_pass = SOURCE.split("void for_each_value_eri_shell_quartet(", 1)[1].split(
        "void build_value_eri_shell_quartets(", 1
    )[0]
    assert "system.shells[shell].angular_momentum > 2" in shell_pass
    build = SOURCE.split("IntegralData build_integrals(", 1)[1].split(
        "std::vector<double> build_range_eri(", 1
    )[0]
    assert "else if (include_eri && !include_derivatives)" in build
    assert "if (include_eri && !shared_value_geometry)" in build
    for name in ("ao_i", "ao_j", "ao_k", "ao_l"):
        assert f"{name}.eri_component_index < 10" in build
    assert "if (!include_derivatives && ao_i.eri_component_index < 10" in build
    assert "build_f_value_eri_shell_quartets(system, aos, out.eri)" in build
    assert "if (!include_derivatives && ao_i.shell->angular_momentum <= 3" in build


def test_f_values_reuse_the_retained_recurrence_with_bounded_components() -> None:
    """Keep preparation outside components and keep g/derivatives independent."""
    shared = SOURCE.split("void build_f_value_eri_shell_quartets(", 1)[1].split(
        "void build_value_eri_shell_quartets(", 1
    )[0]
    assert "if (maximum_shell != 3) return" in shared
    assert "prepare_eri_cartesian(" in shared
    assert shared.index("prepare_eri_cartesian(") < shared.index(
        "for (auto& component : components)"
    )
    assert "contract_prepared_eri_value(" in shared
    assert "maxima[slot].fill(shells[slot]->angular_momentum)" in shared
    assert "store_eri_symmetry" in shared
    contraction = SOURCE.split("double contract_prepared_eri_value(", 1)[1].split(
        "struct PreparedCartesianEri", 1
    )[0]
    assert "Jet" not in contraction
    assert "double value = 0.0" in contraction


def test_only_integral_producers_assert_eightfold_projection() -> None:
    """Keep arbitrary tensors on the ordered adapter, including nonsymmetric data."""
    assert "bool eightfold_symmetric = false" in SOURCE
    adapter = SOURCE.split("IntegralData transform_integrals(", 1)[1].split(
        "IntegralData build_integrals(", 1
    )[0]
    assert "target_aos, true" not in adapter
    producer = SOURCE.split("IntegralData build_integrals(", 1)[1]
    assert "transform_eri(out.eri.data(), out.nbf, target_aos, true)" in producer
    assert "out.nbf, target_aos, true)" in producer
    assert "spherical_expansions(system), true)" in producer
