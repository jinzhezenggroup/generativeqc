"""Keep energy-only dense ERIs scalar without changing the integral producer."""

from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2] / "src/integrals/s_integrals.cpp"
).read_text()


def test_generated_value_producer_writes_scalar_storage() -> None:
    quartet = SOURCE.split("void build_value_eri_shell_quartet(", 1)[1].split(
        "void build_value_eri_shell_quartets(", 1
    )[0]
    assert "std::vector<double>& eri" in quartet
    assert "Jet" not in quartet
    assert "generated_eri_cpu::make_geometry" in quartet
    assert "generated_eri_cpu::prepare_coulomb" in quartet
    assert "generated_eri_cpu::prepared_primitive" in quartet
    assert "store_eri_symmetry" in quartet


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
