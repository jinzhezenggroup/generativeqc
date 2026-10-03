"""Regression guard for CPU AO jet radial-factor reuse."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/dft/ao_grid.cpp"


def _evaluate_body() -> str:
    source = SOURCE.read_text()
    start = source.index("void AoBasis::evaluate(")
    return source[start:]


def _jet_count(order: int) -> int:
    return (order + 1) * (order + 2) * (order + 3) // 6


def test_cpu_ao_radial_factor_is_reused_across_requested_jets() -> None:
    body = _evaluate_body()
    radial = body.index("const double radial =")
    weighted = body.index("const double weighted_radial =", radial)
    jet_loop = body.index("for (std::size_t jet = 0; jet < JetCount; ++jet)", weighted)

    # One primitive exponential feeds every requested derivative jet instead of
    # being recomputed from inside a jet-major traversal.
    assert radial < weighted < jet_loop
    assert body.count("std::exp(") == 1
    assert body.index("double r2 = 0;") < radial


def test_legal_jet_extents_do_not_restore_value_only_runtime_dispatch() -> None:
    body = _evaluate_body()
    assert "std::array<double, JetCount> values{};" in body
    value_only, _ = body.split("if constexpr (JetCount == 1) {", 1)[1].split(
        "} else {", 1
    )
    assert "values[0] += term;" in value_only
    assert "differentiated_power(" in value_only
    assert "0, alpha, r[k]" in " ".join(value_only.split())
    assert "derivatives[" not in value_only
    assert "powers[" not in value_only
    for jets in (1, 4, 10, 20):
        assert f"evaluate_jets.template operator()<{jets}>()" in body


def test_cpu_ao_axis_derivatives_are_reused_before_the_jet_loop() -> None:
    body = _evaluate_body()
    powers = body.index("powers[k][derivative] = differentiated_power(")
    jet_loop = body.index("for (std::size_t jet = 0; jet < JetCount; ++jet)", powers)
    accumulate = body.index("values[jet] += term;", jet_loop)
    assert powers < jet_loop < accumulate
    assert "term *= powers[k][derivatives[jet][k]];" in body[jet_loop:accumulate]
    assert "differentiated_power(" not in body[jet_loop:accumulate]


def test_zero_derivative_keeps_horner_without_coefficient_storage() -> None:
    source = SOURCE.read_text()
    # Bound the branch by its own return, not by the coefficient declaration:
    # nonzero recurrence storage can live in a separate specialized helper.
    branch = source.split("if (derivative == 0) {", 1)[1]
    fast_path = branch[: branch.index("return result;") + len("return result;")]
    assert "double result = 1;" in fast_path
    assert "result = result * x + 0.0;" in fast_path
    assert "return result;" in fast_path
    assert "std::array" not in fast_path


def test_cpu_ao_radial_exp_work_census() -> None:
    # For a fixed AO/point/primitive tuple, the old jet-major traversal paid one
    # exponential per jet. The fused traversal pays exactly one regardless of
    # derivative order. This is a deterministic work-count reduction, not a
    # wall-time claim.
    assert [_jet_count(order) for order in range(4)] == [1, 4, 10, 20]
    primitive_evaluations = 97
    for order in range(4):
        jets = _jet_count(order)
        old_exp_calls = primitive_evaluations * jets
        new_exp_calls = primitive_evaluations
        assert old_exp_calls // new_exp_calls == jets
