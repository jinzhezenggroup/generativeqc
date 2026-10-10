"""Independent exact-rational checks of optional projection-admission metadata."""

from __future__ import annotations

import random
import shutil
import struct
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner, write_df_cpu_headers
from generativeqc_compiler.cc.df_spectator_pairs import build_ladder_pair_majorant

from tools import generate_df_ccsd_spectator_pairs as generator

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bound_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a native C++ compiler")
    directory = tmp_path_factory.mktemp("df-pair-bound")
    write_df_cpu_headers(directory)
    (directory / "generated_df_ccsd_spectator_pairs_cpu.hpp").write_text(
        generator.cpu_header()
    )
    executable = directory / "probe"
    compile_owner(
        compiler,
        directory,
        [ROOT / "tests/native/df_cc_pair_bound_probe.cpp"],
        executable,
    )
    return executable


def _run(probe: Path, commands: list[str]) -> list[list[int]]:
    result = subprocess.run(
        [str(probe)],
        input="\n".join(commands) + "\n",
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return [list(map(int, line.split())) for line in result.stdout.splitlines()]


def _power(exponent: int) -> Fraction:
    return Fraction(2) ** exponent


def _bits(value: float) -> int:
    return struct.unpack("Q", struct.pack("d", value))[0]


def _real(bits: int) -> Fraction:
    return Fraction.from_float(struct.unpack("d", struct.pack("Q", bits))[0])


def _upper(result: list[int]) -> Fraction:
    state, exponent = result
    assert state != 2, "finite admitted inputs must not refuse"
    return _power(exponent) if state else Fraction(0)


def test_ieee_upper_includes_subnormals_and_refuses_nonfinite(
    bound_probe: Path,
) -> None:
    bits = [0, 1, 2, (1 << 52) - 1, 1 << 52, 0x7FEFFFFFFFFFFFFF]
    bits += [
        encoded << 52 | fraction
        for encoded in (1, 512, 1023, 2046)
        for fraction in (0, 1, (1 << 51), (1 << 52) - 1)
    ]
    bits += [0x7FF0000000000000, 0x7FF0000000000001]
    results = _run(bound_probe, [f"magnitude {value}" for value in bits])
    for value, result in zip(bits, results, strict=True):
        if value >= 0x7FF0000000000000:
            assert result[0] == 2
        else:
            assert abs(_real(value)) <= _upper(result)
            assert not value or _upper(result) < 2 * abs(_real(value))


def test_factorization_range_is_a_bounded_refusal_not_a_physical_fault(
    bound_probe: Path,
) -> None:
    limit = 1151 << 52
    cases = [
        (128, 128, 65536, 65536, limit, limit, 0, 1),
        (-1074, -1074, 2, 3, 1, 1, 0, 1),
        (129, 0, 2, 3, 0, 0, 0, 0),
        (0, 129, 2, 3, 0, 0, 0, 0),
        (8193, 0, 2, 3, 0, 0, 0, 0),
        (0, 0, 65537, 3, 0, 0, 0, 0),
        (0, 0, 2, 65537, 0, 0, 0, 0),
        (0, 0, 0, 3, 0, 0, 0, 0),
        (0, 0, 2, 3, limit + 1, 0, 0, 0),
        (0, 0, 2, 3, 0, limit + 1, 0, 0),
        (0, 0, 2, 3, 0, 0x7FF0000000000000, 0, 0),
        (0, 0, 2, 3, 0, 0, 1, 0),
    ]
    results = _run(
        bound_probe,
        ["factorization-range " + " ".join(map(str, case[:-1])) for case in cases],
    )
    assert results == [[case[-1]] for case in cases]


@pytest.mark.parametrize("axes", [0, 1, 2, 3])
def test_grouped_norm_bounds_all_exact_rows(bound_probe: Path, axes: int) -> None:
    rng = random.Random(9328 + axes)
    for _ in range(10):
        rows, columns = 7, 5
        values = [
            rng.choice(
                [
                    0,
                    1,
                    17,
                    0x0010000000000000,
                    0x3F10000000000000,
                    0x400123456789ABCD,
                    0x7FEFFFFFFFFFFFFF,
                ]
            )
            | (rng.randrange(2) << 63)
            for _ in range(rows * columns)
        ]
        results = _run(
            bound_probe, [f"norm {rows} {columns} {axes} " + " ".join(map(str, values))]
        )
        exact = [abs(_real(value)) for value in values]
        groups = (
            [
                sum(exact[row * columns + column] for row in range(rows))
                for column in range(columns)
            ]
            if axes == 1
            else [
                sum(exact[row * columns : (row + 1) * columns]) for row in range(rows)
            ]
            if axes == 2
            else [sum(exact)]
            if axes == 3
            else exact
        )
        assert max(groups) <= _upper(results[0])
        assert results[1] == [len(values) * (2 if axes else 1)]


def test_refusal_and_checked_metadata_overflow(bound_probe: Path) -> None:
    assert all(result[0] == 2 for result in _run(bound_probe, ["unsafe"]))
    for axes in (1, 2, 3, 4):
        result = _run(bound_probe, [f"norm 1 1 {axes} 9218868437227405312"])
        assert result[0][0] == 2


def test_margin_does_not_underflow_tolerance_division(bound_probe: Path) -> None:
    for tolerance in (1, 19, 0x0010000000000000, _bits(1e-8)):
        exact = _real(tolerance) / 8
        for exponent in range(-1078, -1068) if tolerance < 32 else (-100, -40, -20, 0):
            admitted = _run(bound_probe, [f"margin {exponent} {tolerance}"])[0][0]
            assert not admitted or _power(exponent) <= exact
    assert _run(bound_probe, ["margin -1077 1", "margin -1076 1"]) == [[1], [0]]
    for invalid in (0, _bits(-1.0), 0x7FF0000000000000, 0x7FF8000000000000):
        assert _run(bound_probe, [f"margin -2000 {invalid}"]) == [[0]]


def test_compiler_coefficient_envelopes_exact_positive_dag(bound_probe: Path) -> None:
    majorant = build_ladder_pair_majorant(generator.hoisted.programs()["auxiliary"])
    rng = random.Random(81912)
    for _ in range(20):
        exponents = [rng.randrange(-100, 100) for _ in majorant.factor_norms]
        feeds = {
            name: _power(exponent)
            for (name, _, _), exponent in zip(
                majorant.factor_norms, exponents, strict=True
            )
        }
        values = {}
        for node in majorant.coefficients.dependency_order:
            if node.op == "input":
                value = feeds[node.attrs["name"]]
            elif node.op == "constant":
                value = Fraction(*node.attrs["values"][0])
            elif node.op == "add":
                value = sum(
                    Fraction(*coefficient) * values[child]
                    for child, coefficient in zip(
                        node.inputs, node.attrs["coefficients"], strict=True
                    )
                )
            else:
                assert node.op == "einsum"
                value = Fraction(*node.attrs["coefficient"])
                for child in node.inputs:
                    value *= values[child]
            values[node] = value
        result = _run(bound_probe, ["coefficient " + " ".join(map(str, exponents))])
        for position, name in enumerate(("coefficient_0", "coefficient_1")):
            assert values[majorant.coefficients.outputs[name]] <= _upper(
                result[position]
            )
        assert result[2] == [1, 0]
    assert generator._ladder_residual_gain() == 1


def test_whole_auxiliary_sum_is_included(bound_probe: Path) -> None:
    for auxiliary in (1, 2, 3, 17, 1025):
        result = _run(bound_probe, [f"aggregate 10 20 -1074 -12 {auxiliary}"])[0]
        exact = auxiliary * _power(-1074) * (_power(10) + _power(20) * _power(-12))
        assert exact <= _upper(result)


@pytest.mark.parametrize(
    "coefficient",
    [Fraction(1, 3), Fraction(7, 10), Fraction(19, 2), Fraction(1, 2**1078)],
)
def test_rational_constants_do_not_round_through_double(coefficient: Fraction) -> None:
    emitted = generator._power_upper(coefficient)
    exponent = int(emitted.removeprefix("pair_bound::Power::of(").removesuffix(")"))
    assert coefficient <= _power(exponent) < 2 * coefficient
