"""Independent point oracles and negative public-contract controls for slice A."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext

import pytest
from generativeqc.extensions import method, xc


@pytest.mark.parametrize(
    "a,b", [(0, 0), (1, 0), (0, 1), (0.2, 0.03), (1e-12, 1e-30), (2, 3)]
)
def test_independent_decimal_energy_and_analytic_canonical_ad(
    a: float, b: float
) -> None:
    expression = xc.expression("(rho_a**2 + rho_b**2)/3 - 0.25*rho_a*rho_b")
    with localcontext() as ctx:
        ctx.prec = 60
        da, db = Decimal(str(a)), Decimal(str(b))
        oracle = (da * da + db * db) / 3 - da * db / 4
        gradient = (2 * da / 3 - db / 4, 2 * db / 3 - da / 4)
    assert expression.evaluate(rho_a=a, rho_b=b) == pytest.approx(
        float(oracle), abs=1e-40, rel=2e-15
    )
    graph, root = expression.scalar_graph()
    values = {"rho_a": a, "rho_b": b}
    for i, feature in enumerate(("rho_a", "rho_b")):
        derivative = graph.differentiate(root, graph.variable(feature))
        assert graph.evaluate(derivative, values) == pytest.approx(
            float(gradient[i]), abs=1e-40, rel=2e-15
        )
        for other in ("rho_a", "rho_b"):
            second = graph.differentiate(derivative, graph.variable(other))
            assert graph.evaluate(second, values) == pytest.approx(
                2 / 3 if feature == other else -0.25
            )


def test_general_polynomial_finite_difference_and_spin_convention() -> None:
    expression = xc.expression("-rho_a**4/7 + rho_b**3/5 + rho_a*rho_b + rho_a")
    a, b, h = 0.4, 0.1, 1e-5
    graph, root = expression.scalar_graph()
    for feature, analytic in (
        ("rho_a", -4 * a**3 / 7 + b + 1),
        ("rho_b", 3 * b * b / 5 + a),
    ):
        values = {"rho_a": a, "rho_b": b}
        derivative = graph.evaluate(
            graph.differentiate(root, graph.variable(feature)), values
        )
        assert derivative == pytest.approx(analytic)
        plus, minus = dict(values), dict(values)
        plus[feature] += h
        minus[feature] -= h
        assert derivative == pytest.approx(
            (expression.evaluate(**plus) - expression.evaluate(**minus)) / (2 * h),
            abs=1e-9,
        )
    # Per-volume output is not silently divided by total density.
    assert xc.expression("rho_a + rho_b").evaluate(rho_a=2, rho_b=3) == 5
    assert xc.expression("rho_a").evaluate(rho_a=2, rho_b=3) == 2


def test_identity_replay_and_detached_exports() -> None:
    original = xc.expression("rho_b + rho_a + 1/2", name="PBE")
    alias = xc.expression("0.5 + (rho_a + rho_b)", name="anything")
    assert original.identity == alias.identity
    assert xc.expression("rho_a + rho_b + 1/3").identity != original.identity
    for restored in (
        copy.copy(original),
        copy.deepcopy(original),
        xc.UserExpression.loads(original.dumps()),
    ):
        assert restored.identity == original.identity
        assert restored.source == original.source
    with pytest.raises(FrozenInstanceError):
        original.source = "0"
    report = xc.inspect(original)
    report["spec"]["canonical"]["nodes"].clear()
    report["capability"]["cpu"]["scf"] = "PASS"
    graph, _ = original.scalar_graph()
    graph.nodes.clear()
    assert original.evaluate(rho_a=1, rho_b=2) == 3.5
    payload = json.loads(original.dumps())
    payload["source"] = "rho_a"
    with pytest.raises(ValueError, match="mismatch"):
        xc.UserExpression.loads(json.dumps(payload))
    payload = json.loads(original.dumps())
    payload["canonical"]["schema"] = "future"
    with pytest.raises(ValueError, match="mismatch"):
        xc.UserExpression.loads(json.dumps(payload))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from generativeqc.extensions import xc; print(xc.expression('rho_b+rho_a+1/2').identity)",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env=os.environ.copy(),
    )
    assert result.stdout.strip() == original.identity


@pytest.mark.parametrize(
    "source",
    [
        "sigma_aa",
        "sigma_ab",
        "sigma_bb",
        "tau_a",
        "tau_b",
        "rho",
        "unknown",
        "rho_a if rho_b else 0",
        "rho_a > rho_b",
        "rho_a and rho_b",
        "[rho_a]",
        "sum(rho_a for i in range(10))",
        "lambda: rho_a",
        "rho_a.real",
        "rho_a[0]",
        "__import__('os')",
        "(x := rho_a)",
        "rho_a / rho_b",
        "0/0",
        "rho_a**-1",
        "rho_a**0.5",
        "rho_a**5",
        "(rho_a**4)**4",
        "1e309",
        "True",
        "None",
        "0 * unknown",
        "rho_a; rho_b",
        "rho_a % 2",
        "rho_a // 2",
        "'rho_a'",
    ],
)
def test_unsupported_declarations_fail_before_execution(source: str) -> None:
    with pytest.raises(ValueError):
        xc.expression(source)


def test_budgets_callbacks_and_foreign_graph_rejected() -> None:
    called = []

    def callback() -> int:
        called.append(True)
        return 1

    for value in (callback, {"rho_a": 1}, xc.expression("rho_a").scalar_graph()[1]):
        with pytest.raises(TypeError):
            xc.expression(value)
    assert not called
    for source in (
        "1" * 4097,
        "+".join(["rho_a"] * 100),
        "-" * 40 + "rho_a",
        str(2**257),
    ):
        with pytest.raises(ValueError):
            xc.expression(source)
    assert xc.expression("(rho_a**4)**2").evaluate(rho_a=2, rho_b=0) == 256


@pytest.mark.parametrize(
    "value", [-1, float("nan"), float("inf"), -float("inf"), 10**1000]
)
def test_point_domain_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        xc.expression("rho_a").evaluate(rho_a=value, rho_b=0)


def test_no_admission_or_name_based_qualification() -> None:
    expression = xc.expression("rho_a**4", name="PBE")
    with pytest.raises(ValueError):
        expression.evaluate(rho_a=1e308, rho_b=0)
    with pytest.raises(TypeError):
        expression.evaluate(rho_a=True, rho_b=0)
    with pytest.raises(TypeError, match="FunctionalSpec"):
        method.compose("custom", xc=expression)
    report = xc.inspect(expression)
    assert report["kind"] == "xc-expression"
    assert report["capability"]["pointwise_validated"] == "UNQUALIFIED"
    for backend in ("cpu", "cuda"):
        assert set(report["capability"][backend].values()) == {"UNQUALIFIED"}
    assert xc.inspect(xc.named("PBE"))["kind"] == "xc"


def test_canonical_coefficient_budget_and_compiler_only_import() -> None:
    with pytest.raises(ValueError, match="coefficient"):
        xc.expression(f"({2**200} * rho_a) * {2**200}")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from generativeqc_compiler.xc.user_expression import UserExpression; UserExpression('rho_a').scalar_graph(); assert not any(n == 'generativeqc' or n.startswith(('generativeqc.', 'pyscf', 'torch', 'cupy')) for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.returncode == 0
