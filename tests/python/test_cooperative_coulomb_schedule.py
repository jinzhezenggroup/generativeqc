"""Qualify the cooperative lowering against the authoritative Coulomb DAG."""

import pytest
from generativeqc_compiler.integral import (
    direct_cooperative_gradient_cuda as cooperative,
)
from generativeqc_compiler.integral.direct_cooperative_gradient_cuda import (
    build_cooperative_coulomb_schedule,
)
from generativeqc_compiler.integral.expr import Graph, Node
from generativeqc_compiler.integral.shell_class import (
    CoulombDerivativeAlgebra,
    build_coulomb_derivative_algebra,
)


def test_cooperative_constants_require_a_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a malformed constant instead of hiding the optional payload type."""
    graph = Graph()
    root = graph.constant(1)
    graph.nodes[root.identifier] = Node("constant")
    algebra = CoulombDerivativeAlgebra(graph, (((0, 0, 0), root),))
    monkeypatch.setattr(
        cooperative, "build_coulomb_derivative_algebra", lambda _: algebra
    )
    with pytest.raises(ValueError, match="constant is missing its value"):
        build_cooperative_coulomb_schedule.__wrapped__()


def test_cooperative_levels_publish_every_dependency() -> None:
    """A lane may only read nodes retired by an earlier CTA publication barrier."""
    schedule = build_cooperative_coulomb_schedule()
    assert schedule.levels[0] == 0
    assert schedule.levels[-1] == len(schedule.instructions)
    assert len(schedule.roots) == 165
    assert len(schedule.variables) == 13
    for begin, end in zip(schedule.levels, schedule.levels[1:]):
        assert begin < end
        for instruction in schedule.instructions[begin:end]:
            opcode = instruction >> 32
            first, second = instruction & 65535, (instruction >> 16) & 65535
            if opcode >= 2:
                assert first < begin and second < begin
            else:
                assert opcode in (0, 1)
    assert build_cooperative_coulomb_schedule() == schedule


@pytest.mark.parametrize("rho", (0.0, 0.37, 1.4))
def test_cooperative_spatial_outputs_match_original_ir(rho: float) -> None:
    """Independent DAG evaluation checks all roots and packed component mappings."""
    algebra = build_coulomb_derivative_algebra(8)
    schedule = build_cooperative_coulomb_schedule()
    leaves = {
        "rho": rho,
        "difference_x": 0.3,
        "difference_y": -0.8,
        "difference_z": 0.6,
    }
    leaves.update({f"boys_{n}": 0.29 / (n + 1) + (-0.07) ** n for n in range(9)})
    values: list[float] = []
    for instruction in schedule.instructions:
        opcode = instruction >> 32
        first, second = instruction & 65535, (instruction >> 16) & 65535
        if opcode == 0:
            value = schedule.constants[first]
        elif opcode == 1:
            value = leaves[schedule.variables[first]]
        elif opcode == 2:
            value = values[first] * values[second]
        else:
            value = values[first] + values[second]
        values.append(value)
    outputs = {
        (mapping & 15, (mapping >> 4) & 15, (mapping >> 8) & 15): values[mapping >> 12]
        for mapping in schedule.roots
    }
    assert len(outputs) == len(algebra.roots)
    for degrees, root in algebra.roots:
        assert outputs[degrees] == pytest.approx(
            algebra.graph.evaluate(root, leaves), abs=2e-12, rel=2e-12
        )
