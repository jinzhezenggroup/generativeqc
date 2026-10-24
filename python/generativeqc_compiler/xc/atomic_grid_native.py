"""Lower retained quadrature rules and the Treutler transform to native C++."""

from __future__ import annotations

import json
import math

from generativeqc_compiler.dft.treutler import angular_rule, tables
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter


def emit_atomic_grid_header() -> str:
    """Emit immutable element tables; native owners build molecule-local storage."""
    data = tables()
    graph = Graph()
    x, xi = graph.variable("x"), graph.variable("xi")
    radius = (
        -xi
        / math.log(2)
        * graph.power(1 + x, 0.6)
        * graph.stable_unary("log", (1 - x) / 2)
    )
    derivative = graph.differentiate(radius, x)
    emitter = ScalarCEmitter(graph, {"x": "x", "xi": "xi"})
    emitter.emit((radius, derivative))
    lines = [
        "// Generated from pinned PySCF 2.14.0 tables; see THIRD_PARTY_NOTICES.md.",
        "#pragma once",
        "#include <algorithm>",
        "#include <array>",
        "#include <cmath>",
        "#include <span>",
        "#include <stdexcept>",
        "namespace generativeqc::generated::atomic_grid {",
        "struct Direction { double x, y, z, weight; };",
    ]
    for name, values in (
        ("xi", data["treutler_xi"]),
        ("bragg", data["bragg_radii_bohr"]),
    ):
        lines.append(
            f"inline constexpr double {name}[] = {{"
            + ",".join(map(repr, values))
            + "};"
        )
    for name, values in (
        ("radial_counts", data["radial_counts"]),
        ("angular_orders", data["angular_orders"]),
    ):
        lines.append(
            f"inline constexpr unsigned {name}[10][7] = "
            + json.dumps(values).replace("[", "{").replace("]", "}")
            + ";"
        )
    counts = sorted(int(key) for key in data["orbits"])
    lines.append(
        "inline constexpr unsigned angular_counts[] = {"
        + ",".join(map(str, counts))
        + "};"
    )
    # Levels 0--9 use at most 1454 points. Larger retained rules remain available
    # to Python consumers but need not inflate every native translation unit.
    for count in counts:
        if count > 1454:
            break
        lines.append(f"inline constexpr Direction angular_{count}[] = {{")
        lines.extend(
            "{" + ",".join(map(repr, row)) + "}," for row in angular_rule(count)
        )
        lines.append("};")
    lines.extend(
        [
            "inline std::span<const Direction> angular(unsigned count) {",
            "switch (count) {",
            *(
                f"case {count}: return angular_{count};"
                for count in counts
                if count <= 1454
            ),
            'default: throw std::invalid_argument("unsupported native Lebedev order");',
            "}}",
            "inline unsigned maximum_angular(unsigned order) { switch (order) {",
            *(
                f"case {order}: return {count};"
                for order, count in data["lebedev_orders"].items()
            ),
            'default: throw std::invalid_argument("unsupported Lebedev degree"); }}',
            "inline std::array<double,2> radial_transform(double x, double xi) {",
            *emitter.lines,
            "return {"
            + emitter.reference(radius)
            + ","
            + emitter.reference(derivative)
            + "};}",
            "inline std::array<unsigned,2> counts(unsigned z, unsigned level) {",
            'if (z < 1 || z >= std::size(xi) || level > 9) throw std::invalid_argument("Treutler grid requires H-Lr and level 0-9");',
            "unsigned period = 0; for (unsigned edge : {2,10,18,36,54,86,118}) period += z > edge;",
            "return {radial_counts[level][period], maximum_angular(angular_orders[level][period])};}",
            "inline unsigned pruned_count(unsigned z, double radius, unsigned maximum) {",
            "if (maximum < 50) return maximum;",
            "constexpr double edges[3][4] = {{.25,.5,1,4.5},{.1667,.5,.9,3.5},{.1,.4,.8,2.5}};",
            "unsigned place = 0; for (double edge : edges[z <= 2 ? 0 : z <= 10 ? 1 : 2]) place += radius / bragg[z] > edge;",
            "const unsigned* start = angular_counts + 3; // 38-point rule",
            "if (maximum == 50) { constexpr unsigned picks[]{1,2,2,2,1}; return start[picks[place]]; }",
            "const auto index = std::find(start, std::end(angular_counts), maximum) - start;",
            "const size_t picks[]{1,3,size_t(index-1),size_t(index),size_t(index-1)}; return start[picks[place]];}",
            "} // namespace generativeqc::generated::atomic_grid",
        ]
    )
    return "\n".join(lines) + "\n"
