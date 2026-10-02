"""Execute compiler-emitted bounded Becke strips with real host-thread barriers.

These are correctness/work-count probes, not GPU performance qualification.
"""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess

import numpy as np
import pytest
import test_becke_cooperative as retained
from generativeqc_compiler.xc.grid_native import emit_grid_adjoint, emit_grid_partials


@pytest.fixture(scope="module", params=[(1, 4), (3, 1), (3, 4), (3, 7), (5, 4)])
def helper(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ct.CDLL:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    iterations, rows = request.param
    directory = tmp_path_factory.mktemp("becke-tiled")
    harness = retained.HARNESS.replace(
        "std::vector<PointPair> states(geometry.size()+2);",
        f"const size_t rows=std::min(size_t({rows}),na-1);\n"
        "  std::vector<PointPair> states(rows*(2*na-rows-1)/2+2);",
    ).replace("contract_point_cooperative(", "contract_point_tiled_cooperative(")
    harness = harness.replace(
        "states.data()+1,Team{", f"states.data()+1,size_t({rows}),Team{{"
    )
    harness = (
        harness.replace(
            "struct Team {", "static std::atomic<bool> team_valid;\nstruct Team {"
        )
        .replace(
            "  size_t rank() const",
            "  bool all(bool value) const { if(!value) team_valid=false; sync(); bool result=team_valid; sync(); return result; }\n"
            "  size_t rank() const",
        )
        .replace(
            "pairs_evaluated=norms_evaluated=logs_evaluated=0;",
            "pairs_evaluated=norms_evaluated=logs_evaluated=0; team_valid=true;",
        )
    )
    source = directory / "probe.cpp"
    source.write_text(emit_grid_adjoint() + emit_grid_partials(iterations) + harness)
    library = directory / "probe.so"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-pthread",
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = ct.CDLL(str(library))
    pointer = ct.POINTER(ct.c_double)
    result.run.argtypes = [
        pointer,
        ct.c_size_t,
        pointer,
        ct.c_size_t,
        ct.POINTER(ct.c_int64),
        pointer,
        ct.c_bool,
        ct.c_size_t,
        pointer,
        ct.POINTER(ct.c_size_t),
    ]
    result.run.restype = ct.c_int
    result.iterations = iterations
    return result


@pytest.mark.parametrize("atoms", [1, 2, 5, 32, 33, 96, 128])
@pytest.mark.parametrize("cached", [False, True])
def test_tiled_large_domains_keep_pair_work_and_order(
    helper: ct.CDLL, atoms: int, cached: bool
) -> None:
    rng = np.random.default_rng(8200 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 2
    points = rng.normal(size=(3, 3)) * 3
    owners = rng.integers(atoms, size=len(points), dtype=np.int64)
    seeds = rng.normal(size=len(points))
    expected = retained.run(helper, points, centers, owners, seeds, cached, 0)
    for threads in (1, 7, 32):
        actual = retained.run(helper, points, centers, owners, seeds, cached, threads)
        assert actual[0] == expected[0] == 0
        np.testing.assert_allclose(actual[1], expected[1], rtol=5e-13, atol=2e-13)
        # Every unordered pair is produced exactly once per pass, independently
        # of strip count. Neither center norms nor logarithms gain extra passes.
        assert actual[2] == expected[2]
        assert actual[2][0] == len(points) * atoms * (atoms - 1)
    moved = centers + rng.normal(size=centers.shape) * 0.1
    for geometry in (moved, centers):
        actual = retained.run(helper, points, geometry, owners, seeds, cached, 7)
        expected = retained.run(helper, points, geometry, owners, seeds, cached, 0)
        assert actual[0] == expected[0] == 0
        np.testing.assert_allclose(actual[1], expected[1], rtol=5e-13, atol=2e-13)


@pytest.mark.parametrize(
    "case",
    [
        "saturated",
        "rounded_zero",
        "positive_zero",
        "negative_zero",
        "coincident",
        "collision",
        "nonfinite",
        "tiny",
        "huge",
        "empty",
    ],
)
def test_tiled_edge_semantics(helper: ct.CDLL, case: str) -> None:
    retained.test_cooperative_edge_semantics(helper, case)


def test_tiled_independent_decimal_oracle_and_invariances(helper: ct.CDLL) -> None:
    retained.test_cooperative_independent_decimal_fd_translation_and_permutation(helper)
    retained.test_single_atom_point_reuse_then_collision_is_collective(helper)


def test_tiled_multi_strip_independent_decimal_directional_derivative(
    helper: ct.CDLL,
) -> None:
    from decimal import Decimal, localcontext

    from test_grid_response import decimal_partition

    rng = np.random.default_rng(9361)
    centers = rng.normal(size=(11, 3))
    points = rng.normal(size=(2, 3)) * 0.7
    owners = np.array([0, 7], dtype=np.int64)
    seeds = np.array([0.3, -0.7])
    direction = rng.normal(size=centers.shape) * 0.2
    status, gradient, _ = retained.run(helper, points, centers, owners, seeds, True, 7)
    assert status == 0
    with localcontext() as context:
        context.prec = 60
        h = Decimal("1e-7")

        def moved(
            values: np.ndarray, motion: np.ndarray, sign: int
        ) -> list[list[Decimal]]:
            return [
                [
                    Decimal(str(x)) + sign * h * Decimal(str(dx))
                    for x, dx in zip(row, dr)
                ]
                for row, dr in zip(values, motion)
            ]

        plus, minus = [
            decimal_partition(
                moved(points, direction[owners], sign),
                moved(centers, direction, sign),
                helper.iterations,
            )
            for sign in (1, -1)
        ]
        expected = sum(
            Decimal(str(seed)) * (p[owner] - m[owner]) / (2 * h)
            for seed, p, m, owner in zip(seeds, plus, minus, owners)
        )
    assert abs(float(expected) - np.sum(gradient * direction)) < 2e-12
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=2e-13)
    order = rng.permutation(len(centers))
    actual = retained.run(
        helper, points, centers[order], np.argsort(order)[owners], seeds, True, 7
    )
    assert actual[0] == 0
    np.testing.assert_allclose(actual[1], gradient[order], rtol=5e-12, atol=2e-13)
