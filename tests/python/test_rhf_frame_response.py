"""Independent dense test oracles for matrix-sized RHF response composition."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc_compiler.cc.gradient_equations import build_hamiltonian_programs
from generativeqc_compiler.method.rhf_orbital_response import build_rhf_frame_response
from generativeqc_compiler.tensor import execute


def sym(a: np.ndarray) -> np.ndarray:
    return (a + a.T) / 2


def symmetric_eri(rng: np.random.Generator, n: int) -> np.ndarray:
    g = rng.normal(scale=0.03, size=(n, n, n, n))
    g = (g + g.transpose(1, 0, 2, 3)) / 2
    g = (g + g.transpose(0, 1, 3, 2)) / 2
    return (g + g.transpose(2, 3, 0, 1)) / 2


def potential(g: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.einsum("uvwx,wx->uv", g, d) - 0.5 * np.einsum("uwvx,wx->uv", g, d)


@pytest.mark.parametrize("o,v", [(1, 2), (2, 2), (2, 3)])
@pytest.mark.parametrize("reference_seed", [0.0, 1.0])
def test_full_frame_hcore_eri_and_metric_directions(
    o: int, v: int, reference_seed: float
) -> None:
    n = o + v
    rng = np.random.default_rng(1806)
    c = np.linalg.qr(rng.normal(size=(n, n)))[0]
    h = sym(rng.normal(size=(n, n)))
    g = symmetric_eri(rng, n)
    d = 2 * c[:, :o] @ c[:, :o].T
    f = h + potential(g, d)
    bar_f = rng.normal(size=(n, n))
    bar_c = rng.normal(scale=0.1, size=(n, n))
    programs = build_rhf_frame_response(o, v)
    feeds = {
        "coefficients": c,
        "hcore": h,
        "fock_ao": f,
        "bar_fock_mo": bar_f,
        "bar_reference_electronic_energy": np.asarray(reference_seed),
        "bar_frame": bar_c,
    }
    p = execute(programs.potential_seed, feeds).outputs["fock_ao_weights"]
    np.testing.assert_allclose(p, p.T, atol=3e-15, rtol=0)
    result = execute(
        programs.weights, {**feeds, "bar_density": potential(g, p)}
    ).outputs
    np.testing.assert_allclose(
        result["rotation_gradient"],
        c.T @ result["frame_cotangent"],
        atol=3e-14,
        rtol=3e-14,
    )
    np.testing.assert_allclose(
        result["orbital_rhs"], -result["stationarity"][:o, o:], atol=3e-14, rtol=0
    )

    def objective(c: np.ndarray, h: np.ndarray, g: np.ndarray) -> float:
        density = 2 * c[:, :o] @ c[:, :o].T
        fock = h + potential(g, density)
        return float(
            reference_seed * 0.5 * np.sum(density * (h + fock))
            + np.sum(bar_f * (c.T @ fock @ c))
            + np.sum(bar_c * c)
        )

    dc = rng.normal(scale=0.2, size=c.shape)
    dh = sym(rng.normal(scale=0.2, size=h.shape))
    dg = symmetric_eri(rng, n)
    ds = sym(rng.normal(scale=0.2, size=h.shape))
    metric_c = -0.5 * c @ (c.T @ ds @ c)
    directions = [
        ("C", dc, float(np.sum(result["frame_cotangent"] * dc))),
        ("h", dh, float(np.sum(result["hcore"] * dh))),
        ("g", dg, float(np.sum(p * potential(dg, d)))),
        ("S", metric_c, float(np.sum(result["overlap"] * ds))),
    ]
    for kind, direction, analytic in directions:
        for step in (1e-4, 3e-5):
            values = [
                objective(
                    c + sign * step * direction if kind in ("C", "S") else c,
                    h + sign * step * direction if kind == "h" else h,
                    g + sign * step * direction if kind == "g" else g,
                )
                for sign in (-1, 1)
            ]
            np.testing.assert_allclose(
                (values[1] - values[0]) / (2 * step), analytic, atol=3e-8, rtol=3e-8
            )


@pytest.mark.parametrize("o,v", [(1, 2), (2, 2), (2, 3)])
@pytest.mark.parametrize("degenerate", [False, True])
def test_orbital_action_matches_independent_rotation_and_dense_generated_oracle(
    o: int, v: int, degenerate: bool
) -> None:
    n = o + v
    rng = np.random.default_rng(1763)
    c = np.linalg.qr(rng.normal(size=(n, n)))[0]
    eps = (
        np.r_[np.full(o, -1.0), np.full(v, 0.5)]
        if degenerate
        else np.linspace(-1, 0.8, n)
    )
    g = symmetric_eri(rng, n)
    density = 2 * c[:, :o] @ c[:, :o].T
    f = c @ np.diag(eps) @ c.T
    h = f - potential(g, density)
    x = rng.normal(scale=0.2, size=(o, v))
    k = np.zeros((n, n))
    k[:o, o:] = x
    k[o:, :o] = -x.T
    programs = build_rhf_frame_response(o, v)
    feeds = {"coefficients": c, "rotation": np.eye(n), "d_rotation": k, "fock_ao": f}
    dd = execute(programs.density_direction, feeds).outputs["density_direction"]
    action = execute(
        programs.orbital_action, {**feeds, "d_fock_ao": potential(g, dd)}
    ).outputs["orbital_action"]
    for step in (1e-4, 3e-5):
        blocks = []
        for sign in (-1, 1):
            moved = c @ (np.eye(n) + sign * step * k)
            fm = h + potential(g, 2 * moved[:, :o] @ moved[:, :o].T)
            blocks.append((moved.T @ fm @ moved)[:o, o:])
        np.testing.assert_allclose(
            -(blocks[1] - blocks[0]) / (2 * step), action, atol=2e-8, rtol=2e-8
        )
    dense = build_hamiltonian_programs(o, v)
    gmo = np.einsum("up,vq,wr,xs,uvwx->pqrs", c, c, c, c, g, optimize=True)
    oracle = execute(
        dense.orbital_jvp.program,
        {"h": c.T @ h @ c, "g": gmo, "rotation": np.eye(n), "d_rotation": k},
    ).outputs["d_fov"]
    np.testing.assert_allclose(action, -oracle, atol=3e-13, rtol=3e-13)


def test_large_shape_inventory_never_materializes_four_index_inputs_or_hessian() -> (
    None
):
    for o, v in ((2, 3), (9, 221), (21, 243)):
        programs = build_rhf_frame_response(o, v)
        for program in (
            programs.primal,
            programs.potential_seed,
            programs.weights,
            programs.density_direction,
            programs.orbital_action,
        ):
            assert all(len(node.spec.indices) <= 2 for node in program.live_nodes)
        inputs = {
            node.attrs["name"]
            for node in programs.potential_seed.live_nodes
            if node.op == "input"
        }
        assert "bar_density" not in inputs and "bar_frame" not in inputs
        tangent_inputs = {
            node.attrs["name"]
            for node in programs.density_direction.live_nodes
            if node.op == "input"
        }
        assert "d_fock_ao" not in tangent_inputs


@pytest.mark.parametrize("o,v", [(1, 2), (2, 3)])
def test_physical_z_composition_restores_stationarity_without_same_space_gaps(
    o: int, v: int
) -> None:
    """Assemble a Hessian only in this tiny oracle, then audit the full pullback."""
    n = o + v
    rng = np.random.default_rng(1806)
    c = np.linalg.qr(rng.normal(size=(n, n)))[0]
    eps = np.r_[np.full(o, -1.0), np.full(v, 0.5)]
    g = symmetric_eri(rng, n)
    density = 2 * c[:, :o] @ c[:, :o].T
    f = c @ np.diag(eps) @ c.T
    h = f - potential(g, density)
    programs = build_rhf_frame_response(o, v)
    base = {
        "coefficients": c,
        "fock_ao": f,
        "hcore": h,
        "rotation": np.eye(n),
        "bar_reference_electronic_energy": np.asarray(1.0),
    }

    def action(x: np.ndarray) -> np.ndarray:
        k = np.zeros((n, n))
        k[:o, o:] = x.reshape(o, v)
        k[o:, :o] = -x.reshape(o, v).T
        feeds = {**base, "d_rotation": k}
        dd = execute(programs.density_direction, feeds).outputs["density_direction"]
        return (
            execute(programs.orbital_action, {**feeds, "d_fock_ao": potential(g, dd)})
            .outputs["orbital_action"]
            .ravel()
        )

    matrix = np.stack([action(e) for e in np.eye(o * v)], axis=1)
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-13, rtol=0)
    assert np.linalg.eigvalsh(matrix)[0] > 0.1
    # A linear independent frame cotangent supplies only an ov torque.
    frame = np.zeros((n, n))
    frame[:o, o:] = rng.normal(scale=0.1, size=(o, v))
    bar_c = c @ frame

    def weights(seed: np.ndarray, frame_seed: np.ndarray) -> dict:
        feeds = {**base, "bar_fock_mo": seed, "bar_frame": frame_seed}
        p = execute(programs.potential_seed, feeds).outputs["fock_ao_weights"]
        return execute(
            programs.weights, {**feeds, "bar_density": potential(g, p)}
        ).outputs

    hf = weights(np.zeros((n, n)), np.zeros((n, n)))
    np.testing.assert_allclose(hf["hcore"], density, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(
        hf["overlap"], -2 * (c[:, :o] * eps[:o]) @ c[:, :o].T, atol=3e-14, rtol=3e-14
    )
    first = weights(np.zeros((n, n)), bar_c)
    z = np.linalg.solve(matrix, first["orbital_rhs"].ravel())
    seed = np.zeros((n, n))
    seed[:o, o:] = -z.reshape(o, v)
    relaxed = weights(seed, bar_c)
    np.testing.assert_allclose(relaxed["stationarity"], 0, atol=4e-13, rtol=0)
    np.testing.assert_allclose(
        matrix @ z, first["orbital_rhs"].ravel(), atol=4e-13, rtol=0
    )
