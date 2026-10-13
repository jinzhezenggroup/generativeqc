"""Independent ordered-MP2 oracle and differentiated factor/spectrum contracts."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc_compiler.mp2.equations import (
    df_energy_program,
    df_energy_response_program,
)
from generativeqc_compiler.tensor import execute


def ordered_energy(inputs: dict[str, np.ndarray]) -> tuple[float, float]:
    """Scalar-loop oracle, independent of the compiler's contraction/VJP graph."""
    factors = inputs["bov"]
    _, occupied, virtuals = factors.shape
    opposite, same = 0.0, 0.0
    for first in range(occupied):
        for second in range(occupied):
            for particle in range(virtuals):
                for partner in range(virtuals):
                    direct = float(
                        factors[:, first, particle] @ factors[:, second, partner]
                    )
                    exchange = float(
                        factors[:, first, partner] @ factors[:, second, particle]
                    )
                    denominator = (
                        inputs["ei"][first]
                        + inputs["ej"][second]
                        - inputs["ea"][particle]
                        - inputs["eb"][partner]
                    )
                    opposite += direct * direct / denominator
                    same += direct * (direct - exchange) / denominator
    return opposite, same


@pytest.mark.parametrize(
    "occupied,virtuals,auxiliaries", ((1, 2, 3), (2, 3, 4), (3, 2, 5))
)
def test_df_mp2_energy_and_factor_spectrum_reverse(
    occupied: int, virtuals: int, auxiliaries: int
) -> None:
    generator = np.random.default_rng(447 + occupied + virtuals)
    inputs = {
        "bov": generator.normal(size=(auxiliaries, occupied, virtuals)),
        "ei": -1.0 - generator.random(occupied),
        "ej": -1.0 - generator.random(occupied),
        "ea": 0.2 + generator.random(virtuals),
        "eb": 0.2 + generator.random(virtuals),
    }
    expected = ordered_energy(inputs)
    primal = execute(df_energy_program(occupied, virtuals, auxiliaries), inputs).outputs
    reverse = execute(
        df_energy_response_program(occupied, virtuals, auxiliaries),
        {
            **inputs,
            "bar_opposite_spin": np.asarray(1.0),
            "bar_same_spin": np.asarray(1.0),
        },
    ).outputs
    for result in (primal, reverse):
        np.testing.assert_allclose(
            [result["opposite_spin"], result["same_spin"]],
            expected,
            rtol=3e-14,
            atol=3e-13,
        )
    for name, values in inputs.items():
        direction = generator.normal(size=values.shape)
        analytic = float(np.vdot(reverse["bar_" + name], direction))
        step = 2e-5
        minus = sum(ordered_energy({**inputs, name: values - step * direction}))
        plus = sum(ordered_energy({**inputs, name: values + step * direction}))
        assert analytic == pytest.approx(
            (plus - minus) / (2 * step), abs=2e-7, rel=3e-8
        )


def test_shared_physical_spectrum_combines_repeated_cotangents() -> None:
    generator = np.random.default_rng(21)
    occupied = np.asarray([-1.1, -0.7])
    virtuals = np.asarray([0.3, 0.8, 1.2])
    inputs = {
        "bov": generator.normal(size=(4, 2, 3)),
        "ei": occupied,
        "ej": occupied,
        "ea": virtuals,
        "eb": virtuals,
    }
    reverse = execute(
        df_energy_response_program(2, 3, 4),
        {
            **inputs,
            "bar_opposite_spin": np.asarray(1.0),
            "bar_same_spin": np.asarray(1.0),
        },
    ).outputs
    for names in (("ei", "ej"), ("ea", "eb")):
        direction = generator.normal(size=inputs[names[0]].shape)
        analytic = float(
            np.vdot(reverse["bar_" + names[0]] + reverse["bar_" + names[1]], direction)
        )
        step = 1e-5
        energies = [
            sum(
                ordered_energy(
                    {
                        **inputs,
                        **{
                            name: inputs[name] + sign * step * direction
                            for name in names
                        },
                    }
                )
            )
            for sign in (-1, 1)
        ]
        assert analytic == pytest.approx(
            (energies[1] - energies[0]) / (2 * step), abs=2e-7
        )


@pytest.mark.parametrize("degenerate", (False, True))
def test_full_fock_response_matches_noncanonical_amplitude_oracle(
    degenerate: bool,
) -> None:
    generator = np.random.default_rng(561)
    factors = generator.normal(size=(4, 2, 3))
    occupied = np.asarray([-0.8, -0.8] if degenerate else [-1.3, -0.8])
    virtuals = np.asarray([0.4, 0.4, 0.4] if degenerate else [0.4, 0.8, 1.1])
    inputs = {
        "bov": factors,
        "ei": occupied,
        "ej": occupied,
        "ea": virtuals,
        "eb": virtuals,
        "bar_opposite_spin": np.asarray(1.0),
        "bar_same_spin": np.asarray(1.0),
    }
    response = execute(df_energy_response_program(2, 3, 4), inputs).outputs
    np.testing.assert_allclose(
        np.diag(response["bar_foo"]),
        response["bar_ei"] + response["bar_ej"],
        atol=2e-13,
    )
    np.testing.assert_allclose(
        np.diag(response["bar_fvv"]),
        response["bar_ea"] + response["bar_eb"],
        atol=2e-13,
    )

    def noncanonical_energy(foo: np.ndarray, fvv: np.ndarray) -> float:
        direct = np.einsum("Qia,Qjb->ijab", factors, factors)
        occupied_identity, virtual_identity = np.eye(2), np.eye(3)
        operator = (
            np.kron(
                np.kron(np.kron(foo, occupied_identity), virtual_identity),
                virtual_identity,
            )
            + np.kron(
                np.kron(np.kron(occupied_identity, foo), virtual_identity),
                virtual_identity,
            )
            - np.kron(
                np.kron(np.kron(occupied_identity, occupied_identity), fvv),
                virtual_identity,
            )
            - np.kron(
                np.kron(
                    np.kron(occupied_identity, occupied_identity), virtual_identity
                ),
                fvv,
            )
        )
        amplitudes = np.linalg.solve(operator, direct.reshape(-1))
        return float(amplitudes @ (2 * direct - direct.swapaxes(2, 3)).reshape(-1))

    for name, diagonal in (("foo", occupied), ("fvv", virtuals)):
        direction = generator.normal(size=(len(diagonal), len(diagonal)))
        direction = (direction + direction.T) / 2
        analytic = float(np.vdot(response["bar_" + name], direction))
        step = 1e-5
        displaced = []
        for sign in (-1, 1):
            foo, fvv = np.diag(occupied), np.diag(virtuals)
            if name == "foo":
                foo += sign * step * direction
            else:
                fvv += sign * step * direction
            displaced.append(noncanonical_energy(foo, fvv))
        assert analytic == pytest.approx(
            (displaced[1] - displaced[0]) / (2 * step), abs=3e-7
        )

    # Same-space rotations must be stationary even with off-diagonal Fock
    # derivatives; diagonal epsilon seeds alone fail this invariant.
    occupied_rotation = generator.normal(size=(2, 2))
    occupied_rotation -= occupied_rotation.T.copy()
    virtual_rotation = generator.normal(size=(3, 3))
    virtual_rotation -= virtual_rotation.T.copy()
    factor_direction = -np.einsum(
        "ik,Qka->Qia", occupied_rotation, factors
    ) + np.einsum("Qic,ca->Qia", factors, virtual_rotation)
    foo, fvv = np.diag(occupied), np.diag(virtuals)
    total = (
        np.vdot(response["bar_bov"], factor_direction)
        + np.vdot(
            response["bar_foo"], foo @ occupied_rotation - occupied_rotation @ foo
        )
        + np.vdot(response["bar_fvv"], fvv @ virtual_rotation - virtual_rotation @ fvv)
    )
    assert abs(total) < 2e-11


@pytest.mark.parametrize("dimensions", ((0, 2, 3), (2, 0, 3), (2, 3, 0)))
def test_df_mp2_refuses_empty_spaces(dimensions: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        df_energy_program(*dimensions)
