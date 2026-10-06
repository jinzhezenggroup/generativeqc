import os
import typing
from fractions import Fraction

import pytest
from generativeqc import Calculator, KsOptions
from generativeqc_compiler.method import MethodSpec, resolve_method

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="opt-in native CUDA DFT gate",
)


@pytest.mark.parametrize(
    ("method", "charge", "multiplicity"),
    (
        ("lda-rks", 0, 1),
        ("pbe-rks", 0, 1),
        ("lda-uks", -1, 2),
        ("pbe-uks", -1, 2),
    ),
)
def test_native_cuda_dft_matches_independently_converged_cpu_endpoint(
    method: typing.Any, charge: typing.Any, multiplicity: typing.Any
) -> None:
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    options = {
        "method": method,
        "basis": "sto-3g",
        "max_iterations": 200,
        "energy_tolerance": 1.0e-12,
        "density_tolerance": 1.0e-10,
    }
    state = {
        "charge": charge,
        "multiplicity": multiplicity,
        "properties": ("energy",),
    }
    cpu = Calculator(device="cpu", **options).singlepoint(atoms, **state)
    cuda_calculator = Calculator(device="cuda", **options)
    cuda = cuda_calculator.singlepoint(atoms, **state)

    assert cpu.converged and cuda.converged
    assert cuda.executed_backend == "cuda"
    assert cuda.energy == pytest.approx(cpu.energy, abs=2.0e-9)
    assert cuda.physical_residual_rms < 1.0e-9
    assert cuda.forces is None


def test_cuda_scaled_dft_force_request_remains_unqualified() -> None:
    # Ordinary LDA/PBE CUDA forces are public. Keep rejection coverage for a
    # genuinely unqualified scaled composition on the same backend.
    graph = resolve_method(
        MethodSpec(
            "half-exchange-PBE",
            (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
        )
    )
    calculator = Calculator(
        method="pbe-rks",
        basis="sto-3g",
        device="cuda",
        ks_options=KsOptions(composition=graph),
    )
    assert calculator.capabilities.supported_properties == frozenset({"energy"})
    with pytest.raises(ValueError, match=r"does not support properties.*forces"):
        calculator.singlepoint(
            [("He", (0.0, 0.0, 0.0))], properties=("energy", "forces")
        )


def test_native_cuda_dft_ragged_batch_replay_and_failure_isolation() -> None:
    systems = [
        [("He", (0.0, 0.0, 0.0))],
        [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    ]
    cpu = Calculator(method="lda-rks", basis="sto-3g", device="cpu")
    expected = [
        cpu.singlepoint(system, properties=("energy",)).energy for system in systems
    ]
    cuda = Calculator(method="lda-rks", basis="sto-3g", device="cuda")
    with cuda.prepare_batch(systems, warm_start=True) as prepared:
        cold = prepared.execute(strict=True, properties=("energy",))
        warm = prepared.execute(strict=True, properties=("energy",))
        isolated = prepared.execute(
            coordinates=[None, [0.0]], strict=False, properties=("energy",)
        )

    assert [item.bucket_id for item in cold.items] == [0, 1]
    assert cold.energies == pytest.approx(expected, abs=2.0e-9)
    assert all(item.executed_backend == "cuda" for item in cold.items)
    assert all(item.warm_start_used for item in warm.items)
    assert isolated.items[0].succeeded and isolated.items[0].warm_start_used
    assert isolated.items[1].status_message == "invalid argument"
