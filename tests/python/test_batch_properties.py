"""Output selection must skip response work and survive later force replays."""

import ctypes
import json
import os
import typing
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import Calculator, _native
from generativeqc.batch import PreparedBatch

SYSTEMS = [
    [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))],
    [(1, (0.0, 0.0, -0.8)), (1, (0.0, 0.0, 0.8))],
]


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize(
    "route", ["cpu-direct", "cpu-df", "cuda-direct", "cuda-df", "cuda-source"]
)
def test_energy_only_batch_preserves_replay_and_force_recovery(
    method: typing.Any, route: typing.Any, monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    device, provider = route.split("-")
    if device == "cuda":
        if os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1":
            pytest.skip("explicit allocated-GPU opt-in")
        assert os.environ.get("SLURM_JOB_ID")
    options = {
        "method": method,
        "device": device,
        "density_fitting": "none" if provider == "direct" else device,
        "density_fitting_memory_budget_bytes": 8 << 20 if provider == "source" else 0,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    actual = Calculator(**options)
    reference = Calculator(**options)
    # Include an empty beta occupied space and a closed-shell neighbor in the
    # UHF fleet, so output selection also crosses independent spin buckets.
    preparation = {"multiplicities": [3, 1]} if method == "uhf" else {}
    with (
        actual.prepare_batch(SYSTEMS, **preparation) as batch,
        reference.prepare_batch(SYSTEMS, **preparation) as full,
    ):
        for replay in range(3):
            expected = full.execute(strict=True)
            trace = tmp_path / f"energy-{replay}.jsonl"
            monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
            result = batch.execute(properties=("energy",), strict=True)
            monkeypatch.delenv("GENERATIVEQC_DF_TRACE")
            np.testing.assert_allclose(
                result.energies, expected.energies, atol=1e-10, rtol=0
            )
            assert all(item.forces is None for item in result.items)
            if device == "cuda" and provider != "direct":
                # Reading only output buffers would pass numerical parity too.
                # Executed traces must prove response was never launched.
                records = [json.loads(line) for line in trace.read_text().splitlines()]
                assert records
                assert not {
                    "force_response",
                    "one_electron_response",
                    "one_electron_derivative_export",
                    "nuclear_derivative_export",
                } & {record["operation"] for record in records}
            expected = full.execute(strict=True)
            recovered = batch.execute(strict=True)
            np.testing.assert_allclose(
                recovered.energies, expected.energies, atol=1e-10, rtol=0
            )
            for item, target in zip(recovered.items, expected.items, strict=True):
                np.testing.assert_allclose(
                    item.forces, target.forces, atol=1e-9, rtol=0
                )
        failed = batch.execute([np.zeros((1, 3)), None], properties=("energy",))
        assert failed.failure_indices == (0,)
        assert failed.items[1].converged and failed.items[1].forces is None


@pytest.mark.parametrize(
    "properties,error",
    [
        ("energy", TypeError),
        ([], ValueError),
        (["forces"], ValueError),
        (["energy", "dipole"], ValueError),
        ([[]], TypeError),
    ],
)
def test_invalid_batch_properties_reject_before_execution(
    properties: typing.Any, error: typing.Any, monkeypatch: typing.Any
) -> None:
    with Calculator().prepare_batch(SYSTEMS) as batch:

        def forbidden(*args: typing.Any) -> None:
            pytest.fail("invalid output request reached native execution")

        monkeypatch.setattr(batch._library, "generativeqc_batch_execute", forbidden)
        with pytest.raises(error):
            batch.execute(properties=properties)


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_unqualified_dft_basis_rejects_forces_before_execution(
    method: typing.Any, monkeypatch: typing.Any
) -> None:
    preparation = (
        {"charges": [-1], "multiplicities": [2]} if method.endswith("uks") else {}
    )
    # The f-containing basis context remains outside the qualified s/p/d CPU
    # force domain, even though the named semilocal methods can expose forces.
    calculator = Calculator(method=method, basis="def2-tzvp")
    assert calculator.capabilities.supported_properties == frozenset({"energy"})
    with calculator.prepare_batch(SYSTEMS[:1], **preparation) as batch:

        def forbidden(*args: typing.Any) -> None:
            pytest.fail("unsupported force request reached native execution")

        monkeypatch.setattr(batch._library, "generativeqc_batch_execute", forbidden)
        with pytest.raises(ValueError, match="does not support properties: forces"):
            batch.execute(properties=("energy", "forces"))


def test_dft_prepared_native_force_query_fails_closed_without_method_whitelist() -> (
    None
):
    seen: list[int] = []

    def qualified(_owner: object, index: int, output: object) -> int:
        seen.append(index)
        value = _native.PROPERTY_ENERGY | _native.PROPERTY_FORCES
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint32))[0] = value
        return _native.STATUS_SUCCESS

    holder = SimpleNamespace(
        _library=SimpleNamespace(
            generativeqc_ks_batch_supported_properties_v1=qualified
        ),
        _batch=object(),
        _context=None,
        _systems=(SYSTEMS[0], SYSTEMS[1]),
    )
    assert PreparedBatch._native_dft_force_eligible(holder)
    assert seen == [0, 1]
    assert holder._library.generativeqc_ks_batch_supported_properties_v1.argtypes == [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]

    def second_unqualified(owner: object, index: int, output: object) -> int:
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint32))[0] = (
            _native.PROPERTY_ENERGY | (_native.PROPERTY_FORCES if index == 0 else 0)
        )
        return _native.STATUS_SUCCESS

    holder._library.generativeqc_ks_batch_supported_properties_v1 = second_unqualified
    assert not PreparedBatch._native_dft_force_eligible(holder)

    def unavailable(_owner: object, _index: int, _output: object) -> int:
        return _native.STATUS_NOT_IMPLEMENTED

    holder._library.generativeqc_ks_batch_supported_properties_v1 = unavailable
    assert not PreparedBatch._native_dft_force_eligible(holder)
    holder._library = SimpleNamespace()
    assert not PreparedBatch._native_dft_force_eligible(holder)


def test_cpu_df_pbe_python_api_consumes_qualified_native_batch_forces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = {
        "method": "pbe-rks",
        "basis": "sto-3g",
        "device": "cpu",
        "density_fitting": "cpu",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-14,
    }
    with (
        Calculator(**options).prepare_batch(SYSTEMS[:1]) as native_batch,
        Calculator(**options).prepare_batch(SYSTEMS[:1]) as python_batch,
    ):
        if not hasattr(
            native_batch._library, "generativeqc_ks_batch_supported_properties_v1"
        ):
            pytest.skip("current native prepared batch force query is not installed")

        def forbidden_python_force(*_args: typing.Any, **_kwargs: typing.Any) -> None:
            pytest.fail("CPU DF-PBE fell back to the Python stationary force driver")

        monkeypatch.setattr(
            native_batch, "_public_dft_cpu_force", forbidden_python_force
        )
        monkeypatch.setattr(python_batch, "_native_dft_force_eligible", lambda: False)
        native = native_batch.execute(
            strict=True, properties=("energy", "forces")
        ).items[0]
        reference = python_batch.execute(
            strict=True, properties=("energy", "forces")
        ).items[0]
        assert native.forces is not None and reference.forces is not None
        np.testing.assert_allclose(native.energy, reference.energy, atol=1e-8, rtol=0)
        np.testing.assert_allclose(native.forces, reference.forces, atol=1e-4, rtol=0)
        assert np.all(np.isfinite(native.forces))
