"""Actual public comparison ABI with independent high-precision observable norms."""

from __future__ import annotations

import ctypes
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import ObservableTarget, TargetAccuracy, _native
from generativeqc._observable_comparison import (
    _Comparison,
    _Context,
    measure_observables,
)

ROOT = Path(__file__).resolve().parents[2]
POINTER = ctypes.POINTER(ctypes.c_double)


@pytest.fixture(scope="module")
def library(required_native_cxx: Any, tmp_path_factory: Any) -> Any:
    """Compile the production C ABI, with no substituted numerical implementation."""
    folder = tmp_path_factory.mktemp("observable-comparison")
    path = required_native_cxx.build_shared(
        [ROOT / "src/api/c_api_precision_admission.cpp"],
        folder / "comparison.so",
        compile_args=("-std=c++20", f"-I{ROOT / 'include'}", f"-I{ROOT / 'src'}"),
    )
    selected = ctypes.CDLL(str(path))
    selected.generativeqc_status_message = lambda status: (
        b"observable comparison refused"
    )
    return selected


def query(library: Any) -> Any:
    function = library.generativeqc_observable_comparison_query_v1
    function.argtypes = [
        ctypes.POINTER(_Context),
        POINTER,
        POINTER,
        ctypes.POINTER(_Comparison),
    ]
    function.restype = ctypes.c_int
    return function


def target() -> TargetAccuracy:
    return TargetAccuracy(
        (
            ObservableTarget("energy", "absolute", "Eh", 1e-6, 1e-7),
            ObservableTarget("forces", "max_abs", "Eh/bohr", 1e-6, 1e-7),
            ObservableTarget("forces", "rms", "Eh/bohr", 1e-6, 1e-7),
        )
    )


def rms_oracle(values: Any) -> float:
    """Use decimal exact input conversion, squares and square root, not NumPy RMS."""
    with localcontext() as context:
        context.prec = 100
        squares = [Decimal.from_float(float(value)) ** 2 for value in values]
        return float((sum(squares) / Decimal(len(squares))).sqrt())


def difference_oracle(values: Any, reference: Any) -> list[float]:
    """Round exact decimal subtraction once, including binary64 tie cases."""
    with localcontext() as context:
        context.prec = 1200
        return [
            float(
                Decimal.from_float(float(value)) - Decimal.from_float(float(expected))
            )
            for value, expected in zip(values, reference, strict=True)
        ]


@pytest.mark.parametrize("scale", (1e-300, 1e-100, 1.0, 1e100, 1e300))
@pytest.mark.parametrize("atoms", (1, 17, 257))
def test_scaled_rms_against_independent_decimal(
    library: Any, scale: float, atoms: int
) -> None:
    generator = np.random.default_rng(322 + atoms)
    reference = generator.uniform(-0.25, 0.25, size=(atoms, 3)) * scale
    values = reference + generator.uniform(-0.5, 0.5, size=(atoms, 3)) * scale
    result = measure_observables(
        target(),
        {"energy": -2.0, "forces": values},
        {"energy": -2.125, "forces": reference},
        library=library,
    )
    for requirement, difference, reference_norm, allowance in result:
        if requirement.observable == "energy":
            assert difference == 0.125 and reference_norm == 2.125
        else:
            delta = difference_oracle(values.flat, reference.flat)
            if requirement.norm == "rms":
                expected_difference = rms_oracle(delta)
                expected_reference = rms_oracle(reference.flat)
                assert abs(difference - expected_difference) <= 4 * math.ulp(
                    expected_difference
                )
                assert abs(reference_norm - expected_reference) <= 4 * math.ulp(
                    expected_reference
                )
            else:
                assert difference == max(abs(value) for value in delta)
                assert reference_norm == max(
                    abs(float(value)) for value in reference.flat
                )
        assert allowance == requirement.absolute + requirement.relative * reference_norm


@pytest.mark.parametrize("norm", ("max_abs", "rms"))
def test_exact_threshold_neighbors_and_extreme_finite_norms(
    library: Any, norm: str
) -> None:
    values = np.ones((3, 3))
    for allowance in (math.nextafter(1.0, 0), 1.0, math.nextafter(1.0, math.inf)):
        contract = TargetAccuracy(
            (ObservableTarget("forces", norm, "Eh/bohr", allowance),)
        )
        measurement = measure_observables(
            contract, {"forces": values}, {"forces": values * 0}, library=library
        )[0]
        assert measurement[1] == 1 and measurement[2] == 0
        assert (measurement[1] <= measurement[3]) == (allowance >= 1)
    values.fill(np.finfo(float).max)
    contract = TargetAccuracy((ObservableTarget("forces", norm, "Eh/bohr", 1),))
    observed = measure_observables(
        contract, {"forces": values}, {"forces": values}, library=library
    )[0]
    assert observed[1] == 0 and observed[2] == np.finfo(float).max
    values.fill(np.nextafter(0.0, 1.0))
    observed = measure_observables(
        contract, {"forces": values}, {"forces": values * 0}, library=library
    )[0]
    assert observed[1] == np.nextafter(0.0, 1.0)


def context(mask: int = 7) -> _Context:
    result = _Context(ctypes.sizeof(_Context), _native.ABI_VERSION, mask)
    result.atom_count = 2 if mask & 6 else 0
    if mask & 1:
        result.energy, result.reference_energy = -1, -1.1
    for slot in range(3):
        if mask & (1 << slot):
            result.absolute[slot], result.relative[slot] = 1e-3, 1e-2
    return result


@pytest.mark.parametrize("scale", (1e-300, 1.0, 1e300))
def test_small_square_contributions_survive_compensated_accumulation(
    library: Any, scale: float
) -> None:
    values = np.full((4097, 3), scale * 2**-27)
    values[0, 0] = scale
    contract = TargetAccuracy((ObservableTarget("forces", "rms", "Eh/bohr", 1e-6),))
    observed = measure_observables(
        contract, {"forces": values}, {"forces": np.zeros_like(values)}, library=library
    )[0]
    expected = rms_oracle(values.flat)
    assert abs(observed[1] - expected) <= 4 * math.ulp(expected)


def test_allowance_keeps_multiply_then_add_rounding(library: Any) -> None:
    descriptor = context(1)
    descriptor.reference_energy = 1 - 2**-27
    descriptor.energy = descriptor.reference_energy
    descriptor.absolute[0], descriptor.relative[0] = 3 * 2**-53, 1 + 2**-27
    output = _Comparison(ctypes.sizeof(_Comparison), _native.ABI_VERSION)
    assert (
        query(library)(ctypes.byref(descriptor), None, None, ctypes.byref(output)) == 0
    )
    assert output.allowances[0] == 1 + 2**-51
    assert output.allowances[0] != 1 + 2**-52


def test_relative_only_zero_reference_and_nonfinite_values(library: Any) -> None:
    contract = TargetAccuracy(
        (ObservableTarget("energy", "absolute", "Eh", relative=0.1),)
    )
    result = measure_observables(
        contract, {"energy": 0}, {"energy": 0}, library=library
    )[0]
    assert result[1:] == (0, 0, 0)
    result = measure_observables(
        contract, {"energy": 1}, {"energy": 0}, library=library
    )[0]
    assert result[1:] == (1, 0, 0)
    for value in (math.nan, math.inf, -math.inf, 1j, "1"):
        with pytest.raises(ValueError):
            measure_observables(
                contract, {"energy": value}, {"energy": 0}, library=library
            )


@pytest.mark.parametrize(
    "mutation",
    (
        "empty_mask",
        "unknown_mask",
        "reserved",
        "context_size",
        "context_version",
        "output_size",
        "output_version",
        "zero_atoms",
        "overflow_atoms",
        "negative_absolute",
        "nonfinite_relative",
        "zero_allowance",
        "energy_overflow",
        "force_overflow",
        "allowance_overflow",
    ),
)
def test_invalid_inputs_leave_receipt_atomic(library: Any, mutation: str) -> None:
    descriptor = context()
    forces = (ctypes.c_double * 6)(*range(6))
    reference = (ctypes.c_double * 6)()
    output = _Comparison(ctypes.sizeof(_Comparison), _native.ABI_VERSION, 0xA5A5)
    if mutation == "empty_mask":
        descriptor.requested_mask = 0
    elif mutation == "unknown_mask":
        descriptor.requested_mask = 8
    elif mutation == "reserved":
        descriptor.reserved = 1
    elif mutation == "context_size":
        descriptor.struct_size = 0
    elif mutation == "context_version":
        descriptor.abi_version += 1
    elif mutation == "output_size":
        output.struct_size = 0
    elif mutation == "output_version":
        output.abi_version += 1
    elif mutation == "zero_atoms":
        descriptor.atom_count = 0
    elif mutation == "overflow_atoms":
        descriptor.atom_count = 2**64 - 1
    elif mutation == "negative_absolute":
        descriptor.absolute[0] = -1
    elif mutation == "nonfinite_relative":
        descriptor.relative[0] = math.nan
    elif mutation == "zero_allowance":
        descriptor.absolute[0] = descriptor.relative[0] = 0
    elif mutation == "energy_overflow":
        descriptor.energy, descriptor.reference_energy = 1e308, -1e308
    elif mutation == "force_overflow":
        forces[0], reference[0] = 1e308, -1e308
    elif mutation == "allowance_overflow":
        descriptor.relative[0], descriptor.reference_energy = 1e308, 1e308
    before = bytes(output)
    status = query(library)(
        ctypes.byref(descriptor), forces, reference, ctypes.byref(output)
    )
    expected = (
        _native.STATUS_ABI_MISMATCH
        if "size" in mutation or "version" in mutation
        else _native.STATUS_INVALID_ARGUMENT
    )
    assert status == expected and bytes(output) == before


def test_null_and_unselected_arrays_fail_without_touching_storage(library: Any) -> None:
    descriptor = context(1)
    output = _Comparison(ctypes.sizeof(_Comparison), _native.ABI_VERSION)
    array = (ctypes.c_double * 3)()
    before = bytes(output)
    for arguments in (
        (None, None, None, ctypes.byref(output)),
        (ctypes.byref(descriptor), None, None, None),
        (ctypes.byref(descriptor), array, None, ctypes.byref(output)),
        (ctypes.byref(descriptor), None, array, ctypes.byref(output)),
    ):
        assert query(library)(*arguments) == _native.STATUS_INVALID_ARGUMENT
        assert bytes(output) == before


def test_overlap_copies_context_and_finishes_array_reads_before_publication(
    library: Any,
) -> None:
    storage = (ctypes.c_ubyte * 256)()
    descriptor = _Context.from_buffer(storage)
    ctypes.memmove(
        ctypes.byref(descriptor), ctypes.byref(context(1)), ctypes.sizeof(_Context)
    )
    output = _Comparison.from_buffer(storage, 24)
    output.struct_size, output.abi_version = (
        ctypes.sizeof(_Comparison),
        _native.ABI_VERSION,
    )
    expected = abs(descriptor.energy - descriptor.reference_energy)
    assert (
        query(library)(ctypes.byref(descriptor), None, None, ctypes.byref(output)) == 0
    )
    assert output.differences[0] == expected
    output = _Comparison.from_buffer(storage)
    output.struct_size, output.abi_version = (
        ctypes.sizeof(_Comparison),
        _native.ABI_VERSION,
    )
    forces = (ctypes.c_double * 3).from_buffer(storage, 16)
    forces[:] = (1, 2, 3)
    descriptor = context(6)
    descriptor.atom_count = 1
    reference = (ctypes.c_double * 3)()
    assert (
        query(library)(
            ctypes.byref(descriptor), forces, reference, ctypes.byref(output)
        )
        == 0
    )
    assert output.differences[1] == 3
    assert abs(output.differences[2] - rms_oracle((1, 2, 3))) <= math.ulp(
        output.differences[2]
    )


def test_missing_refused_and_inconsistent_selected_owners_do_not_fallback(
    library: Any,
) -> None:
    contract = TargetAccuracy((ObservableTarget("energy", "absolute", "Eh", 1),))
    for selected, error in (
        (SimpleNamespace(), NotImplementedError),
        (
            SimpleNamespace(
                generativeqc_observable_comparison_query_v1=lambda *args: 1
            ),
            ValueError,
        ),
        (
            SimpleNamespace(
                generativeqc_observable_comparison_query_v1=lambda *args: 0
            ),
            RuntimeError,
        ),
    ):
        with pytest.raises(error):
            measure_observables(
                contract, {"energy": 0}, {"energy": 0}, library=selected
            )
    real = query(library)

    def forged(descriptor: Any, forces: Any, reference: Any, output: Any) -> int:
        status = real(descriptor, forces, reference, output)
        ctypes.cast(output, ctypes.POINTER(_Comparison)).contents.met_mask = 0
        return status

    with pytest.raises(RuntimeError, match="invalid numerical evidence"):
        measure_observables(
            contract,
            {"energy": 0},
            {"energy": 0},
            library=SimpleNamespace(generativeqc_observable_comparison_query_v1=forged),
        )


def test_selected_owner_cannot_rewrite_the_request_to_forge_coverage() -> None:
    def forged(context: Any, forces: Any, reference: Any, output: Any) -> int:
        ctypes.cast(context, ctypes.POINTER(_Context)).contents.requested_mask = 0
        receipt = ctypes.cast(output, ctypes.POINTER(_Comparison)).contents
        receipt.policy_version = 1
        return 0

    with pytest.raises(RuntimeError, match="inconsistent receipt"):
        measure_observables(
            target(),
            {"energy": 1, "forces": np.ones((1, 3))},
            {"energy": 1, "forces": np.zeros((1, 3))},
            library=SimpleNamespace(generativeqc_observable_comparison_query_v1=forged),
        )


def test_strided_big_endian_and_parallel_inputs_are_detached(library: Any) -> None:
    values = np.arange(18, dtype=">f8").reshape(6, 3)[::2]
    original = values.copy()

    def execute(index: int) -> Any:
        return measure_observables(
            target(),
            {"energy": 1, "forces": values},
            {"energy": 1, "forces": values * 0},
            library=library,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(execute, range(16)))
    assert all(result == results[0] for result in results)
    np.testing.assert_array_equal(values, original)


def test_public_cpp_contract(required_native_cxx: Any, tmp_path: Path) -> None:
    binary = required_native_cxx.build_executable(
        [
            ROOT / "tests/native/test_observable_comparison.cpp",
            ROOT / "src/api/c_api_precision_admission.cpp",
        ],
        tmp_path / "comparison",
        compile_args=("-std=c++20", f"-I{ROOT / 'include'}", f"-I{ROOT / 'src'}"),
    )
    subprocess.run([str(binary)], check=True, timeout=30)
