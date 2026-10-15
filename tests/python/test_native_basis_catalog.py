"""Built-in materialization ownership, full-record identity and explicit compatibility."""

import ctypes
import typing
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import Atom, Calculator, _basis_native, _native
from generativeqc._model_resolution import _named_basis_record
from generativeqc.basis import BasisSet


@pytest.fixture
def native_context() -> typing.Iterator[tuple[typing.Any, ctypes.c_void_p]]:
    library = _native.load_library(device="cpu")
    if not hasattr(library, "generativeqc_system_create_builtin_basis_v1"):
        pytest.skip("requires the installed native built-in basis protocol")
    descriptor = _native.ContextDescriptor(
        ctypes.sizeof(_native.ContextDescriptor), _native.ABI_VERSION, 0, 0
    )
    context = ctypes.c_void_p()
    _native.check(
        library,
        library.generativeqc_context_create(
            ctypes.byref(descriptor), ctypes.byref(context)
        ),
    )
    try:
        yield library, context
    finally:
        library.generativeqc_context_destroy(context)


def overlap(
    library: typing.Any, context: typing.Any, system: typing.Any, size: int
) -> np.ndarray:
    output = np.empty((size, size), dtype=np.float64)
    function = library.generativeqc_system_cross_overlap_cpu
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int
    _native.check(
        library,
        function(
            context,
            system,
            system,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            output.size,
        ),
        context=context,
    )
    return output


@pytest.mark.parametrize("name", ("sto-3g", "def2-svp", "def2-tzvp"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_native_system_avoids_python_primitive_materialization(
    native_context: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    representation: str,
) -> None:
    library, context = native_context
    calculator = Calculator(basis=name, basis_representation=representation)
    atoms = [
        Atom.from_value(value)
        for value in [("O", (0, 0, 0)), ("H", (0, -1.4, 1.1)), ("H", (0, 1.4, 1.1))]
    ]
    size = sum(
        2 * shell.angular_momentum + 1
        if representation == "spherical"
        else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
        for shell in calculator._basis.shells_for(atoms)
    )
    with monkeypatch.context() as compatibility:
        compatibility.setattr(_basis_native, "eligible", lambda *args: False)
        reference = calculator._create_native_system(context, atoms, 0, 1)
    try:
        expected = overlap(library, context, reference, size)
    finally:
        library.generativeqc_system_destroy(reference)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError(
            "native built-in materialization retried Python primitives"
        )

    monkeypatch.setattr(BasisSet, "shells_for", forbidden)
    actual = calculator._create_native_system(context, atoms, 0, 1)
    try:
        np.testing.assert_allclose(
            overlap(library, context, actual, size), expected, atol=2e-14, rtol=0
        )
    finally:
        library.generativeqc_system_destroy(actual)


def changed_record() -> BasisSet:
    original = _named_basis_record("sto-3g", "cartesian")
    first = original.elements[0]
    shell = first.shells[0]
    modified = replace(shell, exponents=("2", *shell.exponents[1:]))
    return replace(
        original,
        elements=(
            replace(first, shells=(modified, *first.shells[1:])),
            *original.elements[1:],
        ),
    )


def test_custom_same_name_never_selects_catalog(native_context: typing.Any) -> None:
    library, _ = native_context
    original = _named_basis_record("sto-3g", "cartesian")
    custom = changed_record()
    assert original.name == custom.name and original.identity != custom.identity
    assert _basis_native.eligible(library, original)
    assert not _basis_native.eligible(library, custom)


def test_old_protocol_keeps_explicit_compatibility() -> None:
    record = _named_basis_record("sto-3g", "cartesian")
    assert not _basis_native.eligible(SimpleNamespace(), record)
    assert not _basis_native.eligible(
        SimpleNamespace(generativeqc_builtin_basis_query_v1=lambda *args: 0), record
    )


@pytest.mark.parametrize("compatibility", ("custom-record", "old-library"))
def test_compatibility_preserves_actual_basis_data(
    native_context: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    compatibility: str,
) -> None:
    library, context = native_context
    atoms = [Atom.from_value(("H", (0, 0, distance))) for distance in (-0.7, 0.7)]
    original = _named_basis_record("sto-3g", "cartesian")
    calculator = Calculator(basis=original)
    reference = calculator._create_native_system(context, atoms, 0, 1)
    try:
        expected = overlap(library, context, reference, 2)
    finally:
        library.generativeqc_system_destroy(reference)

    class CompatibilityLibrary:
        def __getattr__(self, name: str) -> typing.Any:
            if name in {
                "generativeqc_builtin_basis_query_v1",
                "generativeqc_system_create_builtin_basis_v1",
            }:
                if compatibility == "old-library":
                    return None
                raise AssertionError("a custom record selected built-in table data")
            return getattr(library, name)

    if compatibility == "custom-record":
        calculator = Calculator(basis=changed_record())
    monkeypatch.setattr(calculator, "_library", CompatibilityLibrary())
    system = calculator._create_native_system(context, atoms, 0, 1)
    try:
        actual = overlap(library, context, system, 2)
        if compatibility == "custom-record":
            assert abs(actual[0, 1] - expected[0, 1]) > 1e-5
        else:
            np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    finally:
        library.generativeqc_system_destroy(system)


@pytest.mark.parametrize("publication", ("checksum", "abi-version", "canonical-name"))
def test_inconsistent_native_provenance_never_retries(
    native_context: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    publication: str,
) -> None:
    library, context = native_context
    calculator = Calculator(basis="sto-3g")

    class CorruptQuery:
        def __call__(self, descriptor: typing.Any, output: typing.Any) -> int:
            status = library.generativeqc_builtin_basis_query_v1(descriptor, output)
            capacity = ctypes.cast(
                output, ctypes.POINTER(_basis_native.Capacity)
            ).contents
            if publication == "checksum":
                capacity.pack_checksum = b"x" * 64
            elif publication == "abi-version":
                capacity.abi_version += 1
            else:
                capacity.canonical_name = b"def2-svp"
            return status

    class SelectedLibrary:
        generativeqc_builtin_basis_query_v1 = CorruptQuery()

        def __getattr__(self, name: str) -> typing.Any:
            return getattr(library, name)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("provenance refusal retried Python basis materialization")

    monkeypatch.setattr(calculator, "_library", SelectedLibrary())
    monkeypatch.setattr(BasisSet, "shells_for", forbidden)
    with pytest.raises(RuntimeError, match="inconsistent provenance"):
        calculator._create_native_system(
            context, [Atom.from_value(("H", (0, 0, 0)))], 0, 2
        )


def test_correlation_auxiliary_retains_scoped_bridge(
    native_context: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    library, context = native_context
    calculator = Calculator(basis="sto-3g")
    atoms = [Atom.from_value(("H", (0, 0, 0)))]
    auxiliary = _named_basis_record("sto-3g", "cartesian")
    monkeypatch.setattr(calculator, "_density_fitting_backend", lambda: "cuda")
    monkeypatch.setattr(
        calculator, "_density_fitting_mode", _native.DENSITY_FITTING_CORRELATION_CUDA
    )
    assert calculator._auxiliary_basis_consumer("df_metric") == "correlation_df"

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("correlation auxiliary used generic admission")

    class ScopedLibrary:
        def __getattr__(self, name: str) -> typing.Any:
            if name in {
                "generativeqc_builtin_basis_query_v1",
                "generativeqc_system_create_builtin_basis_v1",
                "generativeqc_system_create",
            }:
                return forbidden
            return getattr(library, name)

    monkeypatch.setattr(calculator, "_library", ScopedLibrary())
    system = calculator._create_native_system(context, atoms, 0, 2, basis=auxiliary)
    try:
        np.testing.assert_allclose(
            overlap(library, context, system, 1), [[1]], atol=2e-14, rtol=0
        )
    finally:
        library.generativeqc_system_destroy(system)


@pytest.mark.parametrize("stage", ("query", "create"))
def test_selected_native_failure_never_retries_python(
    native_context: typing.Any, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    library, context = native_context
    calculator = Calculator(basis="sto-3g")
    atoms = [Atom.from_value(("H", (0, 0, 0)))]

    class FailedFunction:
        def __call__(self, *args: typing.Any) -> int:
            return _native.STATUS_NUMERICAL_FAILURE

    class SelectedLibrary:
        def __getattr__(self, name: str) -> typing.Any:
            if (
                name
                == {
                    "query": "generativeqc_builtin_basis_query_v1",
                    "create": "generativeqc_system_create_builtin_basis_v1",
                }[stage]
            ):
                return FailedFunction()
            return getattr(library, name)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("selected native failure retried Python basis data")

    monkeypatch.setattr(calculator, "_library", SelectedLibrary())
    monkeypatch.setattr(BasisSet, "shells_for", forbidden)
    with pytest.raises(RuntimeError, match="GENERATIVEQC error 5"):
        calculator._create_native_system(context, atoms, 0, 2)


def test_native_budget_refusal_keeps_context_usable(native_context: typing.Any) -> None:
    library, context = native_context
    record = _named_basis_record("sto-3g", "cartesian")
    atoms = [Atom.from_value(("H", (0, 0, 0)))]
    with pytest.raises(RuntimeError, match="GENERATIVEQC error 7"):
        _basis_native.create_system(
            library, context, record, atoms, 0, 2, maximum_expansion_bytes=1
        )
    system = _basis_native.create_system(library, context, record, atoms, 0, 2)
    try:
        np.testing.assert_allclose(
            overlap(library, context, system, 1), [[1]], atol=2e-14, rtol=0
        )
    finally:
        library.generativeqc_system_destroy(system)


def test_offline_basis_stays_native_free(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _named_basis_record("sto-3g", "cartesian")

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("offline basis expansion loaded native execution")

    monkeypatch.setattr(_native, "load_library", forbidden)
    assert len(record.shells_for([Atom.from_value(("H", (0, 0, 0)))])) == 1
