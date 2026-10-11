"""Standard eager behavior must not imply unsupported scientific compilation."""

from __future__ import annotations

import operator

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp


def test_standard_discovery_is_concrete_cpu_only() -> None:
    array = xp.asarray([1, 2, 3])
    assert isinstance(array, xp.Array)
    assert array.__array_namespace__() is xp
    assert array.__array_namespace__(api_version="2025.12") is xp
    with pytest.raises(ValueError, match="version"):
        array.__array_namespace__(api_version="2099.12")
    info = xp.__array_namespace_info__()
    assert array.device == info.default_device() == "cpu"
    assert tuple(info.devices()) == ("cpu",)
    assert info.capabilities()["data-dependent shapes"] is True
    assert array.to_device("cpu") is array
    with pytest.raises(ValueError, match="cpu"):
        array.to_device("cuda:0")
    with pytest.raises(ValueError, match="stream"):
        array.to_device("cpu", stream=1)
    missing_dtype = None
    assert xp.float64 != missing_dtype


@pytest.mark.parametrize("dtype", xp.capabilities()["eager_dtypes"])
def test_every_standard_dtype_survives_creation_indexing_and_manipulation(
    dtype: str,
) -> None:
    kind = getattr(xp, dtype)
    array = xp.asarray([[1, 0], [0, 1]], dtype=kind)
    for result in (
        array,
        array[0, 0],
        array.T,
        array.mT,
        xp.reshape(array, (4,)),
        xp.flip(array),
    ):
        assert isinstance(result, xp.Array)
        assert result.dtype == kind
        assert result.__array_namespace__() is xp


def test_scalar_conversions_and_matrix_transposes_are_rank_checked() -> None:
    assert operator.index(xp.asarray(3)) == 3
    assert float(xp.asarray(1.5)) == 1.5
    assert complex(xp.asarray(1 + 2j)) == 1 + 2j
    assert bool(xp.asarray(True))
    for convert in (bool, int, float, complex, operator.index):
        with pytest.raises(TypeError, match="zero-dimensional"):
            convert(xp.asarray([1]))
    with pytest.raises(TypeError, match="integer"):
        operator.index(xp.asarray(1.0))
    with pytest.raises(ValueError, match="two-dimensional"):
        operator.attrgetter("T")(xp.ones((2,)))
    with pytest.raises(ValueError, match="two dimensions"):
        operator.attrgetter("mT")(xp.ones((2,)))


def test_copy_and_dlpack_do_not_hide_transfers_or_allocations() -> None:
    array = xp.asarray([1.0, 2.0])
    assert np.shares_memory(xp.asarray(array, copy=False), array)
    assert not np.shares_memory(xp.asarray(array, copy=True), array)
    assert xp.astype(array, xp.float64, copy=False) is array
    assert not np.shares_memory(xp.astype(array, xp.float64), array)
    imported = xp.from_dlpack(array, copy=False)
    assert imported.__array_namespace__() is xp
    assert np.shares_memory(imported, array)
    with pytest.raises(ValueError):
        xp.asarray([1, 2], copy=False)


@pytest.mark.parametrize("operation", (xp.floor_divide, operator.floordiv))
def test_floor_division_uses_ieee_infinity_and_signed_zero_limits(
    operation: object,
) -> None:
    first = xp.asarray([xp.inf, -xp.inf, 1.0, -1.0])
    second = xp.asarray([-1.0, 1.0, -xp.inf, xp.inf])
    result = operation(first, second)
    assert float(result[0]) == -xp.inf
    assert float(result[1]) == -xp.inf
    assert float(result[2]) == float(result[3]) == 0
    assert bool(xp.all(xp.signbit(result[2:])))


def test_complex_ieee_limits_are_standard_not_numpy_accidents() -> None:
    array = xp.asarray([complex(xp.inf, 0), complex(-xp.inf, xp.inf), complex(-0.0, 0)])
    result = xp.expm1(array)
    assert complex(result[0]) == complex(xp.inf, 0)
    assert complex(result[1]) == complex(-1, 0)
    assert not bool(xp.signbit(xp.real(result[2])))
    result = xp.tanh(xp.asarray(complex(xp.inf, 2653237258533096.0)))
    assert complex(result) == complex(1, 0)
    assert not bool(xp.signbit(xp.imag(result)))
    with pytest.raises(TypeError, match="real or imag"):
        xp.astype(array, xp.float64)
    result = xp.acosh(xp.asarray(complex(0.0, xp.nan)))
    assert bool(xp.isnan(xp.real(result)))
    assert abs(float(xp.imag(result))) == xp.pi / 2


def test_data_dependent_results_and_extensions_keep_array_protocols() -> None:
    array = xp.asarray([3, 1, 3, 2])
    unique = xp.unique_all(array)
    for field in unique:
        assert field.__array_namespace__() is xp
    np.testing.assert_array_equal(unique.values, [1, 2, 3])
    np.testing.assert_array_equal(xp.nonzero(array > 1)[0], [0, 2, 3])
    np.testing.assert_array_equal(xp.where(array > 1, array, 0), [3, 0, 3, 2])
    matrix = xp.eye(2)
    assert xp.linalg.svd(matrix).S.__array_namespace__() is xp
    assert xp.fft.fft(xp.ones((4,))).__array_namespace__() is xp


@pytest.mark.parametrize("operation", (xp.sin, xp.linalg.svd, xp.fft.fft, xp.nonzero))
def test_unsupported_capture_never_executes_an_eager_backend(operation: object) -> None:
    def expression(array: object) -> object:
        return operation(array)

    with pytest.raises(NotImplementedError, match="TensorIR capture"):
        xp.compile(expression).lower(np.ones((2, 2)))


def test_nested_shared_containers_are_not_mistaken_for_cycles() -> None:
    shared = [1.0, 2.0]
    np.testing.assert_array_equal(xp.asarray([shared, shared]), [[1, 2], [1, 2]])
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic"):
        xp.asarray(cyclic)


def test_operators_cannot_bypass_foreign_device_guard() -> None:
    class Foreign:
        def __dlpack_device__(self) -> tuple[int, int]:
            return (2, 0)

        def __array__(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("implicit transfer must not execute")

    with pytest.raises(TypeError, match="explicit handoff"):
        xp.ones((2,)) + Foreign()
    array = xp.ones((2,))
    with pytest.raises(TypeError, match="explicit handoff"):
        array[0] = Foreign()
    with pytest.raises(TypeError, match="explicit handoff"):
        array[Foreign()]


def test_host_ndarray_subclasses_cannot_redirect_eager_execution() -> None:
    class HostArray(np.ndarray):
        def __array_ufunc__(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("standard calls must use explicit CPU storage")

    array = np.asarray([1.0, 2.0]).view(HostArray)
    np.testing.assert_array_equal(xp.add(array, 1), [2.0, 3.0])


@pytest.mark.parametrize("dtype", (xp.float64, xp.int64, xp.bool))
@pytest.mark.parametrize("protocol", ("dlpack", "cuda"))
@pytest.mark.parametrize("copy", (None, True, False))
def test_host_array_protocol_result_rejects_hidden_foreign_leaves(
    dtype: object, protocol: str, copy: bool | None
) -> None:
    class Foreign:
        def __float__(self) -> float:
            raise AssertionError("implicit foreign scalar conversion must not execute")

        def __int__(self) -> int:
            raise AssertionError("implicit foreign scalar conversion must not execute")

        def __bool__(self) -> bool:
            raise AssertionError("implicit foreign scalar conversion must not execute")

    if protocol == "dlpack":
        Foreign.__dlpack_device__ = lambda self: (2, 0)
    else:
        Foreign.__cuda_array_interface__ = {}

    calls: list[bool | None] = []

    class HostArrayLike:
        def __array__(
            self, dtype: object = None, copy: bool | None = None
        ) -> np.ndarray:
            calls.append(copy)
            array = np.empty((1,), dtype=object)
            array[0] = Foreign()
            return array

    with pytest.raises(TypeError, match="explicit handoff"):
        xp.asarray(HostArrayLike(), dtype=dtype, copy=copy)
    assert calls == [copy]


@pytest.mark.parametrize("source_dtype", (object, np.float64))
@pytest.mark.parametrize("copy", (None, True, False))
def test_host_array_protocol_result_preserves_single_call_and_copy_contract(
    source_dtype: object, copy: bool | None
) -> None:
    storage = np.asarray([1.0, 2.0], dtype=source_dtype)
    calls: list[bool | None] = []

    class HostArrayLike:
        def __array__(
            self, dtype: object = None, copy: bool | None = None
        ) -> np.ndarray:
            calls.append(copy)
            return storage.copy() if copy else storage

    if source_dtype is object and copy is False:
        with pytest.raises(ValueError):
            xp.asarray(HostArrayLike(), dtype=xp.float64, copy=copy)
    else:
        result = xp.asarray(HostArrayLike(), dtype=xp.float64, copy=copy)
        np.testing.assert_array_equal(result, [1.0, 2.0])
        assert np.shares_memory(result, storage) == (
            source_dtype is np.float64 and copy is not True
        )
    assert calls == [copy]


@pytest.mark.parametrize("indices", ([True], [0.5]))
def test_gathers_require_integer_indices(indices: list[object]) -> None:
    with pytest.raises(TypeError, match="integer dtype"):
        xp.take(xp.ones((2,)), xp.asarray(indices), axis=0)


def test_inplace_shape_and_dtype_failures_preserve_storage() -> None:
    array = xp.ones((2, 2), dtype=xp.float32)
    with pytest.raises(ValueError, match="shape"):
        array @= xp.ones((2,), dtype=xp.float32)
    with pytest.raises(TypeError, match="dtype"):
        array += xp.ones((2, 2), dtype=xp.float64)
    np.testing.assert_array_equal(array, np.ones((2, 2), dtype=np.float32))
