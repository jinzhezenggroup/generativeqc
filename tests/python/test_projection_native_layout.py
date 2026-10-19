"""Bounded regression controls; no SCF, timing campaign, or GPU work."""

import ctypes
import typing

import numpy as np
import pytest
from generativeqc import _projection_native, projection


def unaligned(value: typing.Any) -> typing.Any:
    array = np.asarray(value, dtype=np.float64)
    result = np.ndarray(
        array.shape, dtype=np.float64, buffer=bytearray(array.nbytes + 1), offset=1
    )
    result[:] = array
    assert result.flags.c_contiguous and not result.flags.aligned
    return result


@pytest.mark.parametrize("position", range(4))
@pytest.mark.parametrize("density_input", [False, True])
def test_unaligned_native_boundary(
    monkeypatch: typing.Any, position: typing.Any, density_input: typing.Any
) -> None:
    arguments = [
        np.eye(2),
        np.eye(2),
        np.eye(2),
        np.eye(2) * (2 if density_input else 1),
    ]
    arguments[position] = unaligned(arguments[position])
    baseline = (
        projection._project_density_python(*arguments, occupied_orbitals=2)
        if density_input
        else projection._project_occupied_python(*arguments)
    )
    calls = []

    class FakeLibrary:
        def generativeqc_basis_projection_query_v1(
            self, descriptor: typing.Any, capacity: typing.Any
        ) -> typing.Any:
            value = capacity._obj
            value.coefficient_count = 4
            value.density_count = 4
            value.numeric_workspace_bytes = 3072
            return 0

        def generativeqc_context_create(
            self, descriptor: typing.Any, context: typing.Any
        ) -> typing.Any:
            context._obj.value = 1
            return 0

        def generativeqc_basis_project_cpu_v1(
            self, context: typing.Any, descriptor: typing.Any, output: typing.Any
        ) -> typing.Any:
            descriptor, output = descriptor._obj, output._obj
            for name in (
                "source_overlap",
                "target_overlap",
                "cross_overlap",
                "source_state",
            ):
                address = ctypes.cast(getattr(descriptor, name), ctypes.c_void_p).value
                assert address % ctypes.alignment(ctypes.c_double) == 0, name
            np.ctypeslib.as_array(output.coefficients, (4,))[:] = np.eye(2).ravel()
            np.ctypeslib.as_array(output.density, (4,))[:] = (2 * np.eye(2)).ravel()
            calls.append("operation")
            return 0

        def generativeqc_context_destroy(self, context: typing.Any) -> typing.Any:
            calls.append("destroy")

    monkeypatch.setattr(_projection_native, "optional_library", lambda: FakeLibrary())
    result = (
        projection.project_density(*arguments, occupied_orbitals=2)
        if density_input
        else projection.project_occupied(*arguments)
    )
    np.testing.assert_array_equal(result.density, baseline.density)
    assert result.execution_owner == "native_cpu"
    assert calls == ["operation", "destroy"]


@pytest.mark.parametrize(
    "layout", ["fortran", "strided", "big_endian", "float32", "integer", "read_only"]
)
def test_numeric_layout_conversion_preserves_inputs(
    monkeypatch: typing.Any, layout: typing.Any
) -> None:
    class CaptureLibrary:
        def generativeqc_basis_projection_query_v1(
            self, descriptor: typing.Any, capacity: typing.Any
        ) -> typing.Any:
            value = descriptor._obj
            for name, count in (
                ("source_overlap", 4),
                ("target_overlap", 4),
                ("cross_overlap", 4),
                ("source_state", 4),
            ):
                pointer = getattr(value, name)
                assert ctypes.cast(pointer, ctypes.c_void_p).value % 8 == 0
                np.testing.assert_array_equal(
                    np.ctypeslib.as_array(pointer, (count,)), np.eye(2).ravel()
                )
            raise RuntimeError("marshalling checked")

    data = np.eye(2)
    if layout == "fortran":
        data = np.asfortranarray(data)
    elif layout == "strided":
        data = np.tile(data, (2, 2))[::2, ::2]
        data[:] = np.eye(2)
    elif layout == "big_endian":
        data = data.astype(">f8")
    elif layout == "float32":
        data = data.astype(np.float32)
    elif layout == "integer":
        data = data.astype(int)
    else:
        data.setflags(write=False)
    before = data.tobytes()
    monkeypatch.setattr(
        _projection_native, "optional_library", lambda: CaptureLibrary()
    )
    with pytest.raises(RuntimeError, match="marshalling checked"):
        projection.project_occupied(data, data, data, data)
    assert data.tobytes() == before
