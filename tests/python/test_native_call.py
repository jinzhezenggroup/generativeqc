"""Status-specific ctypes errors preserve native allocation/error distinctions."""

import typing

import pytest
from generativeqc_compiler.common.native_call import checked_native_call


@pytest.mark.parametrize("status,error_type", [(1, RuntimeError), (7, MemoryError)])
def test_status_mapping_does_not_hide_other_native_errors(
    status: int,
    error_type: type[Exception],
) -> None:
    def native(*args: typing.Any) -> int:
        args[-2].value = b"native diagnostic"
        return status

    with pytest.raises(error_type, match="native diagnostic"):
        checked_native_call(native, status_error_types={7: MemoryError})


def test_unmapped_allocation_status_preserves_legacy_exception_type() -> None:
    def native(*args: typing.Any) -> int:
        return 7

    with pytest.raises(RuntimeError):
        checked_native_call(native)
