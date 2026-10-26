"""Fault actual resource constructors without changing allocation or cleanup code."""

import ctypes
import typing

from generativeqc import _native


def limit_device_admission(
    monkeypatch: typing.Any,
    library: typing.Any,
    capacity: int,
    admission: str,
) -> None:
    """Replace only the input capacity of the selected native constructor.

    Prepared native owners retain their ledger independently of Python metadata,
    so replacing a metadata handle after preparation cannot inject a failure.
    Compatibility cases must explicitly disable the public prepared entry point
    before preparing the calculation; they never stand in for public-path tests.
    """
    if admission == "public_native":
        create = library.generativeqc_device_resources_create

        def limited_public(descriptor: typing.Any, output: typing.Any) -> int:
            buffers = (_native.DeviceResourceBuffer * 1)(
                _native.DeviceResourceBuffer(capacity, 4, 0, 0, 0, 0)
            )
            limited = _native.DeviceResourceDescriptor(
                ctypes.sizeof(_native.DeviceResourceDescriptor),
                _native.ABI_VERSION,
                buffers,
                len(buffers),
            )
            return create(ctypes.byref(limited), output)

        monkeypatch.setattr(
            library, "generativeqc_device_resources_create", limited_public
        )
    elif admission == "native_inventory":
        create = library.generativeqc_resource_ledger_create_from_buffers_v1
        create.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        create.restype = ctypes.c_int

        def limited_inventory(
            buffers: typing.Any,
            count: int,
            handle: typing.Any,
            receipt: typing.Any,
            receipt_count: int,
        ) -> int:
            limited = (ctypes.c_uint64 * 6)(capacity, 4, 0, 0, 0, 0)
            return create(limited, len(limited), handle, receipt, receipt_count)

        monkeypatch.setattr(
            library,
            "generativeqc_resource_ledger_create_from_buffers_v1",
            limited_inventory,
        )
    elif admission == "legacy":
        create = library.generativeqc_resource_ledger_create_v1
        create.argtypes = [ctypes.c_size_t, ctypes.c_int]
        create.restype = ctypes.c_void_p
        monkeypatch.setattr(
            library, "generativeqc_resource_ledger_create_from_buffers_v1", None
        )
        monkeypatch.setattr(
            library,
            "generativeqc_resource_ledger_create_v1",
            lambda requested, device: create(capacity, device),
        )
    else:
        raise ValueError("unknown resource admission test mode")
