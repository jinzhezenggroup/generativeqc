"""Borrow exact-identity compiled programs from an already selected library."""

from __future__ import annotations

import ctypes
import typing

from . import _native


def select_catalog_plan(
    library: typing.Any, method: typing.Any, backend: int, descriptor: typing.Any
) -> bool:
    """Select only a complete matching compiled graph, never a named substitute.

    A successful zero selection explicitly leaves custom/noncatalog graphs with
    the compiler. Missing-symbol compatibility is independent of that route.
    Query failure, mutated unselected output or invalid publication is final;
    no native rejection may retry Python lowering or library discovery.
    """
    query = getattr(library, "generativeqc_ks_catalog_plan_v1", None)
    if query is None:
        return False
    identity = method.identity.encode("ascii")
    prefix = _native.KsOptionsDescriptor.from_buffer(descriptor)
    before = bytes(descriptor)
    expected_domain = prefix.scf_domain
    selected = ctypes.c_uint32(2)
    query.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(_native.KsOptionsDescriptor),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    query.restype = ctypes.c_int
    _native.check(
        library,
        query(
            None,
            0,
            identity,
            len(identity),
            backend,
            ctypes.byref(prefix),
            ctypes.byref(selected),
        ),
    )
    if selected.value == 0:
        if bytes(descriptor) != before:
            raise RuntimeError(
                "unselected native KS catalog plan changed caller controls"
            )
        return False
    after = bytes(descriptor)
    domain_offset = _native.KsOptionsDescriptor.scf_domain.offset
    controls_start = _native.KsOptionsDescriptor.grid_version.offset
    controls_end = _native.KsOptionsDescriptor.spin_channels.offset
    extension_offset = ctypes.sizeof(_native.KsOptionsDescriptor)
    budget_offset = _native.KsOptionsDescriptor.nonlocal_maximum_bytes.offset
    # The catalog owns only domain/scientific fields, not execution controls or
    # size-tagged application extensions beyond the stable descriptor prefix.
    if (
        after[:domain_offset] != before[:domain_offset]
        or after[controls_start:controls_end] != before[controls_start:controls_end]
        or after[extension_offset:] != before[extension_offset:]
        or after[budget_offset:extension_offset]
        != before[budget_offset:extension_offset]
    ):
        raise RuntimeError("native KS catalog changed caller execution controls")
    from generativeqc_compiler.method import NonlocalCorrelationPrimitive

    has_nonlocal = any(
        isinstance(node, NonlocalCorrelationPrimitive) for node in method.primitives
    )
    if (
        selected.value != 1
        or prefix.abi_version != _native.ABI_VERSION
        or prefix.scf_domain != expected_domain
        or prefix.spin_channels != (1 if method.spin == "unpolarized" else 2)
        or not prefix.semilocal_component_count
        or not prefix.semilocal_components
        or (prefix.exchange_term_count and not prefix.exchange_terms)
        or prefix.has_nonlocal_correlation != int(has_nonlocal)
    ):
        raise RuntimeError(
            "native KS catalog returned an inconsistent compiled program"
        )
    descriptor._catalog_library_owner = library
    return True
