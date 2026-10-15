"""Optional named-basis system materialization, never offline metadata resolution."""

from __future__ import annotations

import ctypes
from typing import Any

from . import _native
from .basis import BasisSet
from .elements import checked_integer


class Descriptor(ctypes.Structure):
    """Borrowed named pack input and a raw-expansion-only admission limit."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("name_count", ctypes.c_uint64),
        ("atoms", ctypes.POINTER(_native.AtomDescriptor)),
        ("atom_count", ctypes.c_uint64),
        ("representation", ctypes.c_int32),
        ("charge", ctypes.c_int32),
        ("multiplicity", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("maximum_expansion_bytes", ctypes.c_uint64),
        ("expected_pack_checksum", ctypes.c_char_p),
        ("expected_pack_checksum_count", ctypes.c_uint64),
    ]


class Capacity(ctypes.Structure):
    """Raw result counts and copied ctypes views of immutable native provenance."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("shell_count", ctypes.c_uint64),
        ("primitive_count", ctypes.c_uint64),
        ("ao_count", ctypes.c_uint64),
        ("expansion_bytes", ctypes.c_uint64),
        ("canonical_name", ctypes.c_char_p),
        ("pack_checksum", ctypes.c_char_p),
    ]


def eligible(library: Any, basis: Any) -> bool:
    """Require the complete immutable built-in record, not a custom display name.

    Offline BasisSet APIs stay native-free. Old libraries and explicit custom
    records retain their existing validated-descriptor route. A discovered
    protocol's error is never interpreted as eligibility for a Python retry.
    """
    if (
        type(basis) is not BasisSet
        or basis.provenance.source
        != "generativeqc/data/basis_pack.json; Basis Set Exchange"
    ):
        return False
    from ._model_resolution import _named_basis_record

    try:
        expected = _named_basis_record(basis.name, basis.representation)
    except NotImplementedError:
        return False
    if basis.identity != expected.identity:
        return False
    return all(
        getattr(library, name, None) is not None
        for name in (
            "generativeqc_builtin_basis_query_v1",
            "generativeqc_system_create_builtin_basis_v1",
        )
    )


def create_system(
    library: Any,
    context: Any,
    basis: BasisSet,
    atoms: Any,
    charge: Any,
    multiplicity: Any,
    *,
    maximum_expansion_bytes: int = 0,
) -> ctypes.c_void_p:
    """Marshal atom metadata; native AOT expansion and existing normalization own data.

    The caller must validate consumer capabilities and eligibility first. Pack,
    ABI, admission and physical validation errors propagate without retry. The
    byte limit excludes normalized system/provider and caller storage.
    """
    charge = checked_integer(charge, "ionic charge", low=-(2**31), high=2**31 - 1)
    multiplicity = checked_integer(multiplicity, "multiplicity", low=1, high=2**31 - 1)
    checked_integer(len(atoms), "atom count", low=1, high=2**32 - 1)
    maximum_expansion_bytes = checked_integer(
        maximum_expansion_bytes, "maximum_expansion_bytes", high=2**64 - 1
    )
    atom_array = (_native.AtomDescriptor * len(atoms))(
        *(_native.AtomDescriptor(atom.atomic_number, *atom.position) for atom in atoms)
    )
    name = basis.name.encode("ascii")
    checksum = basis.provenance.checksum.encode("ascii")
    descriptor = Descriptor(
        struct_size=ctypes.sizeof(Descriptor),
        abi_version=_native.ABI_VERSION,
        name=name,
        name_count=len(name),
        atoms=atom_array,
        atom_count=len(atoms),
        representation=(
            _native.BASIS_SPHERICAL
            if basis.representation == "spherical"
            else _native.BASIS_CARTESIAN
        ),
        charge=charge,
        multiplicity=multiplicity,
        maximum_expansion_bytes=maximum_expansion_bytes,
        expected_pack_checksum=checksum,
        expected_pack_checksum_count=len(checksum),
    )
    query = library.generativeqc_builtin_basis_query_v1
    query.argtypes = [ctypes.POINTER(Descriptor), ctypes.POINTER(Capacity)]
    query.restype = ctypes.c_int
    capacity = Capacity(
        struct_size=ctypes.sizeof(Capacity), abi_version=_native.ABI_VERSION
    )
    _native.check(library, query(ctypes.byref(descriptor), ctypes.byref(capacity)))
    if (
        capacity.struct_size < ctypes.sizeof(Capacity)
        or capacity.abi_version != _native.ABI_VERSION
        or capacity.canonical_name != name
        or capacity.pack_checksum != checksum
    ):
        raise RuntimeError(
            "native builtin basis query returned inconsistent provenance"
        )
    create = library.generativeqc_system_create_builtin_basis_v1
    create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Descriptor),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    create.restype = ctypes.c_int
    system = ctypes.c_void_p()
    _native.check(
        library,
        create(context, ctypes.byref(descriptor), ctypes.byref(system)),
        context=context,
    )
    if not system.value:
        raise RuntimeError("native builtin basis creation returned no system")
    return system
