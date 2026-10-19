"""Element-local FP64 descriptor adapters; native code owns atom/contraction expansion."""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import _native
from .elements import checked_integer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._api_types import Atom
    from .basis import BasisSet


def _count(value: int) -> int:
    return checked_integer(value, "basis template count/offset", high=2**32 - 1)


@dataclass(frozen=True)
class NativeBasisTemplate:
    """Own arrays borrowed synchronously by the v1 element-template constructor.

    The caller first admits the immutable snapshot with require_basis. Only the
    unique elements needed by this geometry are marshaled: unrelated retained
    metadata may intentionally describe unsupported domains. No per-atom shell
    or ECP term array is built here; the native owner maps rows and local channels.
    """

    descriptor: _native.BasisTemplateDescriptor
    has_ecp: bool
    _keepalive: tuple[Any, ...]

    @classmethod
    def from_basis(cls, basis: BasisSet, atoms: Sequence[Atom]) -> NativeBasisTemplate:
        """Marshal admitted element templates without normalization or substitution."""
        selected = {atom.atomic_number for atom in atoms}
        elements, shells, exponents, coefficients, terms = [], [], [], [], []
        for element in basis.elements:
            if element.atomic_number not in selected:
                continue
            shell_offset = _count(len(shells))
            for shell in element.shells:
                exponent_offset = _count(len(exponents))
                coefficient_offset = _count(len(coefficients))
                exponents.extend(float(value) for value in shell.exponents)
                coefficients.extend(
                    float(value) for row in shell.coefficients for value in row
                )
                shells.append(
                    _native.BasisShellTemplate(
                        shell.angular_momentum,
                        exponent_offset,
                        _count(len(shell.exponents)),
                        coefficient_offset,
                        _count(len(shell.coefficients)),
                    )
                )
            term_offset = _count(len(terms))
            if element.ecp_core_electrons:
                for potential in json.loads(element.ecp_data):
                    if len(potential["coefficients"]) != 1:
                        raise NotImplementedError(
                            "ECP requires exactly one scalar coefficient row"
                        )
                    for power, exponent, coefficient in zip(
                        potential["r_exponents"],
                        potential["gaussian_exponents"],
                        potential["coefficients"][0],
                        strict=True,
                    ):
                        terms.append(
                            _native.BasisEcpTemplate(
                                potential["angular_momentum"][0],
                                power,
                                float(exponent),
                                float(coefficient),
                            )
                        )
            elements.append(
                _native.BasisElementTemplate(
                    element.atomic_number,
                    element.ecp_core_electrons,
                    shell_offset,
                    _count(len(shells) - shell_offset),
                    term_offset,
                    _count(len(terms) - term_offset),
                )
            )
        element_array = (_native.BasisElementTemplate * _count(len(elements)))(
            *elements
        )
        shell_array = (_native.BasisShellTemplate * _count(len(shells)))(*shells)
        exponent_array = (ctypes.c_double * _count(len(exponents)))(*exponents)
        coefficient_array = (ctypes.c_double * _count(len(coefficients)))(*coefficients)
        term_array = (_native.BasisEcpTemplate * _count(len(terms)))(*terms)
        descriptor = _native.BasisTemplateDescriptor(
            ctypes.sizeof(_native.BasisTemplateDescriptor),
            _native.ABI_VERSION,
            element_array,
            len(element_array),
            shell_array,
            len(shell_array),
            exponent_array,
            len(exponent_array),
            coefficient_array,
            len(coefficient_array),
            term_array if terms else None,
            len(term_array),
        )
        return cls(
            descriptor,
            bool(terms),
            (element_array, shell_array, exponent_array, coefficient_array, term_array),
        )
