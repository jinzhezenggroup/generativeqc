"""Versioned, representation-only semiempirical product requirements (#875).

This module resolves parameter and state *references* for a method DAG and
separates local derivative declarations from whole-method derivative closure.
It does not admit runtime, backend, analytic-force, or public capabilities.
The GFN v1 IR and legacy xTB identities are deliberately unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.method.semiempirical import (
    InvalidSemiempiricalMethod,
    ParameterResource,
    ProductSpec,
    SemiempiricalMethodIR,
    StateField,
)

SEMIEMPIRICAL_REQUIREMENTS_VERSION = "semiempirical-product-requirements-v1"


class DerivativeMechanism(str, Enum):
    """Family-neutral local operations, not whole-method qualification."""

    PRIMAL_ENERGY = "primal-energy"
    HAMILTONIAN = "hamiltonian"
    INTEGRAL_ADJOINT = "integral-adjoint"
    EXPLICIT_COORDINATE_VJP = "explicit-coordinate-vjp"
    STATE_ADJOINT = "state-adjoint"


# GFN's original names remain serialized in XtbMethodIR and the common v1
# PrimitiveNode. Mapping them *only here* avoids accidental cache migration.
_KNOWN_DECLARATIONS: dict[str, DerivativeMechanism] = {
    "energy": DerivativeMechanism.PRIMAL_ENERGY,
    "hamiltonian": DerivativeMechanism.HAMILTONIAN,
    "integral-adjoint": DerivativeMechanism.INTEGRAL_ADJOINT,
    "nuclear-gradient": DerivativeMechanism.EXPLICIT_COORDINATE_VJP,
    "explicit-coordinate-vjp": DerivativeMechanism.EXPLICIT_COORDINATE_VJP,
    "state-adjoint": DerivativeMechanism.STATE_ADJOINT,
}

_LOCAL_COORDINATE_ROUTES = frozenset(
    (
        DerivativeMechanism.EXPLICIT_COORDINATE_VJP,
        DerivativeMechanism.INTEGRAL_ADJOINT,
    )
)


@dataclass(frozen=True)
class PrimitiveRequirements:
    """Resolved immutable dependency and derivative requirements for one node."""

    node_id: str
    equation_identity: str
    depends_on: tuple[str, ...]
    produces: tuple[str, ...]
    parameter_resources: tuple[ParameterResource, ...]
    state_reads: tuple[StateField, ...]
    state_writes: tuple[StateField, ...]
    derivative_mechanisms: tuple[DerivativeMechanism, ...]
    fixed_point: bool

    def to_payload(self) -> dict[str, object]:
        """Serialize bound node requirements deterministically."""
        return {
            "node_id": self.node_id,
            "equation_identity": self.equation_identity,
            "depends_on": self.depends_on,
            "produces": self.produces,
            "parameter_resources": tuple(
                resource.to_payload() for resource in self.parameter_resources
            ),
            "state_reads": tuple(field.to_payload() for field in self.state_reads),
            "state_writes": tuple(field.to_payload() for field in self.state_writes),
            "derivative_mechanisms": tuple(
                mechanism.value for mechanism in self.derivative_mechanisms
            ),
            "fixed_point": self.fixed_point,
        }


@dataclass(frozen=True)
class SemiempiricalProductRequirements:
    """Canonical scientific obligations, never a runtime-admission certificate.

    Local mechanism declarations cannot prove a *total* stationary gradient:
    differentiating the integrals still requires the appropriate contractions,
    state response, and a qualified final force assembly.
    """

    method_identity: str
    product: ProductSpec
    nodes: tuple[PrimitiveRequirements, ...]
    supported_atomic_numbers: tuple[int, ...]
    unresolved_local_derivative_nodes: tuple[str, ...]
    unresolved_global_obligations: tuple[str, ...]
    version: str = SEMIEMPIRICAL_REQUIREMENTS_VERSION

    @property
    def has_closed_derivative_contract(self) -> bool:
        """Static obligations only; never implies public/runtime qualification."""
        return not (
            self.unresolved_local_derivative_nodes or self.unresolved_global_obligations
        )

    def require_closed_derivative_contract(self) -> None:
        """Fail closed when the method graph alone cannot establish closure."""
        if not self.has_closed_derivative_contract:
            raise InvalidSemiempiricalMethod(
                "semiempirical product has unresolved derivative obligations: "
                f"local={self.unresolved_local_derivative_nodes!r}, "
                f"global={self.unresolved_global_obligations!r}"
            )

    def to_payload(self) -> dict[str, object]:
        """Serialize this derived obligation receipt; not runtime proof."""
        return {
            "version": self.version,
            "method_identity": self.method_identity,
            "product": self.product.to_payload(),
            "nodes": tuple(node.to_payload() for node in self.nodes),
            "supported_atomic_numbers": self.supported_atomic_numbers,
            "unresolved_local_derivative_nodes": self.unresolved_local_derivative_nodes,
            "unresolved_global_obligations": self.unresolved_global_obligations,
        }

    @property
    def identity(self) -> str:
        """Hash the immutable bound requirements and method provenance."""
        return canonical_hash(self.to_payload())


def semiempirical_product_requirements(
    method: SemiempiricalMethodIR, product: ProductSpec
) -> SemiempiricalProductRequirements:
    """Resolve typed references and conservative derivative obligations.

    Only two initial product semantics are understood. Future families may add
    typed products without changing the family-neutral primitive vocabulary,
    but unknown product semantics cannot be qualified implicitly.
    """
    if not isinstance(method, SemiempiricalMethodIR):
        raise TypeError("expected audited SemiempiricalMethodIR")
    if not isinstance(product, ProductSpec):
        raise TypeError("expected a typed ProductSpec")
    if product not in method.requested_products:
        raise InvalidSemiempiricalMethod(
            "product was not requested by the canonical semiempirical method"
        )
    key = (product.name, product.derivative_order)
    if key not in {("energy", 0), ("nuclear-gradient", 1)}:
        raise InvalidSemiempiricalMethod(
            f"unsupported semiempirical product contract {key!r}"
        )
    if key == ("nuclear-gradient", 1) and ProductSpec("energy", 0) not in (
        method.requested_products
    ):
        raise InvalidSemiempiricalMethod(
            "nuclear-gradient contract also requires the total energy product"
        )

    resources = {resource.role: resource for resource in method.parameter_set.resources}
    states = {field.name: field for field in method.state_fields}
    state_writers: dict[str, str] = {}
    requirements: list[PrimitiveRequirements] = []
    for node in method.primitives:
        for name in node.state_writes:
            previous = state_writers.setdefault(name, node.node_id)
            if previous != node.node_id:
                raise InvalidSemiempiricalMethod(
                    f"ambiguous writer for state {name!r}: "
                    f"{previous!r} and {node.node_id!r}"
                )
        try:
            mechanisms = tuple(
                sorted(
                    {
                        _KNOWN_DECLARATIONS[name]
                        for name in node.derivative_capabilities
                    },
                    key=lambda mechanism: mechanism.value,
                )
            )
        except KeyError as error:
            raise InvalidSemiempiricalMethod(
                f"{node.node_id} has an unrecognized derivative declaration "
                f"{error.args[0]!r}"
            ) from error
        requirements.append(
            PrimitiveRequirements(
                node_id=node.node_id,
                equation_identity=node.equation_identity,
                depends_on=node.requires,
                produces=node.produces,
                parameter_resources=tuple(
                    resources[name] for name in node.parameter_resources
                ),
                state_reads=tuple(states[name] for name in node.state_reads),
                state_writes=tuple(states[name] for name in node.state_writes),
                derivative_mechanisms=mechanisms,
                fixed_point=node.fixed_point,
            )
        )

    local: tuple[str, ...] = ()
    global_obligations: tuple[str, ...] = ()
    if key == ("nuclear-gradient", 1):
        # No implicit gradient of a node without an explicitly declared route.
        # Such a node might be coordinate-independent, but only an independent
        # scientific owner can prove that. No guess is safe at the IR boundary.
        local = tuple(
            node.node_id
            for node in requirements
            if not _LOCAL_COORDINATE_ROUTES.intersection(node.derivative_mechanisms)
        )
        obligations: list[str] = []
        if any(
            node.fixed_point or node.state_reads or node.state_writes
            for node in requirements
        ):
            obligations.append("stationary-state-response")
        # Even with every local VJP available, graph annotations cannot prove
        # force assembly / Pulay / density and state-adjoint completeness.
        obligations.append("total-gradient-assembly")
        global_obligations = tuple(obligations)

    return SemiempiricalProductRequirements(
        method_identity=method.identity,
        product=product,
        nodes=tuple(requirements),
        supported_atomic_numbers=method.parameter_set.supported_atomic_numbers,
        unresolved_local_derivative_nodes=local,
        unresolved_global_obligations=global_obligations,
    )
