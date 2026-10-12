"""Bind the explicit unpolarized work Graph to the native total-density ABI."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from generativeqc_compiler.common.provenance import canonical_hash

from .bulk_aot import SourceVariant
from .density_work import DENSITY_WORK_DOMAIN, DENSITY_WORK_DOMAIN_VERSION

if TYPE_CHECKING:
    from .bulk_runtime import BulkRuntimeProgram

RESTRICTED_BINDING_SCHEMA = "generativeqc.libxc-restricted-point-program-binding/v1"


@dataclass(frozen=True)
class RestrictedSemilocalPointBinding:
    """Immutable first-order total-density binding, not endpoint admission.

    This ABI is distinct from the polarized descriptor. A restricted consumer
    must evaluate original total density before any work-coordinate projection;
    folding first and reusing a polarized floor is not equivalent in the active
    density tail. Only the currently qualified density-only ingredient layout
    enters this producer.
    """

    variant: SourceVariant
    capability_identity: str
    point_expression_identity: str
    density_threshold: float

    def __post_init__(self) -> None:
        if not isinstance(self.variant, SourceVariant):
            raise TypeError("restricted point binding requires SourceVariant")
        if (
            self.variant.backend != "cpu"
            or self.variant.spin != "unpolarized"
            or self.variant.features != ("rho",)
            or self.variant.derivative_order != 1
            or self.variant.domain != DENSITY_WORK_DOMAIN
        ):
            raise ValueError(
                "restricted point binding requires the density-only work ABI"
            )
        for identity in (self.capability_identity, self.point_expression_identity):
            if not isinstance(identity, str) or not identity:
                raise ValueError(
                    "restricted point binding requires complete identities"
                )
        if (
            isinstance(self.density_threshold, bool)
            or not math.isfinite(self.density_threshold)
            or self.density_threshold <= 0.0
        ):
            raise ValueError(
                "restricted point binding requires a positive density threshold"
            )

    @property
    def identity(self) -> str:
        """Bind total-density normalization, exact source and pinned worker inputs."""
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        """Return the complete restricted execution contract without admission."""
        return {
            "schema": RESTRICTED_BINDING_SCHEMA,
            "name": self.variant.name,
            "capability_identity": self.capability_identity,
            "point_expression_identity": self.point_expression_identity,
            "artifact_emission_identity": self.variant.emission_identity,
            "artifact_source_sha256": self.variant.source_sha256,
            "import_identity": self.variant.import_identity,
            "domain": self.variant.domain,
            "domain_version": DENSITY_WORK_DOMAIN_VERSION,
            "density_threshold": self.density_threshold,
            "features": list(self.variant.features),
            "outputs": [[], [0]],
            "ingredient_mask": 1,
            "normalization": "original-total-density-no-spin-fold",
        }

    def emit_source(self) -> str:
        """Emit an independent native restricted descriptor and complete E/vxc."""
        return (
            '#include <cmath>\n#include <stdexcept>\n#include "dft/xc.hpp"\n'
            + "namespace generativeqc::dft::bulk_generated::restricted_generated {\nnamespace {\n"
            + self.emit_declarations()
            + "}\n}\n"
        )

    def emit_declarations(self) -> str:
        """Emit declarations inside a consumer-owned private namespace.

        Registry shards contain multiple bindings. The consumer supplies a
        distinct namespace per registration without nesting headers or reopening
        a qualified namespace relative to its existing native registry scope.
        """
        quote = json.dumps
        return (
            self.variant.source
            + f"inline constexpr const char* kBindingIdentity = {quote(self.identity)};\n"
            + "inline RestrictedSemilocalPointValue evaluate_point(double rho,\n"
            + "    const double (&gradient)[3], double tau) {\n"
            + "  (void)gradient; (void)tau;\n"
            + "  if (!std::isfinite(rho) || rho < 0.0)\n"
            + '    throw std::invalid_argument("invalid restricted XC density");\n'
            + f"  if (rho == 0.0 || rho < {float(self.density_threshold).hex()}) return {{}};\n"
            + "  const double features[1]{rho};\n  double outputs[2]{};\n"
            + "  bulk_xc_point(features, outputs);\n"
            + "  for (double value : outputs)\n"
            + "    if (!std::isfinite(value))\n"
            + '      throw std::runtime_error("nonfinite restricted XC output");\n'
            + "  RestrictedSemilocalPointValue value{};\n"
            + "  value.energy = outputs[0]; value.rho = outputs[1];\n"
            + "  return value;\n}\n"
            + "inline constexpr RestrictedSemilocalPointProgram kPointProgram{\n"
            + f"    {quote(self.variant.name)}, {quote(self.point_expression_identity)}, "
            + f"1U, {DENSITY_WORK_DOMAIN_VERSION}U, evaluate_point}};\n"
        )


def bind_runtime_restricted_semilocal_point_program(
    program: BulkRuntimeProgram,
) -> RestrictedSemilocalPointBinding:
    """Bind an explicitly unpolarized density-work Graph, retaining its identity."""
    from .bulk_point_program import inspect_runtime_program

    if program.outputs != ((), (0,)):
        raise ValueError("restricted point binding requires complete E/vxc outputs")
    if program.spec.density_threshold is None:
        raise ValueError("restricted point binding requires density screening")
    return RestrictedSemilocalPointBinding(
        inspect_runtime_program(program, "cpu"),
        program.spec.capability_identity,
        program.expression_hash,
        program.spec.density_threshold,
    )


__all__ = [
    "RESTRICTED_BINDING_SCHEMA",
    "RestrictedSemilocalPointBinding",
    "bind_runtime_restricted_semilocal_point_program",
]
