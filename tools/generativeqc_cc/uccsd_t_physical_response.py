"""Corrected physical UCCSD(T) weights through shared normal ordering and UHF.

This is a distinct method owner, not a bypass of standard CCSD qualification.
Both fresh triples endpoints are charged: corrected Lambda qualification and
the complete same-space Fock replay. Full general-frame metric transport is
applied before the shared orbital solve and again before its all-rotation gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from generativeqc_compiler.cc.uccsd_hamiltonian import (
    build_hamiltonian_cotangent_program,
)
from generativeqc_compiler.common.resources import checked_bytes

from tools.generativeqc_response.implicit import ImplicitSolveError

from .uccsd_physical_response import (
    UCCSDPhysicalResponseOptions,
    _PreparedUCCSDPhysicalResponse,
    _UCCSDRelaxedCotangents,
)
from .uccsd_t_lambda import PreparedUCCSDTLambda, UCCSDTLambdaResult
from .uccsd_triples_fock_response import PreparedUCCSDTriplesFockResponse

if TYPE_CHECKING:
    from tools.generativeqc_posthf.providers import ConventionalProvider

    from .uccsd_physical_response import _RelaxedArguments


@dataclass(frozen=True)
class UCCSDTRelaxedCotangents(_UCCSDRelaxedCotangents):
    """Detached corrected raw-MO/overlap weights and freshly replayed energies."""

    ccsd_correlation_energy: float
    triples_energy: float
    correlation_energy: float
    total_energy: float
    triples_response_identity: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            any(
                not math.isfinite(value)
                for value in (
                    self.ccsd_correlation_energy,
                    self.triples_energy,
                    self.correlation_energy,
                    self.total_energy,
                )
            )
            or self.correlation_energy
            != self.ccsd_correlation_energy + self.triples_energy
            or not isinstance(self.triples_response_identity, str)
            or not self.triples_response_identity
            or self.provenance.get("method") != "UCCSD(T)"
        ):
            raise ValueError("inconsistent corrected UCCSD(T) relaxed energies/binding")


class PreparedUCCSDTPhysicalResponse(_PreparedUCCSDPhysicalResponse):
    """Borrow exact-state corrected Lambda/Fock owners and the physical provider.

    Native Hamiltonian VJP, full-frame overlap transport, UHF operator/GMRES,
    generated Fock-JVP comparison and all-rotation stationarity are shared with
    standard CCSD. Closing this owner does not close any borrowed resource.
    """

    def __init__(
        self,
        lambdas: PreparedUCCSDTLambda,
        result: UCCSDTLambdaResult,
        fock: PreparedUCCSDTriplesFockResponse,
        provider: ConventionalProvider,
        *,
        options: UCCSDPhysicalResponseOptions | None = None,
        response_backend: Any = None,
        solver: Any = None,
        nuclear_repulsion: float | None = None,
    ) -> None:
        if not isinstance(fock, PreparedUCCSDTriplesFockResponse):
            raise TypeError("corrected physical response requires a full Fock owner")
        fock._check_open()
        self.fock = fock
        super().__init__(
            lambdas,
            result,
            provider,
            options=options,
            response_backend=response_backend,
            solver=solver,
            nuclear_repulsion=nuclear_repulsion,
        )

    def _validate_lambda(self, lambdas: Any, result: Any) -> None:
        if not isinstance(lambdas, PreparedUCCSDTLambda):
            raise TypeError("corrected physical response requires corrected Lambda")
        lambdas._check_open()
        if (
            not isinstance(result, UCCSDTLambdaResult)
            or result.state_identity != lambdas.state_identity
            or result.correction_identity != lambdas.correction_identity
        ):
            raise ValueError(
                "corrected physical response requires matching Lambda state"
            )
        if (
            self.fock.triples is not lambdas.triples
            or self.fock.prepared is not lambdas.prepared
            or self.fock.state_identity != lambdas.state_identity
        ):
            raise ValueError("corrected physical owners differ in triples/parent/state")

    def _additional_numeric_bytes(self) -> int:
        source_bytes = sum(value.nbytes for value in self.fock.triples.sources.values())
        return checked_bytes(
            self.fock.peak_bytes + 4 * source_bytes + 4 * self.raw_bytes,
            "corrected physical simultaneous replay/bar/transport storage",
        )

    def _additional_programs(self) -> tuple:
        self.correction_program = build_hamiltonian_cotangent_program(
            self.lambdas.prepared.layout, self.hamiltonian
        )
        return (self.correction_program,)

    def _additional_scalar_work(self) -> int:
        return checked_bytes(
            self.fock.total_scalar_work
            + self._resources[self.correction_program.logical_hash].scalar_work,
            "corrected physical full Fock replay and raw transport work",
        )

    def _additional_binding(self) -> tuple:
        return self.fock, self.correction_program

    def _assert_current(self) -> None:
        self.fock._check_open()
        super()._assert_current()

    def _qualify_lambda(self) -> tuple:
        packed, correction, diagnostics = self.lambdas.qualify_with_correction(
            self.result
        )
        return packed, diagnostics, correction

    def _correction_weights(self, fresh: Any, counter: dict) -> dict:
        complete = self.fock.response()
        if not complete.converged:
            raise ImplicitSolveError(
                f"corrected physical full Fock replay failed: {complete.triples_response.energy.reason}"
            )
        replay = complete.triples_response
        for name in (
            "reference_id",
            "hamiltonian_id",
            "integral_hash",
            "state_identity",
            "template_identities",
            "tensor_backend",
            "integral_backend",
            "source_backend",
        ):
            if fresh.energy.provenance.get(name) != replay.energy.provenance.get(name):
                raise ValueError("corrected physical fresh triples bindings differ")
        if (
            fresh.cotangents.keys() != replay.cotangents.keys()
            or fresh.energy.channel_energies.keys()
            != replay.energy.channel_energies.keys()
        ):
            raise ValueError("corrected physical fresh triples inventories differ")
        differences = []
        for name in (
            "ccsd_correlation_energy",
            "triples_energy",
            "correlation_energy",
            "total_energy",
        ):
            first, second = getattr(fresh.energy, name), getattr(replay.energy, name)
            if first is None or second is None:
                raise ValueError(
                    "corrected physical fresh triples energies are incomplete"
                )
            differences.append(abs(first - second))
        differences.extend(
            abs(value - replay.energy.channel_energies[name])
            for name, value in fresh.energy.channel_energies.items()
        )
        for name, value in fresh.cotangents.items():
            other = replay.cotangents[name]
            if value.shape != other.shape:
                raise ValueError("corrected physical fresh triples populations differ")
            differences.append(float(np.max(np.abs(value - other), initial=0)))
        drift = max(differences, default=0.0)
        if not math.isfinite(drift) or drift > self.lambdas.limits.correction_tolerance:
            raise ImplicitSolveError(
                "corrected physical fresh triples energy/bar drift failed"
            )
        counter["work"] += complete.provenance["executed_semantic_scalar_work"]
        counter["native_calls"] += (
            complete.provenance["fock_native_calls"]
            + replay.provenance["completed_tiles"]
            + 1
        )
        if counter["work"] > self.options.max_total_work:
            raise MemoryError("corrected physical replay work budget exhausted")
        bars = complete.hamiltonian_cotangents()
        if not bars.keys() <= set(self.hamiltonian.parameter_names):
            raise ValueError("corrected physical Hamiltonian bars have unknown inputs")
        seeds = {
            f"bar_{name}": bars.get(
                name, np.zeros(self.hamiltonian.primal.outputs[name].spec.shape)
            )
            for name in self.hamiltonian.parameter_names
        }
        weights = self._run(
            self.correction_program,
            {**seeds, "bar_reference_electronic_energy": np.array(0.0)},
            counter,
        )
        return {"weights": weights, "fock": complete, "fresh_replay_drift": drift}

    def _additional_identity(self, correction: Any) -> dict:
        return {
            "method": "UCCSD(T)",
            "full_fock_binding": self.fock.binding_identity,
            "correction_transport": self.correction_program.logical_hash,
        }

    def _additional_provenance(self, correction: Any) -> dict:
        energy = correction["fock"].triples_response.energy
        return {
            "method": "UCCSD(T)",
            "ccsd_correlation_energy": energy.ccsd_correlation_energy,
            "triples_energy": energy.triples_energy,
            "correlation_energy": energy.correlation_energy,
            "total_energy": energy.total_energy,
            "full_fock_qualification": correction["fock"].provenance,
            "fresh_triples_replay_drift": correction["fresh_replay_drift"],
            "correction_raw_transport": self.correction_program.logical_hash,
        }

    def _make_result(
        self, arguments: _RelaxedArguments, correction: Any
    ) -> UCCSDTRelaxedCotangents:
        energy = correction["fock"].triples_response.energy
        return UCCSDTRelaxedCotangents(
            *arguments,
            energy.ccsd_correlation_energy,
            energy.triples_energy,
            energy.correlation_energy,
            energy.total_energy,
            correction["fock"].response_identity,
        )

    def solve(self) -> UCCSDTRelaxedCotangents:
        """Freshly qualify both corrections and complete physical stationarity."""
        return cast("UCCSDTRelaxedCotangents", super().solve())
