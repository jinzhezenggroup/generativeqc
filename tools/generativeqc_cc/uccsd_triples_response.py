"""Bounded complete unrestricted (T) cotangents on the parent CPU/CUDA backend.

Only gathering, compensated scatter and excitation-coordinate maps are host
owned. Every energy and derivative contraction uses generic TensorIR AD and
the borrowed native executor. These fixed-amplitude bars are not corrected
Lambda, orbital-relaxed densities or nuclear forces.
"""

from __future__ import annotations

import time
from dataclasses import InitVar, dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES, UCCSDLayout
from generativeqc_compiler.cc.uccsd_triples import (
    TriplesTileProgram,
    build_tile_response_program,
)
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.resources import checked_bytes

from .uccsd_lambda import _freeze
from .uccsd_t import PreparedUCCSDT, UCCSDTResult

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class UCCSDTriplesResponseResult:
    """Detached full-Frobenius source bars, or no partial response on failure.

    ``packed_amplitude_cotangent`` is U* times the five full amplitude bars;
    same-spin orbit members are summed with signs, never divided by four.
    eps_o/eps_v bars retain their actual spin owners and physical denominators.
    They require an explicit Fock-diagonal chain in a later physical consumer.
    """

    energy: UCCSDTResult
    cotangents: Mapping[str, np.ndarray]
    packed_amplitude_cotangent: np.ndarray | None
    response_identity: str | None
    provenance: Mapping[str, Any]
    _endpoint_started: InitVar[float | None] = None
    _preparation_seconds: InitVar[float] = 0.0

    def __post_init__(
        self, _endpoint_started: float | None, _preparation_seconds: float
    ) -> None:
        if not isinstance(self.energy, UCCSDTResult):
            raise TypeError("triples response requires a complete UCCSD(T) result")
        if not self.energy.converged:
            if (
                self.cotangents
                or self.packed_amplitude_cotangent is not None
                or self.response_identity is not None
            ):
                raise ValueError("failed triples response cannot publish partial bars")
        else:
            if not set(BLOCK_NAMES).issubset(self.cotangents) or not {
                "eps_o_a",
                "eps_v_a",
                "eps_o_b",
                "eps_v_b",
            }.issubset(self.cotangents):
                raise ValueError("triples response lacks amplitudes or denominators")
            for values in (*self.cotangents.values(), self.packed_amplitude_cotangent):
                if (
                    type(values) is not np.ndarray
                    or values.dtype != np.float64
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("triples response requires finite base FP64 bars")
            singles_a, singles_b = self.cotangents["t1a"], self.cotangents["t1b"]
            if singles_a.ndim != 2 or singles_b.ndim != 2:
                raise ValueError("triples amplitude singles must have rank two")
            layout = UCCSDLayout(
                singles_a.shape[0],
                singles_b.shape[0],
                singles_a.shape[1],
                singles_b.shape[1],
            )
            packed = layout.pack_dual(
                {name: self.cotangents[name] for name in BLOCK_NAMES}
            )
            if not np.array_equal(packed, self.packed_amplitude_cotangent):
                raise ValueError("triples response full and packed cotangents disagree")
            if (
                not isinstance(self.response_identity, str)
                or not self.response_identity
            ):
                raise ValueError("triples response requires a response identity")
            object.__setattr__(self, "packed_amplitude_cotangent", immutable(packed))
        object.__setattr__(
            self,
            "cotangents",
            MappingProxyType(
                {name: immutable(value) for name, value in self.cotangents.items()}
            ),
        )
        provenance = dict(self.provenance)
        if _endpoint_started is not None:
            provenance["triples_response_endpoint_seconds"] = (
                _preparation_seconds + time.perf_counter() - _endpoint_started
            )
        object.__setattr__(self, "provenance", _freeze(provenance))

    @property
    def converged(self) -> bool:
        """Whether the fresh parent and all energy/response tiles succeeded."""
        return self.energy.converged

    def hamiltonian_cotangents(self) -> Mapping[str, np.ndarray]:
        """Chain canonical orbital-energy bars to each spin Fock diagonal.

        Return all independent parent Fock/integral bars, excluding amplitudes
        and the now-bound eps operands. This is a coordinate pullback at the
        canonical reference, not orbital relaxation or a normal-ordering/RDM
        pullback. Both occupied and virtual denominator contributions are
        added to existing Fock bars without a restricted occupancy factor.
        """
        if not self.converged:
            raise ValueError("failed triples response has no Hamiltonian cotangents")
        weights = {
            name: value.copy()
            for name, value in self.cotangents.items()
            if name not in BLOCK_NAMES and not name.startswith("eps_")
        }
        with np.errstate(over="raise", invalid="raise"):
            for spin in "ab":
                for domain in "ov":
                    name = f"f{domain * 2}_{spin * 2}"
                    energies = self.cotangents[f"eps_{domain}_{spin}"]
                    if energies.ndim != 1:
                        raise ValueError(
                            "triples response orbital-energy bars must have rank one"
                        )
                    if name not in weights and not np.any(energies):
                        # Empty excitation domains can prune unused Fock inputs;
                        # their zero denominator bars need no invented block.
                        continue
                    if name not in weights or weights[name].shape != (
                        energies.size,
                        energies.size,
                    ):
                        raise ValueError(
                            "triples response Fock/denominator populations disagree"
                        )
                    positions = np.arange(energies.size)
                    weights[name][positions, positions] += energies
                    if not np.isfinite(weights[name]).all():
                        raise FloatingPointError(
                            "nonfinite triples Fock-diagonal pullback"
                        )
        return MappingProxyType(
            {name: immutable(value) for name, value in weights.items()}
        )


class PreparedUCCSDTriplesResponse(PreparedUCCSDT):
    """Use the energy owner's admission, lifetime and complete tile schedule.

    This owner borrows the live parent/executor and copies its converged state.
    Provider closure is allowed because canonical MO feeds are parent-owned.
    Preparation admits full source bars, compensation/publication storage and
    bounded scatter scratch in addition to all derivative graph execution.
    Logical numeric bounds are not process RSS or arbitrary compiler caches.
    """

    def _extra_storage_bytes(self) -> int:
        full_bytes = sum(
            block.full_layout.storage_bytes(8) for block in self.layout.blocks
        )
        feed_bytes = sum(value.nbytes for value in self.prepared.feeds.values())
        denominator_bytes = 16 * self.prepared.reference.nmo
        largest = max(
            (value.nbytes for value in self.prepared.feeds.values()), default=0
        )
        largest = max(largest, full_bytes, denominator_bytes)
        return checked_bytes(
            3 * (feed_bytes + full_bytes + denominator_bytes)
            + 6 * largest
            + 32 * self.layout.packed_size,
            "UCCSD(T) cotangent/compensation/publication/scatter bytes",
        )

    def _build_tile(self, channel: str, shape: tuple[int, ...]) -> TriplesTileProgram:
        return build_tile_response_program(
            self.layout, channel, shape, max_elements=self.options.max_bytes // 8
        )

    def _bind(
        self, tile: TriplesTileProgram, coordinates: tuple[tuple[int, ...], ...]
    ) -> dict:
        return {**super()._bind(tile, coordinates), "bar_triples_energy": np.array(1.0)}

    def response(self) -> UCCSDTriplesResponseResult:
        """Publish complete finite bars only after fresh physical CCSD replay.

        Reversing a binding is transpose plus scatter, not a new scientific
        equation. Repeated source coordinates across permutations, tiles and
        spin sectors accumulate independently with FP64 compensation. No
        fixed-owner mixed-spin pair-exchange symmetry is assumed.
        """
        self._check_open()
        started = time.perf_counter()
        cotangents = {
            name: np.zeros_like(value) for name, value in self.sources.items()
        }
        compensation = {
            name: np.zeros_like(value) for name, value in self.sources.items()
        }

        def accumulate(
            tile: TriplesTileProgram,
            coordinates: tuple[tuple[int, ...], ...],
            outputs: Mapping,
        ) -> None:
            positions = dict(zip("ijkabc", coordinates, strict=True))
            with np.errstate(over="raise", invalid="raise"):
                for binding in tile.operands:
                    target = cotangents[binding.source].transpose(binding.axes)
                    errors = compensation[binding.source].transpose(binding.axes)
                    selected = np.ix_(
                        *(
                            positions.get(label, tuple(range(size)))
                            for label, size in zip(
                                binding.labels, target.shape, strict=True
                            )
                        )
                    )
                    values = np.asarray(outputs[f"bar_{binding.name}"])
                    expected_shape = tuple(
                        len(positions[label]) if label in positions else size
                        for label, size in zip(
                            binding.labels, target.shape, strict=True
                        )
                    )
                    if values.dtype != np.float64 or values.shape != expected_shape:
                        raise ValueError(
                            "UCCSD(T) executor violated a tile cotangent contract"
                        )
                    if not np.isfinite(values).all():
                        raise FloatingPointError("nonfinite UCCSD(T) tile cotangent")
                    previous = target[selected]
                    correction = values - errors[selected]
                    updated = previous + correction
                    errors[selected] = (updated - previous) - correction
                    target[selected] = updated
                    if (
                        not np.isfinite(updated).all()
                        or not np.isfinite(errors[selected]).all()
                    ):
                        raise FloatingPointError(
                            "nonfinite UCCSD(T) cotangent accumulation"
                        )

        energy = self._evaluate_tiles(accumulate)
        packed = identity = None
        if energy.converged:
            packed = self.layout.pack_dual(
                {name: cotangents[name] for name in BLOCK_NAMES}
            )
            identity = canonical_hash(
                {
                    "state_identity": self.state_identity,
                    "template_identities": energy.provenance["template_identities"],
                    "triples_energy": energy.triples_energy,
                    "cotangents": {
                        name: sha256(
                            value.astype("<f8", copy=False).tobytes()
                        ).hexdigest()
                        for name, value in sorted(cotangents.items())
                    },
                }
            )
        else:
            cotangents = {}
        self._check_open()
        return UCCSDTriplesResponseResult(
            energy,
            cotangents,
            packed,
            identity,
            {
                **energy.provenance,
                "response_identity": identity,
                "cotangent_backend": "host-fp64-compensated-binding-scatter",
                "response_contract": "fixed-amplitude (T) bars; not corrected Lambda or nuclear forces",
                "denominator_chain": "raw eps_o/eps_v retained; hamiltonian_cotangents() explicitly binds spin Fock diagonals",
            },
            _endpoint_started=started,
            _preparation_seconds=self.preparation_seconds,
        )
