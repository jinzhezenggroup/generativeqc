"""Bounded canonical UMP2 energy on validated unrestricted references."""

from __future__ import annotations

import math
import threading
import typing
from dataclasses import dataclass
from itertools import product

import numpy as np
from generativeqc_compiler.mp2.equations import (
    cpu_capacity,
    unrestricted_energy_program,
)
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.types import checked_size

from tools.generativeqc_posthf.conventions import SpinMOBlock
from tools.generativeqc_posthf.providers import ConventionalProvider
from tools.generativeqc_response.uhf import UHFReferenceSnapshot


@dataclass(frozen=True)
class UMP2EnergyResult:
    """Published scalar UMP2 result; amplitudes and forces are intentionally absent."""

    energy: float
    correlation_energy: float
    alpha_alpha: float
    beta_beta: float
    alpha_beta: float
    minimum_absolute_denominator: float
    reference_id: str
    hamiltonian_id: str
    tile_count: int
    numeric_capacity_bytes: int
    equation_hashes: tuple[str, ...]
    integral_backend: str
    scalar_fold_backend: str = "cpu-compensated-sum"


def _spin_spaces(snapshot: UHFReferenceSnapshot, spin: str) -> tuple[range, range]:
    occupied = snapshot.nocc(spin)
    return range(occupied), range(occupied, snapshot.nmo)


def denominator_check_ump2(snapshot: typing.Any, threshold: typing.Any) -> float:
    """Check every active spin-pair denominator before any integral source read."""
    if not isinstance(snapshot, UHFReferenceSnapshot) or snapshot.algorithm != "UHF":
        raise TypeError("UMP2 requires a validated canonical UHFReferenceSnapshot")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("denominator threshold must be finite and positive")

    minima: list[float] = []
    for channel, left, right in (
        ("alpha_alpha", "alpha", "alpha"),
        ("beta_beta", "beta", "beta"),
        ("alpha_beta", "alpha", "beta"),
    ):
        left_occ, left_vir = _spin_spaces(snapshot, left)
        right_occ, right_vir = _spin_spaces(snapshot, right)
        if not left_occ or not right_occ or not left_vir or not right_vir:
            continue
        ei = getattr(snapshot, f"orbital_energies_{left}")
        ej = getattr(snapshot, f"orbital_energies_{right}")
        with np.errstate(over="raise", invalid="raise"):
            try:
                largest = (
                    np.max(ei[list(left_occ)])
                    + np.max(ej[list(right_occ)])
                    - np.min(ei[list(left_vir)])
                    - np.min(ej[list(right_vir)])
                )
                smallest = (
                    np.min(ei[list(left_occ)])
                    + np.min(ej[list(right_occ)])
                    - np.max(ei[list(left_vir)])
                    - np.max(ej[list(right_vir)])
                )
            except FloatingPointError as exc:
                raise ValueError(
                    f"nonfinite UMP2 denominator extrema in {channel}"
                ) from exc
        if not all(np.isfinite(value) for value in (largest, smallest)):
            raise ValueError(f"nonfinite UMP2 denominator extrema in {channel}")
        if largest >= 0:
            raise ValueError(
                f"UMP2 {channel} requires occupied energies strictly below virtual energies"
            )
        if -largest <= threshold:
            raise ValueError(
                f"near-zero UMP2 denominator in {channel}: {largest}; "
                "no regularization applied"
            )
        minima.append(float(-largest))
    return min(minima) if minima else 0.0


def _tile_sizes(total: int, requested: int) -> set[int]:
    if total == 0:
        return set()
    return {min(total, requested), total % requested or min(total, requested)}


class PreparedUMP2Energy:
    """Bounded conventional CPU UMP2 using spin-labelled TensorIR tile equations.

    The UHF reference and integral source are borrowed. Integral blocks are
    transformed with the alpha/beta coefficient matrix attached to each
    chemists-ERI slot; every retained block is cleared before the next request.
    This first slice is deliberately conventional CPU FP64 only.
    """

    def __init__(
        self,
        snapshot: typing.Any,
        source: typing.Any,
        *,
        occupied_tile: typing.Any = 1,
        virtual_tile: typing.Any = 2,
        axis_tile: typing.Any = 2,
        budget_bytes: typing.Any = 256 << 20,
        denominator_threshold: typing.Any = 1e-10,
    ) -> None:
        if not isinstance(snapshot, UHFReferenceSnapshot):
            raise TypeError("UMP2 requires a validated UHFReferenceSnapshot")
        if snapshot.algorithm != "UHF":
            raise ValueError(
                "UMP2 accepts UHF references only; UKS is not an MP2 reference"
            )
        for name, value in (
            ("occupied_tile", occupied_tile),
            ("virtual_tile", virtual_tile),
            ("budget_bytes", budget_bytes),
        ):
            checked_size(value, name)
            if value < 1:
                raise ValueError(f"{name} must be positive")
        self._minimum = denominator_check_ump2(snapshot, denominator_threshold)
        self._snapshot, self._source = snapshot, source
        self._source_identity = source.identity
        self._lock = threading.RLock()
        self._provider = ConventionalProvider(
            snapshot,
            source,
            budget_bytes=budget_bytes,
            axis_tile=axis_tile,
            backend="cpu",
        )
        self._ot, self._vt = occupied_tile, virtual_tile
        self._programs: dict[tuple[str, tuple[int, int, int, int]], typing.Any] = {}
        # Even a reference with no active excitation channels must validate the
        # source and charge its resident buffers through the shared planner.
        empty = SpinMOBlock(((),) * 4, ("alpha",) * 4)
        self._capacity = self._provider.plan(empty).peak_bytes
        self._state, self._last_result = "prepared", None

        for channel, left, right in (
            ("alpha_alpha", "alpha", "alpha"),
            ("beta_beta", "beta", "beta"),
            ("alpha_beta", "alpha", "beta"),
        ):
            no_left, no_right = snapshot.nocc(left), snapshot.nocc(right)
            nv_left, nv_right = snapshot.nmo - no_left, snapshot.nmo - no_right
            if not no_left or not no_right or not nv_left or not nv_right:
                continue
            size_sets = (
                _tile_sizes(no_left, occupied_tile),
                _tile_sizes(no_right, occupied_tile),
                _tile_sizes(nv_left, virtual_tile),
                _tile_sizes(nv_right, virtual_tile),
            )
            for shape in product(*size_sets):
                ni, nj, na, nb = shape
                i = tuple(range(ni))
                a = tuple(range(no_left, no_left + na))
                j = tuple(range(nj))
                b = tuple(range(no_right, no_right + nb))
                direct = SpinMOBlock((i, a, j, b), (left, left, right, right))
                plans = [self._provider.plan(direct)]
                if left == right:
                    plans.append(
                        self._provider.plan(
                            SpinMOBlock((i, b, j, a), (left, left, right, right))
                        )
                    )
                provider_peak = max(plan.peak_bytes for plan in plans)
                program = unrestricted_energy_program(shape, channel=channel)
                self._programs[(channel, shape)] = program
                feeds = 32 * math.prod(shape) + 8 * sum(shape) + 64
                needed = checked_size(
                    provider_peak + feeds + cpu_capacity(program),
                    "UMP2 numeric capacity",
                )
                self._capacity = max(self._capacity, needed)
        if self._capacity > budget_bytes:
            raise MemoryError(
                f"UMP2 needs {self._capacity} numeric bytes, budget is {budget_bytes}"
            )

    @property
    def state(self) -> str:
        return self._state

    @property
    def last_result(self) -> UMP2EnergyResult | None:
        return self._last_result

    @property
    def numeric_capacity_bytes(self) -> int:
        return self._capacity

    def _blocks(self) -> typing.Iterator[tuple[str, str, str, SpinMOBlock]]:
        for channel, left, right in (
            ("alpha_alpha", "alpha", "alpha"),
            ("beta_beta", "beta", "beta"),
            ("alpha_beta", "alpha", "beta"),
        ):
            no_left, no_right = self._snapshot.nocc(left), self._snapshot.nocc(right)
            for i0, j0, a0, b0 in product(
                range(0, no_left, self._ot),
                range(0, no_right, self._ot),
                range(no_left, self._snapshot.nmo, self._vt),
                range(no_right, self._snapshot.nmo, self._vt),
            ):
                i = tuple(range(i0, min(i0 + self._ot, no_left)))
                j = tuple(range(j0, min(j0 + self._ot, no_right)))
                a = tuple(range(a0, min(a0 + self._vt, self._snapshot.nmo)))
                b = tuple(range(b0, min(b0 + self._vt, self._snapshot.nmo)))
                yield (
                    channel,
                    left,
                    right,
                    SpinMOBlock((i, a, j, b), (left, left, right, right)),
                )

    def _read(self, block: SpinMOBlock) -> np.ndarray:
        result = self._provider.get(block)
        if (
            result.reference_id != self._snapshot.identity
            or result.hamiltonian_id != self._snapshot.hamiltonian_id
        ):
            raise ValueError("UMP2 block reference/Hamiltonian mismatch")
        values = result.to_host()
        self._provider.clear()
        return values

    def execute(self, *, properties: typing.Any = ("energy",)) -> UMP2EnergyResult:
        """Evaluate all active spin channels; failures never publish partial energy."""
        with self._lock:
            if self._state == "closed":
                raise RuntimeError("UMP2 plan is closed")
            self._last_result = None
            self._state = "running"
            try:
                if tuple(properties) != ("energy",):
                    raise NotImplementedError(
                        "UMP2 supports energy only; forces/amplitudes are unavailable"
                    )
                if self._source.identity != self._source_identity:
                    raise ValueError(
                        "UMP2 source changed; prepare a fresh reference/provider"
                    )
                self._source._check_open()
                totals = {"alpha_alpha": 0.0, "beta_beta": 0.0, "alpha_beta": 0.0}
                corrections = dict.fromkeys(totals, 0.0)
                hashes: set[str] = set()
                count = 0
                for channel, left, right, block in self._blocks():
                    i, a, j, b = block.slots
                    shape = (len(i), len(j), len(a), len(b))
                    if not all(shape):
                        continue
                    g = self._read(block).transpose(0, 2, 1, 3)
                    feeds: dict[str, typing.Any] = {"g": g}
                    if left == right:
                        exchange = SpinMOBlock((i, b, j, a), (left, left, right, right))
                        feeds["x"] = self._read(exchange).transpose(0, 2, 3, 1)
                    left_eps = getattr(self._snapshot, f"orbital_energies_{left}")
                    right_eps = getattr(self._snapshot, f"orbital_energies_{right}")
                    feeds.update(
                        {
                            "ei": left_eps[list(i)],
                            "ej": right_eps[list(j)],
                            "ea": left_eps[list(a)],
                            "eb": right_eps[list(b)],
                        }
                    )
                    program = self._programs[(channel, shape)]
                    hashes.add(program.logical_hash)
                    values = execute(
                        program, feeds, max_bytes=cpu_capacity(program)
                    ).outputs
                    value = float(values["energy"])
                    if not math.isfinite(value):
                        raise ValueError("nonfinite UMP2 tile energy")
                    adjusted = value - corrections[channel]
                    updated = totals[channel] + adjusted
                    corrections[channel] = (updated - totals[channel]) - adjusted
                    totals[channel] = updated
                    count += 1
                correlation = math.fsum(totals.values())
                energy = self._snapshot.reference_energy + correlation
                if not all(
                    math.isfinite(v) for v in (*totals.values(), correlation, energy)
                ):
                    raise ValueError("nonfinite UMP2 accumulated energy")
                result = UMP2EnergyResult(
                    energy=energy,
                    correlation_energy=correlation,
                    alpha_alpha=totals["alpha_alpha"],
                    beta_beta=totals["beta_beta"],
                    alpha_beta=totals["alpha_beta"],
                    minimum_absolute_denominator=self._minimum,
                    reference_id=self._snapshot.identity,
                    hamiltonian_id=self._snapshot.hamiltonian_id,
                    tile_count=count,
                    numeric_capacity_bytes=self._capacity,
                    equation_hashes=tuple(sorted(hashes)),
                    integral_backend="cpu-staged-fp64",
                )
            except BaseException:
                self._state = "failed"
                raise
            finally:
                self._provider.clear()
            self._last_result = result
            self._state = "ready"
            return result

    def close(self) -> None:
        with self._lock:
            self._provider.close()
            self._last_result = None
            self._state = "closed"

    def __enter__(self) -> typing.Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
