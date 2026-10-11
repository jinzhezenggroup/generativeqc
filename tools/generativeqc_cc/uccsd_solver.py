"""Canonical UCCSD with one host-controlled CPU/CUDA iteration policy.

The shared TensorIR graph owns all scientific contractions. This conventional
owner borrows a validated UHF snapshot and spin-capable integral provider,
materializes owned spin-resolved MO feeds, and maintains packed excitation-state
DIIS coordinates. CUDA transforms/residuals never fall back to CPU evaluation;
host Fock transforms, checkpoints, denominators and DIIS are explicit parts of
this contract.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from heapq import nlargest, nsmallest
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from generativeqc_compiler.cc.uccsd_equations import build_program
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.resources import checked_bytes
from generativeqc_compiler.tensor.cpu import emit_cpu
from generativeqc_compiler.tensor.cuda_plan import plan_cuda

from tools.generativeqc_posthf.conventions import SpinMOBlock
from tools.generativeqc_posthf.providers import ConventionalProvider
from tools.generativeqc_response.uhf import UHFReferenceSnapshot

from .native_tensor_cpu import NativeCCTensorExecutor
from .native_tensor_cuda import CudaCCTensorExecutor
from .solver import _DIIS, SolverOptions

if TYPE_CHECKING:
    from typing_extensions import Self

_COULOMB_PERMUTATIONS = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)


def _canonical_coulomb(block: SpinMOBlock) -> tuple[SpinMOBlock, tuple[int, ...]]:
    """Use only chemists' eightfold Coulomb symmetry, including spin owners."""
    permutation = min(
        _COULOMB_PERMUTATIONS,
        key=lambda axes: tuple((block.spins[axis], block.slots[axis]) for axis in axes),
    )
    canonical = SpinMOBlock(
        tuple(block.slots[axis] for axis in permutation),
        tuple(block.spins[axis] for axis in permutation),
    )
    return canonical, tuple(permutation.index(axis) for axis in range(4))


def _integral_requests(name: str, snapshot: UHFReferenceSnapshot) -> tuple:
    domains, labels = name[1:].split("_")
    spins = tuple("alpha" if label == "a" else "beta" for label in labels)
    slots = tuple(
        tuple(range(snapshot.nocc(spin)))
        if domain == "o"
        else tuple(range(snapshot.nocc(spin), snapshot.nmo))
        for domain, spin in zip(domains, spins, strict=True)
    )
    contributions = []
    for coefficient, axes, output_axes in (
        (1, (0, 2, 1, 3), (0, 2, 1, 3)),
        (-1, (0, 3, 1, 2), (0, 2, 3, 1)),
    ):
        if spins[axes[0]] != spins[axes[1]] or spins[axes[2]] != spins[axes[3]]:
            continue
        block = SpinMOBlock(
            tuple(slots[axis] for axis in axes), tuple(spins[axis] for axis in axes)
        )
        canonical, restore = _canonical_coulomb(block)
        contributions.append(
            (coefficient, canonical, tuple(restore[axis] for axis in output_axes))
        )
    return tuple(contributions)


def _representatives(
    layout: UCCSDLayout, values: Mapping[str, np.ndarray], *, residuals: bool = False
) -> np.ndarray:
    """Read raw excitation representatives without projecting numerical residuals.

    Residual roundoff need not satisfy bit-exact full-array antisymmetry. Reading
    the physical representatives is distinct from accepting an invalid supplied
    amplitude: initial amplitudes always pass layout.pack's exact admission.
    """
    pieces = []
    for block in layout.blocks:
        if block.packed_layout.storage_elements == 0:
            pieces.append(np.empty(0))
            continue
        name = block.name.replace("t", "r", 1) if residuals else block.name
        full = values[name]
        if block.same_spin:
            occ_first, occ_second = np.triu_indices(block.spec.shape[0], 1)
            vir_first, vir_second = np.triu_indices(block.spec.shape[2], 1)
            full = full[
                occ_first[:, None],
                occ_second[:, None],
                vir_first[None, :],
                vir_second[None, :],
            ]
        pieces.append(full.ravel())
    return np.concatenate(pieces)


def _denominator_minimum(
    snapshot: UHFReferenceSnapshot, layout: UCCSDLayout, options: SolverOptions
) -> float:
    """Check actual excitation extrema and proposal shifts before allocation."""
    minima = []
    for block in layout.blocks:
        if block.packed_layout.storage_elements == 0:
            continue
        occupied, virtual = [], []
        for axis in block.spec.indices:
            energies = getattr(snapshot, f"orbital_energies_{axis.space.spin}")
            count = snapshot.nocc(axis.space.spin)
            (occupied if axis.space.kind == "occupied" else virtual).append(
                energies[:count] if axis.space.kind == "occupied" else energies[count:]
            )
        try:
            with np.errstate(over="raise", invalid="raise"):
                if block.same_spin:
                    largest = np.sum(nlargest(2, occupied[0])) - np.sum(
                        nsmallest(2, virtual[0])
                    )
                    smallest = np.sum(nsmallest(2, occupied[0])) - np.sum(
                        nlargest(2, virtual[0])
                    )
                else:
                    largest = sum(np.max(energies) for energies in occupied) - sum(
                        np.min(energies) for energies in virtual
                    )
                    smallest = sum(np.min(energies) for energies in occupied) - sum(
                        np.max(energies) for energies in virtual
                    )
                shifted = smallest - len(occupied) * options.level_shift
        except FloatingPointError as error:
            raise ValueError("nonfinite UCCSD denominator extrema") from error
        if not all(np.isfinite(value) for value in (largest, smallest, shifted)):
            raise ValueError("nonfinite UCCSD denominator extrema")
        if largest >= 0 or -largest <= options.denominator_threshold:
            raise ValueError(
                f"near-zero or non-Aufbau UCCSD denominator in {block.name}"
            )
        minima.append(float(-largest))
    return min(minima) if minima else 0.0


def _denominators(
    snapshot: UHFReferenceSnapshot, layout: UCCSDLayout, shift: float
) -> np.ndarray:
    pieces = []
    for block in layout.blocks:
        if block.packed_layout.storage_elements == 0:
            pieces.append(np.empty(0))
            continue
        if block.same_spin:
            spin = block.spec.indices[0].space.spin
            energies = getattr(snapshot, f"orbital_energies_{spin}")
            occupied = snapshot.nocc(spin)
            occ_first, occ_second = np.triu_indices(occupied, 1)
            vir_first, vir_second = np.triu_indices(snapshot.nmo - occupied, 1)
            pairs_occ = energies[occ_first] + energies[occ_second]
            pairs_vir = energies[occupied + vir_first] + energies[occupied + vir_second]
            values = pairs_occ[:, None] - pairs_vir[None, :] - 2 * shift
        else:
            values = np.zeros(block.spec.shape)
            for position, axis in enumerate(block.spec.indices):
                energies = getattr(snapshot, f"orbital_energies_{axis.space.spin}")
                occupied = snapshot.nocc(axis.space.spin)
                chosen = (
                    energies[:occupied]
                    if axis.space.kind == "occupied"
                    else energies[occupied:]
                )
                shape = [1] * len(block.spec.shape)
                shape[position] = len(chosen)
                values += (1 if axis.space.kind == "occupied" else -1) * chosen.reshape(
                    shape
                )
            values -= (1 if block.name.startswith("t1") else 2) * shift
        pieces.append(values.ravel())
    return immutable(np.concatenate(pieces))


@dataclass(frozen=True)
class UCCSDIteration:
    """One evaluated physical state; changes are never inferred from proposals."""

    iteration: int
    correlation_energy: float
    energy_change: float | None
    residual_max: float
    block_residual_max: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class UCCSDResult:
    """Owned last finite excitation state with explicit convergence provenance."""

    status: str
    reason: str
    correlation_energy: float | None
    total_energy: float | None
    amplitudes: Mapping[str, np.ndarray]
    packed_amplitudes: np.ndarray
    history: tuple[UCCSDIteration, ...]
    provenance: Mapping[str, Any]

    @property
    def converged(self) -> bool:
        return self.status == "converged"


class PreparedUCCSD:
    """Own conventional MO feeds, with bounded packed host iteration state.

    Provider/snapshot identities and the complete provider cache transition are
    admitted before AO reads. Provider and solver numeric budgets are separate
    explicit bounds, as in existing correlated owners. Prepared feeds own host
    storage and survive provider clearing/closing. CUDA is explicitly staged:
    four-center integral transforms and CC contractions use CUDA, while Fock
    transforms, feeds, checkpoints and excitation-state DIIS remain on the host.
    No PySCF or CPU equation fallback.
    """

    def __init__(
        self,
        snapshot: UHFReferenceSnapshot,
        provider: ConventionalProvider,
        *,
        options: SolverOptions | None = None,
        executor: NativeCCTensorExecutor | CudaCCTensorExecutor | None = None,
    ) -> None:
        started = time.perf_counter()
        if (
            not isinstance(snapshot, UHFReferenceSnapshot)
            or snapshot.algorithm != "UHF"
        ):
            raise TypeError(
                "UCCSD requires a validated canonical UHF reference, not RHF/UKS"
            )
        if (
            not isinstance(provider, ConventionalProvider)
            or provider.snapshot.identity != snapshot.identity
        ):
            raise ValueError(
                "UCCSD requires an identity-matched conventional integral provider"
            )
        if options is None:
            options = SolverOptions()
        if not isinstance(options, SolverOptions):
            raise TypeError("UCCSD options must be SolverOptions")
        if executor is None:
            if provider.backend != "cpu":
                raise ValueError("CUDA UCCSD requires an explicit CUDA tensor executor")
            executor = NativeCCTensorExecutor(max_bytes=options.max_bytes)
        if not isinstance(executor, (NativeCCTensorExecutor, CudaCCTensorExecutor)):
            raise TypeError("UCCSD requires a native CPU or CUDA TensorIR executor")
        cuda = isinstance(executor, CudaCCTensorExecutor)
        if provider.backend != ("cuda" if cuda else "cpu"):
            raise ValueError(
                "UCCSD integral and tensor backends must match; no CPU fallback"
            )
        if isinstance(executor, CudaCCTensorExecutor) and (
            provider.cuda_artifact is None or provider.device_id != executor.device
        ):
            raise ValueError(
                "CUDA UCCSD requires a transform artifact and matching device ordinal"
            )
        self.reference, self.options, self.executor = snapshot, options, executor
        self.layout = UCCSDLayout(
            snapshot.nocc("alpha"),
            snapshot.nocc("beta"),
            snapshot.nmo - snapshot.nocc("alpha"),
            snapshot.nmo - snapshot.nocc("beta"),
        )
        self.minimum_absolute_denominator = _denominator_minimum(
            snapshot, self.layout, options
        )
        self.program = build_program(
            self.layout.oa, self.layout.ob, self.layout.va, self.layout.vb
        )
        inputs = {
            node.attrs["name"]: node.spec
            for node in self.program.live_nodes
            if node.op == "input"
        }
        conversions = {
            name: _integral_requests(name, snapshot)
            for name in sorted(inputs)
            if name.startswith("g")
        }
        requests = tuple(
            dict.fromkeys(
                block
                for contributions in conversions.values()
                for _, block, _ in contributions
            )
        )
        full_bytes = sum(
            block.full_layout.storage_bytes(8) for block in self.layout.blocks
        )
        packed_bytes = self.layout.packed_size * 8
        feed_bytes = sum(
            spec.size * 8 for name, spec in inputs.items() if not name.startswith("t")
        )
        raw_bytes = sum(math.prod(block.shape) * 8 for block in requests)
        largest = max(
            (spec.size * 8 for name, spec in inputs.items() if name.startswith("g")),
            default=0,
        )
        self.host_iteration_bytes = checked_bytes(
            feed_bytes
            + 4 * full_bytes
            + (12 + 4 * options.diis_size) * packed_bytes
            + 32 * (options.diis_size + 1) ** 2,
            "UCCSD iteration storage",
        )
        host_preparation_bytes = checked_bytes(
            feed_bytes + raw_bytes + 4 * largest + 24 * snapshot.nmo**2,
            "UCCSD preparation storage",
        )
        if isinstance(executor, CudaCCTensorExecutor):
            self.execution_peak_bytes = plan_cuda(
                self.program, executor.compiler.target, max_bytes=executor.max_bytes
            ).peak_bytes
        else:
            _, resources = emit_cpu(
                self.program,
                max_bytes=executor.max_bytes,
                max_work=executor.max_work,
                max_nodes=executor.max_nodes,
            )
            self.execution_peak_bytes = resources["required_bytes"]
        self.logical_required_bytes = checked_bytes(
            max(
                host_preparation_bytes,
                self.host_iteration_bytes + self.execution_peak_bytes,
            ),
            "UCCSD numeric storage",
        )
        if self.logical_required_bytes > options.max_bytes:
            raise MemoryError(
                f"UCCSD solver budget needs {self.logical_required_bytes} numeric bytes before integral conversion"
            )
        self.denominators = _denominators(snapshot, self.layout, options.level_shift)
        feeds = {}
        for spin, suffix in (("alpha", "aa"), ("beta", "bb")):
            coefficients = getattr(snapshot, f"coefficients_{spin}")
            fock = coefficients.T @ getattr(snapshot, f"fock_{spin}") @ coefficients
            occupied = snapshot.nocc(spin)
            for domains, axes in (
                ("oo", (slice(0, occupied), slice(0, occupied))),
                ("ov", (slice(0, occupied), slice(occupied, None))),
                ("vv", (slice(occupied, None), slice(occupied, None))),
            ):
                feeds[f"f{domains}_{suffix}"] = immutable(fock[axes])
        with provider._lock:
            self.provider_peak_bytes = provider.admit(requests)
            previous_statistics = dict(provider.statistics)
            raw = {}
            for block in requests:
                result = provider.get(block)
                if (
                    result.reference_id != snapshot.identity
                    or result.hamiltonian_id != snapshot.hamiltonian_id
                ):
                    raise ValueError(
                        "UCCSD integral block identity does not match reference"
                    )
                raw[block] = immutable(result.to_host(), shape=block.shape)
            for name, contributions in conversions.items():
                value = np.zeros(inputs[name].shape)
                for coefficient, block, axes in contributions:
                    value += coefficient * raw[block].transpose(axes)
                feeds[name] = immutable(value)
            self.provider_statistics = MappingProxyType(
                {
                    name: (
                        self.provider_peak_bytes
                        if name == "peak_bytes"
                        else value - previous_statistics[name]
                    )
                    for name, value in provider.statistics.items()
                }
            )
        self.feeds = MappingProxyType(feeds)
        self.integral_hash = canonical_hash(
            {
                name: sha256(value.astype("<f8", copy=False).tobytes()).hexdigest()
                for name, value in feeds.items()
            }
        )
        self.integral_backend = provider.backend
        self.source_backend = provider.source.backend
        self.integral_source_id = provider.source.identity
        self.integral_request_count = len(requests)
        self._executor_configuration = self._execution_configuration()
        self._closed = False
        self.preparation_seconds = time.perf_counter() - started

    def _execution_configuration(self) -> tuple:
        return (
            self.executor.backend,
            self.executor.max_bytes,
            getattr(self.executor, "max_work", None),
            getattr(self.executor, "max_nodes", None),
            getattr(self.executor, "device", None),
            getattr(self.executor.compiler, "target", None),
        )

    def evaluate(self, packed: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate physical residuals on the selected backend with owned feeds."""
        if self._closed:
            raise RuntimeError("prepared UCCSD is closed")
        if self._execution_configuration() != self._executor_configuration:
            raise RuntimeError(
                "UCCSD executor configuration changed; prepare a fresh owner"
            )
        amplitudes = self.layout.unpack(packed)
        outputs = self.executor.execute(self.program, {**self.feeds, **amplitudes})
        expected = {
            "correlation_energy": (),
            **{
                block.name.replace("t", "r", 1): block.spec.shape
                for block in self.layout.blocks
            },
        }
        if outputs.keys() != expected.keys():
            raise ValueError("UCCSD executor returned incomplete physical outputs")
        validated = {}
        for name, shape in expected.items():
            value = np.asarray(outputs[name])
            if value.dtype != np.float64 or value.shape != shape:
                raise ValueError("UCCSD executor violated the FP64 shape contract")
            if not np.isfinite(value).all():
                raise FloatingPointError(
                    "UCCSD executor returned nonfinite physical outputs"
                )
            validated[name] = value
        return validated

    def solve(
        self, initial: Mapping[str, np.ndarray] | np.ndarray | None = None
    ) -> UCCSDResult:
        """Solve with packed excitation DIIS and fresh physical convergence gates."""
        if self._closed:
            raise RuntimeError("prepared UCCSD is closed")
        if initial is None:
            current = np.zeros(self.layout.packed_size)
        elif isinstance(initial, Mapping):
            current = self.layout.pack(initial)
        else:
            current = self.layout.pack(self.layout.unpack(initial))
        last_finite = current.copy()
        diis = _DIIS(self.options.diis_size)
        history = []
        energy = previous_energy = last_finite_energy = None
        status, reason = "iteration_limit", "maximum UCCSD iterations reached"
        evaluations = 0
        started = time.perf_counter()
        try:
            for iteration in range(self.options.max_iterations + 1):
                outputs = self.evaluate(current)
                evaluations += 1
                current_energy = float(outputs["correlation_energy"])
                maxima = tuple(
                    (
                        block.name.replace("t", "r", 1),
                        float(
                            np.max(
                                np.abs(outputs[block.name.replace("t", "r", 1)]),
                                initial=0,
                            )
                        ),
                    )
                    for block in self.layout.blocks
                )
                residual = max(value for _, value in maxima)
                delta = (
                    None
                    if previous_energy is None
                    else abs(current_energy - previous_energy)
                )
                history.append(
                    UCCSDIteration(iteration, current_energy, delta, residual, maxima)
                )
                last_finite, energy = current.copy(), current_energy
                last_finite_energy = current_energy
                if residual <= self.options.residual_tolerance and (
                    not self.layout.packed_size
                    or (delta is not None and delta <= self.options.energy_tolerance)
                ):
                    recheck = self.evaluate(current)
                    evaluations += 1
                    last_finite_energy = float(recheck["correlation_energy"])
                    fresh_residual = max(
                        float(np.max(np.abs(value), initial=0))
                        for name, value in recheck.items()
                        if name != "correlation_energy"
                    )
                    if (
                        fresh_residual <= self.options.residual_tolerance
                        and abs(float(recheck["correlation_energy"]) - energy)
                        <= self.options.energy_tolerance
                    ):
                        status, reason = (
                            "converged",
                            "energy change and all five freshly evaluated physical residuals passed",
                        )
                        break
                if iteration == self.options.max_iterations:
                    break
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    trial = (
                        current
                        + (1 - self.options.damping)
                        * _representatives(self.layout, outputs, residuals=True)
                        / self.denominators
                    )
                trial_outputs = self.evaluate(trial)
                evaluations += 1
                last_finite, last_finite_energy = (
                    trial.copy(),
                    float(trial_outputs["correlation_energy"]),
                )
                error = _representatives(self.layout, trial_outputs, residuals=True)
                current = diis.update(trial, error)
                previous_energy = energy
        except (FloatingPointError, OverflowError) as error:
            status, reason = "nonfinite", str(error)
        except ValueError as error:
            if not any(word in str(error).lower() for word in ("finite", "overflow")):
                raise
            status, reason = "nonfinite", str(error)
        packed = immutable(last_finite)
        amplitudes = MappingProxyType(
            {
                name: immutable(value)
                for name, value in self.layout.unpack(packed).items()
            }
        )
        provenance = MappingProxyType(
            {
                "reference_id": self.reference.identity,
                "hamiltonian_id": self.reference.hamiltonian_id,
                "integral_source_id": self.integral_source_id,
                "integral_hash": self.integral_hash,
                "equation_hash": self.program.logical_hash,
                "tensor_backend": self.executor.backend,
                "integral_backend": self.integral_backend,
                "source_backend": self.source_backend,
                "fock_backend": "host-fp64",
                "iteration_backend": "host-packed-excitation-diis",
                "provider_statistics": self.provider_statistics,
                "minimum_absolute_denominator": self.minimum_absolute_denominator,
                "solver_numeric_bound_bytes": self.logical_required_bytes,
                "provider_peak_bytes": self.provider_peak_bytes,
                "integral_request_count": self.integral_request_count,
                "physical_evaluations": evaluations,
                "diis_restarts": diis.restarts,
                "options": MappingProxyType(asdict(self.options)),
                "preparation_seconds": self.preparation_seconds,
                "solve_seconds": time.perf_counter() - started,
                "complete_endpoint_seconds": self.preparation_seconds
                + time.perf_counter()
                - started,
            }
        )
        return UCCSDResult(
            status,
            reason,
            last_finite_energy,
            None
            if last_finite_energy is None
            else self.reference.reference_energy + last_finite_energy,
            amplitudes,
            packed,
            tuple(history),
            provenance,
        )

    def close(self) -> None:
        """Release owned feeds; never close borrowed providers or executors."""
        self.feeds = MappingProxyType({})
        self.denominators = np.empty(0)
        self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("prepared UCCSD is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
