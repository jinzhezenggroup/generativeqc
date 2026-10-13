"""Immutable physical KS records copied from one completed native execution."""

import ctypes
import math
import typing
from dataclasses import asdict, dataclass

from generativeqc_compiler.xc._generated_native_semilocal import SCF_DOMAIN_BY_VERSION
from generativeqc_compiler.xc.density_work import (
    DENSITY_WORK_DOMAIN,
    DENSITY_WORK_DOMAIN_VERSION,
)
from generativeqc_compiler.xc.libxc_work import LIBXC_WORK_DOMAIN_VERSION

from . import _native
from .ks import AUTOMATIC_SCF_DOMAIN, SPLIT_HYBRID_SCF_DOMAIN


@dataclass(frozen=True)
class KsEnergyComponents:
    """Physical Hartree energy terms; XC includes exact exchange when present."""

    nuclear: float
    one_electron: float
    hartree: float
    xc: float

    @property
    def total(self) -> typing.Any:
        """Return the nuclear, one-electron, Hartree, and XC energy sum."""
        return self.nuclear + self.one_electron + self.hartree + self.xc


@dataclass(frozen=True)
class KsIteration:
    """Current physical E/F/D and its proposed density change for one step.

    Energy change is None when a native iteration stage has no preceding
    within-stage energy. The initial +infinity and an explicit later-stage
    -1.0 native marker represent this unavailable baseline, never convergence.
    RMS maxima gate each spin separately. An occupation-stabilized proposal
    changes the iteration control only; all reported components are unshifted.
    """

    iteration: int
    components: KsEnergyComponents
    energy_change: float | None
    density_change_max: float
    physical_residual_max: float
    electrons: tuple[float, float]
    occupation_stabilized: bool


@dataclass(frozen=True)
class KsDiagnostic:
    """Final physical state plus the complete iteration history of one solve.

    CPU RKS performs additional final validation rebuilds; final components and
    residual can therefore differ from history[-1]. Neither is overwritten by
    the other. Electron counts use the AO overlap metric, not quadrature.
    A warm retry reports its final attempt; this is not a total-work counter
    for discarded attempts. Execution and transport profiling are separate.
    The optional coulomb_exchange_work is a narrow, observed target-execution
    census including discarded attempts; it does not certify full precision.
    """

    occupations: tuple[int, int]
    electrons: tuple[float, float]
    grid_points: int
    tile_points: int
    ao_order: int
    scf_domain: str
    initial_density_used: bool
    fock_builds: int
    components: KsEnergyComponents
    density_change_max: float
    physical_residual_max: float
    history: tuple[KsIteration, ...]
    coulomb_exchange_work: dict[str, typing.Any] | None = None
    nonlocal_work: dict[str, int] | None = None
    nonlocal_history_scopes: tuple[str, ...] | None = None

    def to_payload(self) -> typing.Any:
        """Copy every iteration; known unavailable stage baselines are null.

        Finite physical records are JSON-compatible. Unexpected nonfinite
        numerical values remain visible for callers to reject, not hidden as
        missing stage baselines.
        """
        payload = asdict(self)
        # The optional native initial-density experiment keeps its work in the
        # same history. Those rows describe the provisional operator, so they
        # must never look like physical WB97M-V target checks to readers.
        provisional = (self.nonlocal_work or {}).get("provisional_focks", 0)
        if self.nonlocal_history_scopes is not None:
            for row, scope in zip(
                payload["history"], self.nonlocal_history_scopes, strict=True
            ):
                row["operator_scope"] = scope
        elif provisional:
            for row in payload["history"]:
                row["operator_scope"] = (
                    "provisional-semilocal-rsh"
                    if row["iteration"] <= provisional
                    else "full-target"
                )
        return payload


@dataclass(frozen=True)
class KsTransportDiagnostic:
    """Cumulative measured transfers and synchronization for prepared CUDA KS."""

    setup_h2d_bytes: int
    density_h2d_bytes: int
    scalar_d2h_bytes: int
    matrix_d2h_bytes: int
    synchronizations: int
    iterations: int
    occupation_stabilized_proposals: int

    def to_payload(self) -> typing.Any:
        """Return transport diagnostics as a dictionary of dataclass fields."""
        return asdict(self)


def _components(value: typing.Any) -> typing.Any:
    return KsEnergyComponents(
        value.nuclear_energy,
        value.one_electron_energy,
        value.hartree_energy,
        value.xc_energy,
    )


def _coulomb_exchange_work(
    library: typing.Any, handle: typing.Any, index: typing.Any
) -> dict[str, typing.Any] | None:
    name = (
        "generativeqc_calculation_get_ks_coulomb_exchange_work_v1"
        if index is None
        else "generativeqc_batch_get_ks_coulomb_exchange_work_v1"
    )
    query = getattr(library, name, None)
    if query is None:
        return None
    record = _native.KsCoulombExchangeWorkDescriptor(
        ctypes.sizeof(_native.KsCoulombExchangeWorkDescriptor), _native.ABI_VERSION
    )
    prefix = (handle,) if index is None else (handle, index)
    status = query(*prefix, ctypes.byref(record))
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status)
    return {
        "complete": bool(record.complete),
        "coulomb_applications": record.coulomb_applications
        if record.complete
        else None,
        "exchange_applications": record.exchange_applications
        if record.complete
        else None,
    }


def read_ks_diagnostic(
    library: typing.Any,
    handle: typing.Any,
    index: typing.Any = None,
    *,
    expected_domain: str | None = None,
) -> typing.Any:
    """Copy the current native record and check its resolved work-domain identity.

    Generated split hybrids have a distinct native domain version. Their
    resolved method is still checked before admitting the diagnostic.
    """
    name = (
        "generativeqc_calculation_get_ks_diagnostic"
        if index is None
        else "generativeqc_batch_get_ks_diagnostic"
    )
    query = getattr(library, name)
    prefix = (handle,) if index is None else (handle, index)
    summary = _native.KsDiagnosticDescriptor(
        ctypes.sizeof(_native.KsDiagnosticDescriptor), _native.ABI_VERSION
    )
    status = query(*prefix, ctypes.byref(summary), None, 0)
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status)
    # Domain IDs identify numerical policies, independently of method aliases.
    if expected_domain == DENSITY_WORK_DOMAIN:
        if summary.scf_domain_version != DENSITY_WORK_DOMAIN_VERSION:
            raise RuntimeError("native density-work Libxc diagnostic domain mismatch")
        domain = DENSITY_WORK_DOMAIN
    elif expected_domain == AUTOMATIC_SCF_DOMAIN:
        if summary.scf_domain_version != LIBXC_WORK_DOMAIN_VERSION:
            raise RuntimeError("native automatic Libxc diagnostic domain mismatch")
        domain = AUTOMATIC_SCF_DOMAIN
    elif expected_domain == SPLIT_HYBRID_SCF_DOMAIN:
        if summary.scf_domain_version != 4:
            raise RuntimeError("native split-hybrid diagnostic domain mismatch")
        domain = SPLIT_HYBRID_SCF_DOMAIN
    elif summary.scf_domain_version in SCF_DOMAIN_BY_VERSION:
        domain = SCF_DOMAIN_BY_VERSION[summary.scf_domain_version]
    else:
        raise RuntimeError("unsupported native KS diagnostic domain version")
    history = (_native.KsIterationDescriptor * summary.history_count)()
    for row in history:
        row.struct_size = ctypes.sizeof(row)
        row.abi_version = _native.ABI_VERSION
    _native.check(library, query(*prefix, ctypes.byref(summary), history, len(history)))
    rows = []
    for row in history:
        change = row.energy_change
        if (row.iteration == 1 and math.isinf(change) and change > 0) or (
            row.iteration > 1 and change == -1.0
        ):
            change = None
        rows.append(
            KsIteration(
                row.iteration,
                _components(row),
                change,
                row.density_change_max,
                row.physical_residual_max,
                tuple(row.electrons),
                bool(row.occupation_stabilized),
            )
        )
    return KsDiagnostic(
        occupations=tuple(summary.occupations),
        electrons=tuple(summary.electrons),
        grid_points=summary.grid_points,
        tile_points=summary.tile_points,
        ao_order=summary.required_ao_order,
        scf_domain=domain,
        initial_density_used=bool(summary.initial_density_used),
        fock_builds=summary.fock_builds,
        components=_components(summary),
        density_change_max=summary.density_change_max,
        physical_residual_max=summary.physical_residual_max,
        history=tuple(rows),
        coulomb_exchange_work=_coulomb_exchange_work(library, handle, index),
        nonlocal_work=_nonlocal_work(library, handle, index),
        nonlocal_history_scopes=_nonlocal_history_scopes(
            library, handle, index, len(history)
        ),
    )


def _nonlocal_history_scopes(
    library: typing.Any, handle: typing.Any, index: typing.Any, count: int
) -> tuple[str, ...] | None:
    """Distinguish interleaved inner problems from physical target evaluations."""
    query = getattr(
        library, "generativeqc_internal_batch_get_nonlocal_history_scopes_v1", None
    )
    if index is None or query is None or not count:
        return None
    values = (ctypes.c_uint64 * count)()
    status = query(handle, index, values, count)
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status)
    names = (
        "full-target",
        "provisional-semilocal-rsh",
        "provisional-frozen-vv10",
        "provisional-frozen-vv10-and-range-exchange",
    )
    if any(value >= len(names) for value in values):
        raise ValueError("invalid native nonlocal operator scope")
    return tuple(names[value] for value in values)


def _nonlocal_work(
    library: typing.Any, handle: typing.Any, index: typing.Any
) -> typing.Any:
    """Read completed run-local VV10 counters; older libraries remain unavailable."""
    query = getattr(library, "generativeqc_internal_batch_get_nonlocal_work_v1", None)
    if index is None or query is None:
        return None
    values = (ctypes.c_uint64 * 6)()
    status = query(handle, index, values, len(values))
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status)
    if values[0] != 1:
        raise ValueError("invalid native nonlocal work availability")
    return dict(
        zip(
            (
                "provisional_focks",
                "target_focks",
                "pair_cache_probes",
                "pair_cache_hits",
                "pair_sweeps",
            ),
            values[1:],
            strict=True,
        )
    )


def read_ks_transport_diagnostic(
    library: typing.Any, handle: typing.Any, index: typing.Any = None
) -> typing.Any:
    """Snapshot cumulative CUDA movement; unsupported/CPU execution returns None."""
    name = (
        "generativeqc_calculation_get_ks_transport_diagnostic"
        if index is None
        else "generativeqc_batch_get_ks_transport_diagnostic"
    )
    query = getattr(library, name)
    prefix = (handle,) if index is None else (handle, index)
    value = _native.KsTransportDiagnosticDescriptor(
        ctypes.sizeof(_native.KsTransportDiagnosticDescriptor), _native.ABI_VERSION
    )
    status = query(*prefix, ctypes.byref(value))
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status)
    return KsTransportDiagnostic(
        **{
            field: getattr(value, field)
            for field, _ in value._fields_
            if field not in ("struct_size", "abi_version")
        }
    )
