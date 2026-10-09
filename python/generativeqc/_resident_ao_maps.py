"""Bound explicit AO maps to immutable geometry, grid and derivative domains.

This private owner supplies storage/lifetime policy only. The caller chooses an
explicit sampled-jet cutoff and qualifies complete energy/force errors. No default
screening policy is registered. The producer and every consumer remain resident;
only compact AO index arrays are retained on the host.
"""

from __future__ import annotations

import math
import typing
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.cuda import CudaAoMapAllocationError
from generativeqc_compiler.dft.indexed_layout import AoGridBlockLayout


@dataclass(frozen=True)
class ResidentAoMapDomain:
    """Exact caller-validated geometry/grid domain; excludes density generation.

    The molecular-grid lease must be token-checked before entering this owner.
    Pointer equality is an additional lifetime check, never a scientific identity.
    A changed basis, geometry, point order, derivative order or tile shape requires
    a new owner. An order-1 map cannot be relabelled for an order-2 force consumer.
    """

    basis_identity: str
    geometry_identity: str
    grid_identity: str
    device: int
    point_pointer: int
    point_count: int
    tile_points: int
    derivative_order: int


_DENSE = object()


class ResidentAoMapCache:
    """Retain immutable maps under an explicit additional numeric host budget.

    The full AO scratch is already charged to CudaGrid. This budget includes
    retained indices and a conservative 20 bytes per global AO for simultaneous
    flag/index/immutable-output staging in the AO-only producer. It also covers
    the smaller host map copy in a subsequent feature lease. Python object and
    list headers are excluded, consistently with the existing grid resource
    contract. Budget/capability misses select the dense map before any omission;
    invalid identity, numerical and device failures propagate.
    """

    def __init__(
        self,
        grid: typing.Any,
        domain: ResidentAoMapDomain,
        *,
        cutoff: float,
        budget_bytes: int,
        producer: str = "sampled-jets",
    ) -> None:
        if type(producer) is not str or producer not in {
            "sampled-jets",
            "pre-ao-envelope",
        }:
            raise ValueError("unsupported resident AO domain producer")
        self._producer = producer
        if type(domain) is not ResidentAoMapDomain:
            raise TypeError("resident AO cache requires an explicit domain")
        if any(
            not isinstance(value, str) or not value
            for value in (
                domain.basis_identity,
                domain.geometry_identity,
                domain.grid_identity,
            )
        ):
            raise ValueError("resident AO cache requires scientific identities")
        if any(
            type(value) is not int or value <= 0
            for value in (domain.point_pointer, domain.point_count, domain.tile_points)
        ):
            raise ValueError("resident AO cache requires positive point dimensions")
        if (
            type(domain.device) is not int
            or domain.device < 0
            or type(domain.derivative_order) is not int
            or not 0 <= domain.derivative_order <= 3
        ):
            raise ValueError(
                "resident AO cache requires a device and supported jet order"
            )
        if type(cutoff) not in (int, float) or not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("resident AO cutoff must be finite and positive")
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("resident AO cache budget must be a nonnegative integer")
        self._grid = grid
        self._domain = domain
        self._cutoff = float(cutoff)
        self._budget_bytes = budget_bytes
        self._geometry_generation = grid.geometry_generation
        self._basis_generation = grid.basis_generation
        self._validate_binding(grid, domain)
        self._maps: dict[int, typing.Any] = {}
        self._block_layouts: dict[int, AoGridBlockLayout] = {}
        self._retained_bytes = 0
        self._transient_bytes = 20 * grid.plan.nao
        self._capability_missing = False
        self.reset_work()

    @property
    def domain(self) -> ResidentAoMapDomain:
        return self._domain

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def producer(self) -> str:
        """Immutable omission mechanism; not interchangeable on a warm cache."""
        return self._producer

    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

    def _validate_binding(self, grid: typing.Any, domain: ResidentAoMapDomain) -> None:
        if (
            grid is not self._grid
            or domain != self.domain
            or grid.basis_identity != domain.basis_identity
            or grid.device_id != domain.device
            or grid.plan.order != domain.derivative_order
            or grid.plan.tile_points != domain.tile_points
            or grid.plan.active_ao_capacity != grid.plan.nao
            or grid.geometry_generation != self._geometry_generation
            or grid.basis_generation != self._basis_generation
        ):
            raise ValueError(
                "resident AO cache geometry/grid/derivative binding mismatch"
            )

    def reset_work(self) -> None:
        """Begin one endpoint's counters without evicting geometry-bound maps."""
        self._work = {
            "lookups": 0,
            "cache_hits": 0,
            "discoveries": 0,
            "discovery_seconds": 0.0,
            "discovery_ao_jet_values": 0,
            "discovery_region_bounds": 0,
            "discovery_density_contractions": 0,
            "dense_budget_tiles": 0,
            "dense_capability_tiles": 0,
            "point_ao_visits": 0,
            "point_ao_square_sum": 0,
            "dense_point_ao_square_sum": 0,
            "active_aos_min": None,
            "active_aos_max": 0,
            "active_aos_sum": 0,
            "tile_count": 0,
            "empty_tile_count": 0,
        }

    @property
    def work(self) -> dict[str, typing.Any]:
        """Detached per-endpoint work and current numeric capacity accounting."""
        return dict(
            self._work,
            discovery_producer=self.producer,
            retained_map_bytes=self._retained_bytes,
            transient_reserve_bytes=self._transient_bytes,
            budget_bytes=self.budget_bytes,
            numeric_peak_bound_bytes=(
                self._retained_bytes + self._transient_bytes
                if self.budget_bytes >= self._transient_bytes
                else 0
            ),
        )

    def select_block(
        self, grid: typing.Any, domain: ResidentAoMapDomain, begin: int, count: int
    ) -> tuple[np.ndarray | None, AoGridBlockLayout]:
        """Publish the existing map with its validated local-domain capability.

        No second coordinate array or sparse cache is created. Epochs remain
        execution bindings, not generated equation/source identity. Holding
        the same grid lock prevents center rebinding during certification.
        Retained maps also retain their immutable descriptor. Dense budget or
        capability fallbacks do not grow a second inventory of cached blocks.
        """
        with grid._lock:
            selected = self.select(grid, domain, begin, count)
            layout = self._block_layouts.get(begin)
            if layout is not None:
                return selected, layout
            layout = AoGridBlockLayout(
                grid.plan.nao,
                grid.plan.nao if selected is None else selected.size,
                count,
                domain.derivative_order,
                domain.basis_identity,
                selected is not None,
                domain.derivative_order,
                begin,
                self._basis_generation,
                self._geometry_generation,
            )
            if begin in self._maps:
                self._block_layouts[begin] = layout
            return selected, layout

    @contextmanager
    def feature_task(
        self,
        grid: typing.Any,
        domain: ResidentAoMapDomain,
        begin: int,
        count: int,
        ingredients: typing.Iterable[str],
        *,
        stamp: typing.Any = None,
    ) -> typing.Any:
        """Keep the diagnostic sampled-jet route behind the same lease API."""
        selected, layout = self.select_block(grid, domain, begin, count)
        with grid.feature_task_device_points(
            domain.point_pointer + 24 * begin,
            count,
            selected,
            ingredients,
            stamp=stamp,
            block_layout=layout,
        ) as task:
            yield task

    def select(
        self, grid: typing.Any, domain: ResidentAoMapDomain, begin: int, count: int
    ) -> np.ndarray | None:
        """Return a retained selected map, or ``None`` for the full AO domain.

        Every call rechecks scientific/lifetime binding, even on a cache hit.
        Discovery is synchronous and must occur before borrowing the AO task.
        Failures never publish a partially discovered map. Geometry rebinding
        invalidates this owner even if the CUDA allocator reuses an address.
        """
        self._validate_binding(grid, domain)
        if (
            type(begin) is not int
            or type(count) is not int
            or begin < 0
            or begin >= domain.point_count
            or begin % domain.tile_points
            or count != min(domain.tile_points, domain.point_count - begin)
        ):
            raise ValueError("resident AO lookup differs from its point tile domain")
        # Share the grid lock with native lease admission and center rebinding.
        with grid._lock:
            grid._check_open()
            self._validate_binding(grid, domain)
            self._work["lookups"] += 1
            if begin in self._maps:
                self._work["cache_hits"] += 1
                selected = self._maps[begin]
            elif self._capability_missing:
                self._work["dense_capability_tiles"] += 1
                selected = _DENSE
            elif (
                self._retained_bytes
                + self._transient_bytes
                + grid.plan.nao * np.dtype(np.uintp).itemsize
                > self.budget_bytes
            ):
                self._work["dense_budget_tiles"] += 1
                selected = _DENSE
            else:
                started = perf_counter()
                try:
                    options = {"cutoff": self.cutoff}
                    if self.producer != "sampled-jets":
                        options["producer"] = self.producer
                    selected = grid.select_ao_device_points(
                        domain.point_pointer + begin * 3 * 8, count, **options
                    )
                except NotImplementedError:
                    self._capability_missing = True
                    self._work["dense_capability_tiles"] += 1
                    selected = _DENSE
                else:
                    if (
                        selected.dtype != np.dtype(np.uintp)
                        or selected.ndim != 1
                        or selected.flags.writeable
                        or selected.size > grid.plan.nao
                        or (selected.size and selected[-1] >= grid.plan.nao)
                        or np.any(selected[1:] <= selected[:-1])
                    ):
                        raise ValueError("resident AO producer returned an invalid map")
                    self._work["discoveries"] += 1
                    order = domain.derivative_order
                    jets = (order + 1) * (order + 2) * (order + 3) // 6
                    if self.producer == "sampled-jets":
                        self._work["discovery_ao_jet_values"] += (
                            count * grid.plan.nao * jets
                        )
                    else:
                        self._work["discovery_region_bounds"] += grid.plan.nao * jets
                    # A full identity map needs no retained numerical array.
                    if selected.size == grid.plan.nao:
                        selected = _DENSE
                    else:
                        self._retained_bytes += selected.nbytes
                    self._maps[begin] = selected
                finally:
                    self._work["discovery_seconds"] += perf_counter() - started
            active = grid.plan.nao if selected is _DENSE else selected.size
            self._work["tile_count"] += 1
            self._work["empty_tile_count"] += int(active == 0)
            prior_min = self._work["active_aos_min"]
            self._work["active_aos_min"] = (
                active if prior_min is None else min(prior_min, active)
            )
            self._work["active_aos_max"] = max(self._work["active_aos_max"], active)
            self._work["active_aos_sum"] += active
            self._work["point_ao_visits"] += count * active
            self._work["point_ao_square_sum"] += count * active**2
            self._work["dense_point_ao_square_sum"] += count * grid.plan.nao**2
            return None if selected is _DENSE else selected


class ResidentDeviceAoMapOwner:
    """Native CSR lifetime adapter, with no Python AO-label inventory.

    Domain identities bind immutable quadrature storage, not just its address.
    Only the native owner resolves tile spans; all downstream leases keep the
    existing AoGridBlockLayout contract. Budget/capability and optional
    high-occupancy misses stay dense; even a declined CSR is ledger-accounted.
    Derivative maps remain exact-capability here until cross-order reuse is
    separately qualified. The caller supplies policy; this adapter owns storage.
    """

    def __init__(
        self,
        grid: typing.Any,
        domain: ResidentAoMapDomain,
        *,
        cutoff: float,
        budget_bytes: int,
        max_active_fraction: float = 1.0,
        producer: str = "pre-ao-envelope-native-csr",
    ) -> None:
        if producer not in {"pre-ao-envelope-native-csr", "exact-jets-native-bitmask"}:
            raise ValueError("unsupported native resident AO producer")
        self._producer = producer
        if type(max_active_fraction) not in (int, float) or not (
            0 < max_active_fraction <= 1
        ):
            raise ValueError("resident AO occupancy limit must be in (0,1]")
        # Reuse admission, not discovery or storage, from the host adapter.
        # This short-lived validator has no selected maps or numerical arrays.
        admission = ResidentAoMapCache(
            grid,
            domain,
            cutoff=cutoff,
            budget_bytes=budget_bytes,
            producer="pre-ao-envelope",
        )
        self._grid = grid
        self._domain = domain
        self._cutoff = admission.cutoff
        self._budget_bytes = budget_bytes
        self._max_active_fraction = float(max_active_fraction)
        self._basis_generation = grid.basis_generation
        self._geometry_generation = grid.geometry_generation
        self._identity = canonical_hash(
            {
                "schema": "generativeqc.resident-ao-domain.v1",
                "domain": asdict(domain),
                "basis_generation": self._basis_generation,
                "geometry_generation": self._geometry_generation,
                "cutoff": self.cutoff,
                "producer": (
                    "pre-ao-envelope"
                    if producer == "pre-ao-envelope-native-csr"
                    else producer
                ),
                "max_active_fraction": self._max_active_fraction,
            }
        )
        self._info: dict[str, int] = {}
        self._ready = False
        self._occupancy_declined = False
        self._allocation_declined = False
        self._fresh_discovery = True
        started = perf_counter()
        with grid._lock:
            grid._check_open()
            try:
                self._info = grid.prepare_ao_map_device_points(
                    domain.point_pointer,
                    domain.point_count,
                    cutoff=self.cutoff,
                    budget_bytes=budget_bytes,
                    identity=self.identity,
                    **(
                        {"exact": True}
                        if producer == "exact-jets-native-bitmask"
                        else {}
                    ),
                )
            except NotImplementedError:
                self._capability_missing = True
            except CudaAoMapAllocationError:
                self._capability_missing = False
                self._allocation_declined = True
            else:
                self._capability_missing = False
                self._ready = bool(self._info["ready"])
                if self._ready:
                    # Occupancy is a scheduling guard, never an extra AO cutoff.
                    # Declining the entire inventory evaluates all AOs instead.
                    capacity = self._info["map_tiles"] * grid.plan.nao
                    self._occupancy_declined = (
                        self._info["map_entries"] / capacity > self._max_active_fraction
                    )
                    self._ready = not self._occupancy_declined
        self._discovery_seconds = perf_counter() - started
        self.reset_work()

    @property
    def domain(self) -> ResidentAoMapDomain:
        return self._domain

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

    @property
    def identity(self) -> str:
        return self._identity

    def reset_work(self) -> None:
        """Report discovery on the creating endpoint only, never on replay."""
        fresh = self._fresh_discovery
        self._discovery_in_endpoint = fresh
        self._work: dict[str, typing.Any] = {
            "discovery_producer": self._producer,
            "discoveries": int(
                fresh and bool(self._info.get("discovery_d2h_bytes", 0))
            ),
            "discovery_seconds": self._discovery_seconds if fresh else 0.0,
            "discovery_ao_jet_values": self._info.get("discovery_ao_jet_values", 0)
            if fresh
            else 0,
            "discovery_ao_panel_write_bytes": 0,
            "map_compaction_launches": 0,
            "discovery_region_bounds": self._info.get("discovery_region_bounds", 0)
            if fresh
            else 0,
            "discovery_d2h_bytes": self._info.get("discovery_d2h_bytes", 0)
            if fresh
            else 0,
            "discovery_offsets_h2d_bytes": self._info.get(
                "discovery_offsets_h2d_bytes", 0
            )
            if fresh
            else 0,
            "ao_map_h2d_bytes": 0,
            "host_ao_label_lookups": 0,
            "tile_count": 0,
            "cache_hits": 0,
            "dense_budget_tiles": 0,
            "dense_capability_tiles": 0,
            "dense_occupancy_tiles": 0,
            "dense_allocation_tiles": 0,
            "point_ao_visits": 0,
            "point_ao_square_sum": 0,
            "dense_point_ao_square_sum": 0,
            "active_aos_min": None,
            "active_aos_max": 0,
            "active_aos_sum": 0,
            "empty_tile_count": 0,
        }

    @property
    def work(self) -> dict[str, typing.Any]:
        """Detach truthful traffic and resident numeric-capacity diagnostics."""
        return dict(
            self._work,
            retained_map_bytes=self._info.get("retained_map_bytes", 0),
            numeric_peak_bound_bytes=self._info.get("numeric_peak_bound_bytes", 0),
            budget_bytes=self.budget_bytes,
            map_entries=self._info.get("map_entries", 0),
            map_tiles=self._info.get("map_tiles", 0),
            native_csr_ready=self._ready,
            native_map_storage=(
                "bitmask-rebased"
                if self._producer == "exact-jets-native-bitmask"
                else "csr"
            ),
            max_active_fraction=self._max_active_fraction,
            occupancy_declined=self._occupancy_declined,
        )

    @contextmanager
    def feature_task(
        self,
        grid: typing.Any,
        domain: ResidentAoMapDomain,
        begin: int,
        count: int,
        ingredients: typing.Iterable[str],
        *,
        stamp: typing.Any = None,
    ) -> typing.Any:
        """Borrow one native span; stale identity never falls back silently."""
        with grid._lock:
            ResidentAoMapCache._validate_binding(self, grid, domain)
            if (
                type(begin) is not int
                or type(count) is not int
                or begin < 0
                or begin >= domain.point_count
                or begin % domain.tile_points
                or count != min(domain.tile_points, domain.point_count - begin)
            ):
                raise ValueError(
                    "resident AO lookup differs from its point tile domain"
                )
            point_pointer = domain.point_pointer + 24 * begin
            if self._ready:
                lease = grid.feature_task_resident_ao_map(
                    point_pointer,
                    count,
                    begin,
                    ingredients,
                    identity=self.identity,
                    stamp=stamp,
                )
                self._work["cache_hits"] += int(not self._discovery_in_endpoint)
            else:
                lease = grid.feature_task_device_points(
                    point_pointer,
                    count,
                    None,
                    ingredients,
                    stamp=stamp,
                )
                self._work[
                    "dense_occupancy_tiles"
                    if self._occupancy_declined
                    else "dense_allocation_tiles"
                    if self._allocation_declined
                    else "dense_capability_tiles"
                    if self._capability_missing
                    else "dense_budget_tiles"
                ] += 1
            self._work["tile_count"] += 1
            with lease as task:
                active = task.layout.nactive
                if (
                    self._ready
                    and self._producer == "exact-jets-native-bitmask"
                    and 0 < active < grid.plan.nao
                ):
                    self._work["map_compaction_launches"] += 1
                self._work["point_ao_visits"] += count * active
                self._work["point_ao_square_sum"] += count * active**2
                self._work["dense_point_ao_square_sum"] += count * grid.plan.nao**2
                self._work["active_aos_sum"] += active
                prior = self._work["active_aos_min"]
                self._work["active_aos_min"] = (
                    active if prior is None else min(prior, active)
                )
                self._work["active_aos_max"] = max(self._work["active_aos_max"], active)
                self._work["empty_tile_count"] += int(not active)
                yield task
            self._fresh_discovery = False
