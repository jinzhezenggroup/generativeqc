"""Device-free identity/lifetime gates for the native CSR lease adapter."""

import typing
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from generativeqc._resident_ao_maps import ResidentDeviceAoMapOwner
from generativeqc_compiler.dft.cuda import CudaAoMapAllocationError
from generativeqc_compiler.dft.indexed_layout import AoGridBlockLayout
from test_resident_ao_map_cache import Grid, domain


class CsrGrid(Grid):
    """Record native-domain operations, without providing any host AO labels."""

    ready = True
    failure = None
    map_entries = 9

    def prepare_ao_map_device_points(
        self,
        pointer: int,
        count: int,
        **kwargs: typing.Any,
    ) -> dict[str, int]:
        self.calls.append(("prepare", pointer, count, kwargs))
        if self.failure is not None:
            raise self.failure
        self.identity = kwargs["identity"]
        return {
            "ready": int(self.ready),
            "retained_map_bytes": 64 if self.ready else 0,
            "numeric_peak_bound_bytes": 256,
            "map_entries": self.map_entries,
            "map_tiles": 3,
            "discovery_d2h_bytes": 36,
            "discovery_offsets_h2d_bytes": 32,
            "discovery_region_bounds": 30,
        }

    @contextmanager
    def feature_task_resident_ao_map(
        self,
        pointer: int,
        count: int,
        begin: int,
        ingredients: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Iterator[SimpleNamespace]:
        assert kwargs["identity"] == self.identity
        self.calls.append(("resident", pointer, count, begin))
        yield SimpleNamespace(
            layout=AoGridBlockLayout(
                10,
                3,
                count,
                2,
                "basis",
                True,
                2,
                begin,
                0,
                0,
            )
        )

    @contextmanager
    def feature_task_device_points(
        self,
        pointer: int,
        count: int,
        ids: typing.Any,
        ingredients: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Iterator[SimpleNamespace]:
        assert ids is None
        self.calls.append(("dense", pointer, count))
        yield SimpleNamespace(
            layout=AoGridBlockLayout(
                10,
                10,
                count,
                2,
                "basis",
                False,
                point_start=(pointer - 8192) // 24,
            )
        )


def owner(grid: CsrGrid) -> ResidentDeviceAoMapOwner:
    return ResidentDeviceAoMapOwner(grid, domain(), cutoff=1e-16, budget_bytes=512)


def test_exact_bitmask_producer_has_distinct_identity_and_discovery_counters() -> None:
    """Exact and conservative inventories may not share immutable map epochs."""
    grid = CsrGrid()
    envelope = owner(grid)
    maps = ResidentDeviceAoMapOwner(
        grid,
        domain(),
        cutoff=1e-16,
        budget_bytes=512,
        producer="exact-jets-native-bitmask",
    )
    assert maps.identity != envelope.identity
    assert grid.calls[-1][3]["exact"] is True
    assert maps.work["discovery_producer"] == "exact-jets-native-bitmask"
    with maps.feature_task(grid, domain(), 0, 4, ("rho",)):
        pass
    assert maps.work["map_compaction_launches"] == 1
    maps.reset_work()
    assert maps.work["discoveries"] == 0
    assert maps.work["map_compaction_launches"] == 0


def test_high_occupancy_declines_the_whole_domain_without_truncating_ao_labels() -> (
    None
):
    grid = CsrGrid()
    grid.map_entries = 27
    maps = ResidentDeviceAoMapOwner(
        grid, domain(), cutoff=1e-16, budget_bytes=512, max_active_fraction=0.8
    )
    assert maps.work["occupancy_declined"]
    assert not maps.work["native_csr_ready"]
    with maps.feature_task(grid, domain(), 0, 4, ("rho",)) as task:
        assert task.layout.nactive == grid.plan.nao
    assert maps.work["dense_occupancy_tiles"] == 1
    assert maps.work["dense_budget_tiles"] == maps.work["dense_capability_tiles"] == 0
    assert maps.work["retained_map_bytes"] == 64
    assert not any(call[0] == "resident" for call in grid.calls)


@pytest.mark.parametrize(
    "limit", [0, -1, 1.01, float("nan"), float("inf"), True, "0.8"]
)
def test_invalid_occupancy_limits_fail_before_native_preparation(
    limit: typing.Any,
) -> None:
    grid = CsrGrid()
    with pytest.raises(ValueError, match="occupancy"):
        ResidentDeviceAoMapOwner(
            grid, domain(), cutoff=1e-16, budget_bytes=512, max_active_fraction=limit
        )
    assert not grid.calls


def test_native_csr_replay_has_no_host_label_inventory_or_transfer() -> None:
    grid = CsrGrid()
    maps = owner(grid)
    assert not hasattr(maps, "_maps") and not hasattr(maps, "select_block")
    for begin, count in ((0, 4), (4, 4), (8, 2)):
        with maps.feature_task(grid, domain(), begin, count, ("rho",)) as task:
            task.layout.require_derivative_order(2)
            assert task.layout.point_start == begin
    assert maps.work["discoveries"] == 1
    assert maps.work["discovery_region_bounds"] == 30
    assert maps.work["discovery_ao_jet_values"] == 0
    assert maps.work["ao_map_h2d_bytes"] == 0
    assert maps.work["host_ao_label_lookups"] == 0
    assert maps.work["cache_hits"] == 0
    maps.reset_work()
    with maps.feature_task(grid, domain(), 4, 4, ("rho",)):
        pass
    assert maps.work["cache_hits"] == 1
    assert maps.work["discoveries"] == 0
    assert maps.work["discovery_seconds"] == 0
    assert maps.work["discovery_d2h_bytes"] == 0
    assert maps.work["discovery_offsets_h2d_bytes"] == 0
    assert sum(call[0] == "prepare" for call in grid.calls) == 1


@pytest.mark.parametrize("attribute", ["domain", "cutoff", "budget_bytes", "identity"])
def test_native_domain_metadata_cannot_be_relabelled(attribute: str) -> None:
    maps = owner(CsrGrid())
    with pytest.raises(AttributeError):
        setattr(maps, attribute, None)


@pytest.mark.parametrize(
    "attribute", ["basis_generation", "geometry_generation", "device_id"]
)
def test_rebind_is_fail_closed_before_native_execution(attribute: str) -> None:
    grid = CsrGrid()
    maps = owner(grid)
    setattr(grid, attribute, 1)
    with (
        pytest.raises(ValueError, match="binding mismatch"),
        maps.feature_task(
            grid,
            domain(),
            0,
            4,
            ("rho",),
        ),
    ):
        pass
    assert len(grid.calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("geometry_identity", "moved"),
        ("grid_identity", "reordered"),
        ("point_pointer", 9216),
        ("derivative_order", 1),
    ],
)
def test_scientific_domain_mismatch_cannot_be_hidden_by_shape(
    field: str, value: typing.Any
) -> None:
    grid = CsrGrid()
    maps = owner(grid)
    with (
        pytest.raises(ValueError, match="binding mismatch"),
        maps.feature_task(
            grid,
            replace(domain(), **{field: value}),
            0,
            4,
            ("rho",),
        ),
    ):
        pass


@pytest.mark.parametrize("begin,count", [(1, 4), (0, 3), (8, 4), (10, 1), (-1, 4)])
def test_native_tiles_keep_exact_point_order_and_tail_shape(
    begin: int, count: int
) -> None:
    grid = CsrGrid()
    maps = owner(grid)
    with (
        pytest.raises(ValueError, match="point tile domain"),
        maps.feature_task(
            grid,
            domain(),
            begin,
            count,
            ("rho",),
        ),
    ):
        pass


@pytest.mark.parametrize("capability", [False, True])
def test_budget_and_capability_misses_keep_dense_endpoint(capability: bool) -> None:
    grid = CsrGrid()
    grid.ready = False
    if capability:
        grid.failure = NotImplementedError("old artifact")
    maps = owner(grid)
    with maps.feature_task(grid, domain(), 0, 4, ("rho",)) as task:
        assert task.layout.nactive == 10 and not task.layout.indexed
    assert (
        maps.work["dense_capability_tiles" if capability else "dense_budget_tiles"] == 1
    )
    assert maps.work["retained_map_bytes"] == 0


def test_native_failure_never_becomes_a_dense_numerical_success() -> None:
    grid = CsrGrid()
    grid.failure = RuntimeError("nonfinite resident points")
    with pytest.raises(RuntimeError, match="nonfinite"):
        owner(grid)


def test_optional_native_allocation_failure_retains_dense_execution() -> None:
    grid = CsrGrid()
    grid.failure = CudaAoMapAllocationError("optional CSR resource allocation")
    maps = owner(grid)
    with maps.feature_task(grid, domain(), 0, 4, ("rho",)) as task:
        assert task.layout.nactive == grid.plan.nao
    assert maps.work["dense_allocation_tiles"] == 1
    assert maps.work["dense_budget_tiles"] == maps.work["dense_capability_tiles"] == 0


def test_unclassified_python_memory_error_is_not_a_native_dense_success() -> None:
    grid = CsrGrid()
    grid.failure = MemoryError("unclassified Python failure")
    with pytest.raises(MemoryError):
        owner(grid)
