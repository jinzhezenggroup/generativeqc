"""Shared D3 sizing authority, legacy planning and prepared-owner boundaries."""

import json
import typing
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import Calculator, D3CorrectionBatch, _native, resources_d3
from generativeqc_compiler.common.resources import ResourceBudget, plan_resources
from generativeqc_compiler.method import resolve_method


@pytest.fixture(scope="module")
def library() -> typing.Any:
    """The source build must expose the additive bridge, not skip qualification."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_d3_layout_v1")
    return loaded


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
@pytest.mark.parametrize("counts", ((1,), (2,), (2, 1, 4), (1,) * 4100, (4096, 1)))
def test_native_matches_independent_portable_inventory_without_provider_loading(
    library: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    counts: tuple[int, ...],
) -> None:
    """CUDA is a shape selector here; no actual CUDA work is run by this test."""
    expected = resources_d3._portable_d3_inventory(counts, backend)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("native D3 sizing used a portable layout or loaded a provider")

    monkeypatch.setattr(resources_d3, "_portable_d3_inventory", forbidden)
    monkeypatch.setattr(resources_d3, "byte_product", forbidden)
    monkeypatch.setattr(_native, "load_library", forbidden)
    assert resources_d3._d3_inventory(counts, backend, library) == expected


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
def test_public_native_and_old_library_keep_bounds_but_distinct_provenance(
    library: typing.Any, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    graph = resolve_method("PBE-D3(BJ)", spin="unpolarized")
    requests = [
        resources_d3.d3_resource_request(
            ((1, 1), (8, 1, 1)), method=graph, backend=backend, library=owner
        )
        for owner in (library, SimpleNamespace(), None)
    ]
    assert requests[0].candidates == requests[1].candidates == requests[2].candidates
    controls = [json.loads(request.identity.schedule) for request in requests]
    assert controls[0]["inventory_owner"] == "native-d3-layout-v1"
    assert (
        controls[1]["inventory_owner"]
        == controls[2]["inventory_owner"]
        == "portable-d3-layout-v1"
    )
    assert requests[0].identity.identity != requests[1].identity.identity
    assert requests[1].identity == requests[2].identity

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("dry-run planning loaded a native provider")

    monkeypatch.setattr(_native, "load_library", forbidden)
    assert (
        resources_d3.d3_resource_request(
            ((1, 1), (8, 1, 1)), method=graph, backend=backend
        )
        == requests[2]
    )


@pytest.mark.parametrize("failure", ("reject", "inconsistent", "overflow"))
def test_discovered_native_failure_never_retries_portable(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    class Query:
        def __call__(self, *arguments: typing.Any) -> int:
            if failure == "reject":
                return -1
            values = arguments[-2]
            values[0] = 1 if failure == "inconsistent" else 1 << 63
            return 0

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("authoritative native failure retried a portable layout")

    monkeypatch.setattr(resources_d3, "_portable_d3_inventory", forbidden)
    with pytest.raises(ValueError, match="native D3|D3 plan_host_bytes"):
        resources_d3.d3_resource_request(
            ((1, 1),),
            method=resolve_method("PBE-D3(BJ)", spin="unpolarized"),
            library=SimpleNamespace(generativeqc_resource_d3_layout_v1=Query()),
        )


@pytest.mark.parametrize("systems", (((1, 87),), ((1,) * 4097,)))
def test_unsupported_domain_precedes_any_native_capacity_query(
    monkeypatch: pytest.MonkeyPatch, systems: tuple[tuple[int, ...], ...]
) -> None:
    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("unsupported D3 systems reached capacity sizing")

    monkeypatch.setattr(resources_d3, "_d3_inventory", forbidden)
    request = resources_d3.d3_resource_request(
        systems,
        method=resolve_method("PBE-D3(BJ)", spin="unpolarized"),
        library=SimpleNamespace(generativeqc_resource_d3_layout_v1=forbidden),
    )
    assert request.unsupported_reason is not None and not request.candidates


@pytest.mark.parametrize("backend", ("cpu", "cuda"))
def test_exact_and_below_local_and_global_capacity_gates(
    library: typing.Any, backend: str
) -> None:
    graph = resolve_method("PBE-D3(BJ)", spin="unpolarized")
    options = {"method": graph, "backend": backend, "library": library}
    request = resources_d3.d3_resource_request(((1, 1), (8, 1, 1)), **options)
    inventory = resources_d3._d3_inventory((2, 3), backend, library)
    for delta in (-1, 0):
        capped = resources_d3.d3_resource_request(
            ((1, 1), (8, 1, 1)),
            maximum_bytes=inventory["retained_bytes"] + delta,
            **options,
        )
        assert bool(capped.candidates) == (delta == 0)
        assert (capped.infeasible_reason is None) == (delta == 0)
    host = inventory["plan_host_bytes"] + inventory["execution_host_bytes"]
    for delta in (-1, 0):
        plan = plan_resources(
            (request,),
            ResourceBudget(
                host_bytes=host + delta, device_bytes=inventory["device_bytes"]
            ),
        )
        assert (plan.status == "feasible") == (delta == 0)
        if backend == "cuda":
            plan = plan_resources(
                (request,),
                ResourceBudget(
                    host_bytes=host, device_bytes=inventory["device_bytes"] + delta
                ),
            )
            assert (plan.status == "feasible") == (delta == 0)


def test_native_budget_matches_ragged_prepared_owner_replay_and_recovery(
    library: typing.Any,
) -> None:
    """Capacity changes preserve moved-geometry, failed-item and owner lifetimes."""
    graph = resolve_method("PBE-D3(BJ)", spin="unpolarized")
    systems = [
        ([1, 1], [[0, 0, -0.7], [0, 0, 0.7]]),
        ([8, 1, 1], [[0, 0, 0], [1.4, 0, 0], [0, 1.4, 0]]),
    ]
    inventory = resources_d3._d3_inventory((2, 3), "cpu", library)
    with pytest.raises(RuntimeError, match="maximum_bytes"):
        D3CorrectionBatch(graph, systems, maximum_bytes=inventory["retained_bytes"] - 1)
    with D3CorrectionBatch(
        graph, systems, maximum_bytes=inventory["retained_bytes"]
    ) as batch:
        diagnostic = batch.diagnostic()
        for key in resources_d3._D3_RESOURCE_KEYS[:-1]:
            assert getattr(diagnostic, key) == inventory[key]
        cold = batch.execute()
        warm = batch.execute()
        assert all(result.ok for result in cold + warm)
        moved = np.asarray(systems[0][1], dtype=float)
        moved[1, 0] += 0.1
        replay = batch.execute([moved, None])
        assert replay[0].energy != cold[0].energy
        failed = moved.copy()
        failed[0, 0] = np.nan
        isolated = batch.execute([failed, None])
        assert not isolated[0].ok and isolated[1].ok
        recovered = batch.execute([moved, None])
        assert recovered[0].ok
        assert recovered[0].energy == replay[0].energy
        np.testing.assert_array_equal(recovered[0].gradient, replay[0].gradient)
        assert all(result.gradient is None for result in batch.execute(gradients=False))
    with pytest.raises(RuntimeError, match="closed"):
        batch.execute()


def test_calculator_passes_loaded_library_to_composed_d3_plan() -> None:
    graph = resolve_method("PBE-D3(BJ)", spin="unpolarized")
    calculator = Calculator(method=graph)
    plan = calculator.estimate_resources([[("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]])
    request = next(request for request in plan.requests if request.name == "d3")
    assert (
        json.loads(request.identity.schedule)["inventory_owner"]
        == "native-d3-layout-v1"
    )
