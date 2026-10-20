"""Compiled CUDA HF composition with synthetic leaves, not GPU execution evidence."""

import json
import typing
from types import SimpleNamespace

import pytest
from generativeqc import ResourceBudget, _native, estimate_hf_resources, resources_hf

H2 = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
WATER = [(8, (0.0, 0.0, 0.0)), (1, (1.43, 0.0, 1.11)), (1, (-1.43, 0.0, 1.11))]


@pytest.fixture(scope="module")
def library() -> typing.Any:
    """Require the real CPU-compilable bridge; missing symbols are not skips."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_hf_cuda_fleet_v1")
    return loaded


def topology(aos: int, cartesian: int, spins: int) -> dict[str, typing.Any]:
    return {
        "atoms": 2,
        "spins": spins,
        "orbital": {
            "nbf": aos,
            "cartesian_nbf": cartesian,
            "shells": 2,
            "primitives": 6,
        },
    }


def layout_library(library: typing.Any = None) -> typing.Any:
    """Replace only the GPU allocator leaf, retaining the actual fleet ABI."""

    def layout(*args: typing.Any) -> int:
        aos, history, spins = args[0], args[6], args[7]
        output, count = args[-2:]
        assert count == 2
        output[0] = 4096 + 128 * aos + 16 * history + 32 * spins
        output[1] = 256 + 64 * aos
        return 0

    selected = SimpleNamespace(generativeqc_resource_small_hf_cuda_v2=layout)
    if library is not None:
        selected.generativeqc_resource_hf_cuda_fleet_v1 = (
            library.generativeqc_resource_hf_cuda_fleet_v1
        )
    return selected


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("aos,cartesian", ((2, 2), (5, 6), (7, 8), (16, 20)))
@pytest.mark.parametrize("ecp", (False, True))
def test_compiled_ragged_fleet_matches_independent_compatibility_bound(
    library: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    spins: int,
    aos: int,
    cartesian: int,
    ecp: bool,
) -> None:
    """Separate arena/ECP maxima must survive public/Cartesian and spin changes."""
    items = (topology(aos, cartesian, spins), topology(2, 2, 1))
    capacities = (
        {"arena": 8192, "host_staging": 2048, "ecp_workspace": 0},
        {"arena": 1024, "host_staging": 512, "ecp_workspace": 16384 if ecp else 0},
    )
    expected = tuple(
        resources_hf._portable_small_cuda_item_inventory(item, capacity)
        for item, capacity in zip(items, capacities, strict=True)
    )

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("selected native fleet retried portable capacity arithmetic")

    monkeypatch.setattr(resources_hf, "_portable_small_cuda_item_inventory", forbidden)
    actual, resident, retry, host = resources_hf._native_cuda_fleet_inventory(
        library, items, capacities
    )
    assert actual == expected
    assert resident == 9216
    assert retry == 8192 + (16384 if ecp else 0)
    assert host == sum(item["host"] for item in expected)
    reversed_items, *reversed_totals = resources_hf._native_cuda_fleet_inventory(
        library, items[::-1], capacities[::-1]
    )
    assert reversed_items == expected[::-1]
    assert reversed_totals == [resident, retry, host]


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("history", (0, 1, 8, 64))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_estimator_uses_compiled_fleet_without_portable_math(
    library: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    history: int,
    representation: str,
) -> None:
    options = {
        "backend": "cuda",
        "method": method,
        "diis_history": history,
        "basis_representation": representation,
    }
    expected = estimate_hf_resources(
        [H2, WATER, H2], library=layout_library(), **options
    ).require_feasible()

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("native estimator evaluated the compatibility host polynomial")

    monkeypatch.setattr(resources_hf, "_portable_small_cuda_item_inventory", forbidden)
    actual = estimate_hf_resources(
        [H2, WATER, H2], library=layout_library(library), **options
    ).require_feasible()
    assert actual.peak_bytes == expected.peak_bytes
    assert actual.resident_bytes == expected.resident_bytes
    assert actual.requests[0].candidates == expected.requests[0].candidates
    schedule = json.loads(actual.requests[0].identity.schedule)
    assert schedule.pop("cuda_hf_fleet_composition") == "native-declared-capacity"
    assert schedule == json.loads(expected.requests[0].identity.schedule)


@pytest.mark.parametrize("space", ("host", "device"))
def test_compiled_inventory_preserves_exact_and_one_byte_short_admission(
    library: typing.Any, space: str
) -> None:
    selected = layout_library(library)
    options = {"backend": "cuda", "library": selected}
    baseline = estimate_hf_resources([H2, WATER], **options).require_feasible()
    for difference in (0, -1):
        limits = {
            name + "_bytes": baseline.peak_bytes[name] for name in ("host", "device")
        }
        limits[space + "_bytes"] += difference
        plan = estimate_hf_resources(
            [H2, WATER], budget=ResourceBudget(**limits), **options
        )
        assert plan.status == ("feasible" if difference == 0 else "infeasible")
        assert plan.peak_bytes == baseline.peak_bytes


@pytest.mark.parametrize("failure", ("refusal", "peak", "item"))
def test_discovered_failure_is_authoritative_not_a_compatibility_retry(
    library: typing.Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    actual = library.generativeqc_resource_hf_cuda_fleet_v1
    selected = layout_library(library)

    def damaged(*args: typing.Any) -> int:
        if failure == "refusal":
            return -1
        status = actual(*args)
        assert status == 0
        values = args[2]
        if failure == "peak":
            values[2] += 1
        else:
            values[5] = 0
        return 0

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("native failure silently restarted portable sizing")

    selected.generativeqc_resource_hf_cuda_fleet_v1 = damaged
    monkeypatch.setattr(resources_hf, "_portable_small_cuda_item_inventory", forbidden)
    with pytest.raises(ValueError, match="native CUDA HF fleet inventory"):
        estimate_hf_resources([H2, WATER], backend="cuda", library=selected)
