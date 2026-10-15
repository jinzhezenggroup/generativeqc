"""Independent portable/native composition and production admission contracts."""

import json
import math
import random
import typing
from types import SimpleNamespace

import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    _native,
    estimate_hf_resources,
    estimate_ks_resources,
    plan_resources,
    resources_planning,
)
from generativeqc_compiler.common.resources import (
    MAX_BYTES,
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourcePlan,
    ResourceRequest,
    _account,
)
from generativeqc_compiler.common.resources import (
    plan_resources as portable_plan_resources,
)

H2 = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]


@pytest.fixture(scope="module")
def library() -> typing.Any:
    """Require the source ABI instead of skipping the ownership cutover."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_compose_v1")
    return loaded


def request(name: str, candidates: tuple[ResourceCandidate, ...]) -> ResourceRequest:
    return ResourceRequest(
        name,
        ResourceIdentity(
            "test", name, "cpu", "fp64", json.dumps({"shape": 1}), ("energy",), "fixed"
        ),
        candidates,
    )


def candidate(name: str, cost: int, *buffers: ResourceEstimate) -> ResourceCandidate:
    return ResourceCandidate(name, "resident", tuple(buffers), relative_cost=cost)


def buffer(
    name: str,
    size: int,
    space: str = "pageable",
    first: int = 0,
    last: int = 0,
    kind: str = "persistent",
) -> ResourceEstimate:
    return ResourceEstimate(name, size, space, first, last, kind=kind)


@pytest.mark.parametrize("seed", range(100))
def test_native_composition_matches_independent_portable_model(
    library: typing.Any, monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    """Ragged lifetimes, device labels, costs and simultaneous coupled limits."""
    generator = random.Random(seed)
    phases = (0, 1, 9, MAX_BYTES - 1, MAX_BYTES)
    spaces = ("pageable", "pinned", "device:0", "device:2", "device:10", "device:1")
    kinds = ("persistent", "workspace", "output", "cache", "library", "runtime")
    requests = []
    for owner in range(generator.randint(1, 3)):
        choices = []
        for choice in range(generator.randint(1, 3)):
            estimates = []
            for index in range(generator.randint(1, 5)):
                first, last = sorted(generator.choices(phases, k=2))
                estimates.append(
                    buffer(
                        f"buffer-{index}",
                        generator.randrange(300),
                        generator.choice(spaces),
                        first,
                        last,
                        generator.choice(kinds),
                    )
                )
            choices.append(
                candidate(
                    f"choice-{choice}",
                    generator.choice((0, 1, 4, MAX_BYTES)),
                    *estimates,
                )
            )
        requests.append(request(f"owner-{owner}", tuple(choices)))
    requests = tuple(reversed(requests))
    budget = ResourceBudget(
        host_bytes=generator.choice((None, 0, 800, MAX_BYTES)),
        device_bytes=generator.choice((None, 0, 900, MAX_BYTES)),
        pinned_host_bytes=generator.choice((None, 0, 500)),
        per_device_bytes=(
            (2, generator.randrange(1000)),
            (10, generator.randrange(1000)),
        ),
        host_reserve_bytes=generator.randrange(50),
        device_reserve_bytes=generator.randrange(50),
        headroom_fraction=generator.choice(
            (0, 0.1, 0.5, math.ulp(0.0), math.nextafter(1.0, 0.0))
        ),
    )
    expected = portable_plan_resources(requests, budget)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail(
            "native composition retried portable selection or loaded a provider"
        )

    monkeypatch.setattr(resources_planning, "_portable_plan_resources", forbidden)
    monkeypatch.setattr(_native, "load_library", forbidden)
    actual = resources_planning.plan_resources(requests, budget, library=library)
    assert actual.to_dict() == expected.to_dict()
    normalized = tuple(sorted(requests, key=lambda item: item.name))
    receipt = resources_planning._native_composition(
        normalized, budget, library.generativeqc_resource_compose_v1, 65536
    )
    assert receipt["peak_bytes"] == expected.peak_bytes
    assert receipt["resident_bytes"] == expected.resident_bytes
    assert receipt["phase_peaks"] == _account(expected.estimates)[1]
    assert (
        receipt["resident_phase_peaks"]
        == _account(expected.estimates, resident=True)[1]
    )
    assert receipt["limits"] == budget.limits()


@pytest.mark.parametrize("kind", ("workspace", "persistent"))
@pytest.mark.parametrize("zero_candidate_host", (0, 100))
def test_zero_device_buffers_preserve_portable_presence_and_diagnostics(
    library: typing.Any, kind: str, zero_candidate_host: int
) -> None:
    """A zero-byte device label is absent from portable peaks and their minima."""
    requests = (
        request(
            "owner",
            (
                candidate(
                    "zero",
                    0,
                    buffer("host", zero_candidate_host),
                    buffer("device", 0, "device:0", kind=kind),
                ),
                candidate(
                    "nonzero",
                    0,
                    buffer("host", 0),
                    buffer("device", 2, "device:0", kind=kind),
                ),
            ),
        ),
    )
    budget = ResourceBudget(host_bytes=0, per_device_bytes=((0, 0),))
    expected = portable_plan_resources(requests, budget)
    actual = resources_planning.plan_resources(requests, budget, library=library)
    assert actual.to_dict() == expected.to_dict()
    receipt = resources_planning._native_composition(
        requests, budget, library.generativeqc_resource_compose_v1, 65536
    )
    assert receipt["peak_bytes"] == expected.peak_bytes
    assert receipt["resident_bytes"] == expected.resident_bytes
    assert receipt["minima"]["device:0"] == 2


@pytest.mark.parametrize(
    "fraction", (0, 0.1, 0.5, math.ulp(0.0), math.nextafter(1.0, 0.0))
)
@pytest.mark.parametrize("capacity", (0, 1, 2**53 + 1, MAX_BYTES))
def test_exact_headroom_reserve_and_budget_boundaries(
    library: typing.Any, fraction: float, capacity: int
) -> None:
    budget = ResourceBudget(
        host_bytes=capacity, headroom_fraction=fraction, host_reserve_bytes=17
    )
    limit = budget.limits()["host"]
    choices = (request("owner", (candidate("fixed", 0, buffer("bytes", limit)),)),)
    plan = resources_planning.plan_resources(choices, budget, library=library)
    assert plan.status == "feasible"
    if limit < MAX_BYTES:
        larger = (
            request("owner", (candidate("fixed", 0, buffer("bytes", limit + 1)),)),
        )
        assert (
            resources_planning.plan_resources(larger, budget, library=library).status
            == "infeasible"
        )


def test_uint64_cost_excess_and_product_boundaries(library: typing.Any) -> None:
    choices = (
        request(
            "a",
            (
                candidate("z-cheaper", 1, buffer("bytes", 1)),
                candidate("a-costlier", 2, buffer("bytes", 1)),
            ),
        ),
        request("b", (candidate("fixed", MAX_BYTES, buffer("bytes", 1)),)),
        request("c", (candidate("fixed", MAX_BYTES, buffer("bytes", 1)),)),
    )
    actual = resources_planning.plan_resources(
        choices, ResourceBudget(), library=library
    )
    assert dict(actual.selections)["a"] == "z-cheaper"
    assert (
        actual.to_dict() == portable_plan_resources(choices, ResourceBudget()).to_dict()
    )
    excessive = (
        request(
            "a",
            tuple(
                candidate(
                    name,
                    0,
                    buffer("host", MAX_BYTES - adjustment),
                    buffer("device0", MAX_BYTES, "device:0", 1, 1),
                    buffer("device1", MAX_BYTES, "device:1", 2, 2),
                    buffer("device2", MAX_BYTES, "device:2", 3, 3),
                )
                for name, adjustment in (("a-larger", 0), ("z-smaller", 1))
            ),
        ),
    )
    budget = ResourceBudget(
        host_bytes=0, device_bytes=0, per_device_bytes=((0, 0), (1, 0), (2, 0))
    )
    assert (
        resources_planning.plan_resources(excessive, budget, library=library).to_dict()
        == portable_plan_resources(excessive, budget).to_dict()
    )
    product = tuple(
        request(
            str(index),
            (
                candidate("a", 0, buffer("zero", 0)),
                candidate("b", 0, buffer("zero", 0)),
            ),
        )
        for index in range(65)
    )
    limited = resources_planning.plan_resources(
        product, ResourceBudget(), library=library
    )
    assert limited.status == "unsupported" and str(2**65) in limited.diagnostic
    assert (
        limited.to_dict()
        == portable_plan_resources(product, ResourceBudget()).to_dict()
    )


def test_later_overflow_is_not_hidden_by_a_feasible_prefix(library: typing.Any) -> None:
    choices = (
        request(
            "owner",
            (
                candidate("cheap", 0, buffer("small", 1)),
                candidate(
                    "later",
                    1,
                    buffer("pageable", MAX_BYTES),
                    buffer("pinned", 1, "pinned"),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError):
        portable_plan_resources(choices, ResourceBudget())
    with pytest.raises(ValueError, match="native resource composition"):
        resources_planning.plan_resources(choices, ResourceBudget(), library=library)
    assert (
        resources_planning.plan_resources(
            choices, ResourceBudget(), library=library, maximum_combinations=1
        ).status
        == "unsupported"
    )


@pytest.mark.parametrize("status", (1, 2, 3))
def test_native_rejection_never_retries_a_portable_selection(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    class Query:
        def __call__(self, *args: typing.Any) -> int:
            return status

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("native query rejection retried portable policy")

    monkeypatch.setattr(resources_planning, "_portable_plan_resources", forbidden)
    with pytest.raises(
        MemoryError if status == 3 else ValueError, match="native resource composition"
    ):
        resources_planning.plan_resources(
            (request("owner", (candidate("fixed", 0, buffer("bytes", 1)),)),),
            ResourceBudget(),
            library=SimpleNamespace(generativeqc_resource_compose_v1=Query()),
        )


def test_old_library_and_library_free_planning_remain_provider_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = (request("owner", (candidate("fixed", 0, buffer("bytes", 1)),)),)
    expected = portable_plan_resources(choices, ResourceBudget())

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("portable dry run loaded a native provider")

    monkeypatch.setattr(_native, "load_library", forbidden)
    for library in (None, SimpleNamespace()):
        assert (
            resources_planning.plan_resources(
                choices, ResourceBudget(), library=library
            )
            == expected
        )


def test_explicit_selection_is_validated_without_replacement(
    library: typing.Any,
) -> None:
    choices = (
        request(
            "owner",
            (
                candidate("cheap", 0, buffer("small", 1)),
                candidate("chosen", 10, buffer("larger", 2)),
            ),
        ),
    )
    explicit = ResourcePlan(
        ResourceBudget(host_bytes=2), choices, (("owner", "chosen"),), "feasible"
    )
    original = explicit.to_dict()
    resources_planning.validate_resource_plan(explicit, library=library)
    assert explicit.to_dict() == original
    assert dict(explicit.selections)["owner"] == "chosen"
    assert (
        dict(
            resources_planning.plan_resources(
                choices, explicit.budget, library=library
            ).selections
        )["owner"]
        == "cheap"
    )


@pytest.mark.parametrize("entry", ("public", "hf", "ks", "calculator"))
def test_public_production_entries_use_the_loaded_native_owner(
    library: typing.Any, entry: str
) -> None:
    class Query:
        def __call__(self, *args: typing.Any) -> int:
            return 1

    class Library:
        generativeqc_resource_compose_v1 = Query()

        def __getattr__(self, name: str) -> typing.Any:
            return getattr(library, name)

    proxy = Library()
    with pytest.raises(ValueError, match="native resource composition"):
        if entry == "public":
            choices = (request("owner", (candidate("fixed", 0, buffer("bytes", 1)),)),)
            plan_resources(choices, ResourceBudget(), library=proxy)
        elif entry == "hf":
            estimate_hf_resources([H2], library=proxy)
        elif entry == "ks":
            estimate_ks_resources([H2], library=proxy)
        else:
            calculator = Calculator(method="pbe-rks")
            calculator._library = proxy
            calculator.estimate_resources([H2])


def test_production_supplied_plan_rejection_precedes_scientific_preparation(
    library: typing.Any,
) -> None:
    calculator = Calculator(method="pbe-rks")
    plan = calculator.estimate_resources([H2], properties=("energy",))

    class Query:
        def __call__(self, *args: typing.Any) -> int:
            return 2

    class Library:
        generativeqc_resource_compose_v1 = Query()

        def __getattr__(self, name: str) -> typing.Any:
            if name.startswith(
                ("generativeqc_system_create", "generativeqc_calculation_create")
            ):

                class ForbiddenPreparation:
                    def __call__(self, *args: typing.Any) -> typing.NoReturn:
                        pytest.fail("scientific preparation preceded native admission")

                return ForbiddenPreparation()
            return getattr(library, name)

    calculator._library = Library()
    with pytest.raises(ValueError, match="native resource composition"):
        calculator.prepare_batch([H2], resource_plan=plan)


@pytest.mark.parametrize("mode", ("native-reject", "below-budget", "accepted"))
def test_force_workspace_replay_uses_native_admission_and_preserves_rollback(
    mode: str,
) -> None:
    options = {
        "method": "pbe-rks",
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        "max_iterations": 150,
    }
    probe = Calculator(**options)
    energy = probe.estimate_resources([H2], properties=("energy",))
    forces = probe.estimate_resources([H2], properties=("energy", "forces"))
    assert forces.peak_bytes["host"] > energy.peak_bytes["host"]
    host = (
        energy.peak_bytes["host"]
        if mode == "below-budget"
        else forces.peak_bytes["host"]
    )
    calculator = Calculator(**options, resource_budget=ResourceBudget(host_bytes=host))
    plan = calculator.estimate_resources([H2], properties=("energy",))
    with calculator.prepare_batch([H2], resource_plan=plan) as batch:
        batch.execute(properties=("energy",), strict=True)
        original = batch.resource_plan.to_dict()
        library = batch._library

        class Query:
            def __call__(self, *args: typing.Any) -> int:
                return 2

        class Library:
            generativeqc_resource_compose_v1 = Query()

            def __getattr__(self, name: str) -> typing.Any:
                if name.startswith(
                    ("generativeqc_batch_execute", "generativeqc_ks_batch_execute")
                ):

                    class ForbiddenExecution:
                        def __call__(self, *args: typing.Any) -> typing.NoReturn:
                            pytest.fail("force execution preceded native admission")

                    return ForbiddenExecution()
                return getattr(library, name)

        if mode == "native-reject":
            batch._library = Library()
        if mode != "accepted":
            with pytest.raises(ValueError if mode == "native-reject" else MemoryError):
                batch.execute(properties=("energy", "forces"), strict=True)
            assert batch.resource_plan.to_dict() == original
            batch._library = library
            recovery = batch.execute(properties=("energy",), strict=True).items[0]
            assert recovery.succeeded and recovery.warm_start_used
        else:
            result = batch.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
            assert result.succeeded and result.forces is not None
            assert batch.resource_plan.to_dict() != original
            assert batch.resource_plan.selections == plan.selections
            step = 2e-4
            energies = []
            for direction in (-1, 1):
                moved = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7 + direction * step))]
                energies.append(
                    Calculator(**options)
                    .singlepoint(moved, properties=("energy",))
                    .energy
                )
            assert result.forces[1, 2] == pytest.approx(
                -(energies[1] - energies[0]) / (2 * step), abs=2e-5
            )


def test_cpu_complete_endpoint_retains_exact_budget_and_replay() -> None:
    options = {
        "method": "pbe-rks",
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
        ),
        "max_iterations": 150,
    }
    calculator = Calculator(**options)
    plan = calculator.estimate_resources([H2], properties=("energy",))
    host = plan.peak_bytes["host"]
    exact = Calculator(**options, resource_budget=ResourceBudget(host_bytes=host))
    exact_plan = exact.estimate_resources([H2], properties=("energy",))
    with exact.prepare_batch([H2], resource_plan=exact_plan) as batch:
        cold = batch.execute(properties=("energy",), strict=True).items[0]
        warm = batch.execute(properties=("energy",), strict=True).items[0]
        moved = [[(0, 0, -0.75), (0, 0, 0.75)]]
        replay = batch.execute(moved, properties=("energy",), strict=True).items[0]
    assert cold.succeeded and warm.succeeded and replay.succeeded
    assert cold.energy == pytest.approx(warm.energy, abs=1e-10)
    independent = Calculator(**options).singlepoint(
        [("H", moved[0][0]), ("H", moved[0][1])], properties=("energy",)
    )
    assert replay.energy == pytest.approx(independent.energy, abs=1e-10)
    below = Calculator(**options, resource_budget=ResourceBudget(host_bytes=host - 1))
    assert below.estimate_resources([H2], properties=("energy",)).status == "infeasible"
    with pytest.raises(MemoryError):
        below.prepare_batch(
            [H2], resource_plan=below.estimate_resources([H2], properties=("energy",))
        )
