"""Resident reuse must obey each call's admission caps before any device work."""

import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from generativeqc import _stationary_cuda as runtime
from generativeqc_compiler.common.prepared_execution import PreparedArtifactBinding
from generativeqc_compiler.common.provenance import canonical_hash


def test_shared_tensor_artifact_is_bound_once_without_hiding_collisions() -> None:
    first = SimpleNamespace(metadata={"key": "shared", "binary_sha256": "same"})
    second = SimpleNamespace(metadata={"key": "shared", "binary_sha256": "same"})
    other = SimpleNamespace(metadata={"key": "other", "binary_sha256": "other"})
    assert runtime._unique_prepared_artifacts((first, second, other)) == (
        first,
        other,
    )
    conflicting = SimpleNamespace(
        metadata={"key": "shared", "binary_sha256": "different"}
    )
    with pytest.raises(ValueError, match="conflicting binaries"):
        runtime._unique_prepared_artifacts((first, conflicting))


@pytest.mark.parametrize("failure", ("artifact-collision", "lease-install"))
def test_prepared_artifact_resources_close_when_publication_fails(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A failed publish closes both acquired owners in reverse order, without a GPU."""
    owner = runtime.PreparedStationaryCudaExecution()
    monkeypatch.setattr(owner, "_request", lambda **_kwargs: object())
    monkeypatch.setattr(runtime, "_layout", lambda _basis: (None, None, (), None))
    monkeypatch.setattr(runtime, "_component_mode", lambda _expansions: False)
    monkeypatch.setattr(runtime, "stationary_runtime_sources", lambda _plan: ())
    stationary = SimpleNamespace(metadata={"key": "a" * 64, "binary_sha256": "b" * 64})
    grid = SimpleNamespace(
        metadata={
            "key": "a" * 64 if failure == "artifact-collision" else "c" * 64,
            "binary_sha256": "d" * 64,
        }
    )
    monkeypatch.setattr(
        runtime, "compile_stationary_cuda", lambda *_a, **_kw: stationary
    )
    monkeypatch.setattr(runtime, "compile_grid", lambda *_a, **_kw: grid)

    events: list[str] = []

    @contextmanager
    def resource(name: str):
        events.append(f"enter:{name}")
        try:
            yield SimpleNamespace()
        finally:
            events.append(f"exit:{name}")

    monkeypatch.setattr(runtime, "CudaGrid", lambda *_a, **_kw: resource("grid"))
    monkeypatch.setattr(runtime, "_CudaSources", lambda *_a, **_kw: resource("sources"))

    if failure == "lease-install":

        def reject_install(*_args, **_kwargs):
            raise RuntimeError("injected lease installation failure")

        monkeypatch.setattr(owner._lease, "install", reject_install)

    error = ValueError if failure == "artifact-collision" else RuntimeError
    message = (
        "conflicting binaries" if failure == "artifact-collision" else "injected lease"
    )
    with pytest.raises(error, match=message):
        owner.ensure(
            state=SimpleNamespace(),
            basis=SimpleNamespace(identity="geometry", natom=1, nao=1),
            contract=SimpleNamespace(),
            plan=SimpleNamespace(spin_blocks=1),
            tensor_plans={},
            compiler=SimpleNamespace(target=SimpleNamespace()),
            cache="unused",
            requests=(),
            functional=0,
            ecp=False,
            device=0,
            spec=SimpleNamespace(partition_iterations=1),
            grid_plan=SimpleNamespace(peak_bytes=1, allocation_bytes=1),
            source_bytes=1,
            tile_points=8,
            primitive_tile=16,
            integral_terms=2,
            page_work_budget=64,
            max_device_bytes=100,
            max_host_bytes=100,
            host_bound=1,
        )

    assert events == ["enter:grid", "enter:sources", "exit:sources", "exit:grid"]
    assert owner._stack is None
    assert owner._lease.contract is None


@pytest.mark.parametrize(
    "failed,changed", [(False, False), (False, True), (True, False)]
)
def test_retained_host_budget_is_rechecked_before_rebind(
    monkeypatch: pytest.MonkeyPatch, failed: bool, changed: bool
) -> None:
    monkeypatch.setattr(runtime, "_basis_topology_identity", lambda basis: "topology")
    source = SimpleNamespace(backend="cuda")
    identity = SimpleNamespace(
        method="pbe-rks", functional_identity="xc", regularization_identity="reg"
    )
    state = SimpleNamespace(identity=identity, _source=source)
    spec = SimpleNamespace(partition_iterations=3)
    target = SimpleNamespace(to_payload=dict)
    owner = runtime.PreparedStationaryCudaExecution()
    owner._bound_basis_identity = "old"
    owner.artifacts = ()
    owner.sources = SimpleNamespace(
        rebind_geometry=lambda basis: pytest.fail("rebind before admission")
    )
    owner.grid = SimpleNamespace(
        _rebind_centers=lambda centers: pytest.fail("grid rebind before admission")
    )
    basis = SimpleNamespace(
        identity="new" if changed else "old",
        packed=[],
        natom=0,
        nao=0,
    )
    kwargs = {
        "state": state,
        "basis": basis,
        "contract": SimpleNamespace(family="gga", spin="unpolarized"),
        "plan": SimpleNamespace(identity="plan"),
        "tensor_plans": {},
        "compiler": SimpleNamespace(target=target),
        "cache": "unused",
        "requests": (),
        "functional": 1,
        "ecp": False,
        "device": 0,
        "spec": spec,
        "grid_plan": SimpleNamespace(allocation_bytes=10, peak_bytes=10),
        "source_bytes": 20,
        "tile_points": 4,
        "primitive_tile": 5,
        "integral_terms": 6,
        "page_work_budget": 7,
        "max_device_bytes": 30,
        "max_host_bytes": 99,
        "host_bound": 80,
    }
    request = owner._request(
        state=state,
        basis=basis,
        contract=kwargs["contract"],
        plan=kwargs["plan"],
        tensor_plans=kwargs["tensor_plans"],
        target=target,
        aot_directory=None,
        native_grid_library=None,
        ecp=False,
        device=0,
        spec=spec,
        grid_plan=kwargs["grid_plan"],
        source_bytes=20,
        tile_points=4,
        primitive_tile=5,
        integral_terms=6,
        page_work_budget=7,
    )
    pruned_request = owner._request(
        **{
            name: value
            for name, value in kwargs.items()
            if name in inspect.signature(owner._request).parameters
        },
        target=target,
        aot_directory=None,
        native_grid_library=None,
        integral_derivatives=False,
    )
    assert pruned_request.scientific_identity == request.scientific_identity
    assert pruned_request.schedule_identity != request.schedule_identity
    owner._lease.install(
        request,
        (
            PreparedArtifactBinding(
                canonical_hash("artifact-key"), canonical_hash("artifact-binary")
            ),
        ),
        host_bytes=100,
        device_bytes=30,
    )
    if failed:
        owner._lease.mark_failure()
    with pytest.raises(ValueError, match="host budget"):
        owner.ensure(**kwargs)
    # An admission failure must leave an unchanged compatible owner reusable.
    if owner._lease.failed:
        owner._lease.mark_refresh()
    kwargs["basis"] = SimpleNamespace(identity="old", packed=[], natom=0, nao=0)
    kwargs["max_host_bytes"] = 100
    owner.ensure(**kwargs)
    assert owner._bound_basis_identity == "old"
