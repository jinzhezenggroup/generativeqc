"""ECP metadata-only native authority, compatibility and capacity boundaries."""

import ctypes
import json
import typing
from types import SimpleNamespace

import pytest
from generativeqc import (
    BasisProvenance,
    BasisSet,
    BasisShell,
    ElementBasis,
    ResourceBudget,
    _native,
    estimate_hf_resources,
    estimate_ks_resources,
    resources_hf,
)


@pytest.fixture(scope="module")
def library() -> typing.Any:
    """Require the installed scalar query, including in CPU-only builds."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_ecp_workspace_v1")
    return loaded


def item(
    nao: int = 16,
    cartesian_nao: int = 19,
    atoms: int = 2,
    primitives: int = 10,
    terms: int = 5,
) -> dict[str, typing.Any]:
    """Public and Cartesian counts deliberately differ at the CUDA tile gate."""
    return {
        "atoms": atoms,
        "orbital": {
            "nbf": nao,
            "cartesian_nbf": cartesian_nao,
            "primitives": primitives,
            "ecp_terms": terms,
        },
    }


@pytest.mark.parametrize("cuda", (False, True))
@pytest.mark.parametrize(
    "shape",
    (
        (1, 1, 1, 1, 1),
        (16, 19, 2, 10, 5),
        (17, 20, 2, 10, 5),
        (9, 15, 3, 20, 7),
        (50, 56, 4, 80, 12),
    ),
)
def test_native_matches_independent_portable_two_grid_bound(
    library: typing.Any, shape: tuple[int, ...], cuda: bool
) -> None:
    """CUDA here selects scalar scheduling arithmetic, never a CUDA context."""
    topology = item(*shape)
    expected = resources_hf._ecp_workspace(topology, cuda=cuda)
    assert resources_hf._ecp_workspace(topology, cuda=cuda, library=library) == expected


@pytest.mark.parametrize("cuda", (False, True))
def test_native_authority_does_not_retry_portable_or_load_a_provider(
    library: typing.Any, monkeypatch: pytest.MonkeyPatch, cuda: bool
) -> None:
    expected = resources_hf._ecp_workspace(item(), cuda=cuda)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("native ECP sizing retried Python policy or loaded a provider")

    monkeypatch.setattr(resources_hf, "_portable_ecp_workspace", forbidden)
    monkeypatch.setattr(_native, "load_library", forbidden)
    assert resources_hf._ecp_workspace(item(), cuda=cuda, library=library) == expected


@pytest.mark.parametrize("loaded", (None, SimpleNamespace()))
def test_legacy_and_library_free_sizing_never_load_native_code(
    monkeypatch: pytest.MonkeyPatch, loaded: typing.Any
) -> None:
    expected = resources_hf._ecp_workspace(item(), cuda=True)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("portable metadata sizing implicitly loaded native code")

    monkeypatch.setattr(_native, "load_library", forbidden)
    assert resources_hf._ecp_workspace(item(), cuda=True, library=loaded) == expected


def test_all_electron_topology_needs_no_ecp_query_or_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("all-electron topology invoked ECP sizing")

    monkeypatch.setattr(resources_hf, "_portable_ecp_workspace", forbidden)
    loaded = SimpleNamespace(generativeqc_resource_ecp_workspace_v1=forbidden)
    assert resources_hf._ecp_workspace(item(terms=0), library=loaded) == 0
    assert resources_hf._ecp_workspace({"orbital": {}}, library=loaded) == 0


def test_discovered_rejection_does_not_retry_portable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*args: typing.Any) -> int:
        return 1

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("rejected ECP query silently retried portable sizing")

    monkeypatch.setattr(resources_hf, "_portable_ecp_workspace", forbidden)
    loaded = SimpleNamespace(generativeqc_resource_ecp_workspace_v1=reject)
    with pytest.raises(ValueError, match="native ECP workspace inventory rejected"):
        resources_hf._ecp_workspace(item(), library=loaded)


@pytest.mark.parametrize("value", (-1, 1 << 63, True, 1.5))
def test_invalid_shape_is_rejected_before_ctypes_conversion(
    value: typing.Any,
) -> None:
    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("invalid ECP dimension reached the native ABI")

    loaded = SimpleNamespace(generativeqc_resource_ecp_workspace_v1=forbidden)
    with pytest.raises((ValueError, TypeError)):
        resources_hf._ecp_workspace(item(primitives=value), library=loaded)


def test_invalid_native_output_cannot_wrap_into_a_budget() -> None:
    def oversized(*args: typing.Any) -> int:
        ctypes.cast(args[-1], ctypes.POINTER(ctypes.c_uint64))[0] = 1 << 63
        return 0

    loaded = SimpleNamespace(generativeqc_resource_ecp_workspace_v1=oversized)
    with pytest.raises(ValueError, match="ECP two-grid workspace"):
        resources_hf._ecp_workspace(item(), library=loaded)


@pytest.mark.parametrize(
    "shape,cuda",
    (
        ((0, 1, 1, 1, 1), 0),
        ((1, 0, 1, 1, 1), 0),
        ((1, 1, 0, 1, 1), 0),
        ((1, 1, 1, 1, 1), -1),
        ((1, 1, 1, 1, 1), 2),
        ((1, 1 << 62, 1, 1, 1), 0),
        ((1, 1, 1 << 62, 1, 1), 0),
        ((1, 1, 1, 1 << 62, 1), 0),
        ((1, 1, 1, 1, 1 << 62), 1),
    ),
)
def test_native_abi_rejection_preserves_output(
    library: typing.Any, shape: tuple[int, ...], cuda: int
) -> None:
    query = library.generativeqc_resource_ecp_workspace_v1
    query.argtypes = [ctypes.c_size_t] * 5 + [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    query.restype = ctypes.c_int
    output = ctypes.c_uint64(12345)
    assert query(*shape, cuda, ctypes.byref(output)) != 0
    assert output.value == 12345
    assert query(*shape, cuda, None) != 0


@pytest.mark.parametrize("cuda", (False, True))
def test_largest_admissible_term_count_and_next_capacity_overflow(
    library: typing.Any, cuda: bool
) -> None:
    topology = item(nao=1, cartesian_nao=1, atoms=1, primitives=1, terms=1)
    base = resources_hf._ecp_workspace(topology, cuda=cuda) - 256
    terms = ((1 << 63) - 1 - base) // 256
    topology["orbital"]["ecp_terms"] = terms
    expected = resources_hf._ecp_workspace(topology, cuda=cuda)
    assert expected <= (1 << 63) - 1 < expected + 256
    assert resources_hf._ecp_workspace(topology, cuda=cuda, library=library) == expected
    topology["orbital"]["ecp_terms"] += 1
    with pytest.raises(ValueError, match="native ECP workspace inventory rejected"):
        resources_hf._ecp_workspace(topology, cuda=cuda, library=library)


@pytest.fixture
def cuda_layout_library(library: typing.Any) -> typing.Any:
    """Stub CUDA layouts while retaining the real CPU-build ECP scalar ABI."""

    def layout(*args: typing.Any) -> int:
        output, count = args[-2:]
        for index in range(count):
            output[index] = 0
        return 0

    def scalar_layout(*args: typing.Any) -> int:
        return 0

    def version() -> int:
        return 1

    return SimpleNamespace(
        generativeqc_resource_ecp_workspace_v1=library.generativeqc_resource_ecp_workspace_v1,
        generativeqc_resource_small_hf_cuda_v2=layout,
        generativeqc_resource_ks_cuda_v2=layout,
        generativeqc_resource_quadrature_cuda_v1=scalar_layout,
        generativeqc_resource_ks_matrix_provider_cuda_v1=scalar_layout,
        generativeqc_ks_options_version=version,
        generativeqc_ks_resource_inventory_version_v1=version,
    )


def test_cuda_frontend_inventories_use_native_ecp_without_gpu_work(
    cuda_layout_library: typing.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import resources_ks

    topology = item(nao=1, cartesian_nao=1, atoms=1, primitives=1, terms=1)
    topology.update(spins=1, grid_points=1)
    topology["orbital"].update(
        shells=1, shell_angular=(0,), shell_primitive_counts=(1,)
    )
    expected = resources_hf._ecp_workspace(topology, cuda=True)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("CUDA resource adapter retried portable ECP policy")

    monkeypatch.setattr(resources_hf, "_portable_ecp_workspace", forbidden)
    hf = resources_hf._small_cuda_item_inventory(
        cuda_layout_library, topology, 8, "fp64", 1e-10, 1e-12
    )
    assert hf["ecp_workspace"] == expected
    assert (
        resources_hf._portable_small_cuda_item_inventory(topology, hf)["host"]
        >= expected
    )
    ks = resources_ks._cuda_item_inventory(
        cuda_layout_library,
        topology,
        diis_history=8,
        requires_first_ao_derivatives=False,
        functional_code=0,
        requires_tau=False,
        tile=1,
    )
    assert ks["setup"] == expected


@pytest.fixture(scope="module")
def basis() -> BasisSet:
    """Synthetic Na/H metadata; no external chemistry package or oracle lookup."""
    shell = BasisShell(0, ("1",), (("1",),))
    potential = {
        "ecp_type": "scalar_ecp",
        "angular_momentum": [0],
        "r_exponents": [2],
        "gaussian_exponents": ["1"],
        "coefficients": [["-1"]],
    }
    return BasisSet(
        "Synthetic ECP resource fixture",
        (
            ElementBasis(
                11,
                (shell,),
                ecp_core_electrons=10,
                ecp_data=json.dumps([potential]),
            ),
            ElementBasis(1, (shell,)),
        ),
        BasisProvenance("Synthetic test data", "1", "test-only", "0" * 64),
    )


@pytest.mark.parametrize("method", ("rhf", "pbe-rks"))
def test_deferred_cuda_library_records_the_actual_ecp_owners(
    basis: BasisSet,
    cuda_layout_library: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """KS sizes host storage before loading the device-layout library."""

    def load_library(*, device: str) -> typing.Any:
        assert device == "cpu"
        return cuda_layout_library

    monkeypatch.setattr(_native, "load_library", load_library)
    estimate = estimate_hf_resources if method == "rhf" else estimate_ks_resources
    plan = estimate(
        [[("Na", (0, 0, 0)), ("H", (0, 0, 3))]],
        basis=basis,
        method=method,
        backend="cuda",
    ).require_feasible()
    controls = json.loads(plan.requests[0].identity.schedule)
    if method == "rhf":
        assert controls["ecp_inventory"] == "native-ecp-v1"
    else:
        assert controls["ecp_host_inventory"] == "portable-metadata-v1"
        assert controls["ecp_device_inventory"] == "native-ecp-v1"


@pytest.mark.parametrize("method", ("rhf", "uhf", "pbe-rks", "pbe-uks"))
@pytest.mark.parametrize("host_owner", ("current", "phase-only", "portable"))
def test_public_plans_route_ecp_independently_of_enclosing_inventory(
    library: typing.Any,
    basis: BasisSet,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    host_owner: str,
) -> None:
    """Mixed-age libraries keep exact bounds and explicit ECP owner identity."""
    hidden = set()
    if host_owner != "current":
        hidden.add("generativeqc_resource_hf_cpu_fleet_v1")
        hidden.add("generativeqc_resource_ks_cpu_fleet_v1")
    if host_owner == "portable":
        hidden.update(
            (
                "generativeqc_resource_hf_cpu_host_v1",
                "generativeqc_resource_ks_cpu_host_v1",
            )
        )

    class LibraryView:
        def __init__(self, *, legacy_ecp: bool = False) -> None:
            self.legacy_ecp = legacy_ecp

        def __getattr__(self, name: str) -> typing.Any:
            if name in hidden or (
                self.legacy_ecp and name == "generativeqc_resource_ecp_workspace_v1"
            ):
                raise AttributeError(name)
            return getattr(library, name)

    unrestricted = method in ("uhf", "pbe-uks")
    options = {
        "basis": basis,
        "method": method,
        "charges": [int(unrestricted), 0],
        "multiplicities": [2 if unrestricted else 1, 1],
    }
    systems = [
        [("Na", (0, 0, 0)), ("H", (0, 0, 3))],
        [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
    ]
    estimate = (
        estimate_hf_resources if method in ("rhf", "uhf") else estimate_ks_resources
    )
    legacy = estimate(systems, library=LibraryView(legacy_ecp=True), **options)
    legacy.require_feasible()

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("production ECP inventory retried portable policy")

    monkeypatch.setattr(resources_hf, "_portable_ecp_workspace", forbidden)
    native = estimate(systems, library=LibraryView(), **options).require_feasible()
    assert native.peak_bytes == legacy.peak_bytes
    assert native.resident_bytes == legacy.resident_bytes
    assert native.identity != legacy.identity
    owner = "ecp_inventory" if method in ("rhf", "uhf") else "ecp_host_inventory"
    assert json.loads(native.requests[0].identity.schedule)[owner] == "native-ecp-v1"
    assert (
        json.loads(legacy.requests[0].identity.schedule)[owner]
        == "portable-metadata-v1"
    )
    exact = ResourceBudget(host_bytes=native.peak_bytes["host"])
    assert (
        estimate(systems, library=LibraryView(), budget=exact, **options).status
        == "feasible"
    )
    below = ResourceBudget(host_bytes=native.peak_bytes["host"] - 1)
    assert (
        estimate(systems, library=LibraryView(), budget=below, **options).status
        == "infeasible"
    )
