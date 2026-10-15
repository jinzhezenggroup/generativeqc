"""CPU HF phase sizing has one native owner and an explicit legacy fallback."""

import json
import typing
from types import SimpleNamespace

import pytest
from generativeqc import _native, estimate_hf_resources, resources_hf


@pytest.fixture(scope="module")
def library() -> typing.Any:
    """Require the source-matched bridge rather than silently skipping it."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_hf_cpu_host_v1")
    return loaded


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("history", (0, 1, 8, 64))
@pytest.mark.parametrize(
    "orbital_aos,cartesian_aos,angular,auxiliary",
    (
        (2, 2, 0, None),
        (7, 8, 1, None),
        (5, 6, 2, None),
        (9, 15, 4, None),
        (7, 8, 1, (12, 13, 2)),
        (5, 6, 2, (9, 15, 4)),
        (500, 500, 0, (1000, 1000, 1)),
    ),
)
def test_native_matches_independent_portable_phase_bound(
    library: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
    spins: int,
    history: int,
    orbital_aos: int,
    cartesian_aos: int,
    angular: int,
    auxiliary: tuple[int, int, int] | None,
) -> None:
    """Cover direct/DF, public/Cartesian coexistence, through-g and ECP storage."""
    item = {
        "atoms": 3,
        "spins": spins,
        "orbital": {
            "nbf": orbital_aos,
            "cartesian_nbf": cartesian_aos,
            "maximum_angular": angular,
        },
        "auxiliary": (
            dict(
                zip(("nbf", "cartesian_nbf", "maximum_angular"), auxiliary, strict=True)
            )
            if auxiliary
            else None
        ),
    }
    monkeypatch.setattr(resources_hf, "_ecp_workspace", lambda item: 768)
    expected = resources_hf._cpu_item_inventory(item, history)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("discovered native sizing retried the portable arithmetic")

    monkeypatch.setattr(resources_hf, "byte_product", forbidden)
    assert resources_hf._cpu_item_inventory(item, history, library=library) == expected


def test_discovered_rejection_never_retries_portable_sizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*args: typing.Any) -> int:
        return -1

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("rejected native inventory silently used portable sizing")

    monkeypatch.setattr(resources_hf, "byte_product", forbidden)
    monkeypatch.setattr(resources_hf, "_ecp_workspace", lambda item: 0)
    item = {
        "atoms": 2,
        "spins": 1,
        "orbital": {"nbf": 2, "cartesian_nbf": 2, "maximum_angular": 0},
    }
    with pytest.raises(ValueError, match="native CPU HF host inventory rejected"):
        resources_hf._cpu_item_inventory(
            item,
            8,
            library=SimpleNamespace(generativeqc_resource_hf_cpu_host_v1=reject),
        )


@pytest.mark.parametrize("method,charge,multiplicity", (("rhf", 0, 1), ("uhf", 1, 2)))
@pytest.mark.parametrize("fitted", (False, True))
def test_legacy_and_native_plans_share_bounds_but_not_owner_identity(
    library: typing.Any, method: str, charge: int, multiplicity: int, fitted: bool
) -> None:
    """Fleet aggregation and old-library dry runs retain their capacity contract."""

    class LegacyLibrary:
        def __getattr__(self, name: str) -> typing.Any:
            if name == "generativeqc_resource_hf_cpu_host_v1":
                raise AttributeError(name)
            return getattr(library, name)

    atoms = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
    controls = {
        "method": method,
        "charges": [charge] * 3,
        "multiplicities": [multiplicity] * 3,
        "density_fitting": "cpu" if fitted else "none",
    }
    native = estimate_hf_resources([atoms] * 3, library=library, **controls)
    legacy = estimate_hf_resources([atoms] * 3, library=LegacyLibrary(), **controls)
    assert native.status == legacy.status == "feasible"
    assert native.peak_bytes == legacy.peak_bytes
    assert native.resident_bytes == legacy.resident_bytes
    assert native.identity != legacy.identity
    assert (
        json.loads(native.requests[0].identity.schedule)["host_inventory"]
        == "native-cpu-v1"
    )
    assert (
        json.loads(legacy.requests[0].identity.schedule)["host_inventory"]
        == "portable-metadata-v1"
    )
