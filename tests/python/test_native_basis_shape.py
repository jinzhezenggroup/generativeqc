"""Native scalar ownership, independent dimensions and transactional rejection."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import (
    Atom,
    Calculator,
    ResourceBudget,
    _native,
    estimate_hf_resources,
    estimate_ks_resources,
)
from generativeqc import resources_basis as shapes
from generativeqc.resources_hf import _basis_record

WATER = (Atom(8, (0.0, 0.0, 0.0)), Atom(1, (0.0, 1.4, 1.1)), Atom(1, (0.0, -1.4, 1.1)))


@pytest.fixture(scope="module")
def library() -> Any:
    """Require the actual native query; host-only builds need no GPU provider."""
    loaded = _native.load_library(device="cpu")
    assert hasattr(loaded, "generativeqc_resource_basis_shape_v1")
    return loaded


@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize(
    "angular,primitives",
    (((0,), (3,)), ((0, 1, 2, 3, 4), (3, 2, 4, 1, 5)), ((2, 2, 2, 0), (3, 3, 3, 1))),
)
def test_expanded_contraction_counts_match_portable_metadata(
    library: Any,
    representation: str,
    angular: tuple[int, ...],
    primitives: tuple[int, ...],
) -> None:
    expected = shapes._basis_shape(angular, primitives, representation)
    assert (
        shapes._basis_shape(angular, primitives, representation, library=library)
        == expected
    )


@pytest.mark.parametrize("basis", ("sto-3g", "def2-svp", "def2-tzvp"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_actual_basis_record_matches_independent_pyscf_dimensions(
    library: Any, basis: str, representation: str
) -> None:
    gto = pytest.importorskip("pyscf.gto")
    arguments = (WATER, basis, representation, "cpu", 0, 1, "orbital")
    record = _basis_record(*arguments, library=library)
    assert record == _basis_record(*arguments)
    molecule = gto.M(
        atom=[(atom.atomic_number, atom.position) for atom in WATER],
        basis=basis,
        unit="Bohr",
        cart=representation == "cartesian",
        verbose=0,
    )
    assert record["nbf"] == molecule.nao_nr()
    assert record["cartesian_nbf"] == molecule.nao_cart()
    assert record["shells"] == sum(
        molecule.bas_nctr(index) for index in range(molecule.nbas)
    )
    assert record["primitives"] == sum(
        molecule.bas_nprim(index) * molecule.bas_nctr(index)
        for index in range(molecule.nbas)
    )


@pytest.mark.parametrize("loaded", (None, SimpleNamespace()))
def test_library_free_and_older_library_paths_do_not_discover_native_code(
    loaded: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("metadata shape query discovered a native provider")

    monkeypatch.setattr(_native, "load_library", forbidden)
    assert shapes._basis_shape((2,), (3,), "spherical", library=loaded) == {
        "nbf": 5,
        "cartesian_nbf": 6,
        "shells": 1,
        "primitives": 3,
        "maximum_angular": 2,
    }


@pytest.mark.parametrize("method", ("rhf", "uhf", "pbe-rks", "pbe-uks"))
def test_real_cpu_planners_do_not_retry_portable_shape_math(
    library: Any, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "CPU native topology reverted to portable sizing or provider discovery"
        )

    monkeypatch.setattr(shapes, "_portable_basis_shape", forbidden)
    monkeypatch.setattr(_native, "load_library", forbidden)
    estimate = estimate_ks_resources if "pbe" in method else estimate_hf_resources
    assert estimate([WATER], method=method, library=library).status == "feasible"


def test_native_shape_covers_distinct_density_fitting_auxiliary_basis(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("CPU DF topology reverted to portable shape math")

    monkeypatch.setattr(shapes, "_portable_basis_shape", forbidden)
    assert (
        estimate_hf_resources(
            [WATER],
            basis="sto-3g",
            density_fitting="cpu",
            auxiliary_basis="def2-svp",
            library=library,
        ).status
        == "feasible"
    )


@pytest.mark.parametrize("status", (1, 2, -1))
def test_discovered_native_rejection_never_retries_python(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(*args: Any) -> int:
        return status

    def forbidden(*args: Any) -> Any:
        pytest.fail("native shape rejection retried Python math")

    monkeypatch.setattr(shapes, "_portable_basis_shape", forbidden)
    loaded = SimpleNamespace(generativeqc_resource_basis_shape_v1=reject)
    with pytest.raises(ValueError, match="native basis topology rejected"):
        _basis_record(WATER, "sto-3g", None, "cpu", 0, 1, "orbital", library=loaded)


@pytest.mark.parametrize(
    "angular,primitives,representation",
    (
        ((), (), "cartesian"),
        ((0,), (), "cartesian"),
        ((0,), (0,), "cartesian"),
        ((-1,), (1,), "cartesian"),
        ((2**32,), (1,), "cartesian"),
        ((True,), (1,), "cartesian"),
        ((0,), (1.5,), "cartesian"),
        ((0,), (2**63,), "cartesian"),
        ((0,), (1,), "unknown"),
    ),
)
def test_invalid_python_values_cannot_wrap_through_ctypes(
    angular: tuple[Any, ...], primitives: tuple[Any, ...], representation: str
) -> None:
    def forbidden(*args: Any) -> Any:
        pytest.fail("invalid shell rows reached the native ABI")

    loaded = SimpleNamespace(generativeqc_resource_basis_shape_v1=forbidden)
    with pytest.raises((ValueError, TypeError)):
        shapes._basis_shape(angular, primitives, representation, library=loaded)


@pytest.mark.parametrize(
    "case",
    (
        "empty",
        "representation",
        "zero",
        "angular_overflow",
        "sum_overflow",
        "short",
        "null",
    ),
)
def test_native_rejection_preserves_output_and_neighbor_recovery(
    library: Any, case: str
) -> None:
    query = library.generativeqc_resource_basis_shape_v1
    shapes._basis_shape((0,), (1,), "cartesian", library=library)
    angular = (ctypes.c_uint32 * 2)(0, 0)
    primitives = (ctypes.c_uint64 * 2)(1, 1)
    output = (ctypes.c_uint64 * 5)(*[12345] * 5)
    shell_count, representation, count = 2, 0, 5
    if case == "empty":
        shell_count = 0
    elif case == "representation":
        representation = 2
    elif case == "zero":
        primitives[1] = 0
    elif case == "angular_overflow":
        angular[1] = 2**32 - 1
    elif case == "sum_overflow":
        primitives[0] = 2**63 - 1
    elif case == "short":
        count = 4
    elif case == "null":
        angular = None
    assert query(angular, primitives, shell_count, representation, output, count) != 0
    assert list(output) == [12345] * 5
    assert shapes._basis_shape((2,), (3,), "spherical", library=library)["nbf"] == 5


def test_exact_signed_count_limit_and_next_count_failure(library: Any) -> None:
    maximum = min(2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1, 2**63 - 1)
    assert (
        shapes._basis_shape((0,), (maximum,), "cartesian", library=library)[
            "primitives"
        ]
        == maximum
    )
    for loaded in (None, library):
        with pytest.raises(ValueError):
            shapes._basis_shape((0, 0), (maximum, 1), "cartesian", library=loaded)


def test_shape_rejection_does_not_invalidate_a_prepared_warm_state(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed estimate must not mutate an already-admitted scientific owner."""
    atoms = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
    calculator = Calculator(
        method="rhf", device="cpu", resource_budget=ResourceBudget(host_bytes=1 << 30)
    )
    with calculator.prepare_batch([atoms]) as prepared:
        cold = prepared.execute(strict=True).items[0]
        plan = prepared.resource_plan

        def reject(*args: Any) -> int:
            return 1

        with monkeypatch.context() as scoped:
            scoped.setattr(
                calculator._library, "generativeqc_resource_basis_shape_v1", reject
            )
            with pytest.raises(ValueError, match="native basis topology rejected"):
                calculator.estimate_resources([atoms])
        warm = prepared.execute(strict=True).items[0]
        assert warm.warm_start_used
        assert warm.energy == pytest.approx(cold.energy, abs=1e-12)
        assert prepared.resource_plan is plan
