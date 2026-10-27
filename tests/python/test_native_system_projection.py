"""Independent molecular projection and native multi-spin publication contracts."""

from __future__ import annotations

import ctypes as ct
import typing
from contextlib import contextmanager

import numpy as np
import pytest
from generativeqc import Calculator, _native, _projection_native, progressive
from generativeqc import _system_projection_native as native
from generativeqc.calculator import Atom

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))]


@contextmanager
def inputs(
    source: Calculator,
    target: Calculator,
    atoms: list,
    charge: int = 0,
    multiplicity: int = 1,
) -> typing.Iterator[tuple]:
    """Borrow SDK systems for a direct, compiler-free native transport call."""
    library = target._library
    if not native.bind(library):
        pytest.skip("requires native system projection protocol")
    context = ct.c_void_p()
    _native.check(
        library,
        library.generativeqc_context_create(
            ct.byref(target._context_descriptor()), ct.byref(context)
        ),
    )
    left = right = None
    try:
        geometry = tuple(Atom.from_value(atom) for atom in atoms)
        left = target._create_native_system(context, geometry, charge, multiplicity)
        right = source._create_native_system(context, geometry, charge, multiplicity)
        controls = native.Controls(
            ct.sizeof(native.Controls),
            _native.ABI_VERSION,
            target._method == _native.METHOD_UHF,
            0,
            4096,
            1 << 30,
            1e-10,
            1e-7,
            0.5,
        )
        capacity = native.Capacity(ct.sizeof(native.Capacity), _native.ABI_VERSION)
        assert (
            library.generativeqc_hf_system_projection_query_v1(
                left, right, ct.byref(controls), ct.byref(capacity)
            )
            == 0
        )
        yield library, context, left, right, controls, capacity
    finally:
        if right:
            library.generativeqc_system_destroy(right)
        if left:
            library.generativeqc_system_destroy(left)
        library.generativeqc_context_destroy(context)


def outputs(capacity: native.Capacity) -> tuple:
    """Allocate sentinel-filled buffers to expose partial writes on refusal."""
    records = (_projection_native.Output * capacity.spin_count)()
    storage = []
    for spin, occupied in enumerate(
        (capacity.alpha_occupied, capacity.beta_occupied)[: capacity.spin_count]
    ):
        coefficients = np.full((capacity.target_ao, occupied), 19.0, dtype=np.float64)
        density = np.full(
            (capacity.target_ao, capacity.target_ao), 23.0, dtype=np.float64
        )
        records[spin] = _projection_native.Output(
            struct_size=ct.sizeof(_projection_native.Output),
            abi_version=_native.ABI_VERSION,
            coefficients=native._pointer(coefficients),
            coefficient_capacity=coefficients.size,
            density=native._pointer(density),
            density_capacity=density.size,
        )
        storage.append((coefficients, density))
    return records, storage


def execute(arguments: tuple, density: np.ndarray, records: typing.Any) -> int:
    """Execute the public ABI without frontend validation hiding a native error."""
    library, context, target, source, controls, capacity = arguments
    return library.generativeqc_hf_system_project_cpu_v1(
        context,
        target,
        source,
        ct.byref(controls),
        native._pointer(density),
        density.size,
        records,
        capacity.spin_count,
    )


def oracle(
    atoms: list,
    method: str,
    representation: str,
    charge: int,
    spin: int,
    source_basis: str = "sto-3g",
) -> tuple:
    """Solve independent PySCF overlap least-squares in its native AO convention."""
    pyscf = pytest.importorskip("pyscf")
    kwargs = {
        "atom": atoms,
        "unit": "Bohr",
        "charge": charge,
        "spin": spin,
        "cart": representation == "cartesian",
        "verbose": 0,
    }
    # Library basis names can resolve to coefficients rounded at different
    # precision (notably STO-3G). Transport exact model metadata to the independent
    # integral oracle; never borrow native overlap or projection results.
    geometry = tuple(Atom.from_value(atom) for atom in atoms)
    labels = [f"{symbol}{index}" for index, (symbol, _) in enumerate(atoms)]
    kwargs["atom"] = [
        (label, position) for label, (_, position) in zip(labels, atoms, strict=True)
    ]

    def basis_data(name: str) -> dict:
        basis = {label: [] for label in labels}
        calculator = Calculator(basis=name)
        for shell in calculator._shells_for_atoms(geometry):
            basis[labels[shell.atom_index]].append(
                [
                    shell.angular_momentum,
                    *[
                        [primitive.exponent, primitive.coefficient]
                        for primitive in shell.primitives
                    ],
                ]
            )
        return basis

    source = pyscf.gto.M(basis=basis_data(source_basis), **kwargs)
    target = pyscf.gto.M(basis=basis_data("def2-svp"), **kwargs)
    ss, tt = source.intor("int1e_ovlp"), target.intor("int1e_ovlp")
    ts = pyscf.gto.intor_cross("int1e_ovlp", target, source)
    source_scale = 1 / np.sqrt(np.diag(ss))
    target_scale = 1 / np.sqrt(np.diag(tt))
    values, vectors = np.linalg.eigh(ss)
    frame = vectors / np.sqrt(values)
    populations = source.nelec if method == "uhf" else [source.nelec[0]]
    occupation = 1 if method == "uhf" else 2
    densities, expected = [], []
    for occupied in populations:
        coefficients = frame[:, :occupied]
        density = occupation * coefficients @ coefficients.T
        densities.append(density / source_scale[:, None] / source_scale[None, :])
        image = np.linalg.solve(tt, ts @ coefficients)
        norms, rotation = np.linalg.eigh(image.T @ tt @ image)
        image = image @ (rotation / np.sqrt(norms)) @ rotation.T
        result = occupation * image @ image.T
        expected.append(result / target_scale[:, None] / target_scale[None, :])
    return np.ascontiguousarray(densities), np.asarray(expected)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("moved", [False, True])
@pytest.mark.parametrize("source_basis", ["sto-3g", "def2-svp"])
def test_independent_molecular_metrics_and_all_spin_projection(
    method: str, representation: str, moved: bool, source_basis: str
) -> None:
    atoms = [
        ("Ne", (0.0, 0.0, 0.0)),
        ("H", (0.0, 0.0, 2.4)),
        ("H", (0.2 if moved else 0.0, 0.1 if moved else 0.0, 3.8)),
    ]
    charge = int(method == "uhf")
    source = Calculator(
        basis=source_basis, method=method, basis_representation=representation
    )
    target = Calculator(
        basis="def2-svp", method=method, basis_representation=representation
    )
    density, expected = oracle(
        atoms, method, representation, charge, charge, source_basis
    )
    original = density.copy()
    with inputs(source, target, atoms, charge, charge + 1) as arguments:
        records, storage = outputs(arguments[-1])
        assert execute(arguments, density, records) == 0
        for record, (_, actual), reference in zip(
            records, storage, expected, strict=True
        ):
            np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=0)
            assert record.metric_orthogonality_error < 1e-10
            assert record.metric_idempotency_error < 1e-10
        np.testing.assert_array_equal(density, original)


@pytest.mark.parametrize(
    "error", ["budget", "nan", "count", "alias", "beta_purity", "abi"]
)
def test_failed_multispin_transfer_preserves_all_publication(error: str) -> None:
    source = Calculator(basis="sto-3g", method="uhf")
    target = Calculator(basis="def2-svp", method="uhf")
    density, _ = oracle(H2, "uhf", "cartesian", 0, 0)
    with inputs(source, target, H2) as arguments:
        controls, capacity = arguments[-2:]
        records, storage = outputs(capacity)
        expected_status = _native.STATUS_INVALID_ARGUMENT
        if error == "budget":
            controls.maximum_host_bytes = capacity.numeric_workspace_bytes - 1
            expected_status = _native.STATUS_OUT_OF_MEMORY
        elif error == "nan":
            density[1, 0, 0] = np.nan
        elif error == "count":
            density = density[:, :-1, :].copy()
        elif error == "alias":
            records[1].density = records[0].density
        elif error == "beta_purity":
            density[1] *= 0.5
        elif error == "abi":
            records[1].abi_version += 1
            expected_status = _native.STATUS_ABI_MISMATCH
        previous = bytes(records)
        assert execute(arguments, density, records) == expected_status
        assert bytes(records) == previous
        for coefficients, result in storage:
            np.testing.assert_array_equal(coefficients, 19.0)
            np.testing.assert_array_equal(result, 23.0)


def test_empty_beta_channel_and_exact_capacity() -> None:
    source = Calculator(basis="sto-3g", method="uhf")
    target = Calculator(basis="def2-svp", method="uhf")
    density, expected = oracle(H2, "uhf", "cartesian", 1, 1)
    with inputs(source, target, H2, 1, 2) as arguments:
        controls, capacity = arguments[-2:]
        controls.maximum_host_bytes = capacity.numeric_workspace_bytes
        records, storage = outputs(capacity)
        assert execute(arguments, density, records) == 0
        assert records[1].occupied_rank == 0
        np.testing.assert_array_equal(storage[1][1], 0.0)
        np.testing.assert_allclose(storage[0][1], expected[0], atol=2e-11, rtol=0)


def test_progressive_endpoint_forbids_python_metric_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = Calculator(basis="sto-3g"), Calculator(basis="def2-svp")
    if not native.bind(target._library):
        pytest.skip("requires native system projection")

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("admitted projection retried Python science")

    monkeypatch.setattr(progressive, "cross_overlap", forbidden)
    monkeypatch.setattr(progressive, "project_density", forbidden)
    result = progressive.projected_singlepoint(target, source, H2)
    assert result.target.converged
    assert result.diagnostics["items"][0]["projection_owners"] == ["native_cpu_system"]


def test_missing_protocol_retains_explicit_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = Calculator(basis="sto-3g"), Calculator(basis="def2-svp")
    monkeypatch.setattr(native, "bind", lambda _: False)
    result = progressive.projected_singlepoint(target, source, H2)
    assert result.target.converged
    assert result.diagnostics["items"][0]["projection_owners"] == ["native_cpu"]


def test_admitted_failure_never_retries_and_closes_systems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = Calculator(basis="sto-3g"), Calculator(basis="def2-svp")
    library = target._library
    if not native.bind(library):
        pytest.skip("requires native system projection")
    destroyed = []
    original = library.generativeqc_system_destroy

    def destroy(system: typing.Any) -> None:
        destroyed.append(system)
        original(system)

    def failed(*args: typing.Any) -> int:
        return _native.STATUS_INTERNAL_ERROR

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("native failure retried Python preparation")

    monkeypatch.setattr(library, "generativeqc_hf_system_project_cpu_v1", failed)
    monkeypatch.setattr(library, "generativeqc_system_destroy", destroy)
    monkeypatch.setattr(progressive, "cross_overlap", forbidden)
    with pytest.raises(RuntimeError):
        progressive.projected_singlepoint(target, source, H2)
    assert len(destroyed) >= 2


@pytest.mark.parametrize("mismatch", ["geometry", "spin"])
def test_native_metadata_identity_refusal_is_transactional(mismatch: str) -> None:
    source = Calculator(basis="sto-3g", method="uhf")
    target = Calculator(basis="def2-svp", method="uhf")
    with inputs(source, target, H2) as arguments:
        library, context, left, _, controls, capacity = arguments
        moved = (
            [("H", (0.0, 0.0, -0.7)), ("H", (0.2, 0.0, 0.7))]
            if mismatch == "geometry"
            else H2
        )
        geometry = tuple(Atom.from_value(atom) for atom in moved)
        right = source._create_native_system(
            context, geometry, int(mismatch == "spin"), 2 if mismatch == "spin" else 1
        )
        before = bytes(capacity)
        try:
            assert (
                library.generativeqc_hf_system_projection_query_v1(
                    left, right, ct.byref(controls), ct.byref(capacity)
                )
                == _native.STATUS_INVALID_ARGUMENT
            )
            assert bytes(capacity) == before
        finally:
            library.generativeqc_system_destroy(right)


def test_query_does_not_publish_inside_extended_input_descriptor() -> None:
    class Extended(ct.Structure):
        _fields_ = [("controls", native.Controls), ("capacity", native.Capacity)]

    with inputs(
        Calculator(basis="sto-3g"), Calculator(basis="def2-svp"), H2
    ) as arguments:
        library, _, left, right, controls, capacity = arguments
        extended = Extended(controls, capacity)
        extended.controls.struct_size = ct.sizeof(Extended)
        before = bytes(extended)
        assert (
            library.generativeqc_hf_system_projection_query_v1(
                left, right, ct.byref(extended.controls), ct.byref(extended.capacity)
            )
            == _native.STATUS_INVALID_ARGUMENT
        )
        assert bytes(extended) == before
