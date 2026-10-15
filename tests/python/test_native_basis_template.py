"""Native element-template expansion against portable data and real CPU consumers."""

from __future__ import annotations

import ctypes
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import (
    Atom,
    BasisProvenance,
    BasisSet,
    BasisShell,
    Calculator,
    ElementBasis,
    ResourceBudget,
    _native,
    import_bse,
)
from generativeqc._basis_native import NativeBasisTemplate
from generativeqc.ecp import resolve_ecp

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
H2 = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
WATER = [("O", (0, 0, 0)), ("H", (0, -1.43, 1.1)), ("H", (0, 1.43, 1.1))]


class ExpansionView(ctypes.Structure):
    _fields_ = [
        ("shells", ctypes.POINTER(_native.ShellDescriptor)),
        ("shell_count", ctypes.c_size_t),
        ("primitives", ctypes.POINTER(_native.PrimitiveDescriptor)),
        ("primitive_count", ctypes.c_size_t),
        ("cores", ctypes.POINTER(ctypes.c_int32)),
        ("core_count", ctypes.c_size_t),
        ("terms", ctypes.POINTER(_native.EcpTermDescriptor)),
        ("term_count", ctypes.c_size_t),
        ("owner", ctypes.c_void_p),
    ]


@pytest.fixture(scope="module")
def probe(
    required_native_cxx: NativeCxx, tmp_path_factory: pytest.TempPathFactory
) -> Any:
    """Read raw metadata from the actual library owner, not a copied implementation."""
    library = Path(_native.load_library(device="cpu")._name).resolve()
    output = tmp_path_factory.mktemp("basis-template-probe") / "probe.so"
    required_native_cxx.build_shared(
        [ROOT / "tests/native/test_basis_template.cpp"],
        output,
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
        link_args=(str(library), f"-Wl,-rpath,{library.parent}"),
    )
    result = ctypes.CDLL(str(output))
    result.test_expand_basis_template.argtypes = [
        ctypes.POINTER(_native.BasisTemplateDescriptor),
        ctypes.POINTER(_native.AtomDescriptor),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ExpansionView),
    ]
    result.test_expand_basis_template.restype = ctypes.c_int
    result.test_free_basis_template.argtypes = [ctypes.c_void_p]
    result.test_free_basis_template.restype = None
    return result


def synthetic(seed: int, representation: str, with_ecp: bool) -> BasisSet:
    """Generate exact immutable inputs with ragged shared exponents and general rows."""
    generator = random.Random(seed)
    elements = []
    for atomic_number in (1, 8):
        shells = []
        for source_group in range(generator.randint(1, 3)):
            exponents = tuple(
                str(generator.uniform(0.2, 2.5)) for _ in range(generator.randint(1, 4))
            )
            rows = []
            for _ in range(generator.randint(1, 3)):
                row = [str(generator.uniform(-1, 1)) for _ in exponents]
                row[0] = "1"
                if len(row) > 1:
                    row[-1] = "0"
                rows.append(tuple(row))
            shells.append(
                BasisShell(generator.randrange(4), exponents, tuple(rows), source_group)
            )
        potentials = None
        if with_ecp and atomic_number == 8:
            channels = [2, 0, 1]
            generator.shuffle(channels)
            potentials = json.dumps(
                [
                    {
                        "ecp_type": "scalar_ecp",
                        "angular_momentum": [channel],
                        "r_exponents": [0, 2],
                        "gaussian_exponents": ["0.4", "1.2"],
                        "coefficients": [["0", str(generator.uniform(-1, 1))]],
                    }
                    for channel in channels
                ]
            )
        elements.append(
            ElementBasis(
                atomic_number,
                tuple(shells),
                ecp_core_electrons=2 if potentials else 0,
                ecp_data=potentials,
            )
        )
    return BasisSet(
        f"synthetic-{seed}",
        tuple(elements),
        BasisProvenance(
            "synthetic input seed",
            "v1",
            "CC0-1.0",
            hashlib.sha256(str(seed).encode()).hexdigest(),
        ),
        representation,
    )


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("with_ecp", [False, True])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_ragged_general_and_ecp_templates_match_portable_oracle(
    probe: Any, seed: int, with_ecp: bool, representation: str
) -> None:
    basis = synthetic(seed, representation, with_ecp)
    atoms = tuple(
        Atom.from_value(value) for value in [("O", (0, 0, 0)), *H2, ("O", (2.1, 0, 0))]
    )
    source = NativeBasisTemplate.from_basis(basis, atoms)
    atom_array = (_native.AtomDescriptor * len(atoms))(
        *(_native.AtomDescriptor(atom.atomic_number, *atom.position) for atom in atoms)
    )
    output = ExpansionView()
    assert (
        probe.test_expand_basis_template(
            ctypes.byref(source.descriptor),
            atom_array,
            len(atoms),
            _native.BASIS_SPHERICAL
            if representation == "spherical"
            else _native.BASIS_CARTESIAN,
            ctypes.byref(output),
        )
        == 0
    )
    try:
        actual = tuple(
            (
                output.shells[index].atom_index,
                output.shells[index].angular_momentum,
                tuple(
                    (
                        output.primitives[primitive].exponent,
                        output.primitives[primitive].coefficient,
                    )
                    for primitive in range(
                        output.shells[index].primitive_offset,
                        output.shells[index].primitive_offset
                        + output.shells[index].primitive_count,
                    )
                ),
            )
            for index in range(output.shell_count)
        )
        expected = tuple(
            (
                shell.atom_index,
                shell.angular_momentum,
                tuple(
                    (primitive.exponent, primitive.coefficient)
                    for primitive in shell.primitives
                ),
            )
            for shell in basis.shells_for(atoms)
        )
        assert actual == expected
        cores, terms = resolve_ecp(basis, atoms)
        assert tuple(output.cores[index] for index in range(output.core_count)) == cores
        assert (
            tuple(
                (
                    output.terms[index].atom_index,
                    output.terms[index].channel,
                    output.terms[index].power,
                    output.terms[index].exponent,
                    output.terms[index].coefficient,
                )
                for index in range(output.term_count)
            )
            == terms
        )
    finally:
        probe.test_free_basis_template(output.owner)


def custom_basis(representation: str = "cartesian") -> BasisSet:
    """Keep imported source provenance while adding an explicit general contraction."""
    basis = import_bse(
        ROOT / "tests/data/external_basis/sto-3g-ho.json",
        source="pinned independent BSE input",
        source_version="4adaf1372c7101620ca1a9f3130be9ae97fb8f30",
        license="BSD-3-Clause",
        representation=representation,
    )
    hydrogen = basis.by_element[1]
    shell = hydrogen.shells[0]
    return replace(
        basis,
        elements=(
            replace(
                hydrogen,
                shells=(
                    replace(
                        shell, coefficients=(*shell.coefficients, ("0", "0.25", "0.75"))
                    ),
                ),
            ),
            *[element for element in basis.elements if element.atomic_number != 1],
        ),
    )


class LibraryView:
    """Probe explicit old-symbol compatibility, rejection, or the production ABI boundary."""

    def __init__(
        self,
        library: Any,
        *,
        legacy: bool = False,
        reject: bool = False,
        forbid_expanded: bool = False,
    ) -> None:
        self.library = library
        self.legacy = legacy
        self.reject = reject
        self.forbid_expanded = forbid_expanded

    def __getattr__(self, name: str) -> Any:
        if name == "generativeqc_system_create_basis_template_v1":
            if self.legacy:
                raise AttributeError(name)
            if self.reject:
                return self.rejected
        if self.forbid_expanded and name in {
            "generativeqc_system_create",
            "generativeqc_system_create_ecp",
        }:
            raise AssertionError(
                "production Python attempted atom-expanded basis construction"
            )
        return getattr(self.library, name)

    def rejected(self, context: Any, descriptor: Any, basis: Any, output: Any) -> int:
        record = ctypes.cast(
            basis, ctypes.POINTER(_native.BasisTemplateDescriptor)
        ).contents
        record.element_count = 0
        return self.library.generativeqc_system_create_basis_template_v1(
            context, descriptor, basis, output
        )


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_general_templates_exact_budget_ragged_replay_and_recovery(
    representation: str, method: str
) -> None:
    basis = custom_basis(representation)
    systems = [H2, WATER]
    charges, multiplicities = ([1, 0], [2, 1]) if method == "uhf" else ([0, 0], [1, 1])
    options = {"basis": basis, "method": method, "device": "cpu"}
    plan = (
        Calculator(**options)
        .estimate_resources(systems, charges=charges, multiplicities=multiplicities)
        .require_feasible()
    )
    with pytest.raises(MemoryError):
        Calculator(
            **options,
            resource_budget=ResourceBudget(host_bytes=plan.peak_bytes["host"] - 1),
        ).prepare_batch(systems, charges=charges, multiplicities=multiplicities)
    native = Calculator(
        **options, resource_budget=ResourceBudget(host_bytes=plan.peak_bytes["host"])
    )
    legacy = Calculator(
        **options, resource_budget=ResourceBudget(host_bytes=plan.peak_bytes["host"])
    )
    native._library = LibraryView(native._library, forbid_expanded=True)
    legacy._library = LibraryView(legacy._library, legacy=True)
    with (
        native.prepare_batch(
            systems, charges=charges, multiplicities=multiplicities
        ) as actual_plan,
        legacy.prepare_batch(
            systems, charges=charges, multiplicities=multiplicities
        ) as oracle_plan,
    ):
        for coordinates in (
            None,
            None,
            [np.full((2, 3), np.nan), None],
            [[(0, 0, -0.72), (0, 0, 0.72)], None],
        ):
            actual = actual_plan.execute(coordinates)
            expected = oracle_plan.execute(coordinates)
            for result, oracle in zip(actual.items, expected.items, strict=True):
                assert result.status == oracle.status
                assert result.succeeded == oracle.succeeded
                if result.succeeded:
                    assert result.energy == oracle.energy
                    np.testing.assert_array_equal(result.forces, oracle.forces)
                    assert result.iterations == oracle.iterations
                    assert result.fock_builds == oracle.fock_builds
                    assert result.warm_start_used == oracle.warm_start_used
                    assert result.basis_metadata == oracle.basis_metadata


def test_native_template_rejection_is_not_retried_as_expanded_python_data() -> None:
    calculator = Calculator(basis=custom_basis(), device="cpu")
    calculator._library = LibraryView(calculator._library, reject=True)
    with pytest.raises(RuntimeError, match="template"):
        calculator.singlepoint(H2)


def test_unused_unqualified_element_metadata_does_not_change_the_selected_model() -> (
    None
):
    base = custom_basis()
    unsupported = ElementBasis(26, (BasisShell(6, ("0.5",), (("1",),)),))
    with_unused = replace(base, elements=(*base.elements, unsupported))
    actual = Calculator(basis=with_unused, device="cpu").singlepoint(H2)
    expected = Calculator(basis=base, device="cpu").singlepoint(H2)
    assert actual.energy == expected.energy
    np.testing.assert_array_equal(actual.forces, expected.forces)
