"""Shared catalog data, provenance, installed ABI and complete CPU endpoint gates."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import Calculator, _basis_builtin, _model_resolution, _native
from generativeqc.profiles import canonical_hash

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def catalog_probe(
    required_native_cxx: NativeCxx, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Compile the public SDK and typed catalog contract against the tested library."""
    library = Path(_native.load_library(device="cpu")._name).resolve()
    output = tmp_path_factory.mktemp("native-basis-catalog") / "probe"
    required_native_cxx.build_executable(
        [ROOT / "tests/native/test_basis_catalog.cpp"],
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
    return output


def test_catalog_matches_independent_complete_source_pack(catalog_probe: Path) -> None:
    """Every bundled element/primitive survives the shared native expansion exactly."""
    result = subprocess.run(
        [str(catalog_probe), "dump"], check=True, capture_output=True, text=True
    )
    actual = {}
    for line in result.stdout.splitlines():
        name, atomic_number, angular, *values = line.split()
        assert len(values) % 2 == 0
        actual.setdefault((name, atomic_number), []).append(
            (
                int(angular),
                tuple(
                    (float(exponent), float(coefficient))
                    for exponent, coefficient in zip(
                        values[::2], values[1::2], strict=True
                    )
                ),
            )
        )
    pack = json.loads((ROOT / "python/generativeqc/data/basis_pack.json").read_text())
    expected = {
        (name, atomic_number): [
            (
                shell["angular_momentum"],
                tuple(
                    (float(exponent), float(coefficient))
                    for exponent, coefficient in zip(
                        shell["exponents"], shell["coefficients"], strict=True
                    )
                ),
            )
            for shell in shells
        ]
        for name, basis in pack["bases"].items()
        for atomic_number, shells in basis["elements"].items()
    }
    assert actual == expected
    assert _native.load_library(
        device="cpu"
    ).generativeqc_bundled_basis_catalog_sha256().decode() == canonical_hash(pack)


def test_public_sdk_provenance_validation_and_copied_lifetimes(
    catalog_probe: Path,
) -> None:
    subprocess.run([str(catalog_probe)], check=True, capture_output=True, text=True)


def test_only_exact_snapshots_are_recognized_without_loading_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_native(**kwargs: Any) -> Any:
        raise AssertionError("basis preflight must not load the native library")

    monkeypatch.setattr(_native, "load_library", forbidden_native)
    for name in ("STO_3G", "def2-svp", "def2-tzvp"):
        for representation in ("cartesian", "spherical"):
            basis = _model_resolution.snapshot_basis(name, representation)
            assert _model_resolution._bundled_basis_name(basis) == basis.name
            assert (
                _model_resolution._bundled_basis_name(
                    replace(
                        basis, provenance=replace(basis.provenance, source="custom")
                    )
                )
                is None
            )
            element = basis.elements[0]
            shell = element.shells[0]
            changed = replace(
                basis,
                elements=(
                    replace(
                        element,
                        shells=(
                            replace(shell, exponents=("123.5", *shell.exponents[1:])),
                            *element.shells[1:],
                        ),
                    ),
                    *basis.elements[1:],
                ),
            )
            assert _model_resolution._bundled_basis_name(changed) is None


class ExplicitBasisLibrary:
    """Keep supplied records on their own route, never the bundled catalog."""

    def __init__(self, library: Any, *, raw_descriptor: bool = False) -> None:
        self.library = library
        self.raw_descriptor = raw_descriptor

    def __getattr__(self, name: str) -> Any:
        if name in {
            "generativeqc_builtin_basis_query_v1",
            "generativeqc_system_create_builtin_basis_v1",
            "generativeqc_system_create_bundled",
        }:
            raise AssertionError("caller-owned basis selected the bundled catalog")
        if (
            self.raw_descriptor
            and name == "generativeqc_system_create_basis_template_v1"
        ):
            raise AttributeError(name)
        return getattr(self.library, name)


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("basis_name", ["sto-3g", "def2-svp", "def2-tzvp"])
def test_native_and_explicit_data_agree_across_complete_ragged_lifecycles(
    basis_name: str, representation: str
) -> None:
    systems = [
        [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
        [("He", (0, 0, 0))],
    ]
    native = Calculator(
        basis=basis_name, basis_representation=representation, device="cpu"
    )
    bundled = _model_resolution.snapshot_basis(basis_name, representation)
    explicit = Calculator(
        basis=replace(
            bundled, provenance=replace(bundled.provenance, source="explicit test data")
        ),
        device="cpu",
    )
    explicit._library = ExplicitBasisLibrary(explicit._library, raw_descriptor=True)
    with (
        native.prepare_batch(systems, warm_start=True) as planned_native,
        explicit.prepare_batch(systems, warm_start=True) as planned_explicit,
    ):
        for coordinates in (None, None, [[(0, 0, -0.72), (0, 0, 0.72)], None]):
            actual = planned_native.execute(coordinates, strict=True)
            expected = planned_explicit.execute(coordinates, strict=True)
            for result, oracle in zip(actual.items, expected.items, strict=True):
                assert result.energy == oracle.energy
                np.testing.assert_array_equal(result.forces, oracle.forces)
                assert result.converged == oracle.converged
                assert result.iterations == oracle.iterations
                assert result.warm_start_used == oracle.warm_start_used
                assert (
                    result.basis_metadata["orbital"]["mathematical_identity"]
                    == oracle.basis_metadata["orbital"]["mathematical_identity"]
                )


def test_native_catalog_rejection_never_retries_portable_expansion() -> None:
    calculator = Calculator(basis="sto-3g", device="cpu")
    library = calculator._library

    class CorruptDigest:
        def __call__(self, descriptor: Any, output: Any) -> int:
            request = ctypes.cast(
                descriptor, ctypes.POINTER(_basis_builtin.Descriptor)
            ).contents
            request.expected_pack_checksum = b"0" * 64
            return library.generativeqc_builtin_basis_query_v1(descriptor, output)

    class CheckedLibrary:
        generativeqc_builtin_basis_query_v1 = CorruptDigest()

        def __getattr__(self, name: str) -> Any:
            if name in {
                "generativeqc_system_create",
                "generativeqc_system_create_bundled",
                "generativeqc_system_create_basis_template_v1",
            }:
                raise AssertionError("checksum rejection selected another constructor")
            return getattr(library, name)

    library.generativeqc_builtin_basis_query_v1.argtypes = [
        ctypes.POINTER(_basis_builtin.Descriptor),
        ctypes.POINTER(_basis_builtin.Capacity),
    ]
    library.generativeqc_builtin_basis_query_v1.restype = ctypes.c_int
    calculator._library = CheckedLibrary()
    with pytest.raises(RuntimeError, match="GENERATIVEQC error 1: invalid argument"):
        calculator.singlepoint([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))])


def test_custom_snapshot_recognition_does_not_require_the_bundled_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled = _model_resolution.snapshot_basis("sto-3g")
    custom = replace(
        bundled, provenance=replace(bundled.provenance, source="local custom data")
    )

    def unavailable_pack() -> Any:
        raise AssertionError("custom data must not require bundled source tables")

    monkeypatch.setattr(_model_resolution, "_basis_pack", unavailable_pack)
    assert _model_resolution._bundled_basis_name(custom) is None


def test_same_name_custom_data_are_executed_instead_of_the_catalog() -> None:
    basis = _model_resolution.snapshot_basis("sto-3g")
    hydrogen = basis.by_element[1]
    shell = hydrogen.shells[0]
    changed = replace(
        basis,
        elements=(
            replace(
                hydrogen,
                shells=(replace(shell, exponents=("4.1", *shell.exponents[1:])),),
            ),
            *(element for element in basis.elements if element.atomic_number != 1),
        ),
    )
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calculator = Calculator(basis=changed, device="cpu")
    calculator._library = ExplicitBasisLibrary(calculator._library)
    actual = calculator.singlepoint(atoms)
    original = Calculator(basis="sto-3g", device="cpu").singlepoint(atoms)
    assert abs(actual.energy - original.energy) > 1e-5


def test_public_sdk_runtime_without_interpreter_compiler_or_data(
    required_native_cxx: NativeCxx, tmp_path: Path
) -> None:
    """The production constructor must not depend on the Python/CLI environment."""
    library = Path(_native.load_library(device="cpu")._name).resolve()
    executable = required_native_cxx.build_executable(
        [ROOT / "tests/native/test_basis_catalog_sdk.cpp"],
        tmp_path / "sdk-probe",
        compile_args=("-std=c++20", "-I", str(ROOT / "include")),
        link_args=(str(library), f"-Wl,-rpath,{library.parent}"),
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    subprocess.run(
        [str(executable)],
        cwd=empty,
        env={**os.environ, "PATH": str(empty), "PYTHONPATH": str(empty)},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("basis_name", ["sto-3g", "def2-svp", "def2-tzvp"])
def test_native_h2_energy_and_force_against_independent_pyscf(basis_name: str) -> None:
    pyscf = pytest.importorskip("pyscf")
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    result = Calculator(basis=basis_name, device="cpu").singlepoint(atoms)
    molecule = pyscf.gto.M(
        atom=atoms, basis=basis_name, unit="Bohr", cart=True, verbose=0
    )
    oracle = pyscf.scf.RHF(molecule)
    oracle.conv_tol = 1e-12
    oracle.conv_tol_grad = 1e-10
    energy = oracle.kernel()
    assert oracle.converged
    assert result.energy == pytest.approx(energy, abs=2e-9)
    np.testing.assert_allclose(
        result.forces, -oracle.nuc_grad_method().kernel(), atol=2e-8, rtol=0
    )
