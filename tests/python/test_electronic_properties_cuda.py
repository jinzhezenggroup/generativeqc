"""Real-device gates for detached CUDA property analysis, without CPU fallback."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import Atom, Calculator, Primitive, Shell, _native
from generativeqc._electronic_properties import _bind, _CudaProperties, _pointer

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires an explicitly Slurm-allocated GPU",
)

ATOMS = [("He", (0.3, -0.4, 0.2)), ("He", (-0.5, 0.8, 0.9))]
SHELLS = tuple(
    Shell(atom, angular, (Primitive(0.7, 0.8), Primitive(0.2, -0.13)))
    for atom in range(2)
    for angular in range(4)
)
REQUESTED = frozenset({"dipole", "charges"})


@contextmanager
def _system(
    calculator: Calculator, atoms: Any = ATOMS
) -> Iterator[tuple[Any, Any, Any]]:
    library = calculator._library
    _bind(library)
    context = ctypes.c_void_p()
    _native.check(
        library,
        library.generativeqc_context_create(
            ctypes.byref(calculator._context_descriptor()), ctypes.byref(context)
        ),
    )
    system = None
    try:
        system = calculator._create_native_system(
            context, tuple(map(Atom.from_value, atoms)), 0, 1
        )
        yield library, context, system
    finally:
        if system is not None:
            library.generativeqc_system_destroy(system)
        library.generativeqc_context_destroy(context)


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_cuda_spdf_operators_and_samples_match_libcint(representation: str) -> None:
    """Off-center contracted f shells exercise polynomial raising and AO order."""
    gto = pytest.importorskip("pyscf.gto")
    molecule = gto.M(
        atom=ATOMS,
        unit="Bohr",
        cart=representation == "cartesian",
        verbose=0,
        basis={"He": [[angular, [0.7, 0.8], [0.2, -0.13]] for angular in range(4)]},
    )
    raw_overlap = molecule.intor("int1e_ovlp")
    normalization = np.sqrt(np.diag(raw_overlap))
    overlap = raw_overlap / normalization[:, None] / normalization[None, :]
    position = (
        molecule.intor("int1e_r")
        / normalization[None, :, None]
        / normalization[None, None, :]
    )
    values, vectors = np.linalg.eigh(overlap)
    root = (vectors * np.sqrt(values)) @ vectors.T
    generator = np.random.default_rng(137)
    factors = generator.normal(size=(2, molecule.nao, 4)) * 0.1
    density = factors @ factors.transpose(0, 2, 1)
    total = density.sum(axis=0)
    expected_dipole = np.einsum(
        "i,ij->j", molecule.atom_charges(), molecule.atom_coords()
    ) - np.einsum("ij,kji->k", total, position)
    expected_mulliken = np.einsum("ij,ji->i", total, overlap)
    expected_lowdin = np.diag(root @ total @ root)
    calculator = Calculator(
        basis=SHELLS, basis_representation=representation, device="cuda"
    )
    with (
        _system(calculator) as (library, context, system),
        _CudaProperties(library, context, system, density, capacity=3) as owner,
    ):
        outputs = owner.analyze(REQUESTED, 2)
        np.testing.assert_allclose(
            outputs["dipole"], expected_dipole, atol=3e-12, rtol=3e-12
        )
        for name, populations in (
            ("charges", expected_mulliken),
            ("lowdin_charges", expected_lowdin),
        ):
            expected = [
                2 - populations[begin:end].sum()
                for _, _, begin, end in molecule.aoslice_by_atom()
            ]
            np.testing.assert_allclose(outputs[name], expected, atol=3e-11, rtol=3e-12)
        points = generator.normal(size=(11, 3))
        orbitals = (
            molecule.eval_gto("GTOval_cart" if molecule.cart else "GTOval_sph", points)
            / normalization
        )
        for spin, matrix in ((False, total), (True, density[0] - density[1])):
            expected = np.einsum("pi,ij,pj->p", orbitals, matrix, orbitals)
            np.testing.assert_allclose(
                owner.sample(points, spin), expected, atol=3e-12, rtol=3e-12
            )
        np.testing.assert_allclose(
            owner.analyze(frozenset({"dipole"}), 2)["dipole"],
            expected_dipole,
            atol=3e-12,
            rtol=3e-12,
        )


def test_cuda_public_endpoints_do_not_use_host_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Runtime failure sentinels protect dispatch, not implementation spelling."""
    from generativeqc_compiler.dft import NativeAO

    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    calculator = Calculator(basis="sto-3g", device="cuda")
    library = calculator._library
    _bind(library)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("CUDA endpoint selected host numerical analysis")

    monkeypatch.setattr(
        library, "generativeqc_system_position_integrals_cpu_v1", forbidden
    )
    monkeypatch.setattr(library, "generativeqc_system_cross_overlap_cpu", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(NativeAO, "evaluate", forbidden)
    result = calculator.singlepoint(
        atoms, properties=("energy", "dipole", "charges", "density")
    )
    assert result.executed_backend == "cuda"
    assert result.density.device == "cuda"
    assert abs(result.charges.sum()) < 1e-11
    calculator.clear_cache()
    del calculator
    assert np.isfinite(result.density.evaluate([[0.1, 0.2, 0.3]])).all()
    result.density.write_cube(
        tmp_path / "gpu.cube", origin=(-1, -1, -1), shape=(3, 4, 5), block_size=7
    )
    assert (tmp_path / "gpu.cube").exists()


def test_cuda_native_rejects_bad_buffers_and_recovers() -> None:
    calculator = Calculator(basis=SHELLS, device="cuda")
    nao = sum(
        (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
        for shell in SHELLS
    )
    density = np.eye(nao)
    with _system(calculator) as (library, context, system):
        for candidate, count, capacity in (
            (density, density.size - 1, 3),
            (density * np.nan, density.size, 3),
            (density + np.triu(np.ones_like(density), 1), density.size, 3),
            (density, density.size, 0),
            (density, density.size, 2**31),
        ):
            handle = ctypes.c_void_p(123)
            status = library.generativeqc_electronic_properties_cuda_create_v1(
                context,
                system,
                _pointer(candidate),
                count,
                capacity,
                ctypes.byref(handle),
            )
            assert status == _native.STATUS_INVALID_ARGUMENT
            assert not handle.value
        with _CudaProperties(library, context, system, density, capacity=3) as owner:
            dipole, charges = np.empty(3), np.empty(2)
            for mulliken, lowdin, count in (
                (None, None, 1),
                (_pointer(charges), None, 2),
                (_pointer(charges), _pointer(charges), 1),
            ):
                status = library.generativeqc_electronic_properties_cuda_analyze_v1(
                    context, owner.handle, _pointer(dipole), mulliken, lowdin, count
                )
                assert status == _native.STATUS_INVALID_ARGUMENT
            output = np.empty(1)
            for points, count, spin, out_count in (
                (np.zeros((1, 3)), 1, 2, 1),
                (np.zeros((1, 3)), 1, 0, 0),
                (np.full((1, 3), np.nan), 1, 0, 1),
            ):
                status = library.generativeqc_electronic_properties_cuda_evaluate_v1(
                    context,
                    owner.handle,
                    _pointer(points),
                    count,
                    spin,
                    _pointer(output),
                    out_count,
                )
                assert status == _native.STATUS_INVALID_ARGUMENT
            status = library.generativeqc_electronic_properties_cuda_evaluate_v1(
                context, owner.handle, None, 0, 0, None, 0
            )
            assert status == _native.STATUS_SUCCESS
            assert np.isfinite(owner.sample(np.array([[0.1, 0.2, 0.3]]), False)).all()
            assert np.isfinite(owner.analyze(REQUESTED, 2)["lowdin_charges"]).all()
        library.generativeqc_electronic_properties_destroy_v1(None)


def test_cuda_owner_does_not_borrow_system_or_context() -> None:
    calculator = Calculator(basis=SHELLS, device="cuda")
    nao = sum(
        (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
        for shell in SHELLS
    )
    with _system(calculator) as (library, context, system):
        owner = _CudaProperties(library, context, system, np.eye(nao), capacity=2)
    replacement = ctypes.c_void_p()
    try:
        _native.check(
            library,
            library.generativeqc_context_create(
                ctypes.byref(calculator._context_descriptor()),
                ctypes.byref(replacement),
            ),
        )
        owner.context = replacement
        with owner:
            assert np.isfinite(owner.analyze(REQUESTED, 2)["dipole"]).all()
            assert np.isfinite(owner.sample(np.array([[0.1, 0.2, 0.3]]), False)).all()
    finally:
        library.generativeqc_context_destroy(replacement)


def test_cuda_lowdin_rejects_nonpositive_spectrum_without_poisoning_owner() -> None:
    """A rejected population request leaves dipole/sampling usable.

    Four identical AOs retain multiple null modes: a two-AO zero eigenvalue
    alone can round to a tiny positive number in the vendor eigensolver.
    """
    shell = Shell(0, 0, (Primitive(0.7, 1.0),))
    calculator = Calculator(basis=(shell,) * 4, device="cuda")
    with (
        _system(calculator, [("He", (0.3, -0.4, 0.2))]) as (library, context, system),
        _CudaProperties(library, context, system, np.eye(4) * 0.5, capacity=2) as owner,
    ):
        with pytest.raises(RuntimeError, match="positive-definite"):
            owner.analyze(frozenset({"charges"}), 1)
        np.testing.assert_allclose(
            owner.analyze(frozenset({"dipole"}), 1)["dipole"], 0, atol=1e-12
        )
        assert np.isfinite(owner.sample(np.array([[0.1, 0.2, 0.3]]), False)).all()


def test_cuda_owner_restores_callers_current_device() -> None:
    """Test actual switching independently of calculator/context setup policy."""
    library_name = ctypes.util.find_library("cudart") or "libcudart.so.12"
    runtime = ctypes.CDLL(library_name)
    runtime.cudaSetDevice.argtypes = [ctypes.c_int]
    runtime.cudaSetDevice.restype = ctypes.c_int
    for name in ("cudaGetDeviceCount", "cudaGetDevice"):
        getattr(runtime, name).argtypes = [ctypes.POINTER(ctypes.c_int)]
        getattr(runtime, name).restype = ctypes.c_int
    count, original = ctypes.c_int(), ctypes.c_int()
    assert runtime.cudaGetDeviceCount(ctypes.byref(count)) == 0
    if count.value < 2:
        pytest.skip("device restoration gate requires two Slurm-visible GPUs")
    assert runtime.cudaGetDevice(ctypes.byref(original)) == 0
    device = (original.value + 1) % count.value
    calculator = Calculator(basis=SHELLS, device="cuda", device_id=device)
    nao = sum(
        (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
        for shell in SHELLS
    )
    with _system(calculator) as (library, context, system):
        assert runtime.cudaSetDevice(original.value) == 0
        with _CudaProperties(
            library, context, system, np.eye(nao), capacity=2
        ) as owner:
            for operation in (
                lambda: owner.analyze(REQUESTED, 2),
                lambda: owner.sample(np.array([[0.1, 0.2, 0.3]]), False),
            ):
                current = ctypes.c_int()
                assert runtime.cudaGetDevice(ctypes.byref(current)) == 0
                assert current.value == original.value
                operation()
                assert runtime.cudaGetDevice(ctypes.byref(current)) == 0
                assert current.value == original.value
        current = ctypes.c_int()
        assert runtime.cudaGetDevice(ctypes.byref(current)) == 0
        assert current.value == original.value
