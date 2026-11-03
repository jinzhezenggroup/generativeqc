"""Opt-in FP64 analysis of a single accepted native mean-field state.

SCF, property analysis and real-space sampling follow the selected backend.
CUDA uses native kernels/cuSOLVER/cuBLAS, never a silent host numerical fallback.
No production operation imports PySCF or reconstructs a second SCF solution.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _native

if TYPE_CHECKING:
    from collections.abc import Iterator

    from generativeqc_compiler.dft import NativeAO
    from typing_extensions import Self

    from ._api_types import Atom, Shell
    from .batch import PreparedBatch
    from .calculator import Calculator

PROPERTY_NAMES = frozenset({"dipole", "charges", "density"})
AU_TO_DEBYE = 2.541746473


def _pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _bind(library: ctypes.CDLL) -> None:
    """Require additive ABI symbols rather than silently replaying old backends."""
    signatures = {
        "generativeqc_electronic_properties_cuda_create_v1": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ],
        "generativeqc_electronic_properties_cuda_analyze_v1": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ],
        "generativeqc_electronic_properties_cuda_evaluate_v1": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ],
        "generativeqc_electronic_properties_destroy_v1": [ctypes.c_void_p],
        "generativeqc_calculation_execute_density_v1": [
            ctypes.c_void_p,
            ctypes.POINTER(_native.ResultDescriptor),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ],
        "generativeqc_ks_batch_copy_density_v1": [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ],
        "generativeqc_system_position_integrals_cpu_v1": [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name, None)
        if function is None:
            raise NotImplementedError(
                "electronic properties require a rebuilt native library"
            )
        function.argtypes = arguments
        function.restype = None if name.endswith("destroy_v1") else ctypes.c_int


class _CudaProperties:
    """Scoped device state; native owner copies inputs and never borrows a system."""

    def __init__(
        self,
        library: ctypes.CDLL,
        context: ctypes.c_void_p,
        system: ctypes.c_void_p,
        density: np.ndarray,
        capacity: int = 4096,
    ) -> None:
        self.library, self.context = library, context
        self.handle = ctypes.c_void_p()
        _bind(library)
        density = np.ascontiguousarray(density, dtype=np.float64)
        _native.check(
            library,
            library.generativeqc_electronic_properties_cuda_create_v1(
                context,
                system,
                _pointer(density),
                density.size,
                capacity,
                ctypes.byref(self.handle),
            ),
            context=context,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.library.generativeqc_electronic_properties_destroy_v1(self.handle)
        self.handle = ctypes.c_void_p()

    def sample(self, points: np.ndarray, spin: bool) -> np.ndarray:
        points = np.ascontiguousarray(points, dtype=np.float64)
        output = np.empty(len(points))
        _native.check(
            self.library,
            self.library.generativeqc_electronic_properties_cuda_evaluate_v1(
                self.context,
                self.handle,
                _pointer(points),
                len(points),
                int(spin),
                _pointer(output),
                len(output),
            ),
            context=self.context,
        )
        return output

    def analyze(self, requested: frozenset[str], natom: int) -> dict[str, np.ndarray]:
        outputs = {}
        if "dipole" in requested:
            outputs["dipole"] = np.empty(3)
        if "charges" in requested:
            outputs["charges"], outputs["lowdin_charges"] = (
                np.empty(natom),
                np.empty(natom),
            )
        if outputs:
            _native.check(
                self.library,
                self.library.generativeqc_electronic_properties_cuda_analyze_v1(
                    self.context,
                    self.handle,
                    _pointer(outputs["dipole"]) if "dipole" in outputs else None,
                    _pointer(outputs["charges"]) if "charges" in outputs else None,
                    _pointer(outputs["lowdin_charges"])
                    if "charges" in outputs
                    else None,
                    natom if "charges" in outputs else 0,
                ),
                context=self.context,
            )
        return outputs


@dataclass(frozen=True)
class ElectronicDensity:
    """Detached mean-field electron density in the normalized public AO basis.

    ``matrix`` is the spin-summed density, shape ``(nao, nao)``. Optional
    ``spin_matrix`` is alpha minus beta; restricted states have zero spin density.
    Arrays are owned read-only copies and survive calculator/native-owner closure.
    Coordinates and sampling use Bohr and electrons/Bohr**3. Sampling explicitly
    uses FP64 native AOs on ``device`` (``"cpu"`` or ``"cuda"``), supports
    all-electron s/p/d/f bases, and does not SCF. ``device_id`` is the CUDA
    ordinal within the scheduler's visible devices, not a physical GPU number.
    """

    matrix: np.ndarray
    spin_matrix: np.ndarray | None
    _atoms: tuple[Atom, ...]
    _shells: tuple[Shell, ...]
    _representation: str
    _charge: int
    _multiplicity: int
    device: str = "cpu"
    device_id: int = 0

    @contextmanager
    def _sampler(self, block_size: int) -> Iterator[NativeAO | _CudaProperties]:
        if self.device == "cpu":
            with self._ao() as basis:
                yield basis
            return
        from .calculator import Calculator

        calculator = Calculator(
            basis=self._shells,
            device="cuda",
            device_id=self.device_id,
            basis_representation=self._representation,
        )
        library = calculator._library
        context = ctypes.c_void_p()
        _native.check(
            library,
            library.generativeqc_context_create(
                ctypes.byref(calculator._context_descriptor()),
                ctypes.byref(context),
            ),
        )
        system = None
        try:
            system = calculator._create_native_system(
                context,
                self._atoms,
                self._charge,
                self._multiplicity,
            )
            density = (
                self.matrix
                if self.spin_matrix is None
                else np.stack(
                    [
                        (self.matrix + self.spin_matrix) * 0.5,
                        (self.matrix - self.spin_matrix) * 0.5,
                    ]
                )
            )
            with _CudaProperties(
                library, context, system, density, block_size
            ) as owner:
                yield owner
        finally:
            if system is not None:
                library.generativeqc_system_destroy(system)
            library.generativeqc_context_destroy(context)

    def _ao(self) -> NativeAO:
        from generativeqc_compiler.dft import NativeAO

        return NativeAO(
            self._atoms,
            basis=self._shells,
            representation=self._representation,
            charge=self._charge,
            multiplicity=self._multiplicity,
        )

    def _sample(
        self, basis: NativeAO | _CudaProperties, points: np.ndarray, spin: bool
    ) -> np.ndarray:
        if isinstance(basis, _CudaProperties):
            return basis.sample(points, spin)
        matrix = self.spin_matrix if spin else self.matrix
        if matrix is None:
            return np.zeros(len(points))
        values = basis.evaluate(points, order=0)[0]
        return np.einsum("pi,ij,pj->p", values, matrix, values, optimize=True)

    def evaluate(
        self, points: Any, *, spin: bool = False, block_size: int = 4096
    ) -> np.ndarray:
        """Sample finite real ``(n,3)`` Bohr points on the recorded CPU/CUDA device.

        Return an owned ``(n,)`` electron density, or alpha-minus-beta spin
        density. Invalid shapes/nonfinite values and nonpositive block sizes
        raise ValueError. Native evaluation/allocation failures propagate.
        Calls prepare independent AO owners and may run concurrently.
        """
        raw = np.asarray(points)
        if np.iscomplexobj(raw) or raw.ndim != 2 or raw.shape[1:] != (3,):
            raise ValueError("density points must be a real (n,3) array")
        points = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(points).all():
            raise ValueError("density points must be finite")
        if type(block_size) is not int or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        result = np.empty(len(points))
        if not len(points):
            return result
        with self._sampler(block_size) as basis:
            for begin in range(0, len(points), block_size):
                end = min(begin + block_size, len(points))
                result[begin:end] = self._sample(basis, points[begin:end], spin)
        return result

    def write_cube(
        self,
        path: str | Path,
        *,
        origin: Any,
        shape: Any,
        spacing: Any = 0.2,
        spin: bool = False,
        block_size: int = 4096,
    ) -> None:
        """Write an axis-aligned Cube file with streamed CPU/CUDA sampling.

        ``origin`` is three finite Bohr values, ``shape`` three positive integers,
        and ``spacing`` a positive Bohr scalar or xyz triple. Cube values use
        electrons/Bohr**3 (alpha-minus-beta for ``spin=True``), z index fastest.
        The caller owns the path; an existing file is replaced. Invalid grid
        parameters raise ValueError before opening; I/O/native failures propagate
        and may leave a partial file. Memory is O(block_size*nao + nao**2).
        """
        raw_origin, raw_spacing = np.asarray(origin), np.asarray(spacing)
        raw_shape = np.asarray(shape)
        if (
            np.iscomplexobj(raw_origin)
            or raw_origin.shape != (3,)
            or np.iscomplexobj(raw_spacing)
            or raw_spacing.shape not in ((), (3,))
            or raw_shape.shape != (3,)
            or raw_shape.dtype.kind not in "iu"
            or np.any(raw_shape < 1)
        ):
            raise ValueError(
                "cube requires real origin/spacing and three positive integer counts"
            )
        origin = np.asarray(raw_origin, dtype=np.float64)
        spacing = np.broadcast_to(np.asarray(raw_spacing, dtype=np.float64), (3,))
        if (
            not np.isfinite(origin).all()
            or not np.isfinite(spacing).all()
            or np.any(spacing <= 0)
        ):
            raise ValueError("cube origin/spacing must be finite with positive spacing")
        if type(block_size) is not int or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        counts = tuple(int(value) for value in raw_shape)
        total = counts[0] * counts[1] * counts[2]
        if total > np.iinfo(np.int64).max:
            raise ValueError("cube grid exceeds the supported index range")
        with (
            self._sampler(block_size) as basis,
            Path(path).open("w", encoding="ascii") as output,
        ):
            output.write("GenerativeQC mean-field density\n")
            output.write(
                "Alpha minus beta spin density\n" if spin else "Electron density\n"
            )
            output.write(
                f"{len(self._atoms):5d}"
                + "".join(f" {value: .10f}" for value in origin)
                + "\n"
            )
            for axis, count in enumerate(counts):
                vector = np.zeros(3)
                vector[axis] = spacing[axis]
                output.write(
                    f"{count:5d}"
                    + "".join(f" {value: .10f}" for value in vector)
                    + "\n"
                )
            output.writelines(
                f"{atom.atomic_number:5d} {float(atom.atomic_number): .10f}"
                + "".join(f" {value: .10f}" for value in atom.position)
                + "\n"
                for atom in self._atoms
            )
            written = 0
            for begin in range(0, total, block_size):
                indices = np.arange(
                    begin, min(begin + block_size, total), dtype=np.int64
                )
                xyz = np.column_stack(np.unravel_index(indices, counts))
                values = self._sample(basis, origin + xyz * spacing, spin)
                for value in values:
                    output.write(f" {value: .10e}")
                    written += 1
                    if written % 6 == 0:
                        output.write("\n")
            if written % 6:
                output.write("\n")


class DensityCapture:
    """Per-call detached buffers; never store density on a shared calculator."""

    def __init__(
        self,
        calculator: Calculator,
        atoms: tuple[Atom, ...],
        charge: int,
        multiplicity: int,
        requested: frozenset[str],
    ) -> None:
        from .ecp import resolve_ecp

        if calculator._capabilities.family not in (
            "hartree_fock",
            "density_functional",
        ):
            raise ValueError(
                "method does not support properties: electronic properties require HF or KS, "
                "not correlated or xTB densities"
            )
        if calculator._resource_budget is not None:
            raise NotImplementedError(
                "electronic-property analysis does not yet support resource budgets"
            )
        if any(resolve_ecp(calculator._basis, atoms)[0]):
            raise NotImplementedError(
                "electronic properties currently require an all-electron basis"
            )
        _bind(calculator._library)
        self.calculator = calculator
        self.atoms, self.charge, self.multiplicity = atoms, charge, multiplicity
        self.requested = requested & PROPERTY_NAMES
        self.shells = calculator._shells_for_atoms(atoms)
        if ("density" in requested or calculator._device_name == "cuda") and any(
            shell.angular_momentum > 3 for shell in self.shells
        ):
            raise NotImplementedError(
                "real-space density and CUDA properties support basis functions through f"
            )
        sizes = [
            2 * shell.angular_momentum + 1
            if calculator._basis_representation == _native.BASIS_SPHERICAL
            else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
            for shell in self.shells
        ]
        self.ao_atoms = np.repeat([shell.atom_index for shell in self.shells], sizes)
        self.nao = sum(sizes)
        unrestricted = calculator._method == _native.METHOD_UHF or (
            calculator._ks_options is not None
            and calculator._ks_options.method_ir.spin == "polarized"
        )
        self.buffer = np.empty((2 if unrestricted else 1, self.nao, self.nao))

    def execute(
        self, calculation: ctypes.c_void_p, descriptor: _native.ResultDescriptor
    ) -> int:
        return self.calculator._library.generativeqc_calculation_execute_density_v1(
            calculation,
            ctypes.byref(descriptor),
            _pointer(self.buffer),
            self.buffer.size,
        )

    def from_batch(self, batch: PreparedBatch) -> dict[str, Any]:
        library = self.calculator._library
        _native.check(
            library,
            library.generativeqc_ks_batch_copy_density_v1(
                batch._batch,
                0,
                _pointer(self.buffer),
                self.buffer.size,
            ),
            context=batch._context,
        )
        system = self.calculator._create_native_system(
            batch._context,
            self.atoms,
            self.charge,
            self.multiplicity,
        )
        try:
            return self.analyze(batch._context, system)
        finally:
            library.generativeqc_system_destroy(system)

    def analyze(
        self, context: ctypes.c_void_p, system: ctypes.c_void_p
    ) -> dict[str, Any]:
        """Contract exactly this accepted state; charge partitioning is explicit."""
        library = self.calculator._library
        if not np.isfinite(self.buffer).all():
            raise RuntimeError("native density contains nonfinite values")
        total = self.buffer.sum(axis=0)
        outputs = {}
        cuda = self.calculator._device_name == "cuda"
        if cuda and self.requested & {"dipole", "charges"}:
            with _CudaProperties(library, context, system, self.buffer) as owner:
                outputs.update(owner.analyze(self.requested, len(self.atoms)))
        if not cuda and "dipole" in self.requested:
            positions = np.empty((3, self.nao, self.nao))
            _native.check(
                library,
                library.generativeqc_system_position_integrals_cpu_v1(
                    context,
                    system,
                    _pointer(positions),
                    positions.size,
                ),
                context=context,
            )
            nuclear = sum(
                (atom.atomic_number * np.asarray(atom.position) for atom in self.atoms),
                np.zeros(3),
            )
            outputs["dipole"] = nuclear - np.einsum("ij,kji->k", total, positions)
        if not cuda and "charges" in self.requested:
            overlap = np.empty_like(total)
            _native.check(
                library,
                library.generativeqc_system_cross_overlap_cpu(
                    context,
                    system,
                    system,
                    _pointer(overlap),
                    overlap.size,
                ),
                context=context,
            )
            eigenvalues, vectors = np.linalg.eigh(overlap)
            if eigenvalues[0] <= 0:
                raise ValueError("Lowdin charges require a positive-definite overlap")
            root = (vectors * np.sqrt(eigenvalues)) @ vectors.T
            nuclear = np.array([atom.atomic_number for atom in self.atoms], dtype=float)
            mulliken = np.einsum("ij,ji->i", total, overlap)
            lowdin = np.einsum("ij,jk,ki->i", root, total, root, optimize=True)
            outputs["charges"] = nuclear - np.bincount(
                self.ao_atoms, weights=mulliken, minlength=len(self.atoms)
            )
            outputs["lowdin_charges"] = nuclear - np.bincount(
                self.ao_atoms, weights=lowdin, minlength=len(self.atoms)
            )
        if "density" in self.requested:
            spin = self.buffer[0] - self.buffer[1] if len(self.buffer) == 2 else None
            total.setflags(write=False)
            if spin is not None:
                spin.setflags(write=False)
            outputs["density"] = ElectronicDensity(
                total,
                spin,
                self.atoms,
                self.shells,
                self.calculator._representation_name,
                self.charge,
                self.multiplicity,
                self.calculator._device_name,
                self.calculator._device_id,
            )
        return outputs
