"""Generated integral-response providers for production semilocal RKS Hessians.

This module owns only the direct all-electron Cartesian integral topology and
generated first/second nuclear derivative contractions required by the bounded
LDA/PBE RKS Hessian path.  It borrows a live NativeAO owner; no post-HF source,
SCF implementation, or method-specific response algebra lives here.
"""

from __future__ import annotations

import shutil
import typing
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.resources import ResourceBudget
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.integral.blocks import TensorLayout, WeightTile
from generativeqc_compiler.integral.first_derivatives_execute import (
    FirstDerivativeEvaluator,
    compile_first_derivative,
)
from generativeqc_compiler.integral.first_derivatives_native import (
    first_component_identity,
)
from generativeqc_compiler.integral.first_directional import DirectionalMatrixTerm
from generativeqc_compiler.integral.one_electron_derivatives import (
    build_one_electron_derivative_ir,
)
from generativeqc_compiler.integral.second_derivatives import (
    build_eri_second_ir,
    build_one_electron_second_ir,
)
from generativeqc_compiler.integral.second_derivatives_execute import (
    PreparedSecondDerivative,
    compile_second_derivative,
)
from generativeqc_compiler.integral.second_derivatives_inputs import (
    prepare_second_shell_stream,
)
from generativeqc_compiler.integral.second_order_layout import (
    SecondAtomMap,
    second_coordinate_tiles,
)
from generativeqc_compiler.integral.shell_spec import cartesian_components
from generativeqc_compiler.integral.weight_pullback import (
    normalized_cartesian_components,
    normalized_radial_primitives,
)
from generativeqc_compiler.integral.weighted_eri import build_weighted_eri_ir

__all__ = [
    "RKSIntegralTopology",
    "checked_direction",
    "generated_directional_semilocal_rks_integral_first_order",
    "generated_weighted_first_integral_gradient",
    "generated_weighted_second_integral_hvp",
    "nuclear_hvp_from_topology",
    "rks_integral_topology",
]

SEMILOCAL_RKS_FIRST_ERI_TERMS = (
    DirectionalMatrixTerm(0, (0, 1), (2, 3), 1.0),
)
_COMPILE_CACHE: dict[object, object] = {}


@dataclass(frozen=True)
class RKSIntegralTopology:
    """Borrowed Cartesian AO topology for generated nuclear integral derivatives."""

    basis: NativeAO
    atoms: tuple[typing.Any, ...]
    shells: tuple[typing.Any, ...]
    shell_sizes: tuple[int, ...]
    nbf: int
    representation: str = "cartesian"
    auxiliary_shells: tuple[typing.Any, ...] = ()

    @classmethod
    def from_basis(cls, basis: typing.Any) -> "RKSIntegralTopology":
        if not isinstance(basis, NativeAO):
            raise TypeError("RKS Hessian integral topology requires NativeAO")
        if not basis._handle:
            raise RuntimeError("RKS Hessian AO basis is closed")
        if basis.representation != "cartesian":
            raise NotImplementedError(
                "RKS Hessian integral topology requires Cartesian AOs"
            )
        sizes = tuple(
            (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
            for shell in basis.shells
        )
        if sum(sizes) != basis.nao or not basis.atoms:
            raise ValueError("RKS Hessian AO topology is inconsistent")
        if any(shell.angular_momentum > 3 for shell in basis.shells):
            raise ValueError("RKS Hessian integral providers support through f")
        return cls(
            basis=basis,
            atoms=tuple(basis.atoms),
            shells=tuple(basis.shells),
            shell_sizes=sizes,
            nbf=basis.nao,
        )

    def check_current(self) -> None:
        if not self.basis._handle:
            raise RuntimeError("RKS Hessian AO basis is closed")


def rks_integral_topology(operator: typing.Any) -> RKSIntegralTopology:
    """Bind the current RKS response's borrowed NativeAO owner."""
    validate = getattr(operator, "validate_current", None)
    if not callable(validate):
        raise TypeError("RKS Hessian requires a live response owner")
    validate()
    basis = getattr(operator, "basis", None)
    if basis is None:
        basis = getattr(operator, "_basis", None)
    return RKSIntegralTopology.from_basis(basis)


def checked_direction(direction: typing.Any, natoms: int) -> np.ndarray:
    value = np.asarray(direction)
    if (
        value.shape != (natoms, 3)
        or value.dtype.kind not in "iuf"
        or np.iscomplexobj(value)
        or not np.isfinite(value).all()
    ):
        raise ValueError("direction must be finite real with shape (natoms, 3)")
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("direction must be representable in FP64")
    return result


def _checked_ao_weight(value: typing.Any, nbf: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (nbf, nbf)
        or array.dtype.kind not in "iuf"
        or np.iscomplexobj(array)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite real AO matrix")
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.allclose(result, result.T, atol=2e-10, rtol=2e-12):
        raise ValueError(f"{name} must be symmetric")
    return result


class _FirstDerivativeProvider:
    def __init__(self, topology: RKSIntegralTopology, cache: typing.Any) -> None:
        topology.check_current()
        self.topology = topology
        self.compiler = CppCompilerAdapter(Path(shutil.which("c++") or "c++"))
        self.cache = Path(cache) / "first-cache"
        self.evaluators: dict[str, FirstDerivativeEvaluator] = {}
        self.shells = topology.shells
        self.offsets = np.cumsum((0, *topology.shell_sizes))
        self.primitives = tuple(
            normalized_radial_primitives(
                shell.angular_momentum,
                tuple((p.exponent, p.coefficient) for p in shell.primitives),
            )
            for shell in topology.shells
        )
        self.coords = np.asarray(
            [atom.position for atom in topology.atoms], dtype=np.float64
        )

    def raw_tiles(
        self,
        ir: typing.Any,
        slots: tuple[int, ...],
        atom_indices: tuple[int, ...],
    ) -> typing.Iterator[tuple[tuple[int, ...], np.ndarray]]:
        angular = ir.signature.angular
        count = ir.signature.component_count
        shape = ir.signature.component_shape
        scales = np.asarray(
            [
                weight
                for _, weight in normalized_cartesian_components(
                    angular, np.ones(shape)
                )
            ]
        )
        for start in range(0, count, 64):
            indices = tuple(range(start, min(start + 64, count)))
            key = first_component_identity(ir, indices)
            if key not in self.evaluators:
                artifact = compile_first_derivative(
                    ir,
                    self.compiler,
                    self.cache,
                    component_indices=indices,
                )
                self.evaluators[key] = FirstDerivativeEvaluator(artifact)
            values = self.evaluators[key].contract(
                tuple(self.primitives[i] for i in slots),
                self.coords[list(atom_indices)],
            )
            values *= scales[list(indices), None]
            for row, index in enumerate(indices):
                yield (
                    np.unravel_index(index, shape),
                    values[row, 1:].reshape(-1, 3),
                )


def _contract_directional_first_order(
    topology: RKSIntegralTopology,
    density: np.ndarray,
    direction: np.ndarray,
    *,
    cache: typing.Any,
) -> tuple[np.ndarray, np.ndarray]:
    provider = _FirstDerivativeProvider(topology, cache)
    shells, offsets = provider.shells, provider.offsets
    charges = np.asarray(
        [atom.atomic_number for atom in topology.atoms], dtype=np.float64
    )
    frozen = np.zeros((topology.nbf, topology.nbf))
    overlap = np.zeros_like(frozen)

    def accumulate(
        out: np.ndarray,
        atoms: tuple[int, ...],
        derivative: np.ndarray,
        u: int,
        v: int,
        coefficient: float = 1.0,
    ) -> None:
        out[u, v] += coefficient * np.einsum(
            "ca,ca->", derivative, direction[list(atoms)]
        )

    if not np.any(direction):
        return immutable(frozen), immutable(overlap)

    for a, b in product(range(len(shells)), repeat=2):
        angular = (shells[a].angular_momentum, shells[b].angular_momentum)
        atoms = (shells[a].atom_index, shells[b].atom_index)
        for family, out in (("overlap", overlap), ("kinetic", frozen)):
            ir = build_one_electron_derivative_ir(family, angular)
            for (u, v), gradient in provider.raw_tiles(ir, (a, b), atoms):
                accumulate(out, atoms, gradient, offsets[a] + u, offsets[b] + v)
        for nucleus, charge in enumerate(charges):
            ir = build_one_electron_derivative_ir(
                "nuclear_attraction", angular, charge=float(charge)
            )
            centers = (*atoms, nucleus)
            for (u, v), gradient in provider.raw_tiles(ir, (a, b), centers):
                accumulate(
                    frozen,
                    centers,
                    gradient,
                    offsets[a] + u,
                    offsets[b] + v,
                )

    for slots in product(range(len(shells)), repeat=4):
        angular = tuple(shells[i].angular_momentum for i in slots)
        atoms = tuple(shells[i].atom_index for i in slots)
        ir = build_weighted_eri_ir(angular)
        for component, gradient in provider.raw_tiles(ir, slots, atoms):
            u, v, w, x = (
                offsets[shell] + c
                for shell, c in zip(slots, component, strict=True)
            )
            ao = (u, v, w, x)
            for term in SEMILOCAL_RKS_FIRST_ERI_TERMS:
                i, j = (ao[k] for k in term.output_pair)
                k, l = (ao[k] for k in term.weight_pair)
                accumulate(
                    frozen,
                    atoms,
                    gradient,
                    i,
                    j,
                    term.coefficient * density[k, l],
                )

    if not np.isfinite(frozen).all() or not np.isfinite(overlap).all():
        raise FloatingPointError("generated RKS nuclear source is nonfinite")
    return immutable(frozen), immutable(overlap)


def generated_directional_semilocal_rks_integral_first_order(
    topology: RKSIntegralTopology,
    density: typing.Any,
    direction: typing.Any,
    *,
    cache: typing.Any = ".artifacts",
) -> tuple[np.ndarray, np.ndarray]:
    topology.check_current()
    vector = checked_direction(direction, len(topology.atoms))
    ao_density = _checked_ao_weight(
        density, topology.nbf, "RKS reference density"
    )
    return _contract_directional_first_order(
        topology, ao_density, vector, cache=cache
    )


def generated_weighted_first_integral_gradient(
    topology: RKSIntegralTopology,
    source_name: str,
    *,
    pair_weights: typing.Any = None,
    eri_shell_weights: typing.Any = None,
    cache: typing.Any = ".artifacts",
) -> np.ndarray:
    if source_name not in ("one_electron", "coulomb", "overlap_pulay"):
        raise ValueError("unknown stationary first-integral source")
    topology.check_current()
    provider = _FirstDerivativeProvider(topology, cache)
    shells, offsets = provider.shells, provider.offsets
    natom, nbf = len(topology.atoms), topology.nbf
    charges = np.asarray(
        [atom.atomic_number for atom in topology.atoms], dtype=np.float64
    )
    result = np.zeros((natom, 3), dtype=np.float64)

    def accumulate(
        atoms: tuple[int, ...], derivative: np.ndarray, coefficient: typing.Any
    ) -> None:
        coefficient = float(coefficient)
        if coefficient == 0.0:
            return
        for center, atom in enumerate(atoms):
            result[atom] += coefficient * derivative[center]

    if source_name != "coulomb":
        if eri_shell_weights is not None:
            raise ValueError("pair first derivative cannot consume ERI shell weights")
        weights = _checked_ao_weight(pair_weights, nbf, "pair first derivative weight")
        for a, b in product(range(len(shells)), repeat=2):
            angular = (shells[a].angular_momentum, shells[b].angular_momentum)
            atoms = (shells[a].atom_index, shells[b].atom_index)
            sa = slice(offsets[a], offsets[a + 1])
            sb = slice(offsets[b], offsets[b + 1])
            block = weights[sa, sb]
            families = (
                ("kinetic",),
                ("overlap",),
            )[source_name == "overlap_pulay"]
            for family in families:
                ir = build_one_electron_derivative_ir(family, angular)
                for (u, v), gradient in provider.raw_tiles(ir, (a, b), atoms):
                    accumulate(atoms, gradient, block[u, v])
            if source_name == "one_electron":
                for nucleus, charge in enumerate(charges):
                    ir = build_one_electron_derivative_ir(
                        "nuclear_attraction", angular, charge=float(charge)
                    )
                    centers = (*atoms, nucleus)
                    for (u, v), gradient in provider.raw_tiles(
                        ir, (a, b), centers
                    ):
                        accumulate(centers, gradient, block[u, v])
    else:
        if pair_weights is not None or not callable(eri_shell_weights):
            raise ValueError(
                "coulomb first derivative requires shell-local ERI weights"
            )
        for slots in product(range(len(shells)), repeat=4):
            angular = tuple(shells[i].angular_momentum for i in slots)
            atoms = tuple(shells[i].atom_index for i in slots)
            shape = tuple(offsets[i + 1] - offsets[i] for i in slots)
            weights = np.asarray(eri_shell_weights(slots), dtype=np.float64)
            if weights.shape != shape or not np.isfinite(weights).all():
                raise ValueError(
                    "ERI shell weights must be finite with the ordered shell shape"
                )
            ir = build_weighted_eri_ir(angular)
            for component, gradient in provider.raw_tiles(ir, slots, atoms):
                accumulate(atoms, gradient, weights[component])

    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite weighted first-integral contraction")
    return immutable(result)


def _compile_cached(
    key: object,
    build_ir: typing.Any,
    ir_extra: dict[str, object],
    adapter: CppCompilerAdapter,
    cache: Path,
    output_indices: tuple[int, ...],
    component_indices: tuple[int, ...],
) -> typing.Any:
    cache_key = (
        Path(cache).resolve(),
        adapter,
        key,
        output_indices,
        component_indices,
    )
    artifact = _COMPILE_CACHE.get(cache_key)
    if artifact is None or not artifact.native.library.is_file():
        ir = build_ir(**ir_extra)
        artifact = compile_second_derivative(
            ir,
            adapter,
            cache,
            output_indices=output_indices,
            component_indices=component_indices,
        )
        _COMPILE_CACHE[cache_key] = artifact
    return artifact


def _component_tiles(count: int, chunk: int = 64) -> typing.Iterator[tuple[int, ...]]:
    for start in range(0, count, chunk):
        yield tuple(range(start, min(start + chunk, count)))


def _scatter_hvp(
    full: np.ndarray,
    center_indices: tuple[int, ...],
    center_atoms: tuple[int, ...],
    natom: int,
) -> np.ndarray:
    mapping = SecondAtomMap(center_indices, center_atoms)
    block = mapping.scatter_hvp(full.reshape(len(center_indices), 3))
    out = np.zeros((natom, 3))
    for i, atom in enumerate(mapping.atom_indices):
        out[atom] += block[i]
    return out


def _run_kernel_hvp(
    data: dict[str, typing.Any],
    key: object,
    build_ir: typing.Any,
    ir_extra: dict[str, object],
    primitives: tuple[object, ...],
    centers: np.ndarray,
    weight_flat: np.ndarray,
    component_count: int,
    direction: np.ndarray,
) -> np.ndarray:
    center_atoms = typing.cast(tuple[int, ...], ir_extra["_center_atoms"])
    extra = {k: v for k, v in ir_extra.items() if k != "_center_atoms"}
    ir = build_ir(**extra)
    center_indices = ir.requested_derivative_centers
    mapping = SecondAtomMap(center_indices, center_atoms)
    center_direction = mapping.expand_direction(
        direction[list(mapping.atom_indices)]
    )
    full = np.zeros(len(center_indices) * 3)
    signature = ir.signature
    for ao_chunk in _component_tiles(component_count):
        weights = np.zeros(component_count)
        weights[list(ao_chunk)] = weight_flat[list(ao_chunk)]
        for output_indices in second_coordinate_tiles(
            center_indices, packing="dense", hvp=True
        ):
            artifact = _compile_cached(
                (key, "hvp"),
                build_ir,
                extra,
                data["adapter"],
                data["cache"],
                tuple(output_indices),
                ao_chunk,
            )
            tile = WeightTile(
                TensorLayout(signature.tensor_indices, signature.component_shape),
                weights,
            )
            stream = prepare_second_shell_stream(
                artifact,
                primitives,
                centers,
                tile,
                public_signature=signature,
                projections=None,
                direction=center_direction,
            )
            with PreparedSecondDerivative(
                artifact,
                record_capacity=8,
                budget=data["resource_budget"],
                device_id=0,
            ) as plan:
                result = plan.contract(stream, profile=True)
            data["second_executions"].append(result.diagnostics)
            full[list(output_indices)] += np.asarray(result.values).sum(axis=0)
    return _scatter_hvp(
        full,
        center_indices,
        center_atoms,
        data["state"].nat,
    )


def _second_data(
    topology: RKSIntegralTopology,
    cache: typing.Any,
    budget_bytes: int,
) -> dict[str, typing.Any]:
    topology.check_current()
    if type(budget_bytes) is not int or not 0 < budget_bytes < 2**63:
        raise ValueError("second-integral budget must be a positive int64")
    natom = len(topology.atoms)
    output_bytes = natom * 3 * np.dtype(np.float64).itemsize
    if output_bytes > budget_bytes:
        raise MemoryError(
            "weighted second-derivative HVP output exceeds the provider budget"
        )
    geometry = SimpleNamespace(
        nat=natom,
        offsets=np.cumsum((0, *topology.shell_sizes)),
        Z=np.asarray(
            [atom.atomic_number for atom in topology.atoms], dtype=np.float64
        ),
        coords=np.asarray(
            [atom.position for atom in topology.atoms], dtype=np.float64
        ),
    )
    primitives = tuple(
        normalized_radial_primitives(
            shell.angular_momentum,
            tuple((p.exponent, p.coefficient) for p in shell.primitives),
        )
        for shell in topology.shells
    )
    return {
        "state": geometry,
        "shells": topology.shells,
        "primitives": primitives,
        "adapter": CppCompilerAdapter(Path(shutil.which("c++") or "c++")),
        "cache": Path(cache) / "second-cache-cpu",
        "backend": "cpu",
        "device_id": None,
        "budget_bytes": budget_bytes,
        "output_accumulator_bytes": output_bytes,
        "resource_budget": ResourceBudget(
            host_bytes=budget_bytes, device_bytes=0
        ),
        "second_executions": [],
    }


def _second_diagnostics(data: dict[str, typing.Any]) -> dict[str, typing.Any]:
    executions = data["second_executions"]
    return {
        "backend": "cpu-generated-weighted-hvp",
        "provider_backend": "cpu",
        "device_id": None,
        "budget_bytes": data["budget_bytes"],
        "output_accumulator_bytes": data["output_accumulator_bytes"],
        "program_identities": tuple(
            sorted({item["program_identity"] for item in executions})
        ),
        "native_artifacts": tuple(
            sorted({item["native_artifact"] for item in executions})
        ),
        "executions": len(executions),
        "primitive_records": sum(item["records"] for item in executions),
        "record_batches": sum(item["chunks"] for item in executions),
        "record_batch_uploads": 0,
        "result_tile_downloads": 0,
        "raw_hessian_downloads": 0,
        "intermediate_matrix_downloads": 0,
        "peak_host_bytes": max(
            (
                item["resources"]["peak_bytes"].get("host", 0)
                for item in executions
            ),
            default=0,
        ),
        "peak_device_bytes": 0,
        "device_timing_ms": None,
    }


def _run_one_electron_hvp(
    data: dict[str, typing.Any],
    family: str,
    weight: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    state = data["state"]
    shells = data["shells"]
    total = np.zeros((state.nat, 3))
    for a, b in product(range(len(shells)), repeat=2):
        la, lb = shells[a].angular_momentum, shells[b].angular_momentum
        na, nb = len(cartesian_components(la)), len(cartesian_components(lb))
        wa = weight[
            state.offsets[a] : state.offsets[a] + na,
            state.offsets[b] : state.offsets[b] + nb,
        ].reshape(na, nb)
        primitives = (data["primitives"][a], data["primitives"][b])
        atom_a, atom_b = shells[a].atom_index, shells[b].atom_index
        if family == "nuclear_attraction":
            for nucleus in range(state.nat):
                centers = np.asarray(
                    [
                        state.coords[atom_a],
                        state.coords[atom_b],
                        state.coords[nucleus],
                    ]
                )
                total += _run_kernel_hvp(
                    data,
                    (family, la, lb, float(state.Z[nucleus])),
                    build_one_electron_second_ir,
                    {
                        "family": family,
                        "angular": (la, lb),
                        "charge": float(state.Z[nucleus]),
                        "output": "weighted_hvp",
                        "_center_atoms": (atom_a, atom_b, nucleus),
                    },
                    primitives,
                    centers,
                    wa.ravel(),
                    na * nb,
                    direction,
                )
        else:
            centers = np.asarray(
                [state.coords[atom_a], state.coords[atom_b]]
            )
            total += _run_kernel_hvp(
                data,
                (family, la, lb),
                build_one_electron_second_ir,
                {
                    "family": family,
                    "angular": (la, lb),
                    "output": "weighted_hvp",
                    "_center_atoms": (atom_a, atom_b),
                },
                primitives,
                centers,
                wa.ravel(),
                na * nb,
                direction,
            )
    return total


def _run_eri_hvp(
    data: dict[str, typing.Any],
    shell_weights: typing.Callable[[tuple[int, int, int, int]], typing.Any],
    direction: np.ndarray,
) -> np.ndarray:
    state, shells = data["state"], data["shells"]
    total = np.zeros((state.nat, 3))
    for slots in product(range(len(shells)), repeat=4):
        angular = tuple(shells[i].angular_momentum for i in slots)
        extents = tuple(len(cartesian_components(value)) for value in angular)
        weights = np.asarray(shell_weights(slots), dtype=np.float64)
        if weights.shape != extents or not np.isfinite(weights).all():
            raise ValueError(
                "four-center shell weights must be finite with the ordered "
                "Cartesian shell shape"
            )
        primitives = tuple(data["primitives"][i] for i in slots)
        atoms = tuple(shells[i].atom_index for i in slots)
        centers = np.asarray([state.coords[atom] for atom in atoms])
        total += _run_kernel_hvp(
            data,
            ("eri", *angular),
            build_eri_second_ir,
            {
                "angular": angular,
                "output": "weighted_hvp",
                "_center_atoms": atoms,
            },
            primitives,
            centers,
            weights.ravel(),
            int(np.prod(extents)),
            direction,
        )
    return total


def generated_weighted_second_integral_hvp(
    topology: RKSIntegralTopology,
    source_name: str,
    direction: typing.Any,
    *,
    pair_weights: typing.Any = None,
    eri_shell_weights: typing.Any = None,
    cache: typing.Any = ".artifacts",
    budget_bytes: int = 64 << 20,
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    if source_name not in ("one_electron", "coulomb", "overlap_pulay"):
        raise ValueError("unknown stationary second-integral source")
    data = _second_data(topology, cache, budget_bytes)
    vector = checked_direction(direction, data["state"].nat)
    if source_name == "coulomb":
        if pair_weights is not None or not callable(eri_shell_weights):
            raise ValueError("coulomb second HVP requires shell-local ERI weights")
        value = _run_eri_hvp(data, eri_shell_weights, vector)
    else:
        if eri_shell_weights is not None:
            raise ValueError("pair second HVP cannot consume ERI shell weights")
        weights = _checked_ao_weight(
            pair_weights, topology.nbf, "pair second HVP weight"
        )
        if source_name == "one_electron":
            value = _run_one_electron_hvp(
                data, "kinetic", weights, vector
            ) + _run_one_electron_hvp(
                data, "nuclear_attraction", weights, vector
            )
        else:
            value = _run_one_electron_hvp(data, "overlap", weights, vector)
    if not np.isfinite(value).all():
        raise FloatingPointError("nonfinite generated second-integral HVP")
    diagnostic = _second_diagnostics(data)
    diagnostic.update(source=source_name, full_ao_rank_four_weights=False)
    return immutable(value), diagnostic


def nuclear_hvp_from_topology(
    topology: RKSIntegralTopology, direction: typing.Any
) -> np.ndarray:
    topology.check_current()
    vector = checked_direction(direction, len(topology.atoms))
    coords = np.asarray(
        [atom.position for atom in topology.atoms], dtype=np.float64
    )
    charges = np.asarray(
        [atom.atomic_number for atom in topology.atoms], dtype=np.float64
    )
    out = np.zeros((len(topology.atoms), 3))
    for a in range(len(topology.atoms)):
        for b in range(a + 1, len(topology.atoms)):
            r = coords[a] - coords[b]
            distance = np.linalg.norm(r)
            if not np.isfinite(distance) or distance <= 0:
                raise ValueError("nuclear Hessian requires distinct finite centers")
            unit = r / distance
            block = (
                charges[a]
                * charges[b]
                / distance**3
                * (3.0 * np.outer(unit, unit) - np.eye(3))
            )
            contribution = block @ (vector[a] - vector[b])
            out[a] += contribution
            out[b] -= contribution
    return immutable(out)
