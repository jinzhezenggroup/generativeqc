"""Coupled unrestricted stationary CPKS and nuclear metric-density response."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from generativeqc_compiler.common.arrays import immutable

from .profiles import canonical_hash
from .response_solver import GMRESOptions, solve
from .rks_response import _basis_hash, _geometry_hash
from .stationary_nuclear import StationaryNuclearResponse


@dataclass(frozen=True)
class SpinRotationLayout:
    """Independent ordered occupied/virtual spaces, including empty spin blocks."""

    occupied: tuple[int, int]
    orbitals: int

    @property
    def sizes(self) -> tuple[int, int]:
        """Number of alpha and beta rotation unknowns."""
        return tuple(count * (self.orbitals - count) for count in self.occupied)

    @property
    def dimension(self) -> int:
        """Combined nonredundant response dimension."""
        return sum(self.sizes)

    def split(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate and unpack the common alpha-then-beta response ABI."""
        vector = np.asarray(vector)
        if (
            vector.shape != (self.dimension,)
            or np.iscomplexobj(vector)
            or not np.isfinite(vector).all()
        ):
            raise ValueError("spin response vector has an invalid shape or values")
        return tuple(
            block.reshape(count, self.orbitals - count)
            for block, count in zip(
                np.split(vector, (self.sizes[0],)), self.occupied, strict=True
            )
        )

    def pack(self, blocks: typing.Iterable[np.ndarray]) -> np.ndarray:
        """Join physical spin blocks without averaging or occupation factors."""
        return np.concatenate([block.reshape(-1) for block in blocks])


class SpinCPKSOperator:
    """Native-state-bound UKS Jacobian using the complete supplied Fock map.

    Spin densities carry occupations 1/0, not restricted occupations 2/0.
    Coulomb and nonlocal XC couple both spin blocks in every response action.
    This owner uses the installed true-residual solver; no SCF tape, orbital
    denominator clamping or reference-oracle implementation enters production.
    """

    def __init__(
        self,
        state: typing.Any,
        basis: typing.Any,
        backend: typing.Any,
        kernel: typing.Any,
    ) -> None:
        self.state, self.backend, self.xc_kernel = state, backend, kernel
        if state.identity.spin != "polarized" or state.density.shape[0] != 2:
            raise ValueError("coupled CPKS requires two physical spin densities")
        self.layout = SpinRotationLayout(
            tuple(int(np.sum(row)) for row in state.occupations), basis.nao
        )
        for spin, count in enumerate(self.layout.occupied):
            expected = np.zeros(basis.nao)
            expected[:count] = 1
            if not np.array_equal(expected, state.occupations[spin]):
                raise ValueError(
                    "UKS Hessian requires canonical integer 1/0 occupations"
                )
        self.dimension = self.layout.dimension
        identity = canonical_hash(
            {
                "schema": "generativeqc.spin-stationary-cpks/v1",
                "state": state.identity.to_payload(),
                "backend": backend.identity,
                "grid": kernel.identity,
                "occupied": self.layout.occupied,
            }
        )
        reference = SimpleNamespace(
            geometry_hash=_geometry_hash(basis),
            basis_hash=_basis_hash(basis),
            functional_identity=kernel.functional_identity,
            grid_identity=kernel.grid_identity,
        )
        self.problem = SimpleNamespace(
            reference=reference,
            dimension=self.dimension,
            identity=identity,
            compatibility_identity=identity,
            operator_identity=identity,
            layout=self.layout,
            method="uks-cpks",
        )
        self.statistics = {
            "actions": 0,
            "transpose_actions": 0,
            "peak_workspace_bytes": 0,
        }

    @property
    def identity(self) -> str:
        """Identity of the coupled response actually used by the solver."""
        return self.problem.operator_identity

    def induced_fock(
        self, density: np.ndarray, *, transpose: bool = False
    ) -> np.ndarray:
        """Apply both-spin J/SR-K/LR-K and XC response, with no spin averaging."""
        del transpose
        density = np.asarray(density)
        if (
            density.shape != self.state.density.shape
            or np.iscomplexobj(density)
            or not np.isfinite(density).all()
        ):
            raise ValueError("UKS response requires finite [2,AO,AO] densities")
        if not np.allclose(density, density.swapaxes(-1, -2), atol=2e-10, rtol=2e-12):
            raise ValueError("UKS density response must be symmetric")
        interaction, _ = self.backend.coulomb_exchange(density)
        return interaction + self.xc_kernel.apply(density)

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """Apply the coupled stationary orbital Hessian in the spin OV metric."""
        self.validate_current()
        blocks = self.layout.split(vector)
        densities = []
        for coefficients, count, block in zip(
            self.state.coefficients, self.layout.occupied, blocks, strict=True
        ):
            left = coefficients[:, count:] @ block.T @ coefficients[:, :count].T
            densities.append(left + left.T)
        induced = self.induced_fock(np.asarray(densities))
        outputs = []
        for spin, (count, block) in enumerate(
            zip(self.layout.occupied, blocks, strict=True)
        ):
            coefficients, energies = (
                self.state.coefficients[spin],
                self.state.orbital_energies[spin],
            )
            response = coefficients.T @ induced[spin] @ coefficients
            outputs.append(
                (energies[count:][None] - energies[:count, None]) * block
                + response[count:, :count].T
            )
        self.statistics["actions"] += 1
        return self.layout.pack(outputs)

    def apply_transpose(self, vector: np.ndarray) -> np.ndarray:
        """The physical fixed-occupation UKS orbital Hessian is self-adjoint."""
        self.statistics["transpose_actions"] += 1
        return self.apply(vector)

    def solve_nuclear(
        self, frozen: np.ndarray, overlap: np.ndarray, options: GMRESOptions
    ) -> StationaryNuclearResponse:
        """Solve once with metric density, then reconstruct both Pulay densities."""
        coefficients, energies = self.state.coefficients, self.state.orbital_energies
        overlap_mo = np.asarray([frame.T @ overlap @ frame for frame in coefficients])
        frozen_mo = np.asarray(
            [
                frame.T @ matrix @ frame
                for frame, matrix in zip(coefficients, frozen, strict=True)
            ]
        )
        metric = []
        for spin, count in enumerate(self.layout.occupied):
            left = (
                -0.5
                * coefficients[spin]
                @ overlap_mo[spin, :, :count]
                @ coefficients[spin, :, :count].T
            )
            metric.append(left + left.T)
        induced_metric = self.induced_fock(np.asarray(metric))
        rhs = []
        for spin, count in enumerate(self.layout.occupied):
            induced = coefficients[spin].T @ induced_metric[spin] @ coefficients[spin]
            rhs.append(
                -(
                    frozen_mo[spin, count:, :count]
                    + induced[count:, :count]
                    - 0.5
                    * overlap_mo[spin, count:, :count]
                    * (energies[spin, count:, None] + energies[spin, None, :count])
                ).T
            )
        rhs = self.layout.pack(rhs)
        result = solve(self, rhs, options=options)
        result.require_converged()
        rotations = self.layout.split(result.solution)
        orbital_motion = np.zeros_like(coefficients)
        density_motion = np.zeros_like(self.state.density)
        mo_motion = []
        for spin, count in enumerate(self.layout.occupied):
            moving = -0.5 * overlap_mo[spin, :, :count].copy()
            moving[count:] += rotations[spin].T
            mo_motion.append(moving)
            orbital_motion[spin, :, :count] = coefficients[spin] @ moving
            left = orbital_motion[spin, :, :count] @ coefficients[spin, :, :count].T
            density_motion[spin] = left + left.T
        induced = self.induced_fock(density_motion)
        energy_motion = np.zeros_like(self.state.weighted_density)
        occupied_energy = np.zeros_like(coefficients)
        for spin, count in enumerate(self.layout.occupied):
            frame, occupied = coefficients[spin], coefficients[spin, :, :count]
            occupied_energies = energies[spin, :count]
            hs = (
                frozen_mo[spin, :, :count]
                - overlap_mo[spin, :, :count] * occupied_energies
            )
            hs += frame.T @ induced[spin] @ occupied
            first_energy = hs[:count] + mo_motion[spin][:count] * (
                occupied_energies[:, None] - occupied_energies[None]
            )
            occupied_energy[spin, :count, :count] = first_energy
            left = (orbital_motion[spin, :, :count] * occupied_energies) @ occupied.T
            energy_motion[spin] = left + left.T + occupied @ first_energy @ occupied.T
        return StationaryNuclearResponse(
            immutable(rhs),
            immutable(orbital_motion),
            immutable(occupied_energy),
            immutable(density_motion),
            immutable(energy_motion),
            result,
        )
