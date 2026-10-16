"""Resident coupled-spin CPKS and D/W reconstruction for compiler-AD Hessians."""

from __future__ import annotations

import typing
from types import SimpleNamespace

from .response_problem import ResponseSolveError


def _norm(cp: typing.Any, value: typing.Any) -> typing.Any:
    """Scale on the device before squaring, including subnormal RHS values."""
    if not value.size:
        return cp.asarray(0.0)
    scale = cp.maximum(cp.max(abs(value)), cp.finfo(cp.float64).tiny)
    result = scale * cp.linalg.norm(value / scale)
    if not bool(cp.isfinite(result)):
        raise ValueError("resident response vector norm is nonfinite or overflows FP64")
    return result


def resident_gmres(
    cp: typing.Any,
    apply: typing.Callable,
    rhs: typing.Any,
    diagonal: typing.Any,
    options: typing.Any,
) -> SimpleNamespace:
    """Restarted, reorthogonalized FP64 GMRES with true-residual acceptance.

    Arnoldi vectors, the small Hessenberg least-squares problem and all updates
    stay on the device. Only norms and stopping decisions reach the host. A
    preconditioned estimate is never accepted instead of the physical residual.
    """
    dimension = rhs.size
    restart = min(options.restart, max(1, dimension))
    required = 8 * ((restart + 8) * dimension + 6 * (restart + 1) ** 2)
    if required > options.max_workspace_bytes:
        raise ValueError("resident CPKS GMRES exceeds max_workspace_bytes")
    solution = cp.zeros_like(rhs)
    rhs_norm = float(_norm(cp, rhs))
    tolerance = max(options.atol, options.rtol * rhs_norm)
    history = [rhs_norm]
    iterations, actions = 0, 0
    residual = rhs.copy()
    while history[-1] > tolerance and iterations < options.max_iterations:
        basis = cp.zeros((dimension, restart + 1))
        hessenberg = cp.zeros((restart + 1, restart))
        preconditioned = residual / diagonal
        beta = _norm(cp, preconditioned)
        if float(beta) == 0:
            break
        basis[:, 0] = preconditioned / beta
        target = cp.zeros(restart + 1)
        target[0] = beta
        candidate = solution
        for column in range(min(restart, options.max_iterations - iterations)):
            work = apply(basis[:, column]) / diagonal
            actions += 1
            for _ in range(options.reorthogonalize):
                projection = basis[:, : column + 1].T @ work
                hessenberg[: column + 1, column] += projection
                work -= basis[:, : column + 1] @ projection
            hessenberg[column + 1, column] = _norm(cp, work)
            breakdown = (
                float(hessenberg[column + 1, column]) <= options.breakdown_tolerance
            )
            if not breakdown:
                basis[:, column + 1] = work / hessenberg[column + 1, column]
            coefficients = cp.linalg.lstsq(
                hessenberg[: column + 2, : column + 1], target[: column + 2], rcond=None
            )[0]
            candidate = solution + basis[:, : column + 1] @ coefficients
            residual = rhs - apply(candidate)
            actions += 1
            norm = float(_norm(cp, residual))
            history.append(norm)
            iterations += 1
            if not bool(cp.isfinite(residual).all()):
                raise ResponseSolveError(
                    SimpleNamespace(
                        iterations=iterations,
                        residual_norm=norm,
                        reason="nonfinite resident CPKS residual",
                    )
                )
            if norm <= tolerance or breakdown:
                break
            if len(history) > options.stagnation_window:
                previous = history[-1 - options.stagnation_window]
                if previous and norm >= previous * (1 - options.stagnation_tolerance):
                    raise ResponseSolveError(
                        SimpleNamespace(
                            iterations=iterations,
                            residual_norm=norm,
                            reason="resident CPKS GMRES stagnated",
                        )
                    )
        solution = candidate
    if history[-1] > tolerance:
        raise ResponseSolveError(
            SimpleNamespace(
                iterations=iterations,
                residual_norm=history[-1],
                reason="resident CPKS iteration limit or breakdown",
            )
        )
    return SimpleNamespace(
        solution=solution,
        residual_norm=history[-1],
        iterations=iterations,
        operator_actions=actions,
        history=tuple(history),
        workspace_bytes=required,
    )


class CudaSpinResponse:
    """Use independent physical spin spaces, including empty occupied/virtual blocks."""

    def __init__(self, owner: typing.Any) -> None:
        self.owner, self.cp, self.state = owner, owner.execution.cp, owner.device_state
        self.counts = tuple(int((row > 0).sum()) for row in owner.state.occupations)
        self.occupation = 1 if len(self.counts) == 2 else 2
        self.sizes = tuple(count * (owner.basis.nao - count) for count in self.counts)
        self.dimension = sum(self.sizes)
        for row, count in zip(owner.state.occupations, self.counts, strict=True):
            if any(row[:count] != self.occupation) or any(row[count:] != 0):
                raise ValueError("resident CPKS requires canonical integer occupations")
        self.diagonal = self.pack(
            [
                energies[count:][None] - energies[:count, None]
                for energies, count in zip(
                    self.state.orbital_energies, self.counts, strict=True
                )
            ]
        )
        if self.dimension and bool(self.cp.any(abs(self.diagonal) <= 1e-12)):
            raise ValueError(
                "resident CPKS requires separated occupied/virtual energies"
            )

    def pack(self, blocks: list) -> typing.Any:
        return self.cp.concatenate([block.reshape(-1) for block in blocks])

    def split(self, vector: typing.Any) -> list:
        cursor, blocks = 0, []
        for count, size in zip(self.counts, self.sizes, strict=True):
            blocks.append(
                vector[cursor : cursor + size].reshape(
                    count, self.owner.basis.nao - count
                )
            )
            cursor += size
        return blocks

    def apply(self, vector: typing.Any) -> typing.Any:
        densities = []
        for frame, count, block in zip(
            self.state.coefficients, self.counts, self.split(vector), strict=True
        ):
            left = frame[:, count:] @ block.T @ frame[:, :count].T
            densities.append(self.occupation * (left + left.T))
        induced = self.owner.induced_fock(self.cp.stack(densities))
        return self.diagonal * vector + self.pack(
            [
                (frame[:, count:].T @ matrix @ frame[:, :count]).T
                for frame, count, matrix in zip(
                    self.state.coefficients, self.counts, induced, strict=True
                )
            ]
        )

    def solve(
        self, frozen: typing.Any, overlap: typing.Any, options: typing.Any
    ) -> SimpleNamespace:
        coefficients, energies = self.state.coefficients, self.state.orbital_energies
        overlap_mo = self.cp.stack(
            [frame.T @ overlap @ frame for frame in coefficients]
        )
        frozen_mo = self.cp.stack(
            [
                frame.T @ matrix @ frame
                for frame, matrix in zip(coefficients, frozen, strict=True)
            ]
        )
        metric = []
        for frame, count, moving in zip(
            coefficients, self.counts, overlap_mo, strict=True
        ):
            left = (
                -0.5 * self.occupation * frame @ moving[:, :count] @ frame[:, :count].T
            )
            metric.append(left + left.T)
        induced_metric = self.owner.induced_fock(self.cp.stack(metric))
        rhs = []
        for spin, count in enumerate(self.counts):
            frame = coefficients[spin]
            induced = frame.T @ induced_metric[spin] @ frame
            rhs.append(
                -(
                    frozen_mo[spin, count:, :count]
                    + induced[count:, :count]
                    - 0.5
                    * overlap_mo[spin, count:, :count]
                    * (energies[spin, count:, None] + energies[spin, None, :count])
                ).T
            )
        rhs = self.pack(rhs)
        result = resident_gmres(self.cp, self.apply, rhs, self.diagonal, options)
        orbital_motion = self.cp.zeros_like(coefficients)
        density_motion = self.cp.zeros_like(self.state.density)
        mo_motion = []
        for spin, (count, rotation) in enumerate(
            zip(self.counts, self.split(result.solution), strict=True)
        ):
            moving = -0.5 * overlap_mo[spin, :, :count].copy()
            moving[count:] += rotation.T
            mo_motion.append(moving)
            orbital_motion[spin, :, :count] = coefficients[spin] @ moving
            left = orbital_motion[spin, :, :count] @ coefficients[spin, :, :count].T
            density_motion[spin] = self.occupation * (left + left.T)
        induced = self.owner.induced_fock(density_motion)
        energy_motion = self.cp.zeros_like(self.state.weighted_density)
        for spin, count in enumerate(self.counts):
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
            left = (orbital_motion[spin, :, :count] * occupied_energies) @ occupied.T
            energy_motion[spin] = self.occupation * (
                left + left.T + occupied @ first_energy @ occupied.T
            )
        return SimpleNamespace(
            density_derivative=density_motion,
            energy_weighted_density_derivative=energy_motion,
            solve_result=result,
        )
