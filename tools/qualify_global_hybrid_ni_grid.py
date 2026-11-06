"""Opt-in installed Ni default-grid and finite-grid rotation qualification.

Run ``reference`` before ``endpoint``. Reference SCFs use only independent
atomic/independent-center guesses. Native grid export specifies the quadrature,
not the electronic state of that oracle. Neither mode changes a production
grid, coefficient, resource gate, compiler key, or response tolerance.
"""

from __future__ import annotations

import argparse
import json
import typing
from pathlib import Path

import numpy as np
from qualify_global_hybrid_ni import ATOMS, EndpointMemory, digest, write_json


def calculator(args: argparse.Namespace, *, default: bool) -> typing.Any:
    """Use the installed physical B3LYP graph and pinned spherical basis."""
    import generativeqc
    import generativeqc_compiler
    from generativeqc import Calculator, GridSpec, KsOptions, _native, import_bse

    locations = [
        Path(module.__file__).resolve()
        for module in (generativeqc, generativeqc_compiler)
    ]
    locations.append(Path(_native.load_library()._name).resolve())
    if not all(path.is_relative_to(args.installed_prefix) for path in locations):
        raise ValueError(f"not an installed-package run: {locations}")
    manifest = json.loads((args.basis_dir / "manifest.json").read_text())
    payload = args.basis_dir / manifest["file"]
    if digest(payload) != manifest["sha256"]:
        raise ValueError("pinned independent basis changed")
    basis = import_bse(
        payload,
        source=manifest["source_repository"],
        source_version=manifest["source_commit"],
        license=manifest["license"],
        representation="spherical",
    )
    return Calculator(
        method="b3lyp-rks",
        device="cpu",
        basis=basis,
        basis_representation="spherical",
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
        ks_options=(
            None
            if default
            else KsOptions(
                grid=GridSpec(radial_points=16, angular_polar=6, angular_azimuth=12)
            )
        ),
    )


def reference(args: argparse.Namespace) -> None:
    """Freeze an independent moving-grid gradient oracle at three steps."""
    import pyscf
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from pyscf import dft, gto
    from pyscf.dft import gen_grid, radi
    from pyscf.grad import rks

    if pyscf.__version__ != "2.14.0":
        raise ValueError("reference requires PySCF 2.14.0")
    public = calculator(args, default=True)
    with (
        public.prepare_batch([ATOMS], charges=[-2], warm_start=False) as batch,
        NativeAO(
            ATOMS, basis=public._basis, representation="spherical", charge=-2
        ) as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
        try:
            np.savez_compressed(
                args.output / "default-grid.npz",
                points=state.grid.points,
                weights=state.grid.weights,
                owners=state.grid.owners,
                atomic_weights=state._source.atomic_weights,
                native_density=state.density[0],
                native_overlap=state.overlap,
                native_energy=state._source.energy(),
            )
        finally:
            state._source.close()
    grid = np.load(args.output / "default-grid.npz")
    if len(grid["points"]) != 59872:
        raise ValueError("native default quadrature changed; requalify explicitly")
    raw = json.loads((args.basis_dir / "def2-svp-ni-h.json").read_text())
    reference_basis = {}
    for number, element in raw["elements"].items():
        reference_basis[gto.mole._symbol(int(number))] = [
            [
                shell["angular_momentum"][0],
                *[
                    [
                        float(exponent),
                        *[float(row[index]) for row in shell["coefficients"]],
                    ]
                    for index, exponent in enumerate(shell["exponents"])
                ],
            ]
            for shell in element["electron_shells"]
        ]
    atomic = {}
    for index, (symbol, position) in enumerate(ATOMS):
        selected = grid["owners"] == index
        atomic.setdefault(
            symbol,
            (grid["points"][selected] - position, grid["atomic_weights"][selected]),
        )
    symbols = [symbol for symbol, _ in ATOMS]
    positions = np.asarray([position for _, position in ATOMS])

    def solve(offset: np.ndarray, density: np.ndarray | None = None) -> typing.Any:
        """Reconverge each displaced state with no native electronic-state seed."""
        molecule = gto.M(
            atom=list(zip(symbols, positions + offset, strict=True)),
            basis=reference_basis,
            unit="Bohr",
            charge=-2,
            verbose=0,
        )
        solver = dft.RKS(molecule)
        solver.xc = "0.08*LDA_X+0.72*GGA_X_B88+0.19*LDA_C_VWN_RPA+0.81*GGA_C_LYP+0.2*HF"
        solver.direct_scf = False
        solver.init_guess, solver.diis_space = "minao", 12
        solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-12, 1e-10, 240
        solver.grids.gen_atomic_grids = lambda *unused, **kwargs: atomic
        solver.grids.radii_adjust = radi.treutler_atomic_radii_adjust
        solver.grids.becke_scheme = gen_grid.original_becke
        quadrature = list(rks.grids_response_cc(solver.grids))
        solver.grids.coords = np.concatenate([item[0] for item in quadrature])
        solver.grids.weights = np.concatenate([item[1] for item in quadrature])
        solver.kernel(dm0=density)
        residual = float(
            np.linalg.norm(solver.get_grad(solver.mo_coeff, solver.mo_occ))
        )
        if not solver.converged or residual > 2e-9:
            raise ValueError(f"independent stationarity gate failed: {residual}")
        print("independent default-grid SCF", solver.e_tot, residual, flush=True)
        return solver

    center = solve(np.zeros_like(positions))
    np.testing.assert_allclose(center.grids.coords, grid["points"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(
        center.grids.weights, grid["weights"], atol=2e-10, rtol=2e-12
    )
    np.testing.assert_allclose(
        center.make_rdm1(), grid["native_density"], atol=2e-9, rtol=0
    )
    if abs(center.e_tot - float(grid["native_energy"])) > 2e-10:
        raise ValueError("independent/default native energy mismatch")
    direction = np.load(args.references / "reference-actions.npz")["direction"]
    steps = np.asarray([3e-3, 1e-3, 3e-4])
    differences = []
    for step in steps:
        gradients = []
        for sign in (1, -1):
            displaced = solve(sign * step * direction, center.make_rdm1())
            gradient = displaced.nuc_grad_method()
            gradient.grid_response = True
            gradients.append(gradient.kernel())
        differences.append((gradients[0] - gradients[1]) / (2 * step))
    np.savez_compressed(
        args.output / "default-reference.npz",
        density=center.make_rdm1(),
        energy=center.e_tot,
        direction=direction,
        steps=steps,
        hvp=np.asarray(differences),
        gate=1e-5,
        pyscf_version=pyscf.__version__,
    )


def endpoint(args: argparse.Namespace) -> None:
    """Check refusal, explicit resource admission and an actual rotated public HVP."""
    import generativeqc.composite_rks_hessian as composite
    from generativeqc.response_solver import GMRESOptions
    from scipy.spatial import cKDTree

    oracle = np.load(args.output / "default-reference.npz")
    grid = np.load(args.output / "default-grid.npz")
    report = {
        "complete": False,
        "oracle_gate": 1e-5,
        "grid_points": len(grid["points"]),
    }
    options = {
        "charge": -2,
        "cache": args.cache,
        "integral_budget_bytes": 64 << 20,
        "solver_options": GMRESOptions(atol=1e-11, rtol=1e-10),
    }
    public = calculator(args, default=True)
    try:
        public.hessian_vector_product(ATOMS, oracle["direction"], **options)
    except MemoryError as error:
        if "grid_budget_bytes" not in str(error):
            raise
        report["default_128mib_refusal"] = str(error)
    else:
        raise ValueError("expected default-grid 128-MiB preflight rejection")
    action, report["default_endpoint"] = EndpointMemory().run(
        lambda: public.hessian_vector_product(
            ATOMS, oracle["direction"], grid_budget_bytes=4 << 30, **options
        )
    )
    errors = np.max(abs(oracle["hvp"] - action.value), axis=(1, 2)) / np.linalg.norm(
        oracle["direction"]
    )
    if max(errors) > 1e-5 or action.diagnostics["response_residual_norm"] > 1e-9:
        raise ValueError(f"default-grid independent HVP gate: {errors}")
    report.update(
        default_errors=errors.tolist(), default_diagnostics=dict(action.diagnostics)
    )
    np.savez_compressed(args.output / "default-action.npz", hvp=action.value)
    write_json(args.output / "receipt.json", report)
    angle = np.pi / 6
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    atoms = [(symbol, np.asarray(position) @ rotation.T) for symbol, position in ATOMS]
    explicit_grid = np.load(args.references / "reference-grid.npz")
    factory = composite.CompositeRKSResponse.from_native.__func__

    def checked_factory(
        cls: typing.Any, *values: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        """Require rotation of the exact discrete grid, not an isotropic-grid assumption."""
        owner = factory(cls, *values, **kwargs)
        try:
            distances, indices = cKDTree(owner.state.grid.points).query(
                explicit_grid["points"] @ rotation.T
            )
            if max(distances) > 2e-12 or len(np.unique(indices)) != len(indices):
                raise ValueError("rotation did not preserve the quadrature set")
            np.testing.assert_allclose(
                owner.state.grid.weights[indices],
                explicit_grid["weights"],
                atol=2e-9,
                rtol=2e-12,
            )
        except BaseException:
            owner.close()
            raise
        return owner

    composite.CompositeRKSResponse.from_native = classmethod(checked_factory)
    try:
        rotated, report["rotation_endpoint"] = EndpointMemory().run(
            lambda: calculator(args, default=False).hessian_vector_product(
                atoms,
                oracle["direction"] @ rotation.T,
                grid_budget_bytes=512 << 20,
                **options,
            )
        )
    finally:
        composite.CompositeRKSResponse.from_native = classmethod(factory)
    np.savez_compressed(
        args.output / "rotation-action.npz", hvp=rotated.value, rotation=rotation
    )
    report.update(complete=True, rotation_diagnostics=dict(rotated.diagnostics))
    write_json(args.output / "receipt.json", report)
    print(
        "default-grid HVP accepted; retained rotated action for native covariance",
        errors,
        flush=True,
    )


def main() -> None:
    """Keep independent and production processes separate and retain exact inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "endpoint"))
    for name in ("basis-dir", "references", "output", "cache", "installed-prefix"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.installed_prefix = args.installed_prefix.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (reference if args.mode == "reference" else endpoint)(args)


if __name__ == "__main__":
    main()
