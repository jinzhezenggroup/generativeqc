"""Qualify installed spherical B3LYP endpoints against retained independent data.

This opt-in CPU experiment is not a CI microbenchmark. ``center`` independently
reconverges PySCF on the declared quadrature; ``endpoint`` runs only the installed
production package. The latter calls the public full Hessian and two public HVPs,
without filling columns from symmetry, translations, or previous native runs.
Run it under an explicit finite outer timeout and retain incomplete checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import threading
import time
import typing
import weakref
from pathlib import Path

import numpy as np

ATOMS = (
    ("Ni", (0.01, -0.02, 0.03)),
    ("H", (2.9, 0.1, 0.03)),
    ("H", (-3.05, 0.05, -0.02)),
    ("H", (0.03, 2.97, 0.07)),
    ("H", (-0.04, -3.1, -0.03)),
)
SOURCES = (
    "one_electron",
    "coulomb",
    "xc_ao",
    "xc_grid",
    "xc_weight",
    "exact_exchange",
    "overlap_pulay",
    "nuclear",
)
REFERENCE_GATE = 1e-5
INVARIANCE_GATE = 2e-7


def digest(path: Path) -> str:
    """Bind actual input bytes, including independently produced references."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Retain readable receipts without rounding numerical acceptance evidence."""
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def reference_center(args: argparse.Namespace, basis_data: dict) -> None:
    """Solve the independent center without any native density or orbitals."""
    import pyscf
    from pyscf import dft, gto
    from pyscf.dft import gen_grid
    from pyscf.grad import rks

    if pyscf.__version__ != "2.14.0":
        raise ValueError("independent reference requires PySCF 2.14.0")
    basis = {}
    for number, element in basis_data["elements"].items():
        basis[gto.mole._symbol(int(number))] = [
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
    grid = np.load(args.references / "reference-grid.npz")
    atomic = {}
    for index, (symbol, position) in enumerate(ATOMS):
        selected = grid["owners"] == index
        if symbol not in atomic:
            atomic[symbol] = (
                grid["points"][selected] - position,
                grid["atomic_weights"][selected],
            )
    molecule = gto.M(atom=ATOMS, basis=basis, unit="Bohr", charge=-2, verbose=0)
    solver = dft.RKS(molecule)
    solver.xc = "0.08*LDA_X+0.72*GGA_X_B88+0.19*LDA_C_VWN_RPA+0.81*GGA_C_LYP+0.2*HF"
    solver.direct_scf = False
    solver.init_guess = "minao"
    solver.diis_space = 12
    solver.conv_tol, solver.conv_tol_grad, solver.max_cycle = 1e-12, 1e-10, 240
    solver.grids.gen_atomic_grids = lambda *unused, **kwargs: atomic
    solver.grids.radii_adjust = None
    solver.grids.becke_scheme = gen_grid.original_becke
    quadrature = list(rks.grids_response_cc(solver.grids))
    solver.grids.coords = np.concatenate([item[0] for item in quadrature])
    solver.grids.weights = np.concatenate([item[1] for item in quadrature])
    solver.kernel()
    if not solver.converged:
        raise RuntimeError("independent center did not converge at frozen controls")
    residual = float(np.linalg.norm(solver.get_grad(solver.mo_coeff, solver.mo_occ)))
    if residual > 2e-9:
        raise ValueError(f"independent center stationarity gate: {residual}")
    np.testing.assert_allclose(solver.grids.coords, grid["points"], atol=2e-12, rtol=0)
    np.testing.assert_allclose(
        solver.grids.weights, grid["weights"], atol=2e-10, rtol=2e-12
    )
    np.savez_compressed(
        args.references / "reference-center.npz",
        density=solver.make_rdm1(),
        overlap=molecule.intor("int1e_ovlp"),
        energy=solver.e_tot,
        orbital_residual=residual,
        pyscf_version=pyscf.__version__,
    )
    print("independent center", solver.e_tot, "orbital residual", residual, flush=True)


class EndpointMemory:
    """Sample the endpoint's entire process tree, including finite JIT workers.

    Summed RSS is a conservative simultaneous bound, not unique physical pages.
    Independent oracle SCFs run in a separate completed command, not this tree.
    """

    def __init__(self) -> None:
        import psutil

        self.process = psutil.Process()
        self.psutil = psutil
        self.peak = 0
        self.finished = threading.Event()

    def sample(self) -> None:
        while not self.finished.is_set():
            total = 0
            for process in (self.process, *self.process.children(recursive=True)):
                try:
                    total += process.memory_info().rss
                except self.psutil.Error:
                    continue
            self.peak = max(self.peak, total)
            self.finished.wait(0.05)

    def run(
        self, operation: typing.Callable[[], typing.Any]
    ) -> tuple[typing.Any, dict[str, float | int]]:
        """Measure the complete public call, never only its response kernels."""
        worker = threading.Thread(target=self.sample, daemon=True)
        worker.start()
        started = time.monotonic()
        try:
            result = operation()
        finally:
            self.finished.set()
            worker.join()
        return result, {
            "wall_seconds": time.monotonic() - started,
            "peak_process_tree_rss_bytes": self.peak,
            "memory_sampling_seconds": 0.05,
        }


def endpoint(args: argparse.Namespace, manifest: dict, basis_file: Path) -> None:
    """Check the real public assembler and fresh public actions without shortcuts."""
    import generativeqc
    import generativeqc.composite_rks_hessian as composite
    import generativeqc_compiler
    from generativeqc import Calculator, GridSpec, KsOptions, _native, import_bse
    from generativeqc.response_solver import GMRESOptions

    locations = [
        Path(module.__file__).resolve()
        for module in (generativeqc, generativeqc_compiler)
    ]
    locations.append(Path(_native.load_library()._name).resolve())
    if not all(
        path.is_relative_to(args.installed_prefix.resolve()) for path in locations
    ):
        raise ValueError(f"installed-package boundary failed: {locations}")
    raw = np.load(args.references / "reference-matrix.npz")
    actions = np.load(args.references / "reference-actions.npz")
    groups = np.load(args.references / "reference-components.npz")
    center = np.load(args.references / "reference-center.npz")
    grid = np.load(args.references / "reference-grid.npz")
    if not raw["complete"].item() or float(raw["raw_matrix_gate"]) != REFERENCE_GATE:
        raise ValueError("reference matrix must be complete at the frozen 1e-5 gate")
    basis = import_bse(
        basis_file,
        source="MolSSI Basis Set Exchange",
        source_version=manifest["source_commit"],
        license="BSD-3-Clause",
        representation="spherical",
    )
    calculator = Calculator(
        method="b3lyp-rks",
        device="cpu",
        basis=basis,
        basis_representation="spherical",
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=16, angular_polar=6, angular_azimuth=12)
        ),
    )
    options = {
        "charge": -2,
        "cache": args.cache,
        "integral_budget_bytes": 64 << 20,
        "grid_budget_bytes": 512 << 20,
        "solver_options": GMRESOptions(atol=1e-11, rtol=1e-10),
    }
    report = {
        "schema": "generativeqc.installed-global-hybrid-qualification/v1",
        "complete": False,
        "locations": list(map(str, locations)),
        "basis_sha256": digest(basis_file),
        "reference_sha256": {
            path.name: digest(path) for path in args.references.glob("*.npz")
        },
        "oracle_gate": REFERENCE_GATE,
        "invariance_gate": INVARIANCE_GATE,
        "units": "Hartree/Bohr^2",
        "ordering": "atom-major xyz",
        "columns": [],
    }
    write_json(args.output / "receipt.json", report)
    owners = []
    original_factory = composite.CompositeRKSResponse.from_native.__func__

    def checked_factory(
        cls: typing.Any, *values: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        operator = original_factory(cls, *values, **kwargs)
        try:
            np.testing.assert_allclose(
                operator.state.density[0], center["density"], atol=2e-9, rtol=0
            )
            np.testing.assert_allclose(
                operator.state.overlap, center["overlap"], atol=2e-12, rtol=0
            )
            np.testing.assert_allclose(
                operator.state.grid.points, grid["points"], atol=2e-12, rtol=0
            )
            np.testing.assert_allclose(
                operator.state.grid.weights, grid["weights"], atol=2e-10, rtol=2e-12
            )
            if abs(operator.state._source.energy() - float(center["energy"])) > 2e-10:
                raise ValueError("matched-state energy gate failed")
            owners.append(weakref.ref(operator))
        except BaseException:
            operator.close()
            raise
        return operator

    composite.CompositeRKSResponse.from_native = classmethod(checked_factory)
    original_hvp = composite.composite_rks_hvp
    columns = []
    recording = True

    def checked_hvp(
        operator: typing.Any, direction: np.ndarray, **kwargs: typing.Any
    ) -> typing.Any:
        if recording:
            print("starting actual public assembler column", len(columns), flush=True)
        result = original_hvp(operator, direction, **kwargs)
        diagnostic = result.diagnostics
        if (
            tuple(result.components) != SOURCES
            or diagnostic["nuclear_response_solves"] != 1
            or diagnostic["displaced_scf_evaluations"] != 0
            or diagnostic["response_residual_norm"] > 1e-9
        ):
            raise ValueError("incomplete source inventory or failed response gate")
        np.testing.assert_allclose(
            sum(result.components.values()), result.value, atol=2e-14, rtol=0
        )
        errors = np.max(
            abs(raw["matrices"] @ direction.ravel() - result.value.ravel()), axis=1
        )
        errors /= np.linalg.norm(direction)
        if max(errors) > REFERENCE_GATE:
            raise ValueError(f"independent action gate failed: {errors}")
        if recording:
            column = len(columns)
            np.testing.assert_array_equal(direction.ravel(), np.eye(15)[column])
            columns.append(result.value.ravel())
            report["columns"].append(
                {
                    "column": column,
                    "errors": errors.tolist(),
                    "diagnostics": dict(diagnostic),
                }
            )
            np.savez_compressed(
                args.output / "partial-columns.npz",
                columns=np.asarray(columns),
                complete=False,
            )
            write_json(args.output / "receipt.json", report)
            print(
                "accepted actual public assembler column",
                column,
                errors.tolist(),
                flush=True,
            )
        return result

    composite.composite_rks_hvp = checked_hvp
    try:
        matrix, report["matrix_endpoint"] = EndpointMemory().run(
            lambda: calculator.hessian(ATOMS, **options)
        )
        recording = False
        if not matrix.diagnostics["public_calculator_endpoint"] or len(columns) != 15:
            raise ValueError("full installed public Hessian did not complete")
        errors = np.max(abs(raw["matrices"] - matrix.matrix), axis=(1, 2))
        symmetry = float(np.max(abs(matrix.matrix - matrix.matrix.T)))
        translation = float(np.max(abs(matrix.matrix @ np.tile(np.eye(3), (5, 1)))))
        if max(errors) > REFERENCE_GATE or max(symmetry, translation) > INVARIANCE_GATE:
            raise ValueError(
                f"raw matrix gate: {errors}, symmetry {symmetry}, translation {translation}"
            )
        np.savez_compressed(
            args.output / "raw-matrix.npz", matrix=matrix.matrix, complete=True
        )
        report.update(
            matrix_complete=True,
            matrix_errors=errors.tolist(),
            raw_symmetry=symmetry,
            translation=translation,
            matrix_diagnostics=dict(matrix.diagnostics),
        )
        write_json(args.output / "receipt.json", report)
        print(
            "accepted complete public raw Hessian",
            errors.tolist(),
            symmetry,
            translation,
            flush=True,
        )
        first, report["first_hvp_endpoint"] = EndpointMemory().run(
            lambda: calculator.hessian_vector_product(
                ATOMS, actions["direction"], **options
            )
        )
        report["first_hvp_diagnostics"] = dict(first.diagnostics)
        np.savez_compressed(args.output / "first-action.npz", value=first.value)
        write_json(args.output / "receipt.json", report)
        second_direction = np.random.default_rng(2849).normal(size=(5, 3)) * 0.03
        second_direction -= second_direction.mean(axis=0)
        second, report["second_hvp_endpoint"] = EndpointMemory().run(
            lambda: calculator.hessian_vector_product(
                ATOMS, second_direction, **options
            )
        )
        norm = np.linalg.norm(actions["direction"])
        gradient_errors = (
            np.max(abs(actions["reference_hvp"] - first.value), axis=(1, 2)) / norm
        )
        components = dict(first.components)
        components["xc_total"] = sum(
            components[name] for name in ("xc_ao", "xc_grid", "xc_weight")
        )
        grouped = np.asarray([components[name] for name in groups["names"]])
        group_errors = np.max(abs(groups["components"] - grouped), axis=(2, 3)) / norm
        parity = [
            float(np.max(abs(matrix.matrix @ direction.ravel() - result.value.ravel())))
            / np.linalg.norm(direction)
            for direction, result in (
                (actions["direction"], first),
                (second_direction, second),
            )
        ]
        bilinear = float(
            abs(
                np.vdot(second_direction, first.value)
                - np.vdot(actions["direction"], second.value)
            )
        ) / (norm * np.linalg.norm(second_direction))
        if max(np.max(gradient_errors), np.max(group_errors)) > REFERENCE_GATE:
            raise ValueError("independent multistep gradient/component gate failed")
        if max(*parity, bilinear) > INVARIANCE_GATE:
            raise ValueError(f"native matrix/HVP/bilinear gate: {parity}, {bilinear}")
        for owner in owners:
            operator = owner()
            if operator is not None:
                try:
                    operator.validate_current()
                except (ValueError, RuntimeError):
                    continue
                raise ValueError("public endpoint retained a live native owner")
        report.update(
            complete=True,
            gradient_errors=gradient_errors.tolist(),
            component_errors=group_errors.tolist(),
            matrix_hvp_errors=parity,
            bilinear_error=bilinear,
            first_hvp_diagnostics=dict(first.diagnostics),
            second_hvp_diagnostics=dict(second.diagnostics),
            parent_process_peak_rss_kib=resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        )
        np.savez_compressed(
            args.output / "actions.npz",
            first=first.value,
            second=second.value,
            first_direction=actions["direction"],
            second_direction=second_direction,
        )
        write_json(args.output / "receipt.json", report)
        print(
            "complete installed public qualification",
            json.dumps(report, default=str),
            flush=True,
        )
    finally:
        composite.composite_rks_hvp = original_hvp
        composite.CompositeRKSResponse.from_native = classmethod(original_factory)
        calculator.clear_cache()


def main() -> None:
    """Select an explicit independent-reference or installed-production run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("center", "endpoint"))
    parser.add_argument("--basis-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--installed-prefix", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.basis_dir / "manifest.json").read_text())
    basis_file = args.basis_dir / manifest["file"]
    if digest(basis_file) != manifest["sha256"]:
        raise ValueError("pinned independent basis identity failed")
    if args.mode == "center":
        reference_center(args, json.loads(basis_file.read_text()))
    else:
        endpoint(args, manifest, basis_file)


if __name__ == "__main__":
    main()
