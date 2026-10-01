"""Fresh-object exact RHF benchmark; same shell decimals/core guess/Delta-E/Delta-D gates."""

import hashlib
import json
import os
import pathlib
import platform
import sys
import time
from typing import Any

ROOT = pathlib.Path(
    os.environ.get("GQC_SOURCE_ROOT", str(pathlib.Path(__file__).resolve().parents[3]))
)
OUT = pathlib.Path(__file__).resolve().parent
engine, case_name = sys.argv[1:3]
case = json.loads((OUT / "cases.json").read_text())[case_name]
sys.meta_path[:] = [
    f
    for f in sys.meta_path
    if not type(f).__module__.startswith("_editable_skbc_generativeqc")
]
sys.path.insert(0, str(ROOT / "python"))
start = time.perf_counter()
startcpu = time.process_time()
if engine == "gqc":
    import generativeqc
    from generativeqc import Calculator, _native
else:
    import pyscf
    from pyscf import gto, lib, scf

    lib.num_threads(1)
import_wall = time.perf_counter() - start
import_cpu = time.process_time() - startcpu
rows: list[dict[str, Any]] = []
for index, phase in enumerate(["cold", "warm", "warm", "changed_geometry"]):
    atoms = [(s, list(xyz)) for s, xyz in case["atoms"]]
    if phase == "changed_geometry":
        atoms[-1][1][0] += 0.02
    w = time.perf_counter()
    c = time.process_time()
    if engine == "gqc":
        calc = Calculator(
            "rhf",
            basis=case["basis"],
            device="cpu",
            basis_representation="spherical",
            density_fitting="none",
            energy_tolerance=1e-10,
            density_tolerance=1e-8,
            diis_history=8,
            max_iterations=100,
            precision="fp64",
            screening_tolerance=1e-13,
        )
        setup_wall = time.perf_counter() - w
        runstart = time.perf_counter()
        r = calc.singlepoint(atoms, charge=0, multiplicity=1, properties=("energy",))
        run_wall = time.perf_counter() - runstart
        result = {
            "energy": r.energy,
            "converged": r.converged,
            "iterations": r.iterations,
            "energy_change": r.energy_change,
            "density_rms": r.density_rms,
            "backend": r.executed_backend,
            "precision": r.precision,
            "incremental_direct_jk": r.incremental_direct_jk,
            "basis_metadata": r.basis_metadata,
            "native_fock_builds_source_derived": r.iterations + 2,
        }
    else:
        mol = gto.M(
            atom=atoms,
            basis=case["pyscf_basis"],
            cart=False,
            unit="Bohr",
            charge=0,
            spin=0,
            verbose=0,
            max_memory=0 if engine == "pyscf-direct" else 4000,
        )
        mf = scf.RHF(mol)
        mf.chkfile = None
        mf.init_guess = "1e"
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-8
        mf.diis_space = 8
        mf.direct_scf = True
        mf.direct_scf_tol = 1e-13
        mf.max_cycle = 100
        checks = []
        jk_calls = [0]
        orig_jk = mf.get_jk

        def get_jk(
            *a: Any, _calls: list[int] = jk_calls, _original: Any = orig_jk, **kw: Any
        ) -> Any:
            _calls[0] += 1
            return _original(*a, **kw)

        mf.get_jk = get_jk

        def check(
            e: dict[str, Any], _checks: list[dict[str, Any]] = checks, _mol: Any = mol
        ) -> bool:
            de = abs(e["e_tot"] - e["last_hf_e"])
            dr = e["norm_ddm"] / _mol.nao_nr()
            _checks.append(
                {
                    "energy_change": de,
                    "density_rms": dr,
                    "orbital_gradient_norm": float(e["norm_gorb"]),
                }
            )
            return de < 1e-10 and dr < 1e-8

        mf.check_convergence = check
        setup_wall = time.perf_counter() - w
        runstart = time.perf_counter()
        mf.kernel()
        run_wall = time.perf_counter() - runstart
        result = dict(
            energy=mf.e_tot,
            converged=mf.converged,
            iterations=mf.cycles,
            post_scf_cycles=len(checks) - mf.cycles,
            jk_calls=jk_calls[0],
            eri_incore=mf._eri is not None,
            nao=mol.nao_nr(),
            backend="pyscf_cpu_libcint",
            **checks[-1],
            convergence_checks=checks,
        )
    wall = time.perf_counter() - w
    cpu = time.process_time() - c
    row = dict(
        engine=engine,
        case=case_name,
        phase=phase,
        index=index,
        wall_seconds=wall,
        cpu_seconds=cpu,
        setup_wall_seconds=setup_wall,
        scf_wall_seconds=run_wall,
        **result,
    )
    rows.append(row)
    print(json.dumps(row, default=str), flush=True)
    if not result["converged"]:
        raise RuntimeError("Unconverged benchmark sample")
from threadpoolctl import threadpool_info

metadata = {
    "engine": engine,
    "case": case_name,
    "import_wall_seconds": import_wall,
    "import_cpu_seconds": import_cpu,
    "cold_import_plus_endpoint_seconds": import_wall + rows[0]["wall_seconds"],
    "python": sys.version,
    "platform": platform.platform(),
    "threadpool_info": threadpool_info(),
    "thread_env": {
        x: os.environ.get(x)
        for x in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "PYTHONHASHSEED",
        ]
    },
}
if engine == "gqc":
    native = pathlib.Path(_native.load_library()._name)
    metadata.update(
        package_file=generativeqc.__file__,
        native_library=str(native),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        native_source_identity=_native.load_library()
        .generativeqc_get_source_identity()
        .decode(),
    )
else:
    metadata.update(package_file=pyscf.__file__, pyscf_version=pyscf.__version__)
if len(sys.argv) > 3:
    pathlib.Path(sys.argv[3]).write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2, default=str)
    )
