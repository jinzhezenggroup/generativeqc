"""Run the existing native fallback contracts through verified sccache probes."""

import ctypes as ct
import os
import subprocess
import sys
from pathlib import Path

assert os.environ.get("SLURM_JOB_ID")
root = Path.cwd()
out = root / ".artifacts/cbs-standard/probes"
out.mkdir(parents=True, exist_ok=True)
library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
subprocess.run(["sccache", "--version"], check=True)
(out / "cache-before.txt").write_bytes(
    subprocess.check_output(["sccache", "--show-stats"])
)
for stem in ("df_cc_solver_probe", "df_cc_lambda_probe"):
    obj = out / (stem + ".o")
    target = out / (stem + (".so" if "lambda" in stem else ""))
    subprocess.run(
        [
            "sccache",
            "c++",
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-Isrc",
            "-Iinclude",
            "-c",
            str(root / "tests/native" / (stem + ".cpp")),
            "-o",
            str(obj),
        ],
        check=True,
    )
    subprocess.run(
        [
            "c++",
            *(["-shared"] if "lambda" in stem else []),
            str(obj),
            str(library),
            "-Wl,-rpath," + str(library.parent),
            "-o",
            str(target),
        ],
        check=True,
    )
(out / "cache-after.txt").write_bytes(
    subprocess.check_output(["sccache", "--show-stats"])
)
sys.path.insert(0, str(root / "tests/python"))
import test_df_cc_mixed_precision as cc_tests
import test_df_lambda_mixed_precision as lambda_tests

solver = (out / "df_cc_solver_probe", True)
cc_tests.test_mixed_resource_refusal_keeps_strict_execution(solver)
for tolerance in (1e-2, 1e-7):
    cc_tests.test_early_switch_preserves_independent_fp64_certificate(solver, tolerance)
for occupied, pairs in ((1, False), (3, False), (3, True)):
    for symmetric in (True, False):
        cc_tests.test_strict_pair_boundary_preserves_supplied_state_contract(
            solver, symmetric, occupied, pairs
        )
call = ct.CDLL(str(out / "df_cc_lambda_probe.so")).df_cc_lambda_probe
dp = ct.POINTER(ct.c_double)
call.argtypes = [
    ct.c_size_t,
    ct.c_size_t,
    ct.c_size_t,
    ct.c_int,
    ct.c_size_t,
    ct.c_size_t,
    ct.POINTER(dp),
    dp,
    dp,
    ct.POINTER(dp),
    dp,
    ct.POINTER(ct.c_size_t),
    ct.c_void_p,
    ct.c_size_t,
]
call.restype = ct.c_int
lambda_tests.test_optional_matrix_refusal_keeps_original_fp64_actions(call)
lambda_tests.test_finite_mixed_exhaustion_is_only_a_warm_start(call)
print(
    "PASS: 11 CCSD/Lambda native refusal and strict-publication contracts", flush=True
)
