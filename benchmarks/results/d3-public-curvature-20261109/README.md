# Installed native D3(BJ) molecular-curvature qualification

This is a CPU/direct/FP64, spherical, all-electron **restricted** B3LYP-D3(BJ)
water/STO-3G receipt on the explicit grid in
`tests/data/d3_second_order/manifest.json`. It is not unrestricted, ECP,
default-grid, def2-SVP/Ni, molecular TS or frequency qualification. It does not
close the complete acceptance contracts of #2844–#2848.

## Producer and identity

`tools/qualify_d3_second_order.py` runs from outside the checkout, with
`PYTHONPATH` and `GENERATIVEQC_LIBRARY` unset, against an installed CPU wheel.
`installed-receipt.json` binds the actual installed Python files, native
library, parameter/source identities and fixture bytes. Its Python and fixture
digests match the committed checkout. The producer imports no PySCF,
simple-dftd3 or geomeTRIC; the receipt verifies those modules remain unloaded.

The locally built, unaudited CPU wheel uses the existing LP64 OpenBLAS provider
from `scipy_openblas32`, explicitly exposed through `LD_LIBRARY_PATH`. That
search path is retained in the receipt. This is an installed-API experiment,
not a self-contained distribution or a published release artifact.

Example, after building/installing the CPU wheel with the verified sccache
launcher and `GENERATIVEQC_BUILD_CLI=OFF`:

```bash
cd /tmp
OPENBLAS_DIR=$("$PYTHON" -c 'import scipy_openblas32; print(scipy_openblas32.get_lib_dir())')
env -u PYTHONPATH -u GENERATIVEQC_LIBRARY \
  LD_LIBRARY_PATH="$OPENBLAS_DIR" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 1800 "$PYTHON" "$CHECKOUT/tools/qualify_d3_second_order.py" \
  --manifest "$CHECKOUT/tests/data/d3_second_order/manifest.json" \
  --cache /tmp/d3-qualification-artifacts --output /tmp/d3-qualification.json
```

## Complete endpoint observations

The retained final experiment replays JIT artifacts but starts a fresh native
SCF for each public call. Preparation through both endpoints and cleanup takes
46.7653 s: the full raw Hessian endpoint takes 41.9056 s and the independent HVP
endpoint 4.65881 s. This is not a speedup claim or a cold-compilation comparison.

Each SCF uses 10 iterations / 12 Fock builds. The matrix has nine electronic
response solves / 81 response iterations and nine D3 actions; the HVP has one
electronic response solve / nine iterations and one D3 action. D3 adds zero
electronic response solves and zero displaced SCFs. Its scalar work is
51,102 / 5,678 respectively, with three physical atom pairs per action.

Raw H@v error is `1.00673e-14`, symmetry error `1.79412e-13`, and translation
error `8.90676e-14`, without symmetrization or projection. Peak process RSS is
428,572 KiB. The `getrusage` observations distinguish process and child
high-water marks; they are not a simultaneous process-tree memory sum.

## Scientific checks and failures

The independent tests use PySCF 2.14.0 with the explicit Gaussian/VWN-RPA
component expression and simple-dftd3 1.4.0 with the same canonical BJ record.
They reconstruct the identical moving-grid quadrature and independently
reconverge at every displaced geometry. No displaced production SCF supplies
the Hessian or its oracle.

The initial three-point stencil at 0.003 Bohr failed the unchanged `1e-5`
full-column gate with error `4.806239251320399e-5`; its original failure is
retained in `failed-three-point-reference.json`. The five-point fourth-order
stencil uses the same steps and physical gates, reducing reference truncation
instead of relaxing native acceptance. Intermediate reference tests are
explicitly labelled in `qualification-summary.json`.

The final installed D3-composition and semilocal-public regression run passes
25 tests in 286.20 s, including all 13 new composition tests. Its full-matrix
reference errors at `0.003`, `0.001` and `0.0003 Bohr` are `1.56051e-9`,
`8.55606e-10` and `1.75315e-8`, below the unchanged `1e-5` gate. The tests also
exercise release of electronic grid storage before correction construction,
wrong geometry/MethodIR rejection, parameter changes, duplicate-source and
force-sign negative controls, output/workspace failures, and recovery after a
public correction failure. An earlier installed regression passes 41 tests;
it is retained separately rather than labelled as the final source snapshot.

Raw logs and the exact JUnit receipt remain ignored locally and recoverable
from existing Git commit `b1d20f9f6ade91f08f27c024177df3209aa10936`, under the same
result directory. For example, recover the original failed trial with
`git show b1d20f9f6:benchmarks/results/d3-public-curvature-20261109/three-point-reference-failure.log.xz`.
The compact failed record binds its archive digest. Current reviewed storage
contains compact scientific receipts rather than transient build/test archives;
the aggregate evidence budget is unchanged.

Compiler-cache observations are from a shared sccache 0.16.0 server. Other
compilations may contribute to their counters; no per-task cache-hit rate is
inferred. The actual native build commands use the verified launcher.
