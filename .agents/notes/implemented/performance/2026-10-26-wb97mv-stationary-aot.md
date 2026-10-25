# Decision: package the default WB97M-V composite stationary wrapper

Status: implemented
Date: 2026-10-26

## Problem

WB97M-V48 cold timings changed by tens of seconds across fresh processes even
when their SCF work was unchanged. Process/open-file tracing established that
the first force call ran sccache, NVCC, cicc and ptxas through
`compile_stationary_cuda`. A second process reused that artifact. Intrusive
profiling measured 80.44 seconds in first compilation and none on replay, while
native SCF execution stayed near 72.5 seconds in both runs. Those intrusive
measurements diagnose compilation; they are not clean endpoint rankings.

A populated writable JIT cache is therefore not evidence of a production cold
optimization. Do not attribute that cache-state difference to the DF transform,
GPU clocks or a better SCF schedule.

## Decision and invariants

Add both spins of WB97M-V to the existing stationary AOT catalog, with the actual
composite point model `libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16` declared
explicitly. The semilocal catalog point model is mathematically distinct and
must not be substituted merely because the functional code matches. Native
stationary admission still follows external-provider source coverage; extending
the package catalog does not expand native method capability.

Default all-electron composite forces select that exact package when both XC
and VV10 partition iterations match. Public execution passes an explicit target
without discovering a compiler. Missing or corrupt declared packages fail closed;
custom domains retain bounded JIT. Native integral and VV10 ownership stays
unchanged. A geometry-only package consumer must use the complete packaged
primitive inventory to identify nuclear dispatch: its nuclear index is not zero.
The host component sum validates the existing compiler source-coverage contract
without building an unused reduction IR.

## Evidence and limitations

Actual GPU endpoints guard compiler discovery, primitive/wrapper emission and
IR generation. RKS/UKS cold/reuse and finite-difference forces pass independent
GPU4PySCF references for sto-3g and spherical def2-SVP. The expanded test also
checks moved-force publication under the same guard. Package auditing verifies
all ordinary and SPD wrappers and native cubin coverage.

Three pre-existing larger-basis reference gates remain failures at their
unchanged 1e-8 Eh energy threshold: water def2-TZVP (about 2.50e-8), local
def2-TZVPD (about 2.56e-8), and NH2 UKS def2-TZVP (about 8.05e-8). Each failure
was reproduced with the preserved baseline source/library; matching baseline
and candidate energies differ only near roundoff. These failures must remain
visible rather than being relabelled as passing independent qualifications.

Clean repeated #2786 timings use a separate frozen default image and retain
constructor, preparation, SCF, analytic forces, host export and three frozen
engine-local warm calls. No cache clearing or precomputed oracle seed is used.

Three default-only rounds give Direct/DF WB48 cold medians 82.4468/67.7175
seconds versus GPU4PySCF 96.5556/90.5603 seconds; all warm populations and
independent dense-reference gates are retained in
[the endpoint evidence](../../../../benchmarks/results/cold-four-defaults-20261026/README.md).
These gains against GPU4PySCF do not turn cache-hit baseline timings into
algorithmic SCF improvements.

## Rejected alternatives and references

Changing nonlocal inner convergence from the old strict residual multiplier
0.01 to 0.1 reduced SCF updates from 28 to 26 but increased pair sweeps from four
to five and regressed the full WB endpoint. A looser provisional pure-J delta
cutoff did not improve Direct r2SCAN-3c96. Both changes were reverted; do not
repeat them based only on SCF iteration counts.

Frozen `wb-startup`, `outer-j-paired`, `wb-aot*` and `default-four-paired`
evidence remains under `n1:/data/jzzeng/cold-opt-20261025/`, bound to base
`82ce9e56a352fcc1d86e7d64eeaae08c6f1eb5c8` plus recorded patches/archive hashes.
Current packaging is described in
[stationary CUDA diagnostics](../../../../docs/developer/stationary_cuda_diagnostic.md).
