# Bounded FP64 XC cross-product proof

Measured baseline: `f088d3fcd30435c125a5b92422e4d91088884977`.
Measured candidate: `1ccc53419acd4c22d9496178a65fffd83f26d6b1`.
Candidate core SHA-256:
`4555f81fe4e277565cec634d642b41af99aa8c99ae108b0997ea78dec47e43db`.
Later archive/PR commits are not relabeled as measured source.

The existing shared-panel barrier proves a conservative operand bound for each
FP64 cross tile. Within that bound and a bounded incoming accumulator, the
identically ordered FMAs are provably finite and need no individual overflow
check. Unbounded, NaN/Inf and large-accumulator cases retain the original grouped
retry. Diagonal admission and FP32 arithmetic are unchanged. There is no new
global storage, altered screening, grid, basis, precision or convergence gate.

## Complete paired endpoints

Both arms enable the separate optional physical-validation GEMM feature from
PR #2718. Other production controls retain their defaults; no spatial ordering,
occupied-XC, DF charge-split or AO-cache experiment is enabled. Each process
includes constructor, preparation, SCF, analytic forces and host export in cold,
then three frozen-own-density warm calls. Five alternating process pairs run
on the same Slurm-owned RTX 5090 for each method/mode.

| Standard water96 | Baseline cold (s) | Candidate cold (s) | Gain |
| --- | ---: | ---: | ---: |
| PBE DF | 19.336354 | 17.795994 | 7.966% |
| PBE Direct | 19.796046 | 18.159974 | 8.265% |
| PBE0 DF | 31.115664 | 29.707450 | 4.526% |
| PBE0 Direct | 23.107939 | 21.659104 | 6.270% |
| r2SCAN-3c DF | 50.228904 | 48.059313 | 4.319% |
| r2SCAN-3c Direct | 44.153591 | 42.045853 | 4.774% |

All six pass `max(0.02, 2*(baseline_MAD/baseline_median +
candidate_MAD/candidate_median))`, which resolves to 2% here. Warm statistics
use one median per process: PBE DF/Direct and PBE0 Direct pass; PBE0 DF and
both r2SCAN-3c warm comparisons do not. Actual iteration/Fock counts remain
in every record and match between paired arms. Screens are separate observations.

All 352 complete calls pass the existing independent energy/full-force gates:
24 pilot, 240 paired, 56 seven-method Standard/Fast Direct/DF cold/warm, and
32 PBE/r2SCAN-3c cold/warm/moved/moved-warm calls. Moved geometries use separately
retained independent references. The seven-method matrix retains slow endpoints,
including omegaB97M-V DF cold at 676.154 seconds. Its Direct cold is 303.948
seconds, below the historical stock GPU4 value of 369.762 seconds, but this
single matrix sample is not a repeated GPU4 comparison or all-method parity.

## Numerical and execution qualification

The actual generated-body host simulator passes two suites. Real-device
arithmetic passes 134 cases, including FMA on/off, cross-leg cancellation near
overflow, large incoming sums across tiles, dyadic boundaries, NaN/Inf, FP32
compatibility and extreme diagonal cases. Native resident/resource, matrix,
variational and capture checks pass, as do 72 host-resource cases and 89
native-backed contraction cases. Prepared PBE0/B3LYP independent E/F/replay/
displaced controls and unprepared PBE0 independent E/F pass. AOT memcheck and
native memcheck/synccheck/racecheck report zero errors; racecheck has zero
hazards and warnings. Source-generation structure checks report zero errors.

The local broader host run passed 959 cases but lacked a native library for 84
cases; those modules were rerun against the matching remote package, passing
all 89 cases there. Original failures remain in provenance. GPU work used
finite Slurm jobs on node1; device visibility was preserved. The full build used
verified sccache 0.16.0 and authenticated 12,426 tracked source files. No oracle
calculation supplies a timed production seed.

## Offline reproduction and storage

From the repository root:

```sh
PYTHONPATH=python:. python benchmarks/results/xc-bounded-cross-20261024/reproduce.py
```

The verifier authenticates all 108 original JSON texts and both source manifests,
rechecks all 352 calls against the retained independent references, checks paired
settings/device identities and recomputes the twelve cold/warm timing verdicts.
The endpoint packet retains every force component; provenance retains the
source/build/package/cache/test identities, allocations and original recipes.

Thirty-three older JSON files in two campaigns are compressed with deterministic
gzip to recover 1,290,910 bytes within unchanged storage caps. Every decompressed
file equals its original Git blob. `history-storage.json` pins both forms and
supports restoration of original names through the shared offline tool. Failed
historical samples and all numerical values are preserved.
