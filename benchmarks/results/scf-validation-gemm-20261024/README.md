# Optional BLAS products in physical SCF validation

Frozen source: `f088d3fcd30435c125a5b92422e4d91088884977`, based on master
`e85e5251e571f5b4a0008a19d83230d3ca8594e2`. Main CUDA library SHA-256:
`639ec53fd7786b909b53be2539724b3c1dc9ae6a85eb94578e4a29ec39bb1288`.

`GENERATIVEQC_SCF_VALIDATION_GEMM=1` selects the existing FP64 CPU tensor
service for physical-validation products. The default remains scalar. This
campaign compares selector 0/1 in the same build with the pinned LP64 OpenBLAS
provider and admitted local thread ownership. It does not change the grid,
basis, screening, precision, convergence, scientific acceptance or reference
multiplication implementation. No reference calculation supplies a timed seed.

## Complete cold results

Each water96 Standard endpoint includes construction, preparation, SCF, analytic
forces and host export. Five alternating fresh-process pairs per method/mode
retain one cold and three frozen-own-density warm calls per process. Stock
production controls are used; earlier combined AO/spatial/DF partition
experiments are separate evidence and are not pooled here.

| Endpoint | Scalar cold median (s) | GEMM cold median (s) | Improvement | Passes gate |
| --- | ---: | ---: | ---: | --- |
| PBE DF | 19.448738 | 19.036041 | 2.12197% | yes |
| PBE Direct | 20.148883 | 19.717227 | 2.14233% | yes |
| r2SCAN-3c DF | 50.740801 | 49.471292 | 2.50195% | yes |
| r2SCAN-3c Direct | 45.198221 | 43.942942 | 2.77727% | yes |
| PBE0 DF | 31.084483 | 30.642720 | 1.42117% | no |
| PBE0 Direct | 23.404243 | 22.946027 | 1.95784% | no |

The gate is `max(0.02, 2*(baseline_MAD/baseline_median +
candidate_MAD/candidate_median))`; it resolves to 2% in these six cold
comparisons. None of the warm comparisons passes. Warm statistics use one
median per process, never a pooled population. Complete times, all 288 force
components, actual iteration/Fock counts, scientific settings, accelerator and
Slurm identity remain in every raw record. These are incremental gains, not
GPU4PySCF cold parity or a global promotion of this optional schedule.

## Qualification

All 328 complete calls independently pass the existing README energy/full-force
gates and applicable reference grid-stability checks:

- 240 paired calls: three methods, Direct/DF, five pairs, four calls per process.
- 56 matrix calls: all seven methods, Standard/Fast, Direct/DF, cold and warm.
- 32 geometry calls: PBE and r2SCAN-3c, Standard/Fast, Direct/DF, cold, warm,
  moved and moved-warm, using separately retained independent moved references.

The seven methods are PBE, PBE0, B3LYP, r2SCAN, r2SCAN-3c, M06-2X and
omegaB97M-V. Numerical-reference reuse retains original producer identities;
no oracle timing is used as a performance baseline. The fitted omegaB97M-V
matrix's roughly 698-second cold calls remain visible: this optional host
product change does not address the full fitted nonlocal bottleneck.

The source-matched GPU package passes resident KS ownership/resource tests,
native XC matrix/variational/capture tests, 72 host-resource tests, prepared
PBE0/B3LYP independent E/F/replay/displaced controls, and unprepared PBE0 E/F.
AOT memcheck and native matrix memcheck/synccheck/racecheck complete with zero
errors; racecheck reports zero hazards and warnings. Local validation-product
and CMake tests pass 35 cases; provider/packaging checks pass 97 with two skips.

The first build exposed an old native test calling a diagnostic that moved to
an optional batch capability. The test-only fix preserves all assertions.
Both the original failed build and successful resume are retained. An AOT
observation harness initially named a prototype-only spatial entry point; the
corrected harness observes the actual resident molecular consumer. Its failed
attempt remains separate from the completed checks. Neither failure is a
successful scientific endpoint or performance observation.

All real-GPU work uses finite Slurm allocations on node1 RTX 5090 with assigned
device visibility. Build commands use verified sccache 0.16.0 and pinned
`scipy-openblas32==0.3.34.0.0`. Shared cache statistics overlap another build
and are not attributed as exclusive job hit counts. The provider admission fix
can affect other automatic CPU consumers, which is why this whole binary was
qualified independently of the earlier combined prototype.

## Reproduce and retention

From the repository root:

```sh
PYTHONPATH=python:. python benchmarks/results/scf-validation-gemm-20261024/reproduce.py
```

`endpoints.json.xz` retains all 96 original endpoint JSON texts and their byte
hashes, including every force component and the moved references.
`provenance.json.xz` retains the complete 12,425-file measured source manifest,
verification receipt, build/cache/package identities, scripts, allocations and
successful/failed test logs. The verifier authenticates retained inputs,
rechecks all 328 calls, recomputes paired statistics and emits actual work counts.
It requires no GPU and does not relabel the measured source as the later
archive/documentation commit.

To stay within existing storage limits, 24 older JSON records are compressed
losslessly with deterministic gzip. `history-storage.json` binds original and
stored sizes/hashes and the original Git snapshot. All original bytes, numerical
results, failure records and logical identities are retained. Existing gzip-aware
consumers continue to work; historical original-name readers can restore the
snapshot with the shared tool as documented in each affected campaign. No
per-file, aggregate or incoming-review limit is increased.
