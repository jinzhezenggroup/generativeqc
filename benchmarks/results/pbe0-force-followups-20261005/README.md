# Force follow-ups: root reuse and cold-cache diagnostics

These exact-direct PBE0/def2-SVP RTX 5090 observations retain a complete
48/96-atom primitive-root reuse experiment and two supplemental 96-atom
force-map policy repetitions. **No implementation or default is promoted.**
The numerical gates pass; warm improvements do not erase moved regressions
or establish the cause of different SCF trajectories. #1895 remains open.

## Primitive-local Coulomb roots: Slurm 5842

Both arms use the same public force-AO profile, generic SCF local-AO admission,
phased Becke and indexed force scheduling. The candidate computes each reached
Cartesian Coulomb root once per primitive order-4/5/6 gradient and retains
the existing scalar root arithmetic and subset accumulation order. The root
table has maximum capacities 36/48/64 doubles. Screening and numerical gates
are unchanged. `root-reuse.patch.gz` preserves the experiment and host tests;
the patch is not applied to shipping source by this evidence PR.

Complete E+F seconds include the harness's preparation charge. Warm rows are
medians of all five samples; cold/moved are individual observations.

| Atoms | Phase | Control | Root reuse | Candidate/control |
|---|---|---:|---:|---:|
| 48 | cold | 124.6324 | 120.4732 | 0.9666 |
| 48 | warm | 10.7442 | 10.4181 | 0.9696 |
| 48 | moved | 54.6495 | 57.8190 | 1.0580 |
| 48 | moved-warm | 10.7468 | 10.4181 | 0.9694 |
| 96 | cold | 259.3527 | 264.6550 | 1.0204 |
| 96 | warm | 32.9062 | 31.6565 | 0.9620 |
| 96 | moved | 144.2289 | 187.5610 | 1.3004 |
| 96 | moved-warm | 32.4909 | 31.1901 | 0.9600 |

Actual cold/moved Fock counts are 25/12 versus 24/13 at 48 atoms and 26/14
versus 27/20 at 96 atoms. Every native warm call performs one build and no
fallback. The 96-atom moved endpoint regresses 30.044%; do not normalize its
time by iterations or omit it. Root arithmetic is force-side work, so these
records alone do not identify a causal explanation for different SCF counts.
The experiment remains unpromoted under the complete-endpoint policy.

Process order is control/candidate at 48 atoms and candidate/control at 96.
Each arm/size gets its own initially empty generated-artifact cache. Matched
GPU4PySCF reference processes use full-density J/K rebuilds, CUDA LibXC and
analytic moving-grid response; all reference samples and iteration totals are retained.
Neither reference nor native uses DF/COSX to satisfy the Direct comparison.

Slurm 5841 passes the independent native through-f full/LR response gates and
memcheck/initcheck with zero errors. Host tests pass 19,440 all-center
displaced-Hermite finite differences and poisoned-root-table checks. Native
receipts and the compiler/test patch are retained. These gates establish
numerical correctness for the exercised domain, not performance promotion.

## Supplemental cold repetitions

These runs use the current control source and compare the shared public
force-map profile (`default`) with an empty registry (`dense-control`). They
use one warm and one moved-warm replay, so they do not replace the complete
five-warm qualification in [PR #1934](https://github.com/jinzhezenggroup/generativeqc/pull/1934).

Slurm 5837 started separate processes but reused one generated-artifact cache:

| Order | Arm | Cold, s | Cold builds | Moved, s | Moved builds |
|---|---|---:|---:|---:|---:|
| 1 | default | 285.9184 | 30 | 128.3276 | 12 |
| 2 | dense-control | 264.1377 | 29 | 144.5390 | 13 |
| 3 | dense-control | 242.8433 | 26 | 136.9613 | 12 |
| 4 | default | 254.8404 | 29 | 135.2013 | 13 |

The first arm's force-owner construction/cache setup costs 26.0746 s, versus
about 3 s in subsequent arms. Thus this ordering is not a clean cold-cache
comparison. Keep the full records; do not subtract that cost to manufacture
a passing cold result.

Slurm 5851 repeats in reversed ABBA order with a separate initially empty
generated-artifact cache for **every** process:

| Order | Arm | Cold, s | Cold builds | Moved, s | Moved builds |
|---|---|---:|---:|---:|---:|
| 1 | dense-control | 321.1362 | 33 | 139.4326 | 12 |
| 2 | default | 267.5110 | 27 | 131.8170 | 12 |
| 3 | default | 253.6115 | 25 | 137.4798 | 13 |
| 4 | dense-control | 282.8244 | 28 | 168.6163 | 16 |

All observed candidate cold/moved times are below the corresponding outer
control arm, but SCF counts vary substantially. These are ordered observations,
not a causal estimate or proof over the profile's full structural range. The
shared compiler ccache remains enabled and is never cleared: "empty cache"
here means the generated-artifact cache, not every system/toolchain cache.
Old negative observations remain in #1934 and the first table above.

## Verification and provenance

```bash
PYTHONPATH=python:. python benchmarks/results/pbe0-force-followups-20261005/verify.py
```

The verifier authenticates every publication member, checks scientific
protocols and actual selected routes, and recomputes all 592 same-geometry
E/F pairings: 432 in the root experiment, 64 cold-repeat/reference pairings
and 96 pairings between cold-repeat native processes. Gates remain 1e-8 Eh
and 1e-7 Eh/Bohr. The gzip files preserve original JSON bytes, including
every sample, component timer, work record, count and reference trajectory.

`identities.json` pins both native libraries and source identities. Apply
`control-source.patch.gz` to permanent base
`9d0f4fc0ddbe9c1dc4539cd4f6ad8a6e019ba990`, then `root-reuse.patch.gz` for the
candidate. Both source identities have been independently reconstructed;
see `reconstruction.json`. Executable hashes are receipts, not claims of
future byte-identical toolchain output. Drivers, source/binary checks,
scheduler records and sanitizer output are retained in `receipts.json`.

These historical runs have actual iteration/Fock totals but did not serialize
per-iteration energy/residual histories or measure allocator peaks. PR #1936
adds history retention for future campaigns; it does not backfill these runs.
No significant interleaved performance pass, default promotion, or Direct
parity pass is asserted by this publication.
