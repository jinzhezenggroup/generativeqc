# Decision: do not promote incremental J/K from the 96-atom PBE0 probe

Status: rejected (default promotion for this configuration, not feature removal)
Date: 2026-10-09

## Question and source boundary

After correcting our benchmark metrics, the user requested further optimization
investigation. An existing exact-direct, strict-FP64 incremental J/K path was
tested without altering production source, convergence thresholds, initial
density policy, force consumers or resource qualification gates.

The native binary belongs to supported CUDA/MINAO source
`4385f72751b829883407c01106186917c344317b`, with the already qualified SM120
build and packaged PBE0 force artifact. It is not a measurement of this older
dirty VibeQC worktree. Both arms use the same binary, 96 atoms/768 spherical
def2-SVP AOs and 2,359,296 explicit unpruned moving-grid points. Each arm is a
fresh process; cold, fixed-density warm, moved and moved-warm complete energy
plus host-force endpoints are retained. No oracle density is imported.

## Evidence

Finite Slurm job 2745 on n2/node2 (`main`, `gpu:pro6000:1`, 15 minutes),
RTX PRO 6000 Blackwell. Device visibility is preserved. Native/reference energy,
density/gradient and screening controls remain unchanged. The experimental
selector is `GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK=1`, with its existing default
cadence and native closure policy. There is no explicit resource budget/plan:
public KS resource planning intentionally rejects this unaccounted experimental
storage. No such rejection is bypassed and no capacity guarantee is claimed.

| Phase | Ordinary E+F seconds | Incremental E+F seconds | Ordinary full/delta builds | Incremental full/delta builds |
| --- | ---: | ---: | --- | --- |
| Cold | 107.334747 | 117.281216 | 17/0 | 13/8 |
| Warm | 18.636754 | 27.923880 | 1/0 | 3/0 |
| Moved | 79.857458 | 86.595493 | 12/0 | 10/6 |
| Moved-warm | 18.608843 | 27.901869 | 1/0 | 3/0 |

All eight complete endpoints pass independent `1e-8 Eh`/`1e-7 Eh/Bohr`
energy/force gates and the unchanged native energy/density/physical residual
checks. Maximum independent errors are `9.550e-12 Eh` and
`3.784e-11 Eh/Bohr`. No warm fallback is taken. Both child processes and the
outer finite allocation exit zero.

Full/delta counts come from actual native run diagnostics; their sum matches
the reported Fock builds. Admitted-quartet counters are unavailable in these
records, so zero counters are **not** a claim of no work or a measured quartet
reduction. Cold replaces four full builds but adds eight delta builds; the
complete endpoint does not improve. Warm follows additional full-density
closure and takes three builds instead of one. Those scientific safeguards
must not be removed to manufacture a favorable benchmark.

There is one process per arm, not a statistical performance campaign. This
supports rejecting immediate default promotion, not a universal conclusion
that incremental J/K cannot help other densities, systems or policies.

## Better next target

The retained independently qualified ordinary baseline in
`.cache/pbe0-96-cold-test/2715/native-round0.json` attributes warm force return
to a `12.352820 s` force endpoint within `18.682003 s` complete E+F:
about two thirds of the warm endpoint. Its measured stationary integral
derivatives take `6.045080 s`; semilocal geometry response takes `6.019443 s`.
These are host-wall component observations, not individual J/K kernel timings.
Cold/moved geometry response is roughly `10.35–10.38 s` in that retained run.

Prioritize a complete force-phase timeline and investigation of source reuse,
resident grid/derivative state, geometry batching and synchronization, rather
than another attempt to optimize only the SCF iteration count. Existing
telemetry exposes 4,608 geometry batches and thousands of transfer/drain calls,
but counts alone do not prove these dominate wall time. The exact moving grid,
bounded owner lifetimes, force oracle gates and lower-memory fallbacks must be
preserved. Do not cache forces across merely similar densities or claim a
speedup without a complete endpoint measurement.

## Retention and revisit conditions

Local evidence: `.cache/benchmark-scf-fix-20261009/2745/`.
Remote evidence: `/data/jzzeng/benchmark-scf-fix-20261009/runs/2745` on n2.
Original JSONs, operator counters, scripts, source/binary hashes, finite job
receipt, subprocess outcomes and a strict `summary.json` are retained.

Revisit incremental promotion only with complete resource accounting and
independent unchanged endpoint gates, repeated interleaved evidence, larger
size/work counts and successful cold/warm/changed-geometry behavior. Force
optimization must be qualified separately rather than inferred from this
rejected selector.
