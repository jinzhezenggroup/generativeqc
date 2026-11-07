# Rejected VV10 FP64 FMA candidate

This is retained **negative evidence**, not a speedup/default-promotion claim.
The production candidate is reverted. Its source and actual runs remain
reconstructible; the final PR changes no production implementation or defaults.

## Complete endpoints

96-atom water, 768 spherical def2-SVP AOs, WB97M-V full-Direct complete
energy plus analytic forces, including moving quadrature and VV10. Every arm
has one cold observation and three frozen-engine-local-density warm calls.
No production PySCF/GPU4PySCF import is permitted. Normal Release O3,
CUDA12.9.86/g++12, verified sccache0.16.0, SM80-real plus SM120-real.

The same ELF uses `GENERATIVEQC_VV10_PAIR_SCHEDULE=parallel-packet` for
the incumbent and unset for the experimental FMA candidate. This controls only
the candidate image; **unset in the final restored source remains incumbent**.

| Actual allocation | Cold incumbent -> candidate, single observations | Three-warm median incumbent -> candidate |
| --- | ---: | ---: |
| A100 / LSF225807 | 111.798143 -> 113.152722 s | 15.038729 -> 15.015639 s |
| RTX5090 / Slurm16020 | 193.947278 -> 194.417461 s | 14.350758 -> 14.376013 s |
| RTX5090 / Slurm16020, r²SCAN-3c control | 28.611057 -> 28.615110 s | 10.536715 -> 10.535370 s |

The control has 1088 spherical def2-mTZVPP AOs and native D4/gCP. Settings
and recorded work match: WB cold 18 Focks/iterations, every warm 1/1; the
control 10 and 1. VV10 provisional/target Focks, pair cache probes/hits/sweeps
and J/K applications match. Exact active scalar-pair microcensus is unavailable
in clean runs, not zero. All paired `1e-9 Eh/atom`, `1e-7` max and `3e-8`
RMS force gates pass. These data do **not** justify a performance change.

Measured source: `c8b75770b78b79ef8b60486d9be1d754698073394affb1f1141c564517b188e0`.
Measured ELF: `a29218a2fb8990cae5099e88443e54c1816506f0d1305ee2043a4c8c343c8791`.
Base: `1dfcaa940193fce064d337dfb63107d595e614e3`; measured implementation
commit `e315194a1` (the immutable test snapshot predates formatting-only
changes in that commit). `candidate.patch.xz` is the exact measured overlay,
**not** the final PR's production diff.

## Actual cold bottleneck

One intrusive diagnostic, Slurm16019, used the previously qualified
`ff6117e3…`/ELF`4c73bcf8…` image. Four feature pair sweeps take 84.536 s,
about 21.13 s each. Cold stationary forces also contain an uncached geometry
sweep. Completed pair/preparation/endpoint journal scopes and the full CUDA
trace are retained in `diagnostic.json.xz`. The verbose append-only journal
remains in ignored local artifacts, with its exact byte length and checksum.

Its 225.251770 s cold and 16.113821 s warm are diagnostic, not clean timings.
`DF_TRACE` requests a quartet census and bypasses optional MD-J. Consequently
the diagnostic's generated-J cost is **not** the default J cost. Inclusive
regions are nested and must not be summed. Four sweeps account for real work;
the next large opportunity is their pair algorithm/work, not another tile sweep.

The FMA hypothesis was checked against the **actual compile command**:
`vv10_runtime_cuda.cu` does not disable implicit FMA. Original and explicit-FMA
feature kernels have identical 144/136-register SM80/SM120 frames and no local
memory. Geometry has 170/170 and 172/176 registers. Explicitly spelling already
contracted arithmetic does not establish reduced GPU work.

## Independent gates and limitations

`qualification.json.xz` retains 209 host passes, two explicit skips, 206 actual
device admission/array cases and 41 native lifetime/automatic/reuse cases.
Initial missing-`PYTHONPATH` collection errors and the obsolete header-global
FMA prohibition failure remain separately recorded. Production/test changes
from that experiment are reverted, rather than weakening current required CI.

All 32 retained-reference observable pairings pass. The pinned independent
GPU4PySCF data are reused **only** for static accuracy; no reference timing,
new GPU4PySCF ranking, original 1.576 s-gap closure, cold distribution or new
sanitizer qualification is claimed.

The separate `compact-probe.json.xz` preserves a bounded two-versus-four-row
diagnostic at 320000 points, not a molecular endpoint. Features 1.953467 ->
1.951296 s and geometry 2.614718 -> 2.611765 s do not establish meaningful benefit.
No second full A100/5090 cohort was spent on that schedule; it too is rejected.
Its first launch lacked the CUDA library path and failed before timing samples;
the corrected scheduled launch and original failure description remain explicit.

## Replay and retention

```bash
python benchmarks/results/vv10-packet-fp64-fma-20261106/replay.py
```

Stdlib-only replay authenticates every retained member, all24 complete native
samples, settings/work/identity pairs, numerical gates and 32 static-reference
comparisons. `build.json.xz` retains the actual VV10 compile command, recipe,
cache stats, source inventory and producers. Its complete verbose build log
remains in ignored local artifacts, authenticated by byte length and checksum;
repetitive unrelated compiler invocations are not republished. No endpoint
timing or observable sample is removed. Source replacement used `tar --touch`; both
devices passed allocation-free native policy2 checks before GPU endpoints.

The evidence budget is unchanged. No Release, tag, external backup or assets
are published. See issue #2976 and the two rejected Agent Notes for rationale.
