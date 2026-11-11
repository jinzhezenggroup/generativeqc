# Decision: keep the erf DF metric on the primitive schedule

Status: implemented
Date: 2026-11-11

## Problem

On H100/SM90, density-fitted ωB97M-V failed every independent gate in the
README 64-pair cohort on source `5a1ae34ad`. The native energy was about 4e-4
Eh/atom from the dense oracle at water 3/12/48/96. GPU4PySCF on the same GPU
qualified. The same source qualified on RTX 5090 (SM120) and earlier on A100 (SM80).

The failure is deterministic and isolated to the fitted long-range exchange.
`GENERATIVEQC_DF_RANGE_EXCHANGE=direct` (exact LR) matches GPU4PySCF's DF−Direct
shift exactly. Default `auto` and explicit `df` both fail.

A standalone probe of the CUDA DF integral source localized the error. Water-3
was compared with def2-SVP and the frozen ωB97M-V default auxiliary set
(naux 113, through g). The reference is PySCF/libcint
`with_range_coulomb(0.3)`:

| Tensor | Mapping | Max error |
| --- | --- | ---: |
| Coulomb metric and A (ω=0) | auxiliary, component, primitive | 6e-13 |
| erf A[μν,P] (ω=0.3) | auxiliary, component, primitive | 3e-14 |
| erf metric M[P,Q] (ω=0.3) | auxiliary (default raw) | 7.9 (39 of 12769 entries) |
| erf metric M[P,Q] (ω=0.3) | component | 1.1 (41 entries) |
| erf metric M[P,Q] (ω=0.3) | primitive | 4e-13 |

The kernel code and recurrence are identical across mappings. Only lane ownership
changes: one thread per output versus a warp cooperating on one output.

Only `df_source_kernels.cu` was recompiled, and the library was relinked:

| ptxas level | erf metric |
| --- | --- |
| `-O3` | 39/41 bad entries, reproduced exactly |
| `-O1` | 39/41 bad entries, reproduced exactly |
| `-O0` | exact, 4e-13 |

The following did not change the result:

- `CUDA_LAUNCH_BLOCKING`
- fresh JIT/driver caches
- a larger CUDA stack limit
- all other DF value-storage, exchange and projection controls

compute-sanitizer memcheck/initcheck report no errors. The values move under
memcheck, which is consistent with optimizer-dependent register/local reuse.
This is a CUDA 12.9 sm_90 code-generation failure of the single-lane erf
metric. It is not a resource-dependent route or a race.

#2929 (2026-11-02) moved the two-center metric from the transformed consumer's
mapping (automatic: primitive) to the raw consumer's mapping (automatic:
auxiliary). It was qualified on A100 and Blackwell only. H100 cohorts after it
pre-excluded WB97M-V DF, so the regression was not observed.

## Decision

A positive `df_range_omega` metric launch always uses the primitive (warp)
mapping. This is the schedule that every target used before #2929. The Coulomb
metric keeps #2929's raw-consumer schedule and explicit mapping controls.
Three-center values, derivatives and every other consumer are unchanged.

## Rejected alternatives

- **Compile the translation unit with ptxas `-O0`.** This slows every DF value
  kernel in the file and still depends on an unexplained compiler outcome.
- **Restrict the change to SM90.** The defect is optimizer-dependent. Keeping the
  previously qualified cooperative schedule for the small erf metric costs little
  on other targets, and avoids a target list.
- **Default fitted LR off (`direct`).** This discards a qualified faster default
  on unaffected targets and does not fix explicit `df`.

## Invariants

- The erf metric must match an independent libcint `with_range_coulomb` metric
  on SM90 under the automatic mapping.
- Full-range metric scheduling and explicit Coulomb mappings stay unchanged.
- No numerical gate, oracle or acceptance threshold is relaxed.

## Evidence

H100 80GB, driver 595.58.03, CUDA 12.9 build of `5a1ae34ad`, with only the patched
`df_source_setup.cpp` object relinked:

- The erf metric probe under automatic mapping is exact to 4e-13.
- ωB97M-V DF cold endpoints pass the unchanged gates against the tracked
  `readme-refresh-20261026` oracles at all four sizes. These errors match the
  RTX 5090 cohort:

  | Water | E/atom | Fmax | Frms |
  | ---: | ---: | ---: | ---: |
  | 3 | 4.7e-6 | 4.6e-6 | 2.0e-6 |
  | 12 | 3.7e-6 | 7.2e-6 | 3.1e-6 |
  | 48 | 3.0e-6 | 2.0e-5 | 9.4e-6 |
  | 96 | 3.0e-6 | 2.0e-5 | 9.4e-6 |

  The unpatched library still fails at water-12 in the same session, with
  E/atom 3.0e-4.
- ωB97M-V Direct and PBE0 DF energies are bitwise unchanged at 3 and 96 atoms,
  and still qualify.

## Revisit when

- A CUDA toolkit fixes the sm_90 code generation. Re-run the probe across all
  mappings before returning the erf metric to the raw schedule.
- A different erf metric implementation replaces the generic g-adapter path.

## References

- #2929 (metric schedule change), #2973 (fitted LR default).
- H100 README64 cohort on `5a1ae34ad` (PR #3067 source).
