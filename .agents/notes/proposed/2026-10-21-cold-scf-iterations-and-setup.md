# Proposal: separate cold SCF closure, initial-guess admission and source setup

Status: proposed
Date: 2026-10-21

## Problem and measured source

The remaining cold deficits cannot be inferred from warm timings or nominal SCF
iterations alone. On #2664 source `ef5edc80e1fdb8eaa61e86c69ddf13d4e7b5c647`,
the native core SHA-256 is
`f95ed636f503ab0f8e1db55c1beb2157c8b4896098b02b6b62a71c97eb96f16d`.
All 5,852 frozen runtime inputs were reverified before this experiment. No
production source, grids, force response or precision acceptance gates changed.

The existing clean Direct/96 receipts show these target SCF iterations:

| Method | Standard | Fast | GPU4PySCF default |
| --- | ---: | ---: | ---: |
| HF | 14 | 14 | 10 |
| PBE | 15 | 9 | 11 |
| PBE0 | 15 | 9 | 9 |
| B3LYP | 16 | 9 | 10 |
| r2SCAN | 14 | 9 | 10 |
| r2SCAN-3c | 22 | 14 | 11 |
| M06-2X | 16 | 10 | 10 |
| WB97M-V | 28 | 16 | 10 |

These counts exclude the native atomic-HF preparation of a Huckel seed.
For ordinary def2-SVP/96 this preparation takes about 1.25 s and seven atomic
iterations; it must not be counted as seven full molecular Fock applications.
WB97M-V counts also include provisional nonlocal operator work and therefore do
not all represent fresh full-VV10 applications. Fast precision solve receipts
retain its strict-tail and post-SCF work separately.

## Controlled experiments

Seventeen diagnostic populations (one complete cold plus one frozen-own-density
warm each) ran through finite Slurm allocations on n1/node1 RTX 5090. All 34 E/F
calls completed and passed the unchanged independent README gates, including
reference grid stability and CPU/GPU cross-checks. Tracing is enabled, so these
are causal diagnostics rather than new README timing populations.

### Standard closure dominates the last iterations

PBE/96 first meets its energy and density gates at iteration 11, but finishes
at 15. PBE0 similarly goes from 11 to 15; B3LYP and M06-2X from 11 to 16;
r2SCAN-3c from 15 to 22. Physical closure additionally requires both RMS and
maximum AO commutator residuals below `min(1e-9, density_tolerance)`.

The history field named `physical_residual_max` currently exposes
`Scalars::residual`, which is the maximum over spin-channel RMS values. The
separate AO-element maximum is enforced but is not this history field. Do not
infer that the unreported maximum passed from a small historical RMS value.

Changing only PBE energy/density controls from `1e-10/1e-8` to either
`1e-9/1e-8` or `1e-7/1e-6` still takes **15** iterations. Cold observations are
27.547, 27.263 and 27.321 s. Thus relaxing just those exposed tolerances does
not remove the fixed residual tail. GPU4PySCF uses energy `1e-9` and a resolved
orbital-gradient tolerance of approximately `3.16e-5`; its gradient and the
native AO commutator have different norms and cannot be equated numerically.

### r2SCAN-3c/96 falls back before constructing its seed

The Huckel numeric admission estimate is 321,801,344 bytes, exceeding the
default 268,435,456-byte cap. The receipt reports `budget_skipped`, zero
preliminary iterations and one target attempt: this is a core-guess fallback,
not a failed seeded solve and retry.

| Direct r2SCAN-3c/96 diagnostic | Target iterations | Cold seconds | Seed seconds |
| --- | ---: | ---: | ---: |
| Standard default | 22 | 78.264 | 0.000012 |
| Standard Huckel, explicit 512 MiB cap | 16 | 67.622 | 2.995 |
| Standard MINAO, explicit 512 MiB cap | 17 | 74.808 | 8.094 |
| Fast default | 14 | 51.876 | 0.000011 |
| Fast Huckel, explicit 512 MiB cap | 10 | 48.564 | 2.993 |
| Fast MINAO, explicit 512 MiB cap | 10 | 54.699 | 8.071 |

The larger cap is diagnostic, not a new default. Revisit the conservative
shape allowance against actual lifetimes before increasing retained resource
limits. Preserve explicit cap refusal and account for the complete CPU/GPU
preparation owner. Replacing Huckel globally with MINAO is not supported:
PBE/96 increases from 15 to 17 target iterations and 27.547 to 31.226 s;
PBE0 increases from 15 to 16 iterations and 41.591 to 45.938 s.

### Fast hybrid cache construction has a substantial, distinct cost

Fast PBE0/96 has nine target iterations, equal to the reference count, yet
cold remains 37.912 s versus the existing GPU4PySCF observation of 19.869 s.
It generates 1,476,473,708 FP32 integral values and rebuilds the same number in
FP64. `ks_begin` takes 4.635 s; the precision-handoff `ks_finish` takes 8.130 s
(8.122 s in the exact source preparation scope). The strict source then serves
two target Fock actions. These nested intervals must not be summed with their
enclosing SCF regions as independent costs.

Disabling all residency is a rejected shortcut. Complete Fast cold becomes
60.324 s for PBE0, 61.108 s for B3LYP and 68.248 s for M06-2X, with the same
9/9/10 target iterations and passing accuracy. The mixed source is useful.
Separately test whether a generated strict provider should handle the short
tail instead of constructing a new strict quartic cache. That proposal was not measured in this diagnostic campaign. A subsequent
implementation and its separate qualification are recorded in
[the strict-tail decision](../implemented/performance/2026-10-21-fast-hybrid-strict-tail.md).

## Proposed follow-up and invariants

- Qualify any revised stationarity policy against common independent energy
  and force targets, displaced geometries and failure behavior. Do not remove
  the final-state validator or relabel unlike residual definitions.
- Retain native initialization; make seed budget admission reflect complete
  resource lifetimes and keep the bounded core fallback.
- Evaluate mixed-source preparation and strict-tail preparation separately.
  Keep explicit residency controls and unsupported-domain fallbacks.
- Large source-only DF still regenerates raw three-center integrals per
  iteration. Fewer iterations cannot by themselves eliminate that per-iteration
  bottleneck; preserve it as a separate optimization task.

## Retained evidence

Local `.artifacts/cold-scf-diagnosis-v1/` and remote
`n1:/data/jzzeng/scf-performance-20261021/cold-scf-diagnosis-v1/` retain launchers,
versioned diagnostic script bytes, source manifest, per-job outcomes, complete
receipts, component traces, all-method baseline counts and independent gates.
Existing GPU4PySCF receipts remain source-bound in
`benchmarks/results/readme-repaired-20261021/gpu4pyscf.json.xz` from #2662.
The preceding shared-provider and pure-J residency repair is documented in
`../implemented/performance/2026-10-21-scf-capacity-and-pure-j-residency.md`.
No README figure or default configuration is replaced by these ablations.
