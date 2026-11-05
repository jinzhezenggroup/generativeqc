# Decision: opt in guarded dense-Direct warm-anchor deferral

Status: implemented opt-in; matched A100/RTX 5090 qualification complete
Date: 2026-11-05

## Problem

The retained A100 water96 full-Direct WB97M-V energy/force endpoint spends
13.210521 seconds rebuilding the incumbent warm-energy baseline after physical
SCF convergence. The accepted immutable range source can already serve warm
replay, but the deferral predicate admits only fitted J/K owners with an exact
occupied factor. Direct owners instead freeze their accepted dense density.

The existing rebasing fallback already restores that density and its own XC
scalars before evaluating the incumbent provider. It does not require a fitted
factor. The opportunity is avoiding an unused provider schedule, not changing
the physical SCF or treating two summation schedules as numerically identical.

## Decision

Extend explicit `GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY=1` to full-Direct RKS.
Keep unset Direct owners eager, fitted defaults unchanged and partially fitted
owners excluded. All existing strict-FP64, device-fused XC, source-policy,
identity/lifetime, and following-force-memory guards remain authoritative.

The retained source's own energy and compensation certify its frozen density.
If that source is lost, restore that same density and its saved XC values and
rebuild the incumbent anchor as counted preparation work. Never revoke the
current stationary final-state lease during optional cache eviction.

## Rejected alternatives

- Deleting the incumbent fallback or comparing different summation schedules at
  sub-ulp energy gates would change numerical acceptance.
- Unconditional deferral can merely transfer cold work to the first warm call.
  Preserve the force-envelope guard and inspect each warm call, not only a median.
- GPU-name dispatch is unnecessary: eligibility follows the owner, precision,
  immutable source, and actual available memory.
- Default promotion before matched endpoint and independent science evidence is
  not justified. This proposal changes only explicit opt-in behavior.

## Qualification contract and results

Use current-master-based, source-authenticated SM80/SM120 Release binaries with
fast-compile off and verified compiler caching. Compare one matched cold and
three-warm lifecycle per policy on actual scheduled A100 and RTX 5090 devices.
Retain public work ledgers and diagnostic traces, including failures. Do not
reuse the old measured image as qualification of current master.

Independent same-grid GPU4PySCF small-system full-Direct E/F gates cover eager,
opt-in, and unset owners, frozen/updating warm seeds, retained replay, policy and
capacity invalidation, and force eviction. Preserve fitted-owner coverage.
Matched large endpoints retain complete construction/preparation/SCF/force/host
export timing, common independent observable gates and actual Fock/VV10 work.
One diagnostic population is not a statistical default-promotion result.

The measured source identity is
`a5e1f58ffe3b605a2396e334a5c3206995f4cdb5a3cce71166592ab154d71cca`;
core ELF SHA256 is
`4fafc4ac3f0ea533673f371c2bd4ff116243a022a6cd592a2bb01a3bd0445433`.
The same SM80/SM120 Release bytes execute on both devices. No old repair image
is substituted for the current-master candidate. The native endpoint rejects
PySCF/GPU4PySCF imports.

Actual allocations: ECNU LSF 225523, `gpu_chem/gpu22`, eight CPU slots, one
exclusive-process A100, finite 35 minutes; RTX 5090 Slurm 15954, `main/node1`,
one GPU/eight CPUs, finite 25 minutes. Assigned visibility is preserved.

| Device | Eager cold (s) | Opt-in cold (s) | Reduction | Cold/three-warm reduction |
| --- | ---: | ---: | ---: | ---: |
| A100-PCIE-40GB | 131.754409 | 119.604375 | 9.222% | 6.152% |
| RTX 5090 | 218.958945 | 202.866657 | 7.349% | 5.700% |

A100 warm opt-in/eager ratios are 1.000116, 0.999896 and 1.002389; RTX 5090
ratios are 1.008376, 1.007512 and 1.002172. All twelve native warm observations
remain one iteration/one Fock; cold iterations remain 18 and Focks decrease
19 to 18, J applications 19 to 18 and K applications 38 to 36. VV10 work stays
13 provisional plus five target Focks/four sweeps cold and zero sweeps warm.
No threshold, force tile, precision or Direct Hamiltonian changes.

The fresh independent A100 GPU4PySCF reference takes 88.380248/28.426633 seconds
cold/warm, with ten/one public iterations. All 32 native/reference cold/warm
pairs pass common gates (energy/atom `1e-5` Ha, maximum force `1e-4` Ha/Bohr,
force RMS `3e-5` Ha/Bohr). Worst errors are `7.13858601860314e-12` Ha/atom,
`3.5521267438554105e-7` maximum force and `1.55706603143627e-7` force RMS.
Native paired policy differences are below `1.90e-14` Ha/atom and `1.04e-12`
Ha/Bohr maximum force, passing the predeclared tighter policy-parity gates.

The large oracle is stock default, not tight/grid-extrapolated science. Both
use declared level-3 rules and moving-grid response; reported point counts are
1,078,336 native and 1,078,528 reference, not identical reported arrays. Native
Huckel and reference MINAO initial guesses differ; the resolved physical MO
target is `sqrt(1e-9)`. Public density tolerance is not the MO-gradient norm.
Reference effective-potential calls are not native J/K counters.

171 focused host lifetime/admission/failure checks pass. Each device has ten
targeted small science/lifetime controls, including two fitted controls and two
default/explicit-strict moved-geometry controls. Direct same-grid E/F limits
remain `2e-8` Ha absolute energy and `2e-7` Ha/Bohr maximum force.

This is one progress-instrumented population, not statistical proof of zero
warm regression. The 5090 observations are 0.22--0.84% longer warm, inside the
predeclared 1.05 per-call guard. **Do not promote unset Direct defaults** from
this evidence. The optimized A100 cold endpoint is still about 31.22 seconds
behind the independent reference; source setup and SCF/VV10 remain targets,
not delivered improvements.

## Retained failures and bounded execution

The initial current-master dependency build reaches its 40-minute CPU cap. Its
foreground timeout leaves a predecessor Ninja alive. A briefly overlapping
incremental attempt fails when the linker reopens a missing generated-grid
object. The owned predecessor/private worker context is identified and reaped,
without touching an unrelated build; single-owner incremental finalization
succeeds and authenticates the complete loaded image. Partial builds are never
timing/science populations. Compiler-cache versions, commands and statistics
are retained; caches are not cleared.

5090 Slurm 15952 stops before large timing: five small cases pass, three fail a
new test's mistaken assumption that eager updating warm seeds have one Fock.
An eager owner publishing a new seed legitimately adds a post-SCF anchor Fock.
Correct only that work assertion and check the explicit post-SCF counter. The
three failed cases pass a narrow retry in 15954, alongside two default/moved
controls; successful earlier small cases and large populations are not rerun.
Science limits and production bytes do not change for this correction.

## Reproducibility

The compact [publication and CPU replay](../../../../benchmarks/results/direct-range-warm-anchor-20261105/README.md)
retain every large E/F/work observation, common inputs, actual device/scheduler
identity, source/core identities and receipt hashes. Ignored task-local raw
archives/traces are under `.artifacts/cold-direct-20261105/`. Remote receipts
remain in `/data/jzzeng/direct-range-anchor-20261105/` on n1 and
`/tmp/generativeqc-ecnu-a100-tzhu-e-20261029/direct-range-anchor-20261105/`
on gpu22; login02 retains the uploaded `results-225523.tar.gz`.

## References

- Engineering memory: issue #2976; implementation is separate from evidence-only
  PR #3002.
- Existing fitted policy and the rejected first-warm regression:
  [guarded default anchor](2026-11-01-default-range-warm-anchor.md).
- Original measured opportunity: A100 LSF job 225489, retained raw trace and
  audit under task-local `.artifacts/issue2987/`.
