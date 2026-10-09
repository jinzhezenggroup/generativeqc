# Unified K selector and task Work qualification

Frozen base: `6acb3e70e3031b4a7ef44ef77df551b354e34e59`, after merged #2135.
Integrates #2133 `76ade1377531cbfe953d8462578c29272dd0c175` and preserves the
reviewed #2135 `e903d467a62b8aa4ace2ab6513912033e571f436` retention checks.
The typed prepared selector alone did not qualify: the initial union's 96-atom
Cold median regressed. The final source adds warp-private Work companions for
the existing task producers, without changing their quartet arithmetic.

Production source identity:
`cb788b62c65e98f375f29a3fbbf35cf58c61a0c329461d4667a51c7794d6eb7e`.
Library SHA256:
`1426e40973b32359dca90d0cfa106eb62d1052eddb07af50ce12c4b86f079376`.
RTX 5090, node4, CUDA 12.9.1/GCC 12, verified ccache 4.5.1. Finite exclusive-node
Slurm job 666 supplies scientific/sanitizer gates; job 668 measures all final
performance policies with the same frozen binary. No overlapping compilation
or allocation is included in clean measurements.

## Complete Cold E+F

Strict FP64 PBE0 RKS, spherical def2-SVP water clusters, 48x16x32 grid,
screening 1e-12, E/D tolerances 1e-12/1e-10, maximum 100 iterations.
Timing includes calculator construction, CUDA context/preparation, first SCF,
analytic forces, host return and teardown. Workers have no supplied density,
priming or warm owner; imports/input decoding are excluded. Persistent compiler
and artifact caches are equally reused: this is execution-Cold, not cache-empty
compilation timing. The separate holdout preflight does not prime worker contexts.

| Policy | Lowering | Task schedule |
| --- | --- | --- |
| Task + Fill (#2133 policy) | unset | `fill` |
| Incumbent + Work (landed #2135) | `incumbent` | `work` |
| Combined | unset | `work` |

The candidate is the actual unset lowering preference, not the explicit
twelve-class experiment. Preference remains nine classes only on sm_120.
Five fresh workers per policy at 96 atoms, three at 48. Predeclared round orders:
`task-fill,work-only,combined`; `combined,work-only,task-fill`;
`work-only,combined,task-fill`; `task-fill,combined,work-only`;
`combined,task-fill,work-only`. The 48-atom study uses the first three rounds.

| Atoms / AOs | Task + Fill median | Incumbent + Work median | Combined median | Less time vs Fill / Work |
| --- | ---: | ---: | ---: | ---: |
| 96 / 768 | 101.927677 s | 110.618048 s | 97.393406 s | 4.45% / 11.96% |
| 48 / 384 | 47.464295 s | 45.116708 s | 42.179566 s | 11.13% / 6.51% |

Mean reductions: 96 atoms 4.30%/7.65%, 48 atoms 10.23%/7.73%.
All 24 valid observations remain included; both mean and median beat both
comparators. Maximum independent GPU4PySCF errors are 9.095e-12 Ha and
3.785e-11 Ha/Bohr, against unchanged 1e-8/1e-7 acceptance gates.

Actual Fock counts (not same-iteration kernel ratios):
- 96, Fill / Work / Combined: `[17,19,17,18,17]` / `[19,18,19,17,19]` / `[19,18,17,17,17]`.
- 48, Fill / Work / Combined: `[19,20,20]` / `[19,19,20]` / `[19,19,19]`.

`summary.json.xz` retains every receipt's timing/work/gates, checksums, effective controls, protocol,
loaded binary, job/device provenance, cold flags, convergence, positive Fock
counts and every numerical gate. Its deliberate joint-policy reducer requires
exact 5/3 populations and gains in both mean and median against both standalone
policies. The existing single-axis benchmark comparator still rejects unmatched
task schedules. This small population is not statistical significance or a
universal-workload performance claim.
`final-receipts.json.xz` supplies the curated, reviewable numerical fixture with
all Cold and holdout times, work counts, energies and full force vectors;
`initial-union-receipts.json.xz` retains the rejected union's numerical observations.
Full run bundles remain ignored local artifacts under the repository's storage
policy, with exact retained locations, sizes and checksums in `ignored-artifacts.json`.

The independent reference is an E/F accuracy oracle, not a backend timing or
equal-residual stopping-policy comparison. Historical reference diagnostics
retain their original schema; GPU4PySCF Frobenius/RMS and GenerativeQC density
stopping rules must not be relabeled as equivalent (see the later #2150 contract).

## Warm and moved holdouts

The same binary/job runs three repeats with fixed post-cold/moved seeds at 48
atoms, preserving both original and explicitly resupplied moved coordinates.
Times include synchronized SCF, analytic forces and host output; preparation is
reported separately. First-sanity and moved-first calls are not Cold samples.

| Phase | Task + Fill | Incumbent + Work | Combined | Fock builds |
| --- | ---: | ---: | ---: | ---: |
| Warm median | 6.298346 s | 6.253223 s | 6.107105 s | 1/1/1 per repeat |
| Moved first | 31.673393 s | 31.269111 s | 29.323675 s | 13/13/13 |
| Moved-warm median | 6.355931 s | 6.318093 s | 6.156997 s | 1/1/1 per repeat |

All 24 holdout rows converge and pass independent original/moving-grid GPU4PySCF
E/F gates, including three first-sanity calls (19 Focks each). Full force vectors,
reference hashes, warm-state flags, work counts and harness identity are retained.

## Complete prepared-K consumer and source retention

Six fresh processes, two per policy in forward/reverse order; three wall repeats
per density scale. Time includes input upload, transformation, K, projection,
export and synchronization, but excludes plan setup. Full-density wall medians:

| AOs | Task + Fill | Incumbent + Work | Combined | Less vs faster standalone |
| --- | ---: | ---: | ---: | ---: |
| 384 | 0.770219 s | 0.729234 s | 0.584464 s | 19.85% |
| 768 | 1.328519 s | 1.367599 s | 1.079546 s | 18.74% |

All six full generated admission vectors agree at density scales 1, 1e-3, 1e-6
and 1e-14. Full-density counts are 23,480,495 / 81,907,624. These are generated
shell-quartet admissions, not primitive/root counts or proof of equal native
fallback work. Instrumented censuses and CUDA-event probes run outside clean
wall timing; no profile time is added to Cold. No active-lane/barrier contribution
is claimed without hardware counters.

Independent emission retains all 21 incumbent/work streams, twelve original
task shards with validation-only Work omission, component-Rys/block leaves and
six retained scientific-source hashes. Deliberate bundle changes affect only
four sm_120 shards and registry source. Portable shards and the generated header
retain their hashes. The final queue has four independently retiring warps with
eight two-batch bins each and bounded 24,736-byte shared storage. Fill retains its
original smaller-scratch kernel; no new device allocation is introduced.

## Scientific gates and retained negatives

- 320 public/prepared K matrices against independent PySCF/Libcint pass both
  spins, symmetric/asymmetric densities, four scales and frozen selector mutation;
  maximum matrix error 6.072e-15.
- 864 standalone matrices execute the actual production task bundle against
  Libcint. Each memcheck/racecheck/synccheck runs 96 standalone and 16 integrated
  cases, with zero errors/hazards/warnings.
- Native through-f values/derivatives and Cartesian order-two CPU finite
  differences pass. A pre-existing generated-J counter assertion needs
  command-scoped `GENERATIVEQC_DISABLE_MD_J=1`: default MD-J, not a generated J
  leaf, owns J. Every initial-union policy reproduces the same census failure.
  All clean endpoints keep MD-J enabled; no unrelated native test is changed.
- Host sets: 635 passed/1314 resource skips, 178 focused passes (overlapping),
  and 31 independent CPU Libcint passes on current source. Ownership, inventory,
  source retention, formatting and whitespace guards pass.
- `prior-union-negative-evidence.tar.gz` retains all 24 scientifically accurate
  initial-union Cold observations: 96-atom median 102.027941 s versus
  100.433881/101.457031 s, despite a winning mean; 48-atom gains did not authorize
  promotion. It includes source/protocol identity, baseline census failures and
  the old failed moved-warm harness. That harness reverted to original geometry
  with `coordinates=None` while comparing to a moved oracle; the corrected
  harness resupplies moved coordinates each time. The failure is not relabeled
  as a production K error or a passing observation.
- Non-exclusive node1 direction-finding K measurements are distinct diagnostic
  evidence, not final timing. A transferred node1 binary failed on node4's older
  glibc before execution; the qualified node4 binary is a native GCC-12 rebuild.
  A missing preflight reference was corrected before any final timing cohort.
  Primitive sorting's retained negative does not justify a default change.

## Reproduction

Use the same CUDA/host-compatible toolchain and ccache, build `generativeqc`,
the native provider test and adjacent stationary PBE0 manifests. All real GPU
commands must use finite Slurm allocations and preserve assigned visibility.
The reproduction capsule supplies the exact source patch against the frozen
base, build/environment drivers, K probe, input fixtures, all qualification and
holdout scripts, deliberate joint reducer and library/source identities.

Extract the retained archives, inspect the environment paths and run the supplied
finite exclusive-node driver on an allowed RTX 5090 node. Restore neither GPU
visibility nor a failed/older library as a fallback. Reduce only the final same-
binary job's 24 Cold receipts; do not pool historical or exploratory revisions.
After extraction, run `python reproduce/summarize_joint.py observations --compact` to
reproduce the decoded `summary.json.xz`; this CPU-only reducer exits nonzero for a failed gate.
The original independent Cold references remain pinned in
`../rys-task-cold-20261008/reference-{48,96}.json`; a moved moving-grid/response
GPU4PySCF reference is retained separately with its generator and full force vector.
The input capsule retains the exact used density/exchange arrays; unused Coulomb
and moved-K arrays remain in ignored original inputs rather than inflating Git.

Full archives are deliberately not reintroduced into Git or uploaded to a Release.
The curated fixtures and source-identified summary are the reviewed Git evidence.

See the [decision note](../../../.agents/notes/implemented/performance/2026-10-09-rys-task-work-buckets.md)
for invariants, discarded approaches and revisit conditions. No release, release
tag, publishing workflow, merge or universal speedup is requested or implied.

## Lossless storage and historical recovery

The summary and two curated receipt files are losslessly XZ-compressed. For example,
`xz -dc benchmarks/results/unified-k-work-20261009/final-receipts.json.xz`
prints every original JSON byte, including all positive and negative numerical
observations. `storage-recovery.json` records compressed and decoded sizes and
SHA256 values, copyable decoding commands, and exact Git recovery commands with
complete member inventories for the six historical archives. The pre-existing
`local-autotune-136/evidence.json` is also losslessly stored as `.json.xz`; its
decoded bytes are identical. Global evidence budgets remain unchanged.

For a shallow or fresh clone, first fetch the exact recovery commit:
`git fetch https://github.com/jinzhezenggroup/generativeqc.git dee3d522b71df9fb55c91478df3c44b7af454488`.
Then run the manifest's `git show` recovery command from the repository root.
