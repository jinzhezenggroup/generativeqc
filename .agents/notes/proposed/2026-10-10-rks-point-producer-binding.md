# Proposal: bind the qualified PBE0 point to its actual grid producer

Status: historical default-off integration; superseded by the default-on promotion
Date: 2026-10-10

The subsequent production default and master-aligned qualification are recorded
in `../implemented/performance/2026-10-10-producer-bound-pbe0-force-point.md`.
This note retains the distinct initial default-off source and failed attempts.

## Decision

GPU4PySCF's RKS route specializes its scalar variables before XC evaluation.
GenerativeQC already projects an owned restricted density only once, then copies
the ordered panel for the existing two-spin consumers. The remaining target is
the full-spin point differential, not another duplicate-GEMM removal.

Reuse the independently qualified numerical core from
`2026-10-10-pbe0-bound-subnormal-exchange.md`. At initial integration the header had its
exact SHA256 `f9ddf9e8ee0c95680735bc812fe6b20e6d488025d8b7f4a4b44155bf4fd3a204`;
existing general/response arithmetic remains the eight-direction specialization.
No general PBE, UKS, response/HVP or complete-endpoint qualification is implied.

The compiler emits a mathematical capability only for explicitly unpolarized
PBE components with exact semilocal weights 3/4 exchange and 1 correlation. A
custom label cannot remove an otherwise valid capability, and a PBE0 label
cannot admit different components. Runtime policy remains in the production
stationary consumer, outside the generic compiler.

## Producer and ABI

`grid_cuda_density_jets_v2` returns an exact-generation jet pointer and bit-zero
proof only for the owned identical-spin density witness with rho AND gradient
features ready. Orbital, stale and incomplete panels are rejected; two independent
spin densities do not acquire a witness by having equal values. Version-one
`GridTaskView` layout and the original getter remain unchanged.

The optional stationary resident-weight enqueue v2 accepts that generation and
flag. Stale generations and unknown flags fail before any geometry enqueue.
Absent v2 artifacts or producers retain the checked v1 path; an unproven producer
may use v2 with zero flags but must still match the current view generation.

The bound and legacy precomputed point kernels are separate template
instantiations. The bound body has no full-spin numerical fallback or external
seed branch. Native admission also requires the compiled mathematical
capability, no external seed, phased scratch, and enough per-point atom storage.
Small-atom and nonphased routes preserve the existing bounded general consumer.
This avoids the guarded dual-body code bloat diagnosed in
`2026-10-10-rks-point-compile-time-binding.md`.

## Policy and evidence

`GENERATIVEQC_STATIONARY_PBE0_RESTRICTED_POINT=on` is an experiment request;
the default is off. Prepared schedule identity includes the request so an owner
cannot silently replay a different policy. New optional cumulative native
counts record actual bound/general point batches and points; endpoint snapshots
difference these counts instead of treating the request or model label as proof.

The emitted point kernel, not merely its scalar helper, passes all original 60
independent PBE0 points and 27 additional boundary/subnormal points at the
unchanged `5e-10*abs(reference)+1e-322` gate. The original general entry's failed
indices `[8,9,65,66,67,68]` remain asserted. The frozen 87-row fixture and offline
original-formula reproduction tool are retained; no failing row is removed.
Existing source/ABI/cooperative/precomputed CPU gates pass. Initial harness
failures from a stale loaded test module and incomplete PYTHONPATH are retained
in the task log and resolved by targeted reruns, not a repeated old GPU matrix.

The first synthetic GPU owner attempt, Slurm 7054, fails compilation before
execution because the probe omitted `--expt-relaxed-constexpr`. The production
compiler already supplies this flag. Its raw v1 source/logs remain under
`xc-restricted-owner-v1`; the corrected finite Slurm v2 attempt uses the production
flag and is a new evidence namespace. Its integral provider is deliberately a
rejecting stub: this module may qualify owner routing but must NEVER become an
endpoint benchmark or production artifact.

Latest inspected master is `8eaaa66b3` (#2193). Its CI/diagnostic changes do not
modify this PBE0 path; no existing matrix is repeated merely because it landed.
The retained native endpoint control is still the distinct #2185 snapshot plus
default-off vector scaffolding, not current master. All six changed native/
compiler source bases match that snapshot; its additional runtime differences
are fitted-only resource policy and must be preserved when applying this delta.

## Subsequent owner and reference qualification

Slurm 7055 compiles the corrected owner and unsupported-functional modules.
Four routing cases pass; two small-atom cases fail a harness assertion that a
requested phased plan must be granted. The planner may legitimately choose
bounded general storage. No production admission rule was weakened to fix this.

Slurm 7057 replays the exact 7055 binaries with the corrected six-case harness:
atoms 3, 4 and 48, each with phased storage requested off/on. Ordinary execution,
memcheck and initcheck each pass all six cases; both sanitizers report zero
errors. Source parity errors are exactly zero. The admitted 48-atom case with
producer proof executes five restricted batches/144 points and no general
points; legacy API, zero proof and nonphased routes execute the general entry.
Stale generations and unknown flags reject without advancing point counters.
Source and replay binary identities remain unchanged after execution.

The entire 7057 shell still exits 1: its final, CPU-only reference command used
module invocation against a different regular `tools` package. Preserve that
failure separately from the successful GPU gates. CPU-only Slurm 7060 runs the
reference generator by file path and exits 0. All 87 fixtures agree at 450/550
digits, bit-identically to the frozen rows. Fixture SHA256 is
`bc90c951594bd8dc710d406341d0896587b960da4324a8beaa1f8318a4c5bcf6`;
original-formula SHA256 is
`f339d95dd40e18f4573cee4d6bef723b7516b8d64873831a64c3c125182c3e0e`.

Within the same synthetic owner module, general/bound point registers are
142/110; both have zero stack/local bytes and 1024 shared bytes. This is static
owner evidence, not timing. In particular, do not compare these register counts
to the old official AOT compiled with a different integral provider/context.
The synthetic rejecting integral provider remains forbidden for endpoints.

Latest inspected master is `4444d0376` (#2192). Its token-checked native
single-system integral-source bridge returns H'/Pulay/J'/K' only; it neither
changes this Python stationary point consumer nor supplies XC/Becke/nuclear
assembly. The preceding #2198 acceptance tests and #2197 correlated DF tile cap
also leave this PBE0/direct force path unchanged. No old GPU matrix is repeated.
The retained #2185 execution snapshot is still distinct from this master and
must be labeled accordingly in the eventual matched endpoint comparison.

Raw owner v1/v2/v3 and CPU reference evidence is copied into ignored
`.artifacts/xc-restricted-stage/evidence/`. This historical qualification does
not assert that official native/grid/AOT artifacts were rebuilt or timed.

## Subsequent official build

An isolated `candidate-rks-point-binding-v1` source snapshot applies only the
seven-file production delta to the qualified control. The runtime's retained
reserve/tile differences remain fitted-only; direct/noDF policy is unchanged.
The patch SHA256 is
`257f649f09fec969e7c8ce5ba7326ac787348d76a10c4cdf99b75a54c38d5e36`.

CPU-only Slurm 7066 stops before compilation: a provenance harness looked for
a nonexistent standalone `cuda_grid.cu.o`. The native grid implementation is
included in generated `generated_grid_policy.cu`. Preserve build v1 as exit 1.
CPU-only Slurm 7067 checks the actual generated grid and PBE0 s/p/d compiler
commands, both invoking verified ccache 4.5.1, and completes the official native
library and stationary AOT build, exit 0. Cache statistics and source checks
before/after are retained; no shared cache is cleared.

Official native SHA256:
`2fe65c91d2d8925595674c6893595c77ac5584be30fd1153de2fc1548eb9cbcd`.
Official PBE0/RKS s/p/d AOT SHA256:
`e1b497f1e69dd5dedfbe598ceec942a69e233f6f192a44b068c9e38e1840dfcc`.
Within that same official module, general/bound point resources are 254/148
registers, 96/0 stack bytes, and zero local/shared bytes. This is now a matched
module comparison, but still not endpoint timing or a promotion gate.

The first endpoint driver (Slurm 7069) stops at setup because its assertion
mistakenly expects the stationary artifact label `native-build-aot`; the
stationary loader labels a verified official library `packaged-aot`, while the
native grid uses `native-build-aot`. No production code or binary is changed.
The corrected v2 driver reuses the same official artifacts and retains actual
selection checks. No endpoint improvement is implied by these failed attempts.

## Matched complete endpoint qualification

Slurm 7070, endpoint v2, completes normally on node1/RTX 5090 with visibility
0 preserved. It uses the official source and binary identities above, not the
rejecting owner probe. Five interleaved measurements per arm and geometry retain
complete PBE0/RKS spherical def2-SVP direct/noDF FP64 energy, moving-grid analytic
force assembly and host result return. Profiling is off. Each measured call has
one converged SCF iteration/one Fock build, actual native XC evaluation and
identical AO/force semantic work between arms.

| Atoms | Geometry replay | Off median (s) | On median (s) | Endpoint reduction |
| --- | --- | ---: | ---: | ---: |
| 48 | warm | 5.347924 | 5.065475 | 5.281% |
| 48 | moved-warm | 5.294727 | 5.066697 | 4.307% |
| 96 | warm | 16.035795 | 15.487671 | 3.418% |
| 96 | moved-warm | 15.978094 | 15.487917 | 3.068% |

All four comparisons pass the existing >2%-and-robust-noise gate. Each 48-atom
call executes 2304 restricted batches/1,179,648 points in the on arm, versus the
same general work in the off arm. At 96 atoms these counts are 4608/2,359,296.
The on arm never executes a general point in these admitted cases. Becke pair
state evaluations remain 1,330,642,944 and 10,758,389,760 respectively: the gain
is not achieved by silently reducing grid, partition or SCF work.

Independent GPU4PySCF references retain the original 1e-8 Eh/1e-7 Eh/Bohr gates.
Maximum E/F errors across setup, priming and measurements are 6.8213e-12 Eh /
2.4653e-11 Eh/Bohr (48) and 1.0460e-11 Eh / 3.7532e-11 Eh/Bohr (96).
Frozen native seeds are bit-identical before/after each arm, but NOT identical
between independently converged arms. Coordinate differences are zero; maximum
density differences are 8.7764e-14 / 6.1951e-14 (48 warm/moved-warm) and
9.9810e-14 / 4.2242e-11 (96). Do not label these identical-density ablations.
Do not promote cold or moving-geometry reconvergence bootstrap timings to a
performance claim; the clean timed populations are warm and moved-warm only.

Raw JSON/immutable-seed checkpoints and hashes are retained in ignored endpoint
v2 evidence, with a separate offline verifier. Master was fetched again at the
official-build/endpoint boundary and remains `4444d0376`. These measurements
still describe the labeled #2185 control plus this delta, not master.

## Remaining gates

The owner, reference, official-build and matched off/on endpoint gates are now
complete. Slurm 7078 also completes the separately scheduled default-off control
guard, exit 0. Relative updated-off versus retained-control medians are +0.415%
and +0.085% (48 warm/moved-warm), -0.123% and -7.347% (96). No default-off
slowdown exceeding 2% is observed. This allocation uses visibility 2, versus 0
in the matched comparison: these non-interleaved, different-device differences
are descriptive regression observations, NOT additional acceleration evidence.
Only the same-binary interleaved 7070 populations support the gain table above.

Reconcile the submission source with current master before promotion/PR. Among
the seven production delta files, master differs from local HEAD only in the
runtime's fitted reserve/tile policy and four stationary-emitter residency
hook/include lines; the shared point/grid/geometry numerical bases are unchanged.
Do not discard those upstream policies/hooks or submit this stale branch's
unrelated removals. No commit, push or PR is made in this qualification stage.
The final master fetch remains `4444d0376`; no unrelated GPU matrix is repeated.
Keep acceptance gates at 1e-8 Eh and 1e-7 Eh/Bohr and retain bounded fallbacks.
