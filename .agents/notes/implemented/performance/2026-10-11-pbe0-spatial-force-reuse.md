# Decision: default qualified PBE0 forces to primitive-local spatial reuse

Status: implemented; scoped automatic promotion, not broad method qualification
Date: 2026-10-11

## Problem

The fused all-center order-six primitive derivative already shares its radial
workspace, but its 64 pair-subset combinations repeatedly request the same
spatial Coulomb states: one base and three raised states each, or 256 scalar
recurrence evaluations. Changing the scheduling of those requests alone did
not establish a complete-endpoint benefit. The
[frozen prototype](../../proposed/2026-10-11-pbe0-spd-spatial-coulomb-reuse.md)
records the investigation and disjoint source-matched pilots.

## Decision

The integral compiler emits a primitive-local Cartesian state box and three
single-response faces using the existing scalar recurrence. At order six this
requires 22 to 54 evaluations, enclosed by at most 64 doubles. Multi-face corners
are neither initialized nor read. Radial moment validation and gradient
accumulation order are unchanged; this is reuse of existing mathematics, not a
second Coulomb algebra or a changed screening approximation.

Native execution has four cache-only spin/output variants of
`direct_force_spatial_spd_low_kernel`. Generic low, order-seven/eight and mixed-f
consumers remain. No retained quartet inventory, additional global traversal or
dense allocation is introduced. The bounded queue-page adapter retains
incumbent pair ordering and rejects inactive blocks before loading pair ranges.
The unmerged class-page, indexed-domain and uncached narrow-low selectors are
removed rather than shipped as an expanding collection of experiments.

The prepared force preference captures unset/empty as `Automatic`, `0` or
unknown strings as incumbent, and `1`/`spatial-reuse` as an explicit guarded
request. Automatic selection requires method-authenticated exact-direct,
full-range PBE0/RKS, an explicit FP64 request and at least 384 public AOs.
Density-fitted, ECP, nonlocal, D4 and range-separated compositions are excluded.
The composition matcher uses the native restricted total-density exchange
coefficient `-0.125`, not the positive physical exchange fraction `0.25`.
Both public DFT callbacks authenticate the resident density token before
forwarding the hint. Generic/HF/RSH consumers do not infer PBE0 from coefficients.

All primitive-pair residency, topology, prefix and derivative-capability guards
remain independent of preference. Explicit opt-out and bounded fallback remain
available. Proven scope should default to the faster route; missing universal
evidence is not a reason to leave every qualified owner opt-in.

## Rejected alternatives

- [Class/indexed domains](../../rejected/2026-10-11-pbe0-spd-class-force-domains.md)
  did not establish their claimed scheduling benefit and widened the adapter.
- [Narrow-low scheduling alone](../../rejected/2026-10-11-pbe0-spd-narrow-low-force.md)
  did not pass complete endpoints. Its earlier 96-atom density failure remains
  unexplained and retained; later cache cohorts do not erase it.
- More center fusion does not remove repeated spatial evaluations because the
  incumbent already accumulates all centers together.
- Universal automatic selection is not supported by the measured method/size
  scope. The AO boundary is an algorithm-informed guard, not a geometry whitelist;
  larger owners retain the same bounded cache, but have no new measured timings.

## Invariants

- Mathematical lowering belongs to the compiler; method authentication and
  resource/topology admission belong to the native owners.
- Scalar recurrence, radial validation, screening and accumulation order remain
  unchanged. Disabled derivative channels are exactly zero.
- Independent derivative gates remain 3e-10; complete E/F gates remain
  1e-8 Eh / 1e-7 Eh/Bohr, with cross-arm density drift at most 2e-11.
- Compare complete synchronized energy-plus-force endpoints with semantic work,
  not isolated kernels, capacity counters or cross-code wall times.
- Keep failures, timeouts, slow samples and source identity; do not pool cohorts.
- Every real-GPU command uses finite Slurm allocation and assigned visibility.

## Evidence

The publication is
`benchmarks/results/pbe0-spd-spatial-reuse-20261011/`. Measured source is dirty
master `637466c624c4c4c4dd4eb7c511a2c88675744325` plus production patch SHA256
`7cfa7717e6682b5ccc95fa8d494d4a0e052b4258f1b33ba37341477618765f44`.
Native SHA256 is
`47209e21c0fb92038475c0f7705f3540d0ea06bc77c0f3afab4201bfd3b26a49`.
Publication/test-adapter edits do not change those production bytes.

There are 561 passing focused host tests and zero compiler/shared-SCF dependency
errors. A further 139 default-inventory/policy/publication tests pass on the final
non-product overlay without rerunning successful device cohorts. The inventory
audits method/size/precision scope, captured automatic preference and opt-out.
Five artifact-specific offline regressions authenticate the new bundle, recompute
all complete E/F and scoped timing gates, and protect the retained failed wrappers.
Cached build 7321 succeeds with verified ccache 4.5.1, actual launcher
commands and retained statistics. GPU 7323 passes 64 independent source-major
derivative calls and five independent FCI force cases. Additional combined-output
CPU adapter build 7332 does not rebuild the native product; GPU 7334 passes
64 more independently contracted calls. No skips count as device evidence.

All-kernel memcheck reports zero errors for both layouts. Unfiltered racecheck
7325 times out with actual driver exit 143; it is not a successful safety run.
Filtered racecheck 7330 checks the new low kernel with zero hazards but its
parent's four-layout assertion fails, exposing separate-output-only coverage.
Observers record `[0,24,0,24]` for separate and `[24,0,24,0]` for the additional
combined probe, with no wrong widths or unknown variants. Together the disjoint
probes non-vacuously cover all four variants. Observers are absent from timing.

Jobs 7339/7342 each complete 30 numerical calls and 20 timed samples: five
interleaved pairs per geometry, one image/context and two independently converged
owners with immutable seeds. Baseline forces selection `0`; candidate is truly
unset. Both arms enable #2220 resident final validation, so its gain is not
attributed to this cache. RTX 5090/sm_120, CUDA 12.9, explicit FP64:

| Atoms / AOs | Geometry | Baseline / candidate median (s) | Reduction | Gate |
| --- | --- | ---: | ---: | ---: |
| 48 / 384 | warm | 4.863446675 / 4.569519117 | 6.0436% | 2% |
| 48 / 384 | moved-warm | 4.806275673 / 4.598930910 | 4.3140% | 2% |
| 96 / 768 | warm | 14.104704417 / 13.352007795 | 5.3365% | 2% |
| 96 / 768 | moved-warm | 14.248137016 / 13.543867871 | 4.9429% | 3.1329% |

All pass `max(2%, 2*(relative_MAD_baseline + relative_MAD_candidate))` and the
independent numerical/work/checkpoint gates. Maximum E/F errors at 48 atoms are
6.366463e-12 / 2.474398e-11, and at 96 atoms 1.136868e-11 / 3.752010e-11.
Maximum cross-arm density drift is 6.978307e-13 / 1.840028e-12. Available semantic
work agrees; capacity counts are not executed screened J/K quartet counts.

Static linked variants have REG255, SHARED4173 and LOCAL0. STACK fields are
16840/25512 bytes for RKS combined/separate and 16824/25176 for UKS. These are
not runtime peak-memory measurements or proof of spill-free execution. The
unchanged generic native section is byte-identical to master. Cold-SCF, other
methods/sizes/devices and broad formal performance/production remain unmeasured.

## Publication storage

The publication retains existing storage budgets by whitespace-only re-encoding
seven historical JSON members. All parsed data, paths, tolerances, failures and
decisions are unchanged; 668,780 bytes are saved. The publication's
`storage-compaction.json` binds original/stored hashes, canonical data identities
and the original Git revision for exact-byte recovery. Large-evidence review rows
bind the new storage bytes. No external archive, release or budget increase is
used, and compaction is not scientific requalification of those old records.

## Revisit when

Independent complete endpoints justify extending automatic scope, or profiler
evidence identifies a better reuse/layout schedule that preserves the same
scientific gates and bounded fallback. Do not rerun successful device cohorts
for unrelated master changes; inspect path compatibility at publication boundaries.
