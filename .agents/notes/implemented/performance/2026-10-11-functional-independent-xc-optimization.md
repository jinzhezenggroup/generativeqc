# Decision: Registry-wide, functional-independent XC optimization

Status: implemented
Date: 2026-10-11

## Problem

The earlier equal-spin candidate excluded LDA/PBE using a numerical code
threshold. Its launch width and feature layout still contained named-family
branches. Individual CUDA generator scripts had to request equal-spin siblings.
The stationary bound-point optimization admitted only one exact PBE0
composition, even though its producer/storage proof is reusable.

The user requested removing functional-specific optimization hardcoding and
making applicable optimizations available to all functionals.

## Decision

- Enumerate every admitted native CUDA point consumer from the canonical
  curated registry and generated split-hybrid DFT metadata projection. Generate
  feature-width traits and equal-spin binding dispatch from that same census.
  Neither numerical code ranges nor method labels determine optimization
  eligibility. DFT still does not import XC scientific implementations.
- Derive point launch width from Cartesian ingredient shape: physical GGA
  uses 32 threads, LDA/MGGA and response use 128. Single, split and batched
  consumers share this policy; both candidate and incumbent algebra use it.
- Every polarized scalar CUDA lowering automatically emits its equal-spin
  sibling. Preserve polarized differentiation before structural substitution,
  and leave sigma_ab independent. Host-only emission remains unchanged unless
  a sibling is explicitly requested.
- LDA/PBE reuse the existing four-channel equal-spin numerical core with the
  admitted exchange/correlation scales, rather than a named hybrid formula.
- Every registered unpolarized stationary consumer gets a restricted point
  lowering through its original scalar/continuation implementation. The shared
  equal-spin environment selector makes these candidates eligible without
  expanding public method/backend support.
- Retain the old exact-composition check only for the already-qualified default
  and established subnormal policy. It is not a candidate whitelist. Add the
  generic stationary request spelling; keep old environment/metrics spellings
  for compatibility, with explicit precedence and a generic capability alias.

## Invariants and fallbacks

Optimization eligibility is not a scientific method definition, a proof that
two supplied feature arrays are equal, or default-promotion evidence. Native
generation/producer witnesses, phased scratch admission, and exclusion of
external nonlocal seeds remain mandatory. UKS/response, legacy artifacts,
unproven producers and bounded/nonphased geometry retain general evaluation.

Scientific component definitions, family-specific continuation policies,
stable native codes and generated ABI symbol names legitimately contain
functional identities. Removing those would conflate distinct mathematics;
they are not tuning whitelists. An unsupported method stays unsupported.

## Rejected alternatives

- Adding more method names or codes to optimization branches repeats the
  original maintenance failure and misses newly registered consumers.
- Importing XC science from DFT to infer features violates compiler ownership;
  use generated immutable execution metadata instead.
- Enabling every newly eligible stationary point by default bypasses complete
  endpoint promotion gates. Preserve its opt-in and existing qualified default.
- Replacing all scientific point implementations with one formula would change
  work-input, vacuum and continuation semantics rather than generalize tuning.

## Evidence and boundaries

Registry-driven tests enumerate every current curated and split-hybrid consumer,
verify dispatch for correct/wrong widths and single/batched launches, and inject
opaque future LDA/GGA/MGGA identities to reject hidden tuning whitelists.
Actual stationary CUDA scalar bodies are compiled as strict cached host C++;
each registered consumer and additional scaled compositions compare every
energy/rho/Cartesian-gradient/tau output to independent Libxc. Generated
selectors preserve opt-out, invalid-value, UKS and legacy-default behavior.
The retained 87-point independent PBE0 subnormal fixture remains a gate.

Final focused regressions: 804 passed, one explicitly Slurm-gated CUDA test
skipped in the host run; the disjoint capacity suite passed 317 tests. Compiler
structure checked 515 modules with zero dependency errors; registry freshness,
Ruff lint/format and whitespace checks passed.

Source-matched GPU qualification ran on node4 through Slurm jobs 753 and 755,
with finite 10-minute limits and the scheduler's unchanged device visibility:

- Seven registered consumers, both single-tile and ragged batched SCF point
  layouts: 14 paired cases passed independent PySCF 2.14.0 / Libxc 7.0.0 gates.
  Maximum absolute oracle error was 8.21e-15. Generated point-source SHA-256:
  `44df10515e7c619d8f7a52030d76f15874c444073f22eaa860060afba4f98b7d`.
- Nine stationary point consumers (all seven registered consumers plus two
  scaled compositions) executed the actual emitted geometry-point kernel and
  passed independent energy and every coefficient gate. Maximum absolute
  oracle error was 9.49e-15. Probe source SHA-256:
  `0ae96c2b24f2e801f5e9672871b42f4d6c947581df32c441861b0a2fc3419915`.
- Separate memcheck and initcheck runs for both producers reported zero errors.
  NVCC 12.9 / sm_120 compilation reused ccache. Failed prerequisite attempts
  (missing public include path, C++17 for C++20 headers, and a scheduler node
  alias) are retained separately from the successful receipts.

Strict host CSE can be bitwise-identical, but the device compiler's specialized
programs need not be: only six SCF layout cases and four stationary consumers
were bitwise-identical on the device. All cases passed the explicit independent
energy (1e-12 absolute) and coefficient (2e-10 relative/absolute) gates. Do not
describe the registry-wide CUDA result as bitwise parity.

Ignored raw sources, receipts, hashes and the verifier report are under
`.artifacts/functional-independent-xc/`. The grid's owned panel-copy lineage
also proves tau equality when requested; a rho-only witness now permits LDA
without requiring unused gradient features. Source-transition, copied-panel
failure and immutable feature-mask tests remain fail-closed.

The capacity qualifier binds each new transitive descriptor helper and the
generic request policy, retains negative mutation tests, and charges the same
extra `8 * (48 * task_capacity + 7 * nao)` host reservation as the producer.
Refreshing reviewed source fingerprints does not waive device/provider or
performance promotion gates and does not modify the frozen input contract.

This supersedes the earlier candidate-coverage exclusions, not its historical
source-matched timing receipts. Earlier frozen native-library endpoint evidence
does not validate these new generated sources. Registry-wide GPU coefficient
and sanitizer checks are separate from complete endpoint timing qualification;
no whole-method speedup or default promotion follows from scalar/codegen tests.

## Master integration follow-up

The candidate was rebased in place onto detached `origin/master` at
`f80f732e2e1d8d4fbf1a25079baf5990d7aeabc3`, preserving the uncommitted changes.
The preceding 1,121-test and Slurm point receipts belong to the earlier
`a9b991f723ae0482c4496d20c9e2dc0649b6ffe5` integration, not to this new base.
The pre-integration patch and file archive remain under the ignored
`master-integration/` evidence directory.

Current-base host regressions passed 1,086 focused tests and 237 disjoint
fallback/lifecycle tests. The latter also had 93 explicitly gated tests skipped;
four tests needing a full native library remain pending and were explicitly
deselected from the host-only retry. The initial wider attempt exposed those
missing-library prerequisites and an old source-substring expectation. Updating
that expectation preserves both original scientific checks: unpolarized method
metadata survives into the geometry consumer, and its emitted polarized point
mathematics matches the same method-owned components and range parameters.
Both spin contracts now have explicit coverage, including generic capability
and non-default qualification checks. No native-library failure was hidden by
changing its test or admission policy.

The compiler-structure checker reports 519 modules and zero dependency errors;
the generated registry check, Ruff lint/format, and whitespace checks pass.
The full native build uses a separate frozen source directory, with ccache,
Release CUDA 12.9/sm_120, and the portable CUDA AOT profile. All 2,725 frozen
build-input SHA-256 values were checked against the current checkout without
mismatches. Building and linking this full library, its source-matched force
packages, and running its molecular endpoints are separate pending gates; the
earlier frozen endpoint library is not substituted for this build.

The new ignored complete-endpoint driver compares equal-spin disabled,
restricted-point off, scalar packing against equal-spin enabled,
restricted-point on, bulk packing. This is a joint ablation, not a per-selector
speed attribution. It retains actual reported work by execution schema, and
fails the timing/promotion gate closed when native integral work is unavailable.
In particular, absent composite-route quartet counters are not interpreted as
zero or replaced with theoretical counts. Driver smoke checks cover unavailable,
matching, and changed measured-work cohorts. Independent molecular gates and
larger-system/UKS qualification remain required before any default promotion.

Ignored current-base logs and reproduction scripts are under
`.artifacts/functional-independent-xc/`, including `master-regression.log`,
`master-fallback-regression.log`, `master-fallback-host-regression.log`,
`full-source-inputs.sha256`, and `full_endpoint_probe.py`.

## Ingredient-policy follow-up

The remaining prepared-grid shortcut `functional != 0` was replaced with the
existing density-jet trait helper. That helper now includes kinetic-density
requirements as well as gradient requirements; either needs first AO derivatives.
Known rho-only consumers use one jet regardless of their numeric code, and
unclassified legacy consumers retain four jets conservatively. All currently
admitted consumers keep their previous AO order and allocation settings.
Opaque rho-only, gradient-only, tau-only, and combined traits have explicit
coverage. This is not admission of a new public method or ingredient domain.

The capacity qualifier refreshes the reviewed prepared-owner fingerprint and
also binds the transitive density-jet policy. Its negative mutation test rejects
dropping the tau requirement. The frozen DFT-MP-v1 input contract, mathematical
work definitions, and allocation equations are unchanged.

The already-running full native build's input directory was not overwritten.
A separately frozen `final-source` directory differs from it only in
`python/generativeqc/_stationary_cuda.py` and the corresponding capacity
qualifier. All 2,725 final-source input hashes match the current checkout; all
native and scientific compiler files are identical between the two snapshots.
The final endpoint driver authenticates that snapshot and uses the full native
binary and force packages built from those identical native/compiler inputs.

Master integration changed some generated expression identities even when the
arithmetic text stayed the same, so point-device validation was regenerated,
recompiled, and rerun instead of relabeling the old receipts. Slurm job 756
passed 14 SCF registry/layout cases, nine stationary consumers, independent
PySCF 2.14.0 / Libxc 7.0.0 gates, and four separate zero-error sanitizer runs.
Its SCF compilation allowed FMA, whereas stationary used `--fmad=false`; only
six SCF cases were bitwise-identical, while all nine stationary consumers were.
Maximum oracle errors were 8.20e-15 and 1.10e-14 respectively. This remains a
point-level source/precision receipt, not a complete molecular endpoint claim.
A separate strict SCF-plus-stationary probe is retained to match the full native
grid/force packages' `--fmad=false` contract; do not mix the two precision runs.

Force packages build in an independent CMake binary directory with the same
frozen native/compiler inputs and precision settings. This overlaps remaining
core-library compilation without running concurrent Ninja owners in one build
directory. Raw package hashes and command inventories remain separate from
complete-endpoint timing.

## Current-source validation closure

The final combined host run passed 1,333 tests, with 93 explicitly gated tests
skipped and the four full-native CPU tests deselected from that run. Those four
disjoint tests subsequently passed against the newly built native library and
the final-source Python snapshot. Do not add earlier overlapping reruns to the
1,337 successful-test count. Structure/generation, Ruff, and whitespace checks
remain clean; the new density-jet helper's mutation rejection is included.

The full native build and independent force-package build completed. The native
library SHA-256 is
`3fc105357e35090ebbe8c10f9f8000a4b11e3887d56f15a9fc98aeed654e8be3`.
The latter CMake build generated the five requested s/p/d RKS packages from the
same immutable compiler/native files, retaining separate manifests, binary
hashes, and command inventories. No old endpoint library was substituted.

Slurm job 757 reran point checks with `--fmad=false` for both SCF and stationary,
matching the full native grid and force-package precision contract. All 14 SCF
registry/layout cases and nine stationary consumers passed independent Libxc
gates; all four separate memcheck/initcheck runs reported zero errors. These 23
specific paired cases were bitwise-identical. Maximum oracle errors were
9.70e-15 and 1.10e-14 respectively. This does not extend bitwise claims to the
FMA-enabled receipt, arbitrary CUDA inputs, or complete molecular forces.
Current strict raw receipts are under
`.artifacts/functional-independent-xc-master-strict/`.

The same finite Slurm job completed 22 public energy-plus-force observations
per method for B3LYP, M06-2X, WB97M-V, and r2SCAN-3c: two cold-including-prepare
observations plus five interleaved pairs each for warm and moved-warm calls.
Every one of the 88 observations passed the independently generated molecular
energy/force gates, including matched VV10 for WB97M-V and prescribed basis,
D4, and gCP for r2SCAN-3c. Maximum absolute errors were 2.28e-13 Eh and 2.17e-9
Eh/Bohr, below the unchanged 1e-8/1e-7 acceptance gates.

Warm complete-endpoint baseline/candidate medians for the water pilot were
19.499/3.994 s (B3LYP), 19.444/4.024 s (M06-2X), and 45.504/9.275 s (r2SCAN-3c).
Moved-warm medians were 20.029/4.286, 20.178/4.450, and 45.244/9.572 s.
Each passed the robust timing gate with matching iterations, Fock builds, and
executed descriptor/primitive/page/grid counts. B3LYP and M06-2X each executed
1,066,527 descriptors, 5,659,930 primitive records, and 262 pages per endpoint;
r2SCAN-3c executed 2,568,000 descriptors, 12,134,769 primitive records, and 628
pages. These are a joint selector ablation, not an individual XC speed claim.
Both arms retained the general stationary point route in this small pilot;
the new restricted-point capability/request did not bypass its storage and
producer-proof gates. Thus the standalone stationary-point validation must not
be represented as execution of that optimization in these molecular calls.

WB97M-V remained a control: warm medians 0.289072/0.289127 s and moved-warm
1.408977/1.408380 s show no significant improvement. Its available counters
matched, but native integral actual work was not exported, so the complete-work
and timing-promotion gate failed closed. Nonlocal dense-pair capacity is also
not a measurement of active pair work. Do not replace unavailable native or
nonlocal work with logical quartet bounds, capacity, or zeros.

Current full raw results and gate reports are under
`.artifacts/functional-independent-xc/`: `full-endpoints.json`,
`full-endpoint-job.txt`, `master-independent-molecular-final.json`,
`independent-molecular-references.json`, `master-combined-final-regression.log`,
and `master-native-cpu-regression.log`. The endpoint raw status records its
measurement-time oracle state; the separate final oracle report closes those
gates without rewriting raw measurements. Full-native endpoint sanitizer,
larger planner-sensitive systems, and broader UKS/method-family performance
qualification remain separate gates. These water-pilot results do not authorize
universal defaults, new public backend admission, or declaring the broader
optimization goal complete.

## Integration checkpoint: bd6f78288 and automatic providers

The next master checkpoint is `bd6f78288` (including generic hybrid phase-source
reuse, direct shared-spatial force responses, and automatic Libxc CUDA point
programs). The pre-integration diff and untracked sources were preserved under
`.artifacts/functional-independent-xc/bd6f-integration/`. No branch, commit, PR,
tag, or release was created. The earlier f80 source snapshots, libraries and
timings remain historical evidence, not qualification of this new checkpoint.

Keep master's one `dft/xc_point_cuda.py` physical-feature traversal. Curated
incumbent global kernels retain three template arguments, so an extra provider
or equal-spin argument cannot silently invalidate their resource classifier.
A distinct restricted entry binds the canonical registered consumer and calls
that shared traversal. Automatic providers generate a sibling for every entry
of their structural registry and use the same opt-in host selector. No positive
functional-name/code whitelist, public CUDA resolver change, or scientific
formula copy is introduced. CPU automatic point source remains byte-identical;
the polarized semantic identity stays unchanged, while exact CUDA provider
source identities cover both bodies and selection policy.

Automatic ordinary and candidate launches deliberately retain 128 threads.
Extending the curated GGA width to imported automatic programs without their
own schedule/resource qualification was rejected. Source availability is not
response, batching, executable-cache, molecular-force or default admission.
The automatic work policy's spin-symmetric floors and FHC clamp preserve paired
work-input equality; sigma-ab remains independent. Substitution still occurs
only after all polarized derivatives are built, preserving every partial.

Capacity metrics were rebound to the combined generic capability alias and
master's strict-zip arity checks; all negative contract checks remain intact.
An initial combined test run correctly rejected an imported compiler module
whose source was formatted during that run. Do not relax that stale-source
gate: rerun source-bound checks only after compiler bytes are stable.

### Source-matched checkpoint evidence

Raw data is retained under
`.artifacts/functional-independent-xc/bd6f-integration/`. The immutable remote
snapshot is `n4:/data/jzzeng/gqc-functional-independent-20261011/bd6f-automatic/source`.
Native/compiler inputs match the checkout; subsequent documentation edits are
not retroactively part of that snapshot. Neither f80 library was reused.

- All **197 registrations** preserve the exact incumbent CPU and ordinary CUDA
  scalar source bytes, semantic identities and scalar-source identities against
  `bd6f78288`. The new provider source identities deliberately change because
  they now bind both bodies and the shared selector. See
  `incumbent-point-preservation.json`.
- All **788 automatic point kernels** compile: 394 ordinary and 394 restricted
  single/batched variants. NVCC 12.9.86, release `-O3`, sm_120, `--fmad=false`,
  and the existing ccache are used. Ordinary resource extrema remain 48–255
  registers, 256-byte stack, 224-byte spill loads and 176-byte spill stores.
  Restricted extrema are 48–255, 224, 168 and 120 respectively. Each arm still
  has eight spilling variants. These are compile/resource observations, not
  broad numerical qualification or a speed claim.
- **Slurm job 759**, node4 RTX 5090, passes all **44 paired cases** across eleven
  registrations, RKS/UKS, and single/batched partial tiles. The 22 RKS cases
  exercise the restricted sibling; the 22 UKS cases are unchanged-route controls.
  These specific paired outputs are bitwise-identical with contraction disabled.
  Independent PySCF 2.14.0 / Libxc 7.0.0 checks retain the point energy gate
  `atol=1e-12` and coefficient gates `rtol=atol=2e-10`; maximum weighted-energy
  and coefficient absolute errors are 1.11e-16 and 1.31e-12. Four independent
  memcheck/initcheck RKS/UKS runs report zero errors. The executable SHA256 is
  `211a4da80c82fbe1c6c7de4f2d1123e5ca948d1290daec1d25be273b9072fb60`.
- **Slurm job 761** recompiles and validates the current curated/static-binding
  traversal: all 14 SCF consumer/layout pairs and nine stationary consumers
  pass polarized Libxc checks, are bitwise-identical in these paired cases, and
  pass four separate memcheck/initcheck runs. Maximum oracle errors are 9.70e-15
  and 1.10e-14. Job 760 stopped because the verifier ran before sanitizer logs
  existed; its raw results remain separately in `curated-gpu-failed-760/`.
- The 125-file related regression yields **2768 passed, 482 conditional skips**:
  2632 pass initially, then all 136 missing-native-library cases pass against a
  newly built source-matched CPU library. This library has CUDA and shell/force
  AOT disabled, not numerical guards disabled. Its SHA256 is
  `db5e4c34d40a366e31af0d3eb0ef750fcc5a7524b168b01be90a09d8f4ca2f80`.
  The separate native/registry/capacity integration run passes **621 tests,
  133 conditional skips**; overlaps must not be summed into a unique total.
  Ruff lint/format, generated split projection, `git diff --check`, and compiler
  ownership checks pass: **522 compiler modules, zero dependency errors**.

This checkpoint does not qualify complete CUDA energy/force endpoints, installed
automatic bundles, full-catalog numerical correctness, response, universal
performance improvement or new defaults. The earlier water endpoints and their
pending larger-system/UKS/wB97-family gates retain their own source boundaries.
The broader optimization goal remains active.

## Final ingredient-join audit

The emitted geometry point setup now consumes `stationary_coefficients > 1`
instead of `stationary_functional != 0`. The coefficient layout already comes
from the registered ingredient contract; a rho-only consumer must not read
inactive gradient channels merely because its code is nonzero, and a gradient
consumer must not lose its inputs merely because its code is zero. All existing
registered consumers retain their previous feature reads and arithmetic.

Six native host executions extract the actual generated feature join and test
opaque codes 0/917 against coefficient widths 1/4/5, poisoning inactive channels
with NaNs. Together with the independent stationary scalar checks, **63 tests
pass**. The separate capacity, component packing, derivative policy, geometry
and strict-zip regression passes **483 tests**; these are overlapping suites,
not an additive unique-test total. Raw outputs are under
`.artifacts/functional-independent-xc/master56-endpoints/ingredient-layout-tests.log`
and `master56-generic-final/final-host-regression.log`.

Source `56bedd457` plus the generic optimizations is frozen separately from the
constructor-only derivative candidate. Its full CUDA library and five RKS force
packages finish successfully, and Slurm 762 validates 28 complete cold/warm/moved
water energy/force observations against the independent molecular references.
The fallback diagnostic reproduces the absent retained Direct derivative lease
for B3LYP, M06-2X and r2SCAN-3c. This is current-source evidence, not extrapolation
from the older f80 water pilot. The orthogonal derivative-ownership note records
the subsequent candidate and its separate acceptance boundary.

## Revisit when

Promote candidates only after source-matched complete endpoints, work counts,
independent molecular references and resource/fallback coverage pass the current
qualification contract. Add new consumers to authoritative science/admission
registries, not to optimization code.

## References

- `docs/maintainer/xc_equal_spin_qualification.md`
- `tests/python/test_functional_independent_xc_optimization.py`
- `2026-10-11-native-xc-functional-binding.md`
- `2026-10-11-equal-spin-xc-scalar-sharing.md`
