# Decision: measured partial canonical ERI residency

Status: implemented, experimental opt-in; scoped three-pair E+F acceptance passes
Date: 2026-10-11
Comparison base: master `637466c624c4c4c4dd4eb7c511a2c88675744325`

## Problem

The existing physical-reference RHF owner rejects its complete canonical FP64
source when the optional allowance cannot hold every angular bucket. Profiler
measurements can identify expensive reusable subsets, but an offline selection
does not eliminate any production work by itself. The
[initial experiment](../../proposed/2026-10-11-selective-eri-cache-experiment.md)
records that diagnosis, source boundaries and discarded qualification shortcuts.

## Decision

Preserve default complete-cache admission and automatic eligibility. An explicit
measured mask may instead retain whole canonical AO-pair angular-sum buckets in
one packed geometry-bound source. Selected buckets replay from that source;
the remaining buckets execute the existing compensated canonical Direct kernel
once. No integral recurrence, physical equation or CPU oracle is copied into
production. The selector emits the corresponding 28-bit mask as well as the
selected inventory and its non-qualified additive prediction.

The mask and optional value ceiling are captured once per phase owner. Published
values and packed offsets are immutable. A moved geometry constructs a new
source. Provider/device/host preparation, correction planes, scratch and packed
host descriptors precede optional value admission. Capacity/allocation refusal
retains exact whole-source Direct; numerical and driver failures propagate.

The existing native RHF frame-response consumer shares this policy. Its
independent final scalar audit and positive-screening actions never borrow the
resident source. Ordinary public HF/UHF and screened PBE0/UKS/RSH SCF are not
changed: the frozen measurement master does not contain the proposed
`DirectJkSourceLease` owner.

## Invariants

- For each physical action, candidates equal the complete canonical inventory
  `N`, and recurrence evaluations equal `N - cached_values`.
- Source construction submits and finite-audits only the selected values.
  There is no second complete-source Direct pass after partial replay.
- Unset controls retain the existing full-cache policy; zero mask/ceiling and
  insufficient optional capacity retain bounded exact fallback.
- A published selection cannot change, including repeated admission; a foreign
  current device is rejected before borrowing source values.
- Cache admission cannot enlarge the complete numeric budget. Context/stack,
  driver allocation rounding and runtime objects retain their existing explicit
  exclusions; mathematical value bytes are not a measured VRAM peak.

## Complete endpoint and rejected shortcuts

The benchmark composes `run_rhf_cuda_cached` physical-reference export with
`rhf_frame_response_cuda` at zero correlation seeds, then adds nuclear repulsion
using the existing shared native assembler. Reference export returns before
ordinary force assembly, so `compute_forces=true` is not an E+F endpoint on this
path. This was caught by a failed smoke receipt, not hidden by reporting energy
timing as E+F. Warm seeds come from the validated reference density rather than
the unused ordinary SCF density destination.

All four phases include complete SCF, physical-reference validation and native
forces. Cold additionally includes normalized native metadata/context creation;
identity hashing, process startup, input deserialization and the independent
CPU PySCF oracle are outside production wall timing. The response conservatively
charges the exported SCF peak while its cached plan remains alive.

One library is loaded per fresh process. Baseline/candidate processes interleave
on the same Slurm-assigned physical GPU. A scoped experimental PR requires at least three pairs,
matched scientific/device/settings identity and exact iteration/Fock/response
work counts, independent E+F gates, at least 2% median Cold E+F improvement and no
median phase regression exceeding 3%. The profiler's predicted gain does not
satisfy any of those endpoint gates. This scoped acceptance does not override the
shared formal comparison protocol or authorize default/global promotion.

## Evidence so far

- Candidate CUDA build uses the frozen master base, CUDA 12.9.1/SM120 and reused
  ccache. Native candidate SHA256 is
  `d77599bc8e96d14533f648767ca0674d172a204c862d44986509916b666f2e34`;
  immutable master SHA256 is
  `1808dc3bd0c3d5a9faab9b69e78f8705f1c63d5c54512f30e6d31e154e3d19a0`.
- Slurm n1 job 7436 passes all 58 resident signed J/K tests, including partial,
  empty, full, moved, allocation-refused, spherical/Cartesian and through-f
  restricted/unrestricted domains against independent PySCF/libcint integrals.
- Slurm n1 job 7451 passes all 88 native RHF frame-response GPU tests, including
  the unchanged derivative, finite-difference and scalar-audit routes.
- Slurm n1 job 7439 memchecks the actual partial production provider with a
  moved through-f source: zero errors and zero bytes leaked.
- Slurm n1 job 7443 completes both native 6-atom TZVPD E+F routes. Maximum
  independent energy error is `2.5579538487363607e-13` Eh and maximum force error
  is `1.2458473428247907e-9` Eh/Bohr. All four phases have matched work counts.
  Partial storage is slower than an admitted complete cache here, so this smoke
  case is not performance qualification or evidence for default enablement.
- Slurm n1 job 7447 completes zero-mask, zero-ceiling and one-byte-ceiling
  6-atom E+F fallbacks. All 12 phases have zero resident response bytes and zero
  submitted SCF cache values; maximum independent force error is
  `1.2459876750151033e-9` Eh/Bohr.
- The complete independent acceptance limits are `1e-8` Eh and `1e-6`
  Eh/Bohr; paired limits are `1e-9` Eh and `1e-7` Eh/Bohr. These gates do not
  relax the existing derivative validation contract.

Full raw receipts, scheduler identities, failures and compiler logs are retained
in `.artifacts/selective-eri-cache/n1-partial-637466c6/` and at
`/data/jzzeng/selective-eri-cache-partial-637466c6-20261011-3e50/` on n1. The
12-atom TZVPD same-device comparison uses a 3-GiB complete budget: master's
4,328,718,848-byte full source cannot fit; the measured mask `166692865` retains
1,068,437,360 bytes under a 1-GiB optional ceiling. Results remain unqualified
until all four phases and repeated paired acceptance finish.

The initial 2-GiB comparison (n1 job 7445) was interrupted after its trace proved
that mandatory state plus the selected values exceed admission. The reference
peak before optional storage is already approximately 1.24 billion bytes. Both
routes correctly retained Direct, so continuing that comparison could not test
partial-cache performance. The replacement raises the **same complete budget
on both sides**, without disabling master's cache; its complete value inventory
alone still exceeds 3 GiB. Preserve that failed planning attempt rather than
reporting an optional 1-GiB allowance as the complete memory requirement.

The 1-GiB subset under a 3-GiB complete budget (n1 job 7454) passes complete
first-pair E+F/work checks, but Cold takes 357.955 s versus 356.296 s. Warm and
moved phases improve; Cold does not justify spending two more screening pairs.
This one-pair screen is retained as inconclusive, not as statistical promotion.
The isolated canonical Direct cost model cannot price the existing SCF shell
schedule: a profitable additive subset need not accelerate complete Cold E+F.

A separately selected 2-GiB subset uses mask `66058240` and 2,146,754,256 bytes.
Its Cold-only 4-GiB-budget preflight (n1 job 7457) takes 301.336 s and passes the
independent energy/force gates, with the same 20 Fock builds and response work.
This different-GPU preflight is not paired qualification. The complete n1 job
7464 interleaves three fresh pairs under identical 4-GiB budgets on one physical
GPU. Its first candidate four-phase solve completes; repeated acceptance remains
pending. Master's value inventory alone exceeds 4 GiB, so its exact fallback is
a real memory-constrained baseline, not an artificially disabled cache.

## Completed scoped comparison

All six fresh processes in n1 job 7464 exit successfully on one Slurm-assigned
RTX 5090. The exact process order is candidate/baseline/baseline/candidate/
candidate/baseline. All 24 phases have matching physical Fock and native response
work. Independent maximum errors are `1.1368683772161603e-12` Eh and
`3.303612036709147e-9` Eh/Bohr. The medians are:

| Complete E+F phase | Master seconds | Partial seconds | Reduction |
| --- | ---: | ---: | ---: |
| Cold | 357.04463781788945 | 302.4558742195368 | 15.2891% |
| Warm | 236.58689338713884 | 141.21249871701002 | 40.3126% |
| Moved | 325.97220000624657 | 236.35782834142447 | 27.4914% |
| Moved-Warm | 212.8356042392552 | 135.33968444168568 | 36.4112% |

The selected source contains 268,344,282 values; all 541,089,856 canonical
candidates are visited per action and exactly 272,745,574 are independently
evaluated. Cold has 20 physical Fock builds on both sides. The observed reference
numeric peak is below the identical 4-GiB budget. This is numeric admission, not
an instrumented context/stack/allocator-rounded VRAM peak.

The self-contained reviewed bundle is
`benchmarks/results/selective-eri-rhf-partial-20261011/`: full numerical outputs,
three paired samples per phase, independent oracle, source observers, dirty-source
reconstruction and reproduction commands. Its numerical publication is accepted;
shared formal/global performance and default promotion remain explicitly not-run.
The partial controls remain opt-in, and no PBE0/public-HF or larger-case speedup
is inferred.

## Master compatibility follow-up

After the qualified run, master `833986298298d95331d127ff9b62b82a091f808f`
introduced the method-neutral phase source and compact screened layouts. Keep
that ownership refactor: `RhfResidentValues` forwards its frozen experimental
policy to `DirectJkSourceLease` rather than retaining a second allocation/audit
owner. The common lease charges packed host metadata and audits the complete
candidate inventory and exactly the uncached recurrences. Its default remains
full residency; non-full masks with positive screening refuse before source
preparation. KS consumers do not read the RHF environment controls.

The frozen reconstruction, binaries and three-pair timing evidence remain
unchanged. Post-merge checks establish numerical/ownership compatibility, not a
new comparison against the later master or a broader default promotion.
The compatibility host suite passes 330 tests, including the shared screened
lease's unchanged automatic admission and explicit partial-screening refusal.

n1 Slurm job 7500 completes the full Release/CUDA 12.9.1/SM120 integration
build with reused ccache, all 88 independent frame-response GPU tests, and a
four-phase 6-atom TZVPD native E+F smoke endpoint through the shared lease.
All nine modified native source/header hashes match the remote build inputs.
The rebuilt library SHA-256 is
`e4a55e4003557a14be3f00c3d8171ad1fbb06b3087db295a51f93a52af3e9ba9`.
Independent maximum energy/force errors are `4.831690603168681e-13` Eh and
`1.2463684815125498e-9` Eh/Bohr. Every SCF phase admits 16,916,331 packed values
(135,330,648 bytes), visits 34,084,896 candidates and evaluates exactly
17,168,565 uncached recurrences per action; every response also uses the partial
source. These are numerical compatibility checks, not additional timed pairs.
Raw outcome, oracle, full forces, source receipts and binary identities remain
under the ignored `.artifacts/selective-eri-cache/n1-integration-83398629/`
and the n1 integration artifact root.

The first integration CI tested a newer synthetic merge with master `5b577dbb`,
whose additional evidence left less than 48 KiB of the aggregate checkout
budget. Further compact only this publication: gzip all JSON summaries and the
exact input, and retain an xz reconstruction containing every compiled native
change plus the endpoint/oracle clients and real-GPU response tests. All native
patch sections remain byte-identical. Optional profiler/selector and host-test
edits remain recoverable from the original complete patch in existing Git
revision `0d27821557ba77bfd6e1a65316ed88bc20061fbd`; its original checksum and
size are recorded in the publication. No timing, force, work-count or scientific
identity is dropped, and no historical bundle or external release is modified.

## Revisit when

The screened-source boundary is extended by the separate
[screened partial experiment](2026-10-11-screened-partial-eri-experiment.md).
That extension does not relabel this note's frozen RHF timings as PBE0 or
default-promotion evidence.

Complete-endpoint evidence justifies a broader admission/profitability model,
screened-source owners become available on master, or profiling identifies
source-build amortization and layout interactions that invalidate the additive
bucket model. Do not extrapolate these RHF results to 24/96 atoms or PBE0.

## References

- Current controls: `docs/developer/scf_module_boundaries.md`.
- Profiler/selector: `benchmarks/selective_eri_cache.py`.
- Complete E+F: `benchmarks/selective_eri_rhf_endpoint.*`.
- Independent/paired gates: `benchmarks/selective_eri_rhf_acceptance.py`.
