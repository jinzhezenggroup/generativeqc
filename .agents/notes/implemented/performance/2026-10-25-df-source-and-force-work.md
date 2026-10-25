# Decision: compact exact-zero DF charge sources and admit larger force tiles

Status: implemented
Date: 2026-10-25

## Problem

The streamed pure-J Hermite owner already avoids radial evaluation for exact
zero precontracted densities, but still traverses every zero source in every
auxiliary-target block. Repeated traversal is measurable at 96 atoms. The
resident PBE owner does not use this charge path and cannot benefit from it.

Ordinary large DF forces also prefer 256 grid points under 256 MiB host and
512 MiB device staging limits. Larger AO tiles reduce repeated force work, but
must coexist with the phased Becke cache and native provider reservations.
Increasing tile size alone can evict that cache or exceed the host budget.

## Exact source compaction

After every orbital Hermite density precontraction, compact only records whose
existing `density_active` flag is true. The flag tests the actual current FP64
Hermite density against exact zero; NaNs and infinities remain active, including
NaN * zero. There is no tolerance, bare-coefficient screen, norm approximation,
reference density or changed metric/rank policy.

One CUDA block owns each angular class and publishes stable source indices and
a refreshed count. Empty classes overwrite their counts with zero. The charge
kernel consumes this list for primal charge and the raised auxiliary charge;
both previously skipped exactly these zero densities. Orbital projection and
raised orbital derivatives retain their complete target domain. A zero primal
transform does not authorize omitting a nonzero derivative.

Lists refresh on the owning stream before every ordinary/captured charge or
force call. Geometry-owned transforms are rebuilt with their original source
owner. No list survives a changed density without refresh, no host readback or
replay allocation is introduced, and all original metric/provider failure
behavior remains. Stable source order does not imply bitwise-identical sums:
removing zero lanes changes the reduction grouping, so independent numerical
and complete-endpoint gates remain mandatory.

The full orbital owner always allocates exactly its charged `size_t` index
array and 16 offset/count slots. Auxiliary and transform-only/raw views do not
reserve them. Layout preflight and the real device ledger therefore agree,
including a one-byte-short refusal. Earlier probes reserved storage even when
they did not allocate it; this implementation does not retain that mismatch.

## Force capacity and tile ownership

For all-electron ordinary DF owners with at least 48 atoms, public execution
selects finite additional force caps of 512 MiB host and 1 GiB device before
building the active-AO workload or allocating scratch. Smaller, direct and ECP
owners keep their existing 256/512 MiB caps; composite caps are unchanged.
`_cuda_force_resources.py` owns the constants used by public execution and the
existing Direct resource inventory.

Large fitted force planners prefer 1024 points only when both larger caps are
available. Dry admission still accounts for the complete grid, AO, Becke and
native-integral owners and tries smaller candidates on capacity rejection.
Explicit tile requests are not silently resized or given larger caps. Runtime
provider/allocation failures do not become an unbounded retry.

These are additional force reservations, not a total calculator memory budget.
The DF value allowance is unchanged. Public DFT DF exact-byte resource plans
remain explicitly unsupported in `Calculator._base_resource_request`; this
change does not pretend otherwise. Direct resource estimates retain their
previous numeric reservations.

## Qualification and limits

Base `6d508c262` includes the metric value/derivative symmetry PRs. The retained
capsule separates exact-zero-only ablation, larger-capacity ablation, and the
final three-round control/candidate/GPU4PySCF populations. Every timed population
uses a fresh process and includes construction, preparation, SCF, analytic forces
and two warm calls. No experimental screening or reduced DF value allowance is
used. Geometry qualification references are not reused as timing populations:
their shared Python process has a different CUDA initialization lifetime.

Independent original/warm/moved/fresh-moved gates pass for PBE48, PBE96 and
r2SCAN-3c96 (12 native calls; maximum force error 1.4103e-6 Eh/Bohr). Target work
remains 10 cold, 1 warm, 6 moved and 10 fresh-moved iterations/Fock builds.
The source suite covers Cartesian/spherical orbital f and auxiliary g, signed
and long contractions, exponent/translation ranges, independent finite-difference
forces, graph refresh, exact resource accounting and failure cleanup.
The final real-GPU source suite passes 188 tests, including changed-density
NaN/infinity propagation. Two long-contraction cases pass compute-sanitizer
memcheck with zero errors (all API-error reporting enabled). These are private
source-owner sanitizer checks, not a whole large-DF endpoint sanitizer run.
Host qualification passes 179 tests covering layout, full-owner capacity
admission, explicit caps, optional-map fallback and active-AO policy. The capsule
replay checks 129 complete retained E/F calls, including all ablation samples.

Performance evidence is limited to the measured water workloads on Slurm RTX
5090 with the declared validation-GEMM and XC spatial-order settings. It is not
an all-method/all-device ranking or a claim that DF cold parity is complete.
No root README timing is changed by this decision.

Three rotated rounds yield the following medians (seconds). Native control and
candidate use identical scientific settings and target work; GPU4PySCF retains
its own solver and default quadratures.

| Case | Control cold | Candidate cold | GPU4 cold | Control warm | Candidate warm | GPU4 warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PBE48 | 5.7998 | 5.3467 | 6.7275 | 1.7751 | 1.3292 | 1.4456 |
| PBE96 | 15.3923 | 14.5239 | 12.5753 | 5.6940 | 4.8601 | 4.5277 |
| r2SCAN-3c96 | 32.4400 | 30.1772 | 25.7387 | 10.3259 | 8.7214 | 12.4609 |

PBE48 wins both populations in this experiment. PBE96 and r2SCAN96 remain
15.5% and 17.2% slower cold than GPU4PySCF; PBE96 warm also remains slower.
The exact-zero-only ablation improves r2SCAN cold from 32.6709 to 32.0264 s and
does not improve resident PBE. The separate force-capacity ablation improves
all three cases. These ablations used different allocations; do not combine
their fastest samples into a synthetic endpoint.

The source/receipt capsule is `benchmarks/results/df-source-force-20261025/`.

## Remaining large work

Exact-zero compaction helps streamed MD but cannot remove nonzero negligible
sources or improve resident PBE preparation. Force scheduling removes a separate
warm/cold cost but does not reduce SCF integral work. The next investigation is
current-density Hermite norm bounds with an aggregate charge/projection error
budget and explicit metric amplification, followed by a work-aware comparison
of streamed MD versus resident raw A for PBE96.

See `../../proposed/2026-10-25-density-weighted-df-source-bounds.md`. Do not promote
old bare-pair threshold probes or lower the public DF memory allowance solely to
force a faster benchmark path. Their independent validation, resource issues,
negative experiments and raw receipts remain in the research artifact tree.
