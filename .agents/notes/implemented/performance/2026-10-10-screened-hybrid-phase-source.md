# Decision: compact screened Global-Hybrid phase sources

Status: implemented
Date: 2026-10-10

KS auto eligibility and its screened-SPD cap are superseded by the
[2026-10-11 opt-in cold-auto decision](2026-10-11-screened-hybrid-cold-auto.md).
The forced-source measurements below are unchanged historical evidence.

## Problem

The retained 96-atom water/PBE0/def2-SVP endpoint has 768 spherical and 800
Cartesian AOs. Its dense canonical FP64 source would require about 410 GB,
so increasing the original 8-GiB cap without changing layout cannot admit it.
The incumbent Direct owner already prepares geometry-only sorted Schwarz rows
at the endpoint's unchanged 1e-12 threshold. Their 28 angular bucket tails
contain 1,873,221,716 values (14,985,773,728 bytes), still above 8 GiB.

## Decision

Freeze the incumbent row-tail counts once, before optional value allocation.
Retain their small host inventory and use identical compact bucket offsets and
work extents in construction and replay. This reuses the existing integral
recurrences, Schwarz predicate, angular ordering and canonical orbit scatter.
Lower replay thresholds are invalid; higher masks retain the prepared source
identity and may screen further. Unscreened RHF retains its existing layout.

Ordinary Direct admission is allowed to omit optional Schwarz rows and still
contract the exact incumbent mask. Compact-source admission must instead
reserve their sort/scan inventory. Query the existing CUB workspace before
constructing a positive-mask source, and add the shared Direct row-capacity
formula to the complete provider bound. Otherwise a small f-shell source can
legitimately have canonical storage but no screened rows. Retain the sizing
refusal/driver distinction; an explicit source request cannot bypass it.

Use a canonical-only zero-order Direct factory for phase sources. An ordinary
SPD provider prioritizes its generated J/K leases before canonical storage,
and can spend even an enlarged budget on operators the phase source never
uses. The real 96-atom forced arm exposed this legitimate row refusal; the
benchmark's positive-construction gate rejected that fallback instead of
calling it reuse. The source-specific factory shares the existing preparation
and kernels but skips generated J/K caches, MD and range leases. Ordinary
provider priority, derivative consumers and automatic RHF eligibility stay
unchanged. This also avoids duplicate optional small-system ERI caches inside
the source clone.

The method-neutral lease defaults to an 8-GiB value ceiling. KS `auto` preserves
that cap and its prior conservative eligibility. Explicit opt-in KS `1` may
admit up to 16 GiB, still under complete numeric admission and a 256-MiB
free-device reserve. Public resource ledgers without a reserved optional-source
allowance keep the exact fallback. Capacity refusal is not a numerical error.

Source metadata already consumes live device memory. Recheck free memory after
its construction, reserving only not-yet-allocated compensation/audit storage
and the unchanged 256 MiB runtime margin. The complete numeric budget still
charges the conservative preparation peak. KS borrows its existing public J/K
slots instead of duplicating them. The source-only factory retains immutable
row prefixes/order but frees sort keys, input order, offsets and CUB workspace
after the sort fence. Ordinary provider staging/priority is unchanged.

The first admitted 96-atom cold E+F run exposed a later-phase resource limit:
after the gradient, live device memory available to a new source fell by about
7 GB. Closing the idle Python stationary force execution recovered only about
200 MiB and did not admit the next source (Slurm job 7219). Its reported
additional-device budget is 512 MiB, not 7 GB. Reject that unrelated cache
eviction rather than change default force lifetimes without resolving the
dominant owner. No force-cache eviction is retained. The exact fallback stays
necessary for later phases; cold source qualification does not qualify warm
or moved source reuse. Do not attribute the remaining memory to a particular
driver/runtime allocation without allocation-level evidence.

## Rejected alternatives

- Dense storage with a larger cap still needs hundreds of GB for this case.
- Relaxing screening, grid, convergence or FP64 would change the endpoint.
- A density-dependent compressed cache would change source identity during SCF.
- A molecule-specific branch or reference-density seed would not qualify the
  general cold native path.
- Treating E-only success as E+F would omit the existing stationary force owner.
- Reducing the 256 MiB runtime reserve to overcome a few missing MB would hide
  duplicate/dead allocations rather than solve resource ownership.
- Evicting only the prepared stationary force owner does not resolve the
  measured later-phase capacity limit and needlessly loses prepared reuse.

## Invariants

- Geometry and preparation mask determine the immutable compact topology.
- Construction/replay use the same extents and completed canonical census.
- Source storage retires before derivatives and subsequent solves/geometries.
- Full source setup and preparation count in complete endpoint time and work.
- No default promotion, lower precision, new integral math or CPU oracle in
  production is introduced.
- Every device gate runs in a finite Slurm allocation preserving visibility.

## Evidence

Slurm job 7183 on node1 RTX 5090 established the incumbent screened extent
without allocating ERIs. Runtime numerical/endpoint qualification is retained
separately in `.artifacts/hybrid-phase-values-96-20261010/`, with matching
source patch, library/AOT hashes, raw outcomes and independent reference.
Earlier energy-only evidence under `.artifacts/hybrid-phase-values-20261010/`
does not qualify this layout or the new capacity ceiling.

### Source-matched 96-atom E+F qualification

Final source identity:
`f45113100414107bde0b9c4d51ce78f98b013aede7c647d5e51f0e6e01520fdb`.
Final native library SHA256:
`e178fcead5b4e0eda223b4d732122d7097f5efa5a06ca0f680d8996f7cc41c07`.
The local inventory, uploaded final source and embedded build identity agree.
Slurm job 7228 used the matching Release/sm120/AOT-auto binary and stationary
PBE0 assets on node1 RTX 5090. Both arms completed the unchanged public Python
E+F protocol and independent energy/force gates for all four phases.

| Phase | Source disabled E+F (s) | Forced source E+F (s) | Fock actions, disabled / forced | Actual source reuse |
|---|---:|---:|---:|---|
| Cold, including preparation | 93.819820 | 75.361041 | 19 / 17 | 1,873,221,716 values, 17 audited replay actions |
| Warm | 16.156740 | 24.670453 | 1 / 3 | No; capacity refusal, exact fallback |
| Moved | 63.648104 | 64.935340 | 12 / 12 | No; capacity refusal, exact fallback |
| Moved-warm | 16.136211 | 17.327509 | 1 / 1 | No; capacity refusal, exact fallback |

Cold construction completed once, with 14,985,773,728 source bytes and no
integral recurrence on its 17 replays. Complete operator inventories remained
strict FP64. Across both arms, maximum energy error was `9.55e-12 Eh` and
maximum force-entry error `3.76e-11 Eh/Bohr`, versus independent gates of
`1e-8` and `1e-7` respectively. No reference density entered production SCF.

The single ordered cold sample ratio is 1.245; the complete four-phase sample
ratio is only 1.041. Changing Fock counts, node activity, warm regression and
the absence of repeated/ABBA sampling preclude a robust speedup or default
promotion. This is **cold 96-atom source/E+F qualification**, not qualification
of resident reuse for every phase. The original all-phase positive-source gate
correctly failed jobs 7217/7219 and remains retained. The explicitly cold-scoped
observer additionally requires positive audited replay actions and bytes, then
records later fallback status rather than relabeling it as source reuse.

Host selection: 683 passed, 58 device-gated skips; dependency/promotion checks,
Ruff, new-file Clang formatting and focused post-format lease checks passed.
Job 7220 passed four water/methane aug-cc-pVTZ Hybrid cases (chunk 1/2, signed
J/K, both PBE0/B3LYP energy, warm/seed, ledger and low-budget gates) and three
RHF geometry/independent-energy cases, including the previously refused SPD
case. Its full water probe passed Compute Sanitizer memcheck with zero errors.
After whitespace-only source formatting changed the build fingerprint, job
7229 rechecked both water chunk widths against the final binary. Source-only
staging retirement and borrowed outputs are covered by those numerical/device
checks and the host ownership tests.

The complete patch, matching binaries/AOT/harness hashes, independent reference,
raw successes/failures and reproduction procedure are in
`.artifacts/hybrid-phase-values-96-20261010/REPRODUCE.md` and its sibling files.
Earlier artifacts remain unchanged; no GitHub publication, commit or release
was performed.

## Consequences

The 96-atom source can fit a 32-GiB device but still requires significant setup
and memory. Explicit forced warm phases rebuild it as designed; it need not be
profitable for warm solves. A compact layout is memory- and work-bounded by the
same geometry-only screened census, not a reduction in the contraction order.

## Revisit when

Whole-endpoint repeated cold/warm E+F measurements support wider/default
admission, or the public planner reserves optional source storage.

## References

- Supersedes the capacity policy in `2026-10-10-hybrid-phase-source-lease.md`.
- `docs/developer/scf_module_boundaries.md`.
- `tests/native/ks_phase_values_probe.cpp`.
- `benchmarks.readme_pbe0` (compiled stationary high-level E+F endpoint).
