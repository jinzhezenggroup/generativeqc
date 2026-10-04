# Experiment: partition bounded full/LR force work by angular order

Status: proposed (implemented opt-in; qualification pending)
Date: 2026-10-04

## Motivation

The [12-atom full-TZVPD diagnosis](2026-10-04-tzvpd-warm-integral-diagnosis.md)
records 33.264 s in the two bounded derivative launches, with 255 registers per
thread and 256 threads per CTA. This is evidence of a theoretical register
occupancy limit, not measured spills or proof that increasing occupancy helps.
The same profile attributes 23.463 s to canonical full/LR values; this force-only
experiment cannot remove the repeated value cost of cold SCF.

## Candidate and ownership

`GENERATIVEQC_BOUNDED_ANGULAR_FORCE=1` (also `angular`) opts the prepared
`GeneratedExchangePlan` into thirteen disjoint total-angular-order passes,
0 through 12. Each pass specializes the existing bounded CUDA consumer by order
and radial operator. It shares the original recurrence, shell/AO orientation,
screening and bounded queue; it introduces no new integral algebra. Full-range
independent J/K sources and the omega=0.3 LR source use the candidate. Other
omega values and the existing fused/short-range paths retain their old route.

Full-range forces retain their optional indexed block domain. LR deliberately
retains its triangular domain, so adding LR indexing is a separate experiment.
The new switch participates in resource/checkpoint provenance. Older checkpoints
may omit it but need explicit warm admission; absence does not mean the new
schedule was selected when the seed was made.

The normal default remains the original single traversal. No performance or
correctness claim for the candidate is established by writing these templates.

## Work and storage

Let B be the number of enumerated block pages, C the number of shell candidates
in surviving pages, and W(q) the actual admitted derivative/contraction work for
shell quartet q. These are semantic counts/weights, not FLOP counts. Ignoring
launch and reduction overhead, the original scheduling work has the form

    O(B + C) + sum_q W(q).

This diagnostic intentionally changes that to

    O(13 B + 13 C) + sum_order sum_{q: order(q)=order} W(q).

The integral subsets are disjoint; the enumeration and screening are repeated.
The exact executed B, C, primitive/AO admissions and FLOPs have not yet been
observed and must remain absent/null in measurements. This is not a claim of
reduced algorithmic integral work. Empty angular orders are still launched.
Large or strongly screened molecules can therefore regress even if smaller
classes have fewer registers. A later class index is justified only by evidence.

Global retained storage is unchanged: the same cursor is reset sequentially on
the owning stream, the same output has O(number of atoms) entries, and each CTA
retains the same fixed-capacity shared queue. Compiler register/local-memory
usage and increased binary size still require measurement; no register cap or
spill policy is imposed.

## Scientific and operational gates

- Both schedules must pass the same independent CPU-qualified full/SR/LR
  derivative tests for restricted/unrestricted densities, including repeated
  centers and mixed f/d/p/s shells. This mixed fixture exercises orders 10/11
  absent from a two-shell s/f fixture. Source and finite-difference tolerances
  remain 3e-10 and 3e-8 respectively.
- Angular/radial specialization must not change J/K coefficient meaning,
  SR=Full-LR reconstruction, translational symmetry, active-system selection,
  or source signs. Atomic accumulation order may differ; complete molecular
  gates remain 1e-8 Eh and 1e-7 Eh/Bohr for every retained call.
- Compare off/on in the same binary on the same Slurm-assigned RTX 5090, with
  fixed self-converged native density. Checkpoint-seeded diagnostic replay is
  labeled as such and cannot substitute for unseeded cold or moved endpoints.
- Record per-order durations/resources and complete E/F timing; launch count
  alone cannot prove less source work. Check a larger size before promotion.
- Build only with verified ccache; keep failed builds and numerical runs.
  Node3 is excluded by the user's overheating instruction.

## Alternatives excluded from this experiment

Do not re-enable the previously rejected bounded through-f **value** route,
change force screening, truncate diffuse/f shells, or add preliminary LDA.
Those changes would confound the angular scheduling comparison. In particular,
the accepted 12-atom LDA lifecycle made prepared cold 15.12% slower despite fewer
target iterations. Existing VV10/Becke and incremental-J work are separate.

## Frozen build and partial endpoint evidence

Commit `85e7a7233` builds on n1 with verified CMake/Ninja and ccache. All 451
compiler commands use the cache launcher. The production source identity is
`5a95de9ca90911010458d2d70b5d5abcf3f8102028a04c058a91583d43a047b3`
(1,373 inputs), and the library SHA-256 is
`d4e316e1a38d43eb97d93cf0070e73f241ac9de8423f0381cdc30b5c36e2a3fe`.
The native test source/executable and checkpoint test bytes are sealed separately.
There are 65 passing host cases, five optional checks skipped, passing repository
hooks, and 108 still-accepted historical README calls. The incremental build
records 34 cache hits and nine misses without clearing the shared cache.
n1 Slurm 5730 passes both `--range-response-only` and `--through-f-response`,
including independent mixed f/d/p/s full/LR sources, both spins, batch/prefix
budget edges and both AO representations. Five checkpoint host cases also pass.
These correctness checks do not establish endpoint performance.

The **linked** library's `cuobjdump --dump-resource-usage` reports 254--255
registers/thread across all 58 bounded-kernel instantiations; low-order candidate
workers report 255. An earlier **relocatable object** reported only 82 for those
low orders. The linked call graph defeats that preliminary occupancy hypothesis:
do not cite the object count as the executable's register requirement.

For the restricted full-range source, the linked original kernel reports a
90,408-byte stack, versus 2,608 for order 3, 1,024 for order 5, 26,808 for order 7
and 34,424 for order 8. These are compiler resource reports, not observed stack
traffic, spills, achieved occupancy or runtime memory peaks. The cost of thirteen
scans must be checked against any benefit from the compiled call/stack structure.
Original receipts,
failed missing-tool setup, source patches and resource dumps remain under the
ignored `.artifacts/tzvpd-angular-force-20261004/` experiment directory.

Slurm 5734 on n1/device 1 completes the 3-atom off/angular/angular/off diagnostic.
Each process imports the identical native density archive, performs one real
target priming solve (two iterations), then freezes the resulting native state.
Each of the four runs records five one-iteration complete warm E/F calls. Across
the ten samples per schedule, off/angular medians are **1.919525 / 2.479084 s**:
the candidate is **29.15% slower**. Reported integral-derivative stage medians are
0.876914 / 1.433977 s. All 24 priming/warm calls pass independent reference gates;
maximum errors are 9.948e-14 Eh / 9.136e-12 Eh/Bohr. The same binary, Slurm
allocation, visible device and donor-density digest are checked. Executed shell
and primitive work and FLOPs remain null; the stage timings do not isolate scan
traffic. This is negative small-system evidence, not a new README plot point or
a comparison to GPU4PySCF timing. The 12-atom continuation is still running.

Failed diagnostic attempts are preserved: 5730/5731 encountered a missing
`ptxas` companion to the ccache NVCC wrapper (the first runner masked it by
formatting absent force diagnostics); 5732 completed valid 3-atom E/F but the
public checkpoint exporter rejected WB97M's unsupported accuracy-model facade;
5733 successfully transferred native density but wrongly required the initial
target priming solve itself to take one iteration. The corrected private runner
uses the same native descriptor/validation API as `ks_preliminary_density`, seals
the method/basis/geometry/binary and array hash, permits a real independently
checked priming solve, and requires all five measured replays to take one
iteration without fallback. No reference density or relaxed E/F gate is used.
All failed Slurm outcomes stay failures, including 5730 after its passing native
checks; missing diagnostics stay null. This does not qualify public WB97M
checkpoint support or a public initialization policy.

## Revisit

Promote only after independent numerical gates, observed semantic work and
complete cold/warm/moved results show a useful tradeoff. If repeated scans or
unchanged register pressure erase the gain, retain the evidence as a rejected
experiment instead of carrying an unqualified default.
