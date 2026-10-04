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

## Revisit

Promote only after independent numerical gates, observed semantic work and
complete cold/warm/moved results show a useful tradeoff. If repeated scans or
unchanged register pressure erase the gain, retain the evidence as a rejected
experiment instead of carrying an unqualified default.
