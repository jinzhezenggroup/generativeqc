# Experiment: same-source PBE0 force-policy baseline

Status: host-qualified integration; complete GPU endpoints pending
Date: 2026-10-04

## Frozen composition

Master `e79efea00`, ordinary force AO consumer #1833 `4c5aa3eb6`, and compiler
Becke phases #1830 `aae46734b`. #1767's indexed Schwarz force pages are already
in master, default off. No #1798 approximate AO-product screen, paused #1803
incremental execution or DF method substitution. #1831's separate JIT-root
cleanup is deliberately not added, so this comparison does not silently include
a new cold-only transformation.

The integrated default keeps full AO membership, the original explicit tile,
and the original unindexed bounded force schedule. Compare on one binary/GPU:

1. Default.
2. Indexed force request plus zero AO-map budget: exercise dense fallback.
3. Indexed force request plus local force AO maps at explicit 1e-16 cutoff.
4. The same local/indexed request plus compiler-planned phased Becke.

The zero-cache row is a fallback control, not an exact isolated paging ablation:
it also performs optional-cache lookups that return dense membership. Cutoff
selection is not a certified force-error bound. SCF AO policy is unchanged.

## Measurement and qualification

`benchmarks.readme_pbe0_integrated` calls the same scientific README runner,
recording requested policy separately from actual normalized work. Every force
call checks native integral route/timer, complete grid batches, local AO work
or dense fallback, and actual phased batches before another replay. The
canonical normalizer is used directly; a host mutation test reproduces the
prior missing-timer failure rather than shadowing a module-local import.
There is no endpoint indexed-page counter in the present ABI. That request
remains explicitly unmeasured; neither task counts nor timing imply FLOPs or
prove a page count. Do not substitute model capacities for observations.

First run a coarse-grid 48-atom molecular numerical preflight of combined
phases/local AO and zero-cache fallback. Its reference and timings are not
README evidence. Then generate a fresh full-grid GPU4PySCF reference and run
all four complete96 policies, each cold + five warm + moved + five moved-warm.
Compare every native row with every same-geometry reference repeat at unchanged
1e-8 Eh / 1e-7 Eh/Bohr gates. Keep real XC backend flags, failed attempts,
source/library hashes, cache receipts and finite Slurm provenance.

These are ordered process runs, not interleaved causal speedup estimates.
Process/geometry cold includes preparation but does not clear shared compiler
or driver caches. Retain SCF iteration and fallback differences rather than
normalizing them away. Do not splice old jobs 5728/5729 into this campaign.
The main run uses n1, never n3; n5 is used only for the verified-ccache build.

Combined host integration passes 573 tests; a separate, overlapping observer
and output-retention suite passes 60. Compiler structure checks 424 modules
with zero dependency errors. This is not a current-source GPU pass or evidence
that the combined performance equals independent percentage improvements.

## Next structural experiment

The code already contains fixed-angular-order workers in
`src/scf/cuda/direct_angular_force.cu`; do not claim to invent that math or
blindly switch back to its topology-capacity queues. The current full-range
independent-J/K provider in `direct_coulomb.cpp` calls the mixed bounded
enumerate/screen/drain kernel instead. Reuse the qualified contraction helper
while investigating compiler-planned bounded homogeneous workers that remove
unrelated high-order instantiations from each kernel's resource footprint.

Class-isolated historical evidence prioritizes DPPP/DPPS, with DDDP/DPDP also
important; it is not an additive endpoint decomposition. A split schedule may
repeat screening, so charge all scans/launches and preserve complete coverage,
separate J/K, symmetry and fallback. Static stack/register limits are not
dynamic spills; #1787 arithmetic calibration cannot predict these irregular
recurrence kernels without matching work and hardware evidence. Prefer this
bounded experiment over repeating rejected scalar weighted DPPP slabs, another
small moment cache, or a supposedly new J/K fusion already in production.
