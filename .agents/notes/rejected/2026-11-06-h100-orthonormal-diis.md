# Decision: reject the H100 restricted orthonormal-DIIS cold candidate

Status: rejected
Date: 2026-11-06

## Problem

The H100 complete cold E/F backlog in #3004 still includes HF96 Direct and the
remaining WB97M-V Direct reference gap. A metric change was proposed to form
`X.T @ (FDS-SDF) @ X` for restricted FP64 DIIS history while preserving physical
AO convergence, canonical checks, operators, grids and analytic-force semantics.
The unmerged d785 candidate covers Direct/DF RHF and restricted KS, with existing
scratch ownership and explicit selector 0/1 control.

## Decision

Do not promote this candidate. One bounded H100 pilot completed WB96 Direct and
HF96 Direct, each with baseline/candidate/reference fresh owners and cold plus
three own-cold-density warm E/F calls. All 24 outputs independently qualify and
both warm guards pass, but cold work remains 18 iterations/Focks for WB and 11
for HF. Paired cold reference/candidate ratios are 0.673100415648897 and
0.5404812097250224. The small one-call native timing differences do not establish
a meaningful cold solver gain. Three-round, five-case and coherent 30-case acceptance
were not reached; the 30/30 goal remains unmet.

The candidate patch is retained as historical evidence and reconstructs the
exact unmerged tree. No experimental production code or default is added to
main. This H100 result adds its own source/image/device scope alongside the
related A100/5090 rejection in #3019, without rewriting that owner's evidence.

## Numerical and provenance boundaries

The original independent energy/force gates remain 1e-5 Eh/atom, 1e-4 maximum force
and 3e-5 RMS force. Reused dense/CPU-spot references retain their original source,
package and grid-stability contracts. No oracle is regenerated or used as native
SCF input. Native settings and source/image identity agree between selectors.

The pre-data v2 amendment makes the tighter native/native thresholds diagnostic
for this DIIS trajectory change alone. WB misses max/RMS diagnostic thresholds
in all four calls; HF passes. All values and original flags remain in the
evidence. These observations do not reclassify #3019's initial parity failures or
waive strict independent algebra, Gram/scratch or physical convergence gates.

The original small admission attempt remains failed: four HF owners passed,
then PBE0 selector 0 lacked a runtime LibXC asset before E/F. The repaired readonly
runtime tree adds exact tracked upstream inputs to unchanged frozen source bytes;
native/AOT images are unchanged. A separate continuation reuses those four HF
receipts and executes only PBE0 off/on. Its six-owner admission coverage spans
two attempts and supplies neither performance nor independent accuracy votes.

## Evidence

- Source `d785a10d1e3d43e7e8167c7efa5c78a3b6f3594c`, base
  `32b4942de70ff4ac6ec4701063f4b537daf096a9`.
- Native identity `4b7a246cb162b8dfbd72a43a82a1f4788e85eea0f6eafb82dd85601a0c8d0a5f`;
  core SHA256 `2610999fc1d6d714e58af533a546b5cd7534ad2523322dd172d00a9878796cbf`.
- H100 UUID `GPU-d36c3a44-fb07-cfa1-bf66-7872e07d3366`, MIG disabled.
- [Complete retained pilot and offline reader](../../../benchmarks/results/h100-orthonormal-diis-rejection-20261106/README.md).
- [Related A100/5090 rejected rationale](2026-11-06-wb97mv-minao-orthogonal-diis.md).

## Revisit when

A materially different bounded solver change demonstrates fewer necessary
physical Focks, initializer/outer actions or operator work on the affected
complete endpoint, while retaining the original independent scientific and warm
guards. Repeating this unchanged metric candidate or pooling old timings does
not supply that evidence. Coordinate any follow-up through #3004.
