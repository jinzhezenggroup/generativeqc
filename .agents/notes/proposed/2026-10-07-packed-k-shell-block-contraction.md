# Packed restricted-K shell-block contraction

Status: qualification candidate; no default-performance claim
Date: 2026-10-07

## Problem

The incumbent generated Direct-K path already reuses prepared primitive-pair geometry,
Schwarz metadata, density screening, and shell-pair streaming.  For packed low-order
classes, however, the final contraction remains ERI-component centric: each retained
canonical Cartesian component enters the generic eightfold symmetry scatter, reads one
density scalar per surviving permutation, and may issue one global atomic add for each
output contribution.

GPU4PySCF's Rys-K path instead contracts a small ERI block against a cached density tile
before publishing K elements.  The distinction is contraction granularity, not the
recurrence family or prepared-owner lifetime.

## Candidate

For restricted raw-K only, packed workers whose complete forward/transposed K block
footprint is at most 32 doubles accumulate the eight exact symmetry channels in
lane-local shared storage and publish each K block element once after all Cartesian
components have been contracted.

The first bounded domain covers the low-order packed classes whose footprint satisfies
that limit (including psss, ppss, psps, and dsss under the current shell catalog).

Preserved behavior:

- identical primitive-pair recurrence and component integrals;
- identical Schwarz and density-conditioned shell admission;
- identical eightfold ERI uniqueness conditions, including diagonal pairs and
  pair-swapped duplicate suppression;
- unrestricted K, Coulomb J, combined/HF-weighted K, and larger packed classes retain
  the incumbent component scatter;
- no new threshold, approximation, precision mode, persistent allocation, or public API.

The additional shared storage is statically bounded to 32 doubles per packed lane.
This branch is not a production promotion.  It requires generated-source tests, CUDA
compilation, independent fixed-density raw-K matrix gates, resource/occupancy checks,
and matched fixed-density/complete PBE0 timing before any default decision.

## Qualification

Measure at least:

1. exact per-class raw-K parity against the incumbent path;
2. shared-memory/register occupancy for the affected generated kernels;
3. fixed-density 48/96-atom K wall/device time under the same density and admitted
   shell-task counts used by #2060;
4. complete cold PBE0-RKS timing with unchanged iteration history;
5. fallbacks for UHF, HF-weighted exchange, unsupported classes, and constrained
   generated-owner coverage.

A win must come from reduced density/scatter traffic at unchanged scientific work, not
from altered screening or SCF trajectory.

Refs #2015, #2020, #2059, #2060.

Agent: ChatGPT
Model: GPT-5.6 Sol
