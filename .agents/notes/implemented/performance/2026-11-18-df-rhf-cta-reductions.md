# Decision: give fitted RHF energy, convergence and DIIS owners a full CTA

Status: implemented
Date: 2026-11-18

## Problem

An H100 nsys capture of the README water-96 DF-HF endpoint (768 AOs, one cold +
three warm calls) attributed 0.68 s to `update_diis_kernel`, 0.24 s to
`update_device_convergence_kernel` and 0.13 s to `compute_device_energy_kernel`:
50 ms, 17 ms and 7 ms per iteration. Each owns one system with a single 32-lane
warp. The cooperative DIIS owner also performs its history copy and Fock
extrapolation over n^2 elements with those 32 lanes, so a warp cannot approach
memory bandwidth at production AO counts.

## Decision

`kDfMatrixReductionThreads = 512`. The DF RHF energy, convergence and warm-seed
energy reductions use one 512-thread CTA per system, a fixed thread-strided
partition and a fixed warp-then-block combination order. The cooperative DF
DIIS owner (`GENERATIVEQC_DF_DIIS_DOTS=auto`, the default) runs the existing
`WideVectors` instantiation with collective dots folded in warp order. The
convergence policy, its nonfinite handling, the resident density advance, the
DIIS solve and dependent-history retirement are unchanged. The DIIS owner is
shared with DF-UHF, whose two spin blocks form one wider vector; its separate
energy/convergence kernels are unchanged, as are Direct/KS owners and the
32-lane launch wrappers.

Sums are deterministic for a given width but not bitwise equal to the warp
order; the warm-seed energy baseline uses the same width as the SCF energy so
the two remain comparable.

## Evidence

`benchmarks/results/h100-df-rhf-cta-reductions-20261118/`. Per-kernel medians:
DIIS 50.0 → 3.0 ms, convergence 17.2 → 0.46 ms, energy 7.3 → 0.29 ms.
Water-96 DF-HF endpoint and qualification: see the results README.

## Rejected alternatives

- Routing DF through the cached ordered-Gram DIIS owner: changes the dot order to
  the generated scalar order, needs new DF state, and that owner's single-CTA
  pending-row Gram is itself 8–16 ms per iteration at these sizes.
- Multi-CTA partial sums: the single wide CTA already brings these kernels below
  0.5 ms (DIIS: 3 ms), so a second pass and scratch are not justified.

## Revisit when

Batches of many small systems dominate: one 512-thread CTA per tiny system may
then waste occupancy relative to the warp owner.
