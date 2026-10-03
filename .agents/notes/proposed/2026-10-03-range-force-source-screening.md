# Proposal: screen range-exchange force tasks by their requested source

Status: proposed; numerical and complete endpoint qualification pending
Date: 2026-10-03

## Problem

The 96-atom WB97M-V complete force endpoint spends about 222 seconds in integral
derivatives. The bounded SR/LR derivative launcher publishes only exchange,
but passes the mixed J/K demand to its queue. Even when a caller explicitly
requests K alone, the force-product screen still admits a quartet from its
Coulomb density product. Distant Coulomb blocks can therefore retain tasks whose
exchange products fall below the existing force cutoff.

## Candidate and boundaries

Honor the existing source masks in the force-product gate, and set the SR/LR
exchange-only derivative launcher to K demand for both RKS and UKS. Keep each
same-spin exchange product independent; do not use cancellation between signed
J/K coefficients or between spin channels. The mixed J/K default and unrelated
full-range consumers retain their existing selection.

The same demand also selects the existing raw-K linear bound. In restricted
calculations it has unit weight, rather than the mixed Fock's one-half weight,
so this first gate can retain some exchange tasks the old mixed bound rejected.
The new overall task set is not claimed to be a strict subset. Screening
thresholds remain unchanged, including `min(screening, 1e-14)` for force density
products. Recurrences, radial operators, FP64 precision, coefficient/scatter
contracts, buffers and bounded/generic fallback availability do not change.

## Evidence and promotion requirements

The isolated branch starts at master
`cd08953d5765d389baba62935f2025816be57075`. Its clean Release CUDA 12.9/sm120
baseline library SHA-256 is
`4e392ae19aa94656f031ef8104a4f7d1c4b4fd5a9417a41ea2bcf5111ae942f9`.
Explicit ccache launchers are verified in all 441 compiler commands; exact build
and cache receipts are retained in `.artifacts/exchange-force-screen/`.

A host harness executes the actual scalar screening function. The baseline
fails the K-only force-product case; the candidate passes 42 checks covering
Coulomb-only, exchange-only, beta-only UKS, independent source admission and
neighbors of the force cutoff. This demonstrates source selection, not the
accuracy of a complete force or a performance improvement.

Before promotion require independent real-device full/SR/LR derivative gates,
RKS/UKS complete energy/force and displaced-geometry gates, and same-allocation
baseline/candidate complete 24/48-atom timings followed by a 96-atom check.
Retain every sample, convergence/work metadata and 1e-8 Eh / 1e-7 Eh/Bohr gates.
Do not infer the number of executed shell quartets from logical capacity or
from canonical-source counters, which do not observe this bounded route.
