# Decision: Differentiate the native D3 owner in the CUDA stationary Hessian

Status: implemented
Date: 2026-11-21

## Decision

D3 is geometry-only but its coordination-dependent C6 is not constant with
respect to the nuclei. Borrow the correction's immutable native atom and table
storage under the same final-state authentication as the electronic result.
Build per-atom CN and Gaussian weight jets, differentiate pair and optional ATM
gradient legs and CN cotangents, then apply the CN pullback. Three ordered GPU
launches let each atom own its FP64 output without inter-atom FP64 atomics. The
correction is composed with the electronic candidate before the existing
publication gate. This avoids repeating CPKS, recomputing an external D3 model,
or allocating a dense correction Hessian. The CLI and Python retain the named
program identity; a separate Python-side dispersion owner is not interchangeable.

## Invariants

- Preserve the primal switch and reference-weight policies. A vanishing or
  reciprocal-overflowing Gaussian norm selects the max CN reference and a
  locally constant jet. Compute the D3(0) overflow-branch curvature separately
  in the log domain: the radial lane can round to zero before its derivative.
- A hard cutoff at an active boundary is not a differentiable model. Reject
  the request without changing the caller's HVP or dense matrix storage.
- Per direction, the D3 arena adds `24N` doubles for atom jets/cotangents and
  `3N` for its action. The work gate reserves `52N(N−1)` pair visits/terms and
  for ATM another `150N(N−1)(N−2)/2` incident-triangle terms. These are
  conservative bounds, not throughput measurements. Linear storage is not a
  justification for admitting unbounded cubic work.

## Qualification

On RTX 5090 (Slurm job 1666), the matrix difference between corrected and bare
electronic endpoints matched a two-step Richardson stencil of independent
`dftd3==1.4.0` gradients. The maximum absolute error was `6.12e-17` Hartree/Bohr²
over ten Direct and full-rank DF cases on STO-3G water: PBE RKS with BJ, BJ+ATM,
and zero damping, and PBE0 UKS water cation with BJ and zero damping. The ATM
signal was `1.93e-9` Hartree/Bohr², above the `2e-10` component gate. Each dense
result reported all nine correction directions; the Python HVP reported one
completed direction and matched the unmodified CLI matrix action. Tests also
check raw symmetry and translation nulls. This comparison isolates correction
curvature; it does not extend the independent electronic qualification.

The same build preserved both output buffers at an exact hard pair cutoff and
passed both existing PBE-D4 frontend cases. Job 1667 passed the eight r²SCAN and
r²SCAN-3c spin/provider frontend regressions. Native host tests separately
cover smooth-switch/CN response, the Gaussian fallback, and the D3(0) limit
where energy and radial gradient underflow independently of curvature. The
artifact was built against local worktree changes on `f8e0945d0`; retained
evidence is `build/hessian-gpu-sm120/d3-final.xml` and the corresponding
`d3-final-gpu.log`, `d3-r2scan-regression-gpu.log` and `d3-numeric-cache-*.log`.
