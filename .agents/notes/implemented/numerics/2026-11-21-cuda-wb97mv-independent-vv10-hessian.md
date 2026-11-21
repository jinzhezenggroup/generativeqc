# Decision: Differentiate each accepted WB97M-V quadrature on CUDA

Status: implemented
Date: 2026-11-21

## Problem and decision

One combined grid response cannot differentiate a state obtained with two
different quadratures. Bind the actual native primary and VV10 quadratures
to the same canonical KS state and reuse the generated point and pair AD
kernels independently. For distinct grids, only the VV10 owner allocates
pair storage and evaluates ordered pairs or the local VV10 beta term.

Sum fixed-density nuclear Fock derivatives before building the CPKS RHS.
Every induced action sums both density responses before physical Fock
assembly. After the single solve, both moving-grid owners contract against
the same solved density response and sum their mixed nuclear contributions.
The combined-grid case keeps its existing one-owner schedule.

## Invariants

- Authenticate actual geometry, quadrature, density and producer stream, not
  merely the model ID or point count. Neither a primary-grid substitute nor
  an independent VV10 grid rebuilt by the response owner is admissible.
- Keep the active VV10 domain fixed at the central SCF state, on both legs.
  Include each mesh's AO/weight motion and the VV10 point-coordinate motion.
- Reserve mandatory geometry tiles for both meshes under the geometry cap;
  the pair-capable and pair-free response layouts likewise share one response
  grid cap. Replay remains work-bounded across the requested action block.
- Publish only the complete finite and authenticated result. Do not change
  the private/public Hessian C ABI or add a numerical host fallback.

## Evidence

Job 2069 passed 15 fixed and moving grid component tests, including signed
density motion, spin coupling, zero-VV10 domains, and independent Libxc and
real-space VV10 comparisons. Job 2070 verified linked Python/CLI parity for
both operations with different primary and VV10 product grids. The independent
Direct RKS displaced-energy matrix differs by at most `9.81e-10 Ha/Bohr²`.
The first center-energy replay omitted the second-grid energy in the test's
return expression; the corrected oracle uses both independent energies and
retains the original density, Fock and Hessian gates.

Job 2071 passed all 26 selected real-device regressions. The independently
reconverged DF/fitted-LR UKS two-grid matrix differs by at most
`4.69e-9 Ha/Bohr²`. The product-grid matrices and stencils are retained
under `build/hessian-gpu-sm120/vv10-final-receipts/`.

Jobs 2072–2074 also exercise the shared minimum geometry reservation and
the public tight Treutler policy. One byte below the joint minimum leaves
caller storage intact; the exact minimum uses bounded AO replay on both
owners and agrees with the resident HVP. For this H₂ geometry, level 3
retains 17,938 active VV10 points (321,771,844 ordered pairs per action), so
the normal cap rejects it before response pair evaluation and reports the active
count. With an explicit 350,000,000-pair cap, Python tight XC and CLI
`--grid-level 4` agree on the HVP, both for default level-3 VV10 and the
explicit level-0 override. The larger test cap does not change defaults.

Job 2076 passes six targeted regressions on the final build and explicitly
rejects output aliases to the atom offsets and radii of either v4 grid.
Its action-only DF/exact-LR Na(ECP)H HVP agrees with the saved dense product
to `3.06e-16 Ha/Bohr²`. The final device receipts are under
`build/hessian-gpu-sm120/vv10-topology-receipts/`.

The independently reconverged DF/exact-LR Na(ECP)H matrix uses Treutler
level 1 for XC and level 0 for VV10, with different atom offsets on both
quadratures. The worst error over all 36 entries is `1.51e-9 Ha/Bohr²`.
The same receipt passes the two central quadrature, Fock and density gates,
raw symmetry, translation nulls and dense/HVP parity without relaxed tolerances.
