# Keep wide Cartesian grid sweeps bounded without discarding small products

The 96-H / Cartesian STO-3G / r²SCAN qualification uses the same unpruned
12 × 6 × 12 grid as its independent GPU4PySCF references. It has 82,944
points and 378,224,640 unordered Becke point/pair visits per sweep. This
exceeds the small default work allowance; it is not an atom-count or a GPU
address-space limit. Older profiling runs occupied up to 7,942 MiB
(Direct) and 6,214 MiB (DF) on cards reporting 32,607 MiB, without completing
the matrices. These are not completion peaks or performance results.

Two separate fallbacks matter:

- The factored XC left contraction can exceed its per-batch affine work
  allowance for 288 rows. Split into at most two batches, charge the
  right-adjoint build only once, and keep the original geometry materialized.
  Both batches use paired left/mixed weights. A singleton uses a different
  weight layout and cannot be inserted into a split column.
- An unscaled wide Becke product can have a representable total but many
  atom products with too small a prefix to qualify for unscaled exclusive
  derivatives. Routing these points to scalar repairs repeats the grid
  normalization for 288 rows. DF exposed this as an apparent CPKS failure:
  the generated finite check overwrote the upstream reason after the grid
  handed it a rejected RHS. Preserve the original transaction flags in a
  private slot. Even a diagnostic 4,096-point repair queue was exhausted.

Keep the fast prefix when its existing gate accepts. Otherwise, on the
globally tiled path, normalize the prefix and suffix right-duals by exact
binary exponents. Align the complete products and all exclusive-factor
derivatives to the same exponent before the Becke quotient; that exponent
is common to both the primal and derivative and cancels. No small term is
screened and no change is made to the geometry, point rule, or partition
acceptance tolerance. The additional exponent plane is bounded by the
existing point tile; the remaining unrepresentable or invalid points still
use the independently bounded checked repair queue.

Keep the small defaults and HVP resource path. For the larger dense profile,
explicit grid, integral, and aggregate budgets admit more finite work; they
do not remove live device ledger accounting. The exact Direct dense replay
still uses the existing certified omission tolerance and caps its envelope
census. This is not an endorsement of a larger atom-count ceiling or a
change to ordinary CUDA SCF, energy or force dispatch.

## Complete 96-H qualification

Slurm 20804 (Direct) and 20803 (DF with STO-3G auxiliary) used the public
Python endpoint and explicit 4 GiB grid, 3 GiB integral, and 28 GiB global
device budgets (1 GiB reserved). No debugger or numerical host fallback.
Both produced all 288 Cartesian columns without post-symmetrization.

| Provider | SCF + Hessian / Hessian (s) | RTX 5090 peak (MiB) | Max error vs matched GPU4PySCF (Ha/Bohr²) |
| --- | ---: | ---: | ---: |
| Direct | 301.43 / 297.09 | 7,895 | 3.37e-9 |
| DF | 586.59 / 582.68 | 6,127 | 4.88e-9 |

Raw asymmetry was 3.53e-14 (Direct) and 6.72e-14 (DF); the largest
rigid-translation residual was 1.01e-13. Direct recorded 1,470,176,973
exact primitive components and 2,880 physical induced-Fock actions. DF
recorded 288 value passes, 288 contraction passes, and 4,608 physical
induced-Fock actions. Peaks are selected-card NVML samples, including
CUDA overhead, from cards reporting 32,607 MiB. The Direct reference
completed in 109–112 s on an RTX 5090; the fitted reference completed in
106.45 s on an RTX PRO 6000. Do not treat the latter cross-card comparison
as a speedup measurement, or extrapolate the fitted result to another
auxiliary basis. The explicit wide profile is a capacity result.
