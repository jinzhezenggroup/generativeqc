# Proposal: scaled Boys moments for the exact long-range Coulomb interval

Status: proposed; standalone device qualification passed, composed timing pending
Date: 2026-10-02

## Problem

Every Cartesian LR primitive currently evaluates a 64-node positive interval
quadrature, including its extra moment for geometry derivatives. The shared
radial owner is `src/integrals/range_moments.hpp`; changing a CUDA consumer alone
would duplicate science and leave CPU/generated consumers inconsistent.

The master `b9c626d15` WB97M-V profile motivating this work has a 1.531 s fixed-D
long-K replay versus 0.259 s full K for six water atoms / spherical def2-TZVP.
The diagnostic 24 x 8 x 16 grid warm energy-plus-force endpoint is 10.097 s,
versus independently measured GPU4PySCF 1.8.1's ~6.80 s. These are diagnostic
six-atom measurements, not the requested large-system advantage. PR #1711
addresses Hermite storage separately; this proposal starts from the same master
and does not incorporate that candidate.

## Candidate numerical algorithm

For `a = omega / hypot(omega, sqrt(rho))`,

`M_n(T,a) = integral_0^a u^(2n) exp(-T*u*u) du = a^(2n+1) F_n(T*a*a)`.

For `x = T*a*a <= nmax + 1.5`, compute the highest Boys order by the positive
series

`F_m(x) = exp(-x)/(2m+1) * sum_k x^k / (m+3/2)_k`,

then recur downwards using addition only. Terms initially have ratio at most
one and decrease thereafter. Stop at a term smaller than 2^-54 times the sum;
keep a finite 128-step bound and the existing interval quadrature as the fallback
if convergence is not reached. The helper touches output only after convergence.

For larger x, start from `sqrt(pi)/(2*sqrt(x))*erf(sqrt(x))` and recur upwards.
Every recurrence multiplier is below one in this branch. All requested orders
0..14 use FP64 storage and accumulation. The scaled argument is evaluated as
`(T*a)*a` so that squaring a tiny a first cannot lose a representable argument;
`0.5/x` avoids overflowing `2*x`. Boundary powers are factored after the Boys
table. The correctly rounded hexadecimal sqrt(pi)/2 constant is independently
checked at 80 decimal digits.

## Invariants and scope

- The existing native, compiler-generated and CUDA consumers share this one
  radial owner. No reference engine, host staging or reference density enters
  production execution.
- Short-range moments retain their positive interval quadrature and rational
  width. They never become a subtraction of nearly equal Full and Long values.
- Standalone Full and established ordinary Coulomb/Boys owners are unchanged.
- Geometry derivatives continue to consume the next moment via `dM_n/dT=-M_n+1`
  at fixed exponents and omega. They do not differentiate a numerical branch.
- Existing input rejection and legacy fourteen-element output bounds remain;
  the second-order API explicitly owns fifteen values for nmax=14.
- Shell/primitive/quartet inventories, screening, precision, source projection,
  provider budgets and SCF convergence criteria are unchanged. This reduces the
  per-radial-evaluation work, not the count of physical integrals.

## Evidence before device qualification

- 24 host native-moment tests pass: adaptive interval quadrature, positive SR
  limits, Full=SR+LR, derivative chains, invalid controls and output canaries.
  Each maximum order 0..14 is independently checked against 80-digit incomplete
  gamma at the series/erf switch and extremes up to T=1e308. Gate: relative
  error 1e-13 plus 2e-323 absolute for subnormal rounding.
- Eight compiled CPU weighted-ERI tests pass against independent Libcint,
  covering psss, dpsp, fsss and ffff, SR/LR separately, arbitrary signed weights,
  and all four centers' analytic derivatives. CPU RSH ERIs sharing this radial
  helper alone would not be an independent radial accuracy oracle.
- A local host diagnostic runs 100,000 calls per sample, three samples at
  nmax=0/2/6/13/14, omega=0.3 and a varying T/rho inventory. Radial-only speedups
  are 10.35/11.12/11.56/12.36/12.52 times. This is not GPU or molecular endpoint
  performance. Source, raw JSONL, compilation and ccache receipts are retained
  in ignored `.artifacts/lr-moments/`.

## Required qualification and revisit conditions

Run the same moment tests on the real RTX 5090 under Slurm with both default FMA
and explicitly disabled FMA. Then require SR/LR matrix/derivative and independent
GPU4PySCF complete molecular energy/force gates, followed by complete cold, warm
and changed-geometry timings at a larger size. Retain the unchanged 1e-8 Eh /
1e-7 Eh/Bohr endpoint gates. CPU microbenchmarks cannot promote the optimization
or establish large-system advantage. Test #1711 and this candidate separately
before timing their composition.

Revisit branch thresholds or the series bound only with independent dense and
extreme-parameter evidence, and never weaken the positive SR or derivative
contracts to obtain a timing win.

## Device qualification and composition

The standalone candidate `dedcbcfec` passed 48 RTX 5090 moment tests (FMA on/off),
canonical full/SR/LR matrix checks, all-center range derivatives, and three
independent complete RKS/UKS WB97M-V tests with reconverged finite differences.
Slurm job 5299 on node1 also completed six- and twelve-atom cold, priming, warm,
fixed-final-state and changed-geometry diagnostics with independent SCF states.
All samples pass 1e-8 Eh / 1e-7 Eh/Bohr against the qualified references:
six atoms use GPU4PySCF, while twelve use CPU PySCF/Libcint after detecting the
GPU4PySCF analytic-gradient discrepancy described below. Maximum errors are
3.07e-12 / 1.34e-9 at six atoms and 5.18e-12 / 9.09e-10 at twelve atoms.

The work-census fixture's opt-in LR expectation was stale: both full and LR
opt-in sources use the bounded provider. Correcting that expectation passes
baseline and both separate candidates at 58/116 public AOs. Default candidate
and radial counts are respectively 2,033,136 and 27,243,271, unchanged by either
optimization. Zero canonical counters on bounded execution do not mean zero
physical shell work. No molecular derivative quartet counter is exposed yet.

The twelve-atom diagnostic found a pre-existing GPU4PySCF 1.8.1 force discrepancy,
not a native optimization regression: baseline and Hermite candidate forces agree
within 1.17e-12, but differ from GPU4PySCF by 4.47635e-5 Eh/Bohr. Independently
reconverged GPU4PySCF directional energies at steps 1e-3/3e-4/1e-4 approach
-0.01302778017 / -0.01302812829 / -0.01302815917 Eh/Bohr. Native analytic gives
-0.01302816394; GPU4PySCF analytic gives -0.01296492998. Independent CPU
PySCF/Libcint complete moving-grid gradients agree with native across both
geometries and all samples to below 9.1e-10 Eh/Bohr. Do not loosen the gate or
silently count the unqualified GPU4PySCF analytic force as an accepted oracle.
The exact third-party defective component has not yet been isolated.

PR #1711 has since merged as master `06459d469`. This radial PR is rebased on that
master, so its eventual production behavior composes both optimizations. The
evidence above identifies the earlier standalone binary; fresh composed device
qualification and matched endpoint timings remain required before promotion.
The complete 24-atom, 48 x 16 x 32 grid cases are still running. Raw scientific
records, input scripts, binary hashes and phase journals are retained locally in
`.artifacts/lr-moments/` and on the explicitly authorized Slurm nodes under
`/home/jzzeng/codes/wb97m-20261002/`. No large-system advantage is established yet.
