# Decision: occupied-page DF triples Fock resolvent

Status: implemented
Date: 2026-10-04

## Problem

#1799 supplies native DF triples amplitude, integral/factor and diagonal
denominator sources. Diagonal epsilon cotangents cannot replace complete
same-space Fock response at an internal degeneracy. Benzene is a concrete
large-system consumer requiring that contract. The existing conventional
virtual-page resolvent requires resident ovvv, which the large DF route avoids.

## Decision and mathematics

Lower the shared standard-(T) inventory's X=PW/D and Y=R3(P(W+V/2))/D onto
fixed occupied (i,j,k), full virtual cubes. P sums all six simultaneous pair
permutations. Scalar TensorIR binds the same R3 and SLOW_TABLE entries as the
existing virtual-page frontend, with explicit inverse occupied-transpose maps.
V is evaluated pointwise; six W cubes come from the existing two-product
occupied moments and DF integral panels. There is no ovvv or rank-six storage.

The complete energy is (1/3)<PW,Y>. Applying the inverse derivative and pair
symmetry gives the symmetric oo moment -0.5(XY^T+YX^T) and vv moment
+0.5(XY^T+YX^T). For fixed j>=k, integrate all virtual b,c and vary i: weight
2-delta_jk is valid for both moments. There is no 6/2/1 occupied-triangle factor
in a resolvent vector. Only occupied/virtual separation is required; same-space
orbital-energy differences are never divided.

The compiler emits two direct BLAS products for each marginal. vv contracts
one [a,bc] cube with another and accumulates once per unique occupied triple.
oo contracts [i,abc] pages, symmetrizes its two factors and scatters a cross-page
block to both matrix positions once. Tail rows are explicitly zeroed before
reuse. They contribute to neither physical response nor mirrored scatter.

## Storage, fallback and complete work

By default all occupied rows share one X/Y page, requiring 2ov³ vector elements
plus six W cubes and at most three cached integral panels. A caller may limit
page rows; when all rows do not fit, two pages of c rows retain 4cv³ elements
and recompute right pages for earlier left pages. The planner first gives up
extra integral panels to preserve more resolvent rows, then reduces c. Both
limits have an explicit one-row/one-panel fallback. All allocations are admitted
before input access or CUDA provider creation.

The complete bound includes borrowed host inputs, detached oo/vv outputs,
CUDA arena, provider allowance and the caller's other live state. Device views
use one stream; detached destinations outlive exception-path drains. Scalar
intermediates, each logical BLAS result and each gradient scatter are audited.
Publication occurs only after the final sticky error check and synchronization.

Let J=o(o+1)/2, R=ceil(o/c), t=o-(R-1)c and B=J R(R+1)/2.
The complete schedule builds B pages, with
C=J[cR(R-1)/2+tR] resolvent cubes, including recomputation. There are U=Jo
unique cubes. For actual panel GEMM count P, the work ledger is:

- W GEMMs: 12C; Fock GEMMs: 2(U+B).
- BLAS summands: PQv³ + 6C(v⁴+ov³) + 2Uv⁴ + 2Bc²v³.
- Scalar X/Y evaluations: Cv³; each evaluation has multiple scalar operations.
- One audit kernel per BLAS product, one resolvent kernel per built cube and
  B matrix scatter kernels. Upload/download counts include final status.

The oo products include zero-padded row work; vv only visits real rows. Full
occupied retention has C=U, with no cross-page recomputation. Small pages are
memory-bounded but can multiply the most expensive W/source work. Report their
actual C, P and complete time rather than claiming unchanged complexity.

## Independent gates

Six host cases compare occupied and conventional virtual-page X/Y pointwise,
then compare paged moments with the independent complete six-axis inverse
derivative. Exact-degenerate recanonicalized energies independently check full
Fock directions. Native tests cover unequal o/v/Q, full/partial/tail pages,
one/two/three panel capacities, exact and near internal degeneracy, rotational
covariance with nonzero off-diagonals, full replay counts, exact budget admission,
fallback and failure without partial publication. Overflow/preflight rejection
precedes null input access. All 26 focused tests pass; the combined 73-case
occupied energy/response/Fock suite passes CUDA memcheck with zero errors in
115.37 s. Hooks and focused types pass.

The parent's Clang/CuMetal fix binds frexp/scalbn/fma to global CUDA overloads,
preserving the existing robust scaled-bilinear response algorithm. No precision
gate or fallback has been weakened to make a backend compile.

### 230-AO ethane component

Qualified complete library SHA-256:
`3ac5599e5387daa4b9bb74afb3eeb00e6a5e800896ace77a582e9e8ccf34a251`.
Slurm job 12207 used one RTX 5090 with an explicit ten-minute limit. The probe
loads frozen native C/T from #1792's cold molecular solve and regenerates native
DF source. Its Fock-MO preparation is the validation fixture's explicit host
reference transformation. It does not perform another cold CC solve.

For (N,o,v,Q)=(230,9,221,488), complete fixed-input full Fock response took
18.524955898 s, including validation, staging, all outputs and teardown. The
numeric bound is 3,029,060,848 bytes, including 2,700,125,952 workspace bytes.
Source regeneration took 1.148844783 s; the complete validation process,
including two native energy evaluations for differences, took 30.41 s.

All nine occupied rows fit in one X/Y page with three integral panels. There
are 45 occupied pairs/pages, 405 unique and 405 built cubes (no replay), 357
panel GEMMs, 4,860 W GEMMs, 900 Fock GEMMs, 9,924,048,505,176 BLAS summands and
4,371,513,705 scalar X/Y evaluations. Transfers are 263,060,792 H2D and
391,380 D2H bytes. Both matrices are finite; maximum asymmetry is
8.131516293641283e-20.

A combined occupied/virtual diagonal direction gives -0.0023550665635954518;
independent native energy differences give -0.0023550665646538543, absolute
error 1.0584024975890394e-12. This large check covers the diagonal contraction;
the full off-diagonal/degeneracy gates are the independent small native tests.

### 264-AO benzene component

The same frozen library and validation executable completed Slurm job 12208
on one RTX 5090 with a fifteen-minute limit. Inputs are the native converged
state from #1792's n2 job 2178, copied locally with SHA-256
`e294af0c496bb33ad0c621448aeba797ac1c3737c838baafb405af0c0125029e`.
For (N,o,v,Q)=(264,21,243,666), complete fixed-input Fock response took
308.813369219 s; source regeneration took 2.306827684 s and the full validation
process including two native energy differences took 479.90 s externally.

Numeric capacity is 7,936,657,840 bytes, including 6,736,695,808 workspace bytes.
All 21 occupied rows fit in one page with three integral panels. There are
231 occupied pairs/pages, 4,851 unique and built cubes (no replay), 4,617 panel
GEMMs, 58,212 W GEMMs, 10,164 Fock GEMMs, 191,130,798,884,238 BLAS summands and
69,606,547,857 scalar evaluations. Transfers are 776,538,744 H2D and 475,924 D2H
bytes. Both full matrices are finite; maximum asymmetry is
2.439454888092385e-19. The combined diagonal direction gives
0.0036898586166357081, versus energy differences 0.0036898586178168991, absolute
error 1.1811909957082367e-12. This is still a frozen-state component, not a cold
CC solve or a nuclear force endpoint.

## Remaining molecular-force boundary

The full oo/vv matrices replace epsilon diagonal sources; adding both would
double-count denominator response. This internal owner completes the triples
fixed-frame canonicalization contract. It is not a molecular force endpoint:
corrected Lambda plus the physical DF source chain still needs conventional-RHF
orbital/Z and overlap/Pulay response and nuclear DF/metric contractions,
including auxiliary g coverage. Public DF forces remain disabled, and #1782's
strict large-source factor gates remain unqualified. Neither a frozen-state
component timing nor agreement of its diagonal derivative certifies a complete
cold molecular force.

## References

- `.agents/notes/implemented/numerics/2026-10-04-df-occupied-triples-response.md`
- `python/generativeqc_compiler/cc/triples_fock_response.py`
- `python/generativeqc_compiler/cc/occupied_triples_fock.py`
- Ignored `.artifacts/fock-response/` build, test and component logs.
