# Decision: complete primitive Hermite Coulomb execution

Status: implemented, opt-in; matched 96-atom E/F qualified
Date: 2026-10-19

## Problem and source audit

PR #2609 preserved an exact public-AO integral mask. Only uniformly admitted
shell quartets entered Hermite precontraction; partially admitted quartets
reconstructed four-index AO integrals. Primitive descriptors also inherited a
contracted-parent bound, so their Gaussian distance decay did not reduce the
source inventory. These are material differences from GPU4PySCF, not a port of
its complete decontracted screening/execution pipeline. The matched r2SCAN-3c
raw J took 3.922 s, including 2.558 s of residual work, versus 0.630 s upstream.

See [the full executed-path audit](2026-10-19-gpu4pyscf-cold-source-audit.md)
for other differences, counterexamples and measurement boundaries.

## Decision and proof boundary

`GENERATIVEQC_MD_J_ENGINE=primitive` selects a separately named complete
primitive recipe. The existing `reference` retains exact AO-mask compatibility;
the default remains `incumbent`. An unavailable optional recipe retains an
explicit existing provider, with that provider's screening contract and
diagnostic identity. No silent CPU/reference dependency is added.

For an unnormalized Hermite Gaussian derivative with exponent p and multi-index
t, the squared Coulomb norm is

    2*pi^(5/2)/(p^2*sqrt(2*p)) * p^|t| * product((2*t_i-1)!!) / (2*|t|+1).

Compiler-emitted outward arithmetic uses the exact conservative constant 35
above 2*pi^(5/2). The absolute E-coefficient weighted sum of these norms bounds
each primitive AO-pair expansion by the triangle inequality. This remains valid
for signed/general contractions and the public Cartesian/spherical mapping.
Invalid/nonrepresentable bound arithmetic retains work; it cannot omit a source.

Each directed primitive product gets epsilon/(M*N), where epsilon is
min(requested screening,1e-12), M the maximum primitive products in an original
pair, and N the complete primitive-pair inventory. The source density envelope
includes all AO components/orientations and spins. A fixed output AO entry has
at most M target primitives and N source primitives. Summing all omitted
absolute bounds therefore gives at most epsilon per entry. This is a screening
bound, not a claim about floating-point accumulation error. Zero tolerance
retains every nonzero contribution, and exactly zero density produces no work.

Bounds are sorted by pair angular class at preparation. Every replay computes
the current maximum density and five active prefixes on device. The task domain
uses these prefixes before quadratic traversal; a captured graph sees changed
zero/tiny/large/negative densities. Full geometry/E/output mapping is retained
regardless of current activity. Optional bounds/state are charged against the
actual owner allowance and rollback with the MD group. K and force providers
are unchanged.

The shared-memory planner also increases component sharing until it can stage
at least a complete quartet-lane rectangle. Previously (2,3)/(3,4) used widths
4/1 because radial scratch nearly exhausted the target, creating millions of
tiny tasks. Width 16 removes that utilization cliff without changing mathematics.

## Evidence

All timings below are on n4 RTX 5090, FP64 direct/no DF, 96 atoms,
48x16x32 unpruned moving grid/full grid response. r2SCAN-3c includes its defining
1088-AO def2-mTZVPP basis, D4 and gCP. PBE0 has 768-AO def2-SVP. Cold/moved n=1;
warm/moved-warm n=5. These are implementation diagnostics, not independently
grid-converged production rankings. Counters and profilers use separate runs.

| Complete endpoint (s) | Reference after capacity fix | Primitive v2 | GPU4PySCF 1.8.1 |
|---|---:|---:|---:|
| r2SCAN-3c cold | 236.820267 | 180.891338 | 153.635193 |
| r2SCAN-3c warm median | 29.344566 | 27.006406 | 31.035172 |
| r2SCAN-3c moved | 140.010216 | 109.797212 | 107.229452 |
| r2SCAN-3c moved-warm median | 29.320627 | 26.978976 | 26.641672 |
| PBE0 cold | 68.312912 | 67.667604 | 76.804166 |
| PBE0 warm median | 11.234087 | 10.615096 | 13.853463 |

Native r2 cold/moved have 24/13 Fock builds; its warm endpoints have one.
Upstream clean r2 cold/moved have 60/40 JK builds; warm has multiple JK calls.
Do not interpret endpoint times as equal per-iteration work.

Primitive v1 r2 cold was 193.712147 s. The utilization repair gives v2 above.
Fixed identical MINAO raw-J v2 is 1.561526 s r2 and 0.487096 s PBE0; upstream
pinned MD is 0.629825/0.348320 s. Remaining raw-J gaps are explicitly unresolved.
All 18 final E/F endpoints pass E<=1e-8 Eh and max F<=1e-7 Eh/bohr;
observed r2 max E=9.09e-12 and max F=4.47e-11. Raw-J matrices pass the declared
atol=1e-9, rtol=1e-10 gate, with max 1.61e-10 against pinned MD.

816 host tests pass, including independent 80-digit Hermite norm comparison,
compiled actual contracts, exact-budget/failure rollback and emitted execution.
n2:3479 has 90 GPU passes, one full-pinned-package deselection. Independent
unscreened libcint gates include signed/general, Cartesian/spherical, RKS/UKS,
nonsymmetric density, zero/finite/loose requested tolerance and changed-density
graph replay. Two actual small kernel/tail/captured cases each pass memcheck,
initcheck, racecheck and synccheck (zero errors; race zero warnings/hazards).
This does not claim 1088-AO four-sanitizer coverage. Prior large-basis racecheck
timeout remains a limitation. The first v1 endpoint deployment lacked force
modules; its partial SCF timing is not an endpoint result.

v2 source identity: `2cec20e70c7c0c0ecb65dc1f648840bfd8a3f018825ab5ab2bfd22fcce4fc793`.
v2 library SHA256: `ea9aa7173e2ac0d2ca7a32cc81f0dc9e518043bfcfd56adc9e8e12c6363994cc`.
Raw receipts: `.artifacts/pr2609/primitive/n4/{1172,1173,1175}` and
`.artifacts/pr2609/primitive/n2/3479`, with source snapshots retained separately.

## Rejected shortcuts and follow-up

Removing the old residual under `reference` would violate its mask semantics.
Replacing only Boys or recurrence cannot repair the work inventory: controlled
upstream generic/radial ablations show no radial penalty, and low-l unrolling
accounts for only 9--11%. Raising the nominal tolerance alone does not explain
the previous gap. Cached native E and resident replay are advantages to retain.

Primitive bounds and cumulative omission are more conservative than upstream
heuristic log screening; the new path is still not a literal GPU4PySCF port.
Further changes need independent oracles and complete endpoint qualification.
The next measured bottleneck is XC; its spatial-grid difference is recorded in
the audit rather than attributed to MD or hidden in a cold timing.

## Subsequent default promotion

The opt-in selection decision above is superseded by
[the default-promotion decision](2026-10-19-md-j-spatial-defaults.md).
Historical measurements and source identities above remain unchanged.
