# Candidate: analytic Gaussian ECP projector seeds without off-center grids

Status: proposed
Date: 2026-11-11

## Motivation and domain inspection

The retained mixed Ni/I and Ni/Cs quadrature failures are recorded in
`../rejected/2026-11-10-denser-ecp-grid-as-mixed-ni-domain-fix.md`. Do not
replace those fixtures or reinterpret further grid densification as their fix.

Every local and projector term for I and Cs in the pinned
`tests/data/global_hybrid_ecp/def2-svp-ni-i-cs-h.json` has `r_exponents=2`.
In the repository's radial convention these are Gaussian radial potentials,
without an additional radial power. This makes a closed-form seed a candidate
for that particular parameter family, not a general scalar-ECP replacement.

## Derived s-pair identity

Let unnormalized s Gaussians have exponents a, b and center vectors A, B
relative to the ECP center, with a Gaussian potential exponent z. Set
`p=a+b+z`, `k=2ab/p`, `x=k|A||B|`, and
`qA=a(b+z)/p`, `qB=b(a+z)/p`. For an angular projector l, the candidate is

```text
V_l(ss) = (pi/p)^(3/2) (2l+1)
          exp(-qA |A|^2-qB |B|^2) i_l(x) P_l(A.B/(|A||B|)).
```

Here `i_l` is the modified spherical Bessel function and `P_l` the Legendre
polynomial. The expression follows by spherical-harmonic projection and the
Gaussian radial integral of the product of two modified spherical Bessels.
It is our derived candidate, not a production implementation or a quoted
external algorithm.

For l=0,1,2 the angular factor can instead use regular solid polynomials and
`F_l(y)=i_l(sqrt(y))/sqrt(y)^l`, `y=k^2|A|^2|B|^2`:

```text
l=0: F_0(y)
l=1: k (A.B) F_1(y)
l=2: k^2/2 [3(A.B)^2-|A|^2|B|^2] F_2(y)
dF_l/dy = F_(l+1)(y)/2.
```

These identities suggest using the existing compiler Graph to differentiate
the seed, rather than independently implementing derivative recurrence math
in the native runtime. Raising a Cartesian Gaussian power obeys
`I_(n+1)=(dI_n/dA+n I_(n-1))/(2a)`. ECP-center derivatives must use the same
relative-coordinate seed and translation identity. Through-f first/second
derivatives and their stable evaluation are **not implemented or qualified**.

## Finite exploratory evidence and a retained oracle disagreement

The scratch script `/tmp/gqc-ecp-r2-projector-proof.py` compares normalized
s-pair values against isolated single-channel libcint `ECPscalar_cart`
matrices. Twenty ordinary/tight/diffuse/coincident-center cases for l=0..3
agree to a maximum absolute error of `2.6922908347e-15`.

Four additional very tight, coincident off-center cases at
`a=100000`, `b=90000`, `z=0.85`, `A=B=(-2.73,0.22,-0.11)` give zero in
that libcint call, versus candidate values from `1.1807244706e-9` to
`8.2650016248e-9`. The complete 24-case experiment therefore **fails** its
preselected `2e-10` absolute comparison gate. Keep that failure visible; do
not exclude those inputs or widen a production gate to declare success.

A separate 60-decimal mpmath numerical radial integral of the two spherical
Bessels agrees with those four candidate values to `1.4889251026e-23`.
It integrates in the Gaussian peak coordinate over +/-12 standard radial
units, preserving the scaled integrand rather than numerically integrating
an exponentially overflowing unscaled product. This supports the seed
identity at those points, but does not establish why the libcint call returns
zero or qualify a full basis, force, Hessian or molecular endpoint.

The scratch receipt explicitly sets `production_qualified=false` and retains
every comparison and failure in `/tmp/gqc-ecp-r2-projector-proof.{json,log}`.

An additional 60-decimal differentiation experiment applies the Cartesian
raising identity to p/s, d/p, f/d and f/f primitive pairs, for projector
l=0,1,2. All twelve normalized values agree with independent libcint matrices
within `1.0755285551e-15`, below the preselected `2e-10` value gate. Its seed
uses the regular solid polynomials and hypergeometric representation of `F_l`,
including the lower-power terms required when raising Cartesian powers.
The script and receipt are `/tmp/gqc-ecp-r2-cartesian-proof.py` and its adjacent
`{json,log}` files. This is a high-precision validation prototype, not a bounded
FP64 generated provider. It does not resolve the retained very-tight reference
disagreement or qualify contracted mixed-basis matrices or nuclear derivatives.

## Required implementation and acceptance boundaries

- Use a scaled Bessel/regular solid-harmonic representation with analytic
  zero-center limits; independently test cancellation and extreme exponents.
- Avoid subtracting large nearly equal exponents. The scaled exponential can
  use `-[az|A|^2+bz|B|^2+ab(|A|-|B|)^2]/p`, but its derivatives still need a
  regular, compiler-owned representation at zero centers.
- Differentiate one canonical primal operator for local/projector matrices,
  first derivatives, induced Fock and raw second order. No mismatched primal,
  force and Hessian Hamiltonians, or hidden reference-engine production work.
- Preserve normalization, spherical transforms, signed contractions, all
  center derivatives and translation conventions. Qualify complete canonical
  Ni/I and Ni/Cs matrices and forces against independent references before
  considering admission.
- Retain explicit bounded fallback/refusal for other radial powers and
  unsupported projectors; do not remove the existing checked-grid domain.
- Freeze gates before accepting mixed molecular endpoints; include both spins,
  continuation, second-order sources, actual work and 25--40 atom scalability.

The local s-pair experiment is evidence for further investigation, not a fix
for #2846 or #2847 and not permission to open their installed public domains.

## Follow-up: tight reference quadrature, not a zero operator

The earlier shorthand "libcint" identifies the normalization/integral interface,
not the owner of this ECP calculation: PySCF 2.14.0 implements it in
`pyscf/lib/gto/nr_ecp.c`. The installed source SHA256 is
`1f13b0ef396236873aecddb3122103a3455736d6622f00274f97ddd79eff4457`.
Its projector integration refines nested Gaussian-Chebyshev radial rules and
can stop after two consecutive relative-agreement observations. Agreement of
two zero sampled integrals is therefore not proof of a zero integral.

The separate diagnostic `/tmp/gqc-ecp-tight-reference-quadrature.py` reads the
actual static radial nodes/weights without changing or rebuilding PySCF. It
evaluates the stable scaled-Bessel integrand on each complete nested rule for
all four retained tight cases. The 31- and 63-point rules both give exactly
zero: their nearest node is 29.155 Gaussian peak units away. The 127-point rule
is still approximately 1e-287. Later rules detect the nonzero peak, but even
the complete 2,047-point rule differs from the 60-decimal localized integral
by 9.2710181672e-10 through 6.4896580097e-9. Thus forcing the current maximum
rule alone would still fail the original 2e-10 comparison gate.

This is a quadrature-resolution diagnosis and a source-supported explanation
for the observed zero, not an instrumented trace proving which C branch ran.
The diagnostic uses SciPy scaled Bessels rather than copying PySCF's Bessel
approximations. Its receipt and log are
`/tmp/gqc-ecp-tight-reference-quadrature.{json,log}`; the receipt hashes the
script, reference C source and unchanged original 24-case receipt. Keep all
four original failed comparisons visible. No gate is relaxed, no reference
value is substituted into production, and no contracted/derivative/molecular
provider is accepted by this diagnosis. It motivates an independently stable
localized reference as well as the still-unimplemented analytic provider;
neither merely sampling a denser grid nor deleting the tight cases resolves
the full qualification problem.

## Follow-up: bounded FP64 Graph arithmetic experiment

An additional scratch candidate generates selected normalized primitive values,
six A/B first derivatives and 36 A/B second derivatives with the existing
canonical `Graph`, Cartesian raising identity and external-leaf derivative
contract. It does not add a second AD engine or import the public runtime or
PySCF during generation. Near zero it evaluates the regular positive-series
`F_l(y)`; above `x=sqrt(y)=50` it uses scaled spherical Bessels and the stable
relative-radius Gaussian exponent. The scaled branch's leaf rule is
`d[exp(-x)F_l(y)] = exp(-x)F_(l+1)(y) dy/2 - exp(-x)F_l(y) dx`.
Zero-center arithmetic therefore never evaluates the singular large-argument
radius derivatives. The experiment uses Python scalar lowering, not a compiled
native provider. Its finite series and node guards are not endpoint work bounds.

The original 12 ordinary s/p/d/f-component cases pass independent PySCF value
and three-direction first/second derivative comparisons at the predeclared
fourth-order steps `[0.003, 0.001, 0.0003]` Bohr. A separate 38-case edge run
adds 18 zero-center cases, 12 resolved off-center scaled cases and eight
`50 +/- 1e-8` branch-boundary cases. Both processes finish zero. The edge-run
maximum errors are `2.6810335041344324e-14` for values,
`2.6291721173713754e-10` for first derivatives and
`2.9042099006851624e-6` for second derivatives. The frozen experimental gates
are respectively `2e-10`, `2e-8` and `5e-6`; these are primitive/directional
experiments, not a replacement or relaxation of any molecular production gate.
The sampled largest generated graph has 339,793 nodes. Reuse/factorization and
complete contracted-consumer work must be addressed before performance claims.

For the four original extremely tight s-pair points, 70-decimal differentiation
of the same closed identity also checks FP64 arithmetic. This is deliberately
labelled **not an independent derivative oracle**. It neither repairs nor
removes the original failed PySCF comparisons or certifies tight derivatives.
All original failed receipts remain byte-identical. Full component coverage,
contracted mixed Ni/I/Cs matrices, independent tight derivatives, native bounded
execution and matching primal/force/Hessian integration remain unqualified.
The proposal stays proposed; no production selector or numerical owner changes.

Scripts, receipts, logs, canonical Graph source and original failed-reference
evidence are member/hash-verified in the ignored local archive
`build/qualification/ecp-fp64-graph-20261111/local-recovery.tar.gz`, 26 entries,
SHA `63b916d65e52a4f33a4efe7d450e5e61557975a9f4a148bd878ca2d3ef759118`.
No external asset is published. The still-running original Ni matrix's eleven
frozen source/library identities are unchanged by these scratch experiments.

## Follow-up: compiled primal and independent mixed-matrix diagnosis

A separate bounded native candidate now compiles 70 canonical-Graph-generated
complete Cartesian primitive blocks, covering local and nonlocal `l=0,1,2`
Gaussian terms for every s/p/d/f shell pair. The largest block has 55,453 DAG
nodes. Grouping equal ECP exponents reuses radial frontiers, and FP64 compensated
contraction consumes the actual normalized NativeAO basis records. A private
standalone ABI preserves output buffers on admission, finite-budget and
nonfinite failures. Only radial-power-two terms are admitted. This remains a
primal experiment, not production dispatch, public ABI or derivative admission.

All 128 complete normalized primitive-block comparisons pass the independent
PySCF matrices and shell interchange gate, including resolved scaled points;
worst error is `8.698075549748441e-11` against the frozen `2e-10` gate. Twelve
failure controls preserve all outputs. The verified sccache 0.16.0 build makes
71 real compile requests, all misses, in 118.87 seconds; no cache-hit or endpoint
speedup is inferred from these build measurements.

The original complete mixed-matrix probe **exits one**, and stays failed.
At the unchanged Ni/I-or-Cs/H geometry, Ni/I spherical and Cartesian matrix
errors are `6.358418053516591e-13` and `4.910285997309811e-13`.
Both Ni/Cs representations differ from PySCF by `3.084613118154323e-9`, exceeding
the unchanged full raw-matrix gate `2e-9`. The 60/67-AO Cs matrices are not
truncated, selected or relabelled as passes. The worst discrepancy is the
off-center Ni core p-x diagonal: native `0.0001479659942117013` versus PySCF
`0.00014796907882481947`.

A separate independent diagnostic integrates **all** 196 entries in the
14-AO same-center Ni s/p block, selected by element/angular family rather than
observed error. It independently normalizes PySCF shell exponent/contraction
data and uses SciPy scaled Bessels with localized one-dimensional radial
Gauss-Legendre quadrature, not the native closed Gaussian radial identity.
Orders `[64,128,192]` converge within the predeclared `5e-13` reference gate;
the largest observed convergence error is `1.509903313490213e-14`.
The p/p transverse angular term is retained. The localized worst p-x value is
`0.00014796599421169543`, agreeing with native rather than the PySCF value.

That diagnostic forms an explicitly labelled hybrid independent reference:
the complete localized Ni s/p block plus unchanged PySCF for every other entry.
It finishes zero. Full-matrix errors remain `6.358418053516591e-13` and
`4.910285997309811e-13` for I, and become `4.715258919131765e-10` and
`6.903696171049571e-10` for Cs spherical/Cartesian, all below the unchanged
`2e-9` gate. This supplies evidence of a scoped reference-resolution limitation;
it does not silently replace the failed original probe or qualify arbitrary
tight derivatives, contracted forces/Hessians, SCF or a molecular endpoint.

All generated/compiled sources and objects, the standalone library, both raw
matrix sets, original failure, independent localized receipt, scripts, logs
and cache statistics are member/hash-verified in the ignored local recovery
`build/qualification/ecp-gaussian-native-20261111/native-primal-recovery.tar.gz`,
169 entries, SHA
`05a198e274fb5aa8ce4bd2a8f8b20634dbdcd2ff49060d14e2e127eb6dde9cde`.
No external asset is published and all eleven original live-Ni numerical
dependencies remain unchanged. Matching native first/second derivatives,
complete contracted mixed-element qualification and production integration
remain necessary; this architectural proposal is not yet implemented.

## Follow-up: compiled first/second jets and retained reference failures

The standalone native candidate now also generates all six relative-center
first and 36 second derivative channels for complete Cartesian primitive blocks,
using exact Gaussian raising/lowering and shared shifted-shell blocks. Nuclear
chain mapping includes both AO centers and the ECP center, including coincident
atoms. Equal-exponent groups reuse radial frontiers, compensated contraction
preserves the normalized NativeAO representation, and every admission/budget/
capacity failure stages outputs transactionally. This private experiment exports
complete raw one-electron nuclear jets, not an SCF energy/force/Hessian provider.

Sixty-three additional canonical Graph blocks, through h/f and g/g, have at
most 334,850 nodes. A monolithic O1 build is interrupted, exit 143: its shared
cache daemon workers outlived client timeouts. Only this candidate's exact
compiler commands are cleaned up; the shared server and original Ni job are
not stopped. The retained private-foreground retry safely times out four large
O0 translation units at 180 seconds and exits one, preserving 60 successful
objects. A further lowering-only retry retains the Graph arithmetic and scalar
assignment order, partitions those four units into 1,024-temporary no-inline
functions, and finishes zero in 184.08 seconds. Maximum explicit per-kernel
scratch is 2,485,304 bytes. It uses the verified sccache 0.16.0 finite owned
compiler runner and existing leased cache slots. Shared-server statistics are
retained but are **not** private-invocation hit/miss counts. No optimized
production endpoint speedup is claimed for these mixed O0/staged-O1 objects.

The original 192-case analytic-PySCF derivative probe **exits one**. It covers
every ordered s/p/d/f pair, all four channels and ordinary, resolved-scaled and
zero-center families. Exactly 191 cases pass the original primitive gates; the
remaining local resolved-scaled f/f case differs by `3.05731476506893e-8` in
first derivatives and `6.228740606656347e-6` in second derivatives, above the
unchanged `2e-8` / `5e-6` gates. A separate exact-order 12-point Gauss-Hermite
Gaussian-moment diagnostic covers all 48 complete local blocks, not just that
failure; all pass, with maximum value/first/second errors
`[6.245004513516506e-17, 3.400058012914542e-16,
2.192690473634684e-15]`. This independently diagnoses the local reference
failure without altering the original failed process.

All four complete mixed Ni/I/Cs spherical/Cartesian raw derivative comparisons
also remain failed at the unchanged `2e-8` first/second gates. Maximum errors
are `1.8315743552431686e-7` / `6.552796647257033e-7` for I and
`1.2921066869357904e-5` / `3.6093541650785484e-5` for Cs. The full jets retain
every matrix entry and all nine first/81 second nuclear channels. Native raw
translation and derivative symmetry errors stay below `4.5e-16`; sixteen
budget/radial-power/nonfinite/capacity controls preserve every output. For I,
437,030 generated blocks produce 6,906,466 Cartesian values with 29,376 shared
radial frontiers; spherical export takes 11.72 seconds. These are complete
raw-integral measurements, not molecular endpoint timings. Full probe process
wall time is 55.53 seconds and peak RSS 501,992 KiB.

A separate three-direction, three-step independent raw-matrix difference
diagnostic also exits one. For I the native first derivatives agree to below
`7.2e-11`, whereas PySCF analytic first derivatives differ by approximately
`1.3e-7`; second differences remain unresolved. This is a consistency diagnosis,
not a full-derivative acceptance or a reason to relax any gate.

A further independently normalized localized radial reference selects the
entire contracted Ni-s/H-s cross block (ten AO pairs) by angular/element family.
It uses SciPy complex Bessels, undoing `ive`'s nonanalytic real scaling before
complex-step differentiation; fourth-order differences of these independent
gradients provide every nuclear second component at the original three steps.
Orders `[64,128,192]` converge below `5e-13`; all four representations finish
zero in 99.96 seconds. Worst native gradient error is `1.4988010832439613e-15`,
second error `8.528948380881474e-12`, and multistep reference spread
`8.570984200151344e-12`, against predeclared `2e-8` derivative / `2e-9`
spread gates. This confirms the native values at the original worst mixed
entries, but it is **only that complete angular-family block**, not every AO
pair or a replacement for the failed complete-matrix derivative gate.

All original failures, successful scoped diagnostics, raw arrays, generated
and staged code, objects/library, source hashes, scripts and compiler lifetime
evidence are preserved in the ignored member/hash-verified local archive
`build/qualification/ecp-gaussian-derivative-20261111/derivative-recovery.tar.gz`,
206 entries, 100,461,478 bytes, SHA
`759ab077c7d305c8093a8c55f6f768f453a053b7857adabb687d508ff73a1e34`.
The prior primal archive is linked unchanged. No production selector changes,
reference values enter no production calculation, and the proposal remains
proposed pending complete independent derivatives, bounded contracted consumers
and public/installed molecular integration.

## Follow-up: independent complete matrix derivative reference accepted

The original complete PySCF analytic-reference process remains failed. A new
independent reference no longer substitutes a hand-selected block into that
matrix: it covers **every AO pair** of the original mixed Ni/I/H and Ni/Cs/H
fixtures in both representations, including d/f functions, coincident AO/ECP
centers, all nine first and all 81 mixed second nuclear components. The native
candidate and its original raw arrays/library are unchanged.

For nonlocal channels, the reference expands normalized Cartesian orbital
polynomials and real spherical projector polynomials into angular moments of
the addition theorem. Scaled spherical Bessels include their analytic origin
series; `ive`'s nonanalytic real scaling is undone before complex-step
differentiation. Unlike the native closed Gaussian-projector radial identity,
the reference numerically integrates on a fixed segmented radial quadrature,
refined around every orbital/ECP-center radius and the tightest primitive
width. The grid is held fixed across nuclear perturbations. Independent local
Gaussian moments, separately normalized contractions and the checked PySCF
Cartesian/spherical transformation complete the same raw AO space. No ECP
integral or derivative from PySCF enters this new reference.

The initial first-only quadrature orders `[8,12,16]` fail the frozen `5e-13`
value/gradient convergence gate at `9.565681580170349e-13`; preserve that actual
failed receipt and original script. A separately identified retry increases
quadrature resolution to `[12,16,24]`, with 1,608/2,144/3,216 radial points for
I and 1,632/2,176/3,264 for Cs. Its complete first-derivative process finishes
zero in 21.91 seconds. No numerical gate is widened. Complex-step first
derivatives and fourth-order differences of those gradients at the original
steps `[0.003,0.001,0.0003]` then furnish the complete independent nuclear jets.

The new complete process actually finishes zero in **479.29 seconds**. All four
raw matrix comparisons pass the unchanged `2e-8` first/second gates. Maximum
value error is `2.1316282072803006e-14`, first error
`3.692879335659427e-14`; second errors at the three steps are:

| Fixture | Second errors, largest to smallest step |
| --- | --- |
| I spherical, 62 AOs | `[8.30488497416404e-10,1.0265857608438012e-11,7.7973183465474e-12]` |
| I Cartesian, 69 AOs | `[1.1314812975449229e-9,1.3986006797139794e-11,8.988365607365267e-12]` |
| Cs spherical, 60 AOs | `[7.568189408502235e-10,9.415579427241028e-12,1.698086116164177e-11]` |
| Cs Cartesian, 67 AOs | `[1.2715061226131752e-9,1.57898139008239e-11,1.9883317214919316e-11]` |

Reference order convergence stays below `2.132e-14` and the maximum multistep
spread is `1.272e-9`, passing the frozen `5e-13` / `2e-9` reference controls.
The earlier extension to the entire 70-entry Ni-s/p-to-H-s/p block also passes
every first/mixed second component in 486.01 seconds, with first errors below
`2.610e-15` and second errors below `3.710e-11`. That scoped result is retained
separately; it is not used as a substitute for the complete reference.

The new member/hash-verified local recovery is
`build/qualification/ecp-gaussian-derivative-20261111/full-reference-recovery.tar.gz`,
28 members, 42,954,908 bytes, SHA
`17b161764d52d3d4039409a2e4a2dedf0e333034f45aa72ea5654b2e601654e5`.
It includes the original failed full probe/raw arrays/native library, all new
full/scoped reference arrays, failed and successful reference attempts, scripts,
logs, pinned input and normalization/transformation helpers. The original
206-member derivative recovery is linked with its unchanged checksum, not
overwritten; generated Graph sources and compiler-lifetime evidence remain there.

This resolves the complete **private one-electron integral-jet** reference gate
for these fixtures without relabeling the original failed processes. It does
not implement a production ECP selector or SCF force/Hessian provider. Early
contraction, bounded resource admission/fallbacks, native provider ownership,
effective-charge/Pulay response, matched RKS/UKS total endpoints, installed
execution and 25–40-atom work remain separate acceptance requirements. In
particular, do not materialize a global `(3N)^2 * nao^2` jet tensor in a large
molecular consumer merely because the private three-atom diagnostic exports it.
The architectural proposal remains proposed until those production boundaries
are implemented and independently qualified.
