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
