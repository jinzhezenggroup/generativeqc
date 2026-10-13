# Decision: native, bounded Modified Hückel preparation

Status: implemented
Date: 2026-10-12

The subsequent default/DF policy is recorded in
[the superseding note](2026-10-12-default-huckel-df.md); this note preserves the
original explicit-provider decision and atomic construction rationale.

## Problem

The PBE0 screening experiment imports a PySCF atomic-RHF Hückel density. Its
successful numerical admission after the eigenpair-ordering repair does not
make this reference-oracle work a production dependency, or establish that it
is profitable for every method. The repaired experiment improved the fresh
48-atom MINAO median by 8.19%, but the 96-atom median by only 1.55%; the old
96-atom 5.48% result cannot qualify a different library/control campaign.

## Decision

Add an explicit native `InitialGuessSpec("huckel")` provider, additive C kind 4
and capability bit 8 without changing the options/diagnostic layouts. The
provider is independent of the target functional. It currently shares MINAO's
FP64, all-electron H–Ar, restricted exact CPU HF/KS and CUDA KS admission domain.
Unsupported spin, ECP, approximation and backend consumers are not silently
reinterpreted as restricted exact calculations.

The construction follows spherical-average neutral atomic RHF with fractional
angular occupations and the updated Wolfsberg–Helmholtz rule. Hydrogen uses
the one-electron Hamiltonian, not a self-interacting fractional RHF surrogate.
All radial functions of occupied angular channels are retained. Empty angular
channels cannot mix into a spherical density and are removed before building
the atomic integrals; this does not truncate an occupied radial variational
space. H–Ar has only occupied s/p channels, so Cartesian and real-spherical
targets share this atomic calculation even with d/f/g polarization functions.

Native CPU atomic integration, DIIS and the occupied-atomic-subspace eigenproblem
are explicitly production preparation for both CPU and CUDA targets. Small
eigenproblems use the existing validated shared-scalar CPU target provider,
not the independent Jacobi oracle or PySCF. The target CUDA owner still performs
strict final same-AO admission. There is no implicit CPU retry of that operation.

## Work and lifetime

An invocation-local cache compares actual normalized ordered shells and
primitives, plus element identity. Different atomic bases cannot alias simply
because the nuclear charge or AO count agrees. Atomic geometry is fixed to
the origin; no molecular density, geometry-dependent overlap, DIIS history or
unbounded process-wide cache survives. Preparation counts sum unique atomic
iterations and Fock builds; hydrogen builds no J/K.

Minimal-subspace overlap uses sparse atomic coefficient support directly,
rather than dense molecular-basis projections. Explicit target AO indices
support noncontiguous/interleaved shells. Only the occupied molecular columns
are recovered before constructing the target density.

Numeric preflight bounds molecular assembly/admission, invocation-local cached
frames and one serialized atomic integral/DIIS owner. The public planner mirrors
this bound, retaining one output matrix per batch member and charging only the
largest remaining serialized workspace. Resource budgets do not grow to enable
the optional provider. An insufficient occupied atomic subspace, failed atomic
solve, singular overlap or invalid candidate preserves the established bounded
Hcore fallback; allocation failures propagate.

## Numerical invariants

- Neutral H–Ar spherical atomic configurations are fixed, independently of
  molecular charge; the final molecular occupation uses the requested count.
- Atomic convergence is at least 1e-10 in energy, 1e-8 in density, and 1e-7 in
  physical residual. Requested tighter controls and per-atom iteration bounds
  remain effective.
- The modified GWH coefficient is `1.75 + delta^2 - 0.75 delta^4`, with
  `delta = (epsilon_i - epsilon_j)/(epsilon_i + epsilon_j)`.
- No occupation clipping, charge normalization, density repair, or relaxation
  of shared eigenframe/seed gates is introduced.
- Existing explicit/imported/warm densities take precedence; a failed seeded
  target has at most one ordinary core retry.

## Rejected alternatives

Promoting the PySCF arm would silently make a reference oracle part of the
production contract. Frozen element/basis tables would restrict supported
custom bases and make regeneration/source identity another numerical owner.
Using a same-element-only cache would silently reuse the wrong atomic basis.
Choosing a universal default from one favorable water size would bypass the
existing complete-endpoint >5% gate and omit unqualified spin/backend domains.

## Evidence

`tests/data/huckel_reference.json` retains independent PySCF raw bases, overlaps
and Modified-Hückel densities for all H–Ar atoms and heterogeneous molecules,
including spherical/Cartesian polarization and interleaved-shell cases.
`tests/python/test_huckel_initial_guess.py` also checks analytic one-Gaussian
hydrogen seeds with identical/different atomic bases, occupation/trace gates,
planner/native capacity agreement, distinct HF/KS targets and budget fallback.
The complete-endpoint runner has a separate native `huckel` arm and a fresh
core/MINAO/native-Hückel summary mode; oracle-backed and native arms are never
conflated, and atomic Fock work is required in the PBE0 native-arm census.

The fresh native campaign is retained under
`benchmarks/results/native-huckel-20261012/`. All 18 complete endpoints pass
independent numerical and paired-density gates without retry. Native Hückel
reduces the MINAO complete median by 9.14% at 48 atoms, but only 2.31% at 96
atoms. Preparation/admission itself improves from 0.317283 to 0.129302 seconds
and from 2.359682 to 0.927277 seconds, respectively. At 96 atoms the target
Fock counts are Hückel 17/18/17 versus MINAO 17/17/22; one slower MINAO sample
does not qualify a universal default. The existing MINAO/Hcore policy remains.
This evidence uses base `4ac8d4517` plus the native provider/ordering repair;
later CUDA HF integration on a different frozen base has separate qualification.

## Revisit when

The [CUDA RHF follow-up](2026-10-12-cuda-hf-huckel-admission.md) subsequently
connects both CUDA HF execution owners, with separate frozen-source numerical
qualification and unchanged automatic selection. Its strict GPU admission and
force-frame fallback must not be inferred from this earlier PBE0 evidence.

Complete cold energy/force evidence establishes profitability across admitted
method/backend domains, or a new spin/element/approximation consumer has its
own independent numerical and work/budget gates. Keep construction cost in
the endpoint even if a future bounded context-owned atomic cache is added.
