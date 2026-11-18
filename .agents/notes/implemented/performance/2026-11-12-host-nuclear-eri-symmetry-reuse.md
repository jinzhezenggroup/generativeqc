# Decision: reuse physical nuclear jets across real-ERI symmetry orbits

Status: implemented
Date: 2026-11-12

## Problem

The native CPU first-integral and frozen/moving-weight curvature sources
expanded every ordered public AO quartet into radial/Cartesian primitive
derivatives. Permuting a real scalar Coulomb ERI changes the compiler's density
incidences, but not its physical nuclear directional derivative or Hessian
action. Repeating these expensive primitive jets obscured the complete-endpoint
cost even though numeric storage was bounded.

## Decision

Share the incumbent real-ERI index permutation and triangular pair-of-pairs
schedule with `s_integrals.cpp`. The native nuclear integral sources default to
one representative per symmetry orbit. Repeated-index permutations are
deduplicated, so each distinct ordered compiler incidence is still consumed
exactly once.

The first source evaluates a canonical directional integral and supplies it to
each ordered generated Fock map/scatter. The curvature source sums the generated
scalar integral weights and explicit moving-density weight tangents across the
orbit, then evaluates and scatters the canonical physical nuclear jet. The
compiler remains the sole owner of spin, occupation, exchange and constraint
coefficients. Physical-center derivatives are reused, not mathematical-slot
derivatives whose indices would require permutation.

`HostNuclearEriSchedule::Ordered` retains an explicit bounded comparison route.
Planning and construction must select the same schedule. No resident integral
cache or additional numeric arena is required; no new optional-storage
precondition or fallback admission is introduced.

For AO expansion sizes `t_i`, each triangular AO pair has weight
`w_(ij) = t_i * t_j`. A checked prefix sum counts unique primitive work as
`sum_p w_p * sum_(q<=p) w_q` in O(nao^2) planning time. The ordered comparison
remains `(sum_i t_i)^4`. Generated map work and the reported `ao_quartets`
counter retain their ordered-incidence meaning. Primitive counters report
actual evaluations, including failed invocations.

## Rejected alternatives

- Multiply a canonical density weight by eight: incorrect for repeated indices,
  exchange incidences and independently supplied spin/D/W response panels.
- Duplicate the J/K formulas in native code: creates a second scientific owner.
- Cache an AO-fourth-power derivative tensor: loses the bounded streaming
  contract and does not address complete-source work at larger basis sizes.
- Advertise eightfold reuse as practical large-molecule scaling: both primitive
  and generated-map work remain quartic; the 25--40 atom complete electronic
  endpoint requirement remains separate.

## Invariants

Preserve raw physical derivatives, normalization, spherical expansions,
repeated physical centers, both spin layouts, explicit moving D/W terms,
immutable-owner authentication, bounded validation cadence, resource refusal,
transactional publication and recovery. No screening, approximate operator,
displaced production evaluation, reference engine or chemistry change is added.

## Evidence

`tests/python/test_host_nuclear_integrals.py` exercises both schedules through
f, Cartesian/spherical RKS/UKS, mixed pairs, full physical-coordinate columns
and explicit D/W right action. Independent libcint primal/multistep derivative
tests qualify the default schedule. Three-, eight- and 32-site complete frozen
source controls retain work/time/memory evidence; they are not electronic
SCF/response endpoints. A source-matched complete native molecular endpoint
must additionally retain raw-matrix, frequency, work and whole-process evidence.

Local qualification in `build/qualification/host-eri-symmetry-20261112` retains
150 integral/spherical/primitive tests, 68 coupled-response/stationary/public
consumer and identity tests, six native CTests, independent frozen
source receipts and both source/installed complete NH3 native CLI results.
The 16 angular/representation/spin schedule comparisons differ by at most
`1.18e-16`. At 32 one-primitive s sites, ordered primitive work is 1,048,576
versus 139,656 canonical evaluations, while both retain 1,048,576 ordered map
incidences and 36,936 bytes of owned numeric staging. Full frozen-source outputs
differ by at most `1.74e-18`; these are not complete electronic endpoints or
whole-process memory bounds.

The NH3 bare Gaussian/VWN-RPA B3LYP/spherical STO-3G RKS control reaches the
original five convergence criteria in three accepted attempts for both actual
native ELFs. Each first/curvature source counter falls from 19,906,560 to
3,236,760 primitive evaluations across 60 response columns. Both final raw
matrices differ from the previously independent-oracle-qualified matrix by
`1.05e-13` or less; all three archived moving-grid gradient stencils retain
their unchanged `1e-5` matrix and `1 cm^-1` frequency gates. Independent Python
mass weighting differs from native frequencies by less than `1e-12 cm^-1`.
No new displaced-SCF oracle was executed for this molecular control.

The measured source core SHA-256 is
`ffd7478a232ef6c66ccec2e3c6950ba3cda531353e2c981fd5c939bd6a5a6704`;
the installed core SHA-256 is
`16abb330ed1d832cb0002d791cd67361a3f4719099d1206fbf9b3e24ba7dc2e7`.
The cohort receipt records both native source identities and the complete
uncommitted source manifest; these hashes identify this measurement only.

Measured complete endpoint times are 265.87 seconds (source) and 265.83 seconds
(installed), with process maxRSS 24,684 and 25,696 KiB respectively. These are
shared-host observations, with independent consumer qualification running in
parallel, not matched speedup claims. The isolated installed CPU environment
uses a local repaired manylinux_2_35 wheel, explicit minimal dependencies and
`--no-deps`; it has no PySCF/D3/CuPy and executes the ELF with Python/compiler
paths disabled. This is not release portability or default-resolver evidence.

No previously failed H3 saddle or timed-out SiH3 run was restarted or relabeled.
The genuinely open-shell positive molecular saddle, real native Ni/I/Cs
def2-SVP electronic/ECP qualification, practical 25--40 atom complete electronic
work, and the requested Ni saddle remain separate gates. Public ECP/DF/CUDA
molecular saddle admission is unchanged.

## Revisit when

Qualified shell-level contraction, controlled screening or another exact
consumer schedule can reduce the remaining quartic generated-map and primitive
work. Retain the ordered comparison until replacement qualification covers
the same scientific and failure contracts.
