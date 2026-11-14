# Decision: canonical ERI orbits in the native stationary gradient

Status: implemented
Date: 2026-11-13

## Problem and diagnosis

The independently stationary I–CH3–I reference does not return its first native
energy/gradient within a bounded endpoint attempt. Initial AO/XC samples only
describe the first minute; they do not explain the full timeout. The separate
source-bound CPU journal shows native RKS **converges** in 30 iterations and 36
Fock builds, 72.168 seconds, at 81 AOs/87458 points. The subsequent native
integral gradient is still incomplete at the diagnostic's 90-second stop.
Three later owned-child debugger samples, at approximately 82/92/102 seconds,
all find packaged ERI first derivatives inside the stationary integral-gradient
consumer, rather than the ECP, XC or unconverged SCF paths. Samples are not
percentages and intentionally killed children are not electronic endpoints.

The gradient consumer submitted all ordered shell quartets, even though its
physical nuclear derivative and generated density pullbacks admit exact
eightfold symmetry. J/K already shared each recurrence within an orientation;
the remaining amplification was between equivalent orientations.

## Decision

Use the existing native `eri_symmetry.hpp` canonical-quartet/permutation owner.
For each representative, retain only permutations giving distinct ordered
shell tuples and evaluate their original Coulomb/exchange pullbacks at the
permuted public AO indices. Sum these cotangents in canonical component order,
then invoke the unchanged packaged derivative provider once for both channels.

No symmetry of D/W is assumed. In particular, do not replace orbit weights with
a factor of eight or change the UKS spin pairing or prepared exchange fraction.
Repeated shells already cover their complete AO blocks. Their stabilizers must
not duplicate those blocks. Slot-to-physical-atom scattering stays with this
nuclear consumer; independent-slot provider jets and all recurrences are
unchanged. Exact same-center translation elision remains a physical contraction
identity, including distinct shells sharing a center.

The original one/two shell-local channel buffers suffice. Orbit metadata is a
fixed eight-by-four index array, not a molecule-wide table or retained ERI
cache. Resource admission stays conservative, counts actual submissions and
primitive recurrence passes, and retains all finite/capacity checks. The journal
now distinguishes hcore, Pulay and two-electron source phases.

## Validation and limits

The existing native independent Jet source/capacity tests pass. New mixed-shell
tests compare all four source panels against the independently retained full
ordered Jet tensor, for both spins, both representations, nonproportional signed
asymmetric D/W and three exchange fractions. Four shells on three centers cover
shared-center and repeated-shell stabilizers; each source also passes nuclear
translation. These are numerical tests, not exact internal call-count tests.

The source-matched complete endpoint still requires its own timing, actual work,
independent state/gradient/raw-Hessian and mode gates. This schedule alone does
not qualify ECP saddle admission, requested Ni transition states, or 25–40-atom
scaling. No physical method, basis, ECP, grid, tolerance or optimizer criterion
is changed. Both public ECP saddle guards remain closed.

### Source-bound terminal result

The `host-stationary-eri-orbits-20261113` cohort binds production source
`40428aca0ef6af0d4ed91caffdbb6493c270011d71f2d37045845b0fc89c8e1b`
and library SHA-256
`a9144531fcb7557f1aa43b551d88ed186d8fd620ff324938d70ed896e994f6ea`.
Its first complete native energy/gradient returns at 545.874 seconds. The
two-electron gradient takes 465.249 seconds, with 195391 canonical shell
submissions and 255078161 actual primitive derivative evaluations. These are
journal counters; the separate Hessian-oriented receipt's zero first-derivative
counters must not be interpreted as zero gradient work. Process wall time is
545.91 seconds and peak RSS is 369028 KiB, single-thread CPU.

The saved native orbital/energy/gradient checkpoint passes independent comparison
with the previously frozen positive stationary reference, without running a
reference SCF or displaced state: energy error 3.411e-13 hartree, maximum density
error 1.156e-11, occupied-subspace overlap 0.9999999999999969, maximum gradient
error 3.929e-12, and native absolute gradient maximum 1.187071e-8. This qualifies
the stationary state/gradient only. The full attempt exits 1 immediately afterward
because the CPU Hessian's complete nuclear primitive-work estimate exceeds the
unchanged one-billion-primitive cap. No raw Hessian or native modes complete.

Source-matched local validation includes 313 Python cases without skips, ten
native/API/CLI groups, and a separate through-f full ordered-Jet source oracle
across both spins, both representations and all three exchange fractions, with
unchanged 2e-10 source/translation gates. These checks do not make the endpoint
practical, qualify a native saddle frame, or authorize opening either public
ECP saddle guard. Do not rerun this unchanged endpoint or increase its work cap
as a substitute for reducing actual complete-consumer work.

## Rejected alternatives

- Do not relax SCF convergence: the measured native SCF already converges.
- Do not target ECP quadrature or initial ERI values based only on the outer
  timeout; later samples identify the two-electron derivative recurrence.
- Do not add another recurrence/permutation table or retain molecular AO4
  derivative panels to hide repeated orientations.
- Do not infer an eightfold endpoint speedup: SCF, one-electron/ECP and XC
  gradients, diagonal stabilizers and later Hessian/response work remain.
