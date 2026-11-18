# Decision: native frozen-density integral nuclear directions

Status: implemented (development component; not a stationary endpoint)
Date: 2026-11-12

## Problem

The native fixed-geometry coupled response and moving-grid XC source need
actual overlap and frozen-density integral Fock directions. Accepting arbitrary
external matrix directions alone does not supply a nuclear source. Promoting
that interface to a complete Hessian would omit integral, residual ECP and
second-source contributions.

## Decision

Add `HostKsNuclearIntegrals` to the optional CPU native KS response static
target. Reuse the unchanged generated CPU S/T/V gradients and the ordinary
core's linked full-range primitive ERI derivative ABI through f. Keep independent
mathematical-center slots until a linked scalar Graph action applies AO
projection, normalized radial coefficients and physical atom displacement.
Translation supplies dependent center gradients, never a displaced production
evaluation. Public Cartesian and real-spherical AO expansions come from the
same normalized packed `AoBasis` as the ordinary AO provider.

Consume every ordered public AO quartet directly. Differentiate the shared
stationary Coulomb/exact-exchange objective first in density and then only in
the integral. Compiler TensorIR maps retain all four density incidences and
their MethodIR spin, occupation and exchange factors; native code only binds
panels and scatters those outputs. Extract the unchanged density-incidence
renamer from WB97M-V into neutral `method.ordered_quartet` and migrate its
existing caller directly. Preserve a one-off generated-source comparison,
not a permanent production-source fingerprint test.

The explicit scalar inventory includes `integral_direction`. The new map
generator requires that declared projection and carries exact method/leaf
identities. Source-build CMake refuses disabled stationary CPU primitive AD;
imported cores must actually supply its linked ABI. Generation does not import
runtime or reference engines. Generated headers are system includes so unused
formal coordinates in shared s/s code do not disable strict warnings on the
native consumer. Their executable numerical contracts remain independently
tested.

## Invariants

The owner authenticates immutable system, normalized basis, actual strategy,
method and density contents/epochs. The component additionally checks shapes,
pointers, shell/primitive layout, centers, exact CPU FP64 full-range model,
positive exponents and finite symmetric densities. Indefinite densities are
valid frozen-source inputs. This component does not itself certify stationarity.

Mandatory caps cover owned FP64 panels, one-electron primitive visits, ERI
primitive gradients and generated quartet terms before execution. With `M`
the sum of primitive-count times expansion-count over public AOs, the scheduled
work is `M^2 * (natoms + 1)` one-electron visits, `M^4` ERI gradients and
`nao^4` generated per-quartet map workloads. Integer metadata, borrowed owners,
provider/scalar stack, allocator overhead and RSS are not hidden inside that
numeric-memory query. Validation/provider instruction counts and wall time
are not bounded by relabeling these semantic counts. This exact ordered path
is quartic, not a favorable-scaling or complete-endpoint performance claim.

Overlap and every spin Fock output publish together after final owner
validation. Failed caps, malformed/nonfinite frames, model changes, stale
owners, aliases and reentry cannot expose a partial output and must permit
recovery once the original owner is restored. An external direction may alias
an output; output/output and borrowed-state output aliases are rejected.

## Rejected alternatives

- A native handwritten J/K derivative equation duplicates the scientific
  authority and risks restricted/unrestricted occupation factors.
- A molecular AO-rank-four ERI tensor or coordinate derivative tensor defeats
  the bounded panel contract. Dense primal tensors are independent test
  oracles only, not production dependencies.
- PySCF, finite differences or runtime compilation inside the source would
  change production dependencies or derivative semantics.
- Compiling another copy of the existing generated ERI provider makes artifact
  ownership ambiguous. The consumer binds the actual core implementation.
- Calling ionic-point-charge attraction a complete ECP potential omits local
  and nonlocal residual projector physics.

## Evidence and qualification boundary

`tests/python/test_host_nuclear_integrals.py` strictly compiles current native
consumer/selected scalar leaves through a verified compiler cache and links
the ordinary CPU core. Independent normalized libcint S/T/rinv/ERI primal
values provide frozen-D Fock differences at steps `1e-3`, `2e-4`, `4e-5` for
both spin layouts and Cartesian/spherical AOs through f. Acceptance requires
`atol=2e-8`, `rtol=3e-7` and more than thirtyfold maximum-error reduction.
Contracted s/p and smaller single-primitive d/f controls keep oracle work
finite; they are not representative full-spdf endpoint benchmarks. The Ni
control tests ionic point charges only, not residual ECP projectors.

Separate executable controls cover projection slots, translation, linearity,
capacity/model refusal, malformed inputs, late owner failure, reentry, aliases
and recovery. The optional CMake executable runs actual nonzero integral
directions and translation for both spins without Python on PATH/PYTHONPATH.
Source-matched command exits, raw matrices' error observations, preliminary
failures, source hashes and local recovery belong under
`build/qualification/host-integral-motion-20261112/`, not in previous accepted
component receipts. Numerical gates are not relaxed to clear compilation or
lifecycle failures.

## Consequences and revisit conditions

This supplies first S/T/ionic-V/J/K directions only. Moving XC remains a
separate source. Residual ECP projectors, all second-source derivatives,
stationary assembly, current-source installed qualification and a Python-free
CPU TS endpoint are still required. Representative mixed Ni/I/Cs and 25--40
atom endpoint work/time/resource gates and the requested Ni saddle/frequencies
are not established here. Revisit ordered primitive consumption when a shared
compiler-owned shell/block lowering has independent scientific and complete
endpoint evidence; preserve explicit work caps rather than claiming that
bounded memory alone makes this path suitable for those systems.

## References

- `docs/developer/dft_hessian.md`, native CPU frozen-integral nuclear source.
- `2026-11-12-native-moving-grid-xc-source.md`.
- `2026-11-12-native-host-coupled-response.md`.

## Subsequent bounded reuse

The optional first-source coordinate panel in
`../performance/2026-11-13-host-coordinate-first-source-panel.md` supersedes the
no-coordinate/AO-storage policy only when explicitly admitted. The original
streamed direction schedule, primitive algebra and scientific gates remain the
default and bounded fallback. Neither decision establishes molecular TS or
complete-electronic scaling acceptance.
