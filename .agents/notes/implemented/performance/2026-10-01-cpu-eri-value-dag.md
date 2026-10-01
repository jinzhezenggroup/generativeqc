# Decision: lower CPU value-only ERIs from the shared scalar DAG

Status: implemented
Date: 2026-10-01

## Problem

A matched single-threaded, FP64 CPU RHF comparison showed the in-core native path
spending almost its entire water/def2-SVP and formaldehyde/def2-SVP endpoint in
integral preparation. The independent dynamic `Jet` recurrence was still the
production four-center evaluator even when the caller requested values only.
The one-electron family already had a compiler-owned production lowering.

Diagnostic interposition of the existing `build_integrals` symbol confirmed that
integral preparation dominates; isolated no-ERI calls cost about 1 ms for water
and 3 ms for formaldehyde. Timing diagnostics are not performance qualification.
The canonical Cartesian quartet loop evaluates 326,255 primitive ERIs for water
and 2,261,946 for formaldehyde. Neither screening nor scientific work is removed
by this change.

## Decision

Reuse `build_shell_class_component_kernel` with explicit value-only `IntegralIR`
and the existing factored scalar-C lowering. Reuse the shared Boys evaluator.
No new recurrence, Gaussian geometry, or force equation is introduced.

All 10,000 ordered Cartesian s/p/d component tuples are represented by 313
functions under eight ERI center permutations and six simultaneous axis
permutations. The compiler emits both the representative functions and the
20,000-byte packed lookup; native orchestration only binds exponents, centers,
and component indices. Exact full-range CPU HF and DFT share this integral
preparation route. The contraction order, normalized primitive weights,
eightfold tensor scatter, spherical projection, and in-core tensor layout remain
owned by the existing native code.

The optimization applies only when no coordinate derivatives are requested and
every Cartesian angular momentum is at most d. Derivative requests, f+ values,
and range-separated ERIs retain their prior explicit paths. The original
`primitive_eri_cartesian` is unchanged and remains the independent recurrence
used by `RawSource`; it is not implemented in terms of the generated production
helper. No runtime source generation or compilation is introduced.

## Rejected alternatives

- Optimizing scalar J/K first: the profile showed it was a small part of the
  complete endpoint in these cases
- Specializing the handwritten `Jet` evaluator for doubles: this would leave a
  second production scientific algebra owner instead of reusing compiler IR
- Emitting every ordered component independently: center/axis canonicalization
  bounds source and binary growth without changing the requested operator
- Reusing derivative programs for values: this would retain unused work and
  extra Coulomb order rather than honoring consumer demand
- Altering basis, convergence, density fitting, screening, or initial density to
  improve timings: these would invalidate the matched comparison

## Invariants

- Strict FP64 with existing numerical acceptance gates; no fast-math or precision
  lowering, and no claim of bitwise identity after algebraic factorization
- Unnormalized primitive values; native radial/component normalization is applied
  exactly once
- A single center and axis permutation is applied consistently to all exponents,
  coordinates, and angular labels
- Derivative and unsupported-domain fallbacks stay explicit and independently
  tested; no silent use of PySCF or another external oracle in production
- Complete scientific source identity includes the new generator CLI and its
  compiler module; all generated outputs remain build artifacts

## Evidence

The compiler test executes the actual generated header against independent
libcint values. Native tests compare complete contracted Cartesian/spherical
value tensors with `RawSource` and with the derivative fallback's values.
Independent review also checked 1,998 Boys values against 100-digit arithmetic
(maximum relative error 6.55e-16) and 60,000 additional compiled primitive values
against libcint. A deliberate million-Bohr whole-system translation exposes the
existing absolute-coordinate conditioning limit: candidate maximum error
1.23e-9 versus unchanged recurrence 1.33e-9. This is retained negative stress
evidence, not silently counted as a passing numerical row. Both pass the same
gate at a 1000-Bohr translation.

Complete endpoint qualification reuses the exact original matched benchmark
and basis payload, with independent processes, alternating engine order,
process-cold/warm/changed-geometry cases, and retained iteration/work counts.
See the PR and retained CPU ERI benchmark evidence for numerical and timing
results. No CUDA hardware measurement or GPU speedup is claimed.

## Consequences and revisit criteria

The in-core O(N^4) storage and primitive enumeration remain unchanged. This is a
bounded value-lowering slice, not a complete CPU integral engine redesign.
Revisit source-driven shell/pair reuse or direct bounded consumers when endpoint
profiles demonstrate the next bottleneck. Extending production derivatives or
higher angular momentum requires independent numerical and resource evidence;
the availability of the shared DAG alone does not promote those domains.
