# Decision: Native UMP2 consumes an owned physical UHF final state

Status: implemented
Date: 2026-10-08

## Problem

The bounded UMP2 TensorIR and four-slot spin-aware AO-to-MO reference consumer
from #1837 were Python validation infrastructure. The public method registry,
CPU post-HF provider, and physical-reference export only admitted RHF. A public
wrapper around the Python tooling would create a second runtime owner and would
not establish a native endpoint.

## Decision

An explicit UHF physical-reference export uses the existing strict final-state
selector with canonicality required. It owns each spin's occupations, physical
Fock, density, canonical coefficients and orbital energies, then exposes the
existing two-channel `ElectronicReferenceView`. Ordinary HF execution does not
request this export. The conventional native AO-to-MO provider accepts four
explicit coefficient owners for UHF blocks and keeps the previous restricted
request path. Native UMP2 lowers the three compiler-owned TensorIR channels to
CPU scalar tiles; its method owner publishes a total energy and diagnostic only
after complete successful evaluation. It does not retain a source across
executions.

## Rejected alternatives

- Routing the public method through `tools/generativeqc_mp2` would make tooling
  a production dependency and a second execution owner.
- Reusing one RHF coefficient matrix for both spins would change the UHF
  determinant and can silently pass equal-size AO/MO tests.
- Altering the ordinary UHF core guess to induce broken symmetry would change
  unrelated HF behavior. The existing prepared-batch HF warm-state transport
  already accepts two-spin density proposals and can select a determinant
  without a new singlepoint or C ABI seed mechanism.

## Invariants

The UHF export preserves the final unshifted physical Fock and its validated
canonical spin frames. Every AO-to-MO request records all four spin owners, and
CPU correlation is conventional, all-electron, real FP64 and energy-only.
Denominator extrema are finite and sufficiently negative before an AO read.
Reference, source, transform and generated tile capacity are admitted within
the correlated-method numeric budget; this is numeric capacity, not measured
allocator or whole-process peak. Failure clears the method's published
diagnostic. A fresh prepared source is borrowed only within one execution.
Imported warm densities are validated against the source topology before
execution; successful UMP2 publication still requires a new physical UHF solve.

## Evidence

The source-matched CPU build on 2026-10-08 used GCC 11.4, CMake 4.4.4,
`sccache 0.16.0`, strict FP64 and the scalar CPU linear-algebra fallback.
The compiled and checkout source identities both equalled
`cca8ae40dae35477304a6c69c8a05eab723d1d7b047e111b83aacf8337aa9533`;
the native library SHA-256 was
`eae5bb8130ac5195f2ea4d4376c450dd13cced40bb88059d88566c029bf8d4fc`.

The focused public/core/manifest suite reported 36 passed and four CUDA-only
skips. `test_ump2_public.py` compares native Li to pinned PySCF 2.14.0 using
the repository's exact STO-3G primitives at `1e-8` Eh. PySCF's stock Li
STO-3G coefficients are rounded differently and shifted the independent HF
reference by about `2.43e-8` Eh; the gate was kept rather than relaxed. The
public broken-symmetry H2 gate uses a genuine native UHF solve from a localized
two-spin seed, transports its density and diagnostics through the existing
public warm-state ABI, then requires a fresh public UMP2 batch solve to match
independent PySCF at `1e-8` Eh. The public restricted limit agrees with public
MP2 energy and OS/SS components at `1e-9` Eh. Native UMP2, existing MP2, and
existing UHF final-state CTest targets all passed; the UMP2 native target also
checks stale geometry, near and overflowed far denominators before AO reads, four-slot spin
coefficients and CUDA backend rejection. Numeric capacity was admitted but no
whole-process allocator peak or endpoint throughput claim was made.

The core-level independent Li and stretched-H2 oracles remain in
`tests/python/test_ump2_energy.py`; the latter exercises an explicitly
localized starting density and does not itself qualify the public Calculator
endpoint.

## Revisit when

Change the public seeded path only with explicit determinant provenance,
complete resource admission, and an independent broken-symmetry endpoint gate.
Extend forces, DF, mixed precision or CUDA only under separate capability work.

## References

#1820, #1837, #1987, #1988; `docs/user/methods.md`;
`tests/python/test_ump2_public.py`.
