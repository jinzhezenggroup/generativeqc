# Decision: exact-stack opt-in Direct-HF matrix profiles

Status: implemented
Date: 2026-11-09

## Problem

The structural #3052 slice makes resident topology resource-aware and moves
angular queue-domain ownership to the compiler, but production still discarded
the small-HF matrix model's calibration opportunity. Re-running its default
resolver inside execution also gave preparation and execution separate policy
decisions. Merely demonstrating different resolver return values does not
exercise optional provider storage or immutable operation bindings.

## Decision

One canonical `HfMatrixProfile` describes an explicitly selected trusted input
file. It binds six positive finite effective-rate/launch measurements and a
nonnegative first-use setup cost to source identity, exact resource facts,
provider/toolkit/runtime/driver/cuBLAS versions, build mode, loaded ELF build
IDs (including the native library's actual compiled image), workload bounds
and a retained evidence digest. Unknown or mismatched
identity rejects the record; no marketing name or benchmark identity selects
a profile. Identity observation lives in the numerical owner, not lifecycle code.

The actual packer resolves the admitted matrix route once. Host topology retains
the entire compatible record and the resolved decision; arena planning and
execution consume that same value. Editing rates, provenance or bounds invalidates
the prepared topology, even if the selected provider remains the same. Every
replay rereads the file, but execution never rereads it or changes its binding.
Matrix-only/physical-reference exports keep their separate initialization and
capacity contracts. The existing optional-provider failure retry remains native.

The native-only resource-query ABIs inspect only whether a profile path was
requested. They refuse rather than load the file, initialize CUDA, or advertise
a native-only bound for an optionally larger provider route. This refusal is
conservative even if the requested file later fails compatibility checks.

## Rejected alternatives

- Ship synthetic rates or inferred hardware peaks as measured calibrations:
  neither qualifies a production winner. Tests deliberately use synthetic
  ordering records and identify them as such.
- Assume `max_iterations` predicts reuse: convergence can finish much earlier.
  Charge the complete provider first-use cost rather than silently amortizing it.
- Key profiles only by compute capability or cuBLAS version: distinct resource
  targets and different loaded builds can have materially different behavior.
- Cache by profile pathname or evidence label: file contents can change between
  replays, and unchanged labels do not prove unchanged rates.
- Infer persistent-ERI or angular-queue profitability from matrix rates: these
  need different work/reuse and scientific endpoint evidence.

## Invariants

- No-profile AO/matrix crossovers and compiler queue-domain defaults stay intact.
- Parsing is bounded and all-or-nothing; invalid inputs never partly override
  the fallback or its capacity plan. Finite inputs with overflowing estimates
  also fall back.
- The profile is an explicit trusted scheduling input, not a precision option
  or proof that the referenced evidence contains valid measurements. No
  production path imports PySCF, performs CPU-oracle work, or autotunes during SCF.
- Loader-identity observers may reject unsupported/trampoline/static images;
  weakening that boundary requires a separately authenticated execution identity.
- Changes to measured-work bounds, source/stack identity or rates must not replay
  previously captured bindings or charge the old route's storage.

## Evidence and limits

Executable host coverage compiles the actual parser, packer, arena and tensor
binding, then executes the native kernel bodies and vendor-call doubles against
NumPy. It covers tiny/batched RHF/UHF domains, different compatible profiles,
resource fallback, alternate synthetic targets, malformed/stale input and cache
invalidation. These probes are not CUDA or cuBLAS numerical qualification.

The separate native CUDA fixture exercises complete ordinary Direct-HF calls
with synthetic ordering records and independent PySCF 2.14.0 energy/force values
already retained by `test_incremental_direct_jk_cuda.cpp`. It checks actual
prepared bindings and cache invalidation, not just cost-model outputs. Device
execution evidence must be retained separately and must identify source and
loaded library identities. Its owned-byte count is not a complete memory peak.

No calibrated production profile, new ERI crossover, angular schedule promotion,
cross-device timing result or endpoint speedup follows from these changes.
Issue #597's screened-quartet/reuse profitability and full matched performance
qualification remain open.

## Revisit when

Measured primitive and complete-endpoint evidence supports a bounded workload
domain on an authenticated execution stack. A profile writer/distributor or
less conservative setup amortization needs its own provenance/reuse contract;
it must not turn the synthetic qualification fixture into a shipped winner.

## References

- Issue #597 and structural PR #3052.
- `docs/maintainer/performance_engineering.md` for the current input contract.
- `2026-11-08-direct-jk-topology-profile-admission.md` for the structural baseline.
