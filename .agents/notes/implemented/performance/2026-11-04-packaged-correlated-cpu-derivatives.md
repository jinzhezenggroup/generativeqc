# Decision: capability-derived packaged correlated CPU derivatives

Status: implemented
Date: 2026-11-04

## Problem

Retaining the CPU prepared source and choosing bounded source tiles removes
repeated AO generation and oversized MO-transform work, but does not select the
packaged derivative artifact. The conventional CPU derivative consumer still
called the independent reference wrapper even for covered shell components.
This consumer also serves RCCSD(T), so an MP2-only numerical test is insufficient.

## Decision

The integral owner exposes a no-throw shell-quartet capability query derived
from its actual compiled Cartesian component table and artifact availability.
Invalid indices and unsupported components return false. Explicit packaged
execution uses the same coverage check and still rejects unsupported inputs;
the capability query does not validate geometry, weights or scientific results.

The shared conventional correlated CPU derivative selects the existing packaged
multi-channel contraction with one channel for covered quartets, otherwise the
existing native reference contraction. It neither hard-codes a consumer-side
angular-momentum whitelist nor initiates JIT. The three-argument single-channel
reference wrapper remains unchanged as an independent execution oracle. RI and
unrestricted MP2 derivative routes are unchanged.

The optional `mp2_cpu_derivative` trace records artifact availability and, on
successful completion, selected packaged/reference quartet contractions and
primitive evaluations reported by the integral owner. Quartet counts can include
zero-weight contractions; they are not kernel-launch counts. They are not a
complete endpoint semantic-work ledger or an allocation measurement.

## Invariants and rejected alternatives

- Preserve the existing equations, normalization, shell-slot restoration,
  spherical expansion, FP64 accumulation and final-weight contraction.
- Keep unsupported/unavailable-artifact fallback explicit in the consumer.
  Do not turn explicit packaged execution into a silent fallback.
- Do not replace the independent reference wrapper with automatic dispatch.
- Do not promote the diagnostic interposer, duplicate a dense AO cache, override
  source tiles, or infer performance qualification from one publication.
- Preserve source-owner lifetime, memory-only retry and unknown complete-endpoint
  allocation telemetry from the prepared-source decision.

## Evidence and limits

Both CPU Release builds use verified sccache 0.16.0 and report native input
identity `22ca174873ff919b1095498ac5208efbc2cceada88677a80d9cc572869dc140d`.
The packaged build passes all 133 native CTests and 135 affected CPU correlated
Python tests, with 55 explicit skips. Its six physical MP2 source/routing tests
also pass separately. The artifact-disabled build
(`GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=OFF`) passes the native MP2
contract and all six physical MP2 tests, proving actual compiled-unavailable
fallback rather than a mocked capability. Native tests compare Cartesian and
spherical contractions, reject invalid/f components without publishing work,
and execute the correlated f-shell fallback against the independent wrapper.
The f-shell fixture is a cotangent contract, not an SCF molecular oracle.

A separate clean-product physical H2 DZ/TZ Standard CCSD(T)/CBS publication
returns all five component E/F results and the composite in 41.682935499586165
seconds, including construction and owner teardown. All independent 1e-9
hartree energy and 2e-6 complete-force gates pass; composite errors are
4.5062265030537674e-10 and 1.1071013151370268e-9 respectively. This observation
uses published parent `682865864e369a90127e5c33fc20886529722e6c` plus its retained
exact dirty-source patch, not a committed candidate. Its native library SHA-256
is `69952bd03605064f1b3832ec4a039ab11ae672fef3613ba569a6b3d4be5d214a`.
All 1,439 trace scopes close successfully. Two high-basis derivative scopes each
report 20,736 packaged quartets, zero reference quartets and 2,085,136 primitive
evaluations. These counts describe those scopes only.

This is independent science evidence, not ABBA/BAAB, a 1.05 performance gate,
whole-endpoint work/allocation qualification, or #2227 completion. The earlier
frozen campaign remains incomplete with all eight CBS blocks missing. No real
GPU execution is included. CI on the preceding published head failed in stale
executable test fixtures; repair their missing enclosing-state stand-ins rather
than removing lifecycle/failure-injection coverage.

All ten repaired retirement/lifetime tests pass. The CPU callback probe injects
both admission and allocation failure into the actual method-layer phase
callback: partial outputs unwind, the borrowing view dies before its owner,
one bounded replay occurs, and later phases retain the raw source. A failing
replay propagates without publishing; scientific errors preserve the resident
owner and are not retried. The boundary implementations are stand-ins: this is
host lifecycle qualification, not real molecular allocation-failure evidence.

A second observation on clean committed source
`f8c78a3700cb8eca6b49128842546990613d9c7d` returns in 41.736168964765966 seconds
with the same native identity, binary and numerical errors. Its compact packet
at `benchmarks/results/packaged-correlated-cpu-20261104/` retains both journal
records, exact original driver, environment/build provenance, audited diagnostic
summary and offline replay of all twelve independent E/F block gates. Acceptance
is numerical-only; performance/production qualification and whole-endpoint
work/allocation measurements remain absent. The first dirty observation and the
earlier failed campaign are not relabeled or replaced.

The prospective GitHub test merge exceeded the aggregate evidence budget when
the new packet was stored as loose JSON/source. Retain its journal, exact driver,
provenance and diagnostic summary in one lossless sample container. Also compress
the earlier packet's evidence/provenance/summary records byte-for-byte; keep its
raw samples, frozen sources, scientific result and exit-2 replay unchanged.
Check the actual test-merge inventory, not only this branch's smaller checkout.

## Revisit when

Additional compiled components or derivative orders become available, or a
source-matched complete endpoint campaign supplies the missing semantic-work,
allocation and unchanged performance gates. Do not infer general f-shell
packaged support or general correlated method qualification from this slice.

## References

- Issue #2227; draft PR #2972.
- [Prepared source decision](2026-11-04-cpu-mp2-prepared-force-source.md).
- [Rejected derivative shortcut](../../rejected/2026-11-04-cbs-generated-derivative-shortcut.md).
- `tests/native/test_mp2_contract.cpp`.
- `tests/python/test_mp2_cpu_reference_source.py`.
