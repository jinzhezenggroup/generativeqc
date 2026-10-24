# Decision: execute the complete DF-CCSD(T) benchmark report contract

Status: implemented, CPU reporting contract only
Date: 2026-10-21

## Problem

The discarded-attempt report test extracted local lambdas and a single receipt
line from the endpoint source, then exercised three invented work fields against
a handwritten result layout. Its source scan classified the remaining fields
without executing their value mappings. Harmless local refactors could fail the
test, while omissions, duplicate JSON keys and incorrect native-member mappings
could escape the executable probe.

## Decision

The benchmark owns `df_ccsdt_force_report.hpp`. Its callable serializer accepts
the actual `methods::detail::DFCCSDTResult` and explicit reporting inputs; the
production endpoint calls the same implementation. File publication preserves
the existing open and close/flush failure messages. Scientific calls, options,
input validation and cumulative previous-endpoint timing remain in the endpoint.

The permanent host test compiles that header and real native data headers. The transitive source-response
header needs only an opaque `cudaStream_t` declaration; the test supplies the
`CUstream_st*` declaration without a CUDA runtime or any CUDA functions. The
production benchmark separately compiles with real CUDA headers. A
separately authored public schema lists native-member mappings and availability;
it is never generated from formatter text. Distinct scalar input sentinels and
boolean bit planes detect incorrect mappings. An independent input receipt uses
the actual C++ member types, without deciding report availability. Python parses
the complete report with a duplicate-key guard and compares its entire key set
and values. This receipt does not implement a second JSON formatter or scientific
result structure.

Requested boolean options also need distinct bit-plane codewords, separate from
the native boolean inputs. The initial fixture correlated several options; a
temporary damaged copy reading `forces` for `lambda_primal_matrix_requested`
incorrectly passed all 576 cases. Independent option codewords close that oracle
gap without changing the production report or scientific call.

Zero scalar measurements vary independently of measurement flags. Integer
receipts retain exact native values, including distinct near-limit 64-bit inputs;
signed non-binary-exact FP64 sentinels detect narrowing and sign loss. Each
nullable program identity has a distinct quoted/backslash input. Earlier coupled
or small/shared fixtures missed these corruptions. These are serialization
fixtures, not physical measurements or scientific admission evidence.

## Invariants

- Either discarded-attempt flag makes all original work/count/time/peak fields
  unavailable (`null`) across cold, warm and moved-warm output. Final-attempt
  measurements cannot reconstruct end-to-end attempted work.
- Energies, independent residuals, metadata, `native_seconds`,
  `previous_endpoint_seconds` and the independently measured
  `resident_jk_retry_seconds` keep their original availability. Overlapping times
  are not added. Request/measurement flags distinguish absent phases from zero.
- Precision-policy optional values, nullable program identities, quoted strings,
  empty/nonempty forces and floating precision retain their existing semantics.
  Nonfinite native values are not silently converted into new null measurements.
- No solver, SCF, CUDA, resource, stream, precision or scientific acceptance
  policy changes are part of this extraction.

## Rejected alternatives and remaining debt

Keeping the extracted lambda probe or pinning new helper names/source positions
would preserve the original failure mode. A copied native result layout would
hide type drift. Neither is used for this report contract.

The first `test_retained_cache_retry_covers_late_phases_once` test still extracts
the cache/retry owner and injects fake scientific phases. It is intentionally
unchanged and remains separate #2296 migration debt. The default-selector test
retains its real selector/call probes; only its obsolete report-spelling assertion
is replaced by the executable `ccsd_diis_history` JSON contract.

Historical attribution publications and their source-recovery companion list
retain their publication identities. A future build/capture must include this
new header as an actual dependency; a current report must never be substituted
for a historical measured source.

## Verification and limits

Run the report test, retained retry test, default-selector test and report
consumers with a verified compiler cache and finite build/run timeouts on Linux.
Bind qualification to LF source blobs and retain earlier failures. Opt-in
migration comparisons and damaged-copy controls belong in ignored receipts, not
permanent source-fingerprint gates. A real benchmark compile/link checks
integration; running this host report does not run or qualify molecular science.
Parent issue #2296 stays open.

The formatted candidate passed 390 related Linux host/consumer cases with G++
12.3 and sccache 0.16.0, including the retained first retry and default-selector
probes. The earlier missing CUDA declaration compile failure and PATH launch
failure remain in task receipts. The actual benchmark TU compiles with CUDA 12.9
headers and its dependency file includes the new report header. The separately
sourced, immutable #2654 SDK accepts the symbols at link time, but its older
`CcPerformanceDiagnostic` layout differs from the current header. This is
`ABI_INCOMPATIBLE`, not link compatibility acceptance; that executable and library
are never run or loaded. A matching build is required for native integration
acceptance. The SDK cannot qualify this slice scientifically.

## Revisit when

The public report schema changes or a separate executable cache-owner seam can
replace the remaining first retry test with equivalent failure/lifetime coverage.
