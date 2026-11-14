# Decision: bounded exact AO prefix in the native nuclear response reader

Status: implemented
Date: 2026-11-14

## Problem

After ordered moving-XC row reuse, a newly solved native I–CH3–I carbon-z
response still spent most coupled-response time in induced Fock actions. A
fresh physical-frame component decomposition measured warm J/K at about
0.041 s and XC at about 1.424 s per 1.468 s complete induced-Fock action.
Optimizing J/K would therefore miss the measured bottleneck. The generic XC
partial bank correctly still reads/authenticates AO-derived fields on hits.

## Decision

Keep generic XC authentication unchanged. The native nuclear owner optionally
memoizes a whole-tile prefix of its fixed-grid AO values and first derivatives,
after reserving source panels, actual GMRES and the XC partial bank. Charge
the actual packed-basis snapshot, retained coordinates and four AO components
to the original response ceiling. There is no additional cap or full-grid
allocation requirement. Entries/environment metadata are bounded by this prefix.

Each hit checks exact basis/coordinate bytes and the complete input floating
environment, and restores the evaluator's resulting environment, including
exception flags. Dimensions are captured/validated before pointer arithmetic;
tail pointers are never formed outside retained storage. Only completed finite
evaluations publish entries. Weights are always current. Both existing nuclear
frame/epoch authentication guards remain around every reader invocation.
Input/environment misses, uncached tails and optional allocation failures stream
the ordinary AO evaluator. This is neither an approximate density screen nor
a change to AO reduction order, XC branches, source physics or solver gates.

## Evidence and rejected alternatives

Source-matched raw records are under
`build/qualification/host-response-ao-prefix-qualified-20261114`; the original
unsafe private prototype and measurements remain separate under
`host-response-ao-prefix-candidate-20261114`. The initial prototype formed
out-of-range tail pointers without dereferencing them; those measurements are
not sanitizer qualification. Compiled tests exercise production-header numeric
parity, prefix/tail capacity, allocation failure, input/environment changes,
empty reads, dimensions and failed/nonfinite publication. ASan/UBSan additionally
instrument the real native AO evaluator.

On the physical component, 98,456,992 numeric bytes fit in the existing
98,797,368-byte remainder, retaining 37,632 of 87,458 points. Corrected warm
complete induced-Fock actions are about 1.300 s rather than 1.469 s, with
bitwise-equal outputs. Three actions evaluate 187,110 AO points and replay
75,264 instead of evaluating all 262,374. These are component measurements,
not a qualified whole-Hessian/TS speedup or permission to retry an unchanged
full matrix with a larger time cap.

Exact-zero AO compaction was rejected: 99.05% of ordered pairs remained.
A private two-point SIMD variant saved only about 5% per induced action and
was not promoted. This host has AVX2 but no AVX512. Naive BLAS remains
scientifically disqualified by vacuum classification differences; changing
reduction order or relaxing the original physical residual is not a fallback.

## Actual coupled-response qualification

The final frozen core/source pair is additionally exercised by
`host-ao-prefix-carbon-response-20261114` and
`host-response-ao-prefix-component-final-20261114`. The latter authenticates
the newly solved, same-source D′ as its component direction; a preserved initial
attempt correctly refuses the old-source direction rather than weakening its
provenance guard. No saved D′/W′ is injected into the actual coupled solve.

The actual solve retains bitwise-equal D′, W′ and separate source panels,
14 iterations / 28 GMRES actions, and the original physical residual
8.406008203576136e-14. Its coupled phase decreases from 50.72137 to 45.14357 s;
GMRES action time decreases from 41.23883 to 35.87398 s. The whole bounded
component only decreases from 242.69285 to 240.57739 s (about 0.87%) in this
single comparison: first-source preparation instead measures 172.22236 s.
This is not a complete-endpoint speedup or acceptance claim.

All 2,623,740 XC point visits remain. AO work is 1,532,412 evaluated points
and 1,091,328 replayed points. Actual response fixed arrays are 132,987,664
bytes plus 889,688 GMRES bytes inside the original 128 MiB response cap;
the aggregate reservation remains 342,467,888 bytes under its 512 MiB cap.
Peak process RSS increases from 437,500 to 531,968 KiB, as expected when using
previously unused cache headroom; these numeric-array caps are not RSS caps.

Qualification covers 107 Python test groups, six compiled native groups,
six documentation groups and thirteen sanitizer scenarios. Two 900-second
whole-suite timeouts are retained; only actually completed setup/call/teardown
triples count, with all missing tests executed in a separate bounded invocation.
No numerical, derivative, response, reference or wall-time gate is relaxed.

## Revisit conditions

Revisit retention order/capacity only with complete coupled-response and endpoint
timing, semantic AO/XC/J/K/Krylov work, independent scientific gates and the
same source/artifact identities. Mixed Ni/I/Cs, larger endpoints and the native
TS contract remain separate unresolved qualification requirements.
