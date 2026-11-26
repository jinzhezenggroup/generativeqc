# Decision: Require native layout for standalone CPU D3 admission

Status: implemented
Date: 2026-11-25

The complete KS correction owner and the standalone ragged D3 owner retain
different allocations. Missing complete-KS sizing may select standalone
execution, but once a CPU library is selected only that owner's native layout
can admit a budget. Matching a portable estimate on current inputs does not
guarantee the same lifetime or capacity for a different build.

Keep library-free estimates for planning and parity tests. Keep the older-CUDA
compatibility route unchanged until it has end-to-end device qualification.
Neither missing sizing nor a native query rejection should cause a CPU plan to
retry with the portable estimate. Unbudgeted execution and the local native
correction cap are unchanged.

`test_d3_native_resources.py` checks exact/one-byte-short caps and the
portable/native bounds. `test_named_ks_resources.py` verifies that losing the
complete-KS query still admits the native standalone layout and that losing both
queries refuses budgeted preparation. Tracked by #2227 (R2/R6).
