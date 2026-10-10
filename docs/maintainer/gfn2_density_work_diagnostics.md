# GFN2 CUDA density work diagnostics

The SCC graph can record actual plain and energy-weighted density contraction
visits without adding a host read inside the iteration loop. Build the internal
CUDA runtime with `-DGENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS=ON`; the default
is `OFF`. Both builds generate the same ordinary weighted-Gram fragment. The
diagnostic build also generates an instrumented sibling, which is a separate
execution artifact. Pin the CUDA architecture, compiler cache, source revision,
compiler and library identities when comparing the builds.

An internal `Gfn2CudaExecutionCache` must call `enable_density_diagnostics()`
before its first prepared topology. After `execute_restricted_gfn2_cuda`
returns, call `read_density_diagnostics()` once the endpoint has settled. A
build without the compile option rejects enable/read. An early request rejection
has no call-owned receipt buffer. A failed replacement candidate may have a
bounded host snapshot after settlement; its device arena is released and the
old prepared topology remains usable.

Each receipt identifies a launched contraction CTA by slot, system, channel
and matrix tile. Its orbital/pair dimensions and counters come from the
executed generated checked-pair loop. `plain_visits` reaches two coefficient
read expressions and one occupation-weight read expression per visit;
`weighted_visits` reaches the energy-weight expression after successful plain
checks. These are semantic visits, not DRAM transactions. Completion and
publication counters expose partial nonfinite work. Inactive, invalid-active,
unused-channel, prior-error and closed-sequence CTAs carry explicit skip status
and zero work.
Zero occupations still require orbital-loop visits.

The runtime reserves at most 64 MiB from the generated tile count, two channel
slots per system and `maximum_iterations + 2` launch slots. It returns the
actual capacity and allocated bytes. `attempted_receipts > receipt_capacity`
means the retained records are truncated; never replace attempted work with
the retained count. `graph_submitted` comes from the actual submitted graph
count, and `endpoint_completed` is separate from graph submission.

`cta_cycles` measures one instrumented CTA on one SM, with barriers around its
contraction body. Do not add cycles across CTAs or divide them by a whole-graph
duration to claim a density fraction. Report full energy plus host-force
off/on/replay timing separately, including diagnostic readback in the on arm.
P and W updates are interleaved, so the fused CTA duration is not separate
plain-versus-weighted elapsed time.
Source-matched independent numerical checks and complete endpoints remain
required before any provider or default-path claim. These receipts do not
resolve the independent 768-AO tblite failure on #560/#1879.

The focused generator, provenance and host transaction tests are
`test_gfn2_density_receipt_codegen.py`, `test_gfn2_cuda_provenance.py` and
`test_gfn2_density_receipt_transaction.py`. The real-device ragged work oracle
is in `test_gfn2_electronic_schedule.py`; complete endpoint and failure/replay
coverage is in `test_gfn2_cuda_bootstrap.py`. The two CUDA pytest runners
require an explicit qualification setting and a real Slurm allocation; never
fabricate `SLURM_JOB_ID` to run them elsewhere. The test-only
`GENERATIVEQC_GFN2_DENSITY_LOWLEVEL_CSV` and
`GENERATIVEQC_GFN2_DENSITY_RECEIPT_CSV` paths retain bounded raw records
after endpoint completion; record their checksums with source and library
identities. The rationale and rejected
alternatives are in the [Agent Note](../../.agents/notes/implemented/performance/2026-10-10-gfn2-density-work-receipts.md).
