# Default retained-shell RHF cross derivative qualification

Frozen master `a50eb54f362a1b27edcb0f06353e7a04aa9979e2`, plus the retained reconstruction patch for
the candidate. Stock is independently built untouched source, not a candidate
opt-out. Four fresh complete ethane230 E+F processes run in ABBA order on the
same Slurm-assigned RTX 5090, strict FP64, CCSD Q32 and Lambda Q32. Candidate
uses `auto`, leaving `RHFFrameResponseOptions` defaults untouched; stock uses
its established symmetric two-pass schedule.

| Mean native timing | Untouched stock | Actual default candidate |
| --- | ---: | ---: |
| Complete E+F | 250.408748 s | 227.374415 s |
| Two-electron nuclear phase | 46.119834 s | 22.998365 s |

Observed endpoint reduction: 9.199%; two
shell derivative traversals become one, with no generic traversal. Two samples
per side support this scoped observation, not a fleet-wide/statistical claim.
Numerical publication acceptance is separate from generalized performance
promotion; the shared envelope reports the latter as not-run.

Real-device independent RHF, signed shell/canonical derivative and complete
small DF-CCSD(T) force tests pass, including all nuclear coordinates of water.
Memcheck covers default RHF and signed derivatives. Tests, Slurm allocations,
binary hashes, unedited endpoint outputs, process timings, GPU telemetry summaries and sampled memory traces,
independent/retained references and selected compiler-cache/build completion receipts are in
`raw-receipts.json.xz`.

Large endpoint force qualification uses a retained independently qualified
complete force and independent energy. Its independent finite-difference
receipt covers **two components at two steps**, not all large-force components.
Every endpoint separately passes Lambda/Z residual, frame stationarity,
translation and retained-force/energy gates. Numeric tolerances and measured
block errors are in `validation.json.xz`.

Reconstruct stock from the exact revision; reconstruct candidate with
`source.patch.xz`. Build both with verified ccache, Release/sm120, CUDA 12.9.1,
stationary CPU/CUDA force AOT and automatic XC AOT disabled, then compile/link
the respective native endpoint adapters. Place them in the stock/candidate
bin/lib layout named by `reproduce.sh`; the original finite Slurm command and
qualification scripts are retained in the raw receipts. Preserve Slurm device
visibility and use the checked-in ethane230/reference inputs.

The reconstruction patch retains every modified executed production/compiler,
endpoint adapter and native/GPU test source. Documentation, inventories and
host-only review tests are omitted from the reconstruction payload, not from
the PR. The full measured review-patch digest is retained in the envelope.

Already tracked ethane230/reference inputs are retained by exact repository
paths, source revisions and byte hashes in the receipts instead of duplicating
their payloads. All endpoint outputs, numerical gates and source patches remain
in this bundle; reproduction depends on Git inputs, not an expiring archive.
