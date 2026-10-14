# Decision: retain complete WB97M-V timings and omit incomplete DF/96

Status: implemented
Date: 2026-10-15

## Problem

On master `810f081ef293256ebe20beb7298f9d21996fec81`, the README DF WB97M-V
series had native timings only through 12 atoms. New measurement completes
DF/48 without a matching reference, but default DF/96 converges SCF and returns
status 1 with no forces: `stationary CUDA byte budget is not representable`.
A missing reference cannot justify publishing energy-only or failed-force calls.

## Diagnosis

The composite consumer reserves the lazy exact-LR first-derivative provider
before admitting semilocal grid scratch. Native capacity queries report a
424,962,557-byte lazy LR reserve and a 434,257,968-byte nonlocal reserve.
The metadata-only planner admits a 1024-point tile with 1,981,354,125 device
bytes and 2,605,403,181 host bytes under explicit 4 GiB device/host allowances;
the default 2 GiB host allowance cannot fit even the smallest supported tile.
Raising both force allowances still fails the generated DF stationary response.

Preparation-only probes leave 1,178,271,744 physically free device bytes under
automatic DF selection, versus 8,818,196,480 with the public 12 GiB workspace
hint. Neither CuPy nor the CUDA asynchronous pool retains unused memory in
these probes; trimming does not help. The probes do not explain the owner of
every allocation. The explicit hint splits into 8,228,165,057 value bytes and
4,656,736,831 response bytes, selecting a bounded streamed value fallback.
Memory-bounded execution need not be work-bounded: repeated three-center value
generation remains charged to the complete endpoint.

## Decision and invariants

The final explicit-budget DF/96 attempt produced no completed cold E+F sample
before the user requested cancellation. Omit only that native point; do not
wait, rerun it, publish partial SCF timing or invent a timeout observation.
Retain the failed resource probes separately from successful timing medians.

Publish all seven complete populations: direct 3/12/48/96 and DF 3/12/48,
with all twelve calls per point. Missing references never suppress successful
native timings. DF/48 remains unchecked, with open markers and null errors.
Only fitted full-range J/K plus exact-direct LR K references can qualify this
DF Hamiltonian; old fully fitted LR references must not be reused.
Other functional panels reference their prior archive rather than duplicate
or relabel its native observations. Rebasing the PR must not relabel the
measured source snapshot as a later master measurement.

## Evidence and consequences

`benchmarks/results/readme-wb97mv-20261014/` retains 84 endpoint observations,
72 independently checked and 12 unchecked, plus the resource probes and
byte-replay audit. No production kernel, force default or fallback is modified.
The aggregate evidence budget leaves too little room for duplicate full-panel
SVG markup. Compact only the new figures with Scour; archive the two baseline
SVGs losslessly as `.svg.xz` and verify their original hashes and byte replay.
Keep the baseline scientific reductions unchanged and historical previews
bound to an existing Git snapshot. No storage policy limit is raised.
Complete timing still requires successful status, converged SCF and finite
exported host forces. A future DF/96 refresh should qualify complete-force
resource admission and independent large-DF numerics, without weakening gates.
