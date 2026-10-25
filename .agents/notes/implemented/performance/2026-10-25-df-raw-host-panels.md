# Decision: Reuse raw generated DF panels from bounded RAM

Status: implemented
Date: 2026-10-25

## Decision and boundaries

Generated streamed K can revisit the same three-center integrals across AO-row
and whitened-auxiliary panels, then repeat that traversal at each SCF density.
Retain an auxiliary-complete prefix of raw A in pageable FP64 RAM within the
existing host value budget and automatic machine/cgroup headroom. A consumer
crossing the cached pair boundary copies the prefix and generates the suffix.
Refusal retains the existing bounded generator. Requiring the entire tensor
was rejected: at many natural capacity boundaries the full dense host A cannot
fit this cap, even though a useful prefix could remove recurring K work.
Setup borrows existing device K scratch, evaluates through the same CUDA source,
downloads rectangles, checks finiteness, then publishes one immutable owner.

GPU-resident B keeps priority and allocates no host A. Dedicated MD Coulomb,
imported-tensor and batch paths are not displaced. Dense J may replay the same
raw reader; packed J keeps its generated traversal. Most targeted reuse is in
dense/occupied K. Force response remains independently generated and retains
its exact forward metric eigensystem; no cached value is a derivative integral.

Both host and device budgets remain distinct hard allowances. The caller supplies
its existing value-side host cap after required DIIS/plan admission. Charge the
old host peak plus actual vector capacity, even though setup temporaries have
already unwound. This conservative rule preserves the DFReference owner's
combined reference/correlation reservation. Unknown automatic RAM information
and optional host allocation failure refuse the cache. Execution faults and
nonfinite values propagate before cache publication.

## Scheduling and alternatives

A pageable source cannot be read inside the current device SCF graph. Admitted
host backing therefore selects eager SCF; the helper separately retains device
regeneration during external capture. Its array remains alive until a stream
fence at owner teardown. Changing geometry or the requested policy rebuilds the
owner. Automatic capacity discovery is not a varying cache-identity token.

Raw A serves both projected and dense schedules without changing metric gauge.
Caching fitted B/E could also remove repeated whitening, but must distinguish
spectral E from symmetric B and preserve discarded metric directions required
by response. The first implementation leaves whitening on device. U=B*C is
orbital-dependent and must never become a geometry-only cache. Pinning the whole
array and randomly reading mapped host memory were rejected: bounded ordinary
uploads avoid locking large RAM inventories and reuse existing device scratch.

Performance qualification must include the cost of losing graph replay, raw
construction, final response, and cold/warm/moved geometries. Removing recurrence
alone is not sufficient evidence to promote a broader policy. Full GPU and
capacity-refusal controls accompany the source-backed tests.

## Qualification

[Source-bound endpoint evidence](../../../../benchmarks/results/df-host-panels-20261025/README.md)
qualifies complete RHF cold/warm/moved E+F and independent UHF E+F. At 192 AO
and a natural 64 MiB DF allowance, an 80.53% raw prefix cuts median cold E+F
from 8.0221 to 5.6391 seconds (29.7%); all arms have 18 SCF iterations and 19
Focks. At 96 AO/20 MiB, the natural packed full-GPU path remains much faster
than dense streamed+RAM and stays selected. 38 GPU tests and five memcheck
cases pass. Baseline trace counters include graph construction, so only the
complete semantic counts and eager RAM traffic are reported as executed work.

After integrating #2765, qualification at `f88f5ac5e` repeats all 38 tests and
five memcheck cases without errors. Natural 192-AO cold E+F is 8.0253 -> 5.6393 s
(29.7%); warm is 1.9324 -> 1.6596 s and moved is 7.2477 -> 5.1328 s. The same
80.53% raw prefix and identical SCF/Fock counts confirm composition with the
current HF endpoint policy.
