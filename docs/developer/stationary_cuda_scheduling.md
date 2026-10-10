# Stationary CUDA grid scheduling

Ordinary CUDA DFT force endpoints, including PBE0, use the complete-owner
byte-budget planner instead of a fixed 256-point grid tile. The consumer prefers
512 points, then tries the existing 256-point fallback and progressively smaller
tiles. Every candidate must fit the simultaneous grid, tensor, stationary source,
resident AO and native integral owners. The ordinary public budgets remain
512 MiB device and 256 MiB host; selecting a larger tile does not raise them.
Known native fitted providers prefer 256 points on large grids so the existing
Becke phase cache can coexist with their separately bounded integral consumers.

The shared compiler planner accepts `preferred_tile_points` as an execution
preference, not an admission override. Its default remains 1024 for other owners,
including the composite path. Explicit `tile_points` requests are attempted once
and fail if the complete owner cannot admit them. They are not silently retiled.
Changing a tile never changes the scientific grid or reduces whole-grid storage.

## Indexed AO policy

Budget-admitted ordinary tiles retain the same active-AO producer selection as
explicit 256-point callers. The existing continuous dense-work crossover,
derivative/spin domains, resident-grid requirement, integral-provider domains,
cutoffs, cache budgets and occupancy gates still apply to direct profiles.
Fitted ordinary forces above the same dense point-times-AO-squared crossover
instead inspect every requested AO jet with the native bitmask producer. They
retain the existing `1e-16` force AO cutoff, a 64 MiB optional map allowance and
an 80% average active-fraction admission gate. This is a sampled AO-jet cutoff,
not a density or grid-weight cutoff; "exact" describes the selected labels at
that explicit cutoff, not unscreened mathematics. Order-two forces discover
their own order-two labels rather than borrowing order-one SCF masks.
Selection is not a promise
that every individual AO union fits: existing capacity and occupancy misses
retain the bounded dense fallback.

The policy has no GPU product, SM-version, atom-count, AO-count or grid-size
whitelist. Device capabilities and complete-owner resources determine admission.
Qualification on a particular device does not establish a speedup on every CUDA
device.

## Cooperative geometry resources

The cooperative Becke geometry consumer uses 128 threads per point and retains
four pair-panel rows. The geometry scratch ceiling is 16 MiB; actual allocation
is shape- and budget-admitted, not an unconditional 16 MiB allocation. Existing
capability checks and generic/serial fallbacks remain available.

Cooperative geometry is distinct from the
[ordered Becke normalization](becke_normalization.md) kernel. The latter keeps its
canonical scalar-AD denominator order and its separate resource admission.
Neither schedule changes XC expressions, derivative formulas, grid weights,
precision or the SCF iteration/convergence contract.

## Qualification

`python -m benchmarks.qualify_cuda_schedule native ...` runs the standard complete
PBE0 endpoint protocol and retains the actual production force-work receipts.
It does not force an AO producer to make a source trial look eligible.
`--intrusive` enables device timers only for separate diagnostic runs; do not mix
those samples into a clean endpoint timing population.

Use `tools/cuda_schedule_trials.py` to prepare source-frozen schedule candidates
and the launch assessment in `generativeqc_compiler.common.cuda_launch` to check
actual kernel attributes, dynamic/shared reservations and finite grid limits.
Analytical occupancy bounds are not achieved occupancy or measured bandwidth.
Build through a verified compiler cache, run GPU work through finite Slurm jobs,
and retain complete cold/warm/moved/moved-warm energy-plus-force timings, actual
solver histories, semantic work counts and independent numerical gates.

The rationale, rejected trials and measured scope are retained in
`.agents/notes/implemented/performance/2026-10-06-pbe0-budget-admitted-cuda-schedule.md`.
