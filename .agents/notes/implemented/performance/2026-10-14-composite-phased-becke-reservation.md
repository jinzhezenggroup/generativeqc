# Decision: reserve the composite owner's requested phased Becke storage

Status: implementation awaiting endpoint qualification
Date: 2026-10-14

The completed 96-atom frozen-VV10 trial (`003e5c7b0`, Slurm 8298) reports a
70.001-second first warm endpoint. Its force endpoint consumes 44.959 seconds,
including 34.076 seconds in the geometry drain and 10.606 seconds in native
integral derivatives. Both semilocal and nonlocal geometry owners report zero
phased storage and zero phased batches, despite the existing ordinary source
policy requesting phased Becke at 48 or more atoms.

The composite dry planner forwarded only the experimental primitive selection.
It consequently returned an ordinary source arena. Each runtime source received
that exact arena size as its budget, so its later phased request had no space
and fell back. Spare capacity in the complete composite budget could not help.

Forward the same resolved ordinary phased selection to the dry planner and both
runtime owners. Charge both optional phase panels inside the existing complete
device bound, keeping the source planner's no-room fallback and point lanes.
Resolve the primitive selection once too, and include both selections in the
prepared identity. No new device equation, grid, precision, convergence rule,
primitive default or user budget is introduced.

Host resource checks cover both spin layouts, exact double-owner accounting and
bounded fallback. Independent complete molecular energy/forces and source-bound
48/96 endpoint populations must qualify the actual selected route before
performance claims or issue closure. The original unphased measurements remain
retained; a requested policy alone is never evidence of executed phased work.
