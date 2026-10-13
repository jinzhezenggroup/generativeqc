# DF setup and SCF workspace lifetimes

Status: implemented
Date: 2026-10-13

## Diagnosis

The 1088-AO, 3712-auxiliary pure-J r2SCAN-3c endpoint cannot keep one packed
fitted tensor under its automatic 32-GiB-device budget. A prior response-floor
experiment failed because its minimum value reservation plus the complete
response floor exceeded the same total. That failed experiment is retained in
`../../rejected/2026-10-13-wide-pure-j-response-floor.md`.

The value planner adds temporary metric diagonalization storage to lazy SCF,
eigenframe and final-state reservations. In the native owner, `SetupBuffers`
expires before setup can return and before any SCF API can allocate those lazy
owners. The retained metric eigenvectors, eigenvalues and inverse square root
are transferred explicitly; they remain in the common live set.

## Initial candidate

Use common persistent storage plus max(setup scratch, lazy SCF state), preserving
all existing provider-query upper bounds and retained tensor accounting. Apply
the same lifetime rule to native peak diagnostics. Host accounting is unchanged.
For an authorized rank-zero singleton pure-J owner, retry its complete response
floor only after the existing preferred response margin cannot admit a packed
value plan. Keep total, free-memory observation, headroom, explicit budget and
response overrides unchanged. Insufficient resources still stream.

Host planner checks now admit the real wide shape with the original
31-GiB-free/32-GiB-total envelope and 100-MiB source reservation. A 28-GiB-free
control remains streamed. This is dry planning, not a GPU-memory observation.

## Invariants and acceptance

No numerical formula, screening or stopping rule changes. The setup routine
must drain before destruction/ownership transfer. Future asynchronous setup or
SCF allocation during setup must re-evaluate this lifetime proof. Metric and
AO solver query bounds remain enforced before allocation. Qualify actual GPU
allocation diagnostics, explicit budgets, direct independent E/F oracles and
complete cold/warm/moved 96-atom endpoints before making speedup claims.

## References

Issues #2411 and #2415; `src/scf/cuda/df_setup_internal.hpp`;
`tests/python/test_df_preparation_budget.py`;
`tests/python/test_df_property_budget_cuda.py`.

## Moved-geometry qualification and revised automatic policy

Source `fedf81b83` passes the independent 3/12/48-atom 12-call populations.
Its 96-atom cold and all five warm calls also pass, but the moved call exposes
another capacity crossover. A separate one-repeat trace observes free memory
fall from 33,110,228,992 to 31,497,519,104 bytes after persistent CUDA modules
and force arenas have been initialized. The generic total target falls from
21,676,204,032 to 20,466,671,616 bytes. This switches single packed B (storage 2)
to streamed dense panels (storage 0), despite ample physical memory after the
absolute reservation. The incomplete first candidate is retained as such.

The revised resolver still tries both existing splits inside the generic target
first. Only an authenticated automatic rank-zero pure-J owner can then borrow
from the additional fractional margin, bounded by observed free memory minus
the unchanged absolute headroom. It bisects the actual packed planner's monotone
byte admission to select the smallest sufficient value allowance, adds the
complete shape-derived response floor, and publishes the final total. This is
a deliberate increase of the automatic target for that crossover, not a change
to an explicit caller cap, solver workspace bound, physical-memory observation,
SCF stopping rule or benchmark deadline. A 23-GiB-free control retains streaming;
a 29-GiB-free rebuild admits B while preserving the absolute headroom. Failure
to fit the minimum resident plan leaves the original bounded fallback.

Dropping a retained force arena alone would not reclaim the loaded-module
footprint and would sacrifice valid same-geometry reuse. Lowering a cuSolver
workspace estimate without querying its actual provider was also rejected.
The current candidate instead keeps the conservative allocation formulas and
requires complete moved/moved-warm qualification before promotion.

## Complete qualification

Frozen integrated source `3660e04d2b3912de1f349e892852e4b19957b96b` and native
library SHA256 `e5911b58c3b1b4bc1b2abd288c0e5672e0b111e81a13dc9956bd930f70134c3e`
pass all 12 independent energy/analytic-force calls at 3, 12, 48 and 96 atoms.
The 96-atom complete cold/warm/moved/moved-warm times are
226.247 / 23.080 / 145.915 / 23.085 seconds; warm values are medians of all
five frozen-seed replays. The original 1800-second whole-point limit, FP64
arithmetic, SCF tolerances, matched grid, and 1e-8 Eh / 1e-7 Eh/Bohr gates
are unchanged. Maximum 96-atom errors are 3.638e-11 Eh and 1.668e-11 Eh/Bohr.
Cold and moved metric diagnostics both report non-streamed values, with
auxiliary tiles 13 and 5 and value-owner device bounds 18,885,954,645 and
18,658,675,797 bytes. The endpoint includes D4/gCP and host force return.

All GPU work used finite Slurm allocations on n1/node1, main partition,
one RTX 5090 per job, preserving assigned CUDA visibility. Independent
resource/property tests pass (13 cases); scoped host planner/resource tests
pass (38 cases). The property crossover fixture now uses 16 MiB: energy
retains values while energy+forces streams. Old 24/22/20-MiB assumptions
stopped separating those cases after correcting the setup lifetime. The
insufficient-capacity host control uses 23 GiB free; 24 GiB actually fits.
Failed earlier fixtures and the source `fedf81b83` moved timeout are retained
with benchmark evidence rather than relabeled as acceptance.

The standalone PR is rebased on `05844f847`; these integrated GPU measurements
retain their actual source identity. The integration also includes the bounded
grid-capacity repairs needed by the large r2SCAN-3c endpoint.
