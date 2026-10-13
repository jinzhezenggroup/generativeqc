# DF setup and SCF workspace lifetimes

Status: proposed (complete GPU qualification pending)
Date: 2026-10-13

## Diagnosis

The 1088-AO, 3712-auxiliary pure-J r2SCAN-3c endpoint cannot keep one packed
fitted tensor under its automatic 32-GiB-device budget. A prior response-floor
experiment failed because its minimum value reservation plus the complete
response floor exceeded the same total. That failed experiment is retained in
`../rejected/2026-10-13-wide-pure-j-response-floor.md`.

The value planner adds temporary metric diagonalization storage to lazy SCF,
eigenframe and final-state reservations. In the native owner, `SetupBuffers`
expires before setup can return and before any SCF API can allocate those lazy
owners. The retained metric eigenvectors, eigenvalues and inverse square root
are transferred explicitly; they remain in the common live set.

## Candidate

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
