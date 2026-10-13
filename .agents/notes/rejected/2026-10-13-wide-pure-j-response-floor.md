# Rejection: response-floor-only promotion of wide pure-J DF

Status: rejected
Date: 2026-10-13

## Candidate and result

The 1088-AO/3712-auxiliary r2SCAN-3c workload still streams at the 32-GiB
capacity boundary. A candidate retried the complete shape-derived pure-J
response floor after the preferred 20% response split, preserving the total
resolved cap and every explicit override. It did not admit the measured shape.
Do not retain this as a purported performance fix or enlarge a fixture's memory
to make it pass.

The host planner's 31-GiB-free/32-GiB-total, 100-MiB-source test resolves
21,743,271,936 total bytes, with 16,582,646,037 value and 5,160,625,899 response
bytes. Binary-searching the actual packed planner (including metric/solver,
lazy SCF, final-state and source owners) gives 20,634,379,567 bytes for its
minimum complete single-B plan. The calculated response floor plus this value
minimum still exceeds the total. This is shape-only admission evidence, not an
observed GPU free-memory value or an endpoint timing.

The candidate and failing diagnostic are retained in ignored local artifacts;
production code and tests were restored. Revisit with actual lifetime/resource
reductions or measured lower source ownership, preserving bounded fallback and
complete E/F endpoint qualification. A packed tensor's byte count alone is not
a complete capacity contract.
