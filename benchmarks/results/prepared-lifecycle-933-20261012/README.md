# Native prepared-endpoint lifecycle qualification

`receipt.json` records the bounded #933 native endpoint/artifact-lease slice,
not issue-wide acceptance. The exact baseline is master
`a2ba59894217215d68a7885235699249333c77b0`. Every endpoint pair retains native
source identity and library SHA256; raw input/trace receipts have SHA256 digests.

CPU and allocated RTX 5090 CUDA RHF/PBE comparisons use the complete requested
energy or energy/force replay, with preparation included for cold. Calculator
construction and process/library startup are outside these intervals. Five
repetitions cover cold, replay, changed input, reset, ragged failure and recovery.
This small H2/He matrix is not broad method qualification or a speedup claim.

Slurm jobs 7647 and 7650 use node1/main, `gpu:5090:1`, and a finite 30-minute
limit without overriding assigned device visibility. The first failing receipt
is preserved, not rewritten as acceptance. The corrected lifecycle/artifact
suite has 21 passes. The separate existing suite has 193 passes, two failures
and three skips; both failures reproduce on the exact baseline. Native CPU
context/lifecycle tests, CPU scientific regressions, documentation and hooks pass.

Separate Nsight traces retain actual copies, synchronization/launch calls and
simultaneous allocation-address liveness. Pool backing, static allocations and
unobserved host/vendor/driver resources are not added as though independent
owned allocations. Missing semantic-work values remain explicitly unmeasured.
Compute Sanitizer memcheck/full leak checking reports zero errors/leaked bytes
for the RHF and PBE energy/force qualification workloads.

Raw source-matched measurements and original passing/failing logs remain in
ignored `build/issue933-evidence/` and node1's
`/data/jzzeng/qc-933-prepared-20261012/evidence/`. Reproduce separate baseline and
candidate processes with `benchmarks/prepared_lifecycle.py`; real-device tests,
benchmarks, profilers and sanitizers must run through finite `srun` allocations.
The durable decision and remaining public-force/AOT/resource boundaries are in
`.agents/notes/implemented/architecture/2026-10-12-native-prepared-endpoint-gate.md`.
