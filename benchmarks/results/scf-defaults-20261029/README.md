# Native SCF defaults qualification

Scoped #2227 R4/R6 work for PR #2864. Final qualification is still running.

This intermediate evidence commit preserves the three unsuccessful CPU timing
campaigns and independent-reference diagnosis, including failed setup/reference
logs. It is not a passing qualification receipt. The original uncached, pinned
CPU and per-Calculator-cache matrices reached maximum candidate/baseline ratios
1.159752268299457, 1.174134781324136 and 1.1758822235658621, respectively.
No timing gate, model, requested output or numerical tolerance was relaxed.

`prequalification-receipts.json.xz` is a JSON map of receipt names to original
compressed-file hashes, decoded UTF-8 hashes, and lossless decoded text. The
three `*-samples.json.xz` entries contain their original decoded JSON records.
Diagnostics are retained with their original source identities and do not prove
a definitive garbage-collector timing mechanism.

The failed GPU M06-2X methane oracle was reproduced on the frozen baseline. GPU
Newton also failed; CPU PySCF Newton at the same model, stock grid and unchanged
tight thresholds succeeded. This is independent test-oracle recovery only.
Production CUDA execution remains mandatory and has no such CPU fallback.

`history.json` provides exact Git commit/path/SHA-256 recovery for earlier
payloads removed from the current tree to meet the unchanged aggregate budget.
Recover each with `git show COMMIT:PATH` and verify its checksum.

`intermediate-gpu-receipts.json.xz` retains job 14840 on source `4ac102f67`,
which passed before the shared binding extraction. It is not final `e8de3e4d9`
qualification. Its 48 tests and 1,792 calls passed with maximum ratio
1.0096753249835795 and identical scientific work.
