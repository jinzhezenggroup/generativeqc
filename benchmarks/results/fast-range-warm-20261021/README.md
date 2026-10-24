# Strict Fast range-source replay

WB97M-V Direct on n1 RTX 5090, native source `3cd1b5ee8`, core
`da2094b0b5edfc811a425999d1c7e7ad376628a03832a3c2195de9a8685de485`.
Cold includes setup and full energy/forces; warm is the median of three
frozen-density complete calls. No residency override is used in these samples.

| Atoms | Standard cold/warm (s) | Fast cold/warm (s) |
| --- | ---: | ---: |
| 12 | 12.038 / 0.882 | 7.345 / 0.872 |
| 48 | 107.981 / 6.895 | 87.212 / 6.672 |
| 96 | 637.585 / 44.755 | 273.163 / 39.498 |

Every sample passes the unchanged independent common energy/force gates.
Denser-grid stability and CPU/GPU spots satisfy quarter gates. The producer
revisions remain distinct; no old measurement or oracle is relabeled. Warm
executes one strict Fock, complete forces and no implicit independent audit.
The small 12-atom warm difference does not establish a robust speed advantage.
This source precedes the separate VV10 density-domain alignment; these are
Standard/Fast comparisons, not a combined-source GPU4PySCF headline.

357 host tests and 26 GPU tests (Slurm 11435) pass. The GPU suite includes
frozen/moved independent observables, real value reuse without warm allocation,
policy changes, capacity refusal and geometry replacement. Its exact test
overlay is retained with its checksum. An earlier expanded test run hit
25 minutes in the CPU water-12 VV10 oracle; the failed log and bounded stack
diagnostic are retained. The final test uses independent GPU4PySCF for water-12
and additionally checks water-3 against CPU PySCF without relaxing error gates.

`receipts.json.xz` contains the complete numerical receipts, denser references,
all-repeat errors, diagnostic profiles and selector probe, failed attempts,
launchers, build/cache records and actual compiler-command provenance. The
large command listing remains on n1 with its path and SHA256 recorded.
`validation.json` contains the readable gates and all-sample errors. See the
[decision and invariants](../../../.agents/notes/implemented/performance/2026-10-20-fast-strict-warm-revalidation.md#follow-up-diagnosis-strict-range-value-leases).
