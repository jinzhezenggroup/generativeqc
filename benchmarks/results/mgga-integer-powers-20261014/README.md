# Historical FP64 meta-GGA integer-power experiment

October 14, 2026: **global and force-only default promotion rejected** on the
common CSE source `786c31a48`. These are not master measurements and must not
double-credit #2464's unmerged CSE prerequisite. Superseding master qualification:
`../m06-rks-force-powers-20261014/`; historical negatives remain unchanged.

RTX 5090/n5, Release/FP64, matrix tile 16, ordinary README moving-grid/DF/basis
protocol, 1,800-second point limit. Each arm/case has cold + five warm + moved +
five moved-warm. Use full `complete_seconds`, including cold preparation; warm
entries are medians of all five samples.

| M06-2X phase | Baseline (s) | Global (s) | Force-only (s) |
| --- | ---: | ---: | ---: |
| DF48 cold | 37.869489 | 35.188102 | 37.431879 |
| DF48 warm | 7.124970 | 6.514494 | 6.670259 |
| DF48 moved | 28.664111 | 26.583964 | 28.288082 |
| DF48 moved-warm | 7.118959 | 6.519952 | 6.669358 |
| DF96 cold | 123.680956 | 114.099151 | 122.614808 |
| DF96 warm | 29.957500 | **31.703341** | 29.000015 |
| DF96 moved | 92.432596 | 85.607959 | 91.500335 |
| DF96 moved-warm | 29.948927 | 27.516511 | 28.989035 |
| Direct96 cold | 136.580794 | 133.262118 | 136.208958 |
| Direct96 warm | 17.251976 | 16.272472 | 16.772929 |
| Direct96 moved | 94.625997 | 88.309197 | **106.983009** |
| Direct96 moved-warm | 17.272345 | 16.271191 | 16.787956 |

Global DF96 warm executes **4 instead of 3** builds (5.8% slower); global
direct96 cold takes 19 instead of 18. Force-only direct96 moved executes
**14 instead of 12** (13.1% slower). Byte-identical SCF source does not establish
identical full-image behavior or deterministic work; the cause is not isolated.
No gate is relaxed and these samples are not removed. All **108 returned E/F
calls** pass E 1e-8 Eh/F 1e-7 Eh/Bohr; maxima 2.0464e-11/1.5727e-10.

M06 point-only timing: 1.131505 → 0.696010 ms at 65,536 RTX 5090 points.
Unpromoted r2SCAN extended chains save ~15.5% on PRO 6000; neither is an endpoint
or README hardware-comparison claim. Four common M06 partial failures remain
against the stricter 113-bit oracle, plus additional baseline-pair exceedances.
The dirty-source point CLI, glibc mismatch and missing-executable sanitizer
attempt remain failed. Existing formal method gates pass; stress gates do not.
Successful memchecks have zero errors. No #2486 scheduling experiment is retried.

## Evidence recovery

`manifest.json` binds the compact `summary.json.xz` ledger: all endpoint rows
except force vectors, with complete timing samples/ranges, E/error/gate values,
SCF/AO/grid/force work, source/library/model identities, point reports and strict
failure rows. Unknown inner J/K counters stay null. Full vectors/references,
intermediate SCF histories, component profiles, logs/build provenance and
reproduction scripts live in the byte-exact original capsule in existing Git
ancestor `3ca3fa0038fbea5350a8bb799cca26d678522678`, not an external backup.

Restore via the checksummed six-file snapshot and offline command documented in
`../m06-rks-force-powers-20261014/README.md`. No failed sample is discarded;
`tools/compact_m06_force_evidence.py` regenerates the ledger from that snapshot.
Original historical capsule: 236108 bytes, SHA256
`951a9b57e034e5408ee1c73fe10a8af73ea1b77f9441a717ab764be0f4402ab3`.

Durable rejection rationale:
`.agents/notes/rejected/2026-10-14-mgga-integer-power-defaults.md`.
Explicit FP64-only compiler experiments still require
`emit_split_hybrid_device_body(..., lower_integer_powers=True)`; float and
non-boolean opt-ins fail closed. Restored source patches are required to reproduce
these historical ablations, not the superseding qualified default.
