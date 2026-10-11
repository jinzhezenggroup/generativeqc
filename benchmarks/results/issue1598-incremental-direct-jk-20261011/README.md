# Incremental Direct-J/K historical negative closeout

This reduced historical capsule supports keeping incremental Direct-J/K
(#990/#1579) opt-in. It is not a default-promotion qualification or a standard
accepted scientific publication. Its storage inventory is `capsule.json`.
No runtime or default routing changes are made.

## What remains auditable

The retained successful population is one matched baseline/incremental round
of 48-atom nested-water RHF (384 AOs, def2-SVP spherical, batch 1). Each record
contains a cold endpoint and three fixed-density warm energy+force samples,
with independent GPU4PySCF warm reference samples.

| Complete endpoint | Baseline | Incremental |
| --- | --- | --- |
| Cold | 71.076 s | 100.190 s |
| Warm median | 6.835 s | 13.862 s |

This pair is about 41% slower cold and 103% slower warm with incremental.
All three retained warm reference comparisons and baseline/incremental pairs
pass |ΔE| ≤ 1e-8 Eh and |ΔF| ≤ 1e-7 Eh/Bohr. Cold native result arrays are
absent, so their numerical comparisons cannot be recomputed offline.

The measured checkout is `d83aa86585a2f72ed852872c533723648b2799c3`, clean;
the library SHA256 is
`30734a0fc741da60389fb817b7429acd4a5164c840728308822fe215eae90a63`.
The native source-content fingerprint is
`09847dbbf578bf414e45dfe02d996573662119f86c98cd21287db79bffb496eb`.
Both records identify NVIDIA GeForce RTX 4090 device UUID
`0639f74dc42b3c2b4a573795decc0888`. These measured identities are distinct from
the master-lineage revisions named in the original campaign report.

Seven other retained outcomes report unsuccessful final attempts with
`attempt=2` and `retried=true`: old 192-AO rounds 1/2 and 768-AO round 2;
new 192-AO rounds 1/2 and 768-AO rounds 1/2. Their exception logs and individual
attempt records are absent. These dispositions do not establish the failure
cause, baseline success, or independent results for each attempt.

Two old source/library receipts remain, but no old successful numerical
records. Other sizes/rounds, KS records, device receipts, resource time-series,
progress journals, reducer summaries and the old dirty-source patch are not
retained. UHF/moved populations were not completed. The original report also
noted a 96-AO cold gain; no claim of universally slower endpoints is made.
The reported broader A-B-B-A campaign cannot be reconstructed from this subset.

## Offline audit

```bash
python benchmarks/results/issue1598-incremental-direct-jk-20261011/verify.py
```

The verifier requires this exact retained inventory, stored hashes, measured
identities, final dispositions, finite values, all three warm numerical pairs,
and the stated timing directions. It does not reconstruct omitted evidence,
certify cold numerical parity, or prove SCF nonconvergence from an exit code.

The runner `benchmarks/run_issue1598_incremental_qualification.sh` and reducer
`tools/reduce_issue1598_incremental_qualification.py` support a future broader
campaign. Promotion still requires independent cold/warm/moved RHF/UHF gates,
matched endpoint wins and a validated workload/resource guard. See the decision
note `.agents/notes/implemented/performance/2026-10-11-incremental-direct-jk-negative-closeout.md`
and inventory entry `scf.incremental-direct-jk` for #1598.
