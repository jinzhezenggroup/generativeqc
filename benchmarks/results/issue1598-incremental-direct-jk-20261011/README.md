# Incremental Direct-J/K default-promotion negative closeout

This historical capsule supports keeping the exact incremental Direct-J/K route
(#990/#1579) **opt-in** and classifying `scf.incremental-direct-jk` as
`negative-evidence`. It is not a default-promotion qualification or an accepted
scientific validation envelope. Its storage inventory is `capsule.json`, kept
separate from standardized scientific publications. No runtime/default routing
changes are made.

## Retained populations and identities

Successful RHF records were measured with `compare_gpu4pyscf_batch` and an
independent GPU4PySCF reference, batch 1, def2-SVP spherical nested waters. Each
has a cold endpoint, three fixed-density warm energy+force samples, convergence
and incremental diagnostics. The reported campaign protocol was fresh-process
A-B-B-A baseline/incremental ordering with one retry per failed point; the
per-process comparator also interleaves its two engines.

| Campaign | RHF AOs | Recorded harness checkout | Native source identity | Library SHA256 |
| --- | --- | --- | --- | --- |
| old | 96/192/384/768 | `96570fd06243de418e2eec4efb570d5510506d03`, dirty | `54981b932caa9b88a4bc334634fed21ffc92ee9659b9128552a7462c806530f3` | `2998014a18547b813149521c51193cd8913602b5d7e4551e326a4d5c91053a77` |
| new | 192/384/768 | `d83aa86585a2f72ed852872c533723648b2799c3`, clean | `09847dbbf578bf414e45dfe02d996573662119f86c98cd21287db79bffb496eb` | `30734a0fc741da60389fb817b7429acd4a5164c840728308822fe215eae90a63` |

The native source identity is the probe's source-content fingerprint, not a Git
commit. Earlier campaign descriptions named master lineage `637466c…` and
`143df078c…` (base `98263e06c…`); those are not the measured harness revisions.
The old dirty checkout's patch was not retained, so this capsule cannot certify
exact source reconstruction or a clean-source promotion.

Successful RHF records identify the same NVIDIA GeForce RTX 4090 device UUID
`0639f74dc42b3c2b4a573795decc0888`; the old device receipt reports driver
570.124.06. Records also retain CUDA 12.9/gcc 13.3 toolchain information and the
portable_cuda native profile. The allocation was `issue-1598-dijk-20261011`.

## Measured results and limits

| Population | Baseline cold / warm | Incremental cold / warm | Outcome |
| --- | --- | --- | --- |
| 96 AO, old round 1 | 2.363 / 0.230 s | 2.288 / 0.384 s | cold 3.1% faster; warm 67% slower |
| 96 AO, old round 2 | 2.359 / 0.230 s | 2.223 / 0.383 s | cold 5.8% faster; warm 67% slower |
| 384 AO, old round 1 | 70.994 / 7.093 s | 100.506 / 13.770 s | cold 42%, warm 94% slower |
| 384 AO, new round 1 | 71.076 / 6.835 s | 100.190 / 13.862 s | cold 41%, warm 103% slower |

Both 384-AO rounds regress in both campaigns. Both 192-AO incremental points
have failed final outcomes with attempt=2/retried=true in each campaign, while
baseline points succeed. At 768 AOs, old incremental round 1 succeeds (30 cold
iterations, warm 51.729 s versus baseline 26.537 s), old round 2 has a failed
retry disposition, and both new incremental points have failed retry
dispositions. Failed RHF attempts retain no exception logs or comparator JSON;
their failure cause and individual first-attempt result cannot be re-audited.
They must not be relabelled as newly verified convergence failures.

All retained successful RHF warm samples pass recomputed independent-reference
and matched baseline/incremental round gates: |ΔE| ≤ 1e-8 Eh and |ΔF| ≤ 1e-7
Eh/Bohr. Cold native energies/forces are absent from these records, so the
offline verifier does not claim to recompute cold numerical comparisons.

The narrower old 96-AO cold gain is real in these records; there is no observed
warm win and no demonstrated guard domain winning across cold and warm
endpoints. That is sufficient negative evidence against broad default promotion.
It is not a claim that every measured endpoint is slower.

The attempted KS/PBE0 records are retained for context but are not accepted
independent-reference evidence. UHF-cation and moved-geometry populations were
not completed. Those gaps do not prevent this negative decision, but all remain
requirements for a future promotion. Per-point exception logs, progress journals,
resource time-series, reducer summaries and the old dirty-source patch are not
in this capsule. Only the old device receipt and the individual records'
after-measurement device snapshots are retained.

## Offline verification and future qualification

```bash
python benchmarks/results/issue1598-incremental-direct-jk-20261011/verify.py
```

The verifier checks the complete retained file inventory and hashes, exact
recorded campaign/native/device identities, selected/attempt agreement, final
retry dispositions, finite values, all three independent-reference warm pairs
and both available matched rounds, and the stated timing directions. It does
not re-run CUDA, reconstruct missing logs, or certify the unaccepted KS control.

The current runner and reducer support a new, broader qualification campaign;
they are not a reconstruction of the missing historical inputs:

```bash
bash benchmarks/run_issue1598_incremental_qualification.sh
python tools/reduce_issue1598_incremental_qualification.py \
  --input .artifacts/issue1598-incremental-direct-jk --output summary.json
```

## References

- Decision note: `.agents/notes/implemented/performance/2026-10-11-incremental-direct-jk-negative-closeout.md`
- Inventory: `manifests/maintenance/default_promotion_inventory.json`, `scf.incremental-direct-jk`
- #1598 (promotion tracker), #990 (implementation), #1579 (merged CUDA route)
