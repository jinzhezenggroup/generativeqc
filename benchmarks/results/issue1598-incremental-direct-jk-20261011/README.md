# Incremental Direct-J/K default-promotion negative closeout

Retains the measured evidence behind the #1598 DoD item 1 decision: the
incremental Direct-J/K route (#990/#1579) stays **opt-in** and the inventory
entry `scf.incremental-direct-jk` is `negative-evidence`. No default/auto
routing is introduced.

## Measured populations

Two campaigns ran the audited `compare_gpu4pyscf_batch` comparator with an
independent GPU4PySCF reference engine on one NVIDIA GeForce RTX 4090
(UUID `GPU-0639f74d-c42b-3c2b-4a57-3795decc0888`, driver 570.124.06, CUDA 12.9
toolchain, gcc 13.3, AOT portable_cuda sm_89, sccache 0.16.0) inside the finite
inspire allocation `issue-1598-dijk-20261011`. Every point is a fresh process;
per size the order is A-B-B-A (baseline / incremental / incremental /
baseline), each with one cold execution plus three fixed-density warm
energy+force replays, with one retry per failed point.

- `old/` — primary campaign on master `637466c62…`, library
  `2998014a…`. RHF nested-water 96/192/384/768 AOs.
- `new/` — post-#2236/#2241 re-verification on rebased master `143df078c…`
  (base `98263e06c…`), library `30734a0f…`. RHF 192/384/768 AOs plus the
  attempted KS control records.

Each record carries its comparator schema, gate verdict, timings, iteration
branches, incremental diagnostics and native-build identity; the `.outcome`
files carry per-point exit codes and retry dispositions.

## Key measured results

96 AO: no delta opportunity (warm 0.23 vs 0.38 s). 192 AO: incremental fails
4/4 attempts on both revisions; baseline converges. 384 AO: warm +94% (old)
/+103% (new), cold +42%/+41% slower; gates pass. 768 AO: run-to-run
instability on the old revision (round 1 converges in 30 iterations, round 2
fails 2/2) and 4/4 failures on the new revision.

All comparable numerical gates pass (|ΔE| ≤ 1e-8 Eh, |ΔF| ≤ 1e-7 Eh/Bohr).
The KS control could not be re-accepted on the new master because the retained
oracle predates #2246; see the decision note.

## Retained scope

The compact publication retains the failure outcomes (the measured negative
dispositions) plus the matched 384-AO round-1 pair on the new master;
successful points whose records were compacted away have their outcomes
removed too, so the retained set is self-consistent. The other sizes are cited
from the decision note. `verify.py` binds every retained file to
`publication.json` and reports the dispositions it cannot re-check.

## Reproduction

The campaign runner and reducer are retained in-tree
(`benchmarks/run_issue1598_incremental_qualification.sh`,
`tools/reduce_issue1598_incremental_qualification.py`); offline verification:

```bash
python benchmarks/results/issue1598-incremental-direct-jk-20261011/verify.py
```

## References

- Decision note:
  `.agents/notes/implemented/performance/2026-10-11-incremental-direct-jk-negative-closeout.md`
- Inventory entry `scf.incremental-direct-jk` in
  `manifests/maintenance/default_promotion_inventory.json`
- #1598 (promotion tracker), #990 (implementation), #1579 (merged CUDA route)
