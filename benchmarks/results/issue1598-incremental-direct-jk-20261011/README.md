# Incremental Direct-J/K default-promotion negative closeout

This directory retains the measured evidence behind the #1598 DoD item 1
decision: the exact incremental Direct-J/K route (#990/#1579) stays **opt-in**
and the inventory entry `scf.incremental-direct-jk` is classified
`negative-evidence`. No default/auto routing is introduced.

## Measured populations

Two campaigns ran the audited `compare_gpu4pyscf_batch` comparator with an
independent GPU4PySCF reference engine on one NVIDIA GeForce RTX 4090
(UUID `GPU-0639f74d-c42b-3c2b-4a57-3795decc0888`, driver 570.124.06, CUDA 12.9
toolchain, gcc 13.3, AOT portable_cuda sm_89, sccache 0.16.0) inside the finite
inspire allocation `issue-1598-dijk-20261011`. Every point is a fresh process;
per size the order is A-B-B-A (baseline / incremental / incremental /
baseline), each with one cold execution plus three fixed-density warm
energy+force replays, with one retry per failed point.

- `old/` — primary campaign on master `637466c624c4c4c4dd4eb7c511a2c88675744325`,
  library SHA256
  `2998014a18547b813149521c51193cd8913602b5d7e4551e326a4d5c91053a77`.
  RHF nested-water 96/192/384/768 AOs (def2-SVP spherical, batch 1).
- `new/` — post-#2236/#2241 re-verification on rebased master
  `143df078c…` (base `98263e06c…`), library SHA256
  `30734a0fc741da60389fb817b7429acd4a5164c840728308822fe215eae90a63`.
  RHF nested-water 192/384/768 AOs plus the attempted KS/PBE0 control records.

Each record carries its comparator schema, gate verdict, cold and warm timings,
iteration branches, incremental diagnostics and the native-build identity; the
`.outcome` files carry per-point exit codes and retry dispositions.

## Key measured results

| Population | Baseline | Incremental | Outcome |
| --- | --- | --- | --- |
| 96 AO (old) | warm 0.23 s | warm 0.38 s | no delta opportunity |
| 192 AO (old) | converges | 4/4 attempts fail ("SCF did not converge") | convergence failure |
| 384 AO (old) | cold 71.0 / warm 7.09 s | cold 100.5 / warm 13.77 s | warm +94%, cold +42% |
| 768 AO (old) | converges | round 1 converges (30 it); round 2 fails 2/2 | instability |
| 192 AO (new) | converges | 4/4 attempts fail | failure reproduces |
| 384 AO (new) | cold 71.1 / warm 6.83 s | cold 100.2 / warm 13.86 s | warm +103%, cold +41% |
| 768 AO (new) | converges | 4/4 attempts fail | failure reproduces |

All comparable numerical gates pass (|ΔE| ≤ 1e-8 Eh, |ΔF| ≤ 1e-7 Eh/Bohr plus
the comparator's independent reference gates). The KS control measurements are
retained but their independent acceptance could not be re-established on the
new master because the retained composed-baseline oracle predates the merged
DFT force changes (#2246); see the decision note.

## Reproduction

The campaign runner and reducer are retained in-tree:

```bash
bash benchmarks/run_issue1598_incremental_qualification.sh   # full matrix
python tools/reduce_issue1598_incremental_qualification.py \
  --input .artifacts/issue1598-incremental-direct-jk --output summary.json
```

Offline verification of the retained gates:

```bash
python benchmarks/results/issue1598-incremental-direct-jk-20261011/verify.py
```

## References

- Decision note:
  `.agents/notes/implemented/performance/2026-10-11-incremental-direct-jk-negative-closeout.md`
- Inventory entry `scf.incremental-direct-jk` in
  `manifests/maintenance/default_promotion_inventory.json`
- #1598 (promotion tracker), #990 (implementation), #1579 (merged CUDA route)
