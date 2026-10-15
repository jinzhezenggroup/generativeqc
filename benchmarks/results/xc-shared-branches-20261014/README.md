# Historical equal-spin XC shared-branch campaign

**Archive index, not a current performance claim.** The complete original
2026-10-14 campaign, with all 237 returned calls, raw-work summaries,
negative outcomes, manifests, source hashes and test receipts, remains
in the [pinned ancestor commit](https://github.com/njzjz-bot/vibeqc/tree/2f5b3396cbcfea548c76ff59d0bab91ef96ab9a9/benchmarks/results/xc-shared-branches-20261014).

Source-matched historical comparison: baseline `0e6dfb453` versus candidate
`786c31a48` (before merged #2505 and before #2464's narrower admission).
At 96 atoms, M06-2X direct/DF complete warm E+F decreased
`20.193 -> 17.347 s` / `36.209 -> 29.699 s`. r2SCAN-3c
direct moved regressed `296.815 -> 403.783 s` with
`14 -> 23` Fock builds. The direct/96 moved-warm series timed out
in both arms. Do not quote the former gains as combined-master results.

To recover the original complete archive from this branch's Git history:

```sh
git restore --source=2f5b3396cbcfea548c76ff59d0bab91ef96ab9a9 --worktree -- benchmarks/results/xc-shared-branches-20261014
```

The original archive's `manifest.json` binds checksums for the original
files. This checkout keeps only the index because restoring the complete
archive here would exceed the repository's unchanged 64 MiB
aggregate `benchmarks/results/` budget. Semilocal admission is tracked
in #2507; the current PR enables only FP64 split-hybrid point CSE.
