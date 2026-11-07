# Rejected frozen-VV10 history pilot

This #2976 follow-up qualifies the existing #3037 implementation, not a new
DIIS algorithm. **Do not promote this candidate:** the first complete cold pair
regresses on both cards, despite one fewer Fock. No reversed pair is run after
this negative screen, and no slow first call is discarded.

## Complete endpoints and actual work

Water96, WB97M-V, full Direct, spherical def2-SVP, Standard FP64, stock grids,
energy1e-9, density1e-5 and screening1e-12. Each separate process retains its
first constructor + prepare + SCF + analytic-force + host-export call, followed
by three frozen-own-density, same-owner warm calls. The only arm selector is
`GENERATIVEQC_KS_VV10_REBASE_HISTORY=0/1`; production oracle imports are blocked.

| Actual allocation | Cold off / on, s | Change | Warm medians off / on, s |
| --- | ---: | ---: | ---: |
| A10040GB, LSF227923 |82.605219 /88.449537 |+7.075% |13.785436 /13.805636 |
| RTX5090, Slurm16695 |165.173598 /184.785585 |+11.874% |12.532695 /12.534536 |

Both cards have the same measured cold work change:

- Focks16 →15: semilocal7 →7, frozen-VV104 →3, full targets5 →5.
- SCF VV10 pair sweeps4 →5; cache probes4 →4 and exact hits1 →0.
- Two successful history transports,14 entries, zero transport refusals.
- Optional retained lease141558528 bytes (about135MiB), charged through force
  and warm lifetime. Warm admissions refuse transport and record zero density
  writes/rebases; each warm endpoint still has one target Fock and zero SCF pair
  sweeps. All three per-call warm ratios stay below the unchanged1.03 gate.

Thus this is not a win hidden by iteration accounting: it removes cheap inner
work while adding an expensive fresh nonlocal sweep. The total time difference
is **not** an exclusive measurement of that sweep. Active compact points,
force pair sweeps and exclusive rebase/Gram time are not published by this
image; replay reports them unavailable rather than inventing equal pair work.

## Scientific results and refusal

The linked native residual/Gram/affine/refusal fixture passes all seven cases
on each card. RTX5090 memcheck reports zero errors; racecheck reports zero
hazards, errors and warnings. Both arms finish all four physical endpoints.

All32 comparisons against the pinned independent GPU4PySCF cold/warm E/F
receipt pass the original Standard gates:1e-5 Eh/atom,1e-4 max and3e-5 RMS
Eh/Bohr. Worst errors are9.005e-12,1.0465e-6 and4.7544e-7, respectively.
Historical reference latency is not used in any ranking.

The tighter paired-native gates remain1e-9 Eh/atom,1e-7 max and3e-8 RMS
Eh/Bohr. The trajectory change has worst differences1.7385e-12,5.8650e-7 and
3.1622e-7: **paired forces fail**. Candidate error against the independent
reference is actually smaller; paired refusal does not prove incorrect
residual transport. It prevents silently weakening the existing preservation
contract. Both scheduler payloads deliberately exit2 at this gate, after their
complete endpoint populations; they are not reported as successful jobs.

## Identity and replay

- Frozen #3037 head:`08f8abe1345e7f4f9a58858842ea63f2d8f7ad27`.
- Native source:`ef7f04bd77cae25a87aca1452c4640ac30ca95c7ff89817ae95e517fe24d9191`.
- ELF:`53b8e6ca2d11ba260d30857d28d1fcea04542756bd5e2e304a6f2ae09e4a2ee4`.
- Original A100 callback archive SHA256:
  `641390603ad3481d57196302f791f6d7bd385903e4abe1215077c3079b4cf91f`.

The frozen branch includes masterf27 and the then-pending #3036 parent; the
evidence publication is based on masterb40, which merges #3036. It is not a
new-main/default or #3038 HF-image timing. The normal Release build contains
SM80-real and SM120-real, uses verified private sccache0.16.0 (reported89.84%
cache hit rate), and restores the borrowed source/runtime. Its original ELF
hash and the owned HTTP transfer shutdown receipt are retained.

`evidence.json.xz` keeps the original per-file bytes and checksums for both raw
populations, identity/scheduler/device receipts, runners, controls, fixtures,
sanitizers, refusal logs and build/restoration/cleanup evidence. Offline replay
uses explicit checks that remain active under optimization:

```bash
python benchmarks/results/vv10-history-cold-20261107/replay.py
python -O benchmarks/results/vv10-history-cold-20261107/replay.py
```

See the durable rejection rationale in
`.agents/notes/rejected/2026-11-07-vv10-history-cold-promotion.md`.
