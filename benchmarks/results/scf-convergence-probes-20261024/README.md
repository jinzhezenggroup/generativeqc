# Strict and relaxed SCF convergence diagnosis

Retain the master solver defaults. The most promising tested proposal reduces
PBE0 Fast target iterations but does not meet the prespecified 2% complete cold
endpoint improvement floor. Production solver code is unchanged.

The frozen base is `48918670f59a8e10c254c822565a17796aa42f54`. It includes the
normalized-DIIS affine repair from #2714; the separate #2716 source-screening
proposal is absent. All measurements use fresh processes on n4/node4's Slurm
RTX 5090, spherical def2-SVP water clusters, stock grids and full analytic forces.
Cold includes construction, preparation, SCF, forces and export. Imports,
CUDA-context setup and AOT compilation precede timing. One warm call reuses each
owner's frozen cold density; no oracle density enters production.

## Fresh confirmation

Three alternating master/candidate rounds follow the exploratory screen. The
candidate switches to FP64 at maximum AO residual `1e-2`, retaining mixed DIIS
history only for proposals with the existing six-strict-step retirement bound.
It resets the energy epoch and preserves every final physical/force gate.

| Atoms | Master target Focks | Candidate target Focks | Master cold median (s) | Candidate cold median (s) | Cold reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 | 10 | 8 | 2.062343 | 2.089179 | −1.30% |
| 48 | 9 | 8 | 8.365975 | 8.492550 | −1.51% |
| 96 | 9 | 8 | 22.186432 | 21.870702 | 1.42% |

All three samples have the displayed Fock counts. The master uses **8/7/7 mixed
plus two strict** Focks; the candidate uses **five mixed plus three strict**.
The extra strict application offsets much of the removed mixed work. All
confirmation cold calls have one target attempt and seven separately reported atomic-HF preparation
Focks; atomic Focks are not molecular target applications. No discarded target
attempt or post-SCF Fock is hidden in these confirmation rows.

Warm calls remain one strict Fock. Master/candidate warm medians are
0.362006/0.363200, 2.236984/2.249214, and 6.352561/6.361572 seconds. The replayer
prints every timing and median absolute deviation. These measurements qualify
a sampled solver diagnosis, not a general method/grid accuracy or engine ranking.

## Other executed controls

- Standard Direct defaults require 10/9 target Focks at 12/96 atoms; explicit
  `energy=1e-10, density=1e-8` requires 15/14. Default Standard resolves to
  `1e-9/1e-5`; Fast resolves to `1e-6/1e-4` in its admitted mixed route.
- Histories 6, 8 and 12 all require nine relaxed target Focks at 96 atoms;
  history 4 requires ten. At 12 atoms, history 12 improves the strict solve
  from 15 to 13 but leaves relaxed convergence at ten. Shortening history to
  four makes both sizes' relaxed solves one iteration worse. Direct/DF and
  Standard/Fast controls are retained.
- Using `X^T(FDS-SDF)X` only for the DIIS metric adds two FP64 products per
  proposal, leaving physical stopping unchanged. It does not reduce 96-atom
  Standard or Fast counts. Its small-system improvements do not establish
  a complete endpoint benefit.
- Earlier handoff at `1e-3` without retaining mixed history reduces the
  12-atom Fast solve from ten to nine but leaves 96 atoms at nine. More
  aggressive `1e-2` and `1e-1` controls, plus the `3e-3` midpoint, demonstrate
  that fewer mixed steps can increase strict work. Every outcome is retained.
- The `1e-3` **retained-history** 96-atom proposal fails the final force-state
  validator and invokes the existing full FP64 fallback. Cold executes
  **9 + 17 = 26 target Focks** in 56.653 s; its following warm endpoint executes
  **4 + 17 = 21** in 60.370 s. The public endpoint counter includes both solves;
  its reported iteration count describes the returned solve. The earlier
  attempt's atomic preparation census is unavailable and is not reconstructed.
  These are rejected proposals with safe returned endpoints, not speed wins.

## Scientific gates and replay

All **152 returned E/F endpoints in 76 populations**, including one smoke pair,
pass the unchanged independent tight GPU4PySCF stock-grid solver gates:
`2.5e-6 Eh/atom`, maximum force `2.5e-5 Eh/Bohr`, RMS force `7.5e-6 Eh/Bohr`.
The two internally rejected force evaluations above remain visible. This screen
does not qualify a production change on displaced geometries, other molecules,
functionals, unrestricted states or ECPs; no default is promoted.

```bash
python benchmarks/results/scf-convergence-probes-20261024/reproduce.py
python -O benchmarks/results/scf-convergence-probes-20261024/reproduce.py
```

The standard-library replay verifies the archive and exact file bytes, source
maps, all four core-image identities, driver/reference bindings, every returned
energy/full-force error, complete molecular Fock censuses, rejected attempts,
and the separate fresh-confirmation medians. Both commands produce the same
summary: **861 executed target Focks**, including rejected precision attempts.

`evidence.json.xz` contains 123 exact text files: complete observations and attempt outcomes,
independent references and auxiliary inputs, native/force-artifact hashes,
6,038 mapped master inputs, three reconstructible experimental patches and
their complete changed source, drivers/launchers, compiler-launch proof and cache
statistics. `manifest.json` authenticates the archive. Compiler caching used
verified sccache 0.16.0; shared-server statistics are not attributed exclusively
to this build. The cancelled first build layout is retained separately from the
completed reused-layout build in ignored storage. Full build and execution logs
remain there; the archive retains their reproducible settings and material
failures. Source synchronization used content checksums.

The full binaries/builds remain in the existing task artifacts under
`n1:/data/jzzeng/convergence-next-20261024/` and
`n4:/data/jzzeng/convergence-next-20261024/`; raw local copies are in the worktree's
ignored `.artifacts/convergence/`. No release, evidence-cap increase, sample
deletion or production policy change is involved.

The [decision note](../../../.agents/notes/rejected/2026-10-24-shorter-ks-convergence-proposals.md)
connects these observations to the historical strict-DIIS advantage.
