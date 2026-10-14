# Decision: select exact resident VV10 SCF component reuse on large grids

Status: implementation awaiting default-route endpoint qualification
Date: 2026-10-14

Frozen replays still repeated the quadratic VV10 energy/potential pair sweep
after force-component reuse had removed the analogous force sweep. Explicit
SCF component reuse on source `9aabd59a3` completed all 36 calls at 3/12/48
atoms with the original independent energy/force gates. Warm medians were
0.428168 / 1.817467 / 18.808182 seconds, versus force-reuse-only medians
0.492156 / 3.010895 / 39.028821 seconds. Every warm target Fock observed an
exact ingredient hit and zero pair sweeps. Semilocal XC, J/K, AO assembly,
diagonalization, physical residuals and acceptance still ran each time.

Select the cache automatically at 65,536 grid points, retaining explicit `0`
and `1` controls for qualification. Its immutable owner identity, seven
bitwise ingredient panels, error-aware validity and pair-schedule identity
remain required. A miss performs the ordinary complete pair evaluation.
The extra scalar/key/counter lease is 40 bytes; copied work counters share
the existing scalar synchronization and charge their 24 bytes. If the prepared
VV10 bound or device allocation refuses the optional lease, disable reuse
and retry the ordinary arena. Host registry failures and other CUDA errors
still propagate.

The native and frozen-potential initialization experiments remain default-off;
their provisional operators are not part of these performance claims. The
opt-in source passed 268 device checks, six complete independent molecular
energy/force cases, and both 32-case memory/synchronization sanitizer runs.
Default selection must additionally retain new source-matched 12-call
populations, smaller-grid controls, original failed populations and source
inventories. This change alone does not establish 96-atom completion or close
#2426.

See `.agents/notes/proposed/2026-10-14-vv10-exact-scf-component-reuse.md`
for cache ownership and rejected shortcuts.
