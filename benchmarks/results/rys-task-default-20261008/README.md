# Qualified default task-parallel K follow-up

Frozen qualification base: `4385f72751b829883407c01106186917c344317b`.
The separately qualified three-root extension after main-branch AO/XC promotions
is retained in the [current-base report](../rys-task-master-20261008/README.md).
These historical five-class measurements are not changed or pooled with it.
RTX 5090, n1/Slurm `node1`, `main`, finite `srun --gres=gpu:5090:1` throughout.
The actual unset default selects five `sm_120` classes: `psps,ppps,dsss,dpss,dsps`.
Other classes/profiles retain incumbent; `incumbent` explicitly rolls back.
Explicit `rys-task` still exposes the eight-class experimental capability.

The winning schedule has 128 threads, four independent 32-lane warp queues,
and one complete quartet per lane. Every warp owns its bra cursor and bounded
two-batch survivor scratch. No CTA barrier can strand an independently retiring
warp. Existing screening, geometry caches, global task head and Rys axis algebra
remain authoritative. Restricted raw K contracts locally before atomic writeback;
there is no atomic-free or quantified hardware-counter claim.

## Complete Cold E+F

Strict PBE0 RKS, offline spherical def2-SVP, 48x16x32 grid, screening 1e-12,
E/D tolerances 1e-12/1e-10, maximum 100 iterations. Fresh processes/calculators/
owners, no supplied density or CUDA priming. Construction, context/preparation,
first SCF, analytic force, host return and owner teardown are timed. Imports and
input decoding are excluded; persistent compilation/artifact caches are equally
reused. The worker actively removes an inherited K lowering selector; the class
filter is unset for all winning cohorts.

| Cohort | Pairs | Incumbent median | Default median | Less time | Job |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96 atoms, initial wide | 3 | 113.514541 s | 110.314194 s | 2.82% | 6672 |
| 96 atoms, independent confirmation | 5 | 123.137497 s | 114.822158 s | 6.75% | 6674 |
| 48 atoms | 3 | 51.677314 s | 49.242787 s | 4.71% | 6672 |

Means also improve: 113.556890/113.308764 s (only 0.22%, initial 96),
119.373880/113.906666 s (4.58%, confirmation), and 51.630873/49.234294 s (48).
Every converged observation remains in its cohort; allocations are not pooled.

Actual incumbent/default Fock counts:
- Initial 96: `[17,17,17]` / `[19,17,17]`.
- Confirmation 96: `[17,17,19,19,19]` / `[17,17,18,18,19]`.
- 48: `[19,19,19]` / `[19,19,19]`.

The confirmation median includes different Fock work, not a same-iteration
kernel ratio. All samples pass the independent cached GPU4PySCF oracle (1e-8 Ha
energy, 1e-7 Ha/Bohr force). References and exact protocol provenance are in the
adjacent `rys-task-cold-20261008` report. These execution-Cold timings are not
cache-empty JIT compilation timings.

## Retained negatives

The original 32-thread/five-class promotion fails both mean/median Cold gates:
job 6665 medians 113.719225/119.570659 s, Focks `[18,17,17]/[17,19,19]`;
job 6668 112.808201/118.552434 s, Focks `[17,18,17]/[19,22,19]`.
They are valid negative observations, not stale-build exclusions. The two
production generated bundles are byte-identical; the changed trajectories'
cause is not established. The earlier opt-in successes are not relabeled as
default measurements. Job 6665 stops before 48 atoms at the failed 96 gate.

The initial wide 96 mean margin is weak, which motivates the independent five
pairs rather than claiming reliability from three pairs alone. Splitting the
wide preference into only `psps/ppps` or only `dsss/dpss/dsps` also fails both
Cold gates (job 6673); all those accurate, slower observations are retained.

## Separate mechanism/numerical gates

Six fixed-density processes, three clean repeats each, same binary/device/inputs:
96-atom incumbent/old-component-Rys/default medians are
1.717099/4.378061/1.522452 s (11.34% less default K time). At 48 atoms they are
0.966058/1.595584/0.853538 s. Full admission vectors match all modes, repeats
and density scales: 81,907,624 quartets at 96, 23,480,495 at 48, matching zero
tails. This does not prove equal primitive/root work. Never add K diagnostics
to complete Cold times.

- 576 independent Libcint matrices pass: all eight classes, both spins, four
  consumers, signed/reversed/coincident variants, 1/33/129 tasks.
- Memcheck/racecheck/synccheck each pass 64 cases with zero findings.
- Native wide integrated public/prepared K passes 16 cases per allocation,
  both spins and density restoration, with maximum error 3.85e-12. Broader
  selector/freeze/rollback qualification retains 176 cases on the prior bundle.
- CPU/compiler: 381 passed, 18 skipped; 497 compiler and 223 shared-SCF modules,
  zero ownership errors; Ruff and diff checks pass.
- Retained incumbent/old-component-Rys/old-block source hashes remain pinned.
  The registry preference and three task-containing shards have deliberate
  correction records; portable and empty task artifacts stay unchanged.
- NCU counters remain unavailable (`ERR_NVGPUCTRPERM`); no global permissions
  are changed, and no active-lane/barrier-stall contribution is quantified.

## Receipts and reproduction

`summary.json` keeps each winning/rejected cohort and its identity/gates.
`observations.tar.gz` retains all measured endpoints, forces, full diagnostics,
matrix/sanitizer/fixed-K observations and build/cache provenance. `validation.tar.gz`
retains tests, drivers and the exact qualified changed-source capsule. Both
archive hashes are recorded in the summary. The qualified binary SHA256 is
`18a7805606ca47f95b8b324a10331bea9c4dd7670cd96bf5b16386207f8b3a4f`;
native source identity is
`b05f71428835c60b17bf813b19364b987eb1700e34b7974f587a7f816c67eeb3`.

Apply the capsule to the frozen base, build with verified ccache and the `sm_120`
profile, and include both stationary PBE0 force companions/manifests beside
the main library. Do not reproduce by copying only the main library. Under one
finite Slurm allocation, unset the class filter and alternate fresh workers:

```bash
python -m benchmarks.rys_task_cold --mode incumbent --atoms 96 \
  --reference reference-96.json --output control0.json
python -m benchmarks.rys_task_cold --mode default --atoms 96 \
  --reference reference-96.json --output candidate0.json
```

Summarize at least three processes per mode using `--samples`; acceptance
requires all numerical/cold/identity gates and both mean and median improvement.
Changes after this frozen qualification need separately labeled validation,
especially the subsequent main-branch AO/XC default promotions.
