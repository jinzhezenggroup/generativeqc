# WB97M-V automatic canonical pair-reuse qualification

Base: fetched master `15e052d800085c61373937ed9c81c06cc6ca3548`, 2026-10-10.
This record qualifies the automatic default, not the earlier opt-in binary in
[historical component-reuse qualification](../wb97mv-canonical-materialized-20261010/README.md).
The PR base is `742dff879aba78678a45760302f4f21a192d9180`: the intervening
master change adds only LR-domain acceptance notes/evidence, not scientific or
build sources. All nine compiled/tested patch files match the PR tree byte-for-byte.

Scalar library SHA256: `9d8ca1634e4f14e03590291273d27b7d1c5ec3ccd34e7aad03c11352e3f837bd`.
Automatic library SHA256: `483a6123090ddb6974552793ac0397fbc377f61f01e1ef886516432b2a6c05fd`.

## Behavior and protocol

Indexed canonical order-five component reuse is admitted automatically from
source capability and remaining provider budget. No user performance option,
environment opt-in, basis-name check or 96-atom special case is required. An
absent legacy HF materialization selector and an explicit zero both admit the
route. The older dense HF/dddd value selectors remain default-off.

The fixture is the README **water proxy**, not an OMol25-distribution sample:
complete spherical def2-TZVPD including diffuse/f shells, strict-FP64 exact
Direct RKS WB97M-V, grid 48 x 16 x 32/atom, three Becke iterations,
E/density/screen gates 1e-12/1e-10/1e-12, 100 maximum iterations,
VV10 density threshold 1e-8 and an explicit 4 GiB incremental force budget.

Comparison uses two clean builds of the same master: unpatched scalar and the
automatic candidate. Compilation uses verified ccache 4.5.1 with shared caches
preserved and checkout-root path normalization. Real-device execution uses
finite Slurm allocations on node1/RTX 5090 and preserves scheduler visibility.

## Acceptance and interpretation

### Complete matched cold endpoints

| Metric | Unpatched scalar | Automatic |
| --- | ---: | ---: |
| 3-atom complete E+F median, seconds (three repeats) | 12.749601 | 12.508062 |
| 3-atom iterations / Focks, each repeat | 15 / 15 | 15 / 15 |
| 12-atom complete E+F, seconds (one matched pair) | 582.201867 | 531.607555 |
| 12-atom prepare, seconds | 1.001422 | 0.930443 |
| 12-atom SCF/publication, seconds | 542.614734 | 491.596855 |
| 12-atom physical forces, seconds | 38.585711 | 39.080256 |
| 12-atom iterations / Focks | 21 / 21 | 21 / 21 |

The 3-atom median reduction is **1.89%**, with all six independent E/F gates
passing (maximum E error 8.356e-12 Eh, F error 1.848e-11 Eh/Bohr).
The 12-atom matched reduction is **8.69%**; this is one pair, not a repeated
median. Its saving is in SCF, not a cheaper force cache. Native/native E/F
differences are 1.137e-13 Eh / 5.234e-11 Eh/Bohr, not independent oracle errors.
Separate instrumented complete 3-atom execution records 90 materialized launches
with no legacy HF selector set and an independent physical E/F gate passing.
The trace is not a clean timing arm.

The [machine-readable evidence](evidence.json.gz) retains complete times, actual
iteration/Fock counts, AO/force work and traffic, source/binary identities,
raw-artifact checksums and unavailable larger gates as null.

### Routing, numerical and resource gates

Native gates verify independent CPU full/SR/LR matrices, RHF/UHF masks,
Cartesian/spherical projection, diffuse/signed/screened components, batches,
displaced geometry and actual canonical work. Budget and real-ledger allocation
denial retain the incumbent source and release partial optional storage.
A separate exact-budget fixture protects the incumbent SPD MD-J allocation
before admitting optional canonical K metadata.
All four memcheck/initcheck/synccheck/racecheck tools report zero errors/hazards.
The pair recurrence test passes orders 5--12, and native canonical gates
separately cover full/SR/LR positive sources. Focused Python acceptance reports
128 passed, one GPU-qualified skip, and 54 passed subtests; compiler structure
reports 510 modules with zero dependency errors. Pre-commit checks pass.

The generated/canonical census suite isolates its intended J owner with the
existing diagnostic `GENERATIVEQC_DISABLE_MD_J=1`; this is not a selector for
the new optimization. The materialized suite also explicitly restores MD-J to
test its production priority. Complete production endpoints use neither switch.
The unmodified master reproduces the same generated/canonical census assertion
failure when MD-J is left enabled; numerical output is not the failing gate.
This pre-existing observer issue is not fixed or hidden as an optimization
result. Its unmodified baseline log and isolated-owner gate output are retained.
Default launch tracing and persistent-CTA sanitizers are separate from clean
timing arms; the latter force one CTA to reuse shared storage across tasks.

Complete cold timing includes prepare, SCF, physical forces, synchronization
and host publication. Imports, context and Calculator construction precede the
clock; teardown and serialization follow it. Existing artifact/compiler caches
are reused rather than cleared. First-use samples are recorded separately;
fresh-owner 3-atom matched repeats alternate engine order on one allocation.
All samples retain SCF iteration/Fock counts and force work records.

Independent 3-atom E/F uses the unchanged GPU4PySCF reference and gates of
1e-8 Eh / 1e-7 Eh/Bohr. The unchanged stock independent 12-atom reference
did not converge in 100 iterations; native/native 12-atom comparison is a
consistency/performance check, not an independent oracle pass. There is still
no complete physical 96-atom E+F result. Its historical selected source timing
must not be presented as a 96-atom endpoint speedup.

## Retained reproduction

Frozen source inputs, original diagnostic `cold.py`, build/qualification recipes
and raw outputs are retained on node1 under
`/data/jzzeng/wb97m-canonical-default-20261010-15e052d80/` and locally under
`.artifacts/wb97m-canonical-default-20261010/`. The scripts preserve assigned
device visibility and finite scheduler limits. Cache statistics are shared-cache
snapshots, not per-task hit counters.

On node1, after configuring the CUDA build, the default source gate runs without
any materialization opt-in:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=00:10:00 \
  ctest --test-dir build/cuda-release-sm120 --output-on-failure \
  -R '^generativeqc_cuda_fock_materialized_tests$'
```

See the [admission decision](../../../.agents/notes/implemented/performance/2026-10-10-automatic-canonical-pair-reuse.md)
and [prepared-pair contract](../../../docs/developer/direct_pair_recurrence.md)
for current ownership, invariants and retained fallbacks.
