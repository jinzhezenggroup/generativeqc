# Bounded native iteration-reuse evidence

This is a CPU `solve_cpu` probe on deterministic synthetic dense `Problem` inputs.
It includes solver allocation, current/trial evaluation, DIIS, independent physical
convergence replay and result copying. It excludes SCF, MO/source construction,
JIT compilation and a complete molecular endpoint. It does not qualify CUDA.

## Retained matched comparison

`solve-diis6.jsonl` holds nine alternating default-off/opt-in pairs for each of
four shapes, using one `-O3` binary and the same ample memory ceiling. Both routes
had exactly equal final energies and amplitude values, five iterations, nine
iteration evaluations and one independent replay. Baseline executed 81 invariant
operations; reuse executed nine and saved 72. Both executed 1,521 dynamic
operations. These are operation counts, not FLOPs or expensive contractions:
the invariant operations are eight permutations/copies and one coefficient-1 add.

`provenance.json` records the base commit, dirty-source fingerprints checked
before/after compilation and execution, actual compiler commands, compiler/cache
versions, generated-header hashes, binary SHA and raw-output hashes. Cache stats
show verified ccache use. Large generated sources, objects and the binary were
retained in the ignored `.artifacts/iteration-reuse-final/` directory.

At (occupied,virtual)=(8,16), full-solve medians were 0.440682 s (default-off) and
0.429415 s (opt-in). Their median absolute deviations were 0.005006 s and
0.002190 s; ranges were 0.433266–0.469313 s and 0.425348–0.435642 s. The median
within-pair change was -2.24%. Other shapes have mixed/noisy small differences;
inspect all raw repeats rather than extrapolating a universal speedup.

## Attribution control

`fixed-work-control.jsonl` contains nine rotating triples at (8,16), each with
nine identical, changing-amplitude evaluations:

1. Original full schedule
2. Pinned schedule with preparation repeated every evaluation
3. The identical pinned buffer addresses with preparation only once

Input/reference construction, allocation and output checks are outside reported
time. Each evaluation's preparation, if selected, and generated evaluation are
inside. Every residual/trial array is byte-compared to the original evaluator;
energy values are exactly equal. This control excludes the solver, DIIS and
physical replay and is not an endpoint-performance substitute.

The medians were 0.294751 s, 0.294309 s and 0.280205 s respectively. Median paired
changes were -0.57% for pinned-reprepare versus original, -4.63% for pinned-reuse
versus original, and -4.77% for pinned-reuse versus pinned-reprepare. These results
do not demonstrate a pinned-layout penalty. Layout, traversal, function splitting
and compiler behavior are not individually identified; no cache-miss mechanism
was measured. The retained `fixed_work_scope` metadata was completed after the run to fix a
combined-mode metadata omission. Numeric compiled sources, the binary, raw times
and recorded hashes were unchanged; the driver now populates this scope for both
standalone and combined controls.

## Exploratory result and decision

`exploratory-diis0.jsonl` preserves the earlier preliminary nine-pair run that
motivated default-off admission. It used DIIS=0 and forced uncached fallback with
a smaller memory budget, before an explicit opt-in existed. It is not the final
matched comparison. At (8,16), separate medians suggested +5.4%, but median paired
change was only +2.14%, with pair changes ranging from -16.88% to +12.88%. The
slower sign did not reproduce in the matched DIIS=6 run. Its full provenance was
not captured prospectively, so it is retained as exploratory negative evidence,
not qualification.

The internal option remains default-off: correctness and actual one-solve reuse
are demonstrated, while representative molecular endpoint benefit and device
qualification remain open. No shape threshold is inferred from these samples.

## Reproduce

Run with a verified `sccache` or `ccache`; the latter was necessary here because
the environment blocked sccache's local server socket:

```sh
CCACHE_DIR=/path/to/existing/cache CCACHE_BASEDIR="$PWD" \
python tools/benchmark_rccsd_iteration_reuse.py \
  --launcher /path/to/ccache --output .artifacts/iteration-reuse-replay \
  --diis-size 6 --repeats 9 --with-fixed-work-control
```

The driver refuses source changes during generation, compilation or execution.
To run the attribution control alone, use `--fixed-work-control --shapes 8,16`.

## Final formatting provenance

After the measured run, clang-format 23.1.2 reformatted the native solver and
benchmark sources. `formatting-equivalence.json` retains before/after hashes;
non-whitespace bytes (apart from include ordering) and the include inventory are
identical. `restore-measured-format.patch` reversibly restores the exact measured
source bytes from the frozen pre-integration checkpoint using `git apply --unidiff-zero`. This is a formatting-only
integration check, not a new timing attribution to changed source hashes.

## Current-master integration boundary

These timings remain bound to the original implementation snapshot at
[3e60fcaa](https://github.com/njzjz-bot/vibeqc/tree/3e60fcaa66fb5b36f7b59b10bc0f80cf89852aeb),
whose tree is `f83246f99d67ebcaa1c680550d6a28e5d6a0069c`. Apply the formatting
restore patch at that checkpoint, not to later integrated solver sources.

The subsequent integration with master
`c3bb6df4468ad75d34a2e2effb6e2450f2838a5a` preserves #2039's accepted-trial
residual reuse and its two-vector no-DIIS capacity. Actual evaluator counts can
therefore be smaller than this frozen run's nine evaluations. The source proof
still identifies nine reference transforms, but per-solve saved work is computed
from actual evaluations. Integration tests are new correctness evidence; no
current-master timing or unchanged nine-evaluation trajectory is claimed. The
reproduction tool follows current solver behavior and reports actual counts.
