# Supplied-MO CPU RCCSD closeout observations

Measured November 4, 2026, against scientific source
`9cfb77727661cc64e47938d9d0ba65a157427439`, extracted with `git archive`.
The identity-enriched rerun overlays only the benchmark harness; its exact SHA-256
is recorded separately. No production scientific equation, convergence setting,
numerical gate, default backend, or compiler-cache identity changes in this run.

This advances **#762 gate B**, not gate A or closure of the whole issue.

## Protocol and provenance

- Complete supplied-MO RCCSD solves, excluding HF, AO/MO generation and fixture
  preparation. Every observation uses a fresh Python process.
- Independent pinned H2O and NH3 endpoint fixtures, frozen-core zero, existing
  `1e-8 Eh` energy and `1e-10` final expanded-equation residual gates.
- NumPy `cpu`, native JIT-artifact miss, and cached native replay at requested
  **1 and 4** BLAS threads. The enriched rerun records the actual OpenBLAS pools.
- AMD EPYC 7K62 host `node3`; CPU-only execution is allowed on this host. No GPU
  test or probe runs on node3. Python 3.13.9, NumPy 2.5.1, GCC 11.4.0;
  verified compiler launcher `sccache 0.16.0`, using its existing cache.
- "Cold" means a new JIT artifact, **not** a cleared C++ compiler cache. Global
  shared-server statistics cannot attribute this JIT campaign's cache hits.
- RSS is the solver process's Linux high-water RSS, **not** the sum of compiler
  child processes, all physical allocations, or a validated resource envelope.

The lossless XZ transport decodes to every original byte of `evidence.json` at
commit `3c077bdbd6e11ff1c9d2d62a7da6752faa485e9c` (Git blob
`51b10eb45c82e2f168202413c4b86765a189bcd7`). No scientific value, numerical
token, observation, original timing or negative disposition is reserialized.
The offline replay uses the existing shared compressed-JSON reader.

`evidence.json.xz` retains the initial campaign and the identity-enriched rerun:
24 observations total, including reference and cold stages. The enriched
records include pinned input hashes and exact JIT binary SHA-256 values before
the temporary artifact cache is removed. Metadata collection is outside the
timed solve. Raw command logs remain under `/tmp/qc-762-*-20261104.*`.

## Observed endpoint results

The following are individual enriched-run observations, not medians or a
universal performance claim. Each fixture/thread pair has one warm sample.

| Fixture | BLAS threads | NumPy (s) | Native artifact miss (s) | Native cached (s) | Iterations |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2O | 1 | 0.981015 | 2.904084 | 1.545241 | 15 |
| H2O | 4 | 0.974004 | 2.885036 | 1.552725 | 15 |
| NH3 | 1 | 1.067180 | 2.924127 | 1.574236 | 17 |
| NH3 | 4 | 1.080632 | 2.915699 | 1.597866 | 17 |

Both backends have identical energy and iteration counts for each fixture.
Maximum absolute independent energy error is `5.755396159656812e-13 Eh`;
all final R1/R2 maxima pass the original `1e-10` gate. Recorded solver-process
RSS spans 59,892–61,056 KiB in the enriched run.

**Disposition: retain the explicitly selected native route; do not promote it
to the default.** The observed cached native solves are slower, and cold latency
does not justify changing the existing default. This preserves the independent
expanded equation as the convergence oracle. Revisit after a changed native
iteration/lowering demonstrates matched complete-endpoint wins, including
cold latency and the same independent numerical/resource checks; do not select
a default from isolated contraction timings.

## Replay and remaining acceptance

From the repository root:

```bash
PYTHONPATH=.:python python benchmarks/results/cpu-rccsd-native-closeout-20261104/replay.py
PYTHONPATH=.:python python -m benchmarks.rccsd_cpu_compiled_endpoint \
  --fixture h2o --threads 1 4 --output /tmp/rccsd-h2o.json
PYTHONPATH=.:python python -m benchmarks.rccsd_cpu_compiled_endpoint \
  --fixture nh3 --threads 1 4 --output /tmp/rccsd-nh3.json
```

The existing native-CPU solver tests pass: **10 passed, 19 deselected**, including
independent amplitudes/oracles, resource rejection before provider reads,
nonfinite failure, immutable configuration and no interpreter fallback.

The versioned current-source **CPU scientific-ownership inventory remains
outstanding under #934**. #762 was closed on 2026-11-04 as superseded tracking,
not as acceptance of that audit. This small measurement is not the audit, is not a
full HF+CCSD pipeline, and does not qualify CUDA, arbitrary larger bases,
compiler-child peak memory, or a repository-wide compiler-first cutover.
