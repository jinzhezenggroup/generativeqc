# Exact Direct force recurrence qualification

`measurements.json.xz` retains lossless primary complete-call seconds and actual
SCF/Fock counts, secondary population summaries, immutable source bindings,
protocol/oracle identities, independent error maxima, and separate counted/clean
profiles. Cold, five warm calls, moved geometry and five moved-warm calls retain
their original 1800-second population limit. No convergence normalization.

Read with `PYTHONPATH=.:python python -m tools.generativeqc_validation.direct_relative_evidence`.
This checks retained observations and protocol identities; it **cannot replay
force errors offline** because force arrays are not retained here. Every original
completed call was independently rechecked against the pinned geometry oracle
before compaction. Original JSON, arrays and SQLite remain in ignored local
`.artifacts/direct-parity-20261017/`, including unsuccessful trials.

Reproduce from the measured source and authenticated core/stationary AOT images:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=01:00:00 \
  env GENERATIVEQC_LIBRARY=/absolute/path/libgenerativeqc.so \
  PYTHONPATH=.:python python -m benchmarks.direct_force_reuse \
  .artifacts/direct-replay --atoms 96 --candidate relative-reuse
```

Both arms use the same measured image with explicit selectors, not unmodified
master versus candidate. Exports were non-Git at compilation; later immutable
commits authenticate every production input. They are not relabelled clean Git
builds. All new selectors remain opt-in. These are complete-endpoint observations,
not formal performance/production promotion: peak allocation, full compile cost
and the generic envelope's interleaved solver-trace gate were not qualified.

Profile clocks are separate views: Python complete exclusive wall owners, NVTX
exclusive host owners, exclusive device-window owners, and summed device kernel
times. Never add these views. Counted split profiles are diagnostic; clean
baseline low+order6 remains combined. Null work counters remain unknown; all work
counts are entered scientific operations, not hardware FLOPs.
