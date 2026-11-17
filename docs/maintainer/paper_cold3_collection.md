# Independent-process cold HF/DFT collection

`benchmarks.readme_cold_repeats` is an opt-in Linux/POSIX acquisition controller.
It invokes `benchmarks.readme_defaults` without building a library, selecting an
AOT profile or changing scientific settings. Use an actual finite allocation and
retain its live scheduler/platform receipt; environment declarations are not
independent proof that a GPU was granted.

## Protocol and configuration

The64method/mode/size pairs use two engines. Separate128prewarm processes each
own one cold and one warm call. The384fresh formal processes each own one cold
and three same-density warm calls, giving1536complete E+F calls. Each round
balances first-engine order32/32; the middle round reverses traversal.
Cold center/range is the median/min/max of three process cold calls. Warm uses
the three within-process medians while retaining all nine raw calls. These are
descriptive ranges, not confidence intervals.

Create a JSON configuration with these fields:

| Field | Meaning |
|---|---|
| `schema` | `generativeqc.readme-cold-repeats-config.v1` |
| `source`, `source_revision` | Actual checkout and full measured commit |
| `inputs` | Existing explicit orbital/reference and auxiliary inputs |
| `interpreters` | `standard` and `gpu4pyscf` Python executables or wrappers |
| `environment` | Explicit common environment, including thread controls |
| `arm_environment` | Explicit per-engine library/runtime environment |
| `deadline_seconds` | Finite monotonic parent budget |
| `process_timeout_seconds` | Finite child budget, also bounded by remaining parent time |
| `qz_job_name` | Optional actual qz owner when not using Slurm/LSF |

```bash
python -m benchmarks.readme_cold_repeats \
  --config .artifacts/cold3-config.json --output .artifacts/cold3-run
```

Use a new ignored output directory. Retained-results destinations are rejected.
Relative source, input and output paths are resolved against the controller's
invocation directory before any child changes working directory. The effective
absolute paths are recorded in `resolved_paths` beside the original configuration.
Explicit relative interpreter/wrapper paths use that same base without
dereferencing venv interpreter symlinks; bare executable names still use the
explicit child `PATH`.
Inherited `GENERATIVEQC_*` variables are cleared before the explicitly recorded
common/per-engine environment is applied, preventing hidden benchmark overrides.
The first failed or interrupted process stops the campaign and preserves its
command, output and status; there is no automatic retry or sample replacement.
On interruption, cleanup waits for the owned POSIX process group, not just its
launcher: TERM is followed by a group grace period of at most15seconds, KILL for
remaining members, and a launcher reap bounded to5seconds. A launcher exiting early
does not leave TERM-ignoring members behind. Cleanup errors and the interrupted
command are retained in the failed campaign; collection never retries or
continues with surviving samples. Partial cohorts cannot produce medians.

Imports/context startup and separate prewarming remain outside endpoint timing.
Cold still includes setup, SCF, forces and host export. No endpoint work is moved
across the timer. Orbital/auxiliary inputs, quadrature, convergence, precision,
screening and correction policies remain those of the endpoint producer.

## Allocation metadata and historical compatibility

Slurm/LSF declarations retain their actual owner fields. A qz launcher binds
`qz_job_name` to `GQC_QZ_JOB_NAME` and `GQC_QZ_PROJECT`. Both existing worker forms
are explicit: UUID batch pods (`job-<uuid>-worker-<rank>`) and legacy notebook
prefixes (`<job-name>--...`). Unbound hosts and mixed qz/Slurm owners are rejected.
Qz never fabricates a Slurm ID and still requires a separate platform receipt.

The measured H100 and RTX recorder overlays differed only in that hostname
check; their three-process controller was identical. The common recorder retains
both valid forms, without changing scientific defaults or historical records.

The controller keeps `generativeqc.readme-cold-repeats.v1`. Aggregation rejects
missing, duplicated, nonfinite, unconverged or mixed source/device/image/producer
populations without filtering survivors. The original version1 cohorts retain
all centers/ranges. Their existing complete-data replay remains:

```bash
python -m benchmarks.paper_hfdft_replay --verify-only
```

That separate reader also checks original hashes and command/repeat ownership
and recomputes independent full-vector errors. Timing completeness does not
constitute overall numerical acceptance.

The initially published controller was byte-identical to both measured
controllers. The current version additionally fixes relative-path ownership and
interrupted process-group cleanup. These operational corrections do not change
endpoint timing boundaries, scientific defaults, the v1 record schema or archived
aggregates, and do not relabel the historical measured-source hashes.
