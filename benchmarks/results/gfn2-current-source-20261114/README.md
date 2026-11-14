# Restricted FRESH GFN2 public endpoint qualification

This campaign separates current-source CPU/CUDA qualification from the
[historical cutoff reconciliation](../gfn2-sharp-cutoff-20261113/README.md).
The production source is commit `95c3e547f96848e2fc3392d85fdfa1a68dff688b`.
The retained tracked-source manifests and patches identify the exact snapshots;
the patches change qualification, tests and documentation, not production science.
Actual shared-library hashes, build/resource identities, commands, exits and
archive hashes belong to `provenance.json`, not an inferred identity from HEAD.

## Results

All **120/120 observations** pass the unchanged independent gates: 36 CPU E+F,
36 CUDA E+F, 36 CUDA energy-only and 12 separately identified peer E+F calls.
Maximum energy error is `1.1800694e-10` Eh; maximum force error is
`8.1464612e-8` Eh/bohr. The native CUDA maxima are `1.7053026e-12` Eh and
`2.1256579e-8` Eh/bohr. This is current-source acceptance, separate from the
208 passing historical/frozen-replay observations.

The real-device campaign is Slurm **4463**, node2, `main`, `gpu:pro6000:1`,
finite 20-minute limit, unchanged scheduler visibility, exit zero. Device:
RTX PRO 6000 Blackwell, UUID `GPU-54595246-dbdc-a633-dc38-7bd8eea3831a`,
driver `595.99.02`. CPU contract tests pass **218** with 15 skipped;
the device-enabled suite passes **103** with one skipped. CPU live pinned
tblite control regeneration passes. Documentation checks pass 17 tests and
58 subtests. Separate real-device Compute Sanitizer reports **zero errors and
zero leaked bytes/allocations** on the public lifetime/recovery test.
Slurm accounting storage is disabled; the allocation record, script outcome
and successful `srun` exit provide the retained job/resource receipt instead.

The 768-AO complete-call seconds are retained without a speedup claim:

| Endpoint | Cold | Five-repeat median | Moved | Five moved-repeat median | Reported iterations original/moved |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU E+F, history 64 | 35.204591 | 34.925851 | 46.577372 | 46.614805 | 108 / 145 |
| CUDA E+F, history 8 | 7.218638 | 7.038194 | 20.243522 | 20.239865 | 98 / 95 |
| CUDA energy, history 8 | 7.132541 | 6.958050 | 20.225379 | 20.226390 | 98 / 95 |
| Peer CUDA E+F | 7.955766 | 7.230871 | 7.088983 | 7.088248 | 98 / 96 |

In particular, current native moved calls are slower than the peer. These are
scientifically qualified observations, **not a performance win**. Reported
iteration conventions are backend-specific; neither stage costs nor hidden
kernel counts are inferred from them.

CPU E+F process peak RSS is 540,545,024 bytes; CUDA E+F process peak RSS is
1,258,291,200 bytes. The missing device/workspace/allocator counters and the
limited interpretation of device memory snapshots are explicit below.

## Scientific and execution contract

- Use the exact retained 48/192/768-AO water inputs. The moved coordinates are
  the original coordinates plus `0.001 * sin(flat_coordinate_index)`, **not**
  the newer mixer diagnostic's `0.015` first-coordinate displacement.
- Independent tblite 0.6.0 explicitly uses
  `xtb_config={"smooth_cutoff": 0.0}`, matching native sharp D4 cutoffs.
  Input, geometry, oracle-library and model identities are retained with the
  historical campaign. Peer agreement is never the scientific oracle.
- Gates remain absolute `5e-7` Eh energy and `5e-7` Eh/bohr maximum force error.
  Convergence, positive SCC iterations, correct backend, finite complete host
  outputs and geometry identity are separate requirements.
- Both backends use 300 K occupations, energy tolerance `1e-10`, charge
  tolerance `1e-8`, and maximum 300 iterations. CPU explicitly selects mixer
  history **64**; CUDA retains the original history **8**. No solver controls
  change after a failed call, and there is no CPU/oracle scientific fallback.
- Every case executes cold, five retained-workspace calls, moved, and five
  moved retained-workspace calls. Every invocation starts **fresh SCC**.
  This does not qualify true electronic WARM reuse or ragged execution.

The CPU original 768-AO history-8/300 request really failed to converge. A
bounded history-8/1200 diagnostic converged in 702 iterations; the history-32
moved diagnostic required 415 iterations. Explicit history 64 converges the
qualified original/moved points in 108/145 iterations with the original
300-iteration budget. These are different convergence trajectories, not a
second dispersion-model discrepancy. They do not justify an implicit solver
retry, tolerance relaxation, or a claim that default controls converge all
admitted molecules.

## Timing, work and memory interpretation

`replay.py` measures synchronized public host-returned calls, including
constructor time on the first call. Imports, manifest validation and memory
queries are outside timing. The initial CUDA memory query initializes the
primary context: **cold means a cold calculator/workspace, not a cold process
or CUDA context**. No profiler is attached to the timing series. Sanitizer
execution is a separate receipt and must not enter the timing denominator.

Rows retain every SCC iteration count; they do not infer inner kernel/Fock
counts or stage fractions. `physical_residual_rms` is unavailable. The public
energy-change and density-RMS fields are NaN, not measured zero residuals.

Host memory is Linux process peak RSS from `getrusage`, including Python,
libraries and providers; it is not exclusive workspace memory and does not
fall after cleanup. CUDA measurements are device-wide **used-memory snapshots**
from `cudaMemGetInfo` before work, after each call and after explicit cache
release. They are neither process-attributed nor transient peaks.
**Device peak, exact workspace bytes and allocation counts are unavailable**
and remain null. Do not reinterpret these missing counters as zero work or
zero allocation. Runtime ownership, failed-request recovery, detached result
lifetime and host/device descriptor ingress are checked independently by the
contract tests; a separate Compute Sanitizer memcheck/leak-check exercises
real-device reuse, failure and cache release.

The unchanged xTBloom binary is a separately identified peer. Its new 768-AO
series runs in the same Slurm job; the previously valid 48/192-AO peer timings
remain in the historical archive. No speedup or method-stage cost attribution
is claimed from either series.

## Reproduction

Verify retained observations without a GPU or optional scientific dependency:

```sh
python benchmarks/results/gfn2-sharp-cutoff-20261113/reconcile.py
python benchmarks/results/gfn2-current-source-20261114/reconcile.py
```

The current-source reducer verifies archived identities and recomputes every
independent error, original/moved repeat set, convergence/backend gate, complete
timing and charged SCC work count. `summary.json` is generated from that reducer.

To repeat execution, reconstruct the appropriate snapshot from the recorded source commit and
the archived patch, retain the new fixture and driver, and verify the source
manifest before using a library built from it. Use a Release build, verified
sccache CXX/CUDA launchers and normal CUDA optimization (`FAST_COMPILE=OFF`).
The CUDA campaign uses CUDA 12.9.86, SM120 and GNU 14.3; the local CPU build uses
GNU 11.4. Compact configuration receipts, cache statistics and actual launcher
commands are retained in Git. Full build logs remain ignored local artifacts;
their exact hashes and local location are recorded in `provenance.json`.
The CUDA build encountered a cache-server loss; its launcher retried one large,
unrelated HF translation unit. Only the identified owned orphan was stopped.
Cache statistics straddle the server restart and cannot establish a reliable
hit/miss delta. The shared cache was not cleared or disabled.

For CPU, decompress the original inputs/oracle and source manifest to new local
paths, then invoke the driver with a new output path:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=python \
GENERATIVEQC_LIBRARY="$PWD/build-cpu/libgenerativeqc.so" python \
  benchmarks/results/gfn2-current-source-20261114/replay.py \
  --inputs INPUTS --oracle ORACLE --source-root "$PWD" \
  --source-manifest SOURCE_MANIFEST --output NEW_RECEIPT \
  --device cpu --history 64 --max-iterations 300 \
  --cases water-8 water-32 water-128
```

`run-cuda.sh` retains the actual n2 environment and command sequence. It requires
a new receipt directory containing the verified `source` snapshot, source
manifest and matched oracle. Run it only through finite Slurm allocation,
preserving scheduler-assigned visibility:

```sh
GFN2_RECEIPT_ROOT=NEW_ROOT srun --partition=main --gres=gpu:pro6000:1 \
  --nodelist=node2 --nodes=1 --ntasks=1 --time=00:20:00 bash \
  benchmarks/results/gfn2-current-source-20261114/run-cuda.sh
```

Scripts refuse to overwrite existing public receipts and preserve completed
rows with a failed status if a later call fails. Follow the contextual
[admission contract](../../../docs/developer/xtb_native_ownership.md), rather
than interpreting this finite campaign as support for additional temperatures,
fractional occupations, unrestricted channels, attachments or new API modes.

## Storage review

This publication retains exact numerical samples, negative diagnostics and the
two frozen tracked-source inventories, rather than full build output. The
inventories total 1,259,176 compressed bytes and are reviewed explicitly in
`benchmarks/evidence-reviews/gfn2-current-source-20261114.json`; dropping them
would remove offline verification of the source identities in the native rows.
Small configuration, test, convergence-failure and completion receipts have
individual owner/reason/SHA-256 entries in `benchmarks/evidence-policy.json`.
These exceptions admit only the recorded bytes, not future routine logs.

The PR explicitly proposes increasing the aggregate checkout evidence budget
from 91,107,328 to 93,204,480 bytes (2 MiB) for these source inventories and both
independent numerical campaigns. This is a storage-policy review, not a
scientific acceptance change. The hard 1 MiB per-file cap and 2 MiB incoming
change budget remain unchanged. Full compilation logs stay under the ignored
`.artifacts/issue-560/unpublished-build-logs/` directory and are not dependencies
of either offline reducer. No release, external upload or tag is created.
