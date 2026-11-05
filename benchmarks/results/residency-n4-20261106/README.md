# Scheduled n4 CUDA residency observations

These are source-bound observations and deliberately synthetic controls, not a
full residency PASS, endpoint speedup, or closeout of issues #1626/#1629.

## Source and allocation

- Source commit: `1affb86a7a34647a4491154555493b53d301e1a1`.
- Frozen working tree: `4f2c2b475e905415baa1f7b9deada5d47eb01f20`.
- `source-manifest.json.xz` contains the independently frozen path, mode, Git
  blob and SHA-256 inventory. `source.patch.xz` reconstructs that tree from the
  source commit. The later CPU receipt-replay tests are not part of that tree.
- SSH host `n4`, Slurm node `node4`, partition `main`, `gpu:5090:1`;
  Slurm-assigned `CUDA_VISIBLE_DEVICES=0` was preserved. No GPU work ran on `n3`.
- NVIDIA GeForce RTX 5090, UUID
  `GPU-5430ba2e-194a-b6d1-f12c-6b99f6d34efd`, driver `580.178.04`;
  CUDA `12.9.86`, CUPTI API `28`.

The machine's default CUDA is older; qualification explicitly uses
`/group/software/cuda-12.9.1`. `control-provenance.json` retains the device
probe, successful control logs, artifact hashes, and the failed first
calibration's postprocessing error. That error was corrected before job 1504;
it was not a CUDA/CUPTI collection failure.

## Real-GPU controls

`calibration.json.xz` retains job **1504**'s raw activity journal, region
declarations and derived diagnostics. Recompute from the journal, not the
saved summary. The preallocated, numerically checked 64-byte workload executes:

| Role | H2D | D2H | D2D | Stream/device fences | Event waits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preparation | 64 | 0 | 0 | 1 | 0 |
| Deliberate host-transform iteration | 64 | 64 | 0 | 2 | 0 |
| Tile | 0 | 0 | 64 | 0 | 1 |
| Final publication | 0 | 64 | 0 | 1 | 0 |
| Observer completion | 0 | 0 | 0 | 1 | 0 |

This is an explicitly constructed round trip, not inferred scientific
producer/transform/consumer lineage. The existing auditor correctly returns
`INCOMPLETE` and no zero-round-trip assertion.

`source-controls.json.xz` retains job **1505**'s four raw source/activity/graph
journals, regions, independently declared policies and derived diagnostics.
A small helper DSO calls the actual production `residency_cuda.cuh` wrappers
and exports their observer ABI; it is **not** the production scientific DSO.
The literal publication allowance is one 64-byte D2H and one stream fence:

- Publication: `WITHIN_OBSERVED_RATCHET`.
- The same work labeled iteration or tile: `VIOLATION` of zero allowances.
- A second publication fence: `VIOLATION` of the one-fence allowance.

All source/API/activity joins are intact. Even the within-budget case remains
residency `INCOMPLETE`; wrapper controls cannot prove production payload links,
unobserved host fallbacks, graph coverage, or implicit-wait completeness.

## Production-source observations

Job **1506** uses the source-matched native DSO, not the helper DSO, under a
15-minute Slurm limit. All four existing optional
`test_cupti_df_compatibility_cuda.py` cases pass against committed independent,
same-Hamiltonian MP2 energy/amplitude fixtures. Exact observed compatibility
traffic agrees with the production source's counters:

| Fixture | Host-staged tiles | D2H bytes | Source event fences |
| --- | ---: | ---: | ---: |
| H2 | 16 | 128 | 32 |
| LiH | 500 | 6,912 | 1,000 |
| Water | 1,296 | 16,464 | 2,592 |
| F-HeH | 1,296 | 34,992 | 2,592 |

These actual production adapters generate CUDA DF values and stage them to a
host MP2 consumer. Their `subsequent_h2d_bytes` and dependency identities remain
unknown. They therefore **do not** qualify an actual scientific GPU producer ->
host transform -> GPU consumer chain. The deliberate calibration demonstrates
that byte pattern separately without inventing scientific lineage.

The same job captures complete prepared RHF and UHF histories over
`[H2, water, H2]`: first/warm energy, first/warm E+F, and moved-geometry E+F.
Ordinary CUDA histories are clean controls; the second reference is the native
CPU FP64 direct backend, not an independent PySCF implementation.

- UHF passes the original `1e-10` energy / `1e-9` force gates. Maximum independent
  errors are `9.9476e-14` energy and `1.9781e-10` force. Collection exits **2**,
  meaning intact observations with residency **INCOMPLETE**, not PASS.
- RHF collection exits **1** on the independent force gate. Job **1507**, under
  a five-minute limit, repeats the same pinned workload with a post-execution
  diagnostic launcher that saves public results before calling the unchanged
  validator. Moved-water force error is `5.604596453601118e-9` against the
  unchanged `1e-9` gate. Captured versus ordinary CUDA forces agree within
  `2.8866e-15`; the rejected comparison is against the CPU reference. The
  moved-water histories use six CUDA versus eight CPU iterations. This records
  the discrepancy without claiming its underlying numerical cause is repaired.
- Both initial captures and the RHF diagnostic have intact raw source/activity/
  graph-inventory records. That does not establish scientific payload links,
  implicit waits, or complete graph/device execution. No matching independently
  reviewed work-ratchet profile is configured for this prepared workload; the
  different historical profile is not relabeled or regenerated from these runs.

`production-captures.tar.xz` retains all raw production journals, independent
contracts, DF numerical arrays, UHF numerical evidence, RHF failure logs and
the diagnostic's full result comparisons. `qualification.json` reports their
recomputed joins and scoped numerical outcomes. The initial RHF capture emits
no `numerical.json` because its unchanged validator rejects before writing it;
that absence is not treated as acceptance. No threshold is relaxed.

`build-evidence.tar.xz` retains the CMake cache, actual compiler commands,
compiler/cache versions, before/after cache statistics and build logs. The
Release SM 120 build disables optional shell/stationary AOT and uses verified
sccache 0.16.0. Build-driver retries and the change from two to eight CUDA
compile slots are retained; final native linkage succeeds. Cache snapshots
span server lifetimes and are not a single-run cache-hit benchmark. Native
build inputs match the frozen inventory; `source-reconstruction.json` also
checks that the retained patch reconstructs its exact Git tree.

## Reproduction

`observer-controls-source.tar.xz` retains the exact calibration/helper scripts,
three collector translation units, production observer headers, endpoint launch
script and RHF diagnostic launcher. It contains source only, not executables.
Build the collector
as described in `docs/maintainer/residency_receipts.md`, using the verified
compiler cache. For the helper, compile an object through the cache before
linking it against the same CUDA runtime. Preserve the source layout below
`src/runtime/`, place scripts/collector under `artifacts/` and the native DSO
under `artifacts/native/`, and run each helper mode in a fresh Python process.
Use a fresh output workspace rather than overwriting independently pinned
contracts or captured records.

Run all CUDA work through a finite scheduled allocation, for example:

```bash
ssh n4 'cd <frozen-workspace> && srun --partition=main --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --nodelist=node4 --time=00:15:00 \
  bash artifacts/run-qualification.sh'
```

No GPU is needed to replay the retained controls in required CI:

```bash
PYTHONPATH=python:. python -m pytest -q \
  tests/python/test_cupti_residency_capture.py \
  tests/python/test_cupti_source_ratchet.py \
  tests/python/test_cupti_df_compatibility_replay.py
```

These tests recompute observed work from raw records and separately assert the
literal transfer/fence budget. They do not promote a saved diagnostic summary,
pin production source spelling, or authenticate provenance through offline
replay. The DF replay also compares retained numerical arrays against the
repository's independent fixture, not merely a saved numerical error summary.
