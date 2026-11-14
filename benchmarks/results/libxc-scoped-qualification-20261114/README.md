# Scoped Libxc endpoint acceptance

These immutable incremental campaigns qualify representative ordinary LDA,
GGA and tau-MGGA CPU/CUDA energy/force endpoints without relabeling the retained
11-registration historical campaign. They exercise the generic path, not a
new production allow-list. The [manifest](../../../manifests/libxc/native-qualification.json)
authenticates both campaigns and the separate resource/lifetime receipt; the
[generated report](../../../docs/reference/libxc_qualification.md) owns current
counts and outstanding context-specific revalidation requests.

## Source and executed binary

- Measured Git **base**: `fb8cbabd90fe77c07609987bfeea06162177ed21`.
  This is a working-tree qualification, not a claim that this base commit
  contains the uncommitted policy/build-manifest patch. The exact content
  inventories below bind the measured sources, including the three newly
  tracked native generator dependencies.
- Frontend/compiler source identity:
  `9c225b7f02856f3a472786fcbe4b135ada91ff634c3809e6669f02e2212002ff`
  (1,967 inputs). Both campaigns embed the authenticated per-file census.
- Canonical native inventory-map identity:
  `ed6235d392d8280ca9aca5a588792c93fdf24166790d3a6c62a07b7df573a3a1`
  (2,140 inputs). This canonical JSON-map hash is distinct from the embedded
  native line-stream digest, retained in `native_build_verification`.
- Executed ELF SHA-256:
  `de2c01a15769c7942338ec9158e0fc0a517f5f447ae23c84475b0a6f5289121e`
  (856,393,376 bytes). All point/force descriptor and executable owners and
  CPU loaded-library receipts bind this identical ELF.
- Node `node1`, NVIDIA GeForce RTX 5090,
  UUID `GPU-9af5c15a-a02d-fa3e-fe5c-ceb8a0a80c68`, driver 580.95.05;
  Python 3.13.9, NumPy 2.3.5, independent PySCF 2.14.0 / Libxc 7.0.0 oracle.
  Slurm assigned `CUDA_VISIBLE_DEVICES=5` for both measured batches and the
  successful safety run; it was never overridden.

## Incremental execution

| Retained file | Slurm job | Complete endpoint registrations |
| --- | --- | --- |
| `primary-campaign.json.xz` | 17386 | LDA_C_VWN_4 |
| `completion-campaign.json.xz` | 17388 | GGA_X_APBE, MGGA_X_GVT4 |

Build job 17385 had a finite 45-minute allocation and timed out during
compilation, before scientific execution. Job 17386 reused its compiler cache
and build objects in a finite 60-minute allocation, finished the Release build,
and completed LDA domain/CPU/CUDA acceptance. It timed out during the subsequent
GGA campaign; its final checkpoint contains GGA domain proof but no admitted
GGA endpoint. Its original requested-name list is not a completion claim.
No numerical failure is recorded in either retained campaign.

Job 17388 used a finite 30-minute allocation and completed in 13 minutes
9 seconds. It measured only the two missing registrations on the identical
verified ELF, then passed the combined paired-endpoint gate and safety checks.
The LDA endpoint campaign was not rerun. Incomplete checkpoint-local backend
gaps remain in campaign metadata, not as a veto on the later complete receipts.
The prerequisite-only safety job 17387 did not execute tests after its producer
timed out; it is not counted as scientific or safety acceptance.

The combined new evidence admits **12 backend/spin contexts and 24 E/F products**:
three registrations, CPU/CUDA and RKS/UKS, each with energy and analytic forces.
There are 18 measured cold/warm/moved lifecycle rows per backend. Every force
vector is compared componentwise with native and independently reconverged
oracle energy finite differences. Exact fixtures, spin states, grids, semantic
J-only/Fock work, precision, capacities, complete CUDA endpoint timings and
artifact identities remain in the raw receipts. No kernel-only speedup or
cross-source performance comparison is claimed.

| Maximum observed error | CPU | CUDA |
| --- | ---: | ---: |
| Energy / hartree | 3.813e-9 | 2.387e-14 |
| Force vs native finite differences / hartree/bohr | 4.870e-9 | 2.860e-9 |
| Force vs oracle finite differences / hartree/bohr | 8.299e-9 | 2.942e-9 |
| Translational force residual / hartree/bohr | 5.552e-16 | 1.721e-15 |
| Physical SCF residual | 2.363e-12 | 1.506e-13 |

Values in the table round upward; full precision is retained in the campaigns.
Existing gates are unchanged: CPU energy `2e-7`, CUDA energy `3e-9` hartree;
force `3e-5`, translation `2e-8` hartree/bohr; physical SCF residual below
`2e-9`, plus the existing domain, fixed-density, density, Fock and directional
gates. This does not qualify physical empty-spin continuation, higher-l/DF/ECP,
mixed precision, response, Hessians or arbitrary molecules. The CPU and CUDA
fixtures and tile sizes differ and remain separately scoped.

## Resource, lifetime and runtime boundary

`resource-acceptance.json` binds job 17388 and the same ELF to:

- **3 pytest passes, no skips/failures/errors**: minimum-workspace force replay
  and one-byte-below-budget rejection in both spin layouts, plus ragged failed
  neighbor recovery and closed-owner rejection.
- An explicit tau-MGGA nonconvergence case using **MGGA_X_GVT4**, the compiled
  representative, not the existing test's MGGA_X_LTA. With `max_iterations=1`,
  two executions publish neither forces, a warm seed nor a force receipt.
- Native dynamic dependencies without runtime Libxc or Maple. Host tests
  separately retain structural default-allow and explicit admission boundaries.
- Hashes of the tested module, local driver, ignored JUnit and pytest log.

The nonconvergence check uses the existing test module's `_calculator`, `ATOMS`
and `PROPERTIES`, substitutes only the functional name, and asserts the same
failure/receipt invariants. Full logs, drivers, JUnit, point-probe builds and the
ELF remain ignored in the local/remote qualification run, not in Git history.

## Build and reproduction

Use the [native catalog procedure](../../../docs/developer/libxc_bulk_capabilities.md#native-endpoint-catalog)
with the campaign's exact `build.configuration`, toolchain and source inventories.
This build uses Release, CUDA 12.9.86, `120-real`, stable AOT shards,
`GENERATIVEQC_CUDA_FAST_COMPILE=OFF`, native stationary CUDA and stationary CPU
forces enabled, Python-consumer stationary force AOT disabled, and the three
representative automatic CUDA registrations. Compiler launchers are explicit
sccache 0.16.0; 760 actual compiler commands are verified by the build receipt.
`compiler-cache-before.txt` precedes the initial build and
`compiler-cache-after.txt` follows the completed build. These are shared-server
counters, not isolated cache-hit or speedup measurements. No cache was cleared.

All real GPU work uses finite `srun --partition=main --gres=gpu:5090:1`
allocations on node1. Preserve scheduler device visibility. Supplement the
manifest with the measured batches and run:

```console
python tools/check_libxc_native_catalog.py --require-current
python -O tools/check_libxc_native_catalog.py --require-current
```

Unchanged historical dependencies retain their evidence; genuine older
SCF/J/K/precision differences remain explicit for the other registrations.
Those pending rows do not erase this representative end-to-end acceptance.
The optional stronger `public-method=0/222` count is not a statement that
222 registrations cannot run or lack all scientific validation.
