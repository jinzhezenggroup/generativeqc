# Current-source native Libxc acceptance

`campaign.json.xz` retains the independently measured, full-precision inputs for
11 registrations, 44 admitted CPU/CUDA spin/domain contexts and 88 paired energy
and force products on measured source `fbf57c111`. The following stronger gate
passed with `stale=False` at PR source head `2a8f137e3`:

```console
python tools/check_libxc_native_catalog.py --require-current
```

The [manifest](../../../manifests/libxc/native-qualification.json) authenticates
the campaign and preserves the preceding campaign as historical evidence. The
[generated report](../../../docs/reference/libxc_qualification.md) owns current
capability counts, exact contexts, missing measurements and blocker reasons.
Optional stronger `public-method` receipts remain zero; structural default-allow
selection is unchanged and does not depend on those receipts.

## Integration with later source

Master `c6967d7a8` includes the subsequent Direct J/K topology, strict HF
stationarity and exact-stack matrix-profile changes. Integrating it changes the checked native/compiler input
identities. The generated catalog therefore reports `stale=True`, zero
current-source observed contexts and zero admitted energy/force products. Its
66 exact stage-demotion acknowledgements compare against the qualified
`2a8f137e3` summary; no measured rows, source identity, binary identity or
numerical gates are rewritten.

The ordinary `python tools/check_libxc_native_catalog.py` check verifies archival
consistency on the integrated tree. `--require-current` intentionally fails until
a fresh source-bound native campaign qualifies that later tree. Historical CPU
and GPU successes below are not new measurements of this integration, and do not
establish broader domain or performance qualification.

## Measured identity and scope

- Source commit: `fbf57c111c7c885efbc0cd1457e6a7732c9e3d95`.
- Checked source inventory: `ba63873931ccffae7d9b2cea01831028f9bcf04a9903b818da76bd1539f64d64` (1,955 inputs).
- Embedded native source identity: `8b1b284f5198dda7bb167f8c3980db36dda5164462f3774a745a5d62553eab93`.
- Executed ELF: SHA-256 `c2d1b725fb754e72d35ca5829ba9a067974703c6a98bc4e4cefb44ea6abc256c`, 666,939,968 bytes.
- Direct/all-electron FP64, STO-3G, qualified maximum angular momentum 0,
  grid `[24,8,16]`; exact geometries, tiles, work counts and owners are retained.
- Both RKS/UKS layouts; cold, warm and changed-geometry execution; full analytic
  force vectors against independently reconverged native and oracle finite
  differences. Independent oracle: PySCF 2.14.0 / Libxc 7.0.0.
- Existing force gates: energy `2e-7` hartree, force `3e-5` hartree/bohr,
  translational residual `2e-8` hartree/bohr. Existing production-domain gates
  are replayed independently. No tolerance changes or speedup claim.

The registrations are LDA_C_VWN_4, LDA_C_CHACHIYO, LDA_C_PZ_MOD;
GGA_X_APBE, GGA_X_PBE_SOL, GGA_X_RPBE, GGA_X_B86, GGA_C_APBE;
MGGA_X_LTA, MGGA_X_GVT4 and MGGA_C_TPSS. They exercise the generic path,
not a new production allow-list.

## Execution and merge provenance

All real GPU work used finite Slurm allocations on RTX 5090 nodes, preserving
Slurm's assigned `CUDA_VISIBLE_DEVICES`. The campaign retains execution records:

| Selected records | Node | Slurm job | Python | NVIDIA driver |
| --- | --- | --- | --- | --- |
| Three LDA registrations | node1 | 16896 | 3.13.9 | 580.95.05 |
| GGA_X_APBE | node1 | 16898 | 3.13.9 | 580.95.05 |
| GGA_X_PBE_SOL, GGA_X_RPBE | node1 | 16899 | 3.13.9 | 580.95.05 |
| GGA_X_B86, GGA_C_APBE | node1 | 16900 | 3.13.9 | 580.95.05 |
| MGGA_X_LTA | node1 | 16895 | 3.13.9 | 580.95.05 |
| MGGA_X_GVT4 | node1 | 16902 | 3.13.9 | 580.95.05 |
| MGGA_C_TPSS | node1 | 16901 | 3.13.9 | 580.95.05 |

The original GGA and MGGA jobs requested their full families. After the first
registration's complete checkpoint was durably saved, each job was stopped to
distribute the remaining registrations across smaller, disjoint GPU jobs. Their
original requested-name lists remain in `executions`; those partial jobs are not
presented as completed full-family runs. Incomplete next-registration checkpoints
are excluded. The other five jobs complete their entire requested matrices.
Only disjoint complete functional records were merged.
Merge validation required identical schema, source commit, inventory/count,
build receipt and embedded-native verification; every selected functional has
both CPU spins and both CUDA spin receipts, and there are no retained failures.
No measured rows were recomputed or relabelled during merging.

`resource-acceptance.json` binds a separate real-GPU run (node1, Slurm job 16897)
to the same source and ELF: **4 passed, 0 skipped, 0 failures, 0 errors**.
It records minimum-workspace replay and one-byte-below-budget rejection for both
spins, ragged failed-neighbor recovery and closed-owner behavior, and
nonconvergence without published forces, warm seeds or force receipts. It also
retains the tested module's hash and hashes of ignored local JUnit/log receipts.
No skipped diagnostic attempt is counted as acceptance.

## Reproduction

Build the measured source using the
[native catalog procedure](../../../docs/developer/libxc_bulk_capabilities.md#native-endpoint-catalog)
and the exact `build.configuration` in the campaign. Verify the selected compiler
cache with `--version`, retain `--show-stats` before/after compilation and verify
actual compiler commands use it (this build used sccache 0.16.0, 607 commands).
Use Release, CUDA 12.9.86, `120-real`, shared automatic XC parameters, and set:

```text
GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=ON
GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=ON
GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=ON
GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF
GENERATIVEQC_CUDA_FAST_COMPILE=OFF
```

Select the 11 names with `GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS` and their
22 RKS/UKS `GENERATIVEQC_STATIONARY_AOT_PROFILES`, discoverable using
`tools/generate_native_stationary_cuda.py --list-automatic-aot-profiles` with
repeated `--functional NAME` arguments. Explicit native-stationary enablement
matters: disabling Python-consumer force AOT otherwise changes its initial
default. Source identity alone cannot establish provider presence.

With that installed library and independent oracle dependencies available:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=01:00:00 bash -lc '
    PYTHONPATH=python:tests/python:. python tools/qualify_libxc_native_catalog.py \
      --build-dir build/cuda-release --output .artifacts/libxc-current \
      --source-commit "$(git rev-parse HEAD)"
  '
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --time=00:10:00 bash -lc '
    GENERATIVEQC_AUTOMATIC_CUDA_FORCE_TEST=1 \
    GENERATIVEQC_LIBRARY="$PWD/build/cuda-release/libgenerativeqc.so" \
    PYTHONPATH=python:tests/python:. python -m pytest -q \
      tests/python/test_libxc_native_cuda_forces.py \
      -k "minimum_and_below_budget or ragged_failure_and_recovery or nonconvergence_has_no_force_receipt"
  '
```

Use repeated `--name NAME` to reproduce bounded family shards. Do not override
Slurm's device visibility. Build products, full logs and diagnostic attempts
remain ignored local artifacts; no release assets are published.

## Explicit exclusions

These receipts do not qualify physical empty-spin or pathological occupied-density
witness continuation, higher angular momentum, DF/ECP, mixed/Fast precision,
response/Hessian/HVP, nonlocal/hybrid/recipe domains or unmeasured registrations.
No runtime Libxc/Maple or CPU/PySCF derivative fallback is introduced. The producer
guards the CUDA endpoint against Python CPU derivatives and runtime AOT generation.
