# Integrated native Libxc qualification

This 2026-10-28/29 campaign connects independent native measurements to the
existing capability DAG for issue #1124. The generated
[qualification report](../../../docs/reference/libxc_qualification.md) and
[catalog manifest](../../../manifests/libxc/native-qualification.json) own the
current counts, exact contexts and revision diff. Structural public selection
remains default-allow; qualification counters do not act as a selector whitelist.

## Measured source and execution

- Measured source: `36085eec7d745ad0e25514468ddfa42aea2c9f86`, including the native Direct/all-electron
  admission correction from #2854 and AOT source reuse from #2836. Scientific native/compiler/frontend inputs
  match that commit; subsequent integration edits affect only offline checking,
  tests, reports and retained data.
- Source inventory: 1836 inputs, canonical SHA-256 `ec311fd6a9bc3a7105ab42ff9f7e154400119a91d79b3b6987e62c3e09cfc0b3`.
- Embedded native source identity, queried from the actual loaded ELF:
  `a47263f92b809a0d6f52165a9eb42541017b879b437a738d608be93cd548db6f`.
- Executed library: 609857912 bytes, SHA-256 `d16bf758445db7fc0b1b043847979e7a12ef851ee98254be2e9a7f47dace0efa`.
- Build: GCC 11.4.0, CUDA 12.9.86, SM120, Release; verified sccache 0.16.0 on
  all 605 compiler commands in the Ninja build graph. The bundle retains the
  toolchain hashes, exact configuration, and command-stream digest. Full source
  censuses, command streams, compiler statistics and build products remain in
  ignored `.artifacts/1124/` storage; no source census is duplicated in Git.
- Runtime: NVIDIA RTX PRO 6000 Blackwell Workstation Edition on `n2`, driver
  595.99.02, Python 3.13.9, NumPy 2.5.1, PySCF 2.14.0 and Libxc 7.0.0.
  Each shard records its actual GPU UUID, unmodified Slurm visibility and job ID.
  Jobs 3743/3745/3744/3746 measured the LDA/GGA/GGA/tau-MGGA shards, respectively.
  Every real GPU operation used a finite Slurm allocation; node `n3` was not used.

## Numerical scope and gates

The eleven registrations are the existing 3 LDA / 5 GGA / 3 tau-MGGA fixture
set. Every registration has both spin layouts on both native backends, with
cold, warm-replay and moved-geometry full force vectors. The resulting scope is
44 backend/spin contexts and 88 qualified energy/force context-products,
with 132 full force vectors and 1188 independent finite-difference directions.

| Backend | Molecule/spin | Basis | Grid | Tile | Hamiltonian/precision |
| --- | --- | --- | --- | --- | --- |
| CPU | neutral H2 RKS; neutral HeH UKS doublet | STO-3G, s | 24/8/16 | 256 | Direct, all-electron, FP64 |
| CUDA | asymmetric H4 RKS; H4+ UKS doublet | STO-3G, s | 24/8/16 | 37 | Direct, all-electron, FP64 |

Coordinates, grid partition iteration count, force-source work, SCF residuals,
actual callback owners, native and independent FD values, and complete endpoint
times are retained at full precision. CPU stencils retain both displaced native
and independent energies. CUDA retains independently reconverged SCF energy,
fixed-density energy/Fock/density/directional gates and native/independent FD
vectors. This is numerical qualification, not a comparative speedup claim.

No gate was relaxed: CPU energy 2e-7 Ha; CUDA SCF energy 3e-9 Ha,
fixed-density energy 3e-10 Ha, density 2e-7, Fock 3e-9 and directional
energy 5e-8; force 3e-5 Ha/bohr at a 1e-4 bohr central step; translation
sum 2e-8 Ha/bohr. Maximum observed independent force errors are
8.304925636082316e-09 Ha/bohr on CPU and 2.8480329206104216e-09 Ha/bohr on CUDA.

The current native point emitter also passes 209 numerical boundary cases
and 44 existing compiler controls with the original rtol=2e-6/atol=1e-8.
Each spin worker independently rejects nonfinite input. These are current
work-domain E/vxc measurements, not relabeled historical raw-candidate results.
Physical empty-spin particle birth remains guarded; response/Hessian/HVP,
Fast/mixed, DF/ECP and higher-l contexts are not inferred from this campaign.

## Native admission and installed executable

The final native library passes 47 real CUDA empty-spin/policy refusal tests,
4 capacity/failure/lifecycle recovery tests and one public capability
non-inflation test (Slurm job 3747). Seven focused compiled-host tests also
exercise the corrected native automatic Direct/all-electron admission helper.

The installed CLI passes 22 RKS/UKS full-force cases with Python and compiler
executables absent from PATH and PTX JIT disabled. It agrees with the current
native campaign at 3e-9 Ha and 3e-8 Ha/bohr, and rejects an omitted static provider.
[Installed CLI receipts](installed-cli.json.xz) retain its executable/library
hashes, authenticated loader choice, full numerical results and a fresh embedded
source query. Installation changes RPATH and therefore the library digest;
its independently verified embedded scientific source identity is unchanged.
These checks do not promote additional DAG endpoints or the optional
`public-method` evidence stage.

Reproduce with the retained
[installed driver](../shared-automatic-aot-20261027/drivers/installed-cli.py).
Its scratch working directory needs `install-narrow` pointing to the CMake
installation, `tests` pointing to the measured checkout's test directory, and
`evidence/forces/NAME-SPIN-True-False-False-h4.json` links to the corresponding
new `NAME-SPIN-True-True-False-h4.json` lifecycle receipts. The driver reads their
cold rows without altering the receipts. Check the installed library's source
identity separately with `tools/qualify_libxc_native_catalog.py
--verify-build-only --build-dir INSTALL/lib --source-commit REV --output SCRATCH`.
Keep all these executable checks in a finite Slurm job.

## Reproduction and retention

Check out the measured revision. With the recorded toolchain and prerequisites,
configure a cached Release build following the
[native build guide](../../../docs/developer/build.md), with these campaign options:

```bash
sccache --version
mkdir -p .artifacts/libxc-catalog-build
sccache --show-stats > .artifacts/libxc-catalog-build/cache-before.txt
cmake -S . -B build/integration -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_CUDA_COMPILER_LAUNCHER=sccache \
  -DGENERATIVEQC_ENABLE_CUDA=ON -DGENERATIVEQC_CUDA_COMPILE_ARCHITECTURES=120-real \
  -DGENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA=ON \
  -DGENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT=OFF \
  -DGENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT=ON \
  -DGENERATIVEQC_ENABLE_AOT_SHELLS=OFF -DGENERATIVEQC_BUILD_CLI=ON \
  -DGENERATIVEQC_BUILD_TESTS=OFF \
  -DGENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT=ON \
  -DGENERATIVEQC_AUTOMATIC_XC_CUDA_PARAMETER_POLICY=discover \
  '-DGENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS=LDA_C_VWN_4;LDA_C_CHACHIYO;LDA_C_PZ_MOD;GGA_X_APBE;GGA_X_PBE_SOL;GGA_X_RPBE;GGA_X_B86;GGA_C_APBE;MGGA_X_LTA;MGGA_X_GVT4;MGGA_C_TPSS'
cmake --build build/integration --target generativeqc generativeqc_cli --parallel 20
sccache --show-stats > .artifacts/libxc-catalog-build/cache-after.txt
srun --partition=main --gres=gpu:pro6000:1 --nodes=1 --ntasks=1 \
  --time=01:30:00 env PYTHONPATH=python:tests/python:. \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tools/qualify_libxc_native_catalog.py \
  --build-dir build/integration --output .artifacts/libxc-catalog \
  --source-commit "$(git rev-parse HEAD)"
```

The same runner supports disjoint repeated `--name` selections. Shards may be
combined only when their source, build and embedded-source identities are equal;
combine their functional and execution records without modifying scientific data.
The current snapshot was generated with `tools/check_libxc_native_catalog.py`;
normal CI runs its read-only check against live source inputs. Its previous
snapshot is the ordinary intrinsic-only catalog at upstream
`7a295c90cc5abd8dc50369e9ee16d98bc40926f1`, before this evidence integration.
An update must rotate the actual retained summary and acknowledge exact
regression tokens; an older baseline cannot hide a later demotion.

The bundle retains all numerical acceptance inputs consumed by the checker and
mutation tests. `benchmarks/evidence-reviews/libxc-native-catalog-20261028.json`
classifies its compact retention; that storage review is not scientific approval.
Earlier unmerged-source attempts and full logs/XML stay in ignored local storage.
The historical #2835 whole-source census uses verified published Git recovery;
its numerical receipts and accepted input count/digest remain in the checkout.
The #2827/#2835 manifests remain explicitly historical and grant no current
endpoint claims. No Release, release tag or external backup was created.
