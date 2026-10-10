# Bounded native CUDA copy-roundtrip elimination

This bundle retains scoped numerical qualification and complete-endpoint timing
observations, not formal/global performance or force promotion. The frozen
candidate parent is `d4b44b06489d44a517d3dfaecc6b7269f5b89b7a`, the head of the
occupied-spectator-pair PR #2178 at admission. The baseline is its already
qualified **pair-enabled** library, not the pre-folding library and not current
master. Do not combine this observation with the earlier fold into a synthetic
matched or current-master speedup.

## Endpoint and observations

Four fresh-process native molecular energy-only DF CCSD(T) endpoints run in
ABBA order in finite Slurm job 2810 on node2's RTX PRO 6000 Blackwell Workstation
GPU. The workload is ethane230: 230 spherical AOs, 9 occupied / 221 virtual
orbitals, 488 auxiliary functions, all electrons active. Both selections use
FP64 pedantic matrix execution, Q8, DIIS8 with ordinary full DIIS storage, and a
64-GiB correlation budget. No production PySCF/reference oracle is invoked.

Process wall includes native molecular RHF, DF-source construction, CCSD
including pair admission/projection, independent expanded physical replay,
standard FP64 (T), startup, and owner/caller teardown. One thread is used per
numerical library. CUDA toolkit 12.9.1, sm_120 Release/portable CUDA AOT, assigned
device UUID/runtime/driver identities and preserved Slurm visibility are retained
in provenance. Other node allocations are not a formal exclusivity guarantee.

| Median | Pair-enabled baseline | Copy-elided candidate |
| --- | ---: | ---: |
| Complete process wall, s | 165.499194 | 160.892676 |
| Complete CCSD, s | 69.408765 | 64.694207 |
| Primary iteration timer, s | 59.527814 | 54.812352 |
| Expanded physical replay, s | 9.672964 | 9.673417 |
| Generated Q operations | 186,538 | 172,630 |
| GEMM calls | 59,498 | 59,498 |
| CCSD contraction summands | 27,891,350,238,256 | 27,891,350,238,256 |
| Solver numeric capacity, bytes | 4,717,423,790 | 4,717,423,790 |

Observed process wall decreases **2.78%**, and complete CCSD
**6.79%**. There are only two fresh observations per selection;
these are scoped endpoint observations, not statistical/global promotion.
`samples.json.gz` preserves every sample rather than just medians.

## Mechanism and exact work gates

The packed and Q-batched paired auxiliary actions each contain three adjacent
audited-producer / unit-copy / unit-copy triples. The final copy's original
arena slot is already the producer's slot. Both launches can be omitted without
new aliases, allocation changes, longer lifetimes, altered graph mathematics,
new GEMMs, or reordered Q accumulation. The intermediate and producer have no
other readers and are not exposed outputs. Unsupported graphs retain copies;
retained, reused, or external kernel schedules reject this opt-in.

Native singleton addition is `1.0 * source`, not TensorIR's zero-seeded
interpreter addition. This is not a generic algebraic singleton-add rewrite.
Only audited native FP64 scalar producers qualify; borrowed inputs and reductions
do not. Original generated kernels and pointer declarations remain available.

All four endpoints have 20 iterations, 38 evaluations, Q488 and 2,318 primary Q8
tiles. Six omitted launches per primary tile remove exactly **13,908 launches**.
Logical read+write traffic avoided is
`4 * 8 * 3 * 9^2 * 221^2 * 488 * 38 = 7,042,781,551,104` bytes, because each
triple eliminates two full copies. This is **not measured DRAM traffic**.
GEMM calls/summands, contraction summands, Q scope, replay, accumulation order,
projection/fallback observations, setup traffic, packing diagnostics, capacity,
and solver trajectory remain unchanged. Existing CUDA operation diagnostics
report emitted work rather than pretending the skipped copies execute.

## Qualification and limitations

- 26 host proof/guard tests pass; CUDA-only proof skips without explicit opt-in.
- Nine targeted CUDA solver tests and five complete six-cut action tests pass,
  including single occupied blocks, uneven Q tails, expanded replay and canaries.
- A separate generated CUDA probe compares original and elided schedules
  bit-for-bit for signed zero, normal/subnormal limits, infinities, quiet and
  signaling NaNs, longer copy chains, zero and preseeded sticky error states.
  Its canaries and first producer fault agree; compute-sanitizer memcheck reports
  zero errors. This is qualification-only code, not a production oracle.
- Eleven original generated artifacts and the pair CPU header remain
  byte-identical. SHA-256 receipts and per-program graph/arena/copy-plan identities
  are retained. No new diagnostic fields or public ABI changes are introduced.
- Every endpoint passes independent total-energy `1e-8`, (T)-energy `1e-10`,
  and physical replay `1e-10` gates. Total and (T) energies are bit-identical
  between all four selections. Per-iteration formal traces and whole-process
  physical allocation/peak memory are not measured.
- `profile-diagnostic.json` retains only attribution from a prior complete
  traced endpoint in job 2807. Its six copies total 4.703272156 traced seconds;
  this is diagnostic only and is never used as promotion timing. Nsight's default
  CUDA-event-trace warning is recorded. Raw report/SQLite and binaries stay in
  ignored scratch; no Release or external archival publication is created.

Master was observed at `ee0ca19d5cade2eddcbd05585af2739ebe3f7619` before these
measurements. Its intervening SCF-force/implicit-Hcore and cuTENSOR-discovery
changes do not change the scoped paired CC action; cuTENSOR is OFF here. This is
not a claim of latest-master recomposition qualification. No unrelated suite is
rerun merely because master advances. Forces, Lambda, response and generic
nonpaired/reused schedules are not performance-qualified by this bundle.

After the four observations, #2178 merged as
`10a86458edca7aab03de4742f56ca87d495a858e`. The follow-up branch is aligned
with that master after checking identical tensor/compiler-generator, CC native
and owner-support source bytes against the frozen parent. This does not change
the saved campaign identity or turn its receipts into a master endpoint rerun.

## Reproduction and frozen identity

`measured-source.patch.gz` applied to an archive of the exact parent reconstructs
the initial candidate-library overlay. The later stronger test files are in the
PR; they do not change that measured library. Parent/overlay archive hashes,
input/oracle hashes, executable/library hashes and the Slurm allocation identify
the original run. The paired baseline's source reconstruction and build settings
are retained in `../df-cc-occupied-spectator-pairs-20261010/`.

The gzip-compressed build, resumed-qualification, bitwise-qualification and
matched-measurement scripts preserve the site-specific paths and environment.
Regenerate the candidate source from the parent plus the patch, build with the
retained ccache-enabled recipe, and provide the frozen paired baseline described
above. Decompress the scripts before use. Run every GPU command through a finite
allocation, for example:

```bash
srun --partition=main --nodelist=node2 --gres=gpu:pro6000:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:20:00 \
  bash measure-gpu.sh
```

The original build receipt includes a comparison-roster failure after successful
linking: conventional RCCSD has no `generated_rccsd_cuda.cuh`. The corrected
retained build recipe omits that nonexistent file; qualification resumed against
the already built library without another rebuild. Preserve original failures
and archives, and do not replace these frozen receipts with a later rebuilt
binary or relabel them as master qualification.
