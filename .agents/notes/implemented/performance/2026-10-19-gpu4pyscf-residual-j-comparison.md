# Evidence: fresh GPU4PySCF and masked-residual J comparison

Status: implemented qualification; production defaults unchanged
Date: 2026-10-19

## Scope and pinned inputs

The [subsequent compacted-source experiment](2026-10-19-compacted-masked-hermite-sources.md)
retains this comparison's evidence and records a newer native schedule,
fresh upstream timings/counters and independent cold SCF/force qualification.

The user requested a fresh GPU4PySCF measurement and source comparison after
the bounded masked-Hermite experiments failed to beat the incumbent. This note
records new measurements, not a restatement of historical PR timings, and does
not authorize changing Schwarz semantics to match another engine.

- Machine: n1, RTX 5090, UUID
  `GPU-4c5ba19e-745a-9f56-6bf8-34d118bcc818`, driver 580.95.05.
- Clean Slurm jobs: 10292 and 10294, each with one GPU, eight CPUs, a finite
  20-minute limit and unchanged assigned `CUDA_VISIBLE_DEVICES=7`.
- Installed GPU4PySCF: `gpu4pyscf-cuda12x` 1.8.1; PySCF 2.14.0; CuPy 13.6.0.
  No upgrade or claim about the latest upstream version is made.
- Native source base: PR #2600 commit
  `aacbccc275274cb70e36b76a646750fe667acf3e`, plus the retained default-off
  experiments. This pins provenance, not current PR merge status.
- Native library SHA256:
  `80f3f92c46391ee28aa532665ca6092fed73d2dffa453092f346302c3412dcb6`.
  Both native policies use this same binary. It preserves the compact E
  accessor and the pinned source's reciprocal/residual scheduling.
- Upstream source comparison: v1.8.1, commit
  `5b284c258a4260baef80e3d150b4e7a81a9dbd57`.
  Installed `j_engine.py` is byte-identical to that tag, SHA256
  `e24c81017dd37e18664113c71bbab88c9cb1b8fe7a12a6cad76aeb3fe2cb21d3`.
- Installed `libgvhf_md.so` SHA256:
  `96000ade4694214b13eb9aedb7de085c843145aedb303cbbf08d69a258c8fe5f`.
  Python-source equality does not prove exact CUDA build-source identity;
  kernel names, runtime layouts and binary checksums are separately retained.
- Same 96 atoms, 1088 spherical AOs and explicit canonical r2SCAN-3c basis.
  Basis SHA256:
  `fb2c5d3a5f1ec149f0bf1d15d1e76cdb8baf042aeaa8be677cf25f33b1b7882b`.
  Frozen density-array SHA256:
  `a0b22059e12adcd39771e5dcfaac0951afb298d8252ef6674889ad0349ed6090`.
  The input is the previously accepted physical SCF density, not random data.

The reference molecule is constructed from the actual explicit shells and
contraction phases, not merely a similarly named PySCF basis. The existing
`benchmarks._md_j_reference.reference_molecule` supplies the AO normalization
map. Apply it on both sides of the density and returned J. Otherwise basis
phase/normalization differences can invalidate both timing and accuracy claims.

## Oracle and timing protocol

Each clean job computes a new independent CPU PySCF/libcint direct J at
`direct_scf_tol=1e-15`. It uses `RHF.init_direct_scf` and `hf.get_jk` with
`with_j=True, with_k=False`, not an AO four-index ERI tensor. Reference work is
outside timed GPU samples and never a production consumer. It takes about
30 seconds and agrees with the retained GPU4PySCF reference within the stated
gate. Every measured output is checked against the new CPU result using
`atol=3e-9, rtol=3e-11`; maximum errors are also reported, not just pass/fail.

Prepare two native owners, with the same `1e-12` native screening, 1 GiB request
budget and original 128 MiB MD allowance. Disable the old masked and resident
preprojection experiments; only `GENERATIVEQC_MD_J_MASKED_TILES=0/1` differs.
Prepare four upstream optimizers at `1e-12`, the verified default `1e-13`,
`1e-14`, and `1e-16`. Upstream reports one visible device and no high-angular
CPU fallback. Each owner gets one untimed initial evaluation, followed by four
samples per job, interleaving all six methods and reversing order each round.
Thus each method has eight clean samples across two allocations.

The primary timing is synchronized host canonical density to host canonical J
with prepared owners. GPU4PySCF includes upload, AO scaling, internal density
transforms, screening, all J kernels, projection, download and output scaling.
Native includes the equivalent adapter call, plus its test-only input-density
readback/assertion and numerical-error check. Do not describe the latter as
production overhead. Secondary CUDA events are retained but have different
boundaries: native events enclose its internal stream work, while upstream
events enclose its Python-driven default-stream interval. Primary host wall
time avoids calling those different event scopes an identical kernel metric.

These are complete prepared-J consumers, not cold SCF energy/force endpoints.
Preparation times are retained separately and are not a controlled cold-start
comparison: earlier optimizers and imports warm process-global caches.

## Fresh complete-J results

| Method | Configured cutoff | Job 10292 median (s) | Job 10294 median (s) | Largest error vs fresh CPU J |
| --- | ---: | ---: | ---: | ---: |
| Native default | `1e-12` | 3.94515 | 3.94964 | `2.456e-11` |
| Native masked tiles + R lookup | `1e-12` | 4.29673 | 4.32172 | `2.456e-11` |
| GPU4PySCF | `1e-12` | 0.63051 | 0.63131 | `2.390e-10` |
| GPU4PySCF default | `1e-13` | 0.67377 | 0.67528 | `2.753e-11` |
| GPU4PySCF | `1e-14` | 0.72175 | 0.72103 | `2.402e-12` |
| GPU4PySCF | `1e-16` | 0.83853 | 0.83918 | `1.990e-13` |

Relative to native default, upstream default is 5.85--5.86 times faster in
these prepared consumers. At `1e-14`, it is 5.47--5.48 times faster with a
smaller observed error. Even `1e-16` is about 4.71 times faster. This rules out
explaining this observed gap solely by a looser nominal cutoff. It does not
make either engine's full screening contract equivalent, nor prove matched
SCF convergence or force accuracy on arbitrary inputs.

In particular, native retains directional AO FP64 product comparisons and its
global density-error policy. Upstream uses float32 log-space decontracted-pair
conditions and local density admission. Equal numeric cutoff strings are not
equal accepted integral domains. Do not replace native screening with upstream
conditions merely because this one density passes an output error gate.

## Source comparison against actual execution

The actual timed native source is the adapted pinned-PR snapshot, not the older
dense-E working-tree variant. Individual source hashes are checked against the
staging snapshot before timing. Upstream source references below use v1.8.1.

| Mechanism | GPU4PySCF | Native residual/default-off tile |
| --- | --- | --- |
| Density projection lifetime | `j_engine.py` produces Hermite density once before angular tasks | Uniform J already has density-first projection; masked suffixes refresh at task/tile strip scope |
| Potential lifetime | Accumulate Hermite potentials across source batches, then separate AO projection | Default residual builds AO components; masked route still projects at opposing-primitive-chunk/task scope |
| Primitive parallelism | `md_j_2_1` / `md_j_2_2` use `(16,16)` threads for independent primitive-pair products | A warp/group owns a shell task; the residual/shared tile recurrence is produced by one lane |
| Small-class algebra | Explicit scalar R recurrence fused immediately into density/potential accumulation | Native helpers retain general indexing and compact E access; the tile caches a small R matrix but retains dynamic masks/maps |
| Pair order and screening | Descending `q_cond`, 2/4/8/16/32 maxima hierarchy | Angular segmentation, native min/max/density bounds and residual coarse prefilter; no equivalent full hierarchy |
| Specialized resource policy | Fixed unrolled tile constants; generic fallback has a shared-memory budget estimator | Fixed bounded 2-by-4 shell-pair tile and 2-by-2 primitive chunks for the experiment |

Observed upstream data preparation expands to 928 one-primitive,
one-contraction Cartesian shells and 1760 sorted AOs. At `1e-14` its accepted
pair mappings hold 376176 Hermite-density doubles, about 3.01 MB. This is only
one buffer, not upstream total memory, and must not be compared with native's
662672188 total owner bytes as if they were the same inventory.

The unrolled `(2,1)` dispatch uses 256 threads, a logical 768-by-480 primitive
pair rectangle and 48896 dynamic shared bytes for nonnegative omega. `(2,2)`
uses a 224-by-224 rectangle and 47360 bytes. Those are primitive-pair units;
native's two-by-four shell tile is not directly comparable by tile area. The
Python `_md_j_engine_quartets_scheme` resource estimate configures the generic
fallback, not runtime autotuning of these fixed unrolled specializations.

Important source anchors:

- `gpu4pyscf/scf/j_engine.py`: density preparation near line 96, sorted pair
  cache near 245, hierarchy near 276, generic scheme near 297.
- `gpu4pyscf/lib/gvhf-md/unrolled_md_j.cu`: `(2,1)` near 914, `(2,2)` near
  1232, fixed dispatch near 4691 and 4697.
- `gpu4pyscf/lib/gvhf-md/contract_Et.cu`: separate density and AO projections.
- Timed native `direct_md_jk.cu`: `contract_source_task`, exact directional
  orbit/mask scatter, and source scheduling; compiler
  `md_j_masked_tile_cuda.py`: chunked suffix reuse and task-local publication.

The installed library has sm70/sm80/sm90 cubins, no native sm120 cubin.
`cuobjdump --list-ptx -all` also reveals embedded sm80 PTX; omitting `-all`
incorrectly reports no PTX. Execution on this RTX 5090 can use driver JIT from
that PTX. A separately rebuilt upstream sm120 object from the earlier audit is
not the binary timed here. Do not equate its static SASS counts with this
wheel's measured instruction counts.

## Captured-kernel attribution and work

Slurm 10294 separately captures one post-warmup complete J per engine with
Nsight Systems 2025.1.3. CUDA profiler start/stop excludes owner preparation.
Nsight warns about CUDA event-completion tracing overhead, so these are
attribution receipts, not replacements for the clean timing distribution.

| Captured kernel work | Native default (s) | Native masked tiles (s) | GPU4PySCF `1e-14` (s) |
| --- | ---: | ---: | ---: |
| All J kernels | 3.93416 | 4.30820 | 0.71337 |
| Full `(2,1)` angular class | 0.88914 | 1.27159 | 0.09795 |
| Full `(2,2)` angular class | 0.44663 | 0.44592 | 0.05105 |

Native class totals include both its reciprocal uniform potential and residual
consumer; comparing only its residual kernel against upstream's complete class
would mix scopes. Upstream `(2,1)` totals two launches and `(2,2)` three launches
for different individual shell-angular combinations. Work/screening domains
still differ, so these are not per-integral throughput ratios.

Native default spends 2.47309 kernel seconds in residual consumers and 1.46045
in uniform potential kernels. Density/projection work is about 0.00061 seconds.
Native launches 39 kernels versus upstream's 133. Native adapter transfer time
is about 1.56 ms versus upstream's 0.83 ms. Neither raw launch count nor host
transfer overhead explains the multi-second gap. Upstream 1.8.1 itself calls
`vj_xyz.get()` before GPU-backed final projection in this case; its intermediate
host round-trip is not a CPU integral/reference fallback or a zero-copy path.

A fresh counted native replay has 2038308216 uniform radial sources and
242399046 residual primitive products, totaling 2280707262 source evaluations.
It contracts 43920982 residual shell tasks. A separately labeled source replay
of upstream's exact float32 hierarchy and fixed/generic tile dispatch counts
1304707584 recurrence slots, of which 1304434004 have nonzero combinatorial
prefactors. It visits 6948715 microtiles, rejects 1696028 by density, and executes
5252687. These upstream numbers are source-replayed, not instrumented kernel
counters. They concern a decontracted and differently screened source domain;
do not divide the counts and call the quotient a kernel-efficiency factor.
Historical native counts from another binary are not substituted for these
fresh source-matched receipts.

## Fresh hardware counters

Slurm 10295 uses the authorized account and sudo, preserving Slurm visibility
after sudo/PAM with a fail-closed receipt check. It profiles the same card,
binary, frozen density and CPU oracle. Each selected launch uses 22 NCU replay
passes. Clocks are unlocked; hardware counters diagnose execution, not a clean
endpoint timing distribution. No password is stored in scripts or receipts.

| Launch domain | Mean active threads / warp instruction | Registers/thread | Achieved occupancy |
| --- | ---: | ---: | ---: |
| Native residual `(2,1)` | 4.39 | 104 | 33.22% |
| Native tiled `(2,1)` | 3.77 | 99 | 33.33% |
| Upstream complete `(2,1)`, two launches | 27.43--27.54 | 109 | 23.86--31.72% |
| Native residual `(2,2)` | 5.87 | 99 | 33.00% |
| Upstream complete `(2,2)`, three launches | 28.26--28.70 | 128 | 22.28--31.18% |

These instruction-wide active-thread ratios are not isolated FP64 lane metrics
or proofs of a unique stall root cause. However, they substantiate the source
mapping difference. Upstream is faster despite higher register counts and
sometimes lower occupancy. Reducing registers or enlarging shared tiles alone
does not address the observed inefficient instruction/dataflow organization.

The current lookup tile executes 320.258 billion instructions versus 269.242
billion for native residual `(2,1)`. DFMA/DMUL/DADD opcode instances are
13770748455 versus 12218058426, while FP64 atomic opcode instances fall from
79282176 to 24946624. These are opcode instances, not FLOPs or scientific work
counts. This is fresh evidence for the final lookup binary, not the earlier
fused variant's counters. Upstream totals are retained per launch and class;
their different work domains preclude a naive same-work instruction ratio.

## Optimization direction, not an unmeasured promise

1. Prioritize compiler-generated primitive-lane schedules and explicit
   small-class R-times-density DAGs, sharing the established scientific algebra.
   Do not copy a second handwritten recurrence or merely lower register limits.
2. Extend Hermite potential lifetime beyond opposing two-primitive chunks.
   Upstream projects only after accumulating across many source partners. Exact
   masks require stable output-bound/direction channels or equivalent remapping
   across partner-dependent cutoff groups; simply accumulating unrelated local
   group IDs is incorrect. Keep bounded storage and AO fallback.
3. Avoid repeatedly building all unused suffixes and sorting/reloading the same
   entries at short tile lifetimes. A useful next experiment must report actual
   refresh counts, executed instructions and complete-J time, not only fewer
   output dot-product terms.
4. Consider descending bounds and conservative hierarchy only under native's
   original FP64 AO mask and global density budget. Upstream float32 log/local
   conditions cannot be substituted as an optimization preserving that contract.

Residual-only optimization is not the whole gap: native uniform potential
kernels still account for about 1.46 seconds of cumulative work here. Keep the
broader consumer in view after a residual improvement is established.

No default promotion, commit or performance PR follows from these comparisons.
No cold SCF/force speedup is claimed. Future production promotion still needs a
reproducible native-vs-native complete-J win and independent source-identity-
correct SCF/convergence/force qualification.

## Reproduction and receipts

Local ignored scripts, raw clean samples, fresh CPU oracles, hashes, CSVs,
source-replay work and summaries are under `build/gpu4pyscf-compare/`.
`benchmark_96.py`, `cpu_reference.py`, `trace_one.py`, `census_gpu4.py` and the
Slurm wrappers are retained on n1 under
`/home/jzzeng/generativeqc-qualification/gpu4pyscf-compare-20261019-49fc/`.
The `census_gpu4.py` copy is the source-replay audit, not a production kernel.

`j96-10292/result.json` and `j96-10294/result.json` pin installed/native source
and library hashes, versions, tolerance semantics, per-sample accuracy, timing
boundaries and preparation. `summary.json` validates paired results, trace
identities and work scopes. `counter-summary.json` preserves all eight selected
NCU launches, and `counter-reports-10295.sha256` identifies the large remote
reports. No Release or external artifact publication is used for retention.

Related rationale:
`2026-10-19-bounded-tiled-masked-hermite.md` and
`../../proposed/2026-10-18-residual-j-tiled-density-first.md`.
