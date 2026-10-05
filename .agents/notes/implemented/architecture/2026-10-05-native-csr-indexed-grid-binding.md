# Decision: bind native KS CSR spans to compiler-emitted indexed AO/grid domains

Status: implemented; host/device correctness-qualified, P0-C performance acceptance pending
Date: 2026-10-05

## Problem

The spatial/resident-force producer already publishes `AoGridBlockLayout`, but
ordinary native KS still passes independent counts and pointers under a
`local_ao` flag. That route must preserve the same point/AO/derivative domain
contract without a second CSR inventory or a Python/GPU coordinate download.

## Decision

`dft.indexed_layout_native` emits a native immutable-owner specialization of the
data-only domain. Both semilocal and nonlocal native consumers bind the existing
CSR span before collocation, local density projection and potential scatter.
Counts and map addresses come from this descriptor rather than independent
late-path arithmetic. The native owner freezes its packed basis and quadrature
for its lifetime; its owner token is not a dimension-only identity and replaces
the separately mutable Python epochs. Descriptors are stack metadata and never
outlive that prepared owner. There is no new numerical map/tensor cache.

CSR producers now declare a complete through-order derivative capability.
Unknown, insufficient and unsupported capabilities are rejected by native
layout admission. Sampled discovery records only the jets actually tested;
evaluating a later consumer's jets cannot promote a discovered map. Explicit
maps declare the caller's chosen scientific domain, not a certified energy or
force error bound. The native independent CPU bilinears still zero omitted
columns in the full global space.

Sorted unique full-occupancy physical maps are exactly the identity. The emitted
binding omits their AO-map dereferences, while preserving the existing charged
arena, offsets, producer work counts, stream, accumulation policy and fallback
admission. Empty spans are legal; a nonempty span requires live index storage.

## Boundaries

This is the native lowering's binding of the shared indexed-domain semantics,
not execution of `AoGridBlockProgram` through a generic TensorIR runtime. The
immutable-owner specialization does not export mutable Python descriptors,
invent a scientific basis hash, shrink conservative arena admission, change
screening or precision, or qualify a new GEMM provider. It does not solve the
remaining profitable tiny/high-occupancy crossover policy. Do not equate
representability or removal of map dereferences with endpoint improvement.

## Evidence and provenance

- Device-free tests compile the emitted host binding with verified ccache.
  Subset, empty, identity and tail tiles match `AoGridBlockLayout` exactly.
- Fifteen negative probes reject unknown/insufficient capabilities, attempted
  Hessian promotion, unsupported jets, invalid point/CSR extents and null maps.
  Equal-shaped independent native owners have distinct binding tokens.
- Combined focused host gates: 173 pass, three explicit GPU opt-ins skip.
- Compiler structure: 444 modules, zero dependency errors.
- Native ccache-launched rebuild succeeds. Native local-map gates additionally
  reject unknown, insufficient and out-of-range producer capabilities.
- This is v6, after the frozen v5 source/native identity
  `65dec73d485ca144714d3b4b26b7b9f51b1b4efd586983ffc5b7b9ceb8329260`.
  Slurm 6070 is v4 evidence; Slurm 6082 is v5 evidence. Neither qualifies these
  new native CSR bindings.

## References

- [Indexed AO/grid domains](2026-10-05-indexed-ao-grid-domains.md)
- [Spatial/resident consumers](2026-10-05-indexed-ao-grid-consumer-binding.md)
- Issue #1893, P0-C. Complete 48/96-atom endpoints, actual force histories and
  independent native E/V/capture/nonlocal gates remain acceptance authorities.

## v6 real-device evidence

The independently verified v6 source/native identity is
`6fb2d9b652e10a78ce921699b95c6a812242adc5ca3da3a649f7541959af7750`.
n1 Slurm 6089 passes 216 grid/spatial tests, native local-AO independent CPU E/V,
capture/nonlocal/resource gates, and 19 cases each under memcheck/racecheck with
zero errors/hazards/warnings. Combined host validation is 201 pass, four explicit
GPU opt-ins skip. The clean 48/96 paired campaign is still running at this
append; no new endpoint improvement is inferred from these gates.

6090 completes an independent kernel census of the first geometry-zero warm
public force invocation and passes 72 reference pairings. It is intrusive and
does not supply clean endpoint performance. The captured API includes internal
preparation (the trace contains eigensolver work); the helper's initial broad
SCF-exclusion annotation must not be treated as proof of a pure-force scope.
The audited receipt in `.artifacts/n1-force-kernel-profile-6090/` corrects that
interpretation without modifying the frozen raw evidence or its harness hashes.

### Completed v6 endpoints and broader gate packaging failure

6089 completes on October 5 at 23:44:16 Asia/Shanghai, exit 0. All four clean
candidate/control campaigns pass 72 numerical/work pairings each. Warm medians
at 48 are 9.005482 s candidate / 8.997381 s control; at 96, 24.734296 /
24.792437. These 0.090% regression / 0.235% improvement are effectively flat,
not material reference-gap closure. Candidate cold/moved histories are 23/12
Focks at 48 and 26/13 at 96; controls are 25/12 and 28/14. Preserve these actual
histories rather than infer a convergence improvement from integer metadata.
Maximum candidate E/force errors are 8.640e-12 / 2.743e-11 at 48 and
1.023e-10 / 2.209e-11 at 96. Domain reconciliation against 6071 passes 24 pairs.

n1 `srun` 6096, main/5090/one GPU with a finite 20-minute limit, passes all 24
indexed TensorIR CUDA/host gates. Its broader force selection has five pass
and seven failures, all `FileNotFoundError` for omitted packaged LDA/PBE/R2SCAN
stationary modules; these are not numerical discrepancies. Preserve exit 1 and
raw logs in `.artifacts/n1-indexed-force-gates-6096/`. The native PBE0 campaigns
do not qualify missing UKS/meta-GGA package routes. Build and identity-check
the six required AOT modules with verified ccache, then qualify in a new expanded
snapshot without overwriting the original v6 snapshot or its receipts.

### Expanded package gates completed on October 6

The six missing LDA/PBE/R2SCAN RKS/UKS AOT modules are built with verified ccache
launchers and independently validated by the canonical packaged loader: exact
plan and contract identities, FP64, `fmad=false`, architecture `sm_120`, and
library/manifest checksums. No scientific source or main-library bytes change.
The new snapshot is `/data/jzzeng/qc-1893-p0c-layout-v6-aot-20261005`; its source
and embedded main-library identity remain the v6 identity above. The original
v6 snapshot, omitted-module failure and clean endpoints are not overwritten.

n1 `srun` 6106 uses main/one 5090 GPU, a finite 20-minute limit, and preserves
the scheduler's visibility. All 12 selected complete-force gates pass in
180.17 s, with 48 unrelated cases deselected: independent analytic RKS/UKS
LDA/PBE/R2SCAN and public active-AO force replay (including PBE0). There are no
skips in the selected gate. This repairs qualification packaging, not a
production numerical discrepancy or a performance regression.

Retained receipts: `.artifacts/n1-indexed-force-gates-6106/`,
`.artifacts/layout-v6/aot-build/identity-receipt.json`,
`.artifacts/layout-v6/aot-binaries.sha256`, and before/after ccache statistics.
Future portable broad-force qualification must preflight the actual requested
profile libraries and identity sidecars, not merely the main library and generic
Python harness. Correctness qualification does not establish P0-C completion:
the measured 48/96 warm endpoints remain essentially flat, and profitable
provider/gather scheduling and separate fused-stage attribution remain open.
