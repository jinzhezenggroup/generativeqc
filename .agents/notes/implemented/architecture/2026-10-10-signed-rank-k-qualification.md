# Decision: keep signed rank-k as a provider-neutral TensorIR projection

Status: implemented (qualification-only; no production caller)
Date: 2026-10-10

## Problem

#1877 still lacks the rank-k half of shared symmetric CUDA lowering. The
rank-2k cross-product path is already owned by #1876 and is a different
contraction. Density and energy-weighted density use the existing ternary
`tensor.scf` equation; the latter may have negative orbital energies, so its
weights are signed.

## Decision

`symmetric_rank_k_request` recognizes the existing `C[...,p,i] w[...,i]
C[...,q,i]` node. It requires the same coefficient node on both legs, matching
scientific index domains, strict FP64, and dense row- or column-major physical
views. A preceding multiplication may materialize the weights. The scientific
request has two explicit roots: overwrite owns `alpha * Gram` without a beta or
old-output input, while update owns `alpha * Gram + beta * old_output`. The
update input is a symmetric logical view bound from the physical upper triangle
by one generated reader used in validation and publication. Both formulas are
scalarized into separate native checked helpers, so runtime
publication has no second algebra. Each request validates and binds the original
Gram program, scientific request and precision identities before composition;
unsupported audit/lowered provenance fails closed rather than being normalized.
Overwrite marks output write-only; update marks it read-write. Both record
upper-triangle ownership, complete mirroring, and one all-batch
transaction: a detected numerical failure publishes no output block.

Two candidates share that request. The generated fallback uses the existing
`tensor.scf` weighted-Gram scalar emitter for signed scaling and ordered
accumulation. It validates and recomputes the upper triangle in two passes,
requiring no retained device scratch. The optional cuBLAS candidate scales one
coefficient panel by the **signed** weights, computes a full GEMM into scratch,
validates the authoritative triangle, then publishes and mirrors it. It charges
`8 * batches * (n*k + n*n)` temporary bytes plus the prepared provider
allowance. It is test-only until complete endpoint qualification and production
consumer review; unknown timing does not justify default promotion. Failed
provider preparation retains an executable generated fallback.

The prepared binding fixes the stream, device, dimensions, physical order and
provider choice. Inputs, output, error storage and retained scratch have
separate lifetime and alias contracts. The caller initializes the error word
on the same stream for each execution and captured replay, and keeps inputs and
output exclusive until the stream or captured graph has finished. The caller
destroys and drains captured graphs before destroying the binding.

## Rejected alternatives

- `sqrt(w)` plus SYRK silently loses negative weights and changes the real
  arithmetic domain.
- A method-local cuBLAS selector would duplicate provider policy and make
  SCF/DF-SCF/GFN2 drift separately.
- Treating the rank-2k cross-product executor as rank-k would misstate its
  operands, work and numerical publication contract.
- Promoting cuBLAS from a descriptor or kernel-only timing would omit scaling,
  full GEMM, validation, mirroring and provider allocation.

## Invariants

- TensorIR and `tensor.scf` own the mathematical expression. Generated scalar
  bodies and native metadata are source-bound to the same original node.
- Mixed precision, gathered axes, aliasing and unsupported physical order
  fail closed. Row/column-major and transpose choices are explicit.
- The generated candidate remains executable without cuBLAS or its scratch.
- The upper input triangle is the only old-output source when `beta != 0`;
  lower input entries may differ and are replaced on successful publication.
- `beta=+0` and `beta=-0` select the overwrite request/helper and never read old
  output; all other beta values select the update request/helper. Both providers
  must agree on the selected request identity.
- Preparation retains immutable diagnostics for both roots. Queries select the
  record from the invocation beta; no mutable last-mode state may race captured
  replay or concurrent submissions.
- A single error word intentionally means all-batch failure isolation, not
  per-system isolation or ragged production admission.

## Evidence and scope

The standalone H100 harness compares both candidates and both physical orders
against a CPU `long double` reference over odd rectangular panels, two batches,
signed and negative-zero weights, nontrivial alpha/beta, capture/replay, alias
and order rejection, dimension-overflow admission, a zero-budget executable
fallback, and nonfinite all-output preservation. Its device-resident endpoint
time includes weight materialization, signed scaling, GEMM or two generated
reduction passes, validation, mirroring and output reset. It reports semantic
work and resource bytes separately. The generated implementation executes
`2 * batches * n * (n + 1) / 2 * k` products and scaling steps because it
validates then recomputes; the library GEMM executes a full
`batches * n * n * k` product plus `batches * n * k` scaling. This is a
capability qualification; it is
not full SCF/SCC endpoint evidence and does not satisfy the two-production-
consumer criterion of #1877.

The first H100 job failed before compilation because the minimal CUDA image
lacked `git`; the second failed before compilation because it lacked Python;
the third reached compilation and exposed an undefined CUDA NaN macro. These
negative receipts remain under the task's qz result directory. A CPU stage now
pins a clean Git commit and SHA-256 manifest for source files and the emitted
header; the GPU job checks byte integrity before compilation. SHA-256 here is
an integrity check, not an authenticated signature.

An initial 16-case H100 device probe used CUDA 12.8.61 and passed, but the
repository's supported CUDA floor is 12.9. The source-frozen qualification
therefore uses an independently copied, hash-recorded CUDA 12.9.86 toolkit,
explicitly resolves its cuBLAS and CUDA runtime libraries, pins the inventoried
host compiler, and wraps both compile and link with sccache. The 12.8 probe is
historical evidence, not a source-matched supported-toolchain acceptance.

At commit `68de92cd32a356c1c05e41c3935cc206da77a7ea`, qz Job
`i1877-rankk-h100-1010z10` completed all 16 cases on H100 with CUDA 12.9.86.
The tracked compact record and all 16 accepted case rows are retained in
`benchmarks/results/rank-k-1877-20261010/`. Full raw JSONL, source/artifact
hashes, 1,581 file checks, cache receipts and negative trials remain at the
task-owned qz result path; pre-`j` raw receipts are also in Git history at commit
`9ec7fc52e408c062802db6e68de0f31eca7eff1f`. They are not implied to have been
independently retrieved merely because their hashes and locations are recorded.
The complete prepared device endpoint measured 12.54–21.45 µs for the generated
route and 25.58–35.78 µs for cuBLAS over the tested small panels. Those receipts
qualify executable alternatives, not a full method endpoint or a profitable
production library default.

Pre-commit bot commit `9ec7fc52e408c062802db6e68de0f31eca7eff1f`
reformatted the native header and device harness after Job `h`, including an
include-order change. Job `i` then failed before compilation because the CPU
stage wrote the generated-header manifest relative to the task root instead of
the verifier's repository root. Job `j` fixed only that manifest path and
source-matched `c513dccb`. The subsequent review fixes bound complete source,
toolchain and compile-recipe inputs, keep layout out of semantic identity, and
compare dummy axes by scientific domain. Job `k` source-matched those final
compiler/native/harness inputs at `f0c02130`. Independent review then found the
host compiler, CUDA headers and ambient override inputs were not fully bound.
Job `l` proved the task-owned Python and expected host bytes; Job `m` rejected
the platform's ambient `LIBRARY_PATH` before source checking. Job `n` ran with
the fixed empty override environment before the cache-version parser repair.
Job `o` regenerated identical metadata from the actual GPU-side inputs and
accepted the numeric sccache floor, but still carried only the unit-Gram
scientific identity. Job `q` bound the complete alpha/beta/old-output update
root and scalar helper but normalized original precision provenance. Job `r`
first validated and bound the original source request/scientific/precision
identities and rejected unsupported audit/lowered provenance. Job `s` qualified
the later overwrite/update split, including zero-beta no-read behavior, but its
compiler digest still omitted CUDA child tools. Dry-run Job `u` identified the
fixed recipe's actual `cudafe++`, `cicc`, `ptxas`, `fatbinary` and `nvlink`
invocations plus libdevice, link stub and device-runtime inputs. Job `v` hashes
that closure but predates the old-output binding and `nvcc.profile` inventory.
Job `w` qualifies those two repairs at `9b9b2a02c`, but a concurrent upstream
merge changed the full repository source inventory. The branch therefore merged
current `origin/master` without rewriting history, and Job `x` source-matches
that resulting `451045b18` tree. Review then found that `17x9` rows still reused
the `3x5` request and that only the host compiler driver was hashed. Job `y`
failed before manifest generation because the CUDA image has no Git; Job `y2`
generated the 3,666-role fixed GCC closure. Job `z` failed closed before timed
rows because the new shape gate preceded the existing overflow classification.
Job `z2` preserves overflow admission and uses distinct per-shape identities.
The CI mode repair marks the manifest generator executable without changing its
bytes. Review then found one prepared diagnostic still labeled every invocation
with the update candidate. The binding now retains immutable overwrite/update
diagnostics and selects by invocation beta without last-mode state. Final review
also required exact invocation `n/k` and flattened batch/spin prefix,
an overwrite-specific semantic formula, and accurate timing language for the
device reset. The flattened batch/spin prefix is also exact, not a capacity
subset. After upstream #2170 changed the full source inventory, the branch merged
latest master and Job `z8` source-matched that exact `c4b561f99` tree. Subsequent
review bound the constructor to the exact request prefix, included the directly
compiled harness in compilation identity schema v4, and aligned bare GCC child
resolution with the final compilation PATH. Job `z9` source-matched the resulting
`c2ab6a57e` tree. Upstream #2175 then changed the full source inventory without
changing the qualified rank-k implementation blobs, so the branch merged
`09bfb1d35` and Job `z10` regenerated the resulting compilation identities for
exact tree `68de92cd3`. Job `z10` is the accepted run. Its generated header,
object, binary and raw hashes are in the compact receipt. Later receipt-only
commits may reuse `z10` only while all qualified implementation blobs and modes
remain identical and latest-head review verifies that boundary.

## Revisit when

Two independent production consumers can adopt the prepared capability
without touching another active owner, with complete endpoint numerical and
performance receipts, bounded resources and independent latest-head review.
Only then consider a default library candidate or closing #1877.

## References

- #1877 and its 2026-10-06 scope reduction comment
- `python/generativeqc_compiler/tensor/symmetric_rank_k.py`
- `src/tensor/cuda_symmetric_rank_k.cuh`
- `tests/native/test_symmetric_rank_k_cuda.cu`
- `tools/qualify_symmetric_rank_k_cuda.sh`
