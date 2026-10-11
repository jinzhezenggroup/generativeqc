# Shared response contract qualification

The bounded [receipt](qualification.json) records the frozen source archive,
actual library hashes, compiler-cache versions, Slurm allocation and distinct
CPU/CUDA evidence stages. Raw logs and the source archive remain ignored under
`build/issue932-response-contract/`; this is not a release or asset publication.

The final real-device run passes 91 existing response/frame tests and the native
Lambda oracle/fallback/budget executable. It preserves the original tolerances
and tests. The initially missing semilocal force artifacts are supplied by four
exact-source s/p stationary CUDA AOT libraries/manifests, not a CPU fallback or
a forces-disabled request. Device visibility is assigned and preserved by Slurm.

The 13-AO cold second-integral JIT test times out after its unchanged 300-second
compile limit on both machines. Its numerical gate is not reached. The subsequent
CPU oracle run explicitly deselects that failed case and reports the redundant
single-method orchestration skip; neither counts as success for the missing gate.

These are scoped qualification results, not completion of #932, production
resident cutover, full shipping/AOT-domain qualification, complete CUDA molecular
HVP support, or a performance comparison. The retained receipt lists the missing
gates. Existing scientific feature issues retain their own acceptance criteria.

## Upstream integration replay

The [integration receipt](qualification-integration.json) preserves a separate
source-matched replay after merging upstream branch `4a8715f35`, including its
transactional problem-assignment fix and master integration. Frozen commit
`ff000b4ad` has tree `349dcb5c17726d32c82a722fe787983743783b94`; only evidence is
added after qualification. The original receipt remains historical evidence for
its own source and is not relabeled as a run of this later tree.

The merge tree passes 153 focused tests, three rebuilt native CPU CTests,
GMRES ASan/UBSan with leak detection and Sphinx warnings-as-errors. Slurm CPU job
7547 passes the explicitly listed three-suite independent PySCF replay: 42 passed,
one redundant-method skip and one previously failed cold-JIT deselection.
The first local replay lacks PySCF and therefore cannot supply those independent
reference gates; its 28 passes and 15 skips are recorded separately.

Slurm GPU job 7552 passes all 91 existing device suites and the native Lambda
oracle/fallback/budget gates against the rebuilt library. The four exact-source
s/p stationary profile artifacts retain their original binary hashes. No
precision, tolerance, preconditioning default or scientific test changes.

For replay, use the suite lists in the integration receipt, set
`GENERATIVEQC_LIBRARY` to the matching native library, and build the four
`generativeqc_stationary_{lda,pbe}_{rks,uks}_manifest` targets. Their s/p libraries
and manifests must be adjacent to the CUDA library. The native Lambda executable
receives `--cuda`; RHF frame tests require a source-matched compiled probe through
`GENERATIVEQC_RHF_FRAME_PROBE`. Set `GENERATIVEQC_RESPONSE_CUDA_TEST=1` and
`GENERATIVEQC_RHF_FRAME_CUDA_TEST=1` only inside a finite real-device allocation:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --time=00:30:00 bash -lc '<native oracle and pytest suites>'
```

Preserve Slurm-assigned device visibility. The 13-AO numerical gate, production
resident migration and matched complete-endpoint acceptance are still missing;
neither receipt closes #932 or authorizes a default promotion.

The integration receipt also records a subsequent independent review fix at
`6fe3b911d`: bound solves borrow request callbacks rather than copying their
state before binding/budget admission. Its new copy-refusing/stateful callback
regressions pass both overloads and batches. The new frozen source/library hashes
remain distinct from the first merge replay. Rebuilt native tests and sanitizers
pass; Slurm CPU job 7562 repeats 42 passes with the same skip/deselection, and GPU
job 7563 repeats all 91 passes plus the native Lambda gates. That GPU allocation
assigns `CUDA_VISIBLE_DEVICES=3`, which is preserved unchanged.

## Opt-in conventional Lambda residency

The [resident receipt](qualification-resident-lambda.json) qualifies frozen
source `8e8cf9aab32c22d876aaef09c0c97ff02f66a3c9`. The existing resident controller
now executes method-owned right preconditioning, and the conventional CUDA
Lambda adapter keeps vectors and weighted pair projection on its existing
stream. It preserves terminal replay, unsafe-diagonal fallback and independent
physical Lambda/orbital acceptance. Full tensor-provider admission is charged;
the candidate remains opt-in, including in the native complete-force overloads.

The source passes 131 focused tests, three native CPU CTests, resident-controller
ASan/UBSan/leak detection, Sphinx warnings-as-errors and all applicable hooks.
Finite Slurm jobs 7582/7587 qualify independent noninteracting Lambda and complete
native RCCSD E+F pairs for original/changed H2 and two valid budgets. The CUDA
fixture consumes its native reference interaction source, not host raw ERIs.
Job 7589 repeats all 91 existing response/frame GPU tests without skips.

The isolated process A/B order is baseline/resident/resident/baseline. Only each
process's first 256-MiB case is process-cold; later cold labels are the first
call for that budget in a warm process. None of these calls is prepared-plan
warm reuse. Two samples and this tiny molecular domain do not prove performance
non-regression. The 512-MiB process-warm median is about 17% slower, resident
provider memory/fences increase, and **no default is promoted**. Larger domains
and a complete transfer/fence/provider/peak-memory census remain required.

Failed exploratory MO-budget admission, a missing executable alias, local
collection setup and unrelated frozen-gzip header comparisons remain in the
receipt. No acceptance gate or retained evidence bytes are weakened. DF Lambda,
HF/CPKS/MP2 residency, prepared leases/ragged recovery and the historical finite
13-AO cold-JIT gate remain unfinished; this receipt does not close #932.

## Master follow-up replay

PR #2272 was externally merged while the resident slice was being qualified.
The [follow-up receipt](qualification-resident-integration.json) freezes
`b9617de02ab3664fc24532b742c88ac647e4fb0b` on master `f90d79fcc`. Integration
retains the upstream default-off serial-scratch candidate and the separate
default-off resident-vector candidate; the earlier source receipt is not
relabeled as qualification of this later tree.

Rebuilt master bindings pass 131 focused tests, three native CPU CTests, Sphinx
and hooks. Job 7595 passes native Lambda and complete E+F pairs, but its initial
existing-regression attempt correctly rejects stale stationary profile
identities. Supported CMake targets regenerate current manifests; unchanged
compiled s/p kernels replay by artifact identity. Job 7601 then passes all 91
existing CUDA tests without skips. Neither contract validation nor force
requests are disabled to obtain that result.

Finite Slurm job 7596 passes six resident E+F calls under Compute Sanitizer:
zero errors and zero leaked bytes. Separate Nsight Systems baseline/resident
processes each trace six full E+F calls. Both have a 12,716,112-byte peak of
simultaneously-live traced dynamic device allocations and balance to zero.
Resident transfer, synchronization and kernel counts increase for this tiny
domain. This is a CUDA allocation-event census, not total physical VRAM,
module/static storage or host-heap accounting, and does not promote a default.
