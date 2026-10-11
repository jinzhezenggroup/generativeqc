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
