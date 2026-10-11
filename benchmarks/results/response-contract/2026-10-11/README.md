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
