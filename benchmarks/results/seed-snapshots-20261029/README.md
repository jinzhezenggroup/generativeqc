# Native warm-seed snapshot qualification

PR #2868 moves precision-audit numeric seed copying and lifetime into immutable
native host snapshots. The shared physical validator and target resource owner
retain restoration. Python still owns model identity, history selection and audit
scheduling. This scoped #2227 R2/R4/R6 migration excludes R3 and does not complete
the parent.

Production source is `f7c7d6ba9`, compared with #2865 production `115cc3c72`.
The later public API census fix changes a documentation test only.

The CPU matrix retains all 1,792 audited complete calls, identical semantic/audit/
seed work and exact energy/force parity. Its maximum median time ratio is
1.0202404723601008, below the unchanged 1.05 gate. Raw samples are retained here;
`identities.json` binds sources, binaries and worker settings. The protocol uses
`../seed-capacity-20261029/qualify.py` and `endpoint_worker.py` unchanged.

Real-GPU qualification is still pending. This CPU receipt is not CUDA acceptance.
