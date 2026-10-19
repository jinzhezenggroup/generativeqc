# Decision: bind CC semantic configuration separately from observation controls

Status: implemented in benchmark tooling; scientific qualification is separate
Date: 2026-10-19

## Problem

The old campaign hash covered numerical thread counts, visibility and loader
paths but omitted CUBLAS_WORKSPACE_CONFIG. A hash string alone could not establish
the actual child configuration. Historical absent fields remain UNKNOWN; neither
an empty string nor absence can be reconstructed from them.

An independently bound four-process H200 diagnostic on frozen native source
129d measured 354 MiB retained with explicit unset (native exits 1/1), and
34 MiB with :16:8 (native exits 0/0), under the same original 192 MiB allowance.
The latter two outputs pass post-hoc energy/(T)/physical-replay scalar checks
against the committed PySCF oracle and have equal energy bits/work. They are
one unprofiled plus one guard-observed sample, not a formal campaign or full
producer/copy/wait capture. This configuration contrast does not attribute all
retained bytes to one provider or promote a production default.

## Contract

Campaign and attribution v2 require the complete named semantic environment
object and recompute its canonical hash. Every key has explicit Boolean presence
and string/null value. Missing fields, illegal combinations and rehashed equal
malformed objects fail closed. Actual observation/injection controls remain in a
separate retained execution environment; intentional diagnostic differences do
not erase the frozen semantic profile or become matching execution hashes.
The independent identity also freezes complete clean and diagnostic observation
roles. Clean samples cannot preload/inject an observer or silently change debug
logging costs. Diagnostic injection must match the declared role. Only designated
log-file controls may use a restricted original-run-directory template; moved
raw packets retain that original path identity rather than acquiring a new one.

The runner freezes both child-launcher dependencies and the plan in each run.
A same-PID pre-exec witness samples only the declared keys, verifies the planned
mapping, and execs the fixed native argv with it. Parent-side Linux executable
and mapped library device/inode/stat binding are retained. The reporter rehashes
the artifacts and repeats environment, child PID/argv, launcher and library
checks independently of recorded PASS flags. Successful registration, equal
hash labels or three equally invalid receipts cannot substitute for the schema.

Content hashes are read before/after native execution. Startup attestation
inside the run reads process mappings and file identity only. An independent
supervisor enforces the absolute child deadline and owns bounded kill/reap even
when parent attestation is delayed. Reported child wall includes witness/exec
startup and wait observation; it excludes the parent's attestation/finalization
tail. Parent elapsed/attestation wall and Nsight's diagnostic window stay separate.
This is an unprofiled supervised invocation, not a bare C++ kernel timing.

## Boundaries

No SCF/CC production source, provider enablement or guard allowance changes.
Old raw packets remain immutable and unbound rather than being migrated into
an invented unset profile. The 1cc source approval retains its old identity and
does not approve a later tooling delta. Required CI, latest-head formal review,
two clean endpoints plus an independent capture, full producer attribution and
normal protected delivery remain separate gates. Double buffering, forces,
response and default promotion remain parent scope.

## Evidence and references

The configuration diagnostic and independent raw/child-binding verification are
retained locally in .artifacts/pr2581-workspace-20261019 and in the same-named
issue1765-workspace-129d-20261019 project shared directory. Root receipt SHA-256:
fa545fd0928cc1d69aa5cd8c7d6fea9b818c7c55bbf9fa371356c3319cfe3111.

NVIDIA CUDA 12.9.1 cuBLAS section 2.1.4 documents :16:8 as a debug reproducibility
setting that may limit performance. The cuSolverDN create documentation describes
an initial workspace on SM >=9; it does not prove ownership of the full observed
354 MiB. These are diagnostic hypotheses, not automatic scientific acceptance.
