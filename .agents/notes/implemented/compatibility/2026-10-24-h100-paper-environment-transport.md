# Decision: preserve frozen scientific functions behind a qz setup adapter

Status: implemented for CPU preparation and host tests; device acceptance incomplete
Date: 2026-10-24

## Problem

The paper's archived HF and CBS drivers require real Slurm metadata. qz uses a
different allocation system. The HF CLI also requires positive warm repeats,
while #2726's setup qualification calls for one tiny complete endpoint per
engine. Inventing Slurm fields or changing frozen scientific functions would
invalidate the intended source and allocation provenance.

The frozen CBS production-profile resolver also rejects `auto` on sm90. A tuned
sm120 schedule cannot establish H100 qualification.

## Decision

The setup family owns a distinct, hash-bound qz transport adapter. It calls the
original HF `run_native` / `run_reference` functions with zero warm repeats, or
the original CBS `geometry` / `setup` / `native_call` functions for C2, full RI-JK,
Standard and 32 GiB numeric budget. It does not call or modify the Slurm-only
CLI. Source-driver, adapter, protocol, environment and native-library identities
remain distinct. Only the coordinator can submit the actual finite GPU Job.

After actual host preflight established that the original `portable_cuda` branch
supports sm90, the coordinator authorized an explicitly named
same-scientific-source / portable-schedule H100 cohort. This preserves the source
while recording the schedule change. It cannot be described as an identical
generated-schedule hardware-only comparison. The original auto-profile refusal
is retained. No runtime, provider-admission, mathematical or precision code is
changed by this slice.

## Evidence and limits

The family at `benchmarks/results/h100-paper-environment-20261024/` retains CPU
readiness, offline package preparation, failed dependency attempts, no-driver
GPU4PySCF import, authenticated CBS source reconstruction and profile preflights.
Host transport tests verify one-call parameters, complete-force validation,
hash rejection, no-clobber output and saving returned values before rejection.
Independent numeric byte/index files preserve original NaN payloads, infinities
and signed zero; JSON nonfinite tags alone do not provide byte evidence.

The accepted prepared protocol is commit
`df5c44a81c5d73210a8d2f7f09ba4930d7f67210`. CBS reconstruction uses base
`976f756b391bc65edb79f76df1a8c3fe807518a7` and decoded patch SHA-256
`5a1a974bab09e0fe31e6186e6ec2129abf26961113aafcd3d9c81616eb2d7cad`.
All six archived pinned files match. A new build, imports, finite E/F and
scientific/oracle acceptance are separate statuses. H100 execution remains
NOT_RUN at this note's creation; a CPU build is not device acceptance.

## Revisit when

Revisit only if the frozen source's existing portable branch cannot execute the
complete public endpoint, or an actual allocated H100 exposes a provider or
scientific failure. Such failures belong to the production owner and require an
explicitly identified follow-up cohort if source changes are necessary. They do
not authorize relaxing admission, changing the model/budget, or silently using a
newer runtime.

## Independent HF cohort authorization

The coordinator subsequently accepted the measured HF/DFT source
`cbd4d8074de35105717c8aac7cd1f86babbe90ec` for a separate source root,
inventory, profile preflight and DSO. Its build must wait for the live CBS
attempt's natural terminal and CBS artifact validation, then execute once under
the same finite engineering bounds. Existing legal portable sm90 selection may
be used only after actual HF profile preflight; identical resolver/manifest
hashes observed by read-only inspection are not themselves a completed HF
preflight. The unresolved historical 5,852-file package/archive provenance stays
UNAVAILABLE. This CPU authorization grants no GPU call and does not permit
using the CBS DSO as the HF implementation.

## Public-project environment relocation

The later public-project route retains the same source trees, driver hashes,
two DSOs, portable sm90 profiles and scientific parameters. Its separate caller
changes the admitted Project and binds new absolute paths and a new expected
Job name. The original transports, failed attempts and scientific limits remain
part of the record.

A venv created with `--system-site-packages` from another venv did not inherit
that parent venv's packages. Both the original creation receipt and a real
pre-materialization import failure demonstrated this. The minimal network
closure therefore carries the private overlay, while an exact file inventory
binds reusable image assets. Only those known files are materialized locally
with exclusive creation and their original modes. The two original startup
hooks are restored byte-for-byte; no new `.pth` or `PYTHONPATH` search surface
is added. Actual loaded module paths, hashes and normalized `sys.path` must
match the original private environment.

Hashing a copy's input stream and calling `fsync` does not establish a target
readback. Qualification additionally rereads the actual target files and checks
their sizes, hashes, modes and approved symlink targets. The public preparation
reread 40,661 files totaling 5,717,381,795 bytes and checked eight links. This is
environment integrity evidence, not device or numerical qualification.

Subprocess diagnostics must retain stdout and stderr separately. A GPU4PySCF
warning appended to otherwise valid import JSON exposed a collector failure;
the repair parses stdout without suppressing or discarding the warning. The
CPU's `cudaErrorInsufficientDriver` result remains a negative result. Passing
host tests, relocated imports and dependency checks do not establish H100 E+F
acceptance.
