# Decision: establish native grid identity in its consuming owner

Status: proposed
Date: 2026-10-24

## Problem

The Stage14 PBE DF profile reads and SHA256-hashes the same large native library
in `_native_grid_artifact` (0.543 seconds) and `CudaGrid` construction (0.547
seconds). The first read creates an artifact with a freshly computed digest;
the second verifies that newly constructed artifact. This is separate from the
benchmark's binary-identity hashing, which is outside the timed endpoint.

## Candidate

Represent a native-build input as an unbound `NativeGridLibrary` request. After
resource admission, the consuming grid reads the complete binary and creates
its ordinary `CudaArtifact` immediately before loading. The prepared stationary
owner stores this bound artifact, including its unchanged schema, SHA256 and
key. An unprepared consumer similarly publishes only the bound artifact.

This is not a file-digest cache. Every new owner hashes current complete bytes;
reusing a request does not reuse a digest. Externally supplied compiled artifacts
retain their independent expected-digest verification. Prepared geometry-refresh
rehashing, artifact mismatch failures, allocation preflight and cleanup order
remain unchanged. No stat-only identity or mtime/size shortcut is introduced.

## Acceptance

Host gates check current-byte binding, replacement before admission, external
artifact corruption, missing files and preflight budgets. Existing complete
owner/lifetime/composition tests and independent GPU energy/force/moved controls
must pass. Compare complete cold endpoints before claiming a gain; profile time
is a diagnosis, not an endpoint speedup. Compiler provenance must be rebuilt for
the changed Python source closure even though scientific CUDA formulas do not
change.
