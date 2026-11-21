# Decision: Expose the prepared-batch CUDA Hessian through the native SDK

Status: implemented
Date: 2026-11-21

## Decision

Publish the existing synchronous KS response bridge in the installed C
header, preserving the Python ctypes layout and using one numerical owner.
The batch supplies the accepted physical state, stream and optional shared
device ledger. Do not introduce a second SCF, a snapshot-based route or a
device-pointer SDK contract. Only terminal host output and a detached scalar
receipt are published. Validate disjoint buffers before entering the response
owner, and leave both outputs unchanged on failure.

The C++ wrapper accepts caller-owned spans for dense and directional output.
It exposes an energy-only option on `Batch::execute` because the C batch
requests forces whenever any result provides force storage. Keep the
existing default for callers consuming forces; absence of a force provider
does not disqualify the SCF state needed by a Hessian.

## Evidence

- Slurm 2296 (RTX 5090, SM120): the public-header-only C++ client executes
  Direct range-PBE, preserves its HVP output before SCF and on budget
  exhaustion, then publishes a symmetric raw Hessian and matching HVP.
  Independently reconverged Libcint/Libxc energies meet the
  `2e-6` Hartree/Bohr² Richardson gate. The same library passes four Python
  and CLI range-PBE regressions, including two fitted radial sources with ECP.
- Strict C11 compile/link and metadata calls, four host test targets and
  the native API contract suite pass. CPU KS cannot enter CUDA curvature.
- Binary checksums and logs are under `build/hessian-gpu-sm120/ks-hessian-sdk-*`.
  Neither this endpoint nor the CLI transport claims non-SM120 validation.
