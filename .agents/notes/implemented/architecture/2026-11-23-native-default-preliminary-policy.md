# Decision: Select the default cold-start descriptor in native code

Status: implemented
Date: 2026-11-23

## Decision

`generativeqc_initial_guess_default_kind_v1` owns default model selection and
the execution descriptor kind. Python and opt-in CLI use the same query; SDK
callers may pass the returned kind directly into the existing options. The old
model and system queries keep their provider-specific behavior. The default
query takes seed precision, not the target arithmetic mode, and cannot admit
an atom list, a density, or a budget.

Returning an executable kind rather than only an eligible provider is important:
the CPU automatic descriptors resolve each system in the prepared owner, while
the existing CUDA contract uses an explicit descriptor only after conservative
system admission. The native capability mask owns provider preferences; Python
does not revalidate the admitted model or rediscover its transport on current
images. Explicit options retain their checks, and a missing combined symbol
retains older-library discovery. A zero or refused native result is not a
request to retry the older policy. The CLI remains opt-in and CPU-only.

## Evidence

On base `cae2959e1`, the SDK used the discovered automatic descriptor against
installed PBE/H2 at the exact local cap and one byte below, without Python or
toolchains on its runtime path. Energy, force, capacity and fallback matched
the CLI and Python boundaries. 152 CPU CTests, 387 installed Python tests,
the ABI and ownership inventories, and strict Sphinx passed. The separate
older-library suite passed 68 tests and skipped three new-symbol cases.

Seven interleaved PBE0/water force endpoint pairs after warmup measured
Python construction, exact-host planning, cold/warm/moved execution and
teardown, plus installed CLI `resources` and exact-cap `run`. Median changes
were +0.08% for Python and +0.26% for CLI. Energies, forces, convergence,
modeled peak and seed diagnostics (excluding time) matched. Target Fock
counts were 13/4/12 on the Python sequence and 13 on the CLI cold run, with
7 cold preliminary Fock builds. CUDA was exercised only as a context-free
model query; there is no real-device execution claim.

Raw receipts, binary digests, and compiler-cache statistics are under
`/data/jzzeng/gqc-2227-sdk-JsefEq0x/native-default-guess`.

## References

- [CLI automatic admission](2026-11-23-cli-automatic-preliminary-admission.md)
- [Initial-guess contract](../../../../docs/user/initial_guesses.md)
