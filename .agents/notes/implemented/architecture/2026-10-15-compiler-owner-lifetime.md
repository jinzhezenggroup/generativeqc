# Decision: opt-in outer lifetime ownership for isolated compiler sessions

Status: implemented
Date: 2026-10-15

## Problem

The Hessian CLI owned its Python worker's process group, but runtime compilers
and private foreground sccache servers intentionally start separate sessions.
Killing only the Python group therefore left those numerical compilation workers
alive. A successful interpreter exit could also leave ordinary group descendants
running because the frontend only cleaned its group after failure.

## Decision

The CLI closes an exclusively owned pipe writer whenever its invocation ends.
Only the read end is lent to the Python worker. The compiler's existing process
and private sccache launch points opt into a stdlib-only watchdog when that
explicit descriptor exists; ordinary Python/JIT callers keep their prior launch
behavior. The watchdog retains ownership of each isolated compiler group and
watches both the outer owner channel and a per-launch cancellation channel.
EOF or cancellation reclaims that group even if the caller was SIGKILLed or was
blocked inside a native call. Success and failure also reclaim descendants.

Neither the writer nor the local cancellation writer is inherited by compiler
children. Private sccache's exclusive cache-slot lease is still inherited and
remains held until its server/children and watchdog are gone. The ordinary shared
sccache server, other compiler invocations and CI groups are never selected by
name or touched. No compiler cache is cleared and scientific artifact identity
rules are unchanged. Native source identity naturally includes this source edit.

The watchdog uses isolated, no-site interpreter startup so user site hooks cannot
run before its owner-death monitor. A bounded cancellation acknowledgement guard
uses its explicit child-group registration if a watchdog is stopped or fails;
that condition is an error, never a successful compilation.

The watchdog is execution plumbing, not an additional compiler or mathematical
owner. It forwards the original command, environment, standard streams and
explicit inherited descriptors. Its interpreter startup is counted within the
existing compiler timeout. Abruptly killing a caller can leave an empty private
socket directory; the lifetime guarantee does not claim crash-time filesystem
cleanup. No geometry or numerical result is written to such directories.

## Rejected alternatives

- Removing compiler session isolation would break per-invocation cancellation.
- Signalling all compiler/cache processes would kill unrelated user work.
- A Python signal handler alone cannot clean up after SIGKILL or while blocked
  inside native code; a grace period does not repair that ownership boundary.
- Registering a child PID after launch leaves a spawn-to-registration death race.
  The watchdog instead creates the child and monitors owner death itself.

## Evidence

Against PR head `2b1d6d1c72a884ceaebf89b28098fa374ec01372`, the compiled
`hessian.cpp`/`json.cpp` transport owner returned valid JSON while a redirected
child survived successful exit. A second probe used the unmodified production
`run_compiler`: a one-second CLI timeout returned failure but its isolated
compiler continued running. Both failures are reproduced by executable tests;
the successful-exit regression also fails when the original source is restored.

The scoped repair passes the transport owner's success/failure, timeout,
blocking-libc-call and SIGKILL-worker regressions, while an unrelated process
stays alive. Private sccache server/child cleanup and slot release are tested on
outer-owner EOF and worker crash. Existing default launcher and concurrent
isolation regressions remain in place. No native scientific or GPU qualification
is inferred from these process-lifetime tests.
