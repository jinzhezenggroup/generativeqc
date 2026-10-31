# Decision: retain the general CUDA owner for unqualified shell RSH forces

Status: proposed
Date: 2026-10-30

## Problem

The existing H100 native composition test fails for the shell RSH derivative
adapter at omega 0.4. A source-matched diagnostic reproduces the same SR/LR
exchange error with both `000517f82dc62130459a6fd92a27518a158c49cc` and
`80bb2f910ad7354a84ad69e3154c3b5758ed411a`. J agrees with the independent
CPU derivative. The retained general CUDA provider agrees with independent
CPU analytic J and five-point differences of positive-interval SR/LR values
within 4e-13 for the s/p fixture at omega 0.3, 0.4 and 0.5. The exact device-side
cause of the shell exchange discrepancy is unresolved.

## Proposed decision

Keep the qualified omega=0.3 full/LR host decomposition and isolated LR callback
schedule. Route other RSH host exports, and general three-channel callbacks,
through the existing retained general CUDA derivative owner. Its derivative
arena is already reserved by the Direct plan. The callback publishes positive
energy gradients on the owning stream and borrows three contiguous rows; no
host numeric array or CPU scientific consumer participates in production.

The general callback preserves exact radial/coefficient identity, spin and
extent validation, capture refusal before a drain, typed callback failures,
failure cleanup and ordinary asynchronous borrowed lifetime. A trace operation
identifies `direct_rsh_force_general_fallback`. The default composed omega=0.3
consumer continues to use its qualified full-range and isolated-LR shell paths.

## Rejected alternatives

- Relaxing the native 2e-9 gate or accepting zero-error memcheck as numerical
  qualification would hide a reproduced scientific failure.
- Treating the resident host-array adapter as the callback implementation
  would introduce a production D2H/host numeric consumer.
- Promoting the unqualified fused source for arbitrary omega because primitive
  host controls pass would overstate the evidence. Separate primitive and
  contracted-source host checks pass, while the actual H100 shell consumer fails.
- Quantizing runtime omega to 0.3 would change the mathematical operator.

## Evidence and required gates

Issue [#2901](https://github.com/jinzhezenggroup/generativeqc/issues/2901) records
the complete source/core/device identities, component errors and retained
failed-run digests. The private diagnostic archive SHA-256 is
`bc836f588b25780612258603e55d8b89576da972c876e5f966213793bd0c8c63`;
this digest is provenance, not a public archive recovery claim.

New native tests compare host, resident and borrowed sources with an analytic
CPU J reference plus two-step five-point CPU range-value derivatives. Cases
cover restricted/unrestricted s/p at omega 0.3/0.4/0.5, Cartesian/spherical d
at omega 0.4, typed callback failure/drain/recovery and capture refusal. They
retain the existing 2e-9 numerical gate. Device validation, unfiltered memcheck
and affected complete molecular E+F qualification are pending; this note does
not establish production acceptance or a performance result.

## Consequences and revisit condition

General custom-omega forces can do more GPU work than the shell schedule.
The choice favors independently qualified results with the existing bounded
owner. Do not claim custom-omega acceleration from default-omega timings.
Revisit shell admission only after the actual device failure is isolated and
the independent source, molecular, lifetime and resource gates pass.
