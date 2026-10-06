# Decision: resolve incremental Direct-J/K policy before backend lowering

Status: implemented
Date: 2026-10-07

## Problem

The exact incremental Direct-J/K controller had one mathematical contract but two
policy owners. CPU SCF decided eligibility and periodic refresh in
`mean_field_driver.cpp`, while CUDA independently combined provider admission,
mixed-precision exclusion, and a special screened-delta refresh clamp in
`cuda_rhf.cpp`. A caller could therefore request the same incremental control
without one shared statement of the effective accepted-iterate policy.

The numerical distinction is real but belongs in lower capabilities rather than
backend-specific policy. CPU's current exact Direct lower performs no
density-weighted quartet skipping, so exact linearity permits the caller's wider
anchor interval. CUDA Direct compaction can screen from the active density
magnitude; with nonzero screening tolerance, repeated delta updates must not
accumulate omitted contributions through an arbitrarily long anchor chain.

## Decision

Resolve incremental Direct-J/K once in the shared SCF layer from
`ScfOptions` plus explicit lower capabilities:

- provider eligibility for the exact incremental contract;
- whether the selected lower performs density-weighted screening; and
- whether another precision policy currently owns the iterative Fock error
  budget.

The resulting policy publishes `requested`, `active`, the caller's requested
refresh interval, and the effective interval. Exact-linear lowers preserve the
requested interval, including zero for no periodic refresh. A lower advertising
density-weighted screening with a nonzero screening tolerance is limited to one
accepted delta build before a full-density refresh. A conflicting precision
policy fails closed to the ordinary full-build route.

CPU and CUDA consume this same resolution. CPU retains host-side anchor/delta
storage and declares exact-linear execution. CUDA retains device-resident
anchor/delta preparation and quartet compaction and declares density-weighted
screening. Backend lowering still owns execution mechanics; it no longer owns
the numerical refresh rule.

This change does not promote incremental Direct-J/K to a default and does not
add an adaptive profitability heuristic.

## Rejected alternatives

- **Force the CUDA screened cadence onto CPU unconditionally.** This would make
  the visible intervals look identical while discarding the CPU lower's stronger
  exact-linear capability and adding needless full builds.
- **Keep the duplicate backend rules.** That preserves today's behavior but
  prevents CPU tests from protecting the policy that CUDA is expected to
  execute and invites future drift.
- **Combine policy unification with default promotion or profitability
  heuristics.** Those require matched complete-endpoint evidence and should not
  be hidden inside an ownership refactor.

## Invariants

- Only accepted SCF iterates may advance the incremental anchor.
- Proposal/audit work must remain full-build work and cannot mutate the anchor.
- Geometry or topology changes must establish a new full-density anchor.
- Final physical-state validation remains strict full-density work.
- Approximate J/K providers remain outside the exact incremental controller.
- Mixed iterative Fock remains mutually exclusive until its error budget is
  explicitly composed with incremental Direct-J/K.
- A density-weighted screened lower may not accumulate more than one accepted
  delta update before a full-density refresh.

## Evidence

The host provider test now directly exercises shared policy resolution for
exact-linear cadence preservation, screened-delta refresh bounding, disabled
screening, conflicting precision policy, provider refusal, and exact-versus-DF
eligibility.

The existing real-CUDA test remains the execution gate for RHF/UHF,
screened/unscreened delta builds, strict final full rebuilds, quartet work
accounting, changed-geometry anchor reset, and cached-plan policy isolation.
No new device timing or default-promotion claim is made by this refactor.

## Consequences

Policy behavior can now be tested without invoking a backend, while lower-specific
execution differences remain explicit capabilities. Future adaptive profitability
can extend the same shared policy rather than adding another CPU/CUDA branch.

## Revisit when

Add automatic/default activation only after complete cold, warm, and
changed-geometry endpoint evidence establishes a durable benefit and a
work/resource guard. If a future CPU lower adds density-weighted delta screening,
it must advertise that capability and automatically receives the same bounded
refresh cadence.

## References

- #990: exact incremental Direct-J/K implementation.
- #1598: default-promotion qualification.
- `src/scf/types.hpp`
- `src/scf/solver/mean_field_driver.cpp`
- `src/scf/cuda_rhf.cpp`

Agent: ChatGPT
Model: GPT-5.6 Sol
