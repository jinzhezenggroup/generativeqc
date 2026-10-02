# Decision: bounded multi-block stationary geometry lanes

Status: implemented, NVIDIA qualification pending
Date: 2026-10-01

## Problem

The stationary geometry owner launched exactly one 32-thread block and reduced
32 partials. Each lane independently traversed its points, with a nine-coordinate
partial and a nine-coordinate adjoint workspace per atom. More point work could
not occupy additional blocks. Issue #1479 owns the broader resident semilocal
response pipeline; this is only its point-lane scheduling slice.

## Why prior compiler checks did not flag this

The emitted contraction and native launch were separate contracts; no resource
plan connected independent grid-point extent to block count. Existing ABI,
numerical and byte-budget tests admitted a scientifically correct single block.
`compiled_gpu_profitability` receives resources, target and block threads; its
per-SM theoretical occupancy estimate cannot detect a one-block launch grid, and
this stationary path did not call it. The advisory work scanner's Python route
examines NumPy AST patterns rather than the CUDA inside emitted string literals.
A blanket warning about 32-thread blocks would also flag this new eight-block
schedule incorrectly. A future general advisory needs independent work extent,
legal reduction regrouping, budget/target-admitted concurrency and the actual
launch grid; unknown inputs must remain advisory-only. The focused regression
here requires 256 admitted points to expose eight blocks without forbidding
small-work or budget-limited single-block fallbacks.

## Decision

The method compiler owns `stationary_resources.py`: exact arena accounting and a
bounded geometry schedule shared by semilocal, full-range hybrid and composite
stationary owners. Native code validates the selected capacity and physical device
limits, allocates both panels once, and launches that schedule on the existing
borrowed stream. No functional-name-specific scientific implementation is added.

The capacity is at most one lane per tile point and 2,048 lanes, capped by an
8 MiB combined partial/workspace allowance and the remaining admitted owner
budget. One 32-thread block per warp exposes more independent blocks for the
usual 256-point tiles. This block-size choice is an unmeasured initial policy;
there is no occupancy or endpoint speedup claim. Small tiles/tight budgets retain
a single-block fallback. Every tail uses only min(capacity, point count) lanes;
zero-point leases launch no kernels. An extra launched thread returns before
addressing either panel.

Each reduction sums the live lanes in increasing order, and no atomics accumulate
forces. The pointwise AO pullback, generated XC expressions, generated Becke
partials and `generativeqc_grid_adjoint::contract_point` are unchanged. Changing
lane count changes floating-point grouping: reproducibility for a fixed plan is
not bitwise equivalence to the old 32-lane sum.

## Resource and lifetime invariants

- Two panels use exactly `18 * lanes * atoms * sizeof(double)` bytes. Native
  allocation, compiler admission, pointer layout and metrics agree
- Optional geometry expansion reserves all other planned tensor owners first
- The existing native integral provider receives at least its old residual, up
  to the shared paired-one-electron conservative admission bound. This matters
  even for a retained Direct provider: its host output check requires 48 bytes
  per atom. Consuming the residual could otherwise turn a previously admitted
  endpoint into OOM or bypass its prepared source
- The reservation formula is checked against the actual native expression and
  current AO expansion limit. It is a resource bound, not a second science kernel
- Composite owners retain their existing integral/nonlocal reservations before
  dividing the remaining budget between semilocal and nonlocal accumulators
- All eight synchronous/deferred/resident/explicit-owner entrypaths use the same
  lane capacity and live-tail reducer. Existing same-stream ordering, drains,
  sticky failures, reset/rebind, and borrowed-owner lifetimes remain in place
- AOT identities explicitly hash the resource module, and CMake tracks its
  generated-wrapper and post-link contract dependencies. Prepared execution
  provenance includes the selected resource schedule

## Evidence and limits

Host tests execute the emitted kernel and reducer with simulated block/thread
indices, including 1/17/32/64/256/2,048 lanes, multiple irregular tile widths,
nonbinary weights, reused dirty capacity, guard values, explicit/implicit owners,
external seeds, empty work and sticky producer failure. Scientific helper stubs
in that harness isolate scheduling; they are not a physical force oracle.

Separate host-compiled tests execute native allocation/create admission and
verify exact panel offsets, output-null behavior, budget/shape/device rejection,
simulated allocation failure cleanup, and the native pair-resource formula.
Budget-edge tests preserve the former native allowance at the +432/+433-byte
boundary that could otherwise incorrectly select a 33rd lane.

The opt-in real-CUDA suite adds 32-vs-256-vs-2,048-lane equivalence, tails,
empty work, repeated scratch reuse and same-size changed geometry for LDA/PBE,
r2SCAN and PBE0 in both spin modes. Existing complete endpoint independent
analytic and reconverged finite-difference tolerances are unchanged. The new
multi-block route is asserted in the existing LDA/GGA/meta-GGA force gates.

No NVCC compiler or NVIDIA device was available locally. These CUDA tests remain
unrun here, and the PR must stay Draft until its exact head is qualified on NVIDIA.
CuMetal or host-emulated scheduling checks cannot substitute for those gates.

No new performance result is reported. The supplied 12-atom geometry timing and
sub-3-second target are context, not measurements of this patch. Launch counts,
point visits and atom-pair visits do not decrease in this slice. Evaluate complete
cold/warm/changed-geometry endpoints and semantic counters before promotion.

## Rejected alternatives and follow-ups

- A fixed larger block alone still leaves most SMs idle and does not plan scratch
- Blindly growing scratch can steal the native provider's already-needed budget
- Pair-geometry residency and cooperative Becke adjoints are separate follow-up
  changes; no pair caching, screening, locality approximation, grid or quadrature
  change is introduced here
- The legacy standalone `xc/geometry_cuda.py` XC-gradient owner is a different
  fixed-worker resource contract and is not changed by this stationary-owner PR

References: #1479; related resident-grid transport work #1659

## Integration addendum: 2026-10-02

The landed DF provider (#1654) selects its fitted derivative publication through
a separate method and explicitly excludes DF response scratch from its reported
resource scope. Optional geometry expansion therefore retains the fitted
provider's entire previously admitted remainder. Reusing the Direct one-electron
estimate, or checking only whether the Direct method exists, would otherwise
shrink a different provider's allowance and could turn an admitted fitted force
into an allocation failure. Fitted execution keeps the bounded 32-lane schedule
until it exposes a resource contract that admits additional geometry scratch.
Executed-source budget tests cover fitted-only and combined-provider owners.

The maintainer subsequently permitted merging changes whose only gap is missing
acceptance evidence. NVIDIA execution remains unrun locally; this supersedes the
earlier Draft-only requirement without changing numerical gates or claiming a
device result.
