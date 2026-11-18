# Decision: run the DF-RHF force fixed-point probe on resident device data

Status: implemented
Date: 2026-11-18

## Problem

Ordinary CUDA DF-RHF force selection proves the retained determinant with a
fresh physical-Fock projector (`allow_projector_checked_reuse`). On the H100
water-96 README endpoint that probe took 79–84 ms of a 0.95 s warm call, of
which the eigensolve was 8 ms. The rest was host staging:
- The device physical Fock was downloaded and transposed (`materialize`).
- It was re-uploaded for the provider solve, and the frame was downloaded.
- The frame was validated twice, once inside the provider and once by `operations->eigen`, each with fresh S/F/C/epsilon uploads.
- The projector was downloaded and compared on the host.

The Pulay weight then re-staged D and H that the validation had just uploaded.

## Decision

`FinalStateOperations::fixed_point` is an optional backend probe. After
`products()` has validated the single-spin RHF density, its validation
workspace still holds S, H, F[D] and D. The CUDA DF probe borrows exactly that
staged state; the borrow is keyed by identity and by the host S, H and D
objects, and is cleared by every other workspace user.

The probe:
- solves F with the same Xsyevd provider and X transform (`solve_cuda_density_fitting_eigen_resident`);
- runs the same eigenframe kernel and `accept_eigen_frame` gate;
- forms the same occupied projector, and compares it with D on the device (max and hypot-RMS).

Only a diagnostic packet crosses the host boundary. Rules for when the probe's result is used:
- If it is accepted, selection uses its defects with the unchanged tolerance.
- If the frame is accepted but the defects exceed that tolerance, the probe downloads the frame and projector. The correction then promotes them, as with the host probe.
- Any failed frame gate, unsorted values or nonzero solver info falls through to the complete host route.
- UHF, the stationary route, energy-only requests, forced rebuilds and reference export are unchanged.

W = D F[D] D / 2 reuses the same staged F and D.

Acceptance cannot loosen: the probe accepts only frames the host route accepts, and the projector arithmetic is identical. RMS is merged per thread with hypot rather than sequentially, so it can differ in the last bits.

## Evidence

`benchmarks/results/h100-df-final-resident-fixed-point-20261118/`: fixed point
79–84 → 13 ms, W 11–14 → 4–7 ms. Water-96 DF-HF warm 0.91–0.92 → 0.84–0.85 s,
cold 3.22–3.25 → 3.15–3.16 s. Iteration/Fock counts and reference errors are unchanged.

## Rejected alternatives

- Extending stationary (commutator-only) reuse to RHF would drop the deliberate projector proof; that is an acceptance-gate change, not an execution change.
- Caching S/H across calls by content would save the remaining validation staging, but needs a content witness for every call.

## Revisit when

The retained-converged-pair design lets DF-RHF keep D_k with its already built J/K. The 47 ms final J/K and this probe's staging can then share one resident state.
