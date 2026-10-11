# Decision: copy distinct forward W values without aliasing force cotangents

Status: implemented
Date: 2026-10-12

## Problem

#2232 safely identifies equal physical occupied W tuples for energy-only
evaluation. Directly applying its scalar-load remapping to response storage
would risk collapsing six independent cotangents or omitting derivative terms.
The joint force/Fock page owner instead needs every original canonical W slot.

## Decision

Use the existing compiler-owned physical-tuple map only when actual W storage,
compute and accumulation are all FP64. Build the first identical forward seed
once and copy it into each duplicate canonical slot on the original owned CUDA
stream. Keep every resolvent load, six independent W/V cotangents, pullback,
occupied/virtual multiplicity, finite check and canonical reduction unchanged.
Numerically coincident values from different tuples do not authorize reuse.

Copies add no numeric scratch or new provider. Report exact `w_copy_calls` and
`w_copy_bytes`; preflight their complete worst-case counter bound before input
access. Preserve original non-FP64 computation and all single/two-panel,
partial-page, strict/generated and budget fallbacks. Standalone response owners
are unchanged.

## Evidence and source boundary

Frozen clean baseline `34dd51ed3ebc5b03b17f095de2cd7b39903923c8`, plus the
retained owner/diagnostic/benchmark overlay. The experiment is independent of
#2298/#2299/#2300: trial-state DIIS, unbatched replay and force Q8 are unchanged.
No generated mathematical artifacts change; exact legacy math hashes remain
pinned by the existing independent tests.

Fresh cold ethane230 complete-force ABBA runs on node2's assigned RTX PRO 6000
in finite Slurm 2905. The referenced publication retains every process/resource
receipt and independent gate. Complete median wall decreases
275.614076 -> 274.137995 seconds, **0.5356%**; both candidates beat both controls.
Joint triples median decreases 39.289207 -> 37.652091 seconds. Timing is
descriptive scoped evidence, not
statistical/global performance or latest-master whole-library timing.

There are 405 Fock vector cubes. The original six seeds per cube require 4,860
W GEMMs; the strict physical tuple inventory has 1,953 distinct seeds and the
candidate reports **3,906 GEMMs**, **477 copies**, **41,189,373,576 copy bytes**.
Copies are logical payload, not measured DRAM traffic. Native Fock contraction
summands decrease **9,924,048,505,176 -> 8,739,854,014,866**, exactly the removed
477 seeds' **1,184,194,490,310** summands. Reconstructing both ledgers from the
original canonical inventory and candidate's 357 measured panel GEMMs reconciles
the complete totals; baseline did not expose separate per-W counters.

Reverse triples work is unchanged at **9,224,865,365,040** summands. CCSD work,
Lambda/orbital actions and complete logical numeric capacity are unchanged.
Retaining every reverse contribution is a scientific invariant, not merely an
energy gate. Physical peak VRAM is not measured.

All **38** independent complete-force/public tests and **17** joint response,
mixed-precision exclusion, exact-copy-count, one/two-panel/partial-page,
budget/sentinel and all-equal overflow tests pass. Large energy/residual/
Lambda/Z and independent force-direction gates pass; water memcheck reports
zero errors. Maximum matched force difference is 1.40e-14 Eh/bohr and maximum
independent large directional error 7.11e-9 Eh/bohr. All 133 default-inventory
tests pass, including six new precision/copy/cotangent drift rejections.

The two new diagnostic fields change a **private C++ result layout**. Rebuild
the library, complete endpoint, force probe and primitive response probes
against their own matching headers. Do not pair a baseline object with a
candidate library. Public C/Python ABI and scientific inputs remain unchanged.

## Alternatives and consequences

Do not alias or skip cotangents because primal W values agree. Combining bars
and one pullback would be a different derivative/reassociation change needing
independent finite/failure proofs and force qualification. The conservative D2D
copies preserve the original response graph at the cost of explicit copy traffic.

## Revisit when

Requalify broader precision/provider domains with independent complete forces.
Consider remapped response storage only with an explicit cotangent accumulation
proof, finite/failure gates, complete memory ownership and endpoint benefit.

## References

- #2232; force consumer extensions #2298, #2299 and #2300.
- `benchmarks/results/df-triples-force-distinct-moments-20261012/publication.json`.
- `tests/python/test_df_occupied_triples_fock.py` and default-promotion guards.
