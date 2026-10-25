# Decision: whiten large retained DF factors with triangular solves

Status: implemented
Date: 2026-10-26

## Problem

The cold DF HF96 and PBE0-96 endpoints have 768 AOs, 3,616 auxiliary functions
and 295,296 packed AO pairs. Materializing their retained factor took about
6.6 seconds: approximately 2.0 seconds generated raw integrals and 4.6 seconds
applied the spectral metric projection. The incumbent already omitted rotation
back to the original auxiliary basis. Removing another supposed rotation would
not address the measured work.

## Decision and invariants

Keep the eigensystem, rank cutoff, symmetric inverse square root and physical
integral source. A full-rank singleton packed-single owner may additionally
factor `M=L L^T` and retain `F=A L^-T` instead of the spectral factor. The
large panel transform is then a triangular solve with half the dense arithmetic.
J/K are invariant under the auxiliary orthogonal gauge. Force response is not:
its physical fitted source is `F L^-1`. Coulomb, occupied/final-projection and
bounded general-density consumers must carry the same explicit reverse map.

Automatic admission requires at least 1,024 auxiliary functions, a metric-sized
existing panel and enough existing eigensolver scratch for potrf. Native tests
can select spectral or Cholesky whitening explicitly through a typed immutable
storage option. No ambient research switch enters production. Rank truncation,
batches, inadequate scratch or positive potrf status preserve spectral setup;
provider errors remain errors. Setup borrows the scaled-eigenvector slot for L,
then replaces it with L^-1 only after all retained panels are complete. The
persistent metric reservation already covers this transferred slot. Error paths
and destruction retain single ownership.

Cholesky workspace queries and submission belong to the shared CUDA solver
provider, borrowing the existing stream-bound handle without allocation or
synchronization. The DF adapter retains bounded scratch admission, error
translation and numerical fallback policy. This preserves the earlier retirement
of method-owned raw cuSOLVER submissions; adding whitening does not reintroduce
them. Triangular matrix solves remain in the existing DF matrix provider.

Small-panel qualification also exposed an incumbent fitted-response capacity
error: one AO slice was reserved, but the simultaneous B*C temporary was not.
The response planner now reserves both during projection, independently of the
later derivative-panel extent. It still honors the original caller allowance.

## Evidence

The research three-round same-card #2786 campaign, before promotion to the
ordinary default, gave cold medians of 20.9577 -> 19.2429 seconds for DF HF96
and 26.3150 -> 24.5842 seconds for DF PBE0-96. Corresponding GPU4PySCF medians
were 20.0063 and 26.1731 seconds. Warm medians stayed near 4.09 and 8.13 seconds.
Across all cold and warm comparisons, maximum energy changes were 7.28e-12 Eh
and 5.46e-12 Eh; maximum force changes were 3.07e-12 and 2.30e-12 Eh/Bohr.
These research timings are not substituted for the separate default campaign.

The final default-only three-round campaign has DF HF96/PBE0-96 medians
19.3003/24.6200 seconds against same-card GPU4PySCF 20.0240/26.1959 seconds.
All 48 native cold/warm calls across the four promoted endpoints differ from
the preserved baseline by at most 7.28e-12 Eh and 3.03e-12 Eh/Bohr. Both
engines' DF HF96 fitting error still fails the common direct-reference energy
gate; this performance point and its failure stay visible. Source-bound
statistics and unchanged gates are in
[the endpoint evidence](../../../../benchmarks/results/cold-four-defaults-20261026/README.md).

The production library passes 108 independent auxiliary-gauge adjoints covering
condition numbers through 1e10, ragged panels, pure J, occupied K and final
projection reuse. Native physical raw-integral oracles cover RHF/UHF, truncated
metrics, insufficient Cholesky scratch, one-Q response, general density and
final leases. Full native DF/provider suites and 48 finite-difference force
gates pass. The complete occupied-response suite passes Compute Sanitizer
memcheck with zero errors.

The old occupied-response fixture also failed on the unchanged baseline because
it reported truncated response as an occupied contraction. Its expectations now
distinguish validated factor leases from the general spectral derivative and
actual borrowed staging. Independent numerical checks were retained.

## Rejected alternatives and references

A smaller allocation alone is not a work reduction. Keep the raw-generation and
metric-transform timings separate, and do not claim removal of an eigenbasis
rotation that was already absent. Never feed F directly into a consumer that
expects symmetric B or spectral E.

Frozen experiments, commands, source patches, binaries and failure logs remain
under `n1:/data/jzzeng/cold-opt-20261025/`. `cholesky-*` names the research image;
`cholesky-default-*` first promotes the typed policy, and `wb-aot2-*` includes the
bounded response-scratch correction and the separately qualified WB package.
The base is `82ce9e56a352fcc1d86e7d64eeaae08c6f1eb5c8`; each source archive has a
patch and archive digest. Compiler-cache provenance and Slurm allocations are
retained. Current behavior is documented in
[packed DF storage](../../../../docs/developer/df_packed_storage.md).
