# Decision: reject a fully finalized public DF solve as a cold Direct proposal

Status: rejected as measured; not a rejection of a separate bounded density-only proposal API
Date: 2026-11-08

## Problem

The non-CBS A100 README census still loses HF Direct at48/96 atoms. The
separate adaptive-delta experiment saves about12% at96 atoms but remains far
behind GPU4PySCF and is not qualified for default promotion. Test whether a
native fitted density can cheaply remove expensive exact iterations, without
changing the final Direct Hamiltonian, force calculation or stopping policy.

## Experiment and ownership

One freshly constructed CUDA DF owner produces its own density. Capture that
density with the native snapshot transport, release the entire DF owner, then
restore/physically validate it in a freshly constructed ordinary Direct owner.
This is an explicit diagnostic cross-approximation proposal, not an assertion
of same-model checkpoint compatibility. DF Fock matrices, DIIS history,
orbitals and trusted target-energy baselines do not cross the boundary. The
new target still executes strict physical Direct SCF and analytic forces.

Constructor, auxiliary input preparation, DF solve, detached transport,
validation/teardown, Direct solve, forces and host export all enter cold time.
Three warm calls retain only the target's own frozen cold density. Native
processes reject PySCF/GPU4PySCF imports. The existing CUDA DF bridge has its
documented native host-side preparation; this does not establish an all-device
replacement for the ordinary Direct constructor. No new production path,
selector, numerical gate or GPU-product-name dispatch is added.

The v1 proposer used Standard precision, which clamps caller-loose controls;
retain those failed5090 results rather than relabeling them as a loose solve.
The v2 proposer uses ordinary FP64 Calculator and records actual resolved
energy/density controls, both1e-4, with max_iterations20. The target's Standard
Direct controls remain unmodified. All source, scripts and image identities
are frozen before submission.

## Complete endpoint evidence

All times are seconds. These are point observations, not a balanced promotion
cohort. The A10048 off/on pair shares one exclusive allocation; the5090 v2
comparison reuses the v1 off population from a different allocation. Earlier
HF96 controls below also come from separate allocations of the same image.

|card/job|case|ordinary cold|DF-proposed cold|target iterations off/on|
|---|---|---:|---:|---:|
|A100 LSF228622|HF Direct48|6.120849|5.955743|10/4|
|A100 LSF228622|HF Direct96|16.198474 (earlier228595)|19.802837|15/4|
|5090 Slurm16774/16775|HF Direct48|6.250384|7.373423|10/4|
|5090 Slurm16775|HF Direct96|16.829320 (earlier16772)|32.226876|15/4|

The v2 proposer executes9 DF iterations but **23 molecular DF Fock builds**
at both sizes on both cards. The other14 are post-SCF final-state work, not
zero-cost preparation. Target Direct executes4 builds; serialized atomic
Huckel work remains separately reported7. Thus four visible Direct iterations
do not mean four total Focks. DF preparation alone costs A1001.460552/7.913913
and50902.918225/20.681852 at48/96 atoms.

Candidate warm populations, all one target Fock:

- A10048:0.655234/0.587502/0.585740;96:2.335105/2.265594/2.265736.
- 509048:0.743227/0.738584/0.742012;96:2.689748/2.689741/2.686603.

Retain v1 Slurm16774:48 cold7.444501 and96 cold31.112110, with23/21 DF
builds respectively. Do not average these distinct proposer policies together.

## Scientific gates and diagnosis

The unchanged native-control parity gate at48 atoms fails on both cards:
all16 cold/warm cross-comparisons have maximum force spread1.84303e-7Eh/Bohr,
above1e-7; maximum energy spread is below3.6e-11Eh. This is a failed strict
control comparison, **not** an independently established HF48 oracle verdict.
No new tighter independent48 oracle was acquired, and no qualification is
inferred from matching the other engine's looser default force convention.

The original96 geometry passes the authenticated independent tighter GPU4
target from LSF228320 at absE1e-8/maxF1e-7. Largest observed errors are about
1.394e-9Eh/1.782e-8Eh/Bohr. The existing checker authenticates the original
producer/archive and exact scientific recipe. It does not use historical oracle
timings in the new performance comparison. No displaced/restart/lifetime/default
qualification is asserted for this rejected diagnostic.

The public DF endpoint intentionally finalizes a physical state even when it
is later used only for a guess. In `src/scf/rhf.cpp`, the compact loop passes
through `finalize_density_fitting_rhf` and `select_cuda_df_final_state`; their
bounded physical/canonical correction has its own non-waivable acceptance.
Loosening the preliminary SCF control does not make that physical publication
free. Removing those safeguards from public energy/force/reference endpoints
would be an invalid performance fix.

## Decision and revisit conditions

Reject this public-endpoint composition: a2.7% A10048 cold point improvement
neither beats GPU4 nor passes the declared strict native-force comparison,
while both5090 cases and A10096 are slower. Do not promote it, leave a new
opt-in, change the64-pair README census, or ship a hardware-name exception.

A future internal density-only proposal must have an explicit non-physical
result contract, bounded total preliminary work, GPU-resource and detached
density admission, no force/reference publication, and independent strict
target qualification. It must not reuse the physical finalizer's proof by
claiming an unfinalized proposal is a converged energy/force result. This
possibility is not itself evidence of profitability. Repeated strict K remains
the main Direct96 target; no new full DF-guess sweep is justified here.

## Retention

Native source `b5f5f74e34fd5e20d4557a25560121bd90b3ae0f78d95d91727b1a607da4ba8c`,
normal Release SM80/120 ELF
`442e56a153c9e79510d7a48482f92b51d0351ce16e7f3299ccee8f78af6cae59`.
Raw scripts/populations/failures and the actual-control analysis remain in
`.artifacts/hf-df-proposal-v{1,2}-20261108/`; the pre-existing adaptive native
prototype remains separate and uncommitted. No experimental native code is
included with this negative decision.

- A100228622 archive SHA256
  `2060634cf875d13ad03f5b3fdb4d61bd645642fc488fa26c718ef8db16182d5f`.
- 509016775 archive SHA256
  `1bae501c1343f54c2d8f4810446da1dd2c8027c6fa05173678cf145e80a6e1c3`.
- Initial509016774 archive SHA256
  `6fc16c054d5412676153f5a0631dae589d034bed247067292b20f6ace6e9d294`.
- Analysis SHA256
  `5e8e2d9949dcfc110ae5e52c5d22dc07d02052f2796d4753c2c181ee2c07faf1`;
  its exit1 preserves both force-control failures.

Retain both pre-submission transport failures (login/node-local path and nested
script syntax). No GPU population was submitted by either failed attempt.
Shared-home quota then rejects evidence-only CPU copies228625/228628, preserving
their nonzero exits and empty/partial artifacts. Transport-only228634 moves the
**original**228622 archive to a finite owned receiver outside that quota; no
scientific population is rerun and no unrelated files/cache are deleted.
Issue2976 records the decision and PR3041 retains only already qualified native
changes plus negative evidence, not this proposal.
