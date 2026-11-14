# Decision: one native CPU Hessian bridge for public Python and CLI

Status: implemented
Date: 2026-11-12

## Ownership and identity

The public canonical B3LYP RKS/UKS derivatives now consume the accepted native
batch through a synchronous context-locked bridge. The actual prepared system,
Fock, basis, grid and correction owners are borrowed under their current token;
the final physical frame is independently verified before stationary reuse.
Requested elements, coordinates, charge and multiplicity are checked natively
through final publication, not trusted as presentation-only metadata. Python
passes buffers and detaches results; the CLI calls exactly the same bridge.

The response static target can now link into the core without the previous
static-to-core-to-static cycle. Independently linked CPU fixtures retain their
existing dependency direction. Core-linked response uses the actual backend
definitions, rather than treating a CUDA-enabled binary as a CPU-only layout.
This phase validates CPU execution only, not a CUDA-enabled build/device.

B3LYP-D3(BJ) is a native total-method catalog entry built by replacing the
canonical electronic MethodSpec's correction with its existing canonical fit.
No consumer duplicates the electronic coefficients or BJ parameters. The
uncut/two-body/ATM-off dispersion primitive now declares second-order requests;
full corrected MethodIR identities intentionally change and their generated
native catalog is rebuilt. Electronic identities and pinned D3 data stay fixed.
Missing exact electronic or total closure never selects a Python/JIT fallback.

## Publication, evidence and boundaries

The full matrix and receipt are private until every column, physical residual,
energy reconstruction, source authentication and raw symmetry gate succeeds.
Output/receipt/request overlaps and borrowed source aliases refuse. Child ECP
and D3 work are last-source records, so both are accumulated per accepted column.
Integral/XC counters remain cumulative. Native reservations distinguish numerical
storage from borrowed primal sources, export scratch, caller request buffers,
provider/BLAS overhead and empirical RSS. Whole-request nuclear work defaults to
100 million primitive visits; memory-bounded does not mean practical work.

CPU frame export now records the explicit gate it actually verifies:
min(1e-9, requested density tolerance). Canonical density reconstruction is
checked against that gate before export; no unverified tolerance is attached
later by the derivative consumer. The native response default uses absolute
1e-12 alongside relative1e-10, because a symmetry-zero RHS can have rounding
residual3e-18 but an uninformative relative residual. Explicit relative-only
solver options keep their original gates and can legitimately refuse.

Qualification logs live under `build/qualification/host-cpu-endpoint-20261112`.
They retain compiler-cache provenance, public independent multistep RKS/UKS
controls, native C/API transactional/resource/alias tests and an actual CLI
with PATH/PYTHONHOME/PYTHONPATH pointing at nonexistent interpreter locations.
Earlier failing attempts are retained rather than overwritten as successes.

Public ECP and DF domain gates remain closed. Installed-wheel isolation,
real Ni/I/Cs electronic qualification, practical25--40atom complete electronic
time/memory/work, the native saddle migration and final unscaled frequencies
remain separate acceptance work. The frozen predecessor installed library and
accepted component receipts are not replaced by this phase's build.

The saddle migration is subsequently implemented in
[the retained-batch decision](2026-11-12-native-saddle-derivative-batch.md).
That changes derivative ownership, not the heavy/installed/practical acceptance
boundaries recorded here.

## Rejected alternatives

Do not reconstruct AO/grid providers from snapshots, return a corrected label
over an electronic-only owner, add D3 again in Python, repair raw symmetry,
infer whole-matrix work from one final-column counter, or claim reserved bytes
as complete-endpoint RSS. Revisit scheduling with complete electronic evidence,
not microkernel or correction-only timings.
